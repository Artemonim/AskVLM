"""Real STT engines of the worker pool; they run inside a worker process.

The worker process (:mod:`core.stt_pool.worker`) builds an engine with
:func:`create_engine` and drives it through the
:class:`~core.stt_pool.engine_api.PoolEngine` protocol: :meth:`load` once,
:meth:`start`, any number of :meth:`submit` / :meth:`cancel` calls, and
finally :meth:`close`.

Engines (spec section 9):

* :class:`CpuPipelineEngine` - any provider on CPU: one strict-``cpu``
  :class:`~core.pipelines.LocalPipeline` shared by ``capacity`` slot threads;
  every slot runs ``pipeline.process`` exactly like the one-shot
  ``external-transcribe`` path, so the text is the same.
* :class:`GigaamGpuBatchEngine` - ``gigaam-e2e-ctc`` on CUDA: a small audio
  prep pool (ffmpeg only) and ONE batch-loop thread that runs zero-padded
  batches of 20 s windows dealt round-robin across the active jobs.
* :class:`WhisperGpuEngine` - Whisper on CUDA: a prep pool that routes short
  jobs with an explicit language to cross-file clip batches, and ONE
  inference-loop thread that runs those batches and the sequential jobs.

Held work is taken in the admission order of spec section 14 (effective
priority class with aging from the engine's receipt time, round-robin across
calls, then ``seq``); see :func:`admission_order`, :func:`rotate_lanes` (GigaAM
window lanes, rotation carried between batches) and
:func:`next_inference_group` (Whisper inference calls).

Every submitted ``(job_id, attempt_id)`` gets exactly one result until
:meth:`close` starts; from then on nothing is emitted (the supervisor owns the
fate of unfinished attempts of a stopping worker).

Importing this module loads no ML package: the pipeline (torch,
faster-whisper), numpy and ffmpeg helpers are imported inside :meth:`load` and
the engine threads.
"""

from __future__ import annotations

import abc
import contextlib
import itertools
import logging
import math
import shutil
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, NamedTuple, TypeVar

from core.stt_errors import DeviceUnavailableError
from core.stt_pool.config import DEFAULT_PRIORITY_AGING_S
from core.stt_pool.engine_api import EngineResult, LoadInfo
from core.stt_pool.keys import (
    DEFAULT_PRIORITY,
    DEFAULT_WHISPER_MODEL,
    PRIORITY_RANK,
    device_for_backend,
    model_identity,
    provider_family,
)
from core.stt_providers import STT_PROVIDER_GIGAAM_E2E_CTC
from core.whisper_batch import (
    MAX_CLIP_S,
    WHISPER_SAMPLE_RATE,
    load_audio_16k,
    speech_span,
    transcribe_clips_batched,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    import numpy as np
    from numpy.typing import NDArray

    from core.gigaam_ctc_wrapper import WavWindow
    from core.pipelines import LocalPipeline
    from core.stt_pool.engine_api import EmitFn, PoolEngine, WorkerSpec, WorkItem
    from core.stt_pool.keys import Backend, Priority

logger = logging.getLogger(__name__)

# * Prefix of per-job temp directories (listed in core.temp_gc).
TEMP_DIR_PREFIX = "askvlm-pool-"
# * File touched in every active job directory: the temp GC treats a directory
# * as stale only when its newest file is older than 1 h.
KEEPALIVE_FILE_NAME = ".keepalive"
# * Touch interval of the keep-alive files (well below the 10 min requirement).
KEEPALIVE_INTERVAL_S = 300.0
# * Overall bound of the thread joins in close().
CLOSE_JOIN_TIMEOUT_S = 10.0
# * Most audio-prep threads of a GPU engine (prep never touches torch).
MAX_PREP_THREADS = 2
# * Beam size of both Whisper paths (LocalPipeline.process uses 10).
WHISPER_BEAM_SIZE = 10
# * ffprobe limit for a Whisper batch candidate that arrives without duration.
DURATION_PROBE_TIMEOUT_S = 10.0

_KEEPALIVE_STOP_TIMEOUT_S = 1.0
_TEMP_REMOVE_ATTEMPTS = 3
_TEMP_REMOVE_RETRY_S = 0.2

_T = TypeVar("_T")
_K = TypeVar("_K")
WhisperRoute = Literal["batch", "sequential", "empty"]


# * ---------------------------------------------------------------------------
# * Admission order (spec section 14)
# * ---------------------------------------------------------------------------


class AdmissionTicket(NamedTuple):
    """Admission data of one work item held by an engine.

    Attributes:
        priority: Call priority class of the item.
        call_id: Client call of the item (empty when the supervisor sent none).
        seq: Supervisor admission sequence number.
        received_at: Engine receipt time (``time.monotonic``); aging starts here.
        receipt: Engine receipt counter; it breaks the remaining ties.

    """

    priority: Priority
    call_id: str
    seq: int
    received_at: float
    receipt: int


def effective_rank(priority: Priority, *, waited_s: float, aging_s: float) -> int:
    """Return the priority class an item competes in after aging.

    ``rank = PRIORITY_RANK[priority] - floor(waited_s / aging_s)``, clamped at
    0: an item rises one class per ``aging_s`` of waiting. A negative or NaN
    wait counts as no wait, and an ``aging_s`` that is not a positive number
    disables aging.

    Args:
        priority: Base priority class.
        waited_s: Time the item has waited in the engine.
        aging_s: Aging step (``priority_aging_s``).

    Returns:
        The effective rank: 0 (``high``), 1 (``normal``) or 2 (``low``).

    """
    base = PRIORITY_RANK.get(priority, PRIORITY_RANK[DEFAULT_PRIORITY])
    if math.isnan(aging_s) or aging_s <= 0 or math.isnan(waited_s) or waited_s <= 0:
        return base
    steps = waited_s / aging_s
    if math.isinf(steps):
        return 0
    return max(0, base - math.floor(steps))


def _seq_key(ticket: AdmissionTicket) -> tuple[int, int]:
    """Return the in-call order key of a ticket (``seq``, then receipt)."""
    return (ticket.seq, ticket.receipt)


def _class_calls(
    entries: Iterable[_T],
    ticket: Callable[[_T], AdmissionTicket],
    *,
    now: float,
    aging_s: float,
) -> list[tuple[int, list[list[_T]]]]:
    """Group entries by effective class and call.

    Returns:
        ``(effective rank, calls)`` per class in ascending effective rank;
        ``calls`` are ordered by their oldest entry (``seq``, receipt, call
        id); each call is its entries in ``seq`` order.

    """
    classes: dict[int, dict[str, list[_T]]] = {}
    for entry in sorted(entries, key=lambda item: _seq_key(ticket(item))):
        data = ticket(entry)
        rank = effective_rank(
            data.priority, waited_s=now - data.received_at, aging_s=aging_s
        )
        classes.setdefault(rank, {}).setdefault(data.call_id, []).append(entry)
    return [
        (
            rank,
            sorted(
                calls.values(),
                key=lambda members: (
                    _seq_key(ticket(members[0])),
                    ticket(members[0]).call_id,
                ),
            ),
        )
        for rank, calls in sorted(classes.items())
    ]


def _ranked_admission_order(
    entries: Iterable[_T],
    ticket: Callable[[_T], AdmissionTicket],
    *,
    now: float,
    aging_s: float,
    served: Mapping[str, int] | None = None,
) -> list[tuple[int, _T]]:
    """Return ``(effective rank, entry)`` pairs in admission order.

    See :func:`admission_order` for the order; the rank is the class the
    entry competes in at *now*.
    """
    ordered: list[tuple[int, _T]] = []
    for rank, calls in _class_calls(entries, ticket, now=now, aging_s=aging_s):
        keyed: list[tuple[tuple[int, int, int], _T]] = []
        for position, members in enumerate(calls):
            offset = 0
            if served is not None:
                offset = max(0, served.get(ticket(members[0]).call_id, 0))
            keyed.extend(
                ((offset + index, position, index), entry)
                for index, entry in enumerate(members)
            )
        keyed.sort(key=lambda pair: pair[0])
        ordered.extend((rank, entry) for _key, entry in keyed)
    return ordered


def admission_order(
    entries: Iterable[_T],
    ticket: Callable[[_T], AdmissionTicket],
    *,
    now: float,
    aging_s: float,
    served: Mapping[str, int] | None = None,
) -> list[_T]:
    """Return held entries in the admission order of spec section 14.

    1. Classes go in ascending effective rank (:func:`effective_rank`).
    2. Within a class the calls take turns: the k-th entry of a call (in
       ``seq`` order) is in round ``served[call] + k``; rounds go in ascending
       order and, within a round, the calls go by their oldest entry.
    3. Remaining ties go by ``seq`` and the engine receipt order.

    Args:
        entries: Held entries, in any order.
        ticket: Returns the admission data of an entry.
        now: Current ``time.monotonic()``.
        aging_s: Aging step (``priority_aging_s``).
        served: Entries per call that already run in the engine (turns the
            call has had); absent calls count 0.

    Returns:
        The entries in admission order (deterministic for the same input).

    """
    return [
        entry
        for _rank, entry in _ranked_admission_order(
            entries, ticket, now=now, aging_s=aging_s, served=served
        )
    ]


def next_inference_group(
    entries: Iterable[_T],
    ticket: Callable[[_T], AdmissionTicket],
    batch_group: Callable[[_T], str | None],
    *,
    lanes: int,
    now: float,
    aging_s: float,
) -> list[_T]:
    """Choose the entries of the next GPU inference call (spec section 14).

    The first entry in admission order (:func:`admission_order`) decides the
    call: an entry without a batch group (``None``) runs alone; an entry with
    a batch group (the Whisper clip language) is batched with later entries
    of the same group. The batch is filled by walking the admission order
    until it holds *lanes* entries. A same-group entry joins only when no
    entry of a DIFFERENT group (a sequential entry or another language) that
    precedes it in admission order is in a better (lower) effective class:
    a lower class never starts inference while higher-class work of another
    group is waiting, while same-class entries of other groups do not hold
    back the batch.

    Args:
        entries: Ready entries, in any order.
        ticket: Returns the admission data of an entry.
        batch_group: Returns the batch group of an entry (``None``: the entry
            runs alone).
        lanes: Most entries of one batch (the first entry always runs).
        now: Current ``time.monotonic()``.
        aging_s: Aging step (``priority_aging_s``).

    Returns:
        The entries of the call in admission order; empty without entries.

    """
    ranked = _ranked_admission_order(entries, ticket, now=now, aging_s=aging_s)
    if not ranked:
        return []
    first = ranked[0][1]
    group = batch_group(first)
    if group is None:
        return [first]
    chosen = [first]
    # * Best effective rank among the passed-over entries of other groups.
    blocking_rank: int | None = None
    for rank, entry in ranked[1:]:
        if len(chosen) >= lanes:
            break
        if batch_group(entry) != group:
            if blocking_rank is None or rank < blocking_rank:
                blocking_rank = rank
        elif blocking_rank is None or rank <= blocking_rank:
            chosen.append(entry)
    return chosen


@dataclass(frozen=True)
class LaneRotation(Generic[_T]):
    """Lane turns carried from one batch to the next (spec section 14).

    Counts exist only for calls and entries that were active at the last
    deal; :func:`rotate_lanes` drops the others.

    Attributes:
        calls: Lanes dealt per ``(effective rank, call id)``.
        entries: Lanes dealt per ``(effective rank, entry)``.

    """

    calls: Mapping[tuple[int, str], int] = field(default_factory=dict)
    entries: Mapping[tuple[int, _T], int] = field(default_factory=dict)


def _carry_turns(
    keys: Sequence[_K], previous: Mapping[_K, int], into: dict[_K, int]
) -> None:
    """Copy the turn counts of *keys* into *into*.

    A key without a previous count (a newcomer) starts at the lowest previous
    count among *keys* (0 when none has one): it takes its turn in the current
    round, without a catch-up burst and without waiting a full round.
    """
    start = min((previous[key] for key in keys if key in previous), default=0)
    for key in keys:
        into[key] = previous.get(key, start)


def rotate_lanes(
    entries: Iterable[_T],
    ticket: Callable[[_T], AdmissionTicket],
    take: Callable[[_T], bool],
    *,
    lanes: int,
    now: float,
    aging_s: float,
    rotation: LaneRotation[_T],
) -> tuple[list[_T], LaneRotation[_T]]:
    """Deal up to *lanes* batch lanes, continuing *rotation* (spec section 14).

    Classes go in ascending effective rank; a lower class only gets the lanes
    that the higher classes leave free. Within a class every lane goes to the
    call with the fewest lanes dealt (ties: the call with the oldest entry),
    and within that call to its entry with the fewest lanes dealt (ties:
    ``seq`` order). The counts carry over in the returned rotation, so the
    next batch continues the round-robin across calls, and across the jobs of
    one call, where this batch stopped instead of restarting at the oldest
    call and job. A call or entry that joins a class starts at the lowest
    count of its peers there (:func:`_carry_turns`); counts of calls and
    entries that are no longer active in a class are dropped.

    Each turn calls ``take(entry)``: ``True`` means the entry contributed its
    next unit of work (one lane); ``False`` means it has no more work and
    leaves this deal without using the call's turn. A lone entry may take
    every lane.

    Args:
        entries: Active entries, in any order; hashable, and equal across
            deals for the same work.
        ticket: Returns the admission data of an entry.
        take: Contributes the entry's next unit of work to the batch.
        lanes: Batch size.
        now: Current ``time.monotonic()``.
        aging_s: Aging step (``priority_aging_s``).
        rotation: Rotation returned by the previous deal (``LaneRotation()``
            for the first one).

    Returns:
        The entry of every dealt lane in dealing order (an entry appears once
        per lane it took), and the rotation for the next deal.

    """
    classes = _class_calls(entries, ticket, now=now, aging_s=aging_s)
    call_turns: dict[tuple[int, str], int] = {}
    entry_turns: dict[tuple[int, _T], int] = {}
    for rank, calls in classes:
        _carry_turns(
            [(rank, ticket(members[0]).call_id) for members in calls],
            rotation.calls,
            call_turns,
        )
        for members in calls:
            _carry_turns(
                [(rank, entry) for entry in members], rotation.entries, entry_turns
            )
    dealt: list[_T] = []
    for rank, calls in classes:
        # * (call key, position by oldest entry, entries still in this deal).
        open_calls = [
            ((rank, ticket(members[0]).call_id), position, list(members))
            for position, members in enumerate(calls)
        ]
        while open_calls and len(dealt) < lanes:
            *_order, call_index = min(
                (call_turns[key], position, index)
                for index, (key, position, _members) in enumerate(open_calls)
            )
            call_key, _position, members = open_calls[call_index]
            _turns, member_index = min(
                (entry_turns[rank, entry], index) for index, entry in enumerate(members)
            )
            entry = members[member_index]
            if take(entry):
                dealt.append(entry)
                call_turns[call_key] += 1
                entry_turns[rank, entry] += 1
                continue
            del members[member_index]
            if not members:
                del open_calls[call_index]
    return dealt, LaneRotation(calls=call_turns, entries=entry_turns)


def deal_lanes(
    entries: Iterable[_T],
    ticket: Callable[[_T], AdmissionTicket],
    take: Callable[[_T], bool],
    *,
    lanes: int,
    now: float,
    aging_s: float,
) -> list[_T]:
    """Deal up to *lanes* batch lanes from a fresh rotation (spec section 14).

    This is one :func:`rotate_lanes` deal that starts with no carried counts:
    the calls of a class take turns in the order of their oldest entry, and
    the turns of one call rotate over its entries in ``seq`` order.

    Args:
        entries: Active entries, in any order (hashable).
        ticket: Returns the admission data of an entry.
        take: Contributes the entry's next unit of work to the batch.
        lanes: Batch size.
        now: Current ``time.monotonic()``.
        aging_s: Aging step (``priority_aging_s``).

    Returns:
        The entry of every dealt lane, in dealing order (an entry appears once
        per lane it took).

    """
    dealt, _rotation = rotate_lanes(
        entries,
        ticket,
        take,
        lanes=lanes,
        now=now,
        aging_s=aging_s,
        rotation=LaneRotation(),
    )
    return dealt


# * ---------------------------------------------------------------------------
# * Jobs, temp directories and keep-alive
# * ---------------------------------------------------------------------------


@dataclass(eq=False, slots=True)
class _Job:
    """State of one submitted work item inside an engine.

    Ownership: a job is either held (in the engine's queued or ready list,
    ``claimed`` False; :meth:`_EngineBase.cancel` may then finish it at once)
    or claimed by exactly one engine thread, which alone touches its temp
    directory, window iterator and clip.
    """

    item: WorkItem
    ticket: AdmissionTicket
    cancel: threading.Event = field(default_factory=threading.Event)
    claimed: bool = False
    finished: bool = False
    started_at: float | None = None
    temp_dir: Path | None = None
    # * GigaAM GPU batch state.
    wav_path: Path | None = None
    windows: Iterator[WavWindow] | None = None
    lookahead: WavWindow | None = None
    exhausted: bool = False
    read_error: Exception | None = None
    texts: list[str] = field(default_factory=list)
    audio_end_s: float | None = None
    # * Whisper GPU batch state.
    clip: NDArray[np.float32] | None = None
    batch_language: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        """Return ``(job_id, attempt_id)``."""
        return (self.item.job_id, self.item.attempt_id)

    def run_s(self) -> float:
        """Return the wall time since processing started (0 when it did not)."""
        if self.started_at is None:
            return 0.0
        return max(0.0, time.perf_counter() - self.started_at)

    def audio_s(self) -> float | None:
        """Return the item's duration, else the measured audio duration."""
        if self.item.duration_s is not None:
            return self.item.duration_s
        return self.audio_end_s


def _job_ticket(job: _Job) -> AdmissionTicket:
    """Return the admission ticket of *job*."""
    return job.ticket


def _job_batch_language(job: _Job) -> str | None:
    """Return the Whisper batch group of *job* (``None``: sequential path)."""
    return job.batch_language


def _close_windows(job: _Job) -> None:
    """Close the job's window iterator (releases the WAV file handle)."""
    windows, job.windows = job.windows, None
    job.lookahead = None
    close = getattr(windows, "close", None)
    if callable(close):
        with contextlib.suppress(Exception):
            close()


def _touch(path: Path) -> None:
    """Create or refresh *path* (best effort)."""
    try:
        path.touch(exist_ok=True)
    except OSError as exc:
        logger.debug("cannot touch keep-alive file %s: %s", path, exc)


def _remove_tree(path: Path) -> None:
    """Remove a job directory, retrying briefly on transient Windows locks."""
    for attempt in range(_TEMP_REMOVE_ATTEMPTS):
        shutil.rmtree(path, ignore_errors=True)
        if not path.exists():
            return
        if attempt + 1 < _TEMP_REMOVE_ATTEMPTS:
            time.sleep(_TEMP_REMOVE_RETRY_S)
    logger.warning("job temp directory remains after removal attempts: %s", path)


class _KeepAlive:
    """Periodically touches a keep-alive file in every active job directory."""

    def __init__(self, interval_s: float) -> None:
        self._interval_s = interval_s
        self._dirs: set[Path] = set()
        # * Held during every touch, so a discarded directory is never touched
        # * again (a late touch would re-create a file in a directory that is
        # * being removed).
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def add(self, directory: Path) -> None:
        """Register a job directory and create its keep-alive file."""
        with self._lock:
            _touch(directory / KEEPALIVE_FILE_NAME)
            self._dirs.add(directory)

    def discard(self, directory: Path) -> None:
        """Unregister a job directory (before it is removed)."""
        with self._lock:
            self._dirs.discard(directory)

    def touch_all(self) -> None:
        """Refresh the keep-alive file of every registered directory."""
        with self._lock:
            for directory in self._dirs:
                _touch(directory / KEEPALIVE_FILE_NAME)

    def start(self) -> None:
        """Start the keep-alive thread (once)."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="stt-keepalive", daemon=True
        )
        self._thread.start()

    def stop(self, timeout_s: float) -> None:
        """Stop the keep-alive thread (bounded)."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout_s)

    def _run(self) -> None:
        """Touch the registered directories every interval until stopped."""
        while not self._stop.wait(self._interval_s):
            self.touch_all()


# * ---------------------------------------------------------------------------
# * Options and pipeline construction
# * ---------------------------------------------------------------------------


def _positive_int(value: object, name: str) -> int | None:
    """Validate an optional positive integer engine option.

    Raises:
        ValueError: When *value* is present but not an integer >= 1.

    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        msg = f"engine option {name!r} must be a positive integer, got {value!r}"
        raise ValueError(msg)
    return value


def _positive_seconds(
    options: Mapping[str, object], name: str, default: float
) -> float:
    """Return a positive finite number option, or *default* when it is absent.

    Raises:
        ValueError: When the option is present but not a positive finite number.

    """
    value = options.get(name)
    if value is None:
        return default
    number = math.nan
    if not isinstance(value, bool) and isinstance(value, int | float):
        with contextlib.suppress(OverflowError):
            number = float(value)
    if not (math.isfinite(number) and number > 0):
        msg = f"engine option {name!r} must be a positive number, got {value!r}"
        raise ValueError(msg)
    return number


def _batch_lanes(spec: WorkerSpec) -> int:
    """Return the batch size of a GPU engine.

    ``options["batch_lanes"]`` is either the lane count or a mapping per
    provider family; without it the worker capacity is the batch size.

    Raises:
        ValueError: When the option is not a positive integer.

    """
    raw: object = spec.options.get("batch_lanes")
    if isinstance(raw, Mapping):
        raw = raw.get(provider_family(spec.key.provider))
    lanes = _positive_int(raw, "batch_lanes")
    return spec.capacity if lanes is None else lanes


def _new_pipeline(
    spec: WorkerSpec,
    *,
    compute_type: str,
    cpu_threads: int | None = None,
    num_workers: int | None = None,
    torch_threads: int | None = None,
    gigaam_threadsafe: bool = False,
    gigaam_warmup_batch: int = 1,
) -> LocalPipeline:
    """Construct the engine's strict-device :class:`LocalPipeline` (no load).

    Args:
        spec: Worker specification (provider, model and device come from it).
        compute_type: Whisper compute type (``auto`` for GigaAM).
        cpu_threads: Whisper ``cpu_threads``.
        num_workers: Whisper ``num_workers``.
        torch_threads: GigaAM ``torch.set_num_threads`` value.
        gigaam_threadsafe: GigaAM lock-free inference path with warm-up.
        gigaam_warmup_batch: GigaAM CUDA warm-up batch size.

    Returns:
        The unloaded pipeline.

    """
    # ! The pipeline module imports torch and faster-whisper.
    from core import pipelines  # noqa: PLC0415 - heavy ML import inside the worker only

    key = spec.key
    whisper_model = DEFAULT_WHISPER_MODEL
    if provider_family(key.provider) == "whisper":
        # * The key is what the supervisor accounts for and reports.
        whisper_model = key.model
        requested = spec.options.get("whisper_model")
        if isinstance(requested, str) and requested.strip():
            identity = model_identity(key.provider, requested)
            if identity != key.model:
                logger.warning(
                    "spec option whisper_model=%r differs from the worker key "
                    "model %r; the key model is loaded",
                    requested,
                    key.model,
                )
    return pipelines.LocalPipeline(
        whisper_model=whisper_model,
        engine="whisperx",
        stt_provider=key.provider,
        enable_diarization=False,
        enable_dialog_blocks=False,
        language=None,
        device=spec.device,
        compute_type=compute_type,
        strict_device=True,
        cpu_threads=cpu_threads,
        num_workers=num_workers,
        torch_threads=torch_threads,
        gigaam_threadsafe=gigaam_threadsafe,
        gigaam_warmup_batch=gigaam_warmup_batch,
    )


def _active_device(pipeline: LocalPipeline) -> str | None:
    """Return the device the pipeline's STT model was loaded on."""
    gigaam = pipeline.gigaam
    device: object = (
        gigaam.active_device
        if gigaam is not None
        else pipeline.require_whisperx().active_device
    )
    return None if device is None else str(device)


def _close_pipeline(pipeline: LocalPipeline) -> None:
    """Unload the pipeline's models (best effort)."""
    try:
        pipeline.close(aggressive=True)
    except Exception:
        logger.exception("pipeline close failed")


def _probe_duration(path: Path) -> float | None:
    """Return the media duration from ffprobe, or ``None`` when it fails."""
    from core.ffmpeg import (  # noqa: PLC0415 - ffmpeg-python only inside the worker
        MediaProbeError,
        probe_media_duration,
    )

    try:
        return probe_media_duration(path, timeout_s=DURATION_PROBE_TIMEOUT_S)
    except MediaProbeError as exc:
        logger.info("duration probe failed for %s: %s", path, exc)
        return None


def _join_all(threads: Sequence[threading.Thread], timeout_s: float) -> list[str]:
    """Join *threads* within one shared deadline.

    Returns:
        Names of the threads that are still alive.

    """
    deadline = time.monotonic() + timeout_s
    alive: list[str] = []
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
        if thread.is_alive():
            alive.append(thread.name)
    return alive


# * ---------------------------------------------------------------------------
# * Engines
# * ---------------------------------------------------------------------------


class _EngineBase(abc.ABC):
    """Shared job bookkeeping, ordering, cancellation, emission and shutdown.

    Locks: ``_cond`` (a re-entrant condition) guards the job registry and the
    held lists and is never held while a result is emitted or a directory is
    removed; ``_emit_lock`` serializes emission against :meth:`close`;
    ``_close_lock`` serializes :meth:`start` and :meth:`close`.
    """

    def __init__(self, spec: WorkerSpec) -> None:
        self._spec = spec
        self._backend: Backend = spec.key.backend
        self._gigaam = provider_family(spec.key.provider) == "gigaam"
        self._aging_s = _positive_seconds(
            spec.options, "priority_aging_s", DEFAULT_PRIORITY_AGING_S
        )
        self._cond = threading.Condition()
        self._jobs: dict[tuple[str, str], _Job] = {}
        # * Held jobs waiting for a slot / prep thread, in arrival order.
        self._queued: list[_Job] = []
        # * Held (unclaimed) or loop-claimed jobs ready for GPU inference.
        self._ready: list[_Job] = []
        self._receipts = itertools.count()
        self._closing = False
        self._emit: EmitFn | None = None
        self._emit_closed = False
        self._emit_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed = False
        self._threads: list[threading.Thread] = []
        self._pipeline: LocalPipeline | None = None
        self._load_info: LoadInfo | None = None
        self._keepalive = _KeepAlive(KEEPALIVE_INTERVAL_S)

    # * ---- engine-specific hooks --------------------------------------------

    @abc.abstractmethod
    def _build_pipeline(self) -> LocalPipeline:
        """Construct the engine's pipeline (not loaded yet)."""

    @abc.abstractmethod
    def _thread_targets(self) -> list[tuple[str, Callable[[], None]]]:
        """Return ``(thread name, target)`` of every engine thread."""

    def _on_loaded(self, pipeline: LocalPipeline) -> None:  # noqa: B027 - optional hook
        """Keep references the engine needs after a successful load."""

    def _drop_model_refs(self) -> None:  # noqa: B027 - optional hook
        """Drop the engine's own model references before the pipeline unloads."""

    def _details(self) -> dict[str, object]:
        """Return engine-specific ``loaded`` details."""
        return {}

    # * ---- PoolEngine protocol -----------------------------------------------

    def load(self) -> LoadInfo:
        """Load the model on the strict device (GigaAM also warms up).

        Returns:
            Load information for the ``loaded`` message.

        Raises:
            DeviceUnavailableError: The device cannot be used, or the model
                ended up on another device than the spec requires.
            CudaOutOfMemoryError: CUDA ran out of memory while loading.

        """
        if self._load_info is not None:
            return self._load_info
        started = time.perf_counter()
        pipeline = self._build_pipeline()
        try:
            pipeline.ensure_stt_loaded()
            active = self._verified_device(pipeline)
            self._on_loaded(pipeline)
        except BaseException:
            _close_pipeline(pipeline)
            raise
        self._pipeline = pipeline
        spec = self._spec
        details: dict[str, object] = {
            "engine": type(self).__name__,
            "provider": spec.key.provider,
            "model": spec.key.model,
            "compute_type": spec.key.compute_type,
            "backend": spec.key.backend,
            "active_device": active,
            "capacity": spec.capacity,
            "threads_per_slot": spec.threads_per_slot,
            "host_threads": spec.host_threads,
            "priority_aging_s": self._aging_s,
            **self._details(),
        }
        self._load_info = LoadInfo(
            device=spec.device,
            load_s=time.perf_counter() - started,
            details=details,
        )
        logger.info(
            "%s loaded %s on %s in %.2f s",
            type(self).__name__,
            spec.key.label,
            spec.device,
            self._load_info.load_s,
        )
        return self._load_info

    def start(self, emit: EmitFn) -> None:
        """Start the engine threads and the keep-alive thread.

        Args:
            emit: Receives every result (thread-safe).

        Raises:
            RuntimeError: When :meth:`load` has not succeeded.

        """
        with self._close_lock:
            if self._closed:
                logger.warning("start() after close() is ignored")
                return
            if self._threads:
                logger.warning("start() called twice; ignored")
                return
            if self._pipeline is None:
                msg = "load() must succeed before start()"
                raise RuntimeError(msg)
            with self._emit_lock:
                self._emit = emit
            self._keepalive.start()
            for name, target in self._thread_targets():
                thread = threading.Thread(
                    target=self._thread_main,
                    args=(name, target),
                    name=name,
                    daemon=True,
                )
                self._threads.append(thread)
                thread.start()

    def submit(self, item: WorkItem) -> None:
        """Hold one work item for the engine threads (never blocks long).

        Args:
            item: Work item; a duplicate ``(job_id, attempt_id)`` is ignored.

        """
        with self._cond:
            if self._closing:
                logger.warning(
                    "engine is closing; job %s attempt %s is not run",
                    item.job_id,
                    item.attempt_id,
                )
                return
            key = (item.job_id, item.attempt_id)
            if key in self._jobs:
                logger.warning("duplicate submit of %s attempt %s ignored", *key)
                return
            job = _Job(
                item=item,
                ticket=AdmissionTicket(
                    priority=item.priority,
                    call_id=item.call_id,
                    seq=item.seq,
                    received_at=time.monotonic(),
                    receipt=next(self._receipts),
                ),
            )
            self._jobs[key] = job
            self._queued.append(job)
            self._cond.notify_all()
        logger.debug("job %s attempt %s held", item.job_id, item.attempt_id)

    def cancel(self, job_id: str, attempt_id: str) -> None:
        """Cancel a held item at once or flag a running one (cooperative).

        A held item (queued, or prepared and not yet taken by the inference
        thread) is answered ``cancelled`` immediately; a running item stops at
        its next window, segment or batch boundary and is answered then.

        Args:
            job_id: Job identifier.
            attempt_id: Attempt identifier.

        """
        with self._cond:
            job = self._jobs.get((job_id, attempt_id))
            if job is None or job.finished:
                logger.debug("cancel of unknown job %s attempt %s", job_id, attempt_id)
                return
            job.cancel.set()
            self._cond.notify_all()
            if job.claimed:
                return
            # * Held: this thread takes ownership and answers at once.
            if job in self._queued:
                self._queued.remove(job)
            if job in self._ready:
                self._ready.remove(job)
            job.claimed = True
        self._retire(job, self._outcome(job))

    def close(self) -> None:
        """Stop accepting, stop the threads (bounded) and unload; idempotent.

        Emission stops as soon as :meth:`close` starts; unfinished items get
        no result. When an engine thread does not stop within
        :data:`CLOSE_JOIN_TIMEOUT_S`, the model is left loaded (unloading
        under a running inference call is unsafe) and is released when the
        process exits.
        """
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            with self._emit_lock:
                self._emit_closed = True
            with self._cond:
                self._closing = True
                for job in self._jobs.values():
                    job.cancel.set()
                self._cond.notify_all()
            alive = _join_all(self._threads, CLOSE_JOIN_TIMEOUT_S)
            self._keepalive.stop(_KEEPALIVE_STOP_TIMEOUT_S)
            if alive:
                logger.warning(
                    "engine threads still running after %.0f s: %s; the model "
                    "stays loaded until the process exits",
                    CLOSE_JOIN_TIMEOUT_S,
                    ", ".join(alive),
                )
                return
            self._release_leftovers()
            self._drop_model_refs()
            pipeline, self._pipeline = self._pipeline, None
            if pipeline is not None:
                _close_pipeline(pipeline)
            logger.info("%s closed", type(self).__name__)

    # * ---- shared machinery ---------------------------------------------------

    def _verified_device(self, pipeline: LocalPipeline) -> str:
        """Return the model's device after checking it is the strict spec device.

        Raises:
            DeviceUnavailableError: When the model is on another device.

        """
        active = _active_device(pipeline)
        if active != self._spec.device:
            msg = (
                f"{self._spec.key.label}: the model loaded on {active!r} "
                f"instead of the strict device {self._spec.device!r}"
            )
            raise DeviceUnavailableError(msg)
        return active

    @staticmethod
    def _thread_main(name: str, target: Callable[[], None]) -> None:
        """Run an engine thread and log an unexpected crash."""
        try:
            target()
        except Exception:
            logger.exception("engine thread %s crashed", name)

    def _require_pipeline(self) -> LocalPipeline:
        """Return the loaded pipeline.

        Raises:
            RuntimeError: When the engine is not loaded (or already closed).

        """
        pipeline = self._pipeline
        if pipeline is None:
            msg = "engine is not loaded"
            raise RuntimeError(msg)
        return pipeline

    def _should_stop(self, job: _Job) -> bool:
        """Return whether *job* must stop (cancelled or the engine closes)."""
        return self._closing or job.cancel.is_set()

    def _take_queued(self) -> _Job | None:
        """Claim the next queued job in admission order (waits; ``None`` = stop)."""
        with self._cond:
            while not self._closing and not self._queued:
                self._cond.wait()
            if self._closing:
                return None
            served = Counter(
                job.ticket.call_id
                for job in self._jobs.values()
                if job.started_at is not None
            )
            job = admission_order(
                self._queued,
                _job_ticket,
                now=time.monotonic(),
                aging_s=self._aging_s,
                served=served,
            )[0]
            self._queued.remove(job)
            job.claimed = True
            job.started_at = time.perf_counter()
        logger.info(
            "job %s attempt %s started (priority %s, call %s, seq %s)",
            job.item.job_id,
            job.item.attempt_id,
            job.item.priority,
            job.item.call_id or "-",
            job.item.seq,
        )
        return job

    def _release_ready(self, job: _Job) -> None:
        """Hand a prepared job to the inference thread (or finish it if cancelled)."""
        with self._cond:
            if not self._should_stop(job):
                job.claimed = False
                self._ready.append(job)
                self._cond.notify_all()
                return
        self._retire(job, self._outcome(job))

    def _open_temp_dir(self, job: _Job) -> Path:
        """Create the job's ``askvlm-pool-`` temp directory with a keep-alive file."""
        path = Path(tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX))
        job.temp_dir = path
        self._keepalive.add(path)
        return path

    def _close_temp_dir(self, job: _Job) -> None:
        """Remove the job's temp directory (best effort)."""
        path, job.temp_dir = job.temp_dir, None
        if path is None:
            return
        self._keepalive.discard(path)
        _remove_tree(path)

    def _outcome(
        self, job: _Job, *, text: str | None = None, error: Exception | None = None
    ) -> EngineResult:
        """Build the result of *job*: ``cancelled`` wins, then the error, then text."""
        run_s = job.run_s()
        if job.cancel.is_set():
            return EngineResult.for_cancel(job.item, backend=self._backend, run_s=run_s)
        if error is not None:
            logger.warning(
                "job %s attempt %s failed: %s: %s",
                job.item.job_id,
                job.item.attempt_id,
                type(error).__name__,
                error,
                exc_info=error,
            )
            return EngineResult.for_error(
                job.item,
                error,
                backend=self._backend,
                run_s=run_s,
                audio_s=job.audio_s(),
            )
        return EngineResult.for_text(
            job.item,
            text or "",
            backend=self._backend,
            run_s=run_s,
            audio_s=job.audio_s(),
        )

    def _retire(self, job: _Job, result: EngineResult) -> None:
        """Release the job's resources and emit its result (once)."""
        with self._cond:
            if job in self._ready:
                self._ready.remove(job)
        _close_windows(job)
        job.clip = None
        self._close_temp_dir(job)
        self._finish(job, result)

    def _finish(self, job: _Job, result: EngineResult) -> None:
        """Mark *job* finished and emit *result* unless it already finished."""
        with self._cond:
            if job.finished:
                return
            job.finished = True
            if self._jobs.get(job.key) is job:
                del self._jobs[job.key]
        logger.info(
            "job %s attempt %s finished: %s in %.2f s",
            job.item.job_id,
            job.item.attempt_id,
            result.status,
            result.run_s,
        )
        self._deliver(result)

    def _deliver(self, result: EngineResult) -> None:
        """Emit *result* unless :meth:`close` has started."""
        with self._emit_lock:
            emit = None if self._emit_closed else self._emit
            if emit is None:
                logger.debug(
                    "result of job %s attempt %s dropped (engine closed or not started)",
                    result.job_id,
                    result.attempt_id,
                )
                return
            try:
                emit(result)
            except Exception:
                logger.exception("cannot emit the result of job %s", result.job_id)

    def _guarded(self, job: _Job, step: Callable[[_Job], None]) -> None:
        """Run one loop step for a job that the calling thread owns.

        The steps answer their own per-job failures. An exception that still
        escapes (for example from the release of the job's resources) is
        logged and answered with an ``error`` result unless the job already
        finished, so the calling engine thread always survives to take the
        next job. Exceptions that are not :class:`Exception` propagate.

        Args:
            job: Claimed job of the calling thread.
            step: Handles *job*.

        """
        try:
            step(job)
        except Exception as exc:
            logger.exception(
                "unexpected failure while handling job %s attempt %s",
                job.item.job_id,
                job.item.attempt_id,
            )
            self._fail_unfinished((job,), exc)

    def _fail_unfinished(self, jobs: Iterable[_Job], exc: Exception) -> None:
        """Answer every unfinished job of *jobs* with an ``error`` result.

        One job whose answer fails does not keep the others from theirs, and
        nothing is raised.

        Args:
            jobs: Jobs owned by the calling thread.
            exc: Failure that ended their processing.

        """
        for job in jobs:
            try:
                self._answer_failed(job, exc)
            except Exception:
                logger.exception(
                    "cannot answer job %s attempt %s",
                    job.item.job_id,
                    job.item.attempt_id,
                )

    def _answer_failed(self, job: _Job, exc: Exception) -> None:
        """Release what is left of *job* and answer it, even when releasing fails.

        The result is built by :meth:`_outcome` (a requested cancel still
        wins). It is emitted once: :meth:`_finish` ignores a finished job and
        emits nothing after :meth:`close` started.
        """
        if job.finished:
            return
        result = self._outcome(job, error=exc)
        try:
            self._retire(job, result)
        except Exception:
            logger.exception(
                "cannot release the resources of job %s attempt %s",
                job.item.job_id,
                job.item.attempt_id,
            )
        self._finish(job, result)

    def _release_leftovers(self) -> None:
        """Release the resources of unfinished jobs after every thread stopped."""
        with self._cond:
            jobs = list(self._jobs.values())
            self._jobs.clear()
            self._queued.clear()
            self._ready.clear()
            for job in jobs:
                job.finished = True
        for job in jobs:
            _close_windows(job)
            job.clip = None
            self._close_temp_dir(job)

    def _run_pipeline_job(self, job: _Job) -> None:
        """Run one claimed job through ``pipeline.process`` and emit its result.

        The text is ``Document.get_full_text().strip()``, exactly like the
        one-shot ``external-transcribe`` path.
        """
        if self._should_stop(job):
            self._retire(job, self._outcome(job))
            return
        text: str | None = None
        error: Exception | None = None
        try:
            pipeline = self._require_pipeline()
            work_dir = self._open_temp_dir(job)
            document = pipeline.process(
                job.item.input_path,
                work_dir,
                should_cancel=lambda: self._should_stop(job),
                language=job.item.language,
            )
            text = document.get_full_text().strip()
            if self._gigaam and document.segments:
                # * GigaAM segments are the 20 s windows: the last one ends at
                # * the exact audio duration.
                job.audio_end_s = float(document.segments[-1].end_time)
        except Exception as exc:  # noqa: BLE001 - a failed job becomes an error result
            error = exc
        self._retire(job, self._outcome(job, text=text, error=error))


class CpuPipelineEngine(_EngineBase):
    """Any provider on CPU: ``capacity`` slot threads over one shared pipeline.

    Whisper uses ``cpu_threads=threads_per_slot`` and ``num_workers=capacity``
    (CTranslate2 replicas over shared weights); GigaAM uses
    ``torch_threads=threads_per_slot`` and the lock-free thread-safe path
    (warmed up single-threaded in :meth:`load`). Each slot takes the next held
    item in admission order and runs ``pipeline.process`` with the item's own
    language and temp directory; a cancel stops it at the next
    segment/window boundary (or during the ffmpeg preparation).
    """

    def _build_pipeline(self) -> LocalPipeline:
        spec = self._spec
        if self._gigaam:
            return _new_pipeline(
                spec,
                compute_type="auto",
                torch_threads=spec.threads_per_slot,
                gigaam_threadsafe=True,
            )
        return _new_pipeline(
            spec,
            compute_type=spec.key.compute_type,
            cpu_threads=spec.threads_per_slot,
            num_workers=spec.capacity,
        )

    def _details(self) -> dict[str, object]:
        return {"slots": self._spec.capacity}

    def _thread_targets(self) -> list[tuple[str, Callable[[], None]]]:
        return [
            (f"stt-slot-{index}", self._slot_loop)
            for index in range(self._spec.capacity)
        ]

    def _slot_loop(self) -> None:
        """Run held items one at a time until the engine closes."""
        while (job := self._take_queued()) is not None:
            self._guarded(job, self._run_pipeline_job)


class GigaamGpuBatchEngine(_EngineBase):
    """``gigaam-e2e-ctc`` on CUDA with cross-job window batches.

    Threads: up to :data:`MAX_PREP_THREADS` prep threads run ``prepare_audio``
    (ffmpeg, no torch) into each job's temp directory; ONE batch-loop thread
    owns every prepared job, streams its 20 s windows lazily
    (:func:`~core.gigaam_ctc_wrapper.iter_wav_windows`), deals up to ``lanes``
    windows per batch with :func:`rotate_lanes` and runs
    ``transcribe_window_batch``. The lane rotation carries over from batch to
    batch, so consecutive batches keep rotating across calls and across the
    jobs of one call (spec section 14). GigaAM never runs on CUDA from two
    threads.

    A job's text is its window texts joined with newlines and stripped (one
    segment per window, empty windows kept), the structure of
    ``Document.get_full_text()`` on the default path. A failed batch fails
    only the jobs that had windows in it; a cancel drops the job's remaining
    windows at the next batch boundary.
    """

    def __init__(self, spec: WorkerSpec) -> None:
        super().__init__(spec)
        self._lanes = _batch_lanes(spec)
        self._prep_threads = max(1, min(MAX_PREP_THREADS, spec.capacity))
        self._wrapper: Any = None
        # * Lane rotation between batches; only the batch-loop thread uses it.
        self._rotation: LaneRotation[_Job] = LaneRotation()

    def _build_pipeline(self) -> LocalPipeline:
        host_threads = self._spec.host_threads
        return _new_pipeline(
            self._spec,
            compute_type="auto",
            torch_threads=host_threads if host_threads > 0 else None,
            gigaam_threadsafe=True,
            gigaam_warmup_batch=self._lanes,
        )

    def _on_loaded(self, pipeline: LocalPipeline) -> None:
        if pipeline.gigaam is None:
            msg = "GigaAM backend is not configured"
            raise RuntimeError(msg)
        self._wrapper = pipeline.gigaam

    def _drop_model_refs(self) -> None:
        self._wrapper = None

    def _details(self) -> dict[str, object]:
        return {"lanes": self._lanes, "prep_threads": self._prep_threads}

    def _thread_targets(self) -> list[tuple[str, Callable[[], None]]]:
        preps: list[tuple[str, Callable[[], None]]] = [
            (f"stt-prep-{index}", self._prep_loop)
            for index in range(self._prep_threads)
        ]
        return [*preps, ("stt-batch-loop", self._batch_loop)]

    def _prep_loop(self) -> None:
        """Prepare held items in admission order until the engine closes."""
        while (job := self._take_queued()) is not None:
            self._guarded(job, self._prepare)

    def _prepare(self, job: _Job) -> None:
        """Convert the job's media to a 16 kHz mono WAV in its temp directory."""
        from core.audio_io import prepare_audio  # noqa: PLC0415 - worker only

        try:
            work_dir = self._open_temp_dir(job)
            job.wav_path = prepare_audio(
                job.item.input_path,
                work_dir,
                should_cancel=lambda: self._should_stop(job),
            )
        except Exception as exc:  # noqa: BLE001 - a failed prep fails only its job
            self._retire(job, self._outcome(job, error=exc))
            return
        self._release_ready(job)

    def _claim_active(self) -> list[_Job] | None:
        """Wait for prepared jobs and claim them all (``None`` = stop)."""
        with self._cond:
            while not self._closing and not self._ready:
                self._cond.wait()
            if self._closing:
                return None
            for job in self._ready:
                job.claimed = True
            return list(self._ready)

    def _batch_loop(self) -> None:
        """Run window batches over the active jobs until the engine closes."""
        while (active := self._claim_active()) is not None:
            try:
                self._batch_step(active)
            except Exception as exc:
                logger.exception("GigaAM batch step failed")
                self._fail_unfinished(active, exc)

    def _peek(self, job: _Job) -> bool:
        """Read the job's next window into ``lookahead`` (never finishes the job).

        The end of the stream sets ``exhausted`` and a read failure sets
        ``read_error``; :meth:`_settle` acts on both after the batch that may
        still hold earlier windows of the job.

        Returns:
            ``True`` when ``job.lookahead`` holds a window.

        """
        if job.lookahead is not None:
            return True
        if job.exhausted or job.read_error is not None:
            return False
        from core.gigaam_ctc_wrapper import iter_wav_windows  # noqa: PLC0415 - lazy

        try:
            if job.windows is None:
                if job.wav_path is None:
                    msg = "prepared WAV is missing"
                    raise RuntimeError(msg)  # noqa: TRY301 - reported as this job's error
                job.windows = iter_wav_windows(job.wav_path)
            job.lookahead = next(job.windows)
        except StopIteration:
            job.exhausted = True
            _close_windows(job)
            return False
        except Exception as exc:  # noqa: BLE001 - a broken WAV fails only its job
            job.read_error = exc
            _close_windows(job)
            return False
        return True

    def _settle(self, job: _Job) -> None:
        """Finish a job that is cancelled, broken or out of windows.

        Called only when none of the job's windows is pending, so the result
        contains every window text. The read-ahead also detects the end of a
        job that got no lane in this batch, so its result is not delayed by
        higher classes taking every lane.
        """
        if job.finished:
            return
        if self._should_stop(job):
            self._retire(job, self._outcome(job))
            return
        if self._peek(job):
            return
        if job.read_error is not None:
            self._retire(job, self._outcome(job, error=job.read_error))
            return
        self._retire(job, self._outcome(job, text="\n".join(job.texts)))

    def _batch_step(self, active: Sequence[_Job]) -> None:
        """Deal one batch over the active jobs, run it, then settle every job."""
        live: list[_Job] = []
        for job in active:
            if self._should_stop(job):
                self._retire(job, self._outcome(job))
            else:
                live.append(job)
        batch: list[tuple[_Job, WavWindow]] = []

        def take(job: _Job) -> bool:
            if not self._peek(job):
                return False
            window = job.lookahead
            if window is None:
                return False
            job.lookahead = None
            batch.append((job, window))
            return True

        _dealt, self._rotation = rotate_lanes(
            live,
            _job_ticket,
            take,
            lanes=self._lanes,
            now=time.monotonic(),
            aging_s=self._aging_s,
            rotation=self._rotation,
        )
        if batch:
            self._run_batch(batch)
        for job in live:
            self._settle(job)
        if all(job.finished for job in live):
            # * Nothing carries over; an idle engine keeps no finished jobs.
            self._rotation = LaneRotation()

    def _run_batch(self, batch: Sequence[tuple[_Job, WavWindow]]) -> None:
        """Transcribe one batch and append each window text to its job."""
        texts: list[str] = []
        error: Exception | None = None
        try:
            texts = list(
                self._wrapper.transcribe_window_batch(
                    [window.samples for _job, window in batch]
                )
            )
        except Exception as exc:  # noqa: BLE001 - a failed batch fails only its jobs
            error = exc
        if error is None and len(texts) != len(batch):
            error = RuntimeError(
                f"GigaAM batch returned {len(texts)} texts for {len(batch)} windows"
            )
        if error is not None:
            for job in dict.fromkeys(job for job, _window in batch):
                self._retire(job, self._outcome(job, error=error))
            return
        for (job, window), text in zip(batch, texts, strict=True):
            job.texts.append(text)
            job.audio_end_s = window.end_s


class WhisperGpuEngine(_EngineBase):
    """Whisper on CUDA: cross-file clip batches plus the sequential path.

    Threads: up to :data:`MAX_PREP_THREADS` prep threads route each item: an
    item with an explicit language whose duration (``duration_s``, else
    ffprobe) and decoded length are at most ``whisper_batch_max_clip_s`` is
    decoded (:func:`~core.whisper_batch.load_audio_16k`), checked by the
    Silero VAD (no speech -> ``empty`` without inference) and trimmed to its
    speech span; every other item takes the sequential path. ONE inference
    thread takes the first prepared item in admission order: a batch item
    brings up to ``lanes`` prepared batch items of the same language (walking
    the admission order) into one
    :func:`~core.whisper_batch.transcribe_clips_batched` call; a sequential
    item runs ``pipeline.process`` (beam 10, VAD, word timestamps) exactly like
    the one-shot GPU path. A same-language item joins the batch only when no
    prepared item of another language or of the sequential path that precedes
    it in admission order is in a better effective class, so a lower class
    never starts inference while higher-class work waits (spec section 14,
    :func:`next_inference_group`).
    """

    def __init__(self, spec: WorkerSpec) -> None:
        super().__init__(spec)
        self._lanes = _batch_lanes(spec)
        self._prep_threads = max(1, min(MAX_PREP_THREADS, spec.capacity))
        self._max_clip_s = min(
            _positive_seconds(spec.options, "whisper_batch_max_clip_s", MAX_CLIP_S),
            MAX_CLIP_S,
        )
        self._model: Any = None

    def _build_pipeline(self) -> LocalPipeline:
        return _new_pipeline(self._spec, compute_type=self._spec.key.compute_type)

    def _on_loaded(self, pipeline: LocalPipeline) -> None:
        self._model = pipeline.require_whisperx().loaded_model()

    def _drop_model_refs(self) -> None:
        self._model = None

    def _details(self) -> dict[str, object]:
        return {
            "lanes": self._lanes,
            "prep_threads": self._prep_threads,
            "batch_max_clip_s": self._max_clip_s,
        }

    def _thread_targets(self) -> list[tuple[str, Callable[[], None]]]:
        preps: list[tuple[str, Callable[[], None]]] = [
            (f"stt-prep-{index}", self._prep_loop)
            for index in range(self._prep_threads)
        ]
        return [*preps, ("stt-infer-loop", self._infer_loop)]

    def _prep_loop(self) -> None:
        """Route held items in admission order and hand them to the inference thread."""
        while (job := self._take_queued()) is not None:
            self._guarded(job, self._prepare)

    def _prepare(self, job: _Job) -> None:
        """Route one claimed item, then answer it (``empty``) or hand it on."""
        try:
            route = self._route(job)
        except Exception as exc:  # noqa: BLE001 - the sequential path reports its own error
            # * A clip that cannot be decoded or checked here still gets
            # * the one-shot path (and its usual error text).
            logger.warning(
                "batch preparation of job %s failed (%s: %s); using the "
                "sequential path",
                job.item.job_id,
                type(exc).__name__,
                exc,
            )
            job.clip = None
            job.batch_language = None
            route = "sequential"
        if route == "empty":
            self._retire(job, self._outcome(job, text=""))
            return
        self._release_ready(job)

    def _route(self, job: _Job) -> WhisperRoute:
        """Choose the job's path; batch candidates get their trimmed clip."""
        language = job.item.language
        if language is None:
            return "sequential"
        duration = job.item.duration_s
        if duration is None:
            duration = _probe_duration(job.item.input_path)
        if duration is None or duration > self._max_clip_s or self._should_stop(job):
            return "sequential"
        audio = load_audio_16k(job.item.input_path)
        samples = int(audio.shape[0])
        job.audio_end_s = samples / WHISPER_SAMPLE_RATE
        if samples > self._max_clip_s * WHISPER_SAMPLE_RATE:
            return "sequential"
        span = speech_span(audio, sample_rate=WHISPER_SAMPLE_RATE)
        if span is None:
            return "empty"
        begin = max(0, math.floor(span[0] * WHISPER_SAMPLE_RATE))
        end = min(samples, math.ceil(span[1] * WHISPER_SAMPLE_RATE))
        if end <= begin:
            return "empty"
        job.clip = audio[begin:end]
        job.batch_language = language
        return "batch"

    def _claim_group(self) -> list[_Job] | None:
        """Claim the next inference group in admission order (``None`` = stop)."""
        with self._cond:
            while not self._closing and not self._ready:
                self._cond.wait()
            if self._closing:
                return None
            group = next_inference_group(
                self._ready,
                _job_ticket,
                _job_batch_language,
                lanes=self._lanes,
                now=time.monotonic(),
                aging_s=self._aging_s,
            )
            for job in group:
                self._ready.remove(job)
                job.claimed = True
            return group

    def _infer_loop(self) -> None:
        """Run batches and sequential jobs until the engine closes."""
        while (group := self._claim_group()) is not None:
            try:
                if group[0].batch_language is None:
                    self._run_pipeline_job(group[0])
                else:
                    self._run_clip_batch(group)
            except Exception as exc:
                logger.exception("Whisper inference step failed")
                self._fail_unfinished(group, exc)

    def _run_clip_batch(self, group: Sequence[_Job]) -> None:
        """Decode a group of same-language clips in one batched call."""
        live: list[tuple[_Job, NDArray[np.float32]]] = []
        for job in group:
            clip = job.clip
            if self._should_stop(job) or clip is None:
                self._retire(job, self._outcome(job))
            else:
                live.append((job, clip))
        if not live:
            return
        language = live[0][0].batch_language or ""
        texts: list[str] = []
        error: Exception | None = None
        try:
            texts = transcribe_clips_batched(
                self._model,
                [clip for _job, clip in live],
                language=language,
                beam_size=WHISPER_BEAM_SIZE,
                batch_size=self._lanes,
            )
        except Exception as exc:  # noqa: BLE001 - a failed batch fails only its jobs
            error = exc
        if error is None and len(texts) != len(live):
            error = RuntimeError(
                f"Whisper batch returned {len(texts)} texts for {len(live)} clips"
            )
        if error is not None:
            for job, _clip in live:
                self._retire(job, self._outcome(job, error=error))
            return
        for (job, _clip), text in zip(live, texts, strict=True):
            self._retire(job, self._outcome(job, text=text))


def create_engine(spec: WorkerSpec) -> PoolEngine:
    """Build the engine for a worker spec (nothing is loaded here).

    Args:
        spec: Validated worker specification.

    Returns:
        :class:`CpuPipelineEngine` for the CPU backend,
        :class:`GigaamGpuBatchEngine` for ``gigaam-e2e-ctc`` on the GPU, or
        :class:`WhisperGpuEngine` for Whisper on the GPU.

    Raises:
        ValueError: An unsupported provider/backend pair, a device that does
            not match the backend, or an invalid engine option.

    """
    key = spec.key
    expected_device = device_for_backend(key.backend)
    if spec.device != expected_device:
        msg = (
            f"device {spec.device!r} does not match backend {key.backend!r} of "
            f"{key.label} (expected {expected_device!r})"
        )
        raise ValueError(msg)
    if key.backend == "cpu":
        return CpuPipelineEngine(spec)
    if provider_family(key.provider) == "gigaam":
        if key.provider != STT_PROVIDER_GIGAAM_E2E_CTC:
            msg = f"STT provider {key.provider} has no GPU engine"
            raise ValueError(msg)
        return GigaamGpuBatchEngine(spec)
    return WhisperGpuEngine(spec)
