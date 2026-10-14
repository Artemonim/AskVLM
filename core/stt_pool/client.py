"""Thin client of the STT supervisor used by ``external-transcribe``.

One call of :func:`run_client` transcribes one or more inputs through the
shared queue (see :mod:`core.stt_pool.queue`):

1. The call has ONE wall-clock deadline: its start plus ``client_timeout_s``.
   An infinite ``client_timeout_s`` means no deadline: the jobs carry no
   ``deadline_at`` and the call waits for its results (or an interrupt).
2. Each input's duration is probed with ffprobe (optional; a failure leaves it
   unknown) within a bounded share of the call's time budget and within
   :data:`PROBE_TIMEOUT_CAP_S` per input.
3. Every input occurrence becomes its own job; all jobs of the call share one
   ``call_id``, the deadline, the priority and the ``max_parallel`` cap.
4. Without a fresh supervisor heartbeat a detached supervisor is spawned.
5. Results are polled until every input is final or the deadline passes. A
   supervisor that is not alive is respawned at most once per
   :data:`RESPAWN_INTERVAL_S`; when no fresh heartbeat was ever seen within the
   start timeout of the first spawn (also without a deadline), the unfinished
   inputs are ``unavailable`` and the detail names the supervisor's output log.
6. Inputs still unfinished at the deadline get cancel markers and end as
   ``timeout``; an interrupted call writes cancel markers before it unwinds, so
   an abandoned call never leaves the supervisor working for nobody.

The client performs no transcription and no in-process recovery, and imports no
ML packages (nor the supervisor or the engines).
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

from core.stt_pool.config import project_root
from core.stt_pool.keys import (
    normalize_device_request,
    normalize_language,
    normalize_priority,
)
from core.stt_pool.procs import spawn_detached
from core.stt_pool.queue import (
    DEFAULT_HEARTBEAT_STALE_S,
    MAX_PARALLEL_LIMIT,
    JobRecord,
    JobRequest,
    QueuePaths,
    consume_result,
    is_supervisor_alive,
    new_id,
    read_result,
    request_cancel,
    resolve_queue_root,
    submit_job,
)
from core.stt_pool.sysprobe import child_interpreter
from core.stt_providers import normalize_stt_provider

if TYPE_CHECKING:
    from collections.abc import Sequence

    from core.stt_pool.keys import DeviceRequest, Priority
    from core.stt_pool.queue import ResultRecord

logger = logging.getLogger(__name__)

InputStatus = Literal["ok", "empty", "error", "timeout", "unavailable"]

# * Pause between two result polls.
DEFAULT_POLL_INTERVAL_S = 0.2
# * Longest wait for the first fresh heartbeat after the first spawn.
DEFAULT_START_TIMEOUT_S = 30.0
# * Shortest pause between two spawns while no supervisor is alive.
RESPAWN_INTERVAL_S = 10.0
# * Longest ffprobe run for one input.
PROBE_TIMEOUT_CAP_S = 10.0
# * Share of the call's time budget that duration probing may use in total.
PROBE_BUDGET_SHARE = 0.25
# * Spacing of ``submitted_at`` within one call, so the supervisor's
# * ``(submitted_at, job_id)`` scan order follows the input order.
_SUBMIT_SPACING_S = 1e-6

TIMEOUT_DETAIL = "client timeout waiting for transcript"
UNAVAILABLE_DETAIL = "transcription supervisor unavailable"

SpawnFn: TypeAlias = Callable[[QueuePaths], None]
ProbeFn: TypeAlias = Callable[[Path, float], float | None]


@dataclass(frozen=True, slots=True)
class ClientRequest:
    """One ``external-transcribe`` call through the supervisor.

    Attributes:
        inputs: Media files in input order; a repeated path is a separate job.
        provider: STT provider id (normalized by the client).
        whisper_model: Requested Whisper model name; ignored for GigaAM.
        compute_type: Requested compute type (``auto`` resolves per backend).
        device: Device request ``auto``, ``cpu`` or ``cuda``.
        language: Language hint for every input, or ``None`` to auto-detect.
        priority: Priority class shared by every job of the call.
        max_parallel: Cap on concurrently running jobs of this call, or ``None``
            to leave concurrency to the supervisor.
        client_timeout_s: Wall-clock budget of the whole call in seconds,
            probing and queue wait included; ``math.inf`` means no deadline.

    """

    inputs: tuple[Path, ...]
    provider: str
    whisper_model: str
    compute_type: str
    device: DeviceRequest
    language: str | None
    priority: Priority
    max_parallel: int | None
    client_timeout_s: float


@dataclass(frozen=True, slots=True)
class InputOutcome:
    """Final outcome of one input of a call.

    Attributes:
        index: Position of the input in :attr:`ClientRequest.inputs`.
        input_path: The input path exactly as given in the request.
        status: ``ok`` (non-empty transcript), ``empty`` (no speech),
            ``error``, ``timeout`` or ``unavailable``.
        text: Transcript text; empty unless ``status`` is ``ok``.
        detail: Diagnostic text for a failed input (the worker's error text for
            ``error``), ``None`` for ``ok`` and ``empty``.
        warnings: Non-fatal notices of the supervisor for the caller's stderr.
        backend: ``cpu`` or ``gpu`` that ran the input, when known.

    """

    index: int
    input_path: Path
    status: InputStatus
    text: str
    detail: str | None
    warnings: tuple[str, ...]
    backend: str | None


@dataclass(frozen=True, slots=True)
class ClientOutcome:
    """Outcomes of every input of a call.

    Attributes:
        outcomes: One outcome per input, in input order.

    """

    outcomes: tuple[InputOutcome, ...]


OutcomeFn: TypeAlias = Callable[[InputOutcome], None]


@dataclass(frozen=True, slots=True)
class _NormalizedRequest:
    """Validated, canonical fields of a :class:`ClientRequest`."""

    provider: str
    device: DeviceRequest
    language: str | None
    priority: Priority
    max_parallel: int | None


def _normalize_request(request: ClientRequest) -> _NormalizedRequest:
    """Validate *request* and return its canonical job fields.

    Args:
        request: The call to validate.

    Returns:
        Canonical provider, device, language, priority and parallel cap.

    Raises:
        ValueError: When there is no input, the timeout is NaN or not positive
            (positive infinity is allowed and means no deadline),
            ``max_parallel`` is outside ``1..64``, or the provider, device,
            language or priority is invalid.

    """
    if not request.inputs:
        msg = "at least one input is required"
        raise ValueError(msg)
    timeout_s = request.client_timeout_s
    if math.isnan(timeout_s) or timeout_s <= 0:
        msg = (
            "client_timeout_s must be a positive number or infinity "
            f"(no deadline), got {timeout_s!r}"
        )
        raise ValueError(msg)
    max_parallel = request.max_parallel
    if max_parallel is not None and not 1 <= max_parallel <= MAX_PARALLEL_LIMIT:
        msg = f"max_parallel must be in 1..{MAX_PARALLEL_LIMIT}, got {max_parallel}"
        raise ValueError(msg)
    provider = normalize_stt_provider(request.provider)
    return _NormalizedRequest(
        provider=provider,
        device=normalize_device_request(provider, request.device),
        language=normalize_language(request.language),
        priority=normalize_priority(request.priority),
        max_parallel=max_parallel,
    )


def spawn_supervisor(paths: QueuePaths) -> None:
    """Start a detached supervisor process that serves the queue of *paths*.

    The supervisor runs ``cli.py external-transcribe-daemon --queue-dir
    <root>`` from the base interpreter (see
    :func:`~core.stt_pool.sysprobe.child_interpreter`) with the project root as
    its working directory, outlives this client, and appends its output to
    ``control/supervisor.out.log``. A supervisor that finds the queue lock taken
    exits at once, so a redundant spawn is harmless.

    Args:
        paths: Queue layout the supervisor must serve.

    Raises:
        OSError: When the log cannot be opened or the process cannot start.

    """
    root = project_root()
    interpreter, interpreter_env = child_interpreter()
    env = dict(os.environ)
    env.update(interpreter_env)
    argv = [
        interpreter,
        "-X",
        "utf8",
        str(root / "cli.py"),
        "external-transcribe-daemon",
        "--queue-dir",
        str(paths.root),
    ]
    pid = spawn_detached(argv, cwd=root, log_path=paths.supervisor_out_log, env=env)
    logger.info("supervisor spawned: pid=%s root=%s", pid, paths.root)


def _probe_with_ffprobe(path: Path, timeout_s: float) -> float | None:
    """Return the media duration of *path* from ffprobe, or ``None``.

    Args:
        path: Media file.
        timeout_s: Longest probe run in seconds.

    Returns:
        The duration in seconds, or ``None`` when ffprobe cannot report one.

    """
    # * Imported on use: only a real call needs the ffmpeg helpers.
    from core.ffmpeg import (  # noqa: PLC0415 - lazy media helper import keeps the client import light
        MediaProbeError,
        probe_media_duration,
    )

    try:
        return probe_media_duration(path, timeout_s=timeout_s)
    except MediaProbeError as exc:
        logger.debug("duration probe failed: path=%s error=%s", path, exc)
        return None


def _safe_probe(prober: ProbeFn, path: Path, timeout_s: float) -> float | None:
    """Run *prober* and keep only a finite, non-negative duration.

    Args:
        prober: Duration probe ``(path, timeout_s) -> seconds | None``.
        path: Media file.
        timeout_s: Longest probe run in seconds.

    Returns:
        The duration in seconds, or ``None`` on any probe failure.

    """
    try:
        value = prober(path, timeout_s)
        number = None if value is None else float(value)
    except Exception as exc:  # noqa: BLE001 - a duration is optional; any probe failure leaves it unknown
        logger.debug("duration probe failed: path=%s error=%s", path, exc)
        return None
    if number is None or not math.isfinite(number) or number < 0:
        return None
    return number


def _probe_durations(
    inputs: Sequence[Path],
    prober: ProbeFn,
    *,
    clock: Callable[[], float],
    budget_end: float,
) -> list[float | None]:
    """Probe the duration of every input within a shared time budget.

    Each probe is capped at :data:`PROBE_TIMEOUT_CAP_S` and at the time left
    until *budget_end*; inputs reached after the budget stay unknown. A path
    that occurs several times is probed once.

    Args:
        inputs: Media files in input order.
        prober: Duration probe ``(path, timeout_s) -> seconds | None``.
        clock: Wall-clock source.
        budget_end: Time after which no probe starts.

    Returns:
        One duration (or ``None``) per input, in input order.

    """
    known: dict[Path, float | None] = {}
    durations: list[float | None] = []
    for path in inputs:
        if path not in known:
            remaining = budget_end - clock()
            known[path] = (
                _safe_probe(prober, path, min(PROBE_TIMEOUT_CAP_S, remaining))
                if remaining > 0
                else None
            )
        durations.append(known[path])
    return durations


def _absolute_path(path: Path) -> str:
    """Return the absolute path of an input for its job record.

    Args:
        path: Input path as given.

    Returns:
        The resolved path, or the plain absolute path when resolving fails.

    """
    try:
        return str(path.resolve())
    except OSError:
        return str(path.absolute())


def _outcome_from_result(
    index: int, input_path: Path, result: ResultRecord
) -> InputOutcome:
    """Map a supervisor result onto the client outcome of one input.

    ``ok`` stays ``ok`` (a blank transcript counts as ``empty``), ``empty``
    stays ``empty``, ``error`` and ``cancelled`` become ``error``, ``expired``
    becomes ``timeout`` and ``unavailable`` stays ``unavailable``.

    Args:
        index: Input position.
        input_path: Input path as given in the request.
        result: Terminal supervisor result.

    Returns:
        The input's outcome, with the result's warnings and backend.

    """
    status = result.status
    if status in {"ok", "empty"}:
        has_text = status == "ok" and bool(result.text.strip())
        return InputOutcome(
            index=index,
            input_path=input_path,
            status="ok" if has_text else "empty",
            text=result.text if has_text else "",
            detail=None,
            warnings=result.warnings,
            backend=result.backend,
        )
    mapped: InputStatus
    if status == "expired":
        mapped, detail = "timeout", result.error_detail or TIMEOUT_DETAIL
    elif status == "unavailable":
        mapped, detail = "unavailable", result.error_detail or UNAVAILABLE_DETAIL
    else:
        mapped, detail = "error", result.error_detail or status
    return InputOutcome(
        index=index,
        input_path=input_path,
        status=mapped,
        text="",
        detail=detail,
        warnings=result.warnings,
        backend=result.backend,
    )


class _Call:
    """State of one client call: submitted jobs, final outcomes, spawn bookkeeping."""

    def __init__(  # noqa: PLR0913 - every argument is a run_client seam
        self,
        request: ClientRequest,
        paths: QueuePaths,
        *,
        deadline: float | None,
        spawn: SpawnFn,
        clock: Callable[[], float],
        sleeper: Callable[[float], None],
        on_outcome: OutcomeFn | None,
        poll_interval_s: float,
        start_timeout_s: float,
        stale_after_s: float,
    ) -> None:
        self._request = request
        self._paths = paths
        self._deadline = deadline
        self._spawn = spawn
        self._clock = clock
        self._sleeper = sleeper
        self._on_outcome = on_outcome
        self._poll_interval_s = poll_interval_s
        self._start_timeout_s = start_timeout_s
        self._stale_after_s = stale_after_s
        self._outcomes: dict[int, InputOutcome] = {}
        self._pending: dict[int, str] = {}
        self._seen_alive = False
        self._first_spawn_at: float | None = None
        self._last_spawn_at: float | None = None
        self._spawn_error: str | None = None

    # * Outcomes.

    def finish(self, outcome: InputOutcome) -> None:
        """Record the final outcome of one input and report it."""
        self._outcomes[outcome.index] = outcome
        if self._on_outcome is not None:
            self._on_outcome(outcome)

    def fail(self, index: int, status: InputStatus, detail: str) -> None:
        """Finish one input with a client-side failure."""
        self.finish(
            InputOutcome(
                index=index,
                input_path=self._request.inputs[index],
                status=status,
                text="",
                detail=detail,
                warnings=(),
                backend=None,
            )
        )

    def fail_all(self, status: InputStatus, detail: str) -> None:
        """Finish every input that has no outcome yet with the same failure."""
        for index in range(len(self._request.inputs)):
            if index not in self._outcomes:
                self.fail(index, status, detail)

    def outcome(self) -> ClientOutcome:
        """Return the outcomes in input order (every input must be final)."""
        return ClientOutcome(
            tuple(self._outcomes[index] for index in range(len(self._request.inputs)))
        )

    # * Submission.

    def submit(
        self, fields: _NormalizedRequest, durations: Sequence[float | None]
    ) -> None:
        """Write one job per input; a failed write finishes that input.

        Args:
            fields: Canonical request fields.
            durations: Probed duration per input.

        """
        request = self._request
        call_id = new_id()
        client_pid = os.getpid()
        job_request = JobRequest(
            provider=fields.provider,
            whisper_model=request.whisper_model,
            compute_type=request.compute_type,
            device=fields.device,
        )
        last_submitted = -math.inf
        for index, input_path in enumerate(request.inputs):
            submitted_at = max(self._clock(), last_submitted + _SUBMIT_SPACING_S)
            last_submitted = submitted_at
            job = JobRecord(
                job_id=new_id(),
                call_id=call_id,
                call_index=index,
                max_parallel=fields.max_parallel,
                submitted_at=submitted_at,
                deadline_at=self._deadline,
                input_path=_absolute_path(input_path),
                duration_s=durations[index],
                request=job_request,
                language=fields.language,
                client_pid=client_pid,
                priority=fields.priority,
            )
            try:
                submit_job(self._paths, job)
            except OSError as exc:
                self.fail(
                    index,
                    "unavailable",
                    f"{UNAVAILABLE_DETAIL}: cannot write the job file: {exc}",
                )
                continue
            except ValueError as exc:
                self.fail(index, "error", str(exc))
                continue
            self._pending[index] = job.job_id
        logger.debug(
            "call submitted: call_id=%s jobs=%d root=%s",
            call_id,
            len(self._pending),
            self._paths.root,
        )

    # * Supervisor liveness.

    def _alive(self, now: float) -> bool:
        """Return whether a supervisor with a fresh heartbeat serves the queue."""
        alive = is_supervisor_alive(
            self._paths, now=now, stale_after_s=self._stale_after_s
        )
        if alive:
            self._seen_alive = True
        return alive

    def _spawn_now(self, now: float) -> None:
        """Spawn a supervisor; a failed spawn counts as an attempt."""
        if self._first_spawn_at is None:
            self._first_spawn_at = now
        self._last_spawn_at = now
        try:
            self._spawn(self._paths)
        except Exception as exc:  # noqa: BLE001 - the start timeout turns a failed spawn into "unavailable"
            self._spawn_error = f"{type(exc).__name__}: {exc}"
            logger.warning("supervisor spawn failed: %s", self._spawn_error)

    def ensure_supervisor(self) -> None:
        """Spawn a supervisor when jobs are pending and none is alive."""
        if not self._pending:
            return
        now = self._clock()
        if not self._alive(now):
            self._spawn_now(now)

    def _never_started(self) -> bool:
        """Return whether a spawn happened and no fresh heartbeat was ever seen."""
        return self._first_spawn_at is not None and not self._seen_alive

    def _unavailable_detail(self) -> str:
        """Return the detail of inputs abandoned for lack of a supervisor.

        The detail ends with the path of the supervisor's output log, where a
        startup error of the spawned process (for example a config error) is
        written.
        """
        detail = UNAVAILABLE_DETAIL
        if self._spawn_error is not None:
            detail = f"{detail} (spawn failed: {self._spawn_error})"
        return f"{detail}; supervisor log: {self._paths.supervisor_out_log}"

    # * Waiting.

    def _ready_job_ids(self) -> set[str]:
        """Return the pending job ids whose result file exists.

        One directory scan per poll keeps large batches cheap; when the scan
        fails, every pending job is read directly.
        """
        pending = set(self._pending.values())
        try:
            with os.scandir(self._paths.results) as entries:
                names = {entry.name for entry in entries}
        except OSError:
            return pending
        return {
            job_id
            for job_id in pending
            if self._paths.result_file(job_id).name in names
        }

    def collect_results(self) -> None:
        """Finish every pending input whose result is readable, consuming it."""
        ready = self._ready_job_ids()
        for index, job_id in sorted(self._pending.items()):
            if job_id not in ready:
                continue
            result = read_result(self._paths, job_id)
            if result is None:
                continue
            consume_result(self._paths, job_id)
            del self._pending[index]
            self.finish(
                _outcome_from_result(index, self._request.inputs[index], result)
            )

    def wait(self) -> None:
        """Poll results until every input is final or the call is abandoned."""
        while self._pending:
            self.collect_results()
            if not self._pending:
                return
            now = self._clock()
            deadline = self._deadline
            if deadline is not None and now >= deadline:
                if self._never_started():
                    self.abandon("unavailable", self._unavailable_detail())
                else:
                    self.abandon("timeout", TIMEOUT_DETAIL)
                return
            if not self._alive(now):
                first_spawn_at = self._first_spawn_at
                if (
                    first_spawn_at is not None
                    and not self._seen_alive
                    and now - first_spawn_at >= self._start_timeout_s
                ):
                    self.abandon("unavailable", self._unavailable_detail())
                    return
                last_spawn_at = self._last_spawn_at
                if last_spawn_at is None or now - last_spawn_at >= RESPAWN_INTERVAL_S:
                    self._spawn_now(now)
            pause = self._poll_interval_s
            if deadline is not None:
                pause = min(pause, deadline - now)
            self._sleeper(max(0.0, pause))

    def _cancel_pending(self) -> list[int]:
        """Write cancel markers for every pending job and forget them.

        Returns:
            The indexes of the cancelled inputs, in input order.

        """
        cancelled = sorted(self._pending)
        for index in cancelled:
            job_id = self._pending.pop(index)
            try:
                request_cancel(self._paths, job_id)
            except OSError as exc:
                logger.warning(
                    "cancel marker not written: job_id=%s error=%s", job_id, exc
                )
        return cancelled

    def abandon(self, status: InputStatus, detail: str) -> None:
        """Cancel every pending job and finish its input with *status*."""
        for index in self._cancel_pending():
            self.fail(index, status, detail)

    def cancel_leftovers(self) -> None:
        """Cancel the jobs of an interrupted call without reporting outcomes."""
        if self._pending:
            logger.warning(
                "client call interrupted; cancelling %d job(s)", len(self._pending)
            )
            self._cancel_pending()


def run_client(  # noqa: PLR0913 - the keyword seams are the test and caller contract
    request: ClientRequest,
    *,
    queue_root: Path | None = None,
    spawn: SpawnFn | None = None,
    clock: Callable[[], float] = time.time,
    sleeper: Callable[[float], None] = time.sleep,
    probe_duration: ProbeFn | None = None,
    on_outcome: OutcomeFn | None = None,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    start_timeout_s: float = DEFAULT_START_TIMEOUT_S,
    stale_after_s: float = DEFAULT_HEARTBEAT_STALE_S,
) -> ClientOutcome:
    """Transcribe the inputs of *request* through the supervisor.

    Args:
        request: The call.
        queue_root: Explicit queue root; ``None`` resolves it through
            :func:`~core.stt_pool.queue.resolve_queue_root` (the
            ``ASKVLM_EXTERNAL_QUEUE_DIR`` environment variable is honored).
        spawn: Starts a supervisor for a queue; defaults to
            :func:`spawn_supervisor`.
        clock: Wall-clock source of the deadline, the spawn timing and the job
            timestamps.
        sleeper: Pause between polls.
        probe_duration: Duration probe ``(path, timeout_s) -> seconds | None``;
            defaults to ffprobe. Any failure leaves the duration unknown.
        on_outcome: Called once per input as soon as its outcome is final, in
            completion order.
        poll_interval_s: Pause between result polls.
        start_timeout_s: Longest wait for the first fresh heartbeat after the
            first spawn (bounded by the deadline).
        stale_after_s: Largest heartbeat age that counts as alive.

    Returns:
        One outcome per input, in input order.

    Raises:
        ValueError: When the request is invalid (see :class:`ClientRequest`).

    """
    started_at = clock()
    fields = _normalize_request(request)
    # * An infinite timeout leaves the call without a deadline; the probe budget
    # * then stays unbounded and every probe keeps its own per-input cap.
    deadline = (
        None
        if math.isinf(request.client_timeout_s)
        else started_at + request.client_timeout_s
    )
    paths = QueuePaths(resolve_queue_root(queue_root))
    call = _Call(
        request,
        paths,
        deadline=deadline,
        spawn=spawn if spawn is not None else spawn_supervisor,
        clock=clock,
        sleeper=sleeper,
        on_outcome=on_outcome,
        poll_interval_s=poll_interval_s,
        start_timeout_s=start_timeout_s,
        stale_after_s=stale_after_s,
    )
    try:
        paths.ensure()
    except OSError as exc:
        call.fail_all("unavailable", f"{UNAVAILABLE_DETAIL}: queue {paths.root}: {exc}")
        return call.outcome()
    durations = _probe_durations(
        request.inputs,
        probe_duration if probe_duration is not None else _probe_with_ffprobe,
        clock=clock,
        budget_end=started_at + request.client_timeout_s * PROBE_BUDGET_SHARE,
    )
    try:
        call.submit(fields, durations)
        call.ensure_supervisor()
        call.wait()
    finally:
        # * Normal returns leave nothing pending; an exception (an interrupt, a
        # * failing callback) still tells the supervisor to drop the jobs.
        call.cancel_leftovers()
    return call.outcome()
