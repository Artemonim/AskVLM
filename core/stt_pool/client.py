"""Thin client of the STT supervisor used by ``external-transcribe``.

One call of :func:`run_client` transcribes (or diarizes) one or more inputs
through the shared queue (see :mod:`core.stt_pool.queue`):

1. The call has ONE wall-clock deadline: its start plus ``client_timeout_s``.
   An infinite ``client_timeout_s`` means no deadline: the jobs carry no
   ``deadline_at`` and the call waits for its results (or an interrupt).
2. Each input's duration is probed with ffprobe (optional; a failure leaves it
   unknown) within a bounded share of the call's time budget and within
   :data:`PROBE_TIMEOUT_CAP_S` per input. With a media window the job duration
   is the window length, and a window that starts at or after the known media
   end fails its input before any job is written.
3. Every input occurrence becomes its own job: an ``stt`` job in transcribe
   mode, plus a ``diarize`` job when speaker diarization is requested, or one
   ``diarize`` job in diarize mode. All jobs of the call share one
   ``call_id``, the deadline, the priority and the ``max_parallel`` cap, and
   their ``submitted_at`` follows the input order (``stt`` before ``diarize``).
4. Without a fresh supervisor heartbeat a detached supervisor is spawned.
5. Results are polled until every input is final or the deadline passes. An
   input is final once all of its jobs have results, or as soon as one of them
   fails (its other job is then cancelled); the results of an input are merged
   into ONE outcome (see :class:`InputOutcome`). A supervisor that is not alive
   is respawned at most once per :data:`RESPAWN_INTERVAL_S`; when no fresh
   heartbeat was ever seen within the start timeout of the first spawn (also
   without a deadline), the unfinished inputs are ``unavailable`` and the
   detail names the supervisor's output log.
6. Inputs still unfinished at the deadline get cancel markers on every pending
   job and end as ``timeout``; an interrupted call writes cancel markers before
   it unwinds, so an abandoned call never leaves the supervisor working for
   nobody.

The client performs no transcription and no in-process recovery, and imports no
ML packages (nor the supervisor or the engines).
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

from core.stt_options import DiarizeOptions, MediaWindow, WhisperDecodeOptions
from core.stt_output import (
    MIN_VOICES_FOR_SPEAKER_PREFIXES,
    assign_speakers,
    speaker_text,
    voice_count,
)
from core.stt_pool.config import project_root
from core.stt_pool.keys import (
    DIARIZE_COMPUTE_TYPE,
    DIARIZE_MODEL_IDENTITY,
    DIARIZE_PROVIDER,
    normalize_device_request,
    normalize_language,
    normalize_priority,
)
from core.stt_pool.procs import spawn_detached
from core.stt_pool.queue import (
    DEFAULT_HEARTBEAT_STALE_S,
    JOB_KIND_DIARIZE,
    JOB_KIND_STT,
    MAX_PARALLEL_LIMIT,
    JobRecord,
    JobRequest,
    JobValidationError,
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
from core.stt_providers import is_gigaam_provider, normalize_stt_provider

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from core.stt_pool.keys import DeviceRequest, Priority
    from core.stt_pool.queue import JobKind, ResultRecord

logger = logging.getLogger(__name__)

InputStatus = Literal["ok", "empty", "error", "timeout", "unavailable"]
ClientMode = Literal["transcribe", "diarize"]

# * Call modes: speech-to-text per input (optionally with speaker diarization),
# * or speaker diarization alone.
MODE_TRANSCRIBE: ClientMode = "transcribe"
MODE_DIARIZE: ClientMode = "diarize"
_CLIENT_MODES: tuple[ClientMode, ...] = (MODE_TRANSCRIBE, MODE_DIARIZE)

# * Statuses of a successful input; every other status is a failure.
_SUCCESS_STATUSES: frozenset[InputStatus] = frozenset({"ok", "empty"})

# * Failure statuses of an input whose jobs fail differently, most significant
# * first (the order of the CLI's exit codes).
_FAILURE_PRIORITY: tuple[InputStatus, ...] = ("timeout", "unavailable", "error")

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
DIARIZE_TIMEOUT_DETAIL = "client timeout waiting for speaker diarization"
UNAVAILABLE_DETAIL = "transcription supervisor unavailable"

# * Prefix of the detail of a failed diarization job of an input that also
# * has a transcription job, so the caller sees which part failed.
DIARIZE_DETAIL_PREFIX = "speaker diarization"

SpawnFn: TypeAlias = Callable[[QueuePaths], None]
ProbeFn: TypeAlias = Callable[[Path, float], float | None]


@dataclass(frozen=True, slots=True)
class ClientRequest:
    """One ``external-transcribe`` (or diarization) call through the supervisor.

    Attributes:
        inputs: Media files in input order; a repeated path is a separate job.
        provider: STT provider id (normalized by the client); not used in
            diarize mode.
        whisper_model: Requested Whisper model name; ignored for GigaAM and
            in diarize mode.
        compute_type: Requested compute type (``auto`` resolves per backend);
            not used in diarize mode.
        device: Device request ``auto``, ``cpu`` or ``cuda`` of the
            transcription jobs; in diarize mode only ``auto`` or ``cpu``
            (diarization runs on the CPU only).
        language: Language hint for every input, or ``None`` to auto-detect;
            must be ``None`` (or blank) in diarize mode.
        priority: Priority class shared by every job of the call.
        max_parallel: Cap on concurrently running jobs of this call, or ``None``
            to leave concurrency to the supervisor. A transcription with
            diarization runs two jobs per input, and both count.
        client_timeout_s: Wall-clock budget of the whole call in seconds,
            probing and queue wait included; ``math.inf`` means no deadline.
        mode: ``transcribe`` (speech-to-text per input) or ``diarize``
            (speaker diarization per input, no transcription).
        want_segments: Return the transcript segments in
            :attr:`InputOutcome.data` (transcribe mode only).
        decode: Whisper-only decoding overrides (transcribe mode with the
            Whisper provider only).
        window: Time window of every input to process (both modes).
        diarization: Speaker diarization settings. In transcribe mode a
            diarization job runs on the CPU next to every transcription job
            (whatever ``device`` requests) and the outcome merges both; in
            diarize mode the settings of the diarization jobs (``None``
            selects the defaults).

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
    mode: ClientMode = MODE_TRANSCRIBE
    want_segments: bool = False
    decode: WhisperDecodeOptions = field(default_factory=WhisperDecodeOptions)
    window: MediaWindow = field(default_factory=MediaWindow)
    diarization: DiarizeOptions | None = None


@dataclass(frozen=True, slots=True)
class InputOutcome:
    """Final outcome of one input of a call.

    Attributes:
        index: Position of the input in :attr:`ClientRequest.inputs`.
        input_path: The input path exactly as given in the request.
        status: ``ok`` (non-empty transcript, or any diarization result in
            diarize mode, turns or none), ``empty`` (no speech), ``error``,
            ``timeout`` or ``unavailable``.
        text: Transcript text; empty unless ``status`` is ``ok``, and always
            empty in diarize mode. With diarization and at least two voices
            among the transcript segments it holds one ``speaker_xx: text``
            line per segment; with fewer voices it is the plain transcript.
        detail: Diagnostic text for a failed input (the worker's error text for
            ``error``; a failed diarization job of a transcription is prefixed
            with :data:`DIARIZE_DETAIL_PREFIX`), ``None`` for ``ok`` and
            ``empty``.
        warnings: Non-fatal notices of the supervisor for the caller's stderr.
        backend: ``cpu`` or ``gpu`` that ran the input (its transcription job
            in transcribe mode), when known.
        data: Structured payload of an ``ok`` or ``empty`` input, ``None`` for
            a failed input and for a plain transcription:
            ``{"segments": [...]}`` for a transcription with ``want_segments``;
            ``{"segments": [...], "turns": [...], "speakers": [...]}`` for a
            transcription with diarization, where ``speakers`` holds the label
            (or ``None``) of each segment in segment order; ``{"turns": [...]}``
            in diarize mode. Segments and turns use the media-relative layout
            of :mod:`core.stt_output`.

    """

    index: int
    input_path: Path
    status: InputStatus
    text: str
    detail: str | None
    warnings: tuple[str, ...]
    backend: str | None
    data: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class ClientOutcome:
    """Outcomes of every input of a call.

    Attributes:
        outcomes: One outcome per input, in input order.

    """

    outcomes: tuple[InputOutcome, ...]


OutcomeFn: TypeAlias = Callable[[InputOutcome], None]


@dataclass(frozen=True, slots=True)
class _JobPlan:
    """Kind, model selection and options of one job that every input gets.

    Attributes:
        kind: ``stt`` or ``diarize``.
        request: Provider, model and device selection of the job.
        language: Normalized language hint (``stt`` jobs only).
        want_segments: Ask the worker for structured segments (``stt`` only).
        decode: Whisper decoding overrides (``stt`` only).
        diarize: Diarization settings (``diarize`` only).

    """

    kind: JobKind
    request: JobRequest
    language: str | None = None
    want_segments: bool = False
    decode: WhisperDecodeOptions = field(default_factory=WhisperDecodeOptions)
    diarize: DiarizeOptions | None = None


@dataclass(frozen=True, slots=True)
class _NormalizedRequest:
    """Validated, canonical fields of a :class:`ClientRequest`.

    Attributes:
        mode: Call mode.
        plans: The jobs of every input; the first one is the input's primary
            job (``stt`` in transcribe mode, ``diarize`` in diarize mode).
        priority: Priority class of every job.
        max_parallel: Parallel cap of the call, or ``None``.
        window: Media window of every job.
        want_segments: The caller asked for the transcript segments.

    """

    mode: ClientMode
    plans: tuple[_JobPlan, ...]
    priority: Priority
    max_parallel: int | None
    window: MediaWindow
    want_segments: bool

    @property
    def has_sibling_jobs(self) -> bool:
        """Whether every input runs more than one job."""
        return len(self.plans) > 1


def _diarize_plan(options: DiarizeOptions, device: str) -> _JobPlan:
    """Return the diarization job of every input.

    Args:
        options: Diarization settings.
        device: Raw device request; ``auto`` and ``cpu`` select the CPU.

    Returns:
        The ``diarize`` job plan on the diarization pseudo-provider.

    Raises:
        ValueError: When *device* is ``cuda`` (diarization runs on the CPU
            only) or unknown.

    """
    return _JobPlan(
        kind=JOB_KIND_DIARIZE,
        request=JobRequest(
            provider=DIARIZE_PROVIDER,
            whisper_model=DIARIZE_MODEL_IDENTITY,
            compute_type=DIARIZE_COMPUTE_TYPE,
            device=normalize_device_request(DIARIZE_PROVIDER, device),
        ),
        diarize=options,
    )


def _diarize_mode_plans(request: ClientRequest) -> tuple[_JobPlan, ...]:
    """Validate a diarize-mode request and return its single job plan.

    Args:
        request: The call.

    Returns:
        One ``diarize`` job plan.

    Raises:
        ValueError: When the request carries transcription options (language,
            segments, decode options) or asks for ``cuda``.

    """
    given: list[str] = []
    if request.language is not None and request.language.strip():
        given.append(f"language ({request.language.strip()})")
    if request.want_segments:
        given.append("transcript segments")
    if not request.decode.is_default:
        given.append(f"Whisper decode options ({', '.join(request.decode.to_wire())})")
    if given:
        msg = (
            "speaker diarization does not take the transcription option(s): "
            f"{'; '.join(given)}"
        )
        raise ValueError(msg)
    options = DiarizeOptions() if request.diarization is None else request.diarization
    return (_diarize_plan(options, request.device),)


def _transcribe_plans(request: ClientRequest) -> tuple[_JobPlan, ...]:
    """Validate a transcribe-mode request and return its job plans.

    Args:
        request: The call.

    Returns:
        The ``stt`` job plan, followed by a ``diarize`` job plan on the CPU
        when diarization is requested.

    Raises:
        ValueError: When the provider, device or language is invalid, or
            Whisper-only decode options meet a GigaAM provider.

    """
    provider = normalize_stt_provider(request.provider)
    device = normalize_device_request(provider, request.device)
    language = normalize_language(request.language)
    if is_gigaam_provider(provider) and not request.decode.is_default:
        names = ", ".join(request.decode.to_wire())
        msg = (
            f"Whisper-only decode options ({names}) are not supported by "
            f"{provider}; omit them or use the whisper provider"
        )
        raise ValueError(msg)
    stt = _JobPlan(
        kind=JOB_KIND_STT,
        request=JobRequest(
            provider=provider,
            whisper_model=request.whisper_model,
            compute_type=request.compute_type,
            device=device,
        ),
        language=language,
        # * The diarization merge assigns speakers to the transcript segments.
        want_segments=request.want_segments or request.diarization is not None,
        decode=request.decode,
    )
    if request.diarization is None:
        return (stt,)
    # * Diarization runs on the CPU whatever device the transcription requests.
    return (stt, _diarize_plan(request.diarization, "auto"))


def _normalize_request(request: ClientRequest) -> _NormalizedRequest:
    """Validate *request* and return its canonical job fields.

    Args:
        request: The call to validate.

    Returns:
        Canonical mode, job plans, priority, parallel cap and window.

    Raises:
        ValueError: When there is no input, the timeout is NaN or not positive
            (positive infinity is allowed and means no deadline),
            ``max_parallel`` is outside ``1..64``, the mode is unknown, the
            provider, device, language or priority is invalid, or the options
            do not fit the mode or the provider (see :class:`ClientRequest`).

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
    if request.mode == MODE_DIARIZE:
        plans = _diarize_mode_plans(request)
    elif request.mode == MODE_TRANSCRIBE:
        plans = _transcribe_plans(request)
    else:
        msg = f"mode must be one of {', '.join(_CLIENT_MODES)}, got {request.mode!r}"
        raise ValueError(msg)
    return _NormalizedRequest(
        mode=request.mode,
        plans=plans,
        priority=normalize_priority(request.priority),
        max_parallel=max_parallel,
        window=request.window,
        want_segments=request.want_segments,
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


def _seconds(value: float) -> str:
    """Format seconds for a message, with at most millisecond precision.

    Args:
        value: Seconds.

    Returns:
        The number without trailing zeros, such as ``2`` or ``12.345``.

    """
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _job_duration(window: MediaWindow, media_duration: float | None) -> float | None:
    """Return the audio length one job of an input processes.

    Without a window it is the probed media duration. With a window it is the
    window length, cut at the media end when the media duration is known.

    Args:
        window: Media window of the call.
        media_duration: Probed media duration, or ``None`` when unknown.

    Returns:
        The duration in seconds, or ``None`` when it is unknown.

    Raises:
        ValueError: When the window starts at or after the known media end.

    """
    if window.is_default:
        return media_duration
    if media_duration is None:
        return window.duration_s
    remaining = media_duration - window.start_s
    if remaining <= 0:
        msg = (
            f"window starts at {_seconds(window.start_s)} s, beyond the media "
            f"duration {_seconds(media_duration)} s"
        )
        raise ValueError(msg)
    if window.duration_s is None:
        return remaining
    return min(window.duration_s, remaining)


def _outcome_from_result(
    index: int,
    input_path: Path,
    result: ResultRecord,
    *,
    kind: JobKind = JOB_KIND_STT,
) -> InputOutcome:
    """Map a supervisor result onto the client outcome of one job of an input.

    For an ``stt`` job, ``ok`` stays ``ok`` (a blank transcript counts as
    ``empty``) and ``empty`` stays ``empty``. A ``diarize`` job that succeeded
    is ``ok`` with an empty text, whether it found turns or not. For both
    kinds ``error`` and ``cancelled`` become ``error`` (this includes the
    supervisor's ``result_too_large`` and ``protocol_error`` results, with
    their detail), ``expired`` becomes ``timeout`` and ``unavailable`` stays
    ``unavailable``.

    Args:
        index: Input position.
        input_path: Input path as given in the request.
        result: Terminal supervisor result.
        kind: Kind of the job the result answers.

    Returns:
        The job's outcome, with the result's warnings and backend, and its
        ``data`` when it succeeded.

    """
    status = result.status
    if status in {"ok", "empty"}:
        if kind == JOB_KIND_DIARIZE:
            mapped_ok: InputStatus = "ok"
            text = ""
        else:
            has_text = status == "ok" and bool(result.text.strip())
            mapped_ok = "ok" if has_text else "empty"
            text = result.text if has_text else ""
        return InputOutcome(
            index=index,
            input_path=input_path,
            status=mapped_ok,
            text=text,
            detail=None,
            warnings=result.warnings,
            backend=result.backend,
            data=result.data,
        )
    mapped: InputStatus
    if status == "expired":
        fallback = (
            DIARIZE_TIMEOUT_DETAIL if kind == JOB_KIND_DIARIZE else TIMEOUT_DETAIL
        )
        mapped, detail = "timeout", result.error_detail or fallback
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


def _part_detail(fields: _NormalizedRequest, kind: JobKind, detail: str) -> str:
    """Name the failed part of an input that runs sibling jobs.

    Args:
        fields: Canonical request fields.
        kind: Kind of the job that failed.
        detail: The job's failure detail.

    Returns:
        *detail* prefixed with :data:`DIARIZE_DETAIL_PREFIX` for a failed
        ``diarize`` job next to a transcription, otherwise *detail* unchanged.

    """
    if kind == JOB_KIND_DIARIZE and fields.has_sibling_jobs:
        return f"{DIARIZE_DETAIL_PREFIX}: {detail}"
    return detail


def _merged_warnings(outcomes: Sequence[InputOutcome]) -> tuple[str, ...]:
    """Return the warnings of an input's job outcomes.

    Args:
        outcomes: Job outcomes in job order.

    Returns:
        The warnings of a single outcome unchanged; for several outcomes their
        union in first-seen order without duplicates.

    """
    if len(outcomes) == 1:
        return outcomes[0].warnings
    return tuple(
        dict.fromkeys(warning for outcome in outcomes for warning in outcome.warnings)
    )


def _payload_list(outcome: InputOutcome, name: str) -> list[dict[str, object]]:
    """Return the structured list *name* of a successful job outcome.

    Args:
        outcome: ``ok`` or ``empty`` job outcome.
        name: ``segments`` or ``turns``.

    Returns:
        The list of objects from the outcome's ``data``.

    Raises:
        ValueError: When ``data`` lacks the list or it holds a non-object.

    """
    value = None if outcome.data is None else outcome.data.get(name)
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return [item for item in value if isinstance(item, dict)]
    found = "no data" if outcome.data is None else f"data[{name!r}] = {value!r:.80}"
    msg = f"malformed result: expected a {name!r} list of objects, got {found}"
    raise ValueError(msg)


def _job_failure(
    index: int,
    input_path: Path,
    fields: _NormalizedRequest,
    received: Sequence[tuple[JobKind, InputOutcome]],
) -> InputOutcome | None:
    """Return the failed outcome of an input when one of its jobs failed.

    Args:
        index: Input position.
        input_path: Input path as given in the request.
        fields: Canonical request fields.
        received: The input's job outcomes received so far, in job order.

    Returns:
        ``None`` when no received job failed; otherwise the input's failure
        with the most significant status (``timeout`` > ``unavailable`` >
        ``error``; the earlier job wins a tie) and that job's detail, the
        warnings of every received job and the primary job's backend.

    """
    failed = [
        (kind, outcome)
        for kind, outcome in received
        if outcome.status not in _SUCCESS_STATUSES
    ]
    if not failed:
        return None
    kind, worst = min(failed, key=lambda item: _FAILURE_PRIORITY.index(item[1].status))
    primary = dict(received).get(fields.plans[0].kind)
    return InputOutcome(
        index=index,
        input_path=input_path,
        status=worst.status,
        text="",
        detail=_part_detail(fields, kind, worst.detail or worst.status),
        warnings=_merged_warnings([outcome for _, outcome in received]),
        backend=None if primary is None else primary.backend,
    )


def _diarized_transcript(
    stt: InputOutcome, diarize: InputOutcome, warnings: tuple[str, ...]
) -> InputOutcome:
    """Merge the transcription and the diarization of one input.

    Args:
        stt: Successful ``stt`` job outcome with segments.
        diarize: Successful ``diarize`` job outcome with turns.
        warnings: Warnings of both jobs.

    Returns:
        The input's outcome: the transcription's status and backend, the
        segments, turns and per-segment speakers as ``data``, and a text with
        one ``speaker: text`` line per segment when the segments carry at
        least two voices (otherwise the transcription's text unchanged).

    Raises:
        ValueError: When a payload is malformed (the message names the part).

    """
    try:
        segments = _payload_list(stt, "segments")
    except ValueError as exc:
        msg = f"transcription: {exc}"
        raise ValueError(msg) from exc
    try:
        turns = _payload_list(diarize, "turns")
    except ValueError as exc:
        msg = f"{DIARIZE_DETAIL_PREFIX}: {exc}"
        raise ValueError(msg) from exc
    try:
        speakers = assign_speakers(segments, turns)
    except ValueError as exc:
        msg = f"cannot assign speakers to the transcript segments: {exc}"
        raise ValueError(msg) from exc
    text = stt.text
    # * One voice (or none) keeps the plain transcript, byte for byte.
    if stt.status == "ok" and voice_count(speakers) >= MIN_VOICES_FOR_SPEAKER_PREFIXES:
        labelled = speaker_text(
            [str(segment.get("text") or "") for segment in segments], speakers
        )
        if labelled.strip():
            text = labelled
    return replace(
        stt,
        text=text,
        warnings=warnings,
        data={"segments": segments, "turns": turns, "speakers": speakers},
    )


def _merged_success(
    fields: _NormalizedRequest, received: Mapping[JobKind, InputOutcome]
) -> InputOutcome:
    """Merge the successful job outcomes of one input into its outcome.

    Args:
        fields: Canonical request fields.
        received: Every job outcome of the input, none failed.

    Returns:
        The input's outcome (see :class:`InputOutcome` for ``data``); an
        ``error`` outcome when a payload is malformed.

    """
    ordered = [received[plan.kind] for plan in fields.plans]
    primary = ordered[0]
    warnings = _merged_warnings(ordered)
    try:
        if fields.mode == MODE_DIARIZE:
            data: dict[str, object] = {"turns": _payload_list(primary, "turns")}
            return replace(primary, status="ok", text="", warnings=warnings, data=data)
        if fields.has_sibling_jobs:
            return _diarized_transcript(primary, received[JOB_KIND_DIARIZE], warnings)
        if not fields.want_segments:
            return replace(primary, data=None)
        data = {"segments": _payload_list(primary, "segments")}
        return replace(primary, data=data)
    except ValueError as exc:
        return replace(
            primary,
            status="error",
            text="",
            detail=str(exc),
            warnings=warnings,
            data=None,
        )


class _Call:
    """State of one client call: submitted jobs, final outcomes, spawn bookkeeping.

    Every input with jobs in flight has an entry in ``_pending`` (job kind ->
    job id of the jobs without a result yet) and collects the outcomes of its
    finished jobs in ``_received``. An input leaves both maps exactly once,
    right before its single outcome is reported.
    """

    def __init__(  # noqa: PLR0913 - every argument is a run_client seam
        self,
        request: ClientRequest,
        fields: _NormalizedRequest,
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
        self._fields = fields
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
        self._pending: dict[int, dict[JobKind, str]] = {}
        self._received: dict[int, dict[JobKind, InputOutcome]] = {}
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

    def _received_in_order(self, index: int) -> list[tuple[JobKind, InputOutcome]]:
        """Return the job outcomes received for an input, in job order."""
        received = self._received.get(index, {})
        return [
            (plan.kind, received[plan.kind])
            for plan in self._fields.plans
            if plan.kind in received
        ]

    def fail(self, index: int, status: InputStatus, detail: str) -> None:
        """Finish one input with a client-side failure.

        The warnings of the input's jobs that already finished are kept.
        """
        warnings = _merged_warnings(
            [outcome for _, outcome in self._received_in_order(index)]
        )
        self._received.pop(index, None)
        self.finish(
            InputOutcome(
                index=index,
                input_path=self._request.inputs[index],
                status=status,
                text="",
                detail=detail,
                warnings=warnings,
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

    def timeout_detail(self) -> str:
        """Return the detail of inputs still unfinished at the deadline."""
        if self._fields.mode == MODE_DIARIZE:
            return DIARIZE_TIMEOUT_DETAIL
        return TIMEOUT_DETAIL

    # * Submission.

    def _write_job(self, job: JobRecord) -> tuple[InputStatus, str] | None:
        """Write one job file.

        Args:
            job: The job to enqueue.

        Returns:
            ``None`` when the job was written, else the input's failure status
            and detail.

        """
        try:
            submit_job(self._paths, job)
        except OSError as exc:
            return (
                "unavailable",
                f"{UNAVAILABLE_DETAIL}: cannot write the job file: {exc}",
            )
        except JobValidationError as exc:
            return "error", exc.detail
        except ValueError as exc:
            return "error", str(exc)
        return None

    def submit(self, durations: Sequence[float | None]) -> None:
        """Write the jobs of every input; a failure finishes that input.

        An input whose window starts beyond its known media end fails without
        a job. When a later job of an input cannot be written, the input's
        jobs already written are cancelled and the input fails.

        Args:
            durations: Probed duration per input.

        """
        request = self._request
        fields = self._fields
        call_id = new_id()
        client_pid = os.getpid()
        last_submitted = -math.inf
        for index, input_path in enumerate(request.inputs):
            try:
                duration_s = _job_duration(fields.window, durations[index])
            except ValueError as exc:
                self.fail(index, "error", str(exc))
                continue
            jobs: dict[JobKind, str] = {}
            failure: tuple[JobKind, InputStatus, str] | None = None
            for plan in fields.plans:
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
                    duration_s=duration_s,
                    request=plan.request,
                    language=plan.language,
                    client_pid=client_pid,
                    priority=fields.priority,
                    kind=plan.kind,
                    want_segments=plan.want_segments,
                    decode=plan.decode,
                    window=fields.window,
                    diarize=plan.diarize,
                )
                problem = self._write_job(job)
                if problem is not None:
                    failure = (plan.kind, *problem)
                    break
                # * Registered at once, so an interrupt never leaks a written job.
                jobs[plan.kind] = job.job_id
                self._pending[index] = jobs
            if failure is not None:
                self._pending.pop(index, None)
                self._cancel_jobs(jobs.values())
                kind, status, detail = failure
                self.fail(index, status, _part_detail(fields, kind, detail))
        logger.debug(
            "call submitted: call_id=%s inputs=%d jobs=%d root=%s",
            call_id,
            len(self._pending),
            sum(len(jobs) for jobs in self._pending.values()),
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
        pending = {
            job_id for jobs in self._pending.values() for job_id in jobs.values()
        }
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

    def _settled(self, index: int) -> InputOutcome | None:
        """Return the outcome of an input once it is final.

        Args:
            index: Input position with an entry in ``_pending``.

        Returns:
            The failure as soon as one received job failed, the merged outcome
            once every job succeeded, or ``None`` while jobs are still running.

        """
        input_path = self._request.inputs[index]
        received = self._received_in_order(index)
        failure = _job_failure(index, input_path, self._fields, received)
        if failure is not None:
            return failure
        if self._pending[index]:
            return None
        return _merged_success(self._fields, dict(received))

    def collect_results(self) -> None:
        """Finish every pending input that became final, consuming its results.

        A failed job finishes its input at once; the input's other pending
        job is cancelled and its result is not awaited.
        """
        ready = self._ready_job_ids()
        for index in sorted(self._pending):
            jobs = self._pending[index]
            for kind, job_id in list(jobs.items()):
                if job_id not in ready:
                    continue
                result = read_result(self._paths, job_id)
                if result is None:
                    continue
                consume_result(self._paths, job_id)
                del jobs[kind]
                self._received.setdefault(index, {})[kind] = _outcome_from_result(
                    index, self._request.inputs[index], result, kind=kind
                )
            outcome = self._settled(index)
            if outcome is None:
                continue
            del self._pending[index]
            self._received.pop(index, None)
            self._cancel_jobs(jobs.values())
            self.finish(outcome)

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
                    self.abandon("timeout", self.timeout_detail())
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

    def _cancel_jobs(self, job_ids: Iterable[str]) -> None:
        """Write the cancel marker of every job in *job_ids*.

        A marker that cannot be written is logged; the supervisor then runs
        the job to its deadline or end.

        Args:
            job_ids: Jobs to cancel.

        """
        for job_id in job_ids:
            try:
                request_cancel(self._paths, job_id)
            except OSError as exc:
                logger.warning(
                    "cancel marker not written: job_id=%s error=%s", job_id, exc
                )

    def _cancel_pending(self) -> list[int]:
        """Write cancel markers for every pending job and forget them.

        Returns:
            The indexes of the inputs whose jobs were cancelled, in input order.

        """
        cancelled = sorted(self._pending)
        for index in cancelled:
            self._cancel_jobs(self._pending.pop(index).values())
        return cancelled

    def abandon(self, status: InputStatus, detail: str) -> None:
        """Cancel every pending job and finish its input with *status*."""
        for index in self._cancel_pending():
            self.fail(index, status, detail)

    def cancel_leftovers(self) -> None:
        """Cancel the jobs of an interrupted call without reporting outcomes."""
        if self._pending:
            logger.warning(
                "client call interrupted; cancelling %d job(s)",
                sum(len(jobs) for jobs in self._pending.values()),
            )
            self._cancel_pending()
            self._received.clear()


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
    """Transcribe (or diarize) the inputs of *request* through the supervisor.

    Args:
        request: The call; every option combination is validated before any
            job is written.
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
        fields,
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
        call.submit(durations)
        call.ensure_supervisor()
        call.wait()
    finally:
        # * Normal returns leave nothing pending; an exception (an interrupt, a
        # * failing callback) still tells the supervisor to drop the jobs.
        call.cancel_leftovers()
    return call.outcome()
