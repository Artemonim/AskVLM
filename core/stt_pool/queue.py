"""Queue v2: layout, job and result records, file operations, lock, heartbeat.

This module defines the on-disk contract between the ``external-transcribe``
client and the STT supervisor. Everything lives under ``<root>/v2`` so a legacy
v1 daemon that only scans ``<root>/incoming`` never sees v2 jobs::

    v2/incoming/<job_id>.json      client-written job, deleted after the result
    v2/attempts/<job_id>.json      supervisor-persisted attempt counter
    v2/results/<job_id>.json       supervisor-written terminal result (once)
    v2/cancel/<job_id>             client cancel marker (empty file)
    v2/control/supervisor.lock     OS-locked for the supervisor's lifetime
    v2/control/supervisor.heartbeat.json
    v2/control/supervisor.log      supervisor log
    v2/control/supervisor.out.log  stdout/stderr of the detached supervisor
    v2/logs/<worker_id>.log        per-worker stderr
    v2/state/estimates.json        learned statistics

Every JSON write is atomic and every read tolerates partial files. The module
imports no ML packages.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import re
import sys
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from time import sleep, time
from typing import TYPE_CHECKING, Literal, Self, TypeGuard

from core.settings import get_project_root
from core.stt_options import DiarizeOptions, MediaWindow, WhisperDecodeOptions
from core.stt_pool import fsio
from core.stt_pool.keys import (
    DEFAULT_PRIORITY,
    DIARIZE_PROVIDER,
    normalize_device_request,
    normalize_language,
    normalize_pool_provider,
    normalize_priority,
)
from core.stt_providers import is_gigaam_provider

if TYPE_CHECKING:
    from core.stt_pool.keys import DeviceRequest, Priority

logger = logging.getLogger(__name__)

JobKind = Literal["stt", "diarize"]

PROTOCOL_VERSION = 2
JOB_SCHEMA = "askvlm.stt.job"
RESULT_SCHEMA = "askvlm.stt.result"
HEARTBEAT_SCHEMA = "askvlm.stt.heartbeat"
RESULT_VERSION = 1

# * Job document versions. Version 2 adds job kinds and per-job options; a plain
# * ``stt`` job with default options is still written as version 1, so a
# * supervisor that predates version 2 keeps serving it.
JOB_VERSION = 2
JOB_VERSION_PLAIN = 1
JOB_KIND_STT: JobKind = "stt"
JOB_KIND_DIARIZE: JobKind = "diarize"
JOB_KINDS: frozenset[JobKind] = frozenset({JOB_KIND_STT, JOB_KIND_DIARIZE})

# * ``error_kind`` values of a rejected job file.
ERROR_KIND_INPUT_INVALID = "input_invalid"
ERROR_KIND_UNSUPPORTED_KIND = "unsupported_kind"
ERROR_KIND_UNSUPPORTED_VERSION = "unsupported_version"

# * ``error_kind`` of a result document whose fields cannot be trusted.
ERROR_KIND_PROTOCOL_ERROR = "protocol_error"

# * Override the queue root for tests or multi-tenant deployments.
ENV_QUEUE_DIR = "ASKVLM_EXTERNAL_QUEUE_DIR"

MAX_JOB_BYTES = 64 * 1024
MAX_ERROR_DETAIL = 4000
MAX_PARALLEL_LIMIT = 64

# * Largest result document a reader accepts; the supervisor publishes an
# * explicit ``result_too_large`` error instead of a bigger result.
MAX_RESULT_BYTES = fsio.DEFAULT_MAX_JSON_BYTES

# * Keys of the version-2 ``options`` object (version 1 knows only ``language``).
_OPTION_LANGUAGE = "language"
_OPTION_SEGMENTS = "segments"
_OPTION_DECODE = "decode"
_OPTION_WINDOW = "window"
_OPTION_DIARIZE = "diarize"
_V2_OPTION_KEYS: frozenset[str] = frozenset(
    {
        _OPTION_LANGUAGE,
        _OPTION_SEGMENTS,
        _OPTION_DECODE,
        _OPTION_WINDOW,
        _OPTION_DIARIZE,
    }
)

# * A supervisor is considered dead when its heartbeat is older than this many
# * seconds. Clients use it to decide whether to spawn a fresh supervisor.
DEFAULT_HEARTBEAT_STALE_S = 20.0

# * Pause before the single re-read of an unreadable but existing heartbeat file.
HEARTBEAT_RETRY_DELAY_S = 0.05

# * Unread results and orphaned per-job files older than this are reclaimed.
DEFAULT_PURGE_TTL_S = 3600.0

DEFAULT_WHISPER_MODEL_REQUEST = "small"
DEFAULT_COMPUTE_REQUEST = "auto"

ResultStatus = Literal["ok", "empty", "error", "cancelled", "expired", "unavailable"]

_RESULT_STATUSES: dict[str, ResultStatus] = {
    "ok": "ok",
    "empty": "empty",
    "error": "error",
    "cancelled": "cancelled",
    "expired": "expired",
    "unavailable": "unavailable",
}

_ID_RE = re.compile(r"[0-9a-f]{32}")
_JSON_FILE_RE = re.compile(r"([0-9a-f]{32})\.json")
_WORKER_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


def new_id() -> str:
    """Return a fresh queue-safe identifier.

    Returns:
        A 32-character lowercase hexadecimal string.

    """
    return uuid.uuid4().hex


def is_valid_id(value: object) -> TypeGuard[str]:
    """Return whether *value* is a well-formed job or call identifier.

    Args:
        value: Candidate identifier.

    Returns:
        ``True`` for a string of exactly 32 lowercase hexadecimal characters.

    """
    return isinstance(value, str) and _ID_RE.fullmatch(value) is not None


def _require_id(value: str, what: str = "job id") -> str:
    """Return *value* after validating it as an identifier.

    Args:
        value: Candidate identifier.
        what: Name used in the error message.

    Returns:
        The unchanged identifier.

    Raises:
        ValueError: When *value* is not 32 lowercase hexadecimal characters,
            which also rules out any path traversal.

    """
    if not is_valid_id(value):
        msg = f"Invalid {what}: {value!r}"
        raise ValueError(msg)
    return value


def resolve_queue_root(explicit: Path | None = None) -> Path:
    """Return the queue root directory.

    Resolution order: an explicit argument, then the :data:`ENV_QUEUE_DIR`
    environment variable, then ``<project>/.cache/external_queue``.

    Args:
        explicit: Optional caller-provided queue root.

    Returns:
        The resolved absolute queue root (not guaranteed to exist yet).

    """
    if explicit is not None:
        return explicit.resolve()
    env_value = os.environ.get(ENV_QUEUE_DIR)
    if env_value:
        return Path(env_value).resolve()
    return (get_project_root() / ".cache" / "external_queue").resolve()


@dataclass(frozen=True, slots=True)
class QueuePaths:
    """Paths of the v2 queue layout below a queue root.

    Attributes:
        root: The queue root directory.

    """

    root: Path

    @property
    def v2(self) -> Path:
        """Return the directory that holds the whole v2 layout."""
        return self.root / "v2"

    @property
    def incoming(self) -> Path:
        """Return the directory of client-written job files."""
        return self.v2 / "incoming"

    @property
    def attempts(self) -> Path:
        """Return the directory of persisted attempt counters."""
        return self.v2 / "attempts"

    @property
    def results(self) -> Path:
        """Return the directory of terminal result files."""
        return self.v2 / "results"

    @property
    def cancel(self) -> Path:
        """Return the directory of client cancel markers."""
        return self.v2 / "cancel"

    @property
    def control(self) -> Path:
        """Return the directory of supervisor control files."""
        return self.v2 / "control"

    @property
    def logs(self) -> Path:
        """Return the directory of per-worker logs."""
        return self.v2 / "logs"

    @property
    def state(self) -> Path:
        """Return the directory of persisted supervisor state."""
        return self.v2 / "state"

    @property
    def lock_file(self) -> Path:
        """Return the file the supervisor locks for its lifetime."""
        return self.control / "supervisor.lock"

    @property
    def heartbeat_file(self) -> Path:
        """Return the supervisor liveness beacon file."""
        return self.control / "supervisor.heartbeat.json"

    @property
    def supervisor_log(self) -> Path:
        """Return the rotating supervisor log file."""
        return self.control / "supervisor.log"

    @property
    def supervisor_out_log(self) -> Path:
        """Return the file that receives the detached supervisor's stdout/stderr."""
        return self.control / "supervisor.out.log"

    @property
    def estimates_file(self) -> Path:
        """Return the learned-statistics file."""
        return self.state / "estimates.json"

    def incoming_file(self, job_id: str) -> Path:
        """Return the incoming job file of *job_id*.

        Args:
            job_id: Job identifier.

        Returns:
            The file path.

        Raises:
            ValueError: When *job_id* is malformed.

        """
        return self.incoming / f"{_require_id(job_id)}.json"

    def attempts_file(self, job_id: str) -> Path:
        """Return the attempt counter file of *job_id*.

        Args:
            job_id: Job identifier.

        Returns:
            The file path.

        Raises:
            ValueError: When *job_id* is malformed.

        """
        return self.attempts / f"{_require_id(job_id)}.json"

    def result_file(self, job_id: str) -> Path:
        """Return the terminal result file of *job_id*.

        Args:
            job_id: Job identifier.

        Returns:
            The file path.

        Raises:
            ValueError: When *job_id* is malformed.

        """
        return self.results / f"{_require_id(job_id)}.json"

    def cancel_file(self, job_id: str) -> Path:
        """Return the cancel marker file of *job_id*.

        Args:
            job_id: Job identifier.

        Returns:
            The file path.

        Raises:
            ValueError: When *job_id* is malformed.

        """
        return self.cancel / _require_id(job_id)

    def worker_log(self, worker_id: str) -> Path:
        """Return the stderr log file of a worker.

        Args:
            worker_id: Supervisor-assigned worker identifier.

        Returns:
            The file path.

        Raises:
            ValueError: When *worker_id* is not a safe file-name token.

        """
        if _WORKER_ID_RE.fullmatch(worker_id) is None:
            msg = f"Invalid worker id: {worker_id!r}"
            raise ValueError(msg)
        return self.logs / f"{worker_id}.log"

    def ensure(self) -> None:
        """Create every directory of the layout if it is missing."""
        for directory in (
            self.incoming,
            self.attempts,
            self.results,
            self.cancel,
            self.control,
            self.logs,
            self.state,
        ):
            directory.mkdir(parents=True, exist_ok=True)


class JobValidationError(ValueError):
    """Raised when a job file does not satisfy the job contract.

    Attributes:
        error_kind: ``input_invalid``, ``unsupported_kind`` or
            ``unsupported_version``.
        detail: Human-readable description of the violation.

    """

    def __init__(self, error_kind: str, detail: str) -> None:
        """Create the error.

        Args:
            error_kind: Result ``error_kind`` the supervisor publishes.
            detail: Human-readable description of the violation.

        """
        super().__init__(f"{error_kind}: {detail}")
        self.error_kind = error_kind
        self.detail = detail


def _invalid(detail: str) -> JobValidationError:
    """Build an ``input_invalid`` validation error.

    Args:
        detail: Description of the violation.

    Returns:
        The error to raise.

    """
    return JobValidationError(ERROR_KIND_INPUT_INVALID, detail)


def _int_field(
    value: object, name: str, *, minimum: int, maximum: int | None = None
) -> int:
    """Validate an integer job field (booleans are rejected).

    Args:
        value: Candidate value.
        name: Field name for error messages.
        minimum: Smallest accepted value.
        maximum: Largest accepted value, or ``None`` for unbounded.

    Returns:
        The validated integer.

    Raises:
        JobValidationError: With ``input_invalid`` when the value is invalid.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{name} must be an integer"
        raise _invalid(msg)
    if value < minimum or (maximum is not None and value > maximum):
        bound = f">= {minimum}" if maximum is None else f"in {minimum}..{maximum}"
        msg = f"{name} must be {bound}"
        raise _invalid(msg)
    return value


def _number_field(value: object, name: str) -> float:
    """Validate a finite non-negative number job field.

    Args:
        value: Candidate value.
        name: Field name for error messages.

    Returns:
        The number as a float.

    Raises:
        JobValidationError: With ``input_invalid`` when the value is invalid.

    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{name} must be a number"
        raise _invalid(msg)
    try:
        number = float(value)
    except OverflowError:
        msg = f"{name} is out of range"
        raise _invalid(msg) from None
    if not math.isfinite(number) or number < 0:
        msg = f"{name} must be a finite non-negative number"
        raise _invalid(msg)
    return number


def _optional_number_field(value: object, name: str) -> float | None:
    """Validate an optional finite non-negative number job field.

    Args:
        value: Candidate value; ``None`` means absent.
        name: Field name for error messages.

    Returns:
        The number as a float, or ``None``.

    """
    return None if value is None else _number_field(value, name)


def _text_field(value: object, name: str, default: str) -> str:
    """Validate an optional text field of the request.

    Args:
        value: Candidate value; ``None`` or blank selects *default*.
        name: Field name for error messages.
        default: Value used when the field is absent or blank.

    Returns:
        The stripped text.

    Raises:
        JobValidationError: With ``input_invalid`` when the value is not text.

    """
    if value is None:
        return default
    if not isinstance(value, str):
        msg = f"{name} must be a string"
        raise _invalid(msg)
    return value.strip() or default


def _mapping_field(value: object, name: str) -> Mapping[str, object]:
    """Validate an optional JSON object field.

    Args:
        value: Candidate value; ``None`` means an empty object.
        name: Field name for error messages.

    Returns:
        The mapping.

    Raises:
        JobValidationError: With ``input_invalid`` when the value is not an object.

    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        msg = f"{name} must be an object"
        raise _invalid(msg)
    return value


@dataclass(frozen=True, slots=True)
class JobOptions:
    """Per-job options beyond the language, as carried by the wire forms.

    The job file keeps them in its ``options`` object and the worker ``job``
    message at its top level; both use :func:`options_to_wire` and
    :func:`options_from_wire`.

    Attributes:
        want_segments: Return structured segments next to the text.
        decode: Whisper-only decoding overrides.
        window: Time window of the media to process.
        diarize: Diarization settings of a ``diarize`` job (``None`` selects
            the defaults); never set for an ``stt`` job.

    """

    want_segments: bool = False
    decode: WhisperDecodeOptions = field(default_factory=WhisperDecodeOptions)
    window: MediaWindow = field(default_factory=MediaWindow)
    diarize: DiarizeOptions | None = None


def options_to_wire(
    *,
    want_segments: bool,
    decode: WhisperDecodeOptions,
    window: MediaWindow,
    diarize: DiarizeOptions | None,
) -> dict[str, object]:
    """Return the per-job options that differ from their defaults, as JSON.

    Args:
        want_segments: Structured segments requested.
        decode: Whisper decoding overrides.
        window: Media window.
        diarize: Diarization settings, or ``None``.

    Returns:
        ``segments`` (only ``true``), ``decode`` (the set fields), ``window``
        and ``diarize`` entries, each present only when not default; empty for
        a plain job.

    """
    wire: dict[str, object] = {}
    if want_segments:
        wire[_OPTION_SEGMENTS] = True
    if not decode.is_default:
        wire[_OPTION_DECODE] = decode.to_wire()
    if not window.is_default:
        wire[_OPTION_WINDOW] = window.to_wire()
    if diarize is not None:
        wire[_OPTION_DIARIZE] = diarize.to_wire()
    return wire


def options_from_wire(source: Mapping[str, object], *, where: str) -> JobOptions:
    """Parse the per-job options written by :func:`options_to_wire`.

    Absent and ``null`` entries take the defaults; other keys of *source* are
    not inspected here.

    Args:
        source: Job ``options`` object or ``job`` message.
        where: Prefix of the option names in error messages (``"options."``
            for a job file, ``""`` for a message).

    Returns:
        The validated options.

    Raises:
        ValueError: When an entry has the wrong type or an invalid value; the
            message names the option.

    """
    segments = source.get(_OPTION_SEGMENTS)
    if segments is not None and not isinstance(segments, bool):
        msg = f"{where}{_OPTION_SEGMENTS} must be true or false (got {segments!r})"
        raise ValueError(msg)
    try:
        raw = source.get(_OPTION_DECODE)
        decode = (
            WhisperDecodeOptions()
            if raw is None
            else WhisperDecodeOptions.from_wire(raw)
        )
    except ValueError as exc:
        msg = f"{where}{_OPTION_DECODE}: {exc}"
        raise ValueError(msg) from exc
    try:
        raw = source.get(_OPTION_WINDOW)
        window = MediaWindow() if raw is None else MediaWindow.from_wire(raw)
    except ValueError as exc:
        msg = f"{where}{_OPTION_WINDOW}: {exc}"
        raise ValueError(msg) from exc
    try:
        raw = source.get(_OPTION_DIARIZE)
        diarize = None if raw is None else DiarizeOptions.from_wire(raw)
    except ValueError as exc:
        msg = f"{where}{_OPTION_DIARIZE}: {exc}"
        raise ValueError(msg) from exc
    return JobOptions(
        want_segments=bool(segments),
        decode=decode,
        window=window,
        diarize=diarize,
    )


def job_kind_conflict(
    kind: JobKind,
    *,
    language: str | None,
    want_segments: bool,
    decode: WhisperDecodeOptions,
    diarize: DiarizeOptions | None,
) -> str | None:
    """Describe an option that the job kind does not take.

    A ``diarize`` job takes no language, segments flag or decode options; an
    ``stt`` job takes no diarization settings. The media window suits both.

    Args:
        kind: Job kind.
        language: Language hint.
        want_segments: Structured segments requested.
        decode: Whisper decoding overrides.
        diarize: Diarization settings.

    Returns:
        A description of the conflict, or ``None`` when the options fit.

    """
    if kind == JOB_KIND_DIARIZE:
        given = [
            name
            for name, present in (
                (_OPTION_LANGUAGE, language is not None),
                (_OPTION_SEGMENTS, want_segments),
                (_OPTION_DECODE, not decode.is_default),
            )
            if present
        ]
        if given:
            return f"diarize jobs do not take the STT option(s): {', '.join(given)}"
        return None
    if diarize is not None:
        return f"{_OPTION_DIARIZE} options are valid only for {JOB_KIND_DIARIZE} jobs"
    return None


def parse_job_kind(value: object) -> JobKind:
    """Validate a job kind token.

    Args:
        value: Candidate kind.

    Returns:
        ``stt`` or ``diarize``.

    Raises:
        ValueError: When *value* is not one of the job kinds.

    """
    for kind in (JOB_KIND_STT, JOB_KIND_DIARIZE):
        if value == kind:
            return kind
    msg = f"unsupported job kind {value!r}; expected stt or diarize"
    raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class JobRequest:
    """Provider, model and device selection of one job.

    Attributes:
        provider: Canonical STT provider id, or the diarization pseudo-provider
            of a ``diarize`` job.
        whisper_model: Requested Whisper model name.
        compute_type: Requested compute type.
        device: Normalized device request.

    """

    provider: str
    whisper_model: str
    compute_type: str
    device: DeviceRequest


@dataclass(frozen=True, slots=True)
class JobRecord:
    """One transcription or diarization job as written by a client.

    Attributes:
        job_id: Unique identifier of this job.
        call_id: Identifier shared by all jobs of one client call.
        call_index: Position of the input within the call.
        max_parallel: Cap on concurrently running jobs of the call, or ``None``.
        submitted_at: Unix timestamp of the submission.
        deadline_at: Unix timestamp after which the job expires, or ``None``.
        input_path: Absolute path of the media file.
        duration_s: Probed media duration in seconds, or ``None``.
        request: Provider, model and device selection.
        language: Normalized per-job language hint, or ``None``.
        client_pid: Process id of the submitting client, or ``None``.
        priority: Call priority class shared by all jobs of the call.
        kind: ``stt`` (transcription) or ``diarize`` (speaker diarization).
        want_segments: Return structured segments next to the text (``stt``).
        decode: Whisper-only decoding overrides (``stt`` with Whisper).
        window: Time window of the media to process.
        diarize: Diarization settings of a ``diarize`` job (``None`` selects
            the defaults); never set for an ``stt`` job.

    """

    job_id: str
    call_id: str
    call_index: int
    max_parallel: int | None
    submitted_at: float
    deadline_at: float | None
    input_path: str
    duration_s: float | None
    request: JobRequest
    language: str | None
    client_pid: int | None = None
    priority: Priority = DEFAULT_PRIORITY
    kind: JobKind = JOB_KIND_STT
    want_segments: bool = False
    decode: WhisperDecodeOptions = field(default_factory=WhisperDecodeOptions)
    window: MediaWindow = field(default_factory=MediaWindow)
    diarize: DiarizeOptions | None = None

    def options_wire(self) -> dict[str, object]:
        """Return the non-default per-job options (see :func:`options_to_wire`).

        Returns:
            The ``segments``/``decode``/``window``/``diarize`` entries that are
            set; empty for a plain job.

        """
        return options_to_wire(
            want_segments=self.want_segments,
            decode=self.decode,
            window=self.window,
            diarize=self.diarize,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the JSON document of this job.

        A plain ``stt`` job (every per-job option at its default) is written as
        version 1, exactly the layout that supervisors without job kinds read;
        any other job is version 2 with the set options added to ``options``.

        Returns:
            The ``askvlm.stt.job`` mapping.

        """
        extra = self.options_wire()
        plain = self.kind == JOB_KIND_STT and not extra
        return {
            "schema": JOB_SCHEMA,
            "version": JOB_VERSION_PLAIN if plain else JOB_VERSION,
            "kind": self.kind,
            "job_id": self.job_id,
            "call_id": self.call_id,
            "call_index": self.call_index,
            "max_parallel": self.max_parallel,
            "submitted_at": self.submitted_at,
            "deadline_at": self.deadline_at,
            "input_path": self.input_path,
            "duration_s": self.duration_s,
            "request": {
                "provider": self.request.provider,
                "whisper_model": self.request.whisper_model,
                "compute_type": self.request.compute_type,
                "device": self.request.device,
            },
            "options": {_OPTION_LANGUAGE: self.language, **extra},
            "priority": self.priority,
            "client": {"pid": self.client_pid},
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object], *, expected_job_id: str | None
    ) -> Self:
        """Validate a job document and build the record.

        Unknown top-level fields are ignored for forward compatibility. A
        version 1 document is read as before: its ``options`` carry only the
        language and any other option key is ignored. A version 2 document
        accepts only the known option keys and validates each strictly.

        Args:
            payload: Parsed job JSON.
            expected_job_id: Identifier from the file name that ``job_id`` must
                equal, or ``None`` to skip the comparison (the caller states
                explicitly that no file name vouches for the document).

        Returns:
            The validated record.

        Raises:
            JobValidationError: With ``unsupported_version`` for a newer job
                version, ``unsupported_kind`` for a kind the version does not
                know, and ``input_invalid`` for any other violation (including
                options the job kind or provider does not take).

        """
        version, kind = _check_job_header(payload)
        job_id = payload.get("job_id")
        if not is_valid_id(job_id):
            msg = "job_id must be 32 lowercase hexadecimal characters"
            raise _invalid(msg)
        if expected_job_id is not None and job_id != expected_job_id:
            msg = "job_id does not match the job file name"
            raise _invalid(msg)
        call_id = payload.get("call_id")
        if not is_valid_id(call_id):
            msg = "call_id must be 32 lowercase hexadecimal characters"
            raise _invalid(msg)
        max_parallel_raw = payload.get("max_parallel")
        client = _mapping_field(payload.get("client"), "client")
        pid_raw = client.get("pid")
        options = _mapping_field(payload.get("options"), "options")
        extra = (
            JobOptions() if version == JOB_VERSION_PLAIN else _parse_v2_options(options)
        )
        record = cls(
            job_id=job_id,
            call_id=call_id,
            call_index=_int_field(payload.get("call_index"), "call_index", minimum=0),
            max_parallel=(
                None
                if max_parallel_raw is None
                else _int_field(
                    max_parallel_raw,
                    "max_parallel",
                    minimum=1,
                    maximum=MAX_PARALLEL_LIMIT,
                )
            ),
            submitted_at=_number_field(payload.get("submitted_at"), "submitted_at"),
            deadline_at=_optional_number_field(
                payload.get("deadline_at"), "deadline_at"
            ),
            input_path=_parse_input_path(payload.get("input_path")),
            duration_s=_optional_number_field(payload.get("duration_s"), "duration_s"),
            request=_parse_request(payload.get("request"), kind=kind),
            language=_parse_language(options.get(_OPTION_LANGUAGE)),
            client_pid=(
                None
                if pid_raw is None
                else _int_field(pid_raw, "client.pid", minimum=0)
            ),
            priority=_parse_priority(payload.get("priority")),
            kind=kind,
            want_segments=extra.want_segments,
            decode=extra.decode,
            window=extra.window,
            diarize=extra.diarize,
        )
        check_job_options(record)
        return record


def check_job_options(job: JobRecord) -> None:
    """Validate that a job's kind, provider and per-job options fit together.

    Args:
        job: Job to check.

    Raises:
        JobValidationError: With ``unsupported_kind`` for an unknown kind and
            ``input_invalid`` when the provider does not serve the kind, the
            kind does not take a given option, or Whisper-only decode options
            meet a GigaAM provider.

    """
    try:
        kind = parse_job_kind(job.kind)
    except ValueError as exc:
        raise JobValidationError(ERROR_KIND_UNSUPPORTED_KIND, str(exc)) from exc
    provider_problem = _provider_kind_conflict(job.request.provider, kind)
    if provider_problem is not None:
        raise _invalid(provider_problem)
    conflict = job_kind_conflict(
        kind,
        language=job.language,
        want_segments=job.want_segments,
        decode=job.decode,
        diarize=job.diarize,
    )
    if conflict is not None:
        raise _invalid(conflict)
    if is_gigaam_provider(job.request.provider) and not job.decode.is_default:
        names = ", ".join(job.decode.to_wire())
        msg = (
            f"Whisper-only decode options ({names}) are not supported by "
            f"{job.request.provider}; omit them or use the whisper provider"
        )
        raise _invalid(msg)


def _provider_kind_conflict(provider: str, kind: JobKind) -> str | None:
    """Describe a provider that does not serve the job kind.

    Args:
        provider: Canonical provider of the request.
        kind: Job kind.

    Returns:
        A description of the mismatch, or ``None`` when the provider fits.

    """
    if kind == JOB_KIND_DIARIZE and provider != DIARIZE_PROVIDER:
        return (
            f"request.provider must be {DIARIZE_PROVIDER!r} for a "
            f"{JOB_KIND_DIARIZE} job (got {provider!r})"
        )
    if kind == JOB_KIND_STT and provider == DIARIZE_PROVIDER:
        return (
            f"request.provider {DIARIZE_PROVIDER!r} serves {JOB_KIND_DIARIZE} "
            f"jobs only, not {JOB_KIND_STT} jobs"
        )
    return None


def _parse_v2_options(options: Mapping[str, object]) -> JobOptions:
    """Validate the per-job options of a version 2 job document.

    Args:
        options: The document's ``options`` object.

    Returns:
        The parsed options (the language is parsed separately).

    Raises:
        JobValidationError: With ``input_invalid`` for an unknown option key or
            an invalid option value.

    """
    unknown = sorted(str(key) for key in options if key not in _V2_OPTION_KEYS)
    if unknown:
        msg = (
            f"unknown option(s) options.{', options.'.join(unknown)}; "
            f"expected: {', '.join(sorted(_V2_OPTION_KEYS))}"
        )
        raise _invalid(msg)
    try:
        return options_from_wire(options, where="options.")
    except ValueError as exc:
        raise _invalid(str(exc)) from exc


def _check_job_header(payload: Mapping[str, object]) -> tuple[int, JobKind]:
    """Validate the schema, version and kind markers of a job document.

    Args:
        payload: Parsed job JSON.

    Returns:
        The document version and job kind.

    Raises:
        JobValidationError: With ``unsupported_version``, ``unsupported_kind`` or
            ``input_invalid``.

    """
    if payload.get("schema") != JOB_SCHEMA:
        msg = f"schema must be {JOB_SCHEMA!r}"
        raise _invalid(msg)
    version = payload.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        msg = "version must be a positive integer"
        raise _invalid(msg)
    if version > JOB_VERSION:
        msg = f"job version {version} is newer than the supported {JOB_VERSION}"
        raise JobValidationError(ERROR_KIND_UNSUPPORTED_VERSION, msg)
    kind = payload.get("kind")
    if not isinstance(kind, str):
        msg = "kind must be a string"
        raise _invalid(msg)
    if version == JOB_VERSION_PLAIN and kind != JOB_KIND_STT:
        msg = f"unsupported job kind {kind!r} for job version {version}"
        raise JobValidationError(ERROR_KIND_UNSUPPORTED_KIND, msg)
    try:
        return version, parse_job_kind(kind)
    except ValueError as exc:
        raise JobValidationError(ERROR_KIND_UNSUPPORTED_KIND, str(exc)) from exc


def _parse_input_path(value: object) -> str:
    """Validate the media input path of a job.

    Args:
        value: Candidate ``input_path`` value.

    Returns:
        The path text.

    Raises:
        JobValidationError: With ``input_invalid`` when it is not an absolute path.

    """
    if not isinstance(value, str) or not value or "\x00" in value:
        msg = "input_path must be a non-empty string"
        raise _invalid(msg)
    if not Path(value).is_absolute():
        msg = "input_path must be an absolute path"
        raise _invalid(msg)
    return value


def _parse_request(value: object, *, kind: JobKind) -> JobRequest:
    """Validate the ``request`` object of a job.

    Args:
        value: Candidate ``request`` value.
        kind: Job kind, which decides the providers the request may name.

    Returns:
        The validated request with a canonical provider and device.

    Raises:
        JobValidationError: With ``input_invalid`` when a field is invalid or
            the provider does not serve the job kind.

    """
    if not isinstance(value, Mapping):
        msg = "request must be an object"
        raise _invalid(msg)
    raw_provider = value.get("provider")
    if not isinstance(raw_provider, str) or not raw_provider.strip():
        msg = "request.provider must be a non-empty string"
        raise _invalid(msg)
    try:
        provider = normalize_pool_provider(raw_provider)
    except ValueError as exc:
        msg = f"request.provider: {exc}"
        raise _invalid(msg) from exc
    provider_problem = _provider_kind_conflict(provider, kind)
    if provider_problem is not None:
        raise _invalid(provider_problem)
    raw_device = _text_field(value.get("device"), "request.device", "auto")
    try:
        device = normalize_device_request(provider, raw_device)
    except ValueError as exc:
        msg = f"request.device: {exc}"
        raise _invalid(msg) from exc
    return JobRequest(
        provider=provider,
        whisper_model=_text_field(
            value.get("whisper_model"),
            "request.whisper_model",
            DEFAULT_WHISPER_MODEL_REQUEST,
        ),
        compute_type=_text_field(
            value.get("compute_type"), "request.compute_type", DEFAULT_COMPUTE_REQUEST
        ),
        device=device,
    )


def _parse_language(value: object) -> str | None:
    """Validate the per-job language option.

    Args:
        value: Candidate ``options.language`` value.

    Returns:
        The normalized language, or ``None``.

    Raises:
        JobValidationError: With ``input_invalid`` when it is not a valid code.

    """
    if value is None:
        return None
    if not isinstance(value, str):
        msg = "options.language must be a string or null"
        raise _invalid(msg)
    try:
        return normalize_language(value)
    except ValueError as exc:
        msg = f"options.language: {exc}"
        raise _invalid(msg) from exc


def _parse_priority(value: object) -> Priority:
    """Validate the call priority of a job.

    Args:
        value: Candidate ``priority`` value; ``None`` or blank means ``normal``.

    Returns:
        The normalized priority class.

    Raises:
        JobValidationError: With ``input_invalid`` when it names no priority class.

    """
    try:
        return normalize_priority(value)
    except ValueError as exc:
        msg = f"priority: {exc}"
        raise _invalid(msg) from exc


def _optional_text(value: object) -> str | None:
    """Coerce a JSON value to ``str`` or ``None``.

    Args:
        value: Arbitrary deserialized JSON value.

    Returns:
        ``None`` for ``None``, otherwise the string form.

    """
    return None if value is None else str(value)


def _lenient_float(value: object, default: float) -> float:
    """Coerce a JSON value to a finite float.

    Args:
        value: Arbitrary deserialized JSON value.
        default: Value used when *value* is not a finite number.

    Returns:
        The number or *default*.

    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    try:
        number = float(value)
    except OverflowError:
        return default
    return number if math.isfinite(number) else default


@dataclass(frozen=True, slots=True)
class ResultRecord:
    """The terminal outcome of one job as written by the supervisor.

    Attributes:
        job_id: Identifier of the job this result belongs to.
        status: Terminal status.
        text: Transcript text (empty unless ``status`` is ``ok``).
        error_kind: Short machine label for non-success outcomes.
        error_detail: Worker error text, truncated on serialization.
        warnings: Non-fatal notices for the client's stderr.
        backend: ``cpu`` or ``gpu`` that actually ran the job.
        worker_key: Normalized worker key of the executing worker.
        worker_id: Identifier of the executing worker.
        attempts: Number of executions the job used.
        finished_at: Unix timestamp of the publication.
        timings: Named durations in seconds (``queued_s``, ``run_s``).
        data: Structured payload of an ``ok``/``empty`` result (a JSON object
            such as ``{"segments": [...]}`` or ``{"turns": [...]}``), or
            ``None``.

    """

    job_id: str
    status: ResultStatus
    text: str = ""
    error_kind: str | None = None
    error_detail: str | None = None
    warnings: tuple[str, ...] = ()
    backend: str | None = None
    worker_key: dict[str, str] | None = None
    worker_id: str | None = None
    attempts: int = 0
    finished_at: float = 0.0
    timings: dict[str, float] = field(default_factory=dict)
    data: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        """Return the JSON document of this result.

        Returns:
            The ``askvlm.stt.result`` mapping with ``error_detail`` truncated to
            :data:`MAX_ERROR_DETAIL` characters; ``data`` is present only when
            set.

        """
        detail = self.error_detail
        if detail is not None and len(detail) > MAX_ERROR_DETAIL:
            detail = detail[:MAX_ERROR_DETAIL]
        document: dict[str, object] = {
            "schema": RESULT_SCHEMA,
            "version": RESULT_VERSION,
            "job_id": self.job_id,
            "status": self.status,
            "text": self.text,
            "error_kind": self.error_kind,
            "error_detail": detail,
            "warnings": list(self.warnings),
            "backend": self.backend,
            "worker_key": None if self.worker_key is None else dict(self.worker_key),
            "worker_id": self.worker_id,
            "attempts": self.attempts,
            "finished_at": self.finished_at,
            "timings": dict(self.timings),
        }
        if self.data is not None:
            document["data"] = dict(self.data)
        return document

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        """Build a result from a document, tolerating missing or odd fields.

        An unknown ``status`` becomes ``error``. A ``data`` value that is
        neither an object nor ``null`` cannot be trusted either: the result
        becomes an ``error`` of kind ``protocol_error`` (its payload is never
        dropped silently into a seemingly complete success).

        Args:
            payload: Parsed result JSON; ``job_id`` must be a string.

        Returns:
            The reconstructed record.

        Raises:
            ValueError: When ``job_id`` is missing or not a string.

        """
        job_id = payload.get("job_id")
        if not isinstance(job_id, str):
            msg = "result job_id must be a string"
            raise ValueError(msg)  # noqa: TRY004 - payload values are untyped JSON
        warnings_raw = payload.get("warnings")
        worker_key_raw = payload.get("worker_key")
        timings_raw = payload.get("timings")
        attempts_raw = payload.get("attempts")
        status = _RESULT_STATUSES.get(str(payload.get("status")), "error")
        text = str(payload.get("text") or "")
        error_kind = _optional_text(payload.get("error_kind"))
        error_detail = _optional_text(payload.get("error_detail"))
        data_raw = payload.get("data")
        data: dict[str, object] | None = None
        if isinstance(data_raw, Mapping):
            data = {str(key): item for key, item in data_raw.items()}
        elif data_raw is not None:
            status = "error"
            text = ""
            error_kind = ERROR_KIND_PROTOCOL_ERROR
            error_detail = (
                "malformed result document: data must be a JSON object or null "
                f"(got {type(data_raw).__name__})"
            )
        return cls(
            job_id=job_id,
            status=status,
            text=text,
            error_kind=error_kind,
            error_detail=error_detail,
            warnings=(
                tuple(str(item) for item in warnings_raw)
                if isinstance(warnings_raw, list)
                else ()
            ),
            backend=_optional_text(payload.get("backend")),
            worker_key=(
                {str(key): str(item) for key, item in worker_key_raw.items()}
                if isinstance(worker_key_raw, Mapping)
                else None
            ),
            worker_id=_optional_text(payload.get("worker_id")),
            attempts=(
                attempts_raw
                if isinstance(attempts_raw, int) and not isinstance(attempts_raw, bool)
                else 0
            ),
            finished_at=_lenient_float(payload.get("finished_at"), 0.0),
            timings=(
                {
                    str(key): _lenient_float(item, 0.0)
                    for key, item in timings_raw.items()
                    if isinstance(item, (int, float)) and not isinstance(item, bool)
                }
                if isinstance(timings_raw, Mapping)
                else {}
            ),
            data=data,
        )


# * Best-effort file removal.


def _unlink_quiet(path: Path) -> None:
    """Remove *path* if present, logging instead of raising on OS errors.

    Args:
        path: File to remove.

    """
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.debug("queue cannot remove file: path=%s error=%s", path, exc)


# * Jobs.


def submit_job(paths: QueuePaths, job: JobRecord) -> None:
    """Place *job* into the incoming queue atomically.

    Args:
        paths: Queue layout.
        job: The job to enqueue.

    Raises:
        ValueError: When the job id is malformed, the job's kind, provider and
            options do not fit together (a :class:`JobValidationError`, the
            same check the supervisor applies), or the serialized job exceeds
            :data:`MAX_JOB_BYTES`.

    """
    target = paths.incoming_file(job.job_id)
    check_job_options(job)
    payload = job.to_dict()
    size = len(fsio.dumps_json_bytes(payload))
    if size > MAX_JOB_BYTES:
        msg = f"Job {job.job_id} is {size} bytes; the limit is {MAX_JOB_BYTES}"
        raise ValueError(msg)
    fsio.atomic_write_json(target, payload)


def list_incoming(paths: QueuePaths, *, limit: int = 1000) -> list[str]:
    """Return incoming job ids ordered oldest-submitted first.

    Jobs are ordered by ``(submitted_at, job_id)``. Files without a usable
    ``submitted_at`` (unreadable or malformed) come last, ordered by modification
    time. Only file names that are well-formed job ids are listed.

    Args:
        paths: Queue layout.
        limit: Maximum number of ids to return.

    Returns:
        Job identifiers in processing order.

    """
    ranked: list[tuple[int, float, str]] = []
    try:
        scan = os.scandir(paths.incoming)
    except OSError:
        return []
    with scan:
        for entry in scan:
            match = _JSON_FILE_RE.fullmatch(entry.name)
            if match is None:
                continue
            job_id = match.group(1)
            payload = fsio.read_json(Path(entry.path), max_bytes=MAX_JOB_BYTES)
            submitted = _lenient_float(
                None if payload is None else payload.get("submitted_at"), math.nan
            )
            if math.isfinite(submitted):
                ranked.append((0, submitted, job_id))
                continue
            try:
                modified = entry.stat().st_mtime
            except OSError:
                continue
            ranked.append((1, modified, job_id))
    ranked.sort()
    return [job_id for _, _, job_id in ranked[: max(0, limit)]]


def list_incoming_ids(paths: QueuePaths) -> list[str]:
    """Return the well-formed incoming job ids by file name alone.

    No job file is opened, so the scan stays cheap for periodic supervisor polls.

    Args:
        paths: Queue layout.

    Returns:
        Validated job identifiers sorted by name; empty when the directory is
        absent.

    """
    job_ids: list[str] = []
    try:
        scan = os.scandir(paths.incoming)
    except OSError:
        return job_ids
    with scan:
        for entry in scan:
            job_id = _json_file_job_id(entry.name)
            if job_id is not None:
                job_ids.append(job_id)
    job_ids.sort()
    return job_ids


def read_job_payload(paths: QueuePaths, job_id: str) -> dict[str, object] | None:
    """Read the raw job document of *job_id*.

    Args:
        paths: Queue layout.
        job_id: Job identifier.

    Returns:
        The parsed object, or ``None`` when absent, partial, oversized or not an
        object.

    """
    return fsio.read_json(paths.incoming_file(job_id), max_bytes=MAX_JOB_BYTES)


def job_file_age_s(paths: QueuePaths, job_id: str, *, now: float) -> float | None:
    """Return how long the incoming job file has existed unmodified.

    Args:
        paths: Queue layout.
        job_id: Job identifier.
        now: Current Unix timestamp.

    Returns:
        The age in seconds (never negative), or ``None`` when the file is absent.

    """
    try:
        modified = paths.incoming_file(job_id).stat().st_mtime
    except OSError:
        return None
    return max(0.0, now - modified)


# * Results.


def write_result(paths: QueuePaths, result: ResultRecord) -> bool:
    """Publish a terminal result unless one already exists.

    The first result wins; an existing result is never overwritten.

    Args:
        paths: Queue layout.
        result: The result to publish.

    Returns:
        ``True`` when this call published the result, ``False`` when a result
        already existed.

    """
    return fsio.atomic_create_json(paths.result_file(result.job_id), result.to_dict())


def read_result(paths: QueuePaths, job_id: str) -> ResultRecord | None:
    """Read the published result of *job_id*.

    Args:
        paths: Queue layout.
        job_id: Job identifier.

    Returns:
        The result, or ``None`` when it is absent or not yet fully readable.

    """
    payload = fsio.read_json(paths.result_file(job_id), max_bytes=MAX_RESULT_BYTES)
    if payload is None:
        return None
    # * The file name is authoritative for the job the result belongs to.
    return ResultRecord.from_dict({**payload, "job_id": job_id})


def has_result(paths: QueuePaths, job_id: str) -> bool:
    """Return whether a result file exists for *job_id*.

    Args:
        paths: Queue layout.
        job_id: Job identifier.

    Returns:
        ``True`` when the result file exists, readable or not.

    """
    return paths.result_file(job_id).exists()


def consume_result(paths: QueuePaths, job_id: str) -> None:
    """Delete the result file and cancel marker of a consumed job.

    Args:
        paths: Queue layout.
        job_id: Job identifier.

    """
    _unlink_quiet(paths.result_file(job_id))
    _unlink_quiet(paths.cancel_file(job_id))


# * Cancellation and per-job cleanup.


def request_cancel(paths: QueuePaths, job_id: str) -> None:
    """Write the cancel marker of *job_id*.

    Args:
        paths: Queue layout.
        job_id: Job identifier.

    """
    fsio.touch(paths.cancel_file(job_id))


def is_cancel_requested(paths: QueuePaths, job_id: str) -> bool:
    """Return whether a cancel marker exists for *job_id*.

    Args:
        paths: Queue layout.
        job_id: Job identifier.

    Returns:
        ``True`` when the client asked to cancel the job.

    """
    return paths.cancel_file(job_id).exists()


def delete_job_files(paths: QueuePaths, job_id: str) -> None:
    """Delete the incoming job, attempt counter and cancel marker of *job_id*.

    The result file is kept for the client.

    Args:
        paths: Queue layout.
        job_id: Job identifier.

    """
    _unlink_quiet(paths.incoming_file(job_id))
    _unlink_quiet(paths.attempts_file(job_id))
    _unlink_quiet(paths.cancel_file(job_id))


# * Attempt counters.


def read_attempts(paths: QueuePaths, job_id: str) -> int:
    """Read the persisted attempt counter of *job_id*.

    Args:
        paths: Queue layout.
        job_id: Job identifier.

    Returns:
        The stored counter, or ``0`` when absent or malformed.

    """
    payload = fsio.read_json(paths.attempts_file(job_id), max_bytes=MAX_JOB_BYTES)
    value = None if payload is None else payload.get("attempts")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0


def write_attempts(paths: QueuePaths, job_id: str, attempts: int) -> None:
    """Persist the attempt counter of *job_id*.

    Args:
        paths: Queue layout.
        job_id: Job identifier.
        attempts: Number of executions started so far.

    Raises:
        ValueError: When *attempts* is negative.

    """
    if attempts < 0:
        msg = f"attempts must be >= 0, got {attempts}"
        raise ValueError(msg)
    fsio.atomic_write_json(
        paths.attempts_file(job_id), {"attempts": attempts, "updated_at": time()}
    )


# * Housekeeping.


def _scan_files(directory: Path) -> list[tuple[str, float]]:
    """List regular files of *directory* with their modification times.

    Args:
        directory: Directory to scan.

    Returns:
        ``(file name, mtime)`` pairs; empty when the directory is absent.

    """
    found: list[tuple[str, float]] = []
    try:
        scan = os.scandir(directory)
    except OSError:
        return found
    with scan:
        for entry in scan:
            try:
                if entry.is_file(follow_symlinks=False):
                    found.append((entry.name, entry.stat().st_mtime))
            except OSError:
                continue
    return found


def purge_stale(
    paths: QueuePaths, *, now: float, ttl_s: float = DEFAULT_PURGE_TTL_S
) -> list[str]:
    """Reclaim unread results and orphaned per-job files older than *ttl_s*.

    A result older than the TTL is deleted together with its job file, attempt
    counter and cancel marker. Cancel markers and attempt counters older than
    the TTL that have neither an incoming job nor a result are deleted.
    Temporary files left by interrupted writes in the job, control and state
    directories are removed once they exceed the TTL.

    Args:
        paths: Queue layout.
        now: Current Unix timestamp.
        ttl_s: Minimum age in seconds before a file is reclaimed.

    Returns:
        Sorted identifiers of the jobs whose files were purged.

    """
    purged: set[str] = set()
    for name, modified in _scan_files(paths.results):
        job_id = _json_file_job_id(name)
        if job_id is None or now - modified <= ttl_s:
            continue
        delete_job_files(paths, job_id)
        _unlink_quiet(paths.result_file(job_id))
        purged.add(job_id)
    for name, modified in _scan_files(paths.cancel):
        if is_valid_id(name) and now - modified > ttl_s and _is_orphan(paths, name):
            _unlink_quiet(paths.cancel_file(name))
            purged.add(name)
    for name, modified in _scan_files(paths.attempts):
        job_id = _json_file_job_id(name)
        if job_id and now - modified > ttl_s and _is_orphan(paths, job_id):
            _unlink_quiet(paths.attempts_file(job_id))
            purged.add(job_id)
    for directory in (
        paths.incoming,
        paths.attempts,
        paths.results,
        paths.control,
        paths.state,
    ):
        for name, modified in _scan_files(directory):
            if name.endswith(".tmp") and now - modified > ttl_s:
                _unlink_quiet(directory / name)
    return sorted(purged)


def _json_file_job_id(name: str) -> str | None:
    """Extract the job id from a ``<job_id>.json`` file name.

    Args:
        name: File name.

    Returns:
        The job id, or ``None`` when *name* is not a well-formed job file name.

    """
    match = _JSON_FILE_RE.fullmatch(name)
    return None if match is None else match.group(1)


def _is_orphan(paths: QueuePaths, job_id: str) -> bool:
    """Return whether *job_id* has neither an incoming job nor a result.

    Args:
        paths: Queue layout.
        job_id: Job identifier.

    Returns:
        ``True`` when no job file and no result file exists.

    """
    return not paths.incoming_file(job_id).exists() and not has_result(paths, job_id)


# * Supervisor lock.


if sys.platform == "win32":
    import msvcrt

    def _lock_nonblocking(fd: int) -> bool:
        """Take a non-blocking exclusive lock on the first byte of *fd*.

        Args:
            fd: Open file descriptor.

        Returns:
            ``True`` when the lock was acquired, ``False`` when another holder
            owns it.

        """
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    def _unlock(fd: int) -> None:
        """Release the lock taken by :func:`_lock_nonblocking`.

        Args:
            fd: Open file descriptor that holds the lock.

        """
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError as exc:
            # * Closing the descriptor releases the lock regardless.
            logger.debug("supervisor lock unlock failed: error=%s", exc)

else:
    import fcntl

    def _lock_nonblocking(fd: int) -> bool:
        """Take a non-blocking exclusive lock on *fd*.

        Args:
            fd: Open file descriptor.

        Returns:
            ``True`` when the lock was acquired, ``False`` when another holder
            owns it.

        """
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True

    def _unlock(fd: int) -> None:
        """Release the lock taken by :func:`_lock_nonblocking`.

        Args:
            fd: Open file descriptor that holds the lock.

        """
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError as exc:
            # * Closing the descriptor releases the lock regardless.
            logger.debug("supervisor lock unlock failed: error=%s", exc)


class SupervisorLock:
    """Single-owner OS lock of the queue, held for the supervisor's lifetime.

    The lock is a non-blocking exclusive lock on one byte of
    ``control/supervisor.lock`` (``msvcrt.locking`` on Windows, ``fcntl.flock``
    elsewhere). The operating system releases it when the owning process dies,
    and a stale heartbeat never allows a takeover. The descriptor is not
    inherited by child processes.
    """

    def __init__(self, paths: QueuePaths) -> None:
        """Create an unheld lock for a queue.

        Args:
            paths: Queue layout.

        """
        self._paths = paths
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        """Return whether this object currently holds the lock."""
        return self._fd is not None

    def try_acquire(self) -> bool:
        """Try to take the lock without blocking.

        Returns:
            ``True`` when this object holds the lock (including when it already
            did), ``False`` when another holder owns it.

        """
        if self._fd is not None:
            return True
        lock_path = self._paths.lock_file
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
        if not _lock_nonblocking(fd):
            os.close(fd)
            return False
        self._fd = fd
        return True

    def release(self) -> None:
        """Release the lock; calling it again or without holding is a no-op."""
        fd = self._fd
        if fd is None:
            return
        self._fd = None
        _unlock(fd)
        os.close(fd)

    def __del__(self) -> None:
        """Release a lock that is still held when the object is collected."""
        with contextlib.suppress(Exception):
            self.release()

    def __enter__(self) -> Self:
        """Enter the context without acquiring the lock.

        Returns:
            This lock object.

        """
        return self

    def __exit__(self, *_exc: object) -> None:
        """Leave the context, releasing the lock."""
        self.release()


# * Heartbeat.


def write_heartbeat(paths: QueuePaths, payload: Mapping[str, object]) -> None:
    """Write the supervisor liveness beacon atomically.

    Args:
        paths: Queue layout.
        payload: Diagnostic fields; ``schema``, ``protocol``, ``ts`` and ``pid``
            are set by this function.

    """
    beacon = dict(payload)
    beacon["schema"] = HEARTBEAT_SCHEMA
    beacon["protocol"] = PROTOCOL_VERSION
    beacon["ts"] = time()
    beacon["pid"] = os.getpid()
    fsio.atomic_write_json(paths.heartbeat_file, beacon)


def read_heartbeat(paths: QueuePaths) -> dict[str, object] | None:
    """Read the supervisor liveness beacon.

    A read that fails while the file exists is retried once after
    :data:`HEARTBEAT_RETRY_DELAY_S`, because on Windows a concurrent atomic
    replace can make the file briefly unreadable.

    Args:
        paths: Queue layout.

    Returns:
        The beacon mapping, or ``None`` when absent or not readable.

    """
    beacon = fsio.read_json(paths.heartbeat_file)
    if beacon is None and paths.heartbeat_file.exists():
        sleep(HEARTBEAT_RETRY_DELAY_S)
        beacon = fsio.read_json(paths.heartbeat_file)
    return beacon


def clear_heartbeat(paths: QueuePaths) -> None:
    """Remove the liveness beacon so clients see the supervisor as gone at once.

    Args:
        paths: Queue layout.

    """
    _unlink_quiet(paths.heartbeat_file)


def is_supervisor_alive(
    paths: QueuePaths,
    *,
    now: float,
    stale_after_s: float = DEFAULT_HEARTBEAT_STALE_S,
) -> bool:
    """Return whether a supervisor with a fresh heartbeat serves the queue.

    Args:
        paths: Queue layout.
        now: Current Unix timestamp.
        stale_after_s: Largest heartbeat age considered alive.

    Returns:
        ``True`` when a heartbeat with a numeric ``ts`` no older than
        *stale_after_s* exists.

    """
    beacon = read_heartbeat(paths)
    if beacon is None:
        return False
    stamp = _lenient_float(beacon.get("ts"), math.nan)
    if not math.isfinite(stamp):
        return False
    return (now - stamp) <= stale_after_s
