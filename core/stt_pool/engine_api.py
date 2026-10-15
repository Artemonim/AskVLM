"""Engine contract of the STT worker process (ML-free).

The worker process (:mod:`core.stt_pool.worker`) turns the ``spec`` message
into a :class:`WorkerSpec`, builds its engine through the factory named by
``engine_factory`` and drives it through the :class:`PoolEngine` protocol.
Every message shape here mirrors the worker protocol of :mod:`core.stt_pool.ipc`.

Engines catch per-job exceptions and emit an ``error`` result; the
``error_kind`` comes from :func:`error_kind_for_exception` and the detail from
:func:`error_detail_for_exception`.
"""

from __future__ import annotations

import importlib
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Protocol, Self, cast

from core.stt_errors import CudaOutOfMemoryError, DeviceUnavailableError
from core.stt_options import DiarizeOptions, MediaWindow, WhisperDecodeOptions
from core.stt_pool import ipc
from core.stt_pool.keys import (
    DEFAULT_PRIORITY,
    WorkerKey,
    device_for_backend,
    normalize_language,
    normalize_priority,
)
from core.stt_pool.queue import (
    JOB_KIND_STT,
    MAX_ERROR_DETAIL,
    job_kind_conflict,
    options_from_wire,
    options_to_wire,
    parse_job_kind,
)

if TYPE_CHECKING:
    from core.stt_pool.keys import Backend, Priority
    from core.stt_pool.queue import JobKind

EngineStatus = Literal["ok", "empty", "error", "cancelled"]

# * ``error_kind`` values produced by engines.
ERROR_KIND_TRANSCRIBE_FAILED = "transcribe_failed"
ERROR_KIND_CUDA_OOM = "cuda_oom"
ERROR_KIND_DEVICE_UNAVAILABLE = "device_unavailable"
ERROR_KIND_MEMORY_LIMIT = "memory_limit"
ERROR_KIND_CANCELLED = "cancelled"

# * ``error_kind`` of a finished result that is too large for its transport (a
# * worker protocol line or the result file); the worker or the supervisor
# * publishes it in place of the result.
ERROR_KIND_RESULT_TOO_LARGE = "result_too_large"

# * Production engine factory named in the ``spec`` message.
DEFAULT_ENGINE_FACTORY = "core.stt_pool.engines:create_engine"

_ENGINE_STATUSES: dict[str, EngineStatus] = {
    "ok": "ok",
    "empty": "empty",
    "error": "error",
    "cancelled": "cancelled",
}
_SPEC_DEVICES = frozenset({"cpu", "cuda"})
_IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"
_FACTORY_RE = re.compile(
    rf"({_IDENTIFIER}(?:\.{_IDENTIFIER})*):({_IDENTIFIER}(?:\.{_IDENTIFIER})*)"
)


def _bounded_detail(detail: str | None) -> str | None:
    """Return *detail* cut to :data:`MAX_ERROR_DETAIL` characters (head kept)."""
    if detail is None or len(detail) <= MAX_ERROR_DETAIL:
        return detail
    return detail[:MAX_ERROR_DETAIL]


def _require_type(message: Mapping[str, object], expected: str) -> None:
    """Validate the ``type`` field of a protocol message.

    Raises:
        ValueError: When the type differs from *expected*.

    """
    kind = message.get("type")
    if kind != expected:
        msg = f"expected a {expected!r} message, got type {kind!r}"
        raise ValueError(msg)


def _text(message: Mapping[str, object], name: str) -> str:
    """Return a required non-empty string field.

    Raises:
        ValueError: When the field is missing, not a string, or blank.

    """
    value = message.get(name)
    if not isinstance(value, str) or not value.strip():
        msg = f"field {name!r} must be a non-empty string, got {value!r}"
        raise ValueError(msg)
    return value


def _optional_text(message: Mapping[str, object], name: str) -> str | None:
    """Return an optional string field (``None`` when absent or null).

    Raises:
        ValueError: When the field is present but not a string.

    """
    value = message.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"field {name!r} must be a string or null, got {value!r}"
        raise ValueError(msg)  # noqa: TRY004 - malformed protocol fields are ValueError
    return value


def _integer(message: Mapping[str, object], name: str, minimum: int) -> int:
    """Return a required integer field that is at least *minimum*.

    Raises:
        ValueError: When the field is missing, not an integer (booleans
            included), or below *minimum*.

    """
    value = message.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        msg = f"field {name!r} must be an integer >= {minimum}, got {value!r}"
        raise ValueError(msg)
    return value


def _finite_float(value: object) -> float | None:
    """Return *value* as a finite float, or ``None`` when it is not one.

    Booleans, non-numbers, NaN, infinities and integers beyond the float range
    (``float(10**400)`` raises ``OverflowError``) all yield ``None``.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _number(message: Mapping[str, object], name: str, *, minimum: float) -> float:
    """Return a required finite number that is at least *minimum*.

    Raises:
        ValueError: When the field is missing, not a finite number (integers
            beyond the float range included), or below *minimum*.

    """
    value = message.get(name)
    number = _finite_float(value)
    if number is None or number < minimum:
        msg = f"field {name!r} must be a finite number >= {minimum}, got {value!r}"
        raise ValueError(msg)
    return number


def _optional_number(
    message: Mapping[str, object], name: str, *, minimum: float | None = None
) -> float | None:
    """Return an optional finite number (``None`` when absent or null).

    Raises:
        ValueError: When the field is present but not a finite number
            (integers beyond the float range included), or is below *minimum*.

    """
    if message.get(name) is None:
        return None
    return _number(message, name, minimum=-math.inf if minimum is None else minimum)


def _mapping(message: Mapping[str, object], name: str) -> Mapping[str, object]:
    """Return a required JSON-object field as a read-only mapping.

    Raises:
        ValueError: When the field is missing or not an object with string keys.

    """
    value = message.get(name)
    if not isinstance(value, Mapping) or not all(isinstance(k, str) for k in value):
        msg = f"field {name!r} must be a JSON object, got {value!r}"
        raise ValueError(msg)
    return MappingProxyType(dict(cast("Mapping[str, object]", value)))


def _optional_object(
    message: Mapping[str, object], name: str
) -> dict[str, object] | None:
    """Return an optional JSON-object field as a plain dict (``None`` when absent).

    Raises:
        ValueError: When the field is present but not an object with string keys.

    """
    value = message.get(name)
    if value is None:
        return None
    if not isinstance(value, Mapping) or not all(isinstance(k, str) for k in value):
        msg = (
            f"field {name!r} must be a JSON object or null, got {type(value).__name__}"
        )
        raise ValueError(msg)
    return dict(cast("Mapping[str, object]", value))


def job_message_fields(
    *,
    kind: JobKind,
    want_segments: bool,
    decode: WhisperDecodeOptions,
    window: MediaWindow,
    diarize: DiarizeOptions | None,
) -> dict[str, object]:
    """Return the ``job`` message entries of the per-job kind and options.

    Only entries that differ from their defaults are returned, so the message
    of a plain ``stt`` job is unchanged.

    Args:
        kind: Job kind.
        want_segments: Structured segments requested.
        decode: Whisper decoding overrides.
        window: Media window.
        diarize: Diarization settings, or ``None``.

    Returns:
        ``kind`` (unless ``stt``) plus the entries of
        :func:`~core.stt_pool.queue.options_to_wire`.

    """
    fields: dict[str, object] = {} if kind == JOB_KIND_STT else {"kind": kind}
    fields.update(
        options_to_wire(
            want_segments=want_segments, decode=decode, window=window, diarize=diarize
        )
    )
    return fields


def _backend(value: object) -> Backend:
    """Validate a backend token.

    Raises:
        ValueError: When *value* is neither ``cpu`` nor ``gpu``.

    """
    if value == "cpu":
        return "cpu"
    if value == "gpu":
        return "gpu"
    msg = f"field 'backend' must be 'cpu' or 'gpu', got {value!r}"
    raise ValueError(msg)


def _device(message: Mapping[str, object]) -> str:
    """Return the ``device`` field of a ``spec`` or ``loaded`` message.

    Raises:
        ValueError: When the field is neither ``cpu`` nor ``cuda``.

    """
    device = _text(message, "device")
    if device not in _SPEC_DEVICES:
        msg = f"field 'device' must be 'cpu' or 'cuda', got {device!r}"
        raise ValueError(msg)
    return device


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    """Configuration of one worker process (the ``spec`` message).

    Attributes:
        worker_id: Supervisor-assigned worker identifier.
        instance_id: Identifier of the supervisor instance.
        key: Worker configuration served by this process.
        capacity: Most jobs the supervisor runs on the worker at once.
        threads_per_slot: CPU threads per concurrently running CPU job.
        host_threads: Host CPU threads reserved for a GPU worker (``0`` on CPU).
        device: Strict device the engine loads on (``cpu`` or ``cuda``).
        options: Engine options (``whisper_model``, ``compute_type``,
            ``batch_lanes``, ``whisper_batch_max_clip_s``, ...).
        engine_factory: ``"module.path:callable"`` that builds the engine.

    """

    worker_id: str
    instance_id: str
    key: WorkerKey
    capacity: int
    threads_per_slot: int
    host_threads: int
    device: str
    options: Mapping[str, object]
    engine_factory: str

    @classmethod
    def from_message(cls, message: Mapping[str, object]) -> Self:
        """Parse and validate a ``spec`` message.

        Args:
            message: Decoded protocol message.

        Returns:
            The worker specification.

        Raises:
            ValueError: When the message is not a valid protocol-2 ``spec``, or
                its ``device`` does not match the key's backend.

        """
        _require_type(message, ipc.MSG_SPEC)
        protocol = message.get("protocol")
        if protocol != ipc.PROTOCOL_VERSION:
            msg = (
                f"unsupported worker protocol {protocol!r}; "
                f"expected {ipc.PROTOCOL_VERSION}"
            )
            raise ValueError(msg)
        raw_key = message.get("key")
        if not isinstance(raw_key, Mapping):
            msg = f"field 'key' must be a JSON object, got {raw_key!r}"
            raise ValueError(msg)  # noqa: TRY004 - malformed protocol fields are ValueError
        key = WorkerKey.from_dict(cast("Mapping[str, object]", raw_key))
        device = _device(message)
        expected_device = device_for_backend(key.backend)
        if device != expected_device:
            msg = (
                f"device {device!r} does not match backend {key.backend!r} of "
                f"{key.label} (expected {expected_device!r})"
            )
            raise ValueError(msg)
        return cls(
            worker_id=_text(message, "worker_id"),
            instance_id=_text(message, "instance_id"),
            key=key,
            capacity=_integer(message, "capacity", 1),
            threads_per_slot=_integer(message, "threads_per_slot", 1),
            host_threads=_integer(message, "host_threads", 0),
            device=device,
            options=_mapping(message, "options"),
            engine_factory=_text(message, "engine_factory"),
        )

    def to_message(self) -> dict[str, object]:
        """Return the ``spec`` protocol message.

        Returns:
            A JSON-serializable mapping accepted by :meth:`from_message`.

        """
        return {
            "type": ipc.MSG_SPEC,
            "protocol": ipc.PROTOCOL_VERSION,
            "instance_id": self.instance_id,
            "worker_id": self.worker_id,
            "key": self.key.to_dict(),
            "capacity": self.capacity,
            "threads_per_slot": self.threads_per_slot,
            "host_threads": self.host_threads,
            "device": self.device,
            "options": dict(self.options),
            "engine_factory": self.engine_factory,
        }


@dataclass(frozen=True, slots=True)
class WorkItem:
    """One job attempt dispatched to a worker (the ``job`` message).

    Attributes:
        job_id: Queue job identifier.
        attempt_id: ``"{job_id}:{attempt}"``; results echo it.
        input_path: Absolute path of the media file.
        language: Normalized language hint, or ``None`` for auto-detection.
        duration_s: Probed media duration, when known.
        deadline_at: Unix time after which the job is useless, when set.
        priority: Call priority class; batch engines fill lanes by it.
        call_id: Identifier of the client call the job belongs to (empty when
            the supervisor does not send one).
        seq: Supervisor admission sequence number of the job.
        kind: ``stt`` (transcription) or ``diarize`` (speaker diarization).
        want_segments: Return structured segments in the result ``data``.
        decode: Whisper-only decoding overrides.
        window: Time window of the media to process.
        diarize: Diarization settings of a ``diarize`` item (``None`` selects
            the defaults, see :attr:`diarize_options`).

    """

    job_id: str
    attempt_id: str
    input_path: Path
    language: str | None
    duration_s: float | None
    deadline_at: float | None
    priority: Priority = DEFAULT_PRIORITY
    call_id: str = ""
    seq: int = 0
    kind: JobKind = JOB_KIND_STT
    want_segments: bool = False
    decode: WhisperDecodeOptions = field(default_factory=WhisperDecodeOptions)
    window: MediaWindow = field(default_factory=MediaWindow)
    diarize: DiarizeOptions | None = None

    @property
    def diarize_options(self) -> DiarizeOptions:
        """Return the diarization settings to apply (the defaults when unset)."""
        return DiarizeOptions() if self.diarize is None else self.diarize

    @classmethod
    def from_message(cls, message: Mapping[str, object]) -> Self:
        """Parse and validate a ``job`` message.

        Messages without the per-job kind and option entries describe a plain
        ``stt`` job.

        Args:
            message: Decoded protocol message.

        Returns:
            The work item.

        Raises:
            ValueError: When a field is missing or invalid, or the kind does not
                take a given option.

        """
        _require_type(message, ipc.MSG_JOB)
        raw_kind = message.get("kind")
        kind = JOB_KIND_STT if raw_kind is None else parse_job_kind(raw_kind)
        extra = options_from_wire(message, where="")
        language = normalize_language(_optional_text(message, "language"))
        conflict = job_kind_conflict(
            kind,
            language=language,
            want_segments=extra.want_segments,
            decode=extra.decode,
            diarize=extra.diarize,
        )
        if conflict is not None:
            raise ValueError(conflict)
        return cls(
            job_id=_text(message, "job_id"),
            attempt_id=_text(message, "attempt_id"),
            input_path=Path(_text(message, "input_path")),
            language=language,
            duration_s=_optional_number(message, "duration_s", minimum=0.0),
            deadline_at=_optional_number(message, "deadline_at"),
            priority=normalize_priority(message.get("priority")),
            call_id=_optional_text(message, "call_id") or "",
            seq=0 if message.get("seq") is None else _integer(message, "seq", 0),
            kind=kind,
            want_segments=extra.want_segments,
            decode=extra.decode,
            window=extra.window,
            diarize=extra.diarize,
        )

    def to_message(self) -> dict[str, object]:
        """Return the ``job`` protocol message.

        Returns:
            A JSON-serializable mapping accepted by :meth:`from_message`; the
            kind and per-job options appear only when not default.

        """
        return {
            "type": ipc.MSG_JOB,
            "job_id": self.job_id,
            "attempt_id": self.attempt_id,
            "input_path": str(self.input_path),
            "language": self.language,
            "duration_s": self.duration_s,
            "deadline_at": self.deadline_at,
            "priority": self.priority,
            "call_id": self.call_id,
            "seq": self.seq,
            **job_message_fields(
                kind=self.kind,
                want_segments=self.want_segments,
                decode=self.decode,
                window=self.window,
                diarize=self.diarize,
            ),
        }


@dataclass(frozen=True, slots=True)
class EngineResult:
    """Outcome of one work item (the ``result`` message).

    Attributes:
        job_id: Queue job identifier.
        attempt_id: Attempt identifier of the work item.
        status: ``ok`` (non-empty text, or any finished diarization),
            ``empty``, ``error`` or ``cancelled``.
        text: Stripped transcript (empty unless ``ok``).
        error_kind: Error category for ``error``/``cancelled``, else ``None``.
        error_detail: Exception text (at most 4000 characters), else ``None``.
        backend: Backend class that ran the item.
        run_s: Wall-clock processing time.
        audio_s: Processed audio duration, when known.
        data: Structured payload of an ``ok``/``empty`` result (a JSON object:
            ``{"segments": [...]}`` for transcription, ``{"turns": [...]}`` for
            diarization), or ``None``.

    """

    job_id: str
    attempt_id: str
    status: EngineStatus
    text: str
    error_kind: str | None
    error_detail: str | None
    backend: Backend
    run_s: float
    audio_s: float | None
    data: Mapping[str, object] | None = None

    @classmethod
    def for_text(
        cls,
        item: WorkItem,
        text: str,
        *,
        backend: Backend,
        run_s: float,
        audio_s: float | None,
        data: Mapping[str, object] | None = None,
    ) -> Self:
        """Build the result of a finished transcription.

        Args:
            item: Work item that was processed.
            text: Transcript; it is stripped here.
            backend: Backend class that ran the item.
            run_s: Wall-clock processing time.
            audio_s: Processed audio duration, when known.
            data: Structured payload (for example the segments), kept for both
                ``ok`` and ``empty``.

        Returns:
            An ``ok`` result when the stripped text is non-empty, else ``empty``.

        """
        stripped = text.strip()
        return cls(
            job_id=item.job_id,
            attempt_id=item.attempt_id,
            status="ok" if stripped else "empty",
            text=stripped,
            error_kind=None,
            error_detail=None,
            backend=backend,
            run_s=run_s,
            audio_s=audio_s,
            data=data,
        )

    @classmethod
    def for_diarization(
        cls,
        item: WorkItem,
        data: Mapping[str, object],
        *,
        backend: Backend,
        run_s: float,
        audio_s: float | None,
    ) -> Self:
        """Build the result of a finished diarization.

        A diarization that found no speech is still a success: the status is
        ``ok`` whatever the payload holds, and the text is empty.

        Args:
            item: Work item that was processed.
            data: Structured payload (``{"turns": [...]}``).
            backend: Backend class that ran the item.
            run_s: Wall-clock processing time.
            audio_s: Processed audio duration, when known.

        Returns:
            An ``ok`` result carrying *data*.

        """
        return cls(
            job_id=item.job_id,
            attempt_id=item.attempt_id,
            status="ok",
            text="",
            error_kind=None,
            error_detail=None,
            backend=backend,
            run_s=run_s,
            audio_s=audio_s,
            data=data,
        )

    @classmethod
    def for_error(
        cls,
        item: WorkItem,
        exc: BaseException,
        *,
        backend: Backend,
        run_s: float,
        audio_s: float | None = None,
    ) -> Self:
        """Build the ``error`` result of a failed work item.

        Args:
            item: Work item that failed.
            exc: Exception raised while processing it.
            backend: Backend class that ran the item.
            run_s: Wall-clock time until the failure.
            audio_s: Processed audio duration, when known.

        Returns:
            An ``error`` result with the mapped ``error_kind`` and the bounded
            exception text.

        """
        return cls(
            job_id=item.job_id,
            attempt_id=item.attempt_id,
            status="error",
            text="",
            error_kind=error_kind_for_exception(exc),
            error_detail=error_detail_for_exception(exc),
            backend=backend,
            run_s=run_s,
            audio_s=audio_s,
        )

    @classmethod
    def for_cancel(cls, item: WorkItem, *, backend: Backend, run_s: float) -> Self:
        """Build the result of a cooperatively cancelled work item.

        Args:
            item: Work item that was cancelled.
            backend: Backend class that ran the item.
            run_s: Wall-clock time until the cancellation.

        Returns:
            A ``cancelled`` result with ``error_kind`` ``cancelled``.

        """
        return cls(
            job_id=item.job_id,
            attempt_id=item.attempt_id,
            status="cancelled",
            text="",
            error_kind=ERROR_KIND_CANCELLED,
            error_detail=None,
            backend=backend,
            run_s=run_s,
            audio_s=None,
        )

    @classmethod
    def from_message(cls, message: Mapping[str, object]) -> Self:
        """Parse and validate a ``result`` message.

        Args:
            message: Decoded protocol message.

        Returns:
            The engine result.

        Raises:
            ValueError: When a field is missing or invalid.

        """
        _require_type(message, ipc.MSG_RESULT)
        raw_status = message.get("status")
        status = (
            _ENGINE_STATUSES.get(raw_status) if isinstance(raw_status, str) else None
        )
        if status is None:
            msg = f"field 'status' is not a result status: {raw_status!r}"
            raise ValueError(msg)
        text = message.get("text")
        if not isinstance(text, str):
            msg = f"field 'text' must be a string, got {text!r}"
            raise ValueError(msg)  # noqa: TRY004 - malformed protocol fields are ValueError
        return cls(
            job_id=_text(message, "job_id"),
            attempt_id=_text(message, "attempt_id"),
            status=status,
            text=text,
            error_kind=_optional_text(message, "error_kind"),
            error_detail=_bounded_detail(_optional_text(message, "error_detail")),
            backend=_backend(message.get("backend")),
            run_s=_number(message, "run_s", minimum=0.0),
            audio_s=_optional_number(message, "audio_s", minimum=0.0),
            data=_optional_object(message, "data"),
        )

    def to_message(self) -> dict[str, object]:
        """Return the ``result`` protocol message.

        Returns:
            A JSON-serializable mapping accepted by :meth:`from_message`; the
            error detail is cut to 4000 characters and ``data`` is present only
            when set.

        """
        message: dict[str, object] = {
            "type": ipc.MSG_RESULT,
            "job_id": self.job_id,
            "attempt_id": self.attempt_id,
            "status": self.status,
            "text": self.text,
            "error_kind": self.error_kind,
            "error_detail": _bounded_detail(self.error_detail),
            "backend": self.backend,
            "run_s": self.run_s,
            "audio_s": self.audio_s,
        }
        if self.data is not None:
            message["data"] = dict(self.data)
        return message


@dataclass(frozen=True, slots=True)
class LoadInfo:
    """Outcome of a successful engine load (the ``loaded`` message).

    Attributes:
        device: Device the engine runs on (``cpu`` or ``cuda``).
        load_s: Load plus warm-up time.
        details: Diagnostics (model, threads, versions, ...).

    """

    device: str
    load_s: float
    details: Mapping[str, object]

    @classmethod
    def from_message(cls, message: Mapping[str, object]) -> Self:
        """Parse and validate a ``loaded`` message.

        Args:
            message: Decoded protocol message.

        Returns:
            The load information.

        Raises:
            ValueError: When a field is missing or invalid.

        """
        _require_type(message, ipc.MSG_LOADED)
        return cls(
            device=_device(message),
            load_s=_number(message, "load_s", minimum=0.0),
            details=_mapping(message, "details"),
        )

    def to_message(self) -> dict[str, object]:
        """Return the ``loaded`` protocol message.

        Returns:
            A JSON-serializable mapping accepted by :meth:`from_message`.

        """
        return {
            "type": ipc.MSG_LOADED,
            "device": self.device,
            "load_s": self.load_s,
            "details": dict(self.details),
        }


EmitFn = Callable[[EngineResult], None]


class PoolEngine(Protocol):
    """Speech engine driven by a worker process.

    The worker calls :meth:`load` once, then :meth:`start`, then any number of
    :meth:`submit` and :meth:`cancel` calls, and finally :meth:`close`. Every
    submitted item produces exactly one result through the ``emit`` callback.
    """

    def load(self) -> LoadInfo:
        """Load the model on the strict device and warm it up single-threaded.

        Returns:
            Load information for the ``loaded`` message.

        Raises:
            CudaOutOfMemoryError: When CUDA runs out of memory.
            DeviceUnavailableError: When the device cannot be used.

        """
        ...

    def start(self, emit: EmitFn) -> None:
        """Start the slot threads or the batch loop.

        Args:
            emit: Receives every result; it is thread-safe.

        """
        ...

    def submit(self, item: WorkItem) -> None:
        """Queue one work item without blocking.

        The supervisor never has more than ``capacity`` items in flight.

        Args:
            item: Work item.

        """
        ...

    def cancel(self, job_id: str, attempt_id: str) -> None:
        """Cancel a queued or running item cooperatively (result ``cancelled``).

        Args:
            job_id: Job identifier.
            attempt_id: Attempt identifier.

        """
        ...

    def close(self) -> None:
        """Stop the threads (bounded) and unload the model; idempotent."""
        ...


EngineFactory = Callable[[WorkerSpec], PoolEngine]


def load_engine_factory(dotted: str) -> EngineFactory:
    """Resolve an engine factory from ``"package.module:callable"``.

    The path comes only from the supervisor's own ``spec`` message (never from
    queue files), so importing it does not widen what a client controls.

    Args:
        dotted: Module path and attribute path separated by a colon; the
            attribute path may itself be dotted (``module:Class.create``).

    Returns:
        The factory callable.

    Raises:
        ValueError: When the format is invalid, importing the module raises
            any ``Exception`` (a missing module, a failed DLL load, an error at
            module import time), resolving the attribute path fails, the
            attribute does not exist, or it is not callable. ``KeyboardInterrupt``
            and ``SystemExit`` propagate unchanged.

    """
    text = str(dotted).strip()
    match = _FACTORY_RE.fullmatch(text)
    if match is None:
        msg = f"invalid engine factory {dotted!r}; expected 'package.module:callable'"
        raise ValueError(msg)
    module_name, attribute_path = match.group(1), match.group(2)
    try:
        target: object = importlib.import_module(module_name)
    except Exception as exc:
        # * Import-time code of an engine module may raise anything (OSError from
        # * a DLL load, RuntimeError, ...); the worker reports every case alike.
        msg = (
            f"cannot import engine factory module {module_name!r}: "
            f"{type(exc).__name__}: {exc}"
        )
        raise ValueError(msg) from exc
    for part in attribute_path.split("."):
        try:
            target = getattr(target, part)
        except AttributeError as exc:
            msg = f"engine factory {text!r} not found: {exc}"
            raise ValueError(msg) from exc
        except Exception as exc:
            # * Lazy module attributes may run arbitrary code on access.
            msg = (
                f"cannot resolve engine factory {text!r} at {part!r}: "
                f"{type(exc).__name__}: {exc}"
            )
            raise ValueError(msg) from exc
    if not callable(target):
        msg = f"engine factory {text!r} is not callable"
        raise ValueError(msg)  # noqa: TRY004 - one error type for every bad factory path
    return cast("EngineFactory", target)


def error_kind_for_exception(exc: BaseException) -> str:
    """Map an engine exception to a result ``error_kind``.

    Args:
        exc: Exception raised while loading or running a job.

    Returns:
        ``cuda_oom``, ``device_unavailable``, ``memory_limit`` (``MemoryError``,
        typically the Job Object commit cap) or ``transcribe_failed``.

    """
    if isinstance(exc, CudaOutOfMemoryError):
        return ERROR_KIND_CUDA_OOM
    if isinstance(exc, DeviceUnavailableError):
        return ERROR_KIND_DEVICE_UNAVAILABLE
    if isinstance(exc, MemoryError):
        return ERROR_KIND_MEMORY_LIMIT
    return ERROR_KIND_TRANSCRIBE_FAILED


def error_detail_for_exception(exc: BaseException) -> str:
    """Describe an engine exception for the result ``error_detail``.

    The head of the text is kept, so markers that clients classify on (for
    example ``Too long wav file``) survive the 4000-character bound.

    Args:
        exc: Exception raised while loading or running a job.

    Returns:
        ``"<type name>: <message>"``, at most 4000 characters.

    """
    try:
        message = str(exc)
    except Exception:  # noqa: BLE001 - a broken __str__ must not hide the failure
        message = "<unprintable exception message>"
    detail = f"{type(exc).__name__}: {message}"
    return detail[:MAX_ERROR_DETAIL]
