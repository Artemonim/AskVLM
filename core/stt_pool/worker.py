"""Worker process of the STT pool, started as ``python -m core.stt_pool.worker``.

The supervisor starts one process per worker with piped stdin and stdout and
stderr appended to the worker's log, assigns it to its Job Object, and only
then sends the ``spec`` line. The process:

1. moves the protocol off file descriptor 1 (:meth:`WorkerChannel.install`)
   before anything else can write, ignores console interrupts and logs to
   stderr;
2. reads the ``spec`` (a missing or malformed one is answered with ``fatal``
   stage ``spec``) and answers ``hello``;
3. builds the engine through the factory the spec names, loads it and starts
   it with a thread-safe result emitter, then answers ``loaded``, which
   therefore means "fully started and ready for jobs". Any failure up to that
   point is answered with ``fatal`` stage ``load``; a failing load or device
   check makes no further engine call, while a failing ``start()`` seals the
   emitter and gives the already started engine a bounded ``close()``;
4. dispatches ``job``, ``cancel`` and ``shutdown`` from stdin until a
   ``shutdown`` (bounded ``close()``, ``bye``, exit 0) or stdin EOF (the
   supervisor is gone: exit :data:`~core.stt_pool.ipc.EXIT_PARENT_GONE` at
   once, without waiting for running jobs).

Every exit goes through :func:`os._exit` after stderr is flushed: engine
threads may still run, and a regular interpreter shutdown could hang on them.
Until the spec arrives the process imports no ML package and no engine module;
the engine module is imported only through the factory named by the spec.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import sys
import threading
import time
from typing import IO, TYPE_CHECKING, Literal, NoReturn, TypeGuard

from core.stt_errors import DeviceUnavailableError
from core.stt_pool import ipc
from core.stt_pool.engine_api import (
    ERROR_KIND_RESULT_TOO_LARGE,
    ERROR_KIND_TRANSCRIBE_FAILED,
    EngineResult,
    WorkerSpec,
    WorkItem,
    error_detail_for_exception,
    error_kind_for_exception,
    load_engine_factory,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from core.stt_pool.engine_api import PoolEngine
    from core.stt_pool.keys import Backend

# * Run with ``-m`` the module is ``__main__``; records keep the package name.
logger = logging.getLogger("core.stt_pool.worker")

FatalStage = Literal["spec", "load", "run"]

# * ``error_kind`` values produced by the worker itself (engine failures keep
# * the kinds of :func:`~core.stt_pool.engine_api.error_kind_for_exception`).
ERROR_KIND_WORKER_LOAD_FAILED = "worker_load_failed"
ERROR_KIND_PROTOCOL_ERROR = "protocol_error"

# * Bound of ``engine.close()`` on ``shutdown``. Together with the emitter seal
# * (at most ``_SEAL_TIMEOUT_S``) it stays below the supervisor's default
# * ``stop_grace_s`` (20 s), so ``bye`` arrives before a forced stop.
SHUTDOWN_CLOSE_TIMEOUT_S = 12.0
# * Bound of the best-effort ``engine.close()`` after a fatal error once the
# * engine owns the result emitter (a failing ``start()`` or a run-time failure).
FATAL_CLOSE_TIMEOUT_S = 3.0

# * Longest wait for an in-flight result write before the emitter is sealed.
_SEAL_TIMEOUT_S = 5.0
# * Characters of an undecodable first line quoted in the ``fatal`` detail.
_EXCERPT_CHARS = 200
_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
_CONSOLE_INTERRUPTS = ("SIGINT", "SIGBREAK")
_LOG_FORMAT = (
    "%(asctime)s %(levelname)s [worker {worker_id} pid %(process)d "
    "%(threadName)s] %(name)s: %(message)s"
)


class WorkerProtocolError(ValueError):
    """The supervisor sent a line the worker protocol does not allow."""


# * ---------------------------------------------------------------------------
# * Process plumbing
# * ---------------------------------------------------------------------------


def _log_formatter(worker_id: str) -> logging.Formatter:
    """Return the log formatter that tags every record with *worker_id*."""
    # * The worker id becomes part of a %-style format string.
    escaped = worker_id.replace("%", "%%")
    return logging.Formatter(_LOG_FORMAT.format(worker_id=escaped))


def _configure_logging() -> logging.Handler:
    """Send every log record of the process to stderr (the worker's log).

    Returns:
        The installed handler; its formatter gets the worker id once the spec
        names it.

    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_log_formatter("-"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    return handler


def _ignore_console_interrupts() -> None:
    """Ignore Ctrl+C and Ctrl+Break aimed at the supervisor's console.

    The supervisor stops its workers through the protocol or the Job Object; a
    console interrupt must not kill a worker in the middle of a job.
    """
    for name in _CONSOLE_INTERRUPTS:
        signum = getattr(signal, name, None)
        if signum is None:
            continue
        with contextlib.suppress(OSError, ValueError, RuntimeError):
            signal.signal(signum, signal.SIG_IGN)


def _terminate(code: int) -> NoReturn:
    """Flush the log and end the process at once with *code*.

    ``os._exit`` skips the interpreter shutdown, which could hang on engine
    threads or on a reader blocked in ``readline``.
    """
    for handler in logging.getLogger().handlers:
        with contextlib.suppress(Exception):
            handler.flush()
    with contextlib.suppress(OSError, ValueError, AttributeError):
        sys.stderr.flush()
    os._exit(code)


def _parent_gone(reason: str) -> NoReturn:
    """End the process because the supervisor is gone (stdin EOF or failure)."""
    logger.warning("supervisor is gone (%s); exiting at once", reason)
    _terminate(ipc.EXIT_PARENT_GONE)


def _stdin_stream() -> IO[bytes] | IO[str]:
    """Return the binary stdin stream (the text stream when there is none).

    Without any stdin the supervisor cannot reach the worker, which is
    handled like a supervisor that is gone.
    """
    stdin = sys.stdin
    if stdin is None:
        _parent_gone("no stdin")
    stream: IO[bytes] | IO[str] = getattr(stdin, "buffer", None) or stdin
    return stream


def _excerpt(line: str) -> str:
    """Return a short, printable excerpt of a protocol line."""
    text = line.strip()
    if len(text) > _EXCERPT_CHARS:
        text = text[:_EXCERPT_CHARS] + "..."
    return ascii(text)


def _read_first_message(stream: IO[bytes] | IO[str]) -> dict[str, object]:
    """Read the first supervisor line, which must be one decodable message.

    Unlike later lines, an undecodable first line is not skipped: the worker
    cannot be configured without its spec.

    Args:
        stream: The worker's stdin.

    Returns:
        The decoded message (its type is checked by the spec parser).

    Raises:
        WorkerProtocolError: When the line is oversized or not a supervisor
            message.

    """
    try:
        line = next(ipc.iter_lines(stream))
    except StopIteration:
        _parent_gone("stdin closed before the spec")
    except (OSError, ValueError) as exc:
        _parent_gone(f"stdin read failed before the spec: {exc}")
    if line is None:
        msg = f"the first protocol line exceeds {ipc.MAX_LINE_BYTES} bytes"
        raise WorkerProtocolError(msg)
    message = ipc.decode_message(line, allowed=ipc.SUPERVISOR_TO_WORKER)
    if message is None:
        msg = f"the first protocol line is not a supervisor message: {_excerpt(line)}"
        raise WorkerProtocolError(msg)
    return message


def _supervisor_messages(stream: IO[bytes] | IO[str]) -> Iterator[dict[str, object]]:
    """Yield supervisor messages until EOF; a failed read ends them like EOF.

    Undecodable lines and unknown message types are logged and skipped by
    :func:`~core.stt_pool.ipc.iter_messages`.
    """
    try:
        yield from ipc.iter_messages(
            stream, allowed=ipc.SUPERVISOR_TO_WORKER, source="supervisor"
        )
    except (OSError, ValueError) as exc:
        logger.warning("stdin read failed: %s", exc)


def _call_bounded(action: Callable[[], None], *, timeout_s: float, name: str) -> bool:
    """Run *action* on a daemon thread and wait at most *timeout_s* for it.

    Args:
        action: Callable to run; its exceptions are logged.
        timeout_s: Longest wait in seconds.
        name: Thread name, also used in log records.

    Returns:
        Whether *action* returned or raised within *timeout_s*.

    """

    def _target() -> None:
        try:
            action()
        except Exception:
            logger.exception("%s failed", name)

    thread = threading.Thread(target=_target, name=name, daemon=True)
    thread.start()
    thread.join(timeout_s)
    return not thread.is_alive()


def _usable_id(value: object) -> TypeGuard[str]:
    """Return whether *value* can identify a job or an attempt."""
    return isinstance(value, str) and bool(value.strip())


def _fatal_error_kind(stage: FatalStage, exc: BaseException) -> str:
    """Map the exception that ends the worker to a ``fatal`` ``error_kind``.

    Typed device and memory failures keep their kinds (``cuda_oom``,
    ``device_unavailable``, ``memory_limit``); any other failure is
    ``worker_load_failed`` at the load stage and ``protocol_error`` otherwise.
    """
    kind = error_kind_for_exception(exc)
    if kind != ERROR_KIND_TRANSCRIBE_FAILED:
        return kind
    if stage == "load":
        return ERROR_KIND_WORKER_LOAD_FAILED
    return ERROR_KIND_PROTOCOL_ERROR


def _thread_env() -> dict[str, str | None]:
    """Return the thread-count environment the supervisor set for the worker."""
    return {name: os.environ.get(name) for name in _THREAD_ENV_VARS}


# * ---------------------------------------------------------------------------
# * Result emission
# * ---------------------------------------------------------------------------


class _ResultEmitter:
    """Thread-safe ``result`` writer handed to the engine as its ``emit``.

    After :meth:`seal` nothing more is written, so ``bye`` or ``fatal`` stays
    the last message of the process. A result that cannot be encoded is
    replaced by a ``protocol_error`` result for the same attempt, and a result
    whose protocol line would exceed :data:`~core.stt_pool.ipc.MAX_LINE_BYTES`
    (the supervisor's reader would skip it) by a ``result_too_large`` result,
    so the supervisor never waits for an answer that was lost.
    """

    def __init__(self, channel: ipc.WorkerChannel, backend: Backend) -> None:
        self._channel = channel
        self._backend = backend
        self._lock = threading.Lock()
        self._sealed = False

    def emit(self, result: EngineResult) -> None:
        """Send one engine result (called from engine threads).

        Args:
            result: Engine result.

        """
        with self._lock:
            if self._sealed:
                logger.warning(
                    "result dropped, the worker is stopping: job_id=%s attempt_id=%s",
                    getattr(result, "job_id", None),
                    getattr(result, "attempt_id", None),
                )
                return
            try:
                message = result.to_message()
                size = _line_bytes(message)
                if size <= ipc.MAX_LINE_BYTES:
                    self._channel.send(message)
            except Exception as exc:
                # * The engine thread must survive and the attempt still needs
                # * its one answer.
                logger.exception("engine result cannot be encoded")
                self._send_replacement(result, exc)
                return
            if size > ipc.MAX_LINE_BYTES:
                self._send_oversized(result, size)

    def emit_protocol_error(self, job_id: str, attempt_id: str, detail: str) -> None:
        """Answer one attempt with an ``error`` result of kind ``protocol_error``.

        Args:
            job_id: Job identifier.
            attempt_id: Attempt identifier.
            detail: Error detail for the supervisor.

        """
        self.emit(self._protocol_error(job_id, attempt_id, detail))

    def seal(self) -> None:
        """Stop writing results; waits briefly for a write in progress."""
        acquired = self._lock.acquire(timeout=_SEAL_TIMEOUT_S)
        try:
            self._sealed = True
        finally:
            if acquired:
                self._lock.release()
        if not acquired:
            logger.warning(
                "a result write was still in progress after %.0f s; sealing anyway",
                _SEAL_TIMEOUT_S,
            )

    def _protocol_error(
        self, job_id: str, attempt_id: str, detail: str
    ) -> EngineResult:
        """Build an ``error`` result of kind ``protocol_error``."""
        return EngineResult(
            job_id=job_id,
            attempt_id=attempt_id,
            status="error",
            text="",
            error_kind=ERROR_KIND_PROTOCOL_ERROR,
            error_detail=detail,
            backend=self._backend,
            run_s=0.0,
            audio_s=None,
        )

    def _send_replacement(self, result: object, exc: Exception) -> None:
        """Answer the attempt of an unencodable *result* (the caller holds the lock)."""
        job_id = getattr(result, "job_id", None)
        attempt_id = getattr(result, "attempt_id", None)
        if not (_usable_id(job_id) and _usable_id(attempt_id)):
            logger.error("unencodable engine result has no usable job or attempt id")
            return
        detail = f"engine result cannot be encoded: {error_detail_for_exception(exc)}"
        replacement = self._protocol_error(job_id, attempt_id, detail)
        self._channel.send(replacement.to_message())

    def _send_oversized(self, result: EngineResult, size: int) -> None:
        """Answer the attempt of a too large *result* (the caller holds the lock)."""
        logger.error(
            "engine result too large for the protocol; sending %s instead: "
            "job_id=%s attempt_id=%s status=%s size_bytes=%d limit_bytes=%d",
            ERROR_KIND_RESULT_TOO_LARGE,
            result.job_id,
            result.attempt_id,
            result.status,
            size,
            ipc.MAX_LINE_BYTES,
        )
        replacement = EngineResult(
            job_id=result.job_id,
            attempt_id=result.attempt_id,
            status="error",
            text="",
            error_kind=ERROR_KIND_RESULT_TOO_LARGE,
            error_detail=(
                f"the {result.status} result is {size} bytes, over the worker "
                f"protocol line limit of {ipc.MAX_LINE_BYTES} bytes"
            ),
            backend=self._backend,
            run_s=result.run_s,
            audio_s=result.audio_s,
        )
        self._channel.send(replacement.to_message())


def _line_bytes(message: Mapping[str, object]) -> int:
    """Return the UTF-8 size of *message*'s protocol line without its newline.

    The supervisor's reader accepts a line whose content is at most
    :data:`~core.stt_pool.ipc.MAX_LINE_BYTES` bytes.

    Raises:
        TypeError: When *message* holds a value JSON cannot serialize.
        ValueError: When *message* contains a circular reference.

    """
    return len(ipc.encode_message(message).encode("utf-8")) - 1


# * ---------------------------------------------------------------------------
# * Worker protocol
# * ---------------------------------------------------------------------------


class _Worker:
    """Drives one engine through the worker protocol on the main thread."""

    def __init__(
        self, channel: ipc.WorkerChannel, log_handler: logging.Handler
    ) -> None:
        self._channel = channel
        self._log_handler = log_handler
        self._stage: FatalStage = "spec"
        # * Set right before ``engine.start()``, so a failing ``start()`` can
        # * seal the emitter and close the threads it already started. A failure
        # * before that point makes no further engine call (the process exit
        # * releases the model).
        self._engine: PoolEngine | None = None
        self._emitter: _ResultEmitter | None = None

    def run(self) -> NoReturn:
        """Serve the protocol until ``shutdown`` or stdin EOF; never returns.

        Raises:
            WorkerProtocolError: When the first line is not a supervisor message.
            ValueError: When the spec is invalid or the engine factory cannot be
                resolved.

        """
        stream = _stdin_stream()
        spec = self._receive_spec(stream)
        self._channel.send(
            {
                "type": ipc.MSG_HELLO,
                "protocol": ipc.PROTOCOL_VERSION,
                "worker_id": spec.worker_id,
                "pid": os.getpid(),
            }
        )
        engine, emitter = self._bring_up(spec)
        self._serve(stream, engine, emitter)

    def fail(self, exc: BaseException) -> NoReturn:
        """Report *exc* as ``fatal`` and end the process with ``EXIT_FATAL``.

        Once the engine owns the result emitter (a failing ``start()`` or the
        run stage), results stop first (``fatal`` stays the last message), then
        the engine gets a short, best-effort ``close()``.

        Args:
            exc: Failure that ends the worker.

        """
        stage = self._stage
        try:
            kind = _fatal_error_kind(stage, exc)
            logger.error(
                "fatal error at stage %s (error_kind=%s)", stage, kind, exc_info=exc
            )
            if self._emitter is not None:
                self._emitter.seal()
            self._channel.send(
                {
                    "type": ipc.MSG_FATAL,
                    "stage": stage,
                    "error_kind": kind,
                    "error_detail": error_detail_for_exception(exc),
                }
            )
            engine = self._engine
            if engine is not None and not _call_bounded(
                engine.close, timeout_s=FATAL_CLOSE_TIMEOUT_S, name="stt-worker-close"
            ):
                logger.warning(
                    "engine close did not finish within %.0f s after the fatal error",
                    FATAL_CLOSE_TIMEOUT_S,
                )
        finally:
            _terminate(ipc.EXIT_FATAL)

    def _receive_spec(self, stream: IO[bytes] | IO[str]) -> WorkerSpec:
        """Read and validate the ``spec`` line, then tag the log with the worker id."""
        spec = WorkerSpec.from_message(_read_first_message(stream))
        self._log_handler.setFormatter(_log_formatter(spec.worker_id))
        logger.info(
            "spec received: key=%s capacity=%d device=%s threads_per_slot=%d "
            "host_threads=%d engine_factory=%s thread_env=%s",
            spec.key.label,
            spec.capacity,
            spec.device,
            spec.threads_per_slot,
            spec.host_threads,
            spec.engine_factory,
            _thread_env(),
        )
        return spec

    def _bring_up(self, spec: WorkerSpec) -> tuple[PoolEngine, _ResultEmitter]:
        """Build, load and start the engine; answer ``loaded`` once it is ready.

        ``loaded`` is sent only after ``engine.start()`` returned, so it means
        that the engine accepts jobs. The engine and the emitter are stored
        before ``start()`` runs, which lets :meth:`fail` seal the emitter and
        close the threads that a failing ``start()`` already started.

        Raises:
            DeviceUnavailableError: When the engine reports another device than
                the strict spec device (checked before ``start()``).

        """
        self._stage = "load"
        started = time.perf_counter()
        factory = load_engine_factory(spec.engine_factory)
        engine = factory(spec)
        info = engine.load()
        if info.device != spec.device:
            msg = (
                f"{spec.key.label}: the engine loaded on {info.device!r} instead "
                f"of the strict device {spec.device!r}"
            )
            raise DeviceUnavailableError(msg)
        emitter = _ResultEmitter(self._channel, spec.key.backend)
        self._engine, self._emitter = engine, emitter
        engine.start(emitter.emit)
        self._channel.send(info.to_message())
        self._stage = "run"
        logger.info(
            "engine ready: device=%s load_s=%.2f bring_up_s=%.2f details=%s",
            info.device,
            info.load_s,
            time.perf_counter() - started,
            dict(info.details),
        )
        return engine, emitter

    def _serve(
        self,
        stream: IO[bytes] | IO[str],
        engine: PoolEngine,
        emitter: _ResultEmitter,
    ) -> NoReturn:
        """Dispatch supervisor messages until ``shutdown`` or EOF."""
        for message in _supervisor_messages(stream):
            kind = message.get("type")
            if kind == ipc.MSG_JOB:
                self._on_job(engine, emitter, message)
            elif kind == ipc.MSG_CANCEL:
                self._on_cancel(engine, message)
            elif kind == ipc.MSG_SHUTDOWN:
                self._shutdown(engine, emitter)
            else:
                logger.warning("supervisor message ignored: type=%r", kind)
        _parent_gone("stdin closed without a shutdown")

    def _on_job(
        self,
        engine: PoolEngine,
        emitter: _ResultEmitter,
        message: Mapping[str, object],
    ) -> None:
        """Submit a ``job`` message, or answer a malformed one."""
        try:
            item = WorkItem.from_message(message)
        except (ValueError, TypeError) as exc:
            job_id, attempt_id = message.get("job_id"), message.get("attempt_id")
            detail = f"invalid job message: {error_detail_for_exception(exc)}"
            if _usable_id(job_id) and _usable_id(attempt_id):
                logger.warning(
                    "malformed job answered with %s: job_id=%s attempt_id=%s %s",
                    ERROR_KIND_PROTOCOL_ERROR,
                    job_id,
                    attempt_id,
                    detail,
                )
                emitter.emit_protocol_error(job_id, attempt_id, detail)
            else:
                logger.warning("malformed job without usable ids ignored: %s", detail)
            return
        logger.info(
            "job received: job_id=%s attempt_id=%s priority=%s call_id=%s seq=%d",
            item.job_id,
            item.attempt_id,
            item.priority,
            item.call_id or "-",
            item.seq,
        )
        engine.submit(item)

    @staticmethod
    def _on_cancel(engine: PoolEngine, message: Mapping[str, object]) -> None:
        """Forward a ``cancel`` message to the engine."""
        job_id, attempt_id = message.get("job_id"), message.get("attempt_id")
        if not (_usable_id(job_id) and _usable_id(attempt_id)):
            logger.warning(
                "malformed cancel ignored: job_id=%r attempt_id=%r", job_id, attempt_id
            )
            return
        logger.info("cancel received: job_id=%s attempt_id=%s", job_id, attempt_id)
        engine.cancel(job_id, attempt_id)

    def _shutdown(self, engine: PoolEngine, emitter: _ResultEmitter) -> NoReturn:
        """Close the engine (bounded), answer ``bye`` and exit with code 0."""
        logger.info("shutdown requested; closing the engine")
        if not _call_bounded(
            engine.close, timeout_s=SHUTDOWN_CLOSE_TIMEOUT_S, name="stt-worker-close"
        ):
            logger.warning(
                "engine close did not finish within %.0f s; the process exit "
                "releases the model",
                SHUTDOWN_CLOSE_TIMEOUT_S,
            )
        emitter.seal()
        self._channel.send({"type": ipc.MSG_BYE})
        logger.info("bye sent; exiting")
        _terminate(0)


def main() -> NoReturn:
    """Run one pool worker process; never returns (always ``os._exit``)."""
    try:
        # * First: from here on fd 1 and sys.stdout point to stderr, so no
        # * stray output can reach the protocol pipe.
        channel = ipc.WorkerChannel.install()
    except BaseException as exc:  # noqa: BLE001 - there is no protocol stream to report on
        with contextlib.suppress(OSError, ValueError, AttributeError):
            sys.stderr.write(f"cannot install the worker protocol channel: {exc!r}\n")
        _terminate(ipc.EXIT_FATAL)
    _ignore_console_interrupts()
    handler = _configure_logging()
    logger.info(
        "worker process started: pid=%d python=%s", os.getpid(), sys.version.split()[0]
    )
    worker = _Worker(channel, handler)
    try:
        worker.run()
    except BaseException as exc:  # noqa: BLE001 - every failure ends as a reported fatal
        worker.fail(exc)


if __name__ == "__main__":
    main()
