"""Supervisor-side worker process handles and the detached supervisor spawn.

:class:`WorkerProcess` owns one worker process of the pool: it starts the
process with piped stdin/stdout and stderr in a per-worker log, puts it into
its own :class:`~core.stt_pool.winjob.WorkerJob` on Windows, or a capped
delegated cgroup v2 leaf on managed Linux, BEFORE the ``spec`` line is sent,
forwards decoded worker messages to a callback from a reader thread, writes
supervisor messages from a writer thread, and tears everything down without
leaking processes, threads, pipes or handles.

:func:`spawn_detached` starts the supervisor itself as a headless background
process that outlives its spawning client. The module imports no ML packages.
"""

from __future__ import annotations

import atexit
import contextlib
import errno
import logging
import os
import queue
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import IO, TYPE_CHECKING

import psutil  # type: ignore[import-untyped]

from core.stt_pool import ipc
from core.stt_pool.linux_cgroup import START_GATE, WorkerCgroup, managed_linux
from core.stt_pool.sysprobe import child_interpreter
from core.stt_pool.winjob import WorkerJob, detached_creationflags

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from core.stt_pool.linux_cgroup import MemoryEvents

logger = logging.getLogger(__name__)

# * Module run by worker processes.
DEFAULT_WORKER_MODULE = "core.stt_pool.worker"

EventCallback = Callable[[str, dict[str, object] | None], None]

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_THREAD_JOIN_TIMEOUT_S = 5.0
# * Wait after killing a worker that could not join its memory-cap backend.
_ABORT_WAIT_S = 5.0
# * Bytes read from the end of a log per requested character (UTF-8 worst case).
_LOG_TAIL_BYTES_PER_CHAR = 4
_STOP_WRITER: bytes = b""

# * Detached supervisors stay referenced until they exit, so their ``Popen``
# * objects never emit "subprocess still running" ResourceWarnings; workers that
# * survived a kill and the close timeout are kept for the same reason.
_REGISTRY_LOCK = threading.Lock()
_DETACHED: list[subprocess.Popen[bytes]] = []
_LINGERING: list[subprocess.Popen[bytes]] = []


class WorkerSpawnError(RuntimeError):
    """A worker process could not be started safely."""


class WorkerCgroupBusyError(WorkerSpawnError):
    """A busy delegated cgroup defers startup without a model crash."""


@dataclass(frozen=True, slots=True)
class WorkerLaunch:
    """Command line, environment and working directory of a worker process.

    Attributes:
        argv: Full command line (interpreter first).
        env: Complete environment of the process.
        cwd: Working directory (the project root, so ``-m core...`` resolves).

    """

    argv: tuple[str, ...]
    env: Mapping[str, str]
    cwd: Path


def default_worker_launch(
    *,
    project_root: Path,
    extra_env: Mapping[str, str] | None = None,
    module: str = DEFAULT_WORKER_MODULE,
) -> WorkerLaunch:
    """Return the launch description of a production worker process.

    Args:
        project_root: Directory that contains the ``core`` package.
        extra_env: Variables added last (for example ``OMP_NUM_THREADS``).
        module: Module run with ``-m``.

    Returns:
        ``[interpreter, "-X", "utf8", "-m", module]`` started from the base
        interpreter (see :func:`~core.stt_pool.sysprobe.child_interpreter`)
        with UTF-8, unbuffered stdio.

    """
    interpreter, interpreter_env = child_interpreter()
    env = dict(os.environ)
    env.update(interpreter_env)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)
    return WorkerLaunch(
        argv=(interpreter, "-X", "utf8", "-m", module),
        env=MappingProxyType(env),
        cwd=project_root,
    )


def _remember_lingering(proc: subprocess.Popen[bytes]) -> None:
    """Keep a process that did not exit referenced, and log it."""
    logger.error("worker process %s did not exit after being killed", proc.pid)
    with _REGISTRY_LOCK:
        _LINGERING.append(proc)


def _kill_tree(proc: subprocess.Popen[bytes]) -> bool:
    """Kill *proc* and its descendants without the Job Object (best effort).

    The descendants are collected first: once the root is gone they can no
    longer be found through it. A process that was already reaped is left
    alone: its PID may belong to an unrelated process by now, whose children
    must never be walked.

    Args:
        proc: Root process.

    Returns:
        ``False`` when *proc* had already exited and nothing was touched,
        ``True`` otherwise.

    """
    if proc.poll() is not None:
        return False
    descendants: list[psutil.Process] = []
    with contextlib.suppress(psutil.Error, OSError):
        descendants = psutil.Process(proc.pid).children(recursive=True)
    with contextlib.suppress(OSError):
        proc.kill()
    for child in descendants:
        with contextlib.suppress(psutil.Error, OSError):
            child.kill()
    return True


class WorkerProcess:
    """Handle of one worker process owned by the supervisor.

    Threads: one reader thread delivers worker messages to ``on_event`` (and
    exactly one ``None`` at EOF) without holding any lock of this object; one
    writer thread writes queued supervisor messages to the worker's stdin, so
    :meth:`send` never blocks on a worker that stopped reading.

    Attributes:
        worker_id: Supervisor-assigned worker identifier.

    """

    def __init__(
        self,
        worker_id: str,
        launch: WorkerLaunch,
        *,
        log_path: Path,
        memory_limit_bytes: int | None,
        on_event: EventCallback,
        on_memory_limit: Callable[[str, int], None] | None = None,
        require_job: bool | None = None,
    ) -> None:
        """Prepare a worker handle; nothing starts before :meth:`start`.

        Args:
            worker_id: Supervisor-assigned worker identifier.
            launch: Command line, environment and working directory.
            log_path: File that receives the worker's stderr (appended).
            memory_limit_bytes: Reservation plus child headroom, enforced by
                a Windows Job Object or a managed Linux cgroup; ``None`` for
                no cap. Unmanaged Linux keeps its uncapped behavior.
            on_event: Called from the reader thread with ``(worker_id,
                message)`` for each decoded worker message and once with
                ``(worker_id, None)`` at EOF.
            on_memory_limit: Called from the job monitor thread with
                ``(worker_id, pid)`` when the commit cap is hit: once, and
                once more after each :meth:`raise_memory_limit`.
            require_job: Refuse to run a worker outside its Job Object;
                defaults to ``True`` on Windows and ``False`` elsewhere.
                This flag never disables mandatory managed Linux cgroup caps.

        Raises:
            ValueError: When *memory_limit_bytes* is not positive.

        """
        if memory_limit_bytes is not None and memory_limit_bytes <= 0:
            msg = f"memory_limit_bytes must be positive, got {memory_limit_bytes}"
            raise ValueError(msg)
        self.worker_id = worker_id
        self._launch = launch
        self._log_path = log_path
        self._memory_limit_bytes = memory_limit_bytes
        self._on_event = on_event
        self._on_memory_limit = on_memory_limit
        self._require_job = (
            sys.platform == "win32" if require_job is None else require_job
        )
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._job: WorkerJob | None = None
        self._cgroup: WorkerCgroup | None = None
        self._cgroup_events: MemoryEvents | None = None
        self._termination_requested = False
        self._in_job = False
        self._log_handle: IO[bytes] | None = None
        self._reader: threading.Thread | None = None
        self._writer: threading.Thread | None = None
        self._outbox: queue.SimpleQueue[bytes] = queue.SimpleQueue()
        self._pipe_broken = threading.Event()
        self._started = False
        self._closed = False

    # * State.

    @property
    def pid(self) -> int | None:
        """Return the process id, or ``None`` before a successful spawn."""
        proc = self._proc
        return proc.pid if proc is not None else None

    @property
    def memory_limit_hit(self) -> bool:
        """Return whether the worker's Job Object reported its commit cap."""
        job = self._job
        return job is not None and job.memory_limit_hit

    @property
    def memory_limit_hits(self) -> int:
        """Return how many commit-cap reports the worker's Job Object made.

        At most one per arming of the report: the first hit, and one more
        after each :meth:`raise_memory_limit` (Windows notifies only the
        first denial of each process, see
        :class:`~core.stt_pool.winjob.WorkerJob`). The count is updated before
        the ``on_memory_limit`` callback runs.
        """
        job = self._job
        return job.memory_limit_hits if job is not None else 0

    @property
    def memory_limit_pid(self) -> int | None:
        """Return the process the latest commit-cap report named (``None``: none).

        It is set before :attr:`memory_limit_hits` grows, so a reader that
        sees a report also sees its process.
        """
        job = self._job
        return job.memory_limit_pid if job is not None else None

    def raise_memory_limit(self, limit_bytes: int) -> bool:
        """Set a new commit cap on the live worker's Job Object.

        See :meth:`~core.stt_pool.winjob.WorkerJob.set_memory_limit`: the
        other job limits stay, and the one-shot commit-cap report is re-armed.

        Args:
            limit_bytes: New job-wide commit cap in bytes.

        Returns:
            ``True`` when the new cap is in effect; ``False`` when the worker
            does not run in a capped Job Object or the change failed (logged;
            the old cap stays).

        Raises:
            ValueError: When *limit_bytes* is not positive.

        """
        if limit_bytes <= 0:
            msg = f"limit_bytes must be positive, got {limit_bytes}"
            raise ValueError(msg)
        with self._lock:
            job, in_job = self._job, self._in_job
        if job is None or not in_job:
            logger.warning(
                "worker commit cap not raised: worker_id=%s detail=no Job Object",
                self.worker_id,
            )
            return False
        if not job.set_memory_limit(limit_bytes):
            logger.warning(
                "worker commit cap not raised: worker_id=%s limit_bytes=%d detail=%s",
                self.worker_id,
                limit_bytes,
                job.limit_error,
            )
            return False
        old = self._memory_limit_bytes
        self._memory_limit_bytes = limit_bytes
        logger.info(
            "worker commit cap raised: worker_id=%s old_limit_bytes=%s limit_bytes=%d",
            self.worker_id,
            old,
            limit_bytes,
        )
        return True

    @property
    def in_job(self) -> bool:
        """Return whether the process was assigned to its Job Object."""
        return self._in_job

    @property
    def in_cgroup(self) -> bool:
        """Return whether a managed Linux worker owns a capped cgroup."""
        return self._cgroup is not None

    @property
    def initial_memory_events(self) -> MemoryEvents | None:
        """Return the OOM baseline captured before this worker was spawned."""
        group = self._cgroup
        return group.baseline if group is not None else None

    @property
    def termination_requested(self) -> bool:
        """Return whether this handle killed a live process, including at teardown."""
        return self._termination_requested

    def memory_events(self) -> MemoryEvents | None:
        """Read the worker's OOM counters, preserving the final reading after close.

        A failed refresh preserves the last valid reading, or ``None`` when
        none is available. Each attempt captures its baseline before sending
        work, so returning a cached counter does not create a new kill delta.
        """
        group = self._cgroup
        if group is None or group.closed:
            return self._cgroup_events
        try:
            self._cgroup_events = group.memory_events()
        except (OSError, UnicodeError, ValueError) as exc:
            logger.warning(
                "worker cgroup OOM counter refresh failed: worker_id=%s path=%s "
                "last_events=%s error=%s",
                self.worker_id,
                group.path,
                self._cgroup_events,
                exc,
            )
        return self._cgroup_events

    def poll(self) -> int | None:
        """Return the exit code, or ``None`` while running or before a spawn."""
        proc = self._proc
        return proc.poll() if proc is not None else None

    def wait(self, timeout_s: float) -> int | None:
        """Wait for the process to exit.

        Args:
            timeout_s: Longest wait in seconds.

        Returns:
            The exit code, or ``None`` on timeout or before a spawn.

        """
        proc = self._proc
        if proc is None:
            return None
        try:
            return proc.wait(timeout=max(0.0, timeout_s))
        except subprocess.TimeoutExpired:
            return None

    def peak_memory_bytes(self) -> int | None:
        """Return ``PeakJobMemoryUsed`` of the worker's Job Object, if known.

        The value stays available after :meth:`close`.
        """
        job = self._job
        if job is None or not self._in_job:
            return None
        return job.peak_memory_bytes()

    def log_tail(self, max_chars: int = 1500) -> str:
        """Return the last *max_chars* characters of the worker's log.

        Args:
            max_chars: Longest tail returned.

        Returns:
            The stripped tail, or ``""`` when the log cannot be read.

        """
        if max_chars <= 0:
            return ""
        try:
            with self._log_path.open("rb") as handle:
                size = handle.seek(0, os.SEEK_END)
                handle.seek(max(0, size - max_chars * _LOG_TAIL_BYTES_PER_CHAR))
                raw = handle.read()
        except OSError:
            return ""
        return raw.decode("utf-8", errors="replace")[-max_chars:].strip()

    # * Start.

    def start(self, spec_message: Mapping[str, object]) -> None:
        """Spawn the worker, apply its platform memory cap, then send the spec.

        The ``spec`` line is the worker's first protocol input. Windows
        applies its Job Object before sending it; a gated Linux bootstrap
        waits for cgroup enrollment before executing the worker at all.
        Failed required enrollment kills the process before it receives the
        spec (fail closed). A handle can be started once.

        Args:
            spec_message: The ``spec`` protocol message.

        Raises:
            RuntimeError: When the handle was already started.
            TypeError: When *spec_message* is not JSON-serializable (raised
                before anything is spawned).
            WorkerSpawnError: When the handle is closed, the Job Object cannot
                be created or assigned while required, a managed Linux cap
                cannot be established, or the process cannot be started.
                Everything created so far is released first.

        """
        spec_line = ipc.encode_message(spec_message).encode("utf-8")
        with self._lock:
            if self._started:
                msg = f"worker {self.worker_id} was already started"
                raise RuntimeError(msg)
            if self._closed:
                msg = f"worker {self.worker_id} is closed"
                raise WorkerSpawnError(msg)
            self._started = True
            self._cgroup = self._create_cgroup()
            try:
                self._start_capped(spec_line)
            except (WorkerSpawnError, OSError, ValueError):
                if self._cgroup is not None:
                    self._cgroup.close()
                raise

    def _start_capped(self, spec_line: bytes) -> None:
        """Apply the cap and queue the spec while the caller holds the lock."""
        # * Linux executes only a gated bootstrap until cgroup enrollment succeeds.
        # * Windows keeps its Job Object assignment before the spec is sent.
        job = self._create_job()
        log_handle = self._open_log(job)
        proc = self._spawn(job, log_handle)
        self._proc, self._job, self._log_handle = proc, job, log_handle
        self._assign_or_abort(proc, job)
        stdout, stdin = proc.stdout, proc.stdin
        if stdout is None or stdin is None:
            # ! Unreachable with PIPE; guards the type of the streams.
            self._closed = True
            self._teardown(timeout_s=_ABORT_WAIT_S)
            msg = f"worker {self.worker_id} has no protocol pipes"
            raise WorkerSpawnError(msg)
        self._reader = threading.Thread(
            target=self._read_events,
            args=(stdout,),
            name=f"stt-worker-reader-{self.worker_id}",
            daemon=True,
        )
        self._writer = threading.Thread(
            target=self._write_messages,
            args=(stdin,),
            name=f"stt-worker-writer-{self.worker_id}",
            daemon=True,
        )
        self._reader.start()
        self._writer.start()
        self._outbox.put(spec_line)
        if self._pipe_broken.is_set():
            logger.warning(
                "worker spec not sent (pipe closed): worker_id=%s pid=%s",
                self.worker_id,
                proc.pid,
            )
        logger.info(
            "worker started: worker_id=%s pid=%s in_job=%s memory_limit_bytes=%s "
            "cgroup=%s",
            self.worker_id,
            proc.pid,
            self._in_job,
            self._memory_limit_bytes,
            self._cgroup.path if self._cgroup is not None else None,
        )

    def _create_cgroup(self) -> WorkerCgroup | None:
        """Require a cap for managed Linux reservations; fail closed on any error."""
        if not managed_linux() or self._memory_limit_bytes is None:
            return None
        try:
            return WorkerCgroup.create(self.worker_id, self._memory_limit_bytes)
        except (OSError, UnicodeError, ValueError) as exc:
            msg = f"worker {self.worker_id}: delegated cgroup v2 memory cap unavailable: {exc}"
            if isinstance(exc, OSError) and exc.errno == errno.EBUSY:
                raise WorkerCgroupBusyError(msg) from exc
            raise WorkerSpawnError(msg) from exc

    def _handle_memory_limit(self, pid: int) -> None:
        """Forward a Job Object memory-limit report (monitor thread)."""
        logger.warning(
            "worker hit its commit cap: worker_id=%s pid=%s limit_bytes=%s",
            self.worker_id,
            pid,
            self._memory_limit_bytes,
        )
        callback = self._on_memory_limit
        if callback is not None:
            callback(self.worker_id, pid)

    def _create_job(self) -> WorkerJob | None:
        """Create the worker's Job Object.

        Returns:
            The active job, or ``None`` when jobs are unavailable and not
            required.

        Raises:
            WorkerSpawnError: When the job is required but unavailable.

        """
        if self._cgroup is not None or (
            sys.platform != "win32" and not self._require_job
        ):
            return None
        job = WorkerJob(
            memory_limit_bytes=self._memory_limit_bytes,
            on_memory_limit=self._handle_memory_limit,
        )
        if job.active:
            if job.monitor_error is not None:
                logger.warning(
                    "worker job cap is not reported: worker_id=%s detail=%s",
                    self.worker_id,
                    job.monitor_error,
                )
            return job
        detail = job.error
        job.close()
        if self._require_job:
            msg = f"worker {self.worker_id}: Job Object unavailable: {detail}"
            raise WorkerSpawnError(msg)
        logger.info(
            "worker runs without a Job Object: worker_id=%s detail=%s",
            self.worker_id,
            detail,
        )
        return None

    def _open_log(self, job: WorkerJob | None) -> IO[bytes]:
        """Open the worker's stderr log for appending.

        Raises:
            WorkerSpawnError: When the log cannot be opened (the job is closed).

        """
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            return self._log_path.open("ab")
        except OSError as exc:
            if job is not None:
                job.close()
            msg = f"worker {self.worker_id}: cannot open log {self._log_path}: {exc}"
            raise WorkerSpawnError(msg) from exc

    def _spawn(
        self, job: WorkerJob | None, log_handle: IO[bytes]
    ) -> subprocess.Popen[bytes]:
        """Start the worker process.

        Raises:
            WorkerSpawnError: When the process cannot be started (the job and
                the log are closed).

        """
        launch = self._launch
        argv = list(launch.argv)
        if self._cgroup is not None:
            interpreter, _env = child_interpreter()
            argv = [
                interpreter,
                "-X",
                "utf8",
                "-m",
                "core.stt_pool.linux_cgroup",
                "--",
                *argv,
            ]
        try:
            return subprocess.Popen(  # noqa: S603 - supervisor-built argv, no shell
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log_handle,
                cwd=str(launch.cwd),
                env=dict(launch.env),
                creationflags=_NO_WINDOW,
                close_fds=True,
            )
        except (OSError, ValueError) as exc:
            if job is not None:
                job.close()
            with contextlib.suppress(OSError):
                log_handle.close()
            msg = f"worker {self.worker_id}: cannot start process: {exc}"
            raise WorkerSpawnError(msg) from exc

    def _assign_or_abort(
        self, proc: subprocess.Popen[bytes], job: WorkerJob | None
    ) -> None:
        """Assign the new process to its Job Object, or kill it (fail closed).

        The caller holds ``_lock``.

        Raises:
            WorkerSpawnError: When the assignment fails while a job is required.

        """
        if self._cgroup is not None:
            self._assign_cgroup_or_abort(proc)
            return
        if job is None:
            return
        if job.assign(proc.pid):
            self._in_job = True
            return
        detail = job.assign_error
        if not self._require_job:
            logger.warning(
                "worker runs outside its Job Object: worker_id=%s pid=%s detail=%s",
                self.worker_id,
                proc.pid,
                detail,
            )
            return
        self._closed = True
        self._teardown(timeout_s=_ABORT_WAIT_S)
        msg = (
            f"worker {self.worker_id}: process {proc.pid} could not join its "
            f"Job Object: {detail}"
        )
        raise WorkerSpawnError(msg)

    def _assign_cgroup_or_abort(self, proc: subprocess.Popen[bytes]) -> None:
        """Enroll the waiting bootstrap before releasing it to execute worker code."""
        group = self._cgroup
        if group is None or proc.stdin is None:
            msg = "capped Linux worker has no cgroup or bootstrap input"
            raise WorkerSpawnError(msg)
        try:
            group.assign(proc.pid)
            proc.stdin.write(START_GATE)
            proc.stdin.flush()
        except (OSError, ValueError) as exc:
            self._closed = True
            self._teardown(timeout_s=_ABORT_WAIT_S)
            msg = f"worker {self.worker_id}: process {proc.pid} could not join cgroup: {exc}"
            raise WorkerSpawnError(msg) from exc

    # * Pipes.

    def _deliver(self, message: dict[str, object] | None) -> None:
        """Call ``on_event``; its failures never stop the reader thread."""
        try:
            self._on_event(self.worker_id, message)
        except Exception:
            # ! The reader must keep draining the pipe and still report EOF.
            logger.exception(
                "worker event callback failed: worker_id=%s", self.worker_id
            )

    def _read_events(self, stream: IO[bytes]) -> None:
        """Reader thread: deliver worker messages, then exactly one ``None``."""
        try:
            for message in ipc.iter_messages(
                stream,
                allowed=ipc.WORKER_TO_SUPERVISOR,
                source=f"worker {self.worker_id}",
            ):
                self._deliver(message)
        except (OSError, ValueError) as exc:
            logger.debug(
                "worker protocol read ended: worker_id=%s error=%s",
                self.worker_id,
                exc,
            )
        finally:
            with contextlib.suppress(OSError, ValueError):
                stream.close()
            self._deliver(None)

    def _write_messages(self, stream: IO[bytes]) -> None:
        """Writer thread: write queued lines until the stop marker, then close."""
        try:
            while True:
                data = self._outbox.get()
                if data == _STOP_WRITER:
                    return
                try:
                    stream.write(data)
                    stream.flush()
                except (OSError, ValueError) as exc:
                    self._pipe_broken.set()
                    logger.debug(
                        "worker stdin closed: worker_id=%s error=%s",
                        self.worker_id,
                        exc,
                    )
                    return
        finally:
            with contextlib.suppress(OSError, ValueError):
                stream.close()

    def send(self, message: Mapping[str, object]) -> bool:
        """Queue one message for the worker's stdin.

        Args:
            message: Protocol message.

        Returns:
            ``False`` when the worker was never started or has exited, the
            handle is closed, or the pipe is known to be broken; ``True`` when
            the message was queued. A queued message to a worker that dies
            meanwhile is dropped; the reader's EOF event reports the exit.

        Raises:
            TypeError: When *message* holds a value JSON cannot serialize.
            ValueError: When *message* contains a circular reference.

        """
        return self._enqueue(ipc.encode_message(message).encode("utf-8"))

    def _enqueue(self, data: bytes) -> bool:
        """Queue one encoded line for the writer thread (see :meth:`send`)."""
        with self._lock:
            proc = self._proc
            if (
                proc is None
                or self._writer is None
                or self._closed
                or self._pipe_broken.is_set()
                or proc.poll() is not None
            ):
                return False
            self._outbox.put(data)
        return True

    def request_shutdown(self) -> bool:
        """Ask the worker to finish gracefully (``shutdown`` message).

        Returns:
            Whether the message was queued (see :meth:`send`).

        """
        return self.send({"type": ipc.MSG_SHUTDOWN})

    # * Stop.

    def kill(self) -> None:
        """Kill the worker and all of its descendants (ffmpeg included).

        Terminates the Job Object; without one, kills the process and the
        descendants found through ``psutil`` (only while the process has not
        been reaped). Returns without waiting.
        """
        with self._lock:
            proc, job, in_job = self._proc, self._job, self._in_job
        if proc is not None:
            self._kill_now(proc, job if in_job else None)

    def _kill_now(self, proc: subprocess.Popen[bytes], job: WorkerJob | None) -> None:
        """Terminate the job, or fall back to a process-tree kill."""
        if proc.poll() is None:
            self._termination_requested = True
        if job is not None and job.terminate():
            logger.info(
                "worker job terminated: worker_id=%s pid=%s", self.worker_id, proc.pid
            )
            return
        if _kill_tree(proc):
            logger.info(
                "worker process tree killed: worker_id=%s pid=%s",
                self.worker_id,
                proc.pid,
            )
        else:
            logger.info(
                "worker process already exited, nothing to kill: worker_id=%s pid=%s",
                self.worker_id,
                proc.pid,
            )

    def close(self, *, timeout_s: float = 10.0) -> None:
        """Release everything the handle owns; idempotent.

        Kills the process tree if it still runs, waits for the process, joins
        the reader and writer threads (bounded), and closes the pipes, the log
        and the Job Object (which kills anything left in it). Safe from any
        thread except the reader thread, and after a failed :meth:`start`.

        Args:
            timeout_s: Longest wait for the process to exit after the kill.

        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._teardown(timeout_s=timeout_s)

    def _teardown(self, *, timeout_s: float) -> None:
        """Kill, wait, stop the threads, and close every resource (once).

        Never calls ``on_event`` and never takes ``_lock``, so it may run while
        :meth:`start` holds it.
        """
        proc, job, in_job = self._proc, self._job, self._in_job
        if proc is not None:
            if proc.poll() is None:
                self._kill_now(proc, job if in_job else None)
            try:
                proc.wait(timeout=max(0.0, timeout_s))
            except subprocess.TimeoutExpired:
                _remember_lingering(proc)
        if job is not None and in_job:
            # * A descendant (ffmpeg) can outlive a worker that exited on its own
            # * and keep the stdout pipe open; terminating the (possibly empty)
            # * job first lets the reader reach EOF before it is joined.
            job.terminate()
        self._stop_threads(proc)
        if job is not None:
            job.close()
        if self._cgroup is not None:
            self.memory_events()
            self._cgroup.close()
        log_handle, self._log_handle = self._log_handle, None
        if log_handle is not None:
            with contextlib.suppress(OSError):
                log_handle.close()

    def _stop_threads(self, proc: subprocess.Popen[bytes] | None) -> None:
        """Stop the writer, join both pipe threads, and close unowned pipes."""
        current = threading.current_thread()
        writer, reader = self._writer, self._reader
        if writer is not None:
            self._outbox.put(_STOP_WRITER)
            if writer is not current:
                writer.join(timeout=_THREAD_JOIN_TIMEOUT_S)
                if writer.is_alive():
                    logger.warning(
                        "worker writer thread did not stop: %s", self.worker_id
                    )
        if reader is not None and reader is not current:
            reader.join(timeout=_THREAD_JOIN_TIMEOUT_S)
            if reader.is_alive():
                logger.warning("worker reader thread did not stop: %s", self.worker_id)
        if proc is None:
            return
        # * Pipes without an owning thread (failed start) are closed here.
        if writer is None and proc.stdin is not None:
            with contextlib.suppress(OSError, ValueError):
                proc.stdin.close()
        if reader is None and proc.stdout is not None:
            with contextlib.suppress(OSError, ValueError):
                proc.stdout.close()


def _reap_detached() -> None:
    """Forget detached processes that have exited.

    Polling records the exit code, so a forgotten ``Popen`` never warns. It
    also runs at interpreter exit; a supervisor that is still running then is
    the intended outcome of a detached spawn.
    """
    with _REGISTRY_LOCK:
        _DETACHED[:] = [proc for proc in _DETACHED if proc.poll() is None]


atexit.register(_reap_detached)


def _popen_detached(
    argv: Sequence[str],
    *,
    cwd: Path,
    log_handle: IO[bytes],
    env: Mapping[str, str] | None,
    breakaway: bool,
) -> subprocess.Popen[bytes]:
    """Start one headless background process with output appended to a log."""
    return subprocess.Popen(  # noqa: S603 - caller-built argv, no shell
        list(argv),
        cwd=str(cwd),
        env=dict(env) if env is not None else None,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        close_fds=True,
        creationflags=detached_creationflags(breakaway=breakaway),
        start_new_session=sys.platform != "win32",
    )


def spawn_detached(
    argv: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    env: Mapping[str, str] | None = None,
) -> int:
    """Start a detached background process (the supervisor) and return its pid.

    On Windows the process gets no window, its own process group, and breaks
    away from the caller's Job Object, so it outlives the spawning client. A
    job that forbids breakaway refuses the spawn; it is then retried once
    without breakaway (the process stays in the caller's job).

    Args:
        argv: Full command line.
        cwd: Working directory.
        log_path: File that receives stdout and stderr (appended).
        env: Complete environment, or ``None`` to inherit this process's.

    Returns:
        The process id.

    Raises:
        OSError: When the log cannot be opened or the process cannot start.

    """
    _reap_detached()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log_handle:
        try:
            proc = _popen_detached(
                argv, cwd=cwd, log_handle=log_handle, env=env, breakaway=True
            )
        except OSError as exc:
            if sys.platform != "win32":
                raise
            logger.warning(
                "detached spawn with job breakaway refused (%s); retrying without "
                "breakaway, the process stays in the caller's job",
                exc,
            )
            proc = _popen_detached(
                argv, cwd=cwd, log_handle=log_handle, env=env, breakaway=False
            )
    with _REGISTRY_LOCK:
        _DETACHED.append(proc)
    logger.info("detached process started: pid=%s argv0=%s", proc.pid, argv[0])
    return proc.pid
