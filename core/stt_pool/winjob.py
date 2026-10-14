"""Per-worker Windows Job Objects and creation flags for detached processes.

Every STT worker process (and everything it starts, such as ffmpeg) lives in
its own Job Object owned by the supervisor:

* ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``: when the supervisor closes the handle,
  or dies for any reason and Windows closes it, every process of the job is
  terminated, so no orphan keeps a model in memory;
* ``JOB_OBJECT_LIMIT_JOB_MEMORY`` (optional): the summed commit charge of the
  job's processes is capped. An allocation that would cross the cap fails, and
  Windows posts ``JOB_OBJECT_MSG_JOB_MEMORY_LIMIT`` to the job's completion
  port; a monitor thread reports it once.

Jobs nest on Windows 8 and newer: when the supervisor itself runs inside a job,
a worker's job becomes a child of that job and both limits apply. Off Windows
the job is inactive and :attr:`WorkerJob.error` says why. The module uses
``ctypes`` only and imports no ML packages.
"""

from __future__ import annotations

import ctypes
import functools
import logging
import subprocess
import sys
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# * Exit code of the processes terminated through :meth:`WorkerJob.terminate`.
JOB_TERMINATE_EXIT_CODE = 0xE0000017
# * ``CreateProcess`` flag that starts a process outside the caller's job.
CREATE_BREAKAWAY_FROM_JOB = 0x01000000

_DWORD = ctypes.c_uint32
_SIZE_T = ctypes.c_size_t
_JOB_OBJECT_LIMIT_JOB_MEMORY = 0x0200
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JOB_OBJECT_ASSOCIATE_COMPLETION_PORT_INFORMATION = 7
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_MSG_JOB_MEMORY_LIMIT = 10
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
# * Completion keys of the job's notification port.
_JOB_KEY = 1
_STOP_KEY = 2
_PORT_WAIT_MS = 200
_MONITOR_JOIN_TIMEOUT_S = 5.0


class _IoCounters(ctypes.Structure):
    """``IO_COUNTERS``."""

    _fields_ = tuple(
        (name, ctypes.c_uint64)
        for name in (
            "ReadOperationCount",
            "WriteOperationCount",
            "OtherOperationCount",
            "ReadTransferCount",
            "WriteTransferCount",
            "OtherTransferCount",
        )
    )


class _JobBasicLimitInformation(ctypes.Structure):
    """``JOBOBJECT_BASIC_LIMIT_INFORMATION``."""

    _fields_ = (
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", _DWORD),
        ("MinimumWorkingSetSize", _SIZE_T),
        ("MaximumWorkingSetSize", _SIZE_T),
        ("ActiveProcessLimit", _DWORD),
        ("Affinity", _SIZE_T),
        ("PriorityClass", _DWORD),
        ("SchedulingClass", _DWORD),
    )


class _JobExtendedLimitInformation(ctypes.Structure):
    """``JOBOBJECT_EXTENDED_LIMIT_INFORMATION``."""

    _fields_ = (
        ("BasicLimitInformation", _JobBasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", _SIZE_T),
        ("JobMemoryLimit", _SIZE_T),
        ("PeakProcessMemoryUsed", _SIZE_T),
        ("PeakJobMemoryUsed", _SIZE_T),
    )


class _JobAssociateCompletionPort(ctypes.Structure):
    """``JOBOBJECT_ASSOCIATE_COMPLETION_PORT``."""

    _fields_ = (
        ("CompletionKey", ctypes.c_void_p),
        ("CompletionPort", ctypes.c_void_p),
    )


@functools.cache
def _kernel32() -> ctypes.CDLL:
    """Return ``kernel32`` with the job-object prototypes declared.

    Raises:
        OSError: On non-Windows platforms.

    """
    if sys.platform != "win32":
        msg = "kernel32 is only available on Windows"
        raise OSError(msg)
    lib = ctypes.WinDLL("kernel32", use_last_error=True)
    lib.CreateJobObjectW.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p)
    lib.CreateJobObjectW.restype = ctypes.c_void_p
    lib.SetInformationJobObject.argtypes = (
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        _DWORD,
    )
    lib.SetInformationJobObject.restype = ctypes.c_int
    lib.QueryInformationJobObject.argtypes = (
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        _DWORD,
        ctypes.c_void_p,
    )
    lib.QueryInformationJobObject.restype = ctypes.c_int
    lib.OpenProcess.argtypes = (_DWORD, ctypes.c_int, _DWORD)
    lib.OpenProcess.restype = ctypes.c_void_p
    lib.AssignProcessToJobObject.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    lib.AssignProcessToJobObject.restype = ctypes.c_int
    lib.TerminateJobObject.argtypes = (ctypes.c_void_p, ctypes.c_uint)
    lib.TerminateJobObject.restype = ctypes.c_int
    lib.CloseHandle.argtypes = (ctypes.c_void_p,)
    lib.CloseHandle.restype = ctypes.c_int
    lib.CreateIoCompletionPort.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        _DWORD,
    )
    lib.CreateIoCompletionPort.restype = ctypes.c_void_p
    lib.GetQueuedCompletionStatus.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(_DWORD),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_void_p),
        _DWORD,
    )
    lib.GetQueuedCompletionStatus.restype = ctypes.c_int
    lib.PostQueuedCompletionStatus.argtypes = (
        ctypes.c_void_p,
        _DWORD,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    lib.PostQueuedCompletionStatus.restype = ctypes.c_int
    lib.GetCurrentProcess.argtypes = ()
    lib.GetCurrentProcess.restype = ctypes.c_void_p
    lib.IsProcessInJob.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
    )
    lib.IsProcessInJob.restype = ctypes.c_int
    return lib


def _last_error() -> int:
    """Return the calling thread's last Win32 error (``0`` off Windows)."""
    if sys.platform != "win32":
        return 0
    return int(ctypes.get_last_error())


class WorkerJob:
    """Kill-on-close Job Object of one worker, with an optional commit cap.

    Attributes:
        memory_limit_bytes: Job-wide commit cap, or ``None`` for no cap.
        memory_limit_pid: Process id reported with the first memory-limit
            notification, or ``None`` while the cap was not hit.
        error: Why the job is unusable (``None`` when it is active).
        monitor_error: Why a hit of an enforced cap cannot be reported.
        assign_error: Why the last :meth:`assign` failed, if it did.

    """

    def __init__(
        self,
        *,
        memory_limit_bytes: int | None,
        on_memory_limit: Callable[[int], None] | None = None,
    ) -> None:
        """Create the job; on an OS failure :attr:`error` explains why it is inactive.

        Args:
            memory_limit_bytes: Job-wide commit cap in bytes, or ``None`` for no
                cap.
            on_memory_limit: Called once, from the monitor thread, with the
                offending process id when the cap is hit.

        Raises:
            ValueError: When *memory_limit_bytes* is not positive.

        """
        if memory_limit_bytes is not None and memory_limit_bytes <= 0:
            msg = f"memory_limit_bytes must be positive, got {memory_limit_bytes}"
            raise ValueError(msg)
        self._handle: int | None = None
        self._port: int | None = None
        self._monitor: threading.Thread | None = None
        self._stop = threading.Event()
        self._handle_lock = threading.Lock()
        self._on_memory_limit = on_memory_limit
        self._final_peak: int | None = None
        self._closing = False
        self.memory_limit_bytes = memory_limit_bytes
        self.memory_limit_pid: int | None = None
        self.error: str | None = None
        self.monitor_error: str | None = None
        self.assign_error: str | None = None
        if sys.platform != "win32":
            self.error = "job objects are Windows-only"
            return
        try:
            kernel32 = _kernel32()
        except OSError as exc:
            self.error = f"kernel32 is unavailable: {exc}"
            return
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            self.error = f"CreateJobObjectW failed (WinError {_last_error()})"
            return
        info = _JobExtendedLimitInformation()
        flags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if memory_limit_bytes is not None:
            flags |= _JOB_OBJECT_LIMIT_JOB_MEMORY
            info.JobMemoryLimit = int(memory_limit_bytes)
        info.BasicLimitInformation.LimitFlags = flags
        if not kernel32.SetInformationJobObject(
            handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            self.error = f"SetInformationJobObject failed (WinError {_last_error()})"
            kernel32.CloseHandle(handle)
            return
        self._handle = int(handle)
        if memory_limit_bytes is not None:
            self._start_monitor()

    def _start_monitor(self) -> None:
        """Associate a completion port and start the memory-limit monitor."""
        if sys.platform != "win32" or self._handle is None:
            return
        kernel32 = _kernel32()
        port = kernel32.CreateIoCompletionPort(_INVALID_HANDLE_VALUE, None, 0, 1)
        if not port:
            self.monitor_error = (
                f"CreateIoCompletionPort failed (WinError {_last_error()}); "
                "the cap is enforced but not reported"
            )
            return
        association = _JobAssociateCompletionPort(_JOB_KEY, port)
        if not kernel32.SetInformationJobObject(
            self._handle,
            _JOB_OBJECT_ASSOCIATE_COMPLETION_PORT_INFORMATION,
            ctypes.byref(association),
            ctypes.sizeof(association),
        ):
            self.monitor_error = (
                f"job completion port association failed (WinError {_last_error()}); "
                "the cap is enforced but not reported"
            )
            kernel32.CloseHandle(port)
            return
        self._port = int(port)
        self._monitor = threading.Thread(
            target=self._watch_port,
            args=(self._port,),
            name="worker-job-memory-monitor",
            daemon=True,
        )
        self._monitor.start()

    def _watch_port(self, port: int) -> None:
        """Read job notifications until :meth:`close`; report the memory cap once.

        Args:
            port: Completion port associated with the job.

        """
        if sys.platform != "win32":
            return
        kernel32 = _kernel32()
        message = _DWORD()
        key = ctypes.c_size_t()
        overlapped = ctypes.c_void_p()
        while not self._stop.is_set():
            if not kernel32.GetQueuedCompletionStatus(
                port,
                ctypes.byref(message),
                ctypes.byref(key),
                ctypes.byref(overlapped),
                _PORT_WAIT_MS,
            ):
                continue
            if key.value == _STOP_KEY:
                return
            if (
                message.value == _JOB_OBJECT_MSG_JOB_MEMORY_LIMIT
                and self.memory_limit_pid is None
            ):
                # * For this message the "overlapped" value is the process id.
                self.memory_limit_pid = int(overlapped.value or 0)
                self._report_memory_limit(self.memory_limit_pid)

    def _report_memory_limit(self, pid: int) -> None:
        """Call the memory-limit callback; its failures never stop the monitor."""
        callback = self._on_memory_limit
        if callback is None:
            return
        try:
            callback(pid)
        except Exception:
            # ! The monitor must keep draining the port; the failure is logged.
            logger.exception("worker job memory-limit callback failed: pid=%s", pid)

    @property
    def active(self) -> bool:
        """Return whether processes can be assigned to the job."""
        return self._handle is not None

    @property
    def memory_limit_hit(self) -> bool:
        """Return whether the job reported its memory cap."""
        return self.memory_limit_pid is not None

    def assign(self, pid: int) -> bool:
        """Assign process *pid* to the job.

        Args:
            pid: Process id.

        Returns:
            ``True`` when the process is now in the job; otherwise
            :attr:`assign_error` says why.

        """
        with self._handle_lock:
            if self._handle is None or sys.platform != "win32":
                self.assign_error = self.error or "the job is closed"
                return False
            kernel32 = _kernel32()
            process = kernel32.OpenProcess(
                _PROCESS_SET_QUOTA | _PROCESS_TERMINATE,
                0,
                pid,
            )
            if not process:
                self.assign_error = (
                    f"OpenProcess({pid}) failed (WinError {_last_error()})"
                )
                return False
            try:
                if kernel32.AssignProcessToJobObject(self._handle, process):
                    self.assign_error = None
                    return True
                self.assign_error = (
                    f"AssignProcessToJobObject({pid}) failed (WinError {_last_error()})"
                )
                return False
            finally:
                kernel32.CloseHandle(process)

    def terminate(self) -> bool:
        """Terminate every process of the job at once.

        Returns:
            ``True`` when the job was terminated.

        """
        with self._handle_lock:
            if self._handle is None or sys.platform != "win32":
                return False
            return bool(
                _kernel32().TerminateJobObject(self._handle, JOB_TERMINATE_EXIT_CODE)
            )

    def _query_peak_locked(self) -> int | None:
        """Return ``PeakJobMemoryUsed``; the caller holds the handle lock."""
        if self._handle is None or sys.platform != "win32":
            return None
        info = _JobExtendedLimitInformation()
        if not _kernel32().QueryInformationJobObject(
            self._handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
            None,
        ):
            return None
        return int(info.PeakJobMemoryUsed)

    def peak_memory_bytes(self) -> int | None:
        """Return the peak summed commit of the job's processes (``PeakJobMemoryUsed``).

        Returns:
            The live value while the job is open, the value read at
            :meth:`close` afterwards, or ``None`` when unknown.

        """
        with self._handle_lock:
            if self._handle is None:
                return self._final_peak
            return self._query_peak_locked()

    def close(self) -> None:
        """Stop the monitor and close the job; Windows kills what is still assigned.

        Idempotent and safe from any thread, including the monitor's own
        callback. The monitor is joined (bounded) without holding the handle
        lock, because its callback may call :meth:`terminate`.
        """
        with self._handle_lock:
            if self._closing:
                return
            self._closing = True
            port = self._port
            monitor = self._monitor
        self._stop.set()
        if port is not None and sys.platform == "win32":
            _kernel32().PostQueuedCompletionStatus(port, 0, _STOP_KEY, None)
        if monitor is not None and monitor is not threading.current_thread():
            monitor.join(timeout=_MONITOR_JOIN_TIMEOUT_S)
            if monitor.is_alive():
                logger.warning("worker job monitor did not stop within the timeout")
        with self._handle_lock:
            handle = self._handle
            if handle is not None:
                self._final_peak = self._query_peak_locked()
            self._handle = None
            self._port = None
            self._monitor = None
            if sys.platform == "win32":
                if port is not None:
                    _kernel32().CloseHandle(port)
                if handle is not None:
                    _kernel32().CloseHandle(handle)


def is_current_process_in_job() -> bool | None:
    """Return whether this process already runs inside a Job Object.

    Returns:
        ``IsProcessInJob(GetCurrentProcess(), NULL)``; ``None`` off Windows or
        when the query fails.

    """
    if sys.platform != "win32":
        return None
    try:
        kernel32 = _kernel32()
    except OSError:
        return None
    result = ctypes.c_int(0)
    if not kernel32.IsProcessInJob(
        kernel32.GetCurrentProcess(), None, ctypes.byref(result)
    ):
        return None
    return bool(result.value)


def detached_creationflags(*, breakaway: bool) -> int:
    """Return ``creationflags`` for a headless background process.

    ``CREATE_NO_WINDOW`` (rather than ``DETACHED_PROCESS``) keeps console
    interpreters from surfacing a window; ``CREATE_NEW_PROCESS_GROUP`` keeps the
    caller's Ctrl+C away from the child; ``CREATE_BREAKAWAY_FROM_JOB`` lets the
    child outlive a job that the caller runs in, when that job allows it.

    Args:
        breakaway: Add ``CREATE_BREAKAWAY_FROM_JOB``.

    Returns:
        The flags on Windows, ``0`` elsewhere.

    """
    if sys.platform != "win32":
        return 0
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    if breakaway:
        flags |= CREATE_BREAKAWAY_FROM_JOB
    return flags
