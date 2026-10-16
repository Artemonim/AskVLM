"""Per-worker Windows Job Objects and creation flags for detached processes.

Every STT worker process (and everything it starts, such as ffmpeg) lives in
its own Job Object owned by the supervisor:

* ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``: when the supervisor closes the handle,
  or dies for any reason and Windows closes it, every process of the job is
  terminated, so no orphan keeps a model in memory;
* ``JOB_OBJECT_LIMIT_JOB_MEMORY`` (optional): the summed commit charge of the
  job's processes is capped. An allocation that would cross the cap fails, and
  Windows posts ``JOB_OBJECT_MSG_JOB_MEMORY_LIMIT`` to the job's completion
  port; a monitor thread reports it once. :meth:`WorkerJob.set_memory_limit`
  changes the cap of the live job and re-arms that one-shot report.

Jobs nest on Windows 8 and newer: when the supervisor itself runs inside a job,
a worker's job becomes a child of that job and both limits apply. Off Windows
the job is inactive and :attr:`WorkerJob.error` says why.

:func:`suppress_error_dialogs` turns off the modal Windows error dialogs of a
headless process and of the children it starts afterwards: a child that fails
to start or crashes then exits with its status instead of waiting for a click.
The module uses ``ctypes`` only and imports no ML packages.
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
# * ``SetErrorMode`` flags: no critical-error box (a loader failure such as
# * 0xC000012D STATUS_COMMITMENT_LIMIT included), no Windows Error Reporting box
# * after a crash, no box when ``OpenFile`` misses a file.
_SEM_FAILCRITICALERRORS = 0x0001
_SEM_NOGPFAULTERRORBOX = 0x0002
_SEM_NOOPENFILEERRORBOX = 0x8000
_SUPPRESSED_ERROR_DIALOGS = (
    _SEM_FAILCRITICALERRORS | _SEM_NOGPFAULTERRORBOX | _SEM_NOOPENFILEERRORBOX
)
# * Completion keys of the job's notification port. ``_REARM_KEY`` packets are
# * posted by :meth:`WorkerJob.set_memory_limit`.
_JOB_KEY = 1
_STOP_KEY = 2
_REARM_KEY = 3
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
    lib.GetErrorMode.argtypes = ()
    lib.GetErrorMode.restype = ctypes.c_uint
    lib.SetErrorMode.argtypes = (ctypes.c_uint,)
    lib.SetErrorMode.restype = ctypes.c_uint
    return lib


def _last_error() -> int:
    """Return the calling thread's last Win32 error (``0`` off Windows)."""
    if sys.platform != "win32":
        return 0
    return int(ctypes.get_last_error())


class WorkerJob:
    """Kill-on-close Job Object of one worker, with an optional commit cap.

    The memory-limit report is one-shot: the monitor reports the first
    ``JOB_OBJECT_MSG_JOB_MEMORY_LIMIT`` and ignores later ones until
    :meth:`set_memory_limit` re-arms it, so a burst of denied allocations is
    one report and a hit of a raised cap is one more. Windows itself posts the
    message only for the first denied allocation of each process of the job
    (measured on Windows 11 22631: three denials of one process gave one
    message, and neither a raised cap, a toggled limit flag nor a second port
    association brought another; a child process got its own message). A
    process whose denial was reported is therefore never reported again,
    whatever the cap; a re-armed report sees processes not denied before.

    Attributes:
        memory_limit_bytes: Job-wide commit cap, or ``None`` for no cap.
        memory_limit_pid: Process id reported with the latest memory-limit
            report, or ``None`` while the cap was not hit.
        memory_limit_hits: Number of memory-limit reports so far (at most one
            per arming); it only grows and is updated on the monitor thread
            before the callback runs.
        error: Why the job is unusable (``None`` when it is active).
        monitor_error: Why a hit of an enforced cap cannot be reported.
        assign_error: Why the last :meth:`assign` failed, if it did.
        limit_error: Why the last :meth:`set_memory_limit` failed, if it did.

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
            on_memory_limit: Called from the monitor thread with the offending
                process id when the cap is hit, once per arming of the report
                (see :meth:`set_memory_limit`).

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
        self.memory_limit_hits = 0
        self.error: str | None = None
        self.monitor_error: str | None = None
        self.assign_error: str | None = None
        self.limit_error: str | None = None
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
        """Read job notifications until :meth:`close`; report the cap once per arming.

        The arming lives on this thread only: a ``_REARM_KEY`` packet arms the
        next report. The port delivers packets in FIFO order, so the
        notifications Windows queued before a re-arm packet are drained while
        disarmed and never count as a hit of the new cap.

        Args:
            port: Completion port associated with the job.

        """
        if sys.platform != "win32":
            return
        kernel32 = _kernel32()
        message = _DWORD()
        key = ctypes.c_size_t()
        overlapped = ctypes.c_void_p()
        armed = True
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
            if key.value == _REARM_KEY:
                armed = True
                continue
            if message.value == _JOB_OBJECT_MSG_JOB_MEMORY_LIMIT and armed:
                armed = False
                # * For this message the "overlapped" value is the process id.
                pid = int(overlapped.value or 0)
                self.memory_limit_pid = pid
                self.memory_limit_hits += 1
                self._report_memory_limit(pid)

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

    def set_memory_limit(self, limit_bytes: int) -> bool:
        """Change the job-wide commit cap of the live job and re-arm its report.

        Query-modify-set of ``JOBOBJECT_EXTENDED_LIMIT_INFORMATION`` under the
        handle lock: only ``JobMemoryLimit`` changes, every limit flag
        (``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` included) and every other field
        keep their current values. The re-arm packet is posted before the new
        cap is set, so a notified denial after the change is always reported;
        one denied under the old cap in between may be reported too (never
        missed, at worst counted early). Windows notifies only the first
        denial of each process (see the class docstring), so after the change
        only processes not denied before can be reported.

        Args:
            limit_bytes: New job-wide commit cap in bytes.

        Returns:
            ``True`` when the new cap is in effect. ``False`` (with
            :attr:`limit_error`) when the job is inactive or closed, was
            created without a cap, has no working memory-limit monitor (a hit
            of the new cap could not be reported), or a Win32 call failed; the
            old cap then stays in effect (after a failed set the report stays
            re-armed, so a later hit of the old cap is reported again).

        Raises:
            ValueError: When *limit_bytes* is not positive.

        """
        if limit_bytes <= 0:
            msg = f"limit_bytes must be positive, got {limit_bytes}"
            raise ValueError(msg)
        with self._handle_lock:
            error = self._set_memory_limit_locked(int(limit_bytes))
            self.limit_error = error
            if error is not None:
                return False
            self.memory_limit_bytes = int(limit_bytes)
            return True

    def _limit_change_blocker_locked(self) -> str | None:
        """Return why the cap cannot change; the caller holds the handle lock."""
        if self._handle is None or sys.platform != "win32":
            return self.error or "the job is closed"
        if self.memory_limit_bytes is None:
            return "the job was created without a commit cap"
        if self._port is None:
            return self.monitor_error or "the job has no memory-limit monitor"
        return None

    def _set_memory_limit_locked(self, limit_bytes: int) -> str | None:
        """Query, re-arm and set the cap; the caller holds the handle lock.

        Returns:
            ``None`` when the new cap is in effect, otherwise why not.

        """
        blocker = self._limit_change_blocker_locked()
        if blocker is not None:
            return blocker
        kernel32 = _kernel32()
        info = _JobExtendedLimitInformation()
        if not kernel32.QueryInformationJobObject(
            self._handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
            None,
        ):
            return f"QueryInformationJobObject failed (WinError {_last_error()})"
        if not kernel32.PostQueuedCompletionStatus(self._port, 0, _REARM_KEY, None):
            return f"PostQueuedCompletionStatus failed (WinError {_last_error()})"
        info.BasicLimitInformation.LimitFlags |= _JOB_OBJECT_LIMIT_JOB_MEMORY
        info.JobMemoryLimit = limit_bytes
        if not kernel32.SetInformationJobObject(
            self._handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            return f"SetInformationJobObject failed (WinError {_last_error()})"
        return None

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


def suppress_error_dialogs() -> int | None:
    """Turn off the modal Windows error dialogs of this process and its children.

    Adds ``SEM_FAILCRITICALERRORS``, ``SEM_NOGPFAULTERRORBOX`` and
    ``SEM_NOOPENFILEERRORBOX`` to the process error mode, keeping the flags
    already set. Processes started afterwards inherit the mode unless they are
    created with ``CREATE_DEFAULT_ERROR_MODE``, which ``subprocess`` and
    ffmpeg-python never pass. A child that fails to load (for example
    ``0xC000012D`` STATUS_COMMITMENT_LIMIT under the worker's Job Object cap)
    or crashes then exits with its status instead of blocking on an
    "Application Error" box until someone clicks it. A crash is no longer
    reported to Windows Error Reporting either.

    Returns:
        The error mode in effect afterwards, or ``None`` off Windows or when
        ``kernel32`` is unavailable (nothing is changed then).

    """
    if sys.platform != "win32":
        return None
    try:
        kernel32 = _kernel32()
    except OSError:
        return None
    kernel32.SetErrorMode(int(kernel32.GetErrorMode()) | _SUPPRESSED_ERROR_DIALOGS)
    return int(kernel32.GetErrorMode())


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
