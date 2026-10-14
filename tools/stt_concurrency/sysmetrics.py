"""Host resource probes: commit charge, process-tree memory, GPU, CPU, job objects.

Everything here runs in the parent orchestrator; children never import it.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import psutil  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

GIB = float(1024**3)
MIB = float(1024**2)

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_NVIDIA_SMI_TIMEOUT_S = 5.0
_NVIDIA_SMI_QUERY = "memory.used,memory.total,utilization.gpu"
_NVIDIA_SMI_FIELDS = 3
_SAMPLER_JOIN_TIMEOUT_S = 10.0
# * The VRAM poller's own nvidia-smi processes are not part of the benchmark.
_EXCLUDED_PROCESS_NAMES = frozenset({"nvidia-smi.exe", "nvidia-smi"})

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
# * Exit code of processes terminated by the memory cap or the commit watchdog.
JOB_TERMINATE_EXIT_CODE = 0xE0000017
COMMIT_SOURCE_WINDOWS = "GetPerformanceInfo"
_JOIN_TIMEOUT_S = 5.0


class _PerformanceInformation(ctypes.Structure):
    """``PERFORMANCE_INFORMATION`` of ``GetPerformanceInfo`` (psapi)."""

    _fields_ = (
        ("cb", _DWORD),
        ("CommitTotal", _SIZE_T),
        ("CommitLimit", _SIZE_T),
        ("CommitPeak", _SIZE_T),
        ("PhysicalTotal", _SIZE_T),
        ("PhysicalAvailable", _SIZE_T),
        ("SystemCache", _SIZE_T),
        ("KernelTotal", _SIZE_T),
        ("KernelPaged", _SIZE_T),
        ("KernelNonpaged", _SIZE_T),
        ("PageSize", _SIZE_T),
        ("HandleCount", _DWORD),
        ("ProcessCount", _DWORD),
        ("ThreadCount", _DWORD),
    )


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
    lib.OpenProcess.argtypes = (_DWORD, ctypes.c_int, _DWORD)
    lib.OpenProcess.restype = ctypes.c_void_p
    lib.AssignProcessToJobObject.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    lib.AssignProcessToJobObject.restype = ctypes.c_int
    lib.CloseHandle.argtypes = (ctypes.c_void_p,)
    lib.CloseHandle.restype = ctypes.c_int
    lib.TerminateJobObject.argtypes = (ctypes.c_void_p, ctypes.c_uint)
    lib.TerminateJobObject.restype = ctypes.c_int
    lib.QueryInformationJobObject.argtypes = (
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        _DWORD,
        ctypes.c_void_p,
    )
    lib.QueryInformationJobObject.restype = ctypes.c_int
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
    return lib


@functools.cache
def _psapi() -> ctypes.CDLL:
    """Return ``psapi`` with ``GetPerformanceInfo`` declared.

    Raises:
        OSError: On non-Windows platforms.

    """
    if sys.platform != "win32":
        msg = "psapi is only available on Windows"
        raise OSError(msg)
    lib = ctypes.WinDLL("psapi", use_last_error=True)
    lib.GetPerformanceInfo.argtypes = (ctypes.c_void_p, _DWORD)
    lib.GetPerformanceInfo.restype = ctypes.c_int
    return lib


@dataclass(frozen=True, slots=True)
class CommitInfo:
    """System commit charge snapshot.

    Attributes:
        free_bytes: ``(CommitLimit - CommitTotal) * PageSize`` on Windows.
        limit_bytes: Commit limit, when known.
        used_bytes: Current commit charge, when known.
        source: Probe used.

    """

    free_bytes: int
    limit_bytes: int | None
    used_bytes: int | None
    source: str


def query_commit() -> CommitInfo:
    """Return the free system commit (Win32 ``GetPerformanceInfo``).

    Off Windows (or when the call fails) the free commit is approximated by
    available RAM plus free swap.

    Returns:
        Commit snapshot.

    """
    if sys.platform == "win32":
        info = _PerformanceInformation()
        info.cb = ctypes.sizeof(info)
        with contextlib.suppress(OSError):
            if _psapi().GetPerformanceInfo(ctypes.byref(info), info.cb):
                page = int(info.PageSize)
                limit = int(info.CommitLimit) * page
                used = int(info.CommitTotal) * page
                return CommitInfo(limit - used, limit, used, COMMIT_SOURCE_WINDOWS)
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return CommitInfo(
        int(memory.available) + int(swap.free),
        None,
        None,
        "psutil available+swap",
    )


@dataclass(frozen=True, slots=True)
class GpuSample:
    """One ``nvidia-smi`` reading of GPU 0.

    Attributes:
        t: ``time.perf_counter()`` of the reading.
        used_mb: Used VRAM (whole GPU, all processes).
        total_mb: Total VRAM.
        util_percent: GPU utilization, when reported.

    """

    t: float
    used_mb: float
    total_mb: float
    util_percent: float | None

    @property
    def free_mb(self) -> float:
        """Free VRAM."""
        return self.total_mb - self.used_mb


def find_nvidia_smi() -> str | None:
    """Return the ``nvidia-smi`` path, or ``None`` when it is not installed."""
    return shutil.which("nvidia-smi")


def _float_or_none(raw: str) -> float | None:
    """Parse an ``nvidia-smi`` number; ``[N/A]`` and friends become ``None``."""
    try:
        return float(raw)
    except ValueError:
        return None


def _run_nvidia_smi(executable: str, query: str) -> str | None:
    """Run one ``nvidia-smi --query-gpu`` for GPU 0 and return its first line."""
    try:
        completed = subprocess.run(  # noqa: S603 - resolved nvidia-smi, fixed argv
            [
                executable,
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_S,
            check=False,
            creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = completed.stdout.strip().splitlines()
    if completed.returncode != 0 or not lines:
        return None
    return lines[0]


def query_gpu(executable: str) -> GpuSample | None:
    """Read used/total VRAM and utilization of GPU 0.

    ``--id=0`` follows the nvidia-smi (PCI bus) order, while the workers run on
    ``cuda:0`` of the CUDA order (fastest first by default, and subject to
    ``CUDA_VISIBLE_DEVICES``). On a single-GPU PC both are the same GPU; on a
    multi-GPU machine they can differ, and then the VRAM numbers and the VRAM
    guard describe another GPU.

    Args:
        executable: ``nvidia-smi`` path.

    Returns:
        The sample, or ``None`` when the query fails.

    """
    line = _run_nvidia_smi(executable, _NVIDIA_SMI_QUERY)
    if line is None:
        return None
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < _NVIDIA_SMI_FIELDS:
        return None
    used = _float_or_none(parts[0])
    total = _float_or_none(parts[1])
    if used is None or total is None:
        return None
    return GpuSample(time.perf_counter(), used, total, _float_or_none(parts[2]))


def query_gpu_name(executable: str) -> str | None:
    """Return the name of GPU 0, or ``None``."""
    return _run_nvidia_smi(executable, "name")


class _PeriodicSampler:
    """Background thread that calls ``sample`` every ``interval_s`` seconds."""

    def __init__(
        self, interval_s: float, name: str, sample: Callable[[], None]
    ) -> None:
        self._interval_s = interval_s
        self._sample = sample
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name=name, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self._interval_s)

    def start(self) -> None:
        """Start sampling."""
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and wait for the thread (bounded)."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=_SAMPLER_JOIN_TIMEOUT_S)


@dataclass(frozen=True, slots=True)
class _MemorySample:
    t: float
    private_bytes: int
    working_set_bytes: int
    free_commit_bytes: int
    processes: int


@dataclass(frozen=True, slots=True)
class MemoryStats:
    """Peaks of the summed memory of the parent and all of its descendants.

    Attributes:
        peak_private_bytes: Peak sum of private bytes (load + run).
        peak_private_run_bytes: Peak sum of private bytes in the timed run.
        peak_working_set_bytes: Peak sum of working sets (load + run).
        peak_working_set_run_bytes: Peak sum of working sets in the timed run.
        min_free_commit_bytes: Lowest system free commit seen.
        max_processes: Largest process count of the tree.
        samples: Number of samples.

    """

    peak_private_bytes: int | None
    peak_private_run_bytes: int | None
    peak_working_set_bytes: int | None
    peak_working_set_run_bytes: int | None
    min_free_commit_bytes: int | None
    max_processes: int | None
    samples: int


def _max_or_none(values: Sequence[int]) -> int | None:
    return max(values) if values else None


class ProcessTreeSampler:
    """Samples private bytes and working sets of this process tree."""

    def __init__(self, interval_s: float) -> None:
        """Create a sampler rooted at the current process.

        Args:
            interval_s: Sampling period.

        """
        self._root = psutil.Process()
        self._names: dict[int, str] = {}
        self._samples: list[_MemorySample] = []
        self._lock = threading.Lock()
        self._sampler = _PeriodicSampler(interval_s, "memory-sampler", self.sample)

    def start(self) -> None:
        """Start background sampling."""
        self._sampler.start()

    def stop(self) -> None:
        """Stop background sampling and take a final sample."""
        self._sampler.stop()
        self.sample()

    def _name(self, proc: psutil.Process) -> str:
        name = self._names.get(proc.pid)
        if name is None:
            name = str(proc.name()).lower()
            self._names[proc.pid] = name
        return name

    def sample(self) -> None:
        """Take one sample of the whole tree (safe to call from any thread)."""
        try:
            procs = [self._root, *self._root.children(recursive=True)]
        except psutil.Error:
            procs = [self._root]
        private_total = 0
        working_set_total = 0
        counted = 0
        for proc in procs:
            try:
                if (
                    proc.pid != self._root.pid
                    and self._name(proc) in _EXCLUDED_PROCESS_NAMES
                ):
                    continue
                memory = proc.memory_info()
            except psutil.Error:
                continue
            private_total += int(getattr(memory, "private", memory.rss))
            working_set_total += int(memory.rss)
            counted += 1
        free_commit = query_commit().free_bytes
        with self._lock:
            self._samples.append(
                _MemorySample(
                    time.perf_counter(),
                    private_total,
                    working_set_total,
                    free_commit,
                    counted,
                ),
            )

    def stats(self, run_start: float | None, run_end: float | None) -> MemoryStats:
        """Aggregate the samples.

        Args:
            run_start: ``perf_counter`` of the clock start (all children ready).
            run_end: ``perf_counter`` of the last result.

        Returns:
            Peaks for the whole configuration and for the timed run.

        """
        with self._lock:
            samples = list(self._samples)
        in_run = [
            s
            for s in samples
            if run_start is not None
            and s.t >= run_start
            and (run_end is None or s.t <= run_end)
        ]
        return MemoryStats(
            peak_private_bytes=_max_or_none([s.private_bytes for s in samples]),
            peak_private_run_bytes=_max_or_none([s.private_bytes for s in in_run]),
            peak_working_set_bytes=_max_or_none([s.working_set_bytes for s in samples]),
            peak_working_set_run_bytes=_max_or_none(
                [s.working_set_bytes for s in in_run]
            ),
            min_free_commit_bytes=(
                min(s.free_commit_bytes for s in samples) if samples else None
            ),
            max_processes=_max_or_none([s.processes for s in samples]),
            samples=len(samples),
        )


@dataclass(frozen=True, slots=True)
class GpuStats:
    """Whole-GPU VRAM and utilization over one configuration.

    Attributes:
        baseline_mb: Used VRAM before the children were spawned.
        peak_mb: Peak used VRAM (load + run).
        peak_run_mb: Peak used VRAM in the timed run.
        total_mb: Total VRAM.
        util_avg_run_percent: Mean GPU utilization in the timed run.
        samples: Number of samples.

    """

    baseline_mb: float | None
    peak_mb: float | None
    peak_run_mb: float | None
    total_mb: float | None
    util_avg_run_percent: float | None
    samples: int

    @property
    def delta_mb(self) -> float | None:
        """Peak (load or run) minus baseline."""
        if self.peak_mb is None or self.baseline_mb is None:
            return None
        return self.peak_mb - self.baseline_mb

    @property
    def delta_run_mb(self) -> float | None:
        """Peak of the timed run minus baseline."""
        if self.peak_run_mb is None or self.baseline_mb is None:
            return None
        return self.peak_run_mb - self.baseline_mb


class GpuPoller:
    """Polls ``nvidia-smi`` in a background thread."""

    def __init__(self, executable: str, interval_s: float) -> None:
        """Create a poller.

        Args:
            executable: ``nvidia-smi`` path.
            interval_s: Polling period.

        """
        self._executable = executable
        self._samples: list[GpuSample] = []
        self._lock = threading.Lock()
        self._sampler = _PeriodicSampler(interval_s, "vram-poller", self.sample)

    def start(self) -> None:
        """Start background polling."""
        self._sampler.start()

    def stop(self) -> None:
        """Stop background polling."""
        self._sampler.stop()

    def sample(self) -> None:
        """Take one reading."""
        reading = query_gpu(self._executable)
        if reading is not None:
            with self._lock:
                self._samples.append(reading)

    def stats(
        self,
        baseline: GpuSample | None,
        run_start: float | None,
        run_end: float | None,
    ) -> GpuStats:
        """Aggregate the readings.

        Args:
            baseline: Reading taken before the children were spawned.
            run_start: ``perf_counter`` of the clock start.
            run_end: ``perf_counter`` of the last result.

        Returns:
            VRAM peaks and mean utilization.

        """
        with self._lock:
            samples = list(self._samples)
        in_run = [
            s
            for s in samples
            if run_start is not None
            and s.t >= run_start
            and (run_end is None or s.t <= run_end)
        ]
        utils = [s.util_percent for s in in_run if s.util_percent is not None]
        total = (
            samples[-1].total_mb
            if samples
            else (baseline.total_mb if baseline else None)
        )
        return GpuStats(
            baseline_mb=baseline.used_mb if baseline else None,
            peak_mb=max((s.used_mb for s in samples), default=None),
            peak_run_mb=max((s.used_mb for s in in_run), default=None),
            total_mb=total,
            util_avg_run_percent=sum(utils) / len(utils) if utils else None,
            samples=len(samples),
        )


@dataclass(frozen=True, slots=True)
class CpuSnapshot:
    """System-wide CPU time counters.

    Attributes:
        busy_s: Non-idle CPU seconds summed over all logical CPUs.
        total_s: All CPU seconds summed over all logical CPUs.

    """

    busy_s: float
    total_s: float


def cpu_snapshot() -> CpuSnapshot:
    """Return system-wide CPU counters (psutil semantics: busy = total - idle)."""
    times = psutil.cpu_times()
    total = float(sum(times))
    return CpuSnapshot(busy_s=total - float(times.idle), total_s=total)


def cpu_busy_percent(
    start: CpuSnapshot | None, end: CpuSnapshot | None
) -> float | None:
    """Return the mean system-wide CPU utilization between two snapshots.

    Args:
        start: Snapshot at the clock start.
        end: Snapshot at the last result.

    Returns:
        Busy percentage in ``[0, 100]``, or ``None`` when unknown.

    """
    if start is None or end is None:
        return None
    total = end.total_s - start.total_s
    if total <= 0:
        return None
    return max(0.0, min(100.0, 100.0 * (end.busy_s - start.busy_s) / total))


def kill_process_tree(pid: int, *, timeout_s: float = 10.0) -> None:
    """Kill *pid* and all of its descendants, then wait for them.

    Args:
        pid: Root process id.
        timeout_s: Wait budget after the kill.

    """
    try:
        root = psutil.Process(pid)
    except psutil.Error:
        return
    try:
        procs = [*root.children(recursive=True), root]
    except psutil.Error:
        procs = [root]
    for proc in procs:
        with contextlib.suppress(psutil.Error):
            proc.kill()
    with contextlib.suppress(psutil.Error):
        psutil.wait_procs(procs, timeout=timeout_s)


class KillOnCloseJob:
    """Windows job object that kills every assigned process when it is closed.

    The handle lives as long as the parent: if the parent dies for any reason,
    Windows closes the handle and terminates every benchmark child (and its
    ffmpeg helpers), so no orphan keeps a model in memory.

    With ``memory_limit_bytes`` the job also caps the summed commit charge of
    all its processes (``JOB_OBJECT_LIMIT_JOB_MEMORY``): an allocation that
    would cross the cap fails, and Windows posts
    ``JOB_OBJECT_MSG_JOB_MEMORY_LIMIT`` to the job's completion port. A monitor
    thread reads the port and calls ``on_memory_limit(pid)`` once. Jobs nest
    (Windows 8+): a step job created while its processes already belong to the
    run job becomes a child of the run job, and both limits apply.
    """

    def __init__(
        self,
        *,
        memory_limit_bytes: int | None = None,
        on_memory_limit: Callable[[int], None] | None = None,
    ) -> None:
        """Create the job; on failure :attr:`error` explains why it is inactive.

        Args:
            memory_limit_bytes: Job-wide commit cap, or ``None`` for no cap.
            on_memory_limit: Called once (from the monitor thread) with the
                offending process id when the cap is hit.

        """
        self._handle: int | None = None
        self._port: int | None = None
        self._monitor: threading.Thread | None = None
        self._stop = threading.Event()
        self._handle_lock = threading.Lock()
        self._on_memory_limit = on_memory_limit
        self.memory_limit_bytes = memory_limit_bytes
        self.memory_limit_pid: int | None = None
        # * error: the job is unusable; monitor_error: the cap holds, unreported.
        self.error: str | None = None
        self.monitor_error: str | None = None
        if sys.platform != "win32":
            self.error = "job objects are Windows-only"
            return
        kernel32 = _kernel32()
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            self.error = f"CreateJobObjectW failed (WinError {ctypes.get_last_error()})"
            return
        info = _JobExtendedLimitInformation()
        flags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if memory_limit_bytes is not None:
            flags |= _JOB_OBJECT_LIMIT_JOB_MEMORY
            info.JobMemoryLimit = max(1, int(memory_limit_bytes))
        info.BasicLimitInformation.LimitFlags = flags
        if not kernel32.SetInformationJobObject(
            handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            self.error = (
                f"SetInformationJobObject failed (WinError {ctypes.get_last_error()})"
            )
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
                f"CreateIoCompletionPort failed (WinError {ctypes.get_last_error()}); "
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
                "Job completion port association failed "
                f"(WinError {ctypes.get_last_error()}); the cap is enforced but "
                "not reported"
            )
            kernel32.CloseHandle(port)
            return
        self._port = int(port)
        self._monitor = threading.Thread(
            target=self._watch_port,
            name="job-memory-monitor",
            daemon=True,
        )
        self._monitor.start()

    def _watch_port(self) -> None:
        """Read job notifications until :meth:`close`; report the memory cap once."""
        port = self._port
        if sys.platform != "win32" or port is None:
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
                if self._on_memory_limit is not None:
                    self._on_memory_limit(self.memory_limit_pid)

    @property
    def active(self) -> bool:
        """Whether children can be assigned."""
        return self._handle is not None

    @property
    def memory_limit_hit(self) -> bool:
        """Whether the job reported its memory cap."""
        return self.memory_limit_pid is not None

    def terminate(self) -> bool:
        """Terminate every process of the job at once.

        Returns:
            ``True`` when the job was terminated.

        """
        with self._handle_lock:
            if self._handle is None or sys.platform != "win32":
                return False
            return bool(
                _kernel32().TerminateJobObject(self._handle, JOB_TERMINATE_EXIT_CODE),
            )

    def peak_memory_bytes(self) -> int | None:
        """Return the peak summed commit of the job's processes, if known."""
        with self._handle_lock:
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

    def assign(self, pid: int) -> bool:
        """Assign process *pid* to the job.

        Args:
            pid: Process id.

        Returns:
            ``True`` when the process is now in the job.

        """
        with self._handle_lock:
            if self._handle is None or sys.platform != "win32":
                return False
            kernel32 = _kernel32()
            process = kernel32.OpenProcess(
                _PROCESS_SET_QUOTA | _PROCESS_TERMINATE,
                0,
                pid,
            )
            if not process:
                return False
            try:
                return bool(kernel32.AssignProcessToJobObject(self._handle, process))
            finally:
                kernel32.CloseHandle(process)

    def close(self) -> None:
        """Stop the monitor and close the job; Windows kills what is still assigned.

        The monitor is joined without holding the handle lock, because its
        callback may call :meth:`terminate`.
        """
        self._stop.set()
        port = self._port
        if port is not None and sys.platform == "win32":
            _kernel32().PostQueuedCompletionStatus(port, 0, _STOP_KEY, None)
        if self._monitor is not None:
            self._monitor.join(timeout=_JOIN_TIMEOUT_S)
            self._monitor = None
        with self._handle_lock:
            self._port = None
            if port is not None and sys.platform == "win32":
                _kernel32().CloseHandle(port)
            handle, self._handle = self._handle, None
            if handle is not None and sys.platform == "win32":
                _kernel32().CloseHandle(handle)


class CommitWatchdog:
    """Polls the system free commit and fires once when it drops below a reserve.

    It measures the whole system (``GetPerformanceInfo``), so it also reacts to
    growth of processes outside the benchmark. Off Windows (no commit charge
    probe) it is inactive and :attr:`error` says why.
    """

    def __init__(
        self,
        reserve_bytes: int,
        interval_s: float,
        on_low: Callable[[CommitInfo], None],
    ) -> None:
        """Create a watchdog (not started).

        Args:
            reserve_bytes: Fire when free commit falls below this.
            interval_s: Polling period (keep it at or below 0.5 s).
            on_low: Called once, from the watchdog thread, with the reading.

        """
        self._reserve_bytes = reserve_bytes
        self._interval_s = interval_s
        self._on_low = on_low
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.fired: CommitInfo | None = None
        self.error: str | None = None
        if query_commit().source != COMMIT_SOURCE_WINDOWS:
            self.error = "system commit charge is only measured on Windows"

    def _loop(self) -> None:
        while not self._stop.wait(self._interval_s):
            reading = query_commit()
            if reading.free_bytes < self._reserve_bytes:
                self.fired = reading
                self._on_low(reading)
                return

    def start(self) -> None:
        """Start polling (no-op when inactive)."""
        if self.error is None and self._thread is None:
            self._thread = threading.Thread(
                target=self._loop,
                name="commit-watchdog",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop polling and wait for the thread (bounded)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=_JOIN_TIMEOUT_S)
            self._thread = None
