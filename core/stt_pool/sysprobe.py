"""Host probes for the STT supervisor: commit charge, GPU, CPU and interpreter.

Every probe is ML-free and never raises on a probe failure: it returns a
conservative value or ``None`` instead. The supervisor samples them on its own
threads and feeds the readings to the pure scheduler.
"""

from __future__ import annotations

import atexit
import contextlib
import ctypes
import functools
import math
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import psutil  # type: ignore[import-untyped]

from core.stt_pool.config import is_supervisor_managed

if TYPE_CHECKING:
    from collections.abc import Callable

if sys.platform == "win32":
    import winreg

GIB = 1024**3
MIB = 1024**2

# * Sources of a :class:`CommitInfo` reading.
COMMIT_SOURCE_WINDOWS = "GetPerformanceInfo"
COMMIT_SOURCE_PSUTIL = "psutil available+swap"
COMMIT_SOURCE_MEMAVAILABLE = "/proc/meminfo MemAvailable"
COMMIT_SOURCE_UNAVAILABLE = "unavailable"

_MEMINFO = Path("/proc/meminfo")

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_NVIDIA_SMI_QUERY = "name,memory.used,memory.total"
_NVIDIA_SMI_FIELDS = 3
_CPU_REGISTRY_KEY = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
_CPU_REGISTRY_VALUE = "ProcessorNameString"
_UNKNOWN_CPU = "unknown"
# * Upper bound (1 PiB) of an accepted ``nvidia-smi`` memory reading.
_MAX_PLAUSIBLE_VRAM_BYTES = 1024**5

_DWORD = ctypes.c_uint32
_SIZE_T = ctypes.c_size_t

# * nvidia-smi processes that outlived their probe timeout; reaped on later calls
# * so their ``Popen`` objects never emit "subprocess still running" warnings.
_ABANDONED_LOCK = threading.Lock()
_ABANDONED: list[subprocess.Popen[bytes]] = []


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
        free_bytes: ``(CommitLimit - CommitTotal) * PageSize`` on Windows;
            ``MemAvailable`` on managed Linux, available RAM plus free swap
            elsewhere; ``0`` when no probe works.
        limit_bytes: Commit limit, when known.
        used_bytes: Current commit charge, when known.
        source: Probe used (``COMMIT_SOURCE_*``).

    """

    free_bytes: int
    limit_bytes: int | None
    used_bytes: int | None
    source: str

    @property
    def known(self) -> bool:
        """Return whether :attr:`free_bytes` comes from a working probe."""
        return self.source != COMMIT_SOURCE_UNAVAILABLE


def _query_windows_commit() -> CommitInfo | None:
    """Return the Windows commit reading, or ``None`` when the call fails."""
    if sys.platform != "win32":
        return None
    info = _PerformanceInformation()
    info.cb = ctypes.sizeof(info)
    try:
        if not _psapi().GetPerformanceInfo(ctypes.byref(info), info.cb):
            return None
    except OSError:
        return None
    page = int(info.PageSize)
    limit = int(info.CommitLimit) * page
    used = int(info.CommitTotal) * page
    return CommitInfo(max(0, limit - used), limit, used, COMMIT_SOURCE_WINDOWS)


def _query_linux_available() -> CommitInfo:
    """Read managed admission headroom from RAM only, never adding free swap."""
    try:
        for line in _MEMINFO.read_text(encoding="ascii").splitlines():
            label, sep, value = line.partition(":")
            if sep and label == "MemAvailable":
                number, unit = value.split()
                if unit != "kB" or not number.isascii() or not number.isdecimal():
                    break
                return CommitInfo(
                    int(number) * 1024, None, None, COMMIT_SOURCE_MEMAVAILABLE
                )
    except (OSError, UnicodeError, ValueError):
        return CommitInfo(0, None, None, COMMIT_SOURCE_UNAVAILABLE)
    return CommitInfo(0, None, None, COMMIT_SOURCE_UNAVAILABLE)


def query_commit() -> CommitInfo:
    """Return the free system commit.

    Windows uses Win32 ``GetPerformanceInfo``. Managed Linux uses
    ``/proc/meminfo`` ``MemAvailable`` with no swap fallback. Elsewhere, or
    when the Windows call fails, free commit is approximated by available RAM
    plus free swap.

    Returns:
        Commit snapshot. When every probe fails, ``free_bytes`` is ``0`` and
        :attr:`CommitInfo.known` is ``False``; callers treat that as unknown.

    """
    if sys.platform == "linux" and is_supervisor_managed():
        return _query_linux_available()
    reading = _query_windows_commit()
    if reading is not None:
        return reading
    try:
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
    except (OSError, RuntimeError, psutil.Error):
        return CommitInfo(0, None, None, COMMIT_SOURCE_UNAVAILABLE)
    return CommitInfo(
        int(memory.available) + int(swap.free),
        None,
        None,
        COMMIT_SOURCE_PSUTIL,
    )


@dataclass(frozen=True, slots=True)
class GpuSnapshot:
    """One ``nvidia-smi`` reading of GPU 0.

    Attributes:
        name: GPU name, when reported.
        used_bytes: Used VRAM of the whole GPU (all processes).
        total_bytes: Total VRAM.
        taken_at: Clock value (``time.time()`` by default) of the reading.

    """

    name: str | None
    used_bytes: int
    total_bytes: int
    taken_at: float

    @property
    def free_bytes(self) -> int:
        """Return the free VRAM (never negative)."""
        return max(0, self.total_bytes - self.used_bytes)


def find_nvidia_smi() -> str | None:
    """Return the ``nvidia-smi`` path, or ``None`` when it is not installed."""
    return shutil.which("nvidia-smi")


def _mib_to_bytes(raw: str) -> int | None:
    """Parse an ``nvidia-smi`` MiB value; ``[N/A]`` and friends become ``None``."""
    try:
        value = float(raw)
    except ValueError:
        return None
    if not math.isfinite(value) or value < 0:
        return None
    # * The scaled value can overflow to infinity (``1e308`` MiB); anything past
    # * the sanity bound is a garbled reading, not a real GPU.
    scaled = value * MIB
    if not math.isfinite(scaled) or scaled > _MAX_PLAUSIBLE_VRAM_BYTES:
        return None
    return int(scaled)


def _close_pipes(proc: subprocess.Popen[bytes]) -> None:
    """Close the pipes of a finished probe process (idempotent)."""
    for stream in (proc.stdout, proc.stderr):
        if stream is not None:
            with contextlib.suppress(OSError, ValueError):
                stream.close()


def _reap_abandoned() -> None:
    """Forget abandoned probe processes that have exited, closing their pipes."""
    with _ABANDONED_LOCK:
        alive: list[subprocess.Popen[bytes]] = []
        for proc in _ABANDONED:
            if proc.poll() is None:
                alive.append(proc)
            else:
                _close_pipes(proc)
        _ABANDONED[:] = alive


atexit.register(_reap_abandoned)


def _abandon(proc: subprocess.Popen[bytes]) -> None:
    """Kill a probe process that overran its timeout without waiting for it."""
    with contextlib.suppress(OSError):
        proc.kill()
    if proc.poll() is not None:
        _close_pipes(proc)
        return
    with _ABANDONED_LOCK:
        _ABANDONED.append(proc)


def _run_nvidia_smi(executable: str, timeout_s: float) -> str | None:
    """Run one ``nvidia-smi --query-gpu`` for GPU 0 and return its first line.

    A process that overruns *timeout_s* is killed and abandoned (reaped by a
    later call), so this call never waits for a hung driver query.

    Args:
        executable: ``nvidia-smi`` path.
        timeout_s: Wall-clock budget of the query.

    Returns:
        The first output line, or ``None`` on any failure.

    """
    _reap_abandoned()
    argv = [
        executable,
        f"--query-gpu={_NVIDIA_SMI_QUERY}",
        "--format=csv,noheader,nounits",
        "--id=0",
    ]
    try:
        proc = subprocess.Popen(  # noqa: S603 - resolved nvidia-smi, fixed argv
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            creationflags=_NO_WINDOW,
            close_fds=True,
        )
    except (OSError, ValueError):
        return None
    try:
        stdout, _stderr = proc.communicate(timeout=max(0.0, timeout_s))
    except subprocess.TimeoutExpired:
        _abandon(proc)
        return None
    except (OSError, ValueError):
        _abandon(proc)
        return None
    if proc.returncode != 0:
        return None
    lines = stdout.decode("utf-8", errors="replace").strip().splitlines()
    return lines[0] if lines else None


def query_gpu(
    exe: str | None = None,
    *,
    timeout_s: float = 5.0,
    clock: Callable[[], float] = time.time,
) -> GpuSnapshot | None:
    """Read the name and the used/total VRAM of GPU 0 through ``nvidia-smi``.

    ``--id=0`` follows the nvidia-smi (PCI bus) order, while workers run on
    ``cuda:0`` of the CUDA order. On a single-GPU PC both are the same GPU; on a
    multi-GPU machine the reading may describe another GPU.

    Args:
        exe: ``nvidia-smi`` path; resolved from ``PATH`` when ``None``.
        timeout_s: Wall-clock budget; the call never blocks longer.
        clock: Source of :attr:`GpuSnapshot.taken_at`.

    Returns:
        The snapshot, or ``None`` when ``nvidia-smi`` is missing, fails, times
        out, or prints something unexpected.

    """
    executable = exe if exe is not None else find_nvidia_smi()
    if not executable:
        return None
    line = _run_nvidia_smi(executable, timeout_s)
    if line is None:
        return None
    # * GPU names never contain the two numeric fields, so split from the right.
    parts = [part.strip() for part in line.rsplit(",", _NVIDIA_SMI_FIELDS - 1)]
    if len(parts) != _NVIDIA_SMI_FIELDS:
        return None
    used = _mib_to_bytes(parts[1])
    total = _mib_to_bytes(parts[2])
    if used is None or total is None or total <= 0:
        return None
    name = parts[0] or None
    return GpuSnapshot(name=name, used_bytes=used, total_bytes=total, taken_at=clock())


def logical_cpu_count() -> int:
    """Return the number of logical CPUs (at least 1)."""
    return max(1, os.cpu_count() or 1)


def total_memory_bytes() -> int | None:
    """Return the installed physical memory, or ``None`` when unknown."""
    try:
        total = int(psutil.virtual_memory().total)
    except (OSError, RuntimeError, psutil.Error):
        return None
    return total if total > 0 else None


def process_private_bytes() -> int | None:
    """Return the private bytes (Windows commit) of the calling process.

    One ``GetProcessMemoryInfo`` call through ``psutil``; cheap enough to run
    after every job.

    Returns:
        ``PrivateUsage`` of this process, or ``None`` on a platform without it
        (``psutil`` reports ``private`` only on Windows) or when the query
        fails.

    """
    try:
        info = psutil.Process().memory_info()
    except (OSError, RuntimeError, psutil.Error):
        return None
    private = getattr(info, "private", None)
    if isinstance(private, bool) or not isinstance(private, int) or private < 0:
        return None
    return private


def _windows_cpu_name() -> str | None:
    """Return the marketing CPU name from the Windows registry, if available."""
    if sys.platform != "win32":
        return None
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _CPU_REGISTRY_KEY) as key:
            value, _kind = winreg.QueryValueEx(key, _CPU_REGISTRY_VALUE)
    except OSError:
        return None
    text = str(value).strip()
    return text or None


def _proc_cpuinfo_name() -> str | None:
    """Return the ``model name`` of ``/proc/cpuinfo``, if available."""
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        label, sep, value = line.partition(":")
        if sep and label.strip() == "model name" and value.strip():
            return value.strip()
    return None


def cpu_name() -> str:
    """Return a human-readable CPU name, or ``unknown``."""
    name = _windows_cpu_name() or _proc_cpuinfo_name() or platform.processor()
    return " ".join(str(name).split()) or _UNKNOWN_CPU


def hardware_fingerprint(gpu_name: str | None) -> dict[str, object]:
    """Describe the host for learned-estimate invalidation.

    Args:
        gpu_name: GPU name from :func:`query_gpu`, or ``None`` without a GPU.

    Returns:
        ``{"cpu", "logical_cpus", "ram_gib", "gpu"}``; ``ram_gib`` is the
        installed RAM rounded to whole GiB (``0`` when unknown).

    """
    total = total_memory_bytes()
    gpu = gpu_name.strip() if gpu_name is not None else None
    return {
        "cpu": cpu_name(),
        "logical_cpus": logical_cpu_count(),
        "ram_gib": round(total / GIB) if total is not None else 0,
        "gpu": gpu or None,
    }


def child_interpreter() -> tuple[str, dict[str, str]]:
    """Return the interpreter that starts pool processes and its extra environment.

    In a Windows venv ``sys.executable`` is a launcher that starts the real
    interpreter as its own child, which could allocate before the launcher is
    put into a Job Object and would add an extra process level. Pool processes
    therefore start from the base interpreter with ``__PYVENV_LAUNCHER__`` set
    (as :mod:`multiprocessing` does), so the spawned process is the interpreter
    itself and still sees the venv's packages.

    Returns:
        ``(executable, environment additions)``.

    """
    base = getattr(sys, "_base_executable", None) or sys.executable
    if (
        sys.platform == "win32"
        and Path(base) != Path(sys.executable)
        and Path(base).is_file()
    ):
        return str(base), {"__PYVENV_LAUNCHER__": sys.executable}
    return sys.executable, {}
