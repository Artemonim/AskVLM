"""Win32 signals and bounded loader probes for isolated pool scenarios."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import mmap
import os
import queue
import subprocess
import sys
import threading
import uuid
from typing import TYPE_CHECKING, Any, Self, cast

import psutil  # type: ignore[import-untyped]

from core.stt_pool import winjob
from core.stt_pool.sysprobe import child_interpreter

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import TracebackType

WAIT_S = 30.0
SUPPRESSED_MODE = 0x8003
_CREATE_SUSPENDED = 0x00000004


class WinEvent:
    """Own a named manual-reset event shared by a test and its fake engine."""

    def __init__(self, name: str | None = None) -> None:
        self.name = name or f"Local\\AskVLM-test-{uuid.uuid4().hex}"
        self._kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel.CreateEventW.argtypes = (
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_wchar_p,
        )
        self._kernel.CreateEventW.restype = ctypes.c_void_p
        self._kernel.SetEvent.argtypes = (ctypes.c_void_p,)
        self._kernel.SetEvent.restype = ctypes.c_int
        self._kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint)
        self._kernel.WaitForSingleObject.restype = ctypes.c_uint
        self._kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
        self._kernel.CloseHandle.restype = ctypes.c_int
        self._handle = self._kernel.CreateEventW(None, 1, 0, self.name)
        if not self._handle:
            raise ctypes.WinError(ctypes.get_last_error())

    def __enter__(self) -> Self:
        """Keep the named signal alive for the scenario."""
        return self

    def __exit__(
        self,
        _kind: type[BaseException] | None,
        _value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Release waiting engines and close only this owned event handle."""
        # * Release an engine gate on assertion failures before closing the handle.
        self.set()
        self._kernel.CloseHandle(self._handle)

    def set(self) -> None:
        """Signal the event without relying on a scheduling delay."""
        if not self._kernel.SetEvent(self._handle):
            raise ctypes.WinError(ctypes.get_last_error())

    def wait(self) -> None:
        """Wait for a signal; the deadline only bounds a failed scenario."""
        status = self._kernel.WaitForSingleObject(self._handle, int(WAIT_S * 1000))
        if status == 0x102:
            msg = f"test event was not signalled: {self.name}"
            raise TimeoutError(msg)
        if status != 0:
            raise ctypes.WinError(ctypes.get_last_error())


def signal(name: object) -> None:
    """Notify a controller when its instructions name a Win32 event."""
    if isinstance(name, str):
        with WinEvent(name) as event:
            event.set()


def wait_for_signal(name: object) -> None:
    """Hold a fake engine at an explicit process boundary."""
    if isinstance(name, str):
        with WinEvent(name) as event:
            event.wait()


def error_mode_details() -> dict[str, object]:
    """Observe the actual worker mode and the mode inherited by a real child."""
    executable, environment = child_interpreter()
    child = subprocess.run(  # noqa: S603 - fixed test module, no shell
        [executable, "-m", "tests.stt_pool_windows_support", "--mode", "error-mode"],
        env={**os.environ, **environment},
        capture_output=True,
        check=True,
        timeout=WAIT_S,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    return {
        "initial_mode": int(os.environ["ASKVLM_TEST_INITIAL_ERROR_MODE"]),
        "worker_mode": int(winjob._kernel32().GetErrorMode()),  # noqa: SLF001 - observe external Win32 state
        "child_mode": int(child.stdout),
    }


@contextlib.contextmanager
def _hidden_desktop() -> Iterator[str]:
    """Keep even a regressed hard-error dialog off the interactive desktop."""
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.CreateDesktopW.argtypes = (
        ctypes.c_wchar_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
    )
    user32.CreateDesktopW.restype = ctypes.c_void_p
    user32.CloseDesktop.argtypes = (ctypes.c_void_p,)
    user32.CloseDesktop.restype = ctypes.c_int
    name = f"AskVLM-loader-test-{uuid.uuid4().hex}"
    handle = user32.CreateDesktopW(name, None, None, 0, 0x10000000, None)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield name
    finally:
        user32.CloseDesktop(handle)


class AllocationChild:
    """Own a tiny allocator on a hidden desktop, controlled through JSON pipes."""

    def __init__(self, desktop: str) -> None:
        executable, environment = child_interpreter()
        startup = subprocess.STARTUPINFO()
        startup.lpDesktop = desktop
        self.process = subprocess.Popen(  # noqa: S603 - fixed test module on an invisible desktop
            [executable, "-m", "tests.stt_pool_windows_support", "--mode", "allocate"],
            env={**os.environ, **environment},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            startupinfo=startup,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        self._messages: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        """Read pipe responses without unbounded readline waits in the controller."""
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self._messages.put(json.loads(line))
        self._messages.put(None)

    def response(self) -> dict[str, Any]:
        """Receive a child acknowledgement with a failure-only deadline."""
        message = self._messages.get(timeout=WAIT_S)
        if message is None:
            msg = f"allocation child exited unexpectedly: {self.process.poll()}"
            raise AssertionError(msg)
        return message

    def allocate(self, size: int) -> dict[str, Any]:
        """Request one bounded VirtualAlloc probe, freeing any successful allocation."""
        assert self.process.stdin is not None
        self.process.stdin.write((json.dumps({"bytes": size}) + "\n").encode())
        self.process.stdin.flush()
        return self.response()

    def close(self) -> None:
        """Reap this process and join its pipe reader on every outcome."""
        if self.process.stdin is not None:
            with contextlib.suppress(OSError):
                self.process.stdin.close()
        try:
            self.process.wait(timeout=WAIT_S)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=WAIT_S)
        self._reader.join(timeout=WAIT_S)
        assert not self._reader.is_alive()
        if self.process.stdout is not None:
            self.process.stdout.close()


@contextlib.contextmanager
def allocation_child() -> Iterator[AllocationChild]:
    """Protect the interactive desktop even if a child fails in its loader."""
    with _hidden_desktop() as desktop:
        child = AllocationChild(desktop)
        try:
            ready = child.response()
            assert ready["pid"] == child.process.pid
            assert ready["error_mode"] & SUPPRESSED_MODE == SUPPRESSED_MODE
            yield child
        finally:
            child.close()


class JobCapKernel:
    """Delegate real Win32 calls with optional small caps and observable port fences."""

    _FENCE_KEY = 0xA5A5

    def __init__(self, *, initial_cap: int | None = None) -> None:
        self.kernel = winjob._kernel32()  # noqa: SLF001 - external Win32 boundary
        self.initial_cap = initial_cap
        self.raise_floor_bytes = 0
        self.fail_update: str | None = None
        self.unknown_pid = False
        self.updates: list[dict[str, int]] = []
        self.raw_hits: list[int] = []
        self._initial: dict[int, int] = {}
        self._fence = threading.Event()

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - Win32 API delegation
        """Forward every unchanged API to the real kernel32 library."""
        return getattr(self.kernel, name)

    def query(self, handle: int) -> Any:  # noqa: ANN401 - ctypes ABI structure
        """Read the real cap, flags and peak without substituting their values."""
        info = winjob._JobExtendedLimitInformation()  # noqa: SLF001 - external ABI
        assert self.kernel.QueryInformationJobObject(
            handle, 9, ctypes.byref(info), ctypes.sizeof(info), None
        ), ctypes.get_last_error()
        return info

    def SetInformationJobObject(  # noqa: N802 - external Win32 symbol
        self, handle: int, info_class: int, pointer: object, size: int
    ) -> int:
        """Scale only capped jobs; preserve real growth and every other limit field."""
        if info_class != 9:
            return int(
                self.kernel.SetInformationJobObject(handle, info_class, pointer, size)
            )
        info = ctypes.cast(
            pointer,
            ctypes.POINTER(winjob._JobExtendedLimitInformation),  # noqa: SLF001 - external ABI
        ).contents
        requested = int(info.JobMemoryLimit)
        if not requested:
            return int(
                self.kernel.SetInformationJobObject(handle, info_class, pointer, size)
            )
        first = handle not in self._initial
        original = self._initial.setdefault(handle, requested)
        if self.initial_cap is not None:
            info.JobMemoryLimit = round(self.initial_cap * requested / original)
        if not first:
            info.JobMemoryLimit = max(info.JobMemoryLimit, self.raise_floor_bytes)
        # * Windows enforces a whole number of commit pages.
        info.JobMemoryLimit = info.JobMemoryLimit // mmap.PAGESIZE * mmap.PAGESIZE
        success = (
            0
            if not first and self.fail_update == "set"
            else int(
                self.kernel.SetInformationJobObject(handle, info_class, pointer, size)
            )
        )
        self.updates.append(
            {
                "handle": handle,
                "requested": requested,
                "applied": int(info.JobMemoryLimit),
                "flags": int(info.BasicLimitInformation.LimitFlags),
                "success": success,
            }
        )
        return success

    def QueryInformationJobObject(self, *args: Any) -> int:  # noqa: ANN401, N802 - external ABI
        """Allow a deterministic query failure only after a cap is installed."""
        if self.fail_update == "query":
            return 0
        return int(self.kernel.QueryInformationJobObject(*args))

    def PostQueuedCompletionStatus(self, *args: Any) -> int:  # noqa: ANN401, N802 - external ABI
        """Allow a deterministic re-arm failure without breaking stop packets."""
        if self.fail_update == "post" and args[2] == 3:
            return 0
        return int(self.kernel.PostQueuedCompletionStatus(*args))

    def GetQueuedCompletionStatus(  # noqa: N802 - external Win32 symbol
        self, port: int, message: object, key: object, overlapped: object, timeout: int
    ) -> int:
        """Observe real notifications and acknowledge a FIFO completion-port fence."""
        success = int(
            self.kernel.GetQueuedCompletionStatus(
                port, message, key, overlapped, timeout
            )
        )
        if not success:
            return success
        completion_key = ctypes.cast(
            key, ctypes.POINTER(ctypes.c_size_t)
        ).contents.value
        code = ctypes.cast(message, ctypes.POINTER(ctypes.c_uint32)).contents.value
        if completion_key == self._FENCE_KEY:
            self._fence.set()
        elif code == 10:
            process = ctypes.cast(overlapped, ctypes.POINTER(ctypes.c_void_p))
            self.raw_hits.append(int(process.contents.value or 0))
            if self.unknown_pid:
                process.contents.value = 0
        return success

    def fence(self, job: winjob.WorkerJob) -> None:
        """Await consumption of all earlier port packets without scheduling sleeps."""
        self._fence.clear()
        assert self.kernel.PostQueuedCompletionStatus(
            job._port,  # noqa: SLF001 - real external completion-port handle
            0,
            self._FENCE_KEY,
            None,
        )
        assert self._fence.wait(WAIT_S)


def _allocation_loop() -> None:
    """Probe job commit with VirtualAlloc, never touching or retaining the pages."""
    winjob.suppress_error_dialogs()
    kernel = winjob._kernel32()  # noqa: SLF001 - real Win32 allocation boundary
    kernel.VirtualAlloc.argtypes = (
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint,
        ctypes.c_uint,
    )
    kernel.VirtualAlloc.restype = ctypes.c_void_p
    kernel.VirtualFree.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint)
    kernel.VirtualFree.restype = ctypes.c_int
    print(  # noqa: T201 - child protocol
        json.dumps({"pid": os.getpid(), "error_mode": kernel.GetErrorMode()}),
        flush=True,
    )
    for line in sys.stdin:
        size = json.loads(line)["bytes"]
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or not 0 < size <= 256 * 1024**2
        ):
            msg = "allocation probes must be between one byte and 256 MiB"
            raise ValueError(msg)
        address = kernel.VirtualAlloc(None, size, 0x3000, 0x04)
        error = ctypes.get_last_error() if not address else 0
        if address:
            assert kernel.VirtualFree(address, 0, 0x8000), ctypes.get_last_error()
        print(  # noqa: T201 - child protocol
            json.dumps(
                {"pid": os.getpid(), "denied": not bool(address), "error": error}
            ),
            flush=True,
        )


def loader_failure_details() -> dict[str, object]:
    """Leave a suspended loader one page of commit without starving the test host."""
    details = error_mode_details()
    if cast("int", details["worker_mode"]) & SUPPRESSED_MODE != SUPPRESSED_MODE:
        msg = "worker error mode is unsafe for a loader-failure probe"
        raise AssertionError(msg)
    kernel = winjob._kernel32()  # noqa: SLF001 - real Win32 boundary
    kernel.OpenThread.argtypes = (ctypes.c_uint, ctypes.c_int, ctypes.c_uint)
    kernel.OpenThread.restype = ctypes.c_void_p
    kernel.ResumeThread.argtypes = (ctypes.c_void_p,)
    kernel.ResumeThread.restype = ctypes.c_uint
    executable, environment = child_interpreter()
    with _hidden_desktop() as desktop:
        startup = subprocess.STARTUPINFO()
        startup.lpDesktop = desktop
        child = subprocess.Popen(  # noqa: S603 - fixed module on an invisible test desktop
            [
                executable,
                "-m",
                "tests.stt_pool_windows_support",
                "--mode",
                "error-mode",
            ],
            env={**os.environ, **environment},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=startup,
            creationflags=subprocess.CREATE_NO_WINDOW | _CREATE_SUSPENDED,
        )
        cap_reports: queue.Queue[int] = queue.Queue()
        job = winjob.WorkerJob(
            memory_limit_bytes=128 * 1024**2, on_memory_limit=cap_reports.put
        )
        try:
            assert job.active, job.error
            assert job.assign(child.pid), job.assign_error
            # * Only loader-time allocations can spend the remaining commit page.
            handle = job._handle  # noqa: SLF001 - tighten only this isolated child's real job
            info = winjob._JobExtendedLimitInformation()  # noqa: SLF001 - external ABI
            assert kernel.QueryInformationJobObject(
                handle, 9, ctypes.byref(info), ctypes.sizeof(info), None
            )
            info.JobMemoryLimit = info.PeakJobMemoryUsed + 4096
            assert kernel.SetInformationJobObject(
                handle, 9, ctypes.byref(info), ctypes.sizeof(info)
            ), ctypes.get_last_error()
            threads = psutil.Process(child.pid).threads()
            assert len(threads) == 1
            thread = kernel.OpenThread(0x0002, 0, threads[0].id)
            assert thread
            try:
                assert kernel.ResumeThread(thread) == 1
            finally:
                kernel.CloseHandle(thread)
            stdout, _stderr = child.communicate(timeout=WAIT_S)
            details.update(
                loader_exit=child.returncode,
                loader_stdout=stdout.decode(),
                loader_cap_reported=cap_reports.get(timeout=WAIT_S) == child.pid,
            )
        finally:
            job.close()
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=WAIT_S)
    return details


def main() -> None:
    """Report a child's inherited mode or enter the production worker from zero."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("worker", "error-mode", "allocate"), required=True
    )
    args = parser.parse_args()
    kernel = winjob._kernel32()  # noqa: SLF001 - observe the mode before production main
    if args.mode == "worker":
        os.environ["ASKVLM_TEST_INITIAL_ERROR_MODE"] = str(kernel.GetErrorMode())
        from core.stt_pool.worker import (  # noqa: PLC0415 - observe before entry
            main as worker_main,
        )

        worker_main()
    elif args.mode == "allocate":
        _allocation_loop()
    else:
        print(json.dumps(int(kernel.GetErrorMode())), flush=True)  # noqa: T201 - child protocol


if __name__ == "__main__":
    main()
