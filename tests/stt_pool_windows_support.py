"""Win32 signals and bounded loader probes for isolated pool scenarios."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import subprocess
import uuid
from typing import TYPE_CHECKING, Self, cast

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
        job = winjob.WorkerJob(memory_limit_bytes=128 * 1024**2)
        try:
            assert job.active, job.error
            assert job.assign(child.pid), job.assign_error
            # * Assignment precedes the tiny cap, so only loader allocations fail.
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
            details.update(loader_exit=child.returncode, loader_stdout=stdout.decode())
        finally:
            job.close()
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=WAIT_S)
    return details


def main() -> None:
    """Report a child's inherited mode or enter the production worker from zero."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("worker", "error-mode"), required=True)
    args = parser.parse_args()
    kernel = winjob._kernel32()  # noqa: SLF001 - observe the mode before production main
    if args.mode == "worker":
        os.environ["ASKVLM_TEST_INITIAL_ERROR_MODE"] = str(kernel.GetErrorMode())
        from core.stt_pool.worker import (  # noqa: PLC0415 - observe before entry
            main as worker_main,
        )

        worker_main()
    else:
        print(json.dumps(int(kernel.GetErrorMode())), flush=True)  # noqa: T201 - child protocol


if __name__ == "__main__":
    main()
