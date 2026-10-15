"""Tiny protocol peer for pool lifecycle tests; no ML imports or production queue."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.abc
import json
import os
import pkgutil
import shutil
import subprocess
import sys
import threading
import wave
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from importlib.machinery import ModuleSpec

    from core.stt_pool.engine_api import EmitFn, PoolEngine

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_ALLOCATION_BYTES = 128 * 1024**2


class MLImportGuard(importlib.abc.MetaPathFinder):
    """Reject heavy dependencies in a fresh pool-import process."""

    def find_spec(
        self, fullname: str, _path: object = None, _target: object = None
    ) -> ModuleSpec | None:
        """Fail at the Python import boundary before any heavy allocation occurs."""
        if fullname.split(".", 1)[0] in {
            "torch",
            "transformers",
            "hydra",
            "faster_whisper",
            "ctranslate2",
            "numpy",
            "sherpa_onnx",
            "onnxruntime",
        }:
            msg = f"thin pool imports forbidden dependency {fullname}"
            raise RuntimeError(msg)
        return None


# * The fresh-process audit guards dependencies before any project import.
if "--import-pool" in sys.argv:
    sys.meta_path.insert(0, MLImportGuard())

from core.stt_pool.engine_api import (  # noqa: E402 - audit guard precedes imports
    EngineResult,
    LoadInfo,
    WorkerSpec,
    WorkItem,
    load_engine_factory,
)
from core.stt_pool.ipc import WorkerChannel  # noqa: E402 - audit guard
from core.stt_pool.queue import (  # noqa: E402 - audit guard precedes imports
    QueuePaths,
    SupervisorLock,
)
from core.stt_pool.sysprobe import child_interpreter  # noqa: E402 - audit guard


def _start_child(*, branch: bool) -> subprocess.Popen[bytes]:
    """Start another tiny peer with a directly addressable base interpreter."""
    executable, environment = child_interpreter()
    return subprocess.Popen(  # noqa: S603 - fixed test module and no shell
        [
            executable,
            "-X",
            "utf8",
            "-m",
            "tests.stt_pool_fake_worker",
            "--branch" if branch else "--hold",
        ],
        env={**os.environ, **environment},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE if branch else subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=_NO_WINDOW,
        close_fds=True,
    )


def _close_child(child: subprocess.Popen[bytes]) -> None:
    """Reap an owned fake child and close its pipes, including failure paths."""
    if child.stdin is not None:
        with contextlib.suppress(OSError):
            child.stdin.close()
    try:
        child.wait(timeout=2)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=5)
    if child.stdout is not None:
        child.stdout.close()


class FakeEngine:
    """External engine stand-in that exercises real protocol and process boundaries."""

    def __init__(self, spec: WorkerSpec) -> None:
        self._spec = spec
        self._emit: EmitFn | None = None
        self._child: subprocess.Popen[bytes] | None = None
        self._lock: SupervisorLock | None = None

    def load(self) -> LoadInfo:
        """Acquire optional real test resources after receiving a validated spec."""
        options = self._spec.options
        marker = options.get("marker")
        if isinstance(marker, str):
            Path(marker).write_text(str(os.getpid()), encoding="utf-8")
        details: dict[str, object] = {"pid": os.getpid()}
        if options.get("mode") == "tree":
            self._child = _start_child(branch=True)
            stdout = self._child.stdout
            if stdout is None:
                msg = "fake branch has no stdout"
                raise RuntimeError(msg)
            details.update(json.loads(stdout.readline()))
            details["child_pid"] = self._child.pid
        elif options.get("mode") == "lock":
            self._lock = SupervisorLock(QueuePaths(Path(str(options["root"]))))
            if not self._lock.try_acquire():
                msg = "fake peer cannot acquire test lock"
                raise RuntimeError(msg)
        return LoadInfo(self._spec.device, 0.0, details)

    def start(self, emit: EmitFn) -> None:
        """Install the result sink."""
        self._emit = emit

    def submit(self, item: WorkItem) -> None:
        """Emit a deterministic echo, or exercise a capped allocation below 512 MiB."""
        emit = self._emit
        if emit is None:
            msg = "fake engine is not started"
            raise RuntimeError(msg)
        try:
            if self._spec.options.get("mode") == "memory":
                allocation = bytearray(_ALLOCATION_BYTES)
                del allocation
            text = f"{item.language or 'auto'}:{item.input_path.name}"
            result = EngineResult.for_text(
                item,
                text,
                backend=self._spec.key.backend,
                run_s=0.0,
                audio_s=item.duration_s,
            )
        except MemoryError as exc:
            result = EngineResult.for_error(
                item, exc, backend=self._spec.key.backend, run_s=0.0
            )
        emit(result)

    def cancel(self, job_id: str, attempt_id: str) -> None:
        """Ignore cancellation of work that has already produced its terminal result."""

    def close(self) -> None:
        """Release only the fake engine's own resources."""
        if self._lock is not None:
            self._lock.release()
        if self._child is not None:
            _close_child(self._child)
            self._child = None


def create_engine(spec: WorkerSpec) -> PoolEngine:
    """Return the external engine substitute referenced by a trusted test spec."""
    return FakeEngine(spec)


def _run_peer() -> int:
    """Drive the real pool codec and API with the tiny external engine substitute."""
    channel = WorkerChannel.install()
    engine: PoolEngine | None = None
    try:
        incoming = channel.iter_stdin()
        first = next(incoming)
        spec = WorkerSpec.from_message(first)
        channel.send(
            {
                "type": "hello",
                "protocol": 2,
                "worker_id": spec.worker_id,
                "pid": os.getpid(),
            }
        )
        # * A valid-looking stray print must go to the log, not the protocol pipe.
        print(json.dumps({"type": "result", "job_id": "diagnostic-only"}), flush=True)  # noqa: T201 - intentional third-party stdout contamination
        engine = load_engine_factory(spec.engine_factory)(spec)
        fault = _protocol_fault()
        info = engine.load()
        engine.start(lambda result: channel.send(result.to_message()))
        channel.send(info.to_message())
        if fault == "late_loaded_exit":
            root = Path(os.environ["ASKVLM_TEST_ENGINE_ROOT"])
            (root / f"loaded-{spec.worker_id}").touch()
            if not _await_marker(root / "exit-release"):
                return 9
            engine.close()
            engine = None
            channel.send({"type": "bye"})
            return 0
        for message in incoming:
            if message["type"] == "shutdown":
                engine.close()
                engine = None
                channel.send({"type": "bye"})
                return 0
            if message["type"] == "job":
                engine.submit(WorkItem.from_message(message))
                if fault == "duplicate_loaded":
                    channel.send({**info.to_message(), "load_s": -1.0})
                    fault = ""
        return 3
    finally:
        if engine is not None:
            engine.close()
        channel.close()


def _await_marker(path: Path) -> bool:
    """Wait for only a test-owned filesystem gate with a failure-only deadline."""
    for _ in range(3000):
        if path.exists():
            return True
        threading.Event().wait(0.01)
    return False


def _protocol_fault() -> str:
    """Choose one faulty protocol peer per temporary root."""
    mode = os.environ.get("ASKVLM_TEST_PROTOCOL", "")
    if not mode:
        return ""
    root = Path(os.environ["ASKVLM_TEST_ENGINE_ROOT"])
    try:
        with (root / "fault-claimed").open("x", encoding="utf-8") as marker:
            marker.write(mode)
    except FileExistsError:
        return ""
    return mode


def main() -> int:
    """Select a tiny protocol worker, held child, branching child or immediate exit."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--hold", action="store_true", help="Wait for parent pipe EOF.")
    modes.add_argument(
        "--branch", action="store_true", help="Hold one more child and report its PID."
    )
    modes.add_argument(
        "--exit", action="store_true", help="Exit immediately with code zero."
    )
    modes.add_argument(
        "--media-args",
        help="Handle a canonical-WAV ffmpeg/ffprobe command vector encoded as JSON.",
    )
    modes.add_argument("--smi", help="Print one stand-in nvidia-smi CSV record.")
    modes.add_argument(
        "--import-pool",
        action="store_true",
        help="Audit pool imports while denying ML dependencies.",
    )
    args = parser.parse_args()
    if args.import_pool:
        pool = importlib.import_module("core.stt_pool")
        modules = list(pkgutil.iter_modules(pool.__path__, prefix="core.stt_pool."))
        for module in modules:
            importlib.import_module(module.name)
        importlib.import_module("core.whisper_batch")
        for name in (
            "core.stt_options",
            "core.stt_output",
            "core.diarization_models",
            "core.sherpa_diarization",
        ):
            importlib.import_module(name)
        sys.stdout.write(json.dumps({"imported": len(modules)}) + "\n")
        sys.stdout.flush()
        return 0
    if args.smi is not None:
        sys.stdout.write(args.smi + "\n")
        sys.stdout.flush()
        return 0
    if args.media_args is not None:
        command = json.loads(args.media_args)
        if Path(command[0]).stem == "ffprobe":
            with wave.open(command[-1], "rb") as reader:
                duration = reader.getnframes() / reader.getframerate()
            sys.stdout.write(json.dumps({"format": {"duration": str(duration)}}) + "\n")
            sys.stdout.flush()
        else:
            source = command[command.index("-i") + 1]
            destination = [part for part in command if part.endswith(".wav")][-1]
            shutil.copyfile(source, destination)
        return 0
    if args.exit or args.hold:
        if args.hold:
            sys.stdin.buffer.read(1)
        return 0
    if args.branch:
        child = _start_child(branch=False)
        try:
            sys.stdout.write(json.dumps({"grandchild_pid": child.pid}) + "\n")
            sys.stdout.flush()
            sys.stdin.buffer.read(1)
        finally:
            _close_child(child)
        return 0
    return _run_peer()


if __name__ == "__main__":
    raise SystemExit(main())
