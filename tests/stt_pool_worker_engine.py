"""Tiny injected engine for scenarios through the production worker entry."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import threading
import time
import types
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import psutil  # type: ignore[import-untyped]

from core.stt_errors import CudaOutOfMemoryError, DeviceUnavailableError
from core.stt_pool import ipc
from core.stt_pool.engine_api import EngineResult, LoadInfo
from core.stt_pool.sysprobe import child_interpreter
from tests.stt_pool_windows_support import allocation_child, signal, wait_for_signal

if TYPE_CHECKING:
    from core.stt_pool.engine_api import EmitFn, PoolEngine, WorkerSpec, WorkItem


def install_memory_probe(mode: object, marker: object) -> None:
    """Install a tiny torch boundary that records forbidden CUDA initialization."""
    if mode is None:
        return
    if mode == "unimported":
        guard = importlib.import_module("tests.stt_pool_fake_worker").MLImportGuard
        sys.meta_path.insert(0, guard())
        return

    def cuda_access(*_args: object) -> int:
        if mode != "initialized":
            Path(str(marker)).write_text("CUDA accessed", encoding="utf-8")
            msg = "telemetry touches an uninitialized CUDA runtime"
            raise AssertionError(msg)
        return 1

    def figure(device: int, value: int) -> int:
        if device != cuda_access():
            msg = "telemetry reads a different CUDA device"
            raise AssertionError(msg)
        return value

    module = types.ModuleType("torch")
    module.cuda = types.SimpleNamespace(
        is_initialized=lambda: mode == "initialized",
        init=cuda_access,
        _lazy_init=cuda_access,
        is_available=cuda_access,
        current_device=cuda_access,
        memory_reserved=lambda device: figure(device, 8192),
        memory_allocated=lambda device: figure(device, 1024),
        max_memory_reserved=lambda device: figure(device, 16384),
    )
    module.backends = types.SimpleNamespace(
        cuda=types.SimpleNamespace(cufft_plan_cache={1: types.SimpleNamespace(size=7)})
    )
    sys.modules["torch"] = module


class _MemoryPacket:
    """Emit deliberately malformed telemetry at the trusted worker protocol boundary."""

    def __init__(self, result: EngineResult, memory: object) -> None:
        self._result = result
        self._memory = memory
        self.job_id = result.job_id
        self.attempt_id = result.attempt_id

    def to_message(self) -> dict[str, object]:
        """Keep the outcome valid while corrupting only its optional memory field."""
        message = self._result.to_message()
        message["memory"] = self._memory
        return message


class ScenarioEngine:
    """Expose deterministic lifecycle failures without importing any ML stack."""

    def __init__(self, spec: WorkerSpec) -> None:
        self._spec = spec
        self._emit: EmitFn | None = None
        self._held: dict[tuple[str, str], WorkItem] = {}
        self._threads: list[threading.Thread] = []

    def load(self) -> LoadInfo:
        """Report the configured device or a typed load failure."""
        options = self._spec.options
        install_memory_probe(options.get("memory_probe"), options.get("cuda_marker"))
        error = options.get("load_error")
        if error == "cuda_oom":
            msg = "CUDA out of memory while loading fake weights"
            raise CudaOutOfMemoryError(msg)
        if error == "device_unavailable":
            msg = "fake device is unavailable"
            raise DeviceUnavailableError(msg)
        if error == "memory_limit":
            msg = "fake capped allocation failed"
            raise MemoryError(msg)
        if error:
            msg = "fake load failed"
            raise RuntimeError(msg)
        if options.get("mode") in {"error_mode", "loader_failure"}:
            from tests.stt_pool_windows_support import (  # noqa: PLC0415 - Windows-only test boundary
                error_mode_details,
                loader_failure_details,
            )

            probe = (
                loader_failure_details
                if options["mode"] == "loader_failure"
                else error_mode_details
            )
            return LoadInfo(self._spec.device, 0.0, probe())
        return LoadInfo(str(options.get("loaded_device", self._spec.device)), 0.0, {})

    def start(self, emit: EmitFn) -> None:
        """Install emission before allowing submitted jobs; optionally fail startup."""
        self._emit = emit
        if self._spec.options.get("start_error"):
            msg = "fake engine cannot start"
            raise RuntimeError(msg)

    def submit(self, item: WorkItem) -> None:
        """Echo the item, retain it for cancel, or exercise worker error reporting."""
        emit = self._emit
        if emit is None:
            msg = "job accepted before start"
            raise RuntimeError(msg)
        mode = self._spec.options.get("mode")
        if mode == "submit_error":
            msg = "fake submit failed"
            raise RuntimeError(msg)
        if mode == "hold":
            self._held[(item.job_id, item.attempt_id)] = item
            return
        # * Both Python prints and native fd writes must stay in the worker log.
        print('{"type":"result","job_id":"stray-python"}', flush=True)  # noqa: T201 - deliberate external stdout contamination
        os.write(1, b'{"type":"result","job_id":"stray-native"}\n')
        text = (
            ""
            if mode == "empty"
            else f"{item.language or 'auto'}:{item.input_path.name}"
        )
        result = EngineResult.for_text(
            item,
            text,
            backend=self._spec.key.backend,
            run_s=0.0,
            audio_s=item.duration_s,
        )
        if mode == "oversized_result":
            char = str(self._spec.options.get("payload_char", "x"))
            count = ipc.MAX_LINE_BYTES // len(char.encode("utf-8")) + 1
            result = replace(result, data={"blob": char * count})
        elif mode == "bad_result":
            result = replace(result, text=object())  # type: ignore[arg-type]
        elif mode == "error":
            result = EngineResult.for_error(
                item,
                RuntimeError("Too long wav file: transcribe_longform"),
                backend=self._spec.key.backend,
                run_s=0.0,
            )
        if mode == "parallel":
            thread = threading.Thread(target=emit, args=(result,), daemon=True)
            self._threads.append(thread)
            thread.start()
        else:
            emit(result)

    def cancel(self, job_id: str, attempt_id: str) -> None:
        """Answer a retained attempt once; repeat and late cancels have no effect."""
        item = self._held.pop((job_id, attempt_id), None)
        if item is not None and self._emit is not None:
            self._emit(
                EngineResult.for_cancel(item, backend=self._spec.key.backend, run_s=0.0)
            )

    def close(self) -> None:
        """Expose normal, failing or stuck engine close."""
        marker = self._spec.options.get("close_marker")
        if isinstance(marker, str):
            Path(marker).write_text("closed", encoding="utf-8")
        if self._spec.options.get("close_hangs"):
            threading.Event().wait()
        if self._spec.options.get("close_error"):
            msg = "fake close failed"
            raise RuntimeError(msg)
        for thread in self._threads:
            thread.join(timeout=2)
        self._held.clear()


def create_engine(spec: WorkerSpec) -> PoolEngine:
    """Build the fake ML boundary requested by a trusted worker test spec."""
    return ScenarioEngine(spec)


class SupervisorEngine:
    """Fake ML boundary with observable events and file-controlled job completion."""

    def __init__(self, spec: WorkerSpec) -> None:
        self._spec = spec
        self._root = Path(os.environ["ASKVLM_TEST_ENGINE_ROOT"])
        self._journal = self._root / "events" / f"{spec.worker_id}.jsonl"
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._emit: EmitFn | None = None
        self._threads: list[threading.Thread] = []
        self._cancels: dict[tuple[str, str], tuple[threading.Event, bool]] = {}
        self._cancel_signals: dict[tuple[str, str], str] = {}
        self._child: subprocess.Popen[bytes] | None = None
        self._close_gate = False
        self._record(
            "constructed",
            pid=os.getpid(),
            created_at=psutil.Process().create_time(),
            capacity=spec.capacity,
            key=spec.key.to_dict(),
            threads_per_slot=spec.threads_per_slot,
            options=dict(spec.options),
            cudnn_v8_lru_cache_limit=os.environ.get(
                "TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT"
            ),
            thread_env={
                name: os.environ.get(name)
                for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
            },
        )

    def _record(self, event: str, **fields: object) -> None:
        """Append one complete observable boundary event and release the file handle."""
        record = {
            "event": event,
            "worker_id": self._spec.worker_id,
            "stamp": time.monotonic_ns(),
            **fields,
        }
        with self._lock, self._journal.open("a", encoding="utf-8") as journal:
            journal.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load(self) -> LoadInfo:
        """Expose load failure, timeout or malformed metadata."""
        plan = json.loads((self._root / "plan.json").read_text(encoding="utf-8"))
        install_memory_probe(plan.get("memory_probe"), plan.get("cuda_marker"))
        self._close_gate = bool(plan.get("close_gate"))
        mode = plan.get("load_modes", {}).get(self._spec.key.model)
        self._record("loading", mode=mode)
        if mode == "cap_report":
            self._cap_gate(plan)
        if mode == "hang":
            threading.Event().wait()
        if mode == "gate":
            while not (self._root / "load-release").exists():
                threading.Event().wait(0.01)
        if mode == "fail":
            msg = "Too long wav file: transcribe_longform"
            raise RuntimeError(msg)
        if mode == "cuda_oom":
            msg = "CUDA out of memory while loading fake weights"
            raise CudaOutOfMemoryError(msg)
        if plan.get("spawn_child"):
            executable, environment = child_interpreter()
            self._child = subprocess.Popen(  # noqa: S603 - fixed tiny fake Python module, no shell
                [executable, "-m", "tests.stt_pool_fake_worker", "--hold"],
                env={**os.environ, **environment},
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                close_fds=True,
            )
            self._record(
                "child",
                pid=self._child.pid,
                created_at=psutil.Process(self._child.pid).create_time(),
            )
        return LoadInfo(self._spec.device, -1.0 if mode == "bad_loaded" else 0.0, {})

    def start(self, emit: EmitFn) -> None:
        """Install the production worker's protected emitter."""
        self._emit = emit
        self._record("started")

    def submit(self, item: WorkItem) -> None:
        """Start fake recognition without blocking dispatch."""
        instructions = json.loads(item.input_path.read_text(encoding="utf-8"))
        instructions = instructions.get(self._spec.key.backend, instructions)
        instructions = instructions.get(item.kind, instructions)
        cancel = threading.Event()
        self._cancels[item.job_id, item.attempt_id] = (
            cancel,
            bool(instructions.get("ignore_cancel")),
        )
        cancel_signal = instructions.get("cancel_signal")
        if isinstance(cancel_signal, str):
            self._cancel_signals[item.job_id, item.attempt_id] = cancel_signal
        self._record(
            "submitted",
            job_id=item.job_id,
            attempt_id=item.attempt_id,
            backend=self._spec.key.backend,
            input_path=str(item.input_path),
            language=item.language,
            priority=item.priority,
            call_id=item.call_id,
            seq=item.seq,
            kind=item.kind,
            want_segments=item.want_segments,
            decode=item.decode.to_wire(),
            window=item.window.to_wire(),
            diarize=None if item.diarize is None else item.diarize.to_wire(),
        )
        thread = threading.Thread(
            target=self._recognize, args=(item, instructions, cancel), daemon=True
        )
        self._threads.append(thread)
        thread.start()

    def _recognize(
        self, item: WorkItem, instructions: dict[str, object], cancel: threading.Event
    ) -> None:
        """Recognize at the fake external model boundary."""
        mode = self._attempt_mode(item, instructions)
        self._chatter(instructions)
        if mode == "hold" or instructions.get("wait"):
            release = self._root / "release" / item.job_id
            while not release.exists() and not cancel.is_set():
                if self._closed.wait(0.01):
                    return
        if self._closed.is_set():
            return
        if mode == "crash":
            sys.stderr.write(f"fake recognition crash: {item.input_path.name} code=7\n")
            sys.stderr.flush()
            os._exit(7)
        if mode == "allocation_crash":
            self._allocation_crash(item)
        result = self._result(item, instructions, mode, cancelled=cancel.is_set())
        emit = self._emit
        if emit is not None:
            self._record(
                "emitted",
                job_id=item.job_id,
                attempt_id=item.attempt_id,
                status=result.status,
                error_detail=result.error_detail,
            )
            if "wire_memory" in instructions:
                emit(_MemoryPacket(result, instructions["wire_memory"]))  # type: ignore[arg-type]
            else:
                emit(result)
            if mode == "duplicate":
                emit(result)

    def _attempt_mode(self, item: WorkItem, instructions: dict[str, object]) -> object:
        """Fail only the configured attempts and gate real cap reports by events."""
        attempt = int(item.attempt_id.rsplit(":", 1)[1])
        if attempt > int(instructions.get("fail_attempts", 1000000)):
            return "ok"
        mode = instructions.get("mode", "ok")
        if mode == "child_cap_report":
            wait_for_signal(instructions.get("before_cap_gate"))
            with allocation_child() as child:
                self._record(
                    "child",
                    pid=child.process.pid,
                    created_at=psutil.Process(child.process.pid).create_time(),
                )
                denied = child.allocate(256 * 1024**2)
                self._record("child_allocation", **denied)
                if not denied["denied"]:
                    msg = "test Job Object did not reject the child's allocation"
                    raise AssertionError(msg)
                signal(instructions.get("cap_signal"))
                wait_for_signal(instructions.get("result_gate"))
            return instructions.get("after_cap", "error")
        if mode == "cap_report":
            SupervisorEngine._cap_gate(instructions)
            return instructions.get("after_cap", "error")
        return mode

    @staticmethod
    def _cap_gate(instructions: dict[str, object]) -> None:
        """Report a real denied allocation before an event releases the outcome."""
        SupervisorEngine._capped_allocation_error()
        signal(instructions.get("cap_signal"))
        wait_for_signal(instructions.get("result_gate"))

    @staticmethod
    def _chatter(instructions: dict[str, object]) -> None:
        """Expose both Python and native stdout contamination at the model boundary."""
        if instructions.get("stray_stdout"):
            print("model chatter", flush=True)  # noqa: T201 - intentional external stdout contamination
            os.write(1, b"native model chatter\n")

    @staticmethod
    def _allocation_crash(item: WorkItem) -> None:
        """Crash after a real allocation rejected by the tiny test Job cap."""
        try:
            allocation = bytearray(96 * 1024**2)
            del allocation
        except MemoryError:
            sys.stderr.write(f"fake capped allocation: {item.input_path.name}\n")
            sys.stderr.flush()
        os._exit(8)

    def _result(
        self,
        item: WorkItem,
        instructions: dict[str, object],
        mode: object,
        *,
        cancelled: bool,
    ) -> EngineResult:
        """Build an external model outcome without changing supervisor bookkeeping."""
        if cancelled or mode == "unsolicited_cancel":
            result = EngineResult.for_cancel(
                item, backend=self._spec.key.backend, run_s=0.0
            )
        elif mode in {
            "cuda_oom",
            "memory_limit",
            "allocation_error",
            "error",
            "device_unavailable",
            "protocol_error",
            "result_too_large",
        }:
            result = self._error_result(item, instructions, mode)
        else:
            data = instructions.get("data")
            if data is not None and not isinstance(data, dict):
                msg = "fake result data must be an object"
                raise ValueError(msg)
            if instructions.get("large_segments"):
                data = {"segments": [{"text": "x" * (17 * 1024**2)}]}
            if item.kind == "diarize" and data is not None:
                result = EngineResult.for_diarization(
                    item,
                    data,
                    backend=self._spec.key.backend,
                    run_s=0.0,
                    audio_s=item.duration_s,
                )
            else:
                result = EngineResult.for_text(
                    item,
                    ""
                    if mode == "empty"
                    else str(instructions.get("text", item.input_path.stem)),
                    backend=self._spec.key.backend,
                    run_s=0.0,
                    audio_s=item.duration_s,
                    data=data,
                )
            if mode == "blank_ok":
                result = replace(result, status="ok", text=" \n\t ")
        if "memory" in instructions:
            result = replace(result, memory=instructions["memory"])  # type: ignore[arg-type]
        return result

    def _error_result(
        self, item: WorkItem, instructions: dict[str, object], mode: object
    ) -> EngineResult:
        """Expose typed external failures without supervisor logic in the fake."""
        error: Exception
        if mode == "cuda_oom":
            error = CudaOutOfMemoryError("CUDA out of memory in fake recognition")
        elif mode == "memory_limit":
            error = MemoryError("fake capped allocation failed")
        elif mode == "allocation_error":
            error = self._capped_allocation_error()
        elif mode == "device_unavailable":
            error = DeviceUnavailableError("fake device is unavailable")
        else:
            error = RuntimeError(
                str(
                    instructions.get(
                        "error_text", "Too long wav file: transcribe_longform"
                    )
                )
            )
        result = EngineResult.for_error(
            item, error, backend=self._spec.key.backend, run_s=0.0
        )
        if mode in {"protocol_error", "result_too_large"}:
            result = replace(result, error_kind=str(mode))
        return result

    @staticmethod
    def _capped_allocation_error() -> MemoryError:
        """Return a real small Job-cap allocation failure without killing the worker."""
        try:
            allocation = bytearray(96 * 1024**2)
            del allocation
        except MemoryError as exc:
            return exc
        msg = "test Job Object did not reject its small fake allocation"
        raise AssertionError(msg)

    def cancel(self, job_id: str, attempt_id: str) -> None:
        """Deliver cancellation or keep a deliberately uncooperative job running."""
        self._record("cancel", job_id=job_id, attempt_id=attempt_id)
        signal(self._cancel_signals.get((job_id, attempt_id)))
        entry = self._cancels.get((job_id, attempt_id))
        if entry is not None and not entry[1]:
            entry[0].set()

    def close(self) -> None:
        """Stop all owned fake recognition threads and the optional tiny child."""
        self._record("closed")
        self._closed.set()
        while self._close_gate and not (self._root / "close-release").exists():
            threading.Event().wait(0.01)
        for thread in self._threads:
            thread.join(timeout=30)
        child = self._child
        if child is not None:
            if child.stdin is not None:
                child.stdin.close()
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)


def create_supervisor_engine(spec: WorkerSpec) -> PoolEngine:
    """Build the fake engine for a supervisor with an explicitly injected test root."""
    return SupervisorEngine(spec)
