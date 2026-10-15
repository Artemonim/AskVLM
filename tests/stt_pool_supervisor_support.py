"""Bounded supervisor scenarios with real workers and explicit fake hardware."""

from __future__ import annotations

import contextlib
import gc
import json
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import psutil  # type: ignore[import-untyped]

from core.stt_pool.config import parse_pool_config
from core.stt_pool.procs import default_worker_launch
from core.stt_pool.queue import (
    JobOptions,
    JobRecord,
    JobRequest,
    QueuePaths,
    SupervisorLock,
    read_heartbeat,
    read_result,
    submit_job,
)
from core.stt_pool.supervisor import SupervisorOptions, run_supervisor
from core.stt_pool.sysprobe import CommitInfo, GpuSnapshot
from core.temp_gc import TempGcSummary

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from core.stt_pool.config import PoolConfig
    from core.stt_pool.keys import DeviceRequest, Priority, WorkerKey
    from core.stt_pool.procs import WorkerLaunch
    from core.stt_pool.queue import JobKind, ResultRecord

WAIT_TIMEOUT_S = 30.0
GIB = 1024**3
ROOT = Path(__file__).resolve().parents[1]
_T = TypeVar("_T")


class Clock:
    """Logical scheduling time with unique reads and explicit TTL/deadline jumps."""

    def __init__(self) -> None:
        self._value = time.time()
        self._lock = threading.Lock()

    def __call__(self) -> float:
        """Advance minimally so refreshes and assignments have distinct timestamps."""
        with self._lock:
            self._value += 0.0001
            return self._value

    def peek(self) -> float:
        """Read without advancing scheduling time."""
        with self._lock:
            return self._value

    def advance(self, seconds: float) -> None:
        """Jump a policy deadline without waiting in real time."""
        with self._lock:
            self._value += seconds


class Hardware:
    """Fake commit/GPU boundary, including probe failure and controllable freshness."""

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.free_commit = 64 * GIB
        self.fail_commit = False
        self.commit_calls = 0
        self.gpu_calls = 0
        self.gpu_available = False
        self.block_gpu_after: int | None = None
        self.gpu_gate = threading.Event()
        self.gpu_gate.set()

    def commit(self) -> CommitInfo:
        """Supply commit metadata, pressure or probe failure."""
        self.commit_calls += 1
        if self.fail_commit:
            msg = "test commit probe failure"
            raise RuntimeError(msg)
        return CommitInfo(
            self.free_commit, 128 * GIB, 128 * GIB - self.free_commit, "fake"
        )

    def gpu(self) -> GpuSnapshot | None:
        """Hold a driver query or return a tiny metadata-only GPU snapshot."""
        self.gpu_calls += 1
        if (
            self.block_gpu_after is not None
            and self.gpu_calls > self.block_gpu_after
            and not self.gpu_gate.wait(WAIT_TIMEOUT_S)
        ):
            msg = "test did not release fake GPU query"
            raise TimeoutError(msg)
        if not self.gpu_available:
            return None
        return GpuSnapshot("test GPU", 0, 8 * GIB, self.clock.peek())


def configuration(**sections: dict[str, object]) -> PoolConfig:
    """Parse explicit policy limits without reading any machine configuration."""
    payload: dict[str, Any] = {
        "cpu": {"max_slots": 1, "threads_per_slot": 1, "max_slots_per_worker": 1},
        "gpu": {
            "max_workers": 0,
            "batch_lanes": {"whisper": 1, "gigaam": 1},
            "host_threads": {"whisper": 1, "gigaam": 1},
        },
        "threads": {"budget": 8},
        "memory": {
            "commit_budget_gib": 12,
            "free_commit_guard_gib": 0.25,
            "critical_free_commit_gib": 0.1,
        },
        "vram": {"budget_gib": 7, "free_guard_gib": 0.1},
        "lifecycle": {
            "worker_idle_ttl_s": 30,
            "supervisor_linger_s": 5,
            "stop_grace_s": 0.5,
            "cancel_grace_s": 5,
        },
        "scheduling": {
            "priority_aging_s": 300,
            "starvation_after_s": 60,
            "max_attempts": 2,
            "load_failure_cooldown_s": 40,
            "crash_cooldown_s": 30,
        },
    }
    for section, values in sections.items():
        payload.setdefault(section, {}).update(values)
    return parse_pool_config(payload, source="explicit supervisor test", logical_cpus=8)


class SupervisorHarness:
    """Own one supervisor, its queue and fake process identities."""

    def __init__(self, tmp_path: Path) -> None:
        self.paths = QueuePaths(tmp_path / "queue")
        self.paths.ensure()
        self.engine_root = tmp_path / "fake-engine"
        (self.engine_root / "events").mkdir(parents=True)
        (self.engine_root / "release").mkdir()
        self.clock = Clock()
        self.hardware = Hardware(self.clock)
        self.config = configuration()
        self.options = SupervisorOptions(
            engine_factory="tests.stt_pool_worker_engine:create_supervisor_engine",
            worker_launch=self.launch,
            commit_probe=self.hardware.commit,
            gpu_probe=self.hardware.gpu,
            clock=self.clock,
            tick_s=0.01,
            scan_interval_s=0.01,
            heartbeat_interval_s=0.01,
            gpu_probe_interval_s=0.05,
            load_timeout_s=300,
            purge_interval_s=1000,
            temp_gc=TempGcSummary,
            cuda_absent_grace_s=30,
        )
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.code: int | None = None
        self._sequence = 0
        self._call_indexes: dict[int, int] = {}
        self._processes: dict[int, psutil.Process] = {}
        self.plan()

    def launch(self, _key: WorkerKey, env: Mapping[str, str]) -> WorkerLaunch:
        """Launch the real worker with fake models and capped threads."""
        return default_worker_launch(
            project_root=ROOT,
            extra_env={**env, "ASKVLM_TEST_ENGINE_ROOT": str(self.engine_root)},
        )

    def plan(self, **instructions: object) -> None:
        """Write only the fake model loader's test instructions."""
        (self.engine_root / "plan.json").write_text(
            json.dumps(instructions), encoding="utf-8"
        )

    def job(  # noqa: PLR0913 - keyword-only options mirror the job wire fields
        self,
        *,
        model: str = "small",
        device: DeviceRequest = "cpu",
        provider: str = "whisper",
        call: int = 1,
        priority: Priority = "normal",
        duration: float | None = 1.0,
        deadline: float | None = None,
        max_parallel: int | None = None,
        kind: JobKind = "stt",
        job_options: JobOptions | None = None,
        **instructions: object,
    ) -> JobRecord:
        """Submit v2 jobs that control fake recognition."""
        self._sequence += 1
        index = self._call_indexes.get(call, 0)
        self._call_indexes[call] = index + 1
        job_id = f"{self._sequence:032x}"
        path = self.engine_root / f"media-{self._sequence}.json"
        path.write_text(json.dumps(instructions), encoding="utf-8")
        options = JobOptions() if job_options is None else job_options
        record = JobRecord(
            job_id,
            f"{call:032x}",
            index,
            max_parallel,
            self.clock.peek() + self._sequence * 0.00001,
            deadline,
            str(path),
            duration,
            JobRequest(provider, model, "auto", device),
            "ru" if kind == "stt" else None,
            os.getpid(),
            priority,
            kind=kind,
            want_segments=options.want_segments,
            decode=options.decode,
            window=options.window,
            diarize=options.diarize,
        )
        submit_job(self.paths, record)
        return record

    def start(self) -> None:
        """Run only the public supervisor entry, never a production daemon command."""
        if self.thread is not None and self.thread.is_alive():
            msg = "test already owns an active supervisor"
            raise RuntimeError(msg)
        self.stop_event = threading.Event()
        self.code = None

        def serve() -> None:
            self.code = run_supervisor(
                self.paths.root,
                self.config,
                options=self.options,
                stop_event=self.stop_event,
            )

        self.thread = threading.Thread(
            target=serve, name="test-stt-supervisor", daemon=True
        )
        self.thread.start()

    def until(self, read: Callable[[], _T], ready: Callable[[_T], bool]) -> _T:
        """Await a filesystem effect with a failure-only deadline."""
        deadline = time.monotonic() + WAIT_TIMEOUT_S
        wake = threading.Event()
        while time.monotonic() < deadline:
            value = read()
            if ready(value):
                return value
            if self.code is not None:
                msg = f"supervisor exited with {self.code} before the expected effect"
                raise AssertionError(msg)
            wake.wait(0.01)
        msg = f"supervisor effect timed out; heartbeat={read_heartbeat(self.paths)!r}"
        raise TimeoutError(msg)

    def events(self, kind: str, job: JobRecord | None = None) -> list[dict[str, Any]]:
        """Read model events and capture owned process identities."""
        records = []
        for path in sorted((self.engine_root / "events").glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = record.get("pid")
                if pid is not None and pid not in self._processes:
                    with contextlib.suppress(psutil.NoSuchProcess):
                        process = psutil.Process(pid)
                        # * A reused PID is not a process owned by this scenario.
                        if process.create_time() == record.get("created_at"):
                            self._processes[pid] = process
                if record["event"] == kind and (
                    job is None or record.get("job_id") == job.job_id
                ):
                    records.append(record)
        return sorted(records, key=lambda record: record["stamp"])

    def submitted(self, job: JobRecord, count: int = 1) -> dict[str, Any]:
        """Observe a dispatched attempt at the external fake recognition boundary."""
        return self.until(
            lambda: self.events("submitted", job), lambda records: len(records) >= count
        )[count - 1]

    def result(self, job: JobRecord) -> ResultRecord:
        """Read a persisted terminal result through the real queue API."""
        result = self.until(
            lambda: read_result(self.paths, job.job_id),
            lambda result: result is not None
            and not self.paths.incoming_file(job.job_id).exists(),
        )
        assert result is not None
        return result

    def heartbeat(self, ready: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
        """Observe reservations and lifecycle through the public heartbeat."""
        value = self.until(
            lambda: read_heartbeat(self.paths),
            lambda value: value is not None and ready(value),
        )
        assert value is not None
        return value

    def release(self, job: JobRecord) -> None:
        """Release a job while preserving distinct scheduling times."""
        self.clock.advance(0.01)
        (self.engine_root / "release" / job.job_id).touch()

    def finish(self, expected: int = 0, *, request_stop: bool = True) -> None:
        """Reap workers and verify lock, heartbeat and process cleanup."""
        if request_stop:
            self.stop_event.set()
        self.hardware.gpu_gate.set()
        (self.engine_root / "close-release").touch()
        if self.thread is not None:
            self.thread.join(timeout=WAIT_TIMEOUT_S)
            assert not self.thread.is_alive()
        assert self.code == expected
        assert not self.paths.heartbeat_file.exists()
        with SupervisorLock(self.paths) as lock:
            assert lock.try_acquire()
        self.assert_processes_stopped()
        gc.collect()

    def assert_processes_stopped(self) -> None:
        """Verify no process survives before fallback cleanup."""
        self.events("constructed")
        for process in self._processes.values():
            assert not process.is_running()

    def cleanup(self) -> None:
        """Reap only captured test process identities."""
        self.stop_event.set()
        self.hardware.gpu_gate.set()
        (self.engine_root / "close-release").touch()
        if self.thread is not None:
            self.thread.join(timeout=WAIT_TIMEOUT_S)
        self.events("constructed")
        for process in self._processes.values():
            if process.is_running():
                process.kill()
                process.wait(timeout=WAIT_TIMEOUT_S)
        gc.collect()
        assert self.thread is None or not self.thread.is_alive()
