"""Bounded supervisor scenarios with real workers and explicit fake hardware."""

from __future__ import annotations

import contextlib
import gc
import io
import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, TypeVar

import psutil  # type: ignore[import-untyped]

from core.stt_pool import linux_cgroup, procs, supervisor
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

    import pytest

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


@dataclass
class CgroupNode:
    """Kernel control files and identity of one simulated cgroup directory."""

    path: PurePosixPath
    inode: int
    files: dict[str, str] = field(
        default_factory=lambda: {
            "cgroup.procs": "",
            "cgroup.controllers": "memory pids\n",
            "cgroup.subtree_control": "",
            "memory.max": "max\n",
            "memory.events": "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\n",
        }
    )
    event_reads: int = 0


class CgroupKernel:
    """Simulate only the Linux descriptor/control-file boundary, on any host.

    Directory handles retain identity across replacement. Enrollment moves
    one PID and enabling memory enforces the no-internal-process constraint.
    No method accesses a real cgroup or a real process.
    """

    O_RDONLY = 0
    O_WRONLY = 1
    O_DIRECTORY = 0x10000
    O_NOFOLLOW = 0x20000
    O_CLOEXEC = 0x80000

    def __init__(self) -> None:
        self.root = PurePosixPath("/sys/fs/cgroup")
        self.service_path = self.root / "system.slice" / "askvlm-test.service"
        self.nodes: dict[PurePosixPath, CgroupNode] = {}
        self.handles: dict[int, tuple[CgroupNode, str | None]] = {}
        self.symlinks: set[PurePosixPath] = set()
        self.moves: list[tuple[int, PurePosixPath]] = []
        self.fail_events_after: int | None = None
        self.refuse_control: str | None = None
        self.initial_events = CgroupNode(self.service_path, 0).files["memory.events"]
        self.busy_reads = 0
        self._next_fd = 100
        self._next_inode = 0
        for path in reversed([self.service_path, *self.service_path.parents]):
            self.replace(path)
        self.service.files["cgroup.procs"] = f"{self.getpid()}\n"

    @property
    def service(self) -> CgroupNode:
        """Return the service root, including any externally changed membership."""
        return self.nodes[self.service_path]

    @staticmethod
    def getpid() -> int:
        """Use a synthetic supervisor identity, never a host PID."""
        return 42

    def replace(self, path: PurePosixPath) -> CgroupNode:
        """Model a directory replacement without changing already-open handles."""
        self._next_inode += 1
        node = CgroupNode(path, self._next_inode)
        self.nodes[path] = node
        return node

    def _path(self, path: object, dir_fd: int | None) -> PurePosixPath:
        resolved = PurePosixPath(str(path))
        if dir_fd is not None:
            resolved = self.handles[dir_fd][0].path / resolved
        return resolved

    def open(self, path: object, flags: int, *, dir_fd: int | None = None) -> int:
        """Open a virtual directory or existing control file, honoring no-follow."""
        target = self._path(path, dir_fd)
        name: str | None
        if target in self.symlinks and flags & self.O_NOFOLLOW:
            raise OSError(40, "symlink refused")
        if flags & self.O_DIRECTORY:
            node, name = self.nodes[target], None
        else:
            node, name = self.nodes[target.parent], target.name
            if name not in node.files:
                raise FileNotFoundError(str(target))
            if name == "memory.events" and not flags & self.O_WRONLY:
                node.event_reads += 1
                if (
                    self.fail_events_after is not None
                    and node.event_reads > self.fail_events_after
                ):
                    raise PermissionError(str(target))
        self._next_fd += 1
        self.handles[self._next_fd] = (node, name)
        return self._next_fd

    def fdopen(self, fd: int, *, encoding: str) -> io.StringIO:
        """Read a control snapshot and close its virtual descriptor with the stream."""
        node, name = self.handles.pop(fd)
        assert encoding == "ascii"
        assert name is not None
        if node is self.service and name == "cgroup.procs":
            self.busy_reads += 1
        return io.StringIO(node.files[name])

    def close(self, fd: int) -> None:
        """Release a descriptor without changing the group's lifetime."""
        del self.handles[fd]

    def mkdir(self, name: str, *, mode: int, dir_fd: int) -> None:
        """Create a fresh group with the kernel's control files already present."""
        assert mode == 0o700
        path = self._path(name, dir_fd)
        if path in self.nodes:
            raise FileExistsError(str(path))
        node = self.replace(path)
        node.files["memory.events"] = self.initial_events

    def write(self, fd: int, data: bytes) -> int:
        """Apply a control write or a deterministic permission failure."""
        node, name = self.handles[fd]
        assert name is not None
        if name == self.refuse_control:
            raise PermissionError(str(node.path / name))
        value = data.decode("ascii")
        if name == "cgroup.procs":
            pid = int(value)
            for group in self.nodes.values():
                members = group.files["cgroup.procs"].split()
                group.files["cgroup.procs"] = "\n".join(
                    member for member in members if member != str(pid)
                )
            self.moves.append((pid, node.path))
        if name == "cgroup.subtree_control":
            if node.files["cgroup.procs"].strip():
                raise OSError(16, "populated root")
            value = value.replace("+", "")
        node.files[name] = value
        return len(data)

    def fstat(self, fd: int) -> SimpleNamespace:
        """Return the identity captured by an open directory descriptor."""
        return SimpleNamespace(st_dev=1, st_ino=self.handles[fd][0].inode)

    def stat(self, name: str, *, dir_fd: int, follow_symlinks: bool) -> SimpleNamespace:
        """Observe the current directory identity without following links."""
        assert not follow_symlinks
        return SimpleNamespace(
            st_dev=1, st_ino=self.nodes[self._path(name, dir_fd)].inode
        )

    def rmdir(self, name: str, *, dir_fd: int) -> None:
        """Remove an empty leaf only, as cgroupfs does despite virtual files."""
        path = self._path(name, dir_fd)
        node = self.nodes[path]
        if node.files["cgroup.procs"].strip() or any(
            other.parent == path for other in self.nodes
        ):
            raise OSError(16, "group is populated")
        del self.nodes[path]

    def member(self, pid: int) -> CgroupNode | None:
        """Find a simulated process's current group."""
        return next(
            (
                node
                for node in self.nodes.values()
                if str(pid) in node.files["cgroup.procs"].split()
            ),
            None,
        )

    def departed(self, pid: int) -> None:
        """Model process exit without moving it to another group."""
        node = self.member(pid)
        if node is not None:
            node.files["cgroup.procs"] = "\n".join(
                member
                for member in node.files["cgroup.procs"].split()
                if member != str(pid)
            )


class _ScriptedOutput:
    """A blocking byte pipe with explicit peer messages and EOF."""

    def __init__(self) -> None:
        self.lines: queue.Queue[bytes] = queue.Queue()

    def readline(self, _size: int = -1) -> bytes:
        """Synchronize on protocol effects rather than a scheduling sleep."""
        return self.lines.get(timeout=WAIT_TIMEOUT_S)

    def close(self) -> None:
        """Satisfy the stream owner without touching an OS handle."""


class _ScriptedInput:
    """Receive the gate and JSON-lines at the external stdin boundary."""

    def __init__(self, process: ScriptedPoolProcess) -> None:
        self.process = process

    def write(self, data: bytes) -> int:
        """Observe membership at execution and feed protocol responses."""
        process = self.process
        if data == b"\0":
            process.executed_group = process.boundary.kernel.member(process.pid)
            if process.executed_group is not None:
                process.execution_cap = process.executed_group.files["memory.max"]
            process.executed = True
        else:
            message = json.loads(data)
            kind = message["type"]
            if kind == "spec":
                process.spec_before_execution = not process.executed
                process.spec = message
                process.emit({"type": "hello", "protocol": 2, "pid": process.pid})
                process.loaded()
            elif kind == "job":
                process.boundary.jobs.put((process, message))
            elif kind == "cancel":
                process.boundary.cancels.put((process, message))
            elif kind == "shutdown":
                process.emit({"type": "bye"})
                process.finish(process.boundary.shutdown_code)
        return len(data)

    def flush(self) -> None:
        """Make writes immediately visible to the pipe boundary."""

    def close(self) -> None:
        """Satisfy the stdin owner after shutdown or a failed enrollment."""


class ScriptedPoolProcess:
    """A subprocess boundary with explicit exit codes, never a host process."""

    def __init__(self, boundary: LinuxPoolBoundary, argv: list[str]) -> None:
        self.boundary = boundary
        self.args = argv
        self.pid = 1000000 + len(boundary.processes)
        self.stdout = _ScriptedOutput()
        self.stdin = _ScriptedInput(self)
        self.spec: dict[str, Any] | None = None
        self.returncode: int | None = None
        self.exited = threading.Event()
        self.executed = "core.stt_pool.linux_cgroup" not in argv
        self.executed_group: CgroupNode | None = None
        self.execution_cap: str | None = None
        self.spec_before_execution = False

    def emit(self, message: dict[str, object]) -> None:
        """Send a complete worker JSON-line to the real protocol reader."""
        self.stdout.lines.put((json.dumps(message) + "\n").encode("utf-8"))

    def loaded(self) -> None:
        """Publish readiness after the parent's gate and spec."""
        self.emit({"type": "loaded", "device": "cpu", "load_s": 0, "details": {}})

    def reply(self, job: dict[str, Any], error: str | None = None) -> None:
        """Report a result at the protocol boundary, preserving its attempt identity."""
        self.emit(
            {
                "type": "result",
                "job_id": job["job_id"],
                "attempt_id": job["attempt_id"],
                "status": "error" if error else "ok",
                "text": "" if error else "recognized",
                "error_kind": error,
                "error_detail": "decoder failed" if error else None,
                "backend": "cpu",
                "run_s": 0.01,
                "audio_s": job["duration_s"],
            }
        )

    def finish(self, code: int) -> None:
        """Exit only the simulated PID and close its stdout exactly once."""
        if self.returncode is None:
            self.returncode = code
            self.boundary.kernel.departed(self.pid)
            self.exited.set()
            self.stdout.lines.put(b"")

    def poll(self) -> int | None:
        """Return the externally controlled exit status."""
        return self.returncode

    def wait(self, *, timeout: float) -> int:
        """Bound hangs without simulating elapsed policy time."""
        if not self.exited.wait(timeout):
            raise subprocess.TimeoutExpired(self.args, timeout)
        assert self.returncode is not None
        return self.returncode

    def kill(self) -> None:
        """Simulate SIGKILL without sending any signal."""
        self.finish(-9)


class LinuxPoolBoundary:
    """Keep production supervisor, scheduler, process handle and adapter real."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self.kernel = CgroupKernel()
        self.processes: list[ScriptedPoolProcess] = []
        self.jobs: queue.Queue[tuple[ScriptedPoolProcess, dict[str, Any]]] = (
            queue.Queue()
        )
        self.cancels: queue.Queue[tuple[ScriptedPoolProcess, dict[str, Any]]] = (
            queue.Queue()
        )
        self.shutdown_code = 0
        membership = tmp_path / "self-cgroup"
        membership.write_text(
            "0::/system.slice/askvlm-test.service\n", encoding="ascii"
        )
        monkeypatch.setenv("ASKVLM_EXTERNAL_SUPERVISOR_MANAGED", "1")
        monkeypatch.setattr(linux_cgroup, "SELF_CGROUP", membership)
        monkeypatch.setattr(linux_cgroup, "CGROUP_ROOT", self.kernel.root)
        monkeypatch.setattr(linux_cgroup, "os", self.kernel)
        for module in (linux_cgroup, procs, supervisor):
            monkeypatch.setattr(module, "sys", SimpleNamespace(platform="linux"))
        monkeypatch.setattr(
            procs,
            "subprocess",
            SimpleNamespace(
                Popen=self.popen,
                PIPE=subprocess.PIPE,
                TimeoutExpired=subprocess.TimeoutExpired,
            ),
        )
        monkeypatch.setattr(
            procs,
            "psutil",
            SimpleNamespace(
                Process=lambda _pid: SimpleNamespace(children=lambda **_kwargs: []),
                Error=psutil.Error,
            ),
        )

    def popen(self, argv: list[str], **_kwargs: object) -> ScriptedPoolProcess:
        """Create an inert process and retain its external observations."""
        process = ScriptedPoolProcess(self, argv)
        self.processes.append(process)
        return process
