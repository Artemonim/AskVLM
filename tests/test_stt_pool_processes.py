"""Windows lifecycle integrations with tiny peers, never real ML workers."""

from __future__ import annotations

import contextlib
import gc
import json
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psutil  # type: ignore[import-untyped]
import pytest

from core.stt_pool import procs, winjob
from core.stt_pool.engine_api import WorkerSpec, WorkItem
from core.stt_pool.keys import make_key
from core.stt_pool.procs import WorkerProcess, WorkerSpawnError, default_worker_launch
from core.stt_pool.queue import QueuePaths, SupervisorLock
from core.stt_pool.sysprobe import query_gpu
from tests.stt_pool_supervisor_support import LinuxPoolBoundary
from tests.stt_pool_windows_support import JobCapKernel, allocation_child

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = [pytest.mark.integration, pytest.mark.xdist_group(name="pool_processes")]
_ROOT = Path(__file__).resolve().parents[1]
_TIMEOUT_S = 8.0
_MEMORY_CAP = 128 * 1024**2


class Peer:
    """Own a real WorkerProcess and a bounded event queue."""

    def __init__(self, tmp_path: Path) -> None:
        self.events: queue.Queue[dict[str, object] | None] = queue.Queue()
        self.memory_hits: queue.Queue[int] = queue.Queue()
        self.worker = WorkerProcess(
            "fake-peer",
            default_worker_launch(
                project_root=_ROOT, module="tests.stt_pool_fake_worker"
            ),
            log_path=tmp_path / "peer.log",
            memory_limit_bytes=_MEMORY_CAP,
            on_event=lambda _worker_id, message: self.events.put(message),
            on_memory_limit=lambda _worker_id, pid: self.memory_hits.put(pid),
        )

    def start(self, **options: object) -> dict[str, object]:
        """Start the tiny peer after its real Job Object has been assigned."""
        spec = WorkerSpec(
            "fake-peer",
            "test-instance",
            make_key("whisper", "small", "auto", "cpu"),
            2,
            2,
            0,
            "cpu",
            options,
            "tests.stt_pool_fake_worker:create_engine",
        )
        self.worker.start(spec.to_message())
        hello = self.next_message("hello")
        assert hello["pid"] == self.worker.pid
        return self.next_message("loaded")

    def next_message(self, kind: str) -> dict[str, object]:
        """Read a bounded message; fail on EOF or stray data."""
        message = self.events.get(timeout=_TIMEOUT_S)
        assert message is not None
        assert message["type"] == kind
        return message


@pytest.fixture
def peer(tmp_path: Path) -> Iterator[Peer]:
    """Kill and reap only the tiny peer and its own descendants on every exit."""
    instance = Peer(tmp_path)
    try:
        yield instance
    finally:
        instance.worker.close(timeout_s=_TIMEOUT_S)
        gc.collect()


def test_protocol_round_trip_protects_stdout_and_delivers_one_eof(
    peer: Peer, tmp_path: Path
) -> None:
    """Catch protocol noise, Unicode damage, duplicate EOF or leaked handles."""
    peer.start()
    item = WorkItem("a" * 32, "attempt:2", tmp_path / "русский.wav", "ru", 1.0, None)
    assert peer.worker.send(item.to_message())
    result = peer.next_message("result")
    assert result["job_id"] == item.job_id
    assert result["attempt_id"] == item.attempt_id
    assert result["text"] == "ru:русский.wav"
    assert peer.worker.request_shutdown()
    peer.next_message("bye")
    assert peer.events.get(timeout=_TIMEOUT_S) is None
    assert peer.worker.wait(_TIMEOUT_S) == 0
    peer.worker.close(timeout_s=_TIMEOUT_S)
    peer.worker.close(timeout_s=_TIMEOUT_S)
    assert peer.events.empty()
    assert not peer.worker.send(item.to_message())
    assert not any(t.name.endswith("fake-peer") for t in threading.enumerate())
    # * Exclusive reopening checks that the parent has released its log handle.
    (tmp_path / "peer.log").rename(tmp_path / "closed.log")


@pytest.mark.skipif(sys.platform != "win32", reason="Real Windows Job Object contract")
def test_job_termination_reaps_worker_child_and_grandchild(peer: Peer) -> None:
    """Catch descendants retaining commit or pipes after worker termination."""
    loaded = peer.start(mode="tree")
    details = loaded["details"]
    assert isinstance(details, dict)
    child = psutil.Process(int(details["child_pid"]))
    grandchild = psutil.Process(int(details["grandchild_pid"]))
    assert child.is_running()
    assert grandchild.is_running()
    assert peer.worker.in_job
    peer.worker.kill()
    assert peer.worker.wait(_TIMEOUT_S) is not None
    child.wait(timeout=_TIMEOUT_S)
    grandchild.wait(timeout=_TIMEOUT_S)
    peer.worker.close(timeout_s=_TIMEOUT_S)
    assert peer.events.get(timeout=_TIMEOUT_S) is None
    assert not psutil.pid_exists(child.pid)
    assert not psutil.pid_exists(grandchild.pid)


@pytest.mark.skipif(
    sys.platform != "win32", reason="Real Windows commit cap notification"
)
def test_job_memory_limit_is_reported_without_unbounded_allocation(
    peer: Peer, tmp_path: Path
) -> None:
    """Catch missing Job memory caps or OOM notifications."""
    peer.start(mode="memory")
    item = WorkItem("b" * 32, "attempt:1", tmp_path / "clip.wav", None, 1.0, None)
    assert peer.worker.send(item.to_message())
    result = peer.next_message("result")
    assert result["status"] == "error"
    assert result["error_kind"] == "memory_limit"
    assert peer.memory_hits.get(timeout=_TIMEOUT_S) == peer.worker.pid
    assert peer.worker.memory_limit_hit
    peak = peer.worker.peak_memory_bytes()
    assert peak is not None
    assert peak > 0
    assert peer.worker.pid is not None
    assert psutil.Process(peer.worker.pid).memory_info().private <= _MEMORY_CAP


@pytest.mark.skipif(sys.platform != "win32", reason="Real Windows Job Object reports")
def test_live_cap_rearms_for_new_processes_and_preserves_kill_on_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch repeated-process notifications, a lost re-arm or orphaned raised jobs."""
    kernel = JobCapKernel()
    monkeypatch.setattr(winjob, "_kernel32", lambda: kernel)
    reports: queue.Queue[int] = queue.Queue()
    job = winjob.WorkerJob(memory_limit_bytes=96 * 1024**2, on_memory_limit=reports.put)
    try:
        assert job.active, job.error
        with allocation_child() as first:
            assert job.assign(first.process.pid), job.assign_error
            assert first.allocate(256 * 1024**2)["denied"]
            assert reports.get(timeout=30) == first.process.pid
            assert job.memory_limit_hits == 1
            for _ in range(2):
                assert first.allocate(256 * 1024**2)["denied"]
            kernel.fence(job)
            assert kernel.raw_hits == [first.process.pid]
            original = kernel.query(job._handle)  # noqa: SLF001 - observe real external job limits
            assert job.set_memory_limit(128 * 1024**2), job.limit_error
            raised = kernel.query(job._handle)  # noqa: SLF001 - observe real external job limits
            assert raised.JobMemoryLimit == 128 * 1024**2
            assert (
                raised.BasicLimitInformation.LimitFlags
                == original.BasicLimitInformation.LimitFlags
            )
            assert raised.BasicLimitInformation.LimitFlags & 0x2000
            assert first.allocate(256 * 1024**2)["denied"]
            kernel.fence(job)
            assert kernel.raw_hits == [first.process.pid]
            assert reports.empty()
            assert job.memory_limit_hits == 1
            with allocation_child() as second:
                assert job.assign(second.process.pid), job.assign_error
                assert second.allocate(256 * 1024**2)["denied"]
                assert reports.get(timeout=30) == second.process.pid
                kernel.fence(job)
                assert kernel.raw_hits == [first.process.pid, second.process.pid]
                assert job.memory_limit_hits == 2
                assert job.memory_limit_pid == second.process.pid
                assert job.memory_limit_hit
                job.close()
                # * Both children must exit before the controller sends any EOF.
                first.process.wait(timeout=30)
                second.process.wait(timeout=30)
                assert first.process.poll() is not None
                assert second.process.poll() is not None
        assert job.memory_limit_hits == 2
    finally:
        job.close()


@pytest.mark.skipif(sys.platform != "win32", reason="Real Job Object update failures")
@pytest.mark.parametrize("failure", ["query", "post", "set"])
def test_failed_live_cap_update_keeps_the_enforced_limit_and_flags(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Catch partial Win32 failures silently removing the cap or kill-on-close."""
    kernel = JobCapKernel()
    monkeypatch.setattr(winjob, "_kernel32", lambda: kernel)
    job = winjob.WorkerJob(memory_limit_bytes=96 * 1024**2)
    try:
        assert job.active, job.error
        original = kernel.query(job._handle)  # noqa: SLF001 - real external state
        kernel.fail_update = failure
        assert not job.set_memory_limit(128 * 1024**2)
        assert job.limit_error is not None
        unchanged = kernel.query(job._handle)  # noqa: SLF001 - real external state
        assert unchanged.JobMemoryLimit == original.JobMemoryLimit
        assert (
            unchanged.BasicLimitInformation.LimitFlags
            == original.BasicLimitInformation.LimitFlags
        )
        assert job.memory_limit_bytes == 96 * 1024**2
        kernel.fail_update = None
        assert job.set_memory_limit(128 * 1024**2), job.limit_error
        assert job.limit_error is None
        assert kernel.query(job._handle).JobMemoryLimit == 128 * 1024**2  # noqa: SLF001 - real external state
    finally:
        job.close()


@pytest.mark.skipif(sys.platform != "win32", reason="Real completion-port FIFO")
def test_rearm_discards_queued_old_denials_before_reporting_a_new_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch stale packets consuming the raised cap's one-shot report."""
    kernel = JobCapKernel()
    monkeypatch.setattr(winjob, "_kernel32", lambda: kernel)
    reports: queue.Queue[int] = queue.Queue()
    release_monitor = threading.Event()

    def report(pid: int) -> None:
        reports.put(pid)
        if not release_monitor.wait(30):
            msg = "test did not release the first completion-port callback"
            raise TimeoutError(msg)

    job = winjob.WorkerJob(memory_limit_bytes=128 * 1024**2, on_memory_limit=report)
    try:
        assert job.active, job.error
        with contextlib.ExitStack() as stack:
            first, stale, fresh = [
                stack.enter_context(allocation_child()) for _ in range(3)
            ]
            assert job.assign(first.process.pid), job.assign_error
            assert first.allocate(256 * 1024**2)["denied"]
            assert reports.get(timeout=30) == first.process.pid
            # * The monitor remains in its callback while old packets queue.
            assert job.assign(stale.process.pid), job.assign_error
            assert stale.allocate(256 * 1024**2)["denied"]
            assert job.set_memory_limit(192 * 1024**2), job.limit_error
            assert job.assign(fresh.process.pid), job.assign_error
            assert fresh.allocate(256 * 1024**2)["denied"]
            release_monitor.set()
            assert reports.get(timeout=30) == fresh.process.pid
            kernel.fence(job)
            assert kernel.raw_hits == [
                first.process.pid,
                stale.process.pid,
                fresh.process.pid,
            ]
            assert job.memory_limit_hits == 2
            assert job.memory_limit_pid == fresh.process.pid
            assert reports.empty()
    finally:
        release_monitor.set()
        job.close()


def test_supervisor_lock_is_exclusive_across_processes_and_released_on_death(
    peer: Peer, tmp_path: Path
) -> None:
    """Catch process-local locks or dead owners blocking takeover."""
    paths = QueuePaths(tmp_path / "queue")
    peer.start(mode="lock", root=str(paths.root))
    with SupervisorLock(paths) as contender:
        assert not contender.try_acquire()
        assert not contender.held
        peer.worker.kill()
        assert peer.worker.wait(_TIMEOUT_S) is not None
        assert contender.try_acquire()
        with SupervisorLock(paths) as other:
            assert not other.try_acquire()
    with SupervisorLock(paths) as next_owner:
        assert next_owner.try_acquire()


@pytest.mark.skipif(sys.platform != "win32", reason="Win32 assignment failure boundary")
def test_failed_job_assignment_kills_the_peer_before_sending_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch failed Job assignment permitting uncapped inference or leaks."""
    kernel = winjob._kernel32()  # noqa: SLF001 - substitute only the Win32 API boundary

    class RefusingKernel:
        """Delegate real Win32 operations except the injected assignment failure."""

        def __getattr__(self, name: str) -> object:
            return getattr(kernel, name)

        @staticmethod
        def AssignProcessToJobObject(_job: object, _process: object) -> int:  # noqa: N802 - Win32 symbol
            """Return a deterministic denied assignment."""
            return 0

    monkeypatch.setattr(winjob, "_kernel32", RefusingKernel)
    instance = Peer(tmp_path)
    marker = tmp_path / "spec-received"
    try:
        with pytest.raises(WorkerSpawnError):
            instance.start(marker=str(marker))
        assert instance.worker.pid is not None
        assert instance.worker.poll() is not None
        assert not marker.exists()
    finally:
        instance.worker.close(timeout_s=_TIMEOUT_S)
        gc.collect()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows breakaway creation flags")
def test_detached_spawn_retries_refused_breakaway_at_the_process_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch refused breakaway, visible consoles or unreaped children."""
    real_popen = subprocess.Popen
    flags: list[int] = []
    children: list[subprocess.Popen[bytes]] = []

    def popen_at_os_boundary(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:  # noqa: ANN401 - Popen boundary signature
        flags.append(int(kwargs["creationflags"]))
        if len(flags) == 1:
            error = OSError("job refuses breakaway")
            error.winerror = 5
            raise error
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    launch = default_worker_launch(
        project_root=_ROOT, module="tests.stt_pool_fake_worker"
    )
    monkeypatch.setattr(subprocess, "Popen", popen_at_os_boundary)
    try:
        pid = procs.spawn_detached(
            (*launch.argv, "--exit"),
            cwd=_ROOT,
            env=launch.env,
            log_path=tmp_path / "detached.log",
        )
        assert children[0].pid == pid
        assert children[0].wait(timeout=_TIMEOUT_S) == 0
        assert len(flags) == 2
        assert flags[0] & winjob.CREATE_BREAKAWAY_FROM_JOB
        assert not flags[1] & winjob.CREATE_BREAKAWAY_FROM_JOB
        assert all(value & subprocess.CREATE_NO_WINDOW for value in flags)
        assert all(value & subprocess.CREATE_NEW_PROCESS_GROUP for value in flags)
        assert all(not value & subprocess.DETACHED_PROCESS for value in flags)
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=_TIMEOUT_S)
        gc.collect()


@pytest.mark.parametrize(
    ("record", "valid"),
    [
        ("GPU name, with comma, 10, 100", True),
        ("GPU, NaN, 100", False),
        ("GPU, 1e308, 100", False),
        ("GPU, 1, 1e308", False),
        ("GPU, 1073741825, 1073741825", False),
        ("GPU, 1, 1073741825", False),
        ("GPU, 10, 0", False),
        ("junk", False),
    ],
)
def test_gpu_probe_uses_csv_units_and_rejects_invalid_readings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, record: str, *, valid: bool
) -> None:
    """Catch invalid or mis-scaled VRAM probes authorizing an unsafe GPU spawn."""
    real_popen = subprocess.Popen
    children: list[subprocess.Popen[bytes]] = []
    launch = default_worker_launch(
        project_root=_ROOT, module="tests.stt_pool_fake_worker"
    )

    def smi_boundary(_argv: object, **kwargs: Any) -> subprocess.Popen[bytes]:  # noqa: ANN401 - Popen boundary signature
        child = real_popen(
            (*launch.argv, "--smi", record), cwd=_ROOT, env=dict(launch.env), **kwargs
        )
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", smi_boundary)
    try:
        snapshot = query_gpu("fake-nvidia-smi", timeout_s=5, clock=lambda: 1234.0)
        if valid:
            assert snapshot is not None
            assert snapshot.name == "GPU name, with comma"
            assert snapshot.free_bytes == 90 * 1024**2
            assert snapshot.taken_at == 1234.0
        else:
            assert snapshot is None
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
            for stream in (child.stdin, child.stdout, child.stderr):
                if stream is not None:
                    stream.close()


def test_pool_imports_in_a_fresh_process_cannot_load_ml_dependencies() -> None:
    """Catch ML imports allocating commit before worker admission."""
    launch = default_worker_launch(
        project_root=_ROOT, module="tests.stt_pool_fake_worker"
    )
    result = subprocess.run(  # noqa: S603 - trusted tiny import guard, no shell
        (*launch.argv, "--import-pool"),
        cwd=_ROOT,
        env=dict(launch.env),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        timeout=10,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    assert json.loads(result.stdout)["imported"] > 0


def test_managed_worker_is_capped_and_enrolled_before_exec_and_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch worker execution or model input preceding mandatory PID enrollment."""
    boundary = LinuxPoolBoundary(monkeypatch, tmp_path)
    instance = Peer(tmp_path)
    try:
        instance.start()
        process = boundary.processes[0]
        group = process.executed_group
        assert group is not None
        assert group.path.parent == boundary.kernel.service_path
        assert process.execution_cap is not None
        assert int(process.execution_cap) == _MEMORY_CAP
        assert not process.spec_before_execution
        assert instance.worker.in_cgroup
        assert not instance.worker.in_job
        assert boundary.kernel.member(process.pid) is group
        assert not boundary.kernel.service.files["cgroup.procs"].strip()
        assert boundary.kernel.moves[0][0] == boundary.kernel.getpid()
        assert all(
            pid in {boundary.kernel.getpid(), process.pid}
            for pid, _ in boundary.kernel.moves
        )
        process.finish(0)
        assert instance.events.get(timeout=_TIMEOUT_S) is None
        instance.worker.close(timeout_s=0)
        assert group.path not in boundary.kernel.nodes
        assert not boundary.kernel.handles
    finally:
        instance.worker.close(timeout_s=0)


@pytest.mark.parametrize("failure", ["controller", "memory.max", "cgroup.procs"])
def test_managed_cap_failure_never_executes_or_sends_spec_uncapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Catch missing delegation or denied writes silently allowing unbounded work."""
    boundary = LinuxPoolBoundary(monkeypatch, tmp_path)
    if failure == "controller":
        boundary.kernel.service.files["cgroup.controllers"] = "pids\n"
    else:
        boundary.kernel.refuse_control = failure
        if failure == "cgroup.procs":
            boundary.kernel.service.files["cgroup.procs"] = ""
    instance = Peer(tmp_path)
    try:
        with pytest.raises(WorkerSpawnError):
            instance.start()
        assert all(not process.executed for process in boundary.processes)
        assert all(process.spec is None for process in boundary.processes)
        assert all(process.poll() is not None for process in boundary.processes)
        assert not boundary.kernel.handles
        assert not any(
            path.name.startswith("askvlm-worker-") for path in boundary.kernel.nodes
        )
    finally:
        instance.worker.close(timeout_s=0)


@pytest.mark.parametrize("managed", [None, "0", "true"])
def test_unmanaged_linux_worker_keeps_its_uncapped_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, managed: str | None
) -> None:
    """Catch applying service cgroups to a standalone Linux worker."""
    boundary = LinuxPoolBoundary(monkeypatch, tmp_path)
    if managed is None:
        monkeypatch.delenv("ASKVLM_EXTERNAL_SUPERVISOR_MANAGED")
    else:
        monkeypatch.setenv("ASKVLM_EXTERNAL_SUPERVISOR_MANAGED", managed)
    monkeypatch.setenv("INVOCATION_ID", "parent-service-invocation")
    instance = Peer(tmp_path)
    try:
        instance.start()
        process = boundary.processes[0]
        assert not instance.worker.in_cgroup
        assert process.executed
        assert process.executed_group is None
        assert "core.stt_pool.linux_cgroup" not in process.args
        assert not boundary.kernel.moves
        process.finish(0)
        assert instance.events.get(timeout=_TIMEOUT_S) is None
    finally:
        instance.worker.close(timeout_s=0)
