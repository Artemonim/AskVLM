"""Scenarios through the real worker entry and real process/Job Object owner."""

from __future__ import annotations

import gc
import queue
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psutil  # type: ignore[import-untyped]
import pytest

from core.stt_pool import procs
from core.stt_pool.engine_api import WorkerSpec, WorkItem
from core.stt_pool.keys import make_key
from core.stt_pool.procs import WorkerProcess, default_worker_launch

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from core.stt_pool.procs import WorkerLaunch

pytestmark = [pytest.mark.integration, pytest.mark.xdist_group(name="pool_processes")]
_ROOT = Path(__file__).resolve().parents[1]
_TIMEOUT_S = 18.0
_MEMORY_CAP = 128 * 1024**2


class WorkerPeer:
    """Drive production worker.py through production process ownership and IPC."""

    def __init__(
        self,
        tmp_path: Path,
        memory_cap: int = _MEMORY_CAP,
        *,
        launch: WorkerLaunch | None = None,
    ) -> None:
        self.events: queue.Queue[dict[str, object] | None] = queue.Queue()
        self.log_path = tmp_path / "worker.log"
        self.close_marker = tmp_path / "engine-closed"
        self.worker = WorkerProcess(
            "production-entry-test",
            default_worker_launch(project_root=_ROOT) if launch is None else launch,
            log_path=self.log_path,
            memory_limit_bytes=memory_cap,
            on_event=lambda _worker_id, message: self.events.put(message),
        )

    def specification(self, **options: object) -> dict[str, object]:
        """Build a trusted spec that imports only the tiny fake engine."""
        return WorkerSpec(
            "production-entry-test",
            "test-instance",
            make_key("whisper", "small", "auto", "cpu"),
            2,
            2,
            0,
            "cpu",
            {"close_marker": str(self.close_marker), **options},
            "tests.stt_pool_worker_engine:create_engine",
        ).to_message()

    def start(self, **options: object) -> None:
        """Complete the real spec, hello and loaded handshake."""
        self.worker.start(self.specification(**options))
        self.hello()
        loaded = self.next_message("loaded")
        assert loaded["device"] == "cpu"

    def hello(self) -> None:
        """Verify identity from the live subprocess, including its Windows cap."""
        hello = self.next_message("hello")
        assert hello["pid"] == self.worker.pid
        if sys.platform == "win32":
            assert self.worker.in_job

    def next_message(self, kind: str) -> dict[str, object]:
        """Take the next protocol record and fail on a missing or unexpected event."""
        message = self.events.get(timeout=_TIMEOUT_S)
        assert message is not None
        assert message["type"] == kind, (message, self.worker.log_tail(6000))
        return message

    def finish(self, code: int) -> None:
        """Prove no late messages, exactly one EOF, and a reaped worker process."""
        assert self.events.get(timeout=_TIMEOUT_S) is None
        assert self.worker.wait(_TIMEOUT_S) == code
        self.worker.close(timeout_s=_TIMEOUT_S)
        assert self.events.empty()
        assert self.worker.pid is not None
        assert not psutil.pid_exists(self.worker.pid)

    def shutdown(self) -> None:
        """Request graceful shutdown and verify the final bye/exit contract."""
        assert self.worker.request_shutdown()
        self.next_message("bye")
        self.finish(0)

    def close_input(self) -> None:
        """Close the parent's owned stdin pipe without killing the Job Object."""
        # * The real writer closes its pipe after draining earlier queued messages.
        self.worker._outbox.put(procs._STOP_WRITER)  # noqa: SLF001 - simulate parent EOF at the owned process boundary


@pytest.fixture
def worker_peer(tmp_path: Path) -> Iterator[WorkerPeer]:
    """Reap only the worker created by this test on success and every failure path."""
    peer = WorkerPeer(tmp_path)
    try:
        yield peer
    finally:
        peer.worker.close(timeout_s=_TIMEOUT_S)
        gc.collect()


def _item(tmp_path: Path, seq: int = 1) -> WorkItem:
    """Create one Unicode input with an explicit, unique attempt identity."""
    job_id = f"{seq:032x}"
    return WorkItem(job_id, f"{job_id}:2", tmp_path / "русский.wav", "ru", 1.0, None)


@pytest.mark.skipif(
    sys.platform != "win32", reason="Real Windows error-mode inheritance"
)
@pytest.mark.parametrize("probe", ["error_mode", "loader_failure"])
def test_worker_from_zero_mode_protects_children_from_modal_loader_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, probe: str
) -> None:
    """Catch shell flags masking a worker defect or a modal loader hang."""
    native_popen = subprocess.Popen

    def popen(
        argv: Sequence[str],
        **kwargs: Any,  # noqa: ANN401 - external subprocess keyword options
    ) -> subprocess.Popen[bytes]:
        flags = kwargs.get("creationflags", 0)
        assert isinstance(flags, int)
        kwargs["creationflags"] = flags | subprocess.CREATE_DEFAULT_ERROR_MODE
        return native_popen(argv, **kwargs)

    monkeypatch.setattr(procs.subprocess, "Popen", popen)
    launch = default_worker_launch(
        project_root=_ROOT, module="tests.stt_pool_windows_support"
    )
    launch = replace(launch, argv=(*launch.argv, "--mode", "worker"))
    peer = WorkerPeer(tmp_path, launch=launch)
    try:
        peer.worker.start(peer.specification(mode=probe))
        peer.hello()
        details = peer.next_message("loaded")["details"]
        assert isinstance(details, dict)
        assert details["initial_mode"] == 0
        assert details["worker_mode"] & 0x8003 == 0x8003
        assert details["child_mode"] & 0x8003 == 0x8003
        if probe == "loader_failure":
            assert details["loader_exit"] == 0xC000012D
            assert details["loader_stdout"] == ""
        peer.shutdown()
    finally:
        peer.worker.close(timeout_s=_TIMEOUT_S)


@pytest.mark.parametrize("payload_char", ["x", "汉"])
def test_real_worker_replaces_oversized_utf8_payload_for_the_same_attempt(
    tmp_path: Path, payload_char: str
) -> None:
    """Catch oversized IPC payloads being silently skipped or byte limits being counted as characters."""
    peer = WorkerPeer(tmp_path, memory_cap=384 * 1024**2)
    try:
        peer.start(mode="oversized_result", payload_char=payload_char)
        item = _item(tmp_path)
        assert peer.worker.send(item.to_message())
        result = peer.next_message("result")
        assert result["status"] == "error"
        assert result["error_kind"] == "result_too_large"
        assert result["job_id"] == item.job_id
        assert result["attempt_id"] == item.attempt_id
        assert not result.get("data")
        following = _item(tmp_path, 2)
        assert peer.worker.send(following.to_message())
        assert peer.next_message("result")["attempt_id"] == following.attempt_id
        peer.shutdown()
    finally:
        peer.worker.close(timeout_s=_TIMEOUT_S)


def test_real_worker_handshake_result_stdout_and_bye(
    worker_peer: WorkerPeer, tmp_path: Path
) -> None:
    """Catch missing engine wiring, native stdout leakage, Unicode damage or duplicate results."""
    worker_peer.start()
    item = _item(tmp_path)
    assert worker_peer.worker.send(item.to_message())
    result = worker_peer.next_message("result")
    assert result["job_id"] == item.job_id
    assert result["attempt_id"] == item.attempt_id
    assert result["status"] == "ok"
    assert result["text"] == "ru:русский.wav"
    assert result["backend"] == "cpu"
    worker_peer.shutdown()
    assert worker_peer.close_marker.is_file()
    log = worker_peer.log_path.read_text(encoding="utf-8")
    assert "stray-python" in log
    assert "stray-native" in log
    worker_peer.log_path.rename(tmp_path / "released-worker.log")


def test_real_worker_parallel_engine_emissions_keep_attempts_separate(
    worker_peer: WorkerPeer, tmp_path: Path
) -> None:
    """Catch concurrent callback writes interleaving JSON frames or losing one attempt's result."""
    worker_peer.start(mode="parallel")
    items = [_item(tmp_path, seq) for seq in range(1, 5)]
    for item in items:
        assert worker_peer.worker.send(item.to_message())
    results = [worker_peer.next_message("result") for _ in items]
    assert {(result["job_id"], result["attempt_id"]) for result in results} == {
        (item.job_id, item.attempt_id) for item in items
    }
    assert all(result["status"] == "ok" for result in results)
    worker_peer.shutdown()


@pytest.mark.skipif(
    sys.platform != "win32", reason="Real Windows Job Object tree ownership"
)
def test_real_worker_parent_eof_reaps_its_descendants(
    worker_peer: WorkerPeer,
) -> None:
    """Catch an EOF-exited worker's ffmpeg-like descendants surviving Job Object teardown."""
    message = worker_peer.specification(mode="tree")
    message["engine_factory"] = "tests.stt_pool_fake_worker:create_engine"
    worker_peer.worker.start(message)
    worker_peer.hello()
    details = worker_peer.next_message("loaded")["details"]
    assert isinstance(details, dict)
    child = psutil.Process(int(details["child_pid"]))
    grandchild = psutil.Process(int(details["grandchild_pid"]))
    worker_peer.close_input()
    worker_peer.finish(3)
    child.wait(timeout=_TIMEOUT_S)
    grandchild.wait(timeout=_TIMEOUT_S)
    assert not psutil.pid_exists(child.pid)
    assert not psutil.pid_exists(grandchild.pid)


@pytest.mark.parametrize("mode", ["empty", "error"])
def test_real_worker_preserves_empty_and_consumer_error_markers(
    worker_peer: WorkerPeer, tmp_path: Path, mode: str
) -> None:
    """Catch empty becoming failure or the bot's external error classification markers being lost."""
    worker_peer.start(mode=mode)
    assert worker_peer.worker.send(_item(tmp_path).to_message())
    result = worker_peer.next_message("result")
    if mode == "empty":
        assert result["status"] == "empty"
        assert result["text"] == ""
        assert result["error_kind"] is None
    else:
        assert result["status"] == "error"
        assert result["error_kind"] == "transcribe_failed"
        assert "Too long wav file" in str(result["error_detail"])
        assert "transcribe_longform" in str(result["error_detail"])
    worker_peer.shutdown()


@pytest.mark.parametrize(
    ("load_error", "kind"),
    [
        ("generic", "worker_load_failed"),
        ("cuda_oom", "cuda_oom"),
        ("device_unavailable", "device_unavailable"),
        ("memory_limit", "memory_limit"),
    ],
)
def test_real_worker_load_failure_is_fatal_before_loaded(
    worker_peer: WorkerPeer, load_error: str, kind: str
) -> None:
    """Catch failed weights being advertised as ready or typed resource failures losing their kind."""
    worker_peer.worker.start(worker_peer.specification(load_error=load_error))
    worker_peer.hello()
    fatal = worker_peer.next_message("fatal")
    assert fatal["stage"] == "load"
    assert fatal["error_kind"] == kind
    assert fatal["error_detail"]
    worker_peer.finish(4)
    assert not worker_peer.close_marker.exists()


def test_real_worker_rejects_actual_device_mismatch(worker_peer: WorkerPeer) -> None:
    """Catch a worker silently serving another device class than the reserved key."""
    worker_peer.worker.start(worker_peer.specification(loaded_device="cuda"))
    worker_peer.hello()
    fatal = worker_peer.next_message("fatal")
    assert fatal["stage"] == "load"
    assert fatal["error_kind"] == "device_unavailable"
    assert "cpu" in str(fatal["error_detail"])
    assert "cuda" in str(fatal["error_detail"])
    worker_peer.finish(4)


def test_real_worker_start_failure_closes_without_advertising_loaded(
    worker_peer: WorkerPeer,
) -> None:
    """Catch partially started engines reporting readiness or surviving a startup failure."""
    worker_peer.worker.start(worker_peer.specification(start_error=True))
    worker_peer.hello()
    fatal = worker_peer.next_message("fatal")
    assert fatal["stage"] == "load"
    assert fatal["error_kind"] == "worker_load_failed"
    worker_peer.finish(4)
    assert worker_peer.close_marker.is_file()


@pytest.mark.parametrize(
    "invalid_fields", [{"protocol": 1}, {"capacity": 0}, {"type": "shutdown"}]
)
def test_real_worker_invalid_spec_has_no_hello(
    worker_peer: WorkerPeer, invalid_fields: dict[str, object]
) -> None:
    """Catch malformed initial configuration importing an engine or reaching the ready state."""
    message = worker_peer.specification()
    message.update(invalid_fields)
    worker_peer.worker.start(message)
    fatal = worker_peer.next_message("fatal")
    assert fatal["stage"] == "spec"
    assert fatal["error_kind"] == "protocol_error"
    worker_peer.finish(4)
    assert not worker_peer.close_marker.exists()


def test_real_worker_missing_factory_is_a_load_failure(worker_peer: WorkerPeer) -> None:
    """Catch factory resolution failures escaping without a load-stage fatal response."""
    message = worker_peer.specification()
    message["engine_factory"] = "tests.stt_pool_worker_engine:missing_factory"
    worker_peer.worker.start(message)
    worker_peer.hello()
    fatal = worker_peer.next_message("fatal")
    assert fatal["stage"] == "load"
    assert fatal["error_kind"] == "worker_load_failed"
    worker_peer.finish(4)


def test_real_worker_cancel_targets_attempt_and_answers_once(
    worker_peer: WorkerPeer, tmp_path: Path
) -> None:
    """Catch a late attempt cancel killing current work or repeated cancels duplicating its result."""
    worker_peer.start(mode="hold")
    item = _item(tmp_path)
    assert worker_peer.worker.send(item.to_message())
    cancel = {"type": "cancel", "job_id": item.job_id, "attempt_id": "obsolete:1"}
    assert worker_peer.worker.send(cancel)
    cancel["attempt_id"] = item.attempt_id
    assert worker_peer.worker.send(cancel)
    result = worker_peer.next_message("result")
    assert result["job_id"] == item.job_id
    assert result["attempt_id"] == item.attempt_id
    assert result["status"] == "cancelled"
    assert worker_peer.worker.send(cancel)
    worker_peer.shutdown()


def test_real_worker_unencodable_result_is_replaced_for_the_same_attempt(
    worker_peer: WorkerPeer, tmp_path: Path
) -> None:
    """Catch unserializable model output leaving a submitted attempt waiting forever."""
    worker_peer.start(mode="bad_result")
    for seq in (1, 2):
        item = _item(tmp_path, seq)
        assert worker_peer.worker.send(item.to_message())
        result = worker_peer.next_message("result")
        assert result["job_id"] == item.job_id
        assert result["attempt_id"] == item.attempt_id
        assert result["status"] == "error"
        assert result["error_kind"] == "protocol_error"
        assert result["text"] == ""
    worker_peer.shutdown()


def test_real_worker_malformed_jobs_are_bounded_and_do_not_kill_dispatch(
    worker_peer: WorkerPeer, tmp_path: Path
) -> None:
    """Catch one malformed job dropping other attempts or fabricating unusable result identities."""
    worker_peer.start()
    invalid = _item(tmp_path).to_message()
    invalid["duration_s"] = -1
    assert worker_peer.worker.send(invalid)
    result = worker_peer.next_message("result")
    assert result["job_id"] == invalid["job_id"]
    assert result["attempt_id"] == invalid["attempt_id"]
    assert result["error_kind"] == "protocol_error"
    assert worker_peer.worker.send({"type": "job"})
    assert worker_peer.worker.send({"type": "cancel"})
    next_item = _item(tmp_path, 2)
    assert worker_peer.worker.send(next_item.to_message())
    assert worker_peer.next_message("result")["job_id"] == next_item.job_id
    worker_peer.shutdown()


def test_real_worker_unexpected_submit_failure_is_fatal_at_run(
    worker_peer: WorkerPeer, tmp_path: Path
) -> None:
    """Catch run-stage engine exceptions exiting without an actionable fatal event."""
    worker_peer.start(mode="submit_error")
    assert worker_peer.worker.send(_item(tmp_path).to_message())
    fatal = worker_peer.next_message("fatal")
    assert fatal["stage"] == "run"
    assert fatal["error_kind"] == "protocol_error"
    worker_peer.finish(4)
    assert worker_peer.close_marker.is_file()


def test_real_worker_parent_eof_exits_without_waiting_for_work(
    worker_peer: WorkerPeer, tmp_path: Path
) -> None:
    """Catch a worker orphan lingering after the parent closes stdin with unfinished work."""
    worker_peer.start(mode="hold")
    assert worker_peer.worker.send(_item(tmp_path).to_message())
    worker_peer.close_input()
    worker_peer.finish(3)
    assert not worker_peer.close_marker.exists()


@pytest.mark.parametrize("close_error", [False, True])
def test_real_worker_shutdown_relinquishes_unfinished_attempts(
    worker_peer: WorkerPeer, tmp_path: Path, *, close_error: bool
) -> None:
    """Catch shutdown inventing terminal outcomes that prevent supervisor requeue, or losing bye."""
    worker_peer.start(mode="hold", close_error=close_error)
    assert worker_peer.worker.send(_item(tmp_path).to_message())
    worker_peer.shutdown()
    assert worker_peer.close_marker.is_file()


@pytest.mark.slow
def test_real_worker_bounds_a_stuck_engine_close(worker_peer: WorkerPeer) -> None:
    """Catch shutdown hanging permanently on a model thread instead of exiting and freeing memory."""
    worker_peer.start(close_hangs=True)
    worker_peer.shutdown()
    assert worker_peer.close_marker.is_file()
