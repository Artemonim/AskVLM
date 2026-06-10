"""Unit tests for the file-based transcription queue protocol."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from core import external_queue as q

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _make_job(
    job_id: str = "job1", *, deadline_at: float | None = None
) -> q.TranscribeJob:
    """Build a minimal job for tests.

    Args:
        job_id: Identifier for the job.
        deadline_at: Optional deadline timestamp.

    Returns:
        A populated :class:`~core.external_queue.TranscribeJob`.

    """
    return q.TranscribeJob(
        job_id=job_id,
        input_path="input.wav",
        language="ru",
        device="cuda",
        compute_type="auto",
        whisper_model="small",
        diarization=False,
        dialog_blocks=False,
        submitted_at=100.0,
        deadline_at=deadline_at,
    )


def test_job_roundtrip_via_dict() -> None:
    """A job survives a to_dict/from_dict roundtrip unchanged."""
    job = _make_job(deadline_at=200.0)
    assert q.TranscribeJob.from_dict(job.to_dict()) == job


def test_result_roundtrip_and_unknown_status_falls_back_to_error() -> None:
    """Results roundtrip, and an unknown status decodes as ``error``."""
    result = q.TranscribeResult("j", "ok", "text", None, None, 1.0)
    assert q.TranscribeResult.from_dict(result.to_dict()) == result
    decoded = q.TranscribeResult.from_dict({"job_id": "j", "status": "weird"})
    assert decoded.status == "error"


def test_submit_list_and_claim(tmp_path: Path) -> None:
    """Submitted jobs are listed oldest-first and claim moves them."""
    older = _make_job("old")
    newer = q.TranscribeJob(
        job_id="new",
        input_path="input.wav",
        language=None,
        device="cuda",
        compute_type="auto",
        whisper_model="small",
        diarization=False,
        dialog_blocks=False,
        submitted_at=300.0,
        deadline_at=None,
    )
    q.submit_job(tmp_path, newer)
    q.submit_job(tmp_path, older)
    assert q.list_incoming_job_ids(tmp_path) == ["old", "new"]

    claimed = q.claim_job(tmp_path, "old")
    assert claimed is not None
    assert claimed.job_id == "old"
    assert q.list_incoming_job_ids(tmp_path) == ["new"]


def test_claim_missing_job_returns_none(tmp_path: Path) -> None:
    """Claiming a job that is not present returns ``None``."""
    q.ensure_layout(tmp_path)
    assert q.claim_job(tmp_path, "ghost") is None


def test_result_write_read_and_corrupt_is_ignored(tmp_path: Path) -> None:
    """Results roundtrip on disk and a corrupt result reads as missing."""
    q.write_result(tmp_path, q.TranscribeResult("j", "ok", "hi", None, None, 9.0))
    loaded = q.read_result(tmp_path, "j")
    assert loaded is not None
    assert loaded.text == "hi"

    corrupt = tmp_path / "results" / "bad.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert q.read_result(tmp_path, "bad") is None


def test_cancel_marker_lifecycle(tmp_path: Path) -> None:
    """Cancellation markers are observable and removed by cleanup."""
    q.ensure_layout(tmp_path)
    assert not q.is_cancelled(tmp_path, "j")
    q.request_cancel(tmp_path, "j")
    assert q.is_cancelled(tmp_path, "j")
    q.cleanup_job(tmp_path, "j")
    assert not q.is_cancelled(tmp_path, "j")


def test_cleanup_removes_all_job_files(tmp_path: Path) -> None:
    """Cleanup removes incoming, claimed, result, and cancel artifacts."""
    job = _make_job("j")
    q.submit_job(tmp_path, job)
    q.claim_job(tmp_path, "j")
    q.write_result(tmp_path, q.TranscribeResult("j", "ok", "t", None, None, 1.0))
    q.request_cancel(tmp_path, "j")
    q.cleanup_job(tmp_path, "j")
    assert q.read_result(tmp_path, "j") is None
    assert q.list_incoming_job_ids(tmp_path) == []
    assert not q.is_cancelled(tmp_path, "j")


def test_purge_expired_incoming(tmp_path: Path) -> None:
    """Incoming jobs past their deadline are dropped and reported."""
    q.submit_job(tmp_path, _make_job("live", deadline_at=500.0))
    q.submit_job(tmp_path, _make_job("dead", deadline_at=100.0))
    purged = q.purge_expired_incoming(tmp_path, now=200.0)
    assert purged == ["dead"]
    assert q.list_incoming_job_ids(tmp_path) == ["live"]


def test_purge_stale_results(tmp_path: Path) -> None:
    """Results older than the TTL are reclaimed."""
    q.write_result(tmp_path, q.TranscribeResult("j", "ok", "t", None, None, 1.0))
    result_path = tmp_path / "results" / "j.json"
    now = result_path.stat().st_mtime
    old = now - 1000.0
    os.utime(result_path, (old, old))
    purged = q.purge_stale_results(tmp_path, now=now, ttl_s=10.0)
    assert "j" in purged
    assert q.read_result(tmp_path, "j") is None


def test_heartbeat_write_read_clear_and_liveness(tmp_path: Path) -> None:
    """Heartbeat write/read/clear drive the liveness check."""
    assert q.read_heartbeat(tmp_path) is None
    assert not q.is_daemon_alive(tmp_path, now=0.0)
    q.write_heartbeat(tmp_path, {"model": "small"})
    beacon = q.read_heartbeat(tmp_path)
    assert beacon is not None
    ts = float(beacon["ts"])
    assert q.is_daemon_alive(tmp_path, now=ts + 1.0, stale_after_s=30.0)
    assert not q.is_daemon_alive(tmp_path, now=ts + 100.0, stale_after_s=30.0)
    q.clear_heartbeat(tmp_path)
    assert q.read_heartbeat(tmp_path) is None


def test_daemon_lock_blocks_second_holder_when_alive(tmp_path: Path) -> None:
    """A second lock attempt fails while the first holder is alive."""
    q.ensure_layout(tmp_path)
    first = q.DaemonLock(tmp_path)
    assert first.try_acquire()
    q.write_heartbeat(tmp_path, {"model": "small"})
    second = q.DaemonLock(tmp_path)
    assert not second.try_acquire()
    first.release()


def test_daemon_lock_reclaims_stale_lock(tmp_path: Path) -> None:
    """A lock with no fresh heartbeat is reclaimed by a new daemon."""
    q.ensure_layout(tmp_path)
    stale_lock = tmp_path / "control" / "daemon.lock"
    stale_lock.write_text("999999", encoding="utf-8")
    lock = q.DaemonLock(tmp_path)
    assert lock.try_acquire()
    lock.release()
    assert not stale_lock.exists()


def test_daemon_lock_context_manager(tmp_path: Path) -> None:
    """The lock releases on context exit."""
    q.ensure_layout(tmp_path)
    with q.DaemonLock(tmp_path) as lock:
        assert lock.try_acquire()
    assert not (tmp_path / "control" / "daemon.lock").exists()


def test_resolve_queue_dir_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit argument wins over env which wins over the default."""
    explicit = tmp_path / "explicit"
    assert q.resolve_queue_dir(explicit) == explicit.resolve()

    env_dir = tmp_path / "from_env"
    monkeypatch.setenv(q.ENV_QUEUE_DIR, str(env_dir))
    assert q.resolve_queue_dir() == env_dir.resolve()

    monkeypatch.delenv(q.ENV_QUEUE_DIR, raising=False)
    assert q.resolve_queue_dir().name == "external_queue"


def test_new_job_id_is_unique() -> None:
    """Fresh job identifiers do not collide."""
    assert q.new_job_id() != q.new_job_id()


def test_is_windows_matches_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """The platform helper reflects ``sys.platform``."""
    monkeypatch.setattr(q.sys, "platform", "win32")
    assert q.is_windows()
    monkeypatch.setattr(q.sys, "platform", "linux")
    assert not q.is_windows()
