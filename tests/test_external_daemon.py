"""Unit tests for the transcription daemon scheduling and lifecycle."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

from core import external_queue as q
from core.external_daemon import DaemonConfig, process_claimed_job, run_daemon

if TYPE_CHECKING:
    from pathlib import Path

TranscribeFn = Callable[[q.TranscribeJob, Callable[[], bool]], q.TranscribeResult]


def _job(job_id: str = "j", *, deadline_at: float | None = None) -> q.TranscribeJob:
    """Build a test job.

    Args:
        job_id: Identifier for the job.
        deadline_at: Optional deadline timestamp.

    Returns:
        A populated :class:`~core.external_queue.TranscribeJob`.

    """
    return q.TranscribeJob(
        job_id=job_id,
        input_path="input.wav",
        language=None,
        device="cpu",
        compute_type="auto",
        whisper_model="small",
        diarization=False,
        dialog_blocks=False,
        submitted_at=0.0,
        deadline_at=deadline_at,
    )


def _ok_fn(text: str) -> TranscribeFn:
    """Return a transcribe function that always succeeds with *text*.

    Args:
        text: The transcript text to return.

    Returns:
        A transcribe callable.

    """

    def _fn(
        job: q.TranscribeJob, should_cancel: Callable[[], bool]
    ) -> q.TranscribeResult:
        return q.TranscribeResult(job.job_id, "ok", text, None, None, 1.0)

    return _fn


def test_process_claimed_job_success(tmp_path: Path) -> None:
    """A successful job publishes an ``ok`` result."""
    result = process_claimed_job(tmp_path, _job(), _ok_fn("hi"), clock=lambda: 0.0)
    assert result.status == "ok"
    stored = q.read_result(tmp_path, "j")
    assert stored is not None
    assert stored.text == "hi"


def test_process_claimed_job_cancel_before_start(tmp_path: Path) -> None:
    """A pre-existing cancel marker skips the transcribe call."""
    q.ensure_layout(tmp_path)
    q.request_cancel(tmp_path, "j")
    calls = {"n": 0}

    def _fn(
        job: q.TranscribeJob, should_cancel: Callable[[], bool]
    ) -> q.TranscribeResult:
        calls["n"] += 1
        return q.TranscribeResult(job.job_id, "ok", "x", None, None, 1.0)

    result = process_claimed_job(tmp_path, _job(), _fn, clock=lambda: 0.0)
    assert result.status == "cancelled"
    assert calls["n"] == 0


def test_process_claimed_job_error(tmp_path: Path) -> None:
    """An exception in the transcribe call yields a terminal ``error``."""

    def _boom(
        job: q.TranscribeJob, should_cancel: Callable[[], bool]
    ) -> q.TranscribeResult:
        message = "boom"
        raise RuntimeError(message)

    result = process_claimed_job(tmp_path, _job(), _boom, clock=lambda: 0.0)
    assert result.status == "error"
    assert result.error_kind == "RuntimeError"


def test_process_claimed_job_deadline_is_cancelled(tmp_path: Path) -> None:
    """A job past its deadline is cancelled before any work starts."""
    calls = {"n": 0}

    def _fn(
        job: q.TranscribeJob, should_cancel: Callable[[], bool]
    ) -> q.TranscribeResult:
        calls["n"] += 1
        return q.TranscribeResult(job.job_id, "ok", "x", None, None, 1.0)

    result = process_claimed_job(
        tmp_path, _job(deadline_at=50.0), _fn, clock=lambda: 100.0
    )
    assert result.status == "cancelled"
    assert calls["n"] == 0


def test_run_daemon_exits_when_already_running(tmp_path: Path) -> None:
    """A second daemon exits immediately while the first is alive."""
    q.ensure_layout(tmp_path)
    (tmp_path / "control" / "daemon.lock").write_text("123", encoding="utf-8")
    q.write_heartbeat(tmp_path, {"model": "small"})
    calls = {"n": 0}

    def _fn(
        job: q.TranscribeJob, should_cancel: Callable[[], bool]
    ) -> q.TranscribeResult:
        calls["n"] += 1
        return q.TranscribeResult(job.job_id, "ok", "x", None, None, 1.0)

    code = run_daemon(tmp_path, config=DaemonConfig(), transcribe_fn=_fn)
    assert code == 0
    assert calls["n"] == 0


def test_run_daemon_processes_queued_job(tmp_path: Path) -> None:
    """The daemon claims, processes, and publishes a queued job, then stops."""
    q.submit_job(tmp_path, _job("queued"))
    stop = threading.Event()

    def _fn(
        job: q.TranscribeJob, should_cancel: Callable[[], bool]
    ) -> q.TranscribeResult:
        result = q.TranscribeResult(job.job_id, "ok", "done", None, None, 1.0)
        stop.set()
        return result

    code = run_daemon(
        tmp_path,
        config=DaemonConfig(poll_interval_s=0.01, idle_shutdown_s=999.0),
        transcribe_fn=_fn,
        stop_event=stop,
    )
    assert code == 0
    stored = q.read_result(tmp_path, "queued")
    assert stored is not None
    assert stored.status == "ok"
    assert stored.text == "done"


def test_run_daemon_idle_shutdown_and_tidies_up(tmp_path: Path) -> None:
    """An idle daemon shuts itself down and clears its heartbeat."""
    code = run_daemon(
        tmp_path,
        config=DaemonConfig(
            poll_interval_s=0.01, heartbeat_interval_s=0.01, idle_shutdown_s=0.05
        ),
        transcribe_fn=_ok_fn("unused"),
    )
    assert code == 0
    assert q.read_heartbeat(tmp_path) is None
    assert not (tmp_path / "control" / "daemon.lock").exists()
