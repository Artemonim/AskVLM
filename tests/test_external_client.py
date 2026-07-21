"""Unit tests for the transcription queue client orchestration."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest

from core import external_client as ec
from core import external_queue as q

if TYPE_CHECKING:
    from pathlib import Path


def _request(tmp_path: Path, *, client_timeout_s: float = 5.0) -> ec.ClientRequest:
    """Build a client request rooted at *tmp_path*.

    Args:
        tmp_path: Base directory for the input media file.
        client_timeout_s: How long the client waits for a transcript.

    Returns:
        A populated :class:`~core.external_client.ClientRequest`.

    """
    media = tmp_path / "clip.wav"
    media.write_text("stub", encoding="utf-8")
    return ec.ClientRequest(
        input_path=media,
        language=None,
        device="cpu",
        compute_type="auto",
        whisper_model="small",
        client_timeout_s=client_timeout_s,
        daemon_workers=1,
    )


def _write_results(
    queue_dir: Path, status: q.ResultStatus, text: str, detail: str | None
) -> None:
    """Publish a result for every currently-incoming job.

    Args:
        queue_dir: The queue root directory.
        status: Terminal status to publish.
        text: Transcript text to publish.
        detail: Optional error detail to publish.

    """
    for job_id in q.list_incoming_job_ids(queue_dir):
        q.write_result(
            queue_dir,
            q.TranscribeResult(job_id, status, text, None, detail, 1.0),
        )


def test_run_client_transcribe_ok(tmp_path: Path) -> None:
    """A published ``ok`` result returns the transcript text."""

    def _spawn() -> None:
        q.write_heartbeat(tmp_path, {"model": "small"})

    def _sleeper(_seconds: float) -> None:
        _write_results(tmp_path, "ok", "hello", None)

    outcome = ec.run_client_transcribe(
        _request(tmp_path),
        queue_dir=tmp_path,
        spawn=_spawn,
        sleeper=_sleeper,
        poll_interval_s=0.0,
    )
    assert outcome.status == "ok"
    assert outcome.text == "hello"


def test_run_client_transcribe_empty(tmp_path: Path) -> None:
    """An ``empty`` result returns the empty status with no text."""

    def _spawn() -> None:
        q.write_heartbeat(tmp_path, {"model": "small"})

    def _sleeper(_seconds: float) -> None:
        _write_results(tmp_path, "empty", "", None)

    outcome = ec.run_client_transcribe(
        _request(tmp_path),
        queue_dir=tmp_path,
        spawn=_spawn,
        sleeper=_sleeper,
        poll_interval_s=0.0,
    )
    assert outcome.status == "empty"
    assert outcome.text == ""


def test_run_client_transcribe_error(tmp_path: Path) -> None:
    """An ``error`` result surfaces as an error outcome with detail."""

    def _spawn() -> None:
        q.write_heartbeat(tmp_path, {"model": "small"})

    def _sleeper(_seconds: float) -> None:
        _write_results(tmp_path, "error", "", "bad input")

    outcome = ec.run_client_transcribe(
        _request(tmp_path),
        queue_dir=tmp_path,
        spawn=_spawn,
        sleeper=_sleeper,
        poll_interval_s=0.0,
    )
    assert outcome.status == "error"
    assert outcome.detail == "bad input"


def test_run_client_transcribe_timeout_signals_cancel(tmp_path: Path) -> None:
    """Giving up writes a cancel marker and returns a timeout outcome."""

    def _spawn() -> None:
        q.write_heartbeat(tmp_path, {"model": "small"})

    outcome = ec.run_client_transcribe(
        _request(tmp_path, client_timeout_s=0.02),
        queue_dir=tmp_path,
        spawn=_spawn,
        sleeper=lambda _seconds: None,
        poll_interval_s=0.0,
    )
    assert outcome.status == "timeout"
    assert any((tmp_path / "cancel").iterdir())


def test_run_client_transcribe_unavailable(tmp_path: Path) -> None:
    """When no daemon comes up, the client reports it as unavailable."""
    clock_value = {"t": 0.0}

    def _clock() -> float:
        return clock_value["t"]

    def _sleeper(_seconds: float) -> None:
        clock_value["t"] += 5.0

    outcome = ec.run_client_transcribe(
        _request(tmp_path),
        queue_dir=tmp_path,
        spawn=lambda: None,
        clock=_clock,
        sleeper=_sleeper,
    )
    assert outcome.status == "unavailable"


def test_ensure_daemon_running_already_alive(tmp_path: Path) -> None:
    """A live daemon means no spawn is attempted."""
    q.write_heartbeat(tmp_path, {"model": "small"})
    spawned = {"n": 0}

    def _spawn() -> None:
        spawned["n"] += 1

    assert ec.ensure_daemon_running(tmp_path, _request(tmp_path), spawn=_spawn)
    assert spawned["n"] == 0


def test_ensure_daemon_running_spawns_and_detects(tmp_path: Path) -> None:
    """A spawned daemon that publishes a heartbeat is detected as alive."""
    spawned = {"n": 0}

    def _spawn() -> None:
        spawned["n"] += 1
        q.write_heartbeat(tmp_path, {"model": "small"})

    assert ec.ensure_daemon_running(tmp_path, _request(tmp_path), spawn=_spawn)
    assert spawned["n"] == 1


def test_ensure_daemon_running_gives_up(tmp_path: Path) -> None:
    """A daemon that never appears makes the helper give up."""
    clock_value = {"t": 0.0}

    def _clock() -> float:
        return clock_value["t"]

    def _sleeper(_seconds: float) -> None:
        clock_value["t"] += 5.0

    assert not ec.ensure_daemon_running(
        tmp_path,
        _request(tmp_path),
        spawn=lambda: None,
        clock=_clock,
        sleeper=_sleeper,
        start_timeout_s=20.0,
    )


@pytest.mark.parametrize(
    ("daemon_provider", "requested_provider"),
    [
        ("whisper", "gigaam-ctc"),
        ("gigaam-ctc", "whisper"),
    ],
)
def test_run_client_transcribe_provider_mismatch_is_unavailable(
    tmp_path: Path,
    daemon_provider: str,
    requested_provider: str,
) -> None:
    """A live daemon with a different STT provider is an explicit mismatch."""
    now = time.time()
    q.ensure_layout(tmp_path)
    q.write_heartbeat(
        tmp_path,
        {"model": "small", "device": "cpu", "stt_provider": daemon_provider},
    )
    media = tmp_path / "clip.wav"
    media.write_text("stub", encoding="utf-8")
    request = ec.ClientRequest(
        input_path=media,
        language=None,
        device="cpu",
        compute_type="auto",
        whisper_model="small",
        client_timeout_s=5.0,
        daemon_workers=1,
        stt_provider=requested_provider,
    )

    spawned = {"n": 0}

    def _spawn() -> None:
        spawned["n"] += 1

    outcome = ec.run_client_transcribe(
        request,
        queue_dir=tmp_path,
        spawn=_spawn,
        clock=lambda: now,
        sleeper=lambda _s: None,
        poll_interval_s=0.0,
    )
    assert outcome.status == "unavailable"
    assert "ASKVLM_DAEMON_PROVIDER_MISMATCH" in (outcome.detail or "")
    assert daemon_provider in (outcome.detail or "")
    assert requested_provider in (outcome.detail or "")
    assert spawned["n"] == 0
    assert q.list_incoming_job_ids(tmp_path) == []


def test_build_daemon_command_includes_stt_provider(tmp_path: Path) -> None:
    """The daemon spawn command carries ``--stt-provider``."""
    request = ec.ClientRequest(
        input_path=tmp_path / "a.wav",
        language=None,
        device="cpu",
        compute_type="auto",
        whisper_model="small",
        client_timeout_s=10.0,
        daemon_workers=1,
        stt_provider="gigaam-ctc",
    )
    command = ec._build_daemon_command(request, tmp_path)  # noqa: SLF001
    assert "--stt-provider" in command
    assert "gigaam-ctc" in command


def test_build_daemon_command_includes_options(tmp_path: Path) -> None:
    """The daemon command carries model, device, and queue settings."""
    request = ec.ClientRequest(
        input_path=tmp_path / "a.wav",
        language="ru",
        device="cuda",
        compute_type="auto",
        whisper_model="small",
        client_timeout_s=10.0,
        daemon_workers=2,
    )
    command = ec._build_daemon_command(request, tmp_path)  # noqa: SLF001
    assert "external-transcribe-daemon" in command
    assert "--workers" in command
    assert "2" in command
    assert "--language" in command
    assert "ru" in command
    assert "--stt-provider" in command
    assert command[command.index("--stt-provider") + 1] == "whisper"


def test_spawn_detached_daemon_invokes_popen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spawning launches a detached subprocess with the daemon command."""
    captured: dict[str, object] = {}

    class _FakePopen:
        def __init__(self, command: list[str], **kwargs: object) -> None:
            captured["command"] = command
            captured["kwargs"] = kwargs

    monkeypatch.setattr(ec.subprocess, "Popen", _FakePopen)
    ec._spawn_detached_daemon(_request(tmp_path), tmp_path)  # noqa: SLF001
    command = captured["command"]
    assert isinstance(command, list)
    assert "external-transcribe-daemon" in command


def test_spawn_detached_daemon_uses_headless_windows_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows daemon spawn uses hidden-window flags instead of DETACHED_PROCESS."""
    captured: dict[str, object] = {}
    create_no_window = 0x08000000
    create_new_process_group = 0x00000200
    detached_process = 0x00000008

    class _FakePopen:
        def __init__(self, command: list[str], **kwargs: object) -> None:
            captured["command"] = command
            captured["kwargs"] = kwargs

    monkeypatch.setattr(ec, "is_windows", lambda: True)
    monkeypatch.setattr(
        ec.subprocess, "CREATE_NO_WINDOW", create_no_window, raising=False
    )
    monkeypatch.setattr(
        ec.subprocess,
        "CREATE_NEW_PROCESS_GROUP",
        create_new_process_group,
        raising=False,
    )
    monkeypatch.setattr(
        ec.subprocess, "DETACHED_PROCESS", detached_process, raising=False
    )
    monkeypatch.setattr(ec.subprocess, "Popen", _FakePopen)

    ec._spawn_detached_daemon(_request(tmp_path), tmp_path)  # noqa: SLF001

    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    creationflags = kwargs["creationflags"]
    assert isinstance(creationflags, int)
    assert creationflags & create_no_window
    assert creationflags & create_new_process_group
    assert not (creationflags & detached_process)


def test_heartbeat_stale_seconds_is_positive() -> None:
    """The exported staleness window is a positive number of seconds."""
    assert ec.heartbeat_stale_seconds() > 0
