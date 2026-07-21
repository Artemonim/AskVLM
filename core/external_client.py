"""Client side of the single-orchestrator transcription queue.

A client is short-lived: it makes sure a daemon is running, submits one job,
waits for the result up to a deadline, and — crucially — sends a cancellation
signal if it gives up, so an abandoned client never leaves the daemon doing
wasted work or lets the queue accumulate stale jobs.

The orchestration here performs no transcription itself and imports no ML
packages, so it is fully unit-testable with an injected spawner and clock.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import sleep, time
from typing import Literal

from core.external_queue import (
    DEFAULT_HEARTBEAT_STALE_SECONDS,
    TranscribeJob,
    cleanup_job,
    ensure_layout,
    is_daemon_alive,
    is_windows,
    new_job_id,
    read_heartbeat,
    read_result,
    request_cancel,
    resolve_queue_dir,
    submit_job,
)
from core.stt_providers import STT_PROVIDER_WHISPER, normalize_stt_provider

ClientStatus = Literal["ok", "empty", "error", "timeout", "unavailable"]

# * Maximum time to wait for a freshly spawned daemon to publish a heartbeat.
_DAEMON_START_TIMEOUT_SECONDS = 30.0
_POLL_INTERVAL_SECONDS = 0.25
_PROVIDER_MISMATCH_MARKER = "ASKVLM_DAEMON_PROVIDER_MISMATCH"

SpawnFn = Callable[[], None]


@dataclass(frozen=True, slots=True)
class ClientRequest:
    """A single client transcription request.

    Attributes:
        input_path: Absolute path to the media file to transcribe.
        language: Optional language hint, or ``None`` to auto-detect.
        device: Preferred device for the resident model (advisory).
        compute_type: Preferred faster-whisper compute type (advisory).
        whisper_model: Preferred Whisper model name (advisory).
        client_timeout_s: How long the client waits before giving up.
        daemon_workers: Worker count to request when spawning a new daemon.
        stt_provider: STT backend id (``whisper`` default, or ``gigaam-ctc``).

    """

    input_path: Path
    language: str | None
    device: str
    compute_type: str
    whisper_model: str
    client_timeout_s: float
    daemon_workers: int
    stt_provider: str = STT_PROVIDER_WHISPER


@dataclass(frozen=True, slots=True)
class ClientOutcome:
    """Result of a client transcription attempt.

    Attributes:
        status: Terminal client status.
        text: Transcript text (only meaningful when ``status`` is ``ok``).
        detail: Optional diagnostic message for non-success statuses.

    """

    status: ClientStatus
    text: str
    detail: str | None


def _resolve_cli_script() -> Path:
    """Return the absolute path to this project's ``cli.py``.

    Returns:
        The resolved ``cli.py`` path next to the project root.

    """
    return Path(__file__).resolve().parents[1] / "cli.py"


def _build_daemon_command(request: ClientRequest, queue_dir: Path) -> list[str]:
    """Build the command line that launches the daemon for *request*.

    Args:
        request: The client request whose model/device seed the daemon.
        queue_dir: The queue root the daemon must serve.

    Returns:
        The argument vector for :class:`subprocess.Popen`.

    """
    command = [
        sys.executable,
        str(_resolve_cli_script()),
        "external-transcribe-daemon",
        "--workers",
        str(request.daemon_workers),
        "--stt-provider",
        normalize_stt_provider(request.stt_provider),
        "--whisper-model",
        request.whisper_model,
        "--device",
        request.device,
        "--compute-type",
        request.compute_type,
        "--queue-dir",
        str(queue_dir),
    ]
    if request.language:
        command.extend(["--language", request.language])
    return command


def _daemon_provider(queue_dir: Path) -> str | None:
    """Return the STT provider advertised by a live daemon heartbeat.

    Args:
        queue_dir: The queue root directory.

    Returns:
        Provider id from the heartbeat, defaulting to Whisper when the field is
        absent (legacy daemons), or ``None`` when no heartbeat exists.

    """
    beacon = read_heartbeat(queue_dir)
    if beacon is None:
        return None
    raw = beacon.get("stt_provider")
    if raw is None or not str(raw).strip():
        return STT_PROVIDER_WHISPER
    try:
        return normalize_stt_provider(str(raw))
    except ValueError:
        return str(raw)


def _provider_mismatch_detail(*, daemon_provider: str, requested: str) -> str:
    """Build a clear operational error for a live daemon with the wrong provider.

    Args:
        daemon_provider: Provider id currently resident in the daemon.
        requested: Provider id requested by the client.

    Returns:
        Human-readable mismatch detail including the restart hint.

    """
    return (
        f"{_PROVIDER_MISMATCH_MARKER}: live daemon stt_provider={daemon_provider!r} "
        f"does not match requested {requested!r}; keep the singleton daemon and "
        f"restart it with --stt-provider {requested}"
    )


def _daemon_creationflags() -> int:
    """Return subprocess creation flags for a background daemon launch.

    Returns:
        Windows-specific flags that keep the daemon headless and independent,
        or ``0`` on other platforms.

    """
    if not is_windows():
        return 0
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return flags


def _spawn_detached_daemon(request: ClientRequest, queue_dir: Path) -> None:
    """Launch the daemon as a detached, independent process.

    The daemon must outlive the spawning client (it is a shared singleton), so
    it is started as a background process with no visible console window. On
    Windows ``CREATE_NO_WINDOW`` is used instead of ``DETACHED_PROCESS`` because
    console Python interpreters can still surface a terminal window otherwise.

    Args:
        request: The client request whose model/device seed the daemon.
        queue_dir: The queue root the daemon must serve.

    """
    command = _build_daemon_command(request, queue_dir)
    log_path = queue_dir / "control" / "daemon.out.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log_handle:
        subprocess.Popen(  # noqa: S603 - fixed argv, no shell, trusted interpreter
            command,
            cwd=str(_resolve_cli_script().parent),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=_daemon_creationflags(),
            close_fds=True,
        )


def ensure_daemon_running(
    queue_dir: Path,
    request: ClientRequest,
    *,
    spawn: SpawnFn | None = None,
    clock: Callable[[], float] = time,
    sleeper: Callable[[float], None] = sleep,
    start_timeout_s: float = _DAEMON_START_TIMEOUT_SECONDS,
) -> bool:
    """Ensure a live daemon is serving *queue_dir*, spawning one if needed.

    Args:
        queue_dir: The queue root directory.
        request: The client request whose model/device seed a new daemon.
        spawn: Optional spawner callable (injected for tests). Defaults to a
            detached subprocess launch.
        clock: Wall-clock source (injected for tests).
        sleeper: Sleep function (injected for tests).
        start_timeout_s: How long to wait for a new daemon's first heartbeat.

    Returns:
        ``True`` when a live daemon is available, otherwise ``False``.

    """
    if is_daemon_alive(queue_dir, now=clock()):
        return True
    spawner = spawn or (lambda: _spawn_detached_daemon(request, queue_dir))
    spawner()
    deadline = clock() + start_timeout_s
    while clock() < deadline:
        if is_daemon_alive(queue_dir, now=clock()):
            return True
        sleeper(_POLL_INTERVAL_SECONDS)
    return is_daemon_alive(queue_dir, now=clock())


def _map_result_status(status: str, text: str, detail: str | None) -> ClientOutcome:
    """Map a daemon result status onto a client outcome.

    Args:
        status: The terminal status reported by the daemon.
        text: Transcript text from the result.
        detail: Optional error detail from the result.

    Returns:
        The corresponding :class:`ClientOutcome`.

    """
    if status == "ok":
        return ClientOutcome("ok", text, None)
    if status == "empty":
        return ClientOutcome("empty", "", None)
    # cancelled / expired surface as errors to a client that still waited.
    return ClientOutcome("error", "", detail or status)


def run_client_transcribe(
    request: ClientRequest,
    *,
    queue_dir: Path | None = None,
    spawn: SpawnFn | None = None,
    clock: Callable[[], float] = time,
    sleeper: Callable[[float], None] = sleep,
    poll_interval_s: float = _POLL_INTERVAL_SECONDS,
) -> ClientOutcome:
    """Submit *request* to the daemon and wait for its transcript.

    Args:
        request: The transcription request to run.
        queue_dir: Optional explicit queue root; defaults to the resolved root.
        spawn: Optional spawner callable (injected for tests).
        clock: Wall-clock source (injected for tests).
        sleeper: Sleep function (injected for tests).
        poll_interval_s: Delay between result polls.

    Returns:
        The terminal :class:`ClientOutcome` for the request.

    """
    resolved = resolve_queue_dir(queue_dir)
    ensure_layout(resolved)
    requested_provider = normalize_stt_provider(request.stt_provider)
    if is_daemon_alive(resolved, now=clock()):
        live_provider = _daemon_provider(resolved) or STT_PROVIDER_WHISPER
        if live_provider != requested_provider:
            # * Do not submit to the wrong resident model; keep the singleton.
            return ClientOutcome(
                "unavailable",
                "",
                _provider_mismatch_detail(
                    daemon_provider=live_provider,
                    requested=requested_provider,
                ),
            )
    elif not ensure_daemon_running(
        resolved, request, spawn=spawn, clock=clock, sleeper=sleeper
    ):
        return ClientOutcome("unavailable", "", "transcription daemon unavailable")

    job_id = new_job_id()
    submitted_at = clock()
    deadline_at = submitted_at + request.client_timeout_s
    submit_job(
        resolved,
        TranscribeJob(
            job_id=job_id,
            input_path=str(request.input_path.resolve()),
            language=request.language,
            device=request.device,
            compute_type=request.compute_type,
            whisper_model=request.whisper_model,
            diarization=False,
            dialog_blocks=False,
            submitted_at=submitted_at,
            deadline_at=deadline_at,
            stt_provider=requested_provider,
        ),
    )

    while clock() < deadline_at:
        result = read_result(resolved, job_id)
        if result is not None:
            cleanup_job(resolved, job_id)
            return _map_result_status(result.status, result.text, result.error_detail)
        sleeper(poll_interval_s)

    # * Timed out: signal a drop so the daemon does not waste work on this job.
    request_cancel(resolved, job_id)
    return ClientOutcome("timeout", "", "client timeout waiting for transcript")


def heartbeat_stale_seconds() -> float:
    """Return the configured heartbeat staleness threshold.

    Returns:
        The default heartbeat staleness window in seconds.

    """
    return DEFAULT_HEARTBEAT_STALE_SECONDS
