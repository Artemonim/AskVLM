"""File-based job queue protocol for the single-orchestrator transcription daemon.

This module defines the on-disk contract shared by the ``external-transcribe``
client and the ``external-transcribe-daemon`` orchestrator. It deliberately avoids
any heavy ML imports so the protocol can be unit-tested without a GPU or models.

Layout under the queue root (default ``<project>/.cache/external_queue``)::

    incoming/   <job_id>.json   submitted jobs awaiting a worker
    claimed/    <job_id>.json   jobs a worker has taken
    results/    <job_id>.json   completed job outcomes
    cancel/     <job_id>        drop markers requesting cancellation
    control/    daemon.lock      singleton lock owned by the live daemon
                daemon.heartbeat freshness beacon written by the live daemon

All writes are atomic (temporary file + :func:`os.replace`) so a reader never
observes a partially written JSON document. The protocol is intentionally a
single writer (one daemon) and many short-lived clients.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, Literal, Self

from core.settings import get_project_cache_dir

# * Override the queue root for tests or multi-tenant deployments.
ENV_QUEUE_DIR = "ASKVLM_EXTERNAL_QUEUE_DIR"

_INCOMING_DIR = "incoming"
_CLAIMED_DIR = "claimed"
_RESULTS_DIR = "results"
_CANCEL_DIR = "cancel"
_CONTROL_DIR = "control"

_HEARTBEAT_NAME = "daemon.heartbeat"
_LOCK_NAME = "daemon.lock"

# * A daemon is considered dead when its heartbeat is older than this many
# * seconds. Clients use it to decide whether to spawn a fresh daemon.
DEFAULT_HEARTBEAT_STALE_SECONDS = 30.0

ResultStatus = Literal["ok", "empty", "error", "cancelled", "expired"]


@dataclass(frozen=True, slots=True)
class TranscribeJob:
    """One transcription request placed on the queue by a client.

    Attributes:
        job_id: Unique identifier (hex) used for all per-job files.
        input_path: Absolute path to the media file to transcribe.
        language: Optional language hint (for example ``ru``) or ``None``.
        device: Preferred device requested by the client (advisory once a
            daemon with a resident model is already running).
        compute_type: Preferred faster-whisper compute type (advisory).
        whisper_model: Preferred Whisper model name (advisory).
        diarization: Whether speaker diarization was requested.
        dialog_blocks: Whether LLM dialog formatting was requested.
        submitted_at: Unix timestamp when the job was submitted.
        deadline_at: Unix timestamp after which the job may be dropped, or
            ``None`` for no deadline.
        stt_provider: STT backend id (``whisper`` or ``gigaam-ctc``). Missing
            values in older queued jobs default to Whisper.

    """

    job_id: str
    input_path: str
    language: str | None
    device: str
    compute_type: str
    whisper_model: str
    diarization: bool
    dialog_blocks: bool
    submitted_at: float
    deadline_at: float | None
    stt_provider: str = "whisper"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping for this job.

        Returns:
            A plain dictionary mirroring the dataclass fields.

        """
        return {
            "job_id": self.job_id,
            "input_path": self.input_path,
            "language": self.language,
            "device": self.device,
            "compute_type": self.compute_type,
            "whisper_model": self.whisper_model,
            "diarization": self.diarization,
            "dialog_blocks": self.dialog_blocks,
            "submitted_at": self.submitted_at,
            "deadline_at": self.deadline_at,
            "stt_provider": self.stt_provider,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TranscribeJob:
        """Build a job from a previously serialized mapping.

        Args:
            payload: Mapping produced by :meth:`to_dict`.

        Returns:
            The reconstructed :class:`TranscribeJob`.

        """
        return cls(
            job_id=str(payload["job_id"]),
            input_path=str(payload["input_path"]),
            language=_optional_str(payload.get("language")),
            device=str(payload.get("device", "auto")),
            compute_type=str(payload.get("compute_type", "auto")),
            whisper_model=str(payload.get("whisper_model", "small")),
            diarization=bool(payload.get("diarization", False)),
            dialog_blocks=bool(payload.get("dialog_blocks", False)),
            submitted_at=float(payload.get("submitted_at", 0.0)),
            deadline_at=_optional_float(payload.get("deadline_at")),
            # * Legacy queued jobs without a provider keep Whisper semantics.
            stt_provider=str(payload.get("stt_provider", "whisper")),
        )


@dataclass(frozen=True, slots=True)
class TranscribeResult:
    """The outcome of one transcription job written by the daemon.

    Attributes:
        job_id: Identifier of the job this result belongs to.
        status: Terminal status of the job.
        text: Transcript text (empty unless ``status`` is ``ok``).
        error_kind: Short machine label when ``status`` is ``error``.
        error_detail: Human-readable error message when ``status`` is ``error``.
        finished_at: Unix timestamp when the daemon finished the job.

    """

    job_id: str
    status: ResultStatus
    text: str
    error_kind: str | None
    error_detail: str | None
    finished_at: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping for this result.

        Returns:
            A plain dictionary mirroring the dataclass fields.

        """
        return {
            "job_id": self.job_id,
            "status": self.status,
            "text": self.text,
            "error_kind": self.error_kind,
            "error_detail": self.error_detail,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TranscribeResult:
        """Build a result from a previously serialized mapping.

        Args:
            payload: Mapping produced by :meth:`to_dict`.

        Returns:
            The reconstructed :class:`TranscribeResult`.

        """
        status = str(payload.get("status", "error"))
        normalized: ResultStatus = (
            status  # type: ignore[assignment]
            if status in {"ok", "empty", "error", "cancelled", "expired"}
            else "error"
        )
        return cls(
            job_id=str(payload["job_id"]),
            status=normalized,
            text=str(payload.get("text", "")),
            error_kind=_optional_str(payload.get("error_kind")),
            error_detail=_optional_str(payload.get("error_detail")),
            finished_at=float(payload.get("finished_at", 0.0)),
        )


def _optional_str(value: Any) -> str | None:  # noqa: ANN401 - JSON boundary value
    """Coerce a JSON value to ``str`` or ``None``.

    Args:
        value: Arbitrary deserialized JSON value.

    Returns:
        ``None`` when *value* is ``None``, otherwise its string form.

    """
    return None if value is None else str(value)


def _optional_float(value: Any) -> float | None:  # noqa: ANN401 - JSON boundary value
    """Coerce a JSON value to ``float`` or ``None``.

    Args:
        value: Arbitrary deserialized JSON value.

    Returns:
        ``None`` when *value* is ``None``, otherwise its float form.

    """
    return None if value is None else float(value)


def new_job_id() -> str:
    """Return a fresh queue-safe job identifier.

    Returns:
        A 32-character hexadecimal string.

    """
    return uuid.uuid4().hex


def resolve_queue_dir(explicit: Path | None = None) -> Path:
    """Return the queue root directory.

    Resolution order: an explicit argument, then the :data:`ENV_QUEUE_DIR`
    environment variable, then ``<project>/.cache/external_queue``.

    Args:
        explicit: Optional caller-provided queue root.

    Returns:
        The resolved queue root path (not guaranteed to exist yet).

    """
    if explicit is not None:
        return explicit.resolve()
    env_value = os.environ.get(ENV_QUEUE_DIR)
    if env_value:
        return Path(env_value).resolve()
    return (get_project_cache_dir() / "external_queue").resolve()


def ensure_layout(queue_dir: Path) -> None:
    """Create all queue subdirectories if they are missing.

    Args:
        queue_dir: The queue root directory.

    """
    for name in (
        _INCOMING_DIR,
        _CLAIMED_DIR,
        _RESULTS_DIR,
        _CANCEL_DIR,
        _CONTROL_DIR,
    ):
        (queue_dir / name).mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write *payload* as JSON to *path* atomically.

    Args:
        path: Destination file path.
        payload: JSON-serializable mapping.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f"{path.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON mapping from *path*, tolerating missing or partial files.

    Args:
        path: Source file path.

    Returns:
        The parsed mapping, or ``None`` when the file is absent or not yet a
        complete JSON object.

    """
    try:
        raw = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _job_path(queue_dir: Path, subdir: str, job_id: str) -> Path:
    """Return the path of a per-job JSON file in *subdir*.

    Args:
        queue_dir: The queue root directory.
        subdir: One of the queue subdirectory names.
        job_id: The job identifier.

    Returns:
        The composed file path.

    """
    return queue_dir / subdir / f"{job_id}.json"


def submit_job(queue_dir: Path, job: TranscribeJob) -> None:
    """Place *job* into the incoming queue.

    Args:
        queue_dir: The queue root directory.
        job: The job to enqueue.

    """
    ensure_layout(queue_dir)
    _atomic_write_json(_job_path(queue_dir, _INCOMING_DIR, job.job_id), job.to_dict())


def list_incoming_job_ids(queue_dir: Path) -> list[str]:
    """Return incoming job ids ordered oldest-submitted first.

    Args:
        queue_dir: The queue root directory.

    Returns:
        Job identifiers sorted by submission time (falling back to filename).

    """
    incoming = queue_dir / _INCOMING_DIR
    if not incoming.is_dir():
        return []
    jobs: list[tuple[float, str]] = []
    for path in incoming.glob("*.json"):
        payload = _read_json(path)
        submitted = float(payload.get("submitted_at", 0.0)) if payload else 0.0
        jobs.append((submitted, path.stem))
    jobs.sort(key=lambda item: (item[0], item[1]))
    return [job_id for _, job_id in jobs]


def claim_job(queue_dir: Path, job_id: str) -> TranscribeJob | None:
    """Atomically move a job from incoming to claimed and return it.

    Args:
        queue_dir: The queue root directory.
        job_id: The job to claim.

    Returns:
        The claimed job, or ``None`` if it vanished before it could be claimed.

    """
    incoming = _job_path(queue_dir, _INCOMING_DIR, job_id)
    claimed = _job_path(queue_dir, _CLAIMED_DIR, job_id)
    claimed.parent.mkdir(parents=True, exist_ok=True)
    try:
        incoming.replace(claimed)
    except (FileNotFoundError, OSError):
        return None
    payload = _read_json(claimed)
    return TranscribeJob.from_dict(payload) if payload else None


def write_result(queue_dir: Path, result: TranscribeResult) -> None:
    """Publish a completed job result.

    Args:
        queue_dir: The queue root directory.
        result: The result to publish.

    """
    ensure_layout(queue_dir)
    _atomic_write_json(
        _job_path(queue_dir, _RESULTS_DIR, result.job_id), result.to_dict()
    )


def read_result(queue_dir: Path, job_id: str) -> TranscribeResult | None:
    """Read a job result if it has been published.

    Args:
        queue_dir: The queue root directory.
        job_id: The job whose result is requested.

    Returns:
        The result, or ``None`` when it is not yet available.

    """
    payload = _read_json(_job_path(queue_dir, _RESULTS_DIR, job_id))
    return TranscribeResult.from_dict(payload) if payload else None


def request_cancel(queue_dir: Path, job_id: str) -> None:
    """Write a cancellation marker for *job_id*.

    The daemon checks for the marker before starting a job and cooperatively
    while it runs, so a client that gives up does not leave the daemon doing
    wasted work.

    Args:
        queue_dir: The queue root directory.
        job_id: The job to cancel.

    """
    cancel_dir = queue_dir / _CANCEL_DIR
    cancel_dir.mkdir(parents=True, exist_ok=True)
    marker = cancel_dir / job_id
    marker.touch(exist_ok=True)


def is_cancelled(queue_dir: Path, job_id: str) -> bool:
    """Return whether a cancellation marker exists for *job_id*.

    Args:
        queue_dir: The queue root directory.
        job_id: The job to check.

    Returns:
        ``True`` when the job has been marked for cancellation.

    """
    return (queue_dir / _CANCEL_DIR / job_id).exists()


def cleanup_job(queue_dir: Path, job_id: str) -> None:
    """Remove all per-job files for a finished and consumed job.

    Args:
        queue_dir: The queue root directory.
        job_id: The job to clean up.

    """
    for path in (
        _job_path(queue_dir, _INCOMING_DIR, job_id),
        _job_path(queue_dir, _CLAIMED_DIR, job_id),
        _job_path(queue_dir, _RESULTS_DIR, job_id),
        queue_dir / _CANCEL_DIR / job_id,
    ):
        path.unlink(missing_ok=True)


def purge_expired_incoming(queue_dir: Path, now: float) -> list[str]:
    """Drop incoming jobs whose deadline has passed and report their ids.

    This prevents a backlog of abandoned jobs (for example after a client-side
    timeout cascade) from clogging the queue.

    Args:
        queue_dir: The queue root directory.
        now: The current Unix timestamp.

    Returns:
        Identifiers of the jobs that were purged.

    """
    incoming = queue_dir / _INCOMING_DIR
    if not incoming.is_dir():
        return []
    purged: list[str] = []
    for path in incoming.glob("*.json"):
        payload = _read_json(path)
        if payload is None:
            continue
        deadline = _optional_float(payload.get("deadline_at"))
        if deadline is not None and deadline < now:
            path.unlink(missing_ok=True)
            (queue_dir / _CANCEL_DIR / path.stem).unlink(missing_ok=True)
            purged.append(path.stem)
    return purged


def purge_stale_results(queue_dir: Path, now: float, *, ttl_s: float) -> list[str]:
    """Remove result files (and their job siblings) unread past *ttl_s*.

    A client that timed out never consumes its result; this reclaims those plus
    any orphaned per-job files so the queue cannot grow without bound.

    Args:
        queue_dir: The queue root directory.
        now: The current Unix timestamp.
        ttl_s: Maximum age in seconds before a result is reclaimed.

    Returns:
        Identifiers of the results that were purged.

    """
    results = queue_dir / _RESULTS_DIR
    if not results.is_dir():
        return []
    purged: list[str] = []
    for path in results.glob("*.json"):
        try:
            age = now - path.stat().st_mtime
        except OSError:
            continue
        if age > ttl_s:
            cleanup_job(queue_dir, path.stem)
            purged.append(path.stem)
    return purged


def write_heartbeat(queue_dir: Path, payload: dict[str, Any]) -> None:
    """Write the daemon liveness beacon.

    Args:
        queue_dir: The queue root directory.
        payload: Diagnostic fields; ``ts`` is overwritten with the current time.

    """
    control = queue_dir / _CONTROL_DIR
    control.mkdir(parents=True, exist_ok=True)
    beacon = dict(payload)
    beacon["ts"] = time()
    beacon["pid"] = os.getpid()
    _atomic_write_json(control / _HEARTBEAT_NAME, beacon)


def read_heartbeat(queue_dir: Path) -> dict[str, Any] | None:
    """Read the daemon liveness beacon if present.

    Args:
        queue_dir: The queue root directory.

    Returns:
        The beacon mapping, or ``None`` when no daemon has written one.

    """
    return _read_json(queue_dir / _CONTROL_DIR / _HEARTBEAT_NAME)


def clear_heartbeat(queue_dir: Path) -> None:
    """Remove the daemon liveness beacon so clients see it as gone at once.

    Args:
        queue_dir: The queue root directory.

    """
    (queue_dir / _CONTROL_DIR / _HEARTBEAT_NAME).unlink(missing_ok=True)


def is_daemon_alive(
    queue_dir: Path,
    *,
    now: float,
    stale_after_s: float = DEFAULT_HEARTBEAT_STALE_SECONDS,
) -> bool:
    """Return whether a live daemon is currently serving the queue.

    Args:
        queue_dir: The queue root directory.
        now: The current Unix timestamp.
        stale_after_s: Maximum heartbeat age considered alive.

    Returns:
        ``True`` when a fresh heartbeat exists.

    """
    beacon = read_heartbeat(queue_dir)
    if beacon is None:
        return False
    ts = _optional_float(beacon.get("ts"))
    if ts is None:
        return False
    return (now - ts) <= stale_after_s


class DaemonLock:
    """Best-effort single-writer lock for the transcription daemon.

    It combines an ``O_EXCL`` lock file with heartbeat staleness recovery: a
    daemon that crashed without releasing the lock does not permanently block a
    fresh start, because a stale lock (no fresh heartbeat) is reclaimed.
    """

    def __init__(
        self,
        queue_dir: Path,
        *,
        stale_after_s: float = DEFAULT_HEARTBEAT_STALE_SECONDS,
    ) -> None:
        """Initialize the lock for a queue root.

        Args:
            queue_dir: The queue root directory.
            stale_after_s: Heartbeat age beyond which an existing lock is
                considered abandoned and may be reclaimed.

        """
        self._queue_dir = queue_dir
        self._stale_after_s = stale_after_s
        self._lock_path = queue_dir / _CONTROL_DIR / _LOCK_NAME
        self._fd: int | None = None

    def _open_exclusive(self) -> bool:
        """Try to create the lock file exclusively.

        Returns:
            ``True`` when this process created and now owns the lock file.

        """
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        os.write(fd, str(os.getpid()).encode("ascii"))
        self._fd = fd
        return True

    def try_acquire(self) -> bool:
        """Attempt to acquire the lock, reclaiming a stale one if needed.

        Returns:
            ``True`` when the lock is now held by this process.

        """
        if self._open_exclusive():
            return True
        if is_daemon_alive(
            self._queue_dir, now=time(), stale_after_s=self._stale_after_s
        ):
            return False
        # * No fresh heartbeat: the previous owner is gone. Reclaim the lock.
        self._lock_path.unlink(missing_ok=True)
        return self._open_exclusive()

    def release(self) -> None:
        """Release the lock and remove the lock file if owned."""
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
            self._lock_path.unlink(missing_ok=True)

    def __enter__(self) -> Self:
        """Enter the lock context.

        Returns:
            This lock instance.

        """
        return self

    def __exit__(self, *_exc: object) -> None:
        """Exit the lock context, releasing the lock."""
        self.release()


def is_windows() -> bool:
    """Return whether the current platform is Windows.

    Returns:
        ``True`` on Windows, otherwise ``False``.

    """
    return sys.platform == "win32"
