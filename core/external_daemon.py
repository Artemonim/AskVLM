"""Single-orchestrator daemon that drains the external transcription queue.

One daemon process per machine owns the resident Whisper model(s) and serves all
``external-transcribe`` clients through the file-based queue defined in
:mod:`core.external_queue`. This eliminates the previous "cold-load a model per
message" pattern (repeated multi-gigabyte disk reads) and the orphaned worker
processes that a per-message subprocess left behind on timeout.

The scheduling, cancellation, and lifecycle logic here is free of ML imports and
receives the actual transcription work as an injected callable, so it can be
unit-tested without a GPU. The real model-backed callable is built lazily by
:func:`build_local_transcribe_fn`.
"""

from __future__ import annotations

import logging
import tempfile
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import sleep, time
from typing import TYPE_CHECKING

from core.external_queue import (
    DaemonLock,
    TranscribeJob,
    TranscribeResult,
    claim_job,
    clear_heartbeat,
    ensure_layout,
    is_cancelled,
    list_incoming_job_ids,
    purge_expired_incoming,
    purge_stale_results,
    resolve_queue_dir,
    write_heartbeat,
    write_result,
)

if TYPE_CHECKING:
    from concurrent.futures import Future

    from core.external_queue import ResultStatus

logger = logging.getLogger(__name__)

# * (job, should_cancel) -> result. ``should_cancel`` is polled cooperatively.
TranscribeFn = Callable[[TranscribeJob, Callable[[], bool]], TranscribeResult]


@dataclass(frozen=True, slots=True)
class DaemonConfig:
    """Tunables for the transcription daemon.

    Attributes:
        max_workers: Maximum jobs processed concurrently. Defaults to ``1`` to
            honor the project doctrine of a single active neural network in
            memory; raise it only on hardware that can host that many resident
            models at once.
        poll_interval_s: Sleep between queue scans when idle.
        heartbeat_interval_s: How often the liveness beacon is refreshed.
        idle_shutdown_s: Exit after this many seconds with no work, freeing VRAM.
        result_ttl_s: Remove published results unread for longer than this.
        whisper_model: Resident Whisper model name.
        device: Device the resident model is loaded on.
        compute_type: faster-whisper compute type.
        language: Optional fixed language hint for the resident model.

    """

    max_workers: int = 1
    poll_interval_s: float = 0.5
    heartbeat_interval_s: float = 5.0
    idle_shutdown_s: float = 600.0
    result_ttl_s: float = 300.0
    whisper_model: str = "small"
    device: str = "auto"
    compute_type: str = "auto"
    language: str | None = None


def process_claimed_job(
    queue_dir: Path,
    job: TranscribeJob,
    transcribe_fn: TranscribeFn,
    *,
    clock: Callable[[], float] = time,
) -> TranscribeResult:
    """Run one claimed job to a terminal result and publish it.

    The daemon must survive any single job failure, so every exception from
    *transcribe_fn* is captured and recorded as a terminal ``error`` (or
    ``cancelled`` when a drop/deadline was already in effect).

    Args:
        queue_dir: The queue root directory.
        job: The claimed job to process.
        transcribe_fn: Callable performing the actual transcription.
        clock: Monotonic-ish wall-clock source (injected for tests).

    Returns:
        The terminal :class:`TranscribeResult` that was published.

    """

    def should_cancel() -> bool:
        if is_cancelled(queue_dir, job.job_id):
            return True
        return job.deadline_at is not None and clock() > job.deadline_at

    if should_cancel():
        result = TranscribeResult(
            job_id=job.job_id,
            status="cancelled",
            text="",
            error_kind=None,
            error_detail=None,
            finished_at=clock(),
        )
    else:
        try:
            result = transcribe_fn(job, should_cancel)
        except Exception as exc:  # noqa: BLE001 - one job must never kill the daemon
            status: ResultStatus = "cancelled" if should_cancel() else "error"
            result = TranscribeResult(
                job_id=job.job_id,
                status=status,
                text="",
                error_kind=type(exc).__name__,
                error_detail=str(exc)[:500],
                finished_at=clock(),
            )
    write_result(queue_dir, result)
    return result


def _reap_finished(in_flight: dict[str, Future[TranscribeResult]]) -> None:
    """Drop completed futures from the in-flight map.

    Args:
        in_flight: Mapping of job id to its running future.

    """
    for job_id in [jid for jid, fut in in_flight.items() if fut.done()]:
        in_flight.pop(job_id, None)


def _claim_and_submit(
    queue_dir: Path,
    transcribe_fn: TranscribeFn,
    in_flight: dict[str, Future[TranscribeResult]],
    executor: ThreadPoolExecutor,
    capacity: int,
    clock: Callable[[], float],
) -> int:
    """Claim up to *capacity* incoming jobs and submit them to the pool.

    Args:
        queue_dir: The queue root directory.
        transcribe_fn: Callable performing the actual transcription.
        in_flight: Mapping of job id to its running future (mutated in place).
        executor: The worker thread pool.
        capacity: Maximum number of new jobs to start this pass.
        clock: Wall-clock source passed through to job processing.

    Returns:
        The number of jobs newly submitted.

    """
    submitted = 0
    for job_id in list_incoming_job_ids(queue_dir):
        if submitted >= capacity:
            break
        if job_id in in_flight:
            continue
        job = claim_job(queue_dir, job_id)
        if job is None:
            continue
        future = executor.submit(
            process_claimed_job, queue_dir, job, transcribe_fn, clock=clock
        )
        in_flight[job_id] = future
        submitted += 1
    return submitted


def _serve_loop(
    queue_dir: Path,
    config: DaemonConfig,
    transcribe_fn: TranscribeFn,
    *,
    stop_event: threading.Event,
    executor: ThreadPoolExecutor,
    clock: Callable[[], float] = time,
    sleeper: Callable[[float], None] = sleep,
) -> None:
    """Drain the queue until stopped or idle for ``idle_shutdown_s``.

    Args:
        queue_dir: The queue root directory.
        config: Daemon tunables.
        transcribe_fn: Callable performing the actual transcription.
        stop_event: Set by signal handlers to request a graceful stop.
        executor: The worker thread pool.
        clock: Wall-clock source (injected for tests).
        sleeper: Sleep function (injected for tests).

    """
    in_flight: dict[str, Future[TranscribeResult]] = {}
    last_activity = clock()
    last_heartbeat = 0.0
    while not stop_event.is_set():
        now = clock()
        if now - last_heartbeat >= config.heartbeat_interval_s:
            write_heartbeat(
                queue_dir,
                {
                    "model": config.whisper_model,
                    "device": config.device,
                    "workers": config.max_workers,
                },
            )
            last_heartbeat = now
        purge_expired_incoming(queue_dir, now)
        purge_stale_results(queue_dir, now, ttl_s=config.result_ttl_s)
        _reap_finished(in_flight)
        capacity = config.max_workers - len(in_flight)
        if capacity > 0 and _claim_and_submit(
            queue_dir, transcribe_fn, in_flight, executor, capacity, clock
        ):
            last_activity = now
        if in_flight:
            last_activity = now
        elif (now - last_activity) >= config.idle_shutdown_s:
            break
        sleeper(config.poll_interval_s)
    for future in list(in_flight.values()):
        future.result()


def run_daemon(
    queue_dir: Path | None = None,
    *,
    config: DaemonConfig | None = None,
    transcribe_fn: TranscribeFn | None = None,
    stop_event: threading.Event | None = None,
) -> int:
    """Acquire the singleton lock and serve the queue until stopped.

    Args:
        queue_dir: Optional explicit queue root; defaults to the resolved root.
        config: Optional daemon tunables; defaults to :class:`DaemonConfig`.
        transcribe_fn: Optional injected transcription callable; when omitted a
            resident model-backed callable is built lazily.
        stop_event: Optional externally owned stop flag.

    Returns:
        Process exit code: ``0`` on a clean shutdown or when another daemon
        already owns the queue.

    """
    resolved = resolve_queue_dir(queue_dir)
    resolved_config = config or DaemonConfig()
    ensure_layout(resolved)
    lock = DaemonLock(resolved)
    if not lock.try_acquire():
        logger.info("external-transcribe daemon already running: queue=%s", resolved)
        return 0
    event = stop_event or threading.Event()
    worker = transcribe_fn or build_local_transcribe_fn(resolved_config)
    logger.info(
        "external-transcribe daemon starting: queue=%s model=%s device=%s workers=%d",
        resolved,
        resolved_config.whisper_model,
        resolved_config.device,
        resolved_config.max_workers,
    )
    try:
        with ThreadPoolExecutor(
            max_workers=resolved_config.max_workers,
            thread_name_prefix="askvlm-transcribe",
        ) as executor:
            _serve_loop(
                resolved,
                resolved_config,
                worker,
                stop_event=event,
                executor=executor,
            )
    finally:
        clear_heartbeat(resolved)
        lock.release()
        logger.info("external-transcribe daemon stopped: queue=%s", resolved)
    return 0


def build_local_transcribe_fn(config: DaemonConfig) -> TranscribeFn:  # pragma: no cover
    """Build a resident model-backed transcription callable.

    Each worker thread lazily constructs and then reuses its own
    :class:`~core.pipelines.LocalPipeline`, so the model is loaded once per
    worker instead of once per message.

    Args:
        config: Daemon tunables describing the resident model.

    Returns:
        A :data:`TranscribeFn` that transcribes a job with a resident pipeline.

    """
    from core.pipelines import LocalPipeline  # noqa: PLC0415 - lazy ML import

    thread_local = threading.local()

    def _pipeline() -> LocalPipeline:
        existing: LocalPipeline | None = getattr(thread_local, "pipeline", None)
        if existing is not None:
            return existing
        pipeline = LocalPipeline(
            whisper_model=config.whisper_model,
            engine="whisperx",
            enable_diarization=False,
            enable_dialog_blocks=False,
            language=config.language,
            device=config.device,
            compute_type=config.compute_type,
        )
        thread_local.pipeline = pipeline
        return pipeline

    def _transcribe(
        job: TranscribeJob, should_cancel: Callable[[], bool]
    ) -> TranscribeResult:
        pipeline = _pipeline()
        with tempfile.TemporaryDirectory(prefix="askvlm-daemon-") as work_dir:
            document = pipeline.process(
                Path(job.input_path),
                Path(work_dir),
                should_cancel=should_cancel,
            )
            text = (document.get_full_text() or "").strip()
        return TranscribeResult(
            job_id=job.job_id,
            status="ok" if text else "empty",
            text=text,
            error_kind=None,
            error_detail=None,
            finished_at=time(),
        )

    return _transcribe
