"""Supervisor of the STT worker pool (spec section 8).

:func:`run_supervisor` is the long-lived process behind
``external-transcribe-daemon``. It owns one queue root through the OS lock,
admits v2 jobs from ``incoming/``, runs the pure scheduler of
:mod:`core.stt_pool.scheduler` every tick, starts and stops worker processes
(each in its own Job Object, see :mod:`core.stt_pool.procs`), dispatches jobs,
publishes terminal results, keeps the heartbeat, learns estimates, and exits
after idling for ``lifecycle.supervisor_linger_s`` or when its stop event is
set.

Threads: the main thread owns every piece of state. Worker reader threads and
Job Object monitor threads only put events on a queue that the main loop waits
on (the tick is the wait timeout, so events wake it early); one probe thread
refreshes a cached GPU snapshot. Writes to workers never block, because
``WorkerProcess.send`` hands them to a writer thread.

Clocks: the injectable ``clock`` stamps every scheduling decision (waiting
times, deadlines, idle TTL, grace periods, cooldowns, linger). The cadence of
scans, heartbeats and purges and the bounded waits of a shutdown follow
:func:`time.monotonic`, and file ages compare against :func:`time.time`
because file modification times are wall-clock values.

The module imports no ML packages.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import queue
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, TypeAlias

from core.stt_pool import fsio, ipc
from core.stt_pool.config import config_summary, project_root
from core.stt_pool.engine_api import (
    DEFAULT_ENGINE_FACTORY,
    ERROR_KIND_CUDA_OOM,
    ERROR_KIND_MEMORY_LIMIT,
    ERROR_KIND_RESULT_TOO_LARGE,
    ERROR_KIND_TRANSCRIBE_FAILED,
    MEMORY_SNAPSHOT_KEYS,
    EngineResult,
    LoadInfo,
    WorkerSpec,
    job_message_fields,
)
from core.stt_pool.estimates import EstimateStore
from core.stt_pool.keys import device_for_backend, make_key
from core.stt_pool.procs import WorkerProcess, WorkerSpawnError, default_worker_launch
from core.stt_pool.queue import (
    ERROR_KIND_INPUT_INVALID,
    JOB_KIND_DIARIZE,
    MAX_RESULT_BYTES,
    JobRecord,
    JobValidationError,
    QueuePaths,
    ResultRecord,
    SupervisorLock,
    clear_heartbeat,
    delete_job_files,
    has_result,
    is_valid_id,
    job_file_age_s,
    list_incoming_ids,
    purge_stale,
    read_attempts,
    read_job_payload,
    resolve_queue_root,
    write_attempts,
    write_heartbeat,
    write_result,
)
from core.stt_pool.scheduler import (
    Assign,
    KeyStatus,
    Limits,
    PendingJob,
    Probes,
    Reject,
    RunningJob,
    SchedulerState,
    Stop,
    WorkerView,
    commit_growth_blockers,
    plan,
    reservation_for,
    service_seconds,
    threads_for,
)
from core.stt_pool.sysprobe import (
    find_nvidia_smi,
    hardware_fingerprint,
    query_commit,
    query_gpu,
)
from core.stt_pool.worker_setup import worker_environment, worker_spec_options
from core.temp_gc import purge_stale_askvlm_temp_dirs

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from core.stt_pool.config import PoolConfig
    from core.stt_pool.keys import WorkerKey
    from core.stt_pool.procs import WorkerLaunch
    from core.stt_pool.queue import ResultStatus
    from core.stt_pool.scheduler import Action, Reservation, Spawn, WorkerState
    from core.stt_pool.sysprobe import CommitInfo, GpuSnapshot
    from core.temp_gc import TempGcSummary

logger = logging.getLogger(__name__)

# * ``error_kind`` values the supervisor publishes itself (spec 6.2).
ERROR_KIND_WORKER_CRASHED = "worker_crashed"
ERROR_KIND_RETRY_EXHAUSTED = "retry_exhausted"
ERROR_KIND_PROTOCOL_ERROR = "protocol_error"
ERROR_KIND_CANCELLED = "cancelled"
ERROR_KIND_EXPIRED = "expired"

# * Supervisor log: rotating, 5 MB x 3 (spec 6.2).
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(threadName)s] %(message)s"
# * Logger that receives the file handler: every ``core.*`` module logs there.
_LOG_TARGET = "core"

# * Admission (spec 8). An unreadable job file is retried for this long after
# * its first failed read.
MAX_SCAN_BATCH = 1000
UNREADABLE_RETRY_S = 30.0
TERMINAL_MEMORY = 10_000

# * A terminal result whose write fails is retried every tick for at least
# * this long, or until the job's deadline when that is later.
RESULT_RETRY_S = 60.0

# * GPU presence (spec 11: ``cuda`` falls back to CPU only when CUDA is
# * confirmed absent). Without any probe CUDA is absent at once; with a probe it
# * is absent when no reading succeeded within the grace after the supervisor
# * started despite at least this many failed probes.
CUDA_ABSENT_GRACE_S = 30.0
MIN_FAILED_PROBES_FOR_ABSENT = 2

# * Crash bookkeeping (spec 8): three crashes within ten minutes start a crash
# * cooldown that doubles per repetition up to ten minutes.
CRASH_WINDOW_S = 600.0
CRASHES_FOR_COOLDOWN = 3
MAX_CRASH_COOLDOWN_S = 600.0

# * A worker whose Job Object peak reached this share of its limit is treated
# * as having hit the limit (the monitor notification may lag the exit).
MEMORY_LIMIT_PEAK_RATIO = 0.98

# * Worker log excerpts: crash results carry the last 1500 characters; a load
# * failure without a typed error detail keeps a shorter tail.
LOG_TAIL_CHARS = 1500
LOAD_TAIL_CHARS = 500
FATAL_DETAIL_CHARS = 1500
KEEP_WORKER_LOGS = 50

# * Bounded waits around worker exits.
EXIT_WAIT_S = 3.0
CLOSE_TIMEOUT_S = 10.0
LOST_EXIT_TIMEOUT_S = 15.0
SHUTDOWN_KILL_WAIT_S = 10.0
GPU_PROBE_JOIN_S = 7.0
STARTUP_GPU_PROBES = 2
MAX_EVENTS_PER_PUMP = 1000

_LOADING_STATES: frozenset[str] = frozenset({"spawning", "loading"})
_IDLE_STOP_STATES: frozenset[str] = frozenset({"ready", "retiring"})
_GIB = 1024**3

CancelReason = Literal["cancelled", "expired"]
ExitKind = Literal["stopped", "cancel_kill", "load_failed", "crashed"]
GpuPresence = Literal["present", "absent", "unknown"]


@dataclass(frozen=True, slots=True)
class SupervisorOptions:
    """Seams and cadences of :func:`run_supervisor`; the defaults are production.

    Attributes:
        engine_factory: ``"module:callable"`` named in every worker ``spec``;
            a test seam that is never read from queue files.
        worker_launch: Builds the launch of a worker from its key and the
            extra worker environment (thread caps; for a GPU worker also the
            cuDNN v8 cache bound); ``None`` uses
            :func:`~core.stt_pool.procs.default_worker_launch`.
        commit_probe: Free system commit reader, sampled every tick.
        gpu_probe: GPU reader run on the probe thread; ``None`` uses
            ``nvidia-smi`` (no probe at all when it is not installed).
        clock: Time source of every scheduling decision.
        tick_s: Longest wait of the main loop for worker events.
        scan_interval_s: Interval of ``incoming/`` scans, cancel markers and
            deadlines included.
        heartbeat_interval_s: Interval of heartbeat writes.
        gpu_probe_interval_s: Interval of GPU probes.
        load_timeout_s: Longest cold load before a worker is killed.
        purge_interval_s: Interval of queue and worker-log housekeeping.
        max_plan_jobs: Most pending jobs handed to one planning tick (the
            oldest by admission plus every ``high`` job).
        require_job: Passed to :class:`~core.stt_pool.procs.WorkerProcess`;
            ``None`` requires a Job Object on Windows.
        temp_gc: Startup sweep of stale AskVLM temp directories.
        cuda_absent_grace_s: How long after the start a GPU probe that never
            succeeded keeps the GPU presence ``unknown`` (``cuda`` jobs wait)
            before CUDA counts as absent (``cuda`` jobs run on CPU).
        result_retry_s: Shortest retry window of a terminal result whose
            write fails (extended to the job's deadline when that is later).

    """

    engine_factory: str = DEFAULT_ENGINE_FACTORY
    worker_launch: Callable[[WorkerKey, Mapping[str, str]], WorkerLaunch] | None = None
    commit_probe: Callable[[], CommitInfo] = query_commit
    gpu_probe: Callable[[], GpuSnapshot | None] | None = None
    clock: Callable[[], float] = time.time
    tick_s: float = 0.2
    scan_interval_s: float = 0.5
    heartbeat_interval_s: float = 2.0
    gpu_probe_interval_s: float = 5.0
    load_timeout_s: float = 600.0
    purge_interval_s: float = 60.0
    max_plan_jobs: int = 2000
    require_job: bool | None = None
    temp_gc: Callable[[], TempGcSummary] = purge_stale_askvlm_temp_dirs
    cuda_absent_grace_s: float = CUDA_ABSENT_GRACE_S
    result_retry_s: float = RESULT_RETRY_S


def run_supervisor(
    queue_root: Path | None,
    config: PoolConfig,
    *,
    options: SupervisorOptions | None = None,
    stop_event: threading.Event | None = None,
) -> int:
    """Serve one queue root until it idles out or *stop_event* is set.

    Args:
        queue_root: Queue root; ``None`` resolves it like the client does
            (``ASKVLM_EXTERNAL_QUEUE_DIR``, then the project cache).
        config: Validated pool configuration.
        options: Test seams and cadences; ``None`` uses the production defaults.
        stop_event: Set by signal handlers to stop: every worker is stopped,
            incoming jobs and attempt counters stay for the next supervisor.

    Returns:
        ``0`` after a clean exit, also at once when another supervisor holds
        the queue lock; ``1`` when an unexpected error ended the supervisor
        (logged with its traceback after a full cleanup).

    """
    opts = options if options is not None else SupervisorOptions()
    paths = QueuePaths(resolve_queue_root(queue_root))
    paths.ensure()
    lock = SupervisorLock(paths)
    if not lock.try_acquire():
        logger.info("another supervisor owns the queue; exiting: root=%s", paths.root)
        return 0
    attached = _attach_log(paths)
    try:
        supervisor = _Supervisor(
            paths,
            config,
            opts,
            stop_event if stop_event is not None else threading.Event(),
        )
        return supervisor.run()
    finally:
        lock.release()
        _detach_log(attached)


# * Supervisor log file.


@dataclass(frozen=True, slots=True)
class _AttachedLog:
    """File handler attached for one supervisor run and the level it replaced."""

    handler: logging.Handler
    previous_level: int


def _attach_log(paths: QueuePaths) -> _AttachedLog | None:
    """Attach the rotating ``control/supervisor.log`` handler to ``core`` loggers.

    The ``core`` logger is lowered to INFO for the duration of the run when its
    effective level is higher, so lifecycle records reach the file.

    Returns:
        The attachment, or ``None`` when the log file cannot be opened.

    """
    try:
        handler = logging.handlers.RotatingFileHandler(
            paths.supervisor_log,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("supervisor log unavailable: path=%s error=%s", paths, exc)
        return None
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    target = logging.getLogger(_LOG_TARGET)
    previous = target.level
    if target.getEffectiveLevel() > logging.INFO:
        target.setLevel(logging.INFO)
    target.addHandler(handler)
    return _AttachedLog(handler=handler, previous_level=previous)


def _detach_log(attached: _AttachedLog | None) -> None:
    """Remove and close the supervisor log handler and restore the level."""
    if attached is None:
        return
    target = logging.getLogger(_LOG_TARGET)
    target.removeHandler(attached.handler)
    target.setLevel(attached.previous_level)
    attached.handler.close()


# * GPU probe thread.


class _GpuMonitor:
    """Cached GPU snapshot refreshed on its own thread.

    The sample time of a reading is the clock value taken right before the
    probe ran, so a reading counts as newer than a GPU change only when the
    probe started after that change. ``cuda_present`` turns true with the first
    successful reading and stays true: a failed probe later only makes the free
    VRAM unknown (GPU spawns then wait for a fresh sample).

    :meth:`presence` is three-valued: ``present`` after a successful reading;
    ``absent`` without any probe, or when no reading succeeded within
    *absent_after_s* of *started_at* despite at least
    :data:`MIN_FAILED_PROBES_FOR_ABSENT` failed probes; ``unknown`` otherwise
    (a driver that is slow to answer right after boot, a busy GPU).
    """

    def __init__(
        self,
        probe: Callable[[], GpuSnapshot | None] | None,
        *,
        clock: Callable[[], float],
        interval_s: float,
        started_at: float,
        absent_after_s: float,
    ) -> None:
        self._probe = probe
        self._clock = clock
        self._interval_s = max(0.05, interval_s)
        self._started_at = started_at
        self._absent_after_s = max(0.0, absent_after_s)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._snapshot: GpuSnapshot | None = None
        self._sample_ts: float | None = None
        self._seen = False
        self._failures = 0
        self._thread: threading.Thread | None = None

    @property
    def cuda_present(self) -> bool:
        """Return whether any reading reported a GPU."""
        with self._lock:
            return self._seen

    @property
    def failed_probes(self) -> int:
        """Return how many probes reported no GPU."""
        with self._lock:
            return self._failures

    def presence(self, now: float) -> GpuPresence:
        """Return whether CUDA is present, confirmed absent, or still unknown.

        Args:
            now: Current clock value.

        Returns:
            ``present``, ``absent`` or ``unknown`` (see the class docstring).

        """
        with self._lock:
            if self._seen:
                return "present"
            if self._probe is None:
                return "absent"
            if (
                self._failures >= MIN_FAILED_PROBES_FOR_ABSENT
                and now - self._started_at >= self._absent_after_s
            ):
                return "absent"
            return "unknown"

    def reading(self) -> tuple[int | None, float | None]:
        """Return the free VRAM of the cached reading and its sample time."""
        with self._lock:
            snapshot = self._snapshot
            if snapshot is None:
                return None, None
            return snapshot.free_bytes, self._sample_ts

    def probe_now(self, attempts: int = 1) -> GpuSnapshot | None:
        """Probe synchronously (startup), retrying an empty reading.

        Args:
            attempts: Probes tried until one reports a GPU.

        Returns:
            The last snapshot, or ``None`` when no probe reported a GPU.

        """
        snapshot: GpuSnapshot | None = None
        for _ in range(max(1, attempts)):
            snapshot = self._probe_once()
            if snapshot is not None or self._probe is None:
                break
        return snapshot

    def start(self) -> None:
        """Start the periodic probe thread (nothing to do without a probe)."""
        if self._probe is None or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="stt-supervisor-gpu-probe", daemon=True
        )
        self._thread.start()

    def refresh(self) -> None:
        """Ask the probe thread for a reading right away."""
        self._wake.set()

    def stop(self, timeout_s: float) -> None:
        """Stop the probe thread with a bounded join."""
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout_s)
            if thread.is_alive():
                logger.warning("GPU probe thread did not stop within %.0f s", timeout_s)

    def _run(self) -> None:
        """Probe every interval and whenever a refresh is requested."""
        while not self._stop.is_set():
            self._wake.wait(self._interval_s)
            if self._stop.is_set():
                return
            self._wake.clear()
            self._probe_once()

    def _probe_once(self) -> GpuSnapshot | None:
        """Run the probe once and cache its reading."""
        probe = self._probe
        if probe is None:
            return None
        started = self._clock()
        try:
            snapshot = probe()
        except Exception:
            # * An injected probe must not end the probe thread silently.
            logger.exception("GPU probe failed")
            snapshot = None
        with self._lock:
            self._snapshot = snapshot
            self._sample_ts = started if snapshot is not None else None
            if snapshot is not None:
                self._seen = True
            else:
                self._failures += 1
        return snapshot


def _resolve_gpu_probe(
    options: SupervisorOptions,
) -> Callable[[], GpuSnapshot | None] | None:
    """Return the injected GPU probe, or ``nvidia-smi`` when it is installed."""
    if options.gpu_probe is not None:
        return options.gpu_probe
    executable = find_nvidia_smi()
    if executable is None:
        logger.info("nvidia-smi not found; the pool runs without a GPU probe")
        return None
    clock = options.clock

    def _query() -> GpuSnapshot | None:
        return query_gpu(executable, clock=clock)

    return _query


# * Main-loop events and bookkeeping records.


@dataclass(frozen=True, slots=True)
class _WorkerEvent:
    """A decoded worker message, or ``None`` at the end of its protocol stream."""

    worker_id: str
    message: dict[str, object] | None


@dataclass(frozen=True, slots=True)
class _MemoryLimitEvent:
    """The Job Object of a worker reported its commit cap."""

    worker_id: str
    pid: int


_Event: TypeAlias = _WorkerEvent | _MemoryLimitEvent


@dataclass(frozen=True, slots=True)
class _Fatal:
    """``fatal`` message of a worker."""

    stage: str
    error_kind: str
    detail: str | None


@dataclass(frozen=True, slots=True)
class _Outcome:
    """Terminal outcome of a job before it becomes a result record.

    ``data`` is the structured payload of an ``ok``/``empty`` outcome;
    ``memory`` is the worker's memory snapshot of the result (telemetry for
    the log only, never written to the result file).
    """

    status: ResultStatus
    text: str = ""
    error_kind: str | None = None
    error_detail: str | None = None
    run_s: float = 0.0
    data: Mapping[str, object] | None = None
    memory: Mapping[str, int | None] | None = None


@dataclass(frozen=True, slots=True)
class _Telemetry:
    """Memory figures logged with a published result.

    Attributes:
        memory: The worker's memory snapshot of the result
            (:data:`~core.stt_pool.engine_api.MEMORY_SNAPSHOT_KEYS`), or
            ``None`` when the result carried none.
        job_peak_bytes: ``PeakJobMemoryUsed`` of the worker's Job Object when
            the result was published (the peak over the worker's life so far),
            or ``None`` when unknown.

    """

    memory: Mapping[str, int | None] | None = None
    job_peak_bytes: int | None = None

    def log_fields(self) -> str:
        """Return the figures as ``key=value`` pairs (``None`` when unknown)."""
        memory = self.memory or {}
        pairs = [f"{key}={memory.get(key)}" for key in MEMORY_SNAPSHOT_KEYS]
        pairs.append(f"job_peak_bytes={self.job_peak_bytes}")
        return " ".join(pairs)


_NO_TELEMETRY = _Telemetry()


@dataclass(slots=True, eq=False)
class _Unpublished:
    """A terminal result whose write failed; it is retried every tick.

    Its job is out of the job table and keeps its files, so it is neither
    re-admitted nor re-run while the write is retried.
    """

    record: ResultRecord
    key_label: str | None
    give_up_at: float
    failures: int = 1
    telemetry: _Telemetry = _NO_TELEMETRY


@dataclass(slots=True, eq=False)
class _Job:
    """One admitted job (pending while ``worker_id`` is ``None``)."""

    record: JobRecord
    pending: PendingJob
    attempts: int
    warnings: list[str] = field(default_factory=list)
    worker_id: str | None = None
    attempt_id: str | None = None
    started_at: float | None = None
    assignment: Assign | None = None
    cancel_reason: CancelReason | None = None
    cancel_sent_at: float | None = None

    @property
    def job_id(self) -> str:
        """Return the job id."""
        return self.record.job_id


@dataclass(slots=True, eq=False)
class _Worker:
    """Supervisor-side state of one worker process.

    Lifecycle: ``spawning`` (process started) -> ``loading`` (``hello``) ->
    ``ready`` (``loaded``) -> optionally ``retiring`` (after a CUDA OOM or a
    commit-cap failure of a job, when its Job Object reported the commit cap
    while it was ready and the cap was not raised live, when its own commit
    reached the recycle threshold, or when its uptime reached
    ``lifecycle.worker_max_uptime_h``) -> ``stopping`` (``shutdown`` sent
    or the process killed) -> removed once its exit is confirmed and the
    handle is closed. ``stop_reason`` marks a stop the supervisor asked for
    and ``kill_reason`` why it killed the process; both decide how the exit
    is classified (:func:`_exit_kind`). ``load_failure_recorded`` marks a
    worker whose failed cold load already started its key's cooldown; no
    worker of that key is spawned until its exit is confirmed.
    ``commit_bumped`` and ``vram_bumped`` mark the x1.25 growth of the key's
    commit and VRAM estimates this worker already caused (each at most once
    per worker, see :meth:`_Supervisor._bump_memory_once`).

    Commit-cap reports: ``cap_events`` counts the report events handled and
    ``cap_hits_seen`` the reports already acted on (the handle counts them
    too, see :meth:`_Supervisor._cap_hits`). ``reservation`` is replaced
    when the job cap is raised live (at most once: ``cap_raised_at`` is set);
    ``cap_hits_at_raise`` is the report count at that moment,
    ``pre_raise_attempts`` the attempts that ran then, whose failures still
    count as commit-cap failures, and ``pre_raise_limit_bytes`` the job cap
    they ran under.
    """

    worker_id: str
    key: WorkerKey
    capacity: int
    reservation: Reservation
    proc: WorkerProcess
    spawned_at: float
    state: WorkerState = "spawning"
    last_used_at: float = 0.0
    idle_since: float | None = None
    running: dict[str, None] = field(default_factory=dict)
    loaded: bool = False
    fatal: _Fatal | None = None
    bye: bool = False
    memory_limit_reported: bool = False
    stop_reason: str | None = None
    stop_deadline: float | None = None
    kill_reason: str | None = None
    killed_mono: float | None = None
    overdue_since: float | None = None
    load_failure_recorded: bool = False
    commit_bumped: bool = False
    vram_bumped: bool = False
    cap_events: int = 0
    cap_hits_seen: int = 0
    cap_raised_at: float | None = None
    cap_hits_at_raise: int = 0
    pre_raise_attempts: frozenset[str] = frozenset()
    pre_raise_limit_bytes: int = 0


@dataclass(slots=True)
class _KeyBook:
    """Crash and load-failure bookkeeping of one worker configuration."""

    crash_times: deque[float] = field(default_factory=deque)
    crash_level: int = 0
    crash_cooldown_until: float | None = None
    load_failure_until: float | None = None
    load_failure_detail: str | None = None


def _exit_kind(worker: _Worker) -> ExitKind:
    """Classify the exit of a worker for its running jobs and its key.

    A stop the supervisor asked for wins (the jobs are victims); a kill for a
    job that ignored its cancel only victimizes the other jobs; an exit before
    ``loaded`` is a load failure; anything else is a crash.
    """
    if worker.stop_reason is not None:
        return "stopped"
    if worker.kill_reason == "cancel_grace":
        return "cancel_kill"
    if not worker.loaded:
        return "load_failed"
    return "crashed"


def _clip(text: str, limit: int) -> str:
    """Return *text* cut to *limit* characters."""
    return text if len(text) <= limit else text[:limit] + "..."


def _int_or_none(value: object) -> int | None:
    """Return *value* when it is an integer (booleans excluded)."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _readable_result(record: ResultRecord) -> ResultRecord:
    """Return *record*, or an explicit error when clients could not read it.

    A result document larger than the result-file read limit
    (:data:`~core.stt_pool.queue.MAX_RESULT_BYTES`) would never be read by the
    client, which would wait until its deadline. Such a result becomes an
    ``error`` of kind ``result_too_large`` (without text and data) instead.

    Args:
        record: Result about to be written.

    Returns:
        *record* when it fits the limit, otherwise its error replacement.

    """
    size = len(fsio.dumps_json_bytes(record.to_dict()))
    if size <= MAX_RESULT_BYTES:
        return record
    detail = (
        f"the {record.status} result is {size} bytes, over the result file "
        f"limit of {MAX_RESULT_BYTES} bytes"
    )
    logger.error(
        "result too large; publishing an error instead: job_id=%s status=%s "
        "size_bytes=%d limit_bytes=%d",
        record.job_id,
        record.status,
        size,
        MAX_RESULT_BYTES,
    )
    return replace(
        record,
        status="error",
        text="",
        data=None,
        error_kind=ERROR_KIND_RESULT_TOO_LARGE,
        error_detail=detail,
    )


def _missing_payload(record: JobRecord, result: EngineResult) -> str | None:
    """Describe the structured payload that a successful result lacks.

    A ``diarize`` job needs ``data.turns`` and an ``stt`` job that asked for
    segments needs ``data.segments``, each a JSON list; an empty list is a
    valid answer (no speech found, no segments).

    Args:
        record: Job the result answers.
        result: Worker result of the job's current attempt.

    Returns:
        A description of the missing payload, or ``None`` when the result is
        not ``ok``/``empty`` or carries what the job asked for.

    """
    if result.status not in {"ok", "empty"}:
        return None
    if record.kind == JOB_KIND_DIARIZE:
        name = "turns"
    elif record.want_segments:
        name = "segments"
    else:
        return None
    data = result.data
    if data is None:
        found = "the result has no data"
    elif name not in data:
        found = f"data has no {name!r} entry"
    elif isinstance(data[name], list):
        return None
    else:
        found = f"data[{name!r}] is a {type(data[name]).__name__}"
    return (
        f"{record.kind} job result ({result.status}) lacks the requested "
        f"{name!r} list ({found})"
    )


def _with_requested_payload(
    record: JobRecord, worker: _Worker, result: EngineResult
) -> EngineResult:
    """Return *result*, or a ``protocol_error`` when it lacks its payload.

    A success without the structured payload its job asked for would reach
    the client as a silent drop of the segments or turns, so it is published
    as an explicit error instead (see :func:`_missing_payload`).

    Args:
        record: Job the result answers.
        worker: Worker that sent the result.
        result: Worker result of the job's current attempt.

    Returns:
        *result*, or its ``error`` replacement without text and data.

    """
    problem = _missing_payload(record, result)
    if problem is None:
        return result
    logger.error(
        "result lacks its requested payload; publishing %s instead: job_id=%s "
        "worker_id=%s key=%s detail=%s",
        ERROR_KIND_PROTOCOL_ERROR,
        result.job_id,
        worker.worker_id,
        worker.key.label,
        problem,
    )
    return replace(
        result,
        status="error",
        text="",
        error_kind=ERROR_KIND_PROTOCOL_ERROR,
        error_detail=problem,
        data=None,
    )


def _cap_failure(
    worker: _Worker, result: EngineResult, limit_bytes: int
) -> EngineResult:
    """Return a ``transcribe_failed`` *result* of a capped worker as ``memory_limit``.

    The worker's Job Object reported its commit cap, so an unclassified
    engine failure is taken for a memory failure: a child process such as
    ffmpeg that could not commit memory reaches the engine as an ordinary
    error. The engine's detail stays at the head, because clients classify
    on it, and the cap is noted after it.

    Args:
        worker: Worker whose Job Object reported its commit cap.
        result: ``transcribe_failed`` result of the job's current attempt.
        limit_bytes: The worker's Job Object commit cap.

    Returns:
        *result* with ``error_kind`` ``memory_limit`` and the noted detail.

    """
    limit_gib = limit_bytes / _GIB
    note = f"the worker's Job Object reached its memory limit of {limit_gib:.2f} GiB"
    detail = f"{result.error_detail}; {note}" if result.error_detail else note
    logger.warning(
        "error of a capped worker taken as %s: job_id=%s worker_id=%s key=%s "
        "error_kind=%s",
        ERROR_KIND_MEMORY_LIMIT,
        result.job_id,
        worker.worker_id,
        worker.key.label,
        result.error_kind,
    )
    return replace(result, error_kind=ERROR_KIND_MEMORY_LIMIT, error_detail=detail)


# * The supervisor.
# *
# * One loop iteration: wait up to ``tick_s`` for worker events and handle all
# * of them (results publish or re-queue jobs, EOF confirms an exit, settles
# * the worker's jobs and releases its reservation); then, on the scan cadence,
# * admit new job files and apply cancel markers and deadlines; enforce load,
# * stop and cancel timeouts; build a SchedulerState and execute the plan in
# * order (Reject, Stop, Assign, Spawn); save estimates when due; beat the
# * heart. A commit-cap report of a worker's Job Object grows its key's commit
# * estimate; a ready worker's first report raises its live job cap once to
# * the grown reservation when it names a child process (Windows notifies
# * only a process's first denial, so a denied worker process could not report
# * a second hit) and the commit budget and the free-commit guard allow it
# * (the worker keeps serving), otherwise, and on any later report, the ready
# * worker retires. A ``transcribe_failed`` attempt that ran under a
# * reported cap (unless its cancel is pending) retries like a
# * ``memory_limit`` error, while its successes and other error kinds keep
# * their handling; the failure of an attempt the raise covered does not
# * retire the worker. A ready worker whose own commit in a result snapshot
# * reaches ``memory.recycle_commit_fraction`` of its commit reservation
# * retires too, and so does a ready worker whose uptime since spawn reaches
# * ``lifecycle.worker_max_uptime_h``. Job fate after a worker exit
# * (``_settle_jobs``): a pending cancel wins; victims of a supervisor stop or
# * of a kill caused by another job get their attempt back; a crash costs the
# * attempt (retry while attempts last); a crash that hit the commit cap (after
# * a live raise: the raised cap) pins the re-queued jobs to the worker's
# * backend (victims are never pinned).


class _Supervisor:
    """Main loop and every piece of state of one supervisor run."""

    def __init__(
        self,
        paths: QueuePaths,
        config: PoolConfig,
        options: SupervisorOptions,
        stop_event: threading.Event,
    ) -> None:
        self._paths = paths
        self._config = config
        self._options = options
        self._stop_event = stop_event
        self._clock = options.clock
        self._limits = Limits.from_config(config)
        self._instance_id = uuid.uuid4().hex
        self._events: queue.SimpleQueue[_Event] = queue.SimpleQueue()
        self._jobs: dict[str, _Job] = {}
        self._workers: dict[str, _Worker] = {}
        self._keys: dict[WorkerKey, _KeyBook] = {}
        self._terminal: OrderedDict[str, None] = OrderedDict()
        # * Unreadable job files -> clock value of their first failed read.
        self._unreadable: dict[str, float] = {}
        self._unpublished: dict[str, _Unpublished] = {}
        # * ``cuda`` jobs admitted while the GPU presence was unknown.
        self._cuda_waiting: set[str] = set()
        self._gpu_presence: GpuPresence | None = None
        self._call_last_started: dict[str, float] = {}
        self._seq = 0
        self._worker_counter = 0
        self._heartbeat_seq = 0
        self._started_at = self._clock()
        self._last_gpu_change_ts: float | None = None
        self._idle_since: float | None = None
        self._next_scan = 0.0
        self._next_heartbeat = 0.0
        self._next_purge = 0.0
        self._gpu = _GpuMonitor(
            _resolve_gpu_probe(options),
            clock=self._clock,
            interval_s=options.gpu_probe_interval_s,
            started_at=self._started_at,
            absent_after_s=options.cuda_absent_grace_s,
        )
        # * In-memory until startup loads the persisted store.
        self._estimates = EstimateStore(None, fingerprint={}, clock=self._clock)

    # * Run.

    def run(self) -> int:
        """Start, serve until idle or stopped, and always clean up.

        Returns:
            ``0`` after a clean exit, ``1`` after an unexpected error.

        """
        try:
            self._write_heartbeat("starting")
            self._startup()
            self._serve()
        except Exception:
            logger.exception(
                "supervisor ended by an unexpected error: root=%s instance_id=%s",
                self._paths.root,
                self._instance_id,
            )
            code = 1
        else:
            code = 0
        finally:
            self._teardown()
        return code

    def _startup(self) -> None:
        """Sweep temp dirs, probe the GPU, load estimates, start the probe thread."""
        logger.info(
            "supervisor starting: root=%s instance_id=%s pid=%d config=%s",
            self._paths.root,
            self._instance_id,
            os.getpid(),
            self._config.source,
        )
        summary = self._options.temp_gc()
        logger.info(
            "temp GC: deleted=%d skipped=%d freed_bytes=%d",
            summary.deleted_dirs,
            summary.skipped_dirs,
            summary.freed_bytes,
        )
        snapshot = self._gpu.probe_now(attempts=STARTUP_GPU_PROBES)
        gpu_name = snapshot.name if snapshot is not None else None
        logger.info(
            "GPU probe: present=%s name=%s free_vram_bytes=%s",
            snapshot is not None,
            gpu_name,
            snapshot.free_bytes if snapshot is not None else None,
        )
        self._estimates = EstimateStore.load(
            self._paths.estimates_file,
            fingerprint=hardware_fingerprint(gpu_name),
            clock=self._clock,
        )
        self._gpu.start()
        self._housekeep_files()
        self._next_purge = time.monotonic() + self._options.purge_interval_s
        self._write_heartbeat("running")

    def _serve(self) -> None:
        """Run ticks until the stop event or the idle linger ends the run."""
        while not self._stop_event.is_set():
            self._pump_events(self._options.tick_s)
            if self._stop_event.is_set():
                break
            now = self._clock()
            self._tick(now)
            if self._idle_exit_due(now):
                # * Clients must see the supervisor as gone as soon as it
                # * decides to exit, not after the probe stop and the saves.
                clear_heartbeat(self._paths)
                logger.info(
                    "supervisor idle for %.1f s; exiting",
                    self._config.lifecycle.supervisor_linger_s,
                )
                return
        logger.info("stop requested; stopping %d worker(s)", len(self._workers))
        self._shutdown_workers()

    def _tick(self, now: float) -> None:
        """Retry results, scan, keep house, plan, save estimates, beat the heart."""
        mono = time.monotonic()
        self._retry_unpublished(now)
        if mono >= self._next_scan:
            self._next_scan = mono + self._options.scan_interval_s
            self._scan(now)
        self._settle_cuda_waiting(now)
        self._housekeeping(now, mono)
        self._plan(now)
        self._estimates.save_if_due()
        if mono >= self._next_heartbeat:
            self._write_heartbeat("running")

    def _busy(self) -> bool:
        """Return whether any job, worker or unwritten result still needs the loop."""
        return bool(
            self._workers or self._jobs or self._unreadable or self._unpublished
        )

    def _idle_exit_due(self, now: float) -> bool:
        """Return whether nothing ran for the linger time (after a final scan)."""
        if self._busy():
            self._idle_since = None
            return False
        if self._idle_since is None:
            self._idle_since = now
            return False
        if now - self._idle_since < self._config.lifecycle.supervisor_linger_s:
            return False
        self._scan(now)
        if self._busy():
            self._idle_since = None
            return False
        return True

    def _teardown(self) -> None:
        """Drop the heartbeat, close every worker, stop the probe, save estimates.

        The heartbeat goes first on every exit path, so clients stop counting
        on this supervisor while the bounded closing steps still run.
        """
        clear_heartbeat(self._paths)
        self._retry_unpublished(self._clock(), final=True)
        for worker in list(self._workers.values()):
            logger.warning(
                "worker closed at supervisor exit: worker_id=%s key=%s",
                worker.worker_id,
                worker.key.label,
            )
            worker.proc.close(timeout_s=CLOSE_TIMEOUT_S)
        self._workers.clear()
        self._gpu.stop(GPU_PROBE_JOIN_S)
        self._estimates.save_if_due(force=True)
        logger.info(
            "supervisor stopped: root=%s instance_id=%s",
            self._paths.root,
            self._instance_id,
        )

    # * Events.

    def _on_worker_event(
        self, worker_id: str, message: dict[str, object] | None
    ) -> None:
        """Queue a worker message (reader thread)."""
        self._events.put(_WorkerEvent(worker_id, message))

    def _on_memory_limit(self, worker_id: str, pid: int) -> None:
        """Queue a commit-cap report (Job Object monitor thread)."""
        self._events.put(_MemoryLimitEvent(worker_id, pid))

    def _pump_events(self, timeout_s: float) -> None:
        """Wait up to *timeout_s* for an event, then handle every queued one."""
        try:
            event = self._events.get(timeout=max(0.0, timeout_s))
        except queue.Empty:
            return
        self._handle_event(event)
        for _ in range(MAX_EVENTS_PER_PUMP):
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                return
            self._handle_event(event)

    def _handle_event(self, event: _Event) -> None:
        """Dispatch one event to the worker it belongs to."""
        worker = self._workers.get(event.worker_id)
        if worker is None:
            logger.debug("event of a released worker ignored: %s", event.worker_id)
            return
        if isinstance(event, _MemoryLimitEvent):
            worker.cap_events += 1
            self._note_cap_hits(worker, self._clock(), pid=event.pid)
            return
        message = event.message
        if message is None:
            self._finalize(worker)
            return
        kind = message.get("type")
        now = self._clock()
        if kind == ipc.MSG_RESULT:
            self._on_result(worker, message, now)
        elif kind == ipc.MSG_HELLO:
            self._on_hello(worker, message)
        elif kind == ipc.MSG_LOADED:
            self._on_loaded(worker, message, now)
        elif kind == ipc.MSG_FATAL:
            self._on_fatal(worker, message)
        elif kind == ipc.MSG_BYE:
            worker.bye = True
            logger.info("worker said bye: worker_id=%s", worker.worker_id)

    def _note_cap_hits(
        self, worker: _Worker, now: float, *, pid: int | None = None
    ) -> None:
        """Act once on every commit-cap report of a worker's Job Object.

        Runs for the report's event and before each result of the worker: the
        handle counts a report on its monitor thread before the event is
        queued, so a result that overtakes the event still sees the report,
        and the report is acted on before the result is classified.

        The report names the process whose allocation failed: the worker or a
        child such as ffmpeg, which shares the job's cap. Either way the
        worker's reservation proved too small, so the key's commit estimate
        grows x1.25 at once (at most once per worker) and the next worker
        reserves more. A ready worker's first report, when it names a child
        process, raises its live job cap to the grown reservation
        (:meth:`_raise_job_cap`) and the worker keeps serving; when the raise
        is disabled or blocked (the worker process itself was denied
        included), and on any later report, a ready worker retires: it takes
        no new jobs and stops once idle. A worker still loading keeps loading
        and serves once loaded, so a key whose estimate cannot grow any
        further still makes progress instead of respawning in a loop.
        ``transcribe_failed`` results of capped attempts are retried as
        ``memory_limit`` by :meth:`_on_result`.

        Args:
            worker: Worker whose Job Object may have reported its cap.
            now: Current clock value.
            pid: Process the report named, when an event carried it.

        """
        hits = self._cap_hits(worker)
        if hits <= worker.cap_hits_seen:
            return
        worker.cap_hits_seen = hits
        worker.memory_limit_reported = True
        if pid is None:
            pid = worker.proc.memory_limit_pid
        logger.warning(
            "worker reached its commit cap: worker_id=%s key=%s pid=%s hits=%d "
            "limit_bytes=%d commit_bytes=%d state=%s",
            worker.worker_id,
            worker.key.label,
            pid,
            hits,
            self._job_limit_bytes(worker.reservation),
            worker.reservation.commit_bytes,
            worker.state,
        )
        if (
            worker.state == "ready"
            and worker.cap_raised_at is None
            and hits == 1
            and self._raise_job_cap(worker, hits, pid, now)
        ):
            return
        if worker.state == "ready":
            worker.state = "retiring"
        self._bump_memory_once(worker, vram=False, reason="job object commit cap")

    def _raise_job_cap(
        self, worker: _Worker, hits: int, pid: int | None, now: float
    ) -> bool:
        """Raise a ready worker's live job cap once, to the grown reservation.

        The key's commit estimate grows first (:meth:`_bump_memory_once`), and
        the worker's new commit reservation is exactly what a fresh worker of
        the same capacity would reserve now
        (:func:`~core.stt_pool.scheduler.reservation_for`), so it never exceeds
        the estimate ceiling. The raise needs the report to name a child
        process (:meth:`_raise_blockers`) and the growth to fit the commit
        budget, the free-commit guard and the critical free commit. Only then
        does the Job Object get the new cap (reservation plus child headroom;
        its other limits stay and the one-shot report is re-armed), and the
        new reservation replaces the old one for every later accounting:
        admission, exit classification, crash details and the heartbeat.

        Args:
            worker: Ready worker with its first commit-cap report.
            hits: The worker's report count (``1``).
            pid: Process the report named, or ``None`` when unknown.
            now: Current clock value.

        Returns:
            ``True`` when the cap was raised; ``False`` when the raise is
            disabled (``memory.live_cap_raise``), the worker has no cap, or
            the raise was blocked (logged with the reasons).

        """
        old = worker.reservation
        if not self._config.memory.live_cap_raise or old.commit_bytes <= 0:
            return False
        self._bump_memory_once(worker, vram=False, reason="job object commit cap")
        estimate = self._estimates.estimate(worker.key)
        commit = reservation_for(
            worker.key, worker.capacity, estimate, self._limits
        ).commit_bytes
        grown = replace(old, commit_bytes=commit)
        limit = self._job_limit_bytes(grown)
        reasons = self._raise_blockers(worker, commit, pid, now)
        if not reasons and not worker.proc.raise_memory_limit(limit):
            reasons = ("the Job Object commit cap could not be changed",)
        if reasons:
            logger.warning(
                "live commit cap raise blocked; worker retiring: worker_id=%s key=%s "
                "commit_bytes=%d grown_commit_bytes=%d reasons=%s",
                worker.worker_id,
                worker.key.label,
                old.commit_bytes,
                commit,
                "; ".join(reasons),
            )
            return False
        worker.pre_raise_limit_bytes = self._job_limit_bytes(old)
        worker.reservation = grown
        worker.cap_raised_at = now
        worker.cap_hits_at_raise = hits
        worker.pre_raise_attempts = frozenset(
            attempt_id
            for job_id in worker.running
            if (job := self._jobs.get(job_id)) is not None
            and (attempt_id := job.attempt_id) is not None
        )
        logger.warning(
            "worker commit cap raised live; worker keeps serving: worker_id=%s "
            "key=%s old_commit_bytes=%d commit_bytes=%d job_limit_bytes=%d "
            "running=%d",
            worker.worker_id,
            worker.key.label,
            old.commit_bytes,
            grown.commit_bytes,
            limit,
            len(worker.running),
        )
        return True

    def _raise_blockers(
        self, worker: _Worker, commit: int, pid: int | None, now: float
    ) -> tuple[str, ...]:
        """Return why a worker's job cap cannot be raised to *commit* plus headroom.

        Windows notifies only the first denied allocation of each process of
        a job (:class:`~core.stt_pool.winjob.WorkerJob`). When the worker
        process itself was denied (or the report named no process), a hit of
        the raised cap by the worker would never be reported, so the raise is
        refused and the worker retires as before. A child (ffmpeg) denial
        leaves the worker's own notification unused, and every later child is
        a new process, so a second hit is still reported. The growth itself
        must pass :func:`~core.stt_pool.scheduler.commit_growth_blockers` and
        stay above the current reservation (the estimate is not at its
        ceiling).

        Args:
            worker: Ready worker with its first commit-cap report.
            commit: Commit reservation the raise would set.
            pid: Process the report named, or ``None`` when unknown.
            now: Current clock value.

        Returns:
            The blocking reasons; empty when the raise may proceed.

        """
        if pid is None or pid <= 0 or pid == worker.proc.pid:
            return (
                "the worker process itself reached the cap, and Windows reports "
                "only the first denial of a process",
            )
        if commit <= worker.reservation.commit_bytes:
            return ("the commit estimate of the key cannot grow (at its ceiling)",)
        return commit_growth_blockers(
            self._scheduler_state(now), worker.worker_id, commit
        )

    def _check_recycle(
        self, worker: _Worker, memory: Mapping[str, int | None] | None
    ) -> None:
        """Retire a ready worker whose own commit reached its recycle threshold.

        The threshold is ``memory.recycle_commit_fraction`` of the worker's
        commit reservation (the raised one after a live raise); the measure
        is ``private_bytes`` of the worker process in a result snapshot, so
        its children are not counted. A retiring worker takes no new jobs, so
        under continuous load it finishes its running jobs, becomes idle and
        stops; the waiting jobs get a fresh worker. A snapshot without private
        bytes, a worker without a commit reservation and a worker that is not
        ready change nothing.

        Args:
            worker: Worker that sent a result.
            memory: The result's memory snapshot, or ``None``.

        """
        fraction = self._config.memory.recycle_commit_fraction
        reserved = worker.reservation.commit_bytes
        if fraction <= 0 or memory is None or worker.state != "ready" or reserved <= 0:
            return
        private = memory.get("private_bytes")
        threshold = int(fraction * reserved)
        if private is None or private < threshold:
            return
        worker.state = "retiring"
        logger.warning(
            "worker retiring: its own commit reached the recycle threshold: "
            "worker_id=%s key=%s private_bytes=%d threshold_bytes=%d "
            "commit_bytes=%d job_limit_bytes=%d running=%d",
            worker.worker_id,
            worker.key.label,
            private,
            threshold,
            reserved,
            self._job_limit_bytes(worker.reservation),
            len(worker.running),
        )

    def _on_hello(self, worker: _Worker, message: Mapping[str, object]) -> None:
        """Record the handshake; the cold load is in progress from here on."""
        if worker.state == "spawning":
            worker.state = "loading"
        if message.get("protocol") != ipc.PROTOCOL_VERSION:
            logger.warning(
                "worker speaks protocol %r: worker_id=%s",
                message.get("protocol"),
                worker.worker_id,
            )
        logger.info(
            "worker hello: worker_id=%s key=%s pid=%s",
            worker.worker_id,
            worker.key.label,
            _int_or_none(message.get("pid")),
        )

    def _on_loaded(
        self, worker: _Worker, message: Mapping[str, object], now: float
    ) -> None:
        """Mark a worker ready for jobs (or kill it after an invalid message).

        A ``loaded`` that arrives after the supervisor started stopping or
        killing the worker (a load timeout among others) is ignored: the
        worker never serves, and its load failure stays recorded. An invalid
        ``loaded`` of a worker that has not loaded yet is a failed load and
        cools its key down; the same message from a worker that already
        loaded (a duplicate) only kills the worker, and its exit counts as a
        crash, so the healthy key does not enter a load-failure cooldown.
        """
        if worker.state == "stopping":
            logger.info(
                "loaded message of a stopping worker ignored: worker_id=%s "
                "stop_reason=%s kill_reason=%s",
                worker.worker_id,
                worker.stop_reason,
                worker.kill_reason,
            )
            return
        try:
            info = LoadInfo.from_message(message)
        except ValueError as exc:
            logger.warning(
                "invalid loaded message; killing the worker: worker_id=%s error=%s",
                worker.worker_id,
                exc,
            )
            if worker.loaded:
                # * A duplicate of a loaded worker is no load failure: it crashes.
                self._kill(worker, "protocol")
            else:
                self._fail_load(worker, "protocol", now)
            return
        worker.loaded = True
        if worker.state in _LOADING_STATES:
            worker.state = "ready"
            worker.idle_since = now
            worker.last_used_at = now
        book = self._keys.get(worker.key)
        if book is not None:
            book.load_failure_until = None
            book.load_failure_detail = None
        if worker.key.backend == "gpu":
            self._last_gpu_change_ts = now
            self._gpu.refresh()
        logger.info(
            "worker ready: worker_id=%s key=%s device=%s load_s=%.2f capacity=%d",
            worker.worker_id,
            worker.key.label,
            info.device,
            info.load_s,
            worker.capacity,
        )

    def _on_fatal(self, worker: _Worker, message: Mapping[str, object]) -> None:
        """Remember a worker's fatal error; its exit follows."""
        detail = message.get("error_detail")
        worker.fatal = _Fatal(
            stage=str(message.get("stage")),
            error_kind=str(message.get("error_kind")),
            detail=detail if isinstance(detail, str) else None,
        )
        logger.error(
            "worker fatal: worker_id=%s key=%s stage=%s error_kind=%s detail=%s",
            worker.worker_id,
            worker.key.label,
            worker.fatal.stage,
            worker.fatal.error_kind,
            worker.fatal.detail,
        )

    # * Results.

    def _on_result(
        self, worker: _Worker, message: Mapping[str, object], now: float
    ) -> None:
        """Handle a ``result`` of the current attempt; ignore late replies.

        Every parsed result first lets a pending commit-cap report act (a raise
        or a retirement, :meth:`_note_cap_hits`) and then checks the worker's
        own commit against its recycle threshold (:meth:`_check_recycle`),
        late replies included, because the snapshot measures the worker as it
        is now.
        """
        result = self._parse_result(worker, message)
        if result is None:
            return
        self._note_cap_hits(worker, now)
        self._check_recycle(worker, result.memory)
        job = self._jobs.get(result.job_id)
        if (
            job is None
            or job.worker_id != worker.worker_id
            or job.attempt_id != result.attempt_id
        ):
            logger.info(
                "late or unknown result ignored: job_id=%s attempt_id=%s worker_id=%s",
                result.job_id,
                result.attempt_id,
                worker.worker_id,
            )
            return
        if result.status == "cancelled":
            self._on_cancelled(job, worker, result, now)
        elif result.status == "error" and result.error_kind in {
            ERROR_KIND_CUDA_OOM,
            ERROR_KIND_MEMORY_LIMIT,
        }:
            self._on_resource_error(job, worker, result, now)
        elif (
            result.status == "error"
            and result.error_kind == ERROR_KIND_TRANSCRIBE_FAILED
            and job.cancel_reason is None
            and self._ran_under_reported_cap(worker, job)
        ):
            # * The failing allocation may have been in a child (ffmpeg) whose
            # * failure reached the engine as an unclassified error; every other
            # * error kind keeps its own classification. An attempt covered by
            # * a live raise failed under the cap it ran with, the old one.
            limit = (
                worker.pre_raise_limit_bytes
                if self._raise_covers(worker, job)
                else self._job_limit_bytes(worker.reservation)
            )
            failure = _cap_failure(worker, result, limit)
            self._on_resource_error(job, worker, failure, now)
        else:
            result = _with_requested_payload(job.record, worker, result)
            succeeded = result.status in {"ok", "empty"}
            if succeeded:
                self._observe(job, worker, result)
                self._book(worker.key).crash_level = 0
            outcome = _Outcome(
                status=result.status,
                text=result.text if result.status == "ok" else "",
                error_kind=result.error_kind,
                error_detail=result.error_detail,
                run_s=result.run_s,
                data=result.data if succeeded else None,
                memory=result.memory,
            )
            self._publish(job, outcome, worker=worker, now=now)

    @staticmethod
    def _parse_result(
        worker: _Worker, message: Mapping[str, object]
    ) -> EngineResult | None:
        """Parse a result; a malformed one becomes a ``protocol_error`` result."""
        try:
            return EngineResult.from_message(message)
        except ValueError as exc:
            job_id, attempt_id = message.get("job_id"), message.get("attempt_id")
            logger.warning(
                "malformed result: worker_id=%s job_id=%r error=%s",
                worker.worker_id,
                job_id,
                exc,
            )
            if not (isinstance(job_id, str) and isinstance(attempt_id, str)):
                return None
            return EngineResult(
                job_id=job_id,
                attempt_id=attempt_id,
                status="error",
                text="",
                error_kind=ERROR_KIND_PROTOCOL_ERROR,
                error_detail=f"malformed result message: {exc}",
                backend=worker.key.backend,
                run_s=0.0,
                audio_s=None,
            )

    def _on_cancelled(
        self, job: _Job, worker: _Worker, result: EngineResult, now: float
    ) -> None:
        """Publish a requested cancellation; re-queue an unrequested one.

        An engine that drops a job while the supervisor stops its worker did
        not fail the job: the attempt is given back. Any other unrequested
        cancellation counts as a failed attempt, so a misbehaving engine can
        never re-run a job forever.
        """
        reason = job.cancel_reason
        if reason is not None:
            outcome = _Outcome(
                status=reason,
                error_kind=reason,
                run_s=result.run_s,
                memory=result.memory,
            )
            self._publish(job, outcome, worker=worker, now=now)
            return
        logger.warning(
            "unrequested cancellation: job_id=%s worker_id=%s stop_reason=%s "
            "attempts=%d",
            job.job_id,
            worker.worker_id,
            worker.stop_reason,
            job.attempts,
        )
        if worker.stop_reason is not None:
            self._requeue(job, now=now, restore=True)
        elif job.attempts < self._config.scheduling.max_attempts:
            self._requeue(job, now=now, restore=False)
        else:
            detail = f"worker {worker.worker_id} cancelled the job without a request"
            outcome = _Outcome(
                status="error",
                error_kind=ERROR_KIND_CANCELLED,
                error_detail=detail,
                run_s=result.run_s,
                memory=result.memory,
            )
            self._publish(job, outcome, worker=worker, now=now)

    def _on_resource_error(
        self, job: _Job, worker: _Worker, result: EngineResult, now: float
    ) -> None:
        """Retire the worker after a CUDA OOM or commit-cap failure; maybe retry.

        The key's memory estimates grow before the job is re-queued, so the
        fresh worker of the retry reserves more than the failed one, even
        while the retiring worker is still alive: a commit-cap failure
        (``memory_limit``) grows the commit estimate, a CUDA OOM on a GPU
        worker the commit and VRAM estimates (x1.25, once per worker). A CUDA
        OOM sends an ``auto`` job to CPU; a ``cuda`` job retries on a fresh GPU
        worker. A commit-cap failure pins the job to the backend it failed on,
        ``auto`` jobs included, and retries on a fresh worker of the same key.
        Without attempts left the error is published.

        A commit-cap failure of an attempt that ran when the worker's job cap
        was raised live (:meth:`_raise_covers`) neither retires the worker nor
        grows the estimate again (the raise did): the job retries the same
        way and may run on the raised worker again.
        """
        covered = result.error_kind == ERROR_KIND_MEMORY_LIMIT and self._raise_covers(
            worker, job
        )
        if not covered and worker.state == "ready":
            worker.state = "retiring"
        logger.warning(
            "job hit a resource limit; worker %s: job_id=%s worker_id=%s "
            "key=%s error_kind=%s attempts=%d %s",
            "keeps serving under its raised cap" if covered else "retiring",
            job.job_id,
            worker.worker_id,
            worker.key.label,
            result.error_kind,
            job.attempts,
            _Telemetry(result.memory, worker.proc.peak_memory_bytes()).log_fields(),
        )
        if result.error_kind == ERROR_KIND_MEMORY_LIMIT and not covered:
            self._bump_memory_once(worker, vram=False, reason="job memory_limit")
        elif result.error_kind == ERROR_KIND_CUDA_OOM and worker.key.backend == "gpu":
            self._bump_memory_once(worker, vram=True, reason="job cuda_oom")
        if job.attempts < self._config.scheduling.max_attempts:
            if result.error_kind == ERROR_KIND_MEMORY_LIMIT:
                job.pending = replace(job.pending, force_backend=worker.key.backend)
            elif (
                result.error_kind == ERROR_KIND_CUDA_OOM
                and worker.key.backend == "gpu"
                and job.record.request.device == "auto"
            ):
                job.pending = replace(job.pending, force_backend="cpu")
            self._requeue(job, now=now, restore=False)
            return
        outcome = _Outcome(
            status="error",
            error_kind=result.error_kind,
            error_detail=result.error_detail,
            run_s=result.run_s,
            memory=result.memory,
        )
        self._publish(job, outcome, worker=worker, now=now)

    def _observe(self, job: _Job, worker: _Worker, result: EngineResult) -> None:
        """Teach the estimates the measured run time of a finished job."""
        assignment = job.assignment
        if result.audio_s is None or assignment is None:
            return
        base = assignment.base_service_s
        if job.record.duration_s is None:
            base = service_seconds(
                worker.key,
                self._estimates.estimate(worker.key),
                result.audio_s,
                share=assignment.share,
                cpu_busy=assignment.cpu_busy,
                contention=self._limits.gpu_cpu_contention,
                corrected=False,
            )
        self._estimates.observe_job(worker.key, run_s=result.run_s, base_service_s=base)

    # * Publishing and re-queueing.

    def _publish(
        self, job: _Job, outcome: _Outcome, *, worker: _Worker | None, now: float
    ) -> None:
        """Publish the terminal result of an admitted job and forget the job.

        The worker's memory snapshot of the result and its Job Object's peak
        so far are logged with the publication (``result published``).
        """
        start = job.started_at if job.started_at is not None else now
        queued_s = max(0.0, start - job.record.submitted_at)
        telemetry = _NO_TELEMETRY
        if worker is not None:
            telemetry = _Telemetry(outcome.memory, worker.proc.peak_memory_bytes())
        self._detach(job, now)
        self._jobs.pop(job.job_id, None)
        record = ResultRecord(
            job_id=job.job_id,
            status=outcome.status,
            text=outcome.text,
            error_kind=outcome.error_kind,
            error_detail=outcome.error_detail,
            warnings=tuple(job.warnings),
            backend=worker.key.backend if worker is not None else None,
            worker_key=worker.key.to_dict() if worker is not None else None,
            worker_id=worker.worker_id if worker is not None else None,
            attempts=job.attempts,
            finished_at=now,
            timings={"queued_s": queued_s, "run_s": max(0.0, outcome.run_s)},
            data=None if outcome.data is None else dict(outcome.data),
        )
        self._write_terminal(
            _readable_result(record),
            key_label=worker.key.label if worker else None,
            now=now,
            deadline_at=job.record.deadline_at,
            telemetry=telemetry,
        )

    def _publish_file(
        self, job_id: str, error_kind: str, detail: str, now: float
    ) -> None:
        """Publish the ``error`` result of a job file that was never admitted."""
        record = ResultRecord(
            job_id=job_id,
            status="error",
            error_kind=error_kind,
            error_detail=detail,
            finished_at=now,
            timings={"queued_s": 0.0, "run_s": 0.0},
        )
        self._write_terminal(record, key_label=None, now=now, deadline_at=None)

    def _write_terminal(
        self,
        record: ResultRecord,
        *,
        key_label: str | None,
        now: float,
        deadline_at: float | None,
        telemetry: _Telemetry = _NO_TELEMETRY,
    ) -> None:
        """Write a result once; on failure keep it for retries on later ticks.

        The job files are deleted and the job is remembered as done only after
        the result is on disk (or the retries gave up), so a computed result is
        not lost to a transient write failure and its job is never re-run.
        *telemetry* is logged with the publication, also after a retry.
        """
        error = self._try_write_result(record)
        if error is None:
            self._finish_terminal(record, key_label=key_label, telemetry=telemetry)
            return
        give_up_at = now + max(0.0, self._options.result_retry_s)
        if deadline_at is not None:
            give_up_at = max(give_up_at, deadline_at)
        self._unpublished[record.job_id] = _Unpublished(
            record=record,
            key_label=key_label,
            give_up_at=give_up_at,
            telemetry=telemetry,
        )
        logger.warning(
            "cannot write result; retrying every tick for %.0f s: job_id=%s "
            "status=%s error=%s",
            give_up_at - now,
            record.job_id,
            record.status,
            error,
        )

    def _try_write_result(self, record: ResultRecord) -> OSError | None:
        """Write a result unless one exists; return the write error, if any."""
        try:
            created = write_result(self._paths, record)
        except OSError as exc:
            return exc
        if not created:
            logger.info("result already existed; kept: job_id=%s", record.job_id)
        return None

    def _retry_unpublished(self, now: float, *, final: bool = False) -> None:
        """Retry the results whose write failed; give up after their window.

        Args:
            now: Current clock value.
            final: Last try at supervisor exit: a result that still cannot be
                written keeps its job files (and attempt counter), so the next
                supervisor serves the job instead of the client losing it.

        """
        for job_id, entry in list(self._unpublished.items()):
            error = self._try_write_result(entry.record)
            if error is None:
                del self._unpublished[job_id]
                logger.info(
                    "result written after %d failed attempt(s): job_id=%s",
                    entry.failures,
                    job_id,
                )
                self._finish_terminal(
                    entry.record, key_label=entry.key_label, telemetry=entry.telemetry
                )
                continue
            entry.failures += 1
            if final:
                del self._unpublished[job_id]
                logger.error(
                    "result still unwritable at supervisor exit; job files kept "
                    "for the next supervisor: job_id=%s failures=%d error=%s",
                    job_id,
                    entry.failures,
                    error,
                )
            elif now >= entry.give_up_at:
                del self._unpublished[job_id]
                logger.error(
                    "result could not be written; job dropped: job_id=%s "
                    "status=%s failures=%d error=%s",
                    job_id,
                    entry.record.status,
                    entry.failures,
                    error,
                )
                delete_job_files(self._paths, job_id)
                self._remember_terminal(job_id)
            else:
                logger.debug(
                    "result write retry failed: job_id=%s failures=%d error=%s",
                    job_id,
                    entry.failures,
                    error,
                )

    def _finish_terminal(
        self,
        record: ResultRecord,
        *,
        key_label: str | None,
        telemetry: _Telemetry = _NO_TELEMETRY,
    ) -> None:
        """Delete the job files of a written result and remember the job as done.

        The ``result published`` line ends with the memory figures of
        *telemetry* (``None`` for an unknown figure, for example the CUDA
        figures of a CPU worker or every figure of a never admitted job).
        """
        delete_job_files(self._paths, record.job_id)
        self._remember_terminal(record.job_id)
        logger.info(
            "result published: job_id=%s status=%s error_kind=%s worker_id=%s "
            "key=%s attempts=%d %s",
            record.job_id,
            record.status,
            record.error_kind,
            record.worker_id,
            key_label,
            record.attempts,
            telemetry.log_fields(),
        )

    def _remember_terminal(self, job_id: str) -> None:
        """Add a job to the bounded set of finished jobs."""
        self._terminal[job_id] = None
        self._terminal.move_to_end(job_id)
        while len(self._terminal) > TERMINAL_MEMORY:
            self._terminal.popitem(last=False)
        self._unreadable.pop(job_id, None)

    def _requeue(self, job: _Job, *, now: float, restore: bool) -> None:
        """Put a running job back into the pending set (seq and wait kept).

        Args:
            job: Running job.
            now: Current clock value.
            restore: Give back the attempt the job did not cause to fail.

        """
        worker_id = job.worker_id
        self._detach(job, now)
        if restore and job.attempts > 0:
            job.attempts -= 1
            self._persist_attempts(job)
        logger.info(
            "job re-queued: job_id=%s worker_id=%s attempts=%d restored=%s "
            "force_backend=%s",
            job.job_id,
            worker_id,
            job.attempts,
            restore,
            job.pending.force_backend,
        )

    def _detach(self, job: _Job, now: float) -> None:
        """Release the job's worker slot and clear its running state."""
        worker_id = job.worker_id
        if worker_id is not None:
            worker = self._workers.get(worker_id)
            if worker is not None:
                worker.running.pop(job.job_id, None)
                worker.last_used_at = now
                if not worker.running:
                    worker.idle_since = now
        job.worker_id = None
        job.attempt_id = None
        job.started_at = None
        job.assignment = None
        job.cancel_reason = None
        job.cancel_sent_at = None

    def _persist_attempts(self, job: _Job) -> bool:
        """Write the job's attempt counter; return whether it was written."""
        try:
            write_attempts(self._paths, job.job_id, job.attempts)
        except OSError as exc:
            logger.warning(
                "cannot persist attempts: job_id=%s attempts=%d error=%s",
                job.job_id,
                job.attempts,
                exc,
            )
            return False
        return True

    # * Admission, cancellation and deadlines.

    def _scan(self, now: float) -> None:
        """Admit new job files, then apply cancel markers and deadlines."""
        ids = list_incoming_ids(self._paths)
        fresh: list[str] = []
        for job_id in ids:
            if job_id in self._jobs or job_id in self._unpublished:
                continue
            if job_id in self._terminal:
                delete_job_files(self._paths, job_id)
                continue
            fresh.append(job_id)
            if len(fresh) >= MAX_SCAN_BATCH:
                break
        self._admit(fresh, now)
        self._forget_vanished(set(ids))
        self._apply_cancels(now)

    def _admit(self, job_ids: list[str], now: float) -> None:
        """Read new job files and admit the valid ones by submission order."""
        records: list[JobRecord] = []
        for job_id in job_ids:
            record = self._read_record(job_id, now)
            if record is not None:
                records.append(record)
        records.sort(key=lambda record: (record.submitted_at, record.job_id))
        for record in records:
            self._admit_record(record, now)

    def _read_record(self, job_id: str, now: float) -> JobRecord | None:
        """Read and validate one job file; reject or defer what is not a job."""
        if has_result(self._paths, job_id):
            delete_job_files(self._paths, job_id)
            self._remember_terminal(job_id)
            logger.info("job already has a result; cleaned up: job_id=%s", job_id)
            return None
        payload = read_job_payload(self._paths, job_id)
        if payload is None:
            self._defer_unreadable(job_id, now)
            return None
        self._unreadable.pop(job_id, None)
        try:
            return JobRecord.from_dict(payload, expected_job_id=job_id)
        except JobValidationError as exc:
            logger.warning(
                "invalid job rejected: job_id=%s error_kind=%s detail=%s",
                job_id,
                exc.error_kind,
                exc.detail,
            )
            self._publish_file(job_id, exc.error_kind, exc.detail, now)
            return None

    def _defer_unreadable(self, job_id: str, now: float) -> None:
        """Retry an unreadable job file for 30 s after its first failed read.

        The window starts at the first failed read, not at the file's
        modification time, so an old job file hit by one transient read
        failure (a lock held by a scanner) is retried instead of rejected.
        """
        if job_file_age_s(self._paths, job_id, now=time.time()) is None:
            # * The file is gone (or cannot even be inspected): nothing to defer.
            self._unreadable.pop(job_id, None)
            return
        first_failed = self._unreadable.setdefault(job_id, now)
        if now - first_failed < UNREADABLE_RETRY_S:
            return
        self._unreadable.pop(job_id, None)
        detail = (
            f"job file stayed unreadable for {UNREADABLE_RETRY_S:.0f} s (partial, "
            "larger than 64 KiB or not a JSON object)"
        )
        logger.warning("unreadable job rejected: job_id=%s", job_id)
        self._publish_file(job_id, ERROR_KIND_INPUT_INVALID, detail, now)

    def _admit_record(self, record: JobRecord, now: float) -> None:
        """Admit one valid job (or end it when its attempts are used up)."""
        self._seq += 1
        request = record.request
        job = _Job(
            record=record,
            pending=PendingJob(
                job_id=record.job_id,
                call_id=record.call_id,
                seq=self._seq,
                submitted_at=record.submitted_at,
                waiting_since=now,
                deadline_at=record.deadline_at,
                provider=request.provider,
                whisper_model=request.whisper_model,
                compute_request=request.compute_type,
                device_request=request.device,
                duration_s=record.duration_s,
                max_parallel=record.max_parallel,
                priority=record.priority,
            ),
            attempts=read_attempts(self._paths, record.job_id),
        )
        max_attempts = self._config.scheduling.max_attempts
        if job.attempts >= max_attempts:
            detail = (
                f"the job already started {job.attempts} of {max_attempts} attempts "
                "(a previous supervisor stopped while it ran)"
            )
            outcome = _Outcome(
                status="error",
                error_kind=ERROR_KIND_RETRY_EXHAUSTED,
                error_detail=detail,
            )
            self._publish(job, outcome, worker=None, now=now)
            return
        if request.device == "cuda":
            presence = self._gpu.presence(now)
            if presence == "absent":
                self._force_cpu_without_cuda(job)
            elif presence == "unknown":
                # * The scheduler keeps a ``cuda`` job waiting while no CUDA
                # * device is known; the presence decision settles it later.
                self._cuda_waiting.add(record.job_id)
        self._jobs[record.job_id] = job
        logger.info(
            "job admitted: job_id=%s call_id=%s seq=%d kind=%s provider=%s model=%s "
            "device=%s priority=%s duration_s=%s attempts=%d",
            record.job_id,
            record.call_id,
            self._seq,
            record.kind,
            request.provider,
            request.whisper_model,
            request.device,
            record.priority,
            record.duration_s,
            job.attempts,
        )

    @staticmethod
    def _cpu_label(record: JobRecord) -> str:
        """Return the CPU worker key label of a job's request."""
        request = record.request
        try:
            key = make_key(
                request.provider, request.whisper_model, request.compute_type, "cpu"
            )
        except ValueError:
            return request.provider
        return key.label

    def _force_cpu_without_cuda(self, job: _Job) -> None:
        """Send a ``cuda`` job to CPU because CUDA is confirmed absent (spec 7.3)."""
        job.pending = replace(job.pending, force_backend="cpu")
        job.warnings.append(
            "ASKVLM_CUDA_UNAVAILABLE: no CUDA device detected; "
            f"running {self._cpu_label(job.record)} on CPU"
        )

    def _settle_cuda_waiting(self, now: float) -> None:
        """Decide the ``cuda`` jobs admitted while the GPU presence was unknown.

        ``present`` lets them run on the GPU as requested; ``absent`` sends the
        pending ones to CPU with the ``ASKVLM_CUDA_UNAVAILABLE`` warning.
        """
        presence = self._gpu.presence(now)
        if presence != self._gpu_presence:
            self._gpu_presence = presence
            logger.info(
                "GPU presence: %s (failed probes=%d, %.1f s after start)",
                presence,
                self._gpu.failed_probes,
                now - self._started_at,
            )
        if not self._cuda_waiting:
            return
        if presence == "unknown":
            self._cuda_waiting = {
                job_id for job_id in self._cuda_waiting if job_id in self._jobs
            }
            return
        waiting, self._cuda_waiting = self._cuda_waiting, set()
        if presence == "present":
            return
        for job in self._jobs.values():
            if (
                job.job_id in waiting
                and job.worker_id is None
                and job.pending.force_backend is None
            ):
                self._force_cpu_without_cuda(job)
                logger.warning(
                    "CUDA confirmed absent; cuda job runs on CPU: job_id=%s",
                    job.job_id,
                )

    def _forget_vanished(self, present: set[str]) -> None:
        """Drop pending jobs whose job file disappeared from ``incoming/``."""
        # * The per-file check keeps a failed directory listing (which reads as
        # * an empty directory) from dropping every pending job.
        vanished = [
            job_id
            for job_id, job in self._jobs.items()
            if job.worker_id is None
            and job_id not in present
            and not self._paths.incoming_file(job_id).exists()
        ]
        for job_id in vanished:
            del self._jobs[job_id]
            logger.warning("pending job file disappeared; job dropped: %s", job_id)
        for job_id in [j for j in self._unreadable if j not in present]:
            del self._unreadable[job_id]

    def _apply_cancels(self, now: float) -> None:
        """End cancelled or expired pending jobs; ask workers to stop running ones."""
        markers = self._cancel_markers()
        for job in list(self._jobs.values()):
            reason: CancelReason | None = None
            if job.job_id in markers:
                reason = "cancelled"
            elif job.record.deadline_at is not None and now >= job.record.deadline_at:
                reason = "expired"
            if reason is None:
                continue
            if job.worker_id is None:
                self._publish(
                    job,
                    _Outcome(status=reason, error_kind=reason),
                    worker=None,
                    now=now,
                )
            elif job.cancel_reason is None:
                self._send_cancel(job, reason, now)

    def _cancel_markers(self) -> set[str]:
        """Return the job ids that have a cancel marker."""
        try:
            with os.scandir(self._paths.cancel) as entries:
                return {entry.name for entry in entries if is_valid_id(entry.name)}
        except OSError as exc:
            logger.debug("cancel markers unreadable: %s", exc)
            return set()

    def _send_cancel(self, job: _Job, reason: CancelReason, now: float) -> None:
        """Send ``cancel`` for the job's attempt once and remember why."""
        job.cancel_reason = reason
        job.cancel_sent_at = now
        worker = self._workers.get(job.worker_id or "")
        if worker is None:
            return
        sent = worker.proc.send(
            {
                "type": ipc.MSG_CANCEL,
                "job_id": job.job_id,
                "attempt_id": job.attempt_id,
            }
        )
        logger.info(
            "cancel sent: job_id=%s attempt_id=%s worker_id=%s reason=%s sent=%s",
            job.job_id,
            job.attempt_id,
            worker.worker_id,
            reason,
            sent,
        )
        if not sent:
            self._kill(worker, "broken")

    # * Planning.

    def _plan(self, now: float) -> None:
        """Run the scheduler on the current state and execute its actions."""
        decision = plan(self._scheduler_state(now))
        for action in decision.actions:
            self._execute(action, now)

    def _scheduler_state(self, now: float) -> SchedulerState:
        """Build the immutable scheduler input of this tick."""
        active_calls = {job.record.call_id for job in self._jobs.values()}
        for call_id in [c for c in self._call_last_started if c not in active_calls]:
            del self._call_last_started[call_id]
        commit = self._options.commit_probe()
        free_vram, sample_ts = self._gpu.reading()
        return SchedulerState(
            now=now,
            limits=self._limits,
            probes=Probes(
                free_commit_bytes=commit.free_bytes if commit.known else None,
                free_vram_bytes=free_vram,
                vram_sample_ts=sample_ts,
                cuda_present=self._gpu.cuda_present,
            ),
            pending=self._plan_pending(),
            workers=tuple(self._view(worker) for worker in self._workers.values()),
            key_status={
                key: KeyStatus(
                    crash_cooldown_until=book.crash_cooldown_until,
                    load_failure_until=book.load_failure_until,
                    load_failure_detail=book.load_failure_detail,
                )
                for key, book in self._keys.items()
            },
            estimates=self._estimates,
            last_gpu_change_ts=self._last_gpu_change_ts,
            call_last_started=dict(self._call_last_started),
        )

    def _plan_pending(self) -> tuple[PendingJob, ...]:
        """Return the oldest pending jobs by admission plus every ``high`` one.

        ``_jobs`` keeps admission order (re-queued jobs keep their place), so
        the pending jobs come out by ``seq``.
        """
        pending = [job.pending for job in self._jobs.values() if job.worker_id is None]
        limit = max(1, self._options.max_plan_jobs)
        if len(pending) <= limit:
            return tuple(pending)
        extra = [job for job in pending[limit:] if job.priority == "high"]
        return (*pending[:limit], *extra)

    def _view(self, worker: _Worker) -> WorkerView:
        """Describe a worker for the scheduler."""
        running: list[RunningJob] = []
        for job_id in worker.running:
            job = self._jobs.get(job_id)
            if job is None or job.started_at is None:
                continue
            service = job.assignment.expected_service_s if job.assignment else 0.0
            running.append(
                RunningJob(
                    job_id=job_id,
                    call_id=job.record.call_id,
                    started_at=job.started_at,
                    expected_end_at=job.started_at + service,
                )
            )
        return WorkerView(
            worker_id=worker.worker_id,
            key=worker.key,
            state=worker.state,
            capacity=worker.capacity,
            running=tuple(running),
            idle_since=None if worker.running else worker.idle_since,
            spawned_at=worker.spawned_at,
            last_used_at=worker.last_used_at,
            reservation=worker.reservation,
        )

    def _execute(self, action: Action, now: float) -> None:
        """Execute one scheduler action."""
        if isinstance(action, Reject):
            job = self._jobs.get(action.job_id)
            if job is not None and job.worker_id is None:
                logger.warning(
                    "job rejected by the scheduler: job_id=%s status=%s "
                    "error_kind=%s detail=%s",
                    action.job_id,
                    action.status,
                    action.error_kind,
                    action.detail,
                )
                outcome = _Outcome(
                    status=action.status,
                    error_kind=action.error_kind,
                    error_detail=action.detail,
                )
                self._publish(job, outcome, worker=None, now=now)
        elif isinstance(action, Stop):
            worker = self._workers.get(action.worker_id)
            if (
                worker is not None
                and worker.state in _IDLE_STOP_STATES
                and not worker.running
            ):
                self._stop_worker(worker, action.reason, now)
        elif isinstance(action, Assign):
            self._assign(action, now)
        else:
            self._spawn(action, now)

    def _assign(self, action: Assign, now: float) -> None:
        """Persist the next attempt, then send the job to its worker."""
        job = self._jobs.get(action.job_id)
        worker = self._workers.get(action.worker_id)
        if (
            job is None
            or job.worker_id is not None
            or worker is None
            or worker.state != "ready"
            or len(worker.running) >= worker.capacity
        ):
            logger.warning(
                "stale assignment skipped: job_id=%s worker_id=%s",
                action.job_id,
                action.worker_id,
            )
            return
        job.attempts += 1
        if not self._persist_attempts(job):
            job.attempts -= 1
            return
        attempt_id = f"{job.job_id}:{job.attempts}"
        record = job.record
        message: dict[str, object] = {
            "type": ipc.MSG_JOB,
            "job_id": job.job_id,
            "attempt_id": attempt_id,
            "input_path": record.input_path,
            "language": record.language,
            "duration_s": record.duration_s,
            "deadline_at": record.deadline_at,
            "priority": record.priority,
            "call_id": record.call_id,
            "seq": job.pending.seq,
            **job_message_fields(
                kind=record.kind,
                want_segments=record.want_segments,
                decode=record.decode,
                window=record.window,
                diarize=record.diarize,
            ),
        }
        if not worker.proc.send(message):
            logger.warning(
                "job send failed; worker treated as dead: job_id=%s worker_id=%s",
                job.job_id,
                worker.worker_id,
            )
            job.attempts -= 1
            self._persist_attempts(job)
            self._kill(worker, "broken")
            return
        job.worker_id = worker.worker_id
        job.attempt_id = attempt_id
        job.started_at = now
        job.assignment = action
        worker.running[job.job_id] = None
        worker.last_used_at = now
        worker.idle_since = None
        self._call_last_started[record.call_id] = now
        logger.info(
            "job assigned: job_id=%s attempt_id=%s worker_id=%s key=%s expected_s=%.1f",
            job.job_id,
            attempt_id,
            worker.worker_id,
            worker.key.label,
            action.expected_service_s,
        )

    # * Worker processes.

    def _spawn(self, action: Spawn, now: float) -> None:
        """Start a worker for *action*; a failed start counts as a crash.

        A worker of a key whose failed cold load has not exited yet is not
        started: the failed process still holds its reservation, and a
        load-failure cooldown shorter than its teardown must not let a
        replacement load next to it. The job waits for a later tick.
        """
        key = action.key
        if self._load_failure_unconfirmed(key):
            logger.debug(
                "spawn deferred until the failed cold load exits: key=%s "
                "reason_job_id=%s",
                key.label,
                action.reason_job_id,
            )
            return
        self._worker_counter += 1
        worker_id = f"{self._instance_id[:8]}-{self._worker_counter:04d}"
        spec = WorkerSpec(
            worker_id=worker_id,
            instance_id=self._instance_id,
            key=key,
            capacity=action.capacity,
            threads_per_slot=threads_for(key, self._limits),
            host_threads=action.reservation.host_threads,
            device=device_for_backend(key.backend),
            options=worker_spec_options(
                self._config, self._limits, key, action.capacity, action.reservation
            ),
            engine_factory=self._options.engine_factory,
        )
        proc = self._start_process(spec, action.reservation)
        if proc is None:
            # * The reservation is released with the action: nothing was added.
            self._record_crash(key, now)
            return
        self._workers[worker_id] = _Worker(
            worker_id=worker_id,
            key=key,
            capacity=action.capacity,
            reservation=action.reservation,
            proc=proc,
            spawned_at=now,
            last_used_at=now,
        )
        if key.backend == "gpu":
            self._last_gpu_change_ts = now
        logger.info(
            "worker spawned: worker_id=%s key=%s pid=%s capacity=%d commit_bytes=%d "
            "job_limit_bytes=%d vram_bytes=%d host_threads=%d reason_job_id=%s",
            worker_id,
            key.label,
            proc.pid,
            action.capacity,
            action.reservation.commit_bytes,
            self._job_limit_bytes(action.reservation),
            action.reservation.vram_bytes,
            action.reservation.host_threads,
            action.reason_job_id,
        )

    def _job_limit_bytes(self, reservation: Reservation) -> int:
        """Return the Job Object commit cap of a worker with *reservation*.

        The cap is the commit reservation plus ``memory.child_headroom_gib``,
        because child processes such as ffmpeg share the job's cap. The
        headroom is not reserved: the scheduler and the commit budget count
        only the reservation.

        Returns:
            The cap in bytes, or ``0`` (no cap) for a zero reservation.

        """
        if reservation.commit_bytes <= 0:
            return 0
        return reservation.commit_bytes + self._config.memory.child_headroom_bytes

    def _start_process(
        self, spec: WorkerSpec, reservation: Reservation
    ) -> WorkerProcess | None:
        """Start a worker process in its Job Object and send its spec.

        Returns:
            The running handle, or ``None`` when the process could not be
            started safely (everything it created is closed).

        """
        key = spec.key
        limit = self._job_limit_bytes(reservation)
        try:
            proc = WorkerProcess(
                spec.worker_id,
                self._launch(
                    key,
                    worker_environment(self._config, self._limits, key, reservation),
                ),
                log_path=self._paths.worker_log(spec.worker_id),
                memory_limit_bytes=limit if limit > 0 else None,
                on_event=self._on_worker_event,
                on_memory_limit=self._on_memory_limit,
                require_job=self._options.require_job,
            )
        except (OSError, ValueError) as exc:
            logger.warning(
                "worker launch failed: worker_id=%s key=%s error=%s",
                spec.worker_id,
                key.label,
                exc,
            )
            return None
        try:
            proc.start(spec.to_message())
        except (WorkerSpawnError, OSError, TypeError, ValueError) as exc:
            logger.warning(
                "worker spawn failed: worker_id=%s key=%s error=%s",
                spec.worker_id,
                key.label,
                exc,
            )
            proc.close(timeout_s=CLOSE_TIMEOUT_S)
            return None
        return proc

    def _launch(self, key: WorkerKey, env: Mapping[str, str]) -> WorkerLaunch:
        """Return the launch of a worker (injected or production)."""
        factory = self._options.worker_launch
        if factory is not None:
            return factory(key, env)
        return default_worker_launch(project_root=project_root(), extra_env=env)

    def _stop_worker(self, worker: _Worker, reason: str, now: float) -> None:
        """Ask a worker to shut down; kill it when the request cannot be sent."""
        if worker.stop_reason is not None:
            return
        worker.stop_reason = reason
        worker.state = "stopping"
        worker.stop_deadline = now + self._config.lifecycle.stop_grace_s
        sent = worker.proc.request_shutdown()
        logger.info(
            "worker stopping: worker_id=%s key=%s reason=%s running=%d sent=%s",
            worker.worker_id,
            worker.key.label,
            reason,
            len(worker.running),
            sent,
        )
        if not sent:
            self._kill(worker, "shutdown_unsent")

    def _kill(self, worker: _Worker, reason: str) -> None:
        """Kill a worker's process tree once (its exit is confirmed by EOF)."""
        if worker.kill_reason is None:
            worker.kill_reason = reason
        worker.state = "stopping"
        if worker.killed_mono is not None:
            return
        worker.killed_mono = time.monotonic()
        logger.warning(
            "killing worker: worker_id=%s key=%s reason=%s running=%d",
            worker.worker_id,
            worker.key.label,
            reason,
            len(worker.running),
        )
        worker.proc.kill()

    def _finalize(self, worker: _Worker) -> None:
        """Confirm a worker's exit, settle its jobs, then release it."""
        if self._workers.get(worker.worker_id) is not worker:
            return
        proc = worker.proc
        code = proc.wait(EXIT_WAIT_S)
        proc.close(timeout_s=CLOSE_TIMEOUT_S)
        if code is None:
            code = proc.poll()
        peak = proc.peak_memory_bytes()
        limit_hit = self._memory_limit_hit(worker, peak)
        tail = proc.log_tail(LOG_TAIL_CHARS).replace("\r\n", "\n")
        now = self._clock()
        kind = _exit_kind(worker)
        self._settle_jobs(
            worker,
            kind,
            self._crash_outcome(worker, code, tail, limit_hit=limit_hit),
            now,
            limit_hit=limit_hit,
        )
        if kind == "load_failed":
            self._record_load_failure(
                worker, self._load_detail(worker, code, tail), now
            )
            fatal = worker.fatal
            if fatal is not None and fatal.error_kind == ERROR_KIND_CUDA_OOM:
                self._bump_memory_once(worker, vram=True, reason="load cuda_oom")
        elif kind == "crashed":
            self._record_crash(worker.key, now)
        # * The peak still raises the learned commit floor; the x1.25 growth of
        # * a cap hit applies only when this worker has not grown the commit
        # * estimate already (the cap report itself, a job's ``memory_limit``
        # * error, a CUDA OOM).
        self._estimates.observe_worker_exit(
            worker.key,
            capacity=worker.capacity,
            peak_commit_bytes=peak,
            memory_limit_hit=limit_hit and not worker.commit_bumped,
        )
        del self._workers[worker.worker_id]
        if worker.key.backend == "gpu":
            self._last_gpu_change_ts = now
            self._gpu.refresh()
        logger.log(
            logging.INFO if kind == "stopped" else logging.WARNING,
            "worker exited: worker_id=%s key=%s kind=%s code=%s bye=%s "
            "stop_reason=%s kill_reason=%s peak_bytes=%s limit_hit=%s",
            worker.worker_id,
            worker.key.label,
            kind,
            code,
            worker.bye,
            worker.stop_reason,
            worker.kill_reason,
            peak,
            limit_hit,
        )

    def _cap_reported(self, worker: _Worker) -> bool:
        """Return whether the worker's Job Object reported its current commit cap.

        The handle's flag is set on the monitor thread before the event is
        queued, so it also covers a report whose event is still waiting behind
        the worker's result. After a live raise only a report of the raised
        cap counts: the report that led to the raise was answered by it, so a
        later unrelated exit is not blamed on memory (its failed pre-raise
        attempts already retried, see :meth:`_raise_covers`).
        """
        if worker.cap_raised_at is not None:
            return self._cap_hits(worker) > worker.cap_hits_at_raise
        return worker.memory_limit_reported or worker.proc.memory_limit_hit

    @staticmethod
    def _cap_hits(worker: _Worker) -> int:
        """Return how many commit-cap reports the worker's Job Object made.

        The handle counts a report on its monitor thread before the event is
        queued, so the count also covers reports whose events still wait.
        """
        return max(worker.cap_events, worker.proc.memory_limit_hits)

    def _raise_covers(self, worker: _Worker, job: _Job) -> bool:
        """Return whether *job*'s attempt ran when the worker's cap was raised live.

        Such an attempt may have failed under the old cap, and no report of
        the raised cap followed: its failure does not retire the worker.
        """
        return (
            worker.cap_raised_at is not None
            and self._cap_hits(worker) <= worker.cap_hits_at_raise
            and job.attempt_id in worker.pre_raise_attempts
        )

    def _ran_under_reported_cap(self, worker: _Worker, job: _Job) -> bool:
        """Return whether *job*'s attempt ran under a commit cap that was reported.

        Without a live raise any report of the worker counts. After a raise, a
        report of the raised cap counts for every attempt, and the report
        that led to the raise only for the attempts that ran at the raise
        (the raise does not erase their failure).
        """
        hits = self._cap_hits(worker)
        if hits == 0:
            return False
        if worker.cap_raised_at is None or hits > worker.cap_hits_at_raise:
            return True
        return job.attempt_id in worker.pre_raise_attempts

    def _memory_limit_hit(self, worker: _Worker, peak: int | None) -> bool:
        """Return whether the worker reached its Job Object commit cap.

        A report of the current cap (:meth:`_cap_reported`; after a live raise,
        a report of the raised cap) or a peak at
        :data:`MEMORY_LIMIT_PEAK_RATIO` of the current (raised) cap counts.
        The verdict decides how an exit is classified and whether its
        re-queued jobs are pinned to the worker's backend.
        """
        if self._cap_reported(worker):
            return True
        limit = self._job_limit_bytes(worker.reservation)
        return (
            peak is not None and limit > 0 and peak >= MEMORY_LIMIT_PEAK_RATIO * limit
        )

    def _bump_memory_once(self, worker: _Worker, *, vram: bool, reason: str) -> None:
        """Grow the key's memory estimates x1.25 at most once per worker.

        The commit estimate grows on the worker's first memory event, the VRAM
        estimate (when *vram*) on its first GPU memory event; later events of
        the same worker, its exit included, add no further growth.

        Args:
            worker: Worker whose reservation proved too small.
            vram: Whether the VRAM estimate grows too (a CUDA OOM).
            reason: Event named in the log.

        """
        commit = not worker.commit_bumped
        vram = vram and not worker.vram_bumped
        if not (commit or vram):
            return
        worker.commit_bumped = True
        worker.vram_bumped = worker.vram_bumped or vram
        self._estimates.bump_memory(worker.key, commit=commit, vram=vram)
        estimate = self._estimates.estimate(worker.key)
        logger.warning(
            "memory estimate grown: worker_id=%s key=%s reason=%s commit=%s "
            "vram=%s commit_base_bytes=%d commit_per_slot_bytes=%d "
            "vram_base_bytes=%d vram_per_lane_bytes=%d",
            worker.worker_id,
            worker.key.label,
            reason,
            commit,
            vram,
            estimate.commit_base_bytes,
            estimate.commit_per_slot_bytes,
            estimate.vram_base_bytes,
            estimate.vram_per_lane_bytes,
        )

    def _settle_jobs(
        self,
        worker: _Worker,
        kind: ExitKind,
        crash: _Outcome,
        now: float,
        *,
        limit_hit: bool,
    ) -> None:
        """Decide the fate of the jobs that ran on an exited worker.

        A job with a pending cancel ends as ``cancelled``/``expired``. After a
        stop the supervisor asked for, or a kill caused by another job, the
        jobs are victims: re-queued with the attempt given back. After a crash
        a job retries while attempts remain, else it ends with *crash*. When
        the worker reached its commit cap (*limit_hit*), the crash is a
        ``memory_limit`` one and every job it re-queues is pinned to the
        worker's backend, ``auto`` jobs included, like a job that failed with
        ``memory_limit`` itself. A victim of a supervisor stop or of a kill
        caused by another job keeps its ``force_backend`` even when the cap was
        reached during the stop, because the worker's memory did not fail it.
        """
        for job_id in list(worker.running):
            job = self._jobs.get(job_id)
            if job is None:
                worker.running.pop(job_id, None)
                continue
            if job.cancel_reason is not None:
                reason = job.cancel_reason
                outcome = _Outcome(status=reason, error_kind=reason)
                self._publish(job, outcome, worker=worker, now=now)
                continue
            restore = kind in {"stopped", "cancel_kill"}
            if not restore and job.attempts >= self._config.scheduling.max_attempts:
                self._publish(job, crash, worker=worker, now=now)
                continue
            if limit_hit and not restore:
                job.pending = replace(job.pending, force_backend=worker.key.backend)
            self._requeue(job, now=now, restore=restore)

    def _crash_outcome(
        self, worker: _Worker, code: int | None, tail: str, *, limit_hit: bool
    ) -> _Outcome:
        """Return the ``error`` outcome of a job whose worker crashed."""
        summary = (
            f"worker {worker.worker_id} ({worker.key.label}) exited unexpectedly "
            f"with code {code}"
        )
        fatal = worker.fatal
        if fatal is not None:
            summary += (
                f"; fatal {fatal.stage} error {fatal.error_kind}: "
                f"{_clip(fatal.detail or '', FATAL_DETAIL_CHARS)}"
            )
        if limit_hit:
            limit_gib = self._job_limit_bytes(worker.reservation) / _GIB
            summary += f"; it reached its memory limit of {limit_gib:.2f} GiB"
        if worker.kill_reason == "broken":
            summary += "; its protocol pipe was broken"
        detail = f"{summary}\nworker log tail:\n{tail}" if tail else summary
        return _Outcome(
            status="error",
            error_kind=ERROR_KIND_MEMORY_LIMIT
            if limit_hit
            else ERROR_KIND_WORKER_CRASHED,
            error_detail=detail,
        )

    def _load_detail(self, worker: _Worker, code: int | None, tail: str) -> str:
        """Describe why a worker failed to load."""
        fatal = worker.fatal
        if fatal is not None and fatal.detail:
            return fatal.detail
        if worker.kill_reason == "load_timeout":
            base = (
                "worker did not finish loading within "
                f"{self._options.load_timeout_s:.0f} s"
            )
        elif worker.kill_reason == "protocol":
            base = "worker sent an invalid 'loaded' message"
        else:
            base = f"worker exited with code {code} before it finished loading"
        if tail:
            base += f"; log tail: {tail[-LOAD_TAIL_CHARS:]}"
        return base

    def _book(self, key: WorkerKey) -> _KeyBook:
        """Return the bookkeeping of *key*, created on first use."""
        book = self._keys.get(key)
        if book is None:
            book = self._keys[key] = _KeyBook()
        return book

    def _fail_load(self, worker: _Worker, reason: str, now: float) -> None:
        """Kill a worker whose cold load failed and cool its key down at once.

        The cooldown starts with the kill decision, not with the confirmed
        exit, so the scheduler at once rejects the jobs only this key could
        serve (``auto`` jobs take the other backend) instead of spawning a
        replacement while the killed process still holds its reservation.

        Args:
            worker: Worker that has not sent a valid ``loaded``.
            reason: Kill reason (``load_timeout`` or ``protocol``).
            now: Current clock value.

        """
        self._kill(worker, reason)
        tail = worker.proc.log_tail(LOG_TAIL_CHARS).replace("\r\n", "\n")
        self._record_load_failure(worker, self._load_detail(worker, None, tail), now)

    def _load_failure_unconfirmed(self, key: WorkerKey) -> bool:
        """Return whether a worker of *key* failed its load but has not exited."""
        return any(
            worker.key == key and worker.load_failure_recorded
            for worker in self._workers.values()
        )

    def _record_load_failure(self, worker: _Worker, detail: str, now: float) -> None:
        """Start the load-failure cooldown of the worker's key (once per worker)."""
        if worker.load_failure_recorded:
            return
        worker.load_failure_recorded = True
        book = self._book(worker.key)
        cooldown = self._config.scheduling.load_failure_cooldown_s
        book.load_failure_until = now + cooldown
        book.load_failure_detail = detail
        logger.warning(
            "worker load failed; key cooling down for %.0f s: worker_id=%s key=%s "
            "detail=%s",
            cooldown,
            worker.worker_id,
            worker.key.label,
            detail,
        )

    def _record_crash(self, key: WorkerKey, now: float) -> None:
        """Count a crash; three within ten minutes start a doubling cooldown."""
        book = self._book(key)
        book.crash_times.append(now)
        while book.crash_times and now - book.crash_times[0] > CRASH_WINDOW_S:
            book.crash_times.popleft()
        if len(book.crash_times) < CRASHES_FOR_COOLDOWN:
            return
        book.crash_times.clear()
        book.crash_level += 1
        base = self._config.scheduling.crash_cooldown_s
        cooldown = min(
            max(base, MAX_CRASH_COOLDOWN_S), base * 2 ** (book.crash_level - 1)
        )
        book.crash_cooldown_until = now + cooldown
        logger.warning(
            "worker key crashed repeatedly; cooling down for %.0f s: key=%s level=%d",
            cooldown,
            key.label,
            book.crash_level,
        )

    # * Housekeeping.

    def _housekeeping(self, now: float, mono: float) -> None:
        """Enforce load, stop and cancel timeouts, the uptime cap and purges."""
        for worker in list(self._workers.values()):
            self._check_max_uptime(worker, now)
            self._check_worker_timers(worker, now, mono)
        self._check_cancel_grace(now)
        if mono >= self._next_purge:
            self._next_purge = mono + self._options.purge_interval_s
            self._housekeep_files()

    def _check_max_uptime(self, worker: _Worker, now: float) -> None:
        """Retire a ready worker whose uptime since spawn reached the cap.

        ``lifecycle.worker_max_uptime_h`` bounds whatever growth no other
        limit covers (the CUDA context, CTranslate2's own allocator, native
        leaks below the recycle threshold). A retiring worker takes no new
        jobs, finishes the running ones and stops once idle, so no job is
        interrupted; the uptime counts from the spawn and a worker leaves
        ``ready`` once, so each worker is replaced at most once per uptime
        cap. ``0`` disables the cap.

        Args:
            worker: Worker to check.
            now: Current clock value.

        """
        max_uptime = self._config.lifecycle.worker_max_uptime_s
        if max_uptime <= 0 or worker.state != "ready":
            return
        uptime = now - worker.spawned_at
        if uptime < max_uptime:
            return
        worker.state = "retiring"
        logger.info(
            "worker retiring: it reached the maximum uptime: worker_id=%s key=%s "
            "uptime_s=%.0f max_uptime_s=%.0f running=%d",
            worker.worker_id,
            worker.key.label,
            uptime,
            max_uptime,
            len(worker.running),
        )

    def _check_worker_timers(self, worker: _Worker, now: float, mono: float) -> None:
        """Kill a worker past its load or stop deadline; finalize a lost exit."""
        if worker.kill_reason is None:
            if (
                worker.state in _LOADING_STATES
                and now - worker.spawned_at >= self._options.load_timeout_s
            ):
                self._fail_load(worker, "load_timeout", now)
            elif (
                worker.state == "stopping"
                and worker.stop_deadline is not None
                and now >= worker.stop_deadline
            ):
                self._kill(worker, "stop_grace")
        if worker.overdue_since is None:
            if worker.killed_mono is not None or worker.proc.poll() is not None:
                worker.overdue_since = mono
            return
        if mono - worker.overdue_since >= LOST_EXIT_TIMEOUT_S:
            logger.warning(
                "worker exit not reported by its pipe; finalizing: worker_id=%s",
                worker.worker_id,
            )
            self._finalize(worker)

    def _check_cancel_grace(self, now: float) -> None:
        """Kill workers whose cancelled jobs did not answer within the grace."""
        grace = self._config.lifecycle.cancel_grace_s
        for worker in list(self._workers.values()):
            if worker.killed_mono is not None:
                continue
            for job_id in worker.running:
                job = self._jobs.get(job_id)
                if job is None or job.cancel_sent_at is None:
                    continue
                if now - job.cancel_sent_at >= grace:
                    logger.warning(
                        "job ignored its cancel for %.0f s: job_id=%s worker_id=%s",
                        grace,
                        job_id,
                        worker.worker_id,
                    )
                    self._kill(worker, "cancel_grace")
                    break

    def _housekeep_files(self) -> None:
        """Purge stale queue files and keep the newest worker logs."""
        purged = purge_stale(self._paths, now=time.time())
        if purged:
            logger.info("purged stale queue files of %d job(s)", len(purged))
        self._prune_worker_logs()

    def _prune_worker_logs(self) -> None:
        """Delete all but the newest worker logs (logs of live workers stay)."""
        live = {f"{worker_id}.log" for worker_id in self._workers}
        logs: list[tuple[float, str]] = []
        try:
            with os.scandir(self._paths.logs) as entries:
                for entry in entries:
                    if not entry.name.endswith(".log") or entry.name in live:
                        continue
                    try:
                        logs.append((entry.stat().st_mtime, entry.name))
                    except OSError:
                        continue
        except OSError as exc:
            logger.debug("worker logs unreadable: %s", exc)
            return
        logs.sort(reverse=True)
        for _mtime, name in logs[KEEP_WORKER_LOGS:]:
            try:
                (self._paths.logs / name).unlink(missing_ok=True)
            except OSError as exc:
                logger.debug("cannot delete worker log %s: %s", name, exc)

    # * Heartbeat and shutdown.

    def _write_heartbeat(self, state: str) -> None:
        """Write the heartbeat (failures are logged, never fatal)."""
        self._heartbeat_seq += 1
        self._next_heartbeat = time.monotonic() + self._options.heartbeat_interval_s
        running = sum(1 for job in self._jobs.values() if job.worker_id is not None)
        payload: dict[str, object] = {
            "instance_id": self._instance_id,
            "started_at": self._started_at,
            "sequence": self._heartbeat_seq,
            "state": state,
            "config_source": self._config.source,
            "limits": config_summary(self._config),
            "workers": [
                {
                    "worker_id": worker.worker_id,
                    "key": worker.key.to_dict(),
                    "state": worker.state,
                    "capacity": worker.capacity,
                    "running": len(worker.running),
                    "pid": worker.proc.pid,
                    "spawned_at": worker.spawned_at,
                    "commit_bytes": worker.reservation.commit_bytes,
                    "job_limit_bytes": self._job_limit_bytes(worker.reservation),
                    "cap_raised": worker.cap_raised_at is not None,
                }
                for worker in self._workers.values()
            ],
            "queue": {"pending": len(self._jobs) - running, "running": running},
        }
        try:
            write_heartbeat(self._paths, payload)
        except OSError as exc:
            logger.warning("cannot write heartbeat: %s", exc)

    def _shutdown_workers(self) -> None:
        """Stop every worker (bounded), keeping unfinished jobs for a successor."""
        self._write_heartbeat("stopping")
        now = self._clock()
        for worker in list(self._workers.values()):
            self._stop_worker(worker, "shutdown", now)
        self._await_exits(self._config.lifecycle.stop_grace_s)
        for worker in list(self._workers.values()):
            self._kill(worker, "shutdown")
        self._await_exits(SHUTDOWN_KILL_WAIT_S)
        for worker in list(self._workers.values()):
            logger.warning(
                "worker exit not confirmed after the kill: worker_id=%s",
                worker.worker_id,
            )
            self._finalize(worker)
        logger.info(
            "workers stopped; %d unfinished job(s) stay in incoming", len(self._jobs)
        )

    def _await_exits(self, timeout_s: float) -> None:
        """Handle worker events until every worker exited or *timeout_s* passed."""
        deadline = time.monotonic() + max(0.0, timeout_s)
        while self._workers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self._pump_events(min(self._options.tick_s, remaining))
