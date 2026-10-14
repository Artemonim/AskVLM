"""Pure scheduler of the STT worker pool (spec sections 7 and 14).

:func:`plan` turns one immutable :class:`SchedulerState` snapshot into a
:class:`Plan` of actions for the supervisor: reject jobs that can never run,
stop workers, assign jobs to ready workers and spawn new workers. The module
performs no I/O, never reads the clock (time is ``state.now``), uses no
randomness and no threads, so the same state always yields the same plan.

Pending jobs are walked in the admission order of spec section 14
(:func:`admission_order`): priority classes first, where a job rises one class
per ``priority_aging_s`` of waiting (:func:`effective_rank`), then a
round-robin across the calls of one class, so a single-file call is not queued
behind every input of a batch. The order decides only which pending job gets
the next free slot, lane, spawn or eviction; running jobs are never preempted
and busy workers are never stopped.

Terminology (spec 6.5): a CPU *slot* is one concurrently running CPU job; a GPU
*worker* is one GPU process that batches up to its *lanes*.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Protocol, Self, TypeAlias

from core.stt_pool.config import DEFAULT_PRIORITY_AGING_S
from core.stt_pool.keys import (
    DEFAULT_PRIORITY,
    PRIORITY_RANK,
    allowed_backends,
    make_key,
    provider_family,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from core.stt_pool.config import PoolConfig
    from core.stt_pool.keys import Backend, Priority, WorkerKey

WorkerState = Literal["spawning", "loading", "ready", "retiring", "stopping"]
StopReason = Literal["ttl", "evict", "retired", "pressure"]
RejectStatus = Literal["unavailable", "error"]

# * ``error_kind`` values of scheduler rejections (spec 6.2).
ERROR_KIND_INFEASIBLE = "infeasible"
ERROR_KIND_WORKER_LOAD_FAILED = "worker_load_failed"
ERROR_KIND_INPUT_INVALID = "input_invalid"

# * Duration assumed for backlog and expected-end arithmetic of a job whose
# * audio length is unknown; never used for the CPU spill decision.
DEFAULT_UNKNOWN_DURATION_S = 60.0

# * Scale-out hysteresis of GPU workers: a new or still loading worker is
# * preferred over joining a backlogged ready worker of the same key only when
# * its ETA is below this fraction of the join ETA, so a marginal gain never
# * pays for a cold load.
SCALE_OUT_MARGIN = 0.8

_GIB = 1024**3
_MIN_XRT = 1e-6
_LOADING_STATES: frozenset[str] = frozenset({"spawning", "loading"})
_SERVING_STATES: frozenset[str] = frozenset({"spawning", "loading", "ready"})


# * Public data contracts (spec 7.1).


@dataclass(frozen=True, slots=True)
class Limits:
    """Numeric limits of the pool, taken from :class:`PoolConfig`.

    The per-family mappings are stored as read-only views over private copies.

    Attributes:
        cpu_slots: Concurrent CPU jobs across all CPU workers.
        threads_per_slot: CPU threads of one running CPU job.
        cpu_max_slots_per_worker: Largest slot capacity of one CPU worker.
        gpu_workers: Concurrent GPU worker processes.
        batch_lanes: Lanes (batch size) of one GPU worker per provider family.
        host_threads: CPU threads reserved for one GPU worker per family.
        thread_budget: CPU threads shared by CPU jobs and GPU host threads.
        commit_budget_bytes: Commit the pool may reserve in total.
        free_commit_guard_bytes: Free system commit that must remain after a spawn.
        critical_free_commit_bytes: Free system commit below which idle workers
            are stopped and nothing is spawned.
        vram_budget_bytes: VRAM the pool may reserve in total.
        free_vram_guard_bytes: Free VRAM that must remain after a GPU spawn.
        worker_idle_ttl_s: Idle time after which a ready worker is stopped.
        spill_ratio: An ``auto`` job spills to CPU when its CPU ETA is below
            this fraction of its GPU ETA.
        starvation_after_s: Wait after which the first qualifying job in
            admission order is protected.
        gpu_cpu_contention: Slowdown of GPU service per unit of CPU thread use.
        scale_out_backlog_s: Predicted remaining work of a GPU worker that
            justifies one more GPU worker of the same key.
        unknown_duration_s: Duration assumed for backlog and expected-end
            arithmetic of jobs without a known duration.
        priority_aging_s: Waiting time that lifts a pending job by one
            priority class (spec 14, :func:`effective_rank`).

    """

    cpu_slots: int
    threads_per_slot: int
    cpu_max_slots_per_worker: int
    gpu_workers: int
    batch_lanes: Mapping[str, int]
    host_threads: Mapping[str, int]
    thread_budget: int
    commit_budget_bytes: int
    free_commit_guard_bytes: int
    critical_free_commit_bytes: int
    vram_budget_bytes: int
    free_vram_guard_bytes: int
    worker_idle_ttl_s: float
    spill_ratio: float
    starvation_after_s: float
    gpu_cpu_contention: float
    scale_out_backlog_s: float
    unknown_duration_s: float = DEFAULT_UNKNOWN_DURATION_S
    priority_aging_s: float = DEFAULT_PRIORITY_AGING_S

    def __post_init__(self) -> None:
        """Replace the per-family mappings with read-only views of copies."""
        object.__setattr__(
            self, "batch_lanes", MappingProxyType(dict(self.batch_lanes))
        )
        object.__setattr__(
            self, "host_threads", MappingProxyType(dict(self.host_threads))
        )

    @classmethod
    def from_config(cls, config: PoolConfig) -> Self:
        """Build the limits from a validated pool configuration.

        Args:
            config: Validated pool configuration.

        Returns:
            The scheduler limits; ``unknown_duration_s`` keeps its default.

        """
        return cls(
            cpu_slots=config.cpu.max_slots,
            threads_per_slot=config.cpu.threads_per_slot,
            cpu_max_slots_per_worker=config.cpu.max_slots_per_worker,
            gpu_workers=config.gpu.max_workers,
            batch_lanes={
                str(family): lanes for family, lanes in config.gpu.batch_lanes.items()
            },
            host_threads={
                str(family): threads
                for family, threads in config.gpu.host_threads.items()
            },
            thread_budget=config.threads.budget,
            commit_budget_bytes=config.memory.commit_budget_bytes,
            free_commit_guard_bytes=config.memory.free_commit_guard_bytes,
            critical_free_commit_bytes=config.memory.critical_free_commit_bytes,
            vram_budget_bytes=config.vram.budget_bytes,
            free_vram_guard_bytes=config.vram.free_guard_bytes,
            worker_idle_ttl_s=config.lifecycle.worker_idle_ttl_s,
            spill_ratio=config.scheduling.spill_ratio,
            starvation_after_s=config.scheduling.starvation_after_s,
            gpu_cpu_contention=config.scheduling.gpu_cpu_contention,
            scale_out_backlog_s=config.gpu.scale_out_backlog_s,
            priority_aging_s=config.scheduling.priority_aging_s,
        )


@dataclass(frozen=True, slots=True)
class Probes:
    """Host readings sampled by the supervisor.

    Attributes:
        free_commit_bytes: Free system commit, or ``None`` when unknown.
        free_vram_bytes: Free VRAM of the GPU, or ``None`` when unknown.
        vram_sample_ts: Clock value of the VRAM reading, or ``None``.
        cuda_present: Whether a CUDA GPU is present.

    """

    free_commit_bytes: int | None
    free_vram_bytes: int | None
    vram_sample_ts: float | None
    cuda_present: bool


@dataclass(frozen=True, slots=True)
class Reservation:
    """Resources held by one worker from spawn until its exit is confirmed.

    Attributes:
        commit_bytes: Reserved Windows commit (also the Job Object memory limit).
        vram_bytes: Reserved VRAM (``0`` for CPU workers).
        host_threads: CPU threads reserved for a GPU worker (``0`` for CPU).

    """

    commit_bytes: int
    vram_bytes: int
    host_threads: int


@dataclass(frozen=True, slots=True)
class PendingJob:
    """Admitted job that has no worker yet.

    Attributes:
        job_id: Job id.
        call_id: Id of the client call that submitted the job.
        seq: Sequence number the supervisor gave the job at admission (FIFO
            order; the tie-breaker of :func:`admission_order`).
        submitted_at: Client submission time.
        waiting_since: Time the job (re)entered the pending state; the base
            of priority aging and of the starvation guard.
        deadline_at: Absolute deadline, handled by the supervisor.
        provider: Requested STT provider.
        whisper_model: Requested Whisper model (ignored for GigaAM).
        compute_request: Requested compute type.
        device_request: Requested device class.
        duration_s: Audio duration, or ``None`` when unknown.
        max_parallel: Per-call cap of running jobs, or ``None``.
        force_backend: Backend forced by the supervisor (no CUDA, GPU OOM).
        priority: Priority class of the job's call (spec 14).

    """

    job_id: str
    call_id: str
    seq: int
    submitted_at: float
    waiting_since: float
    deadline_at: float | None
    provider: str
    whisper_model: str
    compute_request: str
    device_request: Literal["auto", "cpu", "cuda"]
    duration_s: float | None
    max_parallel: int | None
    force_backend: Backend | None = None
    priority: Priority = DEFAULT_PRIORITY


@dataclass(frozen=True, slots=True)
class RunningJob:
    """Job that runs on a worker.

    Attributes:
        job_id: Job id.
        call_id: Id of the client call that submitted the job.
        started_at: Time of the assignment.
        expected_end_at: Predicted completion time.

    """

    job_id: str
    call_id: str
    started_at: float
    expected_end_at: float


@dataclass(frozen=True, slots=True)
class WorkerView:
    """Supervisor-side state of one worker that has not exited yet.

    Attributes:
        worker_id: Worker id.
        key: Worker configuration.
        state: Lifecycle state.
        capacity: Slots (CPU) or lanes (GPU) fixed at spawn.
        running: Jobs that run on the worker.
        idle_since: Time the worker became idle, or ``None`` while busy.
        spawned_at: Spawn time.
        last_used_at: Last time a job was assigned to or finished on the worker.
        reservation: Resources reserved at spawn.

    """

    worker_id: str
    key: WorkerKey
    state: WorkerState
    capacity: int
    running: tuple[RunningJob, ...]
    idle_since: float | None
    spawned_at: float
    last_used_at: float
    reservation: Reservation


@dataclass(frozen=True, slots=True)
class KeyStatus:
    """Cooldowns of one worker configuration.

    Attributes:
        crash_cooldown_until: End of the crash cooldown, or ``None``.
        load_failure_until: End of the load-failure cooldown, or ``None``.
        load_failure_detail: Error detail of the last load failure.

    """

    crash_cooldown_until: float | None = None
    load_failure_until: float | None = None
    load_failure_detail: str | None = None


@dataclass(frozen=True, slots=True)
class KeyEstimate:
    """Timing and memory estimates of one worker configuration (spec 7.7).

    Attributes:
        load_s: Cold load time (process start, model load, warm-up).
        a_s: Fixed per-job overhead.
        xrt_single: Audio seconds per wall second of one job alone.
        xrt_aggregate: Audio seconds per wall second of one full worker.
        commit_base_bytes: Commit of a worker without slots.
        commit_per_slot_bytes: Extra commit per slot or lane.
        vram_base_bytes: VRAM of a GPU worker without lanes.
        vram_per_lane_bytes: Extra VRAM per lane.
        service_correction: Learned ratio of observed to predicted service.

    """

    load_s: float
    a_s: float
    xrt_single: float
    xrt_aggregate: float
    commit_base_bytes: int
    commit_per_slot_bytes: int
    vram_base_bytes: int
    vram_per_lane_bytes: int
    service_correction: float = 1.0


class EstimateSource(Protocol):
    """Provider of per-key estimates (implemented by ``EstimateStore``)."""

    def estimate(self, key: WorkerKey) -> KeyEstimate:
        """Return the current estimate of *key*."""


@dataclass(frozen=True, slots=True)
class SchedulerState:
    """Complete input of one planning tick.

    Attributes:
        now: Current time; the scheduler never reads a clock.
        limits: Pool limits.
        probes: Host readings.
        pending: Pending jobs in any order; :func:`plan` walks them in
            :func:`admission_order`.
        workers: Every worker whose exit is not confirmed yet.
        key_status: Cooldowns per worker configuration.
        estimates: Estimate provider; must be deterministic within a tick.
        last_gpu_change_ts: Last time a GPU worker was spawned, became ready
            or exited; a GPU spawn needs a VRAM sample newer than this.
        call_last_started: Per call id, the last time a job of the call was
            assigned to a worker (same clock as ``now``). It carries the
            round-robin turn of a call across ticks after its running jobs
            have ended; calls missing from it fall back to the start times of
            their running jobs, then to the time they entered the queue (see
            :func:`admission_order`).

    """

    now: float
    limits: Limits
    probes: Probes
    pending: tuple[PendingJob, ...]
    workers: tuple[WorkerView, ...]
    key_status: Mapping[WorkerKey, KeyStatus]
    estimates: EstimateSource
    last_gpu_change_ts: float | None = None
    call_last_started: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Assign:
    """Send a job to a ready worker.

    Attributes:
        job_id: Job id.
        worker_id: Target worker.
        key: Target worker configuration.
        backend: Backend class of the worker.
        expected_service_s: Predicted run time with the learned correction.
        base_service_s: Predicted run time without the correction; the
            denominator of the correction update.
        share: ``n`` of :func:`service_seconds`: jobs on the GPU worker after
            this assignment, or CPU jobs running on all CPU workers after this
            assignment (CPU workers share the cores).
        cpu_busy: CPU busy fraction used for the GPU contention factor, so the
            supervisor can recompute ``base_service_s`` for the measured audio
            length of a job whose duration was unknown (with ``share``).

    """

    job_id: str
    worker_id: str
    key: WorkerKey
    backend: Backend
    expected_service_s: float
    base_service_s: float
    share: int
    cpu_busy: float = 0.0


@dataclass(frozen=True, slots=True)
class Spawn:
    """Start a worker; its reservation counts from this action on.

    Attributes:
        key: Worker configuration.
        capacity: Slots (CPU) or lanes (GPU).
        reservation: Resources to reserve and the Job Object memory limit.
        reason_job_id: Job that triggered the spawn.

    """

    key: WorkerKey
    capacity: int
    reservation: Reservation
    reason_job_id: str


@dataclass(frozen=True, slots=True)
class Stop:
    """Stop an idle worker gracefully.

    Attributes:
        worker_id: Worker to stop.
        reason: ``ttl``, ``evict``, ``retired`` or ``pressure``.

    """

    worker_id: str
    reason: StopReason


@dataclass(frozen=True, slots=True)
class Reject:
    """Publish a terminal result for a job that cannot be scheduled.

    Attributes:
        job_id: Job id.
        status: ``unavailable`` or ``error``.
        error_kind: ``infeasible``, ``worker_load_failed`` or ``input_invalid``.
        detail: Human-readable reason with the numbers behind it.

    """

    job_id: str
    status: RejectStatus
    error_kind: str
    detail: str


Action: TypeAlias = Assign | Spawn | Stop | Reject


@dataclass(frozen=True, slots=True)
class Plan:
    """Actions of one tick, ordered Reject, Stop, Assign, Spawn.

    Attributes:
        actions: Actions; stable within each group.
        notes: Diagnostics explaining waits (not a contract).

    """

    actions: tuple[Action, ...]
    notes: tuple[str, ...]


# * Public helpers (spec 7.2 and 7.4).


def worker_capacity(key: WorkerKey, limits: Limits) -> int:
    """Return the largest slot or lane capacity of a new worker of *key*.

    This is the spec 7.2 capacity and acts as a ceiling: static feasibility is
    checked at capacity 1, and a spawn takes the largest capacity up to this
    value whose reservation fits the memory and VRAM budgets and guards.

    Args:
        key: Worker configuration.
        limits: Pool limits.

    Returns:
        CPU: ``min(cpu_max_slots_per_worker, cpu_slots, thread_budget //
        threads_per_slot)``; GPU: the batch lanes of the provider family.
        Always at least 1.

    """
    if key.backend == "gpu":
        lanes = limits.batch_lanes.get(provider_family(key.provider), 1)
        return max(1, lanes)
    by_threads = limits.thread_budget // max(1, limits.threads_per_slot)
    return max(1, min(limits.cpu_max_slots_per_worker, limits.cpu_slots, by_threads))


def reservation_for(
    key: WorkerKey, capacity: int, estimate: KeyEstimate, limits: Limits
) -> Reservation:
    """Return the resources a worker of *key* reserves at spawn (spec 7.2).

    Args:
        key: Worker configuration.
        capacity: Slots or lanes of the worker.
        estimate: Current estimate of *key*.
        limits: Pool limits.

    Returns:
        ``commit = commit_base + capacity * commit_per_slot``; GPU workers also
        reserve ``vram_base + capacity * vram_per_lane`` and the host threads of
        their provider family. CPU workers reserve no VRAM and no host threads.

    """
    slots = max(1, capacity)
    commit = estimate.commit_base_bytes + slots * estimate.commit_per_slot_bytes
    if key.backend != "gpu":
        return Reservation(commit_bytes=commit, vram_bytes=0, host_threads=0)
    vram = estimate.vram_base_bytes + slots * estimate.vram_per_lane_bytes
    host = limits.host_threads.get(provider_family(key.provider), 0)
    return Reservation(commit_bytes=commit, vram_bytes=vram, host_threads=max(0, host))


def service_seconds(
    key: WorkerKey,
    estimate: KeyEstimate,
    duration_s: float,
    *,
    share: int,
    cpu_busy: float,
    contention: float,
    corrected: bool = True,
) -> float:
    """Predict the run time of one job on a worker of *key* (spec 7.4).

    ``service = corr * (a + d / xrt_eff)`` with ``n = max(1, share)``. GPU:
    ``xrt_eff = xrt_aggregate / n`` and the result is multiplied by ``1 +
    contention * cpu_busy``. CPU: ``xrt_eff = min(xrt_single, xrt_aggregate /
    n)``, because every CPU job shares the same cores: a lone job runs at its
    single-job speed and a loaded pool at its aggregate throughput.

    Args:
        key: Worker configuration.
        estimate: Current estimate of *key*.
        duration_s: Audio duration in seconds (negative and NaN count as 0).
        share: GPU: jobs sharing the GPU worker after the assignment. CPU: CPU
            jobs running on all CPU workers while this job runs.
        cpu_busy: CPU job threads in use divided by the thread budget, clamped
            to ``[0, 1]`` (GPU only).
        contention: ``gpu_cpu_contention`` of the limits.
        corrected: Whether to apply the learned service correction.

    Returns:
        The predicted run time in seconds.

    """
    duration = 0.0 if math.isnan(duration_s) else max(0.0, duration_s)
    jobs = max(1, share)
    if key.backend == "gpu":
        xrt = estimate.xrt_aggregate / jobs
        busy = min(1.0, max(0.0, cpu_busy))
        factor = 1.0 + max(0.0, contention) * busy
    else:
        xrt = min(estimate.xrt_single, estimate.xrt_aggregate / jobs)
        factor = 1.0
    base = (max(0.0, estimate.a_s) + duration / max(xrt, _MIN_XRT)) * factor
    if not corrected:
        return base
    return base * max(0.0, estimate.service_correction)


# * Admission order (spec 14).


def effective_rank(job: PendingJob, *, now: float, aging_s: float) -> int:
    """Return the priority class a pending job competes in (spec 14 step 1).

    ``rank = PRIORITY_RANK[priority] - floor((now - waiting_since) / aging_s)``
    clamped at 0: a job rises one class per ``aging_s`` of waiting, so a
    ``low`` job competes as ``normal`` after one step and as ``high`` after
    two, and a ``normal`` job competes as ``high`` after one step. A negative
    or NaN wait counts as no wait, an infinite wait reaches class 0, an
    unknown priority counts as ``normal``, and an ``aging_s`` that is not a
    positive number disables aging.

    Args:
        job: Pending job.
        now: Current time (same clock as ``waiting_since``).
        aging_s: ``priority_aging_s`` of the limits.

    Returns:
        The effective rank: 0 (``high``), 1 (``normal``) or 2 (``low``).

    """
    base = PRIORITY_RANK.get(job.priority, PRIORITY_RANK[DEFAULT_PRIORITY])
    waited = now - job.waiting_since
    if math.isnan(aging_s) or aging_s <= 0 or math.isnan(waited) or waited <= 0:
        return base
    steps = waited / aging_s
    if math.isnan(steps):
        return base
    if math.isinf(steps):
        return 0
    return max(0, base - math.floor(steps))


_OrderKey: TypeAlias = tuple[int, int, float, int, int, str, int, str]


def admission_order(
    pending: Sequence[PendingJob],
    *,
    now: float,
    aging_s: float,
    running: Mapping[str, int] | None = None,
    last_started: Mapping[str, float] | None = None,
) -> tuple[PendingJob, ...]:
    """Return the pending jobs in the admission order of spec section 14.

    1. Classes: every job competes in its :func:`effective_rank`; classes go
       in ascending rank (``high`` first, aging included).
    2. Fair share within a class: the calls take turns. The k-th pending job
       of a call in the class (``k`` from 0, the call's jobs in ``seq``
       order) is in round ``k``; rounds go in ascending order, and within a
       round the calls go by their oldest job in the class (smallest
       ``seq``). So the walk takes the first job of every call, then the
       second job of every call, and so on.
    3. Remaining ties go by ``seq``, then job id; the call id separates
       calls with equal positions. The order is deterministic and does not
       depend on the order of *pending*.

    The optional history carries the turns over between ticks (:func:`plan`
    passes both), so the fair share also holds when only one slot frees per
    tick:

    - ``running``: the running jobs of a call count as turns it has already
      had, its k-th pending job in a class is in round ``running[call] + k``;
    - ``last_started``: within a round, the calls go by their turn position
      before their oldest job: the last start of a job of the call, or, for a
      call without a known start, the earliest ``waiting_since`` of its jobs
      in the class (when it entered the queue). A call whose job has just
      started thus goes behind the calls that waited before that start, and
      calls that entered the queue after that start go behind it, so a
      stream of new calls cannot keep a call without running jobs waiting.
      At an equal turn position, a call without a known start (not served
      yet) goes before a call that has been served, so a call that entered
      the queue at the very moment another call started still gets its turn
      first.

    Args:
        pending: Pending jobs, in any order.
        now: Current time (same clock as ``waiting_since``).
        aging_s: ``priority_aging_s`` of the limits.
        running: Running jobs per call id (absent calls: 0), or ``None``.
        last_started: Last job start per call id (same clock as ``now``), or
            ``None`` to order the calls of a round by their oldest job only.

    Returns:
        The jobs of *pending* in admission order.

    """
    counts: Mapping[str, int] = running if running is not None else {}
    classes: dict[int, dict[str, list[PendingJob]]] = {}
    for job in sorted(pending, key=lambda job: (job.seq, job.job_id)):
        rank = effective_rank(job, now=now, aging_s=aging_s)
        classes.setdefault(rank, {}).setdefault(job.call_id, []).append(job)
    keyed: list[tuple[_OrderKey, PendingJob]] = []
    for rank, calls in classes.items():
        for call_id, jobs in calls.items():
            turn, served = (
                (0.0, 0)
                if last_started is None
                else _turn_position(last_started.get(call_id), jobs, now)
            )
            offset = max(0, counts.get(call_id, 0))
            oldest = jobs[0].seq
            keyed.extend(
                (
                    (
                        rank,
                        offset + k,
                        turn,
                        served,
                        oldest,
                        call_id,
                        job.seq,
                        job.job_id,
                    ),
                    job,
                )
                for k, job in enumerate(jobs)
            )
    keyed.sort(key=lambda item: item[0])
    return tuple(job for _, job in keyed)


def _turn_position(
    started: float | None, jobs: Sequence[PendingJob], now: float
) -> tuple[float, int]:
    """Return a call's turn position within a round of its class.

    Args:
        started: Last start of a job of the call, or ``None``.
        jobs: The call's pending jobs in the class.
        now: Current time; replaces NaN times so the order stays total.

    Returns:
        The last start and ``1`` (served) when the start is known, otherwise
        the earliest ``waiting_since`` and ``0`` (not served yet); at an equal
        position the call that has not been served goes first.

    """
    if started is not None and not math.isnan(started):
        return started, 1
    since = (job.waiting_since for job in jobs if not math.isnan(job.waiting_since))
    return min(since, default=now), 0


def plan(state: SchedulerState) -> Plan:
    """Compute the actions of one scheduling tick (spec 7.1-7.6 and 14).

    Pending jobs are walked in :func:`admission_order` with the running jobs
    of each call and the last starts of ``state.call_last_started`` (merged
    with the start times of running jobs), so every in-tick decision that
    favors an earlier job (backlog, spawn reservation, eviction, the
    starvation guard) follows that order.

    Args:
        state: Immutable snapshot of the pool.

    Returns:
        The plan; the same state always yields the same plan.

    """
    return _Planner(state).run()


# * Tick-local model.


@dataclass(slots=True)
class _Sim:
    """Mutable tick-local view of one worker (existing or spawned in this tick)."""

    worker_id: str
    key: WorkerKey
    state: WorkerState
    capacity: int
    ends: list[float]
    reservation: Reservation
    spawned_at: float
    last_used_at: float
    idle_since: float | None
    planned: bool = False
    stopped: bool = False
    draining: bool = False

    @classmethod
    def from_view(cls, view: WorkerView) -> _Sim:
        """Copy a worker view into a mutable tick-local view."""
        return cls(
            worker_id=view.worker_id,
            key=view.key,
            state=view.state,
            capacity=max(0, view.capacity),
            ends=[job.expected_end_at for job in view.running],
            reservation=view.reservation,
            spawned_at=view.spawned_at,
            last_used_at=view.last_used_at,
            idle_since=view.idle_since,
        )

    @property
    def is_gpu(self) -> bool:
        """Return whether the worker is a GPU worker."""
        return self.key.backend == "gpu"

    @property
    def load(self) -> int:
        """Return the running jobs, including assignments of this tick."""
        return len(self.ends)

    @property
    def serving(self) -> bool:
        """Return whether the worker is or will be able to take jobs of its key."""
        return self.state in _SERVING_STATES and not self.stopped and not self.draining

    @property
    def is_loading(self) -> bool:
        """Return whether the worker is a cold load in progress."""
        return self.state in _LOADING_STATES and not self.stopped

    @property
    def has_free_capacity(self) -> bool:
        """Return whether the worker can take a job now (ignoring CPU slots)."""
        return self.state == "ready" and self.serving and self.load < self.capacity

    @property
    def idle_ready(self) -> bool:
        """Return whether the worker is ready, not stopped and has no jobs."""
        return self.state == "ready" and not self.stopped and not self.ends

    @property
    def freeing(self) -> bool:
        """Return whether the worker's reservation is about to be released."""
        return self.state == "stopping" or self.stopped

    def remaining_work(self, now: float) -> float:
        """Return the predicted time until the worker finishes its jobs."""
        return max(0.0, max(self.ends) - now) if self.ends else 0.0


@dataclass(frozen=True, slots=True)
class _Freed:
    """Resources assumed to be released before a placement."""

    commit: int = 0
    vram: int = 0
    gpu_workers: int = 0
    host_threads: int = 0

    def plus(self, sim: _Sim) -> _Freed:
        """Return these resources plus the reservation of *sim*."""
        return self.add(
            _Freed(
                commit=sim.reservation.commit_bytes,
                vram=sim.reservation.vram_bytes,
                gpu_workers=1 if sim.is_gpu else 0,
                host_threads=sim.reservation.host_threads,
            )
        )

    def add(self, other: _Freed) -> _Freed:
        """Return the sum of these resources and *other*."""
        return _Freed(
            commit=self.commit + other.commit,
            vram=self.vram + other.vram,
            gpu_workers=self.gpu_workers + other.gpu_workers,
            host_threads=self.host_threads + other.host_threads,
        )


def _fold(base: _Freed, sims: Iterable[_Sim]) -> _Freed:
    """Return *base* plus the reservations of *sims*."""
    freed = base
    for sim in sims:
        freed = freed.plus(sim)
    return freed


@dataclass(frozen=True, slots=True)
class _Need:
    """Resources one placement must find free.

    A spawn needs the reservation of its worker (commit; for GPU also VRAM and
    one GPU worker) and its threads; a CPU job joining an existing worker
    needs only the threads of one slot.

    Attributes:
        commit: Commit bytes the new worker reserves (0 without a spawn).
        vram: VRAM bytes the new worker reserves.
        threads: Host threads of a GPU worker, or the threads of one CPU job.
        gpu_worker: Whether a GPU worker process is added.

    """

    commit: int = 0
    vram: int = 0
    threads: int = 0
    gpu_worker: bool = False


@dataclass(frozen=True, slots=True)
class _Blockers:
    """Reasons a spawn cannot happen now (spec 7.5 step 4).

    Attributes:
        hard: Reasons that neither time nor eviction resolve in the near term
            (memory pressure, no CUDA device).
        timing: Reasons that clear by themselves (a cold load in progress, a
            spawn reserved by an earlier job, cooldowns, a stale VRAM sample,
            no free CPU slot); eviction does not help.
        delay_s: Predicted time until every timing reason clears.
        resources: Budget and probe shortfalls that releasing reservations of
            other workers can resolve.
        cooldown: Whether a crash or load-failure cooldown of the key is among
            the timing reasons.

    """

    hard: tuple[str, ...] = ()
    timing: tuple[str, ...] = ()
    delay_s: float = 0.0
    resources: tuple[str, ...] = ()
    cooldown: bool = False

    @property
    def clear(self) -> bool:
        """Return whether the spawn is feasible now."""
        return not (self.hard or self.timing or self.resources)

    def first(self) -> str:
        """Return the most important reason."""
        reasons = (*self.hard, *self.timing, *self.resources)
        return reasons[0] if reasons else "no capacity"


@dataclass(frozen=True, slots=True)
class _Candidate:
    """Worker configuration a job may run on.

    Attributes:
        key: Worker configuration.
        max_capacity: Capacity ceiling of a new worker (:func:`worker_capacity`).
        min_reservation: Reservation of a new worker with capacity 1, the
            smallest worker that runs the job (feasibility and eviction).
        estimate: Current estimate of the key.

    """

    key: WorkerKey
    max_capacity: int
    min_reservation: Reservation
    estimate: KeyEstimate

    @property
    def is_gpu(self) -> bool:
        """Return whether the candidate is a GPU configuration."""
        return self.key.backend == "gpu"


RouteKind = Literal["assign", "wait", "spawn", "evict"]


@dataclass(frozen=True, slots=True)
class _Route:
    """What happens to a job on one candidate in this tick, and its ETA.

    Attributes:
        kind: Assignment, wait, spawn or eviction.
        candidate: Candidate the route uses.
        eta_s: Predicted time until the job finishes.
        reason: Diagnostic text for the plan notes.
        target: Worker of an assignment.
        victims: Idle workers an eviction stops.
        claims_spawn: Whether the route reserves the next spawn for its key.
        spawn_capacity: Capacity of the worker a spawn starts.
        protectable: Whether the scheduler itself resolves the wait, so that a
            starving job on this route may become protected.
        drain_need: Resources that draining workers of other keys must release
            for the job once it is protected; ``None`` when draining does not
            help.

    """

    kind: RouteKind
    candidate: _Candidate
    eta_s: float
    reason: str
    target: _Sim | None = None
    victims: tuple[_Sim, ...] = ()
    claims_spawn: bool = False
    spawn_capacity: int = 0
    protectable: bool = False
    drain_need: _Need | None = None


@dataclass(frozen=True, slots=True)
class _Claim:
    """Spawn reserved for an earlier job of this tick (eviction in progress)."""

    job_id: str
    key: WorkerKey


@dataclass(frozen=True, slots=True)
class _Protected:
    """Starvation-protected job and the CPU capacity later jobs must leave free."""

    job_id: str
    key: WorkerKey
    slots: int
    threads: int


_NOTHING_FREED = _Freed()
_DRAINABLE_STATES: frozenset[str] = frozenset({"ready", "retiring"})


def _gib(size: float) -> str:
    """Format a byte count in GiB with two decimals."""
    return f"{size / _GIB:.2f}"


def _seconds(value: float) -> str:
    """Format a duration for notes."""
    return "inf" if math.isinf(value) else f"{value:.1f}s"


# * Planning algorithm.
# *
# * One tick runs in four phases over a mutable copy of the worker list:
# *   1. Pressure and retirement: under critical free commit every idle ready
# *      worker stops (LRU first) and spawns are blocked; idle retiring workers
# *      stop. For every effective priority class, the idle ready workers that
# *      the pending jobs of that class and of the higher classes are about to
# *      take are recorded; eviction for a job of the class leaves them alone
# *      in this tick, while idle workers wanted only by lower classes may go.
# *   2. Placement: pending jobs in admission order (spec 14: effective class
# *      with aging, then round-robin across calls). Each job gets its
# *      candidate keys (rejecting infeasible or load-failed requests), a route
# *      per candidate (assign / wait / spawn / evict) with an ETA computed
# *      from the tick's own earlier decisions, the backend choice (an ``auto``
# *      job spills to CPU only when its duration is known and ETA_cpu <
# *      spill_ratio * ETA_gpu), the per-call cap, and finally the chosen route
# *      executes and updates the tick state, so later jobs see its slots,
# *      lanes, backlog, spawn reservation and evictions. A job is never held
# *      back by a decision of a job that comes later in admission order.
# *   3. Starvation guard: the first qualifying job in admission order left
# *      unassigned whose wait the scheduler itself resolves (no cooldown, no
# *      hard blocker, a finite ETA) becomes protected; later jobs may only use
# *      existing ready workers, must leave its CPU slot and threads free, and
# *      must not use workers drained for it.
# *   4. Idle TTL: idle ready workers past the TTL stop, except as many idle
# *      workers of a key (most recently used first) as the capacity the jobs
# *      left waiting for that key in this tick need; jobs held back by their
# *      call's ``max_parallel`` keep no worker. The same per-worker demand
# *      protects idle workers from eviction during placement.
# * Resources are recomputed from the worker list on every check, so every
# * Assign/Spawn/Stop of the tick is accounted for immediately. Priority only
# * orders admission: running jobs are never preempted and busy workers are
# * never stopped.


class _Planner:
    """Single-use planner for one :class:`SchedulerState`."""

    def __init__(self, state: SchedulerState) -> None:
        """Copy the state into tick-local structures."""
        self._state = state
        self._now = state.now
        self._limits = state.limits
        self._probes = state.probes
        self._sims = [
            _Sim.from_view(view)
            for view in sorted(state.workers, key=lambda view: view.worker_id)
        ]
        self._call_running: Counter[str] = Counter(
            job.call_id for view in state.workers for job in view.running
        )
        self._running_ids = {
            job.job_id for view in state.workers for job in view.running
        }
        self._estimates: dict[WorkerKey, KeyEstimate] = {}
        self._backlog_s: dict[WorkerKey, float] = {}
        self._backlog_n: dict[WorkerKey, int] = {}
        self._demand: Counter[WorkerKey] = Counter()
        self._seen: set[str] = set()
        self._usable_idle: frozenset[str] = frozenset()
        self._claim: _Claim | None = None
        self._protected: _Protected | None = None
        self._pressure = False
        self._rejects: list[Reject] = []
        self._stops: list[Stop] = []
        self._assigns: list[Assign] = []
        self._spawns: list[Spawn] = []
        self._notes: list[str] = []

    def run(self) -> Plan:
        """Execute the four phases and return the plan."""
        self._stop_for_pressure()
        self._stop_retired()
        aging_s = self._limits.priority_aging_s
        jobs = admission_order(
            self._state.pending,
            now=self._now,
            aging_s=aging_s,
            running=self._call_running,
            last_started=self._last_starts(),
        )
        ranks = [effective_rank(job, now=self._now, aging_s=aging_s) for job in jobs]
        usable = {
            rank: self._usable_idle_ids(
                [job for job, other in zip(jobs, ranks, strict=True) if other <= rank]
            )
            for rank in sorted(set(ranks))
        }
        for job, rank in zip(jobs, ranks, strict=True):
            self._usable_idle = usable[rank]
            self._place(job)
        self._stop_expired()
        actions: tuple[Action, ...] = (
            *self._rejects,
            *self._stops,
            *self._assigns,
            *self._spawns,
        )
        return Plan(actions=actions, notes=tuple(self._notes))

    # * Phase 1 and 4: lifecycle stops (spec 7.6).

    def _stop_for_pressure(self) -> None:
        """Stop every idle ready worker (LRU first) under critical free commit."""
        free = self._probes.free_commit_bytes
        self._pressure = (
            free is not None and free < self._limits.critical_free_commit_bytes
        )
        if not self._pressure or free is None:
            return
        idle = sorted(
            (sim for sim in self._sims if sim.idle_ready),
            key=lambda sim: (sim.last_used_at, sim.worker_id),
        )
        for sim in idle:
            self._stop(sim, "pressure")
        self._notes.append(
            f"memory pressure: free commit {_gib(free)} GiB < critical "
            f"{_gib(self._limits.critical_free_commit_bytes)} GiB; no spawns"
        )

    def _stop_retired(self) -> None:
        """Stop retiring workers that have no running jobs."""
        for sim in self._sims:
            if sim.state == "retiring" and not sim.ends and not sim.stopped:
                self._stop(sim, "retired")

    def _stop_expired(self) -> None:
        """Stop idle ready workers past the idle TTL that no waiting job needs.

        The idle workers that the waiting jobs of this tick keep
        (:meth:`_spared_idle`) stay; every other idle ready worker past the TTL
        stops, including a surplus worker of a key with demand.
        """
        spared = self._spared_idle()
        for sim in self._sims:
            if not sim.idle_ready or sim.planned or sim.worker_id in spared:
                continue
            if self._ttl_expired(sim):
                self._stop(sim, "ttl")

    def _ttl_expired(self, sim: _Sim) -> bool:
        """Return whether an idle worker has been idle for the idle TTL."""
        return self._now - self._idle_clock(sim) >= self._limits.worker_idle_ttl_s

    @staticmethod
    def _idle_clock(sim: _Sim) -> float:
        """Return the time an idle worker was last used (when it became idle)."""
        return sim.idle_since if sim.idle_since is not None else sim.last_used_at

    def _spared_idle(self) -> frozenset[str]:
        """Return the idle ready workers that the waiting jobs of this tick keep.

        Every job of this tick that chose a key and was left waiting (for
        example for a free CPU slot) counts once for that key; a job held back
        by its call's ``max_parallel`` does not count. Per key, idle ready
        workers are kept until their capacity covers these jobs: first the
        workers the idle TTL keeps anyway, then the most recently used ones,
        then the lowest worker id. The other idle workers of the key are
        surplus: they stop by TTL and may be evicted.

        Returns:
            Worker ids of the idle workers to keep.

        """
        idle: dict[WorkerKey, list[_Sim]] = {}
        for sim in self._sims:
            if (
                sim.idle_ready
                and not sim.planned
                and sim.capacity > 0
                and self._demand[sim.key] > 0
            ):
                idle.setdefault(sim.key, []).append(sim)
        spared: set[str] = set()
        for key, sims in idle.items():
            needed = self._demand[key]
            for sim in sorted(sims, key=self._spare_order):
                if needed <= 0:
                    break
                spared.add(sim.worker_id)
                needed -= sim.capacity
        return frozenset(spared)

    def _spare_order(self, sim: _Sim) -> tuple[bool, float, str]:
        """Sort key of idle workers to keep: unexpired, most recently used, id."""
        since = self._idle_clock(sim)
        recency = -math.inf if math.isnan(since) else since
        return (self._ttl_expired(sim), -recency, sim.worker_id)

    def _stop(self, sim: _Sim, reason: StopReason) -> None:
        """Record a stop; the reservation stays until the exit is confirmed."""
        sim.stopped = True
        self._stops.append(Stop(worker_id=sim.worker_id, reason=reason))

    def _last_starts(self) -> dict[str, float]:
        """Return the last job start per call for the round-robin turns.

        ``state.call_last_started`` merged with the start times of the jobs
        that run now (the later time wins); NaN times are ignored.
        """
        starts = {
            call_id: at
            for call_id, at in self._state.call_last_started.items()
            if not math.isnan(at)
        }
        for view in self._state.workers:
            for job in view.running:
                known = starts.get(job.call_id)
                if not math.isnan(job.started_at) and (
                    known is None or job.started_at > known
                ):
                    starts[job.call_id] = job.started_at
        return starts

    def _usable_idle_ids(self, jobs: Sequence[PendingJob]) -> frozenset[str]:
        """Return the idle ready workers that the given jobs take in this tick.

        Placement prefers idle workers (no running jobs; the lowest worker id
        first), so for every key the first ``min(jobs, idle workers)`` idle
        workers in worker-id order are about to be used. A job counts for every
        key its request may use (both backends of an ``auto`` job) unless its
        call already runs ``max_parallel`` jobs. The planner calls this once
        per effective class with the jobs of that class and of the higher
        classes: eviction for a job of the class never takes these workers in
        this tick, and a drain for a protected job takes them last, while
        idle workers that only lower classes want are not held for them.

        Args:
            jobs: Pending jobs in admission order.

        Returns:
            Worker ids of the idle workers to keep.

        """
        idle: dict[WorkerKey, list[str]] = {}
        for sim in self._sims:
            if sim.idle_ready and sim.capacity > 0:
                idle.setdefault(sim.key, []).append(sim.worker_id)
        if not idle:
            return frozenset()
        wanted: Counter[WorkerKey] = Counter()
        taken: Counter[str] = Counter()
        seen: set[str] = set()
        for job in jobs:
            if job.job_id in seen or job.job_id in self._running_ids:
                continue
            seen.add(job.job_id)
            running = self._call_running[job.call_id] + taken[job.call_id]
            cap = job.max_parallel
            if cap is not None and running >= max(1, cap):
                continue
            taken[job.call_id] += 1
            for key in self._job_keys(job):
                if key in idle:
                    wanted[key] += 1
        usable: set[str] = set()
        for key, worker_ids in idle.items():
            usable.update(worker_ids[: wanted[key]])
        return frozenset(usable)

    def _job_keys(self, job: PendingJob) -> list[WorkerKey]:
        """Return the worker keys a job's request may use (empty when invalid)."""
        backends = self._backends(job)
        if isinstance(backends, Reject):
            return []
        try:
            return [
                make_key(job.provider, job.whisper_model, job.compute_request, backend)
                for backend in backends
            ]
        except ValueError:
            return []

    # * Phase 2: placement.

    def _place(self, job: PendingJob) -> None:
        """Route one pending job (spec 7.3, 7.5) and apply the decision."""
        if job.job_id in self._seen or job.job_id in self._running_ids:
            self._notes.append(f"{job.job_id}: duplicate or already running; skipped")
            return
        self._seen.add(job.job_id)
        candidates = self._candidates(job)
        if isinstance(candidates, Reject):
            self._rejects.append(candidates)
            return
        route, spill_note = self._choose(job, candidates)
        if self._capped(job):
            # * Capped jobs neither queue as backlog, nor count for starvation,
            # * nor keep idle workers of their key: the worker that ends one of
            # * the call's running jobs serves the next one (spec example 4).
            self._notes.append(
                f"{job.job_id}: call {job.call_id} is at max_parallel={job.max_parallel}"
            )
            return
        if spill_note is not None:
            self._notes.append(spill_note)
        waited = self._now - job.waiting_since
        qualifies = (
            self._protected is None and waited >= self._limits.starvation_after_s
        )
        if self._execute(job, route) or not qualifies:
            return
        if route.protectable:
            self._protect(job, route)
            return
        # * A job blocked by a cooldown, a hard blocker or an endless wait would
        # * only freeze the pool; the next qualifying job is protected instead.
        self._notes.append(
            f"{job.job_id}: not protected after waiting {_seconds(waited)}; "
            "scheduling cannot resolve its wait"
        )

    def _capped(self, job: PendingJob) -> bool:
        """Return whether the job's call already runs ``max_parallel`` jobs."""
        cap = job.max_parallel
        return cap is not None and self._call_running[job.call_id] >= max(1, cap)

    def _execute(self, job: PendingJob, route: _Route) -> bool:
        """Apply a route; return whether the job was assigned."""
        if route.kind == "assign" and route.target is not None:
            self._assign(job, route.candidate, route.target)
            return True
        key = route.candidate.key
        work = self._service(job, route.candidate)
        self._backlog_s[key] = self._backlog_s.get(key, 0.0) + work
        self._backlog_n[key] = self._backlog_n.get(key, 0) + 1
        self._demand[key] += 1
        if route.kind == "spawn":
            self._spawn(job, route.candidate, route.spawn_capacity)
        for victim in route.victims:
            self._stop(victim, "evict")
        if route.claims_spawn:
            self._claim = _Claim(job_id=job.job_id, key=key)
        self._notes.append(
            f"{job.job_id} [{key.label}]: {route.reason}; ETA {_seconds(route.eta_s)}"
        )
        return False

    def _assign(self, job: PendingJob, candidate: _Candidate, target: _Sim) -> None:
        """Record an assignment and occupy a slot or lane of *target*."""
        key = candidate.key
        share = self._assign_share(candidate, target)
        cpu_busy = self._cpu_busy()
        duration = self._duration(job)
        contention = self._limits.gpu_cpu_contention
        expected = service_seconds(
            key,
            candidate.estimate,
            duration,
            share=share,
            cpu_busy=cpu_busy,
            contention=contention,
        )
        base = service_seconds(
            key,
            candidate.estimate,
            duration,
            share=share,
            cpu_busy=cpu_busy,
            contention=contention,
            corrected=False,
        )
        target.ends.append(self._now + expected)
        self._call_running[job.call_id] += 1
        self._assigns.append(
            Assign(
                job_id=job.job_id,
                worker_id=target.worker_id,
                key=key,
                backend=key.backend,
                expected_service_s=expected,
                base_service_s=base,
                share=share,
                cpu_busy=cpu_busy,
            )
        )

    def _spawn(self, job: PendingJob, candidate: _Candidate, capacity: int) -> None:
        """Record a spawn and add the new worker to the tick's resources."""
        capacity = max(1, capacity)
        reservation = reservation_for(
            candidate.key, capacity, candidate.estimate, self._limits
        )
        self._sims.append(
            _Sim(
                worker_id=f"(planned for {job.job_id})",
                key=candidate.key,
                state="spawning",
                capacity=capacity,
                ends=[],
                reservation=reservation,
                spawned_at=self._now,
                last_used_at=self._now,
                idle_since=None,
                planned=True,
            )
        )
        self._spawns.append(
            Spawn(
                key=candidate.key,
                capacity=capacity,
                reservation=reservation,
                reason_job_id=job.job_id,
            )
        )

    # * Candidates and rejections (spec 7.3).

    def _candidates(self, job: PendingJob) -> list[_Candidate] | Reject:
        """Return the usable candidates of a job, GPU first, or its rejection."""
        backends = self._backends(job)
        if isinstance(backends, Reject):
            return backends
        try:
            keys = [
                make_key(job.provider, job.whisper_model, job.compute_request, backend)
                for backend in backends
            ]
        except ValueError as exc:
            return _reject(job, "error", ERROR_KIND_INPUT_INVALID, str(exc))
        feasible: list[_Candidate] = []
        reasons: list[str] = []
        for key in keys:
            candidate = self._candidate(key)
            why = self._static_reasons(candidate)
            if why and not self._has_live_worker(key):
                reasons.extend(why)
            else:
                feasible.append(candidate)
        if not feasible:
            return _reject(
                job, "unavailable", ERROR_KIND_INFEASIBLE, "; ".join(reasons)
            )
        usable = [c for c in feasible if self._load_failure(c.key) is None]
        if not usable:
            failures = [self._load_failure(c.key) or "" for c in feasible]
            return _reject(
                job, "unavailable", ERROR_KIND_WORKER_LOAD_FAILED, "; ".join(failures)
            )
        return usable

    def _backends(self, job: PendingJob) -> tuple[Backend, ...] | Reject:
        """Return the backend classes a job may use (spec 7.3 step 1).

        A forced backend the provider cannot run on is an invalid job record
        (``error`` / ``input_invalid``), not a capacity problem.
        """
        try:
            allowed = allowed_backends(job.provider)
        except ValueError as exc:
            return _reject(job, "error", ERROR_KIND_INPUT_INVALID, str(exc))
        force = job.force_backend
        if force is not None:
            if force not in allowed:
                detail = f"STT provider {job.provider} cannot run on backend {force!r}"
                return _reject(job, "error", ERROR_KIND_INPUT_INVALID, detail)
            return (force,)
        return self._requested_backends(job, allowed)

    def _requested_backends(
        self, job: PendingJob, allowed: frozenset[Backend]
    ) -> tuple[Backend, ...] | Reject:
        """Map the device request to backends; ``auto`` lists GPU first.

        ``auto`` offers the GPU only when a CUDA device is present. ``cuda``
        without CUDA keeps the GPU: the supervisor forces CPU at admission when
        CUDA is confirmed absent (spec 7.3 step 1), so the job waits here.
        """
        request = job.device_request
        if request not in {"auto", "cpu", "cuda"}:
            detail = f"unsupported device request {request!r}"
            return _reject(job, "error", ERROR_KIND_INPUT_INVALID, detail)
        if "gpu" not in allowed:
            if request == "cuda":
                detail = f"STT provider {job.provider} cannot run on CUDA"
                return _reject(job, "error", ERROR_KIND_INPUT_INVALID, detail)
            return ("cpu",)
        if request == "cpu":
            return ("cpu",)
        if request == "cuda":
            return ("gpu",)
        return ("gpu", "cpu") if self._probes.cuda_present else ("cpu",)

    def _candidate(self, key: WorkerKey) -> _Candidate:
        """Describe *key* with its capacity ceiling and its smallest worker."""
        estimate = self._estimate(key)
        return _Candidate(
            key=key,
            max_capacity=worker_capacity(key, self._limits),
            min_reservation=reservation_for(key, 1, estimate, self._limits),
            estimate=estimate,
        )

    def _estimate(self, key: WorkerKey) -> KeyEstimate:
        """Return the estimate of *key*, cached for the tick."""
        cached = self._estimates.get(key)
        if cached is None:
            cached = self._state.estimates.estimate(key)
            self._estimates[key] = cached
        return cached

    def _static_reasons(self, candidate: _Candidate) -> list[str]:
        """Return why even a one-slot worker of the candidate never fits.

        Spec 7.3 step 3 checked at capacity 1: the configured limits are
        ceilings, so a worker that does not fit at full capacity still runs
        with fewer slots or lanes.
        """
        limits = self._limits
        res = candidate.min_reservation
        label = candidate.key.label
        unit = "lane" if candidate.is_gpu else "slot"
        reasons: list[str] = []
        if res.commit_bytes > limits.commit_budget_bytes:
            reasons.append(
                f"{label} needs {_gib(res.commit_bytes)} GiB commit for 1 {unit} "
                f"but the commit budget is {_gib(limits.commit_budget_bytes)} GiB"
            )
        if candidate.is_gpu:
            if limits.gpu_workers < 1:
                reasons.append(
                    f"{label} needs a GPU worker but the GPU worker limit is 0"
                )
            if res.vram_bytes > limits.vram_budget_bytes:
                reasons.append(
                    f"{label} needs {_gib(res.vram_bytes)} GiB VRAM for 1 {unit} "
                    f"but the VRAM budget is {_gib(limits.vram_budget_bytes)} GiB"
                )
            if res.host_threads > limits.thread_budget:
                reasons.append(
                    f"{label} needs {res.host_threads} host threads but the thread "
                    f"budget is {limits.thread_budget}"
                )
        else:
            if limits.cpu_slots < 1:
                reasons.append(f"{label} needs a CPU slot but the CPU slot limit is 0")
            if limits.threads_per_slot > limits.thread_budget:
                reasons.append(
                    f"{label} needs {limits.threads_per_slot} threads per slot but the "
                    f"thread budget is {limits.thread_budget}"
                )
        return reasons

    def _has_live_worker(self, key: WorkerKey) -> bool:
        """Return whether a ready, spawning or loading worker of *key* exists."""
        return any(
            sim.key == key and sim.state in _SERVING_STATES and not sim.stopped
            for sim in self._sims
        )

    def _load_failure(self, key: WorkerKey) -> str | None:
        """Return the load-failure rejection detail of *key*, if it applies.

        A key in load-failure cooldown is unusable unless a live worker of it
        exists (that worker can still serve; only spawns are blocked).
        """
        status = self._state.key_status.get(key)
        until = status.load_failure_until if status is not None else None
        if status is None or until is None or until <= self._now:
            return None
        if self._has_live_worker(key):
            return None
        detail = status.load_failure_detail or "worker load failed"
        remaining = math.ceil(until - self._now)
        return f"{key.label}: {detail} (load-failure cooldown, {remaining} s remaining)"

    # * Backend choice (spec 7.3 step 2).

    def _choose(
        self, job: PendingJob, candidates: list[_Candidate]
    ) -> tuple[_Route, str | None]:
        """Pick the route of a job and explain a spill.

        With both candidates (``auto``), the GPU is the default; the job spills
        to CPU only when its duration is known and ``ETA_cpu < spill_ratio *
        ETA_gpu`` (an unreachable GPU has an infinite ETA).

        Returns:
            The chosen route and a spill note, or ``None`` without a spill.

        """
        gpu = next((c for c in candidates if c.is_gpu), None)
        cpu = next((c for c in candidates if not c.is_gpu), None)
        if gpu is None or cpu is None:
            return self._route(job, candidates[0]), None
        gpu_route = self._route(job, gpu)
        cpu_route = self._route(job, cpu)
        known = self._known_duration(job) is not None
        if known and cpu_route.eta_s < self._limits.spill_ratio * gpu_route.eta_s:
            note = (
                f"{job.job_id}: spills to CPU (ETA cpu {_seconds(cpu_route.eta_s)} < "
                f"{self._limits.spill_ratio} x gpu {_seconds(gpu_route.eta_s)})"
            )
            return cpu_route, note
        return gpu_route, None

    def _route(self, job: PendingJob, candidate: _Candidate) -> _Route:
        """Return the route of a job on one candidate without side effects."""
        if candidate.is_gpu:
            return self._gpu_route(job, candidate)
        return self._cpu_route(job, candidate)

    # * Placement per backend (spec 7.5).

    def _cpu_route(self, job: PendingJob, candidate: _Candidate) -> _Route:
        """Route a CPU job: free slot, pool shortage, loading worker or spawn."""
        sims = self._serving(candidate.key)
        free = [sim for sim in sims if sim.has_free_capacity]
        if free:
            if self._cpu_slot_free():
                target = min(free, key=lambda sim: (sim.load, sim.worker_id))
                return self._assign_route(job, candidate, target)
            return self._cpu_shortage_route(job, candidate)
        loading = [sim for sim in sims if sim.is_loading]
        if loading:
            return self._loading_route(job, candidate, loading)
        # * A further CPU worker of the key only when every one is at capacity.
        return self._spawn_route(job, candidate)

    def _cpu_shortage_route(self, job: PendingJob, candidate: _Candidate) -> _Route:
        """Route a CPU job whose ready worker has a free slot the pool cannot fill.

        A CPU slot shortage, and any shortage while a protected job exists,
        clears only when CPU jobs end. A thread shortage is a resource
        shortage like a blocked spawn: host threads of idle GPU workers of
        other keys are evicted (spec 7.5 step 5); otherwise the job waits for
        the earliest release, and once protected it drains (spec 7.6).
        """
        limits = self._limits
        if self._protected is not None or self._cpu_running() >= limits.cpu_slots:
            eta = self._eta(job, candidate, self._earliest_cpu_end())
            return _Route(
                kind="wait",
                candidate=candidate,
                eta_s=eta,
                reason="no free CPU slot or threads",
                protectable=math.isfinite(eta),
            )
        need = _Need(threads=limits.threads_per_slot)
        pending = self._pending_freed()
        if not self._resource_blockers(need, pending):
            return _Route(
                kind="wait",
                candidate=candidate,
                eta_s=self._eta(job, candidate, 0.0),
                reason="waits for stopping workers to release CPU threads",
                protectable=True,
            )
        victims = self._victims(need, pending, self._eviction_pool(candidate.key))
        if victims:
            names = ", ".join(sim.worker_id for sim in victims)
            return _Route(
                kind="evict",
                candidate=candidate,
                eta_s=self._eta(job, candidate, 0.0),
                reason=f"evicts {names} to free CPU threads",
                victims=victims,
                protectable=True,
            )
        eta = self._eta(job, candidate, self._release_wait(need, candidate.key))
        shortfall = self._resource_blockers(need, _NOTHING_FREED)
        return _Route(
            kind="wait",
            candidate=candidate,
            eta_s=eta,
            reason=shortfall[0] if shortfall else "no free CPU threads",
            protectable=math.isfinite(eta),
            drain_need=need,
        )

    def _gpu_route(self, job: PendingJob, candidate: _Candidate) -> _Route:
        """Route a GPU job: join a worker, wait for a loading one or scale out.

        A ready worker with a free lane and at most ``scale_out_backlog_s`` of
        predicted remaining work takes the job. When every ready worker with a
        free lane is backlogged, the job waits for a loading worker of its key
        or spawns another one (scale-out) only when that ETA is below
        ``SCALE_OUT_MARGIN`` times the ETA of joining the best backlogged
        worker; otherwise it joins, so lanes never idle while jobs wait for a
        cold load. Eviction never serves a scale-out while a lane is free.
        """
        sims = self._serving(candidate.key)
        free = [sim for sim in sims if sim.has_free_capacity]
        threshold = self._limits.scale_out_backlog_s
        fresh = [sim for sim in free if sim.remaining_work(self._now) <= threshold]
        if fresh:
            return self._join_route(job, candidate, self._best_gpu(fresh))
        loading = [sim for sim in sims if sim.is_loading]
        if loading:
            alternative = self._loading_route(job, candidate, loading)
        else:
            alternative = self._spawn_route(job, candidate)
        if not free:
            return alternative
        join = self._join_route(job, candidate, self._best_gpu(free))
        scales = bool(loading) or alternative.kind == "spawn"
        if scales and alternative.eta_s < SCALE_OUT_MARGIN * join.eta_s:
            return alternative
        return join

    def _best_gpu(self, sims: list[_Sim]) -> _Sim:
        """Return the GPU worker with the least predicted remaining work."""
        return min(
            sims,
            key=lambda sim: (sim.remaining_work(self._now), sim.load, sim.worker_id),
        )

    def _join_route(
        self, job: PendingJob, candidate: _Candidate, target: _Sim
    ) -> _Route:
        """Return the assignment of a GPU job to *target* with its join ETA.

        A GigaAM worker interleaves the windows of all its jobs, so a joining
        job starts at once. A Whisper worker runs long files sequentially in
        its inference loop, and the scheduler cannot know whether a job takes
        the batch path, so a Whisper join conservatively starts after the
        worker's remaining work.
        """
        family = provider_family(candidate.key.provider)
        wait = target.remaining_work(self._now) if family == "whisper" else 0.0
        return self._assign_route(job, candidate, target, wait_s=wait)

    def _assign_route(
        self,
        job: PendingJob,
        candidate: _Candidate,
        target: _Sim,
        *,
        wait_s: float = 0.0,
    ) -> _Route:
        """Return an immediate assignment route; its ETA is wait plus service."""
        share = self._assign_share(candidate, target)
        return _Route(
            kind="assign",
            candidate=candidate,
            eta_s=wait_s + self._service(job, candidate, share=share),
            reason=f"assigned to {target.worker_id}",
            target=target,
        )

    def _loading_route(
        self, job: PendingJob, candidate: _Candidate, loading: list[_Sim]
    ) -> _Route:
        """Return the route of a job that waits for a loading worker of its key."""
        wait = min(self._remaining_load(sim) for sim in loading)
        return _Route(
            kind="wait",
            candidate=candidate,
            eta_s=self._eta(job, candidate, wait),
            reason=f"waits for {loading[0].worker_id} to load",
            protectable=True,
        )

    def _spawn_route(self, job: PendingJob, candidate: _Candidate) -> _Route:
        """Return a spawn, eviction or wait route (spec 7.5 steps 4-6, 7.6)."""
        claim = self._claim
        load_s = candidate.estimate.load_s
        if claim is not None and claim.key == candidate.key:
            return _Route(
                kind="wait",
                candidate=candidate,
                eta_s=self._eta(job, candidate, load_s),
                reason=f"waits for the spawn reserved by {claim.job_id}",
                protectable=True,
            )
        blockers = self._check_spawn(candidate)
        protected = self._protected
        if protected is not None:
            reason = f"protected job {protected.job_id} has priority for spawns"
            return self._blocked_route(job, candidate, blockers, reason)
        if blockers.clear:
            capacity = self._fit_capacity(candidate, _NOTHING_FREED)
            return _Route(
                kind="spawn",
                candidate=candidate,
                eta_s=self._eta(job, candidate, load_s),
                reason=f"spawns {candidate.key.label} x{capacity}",
                spawn_capacity=capacity,
                protectable=True,
            )
        if blockers.hard or blockers.timing:
            return self._blocked_route(job, candidate, blockers)
        return self._evict_route(job, candidate, blockers)

    def _evict_route(
        self, job: PendingJob, candidate: _Candidate, blockers: _Blockers
    ) -> _Route:
        """Return the eviction route of a spawn blocked only by resources.

        Workers already stopping count as about to be freed; only idle ready
        workers of other keys that the waiting jobs of this tick do not keep
        (and not about to be taken by a later job of the same or a higher
        priority class) are evicted, least recently used first, and never
        more than a one-slot worker needs (spec 7.5 step 5).
        """
        need = self._spawn_need(candidate)
        eta = self._eta(job, candidate, candidate.estimate.load_s)
        pending = self._pending_freed()
        if not self._resource_blockers(need, pending):
            return _Route(
                kind="wait",
                candidate=candidate,
                eta_s=eta,
                reason="waits for stopping workers to exit",
                claims_spawn=True,
                protectable=True,
            )
        victims = self._victims(need, pending, self._eviction_pool(candidate.key))
        if victims:
            names = ", ".join(sim.worker_id for sim in victims)
            return _Route(
                kind="evict",
                candidate=candidate,
                eta_s=eta,
                reason=f"evicts {names} to spawn {candidate.key.label}",
                victims=victims,
                claims_spawn=True,
                protectable=True,
            )
        return self._blocked_route(job, candidate, blockers)

    def _blocked_route(
        self,
        job: PendingJob,
        candidate: _Candidate,
        blockers: _Blockers,
        reason: str | None = None,
    ) -> _Route:
        """Return a wait route with the spec 7.4 ``wait(k)`` estimate.

        Draining other keys' workers helps a protected job only when its spawn
        lacks resources and no worker of its own key serves it.
        """
        eta, resolvable = self._blocked_eta(job, candidate, blockers)
        drains = (
            bool(blockers.resources)
            and not blockers.hard
            and not self._serving(candidate.key)
        )
        return _Route(
            kind="wait",
            candidate=candidate,
            eta_s=eta,
            reason=reason or blockers.first(),
            protectable=resolvable,
            drain_need=self._spawn_need(candidate) if drains else None,
        )

    # * Spawn feasibility (spec 7.5 step 4).

    def _spawn_need(self, candidate: _Candidate, capacity: int = 1) -> _Need:
        """Return what a spawn of the candidate with *capacity* must find free."""
        if capacity == 1:
            res = candidate.min_reservation
        else:
            res = reservation_for(
                candidate.key, capacity, candidate.estimate, self._limits
            )
        if candidate.is_gpu:
            return _Need(
                commit=res.commit_bytes,
                vram=res.vram_bytes,
                threads=res.host_threads,
                gpu_worker=True,
            )
        return _Need(commit=res.commit_bytes, threads=self._limits.threads_per_slot)

    def _fit_capacity(self, candidate: _Candidate, freed: _Freed) -> int:
        """Return the largest capacity whose spawn fits now, or 0 if none does.

        Only commit and VRAM grow with the capacity, so the first fitting
        capacity from the ceiling down is the largest one.
        """
        for capacity in range(candidate.max_capacity, 0, -1):
            if not self._resource_blockers(
                self._spawn_need(candidate, capacity), freed
            ):
                return capacity
        return 0

    def _check_spawn(self, candidate: _Candidate) -> _Blockers:
        """Classify every reason a one-slot worker of the candidate cannot spawn."""
        timing, delay, cooldown = self._timing_blockers(candidate)
        need = self._spawn_need(candidate)
        return _Blockers(
            hard=tuple(self._hard_blockers(candidate)),
            timing=tuple(timing),
            delay_s=delay,
            resources=tuple(self._resource_blockers(need, _NOTHING_FREED)),
            cooldown=cooldown,
        )

    def _hard_blockers(self, candidate: _Candidate) -> list[str]:
        """Return spawn blockers that neither time nor eviction resolve soon."""
        hard: list[str] = []
        if self._pressure:
            hard.append("free system commit is below the critical level")
        if candidate.is_gpu and not self._probes.cuda_present:
            hard.append("no CUDA device detected")
        return hard

    def _timing_blockers(self, candidate: _Candidate) -> tuple[list[str], float, bool]:
        """Return spawn blockers that clear by themselves, their delay, cooldown."""
        timing: list[str] = []
        delays = [0.0]
        cooldown = False
        loading = [sim for sim in self._sims if sim.is_loading]
        if loading:
            timing.append(f"cold load of {loading[0].key.label} in progress")
            delays.append(max(self._remaining_load(sim) for sim in loading))
        if self._claim is not None:
            timing.append(f"spawn reserved for job {self._claim.job_id}")
            delays.append(self._estimate(self._claim.key).load_s)
        status = self._state.key_status.get(candidate.key)
        if status is not None:
            for until, what in (
                (status.crash_cooldown_until, "crash"),
                (status.load_failure_until, "load-failure"),
            ):
                if until is not None and until > self._now:
                    timing.append(f"{candidate.key.label} is in {what} cooldown")
                    delays.append(until - self._now)
                    cooldown = True
        if candidate.is_gpu and not self._vram_sample_fresh():
            timing.append("waiting for a fresh VRAM sample")
        if not candidate.is_gpu and self._cpu_running() >= self._limits.cpu_slots:
            timing.append("no free CPU slot")
            delays.append(self._earliest_cpu_end())
        return timing, max(delays), cooldown

    def _resource_blockers(self, need: _Need, freed: _Freed) -> list[str]:
        """Return budget and probe shortfalls of *need* given released resources."""
        limits = self._limits
        out: list[str] = []
        if need.commit > 0:
            out.extend(self._commit_blockers(need.commit, freed))
        threads_after = self._threads_used() - freed.host_threads + need.threads
        if threads_after > limits.thread_budget:
            out.append(
                f"thread budget: {threads_after} threads after the placement > "
                f"{limits.thread_budget}"
            )
        if need.gpu_worker:
            out.extend(self._gpu_resource_blockers(need, freed))
        return out

    def _commit_blockers(self, commit: int, freed: _Freed) -> list[str]:
        """Return commit-budget and free-commit shortfalls of a new reservation."""
        limits = self._limits
        out: list[str] = []
        commit_after = self._commit_reserved() - freed.commit + commit
        if commit_after > limits.commit_budget_bytes:
            out.append(
                f"commit budget: {_gib(commit_after)} GiB reserved after the spawn > "
                f"{_gib(limits.commit_budget_bytes)} GiB"
            )
        free_commit = self._probes.free_commit_bytes
        guard = limits.free_commit_guard_bytes
        if free_commit is not None and free_commit + freed.commit < commit + guard:
            out.append(
                f"free system commit {_gib(free_commit)} GiB < "
                f"{_gib(commit)} GiB + {_gib(guard)} GiB guard"
            )
        return out

    def _gpu_resource_blockers(self, need: _Need, freed: _Freed) -> list[str]:
        """Return GPU worker-count and VRAM shortfalls of a GPU spawn."""
        limits = self._limits
        out: list[str] = []
        workers_after = self._gpu_workers() - freed.gpu_workers + 1
        if workers_after > limits.gpu_workers:
            out.append(f"GPU worker limit {limits.gpu_workers} reached")
        vram_after = self._vram_reserved() - freed.vram + need.vram
        if vram_after > limits.vram_budget_bytes:
            out.append(
                f"VRAM budget: {_gib(vram_after)} GiB reserved after the spawn > "
                f"{_gib(limits.vram_budget_bytes)} GiB"
            )
        free_vram = self._probes.free_vram_bytes
        guard = limits.free_vram_guard_bytes
        if free_vram is not None and free_vram + freed.vram < need.vram + guard:
            out.append(
                f"free VRAM {_gib(free_vram)} GiB < {_gib(need.vram)} GiB + "
                f"{_gib(guard)} GiB guard"
            )
        return out

    def _vram_sample_fresh(self) -> bool:
        """Return whether a VRAM sample newer than the last GPU change exists."""
        probes = self._probes
        if probes.free_vram_bytes is None or probes.vram_sample_ts is None:
            return False
        last = self._state.last_gpu_change_ts
        return last is None or probes.vram_sample_ts > last

    # * Eviction (spec 7.5 step 5), release estimates and draining (7.6).

    def _pending_freed(self) -> _Freed:
        """Return the reservations of workers that are stopping or stopped now."""
        return _fold(_NOTHING_FREED, (sim for sim in self._sims if sim.freeing))

    def _eviction_pool(self, key: WorkerKey) -> list[_Sim]:
        """Return evictable idle ready workers of other keys, LRU first.

        The idle workers that the earlier waiting jobs of this tick keep
        (:meth:`_spared_idle`, proportional to those jobs) and the workers that
        a pending job of the current job's class or a higher one is about to
        take are kept.
        """
        spared = self._spared_idle()
        pool = [
            sim
            for sim in self._sims
            if sim.idle_ready
            and not sim.planned
            and sim.key != key
            and sim.worker_id not in spared
            and sim.worker_id not in self._usable_idle
        ]
        return sorted(pool, key=lambda sim: (sim.last_used_at, sim.worker_id))

    def _victims(self, need: _Need, base: _Freed, pool: list[_Sim]) -> tuple[_Sim, ...]:
        """Return a minimal prefix-ordered set of *pool* that makes *need* fit.

        Workers are added in pool order until the resources fit, then every
        chosen worker whose release turns out unnecessary is dropped again,
        last-added first. Returns an empty tuple when even the whole pool is
        not enough.
        """
        chosen: list[_Sim] = []
        freed = base
        for sim in pool:
            if not self._resource_blockers(need, freed):
                break
            chosen.append(sim)
            freed = freed.plus(sim)
        if not chosen or self._resource_blockers(need, freed):
            return ()
        for sim in reversed(tuple(chosen)):
            trial = [other for other in chosen if other is not sim]
            if not self._resource_blockers(need, _fold(base, trial)):
                chosen = trial
        return tuple(chosen)

    def _evictable(self, need: _Need, key: WorkerKey) -> bool:
        """Return whether evicting idle workers of other keys makes *need* fit."""
        pool = self._eviction_pool(key)
        return not self._resource_blockers(need, _fold(self._pending_freed(), pool))

    def _release_wait(self, need: _Need, key: WorkerKey) -> float:
        """Return the predicted wait until running work releases enough for *need*.

        Workers already stopping count as released now. Every ready or
        retiring worker of another key (not spawned in this tick, not drained
        for the protected job) releases its reservation once its running jobs
        end, as a drain would make it; every running CPU job releases its
        threads at its expected end. The result is the earliest of these times
        at which the released resources suffice: for worker releases, the
        largest remaining work of the cheapest sufficient set of workers.
        ``inf`` when even every release is not enough.
        """
        freed = self._pending_freed()
        if not self._resource_blockers(need, freed):
            return 0.0
        job_threads = _Freed(host_threads=self._limits.threads_per_slot)
        events: list[tuple[float, str, _Freed]] = []
        for sim in self._sims:
            if sim.freeing or sim.planned:
                continue
            if not sim.is_gpu:
                events.extend(
                    (max(0.0, end - self._now), sim.worker_id, job_threads)
                    for end in sim.ends
                )
            if sim.key != key and sim.state in _DRAINABLE_STATES and not sim.draining:
                events.append(
                    (
                        sim.remaining_work(self._now),
                        sim.worker_id,
                        _NOTHING_FREED.plus(sim),
                    )
                )
        for at, _, release in sorted(events, key=lambda event: (event[0], event[1])):
            freed = freed.add(release)
            if not self._resource_blockers(need, freed):
                return at
        return math.inf

    def _protect(self, job: PendingJob, route: _Route) -> None:
        """Make *job* the protected job of this tick (starvation guard, 7.6).

        Later jobs leave the protected job's CPU slot and threads free (for a
        GPU job waiting for its own spawn: its host threads), may not spawn or
        evict, and may not use workers drained for it. When the protected job
        waits for resources that only other keys' workers release, a minimal
        set of those workers is drained: idle ones are evicted now, busy ones
        get no new jobs and are evicted once idle, so a stream of small jobs
        cannot keep them busy forever.
        """
        candidate = route.candidate
        if candidate.is_gpu:
            needs_spawn = route.kind != "spawn" and not self._serving(candidate.key)
            slots = 0
            threads = candidate.min_reservation.host_threads if needs_spawn else 0
        else:
            slots = 1
            threads = self._limits.threads_per_slot
        self._protected = _Protected(job.job_id, candidate.key, slots, threads)
        self._notes.append(
            f"{job.job_id}: protected after waiting "
            f"{_seconds(self._now - job.waiting_since)}"
        )
        if route.drain_need is not None:
            self._drain(job, route.drain_need, candidate.key)

    def _drain(self, job: PendingJob, need: _Need, key: WorkerKey) -> None:
        """Drain the fewest workers of other keys that make the job's need fit.

        Workers already stopping, retiring workers of other keys and the
        threads of running CPU jobs count as released (the protection keeps
        the threads of ending CPU jobs for the job). Idle workers go first
        (LRU), then busy ones by remaining work. Idle workers that a later job
        of the protected job's class (or a higher one) is about to take go
        last: they are drained only when nothing else suffices, because the
        starvation guard outranks them.
        """
        base = _fold(
            self._pending_freed(),
            (
                sim
                for sim in self._sims
                if sim.state == "retiring" and not sim.stopped and sim.key != key
            ),
        )
        cpu_threads = self._cpu_running() * self._limits.threads_per_slot
        base = base.add(_Freed(host_threads=cpu_threads))
        if not self._resource_blockers(need, base):
            self._notes.append(
                f"{job.job_id}: waits for running jobs and stopping or retiring "
                "workers to release resources"
            )
            return
        pool = sorted(
            (
                sim
                for sim in self._sims
                if sim.state == "ready"
                and not sim.stopped
                and not sim.planned
                and sim.key != key
            ),
            key=self._drain_order,
        )
        victims = self._victims(need, base, pool)
        if not victims:
            self._notes.append(
                f"{job.job_id}: other workers cannot release enough resources"
            )
            return
        for sim in victims:
            if sim.ends:
                sim.draining = True
            else:
                self._stop(sim, "evict")
        names = ", ".join(sim.worker_id for sim in victims)
        self._notes.append(f"{job.job_id}: drains {names}")

    def _drain_order(self, sim: _Sim) -> tuple[int, float, float, str]:
        """Sort key for draining: idle LRU, busy by remaining work, then usable."""
        if not sim.ends:
            rank = 2 if sim.worker_id in self._usable_idle else 0
            return (rank, 0.0, sim.last_used_at, sim.worker_id)
        return (1, sim.remaining_work(self._now), sim.last_used_at, sim.worker_id)

    # * ETA (spec 7.4).

    def _eta(
        self,
        job: PendingJob,
        candidate: _Candidate,
        wait_s: float,
        *,
        share: int | None = None,
    ) -> float:
        """Return ``wait + backlog / parallel + service`` of a waiting job.

        The backlog holds the work of the earlier jobs of this tick that wait
        for the same key; jobs assigned in this tick occupy slots or lanes
        instead and show up in ``wait``, in the GPU share and in the CPU load.
        """
        backlog = self._backlog_s.get(candidate.key, 0.0)
        return (
            wait_s
            + backlog / self._parallel(candidate)
            + self._service(job, candidate, share=share)
        )

    def _blocked_eta(
        self, job: PendingJob, candidate: _Candidate, blockers: _Blockers
    ) -> tuple[float, bool]:
        """Return the ETA of a job whose spawn cannot happen now and its resolvability.

        Options: the spawn once its timing blockers clear and its resources
        are released (at once when evicting idle workers suffices, otherwise
        when the busy workers that hold them finish, see
        :meth:`_release_wait`), or the earliest end of a running job on a
        ready worker of the key (GPU: the job then joins a full batch). The
        wait is resolvable by scheduling when a finite option exists that is
        not behind a crash or load-failure cooldown. ``inf`` when no option
        applies.
        """
        options: list[tuple[float, bool]] = []
        if not blockers.hard:
            need = self._spawn_need(candidate)
            release = 0.0
            if blockers.resources and not self._evictable(need, candidate.key):
                release = self._release_wait(need, candidate.key)
            if math.isfinite(release):
                wait = max(blockers.delay_s, release) + candidate.estimate.load_s
                eta = self._eta(job, candidate, wait)
                options.append((eta, not blockers.cooldown))
        for sim in self._serving(candidate.key):
            if sim.state == "ready" and sim.ends:
                wait = max(0.0, min(sim.ends) - self._now)
                share = sim.capacity if candidate.is_gpu else None
                options.append((self._eta(job, candidate, wait, share=share), True))
        if not options:
            return math.inf, False
        best = min(eta for eta, _ in options)
        resolvable = any(ok and math.isfinite(eta) for eta, ok in options)
        return best, resolvable

    def _parallel(self, candidate: _Candidate) -> int:
        """Return ``parallel(k)`` of spec 7.4 (at least 1)."""
        sims = self._serving(candidate.key)
        if candidate.is_gpu:
            return max(1, len(sims))
        capacity = sum(sim.capacity for sim in sims)
        if not capacity:
            fitted = self._fit_capacity(candidate, _NOTHING_FREED)
            capacity = fitted or candidate.max_capacity
        running = sum(sim.load for sim in sims)
        free_slots = max(0, self._limits.cpu_slots - self._cpu_running())
        return max(1, min(capacity, free_slots + running))

    def _service(
        self, job: PendingJob, candidate: _Candidate, *, share: int | None = None
    ) -> float:
        """Return the corrected service time of a job on the candidate.

        Without an explicit *share*, a GPU job counts alone on its worker (the
        full-throughput work of a waiting job) and a CPU job shares the cores
        with the CPU load predicted for its start (:meth:`_cpu_share`).
        """
        if share is None:
            share = 1 if candidate.is_gpu else self._cpu_share()
        return service_seconds(
            candidate.key,
            candidate.estimate,
            self._duration(job),
            share=share,
            cpu_busy=self._cpu_busy(),
            contention=self._limits.gpu_cpu_contention,
        )

    def _assign_share(self, candidate: _Candidate, target: _Sim) -> int:
        """Return ``n`` of an assignment: GPU worker load or CPU load, plus one."""
        if candidate.is_gpu:
            return target.load + 1
        return self._cpu_running() + 1

    def _cpu_share(self) -> int:
        """Return the CPU jobs predicted to run alongside a waiting CPU job.

        Jobs running now, the jobs of this tick that wait for a CPU worker
        ahead of it and the job itself, at most the CPU slot limit.
        """
        waiting = sum(
            count for key, count in self._backlog_n.items() if key.backend == "cpu"
        )
        return max(1, min(self._limits.cpu_slots, self._cpu_running() + waiting + 1))

    def _duration(self, job: PendingJob) -> float:
        """Return the known duration, or ``unknown_duration_s``."""
        known = self._known_duration(job)
        return self._limits.unknown_duration_s if known is None else known

    @staticmethod
    def _known_duration(job: PendingJob) -> float | None:
        """Return the job's duration when it is a finite non-negative number."""
        duration = job.duration_s
        if duration is None or not math.isfinite(duration) or duration < 0:
            return None
        return duration

    def _remaining_load(self, sim: _Sim) -> float:
        """Return the predicted remaining cold-load time of a loading worker."""
        load_s = self._estimate(sim.key).load_s
        return max(0.0, sim.spawned_at + load_s - self._now)

    def _earliest_cpu_end(self) -> float:
        """Return the time until the first running CPU job ends (``inf`` if none)."""
        ends = [end for sim in self._sims if not sim.is_gpu for end in sim.ends]
        return max(0.0, min(ends) - self._now) if ends else math.inf

    # * Resource accounting (spec 7.2): every worker counts until it exits.

    def _serving(self, key: WorkerKey) -> list[_Sim]:
        """Return the ready, spawning and loading workers of *key*."""
        return [sim for sim in self._sims if sim.key == key and sim.serving]

    def _cpu_running(self) -> int:
        """Return the CPU slots in use (running CPU jobs on workers in any state)."""
        return sum(sim.load for sim in self._sims if not sim.is_gpu)

    def _threads_used(self) -> int:
        """Return CPU job threads plus host threads of every GPU worker."""
        host = sum(sim.reservation.host_threads for sim in self._sims)
        return self._cpu_running() * self._limits.threads_per_slot + host

    def _gpu_workers(self) -> int:
        """Return the GPU workers whose exit is not confirmed."""
        return sum(1 for sim in self._sims if sim.is_gpu)

    def _commit_reserved(self) -> int:
        """Return the commit reserved by every worker."""
        return sum(sim.reservation.commit_bytes for sim in self._sims)

    def _vram_reserved(self) -> int:
        """Return the VRAM reserved by every worker."""
        return sum(sim.reservation.vram_bytes for sim in self._sims)

    def _cpu_busy(self) -> float:
        """Return CPU job threads in use divided by the thread budget, in [0, 1]."""
        used = self._cpu_running() * self._limits.threads_per_slot
        return min(1.0, used / max(1, self._limits.thread_budget))

    def _cpu_slot_free(self) -> bool:
        """Return whether a CPU job may take a slot now.

        A later job must leave the protected job's slot and threads free.
        """
        protected = self._protected
        slots = 1 + (protected.slots if protected is not None else 0)
        threads = self._limits.threads_per_slot + (
            protected.threads if protected is not None else 0
        )
        return (
            self._cpu_running() + slots <= self._limits.cpu_slots
            and self._threads_used() + threads <= self._limits.thread_budget
        )


def _reject(job: PendingJob, status: RejectStatus, kind: str, detail: str) -> Reject:
    """Build a rejection of *job*."""
    return Reject(job_id=job.job_id, status=status, error_kind=kind, detail=detail)
