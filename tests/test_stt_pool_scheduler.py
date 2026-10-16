"""Decision scenarios for the unwired pool planner, using real seed estimates."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from core.stt_pool.config import load_pool_config
from core.stt_pool.estimates import EstimateStore
from core.stt_pool.keys import make_key
from core.stt_pool.scheduler import (
    Assign,
    KeyStatus,
    Limits,
    PendingJob,
    Probes,
    Reject,
    Reservation,
    RunningJob,
    SchedulerState,
    Spawn,
    Stop,
    admission_order,
    commit_growth_blockers,
    plan,
)

if TYPE_CHECKING:
    from core.stt_pool.keys import Backend, DeviceRequest, Priority
    from core.stt_pool.scheduler import WorkerState

from core.stt_pool.scheduler import WorkerView

_GIB = 1024**3
_NOW = 1000.0


@pytest.fixture
def limits() -> Limits:
    """Read the actual example configuration as the planner's input."""
    path = Path(__file__).resolve().parents[1] / "stt_pool.example.json"
    return Limits.from_config(load_pool_config(path, env={}, logical_cpus=32))


@pytest.fixture
def estimates() -> EstimateStore:
    """Use the production estimator, without persistence or hardware probing."""
    return EstimateStore(None, fingerprint={"cpu": "test"}, clock=lambda: _NOW)


def _job(
    seq: int,
    *,
    call: str = "batch",
    device: DeviceRequest = "cpu",
    model: str = "small",
    duration: float | None = 60.0,
    cap: int | None = None,
    priority: Priority = "normal",
    since: float = _NOW,
) -> PendingJob:
    """Build an admitted request with independent job and call identities."""
    return PendingJob(
        job_id=f"{seq:032x}",
        call_id=call,
        seq=seq,
        submitted_at=since,
        waiting_since=since,
        deadline_at=None,
        provider="whisper",
        whisper_model=model,
        compute_request="auto",
        device_request=device,
        duration_s=duration,
        max_parallel=cap,
        priority=priority,
    )


def _worker(
    name: str,
    estimates: EstimateStore,
    *,
    backend: Backend = "cpu",
    model: str = "small",
    capacity: int = 2,
    running: tuple[RunningJob, ...] = (),
    state: WorkerState = "ready",
    idle_since: float | None = _NOW,
) -> WorkerView:
    """Reserve memory for a worker according to the documented resource formula."""
    key = make_key("whisper", model, "auto", backend)
    estimate = estimates.estimate(key)
    return WorkerView(
        worker_id=name,
        key=key,
        state=state,
        capacity=capacity,
        running=running,
        idle_since=idle_since if not running else None,
        spawned_at=_NOW - 1,
        last_used_at=0.0,
        reservation=Reservation(
            commit_bytes=estimate.commit_base_bytes
            + capacity * estimate.commit_per_slot_bytes,
            vram_bytes=(
                estimate.vram_base_bytes + capacity * estimate.vram_per_lane_bytes
                if backend == "gpu"
                else 0
            ),
            host_threads=2 if backend == "gpu" else 0,
        ),
    )


def _state(
    limits: Limits,
    estimates: EstimateStore,
    pending: tuple[PendingJob, ...],
    workers: tuple[WorkerView, ...] = (),
    *,
    now: float = _NOW,
) -> SchedulerState:
    """Build a tick with ample measured memory and a current VRAM sample."""
    return SchedulerState(
        now=now,
        limits=limits,
        probes=Probes(64 * _GIB, 12 * _GIB, now, cuda_present=True),
        pending=pending,
        workers=workers,
        key_status={},
        estimates=estimates,
    )


def _running(call: str, *, end: float = _NOW + 60) -> RunningJob:
    """Build a job already in service, which priority cannot preempt."""
    return RunningJob(f"running-{call}", call, _NOW - 1, end)


@pytest.mark.parametrize(
    ("budget", "free", "guard", "critical", "target", "blocked"),
    [
        (8 * _GIB, 10 * _GIB, _GIB, 0, 5 * _GIB, False),
        (8 * _GIB - 1, 10 * _GIB, _GIB, 0, 5 * _GIB, True),
        (8 * _GIB, 3 * _GIB, 2 * _GIB, 0, 5 * _GIB, False),
        (8 * _GIB, 3 * _GIB - 1, 2 * _GIB, 0, 5 * _GIB, True),
        (8 * _GIB, 2 * _GIB, 0, 2 * _GIB, 4 * _GIB + 1, False),
        (8 * _GIB, 2 * _GIB - 1, 0, 2 * _GIB, 4 * _GIB + 1, True),
        (7 * _GIB, 0, 0, _GIB, 3 * _GIB, False),
    ],
)
def test_live_commit_growth_counts_other_workers_and_inclusive_boundaries(
    limits: Limits,
    estimates: EstimateStore,
    budget: int,
    free: int,
    guard: int,
    critical: int,
    target: int,
    *,
    blocked: bool,
) -> None:
    """Catch overbooking live growth or rejecting an exact budget/guard fit."""
    owner = replace(
        _worker("owner", estimates), reservation=Reservation(4 * _GIB, 0, 0)
    )
    stopping = replace(
        _worker("other", estimates, backend="gpu", state="stopping"),
        reservation=Reservation(3 * _GIB, _GIB, 1),
    )
    state = _state(limits, estimates, (), (owner, stopping))
    state = replace(
        state,
        limits=replace(
            limits,
            commit_budget_bytes=budget,
            free_commit_guard_bytes=guard,
            critical_free_commit_bytes=critical,
        ),
        probes=replace(state.probes, free_commit_bytes=free),
    )
    assert bool(commit_growth_blockers(state, "owner", target)) is blocked


def test_configuration_demand_and_ttl_sequence(
    limits: Limits, estimates: EstimateStore
) -> None:
    """Catch provider/model substitution and failure to retire model A after TTL."""
    first = plan(_state(limits, estimates, (_job(1, model="base"),)))
    spawns = [a for a in first.actions if isinstance(a, Spawn)]
    assert len(spawns) == 1
    assert spawns[0].key.model == "base"

    model_a = _worker("a", estimates, model="base", idle_since=0.0)
    model_b = _worker("b", estimates, capacity=2)
    jobs = (_job(2, cap=2), _job(3, cap=2), _job(4, cap=2))
    result = plan(_state(limits, estimates, jobs, (model_a, model_b)))
    assert [a.job_id for a in result.actions if isinstance(a, Assign)] == [
        jobs[0].job_id,
        jobs[1].job_id,
    ]
    assert Stop("a", "ttl") in result.actions
    assert not any(isinstance(a, Stop) and a.worker_id == "b" for a in result.actions)


def test_reduced_call_cap_retires_unused_duplicate_after_ttl(
    limits: Limits, estimates: EstimateStore
) -> None:
    """Catch an unused second model-B worker retained by a max-parallel-one batch."""
    workers = tuple(
        _worker(name, estimates, backend="gpu", capacity=1, idle_since=0.0)
        for name in ("b1", "b2")
    )
    jobs = (_job(1, device="cuda", cap=1), _job(2, device="cuda", cap=1))
    result = plan(_state(limits, estimates, jobs, workers))
    assigned = [a for a in result.actions if isinstance(a, Assign)]
    assert len(assigned) == 1
    assert assigned[0].worker_id == "b1"
    assert Stop("b2", "ttl") in result.actions


@pytest.mark.parametrize(
    ("waiting_count", "fresher_since", "retired"),
    [
        (1, 1.0, ("older",)),
        (2, 1.0, ("older",)),
        (3, 1.0, ()),
        (1, _NOW - 1, ("older",)),
    ],
)
def test_uncapped_waiting_demand_keeps_only_needed_idle_capacity(
    limits: Limits,
    estimates: EstimateStore,
    waiting_count: int,
    fresher_since: float,
    retired: tuple[str, ...],
) -> None:
    """Catch surplus duplicates retained forever or needed fresher capacity retired by TTL."""
    busy = _worker(
        "busy",
        estimates,
        model="base",
        capacity=limits.cpu_slots,
        running=tuple(_running(f"other-{index}") for index in range(limits.cpu_slots)),
    )
    workers = (
        busy,
        _worker("older", estimates, capacity=2, idle_since=0.0),
        _worker("fresher", estimates, capacity=2, idle_since=fresher_since),
    )
    jobs = tuple(_job(seq) for seq in range(1, waiting_count + 1))
    result = plan(_state(limits, estimates, jobs, workers))
    stops = tuple(a for a in result.actions if isinstance(a, Stop))
    assert stops == tuple(Stop(worker_id, "ttl") for worker_id in retired)
    assert not any(isinstance(a, Assign | Spawn | Reject) for a in result.actions)


@pytest.mark.parametrize(("cpu_slots", "gpu_workers"), [(0, 2), (8, 0), (1, 1)])
def test_cpu_and_gpu_limits_are_independent(
    limits: Limits, estimates: EstimateStore, cpu_slots: int, gpu_workers: int
) -> None:
    """Catch one device class consuming or disabling the other class's quota."""
    limits = replace(limits, cpu_slots=cpu_slots, gpu_workers=gpu_workers)
    workers = tuple(
        worker
        for worker, enabled in (
            (_worker("cpu", estimates, capacity=8), cpu_slots > 0),
            (_worker("gpu", estimates, backend="gpu", capacity=4), gpu_workers > 0),
        )
        if enabled
    )
    jobs = tuple(_job(i, device="cpu" if i < 10 else "cuda") for i in range(1, 14))
    result = plan(_state(limits, estimates, jobs, workers))
    assigned = [a for a in result.actions if isinstance(a, Assign)]
    assert sum(a.backend == "cpu" for a in assigned) == min(cpu_slots, 8)
    assert sum(a.backend == "gpu" for a in assigned) == (4 if gpu_workers else 0)


def test_shared_commit_reservation_survives_stop_until_exit(
    limits: Limits, estimates: EstimateStore
) -> None:
    """Catch spawning against memory still held by a stopping worker."""
    old = _worker("old", estimates, backend="gpu", capacity=1, model="medium")
    limits = replace(limits, commit_budget_bytes=7 * _GIB)
    job = _job(1)
    eviction = plan(_state(limits, estimates, (job,), (old,)))
    assert Stop("old", "evict") in eviction.actions
    assert not any(isinstance(a, Spawn) for a in eviction.actions)
    stopping = replace(old, state="stopping")
    waiting = plan(_state(limits, estimates, (job,), (stopping,)))
    assert not any(isinstance(a, Spawn) for a in waiting.actions)
    after_exit = plan(_state(limits, estimates, (job,)))
    assert len([a for a in after_exit.actions if isinstance(a, Spawn)]) == 1


@pytest.mark.parametrize("resource", ["commit", "vram", "threads"])
def test_spawn_capacity_fits_the_remaining_budget(
    limits: Limits, estimates: EstimateStore, resource: str
) -> None:
    """Catch rejecting a feasible one-lane worker or reserving an oversized batch."""
    if resource == "commit":
        limits = replace(limits, commit_budget_bytes=int(3.35 * _GIB))
    elif resource == "vram":
        limits = replace(limits, vram_budget_bytes=int(1.45 * _GIB))
    else:
        limits = replace(limits, thread_budget=2)
    result = plan(_state(limits, estimates, (_job(1, device="cuda"),)))
    spawns = [a for a in result.actions if isinstance(a, Spawn)]
    assert len(spawns) == 1
    assert spawns[0].reservation.commit_bytes <= limits.commit_budget_bytes
    assert spawns[0].reservation.vram_bytes <= limits.vram_budget_bytes
    assert spawns[0].reservation.host_threads <= limits.thread_budget
    if resource in {"commit", "vram"}:
        assert spawns[0].capacity == 1


@pytest.mark.parametrize(
    ("free_gib", "expected_reason"), [(3, "pressure"), (8, "evict"), (64, None)]
)
def test_commit_guard_and_pressure_preserve_busy_workers(
    limits: Limits, estimates: EstimateStore, free_gib: int, expected_reason: str | None
) -> None:
    """Catch low commit spawning a model or pressure killing an active job."""
    workers = (
        _worker("idle", estimates),
        _worker("busy", estimates, running=(_running("active"),)),
    )
    state = _state(limits, estimates, (_job(1, model="base"),), workers)
    state = replace(
        state, probes=replace(state.probes, free_commit_bytes=free_gib * _GIB)
    )
    result = plan(state)
    stops = [a.worker_id for a in result.actions if isinstance(a, Stop)]
    assert "busy" not in stops
    if expected_reason is None:
        assert "idle" not in stops
    else:
        assert Stop("idle", expected_reason) in result.actions
    assert any(isinstance(a, Spawn) for a in result.actions) is (free_gib == 64)


@pytest.mark.parametrize("sample_time", [None, _NOW - 1, _NOW, _NOW + 1])
def test_gpu_spawn_requires_a_probe_after_the_last_gpu_change(
    limits: Limits, estimates: EstimateStore, sample_time: float | None
) -> None:
    """Catch a second GPU reservation using a stale free-VRAM reading."""
    state = _state(limits, estimates, (_job(1, device="cuda"),))
    state = replace(
        state,
        last_gpu_change_ts=_NOW,
        probes=replace(state.probes, vram_sample_ts=sample_time),
    )
    result = plan(state)
    assert any(isinstance(a, Spawn) for a in result.actions) is (
        sample_time == _NOW + 1
    )


@pytest.mark.parametrize(
    ("duration", "gpu_wait", "backend"),
    [(60.0, 5, "gpu"), (60.0, 50, "cpu"), (None, 50, "gpu")],
)
def test_auto_spills_only_for_a_known_faster_cpu_eta(
    limits: Limits,
    estimates: EstimateStore,
    duration: float | None,
    gpu_wait: int,
    backend: str,
) -> None:
    """Catch slow CPU spill or spill based on an invented media duration."""
    limits = replace(limits, gpu_workers=1)
    workers = (
        _worker(
            "gpu",
            estimates,
            backend="gpu",
            capacity=1,
            running=(_running("other", end=_NOW + gpu_wait),),
        ),
        _worker("cpu", estimates),
    )
    result = plan(
        _state(limits, estimates, (_job(1, device="auto", duration=duration),), workers)
    )
    assignments = [a for a in result.actions if isinstance(a, Assign)]
    if backend == "cpu":
        assert len(assignments) == 1
        assert assignments[0].backend == "cpu"
    else:
        assert not assignments
        assert not any(isinstance(a, Reject) for a in result.actions)


def test_long_auto_queue_uses_gpu_lanes_and_cpu_slots(
    limits: Limits, estimates: EstimateStore
) -> None:
    """Catch an auto batch leaving CPU slots unused behind saturated GPU lanes."""
    limits = replace(limits, gpu_workers=1)
    workers = (
        _worker("gpu", estimates, backend="gpu", capacity=4),
        _worker("cpu", estimates, capacity=8),
    )
    jobs = tuple(_job(i, device="auto") for i in range(1, 81))
    result = plan(_state(limits, estimates, jobs, workers))
    assigned = [a for a in result.actions if isinstance(a, Assign)]
    assert assigned[0].backend == "gpu"
    assert sum(a.backend == "gpu" for a in assigned) == 4
    assert sum(a.backend == "cpu" for a in assigned) == 8


def test_max_parallel_caps_only_its_call_across_backends(
    limits: Limits, estimates: EstimateStore
) -> None:
    """Catch a per-call cap applied globally or independently per backend."""
    workers = (
        _worker("cpu", estimates, capacity=8),
        _worker(
            "gpu", estimates, backend="gpu", capacity=4, running=(_running("limited"),)
        ),
    )
    jobs = (_job(1, call="limited", cap=1), _job(2, call="independent", cap=1))
    result = plan(_state(limits, estimates, jobs, workers))
    assert [a.job_id for a in result.actions if isinstance(a, Assign)] == [
        jobs[1].job_id
    ]


def test_priority_and_round_robin_order_are_independent_of_input_order() -> None:
    """Catch FIFO batches delaying a single-file call or lower classes overtaking high."""
    jobs = (
        _job(1, call="offline"),
        _job(2, call="offline"),
        _job(3, call="offline"),
        _job(4, call="bot"),
        _job(5, call="urgent", priority="high"),
        _job(6, call="background", priority="low"),
    )
    expected = [jobs[i].job_id for i in (4, 0, 3, 1, 2, 5)]
    for pending in (jobs, jobs[::-1], jobs[2:] + jobs[:2]):
        assert [
            j.job_id for j in admission_order(pending, now=_NOW, aging_s=300)
        ] == expected


@pytest.mark.parametrize(
    ("wait_s", "expected"), [(299, (2, 3, 1)), (300, (2, 1, 3)), (600, (1, 2, 3))]
)
def test_low_priority_aging_reaches_eventual_admission(
    wait_s: int, expected: tuple[int, ...]
) -> None:
    """Catch low-priority jobs starving forever under fresh higher-priority calls."""
    jobs = (
        _job(1, call="old", priority="low", since=_NOW - wait_s),
        _job(2, call="high", priority="high"),
        _job(3, call="normal"),
    )
    ordered = admission_order(jobs, now=_NOW, aging_s=300)
    assert tuple(j.seq for j in ordered) == expected


@pytest.mark.parametrize("since", [_NOW - 1, _NOW])
def test_fair_share_survives_one_slot_per_tick(
    limits: Limits, estimates: EstimateStore, since: float
) -> None:
    """Catch a batch taking consecutive cross-tick turns despite a waiting peer call."""
    limits = replace(limits, cpu_slots=1)
    workers = (_worker("cpu", estimates, capacity=1),)
    jobs = tuple(
        _job(i, call="offline" if i < 4 else "bot", since=since) for i in range(1, 5)
    )
    state = _state(limits, estimates, jobs, workers)
    first = next(a for a in plan(state).actions if isinstance(a, Assign))
    assert first.job_id == jobs[0].job_id
    state = replace(
        state, now=_NOW + 1, pending=jobs[1:], call_last_started={"offline": _NOW}
    )
    second = next(a for a in plan(state).actions if isinstance(a, Assign))
    assert second.job_id == jobs[3].job_id


def test_high_priority_does_not_preempt_a_running_low_job(
    limits: Limits, estimates: EstimateStore
) -> None:
    """Catch priority cancelling or replacing work already in service."""
    limits = replace(limits, cpu_slots=1)
    busy = _worker("cpu", estimates, capacity=1, running=(_running("low"),))
    result = plan(_state(limits, estimates, (_job(1, priority="high"),), (busy,)))
    assert not any(isinstance(a, Assign | Stop) for a in result.actions)


def test_starvation_guard_drains_without_killing_busy_workers(
    limits: Limits, estimates: EstimateStore
) -> None:
    """Catch endless admission to another key preventing an old configuration from loading."""
    limits = replace(limits, commit_budget_bytes=3 * _GIB)
    busy = _worker("busy", estimates, capacity=2, running=(_running("ongoing"),))
    old = _job(1, call="old", model="base", since=_NOW - 61)
    later = _job(2, call="stream")
    result = plan(_state(limits, estimates, (old, later), (busy,)))
    assert not any(isinstance(a, Assign | Stop | Spawn) for a in result.actions)
    idle = replace(busy, running=(), idle_since=_NOW)
    result = plan(_state(limits, estimates, (old, later), (idle,)))
    assert Stop("busy", "evict") in result.actions
    after_exit = plan(_state(limits, estimates, (old, later)))
    assert (
        next(a for a in after_exit.actions if isinstance(a, Spawn)).key.model == "base"
    )


def test_cooldowns_do_not_freeze_other_configurations(
    limits: Limits, estimates: EstimateStore
) -> None:
    """Catch a cooling-down request blocking unrelated admission or losing its load error."""
    failed = _job(1, since=_NOW - 100)
    valid = _job(2, model="base")
    key = make_key("whisper", "small", "auto", "cpu")
    state = replace(
        _state(limits, estimates, (failed, valid)),
        key_status={
            key: KeyStatus(
                load_failure_until=_NOW + 30, load_failure_detail="missing-model.bin"
            )
        },
    )
    result = plan(state)
    reject = next(a for a in result.actions if isinstance(a, Reject))
    assert reject.job_id == failed.job_id
    assert reject.error_kind == "worker_load_failed"
    assert "missing-model.bin" in reject.detail
    assert next(a for a in result.actions if isinstance(a, Spawn)).key.model == "base"


def _check_budget_invariants(state: SchedulerState) -> None:
    """Check resource, lifecycle and assignment contracts over a real plan."""
    result = plan(state)
    assert result == plan(state)
    assert result == plan(
        replace(state, pending=state.pending[::-1], workers=state.workers[::-1])
    )
    assigns = [a for a in result.actions if isinstance(a, Assign)]
    spawns = [a for a in result.actions if isinstance(a, Spawn)]
    stops = [a for a in result.actions if isinstance(a, Stop)]
    assert len(spawns) <= 1
    assert len({a.job_id for a in assigns}) == len(assigns)
    assert len({a.worker_id for a in stops}) == len(stops)
    existing = {w.worker_id: w for w in state.workers}
    stopped = {a.worker_id for a in stops}
    per_worker = Counter(a.worker_id for a in assigns)
    for worker_id, count in per_worker.items():
        worker = existing[worker_id]
        assert worker.state == "ready"
        assert len(worker.running) + count <= worker.capacity
        assert worker_id not in stopped
    assert all(not existing[a.worker_id].running for a in stops)
    running_ids = {j.job_id for w in state.workers for j in w.running}
    assert not running_ids.intersection(a.job_id for a in assigns)
    cpu_jobs = sum(
        len(w.running) for w in state.workers if w.key.backend == "cpu"
    ) + sum(a.backend == "cpu" for a in assigns)
    assert cpu_jobs <= state.limits.cpu_slots
    assert (
        cpu_jobs * state.limits.threads_per_slot
        + sum(w.reservation.host_threads for w in state.workers)
        + sum(a.reservation.host_threads for a in spawns)
        <= state.limits.thread_budget
    )
    assert (
        sum(w.key.backend == "gpu" for w in state.workers)
        + sum(a.key.backend == "gpu" for a in spawns)
        <= state.limits.gpu_workers
    )
    assert (
        sum(w.reservation.commit_bytes for w in state.workers)
        + sum(a.reservation.commit_bytes for a in spawns)
        <= state.limits.commit_budget_bytes
    )
    assert (
        sum(w.reservation.vram_bytes for w in state.workers)
        + sum(a.reservation.vram_bytes for a in spawns)
        <= state.limits.vram_budget_bytes
    )
    if any(w.state in {"spawning", "loading"} for w in state.workers):
        assert not spawns
    for action in assigns:
        job = next(j for j in state.pending if j.job_id == action.job_id)
        assert action.key == make_key(
            job.provider, job.whisper_model, job.compute_request, action.backend
        )
        if job.device_request != "auto":
            assert action.backend == ("cpu" if job.device_request == "cpu" else "gpu")


def test_randomized_plans_obey_global_reservation_and_lifecycle_invariants(
    limits: Limits, estimates: EstimateStore
) -> None:
    """Catch mixed-state overcommit, double dispatch, busy eviction and nondeterminism."""
    rng = random.Random(20261014)  # noqa: S311 - reproducible decision-table generation
    for case in range(120):
        workers: list[WorkerView] = []
        for i in range(rng.randrange(4)):
            backend: Backend = "gpu" if i == 0 and rng.choice((True, False)) else "cpu"
            state: WorkerState = rng.choice(("ready", "retiring", "stopping"))
            capacity = rng.randint(1, 3)
            running = tuple(
                _running(f"active-{case}-{i}-{slot}", end=_NOW + rng.randrange(1, 100))
                for slot in range(rng.randrange(capacity + 1))
            )
            workers.append(
                _worker(
                    f"w{i}",
                    estimates,
                    backend=backend,
                    model=rng.choice(("base", "small")),
                    capacity=capacity,
                    running=running,
                    state=state,
                )
            )
        if workers and rng.randrange(4) == 0:
            workers[-1] = replace(workers[-1], state="loading", running=())
        jobs = tuple(
            _job(
                i,
                call=f"call-{i % 4}",
                device=rng.choice(("auto", "cpu", "cuda")),
                model=rng.choice(("base", "small", "medium")),
                duration=rng.choice((None, 1.0, 60.0, 3600.0)),
                cap=2,
                priority=rng.choice(("high", "normal", "low")),
                since=_NOW - rng.randrange(1000),
            )
            for i in range(1, 17)
        )
        if (
            sum(len(w.running) for w in workers if w.key.backend == "cpu")
            > limits.cpu_slots
        ):
            continue
        snapshot = _state(limits, estimates, (*jobs, jobs[0]), tuple(workers))
        _check_budget_invariants(snapshot)
