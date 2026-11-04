"""Supervisor scenarios through real queue files, worker processes and Job Objects."""

from __future__ import annotations

import ctypes
import logging
import os
import queue
import re
import sys
import threading
import time
from dataclasses import replace
from logging.handlers import QueueHandler
from pathlib import Path
from typing import TYPE_CHECKING

import psutil  # type: ignore[import-untyped]
import pytest

from core.stt_options import DiarizeOptions, MediaWindow, WhisperDecodeOptions
from core.stt_pool import fsio, winjob
from core.stt_pool.config import load_pool_config
from core.stt_pool.estimates import EstimateStore, seed_estimate
from core.stt_pool.keys import make_key
from core.stt_pool.queue import (
    JobOptions,
    ResultRecord,
    read_attempts,
    read_result,
    request_cancel,
    submit_job,
    write_attempts,
    write_result,
)
from core.stt_pool.supervisor import run_supervisor
from core.stt_pool.sysprobe import hardware_fingerprint
from tests.stt_pool_supervisor_support import GIB, SupervisorHarness, configuration
from tests.stt_pool_windows_support import JobCapKernel, WinEvent

if TYPE_CHECKING:
    from collections.abc import Iterator

    from core.stt_pool.keys import WorkerKey
    from core.stt_pool.procs import WorkerLaunch

pytestmark = [pytest.mark.integration, pytest.mark.xdist_group(name="pool_supervisor")]


@pytest.fixture
def memory_logs() -> Iterator[queue.Queue[logging.LogRecord]]:
    """Observe publication without scheduling sleeps or assertions on log prose."""
    records: queue.Queue[logging.LogRecord] = queue.Queue()
    handler = QueueHandler(records)
    logger = logging.getLogger("core.stt_pool.supervisor")
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        handler.close()


def _memory_log_fields(
    records: queue.Queue[logging.LogRecord], job_id: str
) -> dict[str, str]:
    """Await memory facts at the external logging boundary with a hang deadline."""
    deadline = time.monotonic() + 30
    while True:
        record = records.get(timeout=max(0.0, deadline - time.monotonic()))
        fields = dict(re.findall(r"\b([a-z_]+)=([^\s]+)", record.getMessage()))
        if fields.get("job_id") == job_id and "job_peak_bytes" in fields:
            return fields


@pytest.mark.parametrize("source", ["cpu", "cuda", "malformed", "not_an_object"])
@pytest.mark.parametrize("outcome", ["ok", "memory_limit"])
def test_worker_memory_facts_survive_publication_and_resource_errors(
    pool: SupervisorHarness,
    memory_logs: queue.Queue[logging.LogRecord],
    source: str,
    outcome: str,
) -> None:
    """Catch telemetry breaking results, resource diagnostics or CUDA isolation."""
    pool.config = configuration(scheduling={"max_attempts": 1})
    marker = pool.engine_root / "unexpected-cuda-access"
    pool.plan(
        memory_probe="initialized" if source == "cuda" else "uninitialized",
        cuda_marker=str(marker),
    )
    options: dict[str, object] = {}
    if source == "malformed":
        options["memory"] = {
            "private_bytes": -1,
            "cuda_reserved_bytes": True,
            "cuda_allocated_bytes": 2048,
            "cuda_max_reserved_bytes": "bad",
            "cufft_plan_cache_size": [],
            "unknown": 99,
        }
    elif source == "not_an_object":
        options["wire_memory"] = []
    job = pool.job(mode=outcome, **options)
    pool.start()
    fields = _memory_log_fields(memory_logs, job.job_id)
    if outcome == "memory_limit":
        assert fields["error_kind"] == "memory_limit"
        assert "status" not in fields
        publication = _memory_log_fields(memory_logs, job.job_id)
        assert all(
            publication[key] == value
            for key, value in fields.items()
            if key != "job_peak_bytes"
        )
        fields = publication
    assert "status" in fields
    result = read_result(pool.paths, job.job_id)
    assert result is not None
    assert result.attempts == 1
    assert result.status == ("ok" if outcome == "ok" else "error")
    if outcome == "ok":
        assert result.text == Path(job.input_path).stem
    else:
        assert result.error_kind == "memory_limit"
    if source in {"malformed", "not_an_object"}:
        assert fields["private_bytes"] == "None"
        assert fields["cuda_allocated_bytes"] == (
            "2048" if source == "malformed" else "None"
        )
    else:
        if sys.platform == "win32":
            assert int(fields["private_bytes"]) > 0
        else:
            assert fields["private_bytes"] == "None"
        assert fields["cuda_allocated_bytes"] == (
            "1024" if source == "cuda" else "None"
        )
    assert fields["cuda_reserved_bytes"] == ("8192" if source == "cuda" else "None")
    assert fields["cuda_max_reserved_bytes"] == (
        "16384" if source == "cuda" else "None"
    )
    assert fields["cufft_plan_cache_size"] == ("7" if source == "cuda" else "None")
    if sys.platform == "win32":
        assert int(fields["job_peak_bytes"]) > 0
    assert not marker.exists()
    pool.finish()


@pytest.mark.parametrize(
    ("device", "limit", "inherited", "expected"),
    [
        ("cuda", None, None, "256"),
        ("cuda", 19, "77", "19"),
        ("cuda", 0, None, None),
        ("cuda", 0, "77", "77"),
        ("cpu", 19, None, None),
        ("cpu", 19, "77", "77"),
    ],
)
@pytest.mark.parametrize("vram_cap", [True, False])
def test_gpu_cache_policy_reaches_the_real_worker_environment(
    pool: SupervisorHarness,
    memory_logs: queue.Queue[logging.LogRecord],
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    limit: int | None,
    inherited: str | None,
    expected: str | None,
    *,
    vram_cap: bool,
) -> None:
    """Catch unbounded cuDNN caches or zero overwriting inherited policy."""
    name = "TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT"
    if inherited is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, inherited)
    gpu: dict[str, object] = {
        "max_workers": 1,
        "cufft_plan_cache_max_size": 7,
        "idle_cache_release_s": 0.125,
        "vram_cap": vram_cap,
        "vram_cap_margin_gib": 0.375,
    }
    if limit is not None:
        gpu["cudnn_v8_lru_cache_limit"] = limit
    pool.config = configuration(gpu=gpu)
    pool.hardware.gpu_available = True
    job = pool.job(provider="gigaam-e2e-ctc", device=device)
    pool.start()
    _memory_log_fields(memory_logs, job.job_id)
    result = read_result(pool.paths, job.job_id)
    assert result is not None
    assert result.status == "ok"
    constructed = pool.events("constructed")
    assert len(constructed) == 1
    assert constructed[0]["cudnn_v8_lru_cache_limit"] == expected
    options = constructed[0]["options"]
    assert options["cufft_plan_cache_max_size"] == (7 if device == "cuda" else None)
    assert options["idle_cache_release_s"] == (0.125 if device == "cuda" else None)
    expected_cap = None
    if device == "cuda" and vram_cap:
        seed = seed_estimate(make_key("gigaam-e2e-ctc", "small", "auto", "gpu"))
        expected_cap = (
            seed.vram_base_bytes
            + constructed[0]["capacity"] * seed.vram_per_lane_bytes
            + 3 * GIB // 8
        )
    assert options["vram_cap_bytes"] == expected_cap
    pool.finish()


@pytest.mark.parametrize(
    ("fraction", "offset", "retires"),
    [(1.0, 0, True), (0.5, 0, True), (1.0, -1, False), (0.0, 1, False)],
)
def test_commit_recycle_drains_busy_worker_and_routes_backlog_to_fresh_process(
    pool: SupervisorHarness, fraction: float, offset: int, *, retires: bool
) -> None:
    """Catch threshold errors, lost running jobs or reuse under continuous load."""
    pool.config = configuration(
        cpu={"max_slots": 2, "max_slots_per_worker": 2},
        memory={"recycle_commit_fraction": fraction},
    )
    warm = pool.job(memory={"private_bytes": 1})
    pool.start()
    original = pool.result(warm)
    heartbeat = pool.heartbeat(lambda value: value["queue"]["running"] == 0)
    reserved = heartbeat["workers"][0]["commit_bytes"]
    private = int(fraction * reserved) + offset if fraction else 2 * reserved
    busy = pool.job(mode="hold", memory={"private_bytes": 1})
    trigger = pool.job(wait=True, memory={"private_bytes": private})
    assert pool.submitted(busy)["worker_id"] == original.worker_id
    assert pool.submitted(trigger)["worker_id"] == original.worker_id
    pending = pool.job(call=2, mode="hold", memory={"private_bytes": 1})
    pool.heartbeat(lambda value: value["queue"] == {"pending": 1, "running": 2})
    pool.release(trigger)
    assert pool.result(trigger).status == "ok"
    if retires:
        pool.heartbeat(
            lambda value: any(
                worker["worker_id"] == original.worker_id
                and worker["state"] == "retiring"
                and worker["running"] == 1
                for worker in value["workers"]
            )
        )
        assert not any(
            event["worker_id"] == original.worker_id
            for event in pool.events("submitted", pending)
        )
        assert not any(
            event["worker_id"] == original.worker_id for event in pool.events("closed")
        )
    else:
        assert pool.submitted(pending)["worker_id"] == original.worker_id
    pool.release(busy)
    completed = pool.result(busy)
    assert completed.status == "ok"
    assert completed.attempts == 1
    following = pool.submitted(pending)
    assert (following["worker_id"] != original.worker_id) is retires
    pool.release(pending)
    assert pool.result(pending).status == "ok"
    if retires:
        pool.heartbeat(
            lambda value: all(
                worker["worker_id"] != original.worker_id for worker in value["workers"]
            )
        )
        assert (
            sum(
                event["worker_id"] == original.worker_id
                for event in pool.events("closed")
            )
            == 1
        )
    assert pool.result(trigger).attempts == 1
    pool.finish()


@pytest.mark.parametrize("busy", [True, False])
@pytest.mark.parametrize("hours", [0, 1])
def test_uptime_cap_replaces_once_without_interrupting_running_work(
    pool: SupervisorHarness, hours: int, *, busy: bool
) -> None:
    """Catch missed busy retirement, a wrong uptime origin or disabled-cap churn."""
    pool.config = configuration(
        lifecycle={"worker_max_uptime_h": hours, "worker_idle_ttl_s": 10000}
    )
    warm = pool.job(memory={"private_bytes": 1})
    pool.start()
    original = pool.result(warm)
    baseline = pool.heartbeat(lambda value: value["queue"]["running"] == 0)
    spawned_at = baseline["workers"][0]["spawned_at"]
    running = pool.job(mode="hold") if busy else None
    if running is not None:
        assert pool.submitted(running)["worker_id"] == original.worker_id
    pool.clock.advance(spawned_at + 3599 - pool.clock.peek())
    # * Heartbeats after the jump form a barrier without real-time policy waits.
    before = pool.heartbeat(lambda value: value["sequence"] >= baseline["sequence"] + 2)
    assert before["workers"][0]["state"] == "ready"
    pool.clock.advance(2)
    if hours and busy:
        pool.heartbeat(
            lambda value: any(
                worker["worker_id"] == original.worker_id
                and worker["state"] == "retiring"
                and worker["running"] == 1
                for worker in value["workers"]
            )
        )
        assert not any(
            event["worker_id"] == original.worker_id for event in pool.events("closed")
        )
    elif hours:
        pool.heartbeat(
            lambda value: all(
                worker["worker_id"] != original.worker_id for worker in value["workers"]
            )
        )
    else:
        pool.heartbeat(lambda value: value["sequence"] >= before["sequence"] + 2)
    following = pool.job(call=2)
    if running is not None:
        pool.release(running)
        result = pool.result(running)
        assert result.status == "ok"
        assert result.attempts == 1
    result = pool.result(following)
    assert result.status == "ok"
    assert (result.worker_id != original.worker_id) is bool(hours)
    if hours:
        pool.heartbeat(
            lambda value: all(
                worker["worker_id"] != original.worker_id for worker in value["workers"]
            )
        )
        assert (
            sum(
                event["worker_id"] == original.worker_id
                for event in pool.events("closed")
            )
            == 1
        )
        assert len(pool.events("constructed")) == 2
    else:
        assert not pool.events("closed")
        assert len(pool.events("constructed")) == 1
    pool.finish()


@pytest.mark.parametrize("kind", ["stt", "diarize"])
@pytest.mark.parametrize("empty", [False, True])
def test_structured_jobs_transport_options_and_payloads_through_real_workers(
    pool: SupervisorHarness, kind: str, *, empty: bool
) -> None:
    """Catch option/data drops across admission, process IPC or result files, including empty successes."""
    pool.config = configuration(
        cpu={"threads_per_slot": 2}, diarization={"threads_per_slot": 3}
    )
    window = MediaWindow(10.25, 2.5)
    payload: dict[str, object]
    if kind == "stt":
        options = JobOptions(
            want_segments=True,
            decode=WhisperDecodeOptions(
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
                word_timestamps=False,
            ),
            window=window,
        )
        payload = {
            "segments": [] if empty else [{"start": 10.25, "end": 11.0, "text": "речь"}]
        }
    else:
        options = JobOptions(window=window, diarize=DiarizeOptions(2, 0.7, 0.1, 0))
        payload = {
            "turns": []
            if empty
            else [{"start": 10.25, "end": 11.0, "speaker": "speaker_00"}]
        }
    job = pool.job(
        kind=kind,
        provider="whisper" if kind == "stt" else "sherpa-diarize",
        job_options=options,
        data=payload,
        mode="empty" if empty else "ok",
    )
    pool.start()
    receipt = pool.submitted(job)
    result = pool.result(job)
    assert receipt["kind"] == kind
    assert receipt["window"] == {"start_s": 10.25, "duration_s": 2.5}
    assert result.status == ("empty" if empty and kind == "stt" else "ok")
    assert result.data == payload
    assert result.attempts == 1
    constructed = pool.events("constructed")[0]
    threads = 2 if kind == "stt" else 3
    assert constructed["threads_per_slot"] == threads
    assert constructed["thread_env"] == {
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
    }
    if kind == "stt":
        assert receipt["want_segments"]
        assert receipt["decode"] == {
            "beam_size": 1,
            "vad_filter": False,
            "condition_on_previous_text": False,
            "word_timestamps": False,
        }
    else:
        assert result.text == ""
        assert receipt["language"] is None
        assert receipt["diarize"] == {
            "num_speakers": 2,
            "threshold": 0.7,
            "min_duration_on": 0.1,
            "min_duration_off": 0.0,
        }
        assert constructed["options"]["threads_per_slot"] == 3
        heartbeat = pool.heartbeat(lambda value: value["queue"]["running"] == 0)
        assert heartbeat["limits"]["diarization"]["threads_per_slot"] == 3
    pool.finish()


@pytest.mark.parametrize("kind", ["stt", "diarize"])
@pytest.mark.parametrize("invalid", ["absent", "wrong_key", "null", "object"])
def test_success_without_the_requested_payload_becomes_protocol_error(
    pool: SupervisorHarness, kind: str, invalid: str
) -> None:
    """Catch successful-looking transcripts or diarizations with silently missing/malformed payload lists."""
    key = "segments" if kind == "stt" else "turns"
    payload = {
        "absent": None,
        "wrong_key": {"other": []},
        "null": {key: None},
        "object": {key: {}},
    }[invalid]
    job = pool.job(
        kind=kind,
        provider="whisper" if kind == "stt" else "sherpa-diarize",
        job_options=JobOptions(want_segments=kind == "stt"),
        data=payload,
    )
    pool.start()
    result = pool.result(job)
    assert result.status == "error"
    assert result.error_kind == "protocol_error"
    assert result.data is None
    assert result.text == ""
    assert result.attempts == 1
    pool.finish()


def test_oversized_structured_result_publishes_a_readable_error_and_keeps_serving(
    pool: SupervisorHarness,
) -> None:
    """Catch 16 MiB result files becoming unreadable and leaving clients waiting until their deadline."""
    job = pool.job(job_options=JobOptions(want_segments=True), large_segments=True)
    pool.start()
    result = pool.result(job)
    assert result.status == "error"
    assert result.error_kind == "result_too_large"
    assert result.data is None
    assert pool.paths.result_file(job.job_id).stat().st_size < 16 * 1024**2
    following = pool.job(call=2, text="next transcript")
    assert pool.result(following).status == "ok"
    assert pool.result(following).text == "next transcript"
    pool.finish()


def test_mixed_stt_diarization_admission_reserves_each_keys_threads(
    pool: SupervisorHarness,
) -> None:
    """Catch diarization borrowing STT's smaller thread charge and exceeding the shared budget."""
    pool.config = configuration(
        cpu={"max_slots": 3, "threads_per_slot": 2, "max_slots_per_worker": 3},
        threads={"budget": 6},
        memory={"commit_budget_gib": 8},
        diarization={"threads_per_slot": 4},
    )
    stt = pool.job(mode="hold")
    first = pool.job(
        kind="diarize",
        provider="sherpa-diarize",
        call=2,
        mode="hold",
        data={"turns": []},
    )
    pool.start()
    pool.submitted(stt)
    pool.submitted(first)
    waiting = pool.job(
        kind="diarize",
        provider="sherpa-diarize",
        call=3,
        mode="hold",
        data={"turns": []},
    )
    pool.heartbeat(lambda value: value["queue"] == {"pending": 1, "running": 2})
    assert not pool.events("submitted", waiting)
    pool.release(stt)
    assert pool.result(stt).status == "ok"
    pool.heartbeat(lambda value: value["queue"] == {"pending": 1, "running": 1})
    assert not pool.events("submitted", waiting)
    pool.release(first)
    assert pool.result(first).status == "ok"
    pool.submitted(waiting)
    pool.release(waiting)
    assert pool.result(waiting).status == "ok"
    pool.finish()


def test_diarization_scales_as_single_slot_processes_with_its_own_memory_seed(
    pool: SupervisorHarness,
) -> None:
    """Catch sherpa jobs sharing a multi-slot worker or falling back to Whisper-large memory estimates."""
    pool.config = configuration(
        cpu={"max_slots": 2, "max_slots_per_worker": 8},
        threads={"budget": 8},
        memory={"commit_budget_gib": 6},
    )
    jobs = [
        pool.job(
            kind="diarize",
            provider="sherpa-diarize",
            call=call,
            mode="hold",
            data={"turns": []},
        )
        for call in (1, 2)
    ]
    pool.start()
    receipts = [pool.submitted(job) for job in jobs]
    assert len({receipt["worker_id"] for receipt in receipts}) == 2
    constructed = pool.events("constructed")
    assert len(constructed) == 2
    assert all(worker["capacity"] == 1 for worker in constructed)
    assert all(worker["threads_per_slot"] == 4 for worker in constructed)
    for job in jobs:
        pool.release(job)
        assert pool.result(job).status == "ok"
    pool.finish()


@pytest.fixture
def pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[SupervisorHarness]:
    """Check lock, heartbeat and fake process teardown even when a scenario fails."""
    monkeypatch.delenv("ASKVLM_EXTERNAL_SUPERVISOR_MANAGED", raising=False)
    harness = SupervisorHarness(tmp_path)
    try:
        yield harness
    finally:
        try:
            if harness.thread is not None:
                harness.finish(harness.code if harness.code is not None else 0)
        finally:
            harness.cleanup()


@pytest.mark.parametrize("managed", [False, True])
def test_architect_example_configuration_workers_ttl_and_idle_exit(
    pool: SupervisorHarness,
    monkeypatch: pytest.MonkeyPatch,
    *,
    managed: bool,
) -> None:
    """Catch configuration monopoly, wrong scale-out or TTL shutdown."""
    if managed:
        monkeypatch.setenv("ASKVLM_EXTERNAL_SUPERVISOR_MANAGED", "1")
    pool.config = configuration(cpu={"max_slots": 3, "max_slots_per_worker": 1})
    first = pool.job(model="tiny", text="model A")
    pool.start()
    first_result = pool.result(first)
    assert first_result.status == "ok"
    second = pool.job(model="base", call=2, max_parallel=2, mode="hold")
    third = pool.job(model="base", call=2, max_parallel=2, mode="hold")
    second_start, third_start = pool.submitted(second), pool.submitted(third)
    assert second_start["worker_id"] != third_start["worker_id"]
    assert first_result.worker_id not in {
        second_start["worker_id"],
        third_start["worker_id"],
    }
    pool.clock.advance(31)
    heartbeat = pool.heartbeat(lambda value: len(value["workers"]) == 2)
    assert {worker["key"]["model"] for worker in heartbeat["workers"]} == {"base"}
    for job in (second, third):
        pool.release(job)
        assert pool.result(job).status == "ok"
    fourth = pool.job(model="base", call=3, max_parallel=1, mode="hold")
    fourth_start = pool.submitted(fourth)
    assert fourth_start["worker_id"] in {
        second_start["worker_id"],
        third_start["worker_id"],
    }
    pool.clock.advance(31)
    heartbeat = pool.heartbeat(lambda value: len(value["workers"]) == 1)
    assert heartbeat["workers"][0]["worker_id"] == fourth_start["worker_id"]
    pool.release(fourth)
    assert pool.result(fourth).status == "ok"
    pool.clock.advance(31)
    idle = pool.heartbeat(lambda value: not value["workers"])
    pool.clock.advance(6)
    if managed:
        current = pool.heartbeat(
            lambda value: value["sequence"] >= idle["sequence"] + 2
        )
        assert current["instance_id"] == idle["instance_id"]
        late = pool.job(call=4, text="after managed idle")
        assert pool.result(late).text == "after managed idle"
        pool.finish()
    else:
        pool.finish(request_stop=False)


def test_linux_cpu_example_serves_stt_and_diarization_with_one_two_thread_slot(
    pool: SupervisorHarness,
) -> None:
    """Catch an invalid CPU deployment example, GPU use or unschedulable diarization."""
    pool.config = load_pool_config(
        Path(__file__).resolve().parents[1] / "stt_pool.linux-server.example.json",
        env={},
        logical_cpus=4,
    )
    assert pool.config.cpu.max_slots == 1
    assert pool.config.cpu.max_slots_per_worker == 1
    assert pool.config.cpu.threads_per_slot == 2
    assert pool.config.diarization.threads_per_slot == 2
    assert pool.config.threads.budget == 2
    assert pool.config.gpu.max_workers == 0
    assert pool.config.memory.commit_budget_bytes == 6 * GIB
    assert pool.config.memory.free_commit_guard_bytes == 6 * GIB
    assert pool.config.memory.critical_free_commit_bytes == 3 * GIB
    pool.hardware.gpu_available = True
    stt = pool.job(device="auto", mode="hold")
    diarize = pool.job(
        kind="diarize",
        provider="sherpa-diarize",
        call=2,
        mode="hold",
        data={"turns": []},
    )
    pool.start()
    pool.submitted(stt)
    pool.heartbeat(lambda value: value["queue"] == {"pending": 1, "running": 1})
    assert not pool.events("submitted", diarize)
    pool.release(stt)
    assert pool.result(stt).status == "ok"
    pool.submitted(diarize)
    constructed = pool.events("constructed")
    assert {worker["key"]["backend"] for worker in constructed} == {"cpu"}
    assert all(worker["capacity"] == 1 for worker in constructed)
    assert all(worker["threads_per_slot"] == 2 for worker in constructed)
    pool.release(diarize)
    assert pool.result(diarize).status == "ok"
    pool.finish()


def test_cpu_gpu_limits_apply_independently(pool: SupervisorHarness) -> None:
    """Catch CPU and GPU quotas blocking or borrowing from each other."""
    pool.config = configuration(gpu={"max_workers": 1})
    pool.hardware.gpu_available = True
    cpu = pool.job(device="cpu", call=1, mode="hold")
    gpu = pool.job(device="cuda", call=2, mode="hold")
    pool.start()
    cpu_start, gpu_start = pool.submitted(cpu), pool.submitted(gpu)
    assert cpu_start["backend"] == "cpu"
    assert gpu_start["backend"] == "gpu"
    cpu_pending = pool.job(device="cpu", call=3, mode="hold")
    gpu_pending = pool.job(device="cuda", call=4, mode="hold")
    snapshot = pool.heartbeat(
        lambda value: value["queue"] == {"pending": 2, "running": 2}
    )
    assert sorted(worker["key"]["backend"] for worker in snapshot["workers"]) == [
        "cpu",
        "gpu",
    ]
    assert not pool.events("submitted", cpu_pending)
    assert not pool.events("submitted", gpu_pending)
    for active, pending in ((cpu, cpu_pending), (gpu, gpu_pending)):
        pool.release(active)
        assert pool.result(active).status == "ok"
        start = pool.submitted(pending)
        assert start["worker_id"] == pool.submitted(active)["worker_id"]
        pool.release(pending)
        assert pool.result(pending).backend == start["backend"]
    pool.finish()


def test_commit_budget_counts_all_worker_processes(pool: SupervisorHarness) -> None:
    """Catch separate backend budgets exceeding total commit."""
    pool.config = configuration(cpu={"max_slots": 2}, memory={"commit_budget_gib": 3})
    first = pool.job(mode="hold")
    pool.start()
    started = pool.submitted(first)
    second = pool.job(call=2, mode="hold")
    heartbeat = pool.heartbeat(
        lambda value: value["queue"] == {"pending": 1, "running": 1}
    )
    assert len(heartbeat["workers"]) == 1
    assert not pool.events("submitted", second)
    pool.release(first)
    assert pool.result(first).status == "ok"
    assert pool.submitted(second)["worker_id"] == started["worker_id"]
    pool.release(second)
    assert pool.result(second).status == "ok"
    pool.finish()


def test_free_commit_guard_blocks_spawn_then_recovers(pool: SupervisorHarness) -> None:
    """Catch admission violating the system free-commit guard."""
    pool.hardware.free_commit = 2 * GIB
    job = pool.job()
    pool.start()
    pool.heartbeat(lambda value: value["queue"]["pending"] == 1)
    assert not pool.events("constructed")
    assert read_result(pool.paths, job.job_id) is None
    pool.hardware.free_commit = 64 * GIB
    assert pool.result(job).status == "ok"
    pool.finish()


def test_invalid_provider_request_is_rejected_before_spawning(
    pool: SupervisorHarness,
) -> None:
    """Catch malformed jobs reaching a model or losing error codes."""
    job = pool.job()
    payload = job.to_dict()
    payload["request"]["provider"] = "unsupported-provider"
    fsio.atomic_write_json(pool.paths.incoming_file(job.job_id), payload)
    pool.start()
    result = pool.result(job)
    assert result.status == "error"
    assert result.error_kind == "input_invalid"
    assert not pool.events("constructed")
    pool.finish()


@pytest.fixture
def tiny_job_caps(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Enforce small real job caps; record the supervisor's requested limits."""
    if sys.platform != "win32":
        pytest.skip("requires real Windows Job Object caps and notifications")
    kernel = winjob._kernel32()  # noqa: SLF001 - substitute only the external Win32 API boundary
    cap_bytes = 96 * 1024**2
    limits: list[int] = []

    class CappedKernel:
        """Delegate Win32 calls while making only this test's worker cap small."""

        def __getattr__(self, name: str) -> object:
            return getattr(kernel, name)

        @staticmethod
        def SetInformationJobObject(  # noqa: N802 - external Win32 symbol
            handle: object, info_class: int, pointer: object, size: int
        ) -> int:
            """Install a real 96 MiB cap without attempting a multi-GiB allocation."""
            if info_class == 9:
                info = ctypes.cast(
                    pointer,
                    ctypes.POINTER(winjob._JobExtendedLimitInformation),  # noqa: SLF001 - real external ABI structure
                ).contents
                limits.append(int(info.JobMemoryLimit))
                info.JobMemoryLimit = cap_bytes
            return kernel.SetInformationJobObject(handle, info_class, pointer, size)

    monkeypatch.setattr(winjob, "_kernel32", CappedKernel)
    return limits


@pytest.fixture
def live_job_caps(monkeypatch: pytest.MonkeyPatch) -> JobCapKernel:
    """Keep real Job Object growth while scaling allocations to a small test cap."""
    if sys.platform != "win32":
        pytest.skip("requires real Windows Job Object caps and notifications")
    kernel = JobCapKernel(initial_cap=96 * 1024**2)
    monkeypatch.setattr(winjob, "_kernel32", lambda: kernel)
    return kernel


def test_child_cap_raises_once_retries_on_live_worker_and_retires_on_second_hit(
    pool: SupervisorHarness,
    live_job_caps: JobCapKernel,
    memory_logs: queue.Queue[logging.LogRecord],
) -> None:
    """Catch unbounded raises, stale accounting, missed hits or lost retries."""
    pool.config = configuration(cpu={"max_slots": 2, "max_slots_per_worker": 2})
    with WinEvent() as capped, WinEvent() as release:
        job = pool.job(
            mode="child_cap_report",
            cap_signal=capped.name,
            result_gate=release.name,
            fail_attempts=1,
        )
        pool.start()
        started = pool.submitted(job)
        capped.wait()
        heartbeat = pool.heartbeat(
            lambda value: any(worker["cap_raised"] for worker in value["workers"])
        )
        worker = next(
            worker
            for worker in heartbeat["workers"]
            if worker["worker_id"] == started["worker_id"]
        )
        assert worker["state"] == "ready"
        changes = list(live_job_caps.updates)
        assert len(changes) == 2
        assert all(change["success"] for change in changes)
        headroom = pool.config.memory.child_headroom_bytes
        assert changes[1]["requested"] - headroom == pytest.approx(
            (changes[0]["requested"] - headroom) * 1.25, abs=2
        )
        assert worker["commit_bytes"] == changes[1]["requested"] - headroom
        assert worker["job_limit_bytes"] == changes[1]["requested"]
        current = live_job_caps.query(changes[0]["handle"])
        assert current.JobMemoryLimit == changes[1]["applied"] > changes[0]["applied"]
        assert current.BasicLimitInformation.LimitFlags == changes[0]["flags"]
        assert current.BasicLimitInformation.LimitFlags & 0x2000
        canary = pool.job(call=2)
        assert pool.result(canary).worker_id == started["worker_id"]
        release.set()
        retry = pool.result(job)
        fields = _memory_log_fields(memory_logs, job.job_id)
        assert fields["error_kind"] == "memory_limit"
        assert retry.status == "ok"
        assert retry.attempts == 2
        assert retry.worker_id == started["worker_id"]
        assert len(pool.events("submitted", job)) == 2
        assert not pool.events("closed")
    with WinEvent() as second_hit, WinEvent() as release:
        second = pool.job(
            call=3,
            mode="child_cap_report",
            cap_signal=second_hit.name,
            result_gate=release.name,
            fail_attempts=1,
        )
        assert pool.submitted(second)["worker_id"] == started["worker_id"]
        second_hit.wait()
        pool.heartbeat(
            lambda value: any(
                worker["worker_id"] == started["worker_id"]
                and worker["state"] == "retiring"
                for worker in value["workers"]
            )
        )
        assert (
            len(
                [
                    change
                    for change in live_job_caps.updates
                    if change["handle"] == changes[0]["handle"]
                ]
            )
            == 2
        )
        release.set()
        result = pool.result(second)
        assert result.status == "ok"
        assert result.attempts == 2
        assert result.worker_id != started["worker_id"]
        pool.heartbeat(
            lambda value: all(
                worker["worker_id"] != started["worker_id"]
                for worker in value["workers"]
            )
        )
        assert (
            sum(
                event["worker_id"] == started["worker_id"]
                for event in pool.events("closed")
            )
            == 1
        )
    assert len(set(live_job_caps.raw_hits)) == 2
    pool.finish()
    learned = fsio.read_json(pool.paths.estimates_file)
    assert learned is not None
    assert (
        learned["keys"][make_key("whisper", "small", "auto", "cpu").label][
            "commit_factor"
        ]
        == 1.25
    )


@pytest.mark.parametrize(
    "blocker",
    [
        "budget",
        "guard",
        "critical",
        "ceiling",
        "disabled",
        "query",
        "post",
        "set",
        "unknown_pid",
    ],
)
def test_blocked_child_cap_raise_retires_and_preserves_the_running_attempt(
    pool: SupervisorHarness, live_job_caps: JobCapKernel, blocker: str
) -> None:
    """Catch unsafe growth, ignored disable switches or failed Win32 updates."""
    key = make_key("whisper", "small", "auto", "cpu")
    seed = seed_estimate(key)
    reserved = seed.commit_base_bytes + seed.commit_per_slot_bytes
    memory: dict[str, object] = {}
    if blocker == "budget":
        memory["commit_budget_gib"] = reserved / GIB
    if blocker == "disabled":
        memory["live_cap_raise"] = False
    pool.config = configuration(memory=memory, scheduling={"max_attempts": 1})
    if blocker == "ceiling":
        store = EstimateStore(
            pool.paths.estimates_file,
            fingerprint=hardware_fingerprint(None),
            clock=pool.clock,
        )
        store.bump_memory(key, factor=4.0)
        assert store.save_if_due(force=True)
    with WinEvent() as before_cap, WinEvent() as capped, WinEvent() as release:
        job = pool.job(
            mode="child_cap_report",
            before_cap_gate=before_cap.name,
            cap_signal=capped.name,
            result_gate=release.name,
        )
        pool.start()
        started = pool.submitted(job)
        initial = pool.heartbeat(lambda value: value["queue"]["running"] == 1)
        original = initial["workers"][0]
        if blocker == "guard":
            growth = round(0.25 * reserved)
            pool.hardware.free_commit = (
                pool.config.memory.free_commit_guard_bytes + growth - 1024
            )
        elif blocker == "critical":
            pool.hardware.free_commit = (
                pool.config.memory.critical_free_commit_bytes - 1024
            )
        if blocker in {"query", "post", "set"}:
            live_job_caps.fail_update = blocker
        live_job_caps.unknown_pid = blocker == "unknown_pid"
        before_cap.set()
        capped.wait()
        heartbeat = pool.heartbeat(
            lambda value: any(
                worker["worker_id"] == started["worker_id"]
                and worker["state"] == "retiring"
                for worker in value["workers"]
            )
        )
        worker = next(
            worker
            for worker in heartbeat["workers"]
            if worker["worker_id"] == started["worker_id"]
        )
        assert not worker["cap_raised"]
        assert worker["commit_bytes"] == original["commit_bytes"]
        assert worker["job_limit_bytes"] == original["job_limit_bytes"]
        assert (
            len([change for change in live_job_caps.updates if change["success"]]) == 1
        )
        assert not pool.events("closed")
        release.set()
        result = pool.result(job)
        assert result.status == "error"
        assert result.error_kind == "memory_limit"
        assert result.attempts == 1
    pool.finish()


@pytest.mark.parametrize("attempts", [1, 2])
def test_unrelated_crash_after_live_raise_keeps_crash_kind_and_auto_backend_choice(
    pool: SupervisorHarness, live_job_caps: JobCapKernel, attempts: int
) -> None:
    """Catch old cap reports reclassifying crashes or pinning auto retries."""
    # * A denied VirtualAlloc can raise the peak above the old enforced cap.
    live_job_caps.raise_floor_bytes = 384 * 1024**2
    pool.config = configuration(
        gpu={"max_workers": 1}, scheduling={"max_attempts": attempts}
    )
    with WinEvent() as capped, WinEvent() as release:
        warm = pool.job(
            mode="child_cap_report",
            cap_signal=capped.name,
            result_gate=release.name,
            after_cap="ok",
        )
        pool.start()
        started = pool.submitted(warm)
        capped.wait()
        pool.heartbeat(
            lambda value: any(worker["cap_raised"] for worker in value["workers"])
        )
        release.set()
        assert pool.result(warm).status == "ok"
    changes = list(live_job_caps.updates)
    assert len(changes) == 2
    real = live_job_caps.query(changes[0]["handle"])
    assert real.PeakJobMemoryUsed < real.JobMemoryLimit * 0.95
    crash = pool.job(
        call=2, device="auto", duration=600, mode="crash", wait=True, fail_attempts=1
    )
    assert pool.submitted(crash)["worker_id"] == started["worker_id"]
    calls = pool.hardware.gpu_calls
    pool.hardware.gpu_available = True
    pool.until(lambda: pool.hardware.gpu_calls, lambda count: count >= calls + 2)
    pool.release(crash)
    result = pool.result(crash)
    assert result.attempts == attempts
    if attempts == 1:
        assert result.status == "error"
        assert result.error_kind == "worker_crashed"
    else:
        assert result.status == "ok"
        assert [event["backend"] for event in pool.events("submitted", crash)] == [
            "cpu",
            "gpu",
        ]
    pool.finish()
    learned = fsio.read_json(pool.paths.estimates_file)
    assert learned is not None
    assert (
        learned["keys"][make_key("whisper", "small", "auto", "cpu").label][
            "commit_factor"
        ]
        == 1.25
    )


@pytest.mark.parametrize("mode", ["crash", "retry_crash", "handled"])
def test_real_job_memory_limit_notification_classifies_worker_exit(
    pool: SupervisorHarness, tiny_job_caps: list[int], mode: str
) -> None:
    """Catch a capped worker crash being classified as an ordinary crash."""
    retry = mode == "retry_crash"
    pool.config = configuration(
        gpu={"max_workers": 1}, scheduling={"max_attempts": 2 if retry else 1}
    )
    job = pool.job(
        device="auto" if retry else "cpu",
        mode="allocation_error" if mode == "handled" else "allocation_crash",
        wait=True,
        fail_attempts=1,
    )
    pool.start()
    started = pool.submitted(job)
    if retry:
        calls = pool.hardware.gpu_calls
        pool.hardware.gpu_available = True
        pool.until(lambda: pool.hardware.gpu_calls, lambda count: count >= calls + 2)
    pool.release(job)
    result = pool.result(job)
    if retry:
        assert result.status == "ok"
        assert result.backend == "cpu"
        assert result.attempts == 2
        assert [event["backend"] for event in pool.events("submitted", job)] == [
            "cpu",
            "cpu",
        ]
    else:
        assert result.status == "error"
        assert result.error_kind == "memory_limit"
        assert result.attempts == 1
    assert not pool.paths.incoming_file(job.job_id).exists()
    pool.finish()
    learned = fsio.read_json(pool.paths.estimates_file)
    assert learned is not None
    assert next(iter(learned["keys"].values()))["commit_factor"] == 1.25
    assert not any(
        psutil.pid_exists(event["pid"])
        for event in pool.events("constructed")
        if event["worker_id"] == started["worker_id"]
    )


@pytest.mark.parametrize(
    ("mode", "attempt_budget", "status", "kind", "attempts"),
    [
        ("error", 2, "ok", None, 2),
        ("error", 1, "error", "memory_limit", 1),
        ("ok", 2, "ok", None, 1),
        ("empty", 2, "empty", None, 1),
        ("device_unavailable", 2, "error", "device_unavailable", 1),
        ("protocol_error", 2, "error", "protocol_error", 1),
        ("result_too_large", 2, "error", "result_too_large", 1),
        ("cancel", 2, "cancelled", "cancelled", 1),
        ("cancel_error", 2, "error", "transcribe_failed", 1),
    ],
)
@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_cap_report_retires_before_result_and_preserves_attempt_outcomes(
    pool: SupervisorHarness,
    tiny_job_caps: list[int],
    backend: str,
    mode: str,
    attempt_budget: int,
    status: str,
    kind: str | None,
    attempts: int,
) -> None:
    """Catch lost retries, capped-worker reuse, or resurrected cancellations."""
    pool.config = configuration(
        cpu={"max_slots": 2},
        gpu={"max_workers": 2},
        scheduling={"max_attempts": attempt_budget},
    )
    pool.hardware.gpu_available = True
    device = "cpu" if backend == "cpu" else "cuda"
    with WinEvent() as capped, WinEvent() as release, WinEvent() as cancelled:
        job = pool.job(
            device=device,
            mode="cap_report",
            after_cap="error" if mode in {"cancel", "cancel_error"} else mode,
            cap_signal=capped.name,
            result_gate=release.name,
            cancel_signal=cancelled.name,
            ignore_cancel=mode == "cancel_error",
            error_text="decode input=incident.wav exit=3221225773",
            fail_attempts=1,
        )
        pool.start()
        first = pool.submitted(job)
        capped.wait()
        pool.heartbeat(
            lambda value: any(
                worker["worker_id"] == first["worker_id"]
                and worker["state"] == "retiring"
                for worker in value["workers"]
            )
        )
        canary = pool.job(device=device, call=2)
        healthy = pool.result(canary)
        assert healthy.status == "ok"
        assert healthy.worker_id != first["worker_id"]
        if mode in {"cancel", "cancel_error"}:
            request_cancel(pool.paths, job.job_id)
            cancelled.wait()
        release.set()
        result = pool.result(job)
        assert result.status == status
        assert result.error_kind == kind
        assert result.attempts == attempts
        assert result.backend == backend
        starts = pool.events("submitted", job)
        assert len(starts) == attempts
        if attempts == 2:
            assert result.worker_id != first["worker_id"]
            assert starts[1]["backend"] == backend
        if mode == "error" and attempt_budget == 1:
            assert "incident.wav" in (result.error_detail or "")
            assert "3221225773" in (result.error_detail or "")
        assert not pool.paths.incoming_file(job.job_id).exists()
        pool.finish()
    learned = fsio.read_json(pool.paths.estimates_file)
    assert learned is not None
    entry = next(iter(learned["keys"].values()))
    assert entry["commit_factor"] == 1.25
    assert entry["vram_factor"] == 1.0


@pytest.mark.parametrize("at_ceiling", [False, True])
def test_load_time_cap_report_still_serves_without_a_respawn_loop(
    pool: SupervisorHarness, tiny_job_caps: list[int], *, at_ceiling: bool
) -> None:
    """Catch a load-end cap report causing a respawn loop at the ceiling."""
    key = make_key("whisper", "small", "auto", "cpu")
    before = pool.clock.peek()
    if at_ceiling:
        store = EstimateStore(
            pool.paths.estimates_file,
            fingerprint=hardware_fingerprint(None),
            clock=pool.clock,
        )
        store.bump_memory(key, factor=4.0)
        assert store.save_if_due(force=True)
    with WinEvent() as capped, WinEvent() as release:
        pool.plan(
            load_modes={"small": "cap_report"},
            cap_signal=capped.name,
            result_gate=release.name,
        )
        job = pool.job()
        pool.start()
        capped.wait()
        pool.until(
            lambda: fsio.read_json(pool.paths.estimates_file),
            lambda saved: saved is not None
            and saved["keys"].get(key.label, {}).get("updated_at", 0) > before,
        )
        pool.heartbeat(
            lambda value: len(value["workers"]) == 1
            and value["workers"][0]["state"] == "loading"
        )
        release.set()
        result = pool.result(job)
        assert result.status == "ok"
        assert result.attempts == 1
        following = pool.job(call=2)
        second = pool.result(following)
        assert second.status == "ok"
        assert second.worker_id == result.worker_id
        assert len(pool.events("constructed")) == 1
        pool.finish()
    learned = fsio.read_json(pool.paths.estimates_file)
    assert learned is not None
    assert learned["keys"][key.label]["commit_factor"] == (4.0 if at_ceiling else 1.25)


def test_evicts_idle_other_configuration_before_ttl(pool: SupervisorHarness) -> None:
    """Catch waiting for TTL instead of evicting an idle other key."""
    pool.config = configuration(
        cpu={"max_slots": 2},
        memory={"commit_budget_gib": 2.2},
        lifecycle={"worker_idle_ttl_s": 1000},
    )
    first = pool.job(model="small")
    pool.start()
    original = pool.result(first)
    second = pool.job(model="base", call=2)
    result = pool.result(second)
    assert result.status == "ok"
    assert result.worker_key is not None
    assert result.worker_key["model"] == "base"
    assert result.worker_id != original.worker_id
    pool.heartbeat(
        lambda value: len(value["workers"]) == 1
        and value["workers"][0]["key"]["model"] == "base"
    )
    assert any(
        event["worker_id"] == original.worker_id for event in pool.events("closed")
    )
    assert pool.clock.peek() - first.submitted_at < 1000
    pool.finish()


@pytest.mark.parametrize(
    ("backlog_s", "expected_backend"), [(1.0, "gpu"), (10000.0, "cpu")]
)
def test_auto_spills_only_when_cpu_finishes_sooner(
    pool: SupervisorHarness, backlog_s: float, expected_backend: str
) -> None:
    """Catch slow CPU spill or loss of beneficial CPU concurrency."""
    pool.config = configuration(gpu={"max_workers": 1})
    pool.hardware.gpu_available = True
    blocker = pool.job(device="cuda", duration=backlog_s, mode="hold")
    pool.start()
    blocker_start = pool.submitted(blocker)
    next_job = pool.job(device="auto", call=2, duration=1.0)
    if expected_backend == "cpu":
        result = pool.result(next_job)
        assert result.backend == "cpu"
        snapshot = pool.heartbeat(lambda value: value["queue"]["running"] == 1)
        assert any(
            worker["worker_id"] == blocker_start["worker_id"] and worker["running"] == 1
            for worker in snapshot["workers"]
        )
    else:
        pool.heartbeat(lambda value: value["queue"] == {"pending": 1, "running": 1})
        assert not pool.events("submitted", next_job)
    pool.release(blocker)
    assert pool.result(blocker).status == "ok"
    assert pool.result(next_job).backend == expected_backend
    assert not pool.events("cancel", blocker)
    pool.finish()


def test_priority_and_round_robin_with_one_slot_never_preempt_started_work(
    pool: SupervisorHarness,
) -> None:
    """Catch preemption, call monopoly or priority inversion."""
    blocker = pool.job(call=9, priority="low", mode="hold")
    pool.start()
    pool.submitted(blocker)
    batch = [pool.job(call=1, mode="hold") for _ in range(4)]
    single = pool.job(call=2, mode="hold")
    high = pool.job(call=3, priority="high", mode="hold")
    pool.heartbeat(lambda value: value["queue"] == {"pending": 6, "running": 1})
    assert read_attempts(pool.paths, blocker.job_id) == 1
    assert not pool.events("cancel", blocker)
    pool.release(blocker)
    assert pool.result(blocker).status == "ok"
    expected = [high, batch[0], single, *batch[1:]]
    for job in expected:
        pool.submitted(job)
        assert [event["job_id"] for event in pool.events("submitted")][-1] == job.job_id
        pool.release(job)
        assert pool.result(job).status == "ok"
    assert [event["job_id"] for event in pool.events("submitted")] == [
        blocker.job_id,
        *(job.job_id for job in expected),
    ]
    pool.finish()


def test_priority_aging_eventually_serves_low_before_new_high(
    pool: SupervisorHarness,
) -> None:
    """Catch lost wait age defeating starvation protection."""
    pool.config = configuration(
        scheduling={"priority_aging_s": 5, "starvation_after_s": 10000}
    )
    blocker = pool.job(priority="high", call=1, mode="hold")
    pool.start()
    pool.submitted(blocker)
    low = pool.job(priority="low", call=2, mode="hold")
    high = pool.job(priority="high", call=3, mode="hold")
    pool.heartbeat(lambda value: value["queue"]["pending"] == 2)
    pool.clock.advance(11)
    pool.release(blocker)
    pool.result(blocker)
    pool.submitted(low)
    assert not pool.events("submitted", high)
    pool.release(low)
    assert pool.result(low).status == "ok"
    pool.submitted(high)
    pool.release(high)
    assert pool.result(high).status == "ok"
    pool.finish()


def test_parallel_cap_applies_to_its_call_only(pool: SupervisorHarness) -> None:
    """Catch call caps applying globally or being exceeded."""
    pool.config = configuration(cpu={"max_slots": 2, "max_slots_per_worker": 2})
    first = pool.job(call=1, max_parallel=1, mode="hold")
    second = pool.job(call=1, max_parallel=1, mode="hold")
    other = pool.job(call=2, max_parallel=1, mode="hold")
    pool.start()
    pool.submitted(first)
    pool.submitted(other)
    pool.heartbeat(lambda value: value["queue"] == {"pending": 1, "running": 2})
    assert not pool.events("submitted", second)
    pool.release(first)
    assert pool.result(first).status == "ok"
    pool.submitted(second)
    for job in (second, other):
        pool.release(job)
        assert pool.result(job).status == "ok"
    pool.finish()


@pytest.mark.parametrize(("fail_attempts", "status"), [(1, "ok"), (100, "error")])
def test_worker_crashes_retry_without_exceeding_attempt_budget(
    pool: SupervisorHarness, fail_attempts: int, status: str
) -> None:
    """Catch lost crash retries, infinite retries or missing worker log details."""
    job = pool.job(mode="crash", fail_attempts=fail_attempts)
    pool.start()
    result = pool.result(job)
    assert result.status == status
    assert result.attempts == 2
    assert [event["attempt_id"] for event in pool.events("submitted", job)] == [
        f"{job.job_id}:1",
        f"{job.job_id}:2",
    ]
    if status == "error":
        assert result.error_kind == "worker_crashed"
        assert Path(job.input_path).name in (result.error_detail or "")
        assert "7" in (result.error_detail or "")
    pool.finish()


@pytest.mark.parametrize("persisted", [1, 2])
def test_restart_recovery_uses_persisted_attempts_and_rejects_exhausted_jobs(
    pool: SupervisorHarness, persisted: int
) -> None:
    """Catch restarts resetting attempts or rerunning exhausted jobs."""
    job = pool.job()
    write_attempts(pool.paths, job.job_id, persisted)
    pool.start()
    result = pool.result(job)
    assert result.attempts == 2
    if persisted == 1:
        assert result.status == "ok"
        assert pool.submitted(job)["attempt_id"] == f"{job.job_id}:2"
    else:
        assert result.status == "error"
        assert result.error_kind == "retry_exhausted"
        assert not pool.events("constructed")
    pool.finish()


def test_stop_and_restart_restore_victim_attempt_and_keep_incoming(
    pool: SupervisorHarness,
) -> None:
    """Catch supervisor stops consuming victim attempts or losing incoming work."""
    job = pool.job(mode="hold")
    pool.start()
    first = pool.submitted(job)
    pool.finish()
    assert pool.paths.incoming_file(job.job_id).exists()
    assert read_attempts(pool.paths, job.job_id) == 0
    assert read_result(pool.paths, job.job_id) is None
    pool.start()
    second = pool.submitted(job, 2)
    assert second["worker_id"] != first["worker_id"]
    assert second["attempt_id"] == first["attempt_id"]
    pool.release(job)
    result = pool.result(job)
    assert result.status == "ok"
    assert result.attempts == 1
    pool.finish()


def test_load_failure_cooldown_is_per_configuration_and_preserves_error(
    pool: SupervisorHarness,
) -> None:
    """Catch cross-key cooldown, repeated failed loads or lost worker detail."""
    pool.plan(load_modes={"small": "fail"})
    first = pool.job(model="small")
    pool.start()
    result = pool.result(first)
    assert result.status == "unavailable"
    assert result.error_kind == "worker_load_failed"
    assert "Too long wav file" in (result.error_detail or "")
    assert "transcribe_longform" in (result.error_detail or "")
    second = pool.job(model="small", call=2)
    assert pool.result(second).status == "unavailable"
    assert len(pool.events("constructed")) == 1
    other = pool.job(model="base", call=3)
    assert pool.result(other).status == "ok"
    pool.plan()
    pool.clock.advance(41)
    recovered = pool.job(model="small", call=4)
    assert pool.result(recovered).status == "ok"
    pool.finish()


def test_malformed_loaded_fails_closed_before_dispatch(pool: SupervisorHarness) -> None:
    """Catch malformed readiness allowing jobs to enter an unready worker."""
    pool.plan(load_modes={"small": "bad_loaded"})
    job = pool.job()
    pool.start()
    result = pool.result(job)
    assert result.status == "unavailable"
    assert result.error_kind == "worker_load_failed"
    assert not pool.events("submitted", job)
    other = pool.job(model="base", call=2)
    assert pool.result(other).status == "ok"
    pool.finish()


def test_load_timeout_terminates_the_hung_worker_process(
    pool: SupervisorHarness,
) -> None:
    """Catch an unwired load watchdog leaving a hung cold-load process alive."""
    pool.plan(load_modes={"small": "hang"})
    pool.options = replace(pool.options, load_timeout_s=5)
    job = pool.job()
    pool.start()
    loading = pool.until(lambda: pool.events("loading"), bool)[0]
    constructed = next(
        event
        for event in pool.events("constructed")
        if event["worker_id"] == loading["worker_id"]
    )
    process = psutil.Process(constructed["pid"])
    pool.clock.advance(6)
    pool.until(process.is_running, lambda running: not running)
    result = pool.result(job)
    assert result.status == "unavailable"
    assert result.error_kind == "worker_load_failed"
    assert "5" in (result.error_detail or "")
    assert not pool.events("submitted", job)
    assert len(pool.events("constructed")) == 1
    later = pool.job(call=2)
    assert pool.result(later).error_kind == "worker_load_failed"
    assert len(pool.events("constructed")) == 1
    pool.plan()
    pool.clock.advance(41)
    recovered = pool.job(call=3)
    assert pool.result(recovered).status == "ok"
    pool.finish()


def _faulty_launch(
    pool: SupervisorHarness, mode: str, key: WorkerKey, env: dict[str, str]
) -> WorkerLaunch:
    """Launch a faulty peer that the trusted worker cannot produce."""
    launch = pool.launch(key, env)
    return replace(
        launch,
        argv=(*launch.argv[:-1], "tests.stt_pool_fake_worker"),
        env={**launch.env, "ASKVLM_TEST_PROTOCOL": mode},
    )


def test_invalid_loaded_after_readiness_is_a_crash_without_load_cooldown(
    pool: SupervisorHarness,
) -> None:
    """Catch duplicate readiness poisoning a healthy worker key."""
    pool.options = replace(
        pool.options,
        worker_launch=lambda key, env: _faulty_launch(
            pool, "duplicate_loaded", key, env
        ),
    )
    job = pool.job(mode="hold", text="healthy retry")
    pool.start()
    pool.submitted(job)
    pool.submitted(job, count=2)
    pool.release(job)
    result = pool.result(job)
    assert result.status == "ok"
    assert result.attempts == 2
    assert len(pool.events("constructed")) == 2
    assert len(pool.events("submitted", job)) == 2
    pool.finish()


@pytest.mark.skipif(
    sys.platform != "win32", reason="requires the Win32 termination boundary"
)
@pytest.mark.parametrize("cooldown", [2.0, 40.0])
def test_late_loaded_cannot_clear_cooldown_or_spawn_before_exit_is_confirmed(
    pool: SupervisorHarness, monkeypatch: pytest.MonkeyPatch, cooldown: float
) -> None:
    """Catch late readiness, repeated cooldown or replacement overlap."""
    kernel = winjob._kernel32()  # noqa: SLF001 - external Win32 API boundary
    permit_kill = threading.Event()
    kill_requested = threading.Event()

    class DelayedKernel:
        def __getattr__(self, name: str) -> object:
            return getattr(kernel, name)

        @staticmethod
        def TerminateJobObject(handle: object, code: int) -> int:  # noqa: N802 - external Win32 symbol
            if not permit_kill.is_set():
                kill_requested.set()
                return 1
            return kernel.TerminateJobObject(handle, code)

    monkeypatch.setattr(winjob, "_kernel32", DelayedKernel)
    pool.config = configuration(
        cpu={"max_slots": 2}, scheduling={"load_failure_cooldown_s": cooldown}
    )
    pool.options = replace(
        pool.options,
        load_timeout_s=5,
        worker_launch=lambda key, env: _faulty_launch(
            pool, "late_loaded_exit", key, env
        ),
    )
    pool.plan(load_modes={"small": "gate"})
    first = pool.job()
    try:
        pool.start()
        loading = pool.until(lambda: pool.events("loading"), bool)[0]
        pool.clock.advance(6)
        assert kill_requested.wait(30)
        assert pool.result(first).error_kind == "worker_load_failed"
        if cooldown == 2:
            pool.clock.advance(3)
            waiting = pool.job(call=2)
            heartbeat = pool.heartbeat(lambda value: value["queue"]["pending"] == 1)
            assert len(heartbeat["workers"]) == 1
            assert heartbeat["workers"][0]["state"] == "stopping"
            assert len(pool.events("constructed")) == 1
        else:
            pool.clock.advance(30)
        (pool.engine_root / "load-release").touch()
        pool.until(
            lambda: (pool.engine_root / f"loaded-{loading['worker_id']}").exists(), bool
        )
        (pool.engine_root / "exit-release").touch()
        pool.heartbeat(
            lambda value: all(
                worker["worker_id"] != loading["worker_id"]
                for worker in value["workers"]
            )
        )
        assert not pool.events("submitted", first)
        if cooldown == 2:
            assert pool.result(waiting).status == "ok"
            assert len(pool.events("constructed")) == 2
        else:
            still_cooling = pool.job(call=2)
            assert pool.result(still_cooling).error_kind == "worker_load_failed"
            pool.clock.advance(11)
            recovered = pool.job(call=3)
            assert pool.result(recovered).status == "ok"
            assert len(pool.events("constructed")) == 2
    finally:
        permit_kill.set()
        (pool.engine_root / "load-release").touch()
        (pool.engine_root / "exit-release").touch()
        pool.finish()


def test_three_crashes_pause_configuration_until_cooldown_expires(
    pool: SupervisorHarness,
) -> None:
    """Catch a repeatedly crashing key producing an unbounded restart storm."""
    pool.config = configuration(scheduling={"max_attempts": 4})
    job = pool.job(mode="crash", fail_attempts=3)
    pool.start()
    pool.submitted(job, 3)
    pool.heartbeat(
        lambda value: value["queue"] == {"pending": 1, "running": 0}
        and not value["workers"]
    )
    assert len(pool.events("submitted", job)) == 3
    assert read_result(pool.paths, job.job_id) is None
    pool.clock.advance(31)
    result = pool.result(job)
    assert result.status == "ok"
    assert result.attempts == 4
    pool.finish()


@pytest.mark.parametrize("phase", ["pending", "running"])
@pytest.mark.parametrize("reason", ["cancelled", "expired"])
def test_cancel_and_deadline_publish_once_at_pending_and_running_stages(
    pool: SupervisorHarness, phase: str, reason: str
) -> None:
    """Catch cancel/deadline loss or wrong terminal reason before and after dispatch."""
    if phase == "pending":
        pool.hardware.free_commit = 0
    job = pool.job(mode="hold", deadline=pool.clock.peek() + 10)
    pool.start()
    if phase == "running":
        pool.submitted(job)
    else:
        pool.heartbeat(lambda value: value["queue"]["pending"] == 1)
    if reason == "cancelled":
        request_cancel(pool.paths, job.job_id)
    else:
        pool.clock.advance(11)
    result = pool.result(job)
    assert result.status == reason
    assert result.error_kind == reason
    assert result.attempts == (1 if phase == "running" else 0)
    assert not pool.paths.incoming_file(job.job_id).exists()
    original = pool.paths.result_file(job.job_id).read_bytes()
    pool.finish()
    assert pool.paths.result_file(job.job_id).read_bytes() == original


def test_cancel_grace_kill_restores_other_running_attempts(
    pool: SupervisorHarness,
) -> None:
    """Catch ignored cancellation consuming collateral victim attempts."""
    pool.config = configuration(
        cpu={"max_slots": 2, "max_slots_per_worker": 2}, scheduling={"max_attempts": 1}
    )
    target = pool.job(mode="hold", ignore_cancel=True)
    victim = pool.job(mode="hold", call=2)
    pool.start()
    target_start = pool.submitted(target)
    victim_start = pool.submitted(victim)
    assert target_start["worker_id"] == victim_start["worker_id"]
    request_cancel(pool.paths, target.job_id)
    pool.until(lambda: pool.events("cancel", target), bool)
    pool.clock.advance(6)
    assert pool.result(target).status == "cancelled"
    restarted = pool.submitted(victim, 2)
    assert restarted["worker_id"] != victim_start["worker_id"]
    assert restarted["attempt_id"] == victim_start["attempt_id"]
    pool.release(victim)
    result = pool.result(victim)
    assert result.status == "ok"
    assert result.attempts == 1
    pool.finish()


def test_cuda_unknown_waits_then_serves_gpu_when_probe_recovers(
    pool: SupervisorHarness,
) -> None:
    """Catch premature CPU fallback before a GPU probe recovers."""
    pool.config = configuration(gpu={"max_workers": 1})
    job = pool.job(device="cuda")
    pool.start()
    pool.heartbeat(lambda value: value["queue"]["pending"] == 1)
    assert pool.hardware.gpu_calls >= 2
    assert not pool.events("constructed")
    pool.hardware.gpu_available = True
    result = pool.result(job)
    assert result.status == "ok"
    assert result.backend == "gpu"
    assert not result.warnings
    pool.finish()


def test_cuda_unknown_becomes_absent_only_after_grace_and_failed_probes(
    pool: SupervisorHarness,
) -> None:
    """Catch unconfirmed CUDA fallback or missing warnings."""
    pool.config = configuration(gpu={"max_workers": 1})
    job = pool.job(device="cuda")
    pool.start()
    pool.heartbeat(lambda value: value["queue"]["pending"] == 1)
    assert not pool.events("constructed")
    pool.clock.advance(31)
    result = pool.result(job)
    assert result.status == "ok"
    assert result.backend == "cpu"
    assert any("ASKVLM_CUDA_UNAVAILABLE" in warning for warning in result.warnings)
    pool.finish()


def test_cuda_presence_stays_present_after_later_probe_failures(
    pool: SupervisorHarness,
) -> None:
    """Catch failed GPU probes changing explicit CUDA provenance."""
    pool.config = configuration(gpu={"max_workers": 1})
    pool.hardware.gpu_available = True
    first = pool.job(device="cuda", mode="hold")
    pool.start()
    first_start = pool.submitted(first)
    pool.hardware.gpu_available = False
    observed = pool.hardware.gpu_calls
    pool.until(lambda: pool.hardware.gpu_calls, lambda count: count > observed)
    second = pool.job(device="cuda", call=2)
    pool.clock.advance(31)
    pool.release(first)
    assert pool.result(first).status == "ok"
    result = pool.result(second)
    assert result.backend == "gpu"
    assert result.worker_id == first_start["worker_id"]
    assert not result.warnings
    pool.finish()


@pytest.mark.parametrize(
    ("device", "error", "backend"),
    [
        ("auto", "cuda_oom", "cpu"),
        ("cuda", "cuda_oom", "gpu"),
        ("cuda", "memory_limit", "gpu"),
        ("cpu", "memory_limit", "cpu"),
    ],
)
def test_resource_error_retires_worker_and_replans_the_correct_backend(
    pool: SupervisorHarness, device: str, error: str, backend: str
) -> None:
    """Catch backend drift or reuse of a retiring worker."""
    pool.config = configuration(gpu={"max_workers": 1})
    pool.hardware.gpu_available = True
    instructions = {"mode": error, "fail_attempts": 1, "wait": True}
    job = pool.job(
        device=device,
        **(
            {"gpu": instructions, "cpu": {"mode": "ok"}}
            if device == "auto"
            else instructions
        ),
    )
    pool.start()
    pool.submitted(job)
    pool.release(job)
    result = pool.result(job)
    assert result.status == "ok"
    assert result.backend == backend
    assert result.attempts == 2
    events = pool.events("submitted", job)
    assert len(events) == 2
    assert events[0]["worker_id"] != events[1]["worker_id"]
    assert events[1]["backend"] == backend
    pool.until(
        lambda: pool.events("closed"),
        lambda closed: any(
            event["worker_id"] == events[0]["worker_id"] for event in closed
        ),
    )
    pool.finish()


def test_gpu_scale_out_waits_for_a_fresh_vram_probe_after_worker_change(
    pool: SupervisorHarness,
) -> None:
    """Catch reusing a pre-load VRAM snapshot to authorize a second GPU process."""
    pool.config = configuration(gpu={"max_workers": 2})
    pool.hardware.gpu_available = True
    pool.hardware.block_gpu_after = 1
    pool.hardware.gpu_gate.clear()
    first = pool.job(device="cuda", mode="hold")
    pool.start()
    pool.submitted(first)
    second = pool.job(device="cuda", call=2)
    pool.heartbeat(lambda value: value["queue"] == {"pending": 1, "running": 1})
    assert len(pool.events("constructed")) == 1
    pool.hardware.gpu_gate.set()
    result = pool.result(second)
    assert result.backend == "gpu"
    assert result.worker_id != pool.submitted(first)["worker_id"]
    pool.release(first)
    assert pool.result(first).status == "ok"
    pool.finish()


@pytest.mark.parametrize("first_backend", ["cpu", "gpu"])
def test_auto_memory_limit_keeps_failed_backend_after_gpu_probe_recovers(
    pool: SupervisorHarness, first_backend: str
) -> None:
    """Catch memory retries drifting to a newly available backend."""
    pool.config = configuration(gpu={"max_workers": 1})
    pool.hardware.gpu_available = first_backend == "gpu"
    job = pool.job(
        device="auto",
        cpu={"mode": "memory_limit", "wait": True, "fail_attempts": 1},
        gpu={"mode": "memory_limit", "wait": True, "fail_attempts": 1},
    )
    pool.start()
    first = pool.submitted(job)
    assert first["backend"] == first_backend
    if first_backend == "gpu":
        warm_cpu = pool.job(device="cpu", call=2)
        assert pool.result(warm_cpu).status == "ok"
    calls = pool.hardware.gpu_calls
    pool.hardware.gpu_available = True
    pool.until(lambda: pool.hardware.gpu_calls, lambda count: count >= calls + 2)
    pool.release(job)
    result = pool.result(job)
    assert result.status == "ok"
    assert result.backend == first_backend
    assert result.attempts == 2
    starts = pool.events("submitted", job)
    assert len(starts) == 2
    assert starts[1]["backend"] == first_backend
    assert starts[1]["worker_id"] != first["worker_id"]
    pool.finish()


@pytest.fixture
def job_caps(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Observe real Win32 memory reservations without changing them."""
    if sys.platform != "win32":
        pytest.skip("requires Windows Job Object reservation boundary")
    kernel = winjob._kernel32()  # noqa: SLF001 - real external Win32 ABI boundary
    caps: list[int] = []

    class ObservedKernel:
        def __getattr__(self, name: str) -> object:
            return getattr(kernel, name)

        @staticmethod
        def SetInformationJobObject(  # noqa: N802 - external Win32 symbol
            handle: object, info_class: int, pointer: object, size: int
        ) -> int:
            if info_class == 9:
                info = ctypes.cast(
                    pointer,
                    ctypes.POINTER(winjob._JobExtendedLimitInformation),  # noqa: SLF001 - real external ABI
                ).contents
                if info.JobMemoryLimit:
                    caps.append(int(info.JobMemoryLimit))
            return kernel.SetInformationJobObject(handle, info_class, pointer, size)

    monkeypatch.setattr(winjob, "_kernel32", ObservedKernel)
    return caps


@pytest.mark.parametrize(
    ("headroom_gib", "extra_bytes"), [(None, GIB), (0, 0), (0.5, GIB // 2)]
)
def test_child_headroom_changes_only_the_job_cap_and_not_admission(
    pool: SupervisorHarness,
    job_caps: list[int],
    headroom_gib: float | None,
    extra_bytes: int,
) -> None:
    """Catch missing child allowance or counting it in the admission budget."""
    pool.config = configuration(memory={"child_headroom_gib": 0})
    baseline = pool.job()
    pool.start()
    assert pool.result(baseline).status == "ok"
    pool.finish()
    assert len(job_caps) == 1
    reservation = job_caps[0]
    memory: dict[str, object] = {"commit_budget_gib": reservation / GIB}
    if headroom_gib is not None:
        memory["child_headroom_gib"] = headroom_gib
    pool.config = configuration(memory=memory)
    assert pool.config.memory.child_headroom_bytes == extra_bytes
    following = pool.job(call=2)
    pool.start()
    assert pool.result(following).status == "ok"
    pool.finish()
    assert job_caps == [reservation, reservation + extra_bytes]


@pytest.mark.parametrize(
    ("backend", "error"), [("cpu", "memory_limit"), ("gpu", "cuda_oom")]
)
def test_retry_reserves_grown_memory_before_retiring_worker_exits(
    pool: SupervisorHarness, job_caps: list[int], backend: str, error: str
) -> None:
    """Catch retry retaining its failed worker memory cap."""
    pool.config = configuration(
        cpu={"max_slots": 2}, gpu={"max_workers": 2}, lifecycle={"stop_grace_s": 60}
    )
    pool.hardware.gpu_available = True
    pool.plan(close_gate=True)
    job = pool.job(
        device="cpu" if backend == "cpu" else "cuda",
        mode=error,
        wait=True,
        fail_attempts=1,
    )
    pool.start()
    first = pool.submitted(job)
    pool.release(job)
    second = pool.submitted(job, count=2)
    result = pool.result(job)
    assert result.status == "ok"
    assert result.attempts == 2
    assert first["worker_id"] != second["worker_id"]
    assert len(job_caps) == 2
    headroom = pool.config.memory.child_headroom_bytes
    assert job_caps[1] - headroom == pytest.approx(
        (job_caps[0] - headroom) * 1.25, abs=2
    )
    first_pid = next(
        event["pid"]
        for event in pool.events("constructed")
        if event["worker_id"] == first["worker_id"]
    )
    assert psutil.pid_exists(first_pid)
    pool.finish()
    learned = fsio.read_json(pool.paths.estimates_file)
    assert learned is not None
    entry = next(iter(learned["keys"].values()))
    assert entry["commit_factor"] == 1.25
    assert entry["vram_factor"] == (1.25 if backend == "gpu" else 1.0)


@pytest.mark.parametrize(
    ("backend", "error"), [("cpu", "memory_limit"), ("gpu", "cuda_oom")]
)
def test_two_resource_errors_of_one_worker_bump_each_memory_estimate_once(
    pool: SupervisorHarness, backend: str, error: str
) -> None:
    """Catch memory growth repeated per slot and on exit."""
    pool.config = configuration(
        cpu={"max_slots": 2, "max_slots_per_worker": 2},
        gpu={"max_workers": 1, "batch_lanes": {"whisper": 2, "gigaam": 2}},
        scheduling={"max_attempts": 1},
    )
    pool.hardware.gpu_available = True
    jobs = [
        pool.job(device="cpu" if backend == "cpu" else "cuda", mode=error, wait=True)
        for _ in range(2)
    ]
    pool.start()
    starts = [pool.submitted(job) for job in jobs]
    assert starts[0]["worker_id"] == starts[1]["worker_id"]
    for job in jobs:
        pool.release(job)
    assert [pool.result(job).error_kind for job in jobs] == [error, error]
    pool.finish()
    learned = fsio.read_json(pool.paths.estimates_file)
    assert learned is not None
    entry = next(iter(learned["keys"].values()))
    assert entry["commit_factor"] == 1.25
    assert entry["vram_factor"] == (1.25 if backend == "gpu" else 1.0)


def test_transient_result_write_failure_retries_without_rerunning_job(
    pool: SupervisorHarness,
    monkeypatch: pytest.MonkeyPatch,
    memory_logs: queue.Queue[logging.LogRecord],
) -> None:
    """Catch lost publication/telemetry, premature deletion or repeated inference."""
    pool.plan(memory_probe="initialized")
    job = pool.job(text="stored transcript")
    target = pool.paths.result_file(job.job_id)
    original_link, original_replace = os.link, os.replace
    blocked = threading.Event()
    blocked.set()
    failed_writes: list[object] = []
    writes_denied = threading.Event()

    def link_at_filesystem(source: object, destination: object) -> None:
        if Path(destination) == target and blocked.is_set():
            msg = "test result directory is temporarily unwritable"
            raise PermissionError(msg)
        original_link(source, destination)

    def replace_at_filesystem(source: object, destination: object) -> None:
        if Path(destination) == target and blocked.is_set():
            failed_writes.append(destination)
            if len(failed_writes) >= 3:
                writes_denied.set()
            msg = "test result directory is temporarily unwritable"
            raise PermissionError(msg)
        original_replace(source, destination)

    monkeypatch.setattr(os, "link", link_at_filesystem)
    monkeypatch.setattr(os, "replace", replace_at_filesystem)
    pool.start()
    assert writes_denied.wait(30)
    assert pool.paths.incoming_file(job.job_id).exists()
    assert read_attempts(pool.paths, job.job_id) == 1
    assert len(pool.events("submitted", job)) == 1
    blocked.clear()
    fields = _memory_log_fields(memory_logs, job.job_id)
    result = read_result(pool.paths, job.job_id)
    assert result is not None
    assert result.status == "ok"
    assert result.text == "stored transcript"
    assert not pool.paths.incoming_file(job.job_id).exists()
    assert len(pool.events("submitted", job)) == 1
    assert fields["cuda_reserved_bytes"] == "8192"
    assert fields["cuda_allocated_bytes"] == "1024"
    assert fields["cuda_max_reserved_bytes"] == "16384"
    assert fields["cufft_plan_cache_size"] == "7"
    if sys.platform == "win32":
        assert int(fields["private_bytes"]) > 0
        assert int(fields["job_peak_bytes"]) > 0
    pool.finish()


@pytest.mark.parametrize("repair", [True, False])
def test_unreadable_retry_window_starts_at_first_read_not_old_mtime(
    pool: SupervisorHarness, *, repair: bool
) -> None:
    """Catch rejection based on mtime instead of first read failure."""
    job = pool.job()
    path = pool.paths.incoming_file(job.job_id)
    path.write_bytes(b'{"schema":')
    old = time.time() - 3600
    os.utime(path, (old, old))
    pool.start()
    pool.until(lambda: pool.hardware.commit_calls, lambda count: count >= 2)
    assert read_result(pool.paths, job.job_id) is None
    pool.clock.advance(20)
    observed = pool.hardware.commit_calls
    pool.until(lambda: pool.hardware.commit_calls, lambda count: count > observed)
    assert read_result(pool.paths, job.job_id) is None
    if repair:
        submit_job(pool.paths, job)
        assert pool.result(job).status == "ok"
    else:
        pool.clock.advance(11)
        result = pool.result(job)
        assert result.status == "error"
        assert result.error_kind == "input_invalid"
        assert not pool.events("constructed")
    pool.finish()


def test_terminal_result_wins_over_duplicate_worker_reply_cancel_and_resubmission(
    pool: SupervisorHarness,
) -> None:
    """Catch replies or replay rewriting a terminal result."""
    job = pool.job(mode="duplicate", text="terminal transcript")
    pool.start()
    result = pool.result(job)
    assert result.status == "ok"
    original = pool.paths.result_file(job.job_id).read_bytes()
    submit_job(pool.paths, job)
    request_cancel(pool.paths, job.job_id)
    observed = pool.hardware.commit_calls
    pool.until(lambda: pool.hardware.commit_calls, lambda count: count >= observed + 2)
    assert pool.paths.result_file(job.job_id).read_bytes() == original
    assert len(pool.events("submitted", job)) == 1
    pool.finish()


def test_existing_result_is_not_recognized_again_after_restart(
    pool: SupervisorHarness,
) -> None:
    """Catch restart replay of a job whose terminal result is already durable."""
    job = pool.job()
    write_result(pool.paths, ResultRecord(job.job_id, "ok", text="already durable"))
    pool.start()
    pool.until(lambda: pool.hardware.commit_calls, lambda count: count >= 2)
    assert not pool.events("constructed")
    assert not pool.paths.incoming_file(job.job_id).exists()
    assert pool.result(job).text == "already durable"
    pool.finish()


def test_unexpected_loop_failure_releases_workers_descendants_lock_and_heartbeat(
    pool: SupervisorHarness,
) -> None:
    """Catch exceptions leaking workers, descendants or ownership."""
    pool.plan(spawn_child=True)
    job = pool.job(mode="hold")
    pool.start()
    pool.submitted(job)
    assert pool.events("child")
    pool.hardware.fail_commit = True
    pool.finish(1, request_stop=False)
    assert pool.paths.incoming_file(job.job_id).exists()
    assert read_result(pool.paths, job.job_id) is None


def test_second_supervisor_returns_without_disturbing_current_owner(
    pool: SupervisorHarness,
) -> None:
    """Catch lock bypass or damage to the current heartbeat."""
    job = pool.job(mode="hold")
    pool.start()
    pool.submitted(job)
    heartbeat = pool.heartbeat(lambda value: value["queue"]["running"] == 1)
    assert run_supervisor(pool.paths.root, pool.config, options=pool.options) == 0
    current = pool.heartbeat(lambda value: value["queue"]["running"] == 1)
    assert current["instance_id"] == heartbeat["instance_id"]
    pool.release(job)
    assert pool.result(job).status == "ok"
    pool.finish()


def test_stop_heartbeat_is_stopping_while_real_workers_close(
    pool: SupervisorHarness,
) -> None:
    """Catch shutdown advertising a ready supervisor."""
    pool.plan(close_gate=True)
    pool.config = configuration(lifecycle={"stop_grace_s": 10})
    job = pool.job(mode="hold")
    pool.start()
    pool.submitted(job)
    pool.stop_event.set()
    pool.heartbeat(lambda value: value["state"] == "stopping")
    pool.until(lambda: pool.events("closed"), bool)
    assert pool.thread is not None
    assert pool.thread.is_alive()
    pool.finish()
    assert read_attempts(pool.paths, job.job_id) == 0


def test_idle_exit_clears_heartbeat_before_bounded_probe_teardown(
    pool: SupervisorHarness,
) -> None:
    """Catch clients trusting a heartbeat after the supervisor has decided to exit."""
    pool.hardware.gpu_available = True
    pool.hardware.block_gpu_after = 1
    pool.hardware.gpu_gate.clear()
    job = pool.job()
    pool.start()
    assert pool.result(job).status == "ok"
    pool.until(lambda: pool.hardware.gpu_calls, lambda count: count >= 2)
    pool.clock.advance(31)
    pool.heartbeat(lambda value: not value["workers"])
    pool.clock.advance(6)
    pool.until(lambda: pool.paths.heartbeat_file.exists(), lambda exists: not exists)
    assert pool.thread is not None
    assert pool.thread.is_alive()
    assert run_supervisor(pool.paths.root, pool.config, options=pool.options) == 0
    pool.finish(request_stop=False)
