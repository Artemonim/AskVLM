"""Real engine orchestration with fake ML endpoints and bounded audio fixtures."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from core.stt_errors import CudaOutOfMemoryError, DeviceUnavailableError
from core.stt_pool import engines
from core.stt_pool.engines import AdmissionTicket, admission_order, deal_lanes
from tests.stt_pool_engine_support import (
    RATE,
    EngineRuntime,
    item,
    spec,
    wav,
)
from tests.stt_pool_engine_support import (
    RESULT_TIMEOUT_S as WAIT_S,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from core.stt_pool.engine_api import WorkItem
    from core.stt_pool.keys import Backend
    from tests.stt_pool_engine_support import EngineHandle


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[EngineRuntime]:
    """Substitute model boundaries and verify all owned threads and directories are released."""
    boundary = EngineRuntime()
    boundary.install(monkeypatch, tmp_path)
    try:
        yield boundary
    finally:
        boundary.release_and_close()
        assert not list(tmp_path.glob("askvlm-pool-*"))
        assert not any(
            thread.name.startswith(
                ("stt-slot-", "stt-prep-", "stt-batch-", "stt-infer-", "stt-keepalive")
            )
            for thread in threading.enumerate()
        )


@pytest.mark.parametrize(
    ("provider", "backend", "engine_type"),
    [
        ("whisper", "cpu", engines.CpuPipelineEngine),
        ("gigaam-ctc", "cpu", engines.CpuPipelineEngine),
        ("gigaam-e2e-ctc", "cpu", engines.CpuPipelineEngine),
        ("gigaam-e2e-ctc", "gpu", engines.GigaamGpuBatchEngine),
        ("whisper", "gpu", engines.WhisperGpuEngine),
    ],
)
def test_engine_load_preserves_provider_key_device_and_shared_weights(
    runtime: EngineRuntime, provider: str, backend: Backend, engine_type: type[object]
) -> None:
    """Catch silent model substitution, non-strict placement or weights loaded once per CPU slot."""
    configuration = spec(provider, backend, whisper_model="different-model")
    handle = runtime.open(configuration)
    assert isinstance(handle.engine, engine_type)
    assert runtime.load_count == 1
    options = runtime.pipelines[0].options
    assert options["stt_provider"] == provider
    assert options["device"] == ("cpu" if backend == "cpu" else "cuda")
    assert options["strict_device"] is True
    if provider == "whisper":
        assert options["whisper_model"] == "Org/RequestedModel"
        if backend == "cpu":
            assert options["num_workers"] == configuration.capacity
    else:
        assert options["gigaam_threadsafe"] is True
    handle.engine.close()
    handle.engine.close()
    assert runtime.unload_count == 1


@pytest.mark.parametrize("provider", ["whisper", "gigaam-e2e-ctc"])
def test_engine_actual_device_mismatch_fails_without_cpu_recovery(
    runtime: EngineRuntime, provider: str
) -> None:
    """Catch mismatched CUDA placement being advertised as a valid reserved GPU worker."""
    runtime.loaded_device = "cpu"
    with pytest.raises(DeviceUnavailableError):
        runtime.open(spec(provider, "gpu"))
    assert runtime.load_count == 1
    assert runtime.unload_count == 1


def test_cpu_slots_keep_language_and_temp_resources_per_job(
    runtime: EngineRuntime, tmp_path: Path
) -> None:
    """Catch shared language state, accidental serial execution or premature temp reclamation."""
    inputs = [item(tmp_path / "A.wav"), item(tmp_path / "B.wav", seq=2, language="en")]
    runtime.process_gates = {job.input_path.name: threading.Event() for job in inputs}
    handle = runtime.open(spec())
    for job in inputs:
        handle.engine.submit(job)
    calls = [runtime.process_calls.get(timeout=WAIT_S) for _ in inputs]
    assert len({call.thread_id for call in calls}) == 2
    assert {call.language for call in calls} == {"ru", "en"}
    assert len({call.work_dir for call in calls}) == 2
    for call in calls:
        assert call.work_dir.name.startswith("askvlm-pool-")
        keepalive = call.work_dir / engines.KEEPALIVE_FILE_NAME
        assert keepalive.is_file()
        os.utime(keepalive, (100, 100))
    # * Exercise the maintenance operation directly; the test never waits for a timer.
    handle.engine._keepalive.touch_all()  # noqa: SLF001 - deterministic keep-alive maintenance entry
    assert all(
        (call.work_dir / engines.KEEPALIVE_FILE_NAME).stat().st_mtime > 100
        for call in calls
    )
    for gate in runtime.process_gates.values():
        gate.set()
    results = {result.job_id: result for result in (handle.result(), handle.result())}
    assert results[inputs[0].job_id].text == "ru:A.wav"
    assert results[inputs[1].job_id].text == "en:B.wav"
    assert all(result.status == "ok" for result in results.values())
    assert all(not call.work_dir.exists() for call in calls)
    assert runtime.load_count == 1


def test_cpu_active_submit_cancel_race_emits_once_and_slot_survives(
    runtime: EngineRuntime, tmp_path: Path
) -> None:
    """Catch duplicate active submission, stale-attempt cancellation or a cancelled slot dying."""
    job = item(tmp_path / "blocked.wav")
    gate = runtime.process_gates[job.input_path.name] = threading.Event()
    handle = runtime.open(spec())
    handle.engine.submit(job)
    runtime.process_calls.get(timeout=WAIT_S)
    handle.engine.submit(job)
    handle.engine.cancel(job.job_id, "obsolete:1")
    handle.engine.cancel(job.job_id, job.attempt_id)
    handle.engine.cancel(job.job_id, job.attempt_id)
    gate.set()
    result = handle.result()
    assert result.status == "cancelled"
    assert result.attempt_id == job.attempt_id
    healthy = item(tmp_path / "healthy.wav", seq=2)
    handle.engine.submit(healthy)
    assert handle.result().text == "ru:healthy.wav"
    handle.engine.close()
    assert handle.results.empty()


def test_cpu_close_during_inference_relinquishes_the_attempt(
    runtime: EngineRuntime, tmp_path: Path
) -> None:
    """Catch a closing CPU slot publishing a late result or unloading weights still in use."""
    job = item(tmp_path / "blocked.wav")
    gate = runtime.process_gates[job.input_path.name] = threading.Event()
    handle = runtime.open(spec())
    handle.engine.submit(job)
    runtime.process_calls.get(timeout=WAIT_S)
    engine = handle.engine
    with ThreadPoolExecutor(max_workers=1) as executor:
        closing = executor.submit(engine.close)
        with engine._cond:  # noqa: SLF001 - synchronize real close before releasing fake ML inference
            assert engine._cond.wait_for(lambda: engine._closing, timeout=WAIT_S)  # noqa: SLF001 - synchronize real close before releasing fake ML inference
        assert runtime.unload_count == 0
        gate.set()
        closing.result(timeout=WAIT_S)
    assert handle.results.empty()
    assert runtime.unload_count == 1


@pytest.mark.parametrize(
    ("outcome", "status", "kind"),
    [
        (" \n\t", "empty", None),
        (
            RuntimeError("Too long wav file: transcribe_longform"),
            "error",
            "transcribe_failed",
        ),
        (CudaOutOfMemoryError("CUDA out of memory"), "error", "cuda_oom"),
        (DeviceUnavailableError("device unavailable"), "error", "device_unavailable"),
        (MemoryError("allocation failed"), "error", "memory_limit"),
    ],
)
def test_cpu_job_failure_preserves_classification_and_next_attempt(
    runtime: EngineRuntime,
    tmp_path: Path,
    outcome: str | Exception,
    status: str,
    kind: str | None,
) -> None:
    """Catch one model exception killing a slot, empty misclassification or lost bot error markers."""
    runtime.pipeline_texts["first.wav"] = outcome
    handle = runtime.open(spec())
    handle.engine.submit(item(tmp_path / "first.wav"))
    result = handle.result()
    assert result.status == status
    assert result.error_kind == kind
    if kind == "transcribe_failed":
        assert "Too long wav file" in (result.error_detail or "")
        assert "transcribe_longform" in (result.error_detail or "")
    handle.engine.submit(item(tmp_path / "next.wav", seq=2))
    assert handle.result().status == "ok"


def test_cpu_cleanup_failure_is_guarded_and_does_not_lose_the_attempt(
    runtime: EngineRuntime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch a real deletion failure dropping a result or permanently losing a slot."""
    unlink = os.unlink
    failed: list[Path] = []

    def filesystem_failure(path: str | Path, **options: object) -> None:
        target = Path(path)
        if target.name == engines.KEEPALIVE_FILE_NAME and not failed:
            failed.append(target)
            msg = "test filesystem refuses one deletion"
            raise PermissionError(msg)
        unlink(path, **options)

    monkeypatch.setattr(os, "unlink", filesystem_failure)
    handle = runtime.open(spec())
    first = item(tmp_path / "first.wav")
    handle.engine.submit(first)
    result = handle.result()
    assert (result.job_id, result.attempt_id) == (first.job_id, first.attempt_id)
    handle.engine.submit(item(tmp_path / "next.wav", seq=2))
    assert handle.result().status == "ok"
    handle.engine.close()
    assert handle.results.empty()
    assert failed


@pytest.mark.integration
def test_gigaam_windows_are_joined_and_inferred_from_one_thread(
    runtime: EngineRuntime, tmp_path: Path
) -> None:
    """Catch concurrent CUDA inference corruption, dropped windows or empty-window separator changes."""
    runtime.gigaam_texts = {1000: " first ", 2000: "", 3000: "last "}
    first = item(wav(tmp_path / "first.wav", [1000, 2000, 3000]), duration_s=None)
    second = item(wav(tmp_path / "second.wav", [4000]), seq=2, duration_s=None)
    handle = runtime.open(spec("gigaam-e2e-ctc", "gpu", batch_lanes={"gigaam": 2}))
    handle.engine.submit(first)
    assert handle.result().text == "first \n\nlast"
    handle.engine.submit(second)
    result = handle.result()
    assert result.text == "4000"
    assert result.audio_s == 20.0
    handle.engine.close()
    batches = list(runtime.batch_calls.queue)
    assert len(batches) >= 3
    assert all(len(batch.markers) <= 2 for batch in batches)
    assert len({batch.thread_id for batch in batches}) == 1


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["preparation", "inference"])
def test_gigaam_failed_job_does_not_kill_batch_loop(
    runtime: EngineRuntime, tmp_path: Path, kind: str
) -> None:
    """Catch preparation or CUDA failure dropping future jobs and their error classification."""
    first_path = tmp_path / "first.wav"
    if kind == "preparation":
        first_path.write_bytes(b"invalid audio")
    else:
        wav(first_path, [1000], window_s=0.1)
        runtime.batch_errors[1000] = CudaOutOfMemoryError("CUDA out of memory")
    handle = runtime.open(spec("gigaam-e2e-ctc", "gpu"))
    handle.engine.submit(item(first_path))
    failed = handle.result()
    assert failed.status == "error"
    if kind == "inference":
        assert failed.error_kind == "cuda_oom"
    healthy = item(wav(tmp_path / "healthy.wav", [2000], window_s=0.1), seq=2)
    handle.engine.submit(healthy)
    assert handle.result().text == "2000"


@pytest.mark.integration
def test_gigaam_prepared_wav_read_error_answers_attempt_and_loop_survives(
    runtime: EngineRuntime, tmp_path: Path
) -> None:
    """Catch an unreadable prepared WAV losing its attempt or killing the GPU loop."""
    gate = runtime.batch_gate = threading.Event()
    handle = runtime.open(spec("gigaam-e2e-ctc", "gpu"))
    first = item(wav(tmp_path / "blocker.wav", [1000], window_s=0.1))
    broken = item(wav(tmp_path / "broken.wav", [2000], window_s=0.1), seq=2)
    handle.engine.submit(first)
    runtime.batch_calls.get(timeout=WAIT_S)
    handle.engine.submit(broken)
    handle.wait_ready(broken)
    engine = handle.engine
    # * The ownership barrier locates the real artifact after FFmpeg has closed it.
    with engine._cond:  # noqa: SLF001 - identify the ready job's owned filesystem artifact
        prepared = next(
            job.wav_path
            for job in engine._ready  # noqa: SLF001 - identify the ready job's owned filesystem artifact
            if job.key == (broken.job_id, broken.attempt_id)
        )
    assert prepared is not None
    prepared.write_bytes(b"invalid WAV header")
    gate.set()
    results = {result.job_id: result for result in (handle.result(), handle.result())}
    assert results[first.job_id].status == "ok"
    failed = results[broken.job_id]
    assert failed.status == "error"
    assert failed.attempt_id == broken.attempt_id
    assert failed.error_kind == "transcribe_failed"
    healthy = item(wav(tmp_path / "healthy.wav", [3000], window_s=0.1), seq=3)
    handle.engine.submit(healthy)
    assert handle.result().text == "3000"


def _gpu_job(runtime: EngineRuntime, tmp_path: Path, seq: int) -> WorkItem:
    """Prepare one short input understood by either real GPU engine."""
    path = wav(tmp_path / f"clip-{seq}.wav", [seq * 1000], window_s=0.1)
    runtime.audio[path.name] = np.full(RATE // 10, seq * 1000 / 32768, dtype=np.float32)
    return item(path, seq=seq)


@pytest.mark.integration
@pytest.mark.parametrize("provider", ["whisper", "gigaam-e2e-ctc"])
def test_gpu_prepared_cancel_answers_before_blocked_inference(
    runtime: EngineRuntime, tmp_path: Path, provider: str
) -> None:
    """Catch prepared cancellation waiting for another job's CUDA call or emitting twice afterward."""
    gate = runtime.batch_gate = threading.Event()
    handle = runtime.open(spec(provider, "gpu"))
    first = _gpu_job(runtime, tmp_path, 1)
    second = _gpu_job(runtime, tmp_path, 2)
    handle.engine.submit(first)
    runtime.batch_calls.get(timeout=WAIT_S)
    handle.engine.submit(second)
    handle.wait_ready(second)
    handle.engine.cancel(second.job_id, second.attempt_id)
    cancelled = handle.result()
    assert cancelled.job_id == second.job_id
    assert cancelled.status == "cancelled"
    handle.engine.cancel(second.job_id, second.attempt_id)
    gate.set()
    assert handle.result().job_id == first.job_id
    handle.engine.close()
    assert handle.results.empty()


@pytest.mark.integration
@pytest.mark.parametrize("provider", ["whisper", "gigaam-e2e-ctc"])
def test_gpu_close_submit_cancel_race_suppresses_inflight_results(
    runtime: EngineRuntime, tmp_path: Path, provider: str
) -> None:
    """Catch late results preventing supervisor requeue or unloading a model during its inference."""
    gate = runtime.batch_gate = threading.Event()
    handle = runtime.open(spec(provider, "gpu"))
    first = _gpu_job(runtime, tmp_path, 1)
    handle.engine.submit(first)
    runtime.batch_calls.get(timeout=WAIT_S)
    engine = handle.engine
    with ThreadPoolExecutor(max_workers=1) as executor:
        closing = executor.submit(engine.close)
        with engine._cond:  # noqa: SLF001 - synchronize close before releasing external inference
            assert engine._cond.wait_for(lambda: engine._closing, timeout=WAIT_S)  # noqa: SLF001 - synchronize close before releasing external inference
        assert runtime.unload_count == 0
        engine.submit(_gpu_job(runtime, tmp_path, 2))
        engine.cancel(first.job_id, first.attempt_id)
        gate.set()
        closing.result(timeout=WAIT_S)
    assert handle.results.empty()
    assert runtime.unload_count == 1


@pytest.mark.integration
@pytest.mark.parametrize(
    ("language", "duration", "decoded_s", "route"),
    [
        ("ru", 30.0, 30.0, "batch"),
        (None, 1.0, 1.0, "sequential"),
        ("ru", 30.001, 1.0, "sequential"),
        ("ru", 1.0, 30.001, "sequential"),
        ("ru", None, 1.0, "batch"),
    ],
)
def test_whisper_routes_by_explicit_language_and_real_clip_length(
    runtime: EngineRuntime,
    tmp_path: Path,
    language: str | None,
    duration: float | None,
    decoded_s: float,
    route: str,
) -> None:
    """Catch auto-language or long clips entering short-clip batching, or the 30 s boundary being excluded."""
    path = wav(tmp_path / "clip.wav", [1000], window_s=1.0)
    runtime.audio[path.name] = np.full(
        round(decoded_s * RATE), 1000 / 32768, dtype=np.float32
    )
    handle = runtime.open(spec("whisper", "gpu"))
    handle.engine.submit(item(path, language=language, duration_s=duration))
    result = handle.result()
    assert result.status == "ok"
    if route == "batch":
        assert result.text == "ru:1000"
        assert runtime.process_calls.empty()
        assert runtime.batch_calls.get(timeout=WAIT_S).language == "ru"
    else:
        assert result.text == f"{language or 'auto'}:clip.wav"
        assert runtime.batch_calls.empty()
        assert runtime.process_calls.get(timeout=WAIT_S).path == path


def test_whisper_vad_empty_skips_every_inference(
    runtime: EngineRuntime, tmp_path: Path
) -> None:
    """Catch silent audio consuming inference capacity or a valid empty transcript being treated as failure."""
    path = tmp_path / "silence.wav"
    runtime.audio[path.name] = np.full(RATE, 1000 / 32768, dtype=np.float32)
    runtime.vad_spans[1000] = []
    handle = runtime.open(spec("whisper", "gpu"))
    handle.engine.submit(item(path))
    result = handle.result()
    assert result.status == "empty"
    assert result.text == ""
    assert runtime.process_calls.empty()
    assert runtime.batch_calls.empty()


def test_whisper_batch_trims_to_first_and_last_speech_boundaries(
    runtime: EngineRuntime, tmp_path: Path
) -> None:
    """Catch silence entering GPU clips or later speech segments being discarded by trimming."""
    path = tmp_path / "speech.wav"
    runtime.audio[path.name] = np.full(10 * RATE, 1000 / 32768, dtype=np.float32)
    runtime.vad_spans[1000] = [
        {"start": RATE, "end": 2 * RATE},
        {"start": 8 * RATE, "end": 9 * RATE},
    ]
    handle = runtime.open(spec("whisper", "gpu"))
    handle.engine.submit(item(path, duration_s=10.0))
    assert handle.result().text == "ru:1000"
    assert runtime.batch_calls.get(timeout=WAIT_S).lengths == [8 * RATE]


@pytest.mark.integration
@pytest.mark.parametrize("failure", ["probe", "decode", "vad"])
def test_whisper_preparation_failure_uses_sequential_error_contract(
    runtime: EngineRuntime, tmp_path: Path, failure: str
) -> None:
    """Catch batch preparation failures bypassing the one-shot path and losing the bot's error detail."""
    path = tmp_path / "clip.wav"
    duration: float | None = 1.0
    if failure == "probe":
        path.write_bytes(b"invalid media")
        duration = None
    runtime.audio[path.name] = np.full(RATE, 1000 / 32768, dtype=np.float32)
    if failure == "decode":
        runtime.audio[path.name] = RuntimeError("decode failed")
    if failure == "vad":
        runtime.vad_spans[1000] = RuntimeError("VAD failed")
    runtime.pipeline_texts[path.name] = RuntimeError(
        "Too long wav file: transcribe_longform"
    )
    handle = runtime.open(spec("whisper", "gpu"))
    handle.engine.submit(item(path, duration_s=duration))
    result = handle.result()
    assert result.status == "error"
    assert result.error_kind == "transcribe_failed"
    assert "Too long wav file" in (result.error_detail or "")
    assert "transcribe_longform" in (result.error_detail or "")
    assert runtime.process_calls.get(timeout=WAIT_S).path == path
    assert runtime.batch_calls.empty()


def _whisper_blocker(
    runtime: EngineRuntime, tmp_path: Path
) -> tuple[EngineHandle, threading.Event]:
    """Hold the single inference thread while subsequent real prep threads become ready."""
    path = tmp_path / "blocker.wav"
    gate = runtime.process_gates[path.name] = threading.Event()
    handle = runtime.open(replace(spec("whisper", "gpu", batch_lanes=4), capacity=4))
    handle.engine.submit(item(path, language=None))
    runtime.process_calls.get(timeout=WAIT_S)
    return handle, gate


def test_whisper_batches_only_matching_languages_and_maps_results_to_jobs(
    runtime: EngineRuntime, tmp_path: Path
) -> None:
    """Catch one language overriding another file or clip texts returning to the wrong attempts."""
    handle, gate = _whisper_blocker(runtime, tmp_path)
    jobs = []
    for seq, language in ((2, "ru"), (3, "en"), (4, "ru")):
        path = tmp_path / f"clip-{seq}.wav"
        runtime.audio[path.name] = np.full(RATE, seq * 1000 / 32768, dtype=np.float32)
        job = item(path, seq=seq, language=language)
        jobs.append(job)
        handle.engine.submit(job)
        handle.wait_ready(job)
    gate.set()
    results = {result.job_id: result for result in (handle.result() for _ in range(4))}
    assert [results[job.job_id].text for job in jobs] == [
        "ru:2000",
        "en:3000",
        "ru:4000",
    ]
    handle.engine.close()
    batches = list(runtime.batch_calls.queue)
    assert [(batch.language, batch.markers) for batch in batches] == [
        ("ru", [2000, 4000]),
        ("en", [3000]),
    ]
    assert len({thread_id for _name, thread_id in runtime.inferences}) == 1


def test_whisper_older_sequential_job_is_not_starved_by_ready_batches(
    runtime: EngineRuntime, tmp_path: Path
) -> None:
    """Catch a batch-first policy repeatedly overtaking an already admitted long sequential job."""
    handle, gate = _whisper_blocker(runtime, tmp_path)
    older = item(tmp_path / "older.wav", seq=2, duration_s=90.0)
    newer = item(tmp_path / "newer.wav", seq=3)
    runtime.audio[newer.input_path.name] = np.full(RATE, 3000 / 32768, dtype=np.float32)
    for job in (older, newer):
        handle.engine.submit(job)
        handle.wait_ready(job)
    gate.set()
    results = [handle.result() for _ in range(3)]
    assert {result.job_id for result in results} == {f"{seq:032x}" for seq in (1, 2, 3)}
    assert [name for name, _thread in runtime.inferences] == [
        "blocker.wav",
        "older.wav",
        "batch:ru",
    ]


@pytest.mark.parametrize(
    ("now", "expected"),
    [(100.0, [3, 4, 1, 5, 2]), (700.0, [1, 4, 5, 2, 3])],
)
def test_engine_admission_priority_aging_and_call_round_robin(
    now: float, expected: list[int]
) -> None:
    """Catch low-priority starvation or a large same-class call monopolizing held work."""
    tickets = {
        1: AdmissionTicket("low", "A", 1, 0.0, 0),
        2: AdmissionTicket("low", "A", 2, 0.0, 1),
        3: AdmissionTicket("high", "A", 3, 100.0, 2),
        4: AdmissionTicket("high", "B", 4, 100.0, 3),
        5: AdmissionTicket("low", "C", 5, 0.0, 4),
    }
    actual = admission_order(
        reversed(tickets), tickets.__getitem__, now=now, aging_s=300.0
    )
    assert actual == expected
    assert (
        admission_order(tickets, tickets.__getitem__, now=now, aging_s=300.0) == actual
    )


def test_engine_lane_dealing_rotates_calls_then_jobs_and_respects_classes() -> None:
    """Catch a multi-file call taking all lanes or a lower class bypassing pending higher work."""
    tickets = {
        "A1": AdmissionTicket("normal", "A", 1, 0.0, 0),
        "A2": AdmissionTicket("normal", "A", 2, 0.0, 1),
        "B1": AdmissionTicket("normal", "B", 3, 0.0, 2),
        "high": AdmissionTicket("high", "H", 4, 0.0, 3),
        "low": AdmissionTicket("low", "L", 5, 0.0, 4),
    }
    remaining = {name: (1 if name == "high" else 4) for name in tickets}

    def take(name: str) -> bool:
        if remaining[name] == 0:
            return False
        remaining[name] -= 1
        return True

    dealt = deal_lanes(
        reversed(tickets), tickets.__getitem__, take, lanes=8, now=0.0, aging_s=300.0
    )
    assert dealt == ["high", "A1", "B1", "A2", "B1", "A1", "B1", "A2"]


@pytest.mark.integration
def test_gigaam_active_calls_share_batch_lanes_in_admission_order(
    runtime: EngineRuntime, tmp_path: Path
) -> None:
    """Catch the real batch loop ignoring call identity or assigning every lane to one call."""
    gate = runtime.batch_gate = threading.Event()
    configuration = replace(spec("gigaam-e2e-ctc", "gpu", batch_lanes=4), capacity=4)
    handle = runtime.open(configuration)
    first = item(wav(tmp_path / "A1.wav", [1000] * 8), call_id="A")
    handle.engine.submit(first)
    runtime.batch_calls.get(timeout=WAIT_S)
    for seq, call in ((2, "A"), (3, "A"), (4, "B")):
        path = wav(tmp_path / f"job-{seq}.wav", [seq * 1000] * 8)
        job = item(path, seq=seq, call_id=call)
        handle.engine.submit(job)
        handle.wait_ready(job)
    gate.set()
    for _ in range(4):
        assert handle.result().status == "ok"
    handle.engine.close()
    batches = list(runtime.batch_calls.queue)
    assert batches[0].markers == [1000, 4000, 2000, 4000]
    assert batches[1].markers == [3000, 4000, 1000, 4000]
    assert batches[2].markers == [2000, 4000, 3000, 4000]
    assert all(len(batch.markers) <= 4 for batch in batches)
    assert len({batch.thread_id for batch in batches}) == 1


def test_whisper_higher_class_keeps_admission_before_lower_language_match(
    runtime: EngineRuntime, tmp_path: Path
) -> None:
    """Catch a lower class filling a language batch ahead of higher-class work."""
    handle, gate = _whisper_blocker(runtime, tmp_path)
    for seq, language, priority in (
        (2, "ru", "high"),
        (3, "en", "high"),
        (4, "ru", "low"),
    ):
        path = tmp_path / f"clip-{seq}.wav"
        runtime.audio[path.name] = np.full(RATE, seq * 1000 / 32768, dtype=np.float32)
        job = item(
            path,
            seq=seq,
            language=language,
            priority=priority,
            call_id=f"class-{priority}",
        )
        handle.engine.submit(job)
        handle.wait_ready(job)
    gate.set()
    results = [handle.result() for _ in range(4)]
    assert all(result.status == "ok" for result in results)
    handle.engine.close()
    batches = list(runtime.batch_calls.queue)
    assert [(batch.language, batch.markers) for batch in batches] == [
        ("ru", [2000]),
        ("en", [3000]),
        ("ru", [4000]),
    ]
