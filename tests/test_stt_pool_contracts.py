"""Pool request contracts, filesystem transactions and configuration selection."""

from __future__ import annotations

import io
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from core.stt_errors import CudaOutOfMemoryError, DeviceUnavailableError
from core.stt_options import DiarizeOptions, MediaWindow, WhisperDecodeOptions
from core.stt_pool import fsio, ipc
from core.stt_pool import queue as pool_queue
from core.stt_pool.config import (
    PoolConfigError,
    apply_overrides,
    load_pool_config,
    parse_pool_config,
)
from core.stt_pool.engine_api import EngineResult, WorkerSpec, WorkItem
from core.stt_pool.keys import (
    DIARIZE_PROVIDER,
    WorkerKey,
    make_key,
    normalize_device_request,
    normalize_language,
    normalize_priority,
)
from core.stt_pool.queue import JobRecord, JobValidationError, QueuePaths, ResultRecord

if TYPE_CHECKING:
    from collections.abc import Mapping

_GIB = 1024**3
_JOB_ID = "a" * 32
_CALL_ID = "b" * 32
# * A generous test budget detects unbounded retry loops without real waits.
_RETRY_TEST_BUDGET = 32


@pytest.mark.integration
@pytest.mark.parametrize(
    ("document", "status"),
    [("{not json", None), ('{"job_id":"other","status":"unrecognized"}', "error")],
)
def test_result_reader_tolerates_corruption_and_unknown_status_without_wrong_job(
    tmp_path: Path, document: str, status: str | None
) -> None:
    """Catch corrupt results crashing clients or false success."""
    paths = QueuePaths(tmp_path)
    paths.ensure()
    paths.result_file(_JOB_ID).write_text(document, encoding="utf-8")
    result = pool_queue.read_result(paths, _JOB_ID)
    if status is None:
        assert result is None
    else:
        assert result is not None
        assert result.status == status
        assert result.job_id == _JOB_ID


def _payload(tmp_path: Path) -> dict[str, object]:
    """Return a documented v1 job document with a temporary absolute media path."""
    return {
        "schema": "askvlm.stt.job",
        "version": 1,
        "kind": "stt",
        "job_id": _JOB_ID,
        "call_id": _CALL_ID,
        "call_index": 0,
        "max_parallel": 2,
        "submitted_at": 1000.0,
        "deadline_at": 1300.0,
        "input_path": str(tmp_path / "speech.wav"),
        "duration_s": 60.0,
        "request": {
            "provider": "whisper",
            "whisper_model": "Org/CaseSensitive",
            "compute_type": "auto",
            "device": "auto",
        },
        "options": {"language": "RU"},
        "priority": "high",
        "client": {"pid": 123},
    }


@pytest.mark.parametrize(
    ("backend", "compute", "expected"),
    [
        ("cpu", "auto", "int8"),
        ("cpu", "float16", "int8"),
        ("cpu", "int8_float16", "int8"),
        ("cpu", "float32", "float32"),
        ("gpu", "auto", "float16"),
        ("gpu", "int8_float16", "int8_float16"),
    ],
)
def test_whisper_keys_resolve_compute_without_losing_model_identity(
    backend: str, compute: str, expected: str
) -> None:
    """Catch wrong backend compute selection or case-folding a custom model path."""
    key = make_key("whisper", " Org/CaseSensitive ", compute, backend)
    assert key.model == "Org/CaseSensitive"
    assert key.compute_type == expected
    assert WorkerKey.from_dict(key.to_dict()) == key


@pytest.mark.parametrize(
    ("provider", "identity"),
    [
        ("gigaam-ctc", "ai-sage/GigaAM-Multilingual@ctc"),
        ("gigaam-e2e-ctc", "ai-sage/GigaAM-v3@e2e_ctc"),
    ],
)
def test_gigaam_keys_serve_the_pinned_head_and_ignore_whisper_options(
    provider: str, identity: str
) -> None:
    """Catch workers sharing a wrong GigaAM head or compute type."""
    key = make_key(provider, "irrelevant", "int8", "cpu")
    assert key.model == identity
    assert key.compute_type == "fp32"
    assert key == make_key(provider, "different", "float16", "cpu")


@pytest.mark.parametrize(
    ("provider", "raw", "expected"),
    [
        ("whisper", " AUTO ", "auto"),
        ("whisper", "cpu", "cpu"),
        ("whisper", "cuda", "cuda"),
        ("gigaam-ctc", "auto", "cpu"),
        ("gigaam-e2e-ctc", "auto", "auto"),
        ("sherpa-diarize", "auto", "cpu"),
        ("sherpa-diarize", "cpu", "cpu"),
    ],
)
def test_device_requests_preserve_the_requested_device_class(
    provider: str, raw: str, expected: str
) -> None:
    """Catch auto being coerced to CPU for a CUDA-capable provider."""
    assert normalize_device_request(provider, raw) == expected


@pytest.mark.parametrize(
    ("provider", "device", "rejected"),
    [
        ("gigaam-ctc", "cuda", "cuda"),
        ("whisper", "gpu", "gpu"),
        ("missing", "cpu", "missing"),
        ("sherpa-diarize", "cuda", "cuda"),
    ],
)
def test_unsupported_provider_device_pairs_fail(
    provider: str, device: str, rejected: str
) -> None:
    """Catch silent model or device substitution for an unsupported request."""
    with pytest.raises(ValueError) as error:  # noqa: PT011 - validate type and the rejected value, not human wording
        normalize_device_request(provider, device)
    assert rejected in str(error.value)


@pytest.mark.parametrize(
    ("raw", "expected"), [(None, None), ("", None), (" RU ", "ru"), ("EN-us", "en-us")]
)
def test_language_normalization_is_per_job(
    raw: str | None, expected: str | None
) -> None:
    """Catch language hints changing workers or leaking between jobs."""
    assert normalize_language(raw) == expected


@pytest.mark.parametrize("raw", ["../ru", "r", "ru\x00", "en-US-extra"])
def test_invalid_language_tokens_fail(raw: str) -> None:
    """Catch unsafe or unsupported language values reaching the inference API."""
    with pytest.raises(ValueError) as error:  # noqa: PT011 - validate the rejected token without cementing prose
        normalize_language(raw)
    assert repr(raw) in str(error.value)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "normal"),
        ("", "normal"),
        (" HIGH ", "high"),
        ("normal", "normal"),
        ("low", "low"),
    ],
)
def test_priority_tokens_select_the_intended_class(raw: object, expected: str) -> None:
    """Catch caller priorities being assigned to the wrong admission class."""
    assert normalize_priority(raw) == expected


@pytest.mark.parametrize("raw", [False, 1, [], {}, "urgent"])
def test_unknown_priority_values_fail(raw: object) -> None:
    """Catch malformed priorities silently elevating or demoting a call."""
    with pytest.raises(ValueError) as error:  # noqa: PT011 - validate the rejected value without cementing prose
        normalize_priority(raw)
    assert repr(raw) in str(error.value)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("cpu.max_slots", True),
        ("cpu.max_slots", -1),
        ("cpu.threads_per_slot", 0),
        ("gpu.max_workers", 1.5),
        ("gpu.batch_lanes.whisper", 65),
        *[
            (f"gpu.{key}", value)
            for key in ("cufft_plan_cache_max_size", "cudnn_v8_lru_cache_limit")
            for value in (True, -1, 1.5, None, "32", 2**31, float("inf"), 10**400)
        ],
        *[
            ("gpu.idle_cache_release_s", value)
            for value in (
                True,
                -0.01,
                None,
                "15",
                float("nan"),
                float("inf"),
                86400.01,
                10**400,
            )
        ],
        *[
            (path, value)
            for path in ("gpu.vram_cap", "memory.live_cap_raise")
            for value in (0, 1, 0.0, None, "true", [], {})
        ],
        *[
            ("gpu.vram_cap_margin_gib", value)
            for value in (True, -0.01, None, "1", float("nan"), float("inf"), 10**400)
        ],
        *[
            ("memory.recycle_commit_fraction", value)
            for value in (
                True,
                -0.01,
                1.0001,
                None,
                "1",
                float("nan"),
                float("inf"),
                10**400,
            )
        ],
        *[
            ("lifecycle.worker_max_uptime_h", value)
            for value in (
                True,
                -1,
                0.001,
                0.9999,
                8760.01,
                None,
                "12",
                float("nan"),
                float("inf"),
                10**400,
            )
        ],
        ("memory.commit_budget_gib", float("nan")),
        ("memory.commit_budget_gib", 0),
        ("memory.free_commit_guard_gib", -1),
        ("memory.child_headroom_gib", -0.5),
        ("memory.child_headroom_gib", True),
        ("memory.child_headroom_gib", None),
        ("memory.child_headroom_gib", "1"),
        ("memory.child_headroom_gib", float("nan")),
        ("memory.child_headroom_gib", float("inf")),
        ("memory.child_headroom_gib", 10**400),
        ("memory.child_headroom_typo", 1),
        ("memory.live_cap_raise_typo", True),
        ("gpu.vram_cap_typo", True),
        ("lifecycle.worker_max_uptime_typo", 12),
        ("vram.budget_gib", 10**400),
        ("threads.budget", 0),
        ("diarization.threads_per_slot", 0),
        ("diarization.threads_per_slot", True),
        ("diarization.threads_per_slot", 1.5),
        ("diarization.threads_per_slot", None),
        ("diarization.threads_per_slot", "4"),
        ("diarization.unknown", 1),
        ("scheduling.spill_ratio", 1.1),
        ("scheduling.priority_aging_s", 0),
        ("lifecycle.stop_grace_s", float("inf")),
        ("gpu.unknown", 1),
        ("unknown", {}),
    ],
)
def test_invalid_configuration_reports_the_offending_json_path(
    path: str, value: object
) -> None:
    """Catch invalid limits reaching admission control."""
    payload: dict[str, object] = {}
    section = payload
    parts = path.split(".")
    for part in parts[:-1]:
        child: dict[str, object] = {}
        section[part] = child
        section = child
    section[parts[-1]] = value
    with pytest.raises(PoolConfigError) as exc_info:
        parse_pool_config(payload, source="test-config", logical_cpus=32)
    assert exc_info.value.json_path == path


@pytest.mark.integration
def test_configuration_resolution_and_overrides_use_real_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch wrong config precedence or overrides mutating the source."""
    explicit = tmp_path / "explicit.json"
    environment = tmp_path / "environment.json"
    project = tmp_path / "stt_pool.json"
    for file, slots in ((explicit, 3), (environment, 4), (project, 5)):
        file.write_text(json.dumps({"cpu": {"max_slots": slots}}), encoding="utf-8-sig")
    monkeypatch.setenv("ASKVLM_STT_POOL_CONFIG", str(environment))
    assert load_pool_config(explicit, root=tmp_path).cpu.max_slots == 3
    assert load_pool_config(root=tmp_path).cpu.max_slots == 4
    monkeypatch.delenv("ASKVLM_STT_POOL_CONFIG")
    selected = load_pool_config(root=tmp_path)
    assert selected.cpu.max_slots == 5
    assert apply_overrides(selected, cpu_slots=7).cpu.max_slots_per_worker == 7
    project.unlink()
    conservative = load_pool_config(root=tmp_path, logical_cpus=32)
    assert conservative.cpu.max_slots <= 2
    assert conservative.gpu.max_workers <= 1
    with pytest.raises(PoolConfigError):
        load_pool_config(explicit.with_name("missing.json"), root=tmp_path)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("fraction", "enabled", "margin_gib", "hours"),
    [(0, False, 0, 0), (0.5, True, 0.125, 1), (1, True, 1, 8760)],
)
def test_memory_policy_file_accepts_boundaries_and_converts_units(
    tmp_path: Path, fraction: float, *, enabled: bool, margin_gib: float, hours: float
) -> None:
    """Catch rejected disable switches, missing policy keys or GiB/hour unit errors."""
    path = tmp_path / "limits.json"
    path.write_text(
        json.dumps(
            {
                "memory": {
                    "recycle_commit_fraction": fraction,
                    "live_cap_raise": enabled,
                },
                "gpu": {"vram_cap": enabled, "vram_cap_margin_gib": margin_gib},
                "lifecycle": {"worker_max_uptime_h": hours},
            }
        ),
        encoding="utf-8",
    )
    loaded = load_pool_config(path, root=tmp_path, env={}, logical_cpus=8)
    assert loaded.memory.recycle_commit_fraction == fraction
    assert loaded.memory.live_cap_raise is enabled
    assert loaded.gpu.vram_cap is enabled
    assert loaded.gpu.vram_cap_margin_bytes == round(margin_gib * _GIB)
    assert loaded.lifecycle.worker_max_uptime_s == hours * 3600


def test_example_configuration_honors_architect_safety_budgets() -> None:
    """Catch unsafe commit, CPU or GPU example limits."""
    config = load_pool_config(
        Path(__file__).resolve().parents[1] / "stt_pool.example.json", env={}
    )
    assert config.cpu.max_slots == 8
    assert config.gpu.max_workers == 2
    assert config.memory.commit_budget_bytes == 24 * _GIB
    assert config.vram.budget_bytes == 7 * _GIB


@pytest.mark.parametrize(
    ("field", "value", "error_kind"),
    [
        ("schema", "other", "input_invalid"),
        ("version", 3, "unsupported_version"),
        ("version", True, "input_invalid"),
        ("kind", "diarization", "unsupported_kind"),
        ("job_id", "../outside", "input_invalid"),
        ("job_id", "c" * 32, "input_invalid"),
        ("call_id", "BAD", "input_invalid"),
        ("call_index", -1, "input_invalid"),
        ("max_parallel", 0, "input_invalid"),
        ("max_parallel", 65, "input_invalid"),
        ("max_parallel", True, "input_invalid"),
        ("input_path", "relative.wav", "input_invalid"),
        ("input_path", "C:\\bad\x00.wav", "input_invalid"),
        ("duration_s", -1, "input_invalid"),
        ("duration_s", float("inf"), "input_invalid"),
        ("submitted_at", 10**400, "input_invalid"),
        ("priority", "urgent", "input_invalid"),
        ("priority", [], "input_invalid"),
        ("request", {"provider": "gigaam-ctc", "device": "cuda"}, "input_invalid"),
    ],
)
def test_invalid_job_documents_fail_before_admission(
    tmp_path: Path, field: str, value: object, error_kind: str
) -> None:
    """Catch malformed jobs reaching workers or losing error codes."""
    payload = _payload(tmp_path)
    payload[field] = value
    with pytest.raises(JobValidationError) as exc_info:
        JobRecord.from_dict(payload, expected_job_id=_JOB_ID)
    assert exc_info.value.error_kind == error_kind


@pytest.mark.integration
def test_queue_round_trip_attempts_and_terminal_result_are_isolated_from_v1(
    tmp_path: Path,
) -> None:
    """Catch mixed v1/v2 queues, wrong attempts or lost provenance."""
    paths = QueuePaths(tmp_path)
    paths.ensure()
    payload = _payload(tmp_path)
    payload["future_extension"] = {"engine_factory": "must-not-be-used"}
    job = JobRecord.from_dict(payload, expected_job_id=_JOB_ID)
    pool_queue.submit_job(paths, job)
    loaded = pool_queue.read_job_payload(paths, _JOB_ID)
    assert loaded is not None
    assert not (tmp_path / "incoming").exists()
    admitted = JobRecord.from_dict(loaded, expected_job_id=_JOB_ID)
    assert admitted.request.whisper_model == "Org/CaseSensitive"
    assert admitted.language == "ru"
    assert admitted.priority == "high"
    pool_queue.write_attempts(paths, _JOB_ID, 2)
    assert pool_queue.read_attempts(paths, _JOB_ID) == 2
    pool_queue.request_cancel(paths, _JOB_ID)
    detail = "RuntimeError: Too long wav file " + "x" * 5000
    result = ResultRecord(
        _JOB_ID, "error", error_kind="transcribe_failed", error_detail=detail
    )
    assert pool_queue.write_result(paths, result)
    original = paths.result_file(_JOB_ID).read_bytes()
    assert not pool_queue.write_result(paths, ResultRecord(_JOB_ID, "ok", text="late"))
    assert paths.result_file(_JOB_ID).read_bytes() == original
    received = pool_queue.read_result(paths, _JOB_ID)
    assert received is not None
    assert received.error_detail is not None
    assert "Too long wav file" in received.error_detail
    assert len(received.error_detail) == 4000
    pool_queue.delete_job_files(paths, _JOB_ID)
    assert pool_queue.has_result(paths, _JOB_ID)
    assert not paths.incoming_file(_JOB_ID).exists()
    pool_queue.consume_result(paths, _JOB_ID)
    assert not pool_queue.has_result(paths, _JOB_ID)
    assert not pool_queue.is_cancel_requested(paths, _JOB_ID)


@pytest.mark.integration
@pytest.mark.parametrize("hard_links", [True, False])
def test_concurrent_terminal_publication_has_exactly_one_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, hard_links: bool
) -> None:
    """Catch simultaneous late replies overwriting a completed transcript."""
    if not hard_links:
        contenders = threading.Barrier(4)

        def unsupported_link(_source: object, _destination: object) -> None:
            # * All first-wave writers reach the filesystem before any can publish.
            contenders.wait(timeout=5)
            msg = "test filesystem does not support hard links"
            raise OSError(msg)

        monkeypatch.setattr(os, "link", unsupported_link)
    paths = QueuePaths(tmp_path)
    results = tuple(ResultRecord(_JOB_ID, "ok", text=f"attempt-{i}") for i in range(12))
    with ThreadPoolExecutor(max_workers=4) as executor:
        published = tuple(
            executor.map(lambda result: pool_queue.write_result(paths, result), results)
        )
    assert sum(published) == 1
    winning = results[published.index(True)]
    received = pool_queue.read_result(paths, _JOB_ID)
    assert received is not None
    assert received.text == winning.text
    assert not tuple(paths.results.glob("*.tmp"))


@pytest.mark.integration
def test_purge_reclaims_only_expired_results_and_orphaned_files(tmp_path: Path) -> None:
    """Catch purge deleting active jobs or retaining stale artifacts."""
    paths = QueuePaths(tmp_path)
    paths.ensure()
    active = JobRecord.from_dict(_payload(tmp_path), expected_job_id=_JOB_ID)
    pool_queue.submit_job(paths, active)
    pool_queue.request_cancel(paths, _JOB_ID)
    pool_queue.write_attempts(paths, _JOB_ID, 1)
    expired_id = "c" * 32
    orphan_id = "d" * 32
    pool_queue.write_result(paths, ResultRecord(expired_id, "empty"))
    pool_queue.request_cancel(paths, expired_id)
    pool_queue.write_attempts(paths, expired_id, 1)
    pool_queue.request_cancel(paths, orphan_id)
    pool_queue.write_attempts(paths, orphan_id, 1)
    stale = (
        tuple(paths.cancel.iterdir())
        + tuple(paths.attempts.iterdir())
        + (paths.result_file(expired_id),)
    )
    for path in stale:
        os.utime(path, (100, 100))
    fresh = "e" * 32
    pool_queue.write_result(paths, ResultRecord(fresh, "empty"))
    purged = pool_queue.purge_stale(paths, now=10000, ttl_s=3600)
    assert purged == [expired_id, orphan_id]
    assert pool_queue.is_cancel_requested(paths, _JOB_ID)
    assert pool_queue.read_attempts(paths, _JOB_ID) == 1
    assert pool_queue.has_result(paths, fresh)
    assert not paths.attempts_file(expired_id).exists()


@pytest.mark.integration
@pytest.mark.parametrize("winerror", [5, 32])
def test_atomic_write_retries_windows_sharing_without_partial_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, winerror: int
) -> None:
    """Catch lost or partial writes after Windows sharing failures."""
    target = tmp_path / "record.json"
    target.write_text('{"old":true}', encoding="utf-8")
    original_replace = os.replace
    attempts: list[bytes] = []
    delays: list[float] = []

    def replace_at_os_boundary(src: object, dst: object) -> None:
        attempts.append(target.read_bytes())
        if len(attempts) <= 2:
            error = OSError("sharing violation")
            error.winerror = winerror
            raise error
        original_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace_at_os_boundary)
    monkeypatch.setattr(fsio, "sleep", delays.append)
    fsio.atomic_write_json(target, {"transcript": "русский\ntext\ud800"})
    assert len(attempts) == 3
    assert len(delays) == 2
    assert all(json.loads(data) == {"old": True} for data in attempts)
    assert fsio.read_json(target) == {"transcript": "русский\ntext\ud800"}
    assert not tuple(tmp_path.glob("*.tmp"))


@pytest.mark.integration
@pytest.mark.parametrize("winerror", [32, 123, None])
def test_atomic_write_propagates_permanent_failure_and_cleans_temp_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    winerror: int | None,
) -> None:
    """Catch infinite retries or loss of an existing document after an OS failure."""
    target = tmp_path / "record.json"
    target.write_text('{"old":true}', encoding="utf-8")
    attempted: list[object] = []

    def failed_replace(src: object, dst: object) -> None:
        attempted.append(src)
        if len(attempted) > _RETRY_TEST_BUDGET:
            pytest.fail("atomic writes must stop retrying a persistent OS failure")
        error = OSError("cannot replace")
        if winerror is not None:
            error.winerror = winerror
        raise error

    monkeypatch.setattr(os, "replace", failed_replace)
    monkeypatch.setattr(fsio, "sleep", lambda _delay: None)
    with pytest.raises(OSError) as error:  # noqa: PT011 - OS status is the contract, not exception wording
        fsio.atomic_write_json(target, {"new": True})
    if winerror is not None:
        assert error.value.winerror == winerror
    if winerror == 32:
        assert 1 < len(attempted) <= _RETRY_TEST_BUDGET
    else:
        assert len(attempted) == 1
    assert fsio.read_json(target) == {"old": True}
    assert not tuple(tmp_path.glob("*.tmp"))


@pytest.mark.integration
def test_heartbeat_freshness_and_clear_use_the_injected_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch stale heartbeats or exit retaining false supervisor liveness."""
    paths = QueuePaths(tmp_path)
    monkeypatch.setattr(pool_queue, "time", lambda: 1000.0)
    pool_queue.write_heartbeat(paths, {"state": "running"})
    assert pool_queue.is_supervisor_alive(paths, now=1020)
    assert not pool_queue.is_supervisor_alive(paths, now=1020.01)
    pool_queue.clear_heartbeat(paths)
    assert not pool_queue.is_supervisor_alive(paths, now=1001)


@pytest.mark.parametrize(
    "line", ["", "[]", "garbage", '{"type":"unknown"}', '{"text":"missing-type"}']
)
def test_ipc_ignores_non_protocol_lines(line: str) -> None:
    """Catch diagnostic output being accepted as a protocol command."""
    assert ipc.decode_message(line) is None


def test_ipc_framing_survives_unicode_and_an_oversized_line() -> None:
    """Catch Unicode loss or oversized lines desynchronizing the protocol."""
    message = {"type": "result", "text": "русский\ntext\ud800"}
    line = ipc.encode_message(message)
    assert line.count("\n") == 1
    assert ipc.decode_message(line.encode("utf-8").decode("utf-8")) == message
    stream = io.BytesIO(b"x" * 100 + b'\n{"type":"bye"}\n')
    lines = list(ipc.iter_lines(stream, max_bytes=20))
    assert lines[0] is None
    assert ipc.decode_message(cast("str", lines[1])) == {"type": "bye"}


def _spec() -> dict[str, object]:
    """Build a protocol spec for the ML-free parser."""
    return {
        "type": "spec",
        "protocol": 2,
        "worker_id": "w",
        "instance_id": "s",
        "key": make_key("whisper", "small", "auto", "cpu").to_dict(),
        "capacity": 2,
        "threads_per_slot": 2,
        "host_threads": 0,
        "device": "cpu",
        "options": {},
        "engine_factory": "tests.stt_pool_fake_worker:create_engine",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("device", "cuda"),
        ("device", "auto"),
        ("capacity", True),
        ("capacity", 0),
        ("protocol", 3),
        ("options", []),
    ],
)
def test_worker_spec_rejects_device_mismatch_and_malformed_limits(
    field: str, value: object
) -> None:
    """Catch a worker running on an unreserved backend or accepting invalid capacity."""
    message = _spec()
    message[field] = value
    with pytest.raises(ValueError) as error:  # noqa: PT011 - structured field names are checked without prose matching
        WorkerSpec.from_message(message)
    assert field in str(error.value)


@pytest.mark.parametrize(
    ("exception", "error_kind"),
    [
        (MemoryError("allocation"), "memory_limit"),
        (CudaOutOfMemoryError("CUDA allocation"), "cuda_oom"),
        (DeviceUnavailableError("CUDA absent"), "device_unavailable"),
        (RuntimeError("Too long wav file"), "transcribe_failed"),
    ],
)
def test_engine_errors_keep_attempt_provenance_and_classification(
    tmp_path: Path, exception: BaseException, error_kind: str
) -> None:
    """Catch engine errors losing attempt identity or typed kinds."""
    message = {
        "type": "job",
        "job_id": _JOB_ID,
        "attempt_id": f"{_JOB_ID}:2",
        "input_path": str(tmp_path / "clip.wav"),
        "language": "RU",
        "priority": "high",
        "call_id": _CALL_ID,
        "seq": 7,
    }
    item = WorkItem.from_message(message)
    assert item.language == "ru"
    result = EngineResult.for_error(item, exception, backend="cpu", run_s=1)
    wire = ipc.decode_message(ipc.encode_message(result.to_message()))
    assert wire is not None
    received = EngineResult.from_message(wire)
    assert received.attempt_id == message["attempt_id"]
    assert received.error_kind == error_kind
    assert received.error_detail is not None
    assert str(exception) in received.error_detail
    assert received.status == "error"


@pytest.mark.parametrize("raw", [float("nan"), -1, True, 10**400])
def test_worker_job_rejects_invalid_duration(tmp_path: Path, raw: object) -> None:
    """Catch NaN or overflow propagating from the wire into scheduling arithmetic."""
    message: Mapping[str, object] = {
        "type": "job",
        "job_id": _JOB_ID,
        "attempt_id": "attempt",
        "input_path": str(tmp_path / "clip.wav"),
        "duration_s": raw,
    }
    with pytest.raises(ValueError) as error:  # noqa: PT011 - check the malformed field, not human wording
        WorkItem.from_message(message)
    assert "duration_s" in str(error.value)


@pytest.mark.parametrize("budget", [1, 2, 4, 8])
def test_diarization_default_threads_fit_small_budgets_and_overrides(
    budget: int,
) -> None:
    """Catch default diarization becoming impossible on small machines or stale after an override."""
    config = parse_pool_config(
        {"threads": {"budget": budget}}, source="test-config", logical_cpus=8
    )
    assert config.diarization.threads_per_slot == min(4, budget)
    adjusted = apply_overrides(config, thread_budget=3)
    assert adjusted.diarization.threads_per_slot == 3
    assert config.threads.budget == budget


@pytest.mark.parametrize("override", [False, True])
def test_explicit_diarization_threads_cannot_exceed_the_effective_budget(
    *, override: bool
) -> None:
    """Catch an explicit unschedulable thread setting being silently clamped instead of rejected."""
    payload: dict[str, object] = {
        "threads": {"budget": 8 if override else 2},
        "diarization": {"threads_per_slot": 3},
    }
    with pytest.raises(PoolConfigError) as error:  # noqa: PT012 - the parse and the override both raise the asserted error
        config = parse_pool_config(payload, source="test-config", logical_cpus=8)
        if override:
            apply_overrides(config, thread_budget=2)
    assert error.value.json_path == "diarization.threads_per_slot"


@pytest.mark.integration
def test_plain_job_document_preserves_the_v1_machine_contract(tmp_path: Path) -> None:
    """Catch ordinary jobs unnecessarily upgrading their schema or changing the legacy wire layout."""
    payload = _payload(tmp_path)
    payload["options"] = {"language": "ru"}
    job = JobRecord.from_dict(payload, expected_job_id=_JOB_ID)
    assert job.to_dict() == payload
    assert list(job.to_dict()) == list(payload)
    paths = QueuePaths(tmp_path / "queue")
    pool_queue.submit_job(paths, job)
    expected = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    assert paths.incoming_file(_JOB_ID).read_bytes() == expected


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["stt", "diarize"])
def test_v2_jobs_round_trip_options_through_real_queue_files(
    tmp_path: Path, kind: str
) -> None:
    """Catch lost window/decode/clustering fields or v2 jobs being written as legacy requests."""
    payload = _payload(tmp_path)
    payload["version"] = 2
    payload["kind"] = kind
    options: dict[str, object] = {"window": {"start_s": 1.25, "duration_s": 2.5}}
    if kind == "stt":
        options.update(
            language="ru", segments=True, decode={"beam_size": 1, "vad_filter": False}
        )
    else:
        payload["request"] = {"provider": "sherpa-diarize", "device": "auto"}
        options["diarize"] = {
            "num_speakers": 2,
            "threshold": 0.7,
            "min_duration_on": 0,
            "min_duration_off": 0.2,
        }
    payload["options"] = options
    paths = QueuePaths(tmp_path / "queue")
    pool_queue.submit_job(paths, JobRecord.from_dict(payload, expected_job_id=_JOB_ID))
    received = pool_queue.read_job_payload(paths, _JOB_ID)
    assert received is not None
    assert received["version"] == 2
    job = JobRecord.from_dict(received, expected_job_id=_JOB_ID)
    assert job.window == MediaWindow(1.25, 2.5)
    assert job.kind == kind
    if kind == "stt":
        assert job.want_segments
        assert job.decode == WhisperDecodeOptions(beam_size=1, vad_filter=False)
    else:
        assert job.request.provider == DIARIZE_PROVIDER
        assert job.request.device == "cpu"
        assert job.diarize == DiarizeOptions(2, 0.7, 0, 0.2)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("unknown", 1),
        ("segments", 1),
        ("decode", []),
        ("decode", {"beam_size": True}),
        ("decode", {"vad_filter": 0}),
        ("decode", {"temperature": 0}),
        ("window", {"start_s": -1}),
        ("window", {"duration_s": 0}),
        ("window", {"start_s": float("nan")}),
        ("window", {"end_s": 1}),
        ("diarize", {"num_speakers": 0}),
        ("diarize", {"threshold": float("inf")}),
    ],
)
def test_v2_job_options_are_strictly_validated_before_admission(
    tmp_path: Path, option: str, value: object
) -> None:
    """Catch malformed or unknown structured options being silently dropped by admission."""
    payload = _payload(tmp_path)
    payload["version"] = 2
    payload["options"] = {option: value}
    with pytest.raises(JobValidationError) as error:
        JobRecord.from_dict(payload, expected_job_id=_JOB_ID)
    assert error.value.error_kind == "input_invalid"
    assert option in str(error.value)


@pytest.mark.parametrize(
    ("provider", "kind", "options", "device"),
    [
        ("gigaam-ctc", "stt", {"decode": {"beam_size": 10}}, "cpu"),
        ("gigaam-e2e-ctc", "stt", {"decode": {"vad_filter": True}}, "auto"),
        ("whisper", "stt", {"diarize": {}}, "cpu"),
        ("whisper", "diarize", {}, "cpu"),
        ("sherpa-diarize", "stt", {}, "cpu"),
        ("sherpa-diarize", "diarize", {}, "cuda"),
        ("sherpa-diarize", "diarize", {"language": "ru"}, "cpu"),
        ("sherpa-diarize", "diarize", {"segments": True}, "cpu"),
        ("sherpa-diarize", "diarize", {"decode": {"word_timestamps": False}}, "cpu"),
    ],
)
def test_job_kind_provider_and_explicit_options_cannot_conflict(
    tmp_path: Path, provider: str, kind: str, options: dict[str, object], device: str
) -> None:
    """Catch CPU-only diarization on CUDA or explicit Whisper options silently accepted by GigaAM."""
    payload = _payload(tmp_path)
    payload.update(version=2, kind=kind, options=options)
    payload["request"] = {"provider": provider, "device": device}
    with pytest.raises(JobValidationError) as error:
        JobRecord.from_dict(payload, expected_job_id=_JOB_ID)
    assert error.value.error_kind == "input_invalid"


@pytest.mark.parametrize("kind", ["stt", "diarize"])
@pytest.mark.parametrize(
    "memory",
    [
        None,
        [],
        "unavailable",
        42,
        False,
        {
            "private_bytes": -1,
            "cuda_reserved_bytes": True,
            "cuda_allocated_bytes": 0,
            "cuda_max_reserved_bytes": "16384",
            "cufft_plan_cache_size": 1.5,
            "unknown": 99,
        },
    ],
)
def test_worker_packets_preserve_options_and_empty_structured_success(
    tmp_path: Path, kind: str, memory: object
) -> None:
    """Catch broken telemetry invalidating empty successes or dropping their data."""
    message: dict[str, object] = {
        "type": "job",
        "job_id": _JOB_ID,
        "attempt_id": f"{_JOB_ID}:1",
        "input_path": str(tmp_path / "a.wav"),
        "kind": kind,
        "window": {"start_s": 10.5, "duration_s": 2.0},
    }
    if kind == "stt":
        message.update(segments=True, decode={"condition_on_previous_text": False})
    else:
        message["diarize"] = {"num_speakers": 2, "threshold": 0.7}
    item = WorkItem.from_message(message)
    wire = ipc.decode_message(ipc.encode_message(item.to_message()))
    assert wire is not None
    received = WorkItem.from_message(wire)
    assert received.window == MediaWindow(10.5, 2.0)
    if kind == "stt":
        assert received.decode.condition_on_previous_text is False
        result = EngineResult.for_text(
            received, "", backend="cpu", run_s=0, audio_s=2, data={"segments": []}
        )
        expected_status = "empty"
    else:
        assert received.diarize_options.num_speakers == 2
        result = EngineResult.for_diarization(
            received, {"turns": []}, backend="cpu", run_s=0, audio_s=2
        )
        expected_status = "ok"
    result_message = result.to_message()
    if memory is not None:
        result_message["memory"] = memory
    result_wire = ipc.decode_message(ipc.encode_message(result_message))
    assert result_wire is not None
    decoded = EngineResult.from_message(result_wire)
    assert decoded.status == expected_status
    assert decoded.text == ""
    assert decoded.data == ({"segments": []} if kind == "stt" else {"turns": []})
    if isinstance(memory, dict):
        assert decoded.memory == {
            "private_bytes": None,
            "cuda_reserved_bytes": None,
            "cuda_allocated_bytes": 0,
            "cuda_max_reserved_bytes": None,
            "cufft_plan_cache_size": None,
        }
    else:
        assert decoded.memory is None


@pytest.mark.parametrize("data", [[], "{}", False])
def test_malformed_structured_result_is_an_explicit_protocol_error(
    data: object,
) -> None:
    """Catch non-object result data being dropped while clients still see a successful transcript."""
    result = ResultRecord.from_dict(
        {"job_id": _JOB_ID, "status": "ok", "text": "speech", "data": data}
    )
    assert result.status == "error"
    assert result.error_kind == "protocol_error"
    assert result.text == ""
    assert result.data is None
