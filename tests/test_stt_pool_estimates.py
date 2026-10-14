"""Learned estimates exercise real persistence and benchmark-backed memory floors."""

from __future__ import annotations

import copy
import json
import logging
import math
from typing import TYPE_CHECKING, cast

import pytest

from core.stt_pool.estimates import EstimateStore, seed_estimate
from core.stt_pool.keys import make_key

if TYPE_CHECKING:
    from pathlib import Path

_GIB = 1024**3
_FINGERPRINT: dict[str, object] = {
    "cpu": "test",
    "logical_cpus": 8,
    "ram_gib": 32,
    "gpu": None,
}


@pytest.mark.parametrize(
    ("provider", "backend", "base_gib", "slot_gib", "xrt"),
    [
        ("whisper", "cpu", 1.8, 0.25, 4.5),
        ("whisper", "gpu", 3.2, 0.1, 20.0),
        ("gigaam-ctc", "cpu", 4.0, 0.15, 16.7),
        ("gigaam-e2e-ctc", "cpu", 4.5, 0.15, 16.7),
        ("gigaam-e2e-ctc", "gpu", 6.6, 0.05, 68.0),
    ],
)
def test_seed_memory_and_throughput_match_the_measured_baselines(
    provider: str, backend: str, base_gib: float, slot_gib: float, xrt: float
) -> None:
    """Catch understated memory reservations or ETA seeds swapped between providers."""
    estimate = seed_estimate(make_key(provider, "small", "auto", backend))
    assert estimate.commit_base_bytes == math.ceil(base_gib * _GIB)
    assert estimate.commit_per_slot_bytes == math.ceil(slot_gib * _GIB)
    assert estimate.xrt_single == xrt


def test_memory_learning_is_monotonic_and_a_limit_hit_grows_the_whole_reservation() -> (
    None
):
    """Catch successful small exits shrinking a learned floor or OOM bumps missing slot memory."""
    key = make_key("whisper", "small", "auto", "cpu")
    store = EstimateStore(None, fingerprint=_FINGERPRINT, clock=lambda: 1000.0)
    store.observe_worker_exit(
        key, capacity=2, peak_commit_bytes=4 * _GIB, memory_limit_hit=False
    )
    learned = store.estimate(key)
    assert learned.commit_base_bytes == int(3.5 * _GIB)
    store.observe_worker_exit(
        key, capacity=2, peak_commit_bytes=2 * _GIB, memory_limit_hit=False
    )
    assert store.estimate(key).commit_base_bytes == learned.commit_base_bytes
    before = learned.commit_base_bytes + 2 * learned.commit_per_slot_bytes
    store.observe_worker_exit(
        key, capacity=2, peak_commit_bytes=None, memory_limit_hit=True
    )
    bumped = store.estimate(key)
    assert bumped.commit_base_bytes + 2 * bumped.commit_per_slot_bytes == math.ceil(
        before * 1.25
    )
    for factor in (0.5, -1.0, float("nan"), float("inf")):
        store.bump_memory(key, factor=factor)
    assert store.estimate(key) == bumped


def test_service_correction_ema_and_clamps_ignore_invalid_samples() -> None:
    """Catch an unstable ETA correction or NaN and zero-runtime poisoning learned timing."""
    key = make_key("whisper", "small", "auto", "cpu")
    store = EstimateStore(None, fingerprint=_FINGERPRINT)
    store.observe_job(key, run_s=20, base_service_s=10)
    assert store.estimate(key).service_correction == pytest.approx(1.2)
    store.observe_job(key, run_s=5, base_service_s=10)
    assert store.estimate(key).service_correction == pytest.approx(1.06)
    for run, base in ((0.0, 1.0), (1.0, 0.0), (float("nan"), 1.0), (1.0, float("inf"))):
        store.observe_job(key, run_s=run, base_service_s=base)
    assert store.estimate(key).service_correction == pytest.approx(1.06)
    for run in (1e-100, 1e100):
        for _ in range(40):
            store.observe_job(key, run_s=run, base_service_s=1)
            assert 0.25 <= store.estimate(key).service_correction <= 4.0


@pytest.mark.integration
def test_estimates_persist_only_when_due_and_restore_for_matching_hardware(
    tmp_path: Path,
) -> None:
    """Catch lost learned limits, repeated disk writes per tick or reuse on a different host."""
    clock = [1000.0]
    path = tmp_path / "estimates.json"
    key = make_key("gigaam-e2e-ctc", "ignored", "auto", "gpu")
    store = EstimateStore(path, fingerprint=_FINGERPRINT, clock=lambda: clock[0])
    store.bump_memory(key, commit=True, vram=True)
    assert store.save_if_due()
    store.observe_job(key, run_s=20, base_service_s=10)
    clock[0] += 29
    assert not store.save_if_due()
    clock[0] += 1
    assert store.save_if_due()
    restored = EstimateStore.load(path, fingerprint=_FINGERPRINT)
    assert restored.estimate(key) == store.estimate(key)
    assert not store.save_if_due(force=True)
    other = EstimateStore.load(path, fingerprint={**_FINGERPRINT, "gpu": "other"})
    assert other.estimate(key) == seed_estimate(key)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["version"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert EstimateStore.load(path, fingerprint=_FINGERPRINT).estimate(
        key
    ) == seed_estimate(key)


@pytest.mark.integration
def test_corrupt_estimate_document_is_ignored_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Catch a partial persisted JSON file preventing supervisor startup."""
    path = tmp_path / "estimates.json"
    path.write_text('{"keys":', encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        store = EstimateStore.load(path, fingerprint=_FINGERPRINT)
    assert any(record.levelno >= logging.WARNING for record in caplog.records)
    key = make_key("whisper", "small", "auto", "cpu")
    assert store.estimate(key) == seed_estimate(key)


@pytest.mark.integration
@pytest.mark.parametrize("bad_value", ["invalid", float("nan"), 10**400])
def test_corrupt_numeric_entry_does_not_discard_valid_keys_or_crash_startup(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, bad_value: object
) -> None:
    """Catch malformed persisted corrections aborting startup instead of skipping one key."""
    path = tmp_path / "estimates.json"
    good_key = make_key("whisper", "base", "auto", "cpu")
    bad_key = make_key("whisper", "small", "auto", "cpu")
    source = EstimateStore(path, fingerprint=_FINGERPRINT)
    source.bump_memory(good_key)
    source.bump_memory(bad_key)
    payload = copy.deepcopy(source.to_dict())
    entries = cast("dict[str, dict[str, object]]", payload["keys"])
    entries[bad_key.label]["service_correction"] = bad_value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        loaded = EstimateStore.load(path, fingerprint=_FINGERPRINT)
    assert loaded.estimate(good_key) == source.estimate(good_key)
    assert loaded.estimate(bad_key) == seed_estimate(bad_key)
    assert any(record.levelno >= logging.WARNING for record in caplog.records)


@pytest.mark.integration
def test_persisted_huge_commit_floor_remains_bounded_after_memory_bump(
    tmp_path: Path,
) -> None:
    """Catch a persisted huge commit floor crashing the next OOM bump or bypassing its cap."""
    path = tmp_path / "estimates.json"
    key = make_key("whisper", "small", "auto", "cpu")
    source = EstimateStore(path, fingerprint=_FINGERPRINT)
    source.observe_job(key, run_s=20, base_service_s=10)
    payload = source.to_dict()
    entries = cast("dict[str, dict[str, object]]", payload["keys"])
    entries[key.label]["commit_floor_bytes"] = 10**400
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = EstimateStore.load(path, fingerprint=_FINGERPRINT)
    loaded.bump_memory(key)
    estimate = loaded.estimate(key)
    assert estimate.service_correction == pytest.approx(1.2)
    assert (
        math.ceil(1.8 * _GIB) <= estimate.commit_base_bytes <= 4 * math.ceil(1.8 * _GIB)
    )
    assert estimate.commit_per_slot_bytes <= 4 * math.ceil(0.25 * _GIB)
