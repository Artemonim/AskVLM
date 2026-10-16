"""Dense-logic tests of the STT concurrency benchmark (``tools/stt_concurrency``).

No models and no child processes: configuration grammar and matrix, resource
estimates and guard decisions, statistics, correctness comparison, and the
resource probes. Expected values are derived by hand from the documented rules
(module docstrings and the benchmark requirements), not from the code output.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from typing import TYPE_CHECKING

import pytest

# * The support module puts `tools/` on sys.path for the imports below.
from tests import stt_concurrency_support as support

# isort: split

from stt_concurrency import growth_inputs, growth_stats, report, runner
from stt_concurrency.configs import (
    DEFAULT_ESTIMATES,
    ConfigError,
    ResourceEstimate,
    ThreadDefaults,
    apply_estimate_overrides,
    default_matrix,
    estimate_need,
    parse_config_name,
    select_configs,
)
from stt_concurrency.engines import chunk_timestamps
from stt_concurrency.growth_stats import _window
from stt_concurrency.sysmetrics import (
    CommitInfo,
    CpuSnapshot,
    GpuPoller,
    GpuSample,
    ProcessTreeSampler,
    cpu_busy_percent,
)
from stt_concurrency.workload import (
    WorkloadError,
    discover_clips,
    normalize_text,
    percentile,
    pick_warmup,
    similarity,
)

if TYPE_CHECKING:
    from pathlib import Path

DEFAULTS = ThreadDefaults(cpu_budget=16, cuda_threads=4)
GIB = 1024**3


# --- configuration grammar -------------------------------------------------


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("whisper-cpu-P2x8", ("whisper", "cpu", "P", 2, 8, False)),
        ("gigaam-cuda-B8", ("gigaam", "cuda", "B", 8, 4, False)),
        ("gigaam-cpu-S4", ("gigaam", "cpu", "S", 4, 4, False)),
        ("gigaam-cpu-B8", ("gigaam", "cpu", "B", 8, 16, False)),
        ("whisper-cpu-P3", ("whisper", "cpu", "P", 3, 5, False)),
        ("whisper-cuda-B8-long", ("whisper", "cuda", "B", 8, 4, True)),
        ("whisper-cuda-B8", ("whisper", "cuda", "B", 8, 4, False)),
        ("whisper-cpu-B4x16", ("whisper", "cpu", "B", 4, 16, False)),
        ("gigaam-ctc-cpu-S4", ("gigaam-ctc", "cpu", "S", 4, 4, False)),
        ("whisper-cuda-P1x2-long", ("whisper", "cuda", "P", 1, 2, True)),
    ],
)
def test_config_name_grammar_and_thread_defaults(
    token: str, expected: tuple[str, str, str, int, int, bool]
) -> None:
    """Catches: wrong thread derivation (CPU budget split, CUDA, batching)."""
    cfg = parse_config_name(token, DEFAULTS)

    assert (
        cfg.engine,
        cfg.device,
        cfg.mode,
        cfg.width,
        cfg.threads,
        cfg.long,
    ) == expected


@pytest.mark.parametrize(
    "token",
    [
        "gigaam-ctc-cuda-P1",  # GigaAM-Multilingual ctc is CPU only.
        "gigaam-ctc-cpu-P1-long-long",
        "whisper-cpu-P0",
        "whisper-cpu-P2x0",
        "whisper-gpu-P1",
        "other-cpu-P1",
        "gigaam-cpu-P2-LONG",
        "gigaam-cpu-P2x",
        "",
    ],
)
def test_invalid_config_names_are_rejected(token: str) -> None:
    """Catches: malformed names silently becoming some other configuration."""
    with pytest.raises(ConfigError):
        parse_config_name(token, DEFAULTS)


def test_derived_shape_of_modes() -> None:
    """Catches: wrong process/thread/batch accounting that feeds guard and report."""
    proc = parse_config_name("whisper-cpu-P2x8", DEFAULTS)
    thread = parse_config_name("gigaam-cpu-S4x2", DEFAULTS)
    batch = parse_config_name("gigaam-cuda-B8x4", DEFAULTS)

    assert (proc.processes, proc.slots, proc.batch_size, proc.total_threads) == (
        2,
        1,
        1,
        16,
    )
    assert (
        thread.processes,
        thread.slots,
        thread.batch_size,
        thread.total_threads,
    ) == (
        1,
        4,
        1,
        8,
    )
    assert (batch.processes, batch.slots, batch.batch_size, batch.total_threads) == (
        1,
        1,
        8,
        4,
    )


@pytest.mark.parametrize(("budget", "has_long"), [(32, True), (16, False), (6, True)])
def test_default_matrix_invariants(budget: int, *, has_long: bool) -> None:
    """Catches: oversubscribed CPU configs, duplicate names, missing P1 reference."""
    matrix = default_matrix(ThreadDefaults(budget, 4), has_long=has_long)

    names = [cfg.name for cfg in matrix]
    assert len(names) == len(set(names))
    for cfg in matrix:
        assert parse_config_name(cfg.name, DEFAULTS) == cfg
        if cfg.device == "cpu":
            assert cfg.total_threads <= budget
        if not has_long:
            assert not cfg.long
    groups = {cfg.group for cfg in matrix}
    for group in groups:
        # * The correctness signal compares against the P1 config of the group.
        assert any(
            cfg.group == group and cfg.mode == "P" and cfg.width == 1 for cfg in matrix
        ), group
    assert ("whisper-cuda-long" in groups) == has_long


def test_default_matrix_content_for_16_cpu_threads() -> None:
    """Catches: a mode missing for an engine/device pair of the requirement."""
    matrix = default_matrix(DEFAULTS, has_long=True)

    def shapes(engine: str, device: str, *, long: bool = False) -> set[tuple[str, int]]:
        return {
            (cfg.mode, cfg.width)
            for cfg in matrix
            if (cfg.engine, cfg.device, cfg.long) == (engine, device, long)
        }

    # * Requirement: P and S for every engine on CPU and CUDA, B for GigaAM
    # * batching and for Whisper BatchedInferencePipeline on long files.
    for engine in ("whisper", "gigaam"):
        for device in ("cpu", "cuda"):
            found = shapes(engine, device)
            assert ("P", 1) in found, (engine, device)
            assert any(m == "P" and w > 1 for m, w in found), (engine, device)
            assert any(m == "S" and w > 1 for m, w in found), (engine, device)
    assert any(mode == "B" for mode, _w in shapes("gigaam", "cuda"))
    assert not any(mode == "B" for mode, _w in shapes("whisper", "cuda"))
    long_shapes = shapes("whisper", "cuda", long=True)
    assert ("P", 1) in long_shapes
    assert any(mode == "B" for mode, _w in long_shapes)
    # * CPU thread split: every CPU config of the same width uses budget // width.
    for cfg in matrix:
        if cfg.device == "cpu" and cfg.mode in {"P", "S"}:
            assert cfg.threads == DEFAULTS.cpu_budget // cfg.width


def test_selection_globs_exact_names_and_errors() -> None:
    """Catches: selection that drops, duplicates, or invents configurations."""
    globbed = select_configs("whisper-cpu-S*", DEFAULTS, has_long=False)
    explicit = select_configs(
        "gigaam-cuda-B8, gigaam-cuda-B8x4,whisper-cpu-P3", DEFAULTS, has_long=False
    )

    assert [c.name for c in globbed] == [
        "whisper-cpu-S2x8",
        "whisper-cpu-S4x4",
        "whisper-cpu-S8x2",
    ]
    assert [c.name for c in explicit] == ["gigaam-cuda-B8x4", "whisper-cpu-P3x5"]
    with pytest.raises(ConfigError):
        select_configs("nothing-*", DEFAULTS, has_long=False)
    # * Whisper cross-file batching on clips is valid on both devices.
    batched = select_configs(
        "whisper-cuda-B8,whisper-cpu-B4x16", DEFAULTS, has_long=False
    )
    assert [(c.name, c.long) for c in batched] == [
        ("whisper-cuda-B8x4", False),
        ("whisper-cpu-B4x16", False),
    ]
    with pytest.raises(ConfigError):
        select_configs("gigaam-ctc-cuda-P1", DEFAULTS, has_long=False)


# --- estimates and guard ---------------------------------------------------


# * Own numbers, so the tests check the documented formula and not the tuning.
TEST_ESTIMATES = {
    **DEFAULT_ESTIMATES,
    # * commit, vram, extra per S thread, extra per B item
    "whisper-cpu": ResourceEstimate(2.0, 0.0, 0.5, 0.25),
    "gigaam-cuda": ResourceEstimate(10.0, 2.0, 1.0, 0.1),
}


@pytest.mark.parametrize(
    ("token", "commit_gb", "vram_gb"),
    [
        ("whisper-cpu-P2x1", 4.0, 0.0),  # 2 x 2.0
        ("gigaam-cuda-P2x1", 20.0, 4.0),  # 2 x 10 ; 2 x 2
        ("gigaam-cuda-S4x1", 13.0, 5.0),  # 10 + 3 x 1 ; 2 + 3 x 1
        ("gigaam-cuda-B8x1", 10.7, 2.7),  # 10 + 7 x 0.1 ; 2 + 7 x 0.1
        ("whisper-cpu-S3x1", 3.0, 0.0),  # 2 + 2 x 0.5
    ],
)
def test_estimate_need(token: str, commit_gb: float, vram_gb: float) -> None:
    """Catches: guard estimates that undercount instances, threads, or batch items."""
    need = estimate_need(parse_config_name(token, DEFAULTS), TEST_ESTIMATES)

    assert need.commit_gb == pytest.approx(commit_gb)
    assert need.vram_gb == pytest.approx(vram_gb)


def test_estimate_overrides_validate_input() -> None:
    """Catches: typos in --estimate silently ignored or accepted as nonsense."""
    result = apply_estimate_overrides(
        DEFAULT_ESTIMATES, ["gigaam-cuda.vram_gb=2.5", "whisper-cpu.commit_gb=1"]
    )

    assert result["gigaam-cuda"].vram_gb == 2.5
    assert result["whisper-cpu"].commit_gb == 1.0
    assert result["gigaam-cpu"] == DEFAULT_ESTIMATES["gigaam-cpu"]
    for bad in ("gigaam-cuda.vram=1", "nope-cpu.vram_gb=1", "gigaam-cuda.vram_gb=x"):
        with pytest.raises(ConfigError):
            apply_estimate_overrides(DEFAULT_ESTIMATES, [bad])
    with pytest.raises(ConfigError):
        apply_estimate_overrides(DEFAULT_ESTIMATES, ["gigaam-cuda.vram_gb=-1"])


def _options(
    tmp_path: Path,
    *,
    min_commit: float = 8.0,
    min_vram: float = 512.0,
    reserve: float = 0.0,
) -> runner.RunOptions:
    return runner.RunOptions(
        clips_dir=tmp_path,
        out_dir=tmp_path,
        max_clips=0,
        long_files=(),
        language="ru",
        selection="",
        repeat=1,
        clear=False,
        retry_failed=False,
        max_config_seconds=60.0,
        min_free_commit_gb=min_commit,
        commit_reserve_gb=reserve,
        min_free_vram_mb=min_vram,
        estimates=TEST_ESTIMATES,
        thread_defaults=DEFAULTS,
        whisper_model="small",
        whisper_options={},
        whisper_chunk_s=30.0,
        warmup_clip=None,
        sample_interval_s=0.5,
        cooldown_s=0.0,
        gigaam_cuda_streams=True,
        dry_run=True,
    )


@pytest.mark.parametrize(
    ("free_gb", "reserve_gb", "allowed"),
    [
        (10.0, 0.0, True),  # * Headroom equals the 8 GB minimum.
        (9.9, 0.0, False),
        (100.0, 0.0, True),
        (14.0, 12.0, True),  # * Headroom equals the 12 GB reserve.
        (13.9, 12.0, False),
        (9.9, 4.0, False),  # * The minimum still applies above a smaller reserve.
    ],
)
def test_guard_commit_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    free_gb: float,
    reserve_gb: float,
    *,
    allowed: bool,
) -> None:
    """Catches: off-by-one in the free-commit decision (run vs skip)."""
    info = CommitInfo(int(free_gb * GIB), None, None, "test")
    monkeypatch.setattr(runner, "query_commit", lambda: info)

    ok, reason, _details = runner.evaluate_guard(
        parse_config_name("whisper-cpu-P1x4", DEFAULTS),
        _options(tmp_path, reserve=reserve_gb),
        None,
    )

    assert ok is allowed
    assert (reason is None) == allowed


@pytest.mark.parametrize(
    ("free_mb", "allowed"),
    [
        (4096.0 + 512.0, True),  # P2 needs 2 x 2 GB = 4096 MB; +512 MB headroom
        (4096.0 + 500.0, False),
    ],
)
def test_guard_vram_boundary_and_missing_nvidia_smi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, free_mb: float, *, allowed: bool
) -> None:
    """Catches: CUDA configs started without enough free VRAM or without a probe."""
    monkeypatch.setattr(
        runner, "query_commit", lambda: CommitInfo(500 * GIB, None, None, "test")
    )
    monkeypatch.setattr(
        runner,
        "query_gpu",
        lambda _exe: GpuSample(0.0, 8192.0 - free_mb, 8192.0, 0.0),
    )
    cfg = parse_config_name("gigaam-cuda-P2x4", DEFAULTS)

    ok, _reason, _details = runner.evaluate_guard(cfg, _options(tmp_path), "nvidia-smi")
    no_probe, _reason, _details = runner.evaluate_guard(cfg, _options(tmp_path), None)

    assert ok is allowed
    assert no_probe is False


# --- statistics and workload -----------------------------------------------


@pytest.mark.parametrize(
    ("values", "q", "expected"),
    [
        ([1.0, 2.0, 3.0, 4.0], 50.0, 2.5),
        ([4.0, 1.0, 3.0, 2.0], 95.0, 3.85),
        ([float(x) for x in range(1, 101)], 95.0, 95.05),
        ([7.0], 95.0, 7.0),
        ([1.0, 9.0], 0.0, 1.0),
        ([1.0, 9.0], 100.0, 9.0),
    ],
)
def test_percentile_linear_interpolation(
    values: list[float], q: float, expected: float
) -> None:
    """Catches: wrong p50/p95 latency (rank off-by-one, unsorted input)."""
    assert percentile(values, q) == pytest.approx(expected)


def test_percentile_of_nothing_is_none() -> None:
    """Catches: a crash or a fake zero latency for configs without results."""
    assert percentile([], 50.0) is None


def _growth_records(values: list[int]) -> list[dict[str, object]]:
    """Place hand-calculated memory observations on a 30-second job clock."""
    return [
        {"index": index, "elapsed_s": 30 * index, "private_bytes": value}
        for index, value in enumerate(values, start=1)
    ]


@pytest.mark.parametrize(
    ("count", "warmups"), [(5, 1), (6, 1), (10, 2), (11, 2), (10_000, 2000)]
)
def test_growth_tail_slope_excludes_warmup_and_converts_hours(
    count: int, warmups: int
) -> None:
    """Catch warm-up contamination and a per-second slope reported as per-hour."""
    mib = 1024**2
    values = [9000 * mib] * warmups + [
        (100 + 2 * index) * mib for index in range(warmups + 1, count + 1)
    ]

    summary = growth_stats.summarize_records(
        _growth_records(values), tolerance_bytes=0, recycle_threshold_bytes=None
    )["metrics"]["private_bytes"]

    # * The retained line adds 2 MiB every 30 s: 120 jobs/hour, 240 MiB/hour.
    assert summary["slope_per_job"] == pytest.approx(2 * mib)
    assert summary["slope_per_hour"] == pytest.approx(240 * mib)


@pytest.mark.parametrize(
    ("count", "tail", "drift", "level"),
    [
        (0, 0, 1, 1),
        (1, 0, 1, 1),
        (2, 0, 1, 1),
        (3, 0, 1, 1),
        (4, 0, 1, 1),
        (5, 1, 1, 1),
        (6, 1, 1, 1),
        (7, 1, 1, 1),
        (8, 1, 1, 1),
        (9, 1, 1, 1),
        (10, 2, 2, 1),
        (11, 2, 2, 1),
        (12, 2, 2, 1),
        (19, 3, 3, 1),
        (20, 4, 4, 1),
        (21, 4, 4, 1),
        (39, 7, 7, 1),
        (40, 8, 8, 2),
        (41, 8, 8, 2),
        (
            10**18 - 1,
            199_999_999_999_999_999,
            199_999_999_999_999_999,
            49_999_999_999_999_999,
        ),
        (
            10**18,
            200_000_000_000_000_000,
            200_000_000_000_000_000,
            50_000_000_000_000_000,
        ),
        (
            10**18 + 1,
            200_000_000_000_000_000,
            200_000_000_000_000_000,
            50_000_000_000_000_000,
        ),
    ],
)
def test_growth_share_windows_preserve_exact_integer_boundaries(
    count: int, tail: int, drift: int, level: int
) -> None:
    """Catch a rounded share admitting warm-up jobs or moving median windows."""
    # * The 80% tail excludes complete groups of five; 20% and 5% windows
    # * contain complete groups of five and twenty, with a one-item minimum.
    assert growth_stats.tail_start(count) == tail
    assert growth_stats.tail_start(count, 0.8) == tail
    assert _window(count, 0.2) == drift
    assert _window(count, 0.05) == level


@pytest.mark.parametrize(
    ("values", "drift", "plateau", "first_level"),
    [
        ([50] * 20 + [100] * 80, 0, True, 21),
        ([50] * 4 + [1000] + [50] * 15 + [100] * 79 + [1000], 0, True, 21),
        ([50] * 20 + [100] * 60 + [116] * 20, 16, True, 21),
        ([50] * 20 + [100] * 60 + [117] * 20, 17, False, 81),
        ([150] * 20 + [100] * 60 + [90] * 20, -10, True, 1),
        ([100 * index for index in range(19)], 300, False, 18),
        ([100 * index for index in range(20)], 400, False, 19),
        ([100 * index for index in range(21)], 400, False, 20),
        ([100 * index for index in range(39)], 700, False, 36),
        ([100 * index for index in range(40)], 800, False, 37),
        ([100 * index for index in range(41)], 800, False, 37),
        ([100 * index for index in range(59)], 1100, False, 55),
        ([100 * index for index in range(60)], 1200, False, 55),
        ([100 * index for index in range(61)], 1200, False, 55),
    ],
)
def test_growth_plateau_uses_window_medians_and_an_inclusive_tolerance(
    values: list[int], drift: int, *, plateau: bool, first_level: int
) -> None:
    """Catch transient spikes flipping a plateau or replacing medians with maxima."""
    summary = growth_stats.metric_summary(
        _growth_records(values), "private_bytes", tolerance=16
    )

    assert summary is not None
    assert summary["tail_drift"] == drift
    assert summary["plateaued"] is plateau
    assert summary["plateau_at_job"] == first_level


@pytest.mark.parametrize(
    ("step", "threshold", "projected", "first_job"),
    [
        (2, 142, 10, None),
        (0, 142, None, None),
        (-2, 142, None, None),
        (2, 110, 0, 5),
        (2, None, None, None),
    ],
)
def test_growth_recycle_projection_requires_a_positive_slope(
    step: int, threshold: int | None, projected: int | None, first_job: int | None
) -> None:
    """Catch invented exhaustion dates for flat/falling memory or missed crossings."""
    records = _growth_records([100 + step * index for index in range(1, 12)])

    recycle = growth_stats.summarize_records(
        records, tolerance_bytes=0, recycle_threshold_bytes=threshold
    )["recycle"]

    assert recycle["projected_jobs_to_threshold"] == projected
    assert recycle["first_job_at_threshold"] == first_job


@pytest.mark.parametrize("values", [[], [None], [True], [123], [None, 123]])
def test_growth_sparse_metrics_do_not_invent_a_slope_or_plateau(
    values: list[object],
) -> None:
    """Catch crashes and fabricated trends when fewer than two jobs report memory."""
    records = [
        {"index": index, "private_bytes": value}
        for index, value in enumerate(values, start=1)
    ]
    summary = growth_stats.summarize_records(
        records, tolerance_bytes=16, recycle_threshold_bytes=1000
    )
    private = summary["metrics"].get("private_bytes")
    if private is not None:
        assert private["points"] == 1
        assert private["slope_per_job"] is None
        assert private["slope_per_hour"] is None
        assert private["tail_drift"] is None
        assert not private["plateaued"]
    assert summary["recycle"]["projected_jobs_to_threshold"] is None


def test_growth_throughput_excludes_idle_time_and_retains_failed_jobs() -> None:
    """Catch idle gaps counted as busy time or failed jobs silently removed."""
    records = [
        {"index": 1, "status": "ok", "wall_s": 1, "audio_s": 5, "elapsed_s": 1},
        {"index": 2, "status": "empty", "wall_s": 3, "audio_s": 7, "elapsed_s": 50},
        {"index": 3, "status": "error", "wall_s": 2, "audio_s": 0, "elapsed_s": 100},
    ]

    summary = growth_stats.summarize_records(
        records, tolerance_bytes=0, recycle_threshold_bytes=None
    )["throughput"]

    assert (summary["jobs"], summary["ok"], summary["errors"]) == (3, 2, 1)
    assert (summary["busy_s"], summary["elapsed_s"]) == (6, 100)
    assert summary["jobs_per_min_busy"] == 30
    assert summary["audio_xrt"] == 2
    assert summary["wall_s_median"] == 2


def test_growth_trajectory_keeps_job_numbers_at_the_documented_shares() -> None:
    """Catch memory values sampled at the wrong jobs or trajectory endpoints lost."""
    summary = growth_stats.metric_summary(
        _growth_records([100 + 2 * index for index in range(1, 14)]),
        "private_bytes",
        tolerance=0,
    )

    assert summary is not None
    assert summary["trajectory"] == [
        [1, 102],
        [2, 104],
        [4, 108],
        [7, 114],
        [10, 120],
        [13, 126],
    ]


@pytest.mark.parametrize(
    ("count", "positions"),
    [
        (1, [0]),
        (2, [0, 1]),
        (3, [0, 1, 2]),
        (6, [0, 1, 2, 4, 5]),
        (7, [0, 1, 2, 3, 4, 6]),
        (11, [0, 1, 2, 5, 8, 10]),
        (16, [0, 2, 4, 8, 11, 15]),
        (26, [0, 2, 6, 12, 19, 25]),
        (36, [0, 4, 9, 18, 26, 35]),
        (
            10**18,
            [
                0,
                100_000_000_000_000_000,
                250_000_000_000_000_000,
                500_000_000_000_000_000,
                749_999_999_999_999_999,
                999_999_999_999_999_999,
            ],
        ),
    ],
)
def test_growth_trajectory_keeps_even_ties_and_large_integer_positions(
    count: int, positions: list[int]
) -> None:
    """Catch upward half-tie bias, duplicate jobs or float-rounded large indices."""
    # * Lazy ranges exercise long runs without allocating their observations.
    expected = [[float(position), position] for position in positions]

    assert growth_stats.trajectory(range(count), range(count)) == expected
    assert (
        growth_stats.trajectory(
            range(count), range(count), (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
        )
        == expected
    )


@pytest.mark.parametrize("mode", ["fixed", "varied"])
def test_growth_clip_plan_is_seeded_and_keeps_duration_and_offset_streams_separate(
    tmp_path: Path, mode: str
) -> None:
    """Catch non-reproducible cuts or source length changing the duration workload."""
    plan = growth_inputs.DurationPlan(mode, 2, 3, 2.5, 17)
    other_seed = growth_inputs.DurationPlan(mode, 2, 3, 2.5, 18)
    fields = {"source_s": 2, "out_dir": tmp_path, "suffix": ".ogg"}

    clips = growth_inputs.plan_clips(plan, 12, **fields)
    repeated = growth_inputs.plan_clips(plan, 12, **fields)
    different = growth_inputs.plan_clips(other_seed, 12, **fields)
    longer_source = growth_inputs.plan_clips(plan, 12, **{**fields, "source_s": 9})

    assert clips == repeated
    assert [clip.offset_s for clip in clips] != [clip.offset_s for clip in different]
    assert [clip.duration_s for clip in clips] == [
        clip.duration_s for clip in longer_source
    ]
    assert all(0 <= clip.offset_s <= 2 for clip in clips)
    durations = {clip.duration_s for clip in clips}
    if mode == "fixed":
        assert durations == {2.5}
    else:
        assert len(durations) > 1
        assert all(2 <= duration <= 3 for duration in durations)
        assert all(
            duration * 1000 == pytest.approx(round(duration * 1000))
            for duration in durations
        )


@pytest.mark.integration
@pytest.mark.parametrize("audio_format", ["opus", "wav"])
@pytest.mark.parametrize("mode", ["fixed", "varied"])
def test_growth_prepared_inputs_obey_ffmpeg_seek_trim_and_codec_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, audio_format: str, mode: str
) -> None:
    """Catch wrong offsets, clips crossing the loop seam or wrong Telegram encoding."""
    tools = support.GrowthTools(monkeypatch)
    source = tmp_path / "source.wav"
    support.write_wav(source, 2)
    work = tmp_path / "inputs"

    clips = growth_inputs.prepare_inputs(
        growth_inputs.DurationPlan(mode, 2, 3, 3, 17),
        4,
        source=str(source),
        work_dir=work,
        audio_format=audio_format,
    )

    cuts = {args[-1]: args for args in tools.calls if "-t" in args}
    assert len(cuts) == len(clips) == 4
    assert growth_inputs.wav_seconds(work / "repeated.wav") == 6
    for clip in clips:
        args = cuts[str(clip.path)]
        assert args[args.index("-i") + 1] == str(work / "repeated.wav")
        assert args.index("-ss") < args.index("-i") < args.index("-t")
        assert float(args[args.index("-ss") + 1]) == clip.offset_s
        assert float(args[args.index("-t") + 1]) == clip.duration_s
        assert args[args.index("-ar") + 1] == "16000"
        assert args[args.index("-ac") + 1] == "1"
        assert "-nostdin" in args
        if audio_format == "opus":
            assert args[args.index("-c:a") + 1] == "libopus"
            assert args[args.index("-b:a") + 1] == "32k"
            assert args[args.index("-application") + 1] == "voip"
            assert clip.path.suffix == ".ogg"
        else:
            assert args[args.index("-c:a") + 1] == "pcm_s16le"
            assert clip.path.suffix == ".wav"
        assert clip.path.is_file()


def test_chunk_timestamps_cover_the_audio_without_empty_tail() -> None:
    """Catches: dropped audio or an empty last chunk in batched Whisper."""
    assert chunk_timestamps(65.0, 30.0) == [
        {"start": 0.0, "end": 30.0},
        {"start": 30.0, "end": 60.0},
        {"start": 60.0, "end": 65.0},
    ]
    assert len(chunk_timestamps(60.0, 30.0)) == 2
    assert chunk_timestamps(0.0, 30.0) == []


def test_text_normalization_and_similarity() -> None:
    """Catches: punctuation or case differences counted as transcript mismatch."""
    assert normalize_text("Привет,  Мир!  ") == normalize_text("привет мир")
    assert normalize_text("a-b, c.") == "a b c"
    assert similarity("", "") == 1.0
    assert similarity("abc", "abc") == 1.0
    assert 0.0 < similarity("hello world", "hello word") < 1.0
    assert similarity("abc", "") == 0.0


def test_discover_clips_is_deterministic_and_truncates(tmp_path: Path) -> None:
    """Catches: a workload that differs between configurations."""
    support.write_wav(tmp_path / "b.wav", 2)
    support.write_wav(tmp_path / "a.wav", 1)
    support.write_wav(tmp_path / "c.WAV", 3)
    (tmp_path / "notes.txt").write_text("x")

    clips = discover_clips(tmp_path, 2)

    assert [c.path.name for c in clips] == ["a.wav", "b.wav"]
    assert [c.duration_s for c in clips] == [1.0, 2.0]
    assert [c.index for c in clips] == [0, 1]
    assert len(discover_clips(tmp_path, 0)) == 3
    with pytest.raises(WorkloadError):
        discover_clips(tmp_path / "missing", 1)


def test_warmup_prefers_a_clip_that_fits_one_gigaam_window(tmp_path: Path) -> None:
    """Catches: a warm-up clip over 20 s that disables the GigaAM parity check."""
    support.write_wav(tmp_path / "a.wav", 25)
    support.write_wav(tmp_path / "b.wav", 5)

    clips = discover_clips(tmp_path, 0)

    assert pick_warmup(clips, None).name == "b.wav"


# --- correctness comparison and summary -------------------------------------


def _result(
    name: str, repeat: int, texts: list[str | None], status: str = "ok"
) -> dict[str, object]:
    """Build a result document; ``None`` text marks a failed item."""
    cfg = parse_config_name(name, DEFAULTS)
    return {
        "config": cfg.to_json(),
        "repeat": repeat,
        "status": status,
        "started_at": f"2026-01-01T00:00:0{repeat}",
        "timing": {"throughput_x": 10.0},
        "items": [
            {
                "index": i,
                "text": text or "",
                "error": None if text is not None else "boom",
            }
            for i, text in enumerate(texts)
        ],
    }


def test_identity_is_measured_against_the_p1_reference_of_the_group() -> None:
    """Catches: wrong identity %, wrong reference, or mixing workloads/devices."""
    reference = _result(
        "gigaam-cuda-P1x4", 0, ["Привет, мир", "один два", "три", "четыре"]
    )
    batched = _result("gigaam-cuda-B4x4", 0, ["привет мир!", "один два", "тры", None])
    unrelated_cpu = _result("gigaam-cpu-P1x4", 0, ["совсем", "другое", "", ""])
    failed = _result("gigaam-cuda-S2x4", 0, ["x"], status="error")

    rows = report.build_rows([batched, unrelated_cpu, failed, reference])
    by_name = {row["name"]: row for row in rows}

    row = by_name["gigaam-cuda-B4x4"]
    # * Item 3 failed and is excluded; items 0 and 1 match after normalization.
    assert row["reference"] == "gigaam-cuda-P1x4"
    assert row["compared"] == 3
    assert row["identical_percent"] == pytest.approx(100.0 * 2 / 3)
    assert row["similarity_min"] == pytest.approx(similarity("тры", "три"))
    assert by_name["gigaam-cuda-P1x4"]["identical_percent"] == 100.0
    # * A different device is its own group with its own reference.
    assert by_name["gigaam-cpu-P1x4"]["reference"] == "gigaam-cpu-P1x4"
    assert "identical_percent" not in by_name["gigaam-cuda-S2x4"]


def test_reference_prefers_p1_over_the_first_config_run() -> None:
    """Catches: a reference that is not the sequential baseline."""
    older_s2 = _result("whisper-cpu-S2x8", 0, ["a", "b"])
    p1 = _result("whisper-cpu-P1x16", 1, ["a", "c"])

    rows = report.build_rows([older_s2, p1])

    by_name = {row["name"]: row for row in rows}
    assert by_name["whisper-cpu-S2x8"]["reference"] == "whisper-cpu-P1x16"
    assert by_name["whisper-cpu-S2x8"]["identical_percent"] == 50.0


def test_summary_files_are_written_from_result_files(tmp_path: Path) -> None:
    """Catches: summary that loses configs or breaks on partial result sets."""
    results = tmp_path / "results"
    results.mkdir()
    for doc in (
        _result("whisper-cpu-P1x16", 0, ["a", "b"]),
        _result("whisper-cpu-S2x8", 0, ["a", "b"]),
    ):
        cfg = doc["config"]
        assert isinstance(cfg, dict)
        (results / f"{cfg['name']}__r{doc['repeat']}.json").write_text(
            json.dumps(doc), encoding="utf-8"
        )
    (results / "broken.json").write_text("{not json", encoding="utf-8")

    md_path, table = report.write_summary(tmp_path)

    rows = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert [row["name"] for row in rows] == ["whisper-cpu-P1x16", "whisper-cpu-S2x8"]
    assert "whisper-cpu-S2x8" in table
    assert md_path.read_text(encoding="utf-8").startswith("# STT concurrency")


# --- resource probes --------------------------------------------------------


def test_cpu_busy_percent() -> None:
    """Catches: wrong CPU utilization math (busy vs total)."""
    start = CpuSnapshot(busy_s=100.0, total_s=1000.0)
    end = CpuSnapshot(busy_s=160.0, total_s=1200.0)

    assert cpu_busy_percent(start, end) == pytest.approx(30.0)
    assert cpu_busy_percent(start, start) is None
    assert cpu_busy_percent(None, end) is None


def test_vram_delta_uses_baseline_and_separates_load_from_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches: VRAM delta taken without the baseline or from the wrong phase."""
    readings = iter(
        [
            GpuSample(1.0, 4000.0, 8192.0, 5.0),  # load peak
            GpuSample(5.0, 3000.0, 8192.0, 80.0),  # run
            GpuSample(6.0, 2800.0, 8192.0, 40.0),  # run
            GpuSample(20.0, 1100.0, 8192.0, 0.0),  # after the run
        ]
    )
    monkeypatch.setattr(
        "stt_concurrency.sysmetrics.query_gpu", lambda _exe: next(readings)
    )
    poller = GpuPoller("nvidia-smi", 1.0)
    for _ in range(4):
        poller.sample()
    baseline = GpuSample(0.0, 1000.0, 8192.0, 0.0)

    stats = poller.stats(baseline, 4.0, 10.0)

    assert stats.peak_mb == 4000.0
    assert stats.delta_mb == 3000.0
    assert stats.peak_run_mb == 3000.0
    assert stats.delta_run_mb == 2000.0
    assert stats.util_avg_run_percent == pytest.approx(60.0)


def test_memory_sampler_sums_private_bytes_over_descendants() -> None:
    """Catches: peak memory that ignores child processes or reports only RSS."""
    sampler = ProcessTreeSampler(60.0)
    sampler.sample()
    before = sampler.stats(None, None).peak_private_bytes
    assert before is not None
    allocate = "b = bytearray(200 * 1024 * 1024); print('up', flush=True); input()"
    child = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", allocate],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "up"
        time.sleep(0.2)
        sampler.sample()
        stats = sampler.stats(None, None)
    finally:
        child.kill()
        child.wait(timeout=30)
        if child.stdin is not None:
            child.stdin.close()
        if child.stdout is not None:
            child.stdout.close()

    assert stats.samples == 2
    assert stats.max_processes is not None
    assert stats.max_processes >= 2
    # * The second sample holds the child's 200 MiB commit on top of the first.
    assert stats.peak_private_bytes is not None
    assert stats.peak_private_bytes - before >= 150 * 1024 * 1024
