"""Typer CLI of the STT concurrency benchmark.

This module deliberately omits ``from __future__ import annotations``: Typer
resolves the parameter annotations at runtime.
"""

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import psutil  # type: ignore[import-untyped]
import typer

from core.stt_pool.engine_api import DEFAULT_ENGINE_FACTORY

from . import report
from .combo import COMBO_DIRNAME, run_combos, write_combo_summary
from .configs import (
    DEFAULT_ESTIMATES,
    ESTIMATE_FIELDS,
    ConfigError,
    ThreadDefaults,
    apply_estimate_overrides,
    default_matrix,
    estimate_need,
)
from .growth_inputs import (
    AUDIO_FORMATS,
    AUDIO_OPUS,
    DURATION_MODES,
    DURATION_VARIED,
    SAPI_VOICE,
    SOURCE_SAPI,
    DurationPlan,
)
from .memory_growth import (
    BOUNDS_POOL,
    DEFAULT_COMMIT_RESERVE_GIB,
    DEFAULT_JOB_TIMEOUT_S,
    DEFAULT_JOBS,
    DEFAULT_LOAD_TIMEOUT_S,
    DEFAULT_MAX_ERRORS,
    DEFAULT_MIN_FREE_COMMIT_GIB,
    DEFAULT_MIN_FREE_VRAM_GIB,
    DEFAULT_OUT_DIR,
    DEFAULT_PLATEAU_TOLERANCE_MIB,
    DEFAULT_PROGRESS_EVERY,
    GPU_MEMORY_BOUNDS,
    GrowthOptions,
    is_growth_run,
    rebuild_summary,
    run_memory_growth,
)
from .paths import RESULTS_DIRNAME
from .quality import ManifestError
from .runner import DEFAULT_COMMIT_RESERVE_GB, RunOptions, run_benchmark
from .store import ResultStore, latest_fingerprint
from .sweep import (
    DEFAULT_THREAD_SET,
    StopRules,
    SweepOptions,
    load_group_documents,
    run_sweep,
    write_sweep_summary,
)
from .workload import WorkloadError

# * User-tunable defaults.
DEFAULT_MAX_CLIPS = 120
DEFAULT_LANGUAGE = "ru"
DEFAULT_WHISPER_MODEL = "small"
DEFAULT_BEAM_SIZE = 5
# * Fixed decode config shared by every Whisper mode so that modes stay comparable.
# * temperature=0.0 disables the sampling fallback, keeping transcripts deterministic.
WHISPER_FIXED_DECODE: dict[str, object] = {
    "vad_filter": False,
    "word_timestamps": False,
    "condition_on_previous_text": False,
    "temperature": 0.0,
}
DEFAULT_WHISPER_CHUNK_S = 30.0
DEFAULT_CUDA_THREADS = 4
DEFAULT_MAX_CONFIG_SECONDS = 900.0
DEFAULT_MIN_FREE_COMMIT_GB = 8.0
DEFAULT_MIN_FREE_VRAM_MB = 512.0
DEFAULT_SAMPLE_INTERVAL_S = 0.5
DEFAULT_COOLDOWN_S = 5.0
DEFAULT_CLIPS_PER_STREAM = 12
DEFAULT_MIN_STEP_SECONDS = 45.0
DEFAULT_GROWTH_MIN_DURATION_S = 1.0
DEFAULT_GROWTH_MAX_DURATION_S = 60.0
DEFAULT_GROWTH_FIXED_DURATION_S = 13.0
DEFAULT_GROWTH_SEED = 1
DEFAULT_GROWTH_LABEL = "run"
_MAX_WHISPER_CHUNK_S = 30.0
_EXIT_USAGE = 2
_EXIT_INTERRUPTED = 130
_KNEES = ("best", "efficient")

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Measure how resident STT engines (faster-whisper small, GigaAM-v3 e2e_ctc, "
        "GigaAM-Multilingual ctc) scale on this PC: P = N processes with N models, "
        "S = one model served by N threads, B = batching inside one process. "
        "'run' measures a fixed matrix, 'sweep' searches the saturation knee per "
        "group, 'combo' runs a GPU and a CPU configuration at the same time, "
        "'memory-growth' feeds sequential jobs to one long-lived GigaAM GPU pool "
        "worker and tracks its memory per job. Every "
        "measurement uses fresh child processes. VRAM is read with "
        "'nvidia-smi --id=0' (whole GPU 0) while CUDA work runs on cuda:0; on a "
        "multi-GPU machine whose CUDA order (or CUDA_VISIBLE_DEVICES) differs from "
        "the nvidia-smi PCI order the VRAM numbers and the VRAM guard may describe "
        "another GPU."
    ),
)

# * Options shared by run / sweep / combo.
ClipsDirOpt = Annotated[
    Path,
    typer.Option(
        "--clips-dir",
        help="Directory of 16 kHz WAV clips (selection: *.wav sorted by name).",
        file_okay=False,
        resolve_path=True,
    ),
]
OutDirOpt = Annotated[
    Path,
    typer.Option(
        "--out-dir",
        help="Output directory (results, logs, summaries); keep it outside the repo.",
        file_okay=False,
        resolve_path=True,
    ),
]
MaxClipsOpt = Annotated[
    int,
    typer.Option(help="Number of clips taken from --clips-dir (<= 0: all)."),
]
ManifestOpt = Annotated[
    Path | None,
    typer.Option(
        help=(
            "JSONL reference manifest (wav, ref_fleurs_norm/ref_raw) for WER; "
            "e.g. a FLEURS test manifest."
        ),
        dir_okay=False,
        resolve_path=True,
    ),
]
LanguageOpt = Annotated[str, typer.Option(help="Whisper language code.")]
RetryOpt = Annotated[
    bool,
    typer.Option(help="Re-run results whose status is not 'ok'."),
]
MaxSecondsOpt = Annotated[
    float,
    typer.Option(
        help="Abort a measurement (kill its children) after this many seconds, "
        "model loading included; the result is recorded as 'timeout'.",
    ),
]
MinCommitOpt = Annotated[
    float,
    typer.Option(help="Skip when free commit minus the estimate is lower (GB)."),
]
MinVramOpt = Annotated[
    float,
    typer.Option(
        help="Skip CUDA work when free VRAM minus the estimate is lower (MB)."
    ),
]
JobLimitOpt = Annotated[
    float | None,
    typer.Option(
        help=(
            "Lower the commit cap (GB) of the Windows job holding all children of "
            "one measurement. Default cap: free commit at step start - "
            "--commit-reserve-gb (the real headroom); this option can only lower "
            "it. Hitting the cap records 'memory_cap' (a saturation point)."
        ),
    ),
]
AllowUncappedOpt = Annotated[
    bool,
    typer.Option(
        help=(
            "Run a step even when its commit cap cannot be set up (Windows job "
            "creation or assignment failed); recorded as cap_enforced=false. "
            "Without it such a step is skipped."
        ),
    ),
]
ReserveOpt = Annotated[
    float,
    typer.Option(
        help=(
            "System free commit (GB) kept for everything else: a measurement "
            "whose start leaves less does not start, and one during which the "
            "free commit drops below it is killed ('system_commit_low')."
        ),
    ),
]
EstimateOpt = Annotated[
    list[str] | None,
    typer.Option(
        help=(
            "Override a guard estimate: <engine>-<device>.<field>=<GB>, fields "
            f"{', '.join(ESTIMATE_FIELDS)} (repeat the option). Defaults: "
            + "; ".join(
                f"{key} {est.commit_gb:g}/{est.vram_gb:g}/{est.extra_thread_gb:g}"
                f"/{est.batch_item_gb:g}"
                for key, est in DEFAULT_ESTIMATES.items()
            )
        ),
    ),
]
CpuBudgetOpt = Annotated[
    int | None,
    typer.Option(
        min=1,
        help="CPU threads shared by a CPU config (default: physical cores).",
    ),
]
CudaThreadsOpt = Annotated[
    int,
    typer.Option(min=1, help="CPU threads per process of CUDA configs."),
]
WhisperModelOpt = Annotated[
    str,
    typer.Option(help="faster-whisper model name (offline, project model root)."),
]
BeamSizeOpt = Annotated[int, typer.Option(min=1, help="Whisper beam size (all modes).")]
WhisperOptionOpt = Annotated[
    list[str] | None,
    typer.Option(
        help=(
            "Override a Whisper decode option KEY=VALUE (JSON value), e.g. "
            "without_timestamps=true. Fixed defaults: "
            + ", ".join(f"{k}={v}" for k, v in WHISPER_FIXED_DECODE.items())
        ),
    ),
]
ChunkOpt = Annotated[
    float,
    typer.Option(help="Chunk length of batched Whisper (<= 30 s)."),
]
WarmupOpt = Annotated[
    Path | None,
    typer.Option(
        help="Warm-up clip (default: first selected clip <= 20 s).",
        dir_okay=False,
        resolve_path=True,
    ),
]
SampleOpt = Annotated[float, typer.Option(help="Memory/VRAM sampling period (s).")]
CooldownOpt = Annotated[float, typer.Option(help="Pause between measurements (s).")]
StreamsOpt = Annotated[
    bool,
    typer.Option(help="Give each GigaAM CUDA S thread its own CUDA stream."),
]
DryRunOpt = Annotated[
    bool,
    typer.Option(help="Print the plan and guard decisions; write nothing."),
]
ClipsPerStreamOpt = Annotated[
    int,
    typer.Option(min=1, help="Minimum items per stream (x N) of one step."),
]
MinStepOpt = Annotated[
    float,
    typer.Option(help="Minimum measured seconds of one step (clips are cycled)."),
]


def _default_cpu_budget() -> int:
    """Physical cores (SMT siblings add little to GEMM-bound inference)."""
    physical = psutil.cpu_count(logical=False)
    if isinstance(physical, int) and physical > 0:
        return physical
    return os.cpu_count() or 1


def _parse_whisper_options(tokens: list[str] | None) -> dict[str, object]:
    """Parse ``KEY=VALUE`` overrides; VALUE is JSON when it parses, else a string.

    Raises:
        ConfigError: For a token without ``=``.

    """
    parsed: dict[str, object] = {}
    for token in tokens or []:
        key, sep, raw = token.partition("=")
        if not sep or not key.strip():
            msg = f"Invalid --whisper-option {token!r}; expected KEY=VALUE"
            raise ConfigError(msg)
        try:
            value: object = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        parsed[key.strip()] = value
    return parsed


def _parse_int_set(raw: str, name: str) -> tuple[int, ...]:
    """Parse a comma-separated set of positive integers.

    Raises:
        ConfigError: For an empty set or a non-positive value.

    """
    try:
        values = tuple(sorted({int(part) for part in raw.split(",") if part.strip()}))
    except ValueError as exc:
        msg = f"{name} must be comma-separated integers (got {raw!r})"
        raise ConfigError(msg) from exc
    if not values or values[0] < 1:
        msg = f"{name} must list positive integers (got {raw!r})"
        raise ConfigError(msg)
    return values


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        msg = f"{name} must be positive (got {value:g})"
        raise ConfigError(msg)


def _validate_at_most(name: str, value: float, limit: float) -> None:
    if value > limit:
        msg = f"{name} must be at most {limit:g} (got {value:g})"
        raise ConfigError(msg)


def _build_run_options(  # noqa: PLR0913 - mirrors the shared CLI options
    *,
    clips_dir: Path,
    out_dir: Path,
    max_clips: int,
    manifest: Path | None,
    language: str,
    retry_failed: bool,
    max_config_seconds: float,
    min_free_commit_gb: float,
    min_free_vram_mb: float,
    estimate: list[str] | None,
    cpu_budget: int | None,
    cuda_threads: int,
    whisper_model: str,
    beam_size: int,
    whisper_option: list[str] | None,
    whisper_chunk_s: float,
    warmup_clip: Path | None,
    sample_interval: float,
    cooldown: float,
    gigaam_cuda_streams: bool,
    dry_run: bool,
    selection: str = "",
    repeat: int = 1,
    clear: bool = False,
    long_files: list[Path] | None = None,
    job_commit_limit_gb: float | None = None,
    commit_reserve_gb: float = DEFAULT_COMMIT_RESERVE_GB,
    allow_uncapped: bool = False,
) -> RunOptions:
    """Validate the shared options and build :class:`RunOptions`.

    Raises:
        ConfigError: For invalid values.

    """
    if job_commit_limit_gb is not None:
        _validate_positive("--job-commit-limit-gb", job_commit_limit_gb)
    if commit_reserve_gb < 0:
        msg = f"--commit-reserve-gb must not be negative (got {commit_reserve_gb:g})"
        raise ConfigError(msg)
    _validate_positive("--max-config-seconds", max_config_seconds)
    _validate_positive("--sample-interval", sample_interval)
    _validate_positive("--whisper-chunk-s", whisper_chunk_s)
    _validate_at_most("--whisper-chunk-s", whisper_chunk_s, _MAX_WHISPER_CHUNK_S)
    whisper_options: dict[str, object] = {
        "language": language,
        "beam_size": beam_size,
        **WHISPER_FIXED_DECODE,
        **_parse_whisper_options(whisper_option),
    }
    return RunOptions(
        clips_dir=clips_dir,
        out_dir=out_dir,
        max_clips=max_clips,
        long_files=tuple(long_files or ()),
        language=language,
        selection=selection,
        repeat=repeat,
        clear=clear,
        retry_failed=retry_failed,
        max_config_seconds=max_config_seconds,
        min_free_commit_gb=min_free_commit_gb,
        min_free_vram_mb=min_free_vram_mb,
        estimates=apply_estimate_overrides(DEFAULT_ESTIMATES, estimate or []),
        thread_defaults=ThreadDefaults(
            cpu_budget=cpu_budget or _default_cpu_budget(),
            cuda_threads=cuda_threads,
        ),
        whisper_model=whisper_model,
        whisper_options=whisper_options,
        whisper_chunk_s=whisper_chunk_s,
        warmup_clip=warmup_clip,
        sample_interval_s=sample_interval,
        cooldown_s=max(0.0, cooldown),
        gigaam_cuda_streams=gigaam_cuda_streams,
        dry_run=dry_run,
        manifest_path=manifest,
        job_commit_limit_gb=job_commit_limit_gb,
        commit_reserve_gb=commit_reserve_gb,
        allow_uncapped=allow_uncapped,
    )


def _sweep_options(
    run: RunOptions,
    *,
    clips_per_stream: int,
    min_step_seconds: float,
    threads: str = "",
    groups: str = "",
    max_steps: int = 0,
    rules: StopRules | None = None,
) -> SweepOptions:
    _validate_positive("--min-step-seconds", min_step_seconds)
    logical = os.cpu_count() or run.thread_defaults.cpu_budget
    physical = psutil.cpu_count(logical=False)
    return SweepOptions(
        run=run,
        clips_per_stream=clips_per_stream,
        min_step_seconds=min_step_seconds,
        thread_set=(
            _parse_int_set(threads, "--threads") if threads else DEFAULT_THREAD_SET
        ),
        groups=groups,
        max_steps=max_steps,
        rules=rules or StopRules(),
        logical_cpus=logical,
        physical_cpus=physical
        if isinstance(physical, int) and physical > 0
        else logical,
    )


def _guarded(action: Callable[[], int]) -> None:
    """Run a command body, mapping input errors and Ctrl+C to exit codes."""
    try:
        code = action()
    except (ConfigError, WorkloadError, ManifestError) as exc:
        typer.secho(f"ERROR: {exc}", err=True, fg="red")
        raise typer.Exit(code=_EXIT_USAGE) from exc
    except KeyboardInterrupt as exc:
        typer.secho(
            "\nInterrupted; children were terminated. Completed results are kept.",
            err=True,
        )
        raise typer.Exit(code=_EXIT_INTERRUPTED) from exc
    raise typer.Exit(code=int(code or 0))


@app.command("run")
def run_command(  # noqa: PLR0913 - a benchmark CLI exposes its knobs as options
    *,
    clips_dir: ClipsDirOpt,
    out_dir: OutDirOpt,
    max_clips: MaxClipsOpt = DEFAULT_MAX_CLIPS,
    long_files: Annotated[
        list[Path] | None,
        typer.Option(
            "--long-files",
            help="Long WAV file for the -long workload (repeat the option).",
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    manifest: ManifestOpt = None,
    language: LanguageOpt = DEFAULT_LANGUAGE,
    configs: Annotated[
        str,
        typer.Option(
            help=(
                "Comma-separated configuration names or globs over the default "
                "matrix, e.g. 'whisper-cpu-*,gigaam-cuda-B8'. Empty: default "
                "matrix. Grammar: <whisper|gigaam|gigaam-ctc>-<cpu|cuda>-<P|S|B><N>"
                "[x<threads>][-long]. See the 'list' command."
            ),
        ),
    ] = "",
    repeat: Annotated[
        int,
        typer.Option(min=1, help="Repeats; odd repeats run the configs in reverse."),
    ] = 1,
    clear: Annotated[
        bool,
        typer.Option(help="Delete previous results/logs/summary in --out-dir first."),
    ] = False,
    retry_failed: RetryOpt = False,
    max_config_seconds: MaxSecondsOpt = DEFAULT_MAX_CONFIG_SECONDS,
    min_free_commit_gb: MinCommitOpt = DEFAULT_MIN_FREE_COMMIT_GB,
    min_free_vram_mb: MinVramOpt = DEFAULT_MIN_FREE_VRAM_MB,
    job_commit_limit_gb: JobLimitOpt = None,
    commit_reserve_gb: ReserveOpt = DEFAULT_COMMIT_RESERVE_GB,
    allow_uncapped: AllowUncappedOpt = False,
    estimate: EstimateOpt = None,
    cpu_budget: CpuBudgetOpt = None,
    cuda_threads: CudaThreadsOpt = DEFAULT_CUDA_THREADS,
    whisper_model: WhisperModelOpt = DEFAULT_WHISPER_MODEL,
    beam_size: BeamSizeOpt = DEFAULT_BEAM_SIZE,
    whisper_option: WhisperOptionOpt = None,
    whisper_chunk_s: ChunkOpt = DEFAULT_WHISPER_CHUNK_S,
    warmup_clip: WarmupOpt = None,
    sample_interval: SampleOpt = DEFAULT_SAMPLE_INTERVAL_S,
    cooldown: CooldownOpt = DEFAULT_COOLDOWN_S,
    gigaam_cuda_streams: StreamsOpt = True,
    dry_run: DryRunOpt = False,
) -> None:
    """Run the benchmark matrix (or a selection) and write the summary."""

    def _action() -> int:
        options = _build_run_options(
            clips_dir=clips_dir,
            out_dir=out_dir,
            max_clips=max_clips,
            manifest=manifest,
            language=language,
            retry_failed=retry_failed,
            max_config_seconds=max_config_seconds,
            min_free_commit_gb=min_free_commit_gb,
            min_free_vram_mb=min_free_vram_mb,
            job_commit_limit_gb=job_commit_limit_gb,
            commit_reserve_gb=commit_reserve_gb,
            allow_uncapped=allow_uncapped,
            estimate=estimate,
            cpu_budget=cpu_budget,
            cuda_threads=cuda_threads,
            whisper_model=whisper_model,
            beam_size=beam_size,
            whisper_option=whisper_option,
            whisper_chunk_s=whisper_chunk_s,
            warmup_clip=warmup_clip,
            sample_interval=sample_interval,
            cooldown=cooldown,
            gigaam_cuda_streams=gigaam_cuda_streams,
            dry_run=dry_run,
            selection=configs,
            repeat=repeat,
            clear=clear,
            long_files=long_files,
        )
        return run_benchmark(options)

    _guarded(_action)


@app.command("sweep")
def sweep_command(  # noqa: PLR0913 - a benchmark CLI exposes its knobs as options
    *,
    clips_dir: ClipsDirOpt,
    out_dir: OutDirOpt,
    max_clips: MaxClipsOpt = DEFAULT_MAX_CLIPS,
    manifest: ManifestOpt = None,
    language: LanguageOpt = DEFAULT_LANGUAGE,
    groups: Annotated[
        str,
        typer.Option(
            help=(
                "Comma-separated group names or globs, e.g. "
                "'gigaam-cpu-S-T2,gigaam-cuda-B' (empty: all; see --dry-run)."
            ),
        ),
    ] = "",
    threads: Annotated[
        str,
        typer.Option(help="Thread counts T of the CPU scan groups, e.g. '1,2,4'."),
    ] = ",".join(str(t) for t in DEFAULT_THREAD_SET),
    max_steps: Annotated[
        int,
        typer.Option(
            min=0, help="Keep only the first N steps of every group (0: all)."
        ),
    ] = 0,
    clips_per_stream: ClipsPerStreamOpt = DEFAULT_CLIPS_PER_STREAM,
    min_step_seconds: MinStepOpt = DEFAULT_MIN_STEP_SECONDS,
    min_gain: Annotated[
        float,
        typer.Option(help="Gain over the best earlier step that counts as progress."),
    ] = 0.05,
    patience: Annotated[
        int,
        typer.Option(min=1, help="Steps without progress that end a scan group."),
    ] = 2,
    max_drop: Annotated[
        float,
        typer.Option(help="Drop below the best earlier step that ends a scan group."),
    ] = 0.10,
    max_wer_delta: Annotated[
        float,
        typer.Option(help="WER increase (pp) vs the group's first step that stops it."),
    ] = 0.5,
    min_similarity: Annotated[
        float,
        typer.Option(help="Mean similarity vs the first step below which it stops."),
    ] = 0.97,
    knee_tolerance: Annotated[
        float,
        typer.Option(help="Distance to the best step for the 'efficient' knee."),
    ] = 0.05,
    retry_failed: RetryOpt = False,
    max_config_seconds: MaxSecondsOpt = DEFAULT_MAX_CONFIG_SECONDS,
    min_free_commit_gb: MinCommitOpt = DEFAULT_MIN_FREE_COMMIT_GB,
    min_free_vram_mb: MinVramOpt = DEFAULT_MIN_FREE_VRAM_MB,
    job_commit_limit_gb: JobLimitOpt = None,
    commit_reserve_gb: ReserveOpt = DEFAULT_COMMIT_RESERVE_GB,
    allow_uncapped: AllowUncappedOpt = False,
    estimate: EstimateOpt = None,
    cpu_budget: CpuBudgetOpt = None,
    cuda_threads: CudaThreadsOpt = DEFAULT_CUDA_THREADS,
    whisper_model: WhisperModelOpt = DEFAULT_WHISPER_MODEL,
    beam_size: BeamSizeOpt = DEFAULT_BEAM_SIZE,
    whisper_option: WhisperOptionOpt = None,
    whisper_chunk_s: ChunkOpt = DEFAULT_WHISPER_CHUNK_S,
    warmup_clip: WarmupOpt = None,
    sample_interval: SampleOpt = DEFAULT_SAMPLE_INTERVAL_S,
    cooldown: CooldownOpt = DEFAULT_COOLDOWN_S,
    gigaam_cuda_streams: StreamsOpt = True,
    dry_run: DryRunOpt = False,
) -> None:
    """Search the saturation knee of every group (resumable; see --dry-run)."""

    def _action() -> int:
        run = _build_run_options(
            clips_dir=clips_dir,
            out_dir=out_dir,
            max_clips=max_clips,
            manifest=manifest,
            language=language,
            retry_failed=retry_failed,
            max_config_seconds=max_config_seconds,
            min_free_commit_gb=min_free_commit_gb,
            min_free_vram_mb=min_free_vram_mb,
            job_commit_limit_gb=job_commit_limit_gb,
            commit_reserve_gb=commit_reserve_gb,
            allow_uncapped=allow_uncapped,
            estimate=estimate,
            cpu_budget=cpu_budget,
            cuda_threads=cuda_threads,
            whisper_model=whisper_model,
            beam_size=beam_size,
            whisper_option=whisper_option,
            whisper_chunk_s=whisper_chunk_s,
            warmup_clip=warmup_clip,
            sample_interval=sample_interval,
            cooldown=cooldown,
            gigaam_cuda_streams=gigaam_cuda_streams,
            dry_run=dry_run,
        )
        options = _sweep_options(
            run,
            clips_per_stream=clips_per_stream,
            min_step_seconds=min_step_seconds,
            threads=threads,
            groups=groups,
            max_steps=max_steps,
            rules=StopRules(
                min_gain=min_gain,
                patience=patience,
                max_drop=max_drop,
                max_wer_delta_pp=max_wer_delta,
                min_similarity_mean=min_similarity,
                knee_tolerance=knee_tolerance,
            ),
        )
        return run_sweep(options)

    _guarded(_action)


@app.command("combo")
def combo_command(  # noqa: PLR0913 - a benchmark CLI exposes its knobs as options
    *,
    clips_dir: ClipsDirOpt,
    out_dir: OutDirOpt,
    pairs: Annotated[
        str,
        typer.Option(
            help=(
                "Comma-separated pairs '<config>+<config>', e.g. "
                "'gigaam-cuda-B16+gigaam-cpu-S8x2'."
            ),
        ),
    ] = "",
    auto_combos: Annotated[
        bool,
        typer.Option(help="Add GPU x CPU pairs of the sweep knees from --out-dir."),
    ] = False,
    knee: Annotated[
        str,
        typer.Option(help="Knee used by --auto-combos: best or efficient."),
    ] = "best",
    max_clips: MaxClipsOpt = DEFAULT_MAX_CLIPS,
    manifest: ManifestOpt = None,
    language: LanguageOpt = DEFAULT_LANGUAGE,
    clips_per_stream: ClipsPerStreamOpt = DEFAULT_CLIPS_PER_STREAM,
    min_step_seconds: MinStepOpt = DEFAULT_MIN_STEP_SECONDS,
    retry_failed: RetryOpt = False,
    max_config_seconds: MaxSecondsOpt = DEFAULT_MAX_CONFIG_SECONDS,
    min_free_commit_gb: MinCommitOpt = DEFAULT_MIN_FREE_COMMIT_GB,
    min_free_vram_mb: MinVramOpt = DEFAULT_MIN_FREE_VRAM_MB,
    job_commit_limit_gb: JobLimitOpt = None,
    commit_reserve_gb: ReserveOpt = DEFAULT_COMMIT_RESERVE_GB,
    allow_uncapped: AllowUncappedOpt = False,
    estimate: EstimateOpt = None,
    cpu_budget: CpuBudgetOpt = None,
    cuda_threads: CudaThreadsOpt = DEFAULT_CUDA_THREADS,
    whisper_model: WhisperModelOpt = DEFAULT_WHISPER_MODEL,
    beam_size: BeamSizeOpt = DEFAULT_BEAM_SIZE,
    whisper_option: WhisperOptionOpt = None,
    whisper_chunk_s: ChunkOpt = DEFAULT_WHISPER_CHUNK_S,
    warmup_clip: WarmupOpt = None,
    sample_interval: SampleOpt = DEFAULT_SAMPLE_INTERVAL_S,
    cooldown: CooldownOpt = DEFAULT_COOLDOWN_S,
    gigaam_cuda_streams: StreamsOpt = True,
    dry_run: DryRunOpt = False,
) -> None:
    """Run two configurations at once on disjoint clip streams under one clock.

    Use the same --clips-dir/--max-clips/--manifest/--clips-per-stream/
    --min-step-seconds as the sweep so the solo comparison applies.
    """

    def _action() -> int:
        if knee not in _KNEES:
            msg = f"--knee must be one of {', '.join(_KNEES)}"
            raise ConfigError(msg)
        run = _build_run_options(
            clips_dir=clips_dir,
            out_dir=out_dir,
            max_clips=max_clips,
            manifest=manifest,
            language=language,
            retry_failed=retry_failed,
            max_config_seconds=max_config_seconds,
            min_free_commit_gb=min_free_commit_gb,
            min_free_vram_mb=min_free_vram_mb,
            job_commit_limit_gb=job_commit_limit_gb,
            commit_reserve_gb=commit_reserve_gb,
            allow_uncapped=allow_uncapped,
            estimate=estimate,
            cpu_budget=cpu_budget,
            cuda_threads=cuda_threads,
            whisper_model=whisper_model,
            beam_size=beam_size,
            whisper_option=whisper_option,
            whisper_chunk_s=whisper_chunk_s,
            warmup_clip=warmup_clip,
            sample_interval=sample_interval,
            cooldown=cooldown,
            gigaam_cuda_streams=gigaam_cuda_streams,
            dry_run=dry_run,
        )
        options = _sweep_options(
            run,
            clips_per_stream=clips_per_stream,
            min_step_seconds=min_step_seconds,
        )
        return run_combos(options, pairs=pairs, auto=auto_combos, knee=knee)

    _guarded(_action)


@app.command("memory-growth")
def memory_growth_command(  # noqa: PLR0913 - a benchmark CLI exposes its knobs as options
    *,
    jobs: Annotated[
        int, typer.Option(min=1, help="Jobs fed one after another to the worker.")
    ] = DEFAULT_JOBS,
    durations: Annotated[
        str,
        typer.Option(
            help=(
                f"{' or '.join(DURATION_MODES)}: varied draws each job's duration "
                "uniformly from [--duration-min-s, --duration-max-s] (seeded, ms "
                "resolution, so nearly every job has its own input shape, like "
                "voice notes); fixed gives every job --fixed-duration-s (constant "
                "input shapes, only the content differs)."
            ),
        ),
    ] = DURATION_VARIED,
    duration_min_s: Annotated[
        float, typer.Option(help="Shortest varied duration (s).")
    ] = DEFAULT_GROWTH_MIN_DURATION_S,
    duration_max_s: Annotated[
        float, typer.Option(help="Longest varied duration (s).")
    ] = DEFAULT_GROWTH_MAX_DURATION_S,
    fixed_duration_s: Annotated[
        float,
        typer.Option(help="Duration of every fixed job (s; below 20 s = one window)."),
    ] = DEFAULT_GROWTH_FIXED_DURATION_S,
    seed: Annotated[
        int, typer.Option(help="Seed of the durations and source offsets.")
    ] = DEFAULT_GROWTH_SEED,
    source: Annotated[
        str,
        typer.Option(
            help=(
                f"Speech source: '{SOURCE_SAPI}' synthesizes Russian narration with "
                f"'{SAPI_VOICE}' through pwsh (PowerShell 7), or an audio/video file "
                "such as tests/fixtures/test_video_short.mp4. Clips are cut from the "
                "looped source at seeded offsets. Never pass user recordings."
            ),
        ),
    ] = SOURCE_SAPI,
    audio_format: Annotated[
        str,
        typer.Option(
            help=(
                f"Clip format: {', '.join(AUDIO_FORMATS)} (opus = Ogg Opus 16 kHz "
                "mono, like a Telegram voice note)."
            ),
        ),
    ] = AUDIO_OPUS,
    gpu_memory_bounds: Annotated[
        str,
        typer.Option(
            help=(
                f"{' or '.join(GPU_MEMORY_BOUNDS)}: pool keeps the GPU memory "
                "bounds of the pool config (gpu.cufft_plan_cache_max_size, "
                "gpu.cudnn_v8_lru_cache_limit, gpu.idle_cache_release_s, "
                "gpu.vram_cap); library switches all four off (cuFFT and cuDNN "
                "plan caches at the torch defaults, no idle release, no VRAM cap), "
                "the behavior before the bounds existed. --pool-set wins over both."
            ),
        ),
    ] = BOUNDS_POOL,
    pool_config: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Base pool config JSON. Default: the code defaults; stt_pool.json "
                "and ASKVLM_STT_POOL_CONFIG are never read."
            ),
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    pool_set: Annotated[
        list[str] | None,
        typer.Option(
            help=(
                "Override a pool config key: PATH=VALUE with a dotted PATH and a "
                "JSON VALUE, e.g. gpu.cufft_plan_cache_max_size=0, "
                "gpu.batch_lanes.gigaam=8, memory.recycle_commit_fraction=0.9 "
                "(repeat the option; validated like stt_pool.json)."
            ),
        ),
    ] = None,
    reservation_commit_gib: Annotated[
        float | None,
        typer.Option(
            help=(
                "Commit reservation (GiB) instead of the seed estimate; the Job "
                "Object cap adds memory.child_headroom_gib."
            ),
        ),
    ] = None,
    reservation_vram_gib: Annotated[
        float | None,
        typer.Option(
            help=(
                "VRAM reservation (GiB) instead of the seed estimate; the VRAM cap "
                "adds gpu.vram_cap_margin_gib."
            ),
        ),
    ] = None,
    idle_gap_s: Annotated[
        float,
        typer.Option(
            help=(
                "Idle pause (s) after every --idle-gap-every-th job; longer than "
                "gpu.idle_cache_release_s lets the worker release its CUDA cache. "
                "0: back to back."
            ),
        ),
    ] = 0.0,
    idle_gap_every: Annotated[
        int, typer.Option(min=1, help="Pause after every N-th job.")
    ] = 1,
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out-dir",
            help=(
                "Parent of the run directory <UTC stamp>-<label> (default inside "
                "the gitignored .cache)."
            ),
            file_okay=False,
            resolve_path=True,
        ),
    ] = DEFAULT_OUT_DIR,
    label: Annotated[
        str, typer.Option(help="Run label (A-Z, a-z, 0-9, '.', '_', '-').")
    ] = DEFAULT_GROWTH_LABEL,
    keep_inputs: Annotated[
        bool, typer.Option(help="Keep the generated clips in <run dir>/inputs.")
    ] = False,
    min_free_commit_gib: Annotated[
        float,
        typer.Option(help="Refuse to start below this free system commit (GiB)."),
    ] = DEFAULT_MIN_FREE_COMMIT_GIB,
    min_free_vram_gib: Annotated[
        float,
        typer.Option(help="Refuse to start below this free VRAM (GiB; 0: no check)."),
    ] = DEFAULT_MIN_FREE_VRAM_GIB,
    commit_reserve_gib: Annotated[
        float,
        typer.Option(
            help=(
                "Stop the run when free system commit drops below this (GiB); "
                "the guard also refuses to start when the worker's Job Object "
                "cap does not fit into free commit minus this reserve."
            ),
        ),
    ] = DEFAULT_COMMIT_RESERVE_GIB,
    job_timeout_s: Annotated[
        float, typer.Option(help="Longest wait for one job's result (s).")
    ] = DEFAULT_JOB_TIMEOUT_S,
    load_timeout_s: Annotated[
        float, typer.Option(help="Longest model load and warm-up (s).")
    ] = DEFAULT_LOAD_TIMEOUT_S,
    max_errors: Annotated[
        int, typer.Option(min=0, help="Stop after more failed jobs than this.")
    ] = DEFAULT_MAX_ERRORS,
    gpu_sample: Annotated[
        bool,
        typer.Option(help="Record whole-GPU used VRAM (nvidia-smi) after each job."),
    ] = True,
    probe_stack: Annotated[
        bool,
        typer.Option(
            help=(
                "Record torch/CUDA/cuDNN/transformers versions in run.json (a "
                "short torch child, run only after the guard passed)."
            ),
        ),
    ] = True,
    plateau_tolerance_mib: Annotated[
        float,
        typer.Option(
            help=(
                "A figure plateaus when the median of the last 20% of the jobs "
                "exceeds the median of the 20% before them by at most this "
                "tolerance (MiB)."
            ),
        ),
    ] = DEFAULT_PLATEAU_TOLERANCE_MIB,
    progress_every: Annotated[
        int, typer.Option(min=0, help="Progress line every N jobs (0: none).")
    ] = DEFAULT_PROGRESS_EVERY,
    engine_factory: Annotated[
        str,
        typer.Option(hidden=True, help="Engine factory of the worker spec (tests)."),
    ] = "",
    dry_run: DryRunOpt = False,
) -> None:
    """Track the memory of one long-lived GigaAM GPU pool worker job by job.

    Starts ONE real pool worker (production launch path, own Job Object,
    GigaamGpuBatchEngine) for gigaam-e2e-ctc/ai-sage/GigaAM-v3@e2e_ctc/fp32/gpu,
    feeds --jobs synthetic jobs one after another and writes per job the
    worker's memory snapshot (private bytes, CUDA reserved/allocated/peak,
    cuFFT plans), the Job Object peak and timings to
    <out-dir>/<stamp>-<label>/results.jsonl and results.csv, then
    summary.json/summary.md: baseline after load, slope per job and per hour
    over the last 80% of the jobs, plateau, maxima, cuFFT and CUDA
    trajectories, throughput, and where the pool's recycle threshold would
    have retired the worker. Runs one real model: check free commit and VRAM
    first (the guard refuses below --min-free-commit-gib/--min-free-vram-gib,
    and when the worker's Job Object cap does not fit into free commit minus
    --commit-reserve-gib; the commit rules are checked again after the inputs
    are cut, right before the worker spawn).
    Exit codes: 0 all jobs ran and the summary was written; 1 the run ended
    early; 2 invalid input, or the inputs could not be built (status
    inputs_failed; the run folder keeps run.json and the summary); 3 the guard
    refused (at the start, before any probe, or right before the worker
    spawn); 4 all jobs ran but the summary could not be written; 130 Ctrl+C.
    """

    def _action() -> int:
        options = GrowthOptions(
            jobs=jobs,
            durations=DurationPlan(
                mode=durations,
                min_s=duration_min_s,
                max_s=duration_max_s,
                fixed_s=fixed_duration_s,
                seed=seed,
            ),
            source=source,
            audio_format=audio_format,
            out_dir=out_dir,
            label=label,
            pool_config=pool_config,
            gpu_memory_bounds=gpu_memory_bounds,
            pool_set=tuple(pool_set or ()),
            reservation_commit_gib=reservation_commit_gib,
            reservation_vram_gib=reservation_vram_gib,
            idle_gap_s=idle_gap_s,
            idle_gap_every=idle_gap_every,
            keep_inputs=keep_inputs,
            min_free_commit_gib=min_free_commit_gib,
            min_free_vram_gib=min_free_vram_gib,
            commit_reserve_gib=commit_reserve_gib,
            job_timeout_s=job_timeout_s,
            load_timeout_s=load_timeout_s,
            max_errors=max_errors,
            gpu_sample=gpu_sample,
            probe_stack=probe_stack,
            plateau_tolerance_mib=plateau_tolerance_mib,
            progress_every=progress_every,
            dry_run=dry_run,
            engine_factory=engine_factory or DEFAULT_ENGINE_FACTORY,
        )
        return run_memory_growth(options)

    _guarded(_action)


@app.command("list")
def list_command(
    *,
    cpu_budget: CpuBudgetOpt = None,
    cuda_threads: CudaThreadsOpt = DEFAULT_CUDA_THREADS,
    with_long: Annotated[
        bool,
        typer.Option(help="Include the long-file configurations."),
    ] = True,
) -> None:
    """Print the default configuration matrix of 'run' with guard estimates."""
    defaults = ThreadDefaults(
        cpu_budget=cpu_budget or _default_cpu_budget(),
        cuda_threads=cuda_threads,
    )
    typer.echo(
        f"CPU budget {defaults.cpu_budget}, CUDA threads {defaults.cuda_threads}",
    )
    for config in default_matrix(defaults, has_long=with_long):
        need = estimate_need(config, DEFAULT_ESTIMATES)
        typer.echo(
            f"  {config.name:<26} commit~{need.commit_gb:5.1f} GB"
            f"  vram~{need.vram_gb:4.1f} GB",
        )


@app.command("summarize")
def summarize_command(
    *,
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out-dir",
            help="Benchmark output directory, or one memory-growth run directory.",
            file_okay=False,
        ),
    ],
    plateau_tolerance_mib: Annotated[
        float,
        typer.Option(help="Plateau tolerance of a memory-growth run (MiB)."),
    ] = DEFAULT_PLATEAU_TOLERANCE_MIB,
) -> None:
    """Rebuild the summaries in --out-dir from the newest fingerprint of each kind.

    A memory-growth run directory (one that holds run.json) gets its
    summary.json, summary.md and results.csv recomputed instead.
    """
    if not out_dir.is_dir():
        typer.secho(f"ERROR: --out-dir does not exist: {out_dir}", err=True, fg="red")
        raise typer.Exit(code=_EXIT_USAGE)
    if is_growth_run(out_dir):
        for line in rebuild_summary(
            out_dir, plateau_tolerance_mib=plateau_tolerance_mib
        ):
            typer.echo(line)
        return
    run_fp = latest_fingerprint(report.load_results(out_dir))
    md_path, table = report.write_summary(out_dir, run_fp)
    typer.echo(table)
    typer.echo(f"\nSummary: {md_path} (fingerprint {run_fp})")
    sweep_fp = latest_fingerprint(load_group_documents(out_dir, None))
    if sweep_fp is not None:
        sweep_md, _knees = write_sweep_summary(out_dir, sweep_fp)
        typer.echo(f"Sweep summary: {sweep_md} (fingerprint {sweep_fp})")
    combo_store = ResultStore(out_dir / COMBO_DIRNAME / RESULTS_DIRNAME, "")
    combo_fp = latest_fingerprint(combo_store.all_documents())
    if combo_fp is not None:
        combo_md, _table = write_combo_summary(out_dir, combo_fp)
        typer.echo(f"Combo summary: {combo_md} (fingerprint {combo_fp})")
