"""Saturation search (``sweep``): raise concurrency per group until the knee.

Groups and steps (defaults; ``--groups`` selects by name or glob):

* CPU scan, per engine, per T in ``--threads`` (default 1, 2, 4), modes P and S:
  ``N = 1, 2, 3, 4, 6, 8, 12, 16, 24, 32`` with ``N x T <= logical CPUs``
  (``<engine>-cpu-<P|S>-T<T>``).
* CPU constant budget, per engine: ``N x T = cpu budget`` with N = 1, 2, 4, 8
  for P and S (``<engine>-cpu-<P|S>-budget<B>``); only hard stops apply.
* CPU Whisper cross-file batching: T = cpu budget, batch 1, 2, 4, 8, 16
  (``whisper-cpu-B-T<B>``).
* CUDA per engine: P 1, 2, 3; S 1, 2, 4, 8; B 1, 2, 4, 8, 16, 32 (Whisper) or
  1 ... 64 (GigaAM batched windows).

An ``S1`` step and a GigaAM ``B1`` step are the same measurement as ``P1`` (one
process, one model, one item at a time), so those groups start from the shared
``P1`` result. Whisper ``B1`` is measured on its own: the batched decode path
differs from the sequential one.

Workload per step: the clips are cycled until at least ``clips_per_stream x N``
items were dispatched and ``min_step_seconds`` elapsed on the measured clock;
repeated clips keep their identity, so quality compares clip by clip.

Stop rules, checked after every step (first match wins): guard refusal; the
step job's commit cap (``memory_cap``, a saturation point); the system free
commit below the reserve (``system_commit_low``); CUDA out of memory (the
saturation point, recorded, the run continues with the next group); timeout;
errors; quality (WER delta versus the group's first step above
``max_wer_delta_pp``, or mean similarity below ``min_similarity_mean``); then,
for scan groups only, a throughput drop of more than ``max_drop`` below the best
earlier step, or a gain below ``min_gain`` over the best earlier step for
``patience`` consecutive steps.

Knee: the best step is the highest throughput among successful steps; the
efficient step is the smallest N within ``knee_tolerance`` of the best. A step
that triggers the quality rule stays in the curve marked ``rejected`` and is
never a knee (nor is any later step).

Resume: step results are keyed by configuration name; a result is reused when
its fingerprint matches (any final status except ``skipped``, which the guard
re-decides), so an interrupted sweep continues with its next unfinished step.
"""

from __future__ import annotations

import fnmatch
import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .configs import CPU_ONLY_ENGINES, ENGINES, BenchConfig
from .paths import LOGS_DIRNAME, RESULTS_DIRNAME
from .quality import compare_items
from .report import atomic_write, gb, host_line, render_table
from .runner import (
    REUSABLE_STATUSES,
    STATUS_MEMORY_CAP,
    STATUS_SYSTEM_COMMIT_LOW,
    Measurement,
    SideSpec,
    collect_host_info,
    document_base,
    echo,
    evaluate_guard,
    fingerprint_payload,
    is_cuda_oom,
    load_workload,
    now_iso,
    quality_block,
    result_line,
    reusable_result,
    single_side_document,
)
from .store import ResultStore, fingerprint_of, read_json, write_json
from .sysmetrics import KillOnCloseJob, find_nvidia_smi
from .workload import WorkStream

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from .quality import Manifest
    from .runner import RunOptions, Workload

JsonDict = dict[str, Any]

SWEEP_DIRNAME = "sweep"
GROUPS_DIRNAME = "groups"
SUMMARY_MD = "summary.md"
SUMMARY_JSON = "summary.json"

CPU_SCAN_WIDTHS: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
CPU_BUDGET_WIDTHS: tuple[int, ...] = (1, 2, 4, 8)
WHISPER_CPU_BATCHES: tuple[int, ...] = (1, 2, 4, 8, 16)
GPU_PROCESS_WIDTHS: tuple[int, ...] = (1, 2, 3)
GPU_THREAD_WIDTHS: tuple[int, ...] = (1, 2, 4, 8)
WHISPER_GPU_BATCHES: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
GIGAAM_GPU_BATCHES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_THREAD_SET: tuple[int, ...] = (1, 2, 4)
# * Hard cap of one step's workload, in passes over the clip list.
MAX_STEP_PASSES = 50
# * Statuses that stop a group by themselves, with the stop-reason prefix.
_STATUS_STOP_PREFIX: dict[str, str] = {
    "skipped": "guard",
    STATUS_MEMORY_CAP: "memory_cap",
    STATUS_SYSTEM_COMMIT_LOW: "system_commit_low",
}
_NOTE_LIMIT = 90
# * Prefix of stop reasons from the quality rule (such steps are never a knee).
QUALITY_STOP_PREFIX = "quality"

# * Seed model for plan estimates only (rough, from the reference PC).
SEED_CPU_XRT_PER_THREAD: dict[str, float] = {
    "whisper": 1.5,
    "gigaam": 6.0,
    "gigaam-ctc": 6.0,
}
SEED_SMT_BONUS = 0.25
SEED_CUDA_XRT: dict[str, float] = {"whisper": 22.0, "gigaam": 32.0}
SEED_CUDA_GAIN: dict[str, float] = {"P": 1.4, "S": 1.5, "B": 3.0}
SEED_CPU_BATCH_GAIN = 1.3
SEED_LOAD_S: dict[str, float] = {"whisper": 5.0, "gigaam": 9.0, "gigaam-ctc": 9.0}
SEED_LOAD_CONTENTION = 0.15
SEED_WARMUP_S = 3.0
SEED_OVERHEAD_S = 4.0
SEED_CLIP_S = 11.6


@dataclass(frozen=True, slots=True)
class StopRules:
    """Thresholds of the saturation search.

    Attributes:
        min_gain: Relative gain over the best earlier step that counts as progress.
        patience: Consecutive steps without progress that end a scan group.
        max_drop: Relative drop below the best earlier step that ends a scan group.
        max_wer_delta_pp: WER increase (percentage points) versus the first step.
        min_similarity_mean: Lowest mean transcript similarity versus the first step.
        knee_tolerance: Relative distance to the best step for the efficient knee.

    """

    min_gain: float = 0.05
    patience: int = 2
    max_drop: float = 0.10
    max_wer_delta_pp: float = 0.5
    min_similarity_mean: float = 0.97
    knee_tolerance: float = 0.05


@dataclass(frozen=True, slots=True)
class SweepOptions:
    """Resolved options of ``sweep`` (and the workload sizing used by ``combo``)."""

    run: RunOptions
    clips_per_stream: int
    min_step_seconds: float
    thread_set: tuple[int, ...]
    groups: str
    max_steps: int
    rules: StopRules
    logical_cpus: int
    physical_cpus: int


@dataclass(frozen=True, slots=True)
class SweepGroup:
    """One curve of the saturation search.

    Attributes:
        name: Group name.
        kind: ``scan`` (all stop rules) or ``budget`` (hard stops only).
        steps: Configurations in increasing concurrency.
        cut: Why the series is shorter than the full width list, if it is.

    """

    name: str
    kind: str
    steps: tuple[BenchConfig, ...]
    cut: str = ""


def _step(engine: str, device: str, mode: str, width: int, threads: int) -> BenchConfig:
    """Return a step; a width-1 ``S`` or GigaAM ``B`` step is the ``P1`` one."""
    if width == 1 and (mode == "S" or (mode == "B" and engine != "whisper")):
        mode = "P"
    return BenchConfig.from_json(
        {
            "engine": engine,
            "device": device,
            "mode": mode,
            "width": width,
            "threads": threads,
        },
    )


def build_groups(options: SweepOptions) -> list[SweepGroup]:
    """Build every sweep group, then apply ``--groups`` and ``--max-steps``.

    Args:
        options: Sweep options.

    Returns:
        Groups in execution order (CUDA first, then CPU).

    """
    defaults = options.run.thread_defaults
    budget = defaults.cpu_budget
    cuda_threads = defaults.cuda_threads
    groups: list[SweepGroup] = []
    for engine in ENGINES:
        if engine in CPU_ONLY_ENGINES:
            continue
        batches = WHISPER_GPU_BATCHES if engine == "whisper" else GIGAAM_GPU_BATCHES
        for mode, gpu_widths in (
            ("P", GPU_PROCESS_WIDTHS),
            ("S", GPU_THREAD_WIDTHS),
            ("B", batches),
        ):
            steps = tuple(
                _step(engine, "cuda", mode, w, cuda_threads) for w in gpu_widths
            )
            groups.append(SweepGroup(f"{engine}-cuda-{mode}", "scan", steps))
    for engine in ENGINES:
        for mode in ("P", "S"):
            steps = tuple(
                _step(engine, "cpu", mode, w, budget // w)
                for w in CPU_BUDGET_WIDTHS
                if budget // w >= 1
            )
            groups.append(
                SweepGroup(f"{engine}-cpu-{mode}-budget{budget}", "budget", steps)
            )
    for engine in ENGINES:
        for mode in ("P", "S"):
            for threads in options.thread_set:
                widths = [
                    w for w in CPU_SCAN_WIDTHS if w * threads <= options.logical_cpus
                ]
                steps = tuple(_step(engine, "cpu", mode, w, threads) for w in widths)
                if steps:
                    groups.append(
                        SweepGroup(
                            f"{engine}-cpu-{mode}-T{threads}",
                            "scan",
                            steps,
                            cut=(
                                "N x T <= logical CPUs"
                                if len(widths) < len(CPU_SCAN_WIDTHS)
                                else ""
                            ),
                        ),
                    )
    groups.append(
        SweepGroup(
            f"whisper-cpu-B-T{budget}",
            "scan",
            tuple(_step("whisper", "cpu", "B", b, budget) for b in WHISPER_CPU_BATCHES),
        ),
    )
    return _select_groups(groups, options)


def _select_groups(
    groups: Sequence[SweepGroup], options: SweepOptions
) -> list[SweepGroup]:
    tokens = [token.strip() for token in options.groups.split(",") if token.strip()]
    selected = [
        group
        for group in groups
        if not tokens or any(fnmatch.fnmatchcase(group.name, t) for t in tokens)
    ]
    if options.max_steps > 0:
        selected = [
            SweepGroup(
                g.name,
                g.kind,
                g.steps[: options.max_steps],
                "--max-steps" if len(g.steps) > options.max_steps else g.cut,
            )
            for g in selected
        ]
    return selected


def sweep_fingerprint(options: SweepOptions, workload: Workload) -> str:
    """Fingerprint of sweep and combo results (workload, decode, sizing).

    Args:
        options: Sweep options.
        workload: Resolved workload.

    Returns:
        Fingerprint string.

    """
    payload = fingerprint_payload(options.run, workload)
    payload["sweep"] = {
        "clips_per_stream": options.clips_per_stream,
        "min_step_seconds": options.min_step_seconds,
    }
    return fingerprint_of(payload)


def step_stream(
    clips: Sequence[Any],
    config: BenchConfig,
    options: SweepOptions,
    *,
    first_id: int = 0,
) -> WorkStream:
    """Return the work stream of one step (or one combo side).

    Args:
        clips: Clip list to cycle.
        config: Configuration (its width scales the minimum size).
        options: Sweep options.
        first_id: Item id of the first occurrence.

    Returns:
        A stream with ``clips_per_stream x width`` items and ``min_step_seconds``.

    """
    min_items = options.clips_per_stream * config.width
    return WorkStream(
        clips,
        min_items=min_items,
        min_wall_s=options.min_step_seconds,
        max_items=max(min_items, MAX_STEP_PASSES * len(clips)),
        first_id=first_id,
    )


# * Step metrics and stop rules.


def step_metrics(
    document: Mapping[str, Any],
    reference: Mapping[str, Any] | None,
    manifest: Manifest | None,
) -> JsonDict:
    """Summarize one step result for the curve and the stop rules.

    Args:
        document: Step result.
        reference: The group's first step result, if any.
        manifest: Reference manifest for the WER delta.

    Returns:
        Flat metrics of the step.

    """
    config = document["config"]
    timing = document.get("timing") or {}
    cpu = document.get("cpu") or {}
    memory = document.get("memory") or {}
    vram = document.get("vram") or {}
    quality = document.get("quality") or {}
    status = document.get("status")
    comparison: JsonDict = {}
    if reference is not None and status in {"ok", "partial"}:
        comparison = compare_items(
            document.get("items", []),
            reference.get("items", []),
            manifest,
        )
    throughput = timing.get("throughput_x")
    private_gb = gb(memory.get("peak_private_bytes"))
    return {
        "name": config["name"],
        "width": config["width"],
        "threads": config["threads"],
        "mode": config["mode"],
        "status": status,
        "reason": document.get("reason"),
        "throughput_x": throughput,
        "wall_s": timing.get("wall_s"),
        "items": timing.get("items_processed"),
        "latency_p50_s": timing.get("latency_p50_s"),
        "latency_p95_s": timing.get("latency_p95_s"),
        "cpu_system_percent": cpu.get("system_busy_percent"),
        "worker_cores": cpu.get("worker_cores_avg"),
        "peak_private_gb": private_gb,
        "peak_working_set_gb": gb(memory.get("peak_working_set_bytes")),
        "vram_delta_mb": vram.get("delta_mb"),
        "gpu_util_percent": vram.get("util_avg_run_percent"),
        "wer_percent": (quality.get("wer") or {}).get("wer_percent"),
        "wer_delta_pp": comparison.get("wer_delta_pp"),
        "identical_percent": comparison.get("identical_percent"),
        "similarity_mean": comparison.get("similarity_mean"),
        "similarity_min": comparison.get("similarity_min"),
        "repeat_consistency_percent": quality.get("repeat_consistency_percent"),
        "error_count": document.get("error_count") or 0,
        "cuda_oom": is_cuda_oom(document),
        "gb_per_xrt": (
            private_gb / throughput if private_gb is not None and throughput else None
        ),
    }


def _ok(step: Mapping[str, Any]) -> bool:
    return step.get("status") == "ok" and step.get("throughput_x") is not None


def _plateau_streak(history: Sequence[Mapping[str, Any]], min_gain: float) -> int:
    """Count trailing steps that gained less than *min_gain* over earlier bests."""
    streak = 0
    for position in range(len(history) - 1, 0, -1):
        prior = [h["throughput_x"] for h in history[:position] if _ok(h)]
        current = history[position]
        if not prior or not _ok(current):
            break
        if current["throughput_x"] >= max(prior) * (1.0 + min_gain):
            break
        streak += 1
    return streak


def evaluate_stop(
    history: Sequence[Mapping[str, Any]],
    rules: StopRules,
    *,
    scan: bool,
) -> str | None:
    """Return why the group stops after its newest step, or ``None`` to continue.

    Args:
        history: Step metrics in execution order (newest last).
        rules: Stop thresholds.
        scan: Apply the throughput rules (``False`` for budget groups).

    Returns:
        Stop reason, or ``None``.

    """
    last = history[-1]
    stop = _hard_stop(last) or _quality_stop(last, rules)
    if stop is None and scan:
        stop = _throughput_stop(history, rules)
    return stop


def _hard_stop(last: Mapping[str, Any]) -> str | None:
    """Guard refusal, memory cap, low system commit, CUDA OOM, timeout, or errors."""
    reason = last.get("reason")
    status = last.get("status")
    prefix = _STATUS_STOP_PREFIX.get(str(status))
    if prefix is not None:
        suffix = " (saturation point)" if status == STATUS_MEMORY_CAP else ""
        return f"{prefix}: {reason}{suffix}"
    if last.get("cuda_oom"):
        return "cuda_oom: CUDA out of memory (saturation point)"
    if status == "timeout":
        return f"timeout: {reason}"
    if status != "ok" or last.get("error_count"):
        return f"errors: {reason}"
    return None


def _quality_stop(last: Mapping[str, Any], rules: StopRules) -> str | None:
    """WER delta or similarity of the newest step versus the group's first step."""
    wer_delta = last.get("wer_delta_pp")
    if wer_delta is not None and wer_delta > rules.max_wer_delta_pp:
        return f"{QUALITY_STOP_PREFIX}: WER +{wer_delta:.2f} pp vs the first step"
    sim = last.get("similarity_mean")
    if sim is not None and sim < rules.min_similarity_mean:
        return f"{QUALITY_STOP_PREFIX}: mean similarity {sim:.3f} vs the first step"
    return None


def _throughput_stop(
    history: Sequence[Mapping[str, Any]],
    rules: StopRules,
) -> str | None:
    """Regression below the best earlier step, or a plateau of small gains."""
    last = history[-1]
    prior = [h["throughput_x"] for h in history[:-1] if _ok(h)]
    if not prior:
        return None
    best = max(prior)
    if last["throughput_x"] < best * (1.0 - rules.max_drop):
        return (
            f"regression: {last['throughput_x']:.1f}x < best {best:.1f}x "
            f"- {rules.max_drop:.0%}"
        )
    if _plateau_streak(history, rules.min_gain) >= rules.patience:
        return f"plateau: gain < {rules.min_gain:.0%} for {rules.patience} steps"
    return None


def knees(history: Sequence[Mapping[str, Any]], tolerance: float) -> JsonDict:
    """Return the best and the efficient step of a curve.

    Args:
        history: Step metrics.
        tolerance: Relative distance to the best throughput for "efficient".

    Steps rejected by the quality rule, and every step after the first such
    step, are never eligible.

    Returns:
        ``best`` and ``efficient`` step names with their throughput.

    """
    eligible: list[Mapping[str, Any]] = []
    for step in history:
        if step.get("rejected"):
            break
        eligible.append(step)
    good = [h for h in eligible if _ok(h)]
    if not good:
        return {"best": None, "efficient": None}
    best = max(good, key=lambda h: h["throughput_x"])
    floor = best["throughput_x"] * (1.0 - tolerance)
    efficient = min(
        (h for h in good if h["throughput_x"] >= floor),
        key=lambda h: (h["width"], h["throughput_x"]),
    )
    return {
        "best": best["name"],
        "best_xrt": best["throughput_x"],
        "efficient": efficient["name"],
        "efficient_xrt": efficient["throughput_x"],
        "efficient_gb_per_xrt": efficient.get("gb_per_xrt"),
    }


def _annotate(history: list[JsonDict]) -> None:
    """Add the gain over earlier steps and a short note for the curve table."""
    for position, step in enumerate(history):
        prior = [h["throughput_x"] for h in history[:position] if _ok(h)]
        step["gain_percent"] = (
            100.0 * (step["throughput_x"] / max(prior) - 1.0)
            if prior and _ok(step)
            else None
        )
        note = step.get("rejected")
        if not note and step.get("status") != "ok":
            note = f"{step.get('status')}: {step.get('reason') or ''}"
        step["note"] = str(note)[:_NOTE_LIMIT] if note else None


# * Execution.


@dataclass(frozen=True, slots=True)
class _Context:
    options: SweepOptions
    workload: Workload
    store: ResultStore
    logs_dir: Path
    groups_dir: Path
    job: KillOnCloseJob
    nvidia_smi: str | None
    host: JsonDict


def _run_step(config: BenchConfig, ctx: _Context) -> JsonDict:
    """Guard and measure one step."""
    run = ctx.options.run
    document = document_base(
        config,
        repeat=0,
        workload=ctx.workload,
        options=run,
        host=ctx.host,
        stream_info={
            "clips_per_stream": ctx.options.clips_per_stream,
            "min_step_seconds": ctx.options.min_step_seconds,
        },
    )
    allowed, reason, guard = evaluate_guard(config, run, ctx.nvidia_smi)
    document["guard"] = guard
    if not allowed:
        document.update(status="skipped", reason=reason, finished_at=now_iso())
        return document
    measurement = Measurement(
        sides=[
            SideSpec(
                config=config,
                stream=step_stream(ctx.workload.clips, config, ctx.options),
            ),
        ],
        run_id=config.name,
        warmup_path=ctx.workload.warmup_path,
        options=run,
        job=ctx.job,
        logs_dir=ctx.logs_dir,
        nvidia_smi=ctx.nvidia_smi,
    )
    document.update(single_side_document(measurement.execute()))
    document["quality"] = quality_block(document["items"], ctx.workload.manifest)
    document["finished_at"] = now_iso()
    return document


def _run_group(group: SweepGroup, ctx: _Context, *, executed: list[int]) -> JsonDict:
    """Walk one group's steps until a stop rule fires; return the group document."""
    rules = ctx.options.rules
    history: list[JsonDict] = []
    reference: JsonDict | None = None
    stop_reason: str | None = None
    echo(f"== group {group.name} ({group.kind}, {len(group.steps)} step(s))")
    for config in group.steps:
        document = reusable_result(
            ctx.store,
            config.name,
            statuses=REUSABLE_STATUSES,
            options=ctx.options.run,
        )
        if document is None or (
            ctx.options.run.retry_failed and document.get("status") != "ok"
        ):
            if executed[0] and ctx.options.run.cooldown_s > 0:
                time.sleep(ctx.options.run.cooldown_s)
            echo(f"[{config.name}] {now_iso()} ({ctx.store.describe(config.name)})")
            document = _run_step(config, ctx)
            ctx.store.save(config.name, document)
            executed[0] += 1
            echo(result_line(document))
        else:
            echo(f"[{config.name}] reused ({document.get('status')})")
        if reference is None:
            reference = document
        metrics = step_metrics(document, reference, ctx.workload.manifest)
        history.append(metrics)
        stop_reason = evaluate_stop(history, rules, scan=group.kind == "scan")
        if stop_reason and stop_reason.startswith(QUALITY_STOP_PREFIX):
            # * Shown in the curve, never eligible as a knee.
            metrics["rejected"] = stop_reason
        if stop_reason:
            echo(f"  stop: {stop_reason}")
            break
    if stop_reason is None:
        stop_reason = f"series end ({group.cut})" if group.cut else "series end"
    _annotate(history)
    return {
        "fingerprint": ctx.store.fingerprint,
        "group": group.name,
        "kind": group.kind,
        "planned": [config.name for config in group.steps],
        "steps": history,
        "stop_reason": stop_reason,
        "knees": knees(history, rules.knee_tolerance),
        "finished_at": now_iso(),
    }


def run_sweep(options: SweepOptions) -> int:
    """Run (or plan) the saturation search and write the sweep summary.

    Args:
        options: Sweep options.

    Returns:
        Process exit code.

    """
    workload = load_workload(options.run)
    groups = build_groups(options)
    fingerprint = sweep_fingerprint(options, workload)
    root = options.run.out_dir / SWEEP_DIRNAME
    store = ResultStore(root / RESULTS_DIRNAME, fingerprint)
    nvidia_smi = find_nvidia_smi()
    echo(
        f"Sweep: {len(groups)} group(s), {len(workload.clips)} clips, "
        f"{options.clips_per_stream} clips/stream, >= {options.min_step_seconds:g} s "
        f"per step; fingerprint {fingerprint}",
    )
    if options.run.dry_run:
        print_plan(groups, options, workload, store, nvidia_smi)
        return 0
    logs_dir = root / LOGS_DIRNAME
    groups_dir = root / GROUPS_DIRNAME
    for directory in (store.directory, logs_dir, groups_dir):
        directory.mkdir(parents=True, exist_ok=True)
    job = KillOnCloseJob()
    if job.error:
        echo(f"WARNING: kill-on-close job unavailable ({job.error})")
    ctx = _Context(
        options=options,
        workload=workload,
        store=store,
        logs_dir=logs_dir,
        groups_dir=groups_dir,
        job=job,
        nvidia_smi=nvidia_smi,
        host=collect_host_info(options.run, nvidia_smi),
    )
    executed = [0]
    try:
        for group in groups:
            document = _run_group(group, ctx, executed=executed)
            write_json(groups_dir / f"{group.name}.json", document)
            write_sweep_summary(options.run.out_dir, fingerprint)
    finally:
        job.close()
    md_path, table = write_sweep_summary(options.run.out_dir, fingerprint)
    echo("")
    echo(table)
    echo(f"\nSweep summary: {md_path}")
    return 0


# * Plan (dry run).


def seed_throughput(config: BenchConfig, options: SweepOptions) -> float:
    """Return the seed throughput (xRT) of a configuration for plan estimates.

    Args:
        config: Configuration.
        options: Sweep options (CPU counts).

    Returns:
        Rough throughput.

    """
    if config.device == "cuda":
        base = SEED_CUDA_XRT.get(config.engine, 20.0)
        gain = SEED_CUDA_GAIN[config.mode]
        return base * (1.0 + (gain - 1.0) * (1.0 - 1.0 / config.width))
    per_thread = SEED_CPU_XRT_PER_THREAD.get(config.engine, 2.0)
    if config.mode == "B":
        threads = min(config.threads, options.physical_cpus)
        return (
            per_thread
            * threads
            * (1.0 + (SEED_CPU_BATCH_GAIN - 1.0) * (1.0 - 1.0 / config.width))
        )
    requested = config.total_threads
    physical = min(requested, options.physical_cpus)
    smt = max(0, min(requested, options.logical_cpus) - options.physical_cpus)
    return per_thread * (physical + SEED_SMT_BONUS * smt)


def seed_step_seconds(config: BenchConfig, options: SweepOptions) -> float:
    """Return the seed duration of one step, load and warm-up included.

    Args:
        config: Configuration.
        options: Sweep options.

    Returns:
        Seconds.

    """
    load = SEED_LOAD_S.get(config.engine, 8.0) * (
        1.0 + SEED_LOAD_CONTENTION * (config.processes - 1)
    )
    warmup = SEED_WARMUP_S * (2.0 if config.slots > 1 else 1.0)
    items = options.clips_per_stream * config.width
    run = max(
        options.min_step_seconds, items * SEED_CLIP_S / seed_throughput(config, options)
    )
    return load + warmup + run + SEED_OVERHEAD_S + options.run.cooldown_s


def _expected_steps(
    group: SweepGroup, allowed: Sequence[bool], options: SweepOptions
) -> int:
    """Simulate the stop rules on seed throughputs; return the expected step count."""
    history: list[JsonDict] = []
    for config, ok in zip(group.steps, allowed, strict=True):
        history.append(
            {
                "name": config.name,
                "width": config.width,
                "status": "ok" if ok else "skipped",
                "reason": None if ok else "guard",
                "throughput_x": seed_throughput(config, options) if ok else None,
                "error_count": 0,
            },
        )
        if evaluate_stop(history, options.rules, scan=group.kind == "scan"):
            return len(history)
    return len(history)


def print_plan(
    groups: Sequence[SweepGroup],
    options: SweepOptions,
    workload: Workload,
    store: ResultStore,
    nvidia_smi: str | None,
) -> None:
    """Print groups, steps, guard decisions, and the seed time estimate (read-only).

    Args:
        groups: Sweep groups.
        options: Sweep options.
        workload: Resolved workload.
        store: Step result store (read only).
        nvidia_smi: ``nvidia-smi`` path, if any.

    """
    echo(
        f"Plan for {len(workload.clips)} clips; seeds: CPU xRT/thread "
        f"{SEED_CPU_XRT_PER_THREAD}, CUDA P1 xRT {SEED_CUDA_XRT}, "
        f"load s {SEED_LOAD_S}.",
    )
    total_max = 0.0
    total_expected = 0.0
    seen: set[str] = set()
    for group in groups:
        allowed: list[bool] = []
        seconds: list[float] = []
        lines: list[str] = []
        for config in group.steps:
            reused = reusable_result(
                store,
                config.name,
                statuses=REUSABLE_STATUSES,
                options=options.run,
            )
            ok, reason, _guard = evaluate_guard(config, options.run, nvidia_smi)
            runnable = ok or reused is not None
            allowed.append(runnable)
            if reused is not None:
                state = "reuse"
            elif config.name in seen:
                state = "shared (measured once)"
            else:
                state = "run" if ok else f"skip: {reason}"
            cost = 0.0
            if reused is None and ok and config.name not in seen:
                cost = seed_step_seconds(config, options)
            seen.add(config.name)
            seconds.append(cost)
            lines.append(
                f"    {config.name:<24} seed {seed_throughput(config, options):6.1f}x "
                f"~{cost / 60.0:4.1f} min  {state}",
            )
            if not runnable:
                break
        planned = SweepGroup(group.name, group.kind, group.steps[: len(allowed)])
        expected_count = _expected_steps(planned, allowed, options)
        max_s = sum(seconds)
        expected_s = sum(seconds[:expected_count])
        total_max += max_s
        total_expected += expected_s
        echo(
            f"  {group.name} ({group.kind}): {len(lines)} step(s) max, "
            f"~{expected_count} expected; ~{expected_s / 60.0:.0f} min expected, "
            f"~{max_s / 60.0:.0f} min max",
        )
        for line in lines:
            echo(line)
    echo(
        f"Estimate (seed model, shared steps counted once): expected "
        f"~{total_expected / 3600.0:.1f} h, upper bound ~{total_max / 3600.0:.1f} h "
        "(no early stop). Nothing was written.",
    )


# * Summary.


_CURVE_COLUMNS: tuple[tuple[str, str, int], ...] = (
    ("N", "width", 0),
    ("config", "name", 0),
    ("status", "status", 0),
    ("xRT", "throughput_x", 1),
    ("gain %", "gain_percent", 1),
    ("p50 s", "latency_p50_s", 2),
    ("p95 s", "latency_p95_s", 2),
    ("CPU %", "cpu_system_percent", 0),
    ("cores", "worker_cores", 1),
    ("priv GB", "peak_private_gb", 2),
    ("WS GB", "peak_working_set_gb", 2),
    ("VRAM d MB", "vram_delta_mb", 0),
    ("GPU %", "gpu_util_percent", 0),
    ("WER %", "wer_percent", 2),
    ("dWER pp", "wer_delta_pp", 2),
    ("ident %", "identical_percent", 1),
    ("sim mean", "similarity_mean", 3),
    ("GB/xRT", "gb_per_xrt", 3),
    ("note", "note", 0),
)
_KNEE_COLUMNS: tuple[tuple[str, str, int], ...] = (
    ("group", "group", 0),
    ("kind", "kind", 0),
    ("steps", "steps", 0),
    ("best", "best", 0),
    ("best xRT", "best_xrt", 1),
    ("efficient", "efficient", 0),
    ("eff xRT", "efficient_xrt", 1),
    ("eff GB/xRT", "efficient_gb_per_xrt", 3),
    ("stop", "stop_reason", 0),
)


def load_group_documents(out_dir: Path, fingerprint: str | None) -> list[JsonDict]:
    """Return the sweep group documents (current fingerprint only when given).

    Args:
        out_dir: Benchmark output directory.
        fingerprint: Fingerprint filter, or ``None`` for all.

    Returns:
        Group documents sorted by name.

    """
    directory = out_dir / SWEEP_DIRNAME / GROUPS_DIRNAME
    if not directory.is_dir():
        return []
    documents = [read_json(path) for path in sorted(directory.glob("*.json"))]
    return [
        d
        for d in documents
        if d is not None
        and (fingerprint is None or d.get("fingerprint") == fingerprint)
    ]


def write_sweep_summary(out_dir: Path, fingerprint: str | None) -> tuple[Path, str]:
    """Write ``sweep/summary.md`` and ``sweep/summary.json``.

    Args:
        out_dir: Benchmark output directory.
        fingerprint: Aggregate only this fingerprint (``None``: all).

    Returns:
        ``(markdown_path, knee_table)``.

    """
    groups = load_group_documents(out_dir, fingerprint)
    knee_rows = [
        {
            "group": g["group"],
            "kind": g.get("kind"),
            "steps": len(g.get("steps") or []),
            "stop_reason": g.get("stop_reason"),
            **(g.get("knees") or {}),
        }
        for g in groups
    ]
    store = ResultStore(out_dir / SWEEP_DIRNAME / RESULTS_DIRNAME, fingerprint or "")
    documents = store.current() if fingerprint else store.all_documents()
    hosts = [d for d in documents if d.get("host")]
    knee_table = render_table(knee_rows, _KNEE_COLUMNS)
    parts = [
        "# STT concurrency sweep",
        "",
        f"Generated {now_iso()}; fingerprint {fingerprint or '-'}. {host_line(hosts)}",
        "",
        "## Knees",
        "",
        knee_table,
        "",
    ]
    for g in groups:
        parts.extend(
            [
                f"## {g['group']} ({g.get('kind')}; stop: {g.get('stop_reason')})",
                "",
                render_table(g.get("steps") or [], _CURVE_COLUMNS),
                "",
            ],
        )
    parts.extend(
        [
            "## Definitions",
            "",
            "- N: processes (P), threads (S) or batch size (B) of the step.",
            "- xRT: audio seconds of successful items / measured wall seconds; "
            "gain %: versus the best earlier step of the group.",
            "- WER %: corpus WER of the step; dWER pp: WER difference versus the "
            "group's first step on the clips both transcribed.",
            "- ident % / sim mean: clip-wise agreement with the group's first step.",
            "- GB/xRT: peak private GB (parent + workers) per unit of throughput.",
            "- best: highest xRT; efficient: smallest N within "
            "the knee tolerance of the best.",
            "- note: 'quality: ...' marks a step rejected by the quality rule; "
            "other notes give the status of a failed step. Neither is ever chosen "
            "as a knee.",
            "- memory_cap: the step's job hit its commit cap (all children of the "
            "step share one capped Windows job; the cap is the free commit at step "
            "start minus --commit-reserve-gb, or a lower --job-commit-limit-gb): a "
            "saturation point; reused on resume only when its cap was at least "
            "today's. system_commit_low: the system free commit fell below "
            "--commit-reserve-gb during the step (also from other processes); "
            "measured again on resume.",
        ],
    )
    md_path = out_dir / SWEEP_DIRNAME / SUMMARY_MD
    md_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(md_path, "\n".join(parts) + "\n")
    atomic_write(
        out_dir / SWEEP_DIRNAME / SUMMARY_JSON,
        json.dumps(
            {"fingerprint": fingerprint, "groups": groups},
            ensure_ascii=False,
            indent=1,
        ),
    )
    return md_path, knee_table
