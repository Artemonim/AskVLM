"""Mixed concurrent runs (``combo``): two configurations under one clock.

A combo such as ``gigaam-cuda-B16+whisper-cpu-S8x2`` starts both configurations
at the same time on disjoint clip streams (side A cycles the even-numbered
clips, side B the odd-numbered ones). Every side keeps receiving work until
*all* sides dispatched ``clips_per_stream x N`` items and ``min_step_seconds``
elapsed, so no side finishes its share alone.

Throughput is measured over the common concurrent window: from the shared clock
start until the first side finished its last item. Inside it both sides were
busy the whole time. Each side's audio in the window counts every item with the
share of its processing interval (dispatch to result) that lies inside the
window. The window throughput is the headline ``throughput_x`` of a side and of
the pair; the whole run of every side stays available as ``full_*`` fields.

Reported per side: window throughput, latency, WER, and the slowdown versus the
same configuration measured alone by ``sweep`` in the same ``--out-dir`` with the
same fingerprint (when present). Reported for the pair: window throughput of
both sides together, CPU, memory, and VRAM.

``--auto-combos`` pairs the knee steps found by ``sweep``: the best (or the
efficient) GPU step of every GPU engine with the best CPU step of every engine.
Results go to ``<out-dir>/combos/results`` with the sweep fingerprint (the same
workload sizing makes the solo comparison meaningful).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from .configs import ConfigError, parse_config_name
from .paths import LOGS_DIRNAME, RESULTS_DIRNAME
from .report import atomic_write, gb, host_line, render_table
from .runner import (
    RESULT_SCHEMA_VERSION,
    REUSABLE_STATUSES,
    Measurement,
    SideSpec,
    collect_host_info,
    decode_info,
    echo,
    evaluate_guard_configs,
    load_workload,
    now_iso,
    quality_block,
    reusable_result,
)
from .store import ResultStore
from .sweep import (
    SWEEP_DIRNAME,
    load_group_documents,
    step_stream,
    sweep_fingerprint,
)
from .sysmetrics import KillOnCloseJob, find_nvidia_smi

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from .configs import BenchConfig, ThreadDefaults
    from .quality import Manifest
    from .sweep import SweepOptions

JsonDict = dict[str, Any]

COMBO_DIRNAME = "combos"
SUMMARY_MD = "summary.md"
_SIDE_LABELS = ("a", "b")
# * Item ids of side B start here so both sides keep unique ids in the record.
_SIDE_B_FIRST_ID = 1_000_000


def parse_combo(
    token: str, defaults: ThreadDefaults
) -> tuple[BenchConfig, BenchConfig]:
    """Parse ``<config>+<config>``.

    Args:
        token: Combo token.
        defaults: Thread defaults.

    Returns:
        The two configurations.

    Raises:
        ConfigError: For anything but exactly two valid, short-clip configurations.

    """
    parts = [part.strip() for part in token.split("+")]
    if len(parts) != len(_SIDE_LABELS) or not all(parts):
        msg = f"Invalid combo {token!r}; expected <config>+<config>"
        raise ConfigError(msg)
    first, second = (parse_config_name(part, defaults) for part in parts)
    if first.long or second.long:
        msg = f"Combo {token!r}: -long configurations are not supported"
        raise ConfigError(msg)
    return first, second


def auto_combos(
    out_dir: Path,
    fingerprint: str,
    defaults: ThreadDefaults,
    *,
    knee: str,
) -> list[tuple[BenchConfig, BenchConfig]]:
    """Pair sweep knees: every GPU engine's best GPU step with every CPU engine's.

    Args:
        out_dir: Benchmark output directory.
        fingerprint: Sweep fingerprint.
        defaults: Thread defaults (for parsing step names).
        knee: ``best`` or ``efficient``.

    Returns:
        Configuration pairs (GPU first).

    """
    best: dict[tuple[str, str], tuple[float, str]] = {}
    for group in load_group_documents(out_dir, fingerprint):
        knees = group.get("knees") or {}
        name = knees.get(knee)
        xrt = knees.get(f"{knee}_xrt")
        if not name or xrt is None:
            continue
        config = parse_config_name(str(name), defaults)
        key = (config.engine, config.device)
        if key not in best or float(xrt) > best[key][0]:
            best[key] = (float(xrt), str(name))
    gpu = sorted((k, v[1]) for k, v in best.items() if k[1] == "cuda")
    cpu = sorted((k, v[1]) for k, v in best.items() if k[1] == "cpu")
    return [
        (parse_config_name(g, defaults), parse_config_name(c, defaults))
        for _gk, g in gpu
        for _ck, c in cpu
    ]


def _solo(store: ResultStore, config: BenchConfig) -> float | None:
    """Return the solo throughput of *config* from the sweep store, if measured."""
    document = store.reusable(config.name, statuses=frozenset({"ok"}))
    if document is None:
        return None
    value = (document.get("timing") or {}).get("throughput_x")
    return float(value) if value is not None else None


def _run_combo(  # noqa: PLR0913 - explicit inputs of one measured pair
    pair: tuple[BenchConfig, BenchConfig],
    *,
    options: SweepOptions,
    clips: Sequence[Any],
    warmup_path: Path,
    manifest: Manifest | None,
    job: KillOnCloseJob,
    logs_dir: Path,
    nvidia_smi: str | None,
    host: JsonDict,
    solo_store: ResultStore,
) -> JsonDict:
    name = f"{pair[0].name}+{pair[1].name}"
    document: JsonDict = {
        "schema": RESULT_SCHEMA_VERSION,
        "kind": "combo",
        "name": name,
        "configs": [config.to_json() for config in pair],
        "started_at": now_iso(),
        "host": host,
        "workload": {
            "clips": len(clips),
            "split": "side a: even clips, side b: odd clips",
            "clips_per_stream": options.clips_per_stream,
            "min_step_seconds": options.min_step_seconds,
        },
        "decode": [decode_info(config, options.run) for config in pair],
    }
    allowed, reason, guard = evaluate_guard_configs(list(pair), options.run, nvidia_smi)
    document["guard"] = guard
    if not allowed:
        document.update(status="skipped", reason=reason, finished_at=now_iso())
        return document
    halves = (list(clips[0::2]), list(clips[1::2]))
    sides = [
        SideSpec(
            config=config,
            stream=step_stream(
                halves[position] or list(clips),
                config,
                options,
                first_id=_SIDE_B_FIRST_ID * position,
            ),
            label=_SIDE_LABELS[position],
        )
        for position, config in enumerate(pair)
    ]
    measured = Measurement(
        sides=sides,
        run_id=name,
        warmup_path=warmup_path,
        options=options.run,
        job=job,
        logs_dir=logs_dir,
        nvidia_smi=nvidia_smi,
    ).execute()
    for side, config in zip(measured["sides"], pair, strict=True):
        side["quality"] = quality_block(side["items"], manifest)
        solo = _solo(solo_store, config)
        own = side["timing"].get("throughput_x")
        side["solo_throughput_x"] = solo
        side["slowdown_percent"] = (
            100.0 * (1.0 - float(own) / solo) if solo and own is not None else None
        )
    statuses = [side["status"] for side in measured["sides"]]
    worst = next((s for s in ("error", "timeout", "partial") if s in statuses), "ok")
    document.update(measured)
    document["status"] = worst if measured["status"] == "ok" else measured["status"]
    document["finished_at"] = now_iso()
    return document


def run_combos(
    options: SweepOptions,
    *,
    pairs: str,
    auto: bool,
    knee: str,
) -> int:
    """Run (or plan) the requested combos and write the combo summary.

    Args:
        options: Sweep options (workload sizing, shared run options).
        pairs: Comma-separated ``<config>+<config>`` tokens.
        auto: Add the pairs derived from the sweep knees.
        knee: ``best`` or ``efficient`` for ``auto``.

    Returns:
        Process exit code.

    Raises:
        ConfigError: When no combo was requested or found.

    """
    run = options.run
    workload = load_workload(run)
    fingerprint = sweep_fingerprint(options, workload)
    combos = [
        parse_combo(token, run.thread_defaults)
        for token in pairs.split(",")
        if token.strip()
    ]
    if auto:
        found = auto_combos(run.out_dir, fingerprint, run.thread_defaults, knee=knee)
        echo(f"Auto combos from sweep knees ({knee}): {len(found)}")
        combos.extend(found)
    if not combos:
        msg = "No combo requested: pass --pairs and/or --auto-combos after a sweep"
        raise ConfigError(msg)
    root = run.out_dir / COMBO_DIRNAME
    store = ResultStore(root / RESULTS_DIRNAME, fingerprint)
    solo_store = ResultStore(run.out_dir / SWEEP_DIRNAME / RESULTS_DIRNAME, fingerprint)
    nvidia_smi = find_nvidia_smi()
    if run.dry_run:
        for first, second in combos:
            key = f"{first.name}+{second.name}"
            allowed, reason, guard = evaluate_guard_configs(
                [first, second],
                run,
                nvidia_smi,
            )
            verdict = "run" if allowed else f"skip: {reason}"
            echo(f"[{key}] {store.describe(key)}; {verdict}; guard {guard}")
        return 0
    logs_dir = root / LOGS_DIRNAME
    for directory in (store.directory, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    job = KillOnCloseJob()
    host = collect_host_info(run, nvidia_smi)
    executed = 0
    try:
        for pair in combos:
            key = f"{pair[0].name}+{pair[1].name}"
            reuse = REUSABLE_STATUSES if not run.retry_failed else frozenset({"ok"})
            if reusable_result(store, key, statuses=reuse, options=run) is not None:
                echo(f"[{key}] {store.describe(key)}; skipped")
                continue
            if executed and run.cooldown_s > 0:
                time.sleep(run.cooldown_s)
            echo(f"[{key}] {now_iso()}")
            document = _run_combo(
                pair,
                options=options,
                clips=workload.clips,
                warmup_path=workload.warmup_path,
                manifest=workload.manifest,
                job=job,
                logs_dir=logs_dir,
                nvidia_smi=nvidia_smi,
                host=host,
                solo_store=solo_store,
            )
            executed += 1
            store.save(key, document)
            echo(f"  -> {document['status']} {document.get('reason') or ''}")
            write_combo_summary(run.out_dir, fingerprint)
    finally:
        job.close()
    md_path, table = write_combo_summary(run.out_dir, fingerprint)
    echo("")
    echo(table)
    echo(f"\nCombo summary: {md_path}")
    return 0


_COLUMNS: tuple[tuple[str, str, int], ...] = (
    ("combo", "combo", 0),
    ("side", "label", 0),
    ("config", "config", 0),
    ("status", "status", 0),
    ("xRT", "throughput_x", 1),
    ("solo xRT", "solo_throughput_x", 1),
    ("slowdown %", "slowdown_percent", 1),
    ("p50 s", "latency_p50_s", 2),
    ("p95 s", "latency_p95_s", 2),
    ("WER %", "wer_percent", 2),
    ("pair xRT", "combined_xrt", 1),
    ("CPU %", "cpu_system_percent", 0),
    ("priv GB", "peak_private_gb", 2),
    ("VRAM d MB", "vram_delta_mb", 0),
)


def write_combo_summary(out_dir: Path, fingerprint: str | None) -> tuple[Path, str]:
    """Write ``combos/summary.md`` from the combo results.

    Args:
        out_dir: Benchmark output directory.
        fingerprint: Aggregate only this fingerprint (``None``: all).

    Returns:
        ``(markdown_path, table)``.

    """
    store = ResultStore(out_dir / COMBO_DIRNAME / RESULTS_DIRNAME, fingerprint or "")
    documents = store.current() if fingerprint else store.all_documents()
    rows: list[JsonDict] = []
    for document in documents:
        if document.get("status") == "skipped":
            rows.append(
                {
                    "combo": document.get("name"),
                    "status": "skipped",
                    "config": document.get("reason"),
                },
            )
            continue
        timing = document.get("timing") or {}
        memory = document.get("memory") or {}
        vram = document.get("vram") or {}
        for side in document.get("sides") or []:
            side_timing = side.get("timing") or {}
            rows.append(
                {
                    "combo": document.get("name"),
                    "label": side.get("label"),
                    "config": side["config"]["name"],
                    "status": side.get("status"),
                    "throughput_x": side_timing.get("throughput_x"),
                    "solo_throughput_x": side.get("solo_throughput_x"),
                    "slowdown_percent": side.get("slowdown_percent"),
                    "latency_p50_s": side_timing.get("latency_p50_s"),
                    "latency_p95_s": side_timing.get("latency_p95_s"),
                    "wer_percent": ((side.get("quality") or {}).get("wer") or {}).get(
                        "wer_percent",
                    ),
                    "combined_xrt": timing.get("throughput_x"),
                    "cpu_system_percent": (document.get("cpu") or {}).get(
                        "system_busy_percent",
                    ),
                    "peak_private_gb": gb(memory.get("peak_private_bytes")),
                    "vram_delta_mb": vram.get("delta_mb"),
                },
            )
    table = render_table(rows, _COLUMNS)
    text = "\n".join(
        [
            "# STT concurrency combos",
            "",
            f"Generated {now_iso()}; fingerprint {fingerprint or '-'}. "
            f"{host_line(documents)}",
            "",
            table,
            "",
            "## Definitions",
            "",
            "- Window: from the shared clock start until the first side finished "
            "its last item; both sides were busy during the whole window (each side "
            "gets work until every side met its minimum items and seconds).",
            "- xRT: the side's audio processed inside the window / window seconds; "
            "an item crossing the window end counts with the share of its "
            "processing interval inside the window. pair xRT: both sides' window "
            "audio / window seconds. Whole-run numbers are the full_* JSON fields.",
            "- slowdown %: 1 - side xRT / solo xRT of the same configuration in the "
            "sweep of this --out-dir (same fingerprint).",
            "- Side a cycles the even-numbered clips, side b the odd-numbered ones.",
            "",
        ],
    )
    md_path = out_dir / COMBO_DIRNAME / SUMMARY_MD
    md_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(md_path, text)
    return md_path, table
