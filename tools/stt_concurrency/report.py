"""Summary of ``run`` results: metrics table, transcript correctness, notes.

Only results with the current fingerprint are aggregated when a fingerprint is
given (see :mod:`stt_concurrency.store`). The table helpers are shared with the
``sweep`` and ``combo`` summaries.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .configs import BenchConfig
from .paths import RESULTS_DIRNAME, SUMMARY_JSON_NAME, SUMMARY_MD_NAME
from .quality import compare_items
from .sysmetrics import GIB

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

JsonDict = dict[str, Any]

_COMPARABLE_STATUSES = frozenset({"ok", "partial"})
_MISSING = "-"

# * Markdown columns: (header, row key, decimals).
_COLUMNS: tuple[tuple[str, str, int], ...] = (
    ("config", "name", 0),
    ("rep", "repeat", 0),
    ("status", "status", 0),
    ("wall s", "wall_s", 1),
    ("audio s", "audio_s", 0),
    ("xRT", "throughput_x", 1),
    ("p50 s", "latency_p50_s", 2),
    ("p95 s", "latency_p95_s", 2),
    ("load max s", "load_max_s", 1),
    ("load mean s", "load_mean_s", 1),
    ("CPU %", "cpu_system_percent", 0),
    ("cores", "worker_cores", 1),
    ("priv GB", "peak_private_gb", 2),
    ("WS GB", "peak_working_set_gb", 2),
    ("VRAM d MB", "vram_delta_mb", 0),
    ("GPU %", "gpu_util_percent", 0),
    ("WER %", "wer_percent", 2),
    ("ident %", "identical_percent", 1),
    ("sim mean", "similarity_mean", 3),
    ("sim min", "similarity_min", 3),
    ("err", "errors", 0),
)

_DEFINITIONS = (
    "wall s: clock from all workers ready (models loaded, warm-up done) to the "
    "last item result.",
    "xRT: total audio seconds of successful items / wall seconds.",
    "p50/p95 s: processing time per dispatched call, including audio decoding; "
    "in B mode every item of a batch gets the batch time.",
    "load: model load time per instance (children load concurrently).",
    "CPU %: system-wide busy share during the timed run (all logical CPUs).",
    "cores: CPU seconds of the worker processes / wall seconds (excludes the "
    "short-lived ffmpeg decoders of GigaAM).",
    "priv GB / WS GB: peak sum of private bytes / working sets over the parent "
    "and all descendants, load phase included (run-only peaks are in the JSON). "
    "With a Windows venv every worker is a launcher + interpreter pair; the "
    "launcher adds a few MB.",
    "VRAM d MB: whole-GPU used VRAM peak (load or run) minus the baseline read "
    "before the children started (nvidia-smi; per-process VRAM is unavailable "
    "under WDDM). The run-only delta is in the JSON.",
    "ident % / sim: per-item normalized transcript equality / character "
    "similarity versus the group reference (P1 config of the same engine, device "
    "and workload, else the first config run). CPU thread counts and batching "
    "change float summation order, so small differences are expected; a low "
    "'sim min' points at corrupted output.",
    "WER %: corpus WER of successful items against --manifest (lowercase, "
    "Cyrillic yo->ye, punctuation removed); every config of a group "
    "transcribes the same clips, so the column compares directly.",
    "Whisper B-long: BatchedInferencePipeline decodes fixed chunks without the "
    "seek loop of the sequential reference, so its transcripts differ by design; "
    "run whisper-cuda-B1-long to separate batching effects from chunking.",
    "memory_cap: all children of a measurement share one Windows job whose "
    "commit cap is the real headroom (free commit at step start - "
    "--commit-reserve-gb; cap_reason 'headroom') or the lower "
    "--job-commit-limit-gb (cap_reason 'explicit'); the cap was hit and the step "
    "was terminated: a real limit of this run (saturation point). A stored "
    "memory_cap result is reused only when its cap was at least the cap a step "
    "would get now. system_commit_low: the system free commit fell below "
    "--commit-reserve-gb during the step (any process can cause it); the step "
    "was terminated and is measured again on resume. The 'job' block of a result "
    "holds the cap, cap_reason, cap_enforced, and the job's peak commit. Without "
    "a working cap a step is skipped unless --allow-uncapped.",
)


def load_results(out_dir: Path, fingerprint: str | None = None) -> list[JsonDict]:
    """Load the result files of *out_dir*.

    Args:
        out_dir: Benchmark output directory.
        fingerprint: Keep only results with this fingerprint (``None``: all).

    Returns:
        Result documents (unreadable files are skipped).

    """
    results_dir = out_dir / RESULTS_DIRNAME
    if not results_dir.is_dir():
        return []
    results: list[JsonDict] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or "config" not in data:
            continue
        if fingerprint is None or data.get("fingerprint") == fingerprint:
            results.append(data)
    return results


def _config(result: JsonDict) -> BenchConfig:
    return BenchConfig.from_json(result["config"])


def _pick_reference(results: Sequence[JsonDict]) -> JsonDict | None:
    """Pick the group reference: a successful P1 config, else the first run."""
    usable = [r for r in results if r.get("status") in _COMPARABLE_STATUSES]
    if not usable:
        return None
    p1 = [r for r in usable if _config(r).mode == "P" and _config(r).width == 1]
    pool = p1 or usable
    return min(
        pool, key=lambda r: (int(r.get("repeat", 0)), str(r.get("started_at", "")))
    )


def _compare(result: JsonDict, reference: JsonDict) -> JsonDict:
    """Compare per-clip transcripts of *result* with *reference*."""
    comparison = compare_items(
        result.get("items", []),
        reference.get("items", []),
        None,
    )
    return {
        "reference": reference["config"]["name"],
        "reference_repeat": reference.get("repeat"),
        **comparison,
    }


def gb(value: object) -> float | None:
    """Convert a byte count to GiB (``None`` passes through)."""
    return float(value) / GIB if isinstance(value, (int, float)) else None


def _row(result: JsonDict, correctness: JsonDict | None) -> JsonDict:
    """Flatten one result document into a summary row."""
    timing = result.get("timing") or {}
    cpu = result.get("cpu") or {}
    memory = result.get("memory") or {}
    vram = result.get("vram") or {}
    config = result["config"]
    row: JsonDict = {
        "name": config["name"],
        "group": _config(result).group,
        "repeat": result.get("repeat"),
        "status": result.get("status"),
        "reason": result.get("reason"),
        "processes": config.get("processes"),
        "slots": config.get("slots"),
        "batch_size": config.get("batch_size"),
        "threads": config.get("threads"),
        "started_at": result.get("started_at"),
        "wall_s": timing.get("wall_s"),
        "audio_s": timing.get("audio_s"),
        "throughput_x": timing.get("throughput_x"),
        "latency_p50_s": timing.get("latency_p50_s"),
        "latency_p95_s": timing.get("latency_p95_s"),
        "load_max_s": timing.get("load_max_s"),
        "load_mean_s": timing.get("load_mean_s"),
        "cpu_system_percent": cpu.get("system_busy_percent"),
        "worker_cores": cpu.get("worker_cores_avg"),
        "peak_private_gb": gb(memory.get("peak_private_bytes")),
        "peak_private_run_gb": gb(memory.get("peak_private_run_bytes")),
        "peak_working_set_gb": gb(memory.get("peak_working_set_bytes")),
        "vram_delta_mb": vram.get("delta_mb"),
        "vram_peak_mb": vram.get("peak_mb"),
        "gpu_util_percent": vram.get("util_avg_run_percent"),
        "errors": result.get("error_count"),
        "api_parity": result.get("api_parity"),
        "wer_percent": ((result.get("quality") or {}).get("wer") or {}).get(
            "wer_percent",
        ),
        "repeat_consistency_percent": (result.get("quality") or {}).get(
            "repeat_consistency_percent",
        ),
    }
    if correctness:
        row.update(correctness)
    return row


def build_rows(results: Sequence[JsonDict]) -> list[JsonDict]:
    """Compute summary rows, including correctness versus each group reference.

    Args:
        results: Result documents.

    Returns:
        Rows sorted by engine, device, workload, mode, width, and repeat.

    """
    groups: dict[str, list[JsonDict]] = {}
    for result in results:
        groups.setdefault(_config(result).group, []).append(result)
    references = {group: _pick_reference(items) for group, items in groups.items()}
    rows: list[JsonDict] = []
    for result in sorted(
        results,
        key=lambda r: (_config(r).sort_key, int(r.get("repeat", 0))),
    ):
        reference = references.get(_config(result).group)
        correctness = (
            _compare(result, reference)
            if reference is not None and result.get("status") in _COMPARABLE_STATUSES
            else None
        )
        rows.append(_row(result, correctness))
    return rows


def fmt(value: object, decimals: int) -> str:
    """Format one table cell (``-`` for missing values)."""
    if value is None:
        return _MISSING
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def render_table(
    rows: Sequence[JsonDict],
    columns: Sequence[tuple[str, str, int]] = _COLUMNS,
) -> str:
    """Render rows as a Markdown table.

    Args:
        rows: Summary rows.
        columns: ``(header, row key, decimals)`` per column.

    Returns:
        Markdown table.

    """
    header = "| " + " | ".join(title for title, _key, _dec in columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, divider]
    lines.extend(
        "| " + " | ".join(fmt(row.get(key), dec) for _t, key, dec in columns) + " |"
        for row in rows
    )
    return "\n".join(lines)


def _notes(rows: Sequence[JsonDict], results: Sequence[JsonDict]) -> list[str]:
    notes: list[str] = []
    references = sorted(
        {f"{row['group']}: {row['reference']}" for row in rows if row.get("reference")},
    )
    notes.extend(f"Reference {ref}" for ref in references)
    skipped = sum(1 for row in rows if row.get("status") == "skipped")
    if skipped:
        notes.append(f"{skipped} of {len(results)} result(s) were skipped (see below).")
    notes.extend(
        f"{row['name']} r{row['repeat']}: {row['status']} - {row['reason']}"
        for row in rows
        if row.get("status") != "ok" and row.get("reason")
    )
    parity = [row for row in rows if row.get("api_parity") is not None]
    mismatched = [row for row in parity if row.get("api_parity") is False]
    notes.extend(
        f"{row['name']} r{row['repeat']}: GigaAM benchmark path differs from "
        "model.transcribe(path) on the warm-up clip"
        for row in mismatched
    )
    if parity and not mismatched:
        notes.append(
            f"GigaAM benchmark path matches model.transcribe(path) on the warm-up "
            f"clip in all {len(parity)} GigaAM result(s).",
        )
    return notes


def host_line(results: Sequence[JsonDict]) -> str:
    """Describe the host of the newest result in one line."""
    hosts: list[JsonDict] = [
        host for r in results if isinstance(host := r.get("host"), dict)
    ]
    if not hosts:
        return "Host: unknown."
    host = hosts[-1]
    processor = host.get("processor") or "?"
    return (
        f"Host: {processor}; logical CPUs {host.get('cpu_logical')}, "
        f"physical {host.get('cpu_physical')}, CPU budget {host.get('cpu_budget')}; "
        f"RAM {fmt(host.get('ram_total_gb'), 1)} GB; GPU {host.get('gpu') or '-'}; "
        f"Python {host.get('python')}."
    )


def render_markdown(rows: Sequence[JsonDict], results: Sequence[JsonDict]) -> str:
    """Render the whole summary document.

    Args:
        rows: Summary rows from :func:`build_rows`.
        results: Result documents.

    Returns:
        Markdown text.

    """
    generated = datetime.now(UTC).isoformat(timespec="seconds")
    parts = [
        "# STT concurrency benchmark",
        "",
        f"Generated {generated}. {host_line(results)}",
        "",
        render_table(rows),
        "",
        "## Notes",
        "",
    ]
    parts.extend(f"- {note}" for note in _notes(rows, results))
    parts.extend(["", "## Definitions", ""])
    parts.extend(f"- {line}" for line in _DEFINITIONS)
    return "\n".join(parts) + "\n"


def atomic_write(path: Path, text: str) -> None:
    """Write text atomically (temp file + replace)."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_summary(out_dir: Path, fingerprint: str | None = None) -> tuple[Path, str]:
    """Write ``summary.md`` and ``summary.json`` from the result files.

    Args:
        out_dir: Benchmark output directory.
        fingerprint: Aggregate only results with this fingerprint (``None``: all).

    Returns:
        ``(markdown_path, table_markdown)``.

    """
    results = load_results(out_dir, fingerprint)
    rows = build_rows(results)
    md_path = out_dir / SUMMARY_MD_NAME
    atomic_write(md_path, render_markdown(rows, results))
    atomic_write(
        out_dir / SUMMARY_JSON_NAME,
        json.dumps(rows, ensure_ascii=False, indent=1),
    )
    return md_path, render_table(rows)
