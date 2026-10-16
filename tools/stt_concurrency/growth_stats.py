"""Statistics and the text summary of a ``memory-growth`` run.

Each job record carries memory figures in bytes (``None`` when unknown). For
every figure the summary reports the first, last, minimum and maximum value,
the least-squares slope per job and per hour over the last
:data:`TAIL_FRACTION` of the jobs (the first jobs warm the caches up), a
plateau verdict and a trajectory sampled at fixed shares of the run.

Plateau (medians, so a single transient spike does not count as growth):
``tail_drift`` is the median of the last :data:`DRIFT_FRACTION` of the jobs
minus the median of the :data:`DRIFT_FRACTION` before it, and the figure
counts as plateaued when that drift stays below the tolerance.
``plateau_at_job`` is the first job of the first :data:`LEVEL_WINDOW_FRACTION`
window whose median comes within the tolerance of the final level (the median
of the last :data:`DRIFT_FRACTION`).
"""

from __future__ import annotations

import math
import statistics
from fractions import Fraction
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

JsonDict = dict[str, Any]

MIB = 1024**2
GIB = 1024**3
SECONDS_PER_HOUR = 3600.0
# * Shares of a run are exact fractions: float shares round at window
# * boundaries (``1.0 - 0.8`` is 0.19999999999999996, so the last 80% of 10
# * jobs would start at the 2nd job instead of the 3rd).
TAIL_FRACTION = Fraction(4, 5)
DRIFT_FRACTION = Fraction(1, 5)
LEVEL_WINDOW_FRACTION = Fraction(1, 20)
# * Shares of the run at which the trajectories are sampled (job numbers).
TRAJECTORY_SHARES = (
    Fraction(0),
    Fraction(1, 10),
    Fraction(1, 4),
    Fraction(1, 2),
    Fraction(3, 4),
    Fraction(1),
)
# * Byte figures of a job record, in report order.
BYTE_METRICS = (
    "private_bytes",
    "pre_job_private_bytes",
    "job_peak_bytes",
    "cuda_reserved_bytes",
    "cuda_allocated_bytes",
    "cuda_max_reserved_bytes",
    "gpu_used_bytes",
)
CUFFT_METRIC = "cufft_plan_cache_size"
OK_STATUSES = frozenset({"ok", "empty"})
_MIN_POINTS_FOR_SLOPE = 2
_SECONDS_PER_MINUTE = 60.0


def least_squares_slope(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Return the least-squares slope of *ys* over *xs*.

    Args:
        xs: Abscissas.
        ys: Ordinates (same length as *xs*).

    Returns:
        The slope, or ``None`` with fewer than two points or a constant *xs*.

    """
    if len(xs) != len(ys) or len(xs) < _MIN_POINTS_FOR_SLOPE:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    sxx = math.fsum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = math.fsum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    return sxy / sxx


Share: TypeAlias = Fraction | float


def exact_share(share: Share) -> Fraction:
    """Return *share* as an exact fraction.

    A float is taken by its shortest decimal form, so ``0.8`` is exactly
    ``4/5`` and ``0.05`` is ``1/20``.
    """
    return share if isinstance(share, Fraction) else Fraction(str(share))


def tail_start(count: int, fraction: Share = TAIL_FRACTION) -> int:
    """Return the first position of the last *fraction* of *count* items.

    The position is ``floor(count * (1 - fraction))``, computed exactly.
    """
    return max(0, min(count, math.floor(count * (1 - exact_share(fraction)))))


def _window(count: int, fraction: Share) -> int:
    """Return the size of a window holding *fraction* of *count* items (>= 1).

    The size is ``floor(count * fraction)``, computed exactly.
    """
    return max(1, math.floor(count * exact_share(fraction)))


def tail_drift(
    values: Sequence[float], fraction: Share = DRIFT_FRACTION
) -> float | None:
    """Return the median of the last *fraction* minus the median of the one before.

    Returns:
        The drift, or ``None`` when the series is shorter than two windows.

    """
    size = _window(len(values), fraction)
    if len(values) < 2 * size:
        return None
    last = values[len(values) - size :]
    before = values[len(values) - 2 * size : len(values) - size]
    return statistics.median(last) - statistics.median(before)


def level_reached_at(
    values: Sequence[float], tolerance: float, fraction: Share = LEVEL_WINDOW_FRACTION
) -> int | None:
    """Return where the series first reaches its final level within *tolerance*.

    The final level is the median of the last :data:`DRIFT_FRACTION` of the
    series; the series is cut into windows of *fraction* of its length.

    Returns:
        The first position of the first window whose median is at least the
        final level minus *tolerance*, or ``None`` for an empty series.

    """
    if not values:
        return None
    level = statistics.median(
        values[len(values) - _window(len(values), DRIFT_FRACTION) :]
    )
    size = _window(len(values), fraction)
    floor = level - max(0.0, tolerance)
    for start in range(0, len(values), size):
        if statistics.median(values[start : start + size]) >= floor:
            return start
    return None


def _number(value: object) -> float | None:
    """Return *value* as a float when it is a real number (booleans excluded)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _series(
    records: Sequence[Mapping[str, Any]], key: str
) -> tuple[list[int], list[float], list[float]]:
    """Return (job numbers, elapsed hours, values) of the records that know *key*."""
    jobs: list[int] = []
    hours: list[float] = []
    values: list[float] = []
    for record in records:
        value = _number(record.get(key))
        if value is None:
            continue
        jobs.append(int(record["index"]))
        hours.append((_number(record.get("elapsed_s")) or 0.0) / SECONDS_PER_HOUR)
        values.append(value)
    return jobs, hours, values


def trajectory(
    jobs: Sequence[int],
    values: Sequence[float],
    shares: Sequence[Share] = TRAJECTORY_SHARES,
) -> list[list[float]]:
    """Sample a series at fixed shares of its length.

    The position of a share is ``share * (len - 1)`` computed exactly and
    rounded half to even.

    Returns:
        ``[job number, value]`` pairs without duplicates, in job order.

    """
    if not values:
        return []
    last = len(values) - 1
    positions = sorted({round(exact_share(share) * last) for share in shares})
    return [[float(jobs[pos]), values[pos]] for pos in positions]


def metric_summary(
    records: Sequence[Mapping[str, Any]], key: str, *, tolerance: float
) -> JsonDict | None:
    """Summarize one figure of the job records.

    Args:
        records: Job records in job order.
        key: Record field.
        tolerance: Plateau tolerance in the field's unit.

    Returns:
        The summary, or ``None`` when no record knows the figure.

    """
    jobs, hours, values = _series(records, key)
    if not values:
        return None
    start = tail_start(len(values))
    plateau = level_reached_at(values, tolerance)
    drift = tail_drift(values)
    return {
        "points": len(values),
        "first": values[0],
        "last": values[-1],
        "min": min(values),
        "max": max(values),
        "growth": values[-1] - values[0],
        "slope_per_job": least_squares_slope(
            [float(job) for job in jobs[start:]], values[start:]
        ),
        "slope_per_hour": least_squares_slope(hours[start:], values[start:]),
        "tail_drift": drift,
        "plateau_at_job": jobs[plateau] if plateau is not None else None,
        "plateaued": drift is not None and drift <= max(0.0, tolerance),
        "trajectory": trajectory(jobs, values),
    }


def throughput(records: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return the job counts, timings and speed of a run.

    ``busy_s`` sums the wall time from sending each job to its result; idle
    gaps between jobs are excluded. ``audio_xrt`` is processed audio seconds
    per busy second.
    """
    statuses = [str(record.get("status")) for record in records]
    walls = [w for r in records if (w := _number(r.get("wall_s"))) is not None]
    audio = [a for r in records if (a := _number(r.get("audio_s"))) is not None]
    busy_s = math.fsum(walls)
    elapsed = [e for r in records if (e := _number(r.get("elapsed_s"))) is not None]
    return {
        "jobs": len(records),
        "ok": sum(status in OK_STATUSES for status in statuses),
        "errors": sum(status not in OK_STATUSES for status in statuses),
        "busy_s": busy_s,
        "elapsed_s": max(elapsed) if elapsed else 0.0,
        "audio_s": math.fsum(audio),
        "wall_s_median": statistics.median(walls) if walls else None,
        "wall_s_max": max(walls) if walls else None,
        "jobs_per_min_busy": (
            len(walls) * _SECONDS_PER_MINUTE / busy_s if busy_s > 0 else None
        ),
        "audio_xrt": math.fsum(audio) / busy_s if busy_s > 0 else None,
    }


def first_job_at_or_above(
    records: Sequence[Mapping[str, Any]], key: str, threshold: float | None
) -> int | None:
    """Return the first job whose *key* reaches *threshold* (``None``: never)."""
    if threshold is None or threshold <= 0:
        return None
    for record in records:
        value = _number(record.get(key))
        if value is not None and value >= threshold:
            return int(record["index"])
    return None


def jobs_until(
    summary: Mapping[str, Any] | None, threshold: float | None
) -> float | None:
    """Project the jobs left until a figure reaches *threshold* at its tail slope.

    Returns:
        ``None`` without a positive slope or a threshold; ``0`` when the last
        value already reached it.

    """
    if summary is None or threshold is None or threshold <= 0:
        return None
    slope = _number(summary.get("slope_per_job"))
    last = _number(summary.get("last"))
    if last is None:
        return None
    if last >= threshold:
        return 0.0
    if slope is None or slope <= 0:
        return None
    return (threshold - last) / slope


def summarize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    tolerance_bytes: float,
    recycle_threshold_bytes: int | None,
) -> JsonDict:
    """Build the statistics of a run from its job records.

    Args:
        records: Job records in job order.
        tolerance_bytes: Plateau tolerance of the byte figures.
        recycle_threshold_bytes: Private bytes at which the pool would retire
            the worker (``memory.recycle_commit_fraction`` x commit
            reservation), or ``None`` when the recycle is off.

    Returns:
        ``throughput``, ``metrics`` (per byte figure), ``cufft`` and the
        recycle projection.

    """
    metrics = {
        key: metric_summary(records, key, tolerance=tolerance_bytes)
        for key in BYTE_METRICS
    }
    private = metrics.get("private_bytes")
    return {
        "throughput": throughput(records),
        "metrics": {key: value for key, value in metrics.items() if value is not None},
        "cufft": metric_summary(records, CUFFT_METRIC, tolerance=0.0),
        "recycle": {
            "threshold_bytes": recycle_threshold_bytes,
            "first_job_at_threshold": first_job_at_or_above(
                records, "private_bytes", recycle_threshold_bytes
            ),
            "projected_jobs_to_threshold": jobs_until(private, recycle_threshold_bytes),
        },
    }


def _mib(value: object, *, signed: bool = False) -> str:
    """Format a byte figure in MiB (``-`` when unknown)."""
    number = _number(value)
    if number is None:
        return "-"
    return f"{number / MIB:+.2f}" if signed else f"{number / MIB:.1f}"


def _metric_row(name: str, summary: Mapping[str, Any]) -> str:
    """Return one table row of a byte figure (MiB)."""
    plateau = summary.get("plateau_at_job")
    verdict = "yes" if summary.get("plateaued") else "no"
    first, last, peak = (_mib(summary.get(key)) for key in ("first", "last", "max"))
    per_job = _mib(summary.get("slope_per_job"), signed=True)
    per_hour = _mib(summary.get("slope_per_hour"), signed=True)
    drift = _mib(summary.get("tail_drift"), signed=True)
    return (
        f"  {name:<24} {first:>9} {last:>9} {peak:>9} {per_job:>10} {per_hour:>10}"
        f" {drift:>9}  {verdict} (job {plateau})"
    )


def _trajectory_text(summary: Mapping[str, Any] | None, *, as_mib: bool) -> str:
    """Return a trajectory as ``job:value`` pairs."""
    if summary is None:
        return "-"
    pairs = summary.get("trajectory") or []
    return ", ".join(
        f"{int(job)}:{value / MIB:.0f}" if as_mib else f"{int(job)}:{value:.0f}"
        for job, value in pairs
    )


def _throughput_line(stats: Mapping[str, Any]) -> str:
    """Return the throughput line of the summary."""
    jobs_per_min = _number(stats.get("jobs_per_min_busy"))
    xrt = _number(stats.get("audio_xrt"))
    median = _number(stats.get("wall_s_median"))
    return (
        f"Jobs: {stats.get('jobs')} ({stats.get('ok')} ok, {stats.get('errors')} "
        f"errors); elapsed {(_number(stats.get('elapsed_s')) or 0.0) / 60:.1f} min, "
        f"busy {(_number(stats.get('busy_s')) or 0.0) / 60:.1f} min; "
        f"{'-' if jobs_per_min is None else f'{jobs_per_min:.1f}'} jobs/min busy, "
        f"audio xRT {'-' if xrt is None else f'{xrt:.1f}'}, "
        f"median job {'-' if median is None else f'{median:.2f}'} s"
    )


def render_summary(summary: Mapping[str, Any]) -> list[str]:
    """Render the summary document of a run as text lines.

    Args:
        summary: Document written to ``summary.json`` (``status``,
            ``baseline``, ``stats`` and the run identity).

    Returns:
        The lines, without trailing newlines.

    """
    stats: Mapping[str, Any] = summary.get("stats") or {}
    baseline: Mapping[str, Any] = summary.get("baseline") or {}
    lines = [
        f"Run {summary.get('run_id')}: status {summary.get('status')}"
        + (f" ({summary.get('detail')})" if summary.get("detail") else ""),
        _throughput_line(stats.get("throughput") or {}),
        "Baseline after load (MiB): "
        f"private {_mib(baseline.get('private_bytes'))}, "
        f"job peak {_mib(baseline.get('job_peak_bytes'))}, "
        f"GPU used {_mib(baseline.get('gpu_used_bytes'))} "
        f"(before spawn {_mib(baseline.get('gpu_used_before_spawn_bytes'))})",
        f"  {'figure (MiB)':<24} {'first':>9} {'last':>9} {'max':>9} "
        f"{'slope/job':>10} {'slope/h':>10} {'drift':>9}  plateau",
    ]
    metrics: Mapping[str, Any] = stats.get("metrics") or {}
    lines.extend(
        _metric_row(name, metrics[name]) for name in BYTE_METRICS if name in metrics
    )
    cufft = stats.get("cufft")
    if cufft is not None:
        sizes = _trajectory_text(cufft, as_mib=False)
        lines.append(
            f"cuFFT plans: first {cufft.get('first'):.0f}, last "
            f"{cufft.get('last'):.0f}, max {cufft.get('max'):.0f}; job:size {sizes}"
        )
    lines.extend(
        f"{name} trajectory (job:MiB): "
        f"{_trajectory_text(metrics.get(name), as_mib=True)}"
        for name in ("cuda_reserved_bytes", "cuda_allocated_bytes", "private_bytes")
    )
    recycle: Mapping[str, Any] = stats.get("recycle") or {}
    projected = _number(recycle.get("projected_jobs_to_threshold"))
    lines.append(
        f"Recycle threshold {_mib(recycle.get('threshold_bytes'))} MiB: reached at job "
        f"{recycle.get('first_job_at_threshold')}; projected jobs left at the tail "
        f"slope: {'-' if projected is None else f'{projected:.0f}'}"
    )
    return lines
