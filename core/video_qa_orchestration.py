"""Planning-only Video QA orchestration: chunking, policies, manifest merge, preflight."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Final, Literal

from .video_qa_manifest import VideoQAChunkRecord
from .video_qa_runtime import (
    build_video_qa_budget_estimate,
    default_video_qa_budget_policy,
    default_video_qa_runtime_policy,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .video_qa_context import VideoQAContextBundle
    from .video_qa_manifest import GraphNodeKind, VideoQARunManifest
    from .video_qa_runtime import (
        VideoQABudgetEstimate,
        VideoQABudgetPolicy,
        VideoQARuntimePolicy,
    )

PlanningMode = Literal["scene", "uniform_grid", "whole_video"]

RepresentativeFrameKind = Literal[
    "middle_timestamp", "first_timestamp", "last_timestamp"
]

OverflowMitigationStep = Literal[
    "reduce_frame_count",
    "reduce_resolution",
    "split_chunk_or_text",
]

_OVERFLOW_STEP_LABELS: Final[dict[OverflowMitigationStep, str]] = {
    "reduce_frame_count": "reduce frame count per chunk",
    "reduce_resolution": "reduce frame resolution",
    "split_chunk_or_text": "split chunk spans or trim/split accompanying text",
}

# * Canonical graph kinds in DAG order; matches ``_build_graph_skeleton`` / ``GraphNodeKind``.
VIDEO_QA_ORCHESTRATION_STAGES: Final[tuple[GraphNodeKind, ...]] = (
    "source_resolve",
    "attachment_prepare",
    "transcript_prepare",
    "chunk_plan",
    "frame_select",
    "llm_pass",
    "answer_aggregate",
)

# * Manifest kinds whose primary output is transcript/subtitle (subtitle-first), not QA bundles.
VIDEO_QA_SUBTITLE_FIRST_GRAPH_KINDS: Final[tuple[GraphNodeKind, ...]] = (
    "transcript_prepare",
)

# * QA-side graph kinds (chunk timing, frames, LLM, aggregate) without transcript product.
VIDEO_QA_QA_CHUNK_GRAPH_KINDS: Final[tuple[GraphNodeKind, ...]] = (
    "chunk_plan",
    "frame_select",
    "llm_pass",
    "answer_aggregate",
)

_PLANNED_CHUNK_ALLOWED_SLOTS: Final[frozenset[str]] = frozenset(
    ("chunk_id", "t_start", "t_end", "planning_mode")
)


@dataclass(frozen=True, slots=True)
class VideoQAPlannedChunk:
    """A planned temporal span for one Video QA chunk (no runtime artifacts)."""

    chunk_id: str
    t_start: float
    t_end: float
    planning_mode: PlanningMode = "scene"


@dataclass(frozen=True, slots=True)
class VideoQARepresentativeFramePolicy:
    """Selects how representative frame timestamps are derived from chunk spans."""

    kind: RepresentativeFrameKind = "middle_timestamp"

    def timestamp_for_span(self, t_start: float, t_end: float) -> float:
        """Return the representative timestamp for ``[t_start, t_end]``."""
        if t_end < t_start:
            msg = "Chunk span end must be >= start."
            raise ValueError(msg)
        if self.kind == "middle_timestamp":
            return (t_start + t_end) / 2.0
        if self.kind == "first_timestamp":
            return t_start
        return t_end


@dataclass(frozen=True, slots=True)
class VideoQAOverflowPolicy:
    """Documents the overflow mitigation order (frames, then resolution, then chunk/text split)."""

    mitigation_steps: tuple[OverflowMitigationStep, ...] = (
        "reduce_frame_count",
        "reduce_resolution",
        "split_chunk_or_text",
    )


@dataclass(frozen=True, slots=True)
class VideoQAPreflightSummary:
    """Backend-only preflight: budget estimate plus chunk plan and combined warnings."""

    budget: VideoQABudgetEstimate
    chunk_plan: tuple[VideoQAPlannedChunk, ...]
    overflow_policy: VideoQAOverflowPolicy
    representative_frame_policy: VideoQARepresentativeFramePolicy
    orchestration_stages: tuple[GraphNodeKind, ...] = field(
        default_factory=lambda: VIDEO_QA_ORCHESTRATION_STAGES
    )
    subtitle_first_graph_kinds: tuple[GraphNodeKind, ...] = field(
        default_factory=lambda: VIDEO_QA_SUBTITLE_FIRST_GRAPH_KINDS
    )
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class VideoQAPreflightReport:
    """Structured, display-ready preflight summary for GUI or CLI layers.

    Built from :class:`VideoQAContextBundle` and :class:`VideoQAPreflightSummary`
    without performing additional I/O or model calls. The chunk plan matches
    :attr:`VideoQAPreflightSummary.chunk_plan` for the same preflight inputs.
    """

    source_summary: str | None
    question: str
    chunk_plan: tuple[VideoQAPlannedChunk, ...]
    budget_fits: bool
    budget_status_line: str
    budget_estimate_summary: str
    warnings: tuple[str, ...]
    overflow_mitigation_order_text: str
    overflow_fallback_explanation: str

    @property
    def chunk_count(self) -> int:
        """Return the number of entries in :attr:`chunk_plan`."""
        return len(self.chunk_plan)


def format_video_qa_overflow_mitigation_order(
    policy: VideoQAOverflowPolicy,
) -> str:
    """Return the overflow mitigation sequence as a readable ordered phrase."""
    labels = [_OVERFLOW_STEP_LABELS[step] for step in policy.mitigation_steps]
    if not labels:
        return "(no mitigation steps configured)"
    parts = [f"{index}. {label}" for index, label in enumerate(labels, start=1)]
    return "; ".join(parts)


def _build_overflow_fallback_explanation(
    *,
    budget_fits: bool,
    overflow_mitigation_order_text: str,
) -> str:
    """Explain when client-side overflow handling applies."""
    if budget_fits:
        return (
            "Offline estimate fits the configured context window; server-side limits "
            "may still differ."
        )
    return (
        "Offline estimate exceeds the configured context window. Apply overflow "
        "mitigation in this order before relying on server-side truncation: "
        f"{overflow_mitigation_order_text}"
    )


def build_video_qa_preflight_report(
    context: VideoQAContextBundle,
    preflight: VideoQAPreflightSummary,
) -> VideoQAPreflightReport:
    """Turn a preflight summary plus context into a user-facing report record."""
    budget = preflight.budget
    frame_status = (
        f", {budget.sampled_frame_count} frames total, "
        f"peak {budget.peak_frames_per_chunk} per chunk "
        f"≈{budget.frame_tokens_estimate} frame tokens"
        if budget.sampled_frame_count > 0
        else ""
    )
    fits = budget.fits
    if fits:
        status_line = (
            f"fits ({budget.total_required_tokens} / "
            f"{budget.context_window_tokens} tokens{frame_status})"
        )
    else:
        status_line = (
            f"over by {budget.overflow_tokens} tokens "
            f"({budget.total_required_tokens} / {budget.context_window_tokens}"
            f"{frame_status})"
        )
    source_summary = context.source.summary if context.source is not None else None
    overflow_text = format_video_qa_overflow_mitigation_order(preflight.overflow_policy)
    return VideoQAPreflightReport(
        source_summary=source_summary,
        question=context.question,
        chunk_plan=preflight.chunk_plan,
        budget_fits=fits,
        budget_status_line=status_line,
        budget_estimate_summary=budget.summary,
        warnings=preflight.warnings,
        overflow_mitigation_order_text=overflow_text,
        overflow_fallback_explanation=_build_overflow_fallback_explanation(
            budget_fits=fits,
            overflow_mitigation_order_text=overflow_text,
        ),
    )


def format_video_qa_preflight_report_text(report: VideoQAPreflightReport) -> str:
    """Format a :class:`VideoQAPreflightReport` as multi-line plain text."""
    lines: list[str] = []
    q = report.question.strip()
    lines.append(f"Question: {q if q else '(empty)'}")
    lines.append(f"Chunks: {report.chunk_count}")
    lines.append("Chunk plan:")
    if report.chunk_plan:
        lines.extend(
            (
                f"  - {chunk.chunk_id} | {chunk.planning_mode} | "
                f"{chunk.t_start:.4f}s - {chunk.t_end:.4f}s"
            )
            for chunk in report.chunk_plan
        )
    else:
        lines.append("  (none)")
    lines.append(f"Budget: {report.budget_status_line}")
    lines.append(f"Budget detail: {report.budget_estimate_summary}")
    lines.append(f"Overflow mitigation order: {report.overflow_mitigation_order_text}")
    lines.append(report.overflow_fallback_explanation)
    if report.warnings:
        lines.append("Info:")
        lines.extend(f"- {warning}" for warning in report.warnings)
    else:
        lines.append(f"Info: {report.overflow_fallback_explanation}")
    return "\n".join(lines)


def default_representative_frame_policy() -> VideoQARepresentativeFramePolicy:
    """Return the default representative-frame policy (middle of each span)."""
    return VideoQARepresentativeFramePolicy()


def default_overflow_policy() -> VideoQAOverflowPolicy:
    """Return the default overflow mitigation policy."""
    return VideoQAOverflowPolicy()


def deterministic_chunk_id(index: int, t_start: float, t_end: float) -> str:
    """Return a stable chunk identifier for planning and manifest rows."""
    return f"chunk-{index:04d}-{t_start:.4f}-{t_end:.4f}"


def _clamp_span(
    t_start: float, t_end: float, duration: float
) -> tuple[float, float] | None:
    """Clamp a span to ``[0, duration]``; return None if it becomes empty."""
    lo = max(0.0, min(t_start, duration))
    hi = max(0.0, min(t_end, duration))
    if hi <= lo:
        return None
    return (lo, hi)


def build_video_qa_chunk_plan(
    duration_seconds: float,
    *,
    scene_spans: Sequence[tuple[float, float]] | None = None,
    uniform_segment_seconds: float = 30.0,
    single_full_span_chunk: bool = False,
) -> tuple[VideoQAPlannedChunk, ...]:
    """Build a chunk plan from scene spans, or a uniform time grid if scenes are absent.

    Scene spans take precedence when the normalized list is non-empty. Otherwise the
    timeline ``[0, duration_seconds]`` is split into segments of at most
    ``uniform_segment_seconds`` (the last segment may be shorter).

    Args:
        duration_seconds: Total media length in seconds.
        scene_spans: Optional sequence of ``(t_start, t_end)`` scene boundaries in seconds.
        uniform_segment_seconds: Target segment length when falling back to a uniform grid.
        single_full_span_chunk: When True, plan one chunk covering the full timeline.

    Returns:
        Planned chunks with deterministic ``chunk_id`` values.

    """
    if duration_seconds <= 0:
        return ()

    if single_full_span_chunk:
        dur = float(duration_seconds)
        cid = deterministic_chunk_id(0, 0.0, dur)
        return (
            VideoQAPlannedChunk(
                chunk_id=cid,
                t_start=0.0,
                t_end=dur,
                planning_mode="whole_video",
            ),
        )

    normalized_scenes: list[tuple[float, float]] = []
    if scene_spans:
        for raw_start, raw_end in scene_spans:
            clamped = _clamp_span(float(raw_start), float(raw_end), duration_seconds)
            if clamped is not None:
                normalized_scenes.append(clamped)

    if normalized_scenes:
        normalized_scenes.sort(key=lambda span: span[0])
        planned: list[VideoQAPlannedChunk] = []
        for index, (t_start, t_end) in enumerate(normalized_scenes):
            cid = deterministic_chunk_id(index, t_start, t_end)
            planned.append(
                VideoQAPlannedChunk(
                    chunk_id=cid,
                    t_start=t_start,
                    t_end=t_end,
                    planning_mode="scene",
                )
            )
        return tuple(planned)

    seg = max(1e-9, float(uniform_segment_seconds))
    segment_count = max(1, math.ceil(duration_seconds / seg))
    step = duration_seconds / segment_count
    planned_uniform: list[VideoQAPlannedChunk] = []
    for index in range(segment_count):
        t_start = index * step
        t_end = duration_seconds if index == segment_count - 1 else (index + 1) * step
        cid = deterministic_chunk_id(index, t_start, t_end)
        planned_uniform.append(
            VideoQAPlannedChunk(
                chunk_id=cid,
                t_start=t_start,
                t_end=t_end,
                planning_mode="uniform_grid",
            )
        )
    return tuple(planned_uniform)


def merge_planned_chunks_into_manifest(
    manifest: VideoQARunManifest,
    planned_chunks: Sequence[VideoQAPlannedChunk],
) -> VideoQARunManifest:
    """Merge a new chunk plan into ``manifest`` by ``chunk_id`` for idempotent resume.

    For each ``chunk_id`` in the new plan: inserts a pending row if missing; keeps
    ``completed`` and ``running`` rows unchanged; updates ``t_start`` / ``t_end`` for
    ``pending`` and ``failed`` rows when coordinates change.

    Chunk records present in the manifest but not in the new plan are appended after
    the planned rows, in original manifest order, **for every status** (including
    ``pending``), so audit trails and resume state are preserved across replans.

    Args:
        manifest: Existing run manifest.
        planned_chunks: New planned chunks in execution order.

    Returns:
        A new manifest with merged ``chunks``; other fields are unchanged.

    """
    existing_by_id = {record.chunk_id: record for record in manifest.chunks}
    planned_ids = {chunk.chunk_id for chunk in planned_chunks}

    merged: list[VideoQAChunkRecord] = []

    for planned in planned_chunks:
        old = existing_by_id.get(planned.chunk_id)
        if old is None:
            merged.append(
                VideoQAChunkRecord(
                    chunk_id=planned.chunk_id,
                    t_start=planned.t_start,
                    t_end=planned.t_end,
                )
            )
            continue
        if old.status in ("completed", "running"):
            merged.append(old)
            continue
        merged.append(
            replace(
                old,
                t_start=planned.t_start,
                t_end=planned.t_end,
            )
        )

    for old in manifest.chunks:
        if old.chunk_id in planned_ids:
            continue
        merged.append(old)

    return replace(manifest, chunks=tuple(merged))


def build_video_qa_preflight_summary(  # noqa: PLR0913
    context: VideoQAContextBundle,
    *,
    duration_seconds: float,
    scene_spans: Sequence[tuple[float, float]] | None = None,
    uniform_segment_seconds: float = 30.0,
    single_full_span_chunk: bool = False,
    budget_policy: VideoQABudgetPolicy | None = None,
    runtime_policy: VideoQARuntimePolicy | None = None,
    overflow_policy: VideoQAOverflowPolicy | None = None,
    representative_policy: VideoQARepresentativeFramePolicy | None = None,
) -> VideoQAPreflightSummary:
    """Combine chunk planning with the existing budget estimate and merged warnings."""
    policy = budget_policy or default_video_qa_budget_policy()
    runtime = runtime_policy or default_video_qa_runtime_policy()
    overflow = overflow_policy or default_overflow_policy()
    rep = representative_policy or default_representative_frame_policy()

    chunk_warnings: list[str] = []
    if duration_seconds <= 0:
        chunk_warnings.append("Video duration is missing or zero; chunk plan is empty.")

    chunk_plan = build_video_qa_chunk_plan(
        duration_seconds,
        scene_spans=scene_spans,
        uniform_segment_seconds=uniform_segment_seconds,
        single_full_span_chunk=single_full_span_chunk,
    )

    if duration_seconds > 0 and not chunk_plan:
        chunk_warnings.append("Chunk plan is empty after normalization.")

    per_chunk_frame_counts = tuple(
        max(
            1,
            int(policy.minimum_frames_per_chunk),
            math.ceil(max(0.0, chunk.t_end - chunk.t_start) * policy.frame_sample_fps),
        )
        for chunk in chunk_plan
    )
    sampled_frame_count = sum(per_chunk_frame_counts)
    max_frames_per_chunk = max(per_chunk_frame_counts) if per_chunk_frame_counts else 0

    budget = build_video_qa_budget_estimate(
        context,
        chunk_count=len(chunk_plan),
        sampled_frame_count=sampled_frame_count,
        max_frames_per_chunk=max_frames_per_chunk,
        budget_policy=policy,
        runtime_policy=runtime,
    )

    combined = tuple(dict.fromkeys((*budget.warnings, *chunk_warnings)))
    return VideoQAPreflightSummary(
        budget=budget,
        chunk_plan=chunk_plan,
        overflow_policy=overflow,
        representative_frame_policy=rep,
        warnings=combined,
    )


def video_qa_planning_subtitle_separation_holds() -> bool:
    """Return True when planning data keeps transcript/subtitle work off chunk payloads.

    Enforces: :class:`VideoQAPlannedChunk` defines only timing/planning fields (no
    transcript or subtitle payload slots), and manifest graph kinds for transcript
    output (``transcript_prepare``) do not overlap QA chunk pipeline kinds
    (``chunk_plan`` … ``answer_aggregate``).
    """
    if frozenset(VideoQAPlannedChunk.__slots__) != _PLANNED_CHUNK_ALLOWED_SLOTS:
        return False
    transcript_kinds = set(VIDEO_QA_SUBTITLE_FIRST_GRAPH_KINDS)
    qa_chunk_kinds = set(VIDEO_QA_QA_CHUNK_GRAPH_KINDS)
    if not transcript_kinds.isdisjoint(qa_chunk_kinds):
        return False
    return transcript_kinds <= set(VIDEO_QA_ORCHESTRATION_STAGES)
