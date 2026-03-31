from __future__ import annotations

from typing import TYPE_CHECKING

from core.video_qa_context import normalize_video_qa_context
from core.video_qa_manifest import (
    SCHEMA_VERSION,
    VideoQAChunkRecord,
    VideoQARunManifest,
)
from core.video_qa_orchestration import (
    VIDEO_QA_ORCHESTRATION_STAGES,
    VIDEO_QA_QA_CHUNK_GRAPH_KINDS,
    VIDEO_QA_SUBTITLE_FIRST_GRAPH_KINDS,
    VideoQAOverflowPolicy,
    VideoQAPlannedChunk,
    VideoQARepresentativeFramePolicy,
    build_video_qa_chunk_plan,
    build_video_qa_preflight_report,
    build_video_qa_preflight_summary,
    deterministic_chunk_id,
    format_video_qa_overflow_mitigation_order,
    format_video_qa_preflight_report_text,
    merge_planned_chunks_into_manifest,
    video_qa_planning_subtitle_separation_holds,
)
from core.video_qa_runtime import VideoQABudgetPolicy, build_video_qa_budget_estimate
from core.video_qa_sources import LocalFileProvider

if TYPE_CHECKING:
    from pathlib import Path


def test_chunk_plan_from_scene_spans(tmp_path: Path) -> None:
    """Scene spans produce one planned chunk per scene in time order."""
    plan = build_video_qa_chunk_plan(
        120.0,
        scene_spans=((0.0, 30.0), (30.0, 90.0), (90.0, 120.0)),
    )
    assert len(plan) == 3
    assert all(c.planning_mode == "scene" for c in plan)
    assert plan[0].t_start == 0.0
    assert plan[0].t_end == 30.0
    assert plan[1].t_start == 30.0
    assert plan[1].t_end == 90.0
    assert plan[2].t_start == 90.0
    assert plan[2].t_end == 120.0


def test_chunk_plan_uniform_grid_fallback() -> None:
    """Without scenes, the timeline splits into a uniform grid by segment length."""
    plan = build_video_qa_chunk_plan(100.0, uniform_segment_seconds=25.0)
    assert len(plan) == 4
    assert all(c.planning_mode == "uniform_grid" for c in plan)
    assert plan[0].t_start == 0.0
    assert plan[0].t_end == 25.0
    assert plan[-1].t_end == 100.0


def test_representative_frame_middle_timestamp() -> None:
    """Default policy picks the midpoint of the chunk span."""
    policy = VideoQARepresentativeFramePolicy(kind="middle_timestamp")
    assert policy.timestamp_for_span(0.0, 10.0) == 5.0
    assert policy.timestamp_for_span(1.0, 3.0) == 2.0


def test_representative_frame_first_and_last() -> None:
    """Extension kinds anchor to span endpoints."""
    first = VideoQARepresentativeFramePolicy(kind="first_timestamp")
    last = VideoQARepresentativeFramePolicy(kind="last_timestamp")
    assert first.timestamp_for_span(2.0, 8.0) == 2.0
    assert last.timestamp_for_span(2.0, 8.0) == 8.0


def test_manifest_merge_preserves_completed_and_updates_pending() -> None:
    """Re-planning updates pending boundaries while freezing completed rows."""
    cid = deterministic_chunk_id(0, 0.0, 25.0)
    manifest = VideoQARunManifest(
        schema_version=SCHEMA_VERSION,
        run_id="run-1",
        created_at="2026-03-29T12:00:00Z",
        source=None,
        question="Q",
        attachments=(),
        graph=(),
        chunks=(
            VideoQAChunkRecord(
                chunk_id=cid,
                t_start=0.0,
                t_end=25.0,
                status="completed",
                attempts=1,
                frames=("a.png",),
            ),
            VideoQAChunkRecord(
                chunk_id="chunk-pending",
                t_start=0.0,
                t_end=1.0,
                status="pending",
            ),
        ),
    )
    new_plan = (
        VideoQAPlannedChunk(
            chunk_id=cid,
            t_start=0.0,
            t_end=99.0,
            planning_mode="uniform_grid",
        ),
        VideoQAPlannedChunk(
            chunk_id="chunk-pending",
            t_start=5.0,
            t_end=10.0,
            planning_mode="uniform_grid",
        ),
    )
    merged = merge_planned_chunks_into_manifest(manifest, new_plan)
    by_id = {c.chunk_id: c for c in merged.chunks}
    assert by_id[cid].t_end == 25.0
    assert by_id[cid].status == "completed"
    assert by_id[cid].frames == ("a.png",)
    assert by_id["chunk-pending"].t_start == 5.0
    assert by_id["chunk-pending"].t_end == 10.0


def test_manifest_merge_idempotent_for_completed_chunks() -> None:
    """Two identical merges leave completed chunk rows unchanged."""
    cid = deterministic_chunk_id(0, 0.0, 10.0)
    manifest = VideoQARunManifest(
        schema_version=SCHEMA_VERSION,
        run_id="run-1",
        created_at="2026-03-29T12:00:00Z",
        source=None,
        question="Q",
        attachments=(),
        graph=(),
        chunks=(
            VideoQAChunkRecord(
                chunk_id=cid,
                t_start=0.0,
                t_end=10.0,
                status="completed",
                attempts=2,
                frames=("x.png",),
            ),
        ),
    )
    plan = (
        VideoQAPlannedChunk(
            chunk_id=cid,
            t_start=0.0,
            t_end=50.0,
            planning_mode="scene",
        ),
    )
    once = merge_planned_chunks_into_manifest(manifest, plan)
    twice = merge_planned_chunks_into_manifest(once, plan)
    assert once.chunks == twice.chunks


def test_manifest_merge_retains_orphan_failed_chunk() -> None:
    """Failed chunks not in the new plan are retained for audit."""
    manifest = VideoQARunManifest(
        schema_version=SCHEMA_VERSION,
        run_id="run-1",
        created_at="2026-03-29T12:00:00Z",
        source=None,
        question="Q",
        attachments=(),
        graph=(),
        chunks=(
            VideoQAChunkRecord(
                chunk_id="orphan-fail",
                t_start=0.0,
                t_end=1.0,
                status="failed",
                attempts=3,
                error="timeout",
            ),
        ),
    )
    plan = (
        VideoQAPlannedChunk(
            chunk_id="new-only",
            t_start=0.0,
            t_end=5.0,
            planning_mode="uniform_grid",
        ),
    )
    merged = merge_planned_chunks_into_manifest(manifest, plan)
    ids = [c.chunk_id for c in merged.chunks]
    assert ids == ["new-only", "orphan-fail"]


def test_manifest_merge_retains_orphan_pending_chunk() -> None:
    """Pending chunks absent from the replan are kept for audit and resume."""
    manifest = VideoQARunManifest(
        schema_version=SCHEMA_VERSION,
        run_id="run-1",
        created_at="2026-03-29T12:00:00Z",
        source=None,
        question="Q",
        attachments=(),
        graph=(),
        chunks=(
            VideoQAChunkRecord(
                chunk_id="orphan-pending",
                t_start=1.0,
                t_end=2.0,
                status="pending",
            ),
        ),
    )
    plan = (
        VideoQAPlannedChunk(
            chunk_id="planned-a",
            t_start=0.0,
            t_end=5.0,
            planning_mode="uniform_grid",
        ),
    )
    merged = merge_planned_chunks_into_manifest(manifest, plan)
    ids = [c.chunk_id for c in merged.chunks]
    assert ids == ["planned-a", "orphan-pending"]
    assert merged.chunks[1].status == "pending"
    assert merged.chunks[1].t_start == 1.0


def test_preflight_summary_merges_budget_and_chunk_warnings(tmp_path: Path) -> None:
    """Preflight combines runtime budget warnings with chunk-plan warnings."""
    source_media = tmp_path / "clip.mp4"
    source_media.write_bytes(b"x")
    source = LocalFileProvider().resolve(source_media)
    bundle = normalize_video_qa_context(
        source=source,
        question="What happens?",
        attachments=(),
    )
    summary = build_video_qa_preflight_summary(
        bundle,
        duration_seconds=0.0,
    )
    assert summary.chunk_plan == ()
    assert any("duration" in w.lower() for w in summary.warnings)
    budget_only = build_video_qa_budget_estimate(bundle, chunk_count=0)
    assert set(summary.warnings) >= set(budget_only.warnings)


def test_preflight_overflow_policy_in_summary(tmp_path: Path) -> None:
    """Preflight exposes overflow policy and ties budget to chunk count."""
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"abc")
    source = LocalFileProvider().resolve(clip)
    bundle = normalize_video_qa_context(source=source, question="Q?", attachments=())
    policy = tiny_budget()
    custom = VideoQAOverflowPolicy(
        mitigation_steps=(
            "reduce_frame_count",
            "reduce_resolution",
            "split_chunk_or_text",
        )
    )
    summary = build_video_qa_preflight_summary(
        bundle,
        duration_seconds=60.0,
        uniform_segment_seconds=30.0,
        budget_policy=policy,
        overflow_policy=custom,
    )
    assert summary.overflow_policy == custom
    assert len(summary.chunk_plan) == 2
    assert summary.budget.sampled_frame_count == 120
    assert summary.budget.peak_frames_per_chunk == 60
    assert summary.budget.frame_tokens_estimate == (60 * policy.frame_tokens_per_sample)
    assert summary.budget.chunk_overhead_tokens == (
        policy.reserved_chunk_overhead_tokens
    )


def test_preflight_report_includes_budget_chunks_warnings_overflow_text(
    tmp_path: Path,
) -> None:
    """Preflight report surfaces budget status, full chunk plan, warnings, and policy."""
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x" * 9000)
    source = LocalFileProvider().resolve(clip)
    bundle = normalize_video_qa_context(
        source=source,
        question="What is in the video?",
        attachments=(),
    )
    # * Narrow window so the offline estimate reliably exceeds the limit.
    bp = VideoQABudgetPolicy(
        context_window_tokens=800,
        reserved_output_tokens=512,
        reserved_instruction_tokens=256,
        reserved_chunk_overhead_tokens=64,
        source_bytes_per_token=2048,
        min_source_tokens=256,
        max_source_tokens=1024,
    )
    overflow = VideoQAOverflowPolicy(
        mitigation_steps=(
            "reduce_frame_count",
            "reduce_resolution",
            "split_chunk_or_text",
        )
    )
    preflight = build_video_qa_preflight_summary(
        bundle,
        duration_seconds=60.0,
        uniform_segment_seconds=30.0,
        budget_policy=bp,
        overflow_policy=overflow,
    )
    report = build_video_qa_preflight_report(bundle, preflight)
    assert report.chunk_plan == preflight.chunk_plan
    assert len(report.chunk_plan) == 2
    assert all(c.planning_mode == "uniform_grid" for c in report.chunk_plan)
    assert report.chunk_count == 2
    assert report.budget_fits is False
    assert "over by" in report.budget_status_line
    assert "frames" in report.budget_status_line
    assert "total=" in report.budget_estimate_summary
    assert "frames_total" in report.budget_estimate_summary
    assert preflight.warnings
    assert report.warnings == preflight.warnings
    assert "1. reduce frame count" in report.overflow_mitigation_order_text
    assert "2. reduce frame resolution" in report.overflow_mitigation_order_text
    assert "3. split chunk" in report.overflow_mitigation_order_text
    assert report.source_summary is not None
    assert str(clip.resolve()) in report.source_summary
    assert report.question == "What is in the video?"
    text = format_video_qa_preflight_report_text(report)
    assert "Chunks: 2" in text
    assert "Chunk plan:" in text
    assert report.chunk_plan[0].chunk_id in text
    assert "frames" in text
    assert "Overflow mitigation order:" in text
    assert "Warnings:" in text


def test_preflight_report_fits_shows_non_overflow_explanation(tmp_path: Path) -> None:
    """When the estimate fits, the fallback explanation stays informational."""
    clip = tmp_path / "small.mp4"
    clip.write_bytes(b"abc")
    source = LocalFileProvider().resolve(clip)
    bundle = normalize_video_qa_context(source=source, question="Hi", attachments=())
    budget_policy = VideoQABudgetPolicy(
        context_window_tokens=50000,
        reserved_output_tokens=512,
        reserved_instruction_tokens=256,
        reserved_chunk_overhead_tokens=64,
        frame_tokens_per_sample=128,
        source_bytes_per_token=2048,
        min_source_tokens=256,
        max_source_tokens=1024,
    )
    preflight = build_video_qa_preflight_summary(
        bundle,
        duration_seconds=30.0,
        uniform_segment_seconds=30.0,
        budget_policy=budget_policy,
    )
    report = build_video_qa_preflight_report(bundle, preflight)
    assert len(report.chunk_plan) == 1
    assert report.chunk_plan[0].t_end == 30.0
    assert report.budget_fits is True
    assert "fits" in report.budget_status_line
    assert "frames" in report.budget_status_line
    assert "Offline estimate fits" in report.overflow_fallback_explanation
    assert "exceeds" not in report.overflow_fallback_explanation


def test_format_overflow_mitigation_order_custom_steps() -> None:
    """Overflow order formatter lists only configured steps in order."""
    policy = VideoQAOverflowPolicy(
        mitigation_steps=("reduce_resolution", "split_chunk_or_text")
    )
    text = format_video_qa_overflow_mitigation_order(policy)
    assert text.startswith("1. reduce frame resolution")
    assert "2. split chunk" in text
    assert "reduce frame count" not in text


def tiny_budget() -> VideoQABudgetPolicy:
    """Small context window for deterministic chunk overhead math."""
    return VideoQABudgetPolicy(
        context_window_tokens=8192,
        reserved_output_tokens=512,
        reserved_instruction_tokens=256,
        reserved_chunk_overhead_tokens=64,
        source_bytes_per_token=2048,
        min_source_tokens=256,
        max_source_tokens=1024,
    )


def test_subtitle_first_separation_invariants() -> None:
    """Planned chunks carry only timing fields; transcript graph kinds stay separate."""
    assert video_qa_planning_subtitle_separation_holds()
    assert VIDEO_QA_ORCHESTRATION_STAGES == (
        "source_resolve",
        "attachment_prepare",
        "transcript_prepare",
        "chunk_plan",
        "frame_select",
        "llm_pass",
        "answer_aggregate",
    )
    assert VIDEO_QA_SUBTITLE_FIRST_GRAPH_KINDS == ("transcript_prepare",)
    assert set(VIDEO_QA_SUBTITLE_FIRST_GRAPH_KINDS).isdisjoint(
        set(VIDEO_QA_QA_CHUNK_GRAPH_KINDS)
    )
    chunk = VideoQAPlannedChunk(
        chunk_id="c1",
        t_start=0.0,
        t_end=1.0,
    )
    assert getattr(chunk, "__slots__", ()) == (
        "chunk_id",
        "t_start",
        "t_end",
        "planning_mode",
    )


def test_scene_spans_clipped_to_duration() -> None:
    """Scene spans are clamped to media duration."""
    plan = build_video_qa_chunk_plan(
        50.0,
        scene_spans=((0.0, 100.0),),
    )
    assert len(plan) == 1
    assert plan[0].t_end == 50.0


def test_empty_scene_spans_falls_back_to_uniform() -> None:
    """All-invalid scenes fall back to the uniform grid."""
    plan = build_video_qa_chunk_plan(
        40.0,
        scene_spans=((2.0, 1.0),),
        uniform_segment_seconds=20.0,
    )
    assert len(plan) == 2
    assert all(c.planning_mode == "uniform_grid" for c in plan)
