from __future__ import annotations

from typing import TYPE_CHECKING

from core.video_qa_context import VideoQAAttachmentRequest, normalize_video_qa_context
from core.video_qa_runtime import (
    VideoQABudgetPolicy,
    build_video_qa_budget_estimate,
    default_video_qa_budget_policy,
    default_video_qa_runtime_policy,
)
from core.video_qa_sources import LocalFileProvider

if TYPE_CHECKING:
    from pathlib import Path


def test_runtime_policy_defaults_to_single_active_model() -> None:
    """Runtime policy defaults to a single active heavy model."""
    policy = default_video_qa_runtime_policy()

    assert policy.max_active_heavy_models == 1
    assert policy.allow_parallel_inference is False
    assert policy.serialize_model_heavy_steps is True
    assert "8 GB VRAM / 64 GB RAM" in policy.summary()


def test_budget_policy_defaults_are_conservative() -> None:
    """Budget policy defaults to a conservative offline heuristic."""
    policy = default_video_qa_budget_policy()

    assert policy.context_window_tokens == 8192
    assert policy.reserved_output_tokens > 0
    assert policy.reserved_instruction_tokens > 0
    assert policy.reserved_chunk_overhead_tokens > 0
    assert policy.source_bytes_per_token > 0
    assert policy.max_source_tokens >= policy.min_source_tokens


def test_budget_estimate_tracks_source_question_attachments_and_chunks(
    tmp_path: Path,
) -> None:
    """Budget estimate includes the source, question, attachments, and chunks."""
    source_media = tmp_path / "clip.mp4"
    source_media.write_bytes(b"abc")
    source = LocalFileProvider().resolve(source_media)

    notes = tmp_path / "notes.md"
    notes.write_text("one two three", encoding="utf-8")
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"\x89PNG\r\n\x1a\n")
    snippet = tmp_path / "snippet.py"
    snippet.write_text("print('hi')", encoding="utf-8")

    bundle = normalize_video_qa_context(
        source=source,
        question="What is shown in the clip?",
        attachments=[
            notes,
            VideoQAAttachmentRequest(path=snippet, enabled=False),
            frame,
        ],
    )
    estimate = build_video_qa_budget_estimate(bundle, chunk_count=3)

    assert estimate.source_tokens_estimate == 256
    assert estimate.question_tokens > 0
    assert estimate.attachment_tokens == bundle.attachment_budget_tokens
    assert (
        estimate.chunk_overhead_tokens
        == 3 * estimate.policy.reserved_chunk_overhead_tokens
    )
    assert estimate.available_tokens == (
        estimate.context_window_tokens - estimate.total_required_tokens
    )
    assert estimate.fits is True
    assert estimate.warnings == ()


def test_budget_estimate_warns_when_chunk_plan_is_missing(
    tmp_path: Path,
) -> None:
    """Budget estimate reports that the chunk plan is still missing."""
    source_media = tmp_path / "clip.mp4"
    source_media.write_bytes(b"abc")
    source = LocalFileProvider().resolve(source_media)

    bundle = normalize_video_qa_context(
        source=source,
        question="What is shown?",
        attachments=(),
    )
    estimate = build_video_qa_budget_estimate(bundle)

    assert any("chunk plan" in warning.lower() for warning in estimate.warnings)
    assert estimate.chunk_overhead_tokens == 0


def test_budget_estimate_marks_overflow_with_small_context_window(
    tmp_path: Path,
) -> None:
    """A tiny budget policy should trigger an overflow warning."""
    source_media = tmp_path / "clip.mp4"
    source_media.write_bytes(b"abcdef")
    source = LocalFileProvider().resolve(source_media)

    bundle = normalize_video_qa_context(
        source=source,
        question="What is shown in the clip?",
        attachments=(),
    )
    policy = VideoQABudgetPolicy(
        context_window_tokens=1024,
        reserved_output_tokens=512,
        reserved_instruction_tokens=256,
        reserved_chunk_overhead_tokens=256,
        source_bytes_per_token=1,
        min_source_tokens=512,
        max_source_tokens=512,
    )
    estimate = build_video_qa_budget_estimate(
        bundle,
        chunk_count=1,
        budget_policy=policy,
    )

    assert estimate.fits is False
    assert estimate.overflow_tokens > 0
    assert any("exceeds" in warning.lower() for warning in estimate.warnings)
