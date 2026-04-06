from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.video_qa_context import VideoQAAttachmentRequest, normalize_video_qa_context
from core.video_qa_runtime import (
    TextTokenCounterKind,
    VideoQABudgetPolicy,
    build_video_qa_budget_estimate,
    build_video_qa_runtime_summary,
    default_video_qa_budget_policy,
    default_video_qa_runtime_policy,
    default_video_qa_runtime_scheduler,
    default_video_qa_target_model_profile,
)
from core.video_qa_sources import LocalFileProvider
from utils.askvlm_defaults import get_default_video_qa_canonical_model_id

if TYPE_CHECKING:
    from pathlib import Path


def test_runtime_policy_defaults_to_single_active_model() -> None:
    """Runtime policy defaults to a single active heavy model."""
    policy = default_video_qa_runtime_policy()

    assert policy.max_active_heavy_models == 1
    assert policy.allow_parallel_inference is False
    assert policy.serialize_model_heavy_steps is True
    assert policy.offload_to_ram_before_unload is True
    assert policy.execution_order == ("active", "offload_to_ram", "unload")
    assert "8 GB VRAM / 64 GB RAM" in policy.summary()


def test_runtime_scheduler_single_heavy_model_and_lifecycle_order() -> None:
    """Scheduler exposes one active heavy model and fixed RAM offload order."""
    sched = default_video_qa_runtime_scheduler()

    assert sched.max_concurrent_active_heavy_models == 1
    assert sched.parallel_inference_enabled is False
    assert sched.single_active_heavy_model_holds() is True
    assert sched.heavy_model_lifecycle_order() == (
        "active",
        "offload_to_ram",
        "unload",
    )


def test_default_model_profile_fields_and_summary() -> None:
    """Target LM Studio profile is data-only with multimodal and best-effort JSON."""
    profile = default_video_qa_target_model_profile()

    assert profile.canonical_model_id == get_default_video_qa_canonical_model_id()
    assert profile.provider == "LM Studio"
    assert "text" in profile.modalities
    assert "image" in profile.modalities
    assert profile.structured_output_is_best_effort is True
    assert profile.canonical_model_id in profile.summary()
    assert "LM Studio" in profile.summary()
    assert len(profile.model_dependent_limitations()) >= 1
    assert len(profile.application_heuristic_notes()) >= 1


def test_build_video_qa_runtime_summary_combines_layers() -> None:
    """Runtime summary bundles profile, counter mode, and scheduler."""
    s = build_video_qa_runtime_summary()
    assert "Qwen" in s.model_profile.canonical_model_id
    assert s.text_token_counter_mode is TextTokenCounterKind.HEURISTIC
    assert s.scheduler.max_concurrent_active_heavy_models == 1
    assert "text_tokens=heuristic" in s.summary()


@dataclass(frozen=True, slots=True)
class _FixedTokenCounter:
    """Test double: fixed token count for any text."""

    n: int

    def count(self, _text: str) -> int:
        return self.n


def test_budget_estimate_uses_custom_text_token_counter(
    tmp_path: Path,
) -> None:
    """Custom counter drives question token line in the budget estimate."""
    source_media = tmp_path / "clip.mp4"
    source_media.write_bytes(b"abc")
    source = LocalFileProvider().resolve(source_media)

    bundle = normalize_video_qa_context(
        source=source,
        question="ignored for token count",
        attachments=(),
    )
    fixed = _FixedTokenCounter(n=99)
    estimate = build_video_qa_budget_estimate(
        bundle,
        chunk_count=1,
        text_token_counter=fixed,
    )

    assert estimate.question_tokens == 99
    assert estimate.text_token_counter_mode is TextTokenCounterKind.CUSTOM


def test_budget_policy_defaults_are_conservative() -> None:
    """Budget policy defaults to a conservative offline heuristic."""
    policy = default_video_qa_budget_policy()

    assert policy.context_window_tokens == 8192
    assert policy.reserved_output_tokens > 0
    assert policy.reserved_instruction_tokens > 0
    assert policy.reserved_chunk_overhead_tokens > 0
    assert policy.frame_sample_fps == 2.0
    assert policy.minimum_frames_per_chunk == 1
    assert policy.frame_tokens_per_sample > 0
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
    estimate = build_video_qa_budget_estimate(
        bundle,
        chunk_count=3,
        sampled_frame_count=6,
    )

    assert estimate.source_tokens_estimate == 256
    assert estimate.question_tokens > 0
    assert estimate.attachment_tokens == bundle.attachment_budget_tokens
    assert estimate.sampled_frame_count == 6
    assert estimate.peak_frames_per_chunk == 2
    assert estimate.frame_tokens_estimate == (
        2 * estimate.policy.frame_tokens_per_sample
    )
    assert (
        estimate.chunk_overhead_tokens == estimate.policy.reserved_chunk_overhead_tokens
    )
    assert estimate.available_tokens == (
        estimate.context_window_tokens - estimate.total_required_tokens
    )
    assert estimate.fits is True
    assert estimate.warnings == ()


def test_budget_estimate_caps_large_text_attachment_preview(
    tmp_path: Path,
) -> None:
    """Large text attachments use a bounded inline preview for budgeting."""
    source_media = tmp_path / "clip.mp4"
    source_media.write_bytes(b"abc")
    source = LocalFileProvider().resolve(source_media)

    large_text = "0123456789abcdef\n" * 6000
    notes = tmp_path / "big.txt"
    notes.write_text(large_text, encoding="utf-8")

    bundle = normalize_video_qa_context(
        source=source,
        question="What is shown?",
        attachments=[notes],
    )

    preview = bundle.attachment_text_previews[0]
    assert preview is not None
    assert len(preview) < len(large_text)
    assert bundle.attachments[0].budget_tokens < 5000

    estimate = build_video_qa_budget_estimate(bundle, chunk_count=1)
    assert estimate.attachment_tokens == bundle.attachment_budget_tokens


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
    assert estimate.sampled_frame_count == 0
    assert estimate.peak_frames_per_chunk == 0
    assert estimate.frame_tokens_estimate == 0
    assert estimate.chunk_overhead_tokens == 0


def test_budget_estimate_respects_explicit_max_frames_per_chunk(
    tmp_path: Path,
) -> None:
    """Peak frames can exceed total/chunk_count when spans are uneven."""
    source_media = tmp_path / "clip.mp4"
    source_media.write_bytes(b"abc")
    source = LocalFileProvider().resolve(source_media)
    bundle = normalize_video_qa_context(
        source=source,
        question="What is shown?",
        attachments=(),
    )
    estimate = build_video_qa_budget_estimate(
        bundle,
        chunk_count=2,
        sampled_frame_count=120,
        max_frames_per_chunk=100,
    )
    assert estimate.sampled_frame_count == 120
    assert estimate.peak_frames_per_chunk == 100
    assert estimate.frame_tokens_estimate == (
        100 * estimate.policy.frame_tokens_per_sample
    )


def test_budget_estimate_uses_minimum_frame_cost_when_only_chunk_count_is_known(
    tmp_path: Path,
) -> None:
    """Budget estimate keeps a visible frame cost even without chunk durations."""
    source_media = tmp_path / "clip.mp4"
    source_media.write_bytes(b"abc")
    source = LocalFileProvider().resolve(source_media)

    bundle = normalize_video_qa_context(
        source=source,
        question="What is shown?",
        attachments=(),
    )
    estimate = build_video_qa_budget_estimate(bundle, chunk_count=3)

    assert estimate.sampled_frame_count == 3
    assert estimate.peak_frames_per_chunk == 1
    assert estimate.frame_tokens_estimate == (
        1 * estimate.policy.frame_tokens_per_sample
    )


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
