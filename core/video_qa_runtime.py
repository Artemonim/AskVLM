"""Runtime and budget policy helpers for Video QA."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .video_qa_context import VideoQAContextBundle


def _estimate_text_tokens(text: str) -> int:
    """Estimate text tokens using a conservative offline heuristic."""
    cleaned = str(text).strip()
    if not cleaned:
        return 0
    return max(16, math.ceil(len(cleaned) / 4))


def _estimate_source_tokens(size_bytes: int, policy: VideoQABudgetPolicy) -> int:
    """Estimate source-video tokens conservatively from file size."""
    if size_bytes <= 0:
        return 0
    raw = math.ceil(size_bytes / max(1, policy.source_bytes_per_token))
    return max(policy.min_source_tokens, min(policy.max_source_tokens, raw))


@dataclass(frozen=True, slots=True)
class VideoQARuntimePolicy:
    """Describe how Video QA model-heavy work is serialized."""

    max_active_heavy_models: int = 1
    allow_parallel_inference: bool = False
    serialize_model_heavy_steps: bool = True
    offload_to_ram_before_unload: bool = True
    profile_name: str = "8 GB VRAM / 64 GB RAM"
    execution_order: tuple[str, ...] = ("active", "offload_to_ram", "unload")

    def summary(self) -> str:
        """Return a compact human-readable description of the policy."""
        order = " -> ".join(self.execution_order)
        parallel = "disabled" if not self.allow_parallel_inference else "enabled"
        return (
            f"Runtime policy {self.profile_name}: max_active_heavy_models="
            f"{self.max_active_heavy_models}, parallel={parallel}, order={order}"
        )


@dataclass(frozen=True, slots=True)
class VideoQABudgetPolicy:
    """Describe the budget heuristic used for Video QA preflight."""

    context_window_tokens: int = 8192
    reserved_output_tokens: int = 1536
    reserved_instruction_tokens: int = 512
    reserved_chunk_overhead_tokens: int = 128
    source_bytes_per_token: int = 2048
    min_source_tokens: int = 256
    max_source_tokens: int = 4096


@dataclass(frozen=True, slots=True)
class VideoQABudgetEstimate:
    """Summarize the estimated Video QA token budget."""

    policy: VideoQABudgetPolicy
    runtime_policy: VideoQARuntimePolicy
    source_tokens_estimate: int
    question_tokens: int
    attachment_tokens: int
    chunk_overhead_tokens: int
    reserved_instruction_tokens: int
    reserved_output_tokens: int
    total_required_tokens: int
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def context_window_tokens(self) -> int:
        """Return the budget limit that was used for this estimate."""
        return self.policy.context_window_tokens

    @property
    def available_tokens(self) -> int:
        """Return the remaining tokens after reserved input/output."""
        return max(0, self.context_window_tokens - self.total_required_tokens)

    @property
    def fits(self) -> bool:
        """Return True when the estimate fits inside the configured window."""
        return self.total_required_tokens <= self.context_window_tokens

    @property
    def overflow_tokens(self) -> int:
        """Return the amount by which the estimate exceeds the context window."""
        return max(0, self.total_required_tokens - self.context_window_tokens)

    @property
    def summary(self) -> str:
        """Return a concise human-readable summary of the estimate."""
        status = "fits" if self.fits else f"over by {self.overflow_tokens} tokens"
        return (
            f"Video QA budget: source={self.source_tokens_estimate}, "
            f"question={self.question_tokens}, attachments={self.attachment_tokens}, "
            f"chunks={self.chunk_overhead_tokens}, reserved_output={self.reserved_output_tokens}, "
            f"total={self.total_required_tokens}/{self.context_window_tokens} ({status})"
        )


def default_video_qa_runtime_policy() -> VideoQARuntimePolicy:
    """Return the default runtime policy for Video QA."""
    return VideoQARuntimePolicy()


def default_video_qa_budget_policy() -> VideoQABudgetPolicy:
    """Return the default token budget policy for Video QA."""
    return VideoQABudgetPolicy()


def build_video_qa_budget_estimate(
    context: VideoQAContextBundle,
    *,
    chunk_count: int = 0,
    budget_policy: VideoQABudgetPolicy | None = None,
    runtime_policy: VideoQARuntimePolicy | None = None,
) -> VideoQABudgetEstimate:
    """Estimate the Video QA budget for the current context bundle."""
    policy = budget_policy or default_video_qa_budget_policy()
    runtime = runtime_policy or default_video_qa_runtime_policy()
    warnings: list[str] = []

    source_tokens = 0
    if context.source is None:
        warnings.append("No source file is selected.")
    else:
        source_tokens = _estimate_source_tokens(context.source.size_bytes, policy)

    question_tokens = _estimate_text_tokens(context.question)
    if not context.question.strip():
        warnings.append("Question text is empty.")

    attachment_tokens = context.attachment_budget_tokens

    chunk_total = max(0, int(chunk_count))
    if chunk_total <= 0:
        warnings.append("Chunk plan is not built yet.")
    chunk_overhead_tokens = chunk_total * policy.reserved_chunk_overhead_tokens

    total_required_tokens = (
        source_tokens
        + question_tokens
        + attachment_tokens
        + chunk_overhead_tokens
        + policy.reserved_instruction_tokens
        + policy.reserved_output_tokens
    )
    if total_required_tokens > policy.context_window_tokens:
        warnings.append("Estimated budget exceeds the configured context window.")

    return VideoQABudgetEstimate(
        policy=policy,
        runtime_policy=runtime,
        source_tokens_estimate=source_tokens,
        question_tokens=question_tokens,
        attachment_tokens=attachment_tokens,
        chunk_overhead_tokens=chunk_overhead_tokens,
        reserved_instruction_tokens=policy.reserved_instruction_tokens,
        reserved_output_tokens=policy.reserved_output_tokens,
        total_required_tokens=total_required_tokens,
        warnings=tuple(warnings),
    )
