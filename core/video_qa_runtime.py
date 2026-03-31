"""Runtime and budget policy helpers for Video QA."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from utils.askvlm_defaults import get_default_video_qa_canonical_model_id

if TYPE_CHECKING:
    from .video_qa_context import VideoQAContextBundle


class TextTokenCounterKind(StrEnum):
    """How question text tokens were counted for a budget estimate."""

    HEURISTIC = "heuristic"
    CUSTOM = "custom"


@runtime_checkable
class TextTokenCounter(Protocol):
    """Pluggable text token counter for budget estimation."""

    def count(self, text: str) -> int:
        """Return the estimated token count for ``text``."""


def _estimate_text_tokens(text: str) -> int:
    """Estimate text tokens using a conservative offline heuristic."""
    cleaned = str(text).strip()
    if not cleaned:
        return 0
    return max(16, math.ceil(len(cleaned) / 4))


@dataclass(frozen=True, slots=True)
class ConservativeHeuristicTextTokenCounter:
    """Default offline counter: ~4 characters per token, minimum floor."""

    def count(self, text: str) -> int:
        """Return the conservative heuristic token estimate."""
        return _estimate_text_tokens(text)


def resolve_text_token_counter(
    counter: TextTokenCounter | None,
) -> tuple[TextTokenCounter, TextTokenCounterKind]:
    """Resolve the counter to use and whether it was caller-supplied."""
    if counter is None:
        return ConservativeHeuristicTextTokenCounter(), TextTokenCounterKind.HEURISTIC
    return counter, TextTokenCounterKind.CUSTOM


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
class VideoQARuntimeScheduler:
    """First-class scheduler: one active heavy model and serialized stages.

    Parallel inference remains disabled unless ``allow_parallel_inference`` is changed
    explicitly on the embedded policy. GUI layers should not run concurrent heavy
    inference without an explicit queue.
    """

    policy: VideoQARuntimePolicy = field(default_factory=lambda: VideoQARuntimePolicy())

    @property
    def max_concurrent_active_heavy_models(self) -> int:
        """Return the cap on simultaneously active heavy models."""
        return self.policy.max_active_heavy_models

    @property
    def parallel_inference_enabled(self) -> bool:
        """Return True when parallel inference is allowed (default False)."""
        return self.policy.allow_parallel_inference

    def heavy_model_lifecycle_order(self) -> tuple[str, ...]:
        """Return the ordered unload path: active -> offload to RAM -> unload."""
        return self.policy.execution_order

    def single_active_heavy_model_holds(self) -> bool:
        """Return True when at most one heavy model may be active at a time."""
        return self.policy.max_active_heavy_models <= 1

    def summary(self) -> str:
        """Return a compact scheduler-focused description."""
        m = self.max_concurrent_active_heavy_models
        p = self.parallel_inference_enabled
        life = " -> ".join(self.heavy_model_lifecycle_order())
        return (
            f"Scheduler: max_concurrent_heavy={m}, parallel_inference={p}, "
            f"lifecycle={life}"
        )


@dataclass(frozen=True, slots=True)
class VideoQAModelProfile:
    """Data-only profile for the default LM Studio multimodal candidate.

    This profile never performs live model calls.
    """

    canonical_model_id: str = field(
        default_factory=get_default_video_qa_canonical_model_id
    )
    provider: str = "LM Studio"
    modalities: tuple[str, ...] = ("text", "image")
    structured_output_is_best_effort: bool = True

    def summary(self) -> str:
        """Return a one-line profile description."""
        modal = "+".join(self.modalities)
        structured = (
            "best-effort" if self.structured_output_is_best_effort else "assumed"
        )
        return (
            f"{self.canonical_model_id} via {self.provider} ({modal}); "
            f"structured_output={structured}"
        )

    def model_dependent_limitations(self) -> tuple[str, ...]:
        """Return model-specific limitations (quantization, build, LM Studio)."""
        return (
            (
                "MoE routing, quantization, and the loaded artifact affect latency "
                "and quality."
            ),
            "Vision/image input behavior is not guaranteed for every local build.",
            (
                "Structured output and JSON contracts are best-effort in the local "
                "stack."
            ),
            (
                "Usable context length and overflow signals depend on LM Studio and "
                "hardware."
            ),
        )

    def application_heuristic_notes(self) -> tuple[str, ...]:
        """Return notes about app-side heuristics that are not model-grounded."""
        return (
            (
                "Offline source-size and image attachment budgets use conservative "
                "heuristics."
            ),
            (
                "Chunk overhead and reserved instruction/output slots are app-level "
                "defaults."
            ),
            (
                "Text token counts use a pluggable counter; default is a conservative "
                "heuristic."
            ),
        )


@dataclass(frozen=True, slots=True)
class VideoQARuntimeSummary:
    """Bundle model profile, token counter mode, and scheduler for consumers."""

    model_profile: VideoQAModelProfile
    text_token_counter_mode: TextTokenCounterKind
    scheduler: VideoQARuntimeScheduler

    def summary(self) -> str:
        """Return a compact multi-line summary."""
        return (
            f"{self.model_profile.summary()}; "
            f"text_tokens={self.text_token_counter_mode}; "
            f"{self.scheduler.summary()}"
        )


def default_video_qa_target_model_profile() -> VideoQAModelProfile:
    """Return the canonical Video QA LM Studio candidate profile."""
    return VideoQAModelProfile()


def build_video_qa_runtime_summary(
    *,
    model_profile: VideoQAModelProfile | None = None,
    text_token_counter: TextTokenCounter | None = None,
    scheduler: VideoQARuntimeScheduler | None = None,
) -> VideoQARuntimeSummary:
    """Combine default or caller-provided profile, counter mode, and scheduler."""
    _, mode = resolve_text_token_counter(text_token_counter)
    return VideoQARuntimeSummary(
        model_profile=model_profile or default_video_qa_target_model_profile(),
        text_token_counter_mode=mode,
        scheduler=scheduler or default_video_qa_runtime_scheduler(),
    )


@dataclass(frozen=True, slots=True)
class VideoQABudgetPolicy:
    """Describe the budget heuristic used for Video QA preflight."""

    context_window_tokens: int = 8192
    reserved_output_tokens: int = 1536
    reserved_instruction_tokens: int = 512
    reserved_chunk_overhead_tokens: int = 128
    frame_sample_fps: float = 2.0
    minimum_frames_per_chunk: int = 1
    frame_tokens_per_sample: int = 256
    source_bytes_per_token: int = 2048
    min_source_tokens: int = 256
    max_source_tokens: int = 4096


def _estimate_source_tokens(size_bytes: int, policy: VideoQABudgetPolicy) -> int:
    """Estimate source-video tokens conservatively from file size."""
    if size_bytes <= 0:
        return 0
    raw = math.ceil(size_bytes / max(1, policy.source_bytes_per_token))
    return max(policy.min_source_tokens, min(policy.max_source_tokens, raw))


@dataclass(frozen=True, slots=True)
class VideoQABudgetEstimate:
    """Summarize the estimated Video QA token budget."""

    policy: VideoQABudgetPolicy
    runtime_policy: VideoQARuntimePolicy
    source_tokens_estimate: int
    question_tokens: int
    attachment_tokens: int
    sampled_frame_count: int
    frame_tokens_estimate: int
    chunk_overhead_tokens: int
    reserved_instruction_tokens: int
    reserved_output_tokens: int
    total_required_tokens: int
    text_token_counter_mode: TextTokenCounterKind = TextTokenCounterKind.HEURISTIC
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
        src = self.source_tokens_estimate
        q = self.question_tokens
        att = self.attachment_tokens
        sampled_frames = self.sampled_frame_count
        frame_tokens = self.frame_tokens_estimate
        ch = self.chunk_overhead_tokens
        ro = self.reserved_output_tokens
        tot = self.total_required_tokens
        cw = self.context_window_tokens
        tc = self.text_token_counter_mode
        return (
            f"Video QA budget: source={src}, question={q}, attachments={att}, "
            f"frames={sampled_frames} (~{frame_tokens} tokens), chunks={ch}, "
            f"reserved_output={ro}, "
            f"total={tot}/{cw} ({status}), text_counter={tc}"
        )


def default_video_qa_runtime_policy() -> VideoQARuntimePolicy:
    """Return the default runtime policy for Video QA."""
    return VideoQARuntimePolicy()


def default_video_qa_runtime_scheduler() -> VideoQARuntimeScheduler:
    """Return the default Video QA runtime scheduler wrapping the default policy."""
    return VideoQARuntimeScheduler(policy=default_video_qa_runtime_policy())


def default_video_qa_budget_policy() -> VideoQABudgetPolicy:
    """Return the default token budget policy for Video QA."""
    return VideoQABudgetPolicy()


def build_video_qa_budget_estimate(
    context: VideoQAContextBundle,
    *,
    chunk_count: int = 0,
    sampled_frame_count: int | None = None,
    budget_policy: VideoQABudgetPolicy | None = None,
    runtime_policy: VideoQARuntimePolicy | None = None,
    text_token_counter: TextTokenCounter | None = None,
) -> VideoQABudgetEstimate:
    """Estimate the Video QA budget for the current context bundle.

    Question text uses ``text_token_counter`` when provided; otherwise the
    conservative heuristic. Attachment totals still come from the context bundle
    (offline heuristics).
    """
    policy = budget_policy or default_video_qa_budget_policy()
    runtime = runtime_policy or default_video_qa_runtime_policy()
    counter, counter_mode = resolve_text_token_counter(text_token_counter)
    warnings: list[str] = []

    source_tokens = 0
    if context.source is None:
        warnings.append("No source file is selected.")
    else:
        source_tokens = _estimate_source_tokens(context.source.size_bytes, policy)

    question_tokens = counter.count(context.question)
    if not context.question.strip():
        warnings.append("Question text is empty.")

    attachment_tokens = context.attachment_budget_tokens

    chunk_total = max(0, int(chunk_count))
    if chunk_total <= 0:
        warnings.append("Chunk plan is not built yet.")
    default_frame_count = chunk_total * max(1, int(policy.minimum_frames_per_chunk))
    if sampled_frame_count is None:
        normalized_frame_count = default_frame_count
    else:
        normalized_frame_count = max(default_frame_count, 0, int(sampled_frame_count))
    frame_tokens_estimate = normalized_frame_count * max(
        0, policy.frame_tokens_per_sample
    )
    chunk_overhead_tokens = chunk_total * policy.reserved_chunk_overhead_tokens

    total_required_tokens = (
        source_tokens
        + question_tokens
        + attachment_tokens
        + frame_tokens_estimate
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
        sampled_frame_count=normalized_frame_count,
        frame_tokens_estimate=frame_tokens_estimate,
        chunk_overhead_tokens=chunk_overhead_tokens,
        reserved_instruction_tokens=policy.reserved_instruction_tokens,
        reserved_output_tokens=policy.reserved_output_tokens,
        total_required_tokens=total_required_tokens,
        text_token_counter_mode=counter_mode,
        warnings=tuple(warnings),
    )
