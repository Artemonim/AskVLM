"""Shared LLM prompt templates and structured schemas for AskVLM."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from .video_qa_context import VideoQAContextBundle


@dataclass(frozen=True, slots=True)
class VideoQAChunkSynthesisInput:
    """Normalized chunk-analysis data used by the final synthesis prompt."""

    t_start: float
    t_end: float
    transcript_excerpt: str
    chunk_summary: str
    observations: tuple[str, ...]
    confidence: str
    frame_refs: tuple[str, ...]


TEXT_FORMATTING_INSTRUCTION: Final[str] = (
    "You are a text formatter for ASR output. Add punctuation, restore casing, "
    "and split into paragraphs only when it improves readability.\n"
    "Keep the original language and do not invent, remove, or summarize content."
)


def build_text_formatting_prompt(text: str) -> str:
    """Build the prompt used by the optional transcription formatter."""
    normalized_text = str(text)
    return f"{TEXT_FORMATTING_INSTRUCTION}\nText:\n{normalized_text}\n\nFormatted:"


CHUNK_ANALYSIS_INSTRUCTION: Final[str] = (
    "Analyze the video chunk using the representative frames and transcript "
    "context below. Ground every observation in the provided evidence; use the "
    "question and attachments when they help disambiguate. If the evidence is "
    "weak or conflicting, lower confidence rather than guessing. Respond strictly "
    "with JSON that matches the provided schema."
)

CHUNK_ANALYSIS_JSON_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "chunk_summary": {
            "type": "string",
            "description": "Concise summary of this chunk.",
        },
        "observations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Short observations grounded in frames and transcript.",
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "Confidence in this chunk-level analysis.",
        },
    },
    "required": ["chunk_summary", "observations", "confidence"],
}


def build_chunk_analysis_prompt(
    context: VideoQAContextBundle,
    *,
    chunk_id: str | None = None,
    chunk_time_span: tuple[float, float] | None = None,
    transcript_summary: str | None = None,
    frame_refs: Iterable[str] = (),
) -> str:
    """Build the per-chunk multimodal prompt for LM Studio."""
    prompt_parts = [CHUNK_ANALYSIS_INSTRUCTION]
    context_block = context.render_prompt_block(
        chunk_id=chunk_id,
        chunk_time_span=chunk_time_span,
        transcript_summary=transcript_summary,
        frame_refs=frame_refs,
    )
    if context_block:
        prompt_parts.append(f"Context:\n{context_block}")
    return "\n\n".join(prompt_parts)


FINAL_SYNTHESIS_INSTRUCTION: Final[str] = (
    "Synthesize one final user-facing answer from the completed chunk analysis "
    "records. Prefer evidence that is consistent across chunks, keep the answer "
    "grounded in the supplied transcript quotes and frame refs, and set "
    "uncertainty when the evidence is incomplete or conflicting. Return strictly "
    "JSON that matches the provided schema. Do not expose chunk ids, hidden "
    "analysis steps, or chain-of-thought."
)

DIRECT_WHOLE_VIDEO_FINAL_INSTRUCTION: Final[str] = (
    "Answer the user's question using the attached video frames (sampled across the "
    "whole timeline) and the full transcript below. Ground every claim in visible "
    "evidence and transcript quotes; set uncertainty when the evidence is weak or "
    "incomplete. Return strictly JSON that matches the provided schema. Do not "
    "expose internal chain-of-thought."
)

FINAL_SYNTHESIS_JSON_SCHEMA: Final[dict[str, object]] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer": {
            "type": "string",
            "description": "One final answer to the user's question.",
        },
        "evidence": {
            "type": "array",
            "description": "Grounded evidence items supporting the final answer.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "transcript_quote": {"type": "string"},
                    "t_start": {"type": "number"},
                    "t_end": {"type": "number"},
                    "frame_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "transcript_quote",
                    "t_start",
                    "t_end",
                    "frame_refs",
                ],
            },
        },
        "is_uncertain": {"type": "boolean"},
        "uncertainty_note": {"type": ["string", "null"]},
    },
    "required": ["answer", "evidence", "is_uncertain", "uncertainty_note"],
}


def _chunk_input_to_payload(
    record: VideoQAChunkSynthesisInput,
) -> dict[str, object]:
    """Convert one chunk synthesis input into prompt-ready JSON data."""
    return {
        "t_start": record.t_start,
        "t_end": record.t_end,
        "transcript_excerpt": record.transcript_excerpt,
        "chunk_summary": record.chunk_summary,
        "observations": list(record.observations),
        "confidence": record.confidence,
        "frame_refs": list(record.frame_refs),
    }


def _render_final_request_frame_lines(frame_refs: Iterable[str]) -> list[str]:
    """Return prompt lines for chunk-start frames attached to the final request."""
    normalized = tuple(str(ref).strip() for ref in frame_refs if str(ref).strip())
    if not normalized:
        return []
    return [
        "Chunk-start frames attached in this request:",
        *(f"- {frame_ref}" for frame_ref in normalized),
    ]


def build_final_synthesis_prompt(
    context: VideoQAContextBundle,
    chunk_inputs: Sequence[VideoQAChunkSynthesisInput],
    *,
    transcript_body: str | None = None,
    final_frame_refs: Iterable[str] = (),
) -> str:
    """Build the final answer synthesis prompt from chunk analysis records."""
    prompt_parts = [FINAL_SYNTHESIS_INSTRUCTION]
    context_block = context.render_prompt_block()
    if context_block:
        prompt_parts.append(f"Context:\n{context_block}")
    body = str(transcript_body or "").strip()
    if body:
        prompt_parts.append(f"Full transcript:\n{body}")
    final_frame_lines = _render_final_request_frame_lines(final_frame_refs)
    if final_frame_lines:
        prompt_parts.append("\n".join(final_frame_lines))
    prompt_parts.append(
        "Completed chunk analysis records:\n"
        + json.dumps(
            [_chunk_input_to_payload(record) for record in chunk_inputs],
            ensure_ascii=False,
            indent=2,
        )
    )
    return "\n\n".join(prompt_parts)


def build_direct_whole_video_final_prompt(
    context: VideoQAContextBundle,
    *,
    transcript_body: str,
    chunk_id: str,
    chunk_time_span: tuple[float, float],
    frame_paths: Iterable[str],
) -> str:
    """Build the final solver prompt for one-shot whole-video multimodal QA."""
    prompt_parts = [DIRECT_WHOLE_VIDEO_FINAL_INSTRUCTION]
    normalized_frames = tuple(str(p).strip() for p in frame_paths if str(p).strip())
    context_block = context.render_prompt_block(
        chunk_id=chunk_id,
        chunk_time_span=chunk_time_span,
        transcript_summary=None,
        frame_refs=normalized_frames,
    )
    if context_block:
        prompt_parts.append(f"Context:\n{context_block}")
    body = str(transcript_body or "").strip()
    if body:
        prompt_parts.append(f"Full transcript:\n{body}")
    return "\n\n".join(prompt_parts)


__all__ = [
    "CHUNK_ANALYSIS_INSTRUCTION",
    "CHUNK_ANALYSIS_JSON_SCHEMA",
    "DIRECT_WHOLE_VIDEO_FINAL_INSTRUCTION",
    "FINAL_SYNTHESIS_INSTRUCTION",
    "FINAL_SYNTHESIS_JSON_SCHEMA",
    "TEXT_FORMATTING_INSTRUCTION",
    "VideoQAChunkSynthesisInput",
    "build_chunk_analysis_prompt",
    "build_direct_whole_video_final_prompt",
    "build_final_synthesis_prompt",
    "build_text_formatting_prompt",
]
