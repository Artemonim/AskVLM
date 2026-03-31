"""LM Studio-backed per-chunk Video QA inference using the shared prompt contract."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

from .video_qa_executor import (
    VideoQAChunkInferenceOutcome,
)
from .video_qa_lm_studio_client import (
    LMStudioClientError,
    _parse_json_candidate,
    request_chat_completion,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from .video_qa_context import VideoQAContextBundle
    from .video_qa_executor import (
        VideoQATranscriptArtifacts,
    )
    from .video_qa_lm_studio_client import (
        LMStudioResponse,
    )
    from .video_qa_manifest import VideoQAChunkRecord, VideoQARunManifest

logger = logging.getLogger(__name__)

CHUNK_ANALYSIS_INSTRUCTION: Final[str] = (
    "Analyze the video chunk using the representative frames and transcript context "
    "below. Respond strictly with JSON that matches the provided schema."
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


def _transcript_summary_for_chunk(
    transcript: VideoQATranscriptArtifacts,
    chunk: VideoQAChunkRecord,
    *,
    max_chars: int = 4000,
) -> str:
    """Build a transcript excerpt for the chunk time span, with a whole-text fallback."""
    lines: list[str] = []
    for t0, t1, text in transcript.segments:
        if t1 >= chunk.t_start and t0 <= chunk.t_end:
            stripped = text.strip()
            if stripped:
                lines.append(stripped)
    if lines:
        joined = "\n".join(lines)
        return joined if len(joined) <= max_chars else joined[: max_chars - 3] + "..."
    body = transcript.transcript_text.strip()
    if not body:
        body = transcript.subtitle_text.strip()
    if not body:
        return ""
    return body if len(body) <= max_chars else body[: max_chars - 3] + "..."


def _parse_json_loose(text: str) -> dict[str, Any] | None:
    """Parse JSON object from model text, including fenced blocks."""
    parsed = _parse_json_candidate(text)
    if isinstance(parsed, dict):
        return parsed
    return None


def _normalize_chunk_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return a stable JSON-serializable payload for manifest artifacts."""
    observations = tuple(str(x) for x in raw["observations"])
    return {
        "chunk_summary": str(raw["chunk_summary"]),
        "observations": observations,
        "confidence": str(raw["confidence"]),
    }


def _validate_chunk_payload(obj: object) -> dict[str, Any] | None:
    """Validate chunk analysis JSON; returns normalized dict or None."""
    if not isinstance(obj, Mapping):
        return None
    summary = obj.get("chunk_summary")
    observations = obj.get("observations")
    confidence = obj.get("confidence")
    if not isinstance(summary, str):
        return None
    if not isinstance(observations, list) or not all(
        isinstance(x, str) for x in observations
    ):
        return None
    if confidence not in ("low", "medium", "high"):
        return None
    return _normalize_chunk_payload(
        {
            "chunk_summary": summary,
            "observations": observations,
            "confidence": confidence,
        }
    )


class VideoQALMStudioChunkInferencer:
    """Chunk-level multimodal inference via LM Studio structured JSON."""

    def __init__(
        self,
        context: VideoQAContextBundle,
        *,
        base_url: str,
        model: str = "local-model",
        temperature: float = 0.0,
        timeout: float | None = None,
        request_chat_fn: Callable[..., LMStudioResponse] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> None:
        self._context = context
        self._base_url = base_url
        self._model = model
        self._temperature = temperature
        self._timeout = timeout
        self._request_fn = request_chat_fn or request_chat_completion
        self._should_cancel = should_cancel

    def infer_chunk(
        self,
        *,
        chunk: VideoQAChunkRecord,
        frames: tuple[str, ...],
        transcript: VideoQATranscriptArtifacts,
        manifest: VideoQARunManifest,
    ) -> VideoQAChunkInferenceOutcome:
        """Run one LM Studio request for the chunk and return normalized artifacts."""
        _ = manifest
        summary = _transcript_summary_for_chunk(transcript, chunk)
        block = self._context.render_prompt_block(
            chunk_id=chunk.chunk_id,
            chunk_time_span=(chunk.t_start, chunk.t_end),
            transcript_summary=summary if summary else None,
            frame_refs=frames,
        )
        prompt = f"{CHUNK_ANALYSIS_INSTRUCTION}\n\n{block}"
        try:
            response = self._request_fn(
                self._base_url,
                prompt,
                image_paths=frames,
                json_schema=CHUNK_ANALYSIS_JSON_SCHEMA,
                model=self._model,
                temperature=self._temperature,
                timeout=self._timeout,
                should_cancel=self._should_cancel,
            )
        except LMStudioClientError as exc:
            return VideoQAChunkInferenceOutcome(ok=False, error=str(exc))

        payload_obj: object | None = response.parsed_json
        if payload_obj is None:
            payload_obj = _parse_json_loose(response.content)
        if payload_obj is None:
            msg = (
                "Could not parse structured chunk analysis from the LM Studio response."
            )
            logger.warning("%s Raw content prefix: %s", msg, response.content[:200])
            return VideoQAChunkInferenceOutcome(ok=False, error=msg)

        normalized = _validate_chunk_payload(payload_obj)
        if normalized is None:
            return VideoQAChunkInferenceOutcome(
                ok=False,
                error="Chunk analysis JSON did not match the expected shape.",
            )
        artifact = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
        return VideoQAChunkInferenceOutcome(ok=True, artifacts=(artifact,))


__all__ = [
    "CHUNK_ANALYSIS_INSTRUCTION",
    "CHUNK_ANALYSIS_JSON_SCHEMA",
    "VideoQALMStudioChunkInferencer",
]
