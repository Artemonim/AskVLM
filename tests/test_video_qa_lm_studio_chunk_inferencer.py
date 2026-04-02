"""Unit tests for the LM Studio Video QA chunk inferencer."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from core.llm_prompts import CHUNK_ANALYSIS_JSON_SCHEMA
from core.video_qa_context import normalize_video_qa_context
from core.video_qa_executor import VideoQATranscriptArtifacts
from core.video_qa_lm_studio_chunk_inferencer import (
    VideoQALMStudioChunkInferencer,
)
from core.video_qa_lm_studio_client import LMStudioClientError, LMStudioResponse
from core.video_qa_manifest import VideoQAChunkRecord
from core.video_qa_preparation import build_video_qa_preparation_manifest
from core.video_qa_sources import LocalFileProvider

if TYPE_CHECKING:
    from pathlib import Path


def test_inferencer_prompt_includes_chunk_sections_and_passes_frame_paths(
    tmp_path: Path,
) -> None:
    """Prompt uses render_prompt_block; client receives frame paths as images."""
    clip = tmp_path / "v.mp4"
    clip.write_bytes(b"x")
    notes = tmp_path / "notes.md"
    notes.write_text("one two three", encoding="utf-8")
    source = LocalFileProvider().resolve(clip)
    bundle = normalize_video_qa_context(
        source=source, question="What happens?", attachments=(notes,)
    )
    manifest = build_video_qa_preparation_manifest(
        bundle, run_id="test-run", created_at="2026-03-30T12:00:00Z"
    )

    captured: dict[str, object] = {}

    def fake_request(
        base_url: str,
        prompt: str,
        image_paths: object = None,
        json_schema: object = None,
        **kwargs: object,
    ) -> LMStudioResponse:
        _ = (base_url, kwargs)
        captured["prompt"] = prompt
        captured["image_paths"] = tuple(image_paths) if image_paths else ()
        captured["json_schema"] = json_schema
        payload = {
            "chunk_summary": "s",
            "observations": ["a"],
            "confidence": "high",
        }
        return LMStudioResponse(
            content=json.dumps(payload),
            parsed_json=payload,
            used_fallback=False,
            finish_reason="stop",
            raw_response={},
        )

    chunk = VideoQAChunkRecord(
        chunk_id="c-1",
        t_start=1.0,
        t_end=3.5,
    )
    frames = (str(tmp_path / "f1.png"), str(tmp_path / "f2.png"))
    transcript = VideoQATranscriptArtifacts(
        transcript_text="full",
        subtitle_text="",
        segments=((1.0, 2.0, "line one"),),
    )

    inf = VideoQALMStudioChunkInferencer(
        bundle,
        base_url="http://127.0.0.1:1234/v1",
        request_chat_fn=fake_request,
    )
    out = inf.infer_chunk(
        chunk=chunk,
        frames=frames,
        transcript=transcript,
        manifest=manifest,
    )

    prompt = str(captured["prompt"])
    assert "Chunk:" in prompt
    assert "- id: c-1" in prompt
    assert "- span: 1.00s to 3.50s" in prompt
    assert "Attachments:" in prompt
    assert "notes.md" in prompt
    assert "Transcript summary:" in prompt
    assert "line one" in prompt
    assert "Representative frames:" in prompt
    assert str(tmp_path / "f1.png") in prompt
    assert str(tmp_path / "f2.png") in prompt

    assert captured["image_paths"] == frames
    assert captured["json_schema"] == CHUNK_ANALYSIS_JSON_SCHEMA
    assert out.ok is True


def test_inferencer_success_normalizes_outcome(tmp_path: Path) -> None:
    """Structured JSON is normalized into a single manifest artifact string."""
    clip = tmp_path / "v.mp4"
    clip.write_bytes(b"x")
    bundle = normalize_video_qa_context(
        source=LocalFileProvider().resolve(clip), question=""
    )
    manifest = build_video_qa_preparation_manifest(
        bundle, run_id="test-run", created_at="2026-03-30T12:00:00Z"
    )

    def fake_request(
        _base_url: str,
        _prompt: str,
        image_paths: object = None,
        json_schema: object = None,
        **kwargs: object,
    ) -> LMStudioResponse:
        _ = (image_paths, json_schema, kwargs)
        payload = {
            "chunk_summary": "Summary text",
            "observations": ["x", "y"],
            "confidence": "medium",
        }
        return LMStudioResponse(
            content=json.dumps(payload),
            parsed_json=payload,
            used_fallback=False,
            finish_reason="stop",
            raw_response={},
        )

    inf = VideoQALMStudioChunkInferencer(
        bundle,
        base_url="http://localhost/v1",
        request_chat_fn=fake_request,
    )
    out = inf.infer_chunk(
        chunk=VideoQAChunkRecord(chunk_id="x", t_start=0.0, t_end=1.0),
        frames=(),
        transcript=VideoQATranscriptArtifacts(transcript_text=""),
        manifest=manifest,
    )
    assert out.ok is True
    assert out.error is None
    assert len(out.artifacts) == 1
    parsed = json.loads(out.artifacts[0])
    assert parsed["chunk_summary"] == "Summary text"
    assert parsed["observations"] == ["x", "y"]
    assert parsed["confidence"] == "medium"


def test_inferencer_surfaces_client_errors(tmp_path: Path) -> None:
    """LMStudioClientError maps to ok=False with a message."""
    clip = tmp_path / "v.mp4"
    clip.write_bytes(b"x")
    bundle = normalize_video_qa_context(
        source=LocalFileProvider().resolve(clip), question=""
    )
    manifest = build_video_qa_preparation_manifest(
        bundle, run_id="test-run", created_at="2026-03-30T12:00:00Z"
    )

    def fake_request(
        *_args: object,
        **_kwargs: object,
    ) -> LMStudioResponse:
        msg = "HTTP 503: service unavailable"
        raise LMStudioClientError(msg)

    inf = VideoQALMStudioChunkInferencer(
        bundle,
        base_url="http://localhost/v1",
        request_chat_fn=fake_request,
    )
    out = inf.infer_chunk(
        chunk=VideoQAChunkRecord(chunk_id="e", t_start=0.0, t_end=1.0),
        frames=(),
        transcript=VideoQATranscriptArtifacts(transcript_text="t"),
        manifest=manifest,
    )
    assert out.ok is False
    assert out.artifacts == ()
    assert out.error is not None
    assert "503" in out.error


def test_inferencer_unparseable_response_fails(tmp_path: Path) -> None:
    """Missing or invalid JSON yields ok=False."""
    clip = tmp_path / "v.mp4"
    clip.write_bytes(b"x")
    bundle = normalize_video_qa_context(
        source=LocalFileProvider().resolve(clip), question=""
    )
    manifest = build_video_qa_preparation_manifest(
        bundle, run_id="test-run", created_at="2026-03-30T12:00:00Z"
    )

    def fake_request(
        *_args: object,
        **_kwargs: object,
    ) -> LMStudioResponse:
        return LMStudioResponse(
            content="not json",
            parsed_json=None,
            used_fallback=True,
            finish_reason="stop",
            raw_response={},
        )

    inf = VideoQALMStudioChunkInferencer(
        bundle,
        base_url="http://localhost/v1",
        request_chat_fn=fake_request,
    )
    out = inf.infer_chunk(
        chunk=VideoQAChunkRecord(chunk_id="u", t_start=0.0, t_end=1.0),
        frames=(),
        transcript=VideoQATranscriptArtifacts(transcript_text=""),
        manifest=manifest,
    )
    assert out.ok is False
    assert out.error is not None
    assert "parse" in (out.error or "").lower()
