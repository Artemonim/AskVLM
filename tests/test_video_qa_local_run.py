from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from core.video_qa_answer_bundle import ANSWER_BUNDLE_SCHEMA_VERSION
from core.video_qa_context import normalize_video_qa_context
from core.video_qa_executor import (
    VideoQAChunkExecutionResult,
    VideoQAChunkInferenceOutcome,
    VideoQAChunkInferencer,
    VideoQAFrameMaterializer,
    VideoQATranscriptArtifacts,
    VideoQATranscriptProvider,
)
from core.video_qa_local_run import (
    VideoQADeterministicAnswerAggregator,
    VideoQALocalRunParams,
    VideoQAPreflightBlockedError,
    ensure_local_video_qa_run_allowed,
    run_local_video_qa,
)
from core.video_qa_manifest import VideoQAChunkRecord, VideoQARunManifest
from core.video_qa_orchestration import VideoQAPlannedChunk, VideoQAPreflightReport
from core.video_qa_sources import LocalFileProvider

if TYPE_CHECKING:
    from pathlib import Path


def test_ensure_local_run_blocked_when_budget_does_not_fit() -> None:
    """Preflight guard rejects runs when the offline budget does not fit."""
    report = VideoQAPreflightReport(
        source_summary="clip.mp4",
        question="What happens?",
        chunk_plan=(
            VideoQAPlannedChunk(
                chunk_id="c1",
                t_start=0.0,
                t_end=1.0,
                planning_mode="uniform_grid",
            ),
        ),
        budget_fits=False,
        budget_status_line="over",
        budget_estimate_summary="x",
        warnings=(),
        overflow_mitigation_order_text="a; b",
        overflow_fallback_explanation="overflow",
    )
    with pytest.raises(VideoQAPreflightBlockedError):
        ensure_local_video_qa_run_allowed(report, report.chunk_plan)


def test_deterministic_aggregator_reads_chunk_json_artifacts(
    tmp_path: Path,
) -> None:
    """Aggregator builds an answer bundle from chunk artifact JSON strings."""
    payload = {
        "chunk_summary": "Person opens the door.",
        "observations": ["Motion visible"],
        "confidence": "low",
    }
    manifest = VideoQARunManifest(
        schema_version=1,
        run_id="run-a",
        created_at="2026-03-30T12:00:00Z",
        source=None,
        question="Q",
        attachments=(),
        graph=(),
        chunks=(
            VideoQAChunkRecord(
                chunk_id="chunk-1",
                t_start=1.0,
                t_end=3.0,
                artifacts=(json.dumps(payload, sort_keys=True),),
                status="completed",
            ),
        ),
        status="completed",
    )
    ctx = normalize_video_qa_context(source=None, question="What?", attachments=())
    transcript = VideoQATranscriptArtifacts("full text")
    frame_path = tmp_path / "frame.png"
    frame_path.write_bytes(b"x")
    chunk_results = (
        VideoQAChunkExecutionResult(
            chunk_id="chunk-1",
            status="completed",
            frames=(str(frame_path),),
        ),
    )
    agg = VideoQADeterministicAnswerAggregator()
    bundle = agg.aggregate(
        context=ctx,
        manifest=manifest,
        transcript=transcript,
        chunk_results=chunk_results,
    )
    assert "Person opens the door." in bundle.answer
    assert bundle.is_uncertain is True
    assert bundle.schema_version == ANSWER_BUNDLE_SCHEMA_VERSION
    assert len(bundle.evidence) == 1
    assert bundle.evidence[0].t_start == 1.0


def test_run_local_video_qa_with_injected_stack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full local run uses injected adapters so CI does not need ffmpeg or a model."""
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"not-a-real-mp4")
    source = LocalFileProvider().resolve(clip)
    ctx = normalize_video_qa_context(source=source, question="Test?", attachments=())

    monkeypatch.setattr(
        "core.video_qa_local_run.get_media_duration_seconds",
        lambda _p: 120.0,
    )

    class _FixedTranscript(VideoQATranscriptProvider):
        def prepare_transcript(
            self, c: object, m: object
        ) -> VideoQATranscriptArtifacts:
            _ = (c, m)
            return VideoQATranscriptArtifacts(
                "hello world",
                segments=((0.0, 2.0, "hello world"),),
            )

    class _FixedFrames(VideoQAFrameMaterializer):
        def materialize_frames(
            self,
            *,
            chunk: VideoQAChunkRecord,
            representative_timestamp: float,
            manifest: VideoQARunManifest,
            transcript: VideoQATranscriptArtifacts,
        ) -> tuple[str, ...]:
            _ = (representative_timestamp, manifest, transcript)
            p = tmp_path / f"{chunk.chunk_id}.png"
            p.write_bytes(b"\x89PNG\r\n\x1a\n")
            return (str(p),)

    class _FixedInferencer(VideoQAChunkInferencer):
        def infer_chunk(
            self,
            *,
            chunk: VideoQAChunkRecord,
            frames: tuple[str, ...],
            transcript: VideoQATranscriptArtifacts,
            manifest: VideoQARunManifest,
        ) -> VideoQAChunkInferenceOutcome:
            _ = (chunk, frames, transcript, manifest)
            pl = {
                "chunk_summary": "seen",
                "observations": [],
                "confidence": "high",
            }
            return VideoQAChunkInferenceOutcome(
                ok=True,
                artifacts=(json.dumps(pl, sort_keys=True),),
            )

    whisper = MagicMock()
    params = VideoQALocalRunParams(
        context=ctx,
        output_dir=tmp_path,
        context_window_tokens=200_000,
        lm_base_url="http://127.0.0.1:9/v1",
        lm_model_id="fake-model",
    )
    outcome = run_local_video_qa(
        params=params,
        whisper=whisper,
        transcript_override=_FixedTranscript(),
        frame_override=_FixedFrames(),
        chunk_inferencer_override=_FixedInferencer(),
    )
    assert outcome.manifest.status == "completed"
    manifest_path = tmp_path / f"{outcome.manifest.run_id}.manifest.json"
    assert manifest_path.is_file()
    assert "seen" in outcome.answer_bundle.answer
