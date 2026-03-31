from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from core.video_qa_answer_bundle import (
    ANSWER_BUNDLE_SCHEMA_VERSION,
    VideoQAAnswerBundle,
)
from core.video_qa_context import normalize_video_qa_context
from core.video_qa_executor import (
    VideoQAChunkExecutionResult,
    VideoQAChunkInferenceOutcome,
    VideoQAChunkInferencer,
    VideoQAFrameMaterializer,
    VideoQATranscriptArtifacts,
    VideoQATranscriptProvider,
)
from core.video_qa_lm_studio_client import LMStudioResponse
from core.video_qa_local_run import (
    FINAL_SYNTHESIS_JSON_SCHEMA,
    VideoQAFFmpegFrameMaterializer,
    VideoQALMHttpTarget,
    VideoQALMStudioAnswerSynthesizer,
    VideoQALocalRunParams,
    VideoQAPreflightBlockedError,
    VideoQAWhisperTranscriptProvider,
    ensure_local_video_qa_run_allowed,
    map_video_qa_progress_frac_to_200,
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


def test_final_synthesizer_uses_chunk_artifacts_and_returns_single_answer(
    tmp_path: Path,
) -> None:
    """Final synthesis pass produces one answer from chunk artifact inputs."""
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

    captured: dict[str, object] = {}

    def fake_request(
        base_url: str,
        prompt: str,
        json_schema: object = None,
        **kwargs: object,
    ) -> LMStudioResponse:
        captured["base_url"] = base_url
        captured["prompt"] = prompt
        captured["json_schema"] = json_schema
        captured["kwargs"] = kwargs
        final_payload = {
            "answer": "A person opens the door and moves into view.",
            "evidence": [
                {
                    "transcript_quote": "door opens",
                    "t_start": 1.0,
                    "t_end": 3.0,
                    "frame_refs": [str(frame_path)],
                }
            ],
            "is_uncertain": False,
            "uncertainty_note": None,
        }
        return LMStudioResponse(
            content=json.dumps(final_payload),
            parsed_json=final_payload,
            used_fallback=False,
            finish_reason="stop",
            raw_response={},
        )

    agg = VideoQALMStudioAnswerSynthesizer(
        base_url="http://127.0.0.1:1234/v1",
        model="fake-model",
        request_chat_fn=fake_request,
    )
    bundle = agg.aggregate(
        context=ctx,
        manifest=manifest,
        transcript=transcript,
        chunk_results=chunk_results,
    )
    assert bundle.answer == "A person opens the door and moves into view."
    assert bundle.is_uncertain is True
    assert bundle.schema_version == ANSWER_BUNDLE_SCHEMA_VERSION
    assert len(bundle.evidence) == 1
    assert bundle.evidence[0].t_start == 1.0
    assert bundle.evidence[0].frame_refs == (str(frame_path),)
    assert captured["base_url"] == "http://127.0.0.1:1234/v1"
    assert captured["json_schema"] == FINAL_SYNTHESIS_JSON_SCHEMA
    assert "Person opens the door." in str(captured["prompt"])
    assert "chunk-1" not in bundle.answer


def test_ffmpeg_frame_materializer_falls_back_to_single_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frame materializer falls back when multi-frame extraction yields nothing."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    frames_dir = tmp_path / "frames"
    materializer = VideoQAFFmpegFrameMaterializer(
        video_path=video,
        frames_dir=frames_dir,
        sample_fps=2.0,
    )
    fallback_output = frames_dir / "chunk-1.png"

    def no_frames(
        _video_file: object,
        _start_s: float,
        _end_s: float,
        _output_pattern: Path,
        *,
        fps: float,
    ) -> tuple[Path, ...]:
        _ = (_video_file, _start_s, _end_s, _output_pattern, fps)
        return ()

    monkeypatch.setattr("core.video_qa_local_run.extract_frames_for_span", no_frames)

    def fake_extract_frame(
        _video_file: object,
        _timestamp_s: float,
        output_file: Path,
    ) -> Path:
        output_file.write_bytes(b"\x89PNG\r\n\x1a\n")
        return output_file

    monkeypatch.setattr(
        "core.video_qa_local_run.extract_frame_to_file", fake_extract_frame
    )

    manifest = VideoQARunManifest(
        schema_version=1,
        run_id="run-a",
        created_at="2026-03-30T12:00:00Z",
        source=None,
        question="Q",
        attachments=(),
        graph=(),
        chunks=(),
        status="pending",
    )
    frames = materializer.materialize_frames(
        chunk=VideoQAChunkRecord(chunk_id="chunk-1", t_start=0.0, t_end=1.0),
        representative_timestamp=0.5,
        manifest=manifest,
        transcript=VideoQATranscriptArtifacts("hello"),
    )

    assert frames == (str(fallback_output.resolve()),)
    assert fallback_output.exists()


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

    class _FixedAggregator:
        def aggregate(
            self,
            *,
            context: object,
            manifest: VideoQARunManifest,
            transcript: VideoQATranscriptArtifacts,
            chunk_results: object,
        ) -> VideoQAAnswerBundle:
            _ = (context, transcript, chunk_results)
            return VideoQAAnswerBundle(
                schema_version=ANSWER_BUNDLE_SCHEMA_VERSION,
                run_id=f"{manifest.run_id}-answer",
                created_at=manifest.created_at,
                question="Test?",
                answer="seen",
                evidence=(),
                is_uncertain=False,
                manifest_run_id=manifest.run_id,
                uncertainty_note=None,
            )

    whisper = MagicMock()
    lm = VideoQALMHttpTarget("http://127.0.0.1:9/v1", "fake-model")
    params = VideoQALocalRunParams(
        context=ctx,
        output_dir=tmp_path,
        context_window_tokens=200_000,
        chunk_lm=lm,
        final_lm=lm,
    )
    outcome = run_local_video_qa(
        params=params,
        whisper=whisper,
        transcript_override=_FixedTranscript(),
        frame_override=_FixedFrames(),
        chunk_inferencer_override=_FixedInferencer(),
        aggregator_override=_FixedAggregator(),
    )
    assert outcome.manifest.status == "completed"
    manifest_path = tmp_path / f"{outcome.manifest.run_id}.manifest.json"
    assert manifest_path.is_file()
    assert "seen" in outcome.answer_bundle.answer


def test_whisper_transcript_provider_unloads_after_transcribe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ASR stage releases GPU weights before chunk LM work (best-effort unload)."""
    clip = tmp_path / "v.mp4"
    clip.write_bytes(b"x")
    work = tmp_path / "w"
    work.mkdir()
    artifacts = VideoQATranscriptArtifacts("hi", segments=((0.0, 1.0, "hi"),))
    monkeypatch.setattr(
        "core.video_qa_local_run._whisper_transcribe_to_artifacts",
        lambda *_a, **_k: artifacts,
    )
    whisper = MagicMock()
    prov = VideoQAWhisperTranscriptProvider(
        whisper,
        media_path=clip,
        work_dir=work,
    )
    prov.prepare_transcript(MagicMock(), MagicMock())
    whisper.unload.assert_called_once_with(safe=False)


def test_map_video_qa_progress_frac_to_200_splits_pre_vlm_and_vlm() -> None:
    """First ~45% of internal progress maps to 0-100; remainder maps to 100-200."""
    assert map_video_qa_progress_frac_to_200(0.0) == 0
    assert map_video_qa_progress_frac_to_200(0.225) == 50
    assert map_video_qa_progress_frac_to_200(0.45) == 100
    assert map_video_qa_progress_frac_to_200(1.0) == 200
