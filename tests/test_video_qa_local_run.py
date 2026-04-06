from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from core.llm_prompts import CHUNK_ANALYSIS_JSON_SCHEMA, FINAL_SYNTHESIS_JSON_SCHEMA
from core.video_qa_answer_bundle import (
    ANSWER_BUNDLE_SCHEMA_VERSION,
    VideoQAAnswerBundle,
    answer_bundle_path_for_manifest,
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
    VIDEO_QA_WHISPER_UNLOAD_MODE_ENV,
    VideoQAFFmpegFrameMaterializer,
    VideoQAFinalRequestOptions,
    VideoQALMHttpTarget,
    VideoQALMStudioAnswerSynthesizer,
    VideoQALMStudioDirectWholeVideoSolver,
    VideoQALocalRunOptions,
    VideoQALocalRunParams,
    VideoQAPreflightBlockedError,
    VideoQAWhisperTranscriptProvider,
    _get_video_qa_whisper_unload_mode,
    _VideoQAFinalSynthesisOptions,
    _VideoQALMStudioRequestOptions,
    _whisper_transcribe_to_artifacts,
    ensure_local_video_qa_run_allowed,
    map_video_qa_progress_frac_to_200,
    preflight_local_video_qa,
    run_local_video_qa,
)
from core.video_qa_manifest import VideoQAChunkRecord, VideoQARunManifest
from core.video_qa_orchestration import (
    VideoQAPlannedChunk,
    VideoQAPreflightReport,
    build_video_qa_chunk_plan,
)
from core.video_qa_sources import LocalFileProvider

if TYPE_CHECKING:
    from core.video_qa_executor import (
        VideoQAExecutorRunOutcome,
    )

_FIXTURE_SHORT_MP4 = (
    Path(__file__).resolve().parent / "fixtures" / "test_video_short.mp4"
)


def _force_uniform_video_qa_chunk_plan(
    monkeypatch: pytest.MonkeyPatch,
    *,
    segment_seconds: float,
) -> None:
    """Force deterministic uniform chunking for short-fixture smoke coverage."""
    original = build_video_qa_chunk_plan

    def _forced_chunk_plan(
        duration_seconds: float,
        *,
        scene_spans: object = None,
        uniform_segment_seconds: float = 30.0,
        single_full_span_chunk: bool = False,
    ) -> tuple[VideoQAPlannedChunk, ...]:
        _ = (scene_spans, uniform_segment_seconds)
        return original(
            duration_seconds,
            scene_spans=None,
            uniform_segment_seconds=segment_seconds,
            single_full_span_chunk=single_full_span_chunk,
        )

    monkeypatch.setattr(
        "core.video_qa_orchestration.build_video_qa_chunk_plan",
        _forced_chunk_plan,
    )


def test_preflight_local_video_qa_single_span_when_chunking_disabled(
    tmp_path: Path,
) -> None:
    """Disabling video chunking plans one whole-timeline chunk."""
    media = tmp_path / "c.mp4"
    media.write_bytes(b"x")
    ctx = normalize_video_qa_context(
        source=LocalFileProvider().resolve(media),
        question="Q?",
        attachments=(),
    )
    _, plan = preflight_local_video_qa(
        ctx,
        duration_seconds=90.0,
        context_window_tokens=100_000,
        frame_sample_fps=1.0,
        video_chunking_enabled=False,
    )
    assert len(plan) == 1
    assert plan[0].planning_mode == "whole_video"
    assert plan[0].t_end == 90.0


def _write_test_png(path: Path) -> Path:
    """Write a tiny PNG-like blob and return the path for fluent test setup."""
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    return path


def _run_direct_whole_video_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[VideoQAExecutorRunOutcome, dict[str, object], Path]:
    """Run the direct whole-video path with mocked transcript, frames, and final call."""
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"not-a-real-mp4")
    source = LocalFileProvider().resolve(clip)
    note = tmp_path / "direct-note.txt"
    note.write_text("direct attachment content", encoding="utf-8")
    image_attachment = _write_test_png(tmp_path / "reference.png")
    ctx = normalize_video_qa_context(
        source=source,
        question="Test?",
        attachments=[note, image_attachment],
    )
    monkeypatch.setattr(
        "core.video_qa_local_run.get_media_duration_seconds",
        lambda _p: 30.0,
    )
    monkeypatch.setattr(
        "core.video_qa_local_run.openai_chat_base_to_local_rest_root",
        lambda _url: "http://127.0.0.1:1234",
    )

    def fail_lm_studio_load_model(*_args: object, **_kwargs: object) -> None:
        msg = "direct whole-video mode must not call lm_studio_load_model"
        raise AssertionError(msg)

    monkeypatch.setattr(
        "core.video_qa_local_run.lm_studio_load_model",
        fail_lm_studio_load_model,
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
            return (str(_write_test_png(tmp_path / f"{chunk.chunk_id}.png")),)

    class _FixedTranscript(VideoQATranscriptProvider):
        def prepare_transcript(
            self, c: object, m: object
        ) -> VideoQATranscriptArtifacts:
            _ = (c, m)
            return VideoQATranscriptArtifacts(
                "hello transcript",
                segments=((0.0, 2.0, "hello transcript"),),
            )

    class _RaisingInferencer(VideoQAChunkInferencer):
        def infer_chunk(
            self,
            *,
            chunk: VideoQAChunkRecord,
            frames: tuple[str, ...],
            transcript: VideoQATranscriptArtifacts,
            manifest: VideoQARunManifest,
        ) -> VideoQAChunkInferenceOutcome:
            _ = (chunk, frames, transcript, manifest)
            msg = "chunk inferencer must not run when video_chunking_enabled is false"
            raise AssertionError(msg)

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
            "answer": "Direct answer",
            "evidence": [
                {
                    "transcript_quote": "hello",
                    "t_start": 0.0,
                    "t_end": 2.0,
                    "frame_refs": [],
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

    solver = VideoQALMStudioDirectWholeVideoSolver(
        base_url="http://127.0.0.1:9/v1",
        model="final-model",
        request_options=_VideoQALMStudioRequestOptions(
            request_chat_fn=fake_request,
            reasoning="on",
        ),
    )

    whisper = MagicMock()
    lm = VideoQALMHttpTarget("http://127.0.0.1:9/v1", "fake-chunk-model")
    final_lm = VideoQALMHttpTarget("http://127.0.0.1:9/v1", "final-model")
    outcome = run_local_video_qa(
        params=VideoQALocalRunParams(
            context=ctx,
            output_dir=tmp_path,
            context_window_tokens=200_000,
            chunk_lm=lm,
            final_lm=final_lm,
            video_chunking_enabled=False,
            run_options=VideoQALocalRunOptions(reasoning_enabled=True),
        ),
        whisper=whisper,
        transcript_override=_FixedTranscript(),
        frame_override=_FixedFrames(),
        chunk_inferencer_override=_RaisingInferencer(),
        direct_whole_video_solver=solver,
    )
    return outcome, captured, image_attachment


def test_run_local_video_qa_no_chunking_skips_inferencer_and_solver_gets_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct whole-video path forwards transcript, frames, image attachments, and reasoning."""
    outcome, captured, image_attachment = _run_direct_whole_video_case(
        tmp_path, monkeypatch
    )

    assert outcome.manifest.status == "completed"
    assert outcome.answer_bundle.answer == "Direct answer"
    assert len(outcome.chunk_results) == 1
    assert outcome.chunk_results[0].inference_attempted is False
    assert not any(s.startswith("llm_pass:") for s in outcome.stage_sequence), (
        "chunk llm_pass must not be emitted"
    )
    assert "answer_aggregate" in outcome.stage_sequence
    assert "transcript_prepare" in outcome.stage_sequence
    assert captured["json_schema"] == FINAL_SYNTHESIS_JSON_SCHEMA
    prompt_str = str(captured["prompt"])
    assert "Full transcript:" in prompt_str
    assert "hello transcript" in prompt_str
    assert "direct attachment content" in prompt_str
    img_paths = (
        captured["kwargs"].get("image_paths")
        if isinstance(captured["kwargs"], dict)
        else None
    )
    assert img_paths is not None
    assert isinstance(img_paths, tuple)
    assert len(img_paths) == 2
    assert outcome.chunk_results[0].frames == (str(img_paths[0]),)
    assert img_paths[1] == image_attachment.resolve()
    assert captured["kwargs"].get("reasoning") == "on"
    manifest_path = tmp_path / f"{outcome.manifest.run_id}.manifest.json"
    assert manifest_path.is_file()
    assert answer_bundle_path_for_manifest(manifest_path).is_file()


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
    """Final synthesis can include transcript, chunk-start frames, attachments, and reasoning."""
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
    attachment = tmp_path / "summary.txt"
    attachment.write_text("attached summary", encoding="utf-8")
    image_attachment = tmp_path / "reference.png"
    image_attachment.write_bytes(b"\x89PNG\r\n\x1a\n")
    ctx = normalize_video_qa_context(
        source=None,
        question="What?",
        attachments=[attachment, image_attachment],
    )
    transcript = VideoQATranscriptArtifacts("full text")
    frame_path = tmp_path / "frame.png"
    frame_path.write_bytes(b"x")
    start_frame_path = tmp_path / "chunk-1-start.png"
    start_frame_path.write_bytes(b"y")
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
        request_options=_VideoQALMStudioRequestOptions(
            request_chat_fn=fake_request,
            reasoning={"effort": "low"},
        ),
        synthesis_options=_VideoQAFinalSynthesisOptions(
            include_transcript_in_final_prompt=True,
            include_start_frame_per_chunk_in_final_request=True,
            chunk_start_frame_paths_provider=lambda _manifest, _results: (
                str(start_frame_path),
            ),
        ),
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
    assert "attached summary" in str(captured["prompt"])
    assert "Full transcript:" in str(captured["prompt"])
    assert "full text" in str(captured["prompt"])
    assert "Chunk-start frames attached in this request:" in str(captured["prompt"])
    assert str(start_frame_path) in str(captured["prompt"])
    assert captured["kwargs"]["image_paths"] == (
        image_attachment.resolve(),
        str(start_frame_path),
    )
    assert captured["kwargs"]["reasoning"] == {"effort": "low"}
    assert "chunk-1" not in bundle.answer


def _run_chunked_run_options_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    VideoQAExecutorRunOutcome,
    dict[str, list[dict[str, object]]],
    list[str],
    Path,
]:
    """Run a chunked scenario that exercises reasoning and final-request enrichments."""
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"video")
    prepared_audio = tmp_path / "prepared.wav"
    prepared_audio.write_bytes(b"wav")
    note = tmp_path / "summary.txt"
    note.write_text("attached summary", encoding="utf-8")
    image_attachment = _write_test_png(tmp_path / "reference.png")
    source = LocalFileProvider().resolve(clip)
    ctx = normalize_video_qa_context(
        source=source,
        question="What happens?",
        attachments=[note, image_attachment],
    )
    monkeypatch.setattr(
        "core.video_qa_local_run.get_media_duration_seconds",
        lambda _p: 16.0,
    )
    monkeypatch.setattr(
        "core.video_qa_local_run.prepare_audio",
        lambda *_args, **_kwargs: prepared_audio,
    )
    monkeypatch.setattr(
        "core.video_qa_local_run.cleanup_intermediate_audio",
        lambda *_args, **_kwargs: None,
    )
    _force_uniform_video_qa_chunk_plan(monkeypatch, segment_seconds=8.0)

    start_frame_calls: list[str] = []

    def fake_extract_frame(
        _video_file: object,
        _timestamp_s: float,
        output_file: Path,
    ) -> Path:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        _write_test_png(output_file)
        start_frame_calls.append(str(output_file.resolve()))
        return output_file

    monkeypatch.setattr(
        "core.video_qa_local_run.extract_frame_to_file",
        fake_extract_frame,
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
            return (str(_write_test_png(tmp_path / f"{chunk.chunk_id}.png")),)

    captured: dict[str, list[dict[str, object]]] = {"chunk": [], "final": []}

    def fake_request(
        base_url: str,
        prompt: str,
        image_paths: object = None,
        json_schema: object = None,
        **kwargs: object,
    ) -> LMStudioResponse:
        _ = base_url
        payload = {
            "image_paths": tuple(image_paths) if image_paths else (),
            "prompt": prompt,
            "kwargs": kwargs,
        }
        if json_schema == CHUNK_ANALYSIS_JSON_SCHEMA:
            captured["chunk"].append(payload)
            chunk_payload = {
                "chunk_summary": "seen",
                "observations": ["obs"],
                "confidence": "high",
            }
            return LMStudioResponse(
                content=json.dumps(chunk_payload),
                parsed_json=chunk_payload,
                used_fallback=False,
                finish_reason="stop",
                raw_response={},
            )
        if json_schema == FINAL_SYNTHESIS_JSON_SCHEMA:
            captured["final"].append(payload)
            final_payload = {
                "answer": "final",
                "evidence": [],
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
        msg = "unexpected schema"
        raise AssertionError(msg)

    monkeypatch.setattr("core.video_qa_local_run.request_chat_completion", fake_request)
    monkeypatch.setattr(
        "core.video_qa_lm_studio_chunk_inferencer.request_chat_completion",
        fake_request,
    )

    whisper = MagicMock()
    whisper.transcribe.return_value = {
        "text": "speaker explains the scene",
        "segments": [
            {"start": 0.0, "end": 4.0, "text": " speaker explains "},
            {"start": 8.0, "end": 12.0, "text": " the scene "},
        ],
    }
    lm = VideoQALMHttpTarget("http://127.0.0.1:9/v1", "fake-model")
    params = VideoQALocalRunParams(
        context=ctx,
        output_dir=tmp_path,
        context_window_tokens=200_000,
        chunk_lm=lm,
        final_lm=lm,
        run_options=VideoQALocalRunOptions(
            final_request=VideoQAFinalRequestOptions(
                include_transcript=True,
                include_start_frame_per_chunk=True,
            ),
            reasoning_enabled=True,
        ),
    )

    outcome = run_local_video_qa(
        params=params,
        whisper=whisper,
        frame_override=_FixedFrames(),
    )
    return outcome, captured, start_frame_calls, image_attachment


def test_run_local_video_qa_chunked_run_options_forward_reasoning_and_final_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chunked run options reach chunk/final requests, transcript, and start-frame images."""
    outcome, captured, start_frame_calls, image_attachment = (
        _run_chunked_run_options_case(tmp_path, monkeypatch)
    )

    assert outcome.manifest.status == "completed"
    assert len(captured["chunk"]) == 2
    assert all(call["kwargs"]["reasoning"] == "on" for call in captured["chunk"])
    assert all(
        call["image_paths"][1] == image_attachment.resolve()
        for call in captured["chunk"]
    )
    assert len(captured["final"]) == 1
    final_call = captured["final"][0]
    assert final_call["kwargs"]["reasoning"] == "on"
    assert "Full transcript:" in str(final_call["prompt"])
    assert "speaker explains the scene" in str(final_call["prompt"])
    assert "Chunk-start frames attached in this request:" in str(final_call["prompt"])
    assert final_call["image_paths"][0] == image_attachment.resolve()
    assert tuple(final_call["image_paths"][1:]) == tuple(start_frame_calls)
    assert len(start_frame_calls) == 2


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


def test_run_local_video_qa_pipeline_smoke_short_fixture_media(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """E2E smoke: short committed mp4 exercises a forced multi-chunk run with mocks."""
    assert _FIXTURE_SHORT_MP4.is_file(), (
        "tests/fixtures/test_video_short.mp4 must exist"
    )
    monkeypatch.setattr(
        "core.video_qa_local_run.get_media_duration_seconds",
        lambda _path: 16.088,
    )
    _force_uniform_video_qa_chunk_plan(monkeypatch, segment_seconds=10.0)
    source = LocalFileProvider().resolve(_FIXTURE_SHORT_MP4)
    ctx = normalize_video_qa_context(
        source=source,
        question="Smoke test question?",
        attachments=(),
    )

    class _FixedTranscript(VideoQATranscriptProvider):
        def prepare_transcript(
            self, c: object, m: object
        ) -> VideoQATranscriptArtifacts:
            _ = (c, m)
            return VideoQATranscriptArtifacts(
                "fixture smoke transcript",
                segments=((0.0, 2.0, "fixture smoke transcript"),),
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
                "chunk_summary": "fixture_chunk_ok",
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
                question="Smoke test question?",
                answer="fixture smoke ok",
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
    assert "transcript_prepare" in outcome.stage_sequence
    assert "answer_aggregate" in outcome.stage_sequence
    manifest_path = tmp_path / f"{outcome.manifest.run_id}.manifest.json"
    assert manifest_path.is_file()
    answer_path = answer_bundle_path_for_manifest(manifest_path)
    assert answer_path.is_file()
    manifest_obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_obj.get("status") == "completed"
    assert isinstance(manifest_obj.get("chunks"), list)
    assert len(manifest_obj["chunks"]) == 2
    assert all(chunk["status"] == "completed" for chunk in manifest_obj["chunks"])
    answer_obj = json.loads(answer_path.read_text(encoding="utf-8"))
    assert isinstance(answer_obj.get("answer"), str)
    assert answer_obj.get("schema_version") == ANSWER_BUNDLE_SCHEMA_VERSION


def test_whisper_transcript_provider_unloads_after_transcribe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ASR stage releases GPU weights before chunk LM work in safe mode."""
    clip = tmp_path / "v.mp4"
    clip.write_bytes(b"x")
    work = tmp_path / "w"
    work.mkdir()
    monkeypatch.setenv(VIDEO_QA_WHISPER_UNLOAD_MODE_ENV, "safe")
    artifacts = VideoQATranscriptArtifacts("hi", segments=((0.0, 1.0, "hi"),))
    monkeypatch.setattr(
        "core.video_qa_local_run._whisper_transcribe_to_artifacts",
        lambda *_a, **_k: artifacts,
    )
    whisper = MagicMock()
    pipeline_messages: list[str] = []
    prov = VideoQAWhisperTranscriptProvider(
        whisper,
        media_path=clip,
        work_dir=work,
        pipeline_log=pipeline_messages.append,
    )
    manifest = MagicMock()
    manifest.run_id = "run-a"
    with caplog.at_level(logging.INFO):
        prov.prepare_transcript(MagicMock(), manifest)
    whisper.unload.assert_called_once_with(safe=True)
    assert pipeline_messages == [
        "→ Stage: transcript_prepare (local Whisper ASR)",
        "✓ Stage: transcript_prepare complete",
        "→ Whisper unload mode selected (mode=safe, env=ASKVLM_VIDEO_QA_WHISPER_UNLOAD_MODE)",
        "→ Initiating Whisper VRAM unload (mode=safe, safe=True)",
        "✓ Whisper VRAM unload finished (mode=safe, safe=True)",
    ]
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "stage=whisper_unload_mode_selected run_id=run-a mode=safe "
        "env_var=ASKVLM_VIDEO_QA_WHISPER_UNLOAD_MODE default_mode=skip "
        "segment_count=1 transcript_chars=2" in message
        for message in messages
    )
    assert any(
        "stage=whisper_unload_start run_id=run-a mode=safe safe=True "
        "segment_count=1 transcript_chars=2" in message
        for message in messages
    )
    assert any(
        "stage=whisper_unload_complete run_id=run-a mode=safe safe=True "
        "segment_count=1 "
        "transcript_chars=2 elapsed_s=" in message
        for message in messages
    )


def test_whisper_transcript_provider_skips_unload_in_default_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Video QA defaults to skip mode and defers Whisper teardown."""
    clip = tmp_path / "v.mp4"
    clip.write_bytes(b"x")
    work = tmp_path / "w"
    work.mkdir()
    monkeypatch.delenv(VIDEO_QA_WHISPER_UNLOAD_MODE_ENV, raising=False)
    artifacts = VideoQATranscriptArtifacts("hi", segments=((0.0, 1.0, "hi"),))
    monkeypatch.setattr(
        "core.video_qa_local_run._whisper_transcribe_to_artifacts",
        lambda *_a, **_k: artifacts,
    )
    whisper = MagicMock()
    pipeline_messages: list[str] = []
    prov = VideoQAWhisperTranscriptProvider(
        whisper,
        media_path=clip,
        work_dir=work,
        pipeline_log=pipeline_messages.append,
    )
    manifest = MagicMock()
    manifest.run_id = "run-a"
    with caplog.at_level(logging.INFO):
        prov.prepare_transcript(MagicMock(), manifest)
    whisper.unload.assert_not_called()
    assert pipeline_messages == [
        "→ Stage: transcript_prepare (local Whisper ASR)",
        "✓ Stage: transcript_prepare complete",
        "→ Whisper unload mode selected (mode=skip, env=ASKVLM_VIDEO_QA_WHISPER_UNLOAD_MODE)",
        "✓ Whisper VRAM unload skipped/deferred (mode=skip, teardown left for process/session end)",
    ]
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "stage=whisper_unload_mode_selected run_id=run-a mode=skip "
        "env_var=ASKVLM_VIDEO_QA_WHISPER_UNLOAD_MODE default_mode=skip "
        "segment_count=1 transcript_chars=2" in message
        for message in messages
    )
    assert any(
        "stage=whisper_unload_skipped run_id=run-a mode=skip action=deferred "
        "segment_count=1 transcript_chars=2" in message
        for message in messages
    )
    assert not any("stage=whisper_unload_start" in message for message in messages)


@pytest.mark.parametrize(
    ("mode", "safe_flag"),
    [("aggressive", False)],
)
def test_whisper_transcript_provider_applies_nondefault_unload_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    mode: str,
    *,
    safe_flag: bool,
) -> None:
    """Configured non-default unload modes call Whisper with the expected flag."""
    clip = tmp_path / "v.mp4"
    clip.write_bytes(b"x")
    work = tmp_path / "w"
    work.mkdir()
    monkeypatch.setenv(VIDEO_QA_WHISPER_UNLOAD_MODE_ENV, mode)
    artifacts = VideoQATranscriptArtifacts("hi", segments=((0.0, 1.0, "hi"),))
    monkeypatch.setattr(
        "core.video_qa_local_run._whisper_transcribe_to_artifacts",
        lambda *_a, **_k: artifacts,
    )
    whisper = MagicMock()
    pipeline_messages: list[str] = []
    prov = VideoQAWhisperTranscriptProvider(
        whisper,
        media_path=clip,
        work_dir=work,
        pipeline_log=pipeline_messages.append,
    )
    manifest = MagicMock()
    manifest.run_id = "run-a"
    with caplog.at_level(logging.INFO):
        prov.prepare_transcript(MagicMock(), manifest)
    whisper.unload.assert_called_once_with(safe=safe_flag)
    assert pipeline_messages == [
        "→ Stage: transcript_prepare (local Whisper ASR)",
        "✓ Stage: transcript_prepare complete",
        f"→ Whisper unload mode selected (mode={mode}, env=ASKVLM_VIDEO_QA_WHISPER_UNLOAD_MODE)",
        f"→ Initiating Whisper VRAM unload (mode={mode}, safe={safe_flag})",
        f"✓ Whisper VRAM unload finished (mode={mode}, safe={safe_flag})",
    ]
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        f"stage=whisper_unload_start run_id=run-a mode={mode} safe={safe_flag} "
        "segment_count=1 transcript_chars=2" in message
        for message in messages
    )
    assert any(
        f"stage=whisper_unload_complete run_id=run-a mode={mode} safe={safe_flag} "
        "segment_count=1 transcript_chars=2 elapsed_s=" in message
        for message in messages
    )


def test_video_qa_whisper_unload_mode_invalid_value_falls_back_to_skip(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Invalid unload mode values fall back to the skip mitigation."""
    monkeypatch.setenv(VIDEO_QA_WHISPER_UNLOAD_MODE_ENV, "  invalid-mode  ")
    with caplog.at_level(logging.WARNING):
        mode = _get_video_qa_whisper_unload_mode()
    assert mode == "skip"
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "stage=whisper_unload_mode_invalid "
        "env_var=ASKVLM_VIDEO_QA_WHISPER_UNLOAD_MODE raw_value='  invalid-mode  ' "
        "fallback_mode=skip allowed_modes=aggressive,safe,skip" in message
        for message in messages
    )


def test_whisper_transcribe_logs_post_asr_cleanup_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Transcribe diagnostics log the post-ASR and cleanup boundaries once."""
    clip = tmp_path / "v.mp4"
    clip.write_bytes(b"x")
    work = tmp_path / "w"
    work.mkdir()
    audio = work / "prepared.wav"
    audio.write_bytes(b"wav")
    monkeypatch.setattr(
        "core.video_qa_local_run.prepare_audio",
        lambda *_args, **_kwargs: audio,
    )
    monkeypatch.setattr(
        "core.video_qa_local_run.cleanup_intermediate_audio",
        lambda *_args, **_kwargs: None,
    )
    whisper = MagicMock()
    whisper.transcribe.return_value = {
        "text": "hello",
        "segments": [{"start": 0.0, "end": 1.0, "text": " hello "}],
    }
    with caplog.at_level(logging.INFO):
        artifacts = _whisper_transcribe_to_artifacts(
            whisper,
            clip,
            work,
            language=None,
            should_cancel=None,
            progress=None,
            run_id="run-a",
        )
    assert artifacts.transcript_text == "hello"
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "stage=whisper_transcribe_complete run_id=run-a raw_segment_count=1 "
        "elapsed_s=" in message
        for message in messages
    )
    assert any(
        "stage=cleanup_intermediate_audio_start run_id=run-a segment_count=1 "
        "transcript_chars=5" in message
        for message in messages
    )
    assert any(
        "stage=cleanup_intermediate_audio_complete run_id=run-a segment_count=1 "
        "transcript_chars=5 elapsed_s=" in message
        for message in messages
    )


def test_map_video_qa_progress_frac_to_200_splits_pre_vlm_and_vlm() -> None:
    """First ~45% of internal progress maps to 0-100; remainder maps to 100-200."""
    assert map_video_qa_progress_frac_to_200(0.0) == 0
    assert map_video_qa_progress_frac_to_200(0.225) == 50
    assert map_video_qa_progress_frac_to_200(0.45) == 100
    assert map_video_qa_progress_frac_to_200(1.0) == 200
