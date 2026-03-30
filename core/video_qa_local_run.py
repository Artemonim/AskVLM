"""Local Video QA run glue: preflight guards, ASR transcript, ffmpeg frames, LM Studio chunks, deterministic aggregate."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final

from .audio_io import cleanup_intermediate_audio, prepare_audio
from .ffmpeg import extract_frame_to_file, get_media_duration_seconds
from .pipelines import CancelledError
from .video_qa_answer_bundle import (
    ANSWER_BUNDLE_SCHEMA_VERSION,
    VideoQAAnswerBundle,
    VideoQAEvidenceItem,
    answer_bundle_path_for_manifest,
    save_answer_bundle_to_json,
)
from .video_qa_executor import (
    VideoQAExecutorDeps,
    VideoQATranscriptArtifacts,
    run_video_qa_executor,
)
from .video_qa_lm_studio_chunk_inferencer import VideoQALMStudioChunkInferencer
from .video_qa_orchestration import (
    build_video_qa_preflight_report,
    build_video_qa_preflight_summary,
    default_representative_frame_policy,
)
from .video_qa_preparation import build_video_qa_preparation_manifest
from .video_qa_runtime import default_video_qa_budget_policy

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from .video_qa_context import VideoQAContextBundle
    from .video_qa_executor import (
        VideoQAAnswerAggregator,
        VideoQAChunkExecutionResult,
        VideoQAChunkInferenceOutcome,
        VideoQAChunkInferencer,
        VideoQAExecutorRunOutcome,
        VideoQAFrameMaterializer,
        VideoQATranscriptProvider,
    )
    from .video_qa_manifest import VideoQAChunkRecord, VideoQARunManifest
    from .video_qa_orchestration import (
        VideoQAPlannedChunk,
        VideoQAPreflightReport,
    )
    from .whisperx_wrapper import WhisperXWrapper

# * OpenAI-compatible LM Studio HTTP API base (chat completions live under /chat/completions).
DEFAULT_LM_STUDIO_OPENAI_BASE_URL: Final[str] = "http://127.0.0.1:1234/v1"


class VideoQAPreflightBlockedError(ValueError):
    """Raised when a Video QA run is not allowed to start (preflight / validation)."""


def preflight_local_video_qa(
    context: VideoQAContextBundle,
    *,
    duration_seconds: float,
    context_window_tokens: int,
) -> tuple[VideoQAPreflightReport, tuple[VideoQAPlannedChunk, ...]]:
    """Build preflight report and chunk plan without executing ASR or LM calls."""
    budget_policy = replace(
        default_video_qa_budget_policy(),
        context_window_tokens=int(context_window_tokens),
    )
    preflight = build_video_qa_preflight_summary(
        context,
        duration_seconds=float(duration_seconds),
        budget_policy=budget_policy,
    )
    report = build_video_qa_preflight_report(context, preflight)
    return report, preflight.chunk_plan


def ensure_local_video_qa_run_allowed(
    report: VideoQAPreflightReport,
    chunk_plan: Sequence[VideoQAPlannedChunk],
    *,
    require_question: bool = True,
) -> None:
    """Raise :class:`VideoQAPreflightBlockedError` when the run must not start."""
    if require_question and not str(report.question or "").strip():
        msg = "Video QA needs a non-empty question."
        raise VideoQAPreflightBlockedError(msg)
    if report.source_summary is None or not str(report.source_summary).strip():
        msg = "Video QA needs a local media source."
        raise VideoQAPreflightBlockedError(msg)
    if not report.budget_fits:
        msg = (
            "Preflight budget does not fit the configured context window. "
            f"{report.budget_status_line}"
        )
        raise VideoQAPreflightBlockedError(msg)
    if not chunk_plan:
        msg = (
            "Chunk plan is empty (duration missing or zero). "
            "Refresh preflight after selecting a readable media file."
        )
        raise VideoQAPreflightBlockedError(msg)


def save_video_qa_manifest_json(path: Path, manifest: VideoQARunManifest) -> None:
    """Write manifest JSON to ``path`` (UTF-8, indented)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


def _coerce_float(value: object, default: float = 0.0) -> float:
    """Convert segment timing values to ``float`` for progress and alignment."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


class VideoQAWhisperTranscriptProvider:
    """ASR transcript via :class:`WhisperXWrapper` and ``prepare_audio`` (no diarization)."""

    def __init__(
        self,
        whisper: WhisperXWrapper,
        *,
        media_path: Path,
        work_dir: Path,
        language: str | None = None,
        should_cancel: Callable[[], bool] | None = None,
        progress: Callable[[str, float], None] | None = None,
    ) -> None:
        self._whisper = whisper
        self._media_path = media_path
        self._work_dir = work_dir
        self._language = language
        self._should_cancel = should_cancel
        self._progress = progress

    def prepare_transcript(
        self,
        _context: VideoQAContextBundle,
        _manifest: VideoQARunManifest,
    ) -> VideoQATranscriptArtifacts:
        """Transcribe prepared audio and return segment-aligned transcript text."""
        return _whisper_transcribe_to_artifacts(
            self._whisper,
            self._media_path,
            self._work_dir,
            language=self._language,
            should_cancel=self._should_cancel,
            progress=self._progress,
        )


def _whisper_transcribe_to_artifacts(
    whisper: WhisperXWrapper,
    media_path: Path,
    work_dir: Path,
    *,
    language: str | None,
    should_cancel: Callable[[], bool] | None,
    progress: Callable[[str, float], None] | None,
) -> VideoQATranscriptArtifacts:
    """Run faster-whisper transcription and normalize segment tuples."""
    if progress:
        progress("Preparing audio for transcription", 0.18)

    def _should_cancel_audio() -> bool:
        return bool(should_cancel and should_cancel())

    audio_path = prepare_audio(
        media_path,
        work_dir,
        should_cancel=_should_cancel_audio,
    )

    def _on_segment(seg: dict[str, object]) -> None:
        if should_cancel and should_cancel():
            msg = "Canceled"
            raise CancelledError(msg)
        if progress:
            try:
                end = _coerce_float(seg.get("end", 0.0))
                dur = max(1e-6, get_media_duration_seconds(media_path))
                inner = max(0.0, min(1.0, end / dur))
            except (TypeError, ValueError):
                inner = 0.5
            progress("Transcribing", 0.18 + inner * 0.27)

    if should_cancel and should_cancel():
        msg = "Canceled"
        raise CancelledError(msg)

    tx = whisper.transcribe(
        audio_path,
        language=language,
        on_segment=_on_segment,
        progress=None,
    )
    segments_raw = tx.get("segments", []) or []
    segments: list[tuple[float, float, str]] = []
    for raw in segments_raw:
        if not isinstance(raw, dict):
            continue
        t0 = _coerce_float(raw.get("start", 0.0))
        t1 = _coerce_float(raw.get("end", 0.0))
        text = str(raw.get("text", "")).strip()
        segments.append((t0, t1, text))
    body = str(tx.get("text", "")).strip()
    cleanup_intermediate_audio(media_path, work_dir)
    return VideoQATranscriptArtifacts(
        transcript_text=body,
        subtitle_text="",
        segments=tuple(segments),
    )


class VideoQAFFmpegFrameMaterializer:
    """Extract one representative PNG per chunk using :func:`extract_frame_to_file`."""

    def __init__(self, *, video_path: Path, frames_dir: Path) -> None:
        self._video_path = video_path
        self._frames_dir = frames_dir

    def materialize_frames(
        self,
        *,
        chunk: VideoQAChunkRecord,
        representative_timestamp: float,
        manifest: VideoQARunManifest,
        transcript: VideoQATranscriptArtifacts,
    ) -> tuple[str, ...]:
        """Write ``<chunk_id>.png`` under ``frames_dir`` and return its path."""
        _ = (manifest, transcript)
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        safe = "".join(
            ch if ch.isalnum() or ch in "-._" else "_" for ch in chunk.chunk_id
        )
        out = self._frames_dir / f"{safe}.png"
        extract_frame_to_file(self._video_path, representative_timestamp, out)
        return (str(out.resolve()),)


def _first_artifact_answer_and_evidence(
    raw_cr: VideoQAChunkExecutionResult,
    rec: VideoQAChunkRecord,
) -> tuple[str | None, VideoQAEvidenceItem | None, bool]:
    """Parse the first JSON chunk artifact into answer text, evidence, and low-confidence flag."""
    for art in rec.artifacts:
        try:
            payload = json.loads(art)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        summary = str(payload.get("chunk_summary", "")).strip()
        conf = str(payload.get("confidence", "medium"))
        uncertain = conf == "low"
        obs = payload.get("observations")
        obs_lines = ""
        if isinstance(obs, list) and obs:
            obs_lines = "\n".join(
                f"- {x}" for x in obs if isinstance(x, str) and str(x).strip()
            )
        block = summary
        if obs_lines:
            block = f"{summary}\n{obs_lines}" if summary else obs_lines
        if not block.strip():
            continue
        answer_line = f"=== {raw_cr.chunk_id} ===\n{block.strip()}"
        quote = summary if summary else block[:2000]
        evidence = VideoQAEvidenceItem(
            transcript_quote=quote[:8000],
            t_start=rec.t_start,
            t_end=rec.t_end,
            frame_refs=tuple(raw_cr.frames),
        )
        return answer_line, evidence, uncertain
    return None, None, False


class VideoQADeterministicAnswerAggregator:
    """Build :class:`VideoQAAnswerBundle` from chunk JSON artifacts (no extra LM call)."""

    def aggregate(
        self,
        *,
        context: VideoQAContextBundle,
        manifest: VideoQARunManifest,
        transcript: VideoQATranscriptArtifacts,
        chunk_results: Sequence[VideoQAChunkExecutionResult],
    ) -> VideoQAAnswerBundle:
        """Concatenate chunk summaries into the answer and mirror them as evidence rows."""
        _ = transcript

        lines: list[str] = []
        evidence_items: list[VideoQAEvidenceItem] = []
        uncertain = False
        for raw_cr in chunk_results:
            if raw_cr.status not in ("completed", "skipped_completed"):
                continue
            rec = next(
                (c for c in manifest.chunks if c.chunk_id == raw_cr.chunk_id),
                None,
            )
            if rec is None:
                continue
            ans, ev, low = _first_artifact_answer_and_evidence(raw_cr, rec)
            if ans is not None:
                lines.append(ans)
            if ev is not None:
                evidence_items.append(ev)
            uncertain = uncertain or low

        answer = (
            "\n\n".join(lines)
            if lines
            else "No chunk-level analysis artifacts were produced."
        )
        return VideoQAAnswerBundle(
            schema_version=ANSWER_BUNDLE_SCHEMA_VERSION,
            run_id=f"{manifest.run_id}-answer",
            created_at=manifest.created_at,
            question=context.question,
            answer=answer,
            evidence=tuple(evidence_items),
            is_uncertain=uncertain,
            manifest_run_id=manifest.run_id,
            uncertainty_note=(
                "Some chunks reported low confidence." if uncertain else None
            ),
        )


@dataclass(frozen=True, slots=True)
class VideoQALocalRunParams:
    """Inputs for :func:`run_local_video_qa`."""

    context: VideoQAContextBundle
    output_dir: Path
    context_window_tokens: int
    lm_base_url: str
    lm_model_id: str


def build_video_qa_local_executor_deps(  # noqa: PLR0913
    *,
    context: VideoQAContextBundle,
    whisper: WhisperXWrapper,
    staging_dir: Path,
    lm_base_url: str,
    lm_model_id: str,
    should_cancel: Callable[[], bool] | None,
    progress: Callable[[str, float], None] | None,
    chunk_inferencer_override: VideoQAChunkInferencer | None = None,
    transcript_override: VideoQATranscriptProvider | None = None,
    frame_override: VideoQAFrameMaterializer | None = None,
    aggregator_override: VideoQAAnswerAggregator | None = None,
) -> VideoQAExecutorDeps:
    """Construct executor dependencies for a local file run (injectable for tests)."""
    if context.source is None:
        msg = "Video QA context has no local source."
        raise ValueError(msg)
    media_path = Path(context.source.path)
    frames_dir = staging_dir / "frames"
    transcript: VideoQATranscriptProvider
    if transcript_override is not None:
        transcript = transcript_override
    else:
        transcript = VideoQAWhisperTranscriptProvider(
            whisper,
            media_path=media_path,
            work_dir=staging_dir,
            should_cancel=should_cancel,
            progress=progress,
        )
    frame_mat: VideoQAFrameMaterializer
    if frame_override is not None:
        frame_mat = frame_override
    else:
        frame_mat = VideoQAFFmpegFrameMaterializer(
            video_path=media_path,
            frames_dir=frames_dir,
        )
    inferencer: VideoQAChunkInferencer
    if chunk_inferencer_override is not None:
        inferencer = chunk_inferencer_override
    else:
        inferencer = VideoQALMStudioChunkInferencer(
            context,
            base_url=lm_base_url,
            model=lm_model_id,
        )
    aggregator: VideoQAAnswerAggregator
    if aggregator_override is not None:
        aggregator = aggregator_override
    else:
        aggregator = VideoQADeterministicAnswerAggregator()
    return VideoQAExecutorDeps(
        transcript=transcript,
        frame_materializer=frame_mat,
        chunk_inferencer=inferencer,
        answer_aggregator=aggregator,
        source_resolver=None,
    )


def run_local_video_qa(
    *,
    params: VideoQALocalRunParams,
    whisper: WhisperXWrapper,
    should_cancel: Callable[[], bool] | None = None,
    progress: Callable[[str, float], None] | None = None,
    chunk_inferencer_override: VideoQAChunkInferencer | None = None,
    transcript_override: VideoQATranscriptProvider | None = None,
    frame_override: VideoQAFrameMaterializer | None = None,
    aggregator_override: VideoQAAnswerAggregator | None = None,
) -> VideoQAExecutorRunOutcome:
    """Run the full Video QA pipeline for a local file and optional attachments.

    Re-runs preflight and blocks via :func:`ensure_local_video_qa_run_allowed`.
    Writes manifest and answer JSON under ``output_dir`` when the run completes.
    """
    ctx = params.context
    if ctx.source is None:
        msg = "Video QA needs a resolved local file source."
        raise VideoQAPreflightBlockedError(msg)

    duration_s = float(get_media_duration_seconds(ctx.source.path))
    report, chunk_plan = preflight_local_video_qa(
        ctx,
        duration_seconds=duration_s,
        context_window_tokens=params.context_window_tokens,
    )
    ensure_local_video_qa_run_allowed(report, chunk_plan)

    manifest = build_video_qa_preparation_manifest(ctx)
    staging_dir = params.output_dir / "_video_qa_work" / manifest.run_id
    staging_dir.mkdir(parents=True, exist_ok=True)

    if progress:
        progress("Starting Video QA executor", 0.05)

    deps = build_video_qa_local_executor_deps(
        context=ctx,
        whisper=whisper,
        staging_dir=staging_dir,
        lm_base_url=params.lm_base_url,
        lm_model_id=params.lm_model_id,
        should_cancel=should_cancel,
        progress=progress,
        chunk_inferencer_override=chunk_inferencer_override,
        transcript_override=transcript_override,
        frame_override=frame_override,
        aggregator_override=aggregator_override,
    )

    total_chunks = max(1, len(chunk_plan))
    chunk_index = {"i": 0}

    class _ProgressInferencer:
        """Wraps chunk inferencer to report coarse progress per chunk."""

        def __init__(self, inner: VideoQAChunkInferencer) -> None:
            self._inner = inner

        def infer_chunk(
            self,
            *,
            chunk: VideoQAChunkRecord,
            frames: tuple[str, ...],
            transcript: VideoQATranscriptArtifacts,
            manifest: VideoQARunManifest,
        ) -> VideoQAChunkInferenceOutcome:
            chunk_index["i"] += 1
            i = chunk_index["i"]
            if progress:
                base = 0.45 + (i - 1) / total_chunks * 0.48
                progress(
                    f"Chunk inference ({i}/{total_chunks}): {chunk.chunk_id}",
                    min(0.93, base),
                )
            return self._inner.infer_chunk(
                chunk=chunk,
                frames=frames,
                transcript=transcript,
                manifest=manifest,
            )

    inner_inf = deps.chunk_inferencer
    wrapped_inf: VideoQAChunkInferencer = (
        _ProgressInferencer(inner_inf) if progress is not None else inner_inf
    )
    deps = replace(deps, chunk_inferencer=wrapped_inf)

    outcome = run_video_qa_executor(
        context=ctx,
        manifest=manifest,
        planned_chunks=chunk_plan,
        deps=deps,
        representative_frame_policy=default_representative_frame_policy(),
    )

    final_manifest = outcome.manifest
    manifest_path = params.output_dir / f"{final_manifest.run_id}.manifest.json"
    save_video_qa_manifest_json(manifest_path, final_manifest)
    answer_path = answer_bundle_path_for_manifest(manifest_path)
    save_answer_bundle_to_json(answer_path, outcome.answer_bundle)

    if progress:
        progress("Video QA run complete", 1.0)

    return outcome
