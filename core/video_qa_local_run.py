"""Local Video QA run glue: preflight guards, ASR transcript, ffmpeg frames, chunk analysis, and final synthesis."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final

from .audio_io import cleanup_intermediate_audio, prepare_audio
from .ffmpeg import (
    extract_frame_to_file,
    extract_frames_for_span,
    get_media_duration_seconds,
)
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
from .video_qa_lm_studio_client import (
    LMStudioClientError,
    request_chat_completion,
)
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
        progress("Preparing", 0.18)

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
    """Extract sampled chunk frames with a single-frame fallback."""

    def __init__(
        self,
        *,
        video_path: Path,
        frames_dir: Path,
        sample_fps: float = 2.0,
    ) -> None:
        self._video_path = video_path
        self._frames_dir = frames_dir
        self._sample_fps = sample_fps

    def materialize_frames(
        self,
        *,
        chunk: VideoQAChunkRecord,
        representative_timestamp: float,
        manifest: VideoQARunManifest,
        transcript: VideoQATranscriptArtifacts,
    ) -> tuple[str, ...]:
        """Write 2 FPS chunk frames under ``frames_dir`` and return their paths."""
        _ = (manifest, transcript)
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        safe = "".join(
            ch if ch.isalnum() or ch in "-._" else "_" for ch in chunk.chunk_id
        )
        output_pattern = self._frames_dir / f"{safe}-%03d.png"
        outputs = extract_frames_for_span(
            self._video_path,
            chunk.t_start,
            chunk.t_end,
            output_pattern,
            fps=self._sample_fps,
        )
        if outputs:
            return tuple(str(path) for path in outputs)
        fallback_output = self._frames_dir / f"{safe}.png"
        extract_frame_to_file(
            self._video_path, representative_timestamp, fallback_output
        )
        return (str(fallback_output.resolve()),)


FINAL_SYNTHESIS_INSTRUCTION: Final[str] = (
    "Synthesize one final user-facing answer from the internal chunk analysis records. "
    "Return strictly JSON that matches the provided schema. Do not expose chunk ids, "
    "hidden analysis steps, or chain-of-thought."
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


@dataclass(frozen=True, slots=True)
class _VideoQAChunkSynthesisInput:
    chunk_id: str
    t_start: float
    t_end: float
    transcript_excerpt: str
    chunk_summary: str
    observations: tuple[str, ...]
    confidence: str
    frame_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ParsedChunkAnalysisArtifact:
    chunk_summary: str
    observations: tuple[str, ...]
    confidence: str


def _transcript_excerpt_for_chunk(
    transcript: VideoQATranscriptArtifacts,
    chunk: VideoQAChunkRecord,
    *,
    max_chars: int = 2000,
) -> str:
    """Build a transcript excerpt for the chunk span, with a whole-text fallback."""
    lines: list[str] = []
    for t0, t1, text in transcript.segments:
        if t1 >= chunk.t_start and t0 <= chunk.t_end:
            stripped = text.strip()
            if stripped:
                lines.append(stripped)
    if lines:
        joined = "\n".join(lines)
        return joined if len(joined) <= max_chars else joined[: max_chars - 3] + "..."
    body = transcript.transcript_text.strip() or transcript.subtitle_text.strip()
    if not body:
        return ""
    return body if len(body) <= max_chars else body[: max_chars - 3] + "..."


def _parse_chunk_analysis_artifact(text: str) -> _ParsedChunkAnalysisArtifact | None:
    """Return a normalized chunk analysis payload or ``None``."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, Mapping):
        return None
    summary = payload.get("chunk_summary")
    observations = payload.get("observations")
    confidence = payload.get("confidence")
    if not isinstance(summary, str):
        return None
    if not isinstance(observations, list) or not all(
        isinstance(item, str) for item in observations
    ):
        return None
    if confidence not in {"low", "medium", "high"}:
        return None
    return _ParsedChunkAnalysisArtifact(
        chunk_summary=summary.strip(),
        observations=tuple(item.strip() for item in observations if item.strip()),
        confidence=str(confidence),
    )


def _collect_chunk_synthesis_inputs(
    *,
    manifest: VideoQARunManifest,
    transcript: VideoQATranscriptArtifacts,
    chunk_results: Sequence[VideoQAChunkExecutionResult],
) -> tuple[tuple[_VideoQAChunkSynthesisInput, ...], bool]:
    """Collect normalized chunk analysis records for the final synthesis pass."""
    manifest_by_id = {chunk.chunk_id: chunk for chunk in manifest.chunks}
    records: list[_VideoQAChunkSynthesisInput] = []
    has_low_confidence = False
    for result in chunk_results:
        if result.status not in ("completed", "skipped_completed"):
            continue
        chunk = manifest_by_id.get(result.chunk_id)
        if chunk is None:
            continue
        parsed = next(
            (
                normalized
                for artifact in chunk.artifacts
                if (normalized := _parse_chunk_analysis_artifact(artifact)) is not None
            ),
            None,
        )
        if parsed is None:
            continue
        confidence = parsed.confidence
        has_low_confidence = has_low_confidence or confidence == "low"
        records.append(
            _VideoQAChunkSynthesisInput(
                chunk_id=chunk.chunk_id,
                t_start=chunk.t_start,
                t_end=chunk.t_end,
                transcript_excerpt=_transcript_excerpt_for_chunk(transcript, chunk),
                chunk_summary=parsed.chunk_summary,
                observations=parsed.observations,
                confidence=confidence,
                frame_refs=tuple(result.frames),
            )
        )
    return tuple(records), has_low_confidence


def _build_final_synthesis_prompt(
    context: VideoQAContextBundle,
    chunk_inputs: Sequence[_VideoQAChunkSynthesisInput],
) -> str:
    """Render the final synthesis prompt from context and chunk analysis records."""
    prompt_parts = [FINAL_SYNTHESIS_INSTRUCTION]
    context_block = context.render_prompt_block()
    if context_block:
        prompt_parts.append(f"Context:\n{context_block}")
    prompt_parts.append(
        "Internal chunk analysis records:\n"
        + json.dumps(
            [
                {
                    "t_start": record.t_start,
                    "t_end": record.t_end,
                    "transcript_excerpt": record.transcript_excerpt,
                    "chunk_summary": record.chunk_summary,
                    "observations": list(record.observations),
                    "confidence": record.confidence,
                    "frame_refs": list(record.frame_refs),
                }
                for record in chunk_inputs
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
    return "\n\n".join(prompt_parts)


def _validate_final_synthesis_payload(
    payload: object,
) -> tuple[str, tuple[VideoQAEvidenceItem, ...], bool, str | None] | None:
    """Validate the final synthesis JSON payload."""
    if not isinstance(payload, Mapping):
        return None
    answer = payload.get("answer")
    evidence_raw = payload.get("evidence")
    is_uncertain = payload.get("is_uncertain")
    uncertainty_note = payload.get("uncertainty_note")
    if not isinstance(answer, str) or not isinstance(is_uncertain, bool):
        return None
    if uncertainty_note is not None and not isinstance(uncertainty_note, str):
        return None
    if not isinstance(evidence_raw, list):
        return None
    evidence = _validate_final_synthesis_evidence_items(evidence_raw)
    if evidence is None:
        return None
    return answer.strip(), tuple(evidence), is_uncertain, uncertainty_note


def _validate_final_synthesis_evidence_items(
    evidence_raw: list[object],
) -> list[VideoQAEvidenceItem] | None:
    """Validate the synthesized evidence list."""
    evidence: list[VideoQAEvidenceItem] = []
    for index, item in enumerate(evidence_raw):
        normalized = _validate_final_synthesis_evidence_item(item, index)
        if normalized is None:
            return None
        evidence.append(normalized)
    return evidence


def _validate_final_synthesis_evidence_item(
    item: object,
    index: int,
) -> VideoQAEvidenceItem | None:
    """Validate one synthesized evidence item."""
    if not isinstance(item, Mapping):
        return None
    transcript_quote = item.get("transcript_quote")
    t_start = item.get("t_start")
    t_end = item.get("t_end")
    frame_refs = item.get("frame_refs")
    if not isinstance(transcript_quote, str):
        return None
    if isinstance(t_start, bool) or not isinstance(t_start, (int, float)):
        return None
    if isinstance(t_end, bool) or not isinstance(t_end, (int, float)):
        return None
    if not isinstance(frame_refs, list) or not all(
        isinstance(frame_ref, str) for frame_ref in frame_refs
    ):
        return None
    return VideoQAEvidenceItem(
        transcript_quote=transcript_quote.strip() or f"Evidence item {index + 1}",
        t_start=float(t_start),
        t_end=float(t_end),
        frame_refs=tuple(frame_refs),
    )


class VideoQALMStudioAnswerSynthesizer:
    """Build the final answer bundle with a synthesis LM Studio pass."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        timeout: float = 120.0,
        request_chat_fn: Callable[..., object] | None = None,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._temperature = temperature
        self._timeout = timeout
        self._request_chat_fn = request_chat_fn or request_chat_completion

    def aggregate(
        self,
        *,
        context: VideoQAContextBundle,
        manifest: VideoQARunManifest,
        transcript: VideoQATranscriptArtifacts,
        chunk_results: Sequence[VideoQAChunkExecutionResult],
    ) -> VideoQAAnswerBundle:
        """Synthesize one final answer from chunk analysis artifacts."""
        chunk_inputs, has_low_confidence = _collect_chunk_synthesis_inputs(
            manifest=manifest,
            transcript=transcript,
            chunk_results=chunk_results,
        )
        if not chunk_inputs:
            return VideoQAAnswerBundle(
                schema_version=ANSWER_BUNDLE_SCHEMA_VERSION,
                run_id=f"{manifest.run_id}-answer",
                created_at=manifest.created_at,
                question=context.question,
                answer="No completed chunk analysis artifacts were available to synthesize a final answer.",
                evidence=(),
                is_uncertain=True,
                manifest_run_id=manifest.run_id,
                uncertainty_note="The synthesis pass had no completed chunk analysis inputs.",
            )
        prompt = _build_final_synthesis_prompt(context, chunk_inputs)
        try:
            response = self._request_chat_fn(
                self._base_url,
                prompt,
                json_schema=FINAL_SYNTHESIS_JSON_SCHEMA,
                model=self._model,
                temperature=self._temperature,
                timeout=self._timeout,
            )
        except LMStudioClientError as exc:
            msg = f"Final Video QA synthesis failed: {exc}"
            raise RuntimeError(msg) from exc
        normalized = _validate_final_synthesis_payload(
            getattr(response, "parsed_json", None)
        )
        if normalized is None:
            msg = "Final Video QA synthesis returned invalid structured output."
            raise RuntimeError(msg)
        answer, evidence_items, is_uncertain, uncertainty_note = normalized
        final_uncertain = is_uncertain or has_low_confidence
        final_uncertainty_note = uncertainty_note
        if final_uncertain and not final_uncertainty_note:
            final_uncertainty_note = (
                "Some internal chunk analyses reported low confidence."
                if has_low_confidence
                else "The synthesized answer is uncertain."
            )
        return VideoQAAnswerBundle(
            schema_version=ANSWER_BUNDLE_SCHEMA_VERSION,
            run_id=f"{manifest.run_id}-answer",
            created_at=manifest.created_at,
            question=context.question,
            answer=answer,
            evidence=tuple(evidence_items),
            is_uncertain=final_uncertain,
            manifest_run_id=manifest.run_id,
            uncertainty_note=final_uncertainty_note,
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
            sample_fps=default_video_qa_budget_policy().frame_sample_fps,
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
        aggregator = VideoQALMStudioAnswerSynthesizer(
            base_url=lm_base_url,
            model=lm_model_id,
        )
    return VideoQAExecutorDeps(
        transcript=transcript,
        frame_materializer=frame_mat,
        chunk_inferencer=inferencer,
        answer_aggregator=aggregator,
        source_resolver=None,
    )


class _ProgressFrameMaterializer:
    """Wrap frame extraction to report coarse sampling progress."""

    def __init__(
        self,
        inner: VideoQAFrameMaterializer,
        *,
        progress: Callable[[str, float], None],
        total_chunks: int,
        chunk_state: dict[str, int],
    ) -> None:
        self._inner = inner
        self._progress = progress
        self._total_chunks = total_chunks
        self._chunk_state = chunk_state

    def materialize_frames(
        self,
        *,
        chunk: VideoQAChunkRecord,
        representative_timestamp: float,
        manifest: VideoQARunManifest,
        transcript: VideoQATranscriptArtifacts,
    ) -> tuple[str, ...]:
        self._chunk_state["i"] += 1
        chunk_index = self._chunk_state["i"]
        per_chunk_span = 0.48 / self._total_chunks
        base = 0.45 + (chunk_index - 1) * per_chunk_span
        self._progress("Sampling", min(0.93, base))
        return self._inner.materialize_frames(
            chunk=chunk,
            representative_timestamp=representative_timestamp,
            manifest=manifest,
            transcript=transcript,
        )


class _ProgressInferencer:
    """Wrap chunk inference to report coarse processing progress."""

    def __init__(
        self,
        inner: VideoQAChunkInferencer,
        *,
        progress: Callable[[str, float], None],
        total_chunks: int,
        chunk_state: dict[str, int],
    ) -> None:
        self._inner = inner
        self._progress = progress
        self._total_chunks = total_chunks
        self._chunk_state = chunk_state

    def infer_chunk(
        self,
        *,
        chunk: VideoQAChunkRecord,
        frames: tuple[str, ...],
        transcript: VideoQATranscriptArtifacts,
        manifest: VideoQARunManifest,
    ) -> VideoQAChunkInferenceOutcome:
        chunk_index = max(1, self._chunk_state["i"])
        per_chunk_span = 0.48 / self._total_chunks
        base = 0.45 + (chunk_index - 1) * per_chunk_span
        self._progress("Processing", min(0.93, base + per_chunk_span * 0.5))
        return self._inner.infer_chunk(
            chunk=chunk,
            frames=frames,
            transcript=transcript,
            manifest=manifest,
        )


class _ProgressAggregator:
    """Wrap final answer aggregation to report synthesis progress."""

    def __init__(
        self,
        inner: VideoQAAnswerAggregator,
        *,
        progress: Callable[[str, float], None],
    ) -> None:
        self._inner = inner
        self._progress = progress

    def aggregate(
        self,
        *,
        context: VideoQAContextBundle,
        manifest: VideoQARunManifest,
        transcript: VideoQATranscriptArtifacts,
        chunk_results: Sequence[VideoQAChunkExecutionResult],
    ) -> VideoQAAnswerBundle:
        self._progress("Synthesizing", 0.95)
        return self._inner.aggregate(
            context=context,
            manifest=manifest,
            transcript=transcript,
            chunk_results=chunk_results,
        )


def _wrap_local_executor_deps_with_progress(
    deps: VideoQAExecutorDeps,
    *,
    progress: Callable[[str, float], None],
    total_chunks: int,
) -> VideoQAExecutorDeps:
    """Wrap executor dependencies with generic local-run progress reporting."""
    chunk_state = {"i": 0}
    return replace(
        deps,
        frame_materializer=_ProgressFrameMaterializer(
            deps.frame_materializer,
            progress=progress,
            total_chunks=total_chunks,
            chunk_state=chunk_state,
        ),
        chunk_inferencer=_ProgressInferencer(
            deps.chunk_inferencer,
            progress=progress,
            total_chunks=total_chunks,
            chunk_state=chunk_state,
        ),
        answer_aggregator=_ProgressAggregator(
            deps.answer_aggregator,
            progress=progress,
        ),
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
        progress("Preparing", 0.05)

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

    if progress is not None:
        deps = _wrap_local_executor_deps_with_progress(
            deps,
            progress=progress,
            total_chunks=max(1, len(chunk_plan)),
        )

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
        progress("Completed", 1.0)

    return outcome
