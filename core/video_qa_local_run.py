"""Local Video QA run glue: preflight guards, ASR transcript, ffmpeg frames, chunk analysis, and final synthesis."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final

from .audio_io import cleanup_intermediate_audio, prepare_audio
from .ffmpeg import (
    extract_frame_to_file,
    extract_frames_for_span,
    get_media_duration_seconds,
)
from .lm_studio_rest import (
    LMStudioRestError,
    lm_studio_load_model,
    lm_studio_unload_model,
    openai_chat_base_to_local_rest_root,
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
# * OpenAI-compatible OpenRouter base (same paths as local OpenAI-compatible servers).
DEFAULT_OPENROUTER_OPENAI_BASE_URL: Final[str] = "https://openrouter.ai/api/v1"
# * Environment variable read by the GUI when Video QA scope is Cloud (Bearer auth).
OPENROUTER_API_KEY_ENV: Final[str] = "OPENROUTER_API_KEY"

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VideoQALMHttpTarget:
    """OpenAI-compatible chat base (…/v1), model id, and optional Bearer token."""

    base_url: str
    model_id: str
    authorization_bearer: str | None = None


# * Fraction of the 0..1 progress curve treated as pre-VLM (maps to 0-100 on a 0-200 bar).
_VIDEO_QA_PRE_VLM_PROGRESS_FRAC: Final[float] = 0.45
_VIDEO_QA_PRE_VLM_FRAC_EPS: Final[float] = 1e-9
_SECONDS_PER_MINUTE: Final[int] = 60


def _format_vlm_eta_seconds(seconds: float | None) -> str:
    """Format a rough ETA string for chunk-level VLM work."""
    if seconds is None or seconds < 0.0 or math.isnan(float(seconds)):
        return "…"
    s = round(float(seconds))
    spm = _SECONDS_PER_MINUTE
    if s < spm:
        return f"~{s}s"
    minutes, sec = divmod(s, spm)
    if minutes < spm:
        return f"~{minutes}m {sec}s"
    hours, m2 = divmod(minutes, spm)
    return f"~{hours}h {m2}m"


def map_video_qa_progress_frac_to_200(frac: float) -> int:
    """Map internal 0..1 progress to a 0..200 scale (first half = pre-VLM, second = VLM)."""
    x = max(0.0, min(1.0, float(frac)))
    pre = float(_VIDEO_QA_PRE_VLM_PROGRESS_FRAC)
    if x <= pre:
        return int(x / pre * 100) if pre > _VIDEO_QA_PRE_VLM_FRAC_EPS else 0
    tail = (x - pre) / (1.0 - pre)
    return 100 + int(max(0.0, min(1.0, tail)) * 100)


class VideoQAPreflightBlockedError(ValueError):
    """Raised when a Video QA run is not allowed to start (preflight / validation)."""


def _default_frame_sample_fps() -> float:
    """Return the default frame sampling rate from the global Video QA budget policy."""
    return float(default_video_qa_budget_policy().frame_sample_fps)


def preflight_local_video_qa(
    context: VideoQAContextBundle,
    *,
    duration_seconds: float,
    context_window_tokens: int,
    frame_sample_fps: float | None = None,
) -> tuple[VideoQAPreflightReport, tuple[VideoQAPlannedChunk, ...]]:
    """Build preflight report and chunk plan without executing ASR or LM calls."""
    base_policy = default_video_qa_budget_policy()
    fps = (
        float(frame_sample_fps)
        if frame_sample_fps is not None
        else float(base_policy.frame_sample_fps)
    )
    budget_policy = replace(
        base_policy,
        context_window_tokens=int(context_window_tokens),
        frame_sample_fps=fps,
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
        pipeline_log: Callable[[str], None] | None = None,
    ) -> None:
        self._whisper = whisper
        self._media_path = media_path
        self._work_dir = work_dir
        self._language = language
        self._should_cancel = should_cancel
        self._progress = progress
        self._pipeline_log = pipeline_log

    def prepare_transcript(
        self,
        _context: VideoQAContextBundle,
        _manifest: VideoQARunManifest,
    ) -> VideoQATranscriptArtifacts:
        """Transcribe prepared audio and return segment-aligned transcript text."""
        if self._pipeline_log:
            self._pipeline_log("→ Stage: transcript_prepare (local Whisper ASR)")
        artifacts = _whisper_transcribe_to_artifacts(
            self._whisper,
            self._media_path,
            self._work_dir,
            language=self._language,
            should_cancel=self._should_cancel,
            progress=self._progress,
            pipeline_log=self._pipeline_log,
        )
        if self._pipeline_log:
            self._pipeline_log("✓ Stage: transcript_prepare complete")
        # * Drop faster-whisper weights from VRAM before chunk VLM / LM HTTP work.
        with contextlib.suppress(Exception):
            self._whisper.unload(safe=False)
        return artifacts


def _whisper_transcribe_to_artifacts(  # noqa: C901
    whisper: WhisperXWrapper,
    media_path: Path,
    work_dir: Path,
    *,
    language: str | None,
    should_cancel: Callable[[], bool] | None,
    progress: Callable[[str, float], None] | None,
    pipeline_log: Callable[[str], None] | None = None,
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

    if pipeline_log:
        pipeline_log("→ Transcribing audio segments…")

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
    if pipeline_log:
        pipeline_log("✓ Transcription finished")
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
        """Write sampled chunk frames under ``frames_dir`` and return their paths."""
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
        timeout: float | None = None,
        request_chat_fn: Callable[..., object] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        authorization_bearer: str | None = None,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._temperature = temperature
        self._timeout = timeout
        self._request_chat_fn = request_chat_fn or request_chat_completion
        self._should_cancel = should_cancel
        self._authorization_bearer = authorization_bearer

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
                should_cancel=self._should_cancel,
                authorization_bearer=self._authorization_bearer,
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
    chunk_lm: VideoQALMHttpTarget
    final_lm: VideoQALMHttpTarget
    frame_sample_fps: float = field(default_factory=_default_frame_sample_fps)


def _build_lm_studio_dual_local_swap_hook(
    *,
    chunk_lm: VideoQALMHttpTarget,
    final_lm: VideoQALMHttpTarget,
    context_window_tokens: int,
    pipeline_log: Callable[[str], None] | None,
    should_cancel: Callable[[], bool] | None,
) -> Callable[[], None] | None:
    """Return a hook that unloads the chunk LM Studio model and loads the final one.

    Runs only when both targets use the same local LM Studio REST origin and use
    different ``model_id`` strings. ``instance_id`` for unload matches LM Studio
    docs (often the same string as the loaded model id).
    """
    rest_c = openai_chat_base_to_local_rest_root(chunk_lm.base_url)
    rest_f = openai_chat_base_to_local_rest_root(final_lm.base_url)
    if rest_c is None or rest_f is None or rest_c != rest_f:
        return None
    if chunk_lm.model_id == final_lm.model_id:
        return None

    def hook() -> None:
        if should_cancel and should_cancel():
            msg = "Canceled"
            raise CancelledError(msg)
        bearer = chunk_lm.authorization_bearer or final_lm.authorization_bearer
        if pipeline_log:
            pipeline_log(
                "LM Studio REST: unload chunk model "
                f"{chunk_lm.model_id!r}, load final model {final_lm.model_id!r} "
                f"(context_length={int(context_window_tokens)})."
            )
        try:
            lm_studio_unload_model(rest_c, chunk_lm.model_id, bearer=bearer)
        except LMStudioRestError as exc:
            logger.warning("LM Studio unload before final synthesis failed: %s", exc)
            if pipeline_log:
                pipeline_log(
                    f"LM Studio unload failed (safe if the model was not loaded): {exc}"
                )
        if should_cancel and should_cancel():
            msg = "Canceled"
            raise CancelledError(msg)
        lm_studio_load_model(
            rest_c,
            final_lm.model_id,
            context_length=int(context_window_tokens),
            bearer=bearer,
        )
        if pipeline_log:
            pipeline_log(f"LM Studio REST: loaded final model {final_lm.model_id!r}.")

    return hook


def build_video_qa_local_executor_deps(  # noqa: PLR0913
    *,
    context: VideoQAContextBundle,
    whisper: WhisperXWrapper,
    staging_dir: Path,
    chunk_lm: VideoQALMHttpTarget,
    final_lm: VideoQALMHttpTarget,
    should_cancel: Callable[[], bool] | None,
    progress: Callable[[str, float], None] | None,
    pipeline_log: Callable[[str], None] | None = None,
    frame_sample_fps: float | None = None,
    chunk_inferencer_override: VideoQAChunkInferencer | None = None,
    transcript_override: VideoQATranscriptProvider | None = None,
    frame_override: VideoQAFrameMaterializer | None = None,
    aggregator_override: VideoQAAnswerAggregator | None = None,
    before_answer_aggregate: Callable[[], None] | None = None,
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
            pipeline_log=pipeline_log,
        )
    frame_mat: VideoQAFrameMaterializer
    if frame_override is not None:
        frame_mat = frame_override
    else:
        fps_val = (
            float(frame_sample_fps)
            if frame_sample_fps is not None
            else float(default_video_qa_budget_policy().frame_sample_fps)
        )
        frame_mat = VideoQAFFmpegFrameMaterializer(
            video_path=media_path,
            frames_dir=frames_dir,
            sample_fps=fps_val,
        )
    inferencer: VideoQAChunkInferencer
    if chunk_inferencer_override is not None:
        inferencer = chunk_inferencer_override
    else:
        inferencer = VideoQALMStudioChunkInferencer(
            context,
            base_url=chunk_lm.base_url,
            model=chunk_lm.model_id,
            should_cancel=should_cancel,
            authorization_bearer=chunk_lm.authorization_bearer,
        )
    aggregator: VideoQAAnswerAggregator
    if aggregator_override is not None:
        aggregator = aggregator_override
    else:
        aggregator = VideoQALMStudioAnswerSynthesizer(
            base_url=final_lm.base_url,
            model=final_lm.model_id,
            should_cancel=should_cancel,
            authorization_bearer=final_lm.authorization_bearer,
        )
    return VideoQAExecutorDeps(
        transcript=transcript,
        frame_materializer=frame_mat,
        chunk_inferencer=inferencer,
        answer_aggregator=aggregator,
        source_resolver=None,
        before_answer_aggregate=before_answer_aggregate,
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
        pipeline_log: Callable[[str], None] | None = None,
    ) -> None:
        self._inner = inner
        self._progress = progress
        self._total_chunks = total_chunks
        self._chunk_state = chunk_state
        self._pipeline_log = pipeline_log
        self._vlm_section_logged = False

    def materialize_frames(
        self,
        *,
        chunk: VideoQAChunkRecord,
        representative_timestamp: float,
        manifest: VideoQARunManifest,
        transcript: VideoQATranscriptArtifacts,
    ) -> tuple[str, ...]:
        if self._pipeline_log and not self._vlm_section_logged:
            self._pipeline_log(
                "→ VLM phase: per-chunk frame sampling and LM Studio requests "
                "(server console may show prompt %; HTTP client does not)."
            )
            self._vlm_section_logged = True
        self._chunk_state["i"] += 1
        chunk_index = self._chunk_state["i"]
        per_chunk_span = 0.48 / self._total_chunks
        base = _VIDEO_QA_PRE_VLM_PROGRESS_FRAC + (chunk_index - 1) * per_chunk_span
        self._progress("Sampling", min(0.93, base))
        if self._pipeline_log:
            self._pipeline_log(
                f"→ frame_select: {chunk.chunk_id} ({chunk_index}/{self._total_chunks})"
            )
        frames = self._inner.materialize_frames(
            chunk=chunk,
            representative_timestamp=representative_timestamp,
            manifest=manifest,
            transcript=transcript,
        )
        if self._pipeline_log:
            self._pipeline_log(
                f"✓ Frames ready for {chunk.chunk_id} ({len(frames)} file(s))"
            )
        return frames


class _ProgressInferencer:
    """Wrap chunk inference to report coarse processing progress."""

    def __init__(
        self,
        inner: VideoQAChunkInferencer,
        *,
        progress: Callable[[str, float], None],
        total_chunks: int,
        chunk_state: dict[str, int],
        pipeline_log: Callable[[str], None] | None = None,
    ) -> None:
        self._inner = inner
        self._progress = progress
        self._total_chunks = max(1, int(total_chunks))
        self._chunk_state = chunk_state
        self._pipeline_log = pipeline_log
        self._infer_durations: list[float] = []

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
        base = _VIDEO_QA_PRE_VLM_PROGRESS_FRAC + (chunk_index - 1) * per_chunk_span
        done_before = len(self._infer_durations)
        remaining = max(0, self._total_chunks - done_before)
        avg = (
            sum(self._infer_durations) / len(self._infer_durations)
            if self._infer_durations
            else None
        )
        eta_s = avg * remaining if avg is not None and remaining else None
        detail = (
            f"Processing · VLM chunks {done_before}/{self._total_chunks} · "
            f"ETA {_format_vlm_eta_seconds(eta_s)}"
        )
        self._progress(detail, min(0.93, base + per_chunk_span * 0.5))
        if self._pipeline_log:
            self._pipeline_log(
                f"→ llm_pass: {chunk.chunk_id} ({chunk_index}/{self._total_chunks})"
            )
        t0 = time.perf_counter()
        outcome = self._inner.infer_chunk(
            chunk=chunk,
            frames=frames,
            transcript=transcript,
            manifest=manifest,
        )
        self._infer_durations.append(time.perf_counter() - t0)
        status = "ok" if outcome.ok else "failed"
        if self._pipeline_log:
            self._pipeline_log(f"✓ llm_pass finished for {chunk.chunk_id} ({status})")
        return outcome


class _ProgressAggregator:
    """Wrap final answer aggregation to report synthesis progress."""

    def __init__(
        self,
        inner: VideoQAAnswerAggregator,
        *,
        progress: Callable[[str, float], None],
        pipeline_log: Callable[[str], None] | None = None,
    ) -> None:
        self._inner = inner
        self._progress = progress
        self._pipeline_log = pipeline_log

    def aggregate(
        self,
        *,
        context: VideoQAContextBundle,
        manifest: VideoQARunManifest,
        transcript: VideoQATranscriptArtifacts,
        chunk_results: Sequence[VideoQAChunkExecutionResult],
    ) -> VideoQAAnswerBundle:
        self._progress("Synthesizing final answer (LM Studio JSON)", 0.95)
        if self._pipeline_log:
            self._pipeline_log("→ Stage: answer_aggregate (final LM Studio JSON call)")
        bundle = self._inner.aggregate(
            context=context,
            manifest=manifest,
            transcript=transcript,
            chunk_results=chunk_results,
        )
        if self._pipeline_log:
            self._pipeline_log("✓ Stage: answer_aggregate complete")
        return bundle


def _wrap_local_executor_deps_with_progress(
    deps: VideoQAExecutorDeps,
    *,
    progress: Callable[[str, float], None],
    total_chunks: int,
    pipeline_log: Callable[[str], None] | None = None,
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
            pipeline_log=pipeline_log,
        ),
        chunk_inferencer=_ProgressInferencer(
            deps.chunk_inferencer,
            progress=progress,
            total_chunks=total_chunks,
            chunk_state=chunk_state,
            pipeline_log=pipeline_log,
        ),
        answer_aggregator=_ProgressAggregator(
            deps.answer_aggregator,
            progress=progress,
            pipeline_log=pipeline_log,
        ),
    )


def run_local_video_qa(  # noqa: PLR0913
    *,
    params: VideoQALocalRunParams,
    whisper: WhisperXWrapper,
    should_cancel: Callable[[], bool] | None = None,
    progress: Callable[[str, float], None] | None = None,
    pipeline_log: Callable[[str], None] | None = None,
    chunk_inferencer_override: VideoQAChunkInferencer | None = None,
    transcript_override: VideoQATranscriptProvider | None = None,
    frame_override: VideoQAFrameMaterializer | None = None,
    aggregator_override: VideoQAAnswerAggregator | None = None,
) -> VideoQAExecutorRunOutcome:
    """Run the full Video QA pipeline for a local file and optional attachments.

    Re-runs preflight and blocks via :func:`ensure_local_video_qa_run_allowed`.
    Writes manifest and answer JSON under ``output_dir`` when the run completes.

    Args:
        params: Run inputs (context, LM URL/model, output dir, sampling fps, …).
        whisper: Loaded Whisper backend for the transcript stage.
        should_cancel: Optional cooperative cancel predicate.
        progress: Optional ``(message, fraction)`` callback; ``fraction`` is 0..1
            (first ~45% maps to pre-VLM work, the rest to VLM for a 0..200 UI bar).
        pipeline_log: Optional line-by-line pipeline log (GUI); explains that LM
            Studio's prompt-% logs are server-side only, not visible over HTTP.
        chunk_inferencer_override: Test hook replacing per-chunk LM calls.
        transcript_override: Test hook replacing ASR.
        frame_override: Test hook replacing ffmpeg frame extraction.
        aggregator_override: Test hook replacing final synthesis.

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
        frame_sample_fps=params.frame_sample_fps,
    )
    ensure_local_video_qa_run_allowed(report, chunk_plan)
    if pipeline_log:
        pipeline_log(f"✓ Preflight OK — {len(chunk_plan)} chunk(s) planned.")

    manifest = build_video_qa_preparation_manifest(ctx)
    staging_dir = params.output_dir / "_video_qa_work" / manifest.run_id
    staging_dir.mkdir(parents=True, exist_ok=True)

    if pipeline_log:
        pipeline_log("=== Video QA · progress log ===")
        pipeline_log(
            "LM Studio prints prompt/token progress in its server console; the "
            "OpenAI-compatible HTTP API does not expose that stream to AskVLM."
        )

    if progress:
        progress("Preparing", 0.05)

    swap_hook = _build_lm_studio_dual_local_swap_hook(
        chunk_lm=params.chunk_lm,
        final_lm=params.final_lm,
        context_window_tokens=params.context_window_tokens,
        pipeline_log=pipeline_log,
        should_cancel=should_cancel,
    )
    deps = build_video_qa_local_executor_deps(
        context=ctx,
        whisper=whisper,
        staging_dir=staging_dir,
        chunk_lm=params.chunk_lm,
        final_lm=params.final_lm,
        should_cancel=should_cancel,
        progress=progress,
        pipeline_log=pipeline_log,
        frame_sample_fps=params.frame_sample_fps,
        chunk_inferencer_override=chunk_inferencer_override,
        transcript_override=transcript_override,
        frame_override=frame_override,
        aggregator_override=aggregator_override,
        before_answer_aggregate=swap_hook,
    )

    if progress is not None or pipeline_log is not None:
        prog_cb = progress or (lambda _m, _f: None)
        deps = _wrap_local_executor_deps_with_progress(
            deps,
            progress=prog_cb,
            total_chunks=max(1, len(chunk_plan)),
            pipeline_log=pipeline_log,
        )

    outcome = run_video_qa_executor(
        context=ctx,
        manifest=manifest,
        planned_chunks=chunk_plan,
        deps=deps,
        representative_frame_policy=default_representative_frame_policy(),
        should_cancel=should_cancel,
    )

    final_manifest = outcome.manifest
    manifest_path = params.output_dir / f"{final_manifest.run_id}.manifest.json"
    save_video_qa_manifest_json(manifest_path, final_manifest)
    answer_path = answer_bundle_path_for_manifest(manifest_path)
    save_answer_bundle_to_json(answer_path, outcome.answer_bundle)

    if progress:
        progress("Completed", 1.0)
    if pipeline_log:
        pipeline_log("✓ Pipeline completed (manifest + answer bundle saved)")

    return outcome
