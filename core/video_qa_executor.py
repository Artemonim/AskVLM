"""Video QA executor: orchestrates planning/runtime hooks with injected backends."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Literal, Protocol, runtime_checkable

from .pipelines import CancelledError
from .video_qa_orchestration import (
    default_representative_frame_policy,
    merge_planned_chunks_into_manifest,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from .video_qa_answer_bundle import VideoQAAnswerBundle
    from .video_qa_context import VideoQAContextBundle
    from .video_qa_manifest import VideoQAChunkRecord, VideoQARunManifest
    from .video_qa_orchestration import (
        VideoQAPlannedChunk,
        VideoQARepresentativeFramePolicy,
    )

ChunkExecutionStatus = Literal["skipped_completed", "completed", "failed"]


@dataclass(frozen=True, slots=True)
class VideoQATranscriptArtifacts:
    """Transcript and subtitle-oriented product separate from QA answer artifacts."""

    transcript_text: str
    subtitle_text: str = ""
    segments: tuple[tuple[float, float, str], ...] = ()


@dataclass(frozen=True, slots=True)
class VideoQAChunkInferenceOutcome:
    """Result of one chunk-level inference pass (no live model in this module)."""

    ok: bool
    artifacts: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True, slots=True)
class VideoQAChunkExecutionResult:
    """Per-chunk execution summary for aggregation and tests."""

    chunk_id: str
    status: ChunkExecutionStatus
    frames: tuple[str, ...] = ()
    error: str | None = None
    inference_attempted: bool = False


@dataclass(frozen=True, slots=True)
class VideoQAExecutorRunOutcome:
    """Executor result: transcript, answer bundle, manifest, chunk rows, stages."""

    transcript: VideoQATranscriptArtifacts
    answer_bundle: VideoQAAnswerBundle
    manifest: VideoQARunManifest
    chunk_results: tuple[VideoQAChunkExecutionResult, ...]
    stage_sequence: tuple[str, ...]


_EXECUTOR_PIPELINE_STAGES: Final[tuple[str, ...]] = (
    "source_resolve",
    "attachment_prepare",
    "transcript_prepare",
    "chunk_plan",
    "answer_aggregate",
)


@runtime_checkable
class VideoQASourceResolver(Protocol):
    """Resolves or validates the media source before transcript work."""

    def resolve_source(
        self,
        context: VideoQAContextBundle,
        manifest: VideoQARunManifest,
    ) -> VideoQARunManifest:
        """Return an updated manifest after source resolution (may be a no-op)."""


@runtime_checkable
class VideoQATranscriptProvider(Protocol):
    """Builds or loads transcript text for subtitle-first outputs."""

    def prepare_transcript(
        self,
        context: VideoQAContextBundle,
        manifest: VideoQARunManifest,
    ) -> VideoQATranscriptArtifacts:
        """Return transcript artifacts that are independent of QA chunk inference."""


@runtime_checkable
class VideoQAFrameMaterializer(Protocol):
    """Materializes representative frame paths for a chunk."""

    def materialize_frames(
        self,
        *,
        chunk: VideoQAChunkRecord,
        representative_timestamp: float,
        manifest: VideoQARunManifest,
        transcript: VideoQATranscriptArtifacts,
    ) -> tuple[str, ...]:
        """Return frame artifact paths for the chunk."""


@runtime_checkable
class VideoQAChunkInferencer(Protocol):
    """Per-chunk multimodal inference (injected; no network in this module)."""

    def infer_chunk(
        self,
        *,
        chunk: VideoQAChunkRecord,
        frames: tuple[str, ...],
        transcript: VideoQATranscriptArtifacts,
        manifest: VideoQARunManifest,
    ) -> VideoQAChunkInferenceOutcome:
        """Run one chunk inference pass and return success or error payload."""


@runtime_checkable
class VideoQAAnswerAggregator(Protocol):
    """Builds the final :class:`VideoQAAnswerBundle` from manifest and chunk results."""

    def aggregate(
        self,
        *,
        context: VideoQAContextBundle,
        manifest: VideoQARunManifest,
        transcript: VideoQATranscriptArtifacts,
        chunk_results: Sequence[VideoQAChunkExecutionResult],
    ) -> VideoQAAnswerBundle:
        """Produce the final answer bundle for export and replay."""


@dataclass(frozen=True, slots=True)
class VideoQAExecutorDeps:
    """Injectable backends for executor stages."""

    transcript: VideoQATranscriptProvider
    frame_materializer: VideoQAFrameMaterializer
    chunk_inferencer: VideoQAChunkInferencer
    answer_aggregator: VideoQAAnswerAggregator
    source_resolver: VideoQASourceResolver | None = None
    before_answer_aggregate: Callable[[], None] | None = None


def _run_status_from_chunk_results(
    chunk_results: Sequence[VideoQAChunkExecutionResult],
) -> tuple[Literal["completed", "failed"], str | None]:
    """Map chunk outcomes to manifest run-level ``status`` and ``error`` summary."""
    failures = [
        (r.chunk_id, (r.error or "unknown_error").strip())
        for r in chunk_results
        if r.status == "failed"
    ]
    if not failures:
        return ("completed", None)
    parts = [f"{cid}: {err}" for cid, err in failures]
    summary = f"Chunk failure(s) in this run ({len(failures)}): " + "; ".join(parts)
    return ("failed", summary)


def _map_chunk(
    manifest: VideoQARunManifest,
    chunk_id: str,
    transform: Callable[[VideoQAChunkRecord], VideoQAChunkRecord],
) -> VideoQARunManifest:
    """Return a manifest whose ``chunk_id`` row is replaced by ``transform``."""
    new_chunks: list[VideoQAChunkRecord] = []
    for record in manifest.chunks:
        if record.chunk_id == chunk_id:
            new_chunks.append(transform(record))
        else:
            new_chunks.append(record)
    return replace(manifest, chunks=tuple(new_chunks))


def _raise_if_user_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    """Raise :class:`CancelledError` when the injected cancel predicate is true."""
    if should_cancel and should_cancel():
        msg = "Canceled"
        raise CancelledError(msg)


def run_video_qa_executor(
    *,
    context: VideoQAContextBundle,
    manifest: VideoQARunManifest,
    planned_chunks: Sequence[VideoQAPlannedChunk],
    deps: VideoQAExecutorDeps,
    representative_frame_policy: VideoQARepresentativeFramePolicy | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> VideoQAExecutorRunOutcome:
    """Run the Video QA pipeline with injected backends and manifest resume semantics.

    Stages are recorded in ``stage_sequence`` in pipeline order, aligned with the
    planning graph: ``attachment_prepare`` is a no-op marker for context already
    normalized into ``manifest`` / ``context``. Chunk-level work emits
    ``frame_select:<chunk_id>`` and ``llm_pass:<chunk_id>`` between ``chunk_plan``
    and ``answer_aggregate``. Chunks with status ``completed`` are skipped so resume
    is idempotent at the executor level. Run-level ``manifest.status`` is
    ``failed`` with an error summary if any chunk failed; otherwise ``completed``.

    Args:
        context: Normalized prompt and source context.
        manifest: Existing manifest (may already contain completed chunks).
        planned_chunks: Planned chunk rows merged into ``manifest`` before execution.
        deps: Injectable providers for transcript, frames, inference, and aggregation.
        representative_frame_policy: Policy for timestamps; defaults to chunk midpoints.
        should_cancel: Optional polling hook that aborts before chunk work, after
            frame extraction, and before the final aggregate LM call.

    Returns:
        Transcript artifacts, final answer bundle, updated manifest, per-chunk
        results, and stage order for auditing and tests.

    """
    rep = representative_frame_policy or default_representative_frame_policy()
    stages: list[str] = []

    working = replace(manifest, status="running", error=None)

    stages.append("source_resolve")
    if deps.source_resolver is not None:
        working = deps.source_resolver.resolve_source(context, working)

    # * Matches graph ``attachment_prepare``: bundle is already normalized before run.
    stages.append("attachment_prepare")

    stages.append("transcript_prepare")
    transcript = deps.transcript.prepare_transcript(context, working)

    stages.append("chunk_plan")
    working = merge_planned_chunks_into_manifest(working, planned_chunks)

    chunk_results: list[VideoQAChunkExecutionResult] = []
    planned_ids = [chunk.chunk_id for chunk in planned_chunks]

    for chunk_id in planned_ids:
        _raise_if_user_cancelled(should_cancel)

        record = next((c for c in working.chunks if c.chunk_id == chunk_id), None)
        if record is None:
            continue

        if record.status == "completed":
            # * Idempotent resume: do not rerun frame selection or inference.
            stages.append(f"chunk_skip_completed:{chunk_id}")
            chunk_results.append(
                VideoQAChunkExecutionResult(
                    chunk_id=chunk_id,
                    status="skipped_completed",
                    frames=record.frames,
                    error=None,
                    inference_attempted=False,
                )
            )
            continue

        rep_ts = rep.timestamp_for_span(record.t_start, record.t_end)
        stages.append(f"frame_select:{chunk_id}")
        frames = deps.frame_materializer.materialize_frames(
            chunk=record,
            representative_timestamp=rep_ts,
            manifest=working,
            transcript=transcript,
        )

        _raise_if_user_cancelled(should_cancel)

        def _patch_running(
            r: VideoQAChunkRecord,
            frames_: tuple[str, ...] = frames,
        ) -> VideoQAChunkRecord:
            return replace(r, frames=frames_, status="running", error=None)

        working = _map_chunk(working, chunk_id, _patch_running)

        stages.append(f"llm_pass:{chunk_id}")
        current = next(c for c in working.chunks if c.chunk_id == chunk_id)
        attempts = int(current.attempts) + 1
        outcome = deps.chunk_inferencer.infer_chunk(
            chunk=current,
            frames=frames,
            transcript=transcript,
            manifest=working,
        )

        if outcome.ok:
            merged_artifacts = tuple(
                dict.fromkeys((*current.artifacts, *outcome.artifacts))
            )

            def _patch_completed(
                r: VideoQAChunkRecord,
                merged: tuple[str, ...] = merged_artifacts,
                att: int = attempts,
            ) -> VideoQAChunkRecord:
                return replace(
                    r,
                    artifacts=merged,
                    status="completed",
                    attempts=att,
                    error=None,
                )

            working = _map_chunk(working, chunk_id, _patch_completed)
            chunk_results.append(
                VideoQAChunkExecutionResult(
                    chunk_id=chunk_id,
                    status="completed",
                    frames=frames,
                    error=None,
                    inference_attempted=True,
                )
            )
        else:
            err = outcome.error or "chunk_inference_failed"

            def _patch_failed(
                r: VideoQAChunkRecord,
                att: int = attempts,
                err_: str = err,
            ) -> VideoQAChunkRecord:
                return replace(
                    r,
                    status="failed",
                    attempts=att,
                    error=err_,
                )

            working = _map_chunk(working, chunk_id, _patch_failed)
            chunk_results.append(
                VideoQAChunkExecutionResult(
                    chunk_id=chunk_id,
                    status="failed",
                    frames=frames,
                    error=err,
                    inference_attempted=True,
                )
            )

    _raise_if_user_cancelled(should_cancel)

    if deps.before_answer_aggregate is not None:
        deps.before_answer_aggregate()

    stages.append("answer_aggregate")
    answer_bundle = deps.answer_aggregator.aggregate(
        context=context,
        manifest=working,
        transcript=transcript,
        chunk_results=chunk_results,
    )

    run_status, run_error = _run_status_from_chunk_results(chunk_results)
    final_manifest = replace(working, status=run_status, error=run_error)
    return VideoQAExecutorRunOutcome(
        transcript=transcript,
        answer_bundle=answer_bundle,
        manifest=final_manifest,
        chunk_results=tuple(chunk_results),
        stage_sequence=tuple(stages),
    )


def video_qa_executor_pipeline_stage_names() -> tuple[str, ...]:
    """Return canonical high-level stage names for documentation and tests."""
    return _EXECUTOR_PIPELINE_STAGES
