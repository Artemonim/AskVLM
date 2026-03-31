from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from core.pipelines import CancelledError
from core.video_qa_answer_bundle import (
    ANSWER_BUNDLE_SCHEMA_VERSION,
    VideoQAAnswerBundle,
)
from core.video_qa_context import normalize_video_qa_context
from core.video_qa_executor import (
    VideoQAChunkInferenceOutcome,
    VideoQAChunkInferencer,
    VideoQAExecutorDeps,
    VideoQAExecutorRunOutcome,
    VideoQAFrameMaterializer,
    VideoQATranscriptArtifacts,
    VideoQATranscriptProvider,
    run_video_qa_executor,
    video_qa_executor_pipeline_stage_names,
)
from core.video_qa_manifest import (
    SCHEMA_VERSION,
    VideoQAChunkRecord,
)
from core.video_qa_orchestration import (
    VideoQAPlannedChunk,
    merge_planned_chunks_into_manifest,
)
from core.video_qa_preparation import build_video_qa_preparation_manifest
from core.video_qa_sources import LocalFileProvider

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from core.video_qa_context import VideoQAContextBundle
    from core.video_qa_executor import (
        VideoQAChunkExecutionResult,
    )
    from core.video_qa_manifest import (
        VideoQARunManifest,
    )


def _minimal_bundle(tmp_path: Path, question: str = "Q?") -> VideoQAContextBundle:
    clip = tmp_path / "v.mp4"
    clip.write_bytes(b"x")
    source = LocalFileProvider().resolve(clip)
    return normalize_video_qa_context(source=source, question=question, attachments=())


def _answer_bundle(
    run_id: str,
    question: str,
    *,
    answer: str = "ok",
    manifest_run_id: str | None = "m1",
) -> VideoQAAnswerBundle:
    return VideoQAAnswerBundle(
        schema_version=ANSWER_BUNDLE_SCHEMA_VERSION,
        run_id=f"{run_id}-answer",
        created_at="2026-03-30T12:00:00Z",
        question=question,
        answer=answer,
        evidence=(),
        is_uncertain=False,
        manifest_run_id=manifest_run_id,
    )


class _RecordingTranscript(VideoQATranscriptProvider):
    def __init__(self, text: str) -> None:
        self.calls = 0
        self._text = text

    def prepare_transcript(
        self,
        _context: VideoQAContextBundle,
        _manifest: VideoQARunManifest,
    ) -> VideoQATranscriptArtifacts:
        self.calls += 1
        return VideoQATranscriptArtifacts(
            transcript_text=self._text,
            subtitle_text="[subtitles]",
        )


class _RecordingFrames(VideoQAFrameMaterializer):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def materialize_frames(
        self,
        *,
        chunk: VideoQAChunkRecord,
        representative_timestamp: float,
        manifest: VideoQARunManifest,
        transcript: VideoQATranscriptArtifacts,
    ) -> tuple[str, ...]:
        self.calls.append(chunk.chunk_id)
        _ = (representative_timestamp, manifest, transcript)
        return (f"{chunk.chunk_id}-frame.png",)


class _GateInferencer(VideoQAChunkInferencer):
    def __init__(self, fail_ids: frozenset[str]) -> None:
        self.calls: list[str] = []
        self._fail_ids = fail_ids

    def infer_chunk(
        self,
        *,
        chunk: VideoQAChunkRecord,
        frames: tuple[str, ...],
        transcript: VideoQATranscriptArtifacts,
        manifest: VideoQARunManifest,
    ) -> VideoQAChunkInferenceOutcome:
        _ = (frames, transcript, manifest)
        self.calls.append(chunk.chunk_id)
        if chunk.chunk_id in self._fail_ids:
            return VideoQAChunkInferenceOutcome(ok=False, error="boom")
        return VideoQAChunkInferenceOutcome(
            ok=True, artifacts=(f"{chunk.chunk_id}-out",)
        )


class _Aggregator:
    def __init__(self, run_id: str) -> None:
        self.calls = 0
        self._run_id = run_id
        self._last_chunk_results: tuple[VideoQAChunkExecutionResult, ...] = ()
        self._last_manifest: VideoQARunManifest | None = None
        self._last_transcript: VideoQATranscriptArtifacts | None = None

    def aggregate(
        self,
        *,
        context: VideoQAContextBundle,
        manifest: VideoQARunManifest,
        transcript: VideoQATranscriptArtifacts,
        chunk_results: Sequence[VideoQAChunkExecutionResult],
    ) -> VideoQAAnswerBundle:
        self.calls += 1
        self._last_chunk_results = tuple(chunk_results)
        self._last_manifest = manifest
        self._last_transcript = transcript
        return _answer_bundle(self._run_id, context.question)


def _manifest_with_chunks(
    bundle: VideoQAContextBundle,
    chunks: tuple[VideoQAChunkRecord, ...],
    *,
    run_id: str = "run-exec",
) -> VideoQARunManifest:
    base = build_video_qa_preparation_manifest(
        bundle, run_id=run_id, created_at="2026-03-30T12:00:00Z"
    )
    return replace(base, chunks=chunks)


def test_before_answer_aggregate_runs_before_synthesis_aggregate(
    tmp_path: Path,
) -> None:
    """Optional hook runs immediately before ``answer_aggregator.aggregate``."""
    order: list[str] = []
    bundle = _minimal_bundle(tmp_path)
    plan = (
        VideoQAPlannedChunk(
            chunk_id="c0",
            t_start=0.0,
            t_end=1.0,
            planning_mode="uniform_grid",
        ),
    )
    manifest = merge_planned_chunks_into_manifest(
        _manifest_with_chunks(bundle, ()),
        plan,
    )

    class OrderAgg(_Aggregator):
        def aggregate(
            self,
            *,
            context: VideoQAContextBundle,
            manifest: VideoQARunManifest,
            transcript: VideoQATranscriptArtifacts,
            chunk_results: Sequence[VideoQAChunkExecutionResult],
        ) -> VideoQAAnswerBundle:
            order.append("aggregate")
            return super().aggregate(
                context=context,
                manifest=manifest,
                transcript=transcript,
                chunk_results=chunk_results,
            )

    def hook() -> None:
        order.append("hook")

    deps = VideoQAExecutorDeps(
        transcript=_RecordingTranscript("t"),
        frame_materializer=_RecordingFrames(),
        chunk_inferencer=_GateInferencer(fail_ids=frozenset()),
        answer_aggregator=OrderAgg("ord"),
        before_answer_aggregate=hook,
    )
    run_video_qa_executor(
        context=bundle,
        manifest=manifest,
        planned_chunks=plan,
        deps=deps,
    )
    assert order == ["hook", "aggregate"]


def test_transcript_separate_from_answer_bundle(tmp_path: Path) -> None:
    """Transcript artifacts are not embedded in the answer bundle object graph."""
    bundle = _minimal_bundle(tmp_path)
    plan = (
        VideoQAPlannedChunk(
            chunk_id="c0",
            t_start=0.0,
            t_end=10.0,
            planning_mode="uniform_grid",
        ),
    )
    chunks = (
        VideoQAChunkRecord(
            chunk_id="c0",
            t_start=0.0,
            t_end=10.0,
            status="pending",
        ),
    )
    manifest = merge_planned_chunks_into_manifest(
        _manifest_with_chunks(bundle, chunks),
        plan,
    )
    tr = _RecordingTranscript("hello transcript")
    fr = _RecordingFrames()
    inf = _GateInferencer(fail_ids=frozenset())
    agg = _Aggregator("run-1")
    deps = VideoQAExecutorDeps(
        transcript=tr,
        frame_materializer=fr,
        chunk_inferencer=inf,
        answer_aggregator=agg,
    )
    outcome = run_video_qa_executor(
        context=bundle,
        manifest=manifest,
        planned_chunks=plan,
        deps=deps,
    )
    assert isinstance(outcome, VideoQAExecutorRunOutcome)
    assert outcome.transcript.transcript_text == "hello transcript"
    assert outcome.transcript.subtitle_text == "[subtitles]"
    assert outcome.answer_bundle.question == bundle.question
    assert outcome.answer_bundle.answer == "ok"
    assert outcome.transcript.transcript_text not in outcome.answer_bundle.answer
    assert outcome.manifest.status == "completed"
    assert outcome.manifest.error is None


def test_executor_pipeline_stage_names_include_attachment_prepare() -> None:
    """Canonical stage list matches planning graph (attachment step explicit)."""
    names = video_qa_executor_pipeline_stage_names()
    assert names[:3] == ("source_resolve", "attachment_prepare", "transcript_prepare")
    assert "chunk_plan" in names
    assert names[-1] == "answer_aggregate"


def test_stage_order_matches_pipeline(tmp_path: Path) -> None:
    """Executor runs source/transcript/plan, per-chunk frames+llm, then aggregate."""
    bundle = _minimal_bundle(tmp_path)
    plan = (
        VideoQAPlannedChunk(
            chunk_id="a", t_start=0.0, t_end=5.0, planning_mode="uniform_grid"
        ),
        VideoQAPlannedChunk(
            chunk_id="b", t_start=5.0, t_end=10.0, planning_mode="uniform_grid"
        ),
    )
    manifest = merge_planned_chunks_into_manifest(
        _manifest_with_chunks(bundle, ()),
        plan,
    )
    deps = VideoQAExecutorDeps(
        transcript=_RecordingTranscript("t"),
        frame_materializer=_RecordingFrames(),
        chunk_inferencer=_GateInferencer(fail_ids=frozenset()),
        answer_aggregator=_Aggregator("run-2"),
    )
    out = run_video_qa_executor(
        context=bundle,
        manifest=manifest,
        planned_chunks=plan,
        deps=deps,
    )
    seq = list(out.stage_sequence)
    assert seq[0] == "source_resolve"
    assert seq[1] == "attachment_prepare"
    assert seq[2] == "transcript_prepare"
    assert seq[3] == "chunk_plan"
    assert seq[-1] == "answer_aggregate"
    assert seq.index("frame_select:a") < seq.index("llm_pass:a")
    assert seq.index("llm_pass:a") < seq.index("frame_select:b")
    assert seq.index("frame_select:b") < seq.index("llm_pass:b")


def test_completed_chunks_skipped_on_resume(tmp_path: Path) -> None:
    """Resume does not invoke frame or inference hooks for completed chunk rows."""
    bundle = _minimal_bundle(tmp_path)
    plan = (
        VideoQAPlannedChunk(
            chunk_id="done", t_start=0.0, t_end=1.0, planning_mode="uniform_grid"
        ),
        VideoQAPlannedChunk(
            chunk_id="next", t_start=1.0, t_end=2.0, planning_mode="uniform_grid"
        ),
    )
    manifest = merge_planned_chunks_into_manifest(
        _manifest_with_chunks(
            bundle,
            (
                VideoQAChunkRecord(
                    chunk_id="done",
                    t_start=0.0,
                    t_end=1.0,
                    status="completed",
                    attempts=1,
                    frames=("old.png",),
                ),
                VideoQAChunkRecord(
                    chunk_id="next",
                    t_start=1.0,
                    t_end=2.0,
                    status="pending",
                ),
            ),
        ),
        plan,
    )
    fr = _RecordingFrames()
    inf = _GateInferencer(fail_ids=frozenset())
    deps = VideoQAExecutorDeps(
        transcript=_RecordingTranscript("t"),
        frame_materializer=fr,
        chunk_inferencer=inf,
        answer_aggregator=_Aggregator("run-3"),
    )
    out = run_video_qa_executor(
        context=bundle,
        manifest=manifest,
        planned_chunks=plan,
        deps=deps,
    )
    assert "chunk_skip_completed:done" in out.stage_sequence
    assert fr.calls == ["next"]
    assert inf.calls == ["next"]
    by_id = {c.chunk_id: c for c in out.manifest.chunks}
    assert by_id["done"].frames == ("old.png",)
    assert by_id["done"].attempts == 1


def test_resume_second_run_is_idempotent_for_completed_chunks(tmp_path: Path) -> None:
    """A second full executor pass does not repeat inference for completed chunks."""
    bundle = _minimal_bundle(tmp_path)
    plan = (
        VideoQAPlannedChunk(
            chunk_id="c0", t_start=0.0, t_end=10.0, planning_mode="uniform_grid"
        ),
    )
    manifest = merge_planned_chunks_into_manifest(
        _manifest_with_chunks(bundle, ()), plan
    )
    fr = _RecordingFrames()
    inf = _GateInferencer(fail_ids=frozenset())
    agg = _Aggregator("run-4")
    deps = VideoQAExecutorDeps(
        transcript=_RecordingTranscript("t"),
        frame_materializer=fr,
        chunk_inferencer=inf,
        answer_aggregator=agg,
    )
    first = run_video_qa_executor(
        context=bundle,
        manifest=manifest,
        planned_chunks=plan,
        deps=deps,
    )
    assert inf.calls == ["c0"]
    merged_again = merge_planned_chunks_into_manifest(first.manifest, plan)
    second = run_video_qa_executor(
        context=bundle,
        manifest=merged_again,
        planned_chunks=plan,
        deps=deps,
    )
    assert inf.calls == ["c0"]
    assert fr.calls == ["c0"]
    assert "chunk_skip_completed:c0" in second.stage_sequence
    c0_first = next(c for c in first.manifest.chunks if c.chunk_id == "c0")
    c0_second = next(c for c in second.manifest.chunks if c.chunk_id == "c0")
    assert c0_first == c0_second


def test_chunk_failure_surfaces_in_manifest_and_results(tmp_path: Path) -> None:
    """Failed inference marks the chunk failed and passes error through."""
    bundle = _minimal_bundle(tmp_path)
    plan = (
        VideoQAPlannedChunk(
            chunk_id="bad", t_start=0.0, t_end=1.0, planning_mode="uniform_grid"
        ),
    )
    manifest = merge_planned_chunks_into_manifest(
        _manifest_with_chunks(bundle, ()), plan
    )
    deps = VideoQAExecutorDeps(
        transcript=_RecordingTranscript("t"),
        frame_materializer=_RecordingFrames(),
        chunk_inferencer=_GateInferencer(fail_ids=frozenset({"bad"})),
        answer_aggregator=_Aggregator("run-5"),
    )
    out = run_video_qa_executor(
        context=bundle,
        manifest=manifest,
        planned_chunks=plan,
        deps=deps,
    )
    chunk = next(c for c in out.manifest.chunks if c.chunk_id == "bad")
    assert chunk.status == "failed"
    assert chunk.error == "boom"
    assert chunk.attempts == 1
    res = next(r for r in out.chunk_results if r.chunk_id == "bad")
    assert res.status == "failed"
    assert res.error == "boom"
    assert out.manifest.status == "failed"
    assert out.manifest.error is not None
    assert "bad" in out.manifest.error
    assert "boom" in out.manifest.error


def test_partial_success_mixed_chunk_outcomes(tmp_path: Path) -> None:
    """One chunk completes and one fails: run manifest is failed with a summary."""
    bundle = _minimal_bundle(tmp_path)
    plan = (
        VideoQAPlannedChunk(
            chunk_id="ok-chunk", t_start=0.0, t_end=5.0, planning_mode="uniform_grid"
        ),
        VideoQAPlannedChunk(
            chunk_id="fail-chunk",
            t_start=5.0,
            t_end=10.0,
            planning_mode="uniform_grid",
        ),
    )
    manifest = merge_planned_chunks_into_manifest(
        _manifest_with_chunks(bundle, ()),
        plan,
    )
    deps = VideoQAExecutorDeps(
        transcript=_RecordingTranscript("t"),
        frame_materializer=_RecordingFrames(),
        chunk_inferencer=_GateInferencer(fail_ids=frozenset({"fail-chunk"})),
        answer_aggregator=_Aggregator("run-mixed"),
    )
    out = run_video_qa_executor(
        context=bundle,
        manifest=manifest,
        planned_chunks=plan,
        deps=deps,
    )
    assert out.manifest.status == "failed"
    assert out.manifest.error is not None
    assert "fail-chunk" in out.manifest.error
    assert "boom" in out.manifest.error

    by_id = {c.chunk_id: c for c in out.manifest.chunks}
    assert by_id["ok-chunk"].status == "completed"
    assert by_id["fail-chunk"].status == "failed"

    statuses = {r.chunk_id: r.status for r in out.chunk_results}
    assert statuses["ok-chunk"] == "completed"
    assert statuses["fail-chunk"] == "failed"


def test_merge_twice_after_executor_idempotent(tmp_path: Path) -> None:
    """Re-merging the same plan after a finished run does not mutate chunk rows."""
    bundle = _minimal_bundle(tmp_path)
    plan = (
        VideoQAPlannedChunk(
            chunk_id="c0", t_start=0.0, t_end=10.0, planning_mode="uniform_grid"
        ),
    )
    manifest = merge_planned_chunks_into_manifest(
        _manifest_with_chunks(bundle, ()), plan
    )
    deps = VideoQAExecutorDeps(
        transcript=_RecordingTranscript("t"),
        frame_materializer=_RecordingFrames(),
        chunk_inferencer=_GateInferencer(fail_ids=frozenset()),
        answer_aggregator=_Aggregator("run-6"),
    )
    finished = run_video_qa_executor(
        context=bundle,
        manifest=manifest,
        planned_chunks=plan,
        deps=deps,
    ).manifest
    once = merge_planned_chunks_into_manifest(finished, plan)
    twice = merge_planned_chunks_into_manifest(once, plan)
    assert once.chunks == twice.chunks


def test_manifest_schema_preserved(tmp_path: Path) -> None:
    """Executor preserves manifest schema version and run identity fields."""
    bundle = _minimal_bundle(tmp_path)
    plan = (
        VideoQAPlannedChunk(
            chunk_id="c0", t_start=0.0, t_end=1.0, planning_mode="uniform_grid"
        ),
    )
    manifest = merge_planned_chunks_into_manifest(
        replace(
            _manifest_with_chunks(bundle, (), run_id="fixed-run"),
            schema_version=SCHEMA_VERSION,
        ),
        plan,
    )
    deps = VideoQAExecutorDeps(
        transcript=_RecordingTranscript("t"),
        frame_materializer=_RecordingFrames(),
        chunk_inferencer=_GateInferencer(fail_ids=frozenset()),
        answer_aggregator=_Aggregator("fixed-run"),
    )
    out = run_video_qa_executor(
        context=bundle,
        manifest=manifest,
        planned_chunks=plan,
        deps=deps,
    )
    assert out.manifest.schema_version == SCHEMA_VERSION
    assert out.manifest.run_id == "fixed-run"


def test_executor_should_cancel_aborts_before_chunk_work(tmp_path: Path) -> None:
    """``should_cancel`` stops the executor before starting pending chunk work."""
    bundle = _minimal_bundle(tmp_path)
    plan = (
        VideoQAPlannedChunk(
            chunk_id="c0", t_start=0.0, t_end=1.0, planning_mode="uniform_grid"
        ),
    )
    manifest = merge_planned_chunks_into_manifest(
        _manifest_with_chunks(bundle, ()),
        plan,
    )
    frames = _RecordingFrames()
    inf = _GateInferencer(fail_ids=frozenset())
    deps = VideoQAExecutorDeps(
        transcript=_RecordingTranscript("t"),
        frame_materializer=frames,
        chunk_inferencer=inf,
        answer_aggregator=_Aggregator("run-cancel"),
    )
    with pytest.raises(CancelledError):
        run_video_qa_executor(
            context=bundle,
            manifest=manifest,
            planned_chunks=plan,
            deps=deps,
            should_cancel=lambda: True,
        )
    assert frames.calls == []
    assert inf.calls == []
