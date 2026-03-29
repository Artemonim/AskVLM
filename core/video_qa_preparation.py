"""Pure preparation helpers for Video QA manifest skeletons."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .video_qa_manifest import (
    SCHEMA_VERSION,
    VideoQAGraphNode,
    VideoQARunManifest,
    VideoQASourceSnapshot,
    create_manifest_timestamp,
    create_video_qa_run_id,
    validate_manifest_timestamp,
)

if TYPE_CHECKING:
    from .video_qa_context import VideoQAContextBundle


def build_video_qa_preparation_manifest(
    bundle: VideoQAContextBundle,
    *,
    run_id: str | None = None,
    created_at: str | None = None,
) -> VideoQARunManifest:
    """Build a data-only Video QA manifest skeleton from normalized context."""
    normalized_run_id = _normalize_run_id(run_id)
    normalized_created_at = _normalize_created_at(created_at)
    source = (
        None
        if bundle.source is None
        else VideoQASourceSnapshot.from_source(bundle.source)
    )
    return VideoQARunManifest(
        schema_version=SCHEMA_VERSION,
        run_id=normalized_run_id,
        created_at=normalized_created_at,
        source=source,
        question=bundle.question,
        attachments=bundle.attachments,
        graph=_build_graph_skeleton(bundle),
        chunks=(),
        status="pending",
        attempts=0,
        error=None,
    )


def _build_graph_skeleton(
    bundle: VideoQAContextBundle,
) -> tuple[VideoQAGraphNode, ...]:
    source_note = (
        "Captures the resolved local source snapshot for future runtime steps."
        if bundle.source is not None
        else "Source remains empty until a local file is selected."
    )
    attachment_note = (
        "Carries normalized attachment metadata into later prompt assembly."
        if bundle.attachments
        else "No attachments are present for this run."
    )
    return (
        VideoQAGraphNode(
            node_id="source.resolve",
            kind="source_resolve",
            note=source_note,
        ),
        VideoQAGraphNode(
            node_id="context.attachments",
            kind="attachment_prepare",
            note=attachment_note,
        ),
        VideoQAGraphNode(
            node_id="transcript.prepare",
            kind="transcript_prepare",
            depends_on=("source.resolve",),
            note="Reserves transcript reuse or build without executing it.",
        ),
        VideoQAGraphNode(
            node_id="chunks.plan",
            kind="chunk_plan",
            depends_on=("source.resolve", "transcript.prepare"),
            note="Reserves chunk planning without selecting boundaries yet.",
        ),
        VideoQAGraphNode(
            node_id="frames.select",
            kind="frame_select",
            depends_on=("chunks.plan",),
            note="Reserves representative frame selection for later waves.",
        ),
        VideoQAGraphNode(
            node_id="qa.run",
            kind="llm_pass",
            depends_on=("context.attachments", "chunks.plan", "frames.select"),
            note="Reserves multimodal chunk analysis without runtime execution.",
        ),
        VideoQAGraphNode(
            node_id="answer.aggregate",
            kind="answer_aggregate",
            depends_on=("qa.run",),
            note="Reserves grounded answer aggregation for later waves.",
        ),
    )


def _normalize_run_id(run_id: str | None) -> str:
    if run_id is None:
        return create_video_qa_run_id()
    normalized = run_id.strip()
    if normalized:
        return normalized
    msg = "Video QA preparation run_id must not be empty."
    raise ValueError(msg)


def _normalize_created_at(created_at: str | None) -> str:
    if created_at is None:
        return create_manifest_timestamp()
    normalized = created_at.strip()
    if not normalized:
        msg = "Video QA preparation created_at must not be empty."
        raise ValueError(msg)
    return validate_manifest_timestamp(normalized)
