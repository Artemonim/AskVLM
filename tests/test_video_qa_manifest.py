from __future__ import annotations

import json
from typing import TYPE_CHECKING

from core.video_qa_context import normalize_video_qa_context
from core.video_qa_manifest import (
    SCHEMA_VERSION,
    VideoQAChunkRecord,
    VideoQAGraphNode,
    VideoQARunManifest,
    VideoQASourceSnapshot,
)
from core.video_qa_sources import LocalFileProvider

if TYPE_CHECKING:
    from pathlib import Path


def test_video_qa_manifest_round_trip_preserves_schema_objects(
    tmp_path: Path,
) -> None:
    """Manifest JSON round-trip preserves the versioned schema payload."""
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"video")
    source = LocalFileProvider().resolve(media)

    notes = tmp_path / "notes.md"
    notes.write_text("speaker enters the room", encoding="utf-8")
    bundle = normalize_video_qa_context(
        source=source,
        question="What happens in the clip?",
        attachments=[notes],
    )

    manifest = VideoQARunManifest(
        schema_version=SCHEMA_VERSION,
        run_id="run-001",
        created_at="2026-03-29T12:00:00Z",
        source=VideoQASourceSnapshot.from_source(source),
        question=bundle.question,
        attachments=bundle.attachments,
        graph=(
            VideoQAGraphNode(
                node_id="source.resolve",
                kind="source_resolve",
                status="completed",
                note="Source metadata is available.",
            ),
            VideoQAGraphNode(
                node_id="chunks.plan",
                kind="chunk_plan",
                depends_on=("source.resolve",),
                status="running",
                note="Chunk planning is queued for a later wave.",
            ),
        ),
        chunks=(
            VideoQAChunkRecord(
                chunk_id="chunk-001",
                t_start=0.0,
                t_end=12.5,
                frames=("frames/0001.png",),
                artifacts=("artifacts/chunk-001.json",),
                status="pending",
                attempts=0,
                error=None,
            ),
        ),
        status="running",
        attempts=1,
        error=None,
    )

    payload = json.loads(json.dumps(manifest.to_dict()))
    restored = VideoQARunManifest.from_dict(payload)

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["chunks"][0]["chunk_id"] == "chunk-001"
    assert restored == manifest
