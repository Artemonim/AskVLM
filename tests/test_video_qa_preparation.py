from __future__ import annotations

from typing import TYPE_CHECKING

from core.video_qa_context import normalize_video_qa_context
from core.video_qa_manifest import SCHEMA_VERSION, VideoQASourceSnapshot
from core.video_qa_preparation import build_video_qa_preparation_manifest
from core.video_qa_sources import LocalFileProvider

if TYPE_CHECKING:
    from pathlib import Path


def test_build_video_qa_preparation_manifest_returns_graph_skeleton(
    tmp_path: Path,
) -> None:
    """Preparation builds a data-only graph skeleton without chunk records."""
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"video")
    source = LocalFileProvider().resolve(media)

    notes = tmp_path / "notes.md"
    notes.write_text("speaker introduces the topic", encoding="utf-8")
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"\x89PNG\r\n\x1a\n")

    bundle = normalize_video_qa_context(
        source=source,
        question="Summarize the scene.",
        attachments=[notes, frame],
    )

    manifest = build_video_qa_preparation_manifest(
        bundle,
        run_id="prep-001",
        created_at="2026-03-29T12:00:00Z",
    )

    assert manifest.schema_version == SCHEMA_VERSION
    assert manifest.run_id == "prep-001"
    assert manifest.created_at == "2026-03-29T12:00:00Z"
    assert manifest.source == VideoQASourceSnapshot.from_source(source)
    assert manifest.question == "Summarize the scene."
    assert manifest.attachments == bundle.attachments
    assert manifest.chunks == ()
    assert manifest.status == "pending"
    assert manifest.attempts == 0
    assert [node.node_id for node in manifest.graph] == [
        "source.resolve",
        "context.attachments",
        "transcript.prepare",
        "chunks.plan",
        "frames.select",
        "qa.run",
        "answer.aggregate",
    ]
    assert [node.depends_on for node in manifest.graph] == [
        (),
        (),
        ("source.resolve",),
        ("source.resolve", "transcript.prepare"),
        ("chunks.plan",),
        ("context.attachments", "chunks.plan", "frames.select"),
        ("qa.run",),
    ]
    assert all(node.status == "pending" for node in manifest.graph)
