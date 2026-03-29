from __future__ import annotations

from typing import TYPE_CHECKING

from core.video_qa_context import (
    VideoQAAttachmentRequest,
    normalize_video_qa_context,
)
from core.video_qa_sources import LocalFileProvider

if TYPE_CHECKING:
    from pathlib import Path


def test_video_qa_context_normalizes_attachments_and_question(
    tmp_path: Path,
) -> None:
    """Context normalization preserves metadata and applies conservative budgets."""
    source_media = tmp_path / "clip.mp4"
    source_media.write_bytes(b"abc")
    source = LocalFileProvider().resolve(source_media)

    notes = tmp_path / "notes.md"
    notes.write_text("one two three", encoding="utf-8")
    snippet = tmp_path / "snippet.py"
    snippet.write_text("print('hi')", encoding="utf-8")
    image = tmp_path / "frame.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    bundle = normalize_video_qa_context(
        source=source,
        question="  What is in the clip?  ",
        attachments=[
            notes,
            VideoQAAttachmentRequest(path=snippet, enabled=False),
            image,
        ],
    )

    assert bundle.source == source
    assert bundle.question == "What is in the clip?"
    assert [item.name for item in bundle.attachments] == [
        "notes.md",
        "snippet.py",
        "frame.png",
    ]
    assert [item.type for item in bundle.attachments] == [
        "text",
        "code",
        "image",
    ]
    assert [item.enabled for item in bundle.attachments] == [True, False, True]
    assert bundle.attachments[1].language == "python"
    assert bundle.attachments[1].budget_tokens == 0
    assert bundle.attachments[2].budget_tokens >= 1024
    assert bundle.attachment_budget_tokens == (
        bundle.attachments[0].budget_tokens + bundle.attachments[2].budget_tokens
    )

    prompt_block = bundle.render_prompt_block()
    assert "Source:" in prompt_block
    assert "Question: What is in the clip?" in prompt_block
    assert "Attachments:" in prompt_block
