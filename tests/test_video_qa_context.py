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
    script = tmp_path / "script.py"
    script.write_text("def greet():\n    return 'hi'\n", encoding="utf-8")
    image = tmp_path / "frame.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    bundle = normalize_video_qa_context(
        source=source,
        question="  What is in the clip?  ",
        attachments=[
            notes,
            VideoQAAttachmentRequest(path=snippet, enabled=False),
            script,
            image,
        ],
    )

    assert bundle.source == source
    assert bundle.question == "What is in the clip?"
    assert [item.name for item in bundle.attachments] == [
        "notes.md",
        "snippet.py",
        "script.py",
        "frame.png",
    ]
    assert [item.type for item in bundle.attachments] == [
        "text",
        "code",
        "code",
        "image",
    ]
    assert [item.enabled for item in bundle.attachments] == [
        True,
        False,
        True,
        True,
    ]
    assert bundle.attachments[1].language == "python"
    assert bundle.attachments[1].budget_tokens == 0
    assert bundle.attachments[2].language == "python"
    assert bundle.attachments[2].budget_tokens >= 48
    assert bundle.attachments[3].budget_tokens >= 1024
    assert bundle.attachments[0].path == notes.resolve()
    assert bundle.attachments[3].path == image.resolve()
    assert bundle.attachment_text_previews == (
        "one two three",
        None,
        "def greet():\n    return 'hi'\n",
        None,
    )
    assert bundle.attachment_budget_tokens == (
        bundle.attachments[0].budget_tokens
        + bundle.attachments[2].budget_tokens
        + bundle.attachments[3].budget_tokens
    )
    assert bundle.enabled_image_attachment_paths == (image.resolve(),)

    prompt_block = bundle.render_prompt_block()
    assert "Source:" in prompt_block
    assert "Question: What is in the clip?" in prompt_block
    assert "Attachments:" in prompt_block
    assert "one two three" in prompt_block
    assert "def greet()" in prompt_block
    assert "print('hi')" not in prompt_block


def test_video_qa_context_render_prompt_block_includes_chunk_sections(
    tmp_path: Path,
) -> None:
    """Prompt blocks include chunk metadata, transcript summary, and frame refs."""
    source_media = tmp_path / "clip.mp4"
    source_media.write_bytes(b"abc")
    source = LocalFileProvider().resolve(source_media)

    attachment = tmp_path / "notes.txt"
    attachment.write_text("hello", encoding="utf-8")

    bundle = normalize_video_qa_context(
        source=source,
        question="What is shown?",
        attachments=[attachment],
    )

    prompt_block = bundle.render_prompt_block(
        chunk_id="chunk-01",
        chunk_time_span=(12.5, 18.0),
        transcript_summary="The speaker points at a diagram.\nThey highlight the left side.",
        frame_refs=("frame-001.png", " frame-002.png "),
    )

    assert "Chunk:" in prompt_block
    assert "- id: chunk-01" in prompt_block
    assert "- span: 12.50s to 18.00s" in prompt_block
    assert "Transcript summary:" in prompt_block
    assert "- The speaker points at a diagram." in prompt_block
    assert "- They highlight the left side." in prompt_block
    assert "Representative frames:" in prompt_block
    assert "- frame-001.png" in prompt_block
    assert "- frame-002.png" in prompt_block
    assert "hello" in prompt_block


def test_video_qa_context_truncates_large_text_attachment_preview(
    tmp_path: Path,
) -> None:
    """Large text attachments are previewed with an explicit prompt limit."""
    source_media = tmp_path / "clip.mp4"
    source_media.write_bytes(b"abc")
    source = LocalFileProvider().resolve(source_media)

    large_text = "0123456789abcdef\n" * 1024
    attachment = tmp_path / "large.txt"
    attachment.write_text(large_text, encoding="utf-8")

    bundle = normalize_video_qa_context(
        source=source,
        question="What is shown?",
        attachments=[attachment],
    )

    preview = bundle.attachment_text_previews[0]
    assert preview is not None
    assert preview.startswith("0123456789abcdef")
    assert len(preview) < len(large_text)

    prompt_block = bundle.render_prompt_block()
    assert "Content preview (" in prompt_block
    assert "0123456789abcdef" in prompt_block
