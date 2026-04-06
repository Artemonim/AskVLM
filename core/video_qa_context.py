"""Video QA attachment and context normalization helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

from .video_qa_sources import PathInput

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .video_qa_sources import LocalFileSource

AttachmentType = Literal["text", "code", "image"]
AttachmentInput: TypeAlias = PathInput

_TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".env",
    ".htm",
    ".html",
    ".ini",
    ".json",
    ".log",
    ".md",
    ".rst",
    ".toml",
    ".tsv",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}

_CODE_SUFFIX_TO_LANGUAGE = {
    ".c": "c",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".go": "go",
    ".h": "c",
    ".hpp": "cpp",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".kt": "kotlin",
    ".lua": "lua",
    ".m": "objective-c",
    ".php": "php",
    ".py": "python",
    ".pyi": "python",
    ".rb": "ruby",
    ".rs": "rust",
    ".scala": "scala",
    ".ps1": "powershell",
    ".sh": "shell",
    ".sql": "sql",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".swift": "swift",
}

_IMAGE_SUFFIXES = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

_TEXT_ATTACHMENT_INLINE_BYTE_LIMIT = 8 * 1024
_CODE_ATTACHMENT_INLINE_BYTE_LIMIT = 12 * 1024


@dataclass(frozen=True, slots=True)
class VideoQAAttachmentRequest:
    """Input request describing an attachment path and optional overrides."""

    path: AttachmentInput
    enabled: bool = True
    language: str | None = None


@dataclass(frozen=True, slots=True)
class VideoQAAttachment:
    """Normalized attachment metadata for prompt preparation."""

    path: Path | None
    name: str
    type: AttachmentType
    size: int
    suffix: str
    language: str | None
    enabled: bool = True

    @property
    def size_bytes(self) -> int:
        """Return the attachment size in bytes."""
        return self.size

    @property
    def budget_tokens(self) -> int:
        """Return a conservative offline budget estimate for the inline preview."""
        if not self.enabled:
            return 0
        if self.type == "image":
            return _estimate_image_budget_tokens(self.size)
        if self.type == "code":
            budget_size = min(self.size, _CODE_ATTACHMENT_INLINE_BYTE_LIMIT)
            return _estimate_code_budget_tokens(budget_size)
        budget_size = min(self.size, _TEXT_ATTACHMENT_INLINE_BYTE_LIMIT)
        return _estimate_text_budget_tokens(budget_size)

    @property
    def summary(self) -> str:
        """Return a compact human-readable description of the attachment."""
        suffix = self.suffix or "no extension"
        language = f", language={self.language}" if self.language else ""
        state = "enabled" if self.enabled else "disabled"
        return (
            f"{self.name} [{self.type}] {self.size:,} bytes, "
            f"suffix={suffix}{language}, {state}"
        )


@dataclass(frozen=True, slots=True)
class _NormalizedAttachmentRecord:
    """Normalized attachment metadata paired with an inline prompt preview."""

    attachment: VideoQAAttachment
    prompt_text: str | None


@dataclass(frozen=True, slots=True)
class VideoQAContextBundle:
    """Normalized prompt context for a Video QA run."""

    source: LocalFileSource | None
    question: str
    attachments: tuple[VideoQAAttachment, ...]
    attachment_text_previews: tuple[str | None, ...] = ()

    @property
    def enabled_attachments(self) -> tuple[VideoQAAttachment, ...]:
        """Return only the attachments that are enabled for prompt building."""
        return tuple(item for item in self.attachments if item.enabled)

    @property
    def disabled_attachments(self) -> tuple[VideoQAAttachment, ...]:
        """Return only the attachments that are disabled."""
        return tuple(item for item in self.attachments if not item.enabled)

    @property
    def enabled_image_attachments(self) -> tuple[VideoQAAttachment, ...]:
        """Return enabled image attachments that can be sent as multimodal inputs."""
        return tuple(
            item
            for item in self.enabled_attachments
            if item.type == "image" and item.path is not None
        )

    @property
    def enabled_image_attachment_paths(self) -> tuple[Path, ...]:
        """Return resolved paths for enabled image attachments."""
        return tuple(
            item.path
            for item in self.enabled_image_attachments
            if item.path is not None
        )

    @property
    def attachment_budget_tokens(self) -> int:
        """Return the combined token budget of enabled attachments."""
        return sum(item.budget_tokens for item in self.enabled_attachments)

    def render_prompt_block(
        self,
        *,
        chunk_id: str | None = None,
        chunk_time_span: tuple[float, float] | None = None,
        transcript_summary: str | None = None,
        frame_refs: Iterable[str] = (),
    ) -> str:
        """Render a compact prompt block for the current context bundle.

        The block keeps user-provided context first and appends chunk-specific data
        when available, so a future chunk inferencer can reuse the same contract.
        Text and code attachments include bounded inline previews.
        """
        lines: list[str] = []
        lines.extend(_render_source_prompt_lines(self.source))
        if self.question:
            lines.append(f"Question: {self.question}")
        lines.extend(_render_attachment_prompt_lines(self))
        lines.extend(_render_chunk_prompt_lines(chunk_id, chunk_time_span))
        lines.extend(_render_transcript_summary_prompt_lines(transcript_summary))
        lines.extend(_render_frame_prompt_lines(frame_refs))
        return "\n".join(lines)


def normalize_video_qa_attachments(
    attachments: Iterable[AttachmentInput | VideoQAAttachmentRequest],
) -> tuple[VideoQAAttachment, ...]:
    """Normalize raw attachment inputs into immutable metadata records."""
    return tuple(
        _normalize_single_attachment_record(raw, include_preview=False).attachment
        for raw in attachments
    )


def normalize_video_qa_context(
    *,
    source: LocalFileSource | None,
    question: str | None,
    attachments: Iterable[AttachmentInput | VideoQAAttachmentRequest] = (),
) -> VideoQAContextBundle:
    """Normalize the current Video QA source, question, and attachments."""
    normalized_question = str(question or "").strip()
    normalized_records = tuple(
        _normalize_single_attachment_record(raw, include_preview=True)
        for raw in attachments
    )
    return VideoQAContextBundle(
        source=source,
        question=normalized_question,
        attachments=tuple(record.attachment for record in normalized_records),
        attachment_text_previews=tuple(
            record.prompt_text for record in normalized_records
        ),
    )


def _normalize_single_attachment_record(
    raw: AttachmentInput | VideoQAAttachmentRequest,
    *,
    include_preview: bool,
) -> _NormalizedAttachmentRecord:
    """Normalize one attachment input into metadata and optional inline text."""
    path_input: AttachmentInput
    enabled = True
    language_hint: str | None = None
    if isinstance(raw, VideoQAAttachmentRequest):
        path_input = raw.path
        enabled = bool(raw.enabled)
        language_hint = raw.language.strip() if raw.language else None
        if language_hint == "":
            language_hint = None
    else:
        path_input = raw

    text = str(path_input).strip()
    if not text:
        msg = "Attachment path is empty."
        raise ValueError(msg)

    path = Path(text).expanduser()
    if not path.exists():
        msg = f"Attachment not found: {path}"
        raise FileNotFoundError(msg)
    if not path.is_file():
        msg = f"Attachment provider expects a file path: {path}"
        raise IsADirectoryError(msg)

    try:
        stat = path.stat()
    except OSError as exc:
        msg = f"Unable to inspect attachment: {path}"
        raise OSError(msg) from exc

    resolved = path.resolve()
    attachment_type, language = _classify_attachment(resolved.suffix.lower())
    language = (language_hint or language) if attachment_type == "code" else None

    prompt_text: str | None = None
    if include_preview and enabled and attachment_type in {"text", "code"}:
        prompt_text = _read_attachment_preview_text(
            resolved,
            _attachment_inline_byte_limit(attachment_type),
        )

    return _NormalizedAttachmentRecord(
        attachment=VideoQAAttachment(
            path=resolved,
            name=resolved.name,
            type=attachment_type,
            size=int(stat.st_size),
            suffix=resolved.suffix.lower(),
            language=language,
            enabled=enabled,
        ),
        prompt_text=prompt_text,
    )


def _classify_attachment(suffix: str) -> tuple[AttachmentType, str | None]:
    """Classify an attachment file suffix into a normalized attachment type."""
    if suffix in _IMAGE_SUFFIXES:
        return "image", None
    if suffix in _CODE_SUFFIX_TO_LANGUAGE:
        return "code", _CODE_SUFFIX_TO_LANGUAGE[suffix]
    if suffix in _TEXT_SUFFIXES:
        return "text", None
    return "text", None


def _estimate_text_budget_tokens(size_bytes: int) -> int:
    """Estimate token budget for a text attachment."""
    return max(32, math.ceil(max(size_bytes, 1) / 4))


def _estimate_code_budget_tokens(size_bytes: int) -> int:
    """Estimate token budget for a code attachment."""
    return max(48, math.ceil(max(size_bytes, 1) / 3))


def _estimate_image_budget_tokens(size_bytes: int) -> int:
    """Estimate a conservative token budget for an image attachment."""
    kib = max(1, math.ceil(max(size_bytes, 1) / 1024))
    return max(1024, math.ceil(kib * 32 * 1.5))


def _attachment_inline_byte_limit(attachment_type: AttachmentType) -> int:
    """Return the byte cap used for an inline attachment preview."""
    if attachment_type == "code":
        return _CODE_ATTACHMENT_INLINE_BYTE_LIMIT
    if attachment_type == "text":
        return _TEXT_ATTACHMENT_INLINE_BYTE_LIMIT
    return 0


def _read_attachment_preview_text(path: Path, limit_bytes: int) -> str:
    """Read a bounded text preview for prompt inlining."""
    if limit_bytes <= 0:
        return ""
    try:
        with path.open("rb") as handle:
            preview_bytes = handle.read(limit_bytes)
    except OSError as exc:
        msg = f"Unable to read attachment content: {path}"
        raise OSError(msg) from exc
    preview_text = preview_bytes.decode("utf-8", errors="replace")
    return preview_text.replace("\r\n", "\n").replace("\r", "\n")


def _format_attachment_prompt_preview(
    attachment: VideoQAAttachment,
    preview_text: str | None,
) -> list[str]:
    """Return prompt lines for one attachment preview when content is available."""
    if preview_text is None or attachment.type not in {"text", "code"}:
        return []
    limit_bytes = min(attachment.size, _attachment_inline_byte_limit(attachment.type))
    header = (
        f"  Content preview (first {limit_bytes:,} of {attachment.size:,} bytes):"
        if attachment.size > limit_bytes
        else "  Content:"
    )
    content_lines = preview_text.splitlines()
    if not content_lines:
        content_lines = ["(empty)"]
    return [header, *(f"  {line}" for line in content_lines)]


def _render_source_prompt_lines(
    source: LocalFileSource | None,
) -> list[str]:
    """Return prompt lines for the resolved source file."""
    if source is None:
        return []
    return [f"Source: {source.path}", f"Source size: {source.size_bytes:,} bytes"]


def _render_attachment_prompt_lines(
    bundle: VideoQAContextBundle,
) -> list[str]:
    """Return prompt lines for enabled attachments and bounded previews."""
    if not any(item.enabled for item in bundle.attachments):
        return []
    lines: list[str] = ["Attachments:"]
    for index, attachment in enumerate(bundle.attachments):
        if not attachment.enabled:
            continue
        language = f", language={attachment.language}" if attachment.language else ""
        suffix = attachment.suffix or "no extension"
        lines.append(
            "- "
            f"{attachment.name} [{attachment.type}] "
            f"{attachment.size:,} bytes, suffix={suffix}"
            f"{language}, budget≈{attachment.budget_tokens}"
        )
        preview_text = (
            bundle.attachment_text_previews[index]
            if index < len(bundle.attachment_text_previews)
            else None
        )
        lines.extend(_format_attachment_prompt_preview(attachment, preview_text))
    return lines


def _render_chunk_prompt_lines(
    chunk_id: str | None,
    chunk_time_span: tuple[float, float] | None,
) -> list[str]:
    """Return prompt lines for chunk metadata when available."""
    if chunk_id is None and chunk_time_span is None:
        return []
    lines: list[str] = ["Chunk:"]
    if chunk_id is not None:
        lines.append(f"- id: {chunk_id}")
    if chunk_time_span is not None:
        start, end = chunk_time_span
        lines.append(f"- span: {start:.2f}s to {end:.2f}s")
    return lines


def _render_transcript_summary_prompt_lines(
    transcript_summary: str | None,
) -> list[str]:
    """Return prompt lines for the transcript summary when available."""
    normalized_summary = str(transcript_summary or "").strip()
    if not normalized_summary:
        return []
    lines = ["Transcript summary:"]
    lines.extend(f"- {line}" for line in normalized_summary.splitlines())
    return lines


def _render_frame_prompt_lines(frame_refs: Iterable[str]) -> list[str]:
    """Return prompt lines for representative frame references when available."""
    normalized_frames = tuple(
        str(ref).strip() for ref in frame_refs if str(ref).strip()
    )
    if not normalized_frames:
        return []
    lines = ["Representative frames:"]
    lines.extend(f"- {frame_ref}" for frame_ref in normalized_frames)
    return lines
