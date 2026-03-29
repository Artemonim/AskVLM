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


@dataclass(frozen=True, slots=True)
class VideoQAAttachmentRequest:
    """Input request describing an attachment path and optional overrides."""

    path: AttachmentInput
    enabled: bool = True
    language: str | None = None


@dataclass(frozen=True, slots=True)
class VideoQAAttachment:
    """Normalized attachment metadata for prompt preparation."""

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
        """Return a conservative offline budget estimate for the attachment."""
        if not self.enabled:
            return 0
        if self.type == "image":
            return _estimate_image_budget_tokens(self.size)
        if self.type == "code":
            return _estimate_code_budget_tokens(self.size)
        return _estimate_text_budget_tokens(self.size)

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
class VideoQAContextBundle:
    """Normalized prompt context for a Video QA run."""

    source: LocalFileSource | None
    question: str
    attachments: tuple[VideoQAAttachment, ...]

    @property
    def enabled_attachments(self) -> tuple[VideoQAAttachment, ...]:
        """Return only the attachments that are enabled for prompt building."""
        return tuple(item for item in self.attachments if item.enabled)

    @property
    def disabled_attachments(self) -> tuple[VideoQAAttachment, ...]:
        """Return only the attachments that are disabled."""
        return tuple(item for item in self.attachments if not item.enabled)

    @property
    def attachment_budget_tokens(self) -> int:
        """Return the combined token budget of enabled attachments."""
        return sum(item.budget_tokens for item in self.enabled_attachments)

    def render_prompt_block(self) -> str:
        """Render a compact prompt block for the current context bundle."""
        lines: list[str] = []
        if self.source is not None:
            lines.append(f"Source: {self.source.path}")
            lines.append(f"Source size: {self.source.size_bytes:,} bytes")
        if self.question:
            lines.append(f"Question: {self.question}")
        if self.enabled_attachments:
            lines.append("Attachments:")
            for attachment in self.enabled_attachments:
                language = (
                    f", language={attachment.language}" if attachment.language else ""
                )
                suffix = attachment.suffix or "no extension"
                lines.append(
                    "- "
                    f"{attachment.name} [{attachment.type}] "
                    f"{attachment.size:,} bytes, suffix={suffix}"
                    f"{language}, budget≈{attachment.budget_tokens}"
                )
        return "\n".join(lines)


def normalize_video_qa_attachments(
    attachments: Iterable[AttachmentInput | VideoQAAttachmentRequest],
) -> tuple[VideoQAAttachment, ...]:
    """Normalize raw attachment inputs into immutable metadata records."""
    return tuple(_normalize_single_attachment(raw) for raw in attachments)


def normalize_video_qa_context(
    *,
    source: LocalFileSource | None,
    question: str | None,
    attachments: Iterable[AttachmentInput | VideoQAAttachmentRequest] = (),
) -> VideoQAContextBundle:
    """Normalize the current Video QA source, question, and attachments."""
    normalized_question = str(question or "").strip()
    normalized_attachments = normalize_video_qa_attachments(attachments)
    return VideoQAContextBundle(
        source=source,
        question=normalized_question,
        attachments=normalized_attachments,
    )


def _normalize_single_attachment(
    raw: AttachmentInput | VideoQAAttachmentRequest,
) -> VideoQAAttachment:
    """Normalize one attachment input into a typed metadata record."""
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
    language = language_hint or language if attachment_type == "code" else None

    return VideoQAAttachment(
        name=resolved.name,
        type=attachment_type,
        size=int(stat.st_size),
        suffix=resolved.suffix.lower(),
        language=language,
        enabled=enabled,
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
