"""Video QA source providers and normalized local source metadata."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Protocol, TypeAlias, runtime_checkable

PathInput: TypeAlias = str | PathLike[str] | Path


@runtime_checkable
class VideoQASourceProvider(Protocol):
    """Contract for resolving a Video QA source into local metadata."""

    name: str

    def resolve(self, raw_source: PathInput) -> LocalFileSource:
        """Resolve a raw source path into normalized local file metadata."""
        ...


@dataclass(frozen=True, slots=True)
class LocalFileSource:
    """Normalized metadata for a local file source."""

    path: Path
    name: str
    size_bytes: int
    suffix: str

    @property
    def summary(self) -> str:
        """Return a concise human-readable description of the source."""
        suffix = self.suffix or "no extension"
        return f"{self.path} | {self.size_bytes:,} bytes | {suffix}"


def _coerce_path(raw_source: PathInput) -> Path:
    """Convert a path-like input into a normalized `Path` object."""
    text = str(raw_source).strip()
    if not text:
        msg = "Local file path is empty."
        raise ValueError(msg)
    return Path(text).expanduser()


class LocalFileProvider:
    """Resolve local files without URL import or remote fetching."""

    name = "LocalFile"

    def resolve(self, raw_source: PathInput) -> LocalFileSource:
        """Resolve `raw_source` into a normalized local file source."""
        path = _coerce_path(raw_source)
        if not path.exists():
            msg = f"Local file not found: {path}"
            raise FileNotFoundError(msg)
        if not path.is_file():
            msg = f"Local file provider expects a file path: {path}"
            raise IsADirectoryError(msg)

        try:
            stat = path.stat()
        except OSError as exc:
            msg = f"Unable to inspect local file: {path}"
            raise OSError(msg) from exc

        resolved = path.resolve()
        return LocalFileSource(
            path=resolved,
            name=resolved.name,
            size_bytes=int(stat.st_size),
            suffix=resolved.suffix.lower(),
        )
