"""Separate URL import stage: remote URL to staged local file for Video QA."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from urllib.parse import urlparse
from uuid import uuid4

from .video_qa_sources import LocalFileProvider

if TYPE_CHECKING:
    from .video_qa_policy import VideoQAUrlImportPolicy
    from .video_qa_sources import LocalFileSource

# * Experimental: only YouTube hosts are admitted by this backend cluster.
_YOUTUBE_HOST_EXACT: frozenset[str] = frozenset(
    {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtu.be",
        "www.youtube-nocookie.com",
    }
)


class VideoQAUrlImportError(RuntimeError):
    """Raised when a URL cannot be imported under the current policy or adapters."""


@runtime_checkable
class VideoUrlDownloader(Protocol):
    """Downloads remote media into a destination directory and returns the file path."""

    def download(self, url: str, destination_dir: Path) -> Path:
        """Write media under ``destination_dir`` and return the primary media file path."""
        ...


@dataclass(frozen=True, slots=True)
class UrlImportStagingHandle:
    """Tracks ephemeral staging state for cleanup after ingestion."""

    staging_root: Path
    """Directory created for this import; safe to remove entirely when done."""

    def cleanup(self) -> None:
        """Remove staged files and directories for this import."""
        if self.staging_root.exists():
            shutil.rmtree(self.staging_root, ignore_errors=False)


@dataclass(frozen=True, slots=True)
class UrlImportResult:
    """Normalized local source plus metadata for lifecycle and downstream QA."""

    local_source: LocalFileSource
    original_url: str
    provider_id: str
    handle: UrlImportStagingHandle

    def cleanup_staging(self) -> None:
        """Remove staged artifacts according to policy (ephemeral staging)."""
        self.handle.cleanup()


def is_youtube_http_url(url: str) -> bool:
    """Return True when ``url`` targets an experimental YouTube host."""
    cleaned = str(url).strip()
    if not cleaned:
        return False
    parsed = urlparse(cleaned)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        return False
    host = _split_host(parsed.netloc)
    if not host:
        return False
    if host in _YOUTUBE_HOST_EXACT:
        return True
    return host.endswith(".youtube.com") and host != "youtube.com"


def _split_host(netloc: str) -> str:
    """Return lowercased hostname without port."""
    if not netloc:
        return ""
    return netloc.lower().split(":")[0]


def _require_policy_allows(policy: VideoQAUrlImportPolicy, raw_url: str) -> None:
    allowed, reason = policy.check(raw_url)
    if not allowed:
        msg = f"URL import rejected by policy: {reason}"
        raise VideoQAUrlImportError(msg)


def _require_youtube_experimental(url: str) -> None:
    if is_youtube_http_url(url):
        return
    msg = (
        "Only experimental YouTube HTTP(S) URLs are supported by this import stage; "
        "other hosts are not enabled yet."
    )
    raise VideoQAUrlImportError(msg)


class VideoQAUrlImportProvider:
    """Provider stage: validate policy, route to YouTube, stage file, return local metadata."""

    name = "UrlImport"

    def __init__(
        self,
        policy: VideoQAUrlImportPolicy,
        downloader: VideoUrlDownloader,
        *,
        staging_parent: Path,
    ) -> None:
        """Build a URL import provider with policy, downloader, and staging parent directory."""
        self._policy = policy
        self._downloader = downloader
        self._staging_parent = staging_parent

    def import_url(self, raw_url: str) -> UrlImportResult:
        """Validate URL, download into a fresh staging directory, return normalized source."""
        cleaned = str(raw_url).strip()
        _require_policy_allows(self._policy, cleaned)
        _require_youtube_experimental(cleaned)

        self._staging_parent.mkdir(parents=True, exist_ok=True)
        staging_root = self._staging_parent / f"video_qa_url_import_{uuid4().hex}"
        staging_root.mkdir(parents=False)

        try:
            media_path = self._downloader.download(cleaned, staging_root)
        except VideoQAUrlImportError:
            shutil.rmtree(staging_root, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(staging_root, ignore_errors=True)
            msg = f"Download failed: {exc}"
            raise VideoQAUrlImportError(msg) from exc

        resolved_media = Path(media_path).resolve()
        staging_resolved = staging_root.resolve()
        try:
            resolved_media.relative_to(staging_resolved)
        except ValueError as exc:
            shutil.rmtree(staging_root, ignore_errors=True)
            msg = f"Downloader path must stay under staging dir: {resolved_media}"
            raise VideoQAUrlImportError(msg) from exc

        if not resolved_media.is_file():
            shutil.rmtree(staging_root, ignore_errors=True)
            msg = f"Downloader did not produce a file: {resolved_media}"
            raise VideoQAUrlImportError(msg)

        try:
            local = LocalFileProvider().resolve(resolved_media)
        except OSError as exc:
            shutil.rmtree(staging_root, ignore_errors=True)
            msg = f"Unable to normalize staged file: {resolved_media}"
            raise VideoQAUrlImportError(msg) from exc

        handle = UrlImportStagingHandle(staging_root=staging_root.resolve())
        return UrlImportResult(
            local_source=local,
            original_url=cleaned,
            provider_id="youtube",
            handle=handle,
        )


# * Prefer these suffixes when multiple `video_qa_import.*` files appear (e.g. sidecars).
_YTDLP_VIDEO_SUFFIXES: frozenset[str] = frozenset(
    {
        ".mp4",
        ".webm",
        ".mkv",
        ".mov",
        ".m4v",
        ".avi",
        ".flv",
    }
)


class YtDlpCliDownloader:
    """Optional downloader that shells out to the ``yt-dlp`` CLI (user-installed)."""

    def __init__(self, *, extra_args: tuple[str, ...] = ()) -> None:
        """Configure optional extra arguments forwarded to ``yt-dlp``."""
        self._extra_args = extra_args

    def download(self, url: str, destination_dir: Path) -> Path:
        """Download best video+audio or best single stream into ``destination_dir``."""
        dest = destination_dir.resolve()
        dest.mkdir(parents=True, exist_ok=True)
        output_template = str(dest / "video_qa_import.%(ext)s")
        cmd = [
            sys.executable,
            "-m",
            "yt_dlp",
            "-o",
            output_template,
            "--no-playlist",
            "--no-write-info-json",
            *self._extra_args,
            url,
        ]
        try:
            proc = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=86_400,
            )
        except FileNotFoundError as exc:
            msg = (
                "yt-dlp is not available: Python cannot run `python -m yt_dlp`. "
                "Install with: pip install yt-dlp"
            )
            raise VideoQAUrlImportError(msg) from exc
        except subprocess.TimeoutExpired as exc:
            msg = "yt-dlp download timed out."
            raise VideoQAUrlImportError(msg) from exc

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            hint = stderr or (proc.stdout or "").strip() or "no output"
            if "No module named yt_dlp" in hint or "No module named 'yt_dlp'" in hint:
                msg = "yt-dlp is not installed. Install the optional package with: pip install yt-dlp"
                raise VideoQAUrlImportError(msg)
            msg = f"yt-dlp failed (exit {proc.returncode}): {hint}"
            raise VideoQAUrlImportError(msg)

        staged = [p for p in dest.glob("video_qa_import.*") if p.is_file()]
        if not staged:
            msg = f"yt-dlp reported success but no media file was written under {dest}"
            raise VideoQAUrlImportError(msg)
        video_files = [p for p in staged if p.suffix.lower() in _YTDLP_VIDEO_SUFFIXES]
        pick_from = video_files if video_files else staged
        chosen = max(pick_from, key=lambda p: p.stat().st_size)
        return chosen.resolve()


def describe_url_import_temp_policy(policy: VideoQAUrlImportPolicy) -> str:
    """Return a human-readable description of temp file handling for URL import."""
    return policy.temp_file_policy_description()
