from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from core.video_qa_context import normalize_video_qa_context
from core.video_qa_policy import VideoQAUrlImportPolicy
from core.video_qa_url_import import (
    VideoQAUrlImportError,
    VideoQAUrlImportProvider,
    YtDlpCliDownloader,
    describe_url_import_temp_policy,
    is_youtube_http_url,
)


def test_youtube_urls_accepted_by_host_rules() -> None:
    """Experimental YouTube hosts are recognized for the URL import stage."""
    assert is_youtube_http_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ") is True
    assert is_youtube_http_url("https://youtu.be/dQw4w9WgXcQ") is True
    assert is_youtube_http_url("https://m.youtube.com/watch?v=abc") is True


def test_non_youtube_or_bad_urls_rejected_for_experimental_provider() -> None:
    """Non-YouTube and malformed URLs are not admitted by the experimental adapter."""
    assert is_youtube_http_url("https://example.com/video.mp4") is False
    assert is_youtube_http_url("ftp://youtube.com/foo") is False
    assert is_youtube_http_url("") is False

    policy = VideoQAUrlImportPolicy(enabled=True)
    provider = VideoQAUrlImportProvider(
        policy=policy,
        downloader=_FakeDownloader(),
        staging_parent=Path("stub-staging-not-used"),
    )
    with pytest.raises(VideoQAUrlImportError, match="Only experimental YouTube"):
        provider.import_url("https://vimeo.com/123")


def test_policy_blocks_when_disabled_or_scheme_invalid() -> None:
    """Policy gate runs before host routing."""
    policy_off = VideoQAUrlImportPolicy(enabled=False)
    provider_off = VideoQAUrlImportProvider(
        policy=policy_off,
        downloader=_FakeDownloader(),
        staging_parent=Path("stub-staging-not-used"),
    )
    with pytest.raises(VideoQAUrlImportError, match="policy"):
        provider_off.import_url("https://www.youtube.com/watch?v=1")

    policy_on = VideoQAUrlImportPolicy(enabled=True)
    provider_on = VideoQAUrlImportProvider(
        policy=policy_on,
        downloader=_FakeDownloader(),
        staging_parent=Path("stub-staging-not-used"),
    )
    with pytest.raises(VideoQAUrlImportError, match="policy"):
        provider_on.import_url("ftp://youtube.com/foo")


def test_provider_stages_with_fake_downloader(
    tmp_path: Path,
) -> None:
    """A downloader produces a file under staging; result matches LocalFileProvider shape."""
    staging_parent = tmp_path / "stage_parent"
    policy = VideoQAUrlImportPolicy(enabled=True)
    provider = VideoQAUrlImportProvider(
        policy=policy,
        downloader=_FakeDownloader(),
        staging_parent=staging_parent,
    )

    result = provider.import_url("https://www.youtube.com/watch?v=test")
    assert result.provider_id == "youtube"
    assert result.local_source.name == "staged.bin"
    assert result.local_source.suffix == ".bin"
    assert result.local_source.size_bytes == 4
    assert result.local_source.path.is_file()
    assert "youtube.com" in result.original_url

    bundle = normalize_video_qa_context(
        source=result.local_source,
        question="q",
        attachments=(),
    )
    assert bundle.source is not None
    assert bundle.source.path == result.local_source.path


def test_cleanup_removes_staging_directory(tmp_path: Path) -> None:
    """Ephemeral staging root is removed entirely on cleanup."""
    staging_parent = tmp_path / "stage_parent"
    policy = VideoQAUrlImportPolicy(enabled=True)
    provider = VideoQAUrlImportProvider(
        policy=policy,
        downloader=_FakeDownloader(),
        staging_parent=staging_parent,
    )
    result = provider.import_url("https://youtu.be/abc123")
    staging_root = result.handle.staging_root
    assert staging_root.exists()
    assert result.local_source.path.exists()

    result.cleanup_staging()

    assert not staging_root.exists()
    assert not result.local_source.path.exists()


def test_downloader_outside_staging_rejected(tmp_path: Path) -> None:
    """Paths outside the staging directory are rejected."""
    staging_parent = tmp_path / "stage_parent"
    policy = VideoQAUrlImportPolicy(enabled=True)
    evil = tmp_path / "outside.mp4"
    evil.write_bytes(b"bad")

    class EvilDownloader:
        def download(self, _url: str, _destination_dir: Path) -> Path:
            return evil

    provider = VideoQAUrlImportProvider(
        policy=policy,
        downloader=EvilDownloader(),
        staging_parent=staging_parent,
    )
    with pytest.raises(VideoQAUrlImportError, match="staging dir"):
        provider.import_url("https://www.youtube.com/watch?v=x")


def test_temp_policy_description_wired() -> None:
    """Temp policy text is exposed for logs and UI strings."""
    text = describe_url_import_temp_policy(VideoQAUrlImportPolicy())
    assert "temporary" in text.lower() or "staged" in text.lower()


def test_ytdlp_downloader_missing_module_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """yt-dlp backend fails with a clear message when the module is absent."""

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="No module named yt_dlp",
        )

    monkeypatch.setattr("core.video_qa_url_import.subprocess.run", fake_run)
    dl = YtDlpCliDownloader()
    with pytest.raises(VideoQAUrlImportError, match="yt-dlp is not installed"):
        dl.download("https://www.youtube.com/watch?v=1", tmp_path)


class _FakeDownloader:
    def download(self, _url: str, destination_dir: Path) -> Path:
        destination_dir.mkdir(parents=True, exist_ok=True)
        target = destination_dir / "staged.bin"
        target.write_bytes(b"VIDE")
        return target
