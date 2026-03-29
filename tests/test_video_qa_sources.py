from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from core.video_qa_sources import LocalFileProvider

if TYPE_CHECKING:
    from pathlib import Path


def test_local_file_provider_resolves_local_file(tmp_path: Path) -> None:
    """LocalFile provider resolves an existing file and rejects invalid inputs."""
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"abc")

    provider = LocalFileProvider()
    source = provider.resolve(media)

    assert source.path == media.resolve()
    assert source.name == "clip.mp4"
    assert source.size_bytes == 3
    assert source.suffix == ".mp4"
    assert source.summary.endswith(".mp4")

    with pytest.raises(FileNotFoundError):
        provider.resolve(tmp_path / "missing.mp4")

    with pytest.raises(IsADirectoryError):
        provider.resolve(tmp_path)
