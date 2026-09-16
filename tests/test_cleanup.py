"""Unit tests for cleanup utility temp handling."""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

import pytest

from utils import cleanup

if TYPE_CHECKING:
    from pathlib import Path


ASKVLM_PREFIXES: tuple[str, ...] = (
    "askvlm-cli-",
    "askvlm-ext-ipc-",
    "askvlm-daemon-",
    "askvlm-gigaam-",
    "askvlm-gigaam-reencode-",
)


def _write_file(path: Path, *, data: bytes, mtime_s: float) -> None:
    """Create file payload and set its mtime."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    os.utime(path, (mtime_s, mtime_s))


@pytest.mark.parametrize("prefix", ASKVLM_PREFIXES)
def test_clean_temp_files_purges_stale_askvlm_dirs_per_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefix: str
) -> None:
    """Deep cleanup removes only stale AskVLM directories for each prefix."""
    now = time.time()
    stale_payload = b"stale-bytes"
    stale_dir = tmp_path / f"{prefix}stale"
    stale_dir.mkdir()
    _write_file(stale_dir / "payload.bin", data=stale_payload, mtime_s=now - 7200.0)

    fresh_dir = tmp_path / f"{prefix}fresh"
    fresh_dir.mkdir()
    _write_file(fresh_dir / "payload.bin", data=b"fresh", mtime_s=now - 60.0)

    foreign_dir = tmp_path / "askvlm-other-stale"
    foreign_dir.mkdir()
    _write_file(foreign_dir / "payload.bin", data=b"other", mtime_s=now - 7200.0)

    monkeypatch.setattr(cleanup.tempfile, "gettempdir", lambda: str(tmp_path))
    freed_mb = cleanup.clean_temp_files()

    assert not stale_dir.exists()
    assert fresh_dir.exists()
    assert foreign_dir.exists()
    assert freed_mb >= len(stale_payload) / (1024 * 1024)


def test_clean_temp_files_survives_askvlm_gc_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Cleanup keeps exit contract when AskVLM temp GC crashes."""

    def _raise_gc(*_args: object, **_kwargs: object) -> object:
        msg = "gc failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(cleanup.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(cleanup, "purge_stale_askvlm_temp_dirs", _raise_gc)
    with caplog.at_level(logging.WARNING, logger="utils.cleanup"):
        freed_mb = cleanup.clean_temp_files()

    assert freed_mb == 0.0
    assert any(
        "cleanup askvlm temp GC failed" in record.getMessage()
        for record in caplog.records
    )
