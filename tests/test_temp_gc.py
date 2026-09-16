"""Unit tests for AskVLM temporary directory garbage collection."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from core import temp_gc

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _write_file(path: Path, *, data: bytes = b"x", mtime_s: float) -> None:
    """Create a file and set its mtime."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    os.utime(path, (mtime_s, mtime_s))


def _set_mtime(path: Path, *, mtime_s: float) -> None:
    """Set directory mtime."""
    os.utime(path, (mtime_s, mtime_s))


def test_gc_deletes_stale_and_keeps_fresh_by_latest_file_mtime(tmp_path: Path) -> None:
    """A stale candidate is deleted while a fresh candidate is preserved."""
    now = 10_000.0
    stale_dir = tmp_path / "askvlm-cli-old"
    stale_file = stale_dir / "payload.bin"
    stale_dir.mkdir()
    _write_file(stale_file, data=b"stale-data", mtime_s=now - 7200.0)

    fresh_dir = tmp_path / "askvlm-gigaam-new"
    fresh_dir.mkdir()
    _write_file(fresh_dir / "nested" / "old.bin", mtime_s=now - 7200.0)
    _write_file(fresh_dir / "nested" / "new.bin", mtime_s=now - 30.0)

    summary = temp_gc.purge_stale_askvlm_temp_dirs(
        temp_root=tmp_path,
        stale_after_s=3600.0,
        clock=lambda: now,
    )

    assert not stale_dir.exists()
    assert fresh_dir.exists()
    assert summary.deleted_dirs == 1
    assert summary.skipped_dirs == 1
    assert summary.freed_bytes >= len(b"stale-data")


def test_gc_ignores_files_and_non_askvlm_prefixes(tmp_path: Path) -> None:
    """Only known askvlm directory prefixes are considered for deletion."""
    now = 10_000.0
    foreign_dir = tmp_path / "pytest-123"
    foreign_dir.mkdir()
    _set_mtime(foreign_dir, mtime_s=now - 7200.0)

    unknown_askvlm = tmp_path / "askvlm-other-123"
    unknown_askvlm.mkdir()
    _set_mtime(unknown_askvlm, mtime_s=now - 7200.0)

    askvlm_file = tmp_path / "askvlm-daemon-file.txt"
    askvlm_file.write_text("payload", encoding="utf-8")
    os.utime(askvlm_file, (now - 7200.0, now - 7200.0))

    video_work_dir = tmp_path / "_video_qa_work"
    video_work_dir.mkdir()
    _set_mtime(video_work_dir, mtime_s=now - 7200.0)

    summary = temp_gc.purge_stale_askvlm_temp_dirs(
        temp_root=tmp_path,
        stale_after_s=1.0,
        clock=lambda: now,
    )

    assert summary.deleted_dirs == 0
    assert summary.skipped_dirs == 0
    assert foreign_dir.exists()
    assert unknown_askvlm.exists()
    assert askvlm_file.exists()
    assert video_work_dir.exists()


def test_gc_deletes_empty_stale_candidate_by_directory_mtime(tmp_path: Path) -> None:
    """An empty stale askvlm directory is aged by directory mtime and removed."""
    now = 10_000.0
    empty_dir = tmp_path / "askvlm-ext-ipc-empty"
    empty_dir.mkdir()
    _set_mtime(empty_dir, mtime_s=now - 7200.0)

    summary = temp_gc.purge_stale_askvlm_temp_dirs(
        temp_root=tmp_path,
        stale_after_s=3600.0,
        clock=lambda: now,
    )

    assert not empty_dir.exists()
    assert summary.deleted_dirs == 1
    assert summary.skipped_dirs == 0


def test_gc_delete_oserror_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A delete callback error is logged and the function still returns."""
    now = 10_000.0
    denied_dir = tmp_path / "askvlm-daemon-denied"
    denied_dir.mkdir()
    _write_file(denied_dir / "chunk.wav", mtime_s=now - 7200.0)

    def _fake_rmtree(_path: Path, **kwargs: object) -> None:
        assert kwargs.get("ignore_errors") is False
        msg = "denied"
        onexc = kwargs.get("onexc")
        if callable(onexc):
            onexc(None, str(_path), PermissionError(msg))
            return
        onerror = kwargs.get("onerror")
        if callable(onerror):
            onerror(None, str(_path), (PermissionError, PermissionError(msg), None))
            return
        msg = "expected onexc or onerror callback"
        raise AssertionError(msg)

    monkeypatch.setattr(temp_gc.shutil, "rmtree", _fake_rmtree)
    with caplog.at_level(logging.WARNING, logger="core.temp_gc"):
        summary = temp_gc.purge_stale_askvlm_temp_dirs(
            temp_root=tmp_path,
            stale_after_s=3600.0,
            clock=lambda: now,
        )

    assert denied_dir.exists()
    assert summary.deleted_dirs == 0
    assert summary.skipped_dirs == 1
    assert summary.freed_bytes == 0
    assert any("delete failed" in record.getMessage() for record in caplog.records)


def test_gc_uses_directory_mtime_when_walk_is_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreadable walk keeps the directory because age is unknown."""
    now = 10_000.0
    unreadable_dir = tmp_path / "askvlm-gigaam-reencode-old"
    unreadable_dir.mkdir()
    _write_file(unreadable_dir / "reencoded.wav", mtime_s=now - 60.0)
    _set_mtime(unreadable_dir, mtime_s=now - 7200.0)

    def _raise_walk(*_args: object, **_kwargs: object) -> object:
        msg = "walk denied"
        raise OSError(msg)

    monkeypatch.setattr(temp_gc.os, "walk", _raise_walk)
    summary = temp_gc.purge_stale_askvlm_temp_dirs(
        temp_root=tmp_path,
        stale_after_s=3600.0,
        clock=lambda: now,
    )

    assert unreadable_dir.exists()
    assert summary.deleted_dirs == 0
    assert summary.skipped_dirs == 1


def test_gc_keeps_directory_when_fresh_file_is_readable_but_sibling_stat_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful fresh file stat dominates sibling stat failures."""
    now = 10_000.0
    live_dir = tmp_path / "askvlm-gigaam-live"
    live_dir.mkdir()
    _write_file(live_dir / "fresh.wav", mtime_s=now - 30.0)
    _set_mtime(live_dir, mtime_s=now - 7200.0)

    def _walk_with_missing(_directory: Path, onerror: object | None = None) -> object:
        del onerror
        yield (str(live_dir), [], ["fresh.wav", "locked.wav"])

    monkeypatch.setattr(temp_gc.os, "walk", _walk_with_missing)
    summary = temp_gc.purge_stale_askvlm_temp_dirs(
        temp_root=tmp_path,
        stale_after_s=3600.0,
        clock=lambda: now,
    )

    assert live_dir.exists()
    assert summary.deleted_dirs == 0
    assert summary.skipped_dirs == 1


def test_gc_never_raises_on_unexpected_top_level_exception(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Unexpected top-level failures are converted to an empty summary."""

    def _boom() -> str:
        msg = "temp root unavailable"
        raise RuntimeError(msg)

    monkeypatch.setattr(temp_gc.tempfile, "gettempdir", _boom)
    with caplog.at_level(logging.WARNING, logger="core.temp_gc"):
        summary = temp_gc.purge_stale_askvlm_temp_dirs()

    assert summary == temp_gc.TempGcSummary()
    assert any("unexpected failure" in record.getMessage() for record in caplog.records)
