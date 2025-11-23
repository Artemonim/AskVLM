from pathlib import Path

import pytest

from core import ffmpeg as ffm
from editing.text_model import Document, TextSegment
from utils import exporters as ex
from utils import logging as logutils


def test_setup_logging_and_get_logger() -> None:
    """setup_logging adds a StreamHandler and sets level; get_logger returns a logger."""
    logutils.setup_logging()
    root = logutils.logging.getLogger()
    assert any(isinstance(h, logutils.logging.StreamHandler) for h in root.handlers)
    lg = logutils.get_logger("x")
    assert lg.name == "x"


def test_ffmpeg_get_media_duration_seconds_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_media_duration_seconds parses float duration from ffprobe result."""
    monkeypatch.setattr(
        ffm.ffmpeg, "probe", lambda _p: {"format": {"duration": "12.5"}}
    )
    assert ffm.get_media_duration_seconds("file") == pytest.approx(12.5)


def test_parse_ts_and_export_document_txt(tmp_path: Path) -> None:
    """_parse_ts_srt_to_seconds parses ms; export_document('txt') omits default speaker."""
    # _parse_ts_srt_to_seconds is used via public exporters; sanity-check indirectly
    assert ex._parse_ts_srt_to_seconds("00:00:00,500") == pytest.approx(0.5)  # type: ignore[attr-defined]  # noqa: SLF001
    assert ex._parse_ts_srt_to_seconds("bad") == 0.0  # type: ignore[attr-defined]  # noqa: SLF001
    doc = Document([TextSegment("speaker_1", 0.0, 0.0, "Hi")])
    out = tmp_path / "a.txt"
    ex.export_document(doc, "txt", out)
    txt = out.read_text(encoding="utf-8")
    assert txt.strip() == "Hi"
    assert "speaker_1:" not in txt
