from pathlib import Path

import pytest

from editing.text_model import Document, TextSegment
from utils import exporters as ex


@pytest.fixture
def sample_doc() -> Document:
    """Return a Document with two segments/speakers."""
    return Document(
        [
            TextSegment("spk1", 0.0, 1.5, "Hello world."),
            TextSegment("spk2", 1.5, 3.0, "Hi there."),
        ]
    )


def test_export_json(sample_doc: Document, tmp_path: Path) -> None:
    """export_document('json') produces valid JSON with segments."""
    out = tmp_path / "test.json"
    ex.export_document(sample_doc, "json", out)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert '"text": "Hello world."' in content
    assert '"speaker_id": "spk1"' in content


def test_export_csv(sample_doc: Document, tmp_path: Path) -> None:
    """export_document('csv') produces comma-separated values with header implicit."""
    out = tmp_path / "test.csv"
    ex.export_document(sample_doc, "csv", out)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "spk1,0.0,1.5,Hello world." in content
    assert "spk2,1.5,3.0,Hi there." in content


def test_export_vtt(sample_doc: Document, tmp_path: Path) -> None:
    """export_document('vtt') produces WEBVTT format."""
    out = tmp_path / "test.vtt"
    ex.export_document(sample_doc, "vtt", out)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "WEBVTT" in content
    # VTT exporter omits hour if 0
    assert "00:00.000 --> 00:01.500" in content
    assert "spk1: Hello world." in content


def test_export_srt(sample_doc: Document, tmp_path: Path) -> None:
    """export_document('srt') produces standard SRT format."""
    out = tmp_path / "test.srt"
    ex.export_document(sample_doc, "srt", out)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "1" in content
    assert "00:00:00,000 --> 00:00:01,500" in content
    assert "spk1: Hello world." in content


def test_export_txt(sample_doc: Document, tmp_path: Path) -> None:
    """export_document('txt') produces plain text."""
    out = tmp_path / "test.txt"
    ex.export_document(sample_doc, "txt", out)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    # Multiple speakers triggers prefixes
    assert "spk1: Hello world." in content
    assert "spk2: Hi there." in content


def test_export_unknown_format_raises(sample_doc: Document, tmp_path: Path) -> None:
    """export_document raises ValueError for unsupported formats."""
    with pytest.raises(ValueError, match="Unknown export format"):
        ex.export_document(sample_doc, "xyz", tmp_path / "test.xyz")
