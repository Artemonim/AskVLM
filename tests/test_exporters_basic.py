from pathlib import Path

from editing.text_model import Document, TextSegment
from utils import exporters as ex


def build_doc() -> Document:
    doc = Document()
    doc.add_segment(TextSegment("speaker_1", 0.0, 1.0, "Hello world"))
    doc.add_segment(TextSegment("speaker_2", 1.0, 3.0, "How are you doing today?"))
    return doc


def test_export_txt_and_json(tmp_path: Path) -> None:
    """TXT concatenates with speakers; JSON contains segments array."""
    doc = build_doc()
    txt = ex.export_txt(doc)
    assert "speaker_1: Hello world" in txt
    data = ex.export_json(doc)
    assert isinstance(data, dict)
    assert "segments" in data
    out = tmp_path / "out.json"
    ex.export_document(doc, "json", out)
    assert out.exists()
    assert out.read_text(encoding="utf-8").strip().startswith("{")


def test_export_srt_and_vtt_formats(tmp_path: Path) -> None:
    """SRT and VTT formats produce time-coded outputs."""
    doc = build_doc()
    srt = ex.export_srt(doc)
    assert "00:00:00,000" in srt
    assert "-->" in srt
    vtt = ex.export_vtt(doc)
    assert vtt.splitlines()[0] == "WEBVTT"
    out_srt = tmp_path / "out.srt"
    ex.export_document(doc, "srt", out_srt)
    assert out_srt.exists()
    assert "-->" in out_srt.read_text(encoding="utf-8")


def test_export_srt_with_rules_splits_long_text() -> None:
    """Long text is split into multiple cues respecting line/lines limits."""
    long_text = " ".join(["word"] * 80)
    doc = Document([TextSegment("speaker_1", 0.0, 0.0, long_text)])
    srt = ex.export_srt_with_rules(doc)
    # Expect multiple indices
    assert "\n2\n" in srt or "\n3\n" in srt


def test_fill_empty_gaps_in_srt_adjusts_end_times() -> None:
    """fill_empty_gaps_in_srt stretches previous cue end to next start time."""
    srt = "1\n00:00:01,000 --> 00:00:02,000\nA\n\n2\n00:00:03,000 --> 00:00:04,000\nB\n"
    fixed = ex.fill_empty_gaps_in_srt(srt)
    assert "00:00:01,000 --> 00:00:03,000" in fixed


def test_metadata_append_extract_and_strip() -> None:
    """ASK metadata helpers append, extract and strip correctly."""
    base = "1\n00:00:00,000 --> 00:00:01,000\nHello\n\n"
    with_meta = ex.append_ask_metadata_to_srt(
        base, tool_name="ASK", quality="fast", completed=True
    )
    meta = ex.extract_ask_metadata_from_srt(with_meta)
    assert meta
    assert meta.get("tool") == "ASK"
    assert meta.get("completed") is True
    stripped = ex.strip_ask_meta_from_srt(with_meta)
    assert stripped.strip().endswith("Hello")


def test_export_document_unknown_format_raises(tmp_path: Path) -> None:
    """export_document raises ValueError for unknown format."""
    doc = build_doc()
    out = tmp_path / "x.bin"
    try:
        ex.export_document(doc, "nope", out)
    except ValueError:
        pass
    else:
        msg = "expected ValueError"
        raise AssertionError(msg)
