from pathlib import Path

from editing.text_model import Document, TextSegment
from utils import exporters as ex


def build_doc() -> Document:
    """Return a small `Document` with two short segments for tests."""
    doc = Document()
    doc.add_segment(TextSegment("speaker_1", 0.0, 1.0, "Hello world"))
    doc.add_segment(TextSegment("speaker_2", 1.0, 3.0, "How are you doing today?"))
    return doc


def test_export_txt_and_json(tmp_path: Path) -> None:
    """TXT prefixes speakers when multiple are present; JSON contains segments array."""
    doc = build_doc()
    txt = ex.export_txt(doc)
    assert "speaker_1: Hello world" in txt
    assert "speaker_2:" in txt
    data = ex.export_json(doc)
    assert isinstance(data, dict)
    assert "segments" in data
    out = tmp_path / "out.json"
    ex.export_document(doc, "json", out)
    assert out.exists()
    assert out.read_text(encoding="utf-8").strip().startswith("{")


def test_dialog_blocks_merge_consecutive_segments_for_txt_and_json(
    tmp_path: Path,
) -> None:
    """Dialog Blocks merge adjacent segments of the same speaker for TXT/JSON exports."""
    doc = Document()
    # * Two blocks for speaker_1 and speaker_2 with two segments each
    doc.add_segment(TextSegment("speaker_1", 0.0, 1.0, "Hello"))
    doc.add_segment(TextSegment("speaker_1", 1.0, 2.0, "world"))
    doc.add_segment(TextSegment("speaker_2", 2.0, 3.0, "Second"))
    doc.add_segment(TextSegment("speaker_2", 3.0, 4.0, "block"))
    doc.dialog_blocks_enabled = True

    # * TXT: expect two lines, one per speaker, with merged text per speaker
    txt = ex.export_txt(doc)
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    assert len(lines) == 2
    assert lines[0].startswith("speaker_1:")
    assert "Hello" in lines[0]
    assert "world" in lines[0]
    assert lines[1].startswith("speaker_2:")
    assert "Second" in lines[1]
    assert "block" in lines[1]

    # * JSON: expect two segments with merged timings and text
    data = ex.export_json(doc)
    segs = data.get("segments", [])
    assert isinstance(segs, list)
    assert len(segs) == 2
    s1, s2 = segs
    assert s1["speaker_id"] == "speaker_1"
    assert s1["start_time"] == 0.0
    assert s1["end_time"] == 2.0
    assert "Hello" in str(s1["text"])
    assert "world" in str(s1["text"])
    assert s2["speaker_id"] == "speaker_2"
    assert s2["start_time"] == 2.0
    assert s2["end_time"] == 4.0
    assert "Second" in str(s2["text"])
    assert "block" in str(s2["text"])

    # * Ensure export_document writes merged JSON structure
    out = tmp_path / "dialog_blocks.json"
    ex.export_document(doc, "json", out)
    content = out.read_text(encoding="utf-8")
    assert '"speaker_1"' in content
    assert '"speaker_2"' in content


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
    """AskVLM metadata helpers append, extract and strip correctly."""
    base = "1\n00:00:00,000 --> 00:00:01,000\nHello\n\n"
    with_meta = ex.append_askvlm_metadata_to_srt(
        base, tool_name="AskVLM", quality="fast", completed=True
    )
    assert ex.ASKVLM_META_PREFIX in with_meta
    meta = ex.extract_askvlm_metadata_from_srt(with_meta)
    assert meta
    assert meta.get("tool") == "AskVLM"
    assert meta.get("completed") is True
    stripped = ex.strip_askvlm_metadata_from_srt(with_meta)
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
