from __future__ import annotations

from PySide6.QtWidgets import QApplication

from editing.text_model import Document, TextSegment
from utils.exporters import (
    SubtitleRules,
    export_srt_with_rules,
    fill_empty_gaps_in_srt,
)


def _ensure_qapp() -> None:
    QApplication.instance() or QApplication([])


def test_fill_empty_gaps_basic() -> None:
    _ensure_qapp()
    # Create three cues with a gap between 1.0..2.0 and 3.0..4.0
    d = Document()
    d.add_segment(TextSegment("speaker_1", 0.0, 1.0, "A"))
    d.add_segment(TextSegment("speaker_1", 2.0, 3.0, "B"))
    d.add_segment(TextSegment("speaker_1", 4.0, 5.0, "C"))
    srt = export_srt_with_rules(
        d, SubtitleRules(max_line_chars=50, max_lines=2, min_duration=0.5)
    )
    filled = fill_empty_gaps_in_srt(srt)
    # Ensure end of first cue equals start of second cue (1.0 -> 2.0)
    assert "00:00:00,000 --> 00:00:02,000" in filled
    # Ensure end of second cue equals start of third cue (3.0 -> 4.0)
    assert "00:00:02,000 --> 00:00:04,000" in filled


def test_fill_empty_gaps_ignores_meta_json_cue() -> None:
    _ensure_qapp()
    # SRT with two cues and a trailing JSON meta line; ensure function does not break
    srt = (
        "1\n00:00:00,000 --> 00:00:01,000\nA\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\nB\n\n"
        '{"tool":"ASK"}\n'
    )
    filled = fill_empty_gaps_in_srt(srt)
    # Gap should be filled (1.0 -> 2.0)
    assert "00:00:00,000 --> 00:00:02,000" in filled
    # Meta JSON line should remain present (function only adjusts cue timings)
    assert '{"tool":"ASK"}' in filled.replace(" ", "")
