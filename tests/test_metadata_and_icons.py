from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QApplication, QTableWidgetItem

from editing.text_model import Document, TextSegment
from gui.main_window import MainWindow, PipelineWorker
from utils.exporters import (
    SubtitleRules,
    append_ask_metadata_to_srt,
    export_srt_with_rules,
    extract_ask_metadata_from_srt,
    strip_ask_meta_from_srt,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_srt_metadata_round_trip() -> None:
    """Append/extract ASK metadata round trip is consistent."""
    # Ensure Qt exists for potential GUI interactions later
    QApplication.instance() or QApplication([])

    doc = Document()
    doc.add_segment(TextSegment("speaker_1", 0.0, 1.0, "hello world"))
    srt = export_srt_with_rules(doc, SubtitleRules(max_line_chars=20, max_lines=2))
    with_meta = append_ask_metadata_to_srt(
        srt,
        tool_name="Artemonim's Speech Kit",
        quality="fast",
        completed=True,
    )
    meta = extract_ask_metadata_from_srt(with_meta)
    assert isinstance(meta, dict)
    assert meta.get("tool") == "Artemonim's Speech Kit"
    assert meta.get("quality") == "fast"
    assert meta.get("completed") is True


def test_input_status_scan_sets_icons(tmp_path: Path) -> None:
    """Scanning output dir sets expected per-row statuses/icons."""
    # Create a minimal Qt app
    QApplication.instance() or QApplication([])

    # Prepare fake inputs in the GUI list
    w = MainWindow()
    # Point output dir to temp
    w.out_dir_edit.setText(str(tmp_path))

    # Create three dummy inputs with different statuses by stem
    fast_media = tmp_path / "fast_input.mp4"
    good_media = tmp_path / "good_input.mp4"
    burn_media = tmp_path / "burn_input.mp4"
    # Touch media files (only paths are needed for table rows)
    for p in (fast_media, good_media, burn_media):
        p.write_bytes(b"")

    # Insert rows into Input table (column 1 = path)
    for p in (fast_media, good_media, burn_media):
        row = w.input_list.rowCount()
        w.input_list.insertRow(row)
        w.input_list.setItem(row, 1, QTableWidgetItem(str(p)))

    # Write SRT files with metadata: fast and good
    def write_srt(stem: str, quality: str) -> None:
        d = Document()
        d.add_segment(TextSegment("speaker_1", 0.0, 1.0, f"{stem}"))
        txt = export_srt_with_rules(d, SubtitleRules())
        txt = append_ask_metadata_to_srt(
            txt, tool_name="Artemonim's Speech Kit", quality=quality, completed=True
        )
        (tmp_path / f"{stem}.srt").write_text(txt, encoding="utf-8")

    write_srt("fast_input", "fast")
    write_srt("good_input", "good")
    # Burned output marker
    (tmp_path / "burn_input_subbed.mp4").write_bytes(b"")

    # Trigger scan
    w._scan_output_statuses()  # noqa: SLF001

    # Verify internal statuses
    assert w._get_input_status(fast_media) == "fast"  # noqa: SLF001
    assert w._get_input_status(good_media) == "good"  # noqa: SLF001
    assert w._get_input_status(burn_media) == "burned"  # noqa: SLF001


def test_pipeline_appends_ask_metadata_and_scan(tmp_path: Path) -> None:
    """Pipeline exports SRT with ASK meta and scanner detects quality."""
    # Ensure Qt exists
    QApplication.instance() or QApplication([])

    media = tmp_path / "pipe_input.mp4"
    media.write_bytes(b"")

    class StubPipeline:
        enable_diarization: bool = False
        enable_dialog_blocks: bool = False

        def process(self, _inp: Path, _out: Path, **_kwargs: object) -> Document:  # type: ignore[override]
            d = Document()
            d.add_segment(TextSegment("speaker_1", 0.0, 1.0, "stub text"))
            return d

    # Run worker synchronously to export SRT
    worker = PipelineWorker(
        pipeline=StubPipeline(),
        inputs=[media],
        out_dir=tmp_path,
        options={
            "export_format": "srt",
            "single_view": False,
            "save_srt": True,
            "subtitle_max_line_width": 42,
            "subtitle_max_lines": 2,
            "quality": "fast",
        },
    )
    worker.run()

    srt_path = tmp_path / "pipe_input.srt"
    assert srt_path.exists()
    meta = extract_ask_metadata_from_srt(srt_path.read_text(encoding="utf-8"))
    assert isinstance(meta, dict)
    assert meta.get("tool") == "Artemonim's Speech Kit"
    assert meta.get("quality") == "fast"
    assert meta.get("completed") is True

    # Scanner should pick it up as fast
    w = MainWindow()
    w.out_dir_edit.setText(str(tmp_path))
    row = w.input_list.rowCount()
    w.input_list.insertRow(row)
    w.input_list.setItem(row, 1, QTableWidgetItem(str(media)))
    w._scan_output_statuses()  # noqa: SLF001
    assert w._get_input_status(media) == "fast"  # noqa: SLF001


def test_viewer_strips_meta_cue_and_comment(tmp_path: Path) -> None:
    """Viewer text strips JSON meta cue and comment meta lines safely."""
    srt = """1
00:00:00,000 --> 00:00:01,000
Hello

2
00:00:02,000 --> 00:00:03,000
World

{\"tool\": \"Artemonim's Speech Kit\", \"quality\": \"fast\", \"completed\": true}
# ASK_META: {"tool": "Artemonim's Speech Kit", "quality": "fast", "completed": true}
"""
    stripped = strip_ask_meta_from_srt(srt)
    # Should not contain JSON metadata nor comment meta line
    assert "Artemonim's Speech Kit" not in stripped
    # Should preserve the original cues
    assert "Hello" in stripped
    assert "World" in stripped
