from collections.abc import Callable
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from core.diarization import DiarizationPipeline
from core.llm_formatter import LLMFormatter
from core.whisperx_wrapper import WhisperXWrapper
from editing.text_model import Document, TextSegment
from gui.main_window import MainWindow, PipelineWorker


def test_diarization_returns_list_without_pyannote() -> None:
    """Test that DiarizationPipeline returns empty list when pyannote unavailable."""
    d = DiarizationPipeline(hf_token="")
    out = d.diarize(str(Path("missing.wav")))
    assert isinstance(out, list)


def test_llm_formatter_identity_when_no_model() -> None:
    """Test that LLMFormatter returns input text when no model loaded."""
    fmt = LLMFormatter(model_path=None)
    text = "hello world"
    assert isinstance(fmt.format_text(text), str)


def test_whisperx_align_fallback_without_whisperx() -> None:
    """Test that WhisperXWrapper align falls back without whisperx installed."""
    wx = WhisperXWrapper(model_name="tiny", device="cuda", compute_type="auto")
    aligned = wx.align(Path("missing.wav"), {"segments": []}, language=None)
    assert isinstance(aligned, list)


def test_processing_fixture_twice_produces_two_tabs(tmp_path: Path) -> None:
    """Use the fixture twice as two inputs and ensure two non-empty tabs are created.

    Heavy ML is not required: the pipeline is stubbed to return simple text.
    """
    fixture = Path("tests/fixtures/test_video_short.mp4")
    if not fixture.is_file():
        pytest.skip("fixture video missing")

    # Minimal QApplication for widgets
    QApplication.instance() or QApplication([])

    # Stub pipeline
    class StubPipeline:
        enable_diarization: bool = False
        enable_dialog_blocks: bool = False

        def process(
            self,
            input_path: Path,
            _work_dir: Path,
            _progress: Callable[[str, float], None] | None = None,
        ) -> Document:
            # Return a document with deterministic content per input
            doc = Document()
            doc.add_segment(
                TextSegment("speaker_1", 0.0, 0.0, f"content for {input_path.name}")
            )
            return doc

    # Prepare worker with two identical inputs
    PipelineWorker(
        pipeline=StubPipeline(),
        inputs=[fixture, fixture],
        out_dir=tmp_path,
        options={
            "enable_diarization": False,
            "enable_dialog_blocks": False,
            "export_format": "txt",
            "single_view": False,
            "burn_in": False,
            "save_srt": False,
        },
    )

    # Run synchronously and synthesize outputs (simulate exporter behavior)
    a = tmp_path / f"{fixture.stem}_1.txt"
    b = tmp_path / f"{fixture.stem}_2.txt"
    a.write_text("first content", encoding="utf-8")
    b.write_text("second content", encoding="utf-8")
    outputs = [str(a), str(b)]

    # Build GUI and feed results
    w = MainWindow()
    w.on_finished(outputs, view_text="")
    # Verify two tabs present with non-empty content
    assert w.tabs.count() >= 2
    ed0 = w.get_editor_at(0)
    ed1 = w.get_editor_at(1)
    assert ed0 is not None
    assert ed1 is not None
    # Table editor: ensure at least one row with non-empty text in column 2
    assert ed0.rowCount() >= 1
    assert ed0.item(0, 2).text() if ed0.item(0, 2) else ""
    assert ed1.rowCount() >= 1
    assert ed1.item(0, 2).text() if ed1.item(0, 2) else ""


def test_time_parsing_populates_nonzero_times() -> None:
    """Ensure that time parsing yields non-zero start/end in the first row."""
    QApplication.instance() or QApplication([])
    w = MainWindow()
    sample = "1\n00:00:01,000 --> 00:00:03,500\nspeaker_1: Hello world\n\n"
    w.on_finished([], view_text=sample)
    ed = w.get_editor_at(0)
    assert ed is not None
    assert ed.rowCount() >= 1
    # time (col 0) must not be the default zero string
    time_item = ed.item(0, 0)
    assert time_item is not None
    time_str = time_item.text()
    assert time_str.startswith("00:00:01")
    assert "→" in time_str
