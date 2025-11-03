from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from editing.text_model import Document, TextSegment
from gui import main_window as gw
from gui.main_window import PipelineWorker


class _StubPipeline:
    """Lightweight stub pipeline for queue integration tests.

    It returns a simple Document without performing any heavy ML work.
    """

    enable_diarization: bool = False
    enable_dialog_blocks: bool = False

    def process(self, _inp: Path, _out: Path, **_kwargs: object) -> Document:
        doc = Document()
        # Minimal non-empty text to allow exporters to write something
        doc.add_segment(TextSegment("speaker_1", 0.0, 1.0, f"ok:{_inp.name}"))
        return doc


def _make_options(quality: str) -> dict[str, object]:
    return {
        "export_format": "txt",
        "single_view": False,
        "save_srt": False,
        "subtitle_max_line_width": 42,
        "subtitle_max_lines": 2,
        "quality": quality,
        "no_empty": False,
    }


@pytest.mark.integration
def test_queue_processing_good_and_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run multi-file queues in good and fast modes ensuring no worker errors."""
    # Ensure Qt app exists (safe for repeated calls)
    QApplication.instance() or QApplication([])

    # Prepare inputs by name (files need not exist for this test)
    inputs_good: list[Path] = [
        Path("tests/fixtures/test_video_first.mp4").resolve(),
        Path("tests/fixtures/test_video_short.mp4").resolve(),
    ]

    inputs_fast: list[Path] = [
        Path("tests/fixtures/test_video_first.mp4").resolve(),
        Path("tests/fixtures/test_video_short.mp4").resolve(),
        Path("tests/fixtures/test_video_second.mp4").resolve(),
        Path("tests/fixtures/test_video_third.mp4").resolve(),
    ]

    # Monkeypatch durations used by the scheduler to control CPU heuristic
    def _fake_duration(p: Path) -> float:
        stem = p.stem
        if stem == "test_video_first":
            return 100.0
        if stem == "test_video_short":
            return 10.0
        if stem == "test_video_second":
            return 20.0
        if stem == "test_video_third":
            return 15.0
        return 30.0

    monkeypatch.setattr(gw, "get_media_duration_seconds", _fake_duration)

    # GOOD mode queue (sequential, no opportunistic CPU scheduling expected by heuristic)
    errs: list[str] = []
    worker_good = PipelineWorker(
        _StubPipeline(), inputs_good, tmp_path, _make_options("good")
    )
    worker_good.error.connect(lambda m: errs.append(str(m)))
    worker_good.run()
    assert not errs, f"Errors occurred in GOOD queue: {errs}"

    # FAST mode queue (heuristic may schedule one CPU job in parallel)
    worker_fast = PipelineWorker(
        _StubPipeline(), inputs_fast, tmp_path, _make_options("fast")
    )
    worker_fast.error.connect(lambda m: errs.append(str(m)))
    worker_fast.run()
    assert not errs, f"Errors occurred in FAST queue: {errs}"
