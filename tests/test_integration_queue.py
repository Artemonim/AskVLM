import threading as _th
import time as _t
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from core.pipelines import CancelledError
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


@pytest.mark.integration
def test_cancel_stops_scheduling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Cancel flag prevents scheduling of new jobs; running job aborts promptly."""
    QApplication.instance() or QApplication([])

    # Prepare three inputs (names only)
    files = [
        Path("tests/fixtures/test_video_first.mp4").resolve(),
        Path("tests/fixtures/test_video_second.mp4").resolve(),
        Path("tests/fixtures/test_video_third.mp4").resolve(),
    ]

    # Slow stub to simulate long-running process with cancel checks
    class _SlowCancelPipeline:
        enable_diarization = False
        enable_dialog_blocks = False

        def process(self, _inp: Path, _out: Path, **kwargs: object) -> Document:
            should_cancel = kwargs.get("should_cancel")
            # Emit a few progress callbacks
            cb = kwargs.get("progress")
            if callable(cb):
                cb("prepare", 0.1)
            # Spin and respect cancel
            t0 = _t.time()
            while _t.time() - t0 < 0.5:
                if callable(should_cancel) and should_cancel():
                    msg = "Canceled"
                    raise CancelledError(msg)
                _t.sleep(0.05)
                if callable(cb):
                    cb("transcribe", 0.2)
            # Return a trivial document if not canceled
            d = Document()
            d.add_segment(TextSegment("speaker_1", 0.0, 1.0, "ok"))
            return d

    opts = {
        "export_format": "txt",
        "single_view": False,
        "save_srt": False,
        "subtitle_max_line_width": 42,
        "subtitle_max_lines": 2,
        "quality": "good",
        "no_empty": False,
    }

    # Collect scheduled CUDA messages to ensure no post-cancel scheduling
    scheduled: list[str] = []

    worker = PipelineWorker(_SlowCancelPipeline(), files, tmp_path, opts)
    worker.log.connect(lambda m: scheduled.append(m) if "Scheduled CUDA" in m else None)

    # Monkeypatch: request cancel shortly after starting
    def _cancel_soon() -> None:
        worker.request_cancel()

    # Run synchronously: trigger cancel after brief delay
    t = _th.Timer(0.1, _cancel_soon)
    t.start()
    worker.run()
    t.cancel()

    # Ensure no scheduling after cancel
    assert not any("Scheduled CUDA" in m for m in scheduled[1:]), scheduled
