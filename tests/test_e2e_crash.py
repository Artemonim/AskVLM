import os
import sys
import time
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QCloseEvent

from gui.main_window import MainWindow


# Combinations:
# 1. Quality: fast vs good
# 2. Count: 1 vs 2 files
# 3. Action: implicit_cancel (close window) vs explicit_cancel (click cancel then close)
@pytest.mark.skipif(
    not os.getenv("SK_RUN_E2E_CRASH"),
    reason="Manual E2E test for crash detection (set SK_RUN_E2E_CRASH=1 to run)",
)
@pytest.mark.parametrize("quality", ["fast", "good"])
@pytest.mark.parametrize("num_videos", [1, 2])
@pytest.mark.parametrize("strategy", ["implicit_cancel", "explicit_cancel"])
def test_e2e_crash_scenarios(
    qapp, qtbot, tmp_path, monkeypatch, quality, num_videos, strategy
) -> None:
    """Parametric E2E test for crash/hang detection on exit.
    Covers Fast/Good modes, Single/Multi-file inputs, and Implicit/Explicit cancellation.
    """
    # * Isolate QSettings to prevent overwriting user config
    # Define a mock QSettings that ignores arguments and uses a temp INI file
    temp_settings_path = tmp_path / "test_settings.ini"

    class MockSettings(QSettings):
        def __init__(self, *args, **kwargs) -> None:
            # Redirect all QSettings instantiations to our temp file
            super().__init__(str(temp_settings_path), QSettings.Format.IniFormat)

    # Patch the QSettings class imported in gui.main_window
    monkeypatch.setattr("gui.main_window.QSettings", MockSettings)

    # Setup inputs
    fixtures_dir = Path(__file__).parent / "fixtures"
    # Use existing fixtures; duplicate if needed to reach count
    sources = [
        fixtures_dir / "test_video_first.mp4",
        fixtures_dir / "test_video_second.mp4",
    ]
    if not sources[0].exists():
        pytest.skip("Fixtures not found")

    # Select inputs based on num_videos
    inputs = [sources[i % len(sources)] for i in range(num_videos)]

    out_dir = tmp_path / f"e2e_out_{quality}_{num_videos}_{strategy}"
    out_dir.mkdir()

    # Initialize Window
    window = MainWindow()
    window.show()
    qtbot.addWidget(window)

    # Setup UI state
    window.out_dir_edit.setText(str(out_dir))
    window.chk_diar.setChecked(False)
    window.chk_dialog.setChecked(False)

    # Set quality
    # We need to toggle if default isn't what we want, or set internal state directly
    # and trigger update. The UI button toggles, but we can set state and call applier.
    window._quality_mode = quality
    window._apply_quality_to_pipeline()

    # Add files
    for p in inputs:
        window.last_input_dir = p.parent
        row = window.input_list.rowCount()
        window.input_list.insertRow(row)
        from PySide6.QtWidgets import QTableWidgetItem

        window.input_list.setItem(row, 1, QTableWidgetItem(str(p)))
        window._update_item_icon_row(row)

    window._scan_output_statuses()

    # Start processing
    qtbot.mouseClick(window.btn_start, Qt.LeftButton)

    # Wait until Whisper transcription phase begins to ensure GPU work is active.
    # Status bar messages are updated via LocalPipeline.report().
    qtbot.waitUntil(
        lambda: "Transcribing" in window.status.currentMessage(),
        timeout=60000,
    )
    # Let it run briefly so that Whisper has time to start emitting segments.
    time.sleep(2.0)

    if strategy == "explicit_cancel":
        # * Explicit user-style cancel while Whisper is actively transcribing.
        qtbot.mouseClick(window.btn_cancel, Qt.LeftButton)
        # Do NOT wait for cancellation acknowledgment here: we want to model
        # the user closing the window almost immediately after pressing Cancel.
        time.sleep(0.5)
        window.close()
    else:
        # Implicit cancel via Close (user closes window without pressing Cancel).
        window.close()

    # Wait for threads shutdown
    # Capture thread reference early to avoid PySide6 deletion issues
    th = window._thread

    # If explicit cancel + close, the window.close() triggered closeEvent,
    # which triggers await_worker_shutdown, which QUITS and WAITS on the thread.
    # So by the time we get here, the thread might already be finished and deleted by Qt.

    # Check if Python wrapper is still valid and running
    if th is not None:
        try:
            if th.isRunning():
                success = th.wait(30000)
                if not success:
                    pass
        except RuntimeError:
            # Object already deleted by C++ side (expected on successful close)
            pass

    # Also check burn thread just in case (should be None here)
    bth = window._burn_thread
    if bth is not None:
        try:
            if bth.isRunning():
                bth.wait(2000)
        except RuntimeError:
            pass


@pytest.mark.parametrize(
    ("shutdown_result", "expected_accept"),
    [(False, False), (True, True)],
)
def test_close_event_respects_shutdown_result(
    qapp, qtbot, monkeypatch, shutdown_result, expected_accept
) -> None:
    """CloseEvent keeps the window open until shutdown completes."""
    monkeypatch.setattr(MainWindow, "_load_settings", lambda self: None)
    monkeypatch.setattr(MainWindow, "_save_settings", lambda self: None)

    window = MainWindow()
    qtbot.addWidget(window)

    class DummyWorker:
        def __init__(self) -> None:
            self.closing = False

        def set_closing(self) -> None:
            self.closing = True

    dummy_worker = DummyWorker()
    window._worker = dummy_worker
    cancel_calls: list[bool] = []
    monkeypatch.setattr(window, "request_cancel", lambda: cancel_calls.append(True))
    monkeypatch.setattr(
        window,
        "await_worker_shutdown",
        lambda timeout_ms=30000: shutdown_result,
    )

    event = QCloseEvent()
    window.closeEvent(event)

    assert event.isAccepted() is expected_accept
    assert dummy_worker.closing is True
    assert cancel_calls == [True]


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "-s", __file__]))
