import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QTableWidgetItem
from pytestqt.qtbot import QtBot

from gui.main_window import MainWindow


def _select_inputs(tmp_path: Path, fixtures_dir: Path, num_videos: int) -> list[Path]:
    short = fixtures_dir / "test_video_short.mp4"
    if not short.is_file():
        pytest.skip("Short fixture not found")
    inputs: list[Path] = []
    for index in range(num_videos):
        copy_path = tmp_path / "e2e_inputs" / f"input_{index}.mp4"
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(short, copy_path)
        inputs.append(copy_path)
    return inputs


def _create_subtitle_crash_window(
    qtbot: QtBot,
    out_dir: Path,
    inputs: list[Path],
    quality: str,
) -> MainWindow:
    """Build a main window configured for the Text + Subtitles transcribe path."""
    window = MainWindow()
    window.show()
    qtbot.addWidget(window)

    window.out_dir_edit.setText(str(out_dir))
    window.chk_diar.setChecked(False)
    window.chk_dialog.setChecked(False)
    if quality == "fast":
        qtbot.mouseClick(window.btn_quality, Qt.LeftButton)

    for media_path in inputs:
        window.last_input_dir = media_path.parent
        row = window.input_list.rowCount()
        window.input_list.insertRow(row)
        window.input_list.setItem(row, 1, QTableWidgetItem(str(media_path)))

    return window


def _create_video_qa_crash_window(
    qtbot: QtBot,
    out_dir: Path,
    media_path: Path,
    quality: str,
) -> MainWindow:
    """Build MainWindow on Video QA tab with source and question (GUI-like)."""
    window = MainWindow()
    window.show()
    qtbot.addWidget(window)

    window.out_dir_edit.setText(str(out_dir))
    window.chk_diar.setChecked(False)
    window.chk_dialog.setChecked(False)
    if quality == "fast":
        qtbot.mouseClick(window.btn_quality, Qt.LeftButton)

    window.shell_tabs.setCurrentIndex(1)
    # * Large token budget so the short fixture passes preflight offline.
    window.video_qa_panel.set_context_window_tokens(200_000)
    window.video_qa_panel.set_source_path(media_path)
    window.video_qa_panel.set_question_text(
        "E2E crash probe: describe one visible detail in this clip.",
    )
    return window


def _video_qa_thread_running_or_terminal_status(window: MainWindow) -> bool:
    """Return True when Video QA is in-flight or has a terminal status message."""
    thread = window._video_qa_thread  # noqa: SLF001
    if thread is not None and thread.isRunning():
        return True
    msg = window.status.currentMessage()
    return (
        "Video QA error" in msg
        or "Video QA completed" in msg
        or "Video QA canceled" in msg
    )


def _make_dummy_thread(*, should_stop: bool) -> SimpleNamespace:
    thread = SimpleNamespace(quitted=False)

    def _is_running() -> bool:
        return True

    def _quit() -> None:
        thread.quitted = True

    def _wait(_timeout_ms: int) -> bool:
        return should_stop

    thread.isRunning = _is_running
    thread.quit = _quit
    thread.wait = _wait
    return thread


# * E2E crash detection covers subtitle and Video QA shutdown paths (manual / heavy).
@pytest.mark.skipif(
    not os.getenv("SK_RUN_E2E_CRASH"),
    reason="Manual E2E test for crash detection (set SK_RUN_E2E_CRASH=1 to run)",
)
@pytest.mark.parametrize("quality", ["fast", "good"])
@pytest.mark.parametrize("num_videos", [1, 2])
@pytest.mark.parametrize("strategy", ["implicit_cancel", "explicit_cancel"])
def test_e2e_subtitles_crash_scenarios(
    qapp: QApplication,
    qtbot: QtBot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quality: str,
    num_videos: int,
    strategy: str,
) -> None:
    """Parametric E2E for crash/hang detection on exit via the subtitle transcribe path.

    Covers Fast and Good modes, single and multi-file inputs, and implicit and
    explicit cancellation (main input table + Start Transcribe).
    """
    temp_settings_path = tmp_path / "test_settings.ini"

    class MockSettings(QSettings):
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            super().__init__(str(temp_settings_path), QSettings.Format.IniFormat)

    monkeypatch.setattr("gui.main_window.QSettings", MockSettings)

    fixtures_dir = Path(__file__).parent / "fixtures"
    inputs = _select_inputs(tmp_path, fixtures_dir, num_videos)

    out_dir = tmp_path / f"e2e_out_{quality}_{num_videos}_{strategy}"
    out_dir.mkdir()

    window = _create_subtitle_crash_window(qtbot, out_dir, inputs, quality)

    qtbot.mouseClick(window.btn_start, Qt.LeftButton)
    qtbot.waitUntil(
        lambda: "Transcribing" in window.status.currentMessage(),
        timeout=60000,
    )
    time.sleep(2.0)

    if strategy == "explicit_cancel":
        qtbot.mouseClick(window.btn_cancel, Qt.LeftButton)
        time.sleep(0.5)

    window.close()
    assert window.await_worker_shutdown(timeout_ms=30000)


@pytest.mark.skipif(
    not os.getenv("SK_RUN_E2E_CRASH"),
    reason="Manual E2E test for crash detection (set SK_RUN_E2E_CRASH=1 to run)",
)
@pytest.mark.parametrize("quality", ["fast", "good"])
@pytest.mark.parametrize("strategy", ["implicit_cancel", "explicit_cancel"])
def test_e2e_video_qa_crash_scenarios(
    qapp: QApplication,
    qtbot: QtBot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quality: str,
    strategy: str,
) -> None:
    """E2E crash/hang check using the real Video QA panel run and cancel wiring.

    Drives ``VideoQAPanel`` (source, question), clicks Run Video QA, waits until
    the worker thread is running or a terminal Video QA status is shown, then
    closes the window. Asserts shutdown without asserting on model output text.
    """
    temp_settings_path = tmp_path / "test_settings.ini"

    class MockSettings(QSettings):
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            super().__init__(str(temp_settings_path), QSettings.Format.IniFormat)

    monkeypatch.setattr("gui.main_window.QSettings", MockSettings)

    fixtures_dir = Path(__file__).parent / "fixtures"
    inputs = _select_inputs(tmp_path, fixtures_dir, 1)
    media_path = inputs[0]

    out_dir = tmp_path / f"e2e_vqa_out_{quality}_{strategy}"
    out_dir.mkdir()

    window = _create_video_qa_crash_window(qtbot, out_dir, media_path, quality)

    qtbot.mouseClick(window.video_qa_panel.btn_run_qa, Qt.LeftButton)
    qtbot.waitUntil(
        lambda: _video_qa_thread_running_or_terminal_status(window),
        timeout=120000,
    )
    time.sleep(2.0)

    if strategy == "explicit_cancel":
        vq_thread = window._video_qa_thread  # noqa: SLF001
        if vq_thread is not None and vq_thread.isRunning():
            qtbot.mouseClick(window.video_qa_panel.btn_cancel_qa, Qt.LeftButton)
            time.sleep(0.5)

    window.close()
    assert window.await_worker_shutdown(timeout_ms=30000)


@pytest.mark.parametrize(
    ("shutdown_result", "expected_accept"),
    [(False, False), (True, True)],
)
def test_close_event_respects_shutdown_result(
    qapp: QApplication,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    *,
    shutdown_result: bool,
    expected_accept: bool,
) -> None:
    """CloseEvent keeps the window open until shutdown completes."""
    monkeypatch.setattr(MainWindow, "_load_settings", lambda _self: None)
    monkeypatch.setattr(MainWindow, "_save_settings", lambda _self: None)

    window = MainWindow()
    qtbot.addWidget(window)

    class DummyWorker:
        def __init__(self) -> None:
            self.closing = False

        def set_closing(self) -> None:
            self.closing = True

    dummy_worker = DummyWorker()
    dummy_thread = _make_dummy_thread(should_stop=shutdown_result)
    monkeypatch.setattr(window, "_worker", dummy_worker)
    monkeypatch.setattr(window, "_thread", dummy_thread)
    monkeypatch.setattr(window, "_burn_thread", None)
    cancel_calls: list[bool] = []
    monkeypatch.setattr(window, "request_cancel", lambda: cancel_calls.append(True))

    event = QCloseEvent()
    window.closeEvent(event)

    assert event.isAccepted() is expected_accept
    assert dummy_worker.closing is True
    assert cancel_calls == [True]


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "-s", __file__]))
