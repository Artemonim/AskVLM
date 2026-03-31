from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QPushButton, QSplitter, QTableWidget

from gui.main_window import MainWindow
from gui.video_qa import VIDEO_QA_PREFLIGHT_FPS_CHOICES, VideoQAPanel

if TYPE_CHECKING:
    from pathlib import Path

    import pytest
    from pytestqt.qtbot import QtBot


def test_video_qa_shell_restores_screen_and_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    qtbot: QtBot,
) -> None:
    """Video QA shell restores its tab and local source state."""
    settings_path = tmp_path / "settings.ini"

    class MockSettings(QSettings):
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            super().__init__(str(settings_path), QSettings.Format.IniFormat)

    monkeypatch.setattr("gui.main_window.QSettings", MockSettings)

    media = tmp_path / "clip.mp4"
    media.write_bytes(b"abc")

    window = MainWindow()
    qtbot.addWidget(window)
    assert window.shell_tabs.tabText(0) == "Text + Subtitles"
    assert window.shell_tabs.tabText(1) == "Video QA"

    window.video_qa_panel.set_source_path(media)
    window.video_qa_panel.set_question_text("What is shown?")
    window.video_qa_panel.set_context_window_tokens(123456)
    window.video_qa_panel.set_frame_sample_fps(0.1)
    window.video_qa_panel.set_answer_area_ratio_percent(72)
    main_splitter_state = window.video_qa_panel.main_splitter_state()
    left_splitter_state = window.video_qa_panel.left_splitter_state()
    window.shell_tabs.setCurrentIndex(1)
    window._save_settings()  # noqa: SLF001

    restored = MainWindow()
    qtbot.addWidget(restored)

    assert restored.shell_tabs.currentIndex() == 1
    assert restored.video_qa_panel.source_path() == media.resolve()
    assert restored.video_qa_panel.question_text() == "What is shown?"
    assert restored.video_qa_panel.context_window_tokens() == 123456
    assert restored.video_qa_panel.frame_sample_fps() == 0.1
    assert restored.video_qa_panel.answer_area_ratio_percent() == 72
    assert restored.video_qa_panel.main_splitter_state() == main_splitter_state
    assert restored.video_qa_panel.left_splitter_state() == left_splitter_state


def test_video_qa_restores_attachments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    qtbot: QtBot,
) -> None:
    """Attachment paths and include flags persist across sessions."""
    settings_path = tmp_path / "settings.ini"

    class MockSettings(QSettings):
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            super().__init__(str(settings_path), QSettings.Format.IniFormat)

    monkeypatch.setattr("gui.main_window.QSettings", MockSettings)

    media = tmp_path / "v.mp4"
    media.write_bytes(b"x")
    note = tmp_path / "note.txt"
    note.write_text("hello", encoding="utf-8")
    extra = tmp_path / "extra.md"
    extra.write_text("# t", encoding="utf-8")

    w = MainWindow()
    qtbot.addWidget(w)
    w.video_qa_panel.set_source_path(media)
    w.video_qa_panel.restore_attachments_state(
        [
            {"path": str(note.resolve()), "enabled": True},
            {"path": str(extra.resolve()), "enabled": False},
        ]
    )
    w._save_settings()  # noqa: SLF001

    w2 = MainWindow()
    qtbot.addWidget(w2)
    persisted = w2.video_qa_panel.attachments_for_persistence()
    paths = {str(x["path"]) for x in persisted}
    assert str(note.resolve()) in paths
    assert str(extra.resolve()) in paths
    by_path = {str(x["path"]): x["enabled"] for x in persisted}
    assert by_path[str(note.resolve())] is True
    assert by_path[str(extra.resolve())] is False


def test_context_bundle_respects_attachment_include(
    tmp_path: Path,
    qtbot: QtBot,
) -> None:
    """Disabled attachments are normalized but excluded from enabled_attachments."""
    media = tmp_path / "v.mp4"
    media.write_bytes(b"x")
    a_ok = tmp_path / "a.txt"
    a_ok.write_text("a", encoding="utf-8")
    b_ok = tmp_path / "b.txt"
    b_ok.write_text("b", encoding="utf-8")

    panel = VideoQAPanel()
    qtbot.addWidget(panel)
    panel.set_source_path(media)
    panel.restore_attachments_state(
        [
            {"path": str(a_ok.resolve()), "enabled": True},
            {"path": str(b_ok.resolve()), "enabled": False},
        ]
    )
    bundle = panel.context_bundle()
    names = {att.name for att in bundle.enabled_attachments}
    assert "a.txt" in names
    assert "b.txt" not in names
    disabled = {att.name for att in bundle.disabled_attachments}
    assert "b.txt" in disabled


def test_video_qa_run_button_enabled_and_emits_request(qtbot: QtBot) -> None:
    """Run Video QA is active and notifies the shell via ``video_qa_run_requested``."""
    panel = VideoQAPanel()
    qtbot.addWidget(panel)
    run_btn = panel.findChild(QPushButton, "video_qa_run")
    assert run_btn is not None
    assert run_btn.isEnabled()
    assert "Run Video QA" in run_btn.text()
    cancel_btn = panel.findChild(QPushButton, "video_qa_cancel")
    assert cancel_btn is not None
    assert cancel_btn.text() == "Cancel"
    assert not cancel_btn.isEnabled()
    with qtbot.waitSignal(panel.video_qa_run_requested, timeout=2000):
        qtbot.mouseClick(run_btn, Qt.MouseButton.LeftButton)


def test_video_qa_cancel_button_emits_when_enabled(qtbot: QtBot) -> None:
    """Cancel notifies the shell via ``video_qa_cancel_requested`` when enabled."""
    panel = VideoQAPanel()
    qtbot.addWidget(panel)
    cancel_btn = panel.findChild(QPushButton, "video_qa_cancel")
    assert cancel_btn is not None
    cancel_btn.setEnabled(True)
    with qtbot.waitSignal(panel.video_qa_cancel_requested, timeout=2000):
        qtbot.mouseClick(cancel_btn, Qt.MouseButton.LeftButton)


def test_main_window_dispatches_video_qa_launch_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    qtbot: QtBot,
) -> None:
    """The shell forwards the run button click into the Video QA launch path."""
    settings_path = tmp_path / "settings.ini"

    class MockSettings(QSettings):
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            super().__init__(str(settings_path), QSettings.Format.IniFormat)

    monkeypatch.setattr("gui.main_window.QSettings", MockSettings)
    monkeypatch.setattr("gui.main_window.get_media_duration_seconds", lambda _p: 60.0)

    media = tmp_path / "clip.mp4"
    media.write_bytes(b"abc")

    window = MainWindow()
    qtbot.addWidget(window)
    window.out_dir_edit.setText(str(tmp_path / "out"))
    window.video_qa_panel.set_source_path(media)
    window.video_qa_panel.set_question_text("What is shown?")
    run_btn = window.video_qa_panel.findChild(QPushButton, "video_qa_run")
    assert run_btn is not None
    assert run_btn.isEnabled()

    launched: list[tuple[object, object]] = []

    def fake_start(ctx: object, out_dir: object) -> None:
        launched.append((ctx, out_dir))

    monkeypatch.setattr(window, "_start_video_qa_worker", fake_start)

    qtbot.mouseClick(run_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(launched) == 1, timeout=2000)

    ctx, out_dir = launched[0]
    assert getattr(ctx, "question", "") == "What is shown?"
    assert getattr(getattr(ctx, "source", None), "path", None) == media.resolve()
    assert str(out_dir) == str((tmp_path / "out").resolve())


def test_main_window_dispatches_video_qa_cancel_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    qtbot: QtBot,
) -> None:
    """The shell forwards the Video QA cancel click to the cooperative cancel path."""
    settings_path = tmp_path / "settings.ini"

    class MockSettings(QSettings):
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            super().__init__(str(settings_path), QSettings.Format.IniFormat)

    monkeypatch.setattr("gui.main_window.QSettings", MockSettings)

    window = MainWindow()
    qtbot.addWidget(window)
    cancel_btn = window.video_qa_panel.findChild(QPushButton, "video_qa_cancel")
    assert cancel_btn is not None
    cancel_btn.setEnabled(True)

    worker = MagicMock()
    window._video_qa_worker = worker  # noqa: SLF001
    fake_thread = MagicMock()
    fake_thread.isRunning.return_value = True
    window._video_qa_thread = fake_thread  # noqa: SLF001

    qtbot.mouseClick(cancel_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: worker.request_cancel.called, timeout=2000)

    assert worker.request_cancel.call_count == 1


def test_video_qa_retry_controls_removed(qtbot: QtBot) -> None:
    """Retry scaffold is removed from the Video QA panel."""
    panel = VideoQAPanel()
    qtbot.addWidget(panel)
    retry_btn = panel.findChild(QPushButton, "video_qa_retry_selected_chunk")
    resume_btn = panel.findChild(QPushButton, "video_qa_resume_last_run")
    assert retry_btn is None
    assert resume_btn is None


def test_video_qa_auto_preflight_refreshes_after_debounce(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    qtbot: QtBot,
) -> None:
    """Question, source, attachment, and budget changes trigger debounced refreshes."""
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"abc")
    note = tmp_path / "note.txt"
    note.write_text("hello", encoding="utf-8")
    monkeypatch.setattr("gui.video_qa.get_media_duration_seconds", lambda _p: 30.0)

    panel = VideoQAPanel(preflight_debounce_ms=40)
    qtbot.addWidget(panel)
    panel.set_source_path(media)
    panel.set_question_text("Auto refresh?")

    qtbot.wait(10)
    assert panel.preflight_edit.toPlainText() == ""

    qtbot.waitUntil(
        lambda: "Auto refresh?" in panel.preflight_edit.toPlainText(),
        timeout=1000,
    )
    assert "frames_total=" in panel.preflight_edit.toPlainText()

    previous_text = panel.preflight_edit.toPlainText()
    panel.restore_attachments_state([{"path": str(note.resolve()), "enabled": True}])
    qtbot.waitUntil(
        lambda: panel.preflight_edit.toPlainText() != previous_text, timeout=1000
    )

    panel.set_context_window_tokens(120000)
    qtbot.waitUntil(
        lambda: "120000" in panel.lbl_preflight_budget.text(),
        timeout=1000,
    )


def test_preflight_fps_changes_budget_estimate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    qtbot: QtBot,
) -> None:
    """Lower fps reduces the offline frame budget shown in the preflight summary."""
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"abc")
    monkeypatch.setattr("gui.video_qa.get_media_duration_seconds", lambda _p: 100.0)

    panel = VideoQAPanel()
    qtbot.addWidget(panel)
    panel.set_source_path(media)
    panel.set_question_text("FPS test")
    panel.refresh_preflight()
    high = panel.lbl_preflight_budget.text()
    panel.set_frame_sample_fps(0.05)
    panel.refresh_preflight()
    low = panel.lbl_preflight_budget.text()
    assert high != low
    assert "frames" in high
    assert "frames" in low


def test_preflight_fps_toggle_refreshes_without_debounce_wait(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    qtbot: QtBot,
) -> None:
    """FPS toggle triggers an immediate preflight refresh (not only the debounced timer)."""
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"abc")
    monkeypatch.setattr("gui.video_qa.get_media_duration_seconds", lambda _p: 30.0)

    panel = VideoQAPanel(preflight_debounce_ms=8000)
    qtbot.addWidget(panel)
    panel.set_source_path(media)
    panel.set_question_text("Immediate fps refresh")
    qtbot.wait(400)
    assert panel.preflight_edit.toPlainText() == ""

    idx = VIDEO_QA_PREFLIGHT_FPS_CHOICES.index(1.0)
    btn = panel._fps_button_group.button(idx)  # noqa: SLF001
    assert btn is not None
    qtbot.mouseClick(btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(
        lambda: "Immediate fps refresh" in panel.preflight_edit.toPlainText(),
        timeout=3000,
    )


def test_preflight_refresh_renders_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    qtbot: QtBot,
) -> None:
    """Preflight output uses backend formatting and mentions budget."""
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"abc")
    monkeypatch.setattr("gui.video_qa.get_media_duration_seconds", lambda _p: 30.0)

    panel = VideoQAPanel()
    qtbot.addWidget(panel)
    panel.set_source_path(media)
    panel.set_question_text("Describe the scene.")
    panel.set_context_window_tokens(100000)
    panel.refresh_preflight()

    text = panel.preflight_edit.toPlainText()
    assert "Budget:" in text
    assert "Question:" in text
    assert "Describe the scene." in text
    assert "frames_total=" in text

    assert panel.lbl_preflight_budget.text() != "-"
    assert "100000" in panel.lbl_preflight_budget.text()
    assert "frames" in panel.lbl_preflight_budget.text()


def test_video_qa_layout_has_splitters(qtbot: QtBot) -> None:
    """Video QA layout is split into two resizable areas."""
    panel = VideoQAPanel()
    qtbot.addWidget(panel)
    splitters = panel.findChildren(QSplitter)
    assert len(splitters) >= 3, "Expected main, left, and answer/progress splitters."
    assert any(
        splitter.orientation() == Qt.Orientation.Horizontal for splitter in splitters
    )
    assert any(
        splitter.orientation() == Qt.Orientation.Vertical for splitter in splitters
    )
    attachments_table = panel.findChild(QTableWidget)
    assert attachments_table is not None
    assert attachments_table.minimumHeight() >= 220
    assert panel.preflight_edit.minimumHeight() >= 120


def test_text_subtitles_shell_layout_unchanged(qtbot: QtBot) -> None:
    """Text + Subtitles workspace keeps expected shell widgets (regression guard)."""
    w = MainWindow()
    qtbot.addWidget(w)
    assert w.shell_tabs.count() == 2
    assert w.shell_tabs.tabText(0) == "Text + Subtitles"
    assert hasattr(w, "splitter")
    assert hasattr(w, "tabs")
    assert hasattr(w, "video_qa_panel")
    assert isinstance(w.video_qa_panel, VideoQAPanel)


def test_preflight_warnings_summary_neutral_when_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    qtbot: QtBot,
) -> None:
    """Preflight warnings row stays non-red when the backend reports no warnings."""
    media = tmp_path / "clean.mp4"
    media.write_bytes(b"a")
    monkeypatch.setattr("gui.video_qa.get_media_duration_seconds", lambda _p: 10.0)
    panel = VideoQAPanel()
    qtbot.addWidget(panel)
    panel.set_source_path(media)
    panel.set_question_text("Q?")
    panel.refresh_preflight()
    html = panel.lbl_preflight_warnings.text()
    assert "none" in html
    assert "#f48771" not in html


def test_preflight_warnings_summary_red_when_source_missing(qtbot: QtBot) -> None:
    """Missing media produces a real warning and uses the warning styling."""
    panel = VideoQAPanel()
    qtbot.addWidget(panel)
    panel.set_question_text("Q?")
    panel.refresh_preflight()
    html = panel.lbl_preflight_warnings.text()
    assert "#f48771" in html


def test_video_qa_answer_uses_markdown_view(qtbot: QtBot) -> None:
    """Answer surface accepts Markdown via QTextEdit."""
    panel = VideoQAPanel()
    qtbot.addWidget(panel)
    panel.set_answer_text("# Heading\n\n**bold** text")
    plain = panel.answer_edit.toPlainText()
    assert "Heading" in plain
    assert "bold" in plain
