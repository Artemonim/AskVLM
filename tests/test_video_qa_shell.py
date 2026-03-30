from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QPushButton, QSplitter, QTableWidget

from gui.main_window import MainWindow
from gui.video_qa import VideoQAPanel

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


def test_video_qa_retry_controls_present_and_disabled(qtbot: QtBot) -> None:
    """Retry controls are visible placeholders until backend wiring exists."""
    panel = VideoQAPanel()
    qtbot.addWidget(panel)
    retry_btn = panel.findChild(QPushButton, "video_qa_retry_selected_chunk")
    resume_btn = panel.findChild(QPushButton, "video_qa_resume_last_run")
    assert retry_btn is not None
    assert resume_btn is not None
    assert retry_btn.isEnabled() is False
    assert resume_btn.isEnabled() is False
    assert retry_btn.text() == "Retry selected chunk"
    assert resume_btn.text() == "Resume last run"


def test_preflight_refresh_renders_report(tmp_path: Path, qtbot: QtBot) -> None:
    """Preflight output uses backend formatting and mentions budget."""
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"abc")

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

    assert panel.lbl_preflight_budget.text() != "-"
    assert "100000" in panel.lbl_preflight_budget.text()


def test_video_qa_layout_has_splitters(qtbot: QtBot) -> None:
    """Video QA layout is split into two resizable areas."""
    panel = VideoQAPanel()
    qtbot.addWidget(panel)
    splitters = panel.findChildren(QSplitter)
    assert len(splitters) >= 2, "Expected horizontal and vertical splitters."
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
