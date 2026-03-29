from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QSettings

from gui.main_window import MainWindow

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
    window.shell_tabs.setCurrentIndex(1)
    window._save_settings()  # noqa: SLF001

    restored = MainWindow()
    qtbot.addWidget(restored)

    assert restored.shell_tabs.currentIndex() == 1
    assert restored.video_qa_panel.source_path() == media.resolve()
    assert restored.video_qa_panel.question_text() == "What is shown?"
