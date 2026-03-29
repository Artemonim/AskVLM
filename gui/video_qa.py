from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.video_qa_context import normalize_video_qa_context
from core.video_qa_policy import (
    default_video_qa_url_import_policy,
)
from core.video_qa_sources import LocalFileProvider

if TYPE_CHECKING:
    from collections.abc import Iterable

    from core.video_qa_context import VideoQAContextBundle
    from core.video_qa_policy import (
        VideoQAUrlImportPolicy,
    )
    from core.video_qa_sources import LocalFileSource


class VideoQAPanel(QWidget):
    """Thin Video QA shell adapter with a local-file source picker."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._provider = LocalFileProvider()
        self._url_import_policy = default_video_qa_url_import_policy()
        self._source: LocalFileSource | None = None

        layout = QVBoxLayout(self)

        title = QLabel("Video QA")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(title)

        guardrails = QLabel(
            "Local file is the active source. URL import, attachments, chunk "
            "planning, LLM orchestration, and budget policy stay backend-only "
            "for now."
        )
        guardrails.setWordWrap(True)
        layout.addWidget(guardrails)

        source_row = QHBoxLayout()
        source_label = QLabel("Local file:")
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("Select a local media file")
        self.source_edit.setClearButtonEnabled(True)
        self.source_edit.editingFinished.connect(self._sync_source_from_edit)
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_for_source)
        source_row.addWidget(source_label)
        source_row.addWidget(self.source_edit, 1)
        source_row.addWidget(self.browse_button)
        layout.addLayout(source_row)

        self.source_details = QLabel("No local file selected.")
        self.source_details.setWordWrap(True)
        self.source_details.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.source_details)

        question_label = QLabel("Question:")
        layout.addWidget(question_label)
        self.question_edit = QLineEdit()
        self.question_edit.setPlaceholderText(
            "Ask a question about the selected local file"
        )
        layout.addWidget(self.question_edit)

        self.answer_placeholder = QLabel(
            "Wave 1 does not execute Video QA yet. This area is reserved for the "
            "final answer surface."
        )
        self.answer_placeholder.setWordWrap(True)
        self.answer_placeholder.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.answer_placeholder)

        layout.addStretch(1)

    def browse_for_source(self) -> None:
        """Open a file dialog and attach the selected local source."""
        start_dir = str(self._source.path.parent) if self._source else str(Path.cwd())
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Choose local media file",
            start_dir,
            "Media Files (*.mp4 *.mkv *.avi *.mov *.webm *.mp3 *.wav);;All Files (*.*)",
        )
        if file_name:
            self.set_source_path(file_name)

    def set_question_text(self, text: str) -> None:
        """Populate the question field."""
        self.question_edit.setText(text)

    def question_text(self) -> str:
        """Return the current question text."""
        return self.question_edit.text()

    def set_source_path(self, path: str | Path | None) -> bool:
        """Set the local source path and refresh the metadata display."""
        if path is None or not str(path).strip():
            self._source = None
            self.source_edit.clear()
            self.source_details.setText("No local file selected.")
            return False

        try:
            source = self._provider.resolve(path)
        except (OSError, ValueError) as exc:
            self._source = None
            self.source_edit.setText(str(path))
            self.source_details.setText(f"Local file unavailable: {exc}")
            return False

        self._source = source
        self.source_edit.setText(str(source.path))
        self.source_details.setText(source.summary)
        return True

    def source_path(self) -> Path | None:
        """Return the resolved source path, if one is selected."""
        if self._source is None:
            return None
        return self._source.path

    def source(self) -> LocalFileSource | None:
        """Return the resolved local source metadata, if available."""
        return self._source

    def context_bundle(
        self,
        attachments: Iterable[str | Path] = (),
    ) -> VideoQAContextBundle:
        """Return a normalized prompt context bundle for the current shell."""
        return normalize_video_qa_context(
            source=self._source,
            question=self.question_text(),
            attachments=attachments,
        )

    def url_import_policy(self) -> VideoQAUrlImportPolicy:
        """Return the backend-only URL import policy."""
        return self._url_import_policy

    def _sync_source_from_edit(self) -> None:
        """Sync the source state from the editable path field."""
        self.set_source_path(self.source_edit.text())
