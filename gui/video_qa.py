from __future__ import annotations

import contextlib
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QByteArray, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.ffmpeg import get_media_duration_seconds
from core.video_qa_context import (
    VideoQAAttachmentRequest,
    normalize_video_qa_context,
)
from core.video_qa_orchestration import (
    build_video_qa_preflight_report,
    build_video_qa_preflight_summary,
    format_video_qa_preflight_report_text,
)
from core.video_qa_policy import (
    default_video_qa_url_import_policy,
)
from core.video_qa_runtime import default_video_qa_budget_policy
from core.video_qa_sources import LocalFileProvider

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from core.video_qa_context import VideoQAContextBundle
    from core.video_qa_policy import (
        VideoQAUrlImportPolicy,
    )
    from core.video_qa_sources import LocalFileSource

# * Supported attachment extensions aligned with core/video_qa_context classification.
_ATTACHMENT_NAME_FILTER = (
    "Attachments (*.txt *.md *.rst *.json *.csv *.xml *.yaml *.yml *.ini *.log *.html "
    "*.htm *.cfg *.env *.tsv "
    "*.py *.pyi *.js *.jsx *.ts *.tsx *.rs *.go *.c *.h *.cpp *.hpp *.cs *.java *.kt "
    "*.lua *.php *.rb *.scala *.swift *.sql *.m *.ps1 *.sh "
    "*.png *.jpg *.jpeg *.gif *.webp *.bmp *.tif *.tiff);;"
    "All files (*.*)"
)

_READ_ONLY_STYLE = (
    "QPlainTextEdit { background-color: #1e1e1e; color: #d4d4d4; "
    "font-family: Consolas, 'Segoe UI', monospace; font-size: 11px; "
    "border: 1px solid #3f3f46; }"
)
_ANSWER_STYLE = (
    "QPlainTextEdit { background-color: #252526; color: #d4d4d4; "
    "font-family: 'Segoe UI', system-ui, sans-serif; font-size: 12px; "
    "border: 1px solid #3f3f46; }"
)
_EVIDENCE_STYLE = (
    "QPlainTextEdit { background-color: #1e1e1e; color: #d4d4d4; "
    "font-family: Consolas, 'Segoe UI', monospace; font-size: 11px; "
    "border: 1px solid #3f3f46; }"
)
_PREFLIGHT_SUMMARY_STYLE = (
    "QLabel { color: #d4d4d4; font-size: 12px; }"
    'QLabel[isWarning="true"] { color: #f48771; font-weight: bold; }'
)
_VIDEO_QA_PANEL_STYLE = (
    "QGroupBox { border: 1px solid #3f3f46; border-radius: 4px; margin-top: 10px; "
    "padding-top: 10px; background-color: #2b2b2f; }"
    "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; "
    "color: #d4d4d4; }"
    "QTableWidget { background-color: #1e1e1e; color: #d4d4d4; gridline-color: #3f3f46; "
    "border: 1px solid #3f3f46; }"
    "QHeaderView::section { background-color: #2d2d30; color: #d4d4d4; "
    "border: 1px solid #3f3f46; padding: 4px 8px; }"
    "QLineEdit, QSpinBox, QComboBox { background-color: #1e1e1e; color: #d4d4d4; "
    "border: 1px solid #3f3f46; }"
    "QPushButton { background-color: #2d2d30; color: #d4d4d4; "
    "border: 1px solid #3f3f46; padding: 6px 10px; }"
    "QPushButton:disabled { color: #7a7a7a; background-color: #252526; }"
    "QCheckBox { color: #d4d4d4; }"
    "QSplitter::handle { background-color: #3f3f46; }"
    "QLabel { color: #d4d4d4; }"
)


class VideoQAPanel(QWidget):
    """Video QA workspace: source, question, attachments, preflight, answer, and evidence."""

    video_qa_run_requested = Signal()
    video_qa_cancel_requested = Signal()

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        preflight_debounce_ms: int = 2000,
    ) -> None:
        super().__init__(parent)
        self._provider = LocalFileProvider()
        self._url_import_policy = default_video_qa_url_import_policy()
        self._source: LocalFileSource | None = None
        self._last_attachment_dir = Path.cwd()
        self._preflight_refresh_timer = QTimer(self)
        self._preflight_refresh_timer.setSingleShot(True)
        self._preflight_refresh_timer.setInterval(max(0, int(preflight_debounce_ms)))
        self._preflight_refresh_timer.timeout.connect(self.refresh_preflight)
        self._build_form(QVBoxLayout(self))
        self.setStyleSheet(_VIDEO_QA_PANEL_STYLE)

    def _build_form(self, root: QVBoxLayout) -> None:
        self._build_header(root)

        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter.setChildrenCollapsible(False)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self._build_source_and_question(left_layout)

        self._left_splitter = QSplitter(Qt.Orientation.Vertical)
        self._left_splitter.setChildrenCollapsible(False)

        att_widget = QWidget()
        att_layout = QVBoxLayout(att_widget)
        att_layout.setContentsMargins(0, 0, 0, 0)
        self._build_attachments_group(att_layout)

        pre_widget = QWidget()
        pre_layout = QVBoxLayout(pre_widget)
        pre_layout.setContentsMargins(0, 0, 0, 0)
        self._build_preflight_group(pre_layout)

        self._left_splitter.addWidget(att_widget)
        self._left_splitter.addWidget(pre_widget)
        self._left_splitter.setStretchFactor(0, 1)
        self._left_splitter.setStretchFactor(1, 2)
        self._left_splitter.setSizes([240, 360])
        left_layout.addWidget(self._left_splitter, 1)

        self._build_answer_evidence_group(right_layout)
        self._build_run_placeholder(right_layout)

        self._main_splitter.addWidget(left_widget)
        self._main_splitter.addWidget(right_widget)
        self._main_splitter.setStretchFactor(0, 1)
        self._main_splitter.setStretchFactor(1, 1)
        self._main_splitter.setSizes([540, 460])

        root.addWidget(self._main_splitter, 1)

    def _build_header(self, root: QVBoxLayout) -> None:
        title = QLabel("Video QA")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        root.addWidget(title)
        hint = QLabel(
            "Local file source, optional text/code/image attachments, and preflight "
            "planning. Run Video QA uses the configured output folder, local ASR, "
            "ffmpeg frames, and LM Studio."
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

    def _build_source_and_question(self, root: QVBoxLayout) -> None:
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Local file:"))
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("Select a local media file")
        self.source_edit.setClearButtonEnabled(True)
        self.source_edit.editingFinished.connect(self._sync_source_from_edit)
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_for_source)
        source_row.addWidget(self.source_edit, 1)
        source_row.addWidget(self.browse_button)
        root.addLayout(source_row)
        self.source_details = QLabel("No local file selected.")
        self.source_details.setWordWrap(True)
        self.source_details.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        root.addWidget(self.source_details)
        root.addWidget(QLabel("Question:"))
        self.question_edit = QLineEdit()
        self.question_edit.setPlaceholderText(
            "Ask a question about the selected local file"
        )
        self.question_edit.textChanged.connect(self._schedule_preflight_refresh)
        root.addWidget(self.question_edit)

    def _build_attachments_group(self, root: QVBoxLayout) -> None:
        att_box = QGroupBox("Attachments")
        att_box.setStyleSheet(_VIDEO_QA_PANEL_STYLE)
        att_layout = QVBoxLayout(att_box)
        self._attachment_table = QTableWidget(0, 2)
        self._attachment_table.setHorizontalHeaderLabels(["Include", "Path"])
        self._attachment_table.setAlternatingRowColors(True)
        self._attachment_table.setMinimumHeight(220)
        self._attachment_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Fixed
        )
        self._attachment_table.setColumnWidth(0, 72)
        self._attachment_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self._attachment_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._attachment_table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._attachment_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        att_layout.addWidget(self._attachment_table, 1)
        att_btns = QHBoxLayout()
        self.btn_add_attachments = QPushButton("Add files…")
        self.btn_add_attachments.clicked.connect(self._add_attachment_files)
        self.btn_remove_attachments = QPushButton("Remove selected")
        self.btn_remove_attachments.clicked.connect(self._remove_selected_attachments)
        att_btns.addWidget(self.btn_add_attachments)
        att_btns.addWidget(self.btn_remove_attachments)
        att_btns.addStretch(1)
        att_layout.addLayout(att_btns)
        root.addWidget(att_box, 1)

    def _build_preflight_group(self, root: QVBoxLayout) -> None:
        pre_box = QGroupBox("Preflight")
        pre_box.setStyleSheet(_VIDEO_QA_PANEL_STYLE)
        pre_layout = QVBoxLayout(pre_box)
        pre_btn_row = QHBoxLayout()
        self.btn_refresh_preflight = QPushButton("Refresh preflight")
        self.btn_refresh_preflight.clicked.connect(self.refresh_preflight)
        pre_btn_row.addWidget(self.btn_refresh_preflight)

        pre_btn_row.addSpacing(16)
        pre_btn_row.addWidget(QLabel("Context window tokens:"))
        self.budget_spin = QSpinBox()
        self.budget_spin.setRange(1024, 262144)
        self.budget_spin.setSingleStep(1024)
        self.budget_spin.setValue(100000)
        self.budget_spin.valueChanged.connect(self._schedule_preflight_refresh)
        pre_btn_row.addWidget(self.budget_spin)

        pre_btn_row.addStretch(1)
        pre_layout.addLayout(pre_btn_row)

        self.preflight_summary_form = QFormLayout()
        self.lbl_preflight_source = QLabel("-")
        self.lbl_preflight_question = QLabel("-")
        self.lbl_preflight_duration = QLabel("-")
        self.lbl_preflight_chunks = QLabel("-")
        self.lbl_preflight_budget = QLabel("-")
        self.lbl_preflight_warnings = QLabel("-")
        self.lbl_preflight_warnings.setWordWrap(True)
        self.lbl_preflight_warnings.setProperty("isWarning", "true")
        self.lbl_preflight_overflow = QLabel("-")
        self.lbl_preflight_overflow.setWordWrap(True)

        self.preflight_summary_form.addRow(QLabel("Source:"), self.lbl_preflight_source)
        self.preflight_summary_form.addRow(
            QLabel("Question:"), self.lbl_preflight_question
        )
        self.preflight_summary_form.addRow(
            QLabel("Duration:"), self.lbl_preflight_duration
        )
        self.preflight_summary_form.addRow(QLabel("Chunks:"), self.lbl_preflight_chunks)
        self.preflight_summary_form.addRow(QLabel("Budget:"), self.lbl_preflight_budget)
        self.preflight_summary_form.addRow(
            QLabel("Warnings:"), self.lbl_preflight_warnings
        )
        self.preflight_summary_form.addRow(
            QLabel("Overflow:"), self.lbl_preflight_overflow
        )

        summary_widget = QWidget()
        summary_widget.setLayout(self.preflight_summary_form)
        summary_widget.setStyleSheet(_PREFLIGHT_SUMMARY_STYLE)
        pre_layout.addWidget(summary_widget)

        self.preflight_edit = QPlainTextEdit()
        self.preflight_edit.setReadOnly(True)
        self.preflight_edit.setPlaceholderText(
            "Click “Refresh preflight” to estimate chunks and context budget."
        )
        self.preflight_edit.setMinimumHeight(120)
        self.preflight_edit.setStyleSheet(_READ_ONLY_STYLE)
        pre_layout.addWidget(self.preflight_edit)
        root.addWidget(pre_box)

    def _build_answer_evidence_group(self, root: QVBoxLayout) -> None:
        out_box = QGroupBox("Answer and evidence")
        out_box.setStyleSheet(_VIDEO_QA_PANEL_STYLE)
        out_layout = QVBoxLayout(out_box)
        out_layout.addWidget(QLabel("Answer (read-only until backend run):"))
        self.answer_edit = QPlainTextEdit()
        self.answer_edit.setReadOnly(True)
        self.answer_edit.setPlaceholderText(
            "Final answer appears here after a successful Video QA run."
        )
        self.answer_edit.setMinimumHeight(100)
        self.answer_edit.setStyleSheet(_ANSWER_STYLE)
        out_layout.addWidget(self.answer_edit)
        out_layout.addWidget(QLabel("Evidence (read-only until backend run):"))
        self.evidence_edit = QPlainTextEdit()
        self.evidence_edit.setReadOnly(True)
        self.evidence_edit.setPlaceholderText(
            "Evidence items (quotes, timecodes, frame refs) will appear here."
        )
        self.evidence_edit.setMinimumHeight(100)
        self.evidence_edit.setStyleSheet(_EVIDENCE_STYLE)
        out_layout.addWidget(self.evidence_edit)
        root.addWidget(out_box)

    def _build_run_placeholder(self, root: QVBoxLayout) -> None:
        run_row = QHBoxLayout()
        self.btn_run_qa = QPushButton("Run Video QA")
        self.btn_run_qa.setObjectName("video_qa_run")
        self.btn_run_qa.setEnabled(True)
        self.btn_run_qa.setToolTip(
            "Run ASR, sample chunk frames, call LM Studio per chunk, and synthesize "
            "one final answer (uses the main output directory)."
        )
        self.btn_run_qa.clicked.connect(self._emit_run_requested)
        self.btn_cancel_qa = QPushButton("Cancel")
        self.btn_cancel_qa.setObjectName("video_qa_cancel")
        self.btn_cancel_qa.setEnabled(False)
        self.btn_cancel_qa.setToolTip(
            "Request stop after the current step (LM Studio may finish the in-flight "
            "chunk first)."
        )
        self.btn_cancel_qa.clicked.connect(self._emit_cancel_requested)
        run_row.addWidget(self.btn_run_qa)
        run_row.addWidget(self.btn_cancel_qa)
        run_row.addStretch(1)
        root.addLayout(run_row)

    def _emit_run_requested(self) -> None:
        """Notify the main window that the user wants to start a Video QA run."""
        self.video_qa_run_requested.emit()

    def _emit_cancel_requested(self) -> None:
        """Notify the main window that the user wants to cancel the Video QA worker."""
        self.video_qa_cancel_requested.emit()

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
            self._schedule_preflight_refresh()
            return False

        try:
            source = self._provider.resolve(path)
        except (OSError, ValueError) as exc:
            self._source = None
            self.source_edit.setText(str(path))
            self.source_details.setText(f"Local file unavailable: {exc}")
            self._schedule_preflight_refresh()
            return False

        self._source = source
        self.source_edit.setText(str(source.path))
        self.source_details.setText(source.summary)
        self._schedule_preflight_refresh()
        return True

    def source_path(self) -> Path | None:
        """Return the resolved source path, if one is selected."""
        if self._source is None:
            return None
        return self._source.path

    def source(self) -> LocalFileSource | None:
        """Return the resolved local source metadata, if available."""
        return self._source

    def attachment_requests(self) -> list[VideoQAAttachmentRequest]:
        """Return attachment requests for rows whose paths resolve to existing files."""
        return list(self._iter_attachment_requests())

    def set_answer_text(self, text: str) -> None:
        """Set the read-only answer surface (for future backend wiring)."""
        self.answer_edit.setPlainText(text)

    def answer_text(self) -> str:
        """Return the current answer text."""
        return self.answer_edit.toPlainText()

    def set_evidence_items(self, items: list[str]) -> None:
        """Set evidence lines (for future backend wiring)."""
        self.evidence_edit.setPlainText("\n".join(items))

    def evidence_items(self) -> list[str]:
        """Return non-empty evidence lines."""
        return [
            ln.strip()
            for ln in self.evidence_edit.toPlainText().splitlines()
            if ln.strip()
        ]

    def context_window_tokens(self) -> int:
        """Return the current GUI budget limit in tokens."""
        return self.budget_spin.value()

    def set_context_window_tokens(self, tokens: int) -> None:
        """Set the current GUI budget limit."""
        self.budget_spin.setValue(tokens)

    def main_splitter_state(self) -> QByteArray:
        """Return the saved state for the main left/right splitter."""
        return self._main_splitter.saveState()

    def left_splitter_state(self) -> QByteArray:
        """Return the saved state for the attachments/preflight splitter."""
        return self._left_splitter.saveState()

    def restore_splitter_states(
        self,
        main_state: object | None,
        left_state: object | None,
    ) -> None:
        """Restore the panel splitter state from persisted settings."""
        if isinstance(main_state, QByteArray):
            with contextlib.suppress(Exception):
                self._main_splitter.restoreState(main_state)
        if isinstance(left_state, QByteArray):
            with contextlib.suppress(Exception):
                self._left_splitter.restoreState(left_state)

    def refresh_preflight(self) -> None:
        """Build and display a preflight report from the current shell state."""
        self._preflight_refresh_timer.stop()
        context = self.context_bundle()
        extra_warnings: list[str] = []
        duration_s = 0.0
        src = self.source_path()
        if src is None:
            extra_warnings.append("No local media source selected.")
        elif not src.exists():
            extra_warnings.append("Media source path is missing on disk.")
        else:
            duration_s = float(get_media_duration_seconds(src))
            if duration_s <= 0.0:
                extra_warnings.append(
                    "Media duration could not be read or is zero; preflight uses 0s."
                )

        budget_policy = default_video_qa_budget_policy()
        budget_policy = replace(
            budget_policy, context_window_tokens=self.budget_spin.value()
        )

        preflight = build_video_qa_preflight_summary(
            context,
            duration_seconds=duration_s,
            budget_policy=budget_policy,
        )
        if extra_warnings:
            merged = tuple(dict.fromkeys((*preflight.warnings, *extra_warnings)))
            preflight = replace(preflight, warnings=merged)

        report = build_video_qa_preflight_report(context, preflight)

        self.lbl_preflight_source.setText(report.source_summary or "(not selected)")
        self.lbl_preflight_question.setText(report.question.strip() or "(empty)")
        self.lbl_preflight_duration.setText(f"{duration_s:.2f}s")
        self.lbl_preflight_chunks.setText(str(report.chunk_count))
        self.lbl_preflight_budget.setText(report.budget_status_line)
        self.lbl_preflight_warnings.setText(
            "\n".join(report.warnings) if report.warnings else "none"
        )
        self.lbl_preflight_overflow.setText(report.overflow_fallback_explanation)

        self.preflight_edit.setPlainText(format_video_qa_preflight_report_text(report))

    def context_bundle(
        self,
        attachments: Iterable[str | Path | VideoQAAttachmentRequest] | None = None,
    ) -> VideoQAContextBundle:
        """Return a normalized prompt context bundle for the current shell.

        Args:
            attachments: Optional override iterable. When omitted, the panel's
                attachment table drives normalization.

        """
        if attachments is not None:
            return normalize_video_qa_context(
                source=self._source,
                question=self.question_text(),
                attachments=attachments,
            )
        return normalize_video_qa_context(
            source=self._source,
            question=self.question_text(),
            attachments=self._iter_attachment_requests(),
        )

    def url_import_policy(self) -> VideoQAUrlImportPolicy:
        """Return the backend-only URL import policy."""
        return self._url_import_policy

    def attachments_for_persistence(self) -> list[dict[str, Any]]:
        """Serialize attachment rows for QSettings."""
        rows: list[dict[str, Any]] = []
        for r in range(self._attachment_table.rowCount()):
            item = self._attachment_table.item(r, 1)
            if item is None:
                continue
            path_str = item.text().strip()
            if not path_str:
                continue
            w = self._attachment_table.cellWidget(r, 0)
            enabled = True
            if isinstance(w, QCheckBox):
                enabled = w.isChecked()
            rows.append({"path": path_str, "enabled": enabled})
        return rows

    def restore_attachments_state(self, entries: list[Any] | None) -> None:
        """Restore attachment rows from persisted data."""
        self._attachment_table.setRowCount(0)
        if not entries:
            self._schedule_preflight_refresh()
            return
        seen: set[str] = set()
        for raw in entries:
            if not isinstance(raw, dict):
                continue
            path_str = str(raw.get("path", "")).strip()
            if not path_str or path_str in seen:
                continue
            seen.add(path_str)
            enabled = bool(raw.get("enabled", True))
            self._add_attachment_row(path_str, enabled=enabled)
        self._schedule_preflight_refresh()

    def _sync_source_from_edit(self) -> None:
        """Sync the source state from the editable path field."""
        self.set_source_path(self.source_edit.text())

    def _schedule_preflight_refresh(self) -> None:
        """Queue a debounced preflight refresh after interactive changes."""
        self._preflight_refresh_timer.start()

    def _iter_attachment_requests(self) -> Iterator[VideoQAAttachmentRequest]:
        for row in range(self._attachment_table.rowCount()):
            item = self._attachment_table.item(row, 1)
            if item is None:
                continue
            path_str = item.text().strip()
            if not path_str:
                continue
            p = Path(path_str)
            w = self._attachment_table.cellWidget(row, 0)
            enabled = True
            if isinstance(w, QCheckBox):
                enabled = w.isChecked()
            if not p.is_file():
                continue
            yield VideoQAAttachmentRequest(path=p, enabled=enabled)

    def _add_attachment_row(self, path: str, *, enabled: bool) -> None:
        row = self._attachment_table.rowCount()
        self._attachment_table.insertRow(row)
        cb = QCheckBox()
        cb.setChecked(enabled)
        cb.stateChanged.connect(self._schedule_preflight_refresh)
        self._attachment_table.setCellWidget(row, 0, cb)
        cell = QTableWidgetItem(path)
        cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self._attachment_table.setItem(row, 1, cell)

    def _add_attachment_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Add attachment files",
            str(self._last_attachment_dir),
            _ATTACHMENT_NAME_FILTER,
        )
        existing: set[str] = set()
        for r in range(self._attachment_table.rowCount()):
            it = self._attachment_table.item(r, 1)
            if it is not None:
                existing.add(it.text())
        added_any = False
        for fn in files:
            p = Path(fn)
            if not p.is_file():
                continue
            self._last_attachment_dir = p.parent
            key = str(p.resolve())
            if key in existing:
                continue
            existing.add(key)
            self._add_attachment_row(key, enabled=True)
            added_any = True
        if added_any:
            self._schedule_preflight_refresh()

    def _remove_selected_attachments(self) -> None:
        rows = sorted(
            {i.row() for i in self._attachment_table.selectedIndexes()},
            reverse=True,
        )
        for r in rows:
            self._attachment_table.removeRow(r)
        if rows:
            self._schedule_preflight_refresh()
