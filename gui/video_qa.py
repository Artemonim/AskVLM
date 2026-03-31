from __future__ import annotations

import contextlib
import json
import os
import urllib.request
from dataclasses import dataclass, replace
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from PySide6.QtCore import QByteArray, Qt, QTimer, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.ffmpeg import get_media_duration_seconds
from core.video_qa_context import (
    VideoQAAttachmentRequest,
    normalize_video_qa_context,
)
from core.video_qa_local_run import (
    DEFAULT_LM_STUDIO_OPENAI_BASE_URL,
    DEFAULT_OPENROUTER_OPENAI_BASE_URL,
    OPENROUTER_API_KEY_ENV,
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
from utils.askvlm_defaults import get_default_video_qa_canonical_model_id

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from core.video_qa_context import VideoQAContextBundle
    from core.video_qa_policy import (
        VideoQAUrlImportPolicy,
    )
    from core.video_qa_sources import LocalFileSource

# * Preflight / run frame sampling rates (uniform grid budget and ffmpeg extraction).
VIDEO_QA_PREFLIGHT_FPS_CHOICES: Final[tuple[float, ...]] = (
    2.0,
    1.0,
    0.5,
    0.25,
    0.1,
    0.05,
)
FPS_VALUE_MATCH_EPSILON: Final[float] = 1e-9

# * Combo entries that are not real LM Studio model ids (errors / empty catalog).
_LOCAL_MODEL_NON_SELECTION_LABELS: Final[frozenset[str]] = frozenset(
    {"LM Studio not running/reachable", "No models found"}
)

_WARN_ROW_DIV_STYLE: Final[str] = (
    "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"
)


def _format_preflight_info_row_html(
    warnings: tuple[str, ...],
    overflow_explanation: str,
) -> str:
    """Build HTML for the preflight Info row (errors vs offline status text)."""
    div_open = f"<div style='{_WARN_ROW_DIV_STYLE}'>"
    if warnings:
        body = ", ".join(escape(str(w)) for w in warnings)
        warn_html = f"<span style='color:#f48771;font-weight:600;'>{body}</span>"
        return f"{div_open}{warn_html}</div>"
    ow = escape((overflow_explanation or "").strip())
    if ow:
        muted = f"<span style='color:#9d9d9d;'>{ow}</span>"
        return f"{div_open}{muted}</div>"
    muted = "<span style='color:#9d9d9d;'>—</span>"
    return f"{div_open}{muted}</div>"


@dataclass(frozen=True, slots=True)
class VideoQALMRuntimeSettings:
    """OpenAI-compatible HTTP target for one Video QA worker run."""

    lm_base_url: str
    lm_model_id: str
    lm_authorization_bearer: str | None = None


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
    "QTextEdit { background-color: #252526; color: #d4d4d4; "
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
    "QTableWidget { background-color: #1e1e1e; color: #d4d4d4; "
    "gridline-color: #3f3f46; border: 1px solid #3f3f46; }"
    "QHeaderView::section { background-color: #2d2d30; color: #d4d4d4; "
    "border: 1px solid #3f3f46; padding: 4px 8px; }"
    "QLineEdit, QComboBox { background-color: #1e1e1e; color: #d4d4d4; "
    "border: 1px solid #3f3f46; }"
    "QSpinBox { background-color: #1e1e1e; color: #d4d4d4; "
    "border: 1px solid #3f3f46; padding-left: 4px; }"
    "QPushButton { background-color: #2d2d30; color: #d4d4d4; "
    "border: 1px solid #3f3f46; padding: 6px 10px; }"
    "QPushButton:checked { background-color: #0e639c; color: #ffffff; "
    "border: 1px solid #1177bb; }"
    "QPushButton:disabled { color: #7a7a7a; background-color: #252526; }"
    "QCheckBox { color: #d4d4d4; }"
    "QSplitter::handle { background-color: #3f3f46; }"
    "QSlider::groove:horizontal { background: #3f3f46; height: 4px; "
    "border-radius: 2px; }"
    "QSlider::handle:horizontal { background: #0e639c; width: 14px; margin: -6px 0; "
    "border-radius: 7px; }"
    "QLabel { color: #d4d4d4; }"
)


class VideoQAPanel(QWidget):
    """Video QA workspace: source, question, attachments, and results."""

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

        self._build_source_row(left_layout)

        question_widget = self._create_question_area_widget()

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
        self._left_splitter.setSizes([200, 200])

        self._question_rest_splitter = QSplitter(Qt.Orientation.Vertical)
        self._question_rest_splitter.setChildrenCollapsible(False)
        self._question_rest_splitter.addWidget(question_widget)
        self._question_rest_splitter.addWidget(self._left_splitter)
        top_default = self._question_area_splitter_top_size()
        self._question_rest_splitter.setSizes([top_default, 320])
        self._question_rest_splitter.setStretchFactor(0, 0)
        self._question_rest_splitter.setStretchFactor(1, 1)
        left_layout.addWidget(self._question_rest_splitter, 1)

        self._build_answer_evidence_group(right_layout)
        self._build_run_placeholder(right_layout)

        self._main_splitter.addWidget(left_widget)
        self._main_splitter.addWidget(right_widget)
        self._main_splitter.setStretchFactor(0, 1)
        self._main_splitter.setStretchFactor(1, 1)
        self._main_splitter.setSizes([400, 400])

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

    def _build_source_row(self, root: QVBoxLayout) -> None:
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

    def _create_question_area_widget(self) -> QWidget:
        """Pack the question label and editor; height is adjusted via splitter."""
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Question:"))
        self.question_edit = QTextEdit()
        self.question_edit.setPlaceholderText(
            "Ask a question about the selected local file"
        )
        fm = QFontMetrics(self.question_edit.font())
        min_body = max(40, fm.lineSpacing() * 2 + 8)
        self.question_edit.setMinimumHeight(min_body)
        self.question_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self.question_edit.textChanged.connect(self._schedule_preflight_refresh)
        layout.addWidget(self.question_edit)
        return holder

    def _question_area_splitter_top_size(self) -> int:
        """Return default splitter height (question label + ~two editor lines)."""
        fm = QFontMetrics(self.question_edit.font())
        label_row = 22
        body = max(40, fm.lineSpacing() * 2 + 8)
        return label_row + body + 8

    def _build_attachments_group(self, root: QVBoxLayout) -> None:
        att_box = QGroupBox("Attachments")
        att_box.setStyleSheet(_VIDEO_QA_PANEL_STYLE)
        att_layout = QVBoxLayout(att_box)
        self._attachment_table = QTableWidget(0, 2)
        self._attachment_table.setHorizontalHeaderLabels(["Include", "Path"])
        self._attachment_table.setAlternatingRowColors(True)
        self._attachment_table.setMinimumHeight(100)
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
        self._build_preflight_toolbar(pre_layout)
        self._build_preflight_summary_section(pre_layout)
        root.addWidget(pre_box)

    def _build_preflight_toolbar(self, pre_layout: QVBoxLayout) -> None:
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
        self._setup_preflight_fps_row(pre_btn_row)
        pre_btn_row.addStretch(1)
        pre_layout.addLayout(pre_btn_row)

    def _build_preflight_summary_section(self, pre_layout: QVBoxLayout) -> None:
        self.preflight_summary_form = QFormLayout()
        self._setup_preflight_model_row()
        self._setup_preflight_summary_label_rows()
        summary_widget = QWidget()
        summary_widget.setLayout(self.preflight_summary_form)
        summary_widget.setStyleSheet(_PREFLIGHT_SUMMARY_STYLE)
        pre_layout.addWidget(summary_widget)
        self.preflight_edit = QPlainTextEdit()
        self.preflight_edit.setReadOnly(True)
        self.preflight_edit.setPlaceholderText(
            "Click “Refresh preflight” to estimate chunks and context budget."
        )
        self.preflight_edit.setMinimumHeight(60)
        self.preflight_edit.setStyleSheet(_READ_ONLY_STYLE)
        pre_layout.addWidget(self.preflight_edit)

    def _setup_preflight_model_row(self) -> None:
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["Local (LM Studio)", "Cloud"])
        self.model_combo = QComboBox()
        self.model_combo.setEditable(False)
        self.model_cloud_edit = QLineEdit()
        self.model_cloud_edit.setPlaceholderText("Enter cloud model name")
        self.model_cloud_edit.setVisible(False)
        self.btn_refresh_models = QPushButton("🔄")
        self.btn_refresh_models.setToolTip("Refresh models from LM Studio")
        self.btn_refresh_models.clicked.connect(self._refresh_local_models)
        self.model_type_combo.currentIndexChanged.connect(self._on_model_scope_changed)
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_type_combo)
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(self.btn_refresh_models)
        model_row.addWidget(self.model_cloud_edit, 1)
        self.preflight_summary_form.addRow(QLabel("Model:"), model_row)

    def _setup_preflight_summary_label_rows(self) -> None:
        self.lbl_preflight_question = QLabel("-")
        self.lbl_preflight_duration = QLabel("-")
        self.lbl_preflight_budget = QLabel("-")
        self.lbl_preflight_info = QLabel("-")
        self.lbl_preflight_info.setWordWrap(False)
        self.lbl_preflight_info.setTextFormat(Qt.TextFormat.RichText)
        self.preflight_summary_form.addRow(
            QLabel("Question:"), self.lbl_preflight_question
        )
        self.preflight_summary_form.addRow(
            QLabel("Video:"), self.lbl_preflight_duration
        )
        self.preflight_summary_form.addRow(QLabel("Budget:"), self.lbl_preflight_budget)
        self.preflight_summary_form.addRow(QLabel("Info:"), self.lbl_preflight_info)

    def _on_model_scope_changed(self, idx: int) -> None:
        """Show LM Studio model picker for local scope; text field for cloud."""
        is_local = idx == 0
        self.model_combo.setVisible(is_local)
        self.btn_refresh_models.setVisible(is_local)
        self.model_cloud_edit.setVisible(not is_local)

    def _setup_preflight_fps_row(self, row: QHBoxLayout) -> None:
        """Add fps toggles; each change runs preflight immediately."""
        row.addSpacing(12)
        self._fps_button_group = QButtonGroup(self)
        self._fps_button_group.setExclusive(True)
        for idx, fps_val in enumerate(VIDEO_QA_PREFLIGHT_FPS_CHOICES):
            btn = QPushButton(f"{fps_val:.2f}")
            btn.setCheckable(True)
            self._fps_button_group.addButton(btn, idx)
            row.addWidget(btn)
        first_fps_btn = self._fps_button_group.button(0)
        if first_fps_btn is not None:
            first_fps_btn.setChecked(True)
        self._fps_button_group.idClicked.connect(self._on_preflight_fps_group_changed)
        row.addWidget(QLabel("fps"))

    def _build_answer_evidence_group(self, root: QVBoxLayout) -> None:
        out_box = QGroupBox("Answer and progress")
        out_box.setStyleSheet(_VIDEO_QA_PANEL_STYLE)
        out_layout = QVBoxLayout(out_box)
        out_layout.addWidget(QLabel("Answer (markdown, read-only until backend run):"))
        self.answer_edit = QTextEdit()
        self.answer_edit.setReadOnly(True)
        self.answer_edit.setAcceptRichText(False)
        self.answer_edit.setPlaceholderText(
            "Final answer appears here after a successful Video QA run (Markdown)."
        )
        self.answer_edit.setMinimumHeight(60)
        self.answer_edit.setStyleSheet(_ANSWER_STYLE)

        self.progress_log_edit = QPlainTextEdit()
        self.progress_log_edit.setReadOnly(True)
        self.progress_log_edit.setPlaceholderText(
            "Pipeline stages and LM Studio steps are logged here during a run."
        )
        self.progress_log_edit.setMinimumHeight(60)
        self.progress_log_edit.setStyleSheet(_EVIDENCE_STYLE)

        self._answer_progress_splitter = QSplitter(Qt.Orientation.Vertical)
        self._answer_progress_splitter.setChildrenCollapsible(False)
        self._answer_progress_splitter.addWidget(self.answer_edit)
        self._answer_progress_splitter.addWidget(self.progress_log_edit)
        self._answer_progress_splitter.setSizes([500, 100])
        out_layout.addWidget(self._answer_progress_splitter, 1)

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
            "Request stop: skips further chunks, aborts an in-flight LM Studio HTTP "
            "request when the stack allows, and stops uploading more images for the "
            "current chunk."
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
        return self.question_edit.toPlainText()

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
        """Set the read-only answer surface as Markdown."""
        self.answer_edit.setMarkdown(text)

    def answer_text(self) -> str:
        """Return the current answer as Markdown when possible, else plain text."""
        md = self.answer_edit.toMarkdown()
        if md.strip():
            return md
        return self.answer_edit.toPlainText()

    def clear_progress_log(self) -> None:
        """Clear the pipeline progress log."""
        self.progress_log_edit.clear()

    def append_progress_log_line(self, line: str) -> None:
        """Append one line to the progress log and scroll to the end."""
        self.progress_log_edit.appendPlainText(line.rstrip("\n"))
        bar = self.progress_log_edit.verticalScrollBar()
        bar.setValue(bar.maximum())

    def set_evidence_items(self, items: list[str]) -> None:
        """Append grounded evidence lines to the progress log after a run."""
        if not items:
            return
        self.append_progress_log_line("")
        self.append_progress_log_line("--- Evidence (from answer bundle) ---")
        for ln in items:
            self.append_progress_log_line(ln)

    def evidence_items(self) -> list[str]:
        """Return an empty list; evidence is no longer edited in a dedicated field."""
        return []

    def answer_progress_splitter_state(self) -> QByteArray:
        """Return the saved state for the answer/progress splitter."""
        return self._answer_progress_splitter.saveState()

    def restore_answer_progress_splitter_state(self, state: object | None) -> None:
        """Restore the answer/progress splitter state."""
        if isinstance(state, QByteArray):
            with contextlib.suppress(Exception):
                self._answer_progress_splitter.restoreState(state)

    def context_window_tokens(self) -> int:
        """Return the current GUI budget limit in tokens."""
        return self.budget_spin.value()

    def frame_sample_fps(self) -> float:
        """Return the selected uniform frame sampling rate for preflight and runs."""
        bid = self._fps_button_group.checkedId()
        if bid < 0:
            return VIDEO_QA_PREFLIGHT_FPS_CHOICES[0]
        return VIDEO_QA_PREFLIGHT_FPS_CHOICES[bid]

    def set_frame_sample_fps(self, fps: float) -> None:
        """Select the sampling toggle that matches ``fps``, or default to 2.0 fps."""
        target = float(fps)
        for idx, val in enumerate(VIDEO_QA_PREFLIGHT_FPS_CHOICES):
            if abs(val - target) < FPS_VALUE_MATCH_EPSILON:
                btn = self._fps_button_group.button(idx)
                if btn is not None:
                    btn.setChecked(True)
                return
        fallback = self._fps_button_group.button(0)
        if fallback is not None:
            fallback.setChecked(True)

    def set_context_window_tokens(self, tokens: int) -> None:
        """Set the current GUI budget limit."""
        self.budget_spin.setValue(tokens)

    def lm_runtime_settings(self) -> VideoQALMRuntimeSettings:
        """Resolve base URL, model id, and optional Bearer for the model scope."""
        default_id = get_default_video_qa_canonical_model_id()
        if self.model_type_combo.currentIndex() == 0:
            base = DEFAULT_LM_STUDIO_OPENAI_BASE_URL
            raw = self.model_combo.currentText().strip()
            if not raw or raw in _LOCAL_MODEL_NON_SELECTION_LABELS:
                model_id = default_id
            else:
                model_id = raw
            return VideoQALMRuntimeSettings(base, model_id, None)
        base = DEFAULT_OPENROUTER_OPENAI_BASE_URL
        cloud_raw = self.model_cloud_edit.text().strip()
        model_id = cloud_raw if cloud_raw else default_id
        token = os.environ.get(OPENROUTER_API_KEY_ENV, "")
        stripped = token.strip()
        bearer = stripped if stripped else None
        return VideoQALMRuntimeSettings(base, model_id, bearer)

    def restore_video_qa_lm_ui(
        self,
        *,
        scope_index: int = 0,
        local_model_text: str = "",
        cloud_model_text: str = "",
    ) -> None:
        """Restore model scope widgets from persisted settings."""
        idx = 0 if int(scope_index) <= 0 else 1
        self.model_type_combo.blockSignals(True)  # noqa: FBT003
        self.model_type_combo.setCurrentIndex(idx)
        self.model_type_combo.blockSignals(False)  # noqa: FBT003
        self._on_model_scope_changed(idx)
        local_t = local_model_text.strip()
        if local_t:
            found = self.model_combo.findText(local_t, Qt.MatchFlag.MatchExactly)
            if found >= 0:
                self.model_combo.setCurrentIndex(found)
            else:
                self.model_combo.insertItem(0, local_t)
                self.model_combo.setCurrentIndex(0)
        self.model_cloud_edit.setText(cloud_model_text)

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
            budget_policy,
            context_window_tokens=self.budget_spin.value(),
            frame_sample_fps=self.frame_sample_fps(),
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

        self.lbl_preflight_question.setText(report.question.strip() or "(empty)")
        self.lbl_preflight_duration.setText(f"{duration_s:.2f}s")
        self.lbl_preflight_budget.setText(report.budget_status_line)
        self.lbl_preflight_info.setText(
            _format_preflight_info_row_html(
                report.warnings,
                report.overflow_fallback_explanation,
            )
        )

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

    def _on_preflight_fps_group_changed(self, _button_id: int) -> None:
        """Recompute preflight immediately when the sampling rate toggle changes."""
        self.refresh_preflight()

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

    def _refresh_local_models(self) -> None:
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:1234/v1/models", method="GET"
            )
            with urllib.request.urlopen(req, timeout=2.0) as response:  # noqa: S310
                data = json.loads(response.read().decode("utf-8"))
                models = [m["id"] for m in data.get("data", []) if "id" in m]
                self.model_combo.clear()
                if models:
                    self.model_combo.addItems(models)
                else:
                    self.model_combo.addItem("No models found")
        except Exception:  # noqa: BLE001
            self.model_combo.clear()
            self.model_combo.addItem("LM Studio not running/reachable")
