from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from PySide6.QtCore import (
    QByteArray,
    QObject,
    QSettings,
    QSize,
    Qt,
    QThread,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import (
    QDesktopServices,
    QFont,
    QFontDatabase,
    QIcon,
    QImage,
    QPainter,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QCompleter,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.ffmpeg import (
    extract_frame_to_file,
    get_media_duration_seconds,
    start_burn_process,
)
from core.pipelines import CancelledError, LocalPipeline
from core.video_qa_executor import VideoQAExecutorRunOutcome
from core.video_qa_local_run import (
    VideoQAPreflightBlockedError,
    ensure_local_video_qa_run_allowed,
    map_video_qa_progress_frac_to_200,
    preflight_local_video_qa,
)
from core.whisperx_wrapper import WhisperXWrapper
from editing.text_model import Document, TextSegment
from gui.speaker_sidebar import SpeakerSidebar
from gui.subtitle_preview import SubtitlePreview
from gui.video_qa import VideoQAPanel
from gui.video_qa_worker import VideoQALocalRunWorker
from gui.wysiwyg_editor import TableRow, WysiwygEditor
from utils.env import load_env_file
from utils.exporters import (
    SubtitleRules,
    append_askvlm_metadata_to_srt,
    export_document,
    export_srt_with_rules,
    extract_askvlm_metadata_from_srt,
    fill_empty_gaps_in_srt,
    strip_askvlm_metadata_from_srt,
)
from utils.logging import get_logger, setup_logging


def _format_eta(seconds: float) -> str:
    """Return HH:MM:SS or MM:SS string for ETA display."""
    try:
        total = int(max(0.0, float(seconds)))
    except Exception:  # noqa: BLE001
        total = 0
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class _InlineBurnWorker(QObject):
    """Background worker for cancellable ffmpeg burn-in operations (inline)."""

    progress = Signal(float, str)
    log = Signal(str)
    error = Signal(str)
    canceled = Signal()
    finished = Signal(list)

    def __init__(
        self,
        paths: list[Path],
        outd: Path,
        *,
        norm: bool,
        font: str | None,
        font_px: int | None = None,
    ) -> None:
        super().__init__()
        self._inputs = paths
        self._out_dir = outd
        self._normalize = norm
        self._font_name = font
        self._font_px = font_px
        self._cancel = False
        self._proc: Popen[bytes] | None = None

    def _is_canceled(self) -> bool:
        """Return True if cancellation was requested.

        * This helper is a separate method to avoid mypy flow analysis
        treating subsequent attribute checks as statically unreachable
        when the flag is tested earlier in the same function.
        """
        return bool(self._cancel)

    def request_cancel(self) -> None:
        self._cancel = True
        if self._proc is not None:
            with contextlib.suppress(Exception):
                self._proc.terminate()
                self._proc.wait(timeout=2)
            self._proc = None

    def run(self) -> None:  # noqa: C901, PLR0912, PLR0915
        try:
            outputs: list[str] = []
            total = max(1, len(self._inputs))
            start_time = time.time()
            for idx, media in enumerate(self._inputs):
                if self._is_canceled():
                    self.canceled.emit()
                    return
                prefix = f"[{idx + 1}/{total}] " if total > 1 else ""
                srt_path = self._out_dir / f"{media.stem}.srt"
                if not srt_path.exists():
                    self.log.emit(f"SRT not found for burn: {srt_path}")
                    continue
                burned_out = self._out_dir / f"{media.stem}_subbed.mp4"
                prog_file = self._out_dir / f".{media.stem}.ffprogress"
                with contextlib.suppress(Exception):
                    if prog_file.exists():
                        prog_file.unlink()
                self.progress.emit((idx + 0.1) / total, prefix + "Burning subtitles")
                # Force style when preview overrides size
                force_style = (
                    f"Fontsize={int(self._font_px or 0)},Outline=2,Shadow=0"
                    if (self._font_px or 0) > 0
                    else None
                )
                proc = start_burn_process(
                    media,
                    srt_path,
                    burned_out,
                    force_style,
                    normalize_audio=self._normalize,
                    font_name=self._font_name,
                    progress_path=prog_file,
                )
                self._proc = proc
                base = idx / total
                end = (idx + 1) / total
                duration_s = get_media_duration_seconds(media)

                def _parse(pf: Path = prog_file) -> float:
                    try:
                        last = -1.0
                        for line in pf.read_text(
                            encoding="utf-8", errors="ignore"
                        ).splitlines():
                            if line.startswith("out_time_ms="):
                                last = float(line.split("=", 1)[1])
                    except (OSError, ValueError):
                        return -1.0
                    return last

                shown = base
                poll_sleep_s = 0.25
                while True:
                    if self._is_canceled():
                        # Try best-effort termination of ffmpeg process
                        with contextlib.suppress(OSError):
                            proc.terminate()
                        self.canceled.emit()
                        break
                    ret = proc.poll()
                    if ret is not None:
                        if ret != 0:
                            self.log.emit("Burn-in failed")
                            try:
                                if burned_out.exists():
                                    burned_out.unlink()
                            except OSError:
                                pass
                        else:
                            outputs.append(str(burned_out))
                        break
                    p_ms = _parse()
                    if p_ms >= 0 and duration_s > 0:
                        ratio = min(1.0, (p_ms / 1000.0) / duration_s)
                        shown = min(0.995, base + ratio * (end - base))
                        # Compute ETA
                        elapsed = time.time() - start_time
                        processed = max(1e-6, (idx + ratio))
                        total_units = float(total)
                        est_total = elapsed / processed * total_units
                        eta = max(0.0, est_total - elapsed)
                        eta_str = _format_eta(eta)
                        self.progress.emit(
                            shown, prefix + f"Burning subtitles (ETA {eta_str})"
                        )
                    else:
                        shown = min(0.995, shown + 0.02)
                    # Provide heartbeat when no structured progress is available
                    self.progress.emit(shown, prefix + "Burning subtitles")
                    time.sleep(poll_sleep_s)
            self.finished.emit(outputs)
        except Exception as e:  # noqa: BLE001
            # Surface error details in both status bar and log
            msg = str(e)
            get_logger(__name__).error("Processing error: %s", msg)
            self.error.emit(msg)


if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Future
    from subprocess import Popen

    from PySide6.QtGui import QCloseEvent

    from core.video_qa_context import VideoQAContextBundle

# * Constants for boolean UI states
_UI_CHECKED = True
_UI_UNCHECKED = False


class CancelledByUserError(Exception):
    """Raised internally to abort processing when user requests cancel."""


# * Main window for AskVLM GUI
class MainWindow(QMainWindow):
    """Application main window with Quick Transcribe controls and text viewer."""

    # * Constants
    MAX_SPEAKER_LEN = 64
    SHELL_SCREEN_TEXT = "text_subtitles"
    SHELL_SCREEN_VIDEO_QA = "video_qa"

    def _can_show_modal(self) -> bool:
        """Return True if it is safe to show modal dialogs (not under pytest/CI)."""
        if os.getenv("PYTEST_CURRENT_TEST") is not None:
            return False
        suppress = os.getenv("SK_SUPPRESS_DIALOGS", "").lower()
        return suppress not in {"1", "true", "yes"}

    def __init__(self) -> None:  # noqa: PLR0915
        super().__init__()
        # Ensure logging to console for high-level operations
        with contextlib.suppress(Exception):
            setup_logging()
        get_logger(__name__).info("MainWindow initializing")
        self.setWindowTitle("AskVLM")
        self.resize(800, 600)
        # * Hide the native menu bar so mode tabs sit directly under the title bar.
        self.menuBar().setVisible(False)

        # * Central widget and main vertical layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        shell_layout = QVBoxLayout(central_widget)

        # * Top-level shell keeps Text + Subtitles separated from Video QA
        self.shell_tabs = QTabWidget()
        shell_layout.addWidget(self.shell_tabs, 1)

        workspace_tab = QWidget()
        workspace_layout = QVBoxLayout(workspace_tab)
        self.shell_tabs.addTab(workspace_tab, "Text + Subtitles")

        self.video_qa_panel = VideoQAPanel()
        self.shell_tabs.addTab(self.video_qa_panel, "Video QA")

        layout = workspace_layout

        # * Output directory row
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output:"))
        self.out_dir_edit = QLineEdit(str((Path.cwd() / "transcriptions").resolve()))
        self.out_dir_btn = QPushButton("Browse…")
        out_row.addWidget(self.out_dir_edit, 1)
        out_row.addWidget(self.out_dir_btn)
        self.btn_open_out = QPushButton("Open...")
        out_row.addWidget(self.btn_open_out)
        layout.addLayout(out_row)

        # * Options row: toggles and format
        opts_row = QHBoxLayout()
        self.chk_diar = QCheckBox("Diarization")
        self.chk_diar.setChecked(_UI_UNCHECKED)
        self.chk_dialog = QCheckBox("Dialog blocks")
        self.chk_dialog.setChecked(_UI_UNCHECKED)
        self.chk_save_srt = QCheckBox("Also save .srt")
        self.chk_save_srt.setChecked(_UI_CHECKED)
        opts_row.addWidget(self.chk_diar)
        opts_row.addWidget(self.chk_dialog)
        opts_row.addWidget(self.chk_save_srt)
        opts_row.addWidget(QLabel("Format:"))
        # Option: avoid empty gaps in generated SRT (stretch previous cue)
        self.chk_no_empty = QCheckBox("No empty")
        self.chk_no_empty.setChecked(False)
        opts_row.addWidget(self.chk_no_empty)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["none", "txt", "srt", "vtt", "json"])
        self.format_combo.setCurrentText("srt")
        opts_row.addWidget(self.format_combo)
        opts_row.addStretch(1)
        # Quality toggle and Start/Cancel
        self.btn_quality = QPushButton("Quality: Good")
        self._quality_mode: Literal["good", "fast"] = "good"
        opts_row.addWidget(self.btn_quality)
        self.btn_start = QPushButton("Start")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        opts_row.addWidget(self.btn_start)
        opts_row.addWidget(self.btn_cancel)
        layout.addLayout(opts_row)

        # * Burn-in row
        burn_row = QHBoxLayout()
        self.btn_burn = QPushButton("Burn")
        self.chk_normalize = QCheckBox("Normalization")
        self.chk_normalize.setChecked(_UI_CHECKED)
        burn_row.addWidget(self.btn_burn)
        burn_row.addWidget(self.chk_normalize)
        # Move Line length controls into burn row
        burn_row.addWidget(QLabel("Line length:"))
        self.spin_line_len = QSpinBox()
        self.spin_line_len.setRange(20, 120)
        self.spin_line_len.setValue(42)
        burn_row.addWidget(self.spin_line_len)
        burn_row.addWidget(QLabel("Lines:"))
        self.spin_max_lines = QSpinBox()
        self.spin_max_lines.setRange(1, 3)
        self.spin_max_lines.setValue(2)
        burn_row.addWidget(self.spin_max_lines)
        burn_row.addWidget(QLabel("Font:"))
        self.font_combo = QComboBox()
        try:
            families = QFontDatabase.families()
            self.font_combo.addItems(families)
            idx = self.font_combo.findText("Open Sans")
            if idx >= 0:
                self.font_combo.setCurrentIndex(idx)
            else:
                idx2 = self.font_combo.findText("Arial")
                if idx2 >= 0:
                    self.font_combo.setCurrentIndex(idx2)
            # Enable type-to-filter with popup completer
            self.font_combo.setEditable(True)
            completer = QCompleter(self.font_combo.model(), self.font_combo)
            completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
            completer.setFilterMode(Qt.MatchFlag.MatchContains)
            completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
            self.font_combo.setCompleter(completer)
        except Exception as exc:  # noqa: BLE001
            get_logger(__name__).debug("Font families enumeration failed: %s", exc)
        burn_row.addWidget(self.font_combo)
        # Font size controls
        burn_row.addWidget(QLabel("Size:"))
        self.btn_font_dec = QPushButton("-")
        self.btn_font_inc = QPushButton("+")
        burn_row.addWidget(self.btn_font_dec)
        burn_row.addWidget(self.btn_font_inc)
        burn_row.addStretch(1)
        layout.addLayout(burn_row)

        # * Left tabs (Speakers, Input) + center editors + right preview
        self.splitter = QSplitter()
        # Left tabs
        self.left_tabs = QTabWidget()
        # Input tab (make first)
        input_tab = QWidget()
        in_layout = QVBoxLayout(input_tab)
        in_layout.setContentsMargins(4, 4, 4, 4)
        # * Input list as two-column table: [status icon] [path]
        self.input_list = QTableWidget(0, 2)
        self.input_list.setHorizontalHeaderLabels(["", "Path"])  # icon, text
        with contextlib.suppress(Exception):
            # Small icon column
            self.input_list.setColumnWidth(0, 24)
            self.input_list.horizontalHeader().setSectionResizeMode(
                0, QHeaderView.ResizeMode.Fixed
            )
            self.input_list.horizontalHeader().setSectionResizeMode(
                1, QHeaderView.ResizeMode.Stretch
            )
            self.input_list.horizontalHeader().setStretchLastSection(True)
            # Do not elide long paths
            self.input_list.setTextElideMode(Qt.TextElideMode.ElideNone)
            # Ensure icons are visible and rows tall enough
            self.input_list.setIconSize(QSize(16, 16))
            self.input_list.verticalHeader().setDefaultSectionSize(22)
        # Select entire rows; allow multi-select
        self.input_list.setSelectionBehavior(
            self.input_list.SelectionBehavior.SelectRows
        )
        self.input_list.setSelectionMode(
            self.input_list.SelectionMode.ExtendedSelection
        )
        # Paths are not edited in-place
        self.input_list.setEditTriggers(self.input_list.EditTrigger.NoEditTriggers)
        in_layout.addWidget(self.input_list, 1)
        in_controls = QHBoxLayout()
        self.btn_in_add_file = QPushButton("Add File(s)…")
        self.btn_in_remove = QPushButton("Remove")
        self.btn_in_reset = QPushButton("Reset")
        for b in (self.btn_in_add_file, self.btn_in_remove, self.btn_in_reset):
            in_controls.addWidget(b)
        in_controls.addStretch(1)
        in_layout.addLayout(in_controls)
        self.left_tabs.addTab(input_tab, "Input")
        # Speakers tab (second), add Sidebar directly without legacy label/wrapper
        speakers_tab = QWidget()
        sp_layout = QVBoxLayout(speakers_tab)
        sp_layout.setContentsMargins(0, 0, 0, 0)
        self.sidebar = SpeakerSidebar()
        sp_layout.addWidget(self.sidebar)
        self.left_tabs.addTab(speakers_tab, "Speakers")
        self.splitter.addWidget(self.left_tabs)
        # Center editors
        self.tabs = QTabWidget()
        self.splitter.addWidget(self.tabs)
        # Right preview
        self.preview = SubtitlePreview()
        self.splitter.addWidget(self.preview)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setStretchFactor(2, 1)
        layout.addWidget(self.splitter, 1)

        # * Status bar with progress
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status.addPermanentWidget(self.progress)
        self.status.showMessage("Ready")

        # * Runtime/session state (respect UI defaults)
        self.pipeline = LocalPipeline(
            enable_diarization=bool(self.chk_diar.isChecked()),
            enable_dialog_blocks=bool(self.chk_dialog.isChecked()),
        )
        self.input_mode: Literal["file", "folder"] | None = None
        self.input_path: Path | None = None
        self.last_input_dir: Path = Path.cwd()
        self.last_output_dir: Path = Path(self.out_dir_edit.text()).resolve()
        self._thread: QThread | None = None
        self._worker: PipelineWorker | None = None
        self._burn_thread: QThread | None = None
        self._burn_worker: object | None = None
        self._video_qa_thread: QThread | None = None
        self._video_qa_worker: VideoQALocalRunWorker | None = None
        self._has_transcript: bool = False
        # * Preview mapping and state
        self._last_inputs: list[Path] = []
        self._tab_to_media: dict[str, Path] = {}
        # * Input status state and icon cache
        self._input_status: dict[str, str] = {}
        self._status_icon_cache: dict[str, QIcon] = {}
        # * Overlay state: per-input secondary icon (session done / spinner)
        self._input_overlay: dict[str, str] = {}
        self._composite_icon_cache: dict[str, QIcon] = {}
        self._spinner_phase: int = 0
        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(1000)
        self._spinner_timer.timeout.connect(self._on_spinner_tick)

        # * Wire up signals
        # (Removed legacy top input actions)
        self.out_dir_btn.clicked.connect(
            self._log_wrap(self.choose_output_dir, "Choose Output Dir")
        )
        self.btn_start.clicked.connect(self._log_wrap(self.start_processing, "Start"))
        self.btn_cancel.clicked.connect(self._log_wrap(self.request_cancel, "Cancel"))
        self.btn_open_out.clicked.connect(
            self._log_wrap(self.open_output_folder, "Open Output Folder")
        )
        self.btn_burn.clicked.connect(self._log_wrap(self.start_burn, "Burn"))
        # Input tab actions
        self.btn_in_add_file.clicked.connect(
            self._log_wrap(self._input_add_file, "Input Add File")
        )
        self.btn_in_remove.clicked.connect(
            self._log_wrap(self._input_remove_selected, "Input Remove")
        )
        self.btn_in_reset.clicked.connect(
            self._log_wrap(self._input_reset_status_selected, "Input Reset")
        )
        # Double-click to open SRT tab for a row
        self.input_list.itemDoubleClicked.connect(self._on_input_item_double_clicked)
        # Quality toggle
        self.btn_quality.clicked.connect(
            self._log_wrap(self._toggle_quality, "Toggle Quality")
        )
        # Preview updates on options change
        self.spin_line_len.valueChanged.connect(
            lambda _v: self._update_preview_for_selection()
        )
        self.spin_max_lines.valueChanged.connect(
            lambda _v: self._update_preview_for_selection()
        )
        self.font_combo.currentTextChanged.connect(self.preview.set_font_family)
        self.btn_font_dec.clicked.connect(lambda: self._nudge_font_size(-2))
        self.btn_font_inc.clicked.connect(lambda: self._nudge_font_size(+2))

        # * Start with no tabs; content tabs are added upon results or user action
        self._clear_tabs()

        # * Connect editor selection changes to preview updates
        self.tabs.currentChanged.connect(self._on_tab_changed)
        # React to output dir change to rescan statuses
        self.out_dir_edit.textChanged.connect(lambda _t: self._scan_output_statuses())

        # * Load persisted settings
        self._load_settings()
        # Burn is disabled until transcript exists
        self.btn_burn.setEnabled(False)
        # Initialize preview font
        self.preview.set_font_family(self.font_combo.currentText())
        self._font_px: int | None = None
        # Apply initial quality
        self._apply_quality_to_pipeline()
        # Initial status scan
        self._scan_output_statuses()

        self.video_qa_panel.video_qa_run_requested.connect(
            self._log_wrap(self._on_video_qa_run_requested, "Video QA run")
        )
        self.video_qa_panel.video_qa_cancel_requested.connect(
            self._log_wrap(self._on_video_qa_cancel_requested, "Video QA cancel")
        )

    def choose_file(self) -> None:
        """Choose a single media file for processing."""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Media File",
            str(self.last_input_dir),
            "Media Files (*.wav *.mp3 *.mp4 *.avi *.mkv)",
        )
        if file_name:
            p = Path(file_name)
            self.input_mode = "file"
            self.input_path = p
            self.last_input_dir = p.parent
            self.status.showMessage(f"Selected file: {p}")

    def choose_folder(self) -> None:
        """Choose an input folder for batch processing."""
        folder = QFileDialog.getExistingDirectory(
            self, "Choose Input Folder", str(self.last_input_dir)
        )
        if folder:
            p = Path(folder)
            self.input_mode = "folder"
            self.input_path = p
            self.last_input_dir = p
            self.status.showMessage(f"Selected folder: {p}")

    def choose_output_dir(self) -> None:
        """Choose the output directory."""
        folder = QFileDialog.getExistingDirectory(
            self, "Choose Output Directory", str(self.last_output_dir)
        )
        if folder:
            p = Path(folder)
            self.out_dir_edit.setText(str(p))
            self.last_output_dir = p
            self.status.showMessage(f"Output directory: {p}")
            # Rescan statuses for the new output folder
            self._scan_output_statuses()

    def _set_controls_enabled(self, enabled: bool) -> None:  # noqa: FBT001
        if hasattr(self, "btn_choose_file"):
            self.btn_choose_file.setEnabled(enabled)
        if hasattr(self, "btn_choose_folder"):
            self.btn_choose_folder.setEnabled(enabled)
        self.out_dir_btn.setEnabled(enabled)
        self.out_dir_edit.setEnabled(enabled)
        self.chk_diar.setEnabled(enabled)
        self.chk_dialog.setEnabled(enabled)
        self.chk_save_srt.setEnabled(enabled)
        self.format_combo.setEnabled(enabled)
        self.btn_quality.setEnabled(enabled)
        self.btn_start.setEnabled(enabled)
        self.btn_cancel.setEnabled(not enabled)
        if self._burn_worker is None:
            self.btn_burn.setEnabled(enabled)
            self.chk_normalize.setEnabled(enabled)
            self.font_combo.setEnabled(enabled)
        vq_idle = self._video_qa_thread is None or (
            not self._video_qa_thread.isRunning()
        )
        self.video_qa_panel.btn_run_qa.setEnabled(bool(enabled) and vq_idle)

    def start_processing(self) -> None:  # noqa: PLR0915, C901
        """Start pipeline processing in a background thread (QThread)."""
        if self._video_qa_thread is not None and self._video_qa_thread.isRunning():
            QMessageBox.information(
                self,
                "Busy",
                "Video QA is running. Wait for it to finish or cancel it.",
            )
            return
        # Validate input
        inputs = self._gather_inputs()
        if not inputs:
            QMessageBox.information(
                self,
                "No input",
                "Please add inputs in the Input tab.",
            )
            return
        # Apply restrictions based on current statuses
        allowed: list[Path] = []
        skipped_restricted = 0
        for p in inputs:
            st = self._get_input_status(p)
            if st == "good":
                skipped_restricted += 1
                continue
            if st == "fast" and self._quality_mode == "fast":
                skipped_restricted += 1
                continue
            allowed.append(p)
        if not allowed:
            QMessageBox.information(
                self,
                "Nothing to process",
                (
                    "All inputs are already processed under current quality. "
                    "Switch to Good or use Burn."
                ),
            )
            return
        if skipped_restricted > 0:
            self.status.showMessage(
                f"Skipping {skipped_restricted} input(s) due to existing status"
            )
        # * Remember inputs for preview/tab mapping
        self._last_inputs = allowed
        out_dir = Path(self.out_dir_edit.text()).resolve()
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, "Output error", f"Cannot create output dir: {e}")
            return

        # Apply model quality on each run
        self._apply_quality_to_pipeline(force_reload=True)

        # Configure worker options
        opts = {
            "enable_diarization": self.chk_diar.isChecked(),
            "enable_dialog_blocks": self.chk_dialog.isChecked(),
            "export_format": str(self.format_combo.currentText()),
            "single_view": len(inputs) == 1,
            "save_srt": self.chk_save_srt.isChecked(),
            # * Subtitle readability options for exporters
            "subtitle_max_line_width": int(self.spin_line_len.value()),
            "subtitle_max_lines": int(self.spin_max_lines.value()),
            # * Quality metadata for SRT
            "quality": self._quality_mode,
            # * NoEmpty: stretch cues to next start to avoid gaps
            "no_empty": bool(self.chk_no_empty.isChecked()),
        }

        # Ensure pipeline flags reflect current UI before processing
        self.pipeline.enable_diarization = bool(self.chk_diar.isChecked())
        self.pipeline.enable_dialog_blocks = bool(self.chk_dialog.isChecked())

        # Spin up worker and thread
        self._set_controls_enabled(enabled=False)
        self.progress.setValue(0)
        self.status.showMessage("Processing…")
        # Reset transcript availability
        self._has_transcript = False
        self.btn_burn.setEnabled(_UI_UNCHECKED)
        # Prepare spinner animation; overlays are set per-file on start events
        self._input_overlay.clear()
        self._spinner_phase = 0
        self._spinner_timer.start()
        # Ensure previous thread is fully stopped before creating new one
        if self._thread is not None:
            try:
                self._thread.quit()
                self._thread.wait(2000)
            except Exception as exc:  # noqa: BLE001
                # * Best-effort cleanup; log for diagnostics
                get_logger(__name__).debug("Thread cleanup issue: %s", exc)
            self._thread = None
        self._thread = QThread(self)
        self._worker = PipelineWorker(self.pipeline, allowed, out_dir, opts)
        self._worker.moveToThread(self._thread)

        # Connect signals
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.on_progress)
        self._worker.log.connect(self.on_log)
        self._worker.error.connect(self.on_error)
        self._worker.canceled.connect(self.on_canceled)
        self._worker.finished.connect(self.on_finished)
        # Per-file lifecycle
        self._worker.file_started.connect(self._on_file_started)
        self._worker.file_finished.connect(self._on_file_finished)
        # Ensure cleanup
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.canceled.connect(self._thread.quit)
        self._worker.canceled.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._on_worker_thread_finished)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    def request_cancel(self) -> None:
        """Request cancellation from the worker (best effort)."""
        worker = self._worker
        if worker is not None:
            try:
                worker.request_cancel()
            except RuntimeError:
                # * Under some shutdown sequences the underlying C++ object
                # * is already deleted by Qt; treat this as "already canceled".
                get_logger(__name__).debug(
                    "Worker object already deleted when requesting cancel",
                )
                self._worker = None
            else:
                with contextlib.suppress(Exception):
                    get_logger(__name__).info("Cancel signal forwarded to worker")
                self.status.showMessage("Cancel requested…")
                self.btn_cancel.setEnabled(_UI_UNCHECKED)
        if isinstance(self._burn_worker, _InlineBurnWorker):
            with contextlib.suppress(RuntimeError):
                self._burn_worker.request_cancel()
        vq = self._video_qa_worker
        if vq is not None:
            with contextlib.suppress(RuntimeError):
                vq.request_cancel()

    def on_progress(self, frac: float, msg: str) -> None:
        """Update progress bar and status message."""
        vqa_running = (
            self._video_qa_thread is not None and self._video_qa_thread.isRunning()
        )
        if vqa_running:
            self.progress.setMaximum(200)
            pct200 = map_video_qa_progress_frac_to_200(frac)
            self.progress.setValue(pct200)
            suffix = f" — {msg}" if msg else ""
            self.status.showMessage(f"{pct200}/200 ({pct200 // 2}%){suffix}")
            return
        self.progress.setMaximum(100)
        self.progress.setValue(int(max(0.0, min(1.0, frac)) * 100))
        if msg:
            self.status.showMessage(msg)

    def on_log(self, line: str) -> None:
        """Handle log message from worker (currently unused)."""
        # Minimal Phase 1.7: surface critical warnings in GUI
        if not line:
            return
        # Mirror into console logger for visibility when launching from terminal
        get_logger(__name__).info("Worker: %s", line)
        if (
            "Burn-in failed" in line or "SRT export failed" in line
        ) and self._can_show_modal():
            QMessageBox.warning(self, "Processing warning", line)
        # Update status with last log line for visibility
        self.status.showMessage(line)
        # * Detect per-file errors to mark input status as error
        m = re.search(r"Skipping '([^']+)'\:", line)
        if m:
            with contextlib.suppress(OSError):
                err_path = Path(m.group(1)).resolve()
                self._set_input_status(err_path, "error")

    def on_error(self, message: str) -> None:
        """Handle processing error."""
        self._set_controls_enabled(enabled=True)
        self.progress.setValue(0)
        # Ensure spinner stops to avoid timer callbacks during shutdown
        with contextlib.suppress(Exception):
            self._spinner_timer.stop()
        # Also print to console for visibility when launched from terminal
        with contextlib.suppress(Exception):
            pass
        if self._can_show_modal():
            QMessageBox.critical(self, "Processing error", message)
        self.status.showMessage("Error")

    def on_canceled(self) -> None:
        """Handle processing cancellation."""
        # Re-enable controls only after worker reported cancellation completion
        self._set_controls_enabled(enabled=True)
        self.progress.setValue(0)
        self.status.showMessage("Canceled")
        # Stop spinner once cancellation is fully handled
        with contextlib.suppress(Exception):
            self._spinner_timer.stop()

    def _re_enable_after_video_qa(self) -> None:
        """Restore main controls after a Video QA worker ends."""
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        self.btn_start.setEnabled(True)
        self.video_qa_panel.btn_run_qa.setEnabled(True)
        self.video_qa_panel.btn_cancel_qa.setEnabled(False)
        self.btn_cancel.setEnabled(False)

    def _on_video_qa_run_requested(self) -> None:
        """Validate preflight and start the Video QA background worker."""
        vq_log = get_logger(__name__)
        vq_log.info("Video QA: run requested (preflight path)")
        if self._thread is not None and self._thread.isRunning():
            self.status.showMessage("Video QA error")
            QMessageBox.information(
                self,
                "Busy",
                "Subtitle processing is running. Wait for it to finish or cancel it.",
            )
            return
        if self._video_qa_thread is not None and self._video_qa_thread.isRunning():
            return
        ctx = self.video_qa_panel.context_bundle()
        if ctx.source is None:
            self.status.showMessage("Video QA error")
            QMessageBox.warning(self, "Video QA", "Select a local media file first.")
            return
        if not str(ctx.question or "").strip():
            self.status.showMessage("Video QA error")
            QMessageBox.warning(self, "Video QA", "Enter a question.")
            return
        out_dir = Path(self.out_dir_edit.text()).resolve()
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.status.showMessage("Video QA error")
            QMessageBox.critical(self, "Output error", f"Cannot use output dir: {exc}")
            return
        try:
            duration_s = float(get_media_duration_seconds(ctx.source.path))
        except OSError:
            duration_s = 0.0
        report, chunk_plan = preflight_local_video_qa(
            ctx,
            duration_seconds=duration_s,
            context_window_tokens=self.video_qa_panel.context_window_tokens(),
            frame_sample_fps=self.video_qa_panel.frame_sample_fps(),
            video_chunking_enabled=self.video_qa_panel.video_chunking_enabled(),
        )
        try:
            ensure_local_video_qa_run_allowed(report, chunk_plan)
        except VideoQAPreflightBlockedError as exc:
            vq_log.info("Video QA: preflight blocked — %s", exc)
            self.status.showMessage("Video QA error")
            QMessageBox.warning(self, "Video QA", str(exc))
            return
        vq_log.info(
            "Video QA: preflight passed (chunks=%d), starting worker",
            len(chunk_plan),
        )
        self._start_video_qa_worker(ctx, out_dir)

    def _on_video_qa_cancel_requested(self) -> None:
        """Cooperatively cancel the Video QA worker if it is running."""
        vq = self._video_qa_worker
        if vq is None:
            return
        if self._video_qa_thread is None or not self._video_qa_thread.isRunning():
            return
        with contextlib.suppress(RuntimeError):
            vq.request_cancel()
        self.status.showMessage("Video QA cancel requested…")

    def _start_video_qa_worker(
        self,
        ctx: VideoQAContextBundle,
        out_dir: Path,
    ) -> None:
        """Spin up :class:`VideoQALocalRunWorker` on a dedicated thread."""
        vq_log = get_logger(__name__)
        thread: QThread | None = None
        worker: VideoQALocalRunWorker | None = None
        try:
            self._apply_quality_to_pipeline(force_reload=False)
            thread = QThread(self)
            lm = self.video_qa_panel.lm_runtime_settings_pair()
            worker = VideoQALocalRunWorker(
                context=ctx,
                output_dir=out_dir,
                context_window_tokens=self.video_qa_panel.context_window_tokens(),
                frame_sample_fps=self.video_qa_panel.frame_sample_fps(),
                whisper=self.pipeline.whisperx,
                chunk_lm=lm.chunk,
                final_lm=lm.final_answer,
                video_chunking_enabled=self.video_qa_panel.video_chunking_enabled(),
                run_options=self.video_qa_panel.video_qa_local_run_options(),
            )
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.progress.connect(self.on_progress)
            worker.pipeline_log_line.connect(
                self.video_qa_panel.append_progress_log_line
            )
            worker.finished.connect(self._on_video_qa_finished)
            worker.error.connect(self._on_video_qa_error)
            worker.canceled.connect(self._on_video_qa_canceled)
            worker.error.connect(worker.deleteLater)
            # * Mirror finished/canceled: quit the thread when the worker errors so the
            # * QThread does not rely only on a queued main-thread slot ordering.
            worker.error.connect(thread.quit)
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            worker.canceled.connect(thread.quit)
            worker.canceled.connect(worker.deleteLater)
            thread.finished.connect(self._on_video_qa_worker_thread_finished)
            thread.finished.connect(thread.deleteLater)
            self._video_qa_thread = thread
            self._video_qa_worker = worker
            self.btn_start.setEnabled(False)
            self.video_qa_panel.btn_run_qa.setEnabled(False)
            self.btn_cancel.setEnabled(False)
            self.video_qa_panel.btn_cancel_qa.setEnabled(True)
            self.progress.setMaximum(200)
            self.progress.setValue(0)
            self.video_qa_panel.clear_progress_log()
            self.status.showMessage("Video QA…")
            vq_log.info("Video QA: worker thread starting (out_dir=%s)", out_dir)
            thread.start()
        except Exception as exc:  # noqa: BLE001
            vq_log.exception("Video QA: failed to start worker thread")
            self._video_qa_worker = None
            self._video_qa_thread = None
            if worker is not None:
                worker.deleteLater()
            if thread is not None:
                thread.deleteLater()
            if self._can_show_modal():
                QMessageBox.critical(
                    self,
                    "Video QA",
                    f"Could not start Video QA. See the log for details.\n\n{exc!s}",
                )
            self._re_enable_after_video_qa()

    def _on_video_qa_finished(self, outcome: object) -> None:
        """Populate answer/evidence from a successful Video QA run."""
        if isinstance(outcome, VideoQAExecutorRunOutcome):
            self.video_qa_panel.set_answer_text(outcome.answer_bundle.answer)
            items = [
                f"[{ev.t_start:.2f}s - {ev.t_end:.2f}s] {ev.transcript_quote}"
                for ev in outcome.answer_bundle.evidence
            ]
            self.video_qa_panel.set_evidence_items(items)
        self.status.showMessage("Video QA completed")

    def _on_video_qa_error(self, message: str) -> None:
        """Surface a Video QA failure and stop the worker thread."""
        if self._can_show_modal():
            QMessageBox.warning(self, "Video QA", message)
        self.status.showMessage("Video QA error")
        if self._video_qa_thread is not None:
            self._video_qa_thread.quit()

    def _on_video_qa_canceled(self) -> None:
        """Reset UI after cooperative Video QA cancellation."""
        self.status.showMessage("Video QA canceled")
        if self._video_qa_thread is not None:
            self._video_qa_thread.quit()

    def _on_video_qa_worker_thread_finished(self) -> None:
        """Clear Video QA thread references after the thread stops."""
        self._re_enable_after_video_qa()
        self._video_qa_thread = None
        self._video_qa_worker = None

    # * Per-file UI updates
    def _on_file_started(self, media_path: str) -> None:
        """Mark a single input row as actively processing (spinner overlay)."""
        try:
            key = str(Path(media_path).resolve())
        except OSError:
            key = str(media_path)
        self._input_overlay[key] = "spin"
        # Ensure spinner is running
        if not self._spinner_timer.isActive():
            self._spinner_phase = 0
            self._spinner_timer.start()
        # Refresh icon for this row only
        for i in range(self.input_list.rowCount()):
            it = self.input_list.item(i, 1)
            if it is not None and (
                it.text() == media_path or str(Path(it.text())) == media_path
            ):
                self._update_item_icon_row(i)
                break

    def _on_file_finished(self, media_path: str, _outputs: list[str]) -> None:
        """Mark a single input row as completed and refresh status immediately."""
        try:
            key = str(Path(media_path).resolve())
        except OSError:
            key = str(media_path)
        if self._input_overlay.get(key) == "spin":
            self._input_overlay[key] = "done"
        # Refresh statuses from disk so icons switch from pending ➜ fast/good
        self._scan_output_statuses()
        # Update just this row's icon
        for i in range(self.input_list.rowCount()):
            it = self.input_list.item(i, 1)
            if it is not None and (
                it.text() == media_path or str(Path(it.text())) == media_path
            ):
                self._update_item_icon_row(i)
                break

    def on_finished(self, _outputs: list[str], view_text: str) -> None:
        """Handle processing completion."""
        self._set_controls_enabled(enabled=True)
        self.progress.setValue(100)
        self._clear_tabs()
        try:
            self._build_result_tabs(_outputs, view_text)
        except Exception as ex:  # noqa: BLE001
            # Surface UI error if tab rendering fails
            if self._can_show_modal():
                QMessageBox.warning(self, "Viewer error", f"Cannot show results: {ex}")
        self.status.showMessage("Done")
        # Explicit success toast (skip under pytest/CI)
        if self._can_show_modal():
            QMessageBox.information(
                self, "Completed", "Processing finished successfully."
            )
        # Enable burn if transcript artifacts exist
        srt_exists = any(Path(x).suffix.lower() == ".srt" for x in _outputs)
        self._has_transcript = bool(view_text.strip()) or srt_exists or bool(_outputs)
        self.btn_burn.setEnabled(self._has_transcript)
        # Rescan statuses based on new outputs
        self._scan_output_statuses()
        # Stop spinner and mark processed as done for this session
        self._spinner_timer.stop()
        for p in self._last_inputs:
            try:
                key = str(p.resolve())
            except OSError:
                key = str(p)
            if self._input_overlay.get(key) == "spin":
                self._input_overlay[key] = "done"
        # Refresh icons
        for i in range(self.input_list.rowCount()):
            self._update_item_icon_row(i)

    def _build_result_tabs(self, outputs: list[str], view_text: str) -> None:
        """Build tabs for results and establish preview mapping to media files."""
        srt_candidates = [x for x in outputs if Path(x).suffix.lower() == ".srt"]
        if srt_candidates:
            p = Path(srt_candidates[0])
            try:
                content = p.read_text(encoding="utf-8")
            except OSError as exc:
                get_logger(__name__).debug("Failed to read SRT '%s': %s", p, exc)
                content = ""
            # * Strip AskVLM metadata lines from viewer text.
            content2 = strip_askvlm_metadata_from_srt(content)
            self._add_tab(p.stem, content2, self._find_input_media_by_stem(p.stem))
            return
        if view_text:
            # * Single-view implies single input; try to map media
            media: Path | None = None
            if len(self._last_inputs) == 1:
                candidate = self._last_inputs[0]
                if candidate.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}:
                    media = candidate
            self._add_tab("Document", view_text, media)
            return
        for out_str in outputs:
            p = Path(out_str)
            try:
                content = p.read_text(encoding="utf-8")
            except OSError as exc:
                get_logger(__name__).debug("Failed to read output '%s': %s", p, exc)
                content = ""
            # * Strip AskVLM metadata lines from viewer text.
            content2 = strip_askvlm_metadata_from_srt(content)
            self._add_tab(p.stem, content2, self._find_input_media_by_stem(p.stem))

    def _remove_placeholder_document_tab_if_present(self) -> None:
        # Remove an initial empty placeholder tab named "Document"
        # when a real SRT tab is added as the first actual content.
        idx = self._find_tab_index_by_title("Document")
        if idx is not None and self.tabs.count() > 1:
            w = self.get_editor_at(idx)
            if w is not None and w.rowCount() == 0:
                self.tabs.removeTab(idx)

    def _find_tab_index_by_title(self, title: str) -> int | None:
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == title:
                return i
        return None

    # * Autosave current tab to SRT
    def _autosave_tab_srt(self, title: str, editor: WysiwygEditor) -> None:
        try:
            doc = Document()
            for row in range(editor.rowCount()):
                tr = editor.get_row_data(row)
                if tr is None:
                    continue
                # Current speaker text from combobox
                sp_widget = editor.cellWidget(row, 1)
                speaker = (
                    sp_widget.currentText()
                    if hasattr(sp_widget, "currentText")
                    else tr.speaker_id
                )
                # Current text from table item
                text_item = editor.item(row, 2)
                text = text_item.text() if text_item is not None else tr.text
                doc.add_segment(
                    TextSegment(speaker, float(tr.start), float(tr.end), text)
                )
            # Export SRT with rules and append metadata
            rules = SubtitleRules(
                max_line_chars=int(self.spin_line_len.value()),
                max_lines=int(self.spin_max_lines.value()),
            )
            srt_text = export_srt_with_rules(doc, rules)
            srt_text = append_askvlm_metadata_to_srt(
                srt_text,
                tool_name="AskVLM",
                quality=str(self._quality_mode),
                completed=True,
            )
            out_dir = Path(self.out_dir_edit.text()).resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{title}.srt").write_text(srt_text, encoding="utf-8")
            # After autosave, refresh statuses from disk
            self._scan_output_statuses()
            self.status.showMessage("Autosaved SRT")
        except OSError as exc:
            get_logger(__name__).debug("Autosave failed: %s", exc)

    def open_output_folder(self) -> None:
        """Open the output directory in the system file manager."""
        out_dir = Path(self.out_dir_edit.text()).resolve()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(out_dir)))

    def _log_wrap(self, func: Callable[..., Any], _name: str) -> Callable[..., Any]:
        def _wrapped(*args: object, **kwargs: object) -> object:
            return func(*args, **kwargs)

        return _wrapped

    # * Preview wiring
    def _on_tab_changed(self, _index: int) -> None:
        self._update_preview_for_selection()

    def _get_current_editor(self) -> WysiwygEditor | None:
        w = self.tabs.currentWidget()
        return w if isinstance(w, WysiwygEditor) else None

    def _update_preview_for_selection(self) -> None:
        editor = self._get_current_editor()
        if editor is None:
            self.preview.clear()
            return
        selected = editor.selectedIndexes()
        if not selected:
            # No selection: clear both text and background to avoid stale frames
            self.preview.clear()
            return
        row = selected[0].row()
        data = editor.get_row_data(row)
        if data is None:
            self.preview.clear()
            return
        # Layout text lines according to current limits
        lines = SubtitlePreview.layout_text(
            data.text, self.spin_line_len.value(), self.spin_max_lines.value()
        )
        self.preview.set_text_lines(lines)
        # Push current font overrides to preview
        self.preview.set_font_family(self.font_combo.currentText())
        self.preview.set_font_size_override(self._font_px)
        # * Try to extract a frame for the media mapped to the current tab
        try:
            media_path = self._get_preview_media_path_for_current_tab()
            if media_path is not None and media_path.suffix.lower() in {
                ".mp4",
                ".mov",
                ".mkv",
                ".avi",
            }:
                # Ensure output directory exists for the preview artifact
                with contextlib.suppress(OSError):
                    self.last_output_dir.mkdir(parents=True, exist_ok=True)
                frame_path = self.last_output_dir / f".{media_path.stem}.preview.png"
                # Prefer start timestamp; fallback to 0
                ts = max(0.0, float(data.start))
                extract_frame_to_file(media_path, ts, frame_path)
                img = QImage(str(frame_path))
                if not img.isNull():
                    self.preview.set_background_image(img)
                    # Ensure preview font override applied after image update
                    self.preview.set_font_size_override(self._font_px)
                    return
        except Exception as exc:  # noqa: BLE001
            # * Best-effort preview; log and continue with plain background
            get_logger(__name__).debug("Preview frame extraction failed: %s", exc)
        # If cannot show frame, keep plain background
        self.preview.set_background_image(None)

    def _get_preview_media_path_for_current_tab(self) -> Path | None:
        """Return media path associated with the current tab or a best-effort fallback.

        The method first resolves mapping by the tab's title, then falls back to
        single-input scenarios and legacy `input_mode`/`input_path` fields.
        """
        # * Primary: by tab title mapping
        idx = self.tabs.currentIndex()
        if idx >= 0:
            title = str(self.tabs.tabText(idx))
            mapped = self._tab_to_media.get(title)
            if mapped is not None:
                return mapped
        # * Fallback: only one known input
        if len(self._last_inputs) == 1:
            return self._last_inputs[0]
        # * Legacy fallback: direct input selectors
        if self.input_mode == "file" and self.input_path is not None:
            return self.input_path
        # * Last resort: if Input tab contains exactly one video file
        items = []
        for i in range(self.input_list.rowCount()):
            it = self.input_list.item(i, 1)
            if it is not None:
                items.append(Path(it.text()))
        videos = [
            p
            for p in items
            if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}
        ]
        if len(videos) == 1:
            return videos[0]
        return None

    def _find_input_media_by_stem(self, stem: str) -> Path | None:
        """Return input media whose stem matches `stem` if available.

        Searches the remembered inputs first, then the current Input tab list.
        """
        # * Prefer remembered inputs from the last run
        for p in self._last_inputs:
            if p.stem == stem:
                return p
        # * Fallback: scan visible Input tab entries
        for i in range(self.input_list.rowCount()):
            it = self.input_list.item(i, 1)
            if it is None:
                continue
            p = Path(it.text())
            if p.is_file() and p.stem == stem:
                return p
        return None

    def _on_input_item_double_clicked(self, _item: QTableWidgetItem) -> None:
        # Open corresponding SRT in a new tab if exists
        out_dir = Path(self.out_dir_edit.text()).resolve()
        sel = self.input_list.selectedIndexes()
        if not sel:
            return
        row = sel[0].row()
        it = self.input_list.item(row, 1)
        if it is None:
            return
        p = Path(it.text())
        srt = out_dir / f"{p.stem}.srt"
        if srt.exists():
            try:
                content = srt.read_text(encoding="utf-8")
            except OSError:
                return
            content2 = strip_askvlm_metadata_from_srt(content)
            self._add_tab(srt.stem, content2, self._find_input_media_by_stem(p.stem))

    def _close_orphan_tabs(self) -> None:
        # Close tabs whose mapped media is not in current Input table
        present: set[str] = set()
        for i in range(self.input_list.rowCount()):
            it = self.input_list.item(i, 1)
            if it is None:
                continue
            with contextlib.suppress(OSError):
                present.add(str(Path(it.text()).resolve()))
        # Iterate titles mapped to media
        to_remove: list[int] = []
        for idx in range(self.tabs.count()):
            title = self.tabs.tabText(idx)
            media = self._tab_to_media.get(title)
            if media is None:
                continue
            with contextlib.suppress(OSError):
                key = str(media.resolve())
            if key not in present:
                to_remove.append(idx)
        for idx in reversed(to_remove):
            self.tabs.removeTab(idx)

    def start_burn(self) -> None:
        """Start burn-in process for selected inputs using burn settings."""
        burn_video_exts = {".mp4", ".mov", ".mkv", ".avi"}
        gathered = self._gather_inputs()
        if not gathered:
            QMessageBox.information(
                self,
                "No input",
                "Add at least one file or folder in the Input tab before burning "
                "subtitles.",
            )
            return
        if not self._has_transcript:
            QMessageBox.information(self, "No transcript", "Please transcribe first.")
            return
        out_dir = Path(self.out_dir_edit.text()).resolve()
        normalize = self.chk_normalize.isChecked()
        font_name = self.font_combo.currentText().strip() or None
        inputs = [
            p for p in gathered if p.is_file() and p.suffix.lower() in burn_video_exts
        ]
        if not inputs:
            QMessageBox.information(
                self,
                "No video",
                "No video files (.mp4, .mov, .mkv, .avi) found among the Input tab "
                "entries.",
            )
            return
        # Start burn worker thread
        if self._burn_thread is not None:
            try:
                self._burn_thread.quit()
                self._burn_thread.wait(2000)
            except Exception as exc:  # noqa: BLE001
                get_logger(__name__).debug("Thread cleanup issue: %s", exc)
            self._burn_thread = None
        self._burn_thread = QThread(self)
        # Inline lightweight burn worker (signals via lambda wrappers)
        self._burn_worker = _InlineBurnWorker(
            inputs, out_dir, norm=normalize, font=font_name, font_px=self._font_px
        )
        self._burn_worker.moveToThread(self._burn_thread)
        self._burn_thread.started.connect(self._burn_worker.run)
        self._burn_worker.progress.connect(self.on_progress)
        self._burn_worker.log.connect(self.on_log)
        self._burn_worker.error.connect(self.on_error)
        self._burn_worker.canceled.connect(self.on_canceled)
        self._burn_worker.finished.connect(lambda _o: self._on_burn_finished())
        self._burn_worker.finished.connect(self._burn_thread.quit)
        self._burn_worker.finished.connect(self._burn_worker.deleteLater)
        self._burn_thread.finished.connect(self._burn_thread.deleteLater)
        # Disable burn controls while running and enable global Cancel
        _disable = False
        _enable = True
        self.btn_burn.setEnabled(_disable)
        self.chk_normalize.setEnabled(_disable)
        self.font_combo.setEnabled(_disable)
        self.btn_cancel.setEnabled(_enable)
        self.status.showMessage("Burning…")
        self._burn_thread.start()

    def _on_burn_finished(self) -> None:
        self.btn_burn.setEnabled(_UI_CHECKED)
        self.chk_normalize.setEnabled(_UI_CHECKED)
        self.font_combo.setEnabled(_UI_CHECKED)
        self.status.showMessage("Burn completed")

    # * Font size helpers
    def _nudge_font_size(self, delta: int) -> None:
        """Increase or decrease preview/burn font size by delta pixels.

        When None, initialize from current auto-estimate (~3% of preview height).
        """
        if self._font_px is None:
            est = max(20, min(38, int(self.preview.height() * 0.03)))
            self._font_px = est
        self._font_px = int(max(10, min(96, (self._font_px or 0) + int(delta))))
        self.preview.set_font_size_override(self._font_px)
        # Trigger re-layout
        self._update_preview_for_selection()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Ensure we cancel background tasks and persist settings on close."""
        with contextlib.suppress(Exception):
            self._save_settings()

        should_close = True
        # Trigger global cancel and wait for background work to finish
        try:
            if self._worker:
                self._worker.set_closing()
            self.request_cancel()
            with contextlib.suppress(Exception):
                get_logger(__name__).info("Application exit requested")
            # Stop spinner and wait for worker threads
            should_close = self.await_worker_shutdown(timeout_ms=30000)
        except Exception:  # noqa: BLE001
            # * Best-effort shutdown: if something unexpected happens, prefer exit
            should_close = True

        if not should_close:
            # * Keep window open to avoid destroying running QThreads,
            # * which would cause Qt to abort the process.
            self.status.showMessage(
                "Background tasks are still stopping; please wait…",
            )
            event.ignore()
            return

        event.accept()

    def _on_worker_thread_finished(self) -> None:
        """Reset worker/thread references after background worker termination."""
        self._worker = None
        self._thread = None

    def _stop_qthread(
        self,
        thread: QThread | None,
        timeout_ms: int,
        *,
        label: str,
    ) -> tuple[QThread | None, bool]:
        if thread is None:
            return None, True
        try:
            if thread.isRunning():
                thread.quit()
                if not thread.wait(timeout_ms):
                    get_logger(__name__).warning(
                        "%s thread did not stop in %d ms",
                        label,
                        timeout_ms,
                    )
                    return thread, False
            if not thread.isRunning():
                return None, True
        except RuntimeError:
            # C++ object already deleted (e.g. via deleteLater)
            get_logger(__name__).debug(
                "%s thread object already deleted during shutdown",
                label,
            )
            return None, True
        return thread, True

    def await_worker_shutdown(self, timeout_ms: int = 30000) -> bool:
        """Stop spinner and await background worker threads termination.

        Returns:
            True if all known worker threads have stopped within the timeout,
            False if any worker is still running.

        """
        with contextlib.suppress(Exception):
            self._spinner_timer.stop()

        self._thread, stopped = self._stop_qthread(
            self._thread,
            timeout_ms,
            label="Worker",
        )
        worker_stopped = stopped
        self._burn_thread, stopped = self._stop_qthread(
            self._burn_thread,
            2000,
            label="Burn",
        )
        burn_stopped = stopped
        self._video_qa_thread, vq_stopped = self._stop_qthread(
            self._video_qa_thread,
            timeout_ms,
            label="VideoQA",
        )
        return worker_stopped and burn_stopped and vq_stopped

    def _restore_video_qa_run_toggles(self, s: QSettings) -> None:
        """Restore Video QA chunking and backend run-option flags from settings."""
        chunking_val = s.value("videoqa/chunking_enabled", defaultValue=True, type=bool)
        if isinstance(chunking_val, bool):
            chunking_on = chunking_val
        else:
            chunking_on = str(chunking_val).strip().lower() in ("1", "true", "yes")

        def _bool_key(key: str, *, default: bool) -> bool:
            v = s.value(key, defaultValue=default, type=bool)
            if isinstance(v, bool):
                return v
            return str(v).strip().lower() in ("1", "true", "yes")

        self.video_qa_panel.set_video_qa_run_toggles(
            chunking_enabled=chunking_on,
            final_include_transcript=_bool_key(
                "videoqa/final_include_transcript",
                default=False,
            ),
            final_include_start_frame_per_chunk=_bool_key(
                "videoqa/final_include_start_frame_per_chunk",
                default=False,
            ),
            reasoning_enabled=_bool_key("videoqa/reasoning_enabled", default=False),
        )

    def _apply_video_qa_settings(self, s: QSettings) -> None:
        """Restore Video QA panel fields from settings."""
        self.video_qa_panel.set_source_path(str(s.value("videoqa/source_path", "")))
        self.video_qa_panel.set_question_text(str(s.value("videoqa/question", "")))
        budget_str = str(s.value("videoqa/context_window_tokens", ""))
        if budget_str.isdigit():
            self.video_qa_panel.set_context_window_tokens(int(budget_str))
        fps_raw = s.value("videoqa/frame_sample_fps", "")
        fps_text = str(fps_raw).strip()
        if fps_text:
            with contextlib.suppress(TypeError, ValueError):
                self.video_qa_panel.set_frame_sample_fps(float(fps_text))
        self._restore_video_qa_run_toggles(s)
        raw_attach = s.value("videoqa/attachments_json", "")
        if isinstance(raw_attach, str) and raw_attach.strip():
            try:
                data = json.loads(raw_attach)
            except json.JSONDecodeError:
                data = None
            if isinstance(data, list):
                self.video_qa_panel.restore_attachments_state(list(data))
        main_splitter_state = s.value("videoqa/main_splitter_state", None)
        left_splitter_state = s.value("videoqa/left_splitter_state", None)
        self.video_qa_panel.restore_splitter_states(
            main_splitter_state,
            left_splitter_state,
        )
        ans_prog_splitter_state = s.value(
            "videoqa/answer_progress_splitter_state", None
        )
        self.video_qa_panel.restore_answer_progress_splitter_state(
            ans_prog_splitter_state
        )
        chunk_scope_raw = s.value("videoqa/lm_chunk_scope", None)
        if chunk_scope_raw is None:
            try:
                legacy_scope = int(str(s.value("videoqa/lm_scope", 0)).strip())
            except (TypeError, ValueError):
                legacy_scope = 0
            legacy_local = str(s.value("videoqa/lm_local_model_text", ""))
            legacy_cloud = str(s.value("videoqa/lm_cloud_model_text", ""))
            self.video_qa_panel.restore_video_qa_lm_ui(
                chunk_scope_index=legacy_scope,
                chunk_local_model_text=legacy_local,
                chunk_cloud_model_text=legacy_cloud,
                final_scope_index=legacy_scope,
                final_local_model_text=legacy_local,
                final_cloud_model_text=legacy_cloud,
            )
        else:
            try:
                cs = int(str(chunk_scope_raw).strip())
            except (TypeError, ValueError):
                cs = 0
            try:
                fs = int(str(s.value("videoqa/lm_final_scope", 0)).strip())
            except (TypeError, ValueError):
                fs = 0
            self.video_qa_panel.restore_video_qa_lm_ui(
                chunk_scope_index=cs,
                chunk_local_model_text=str(
                    s.value("videoqa/lm_chunk_local_model_text", "")
                ),
                chunk_cloud_model_text=str(
                    s.value("videoqa/lm_chunk_cloud_model_text", "")
                ),
                final_scope_index=fs,
                final_local_model_text=str(
                    s.value("videoqa/lm_final_local_model_text", "")
                ),
                final_cloud_model_text=str(
                    s.value("videoqa/lm_final_cloud_model_text", "")
                ),
            )
        get_logger(__name__).debug("Video QA: settings restore finished")

    # * Settings persistence
    def _load_settings(self) -> None:
        s = QSettings("AskVLM", "AskVLM")
        # Window geometry/state
        geo = s.value("ui/window_geometry", None)
        if isinstance(geo, QByteArray):
            with contextlib.suppress(Exception):
                self.restoreGeometry(geo)
        state = s.value("ui/window_state", None)
        if isinstance(state, QByteArray):
            with contextlib.suppress(Exception):
                self.restoreState(state)
        # Splitter state
        sp = s.value("ui/splitter_state", None)
        if isinstance(sp, QByteArray):
            with contextlib.suppress(Exception):
                self.splitter.restoreState(sp)

        shell_screen = str(s.value("ui/shell_screen", self.SHELL_SCREEN_TEXT))
        self._set_shell_screen(shell_screen)
        self._apply_video_qa_settings(s)

        self.chk_diar.setChecked(bool(s.value("opts/diar", type=bool)))
        self.chk_dialog.setChecked(bool(s.value("opts/dialog", type=bool)))
        self.chk_save_srt.setChecked(bool(s.value("opts/save_srt", 1, type=bool)))
        fmt = str(s.value("opts/format", "srt"))
        idx = self.format_combo.findText(fmt)
        if idx >= 0:
            self.format_combo.setCurrentIndex(idx)
        out_dir = str(s.value("paths/out_dir", self.out_dir_edit.text()))
        if out_dir:
            self.out_dir_edit.setText(out_dir)
            self.last_output_dir = Path(out_dir)
        # Last input dir for file/folder dialogs
        last_in = s.value("paths/last_input_dir", None)
        if last_in:
            with contextlib.suppress(Exception):
                self.last_input_dir = Path(str(last_in))
        self.chk_normalize.setChecked(bool(s.value("burn/normalize", 1, type=bool)))
        font = str(s.value("burn/font", ""))
        if font:
            fidx = self.font_combo.findText(font)
            if fidx >= 0:
                self.font_combo.setCurrentIndex(fidx)
        # Phase 1.81 integers with explicit type coercion
        mlc_val = s.value("subs/max_line_chars", 42, type=int)
        mlc: int = int(mlc_val) if isinstance(mlc_val, int) else 42
        self.spin_line_len.setValue(max(20, min(120, mlc)))
        mlines_val = s.value("subs/max_lines", 2, type=int)
        mlines: int = int(mlines_val) if isinstance(mlines_val, int) else 2
        self.spin_max_lines.setValue(max(1, min(3, mlines)))

    def _save_settings(self) -> None:
        s = QSettings("AskVLM", "AskVLM")
        # Window geometry/state
        s.setValue("ui/window_geometry", self.saveGeometry())
        s.setValue("ui/window_state", self.saveState())
        # Splitter state
        s.setValue("ui/splitter_state", self.splitter.saveState())
        # Common options
        s.setValue("opts/diar", self.chk_diar.isChecked())
        s.setValue("opts/dialog", self.chk_dialog.isChecked())
        s.setValue("opts/save_srt", self.chk_save_srt.isChecked())
        s.setValue("opts/format", self.format_combo.currentText())
        s.setValue("paths/out_dir", self.out_dir_edit.text())
        # Last input dir for file/folder dialogs
        s.setValue("paths/last_input_dir", str(self.last_input_dir))
        s.setValue("burn/normalize", self.chk_normalize.isChecked())
        s.setValue("burn/font", self.font_combo.currentText())
        s.setValue("ui/shell_screen", self._shell_screen_key())
        source_path = self.video_qa_panel.source_path()
        if source_path is None:
            s.remove("videoqa/source_path")
        else:
            s.setValue("videoqa/source_path", str(source_path))
        s.setValue("videoqa/question", self.video_qa_panel.question_text())
        s.setValue(
            "videoqa/context_window_tokens", self.video_qa_panel.context_window_tokens()
        )
        s.setValue("videoqa/frame_sample_fps", self.video_qa_panel.frame_sample_fps())
        s.setValue(
            "videoqa/chunking_enabled",
            self.video_qa_panel.video_chunking_enabled(),
        )
        opts = self.video_qa_panel.video_qa_local_run_options()
        s.setValue(
            "videoqa/final_include_transcript",
            opts.final_request.include_transcript,
        )
        s.setValue(
            "videoqa/final_include_start_frame_per_chunk",
            opts.final_request.include_start_frame_per_chunk,
        )
        s.setValue("videoqa/reasoning_enabled", opts.reasoning_enabled)
        s.setValue(
            "videoqa/attachments_json",
            json.dumps(self.video_qa_panel.attachments_for_persistence()),
        )
        s.setValue(
            "videoqa/main_splitter_state",
            self.video_qa_panel.main_splitter_state(),
        )
        s.setValue(
            "videoqa/left_splitter_state",
            self.video_qa_panel.left_splitter_state(),
        )
        s.setValue(
            "videoqa/answer_progress_splitter_state",
            self.video_qa_panel.answer_progress_splitter_state(),
        )
        s.setValue(
            "videoqa/lm_chunk_scope",
            int(self.video_qa_panel.chunk_model_type_combo.currentIndex()),
        )
        s.setValue(
            "videoqa/lm_chunk_local_model_text",
            self.video_qa_panel.chunk_model_combo.currentText(),
        )
        s.setValue(
            "videoqa/lm_chunk_cloud_model_text",
            self.video_qa_panel.chunk_model_cloud_edit.text(),
        )
        s.setValue(
            "videoqa/lm_final_scope",
            int(self.video_qa_panel.final_model_type_combo.currentIndex()),
        )
        s.setValue(
            "videoqa/lm_final_local_model_text",
            self.video_qa_panel.final_model_combo.currentText(),
        )
        s.setValue(
            "videoqa/lm_final_cloud_model_text",
            self.video_qa_panel.final_model_cloud_edit.text(),
        )
        # Phase 1.81
        s.setValue("subs/max_line_chars", int(self.spin_line_len.value()))
        s.setValue("subs/max_lines", int(self.spin_max_lines.value()))

    def _shell_screen_key(self) -> str:
        """Return the current top-level shell key."""
        if self.shell_tabs.currentIndex() == 1:
            return self.SHELL_SCREEN_VIDEO_QA
        return self.SHELL_SCREEN_TEXT

    def _set_shell_screen(self, screen: str) -> None:
        """Select the top-level shell tab from a persisted key."""
        idx = 1 if screen == self.SHELL_SCREEN_VIDEO_QA else 0
        if 0 <= idx < self.shell_tabs.count():
            self.shell_tabs.setCurrentIndex(idx)

    # * Tabs helpers
    def _clear_tabs(self) -> None:
        while self.tabs.count() > 0:
            self.tabs.removeTab(0)
        # Create a hidden placeholder to simplify logic if needed

    def _add_tab(self, title: str, text: str, media: Path | None = None) -> None:  # noqa: PLR0915, C901
        editor = WysiwygEditor()
        # Parse plain text into rows with timestamp-prefixed blocks.
        # Regex for leading time range
        time_pat = re.compile(
            r"^(\d\d:\d\d:\d\d[,\.]\d{3})\s*(?:-->|→|-+>?)\s*(\d\d:\d\d:\d\d[,\.]\d{3})"
        )
        rows: list[tuple[float, float, str, str]] = []
        for block in [x for x in (text or "").split("\n\n") if x.strip()]:
            start = 0.0
            end = 0.0
            speaker = "speaker_1"
            content = block.strip()
            lines = content.splitlines()
            m = time_pat.match(lines[0]) if lines else None
            second_line_index = 2  # * Index of the second line in block
            text_start_idx = 1
            if not m and len(lines) >= second_line_index:
                m = time_pat.match(lines[1])
                text_start_idx = second_line_index if m else 0
            if m:

                def parse_time(s: str) -> float:
                    s2 = s.replace(",", ".")
                    h, m2, rest = s2.split(":")
                    s3, ms = rest.split(".")
                    return int(h) * 3600 + int(m2) * 60 + int(s3) + int(ms) / 1000.0

                start = parse_time(m.group(1))
                end = parse_time(m.group(2))
                content = "\n".join(lines[text_start_idx:]).strip()
            # Try speaker prefix
            if ":" in content and content.split(":", 1)[0].strip():
                sp, txt = content.split(":", 1)
                candidate = sp.strip()
                if 1 <= len(candidate) <= self.MAX_SPEAKER_LEN:
                    speaker = candidate
                    content = txt.strip()
            rows.append((start, end, speaker, content))
        # Feed speakers and rows
        self.sidebar.set_speakers(list({r[2] for r in rows} or {"speaker_1"}))
        editor.set_speakers(self.sidebar.get_speakers())
        editor.set_rows(
            [TableRow(start=s, end=e, speaker_id=sp, text=tx) for s, e, sp, tx in rows]
        )
        # Fixed row height for two lines
        fm_height = editor.fontMetrics().height()
        # Avoid magic number warning by naming the padding constant
        padding_px = 6
        two_line_height = fm_height * 2 + padding_px
        for r in range(editor.rowCount()):
            editor.setRowHeight(r, two_line_height)
        # Hide speaker column if all speakers are default and diarization disabled
        hide_speaker = all(sp == "speaker_1" for _, _, sp, _ in rows)
        editor.setColumnHidden(1, hide_speaker)
        # Update stats
        for _, _, sp, _ in rows:
            self.sidebar.record_usage(title, sp)
        # Restore column widths if saved
        with contextlib.suppress(Exception):
            s = QSettings("AskVLM", "AskVLM")
            w0_val: object = s.value("ui/table_time_width", 0, type=int)
            w1_val: object = s.value("ui/table_speaker_width", 0, type=int)
            w2_val: object = s.value("ui/table_text_width", 0, type=int)
            w0: int = int(w0_val) if isinstance(w0_val, int) else 0
            w1: int = int(w1_val) if isinstance(w1_val, int) else 0
            w2: int = int(w2_val) if isinstance(w2_val, int) else 0
            if w0 > 0:
                editor.setColumnWidth(0, w0)
            if w1 > 0:
                editor.setColumnWidth(1, w1)
            if w2 > 0:
                editor.setColumnWidth(2, w2)
        # Connect selection change to preview
        editor.itemSelectionChanged.connect(self._update_preview_for_selection)

        # Autosave on cell content changes and on speaker changes
        def _autosave_on_change_title_ed(
            _arg: object = None, *, _t: str = title, _ed: WysiwygEditor = editor
        ) -> None:
            self._autosave_tab_srt(_t, _ed)

        editor.itemChanged.connect(_autosave_on_change_title_ed)
        editor.on_speaker_changed(lambda _row, _val: _autosave_on_change_title_ed())
        # Persist column widths on resize changes
        editor.horizontalHeader().sectionResized.connect(
            lambda _i, _o, _n, ed=editor: self._save_table_widths(ed)
        )
        # * Register tab and map to media (for preview frame extraction)
        self.tabs.addTab(editor, title)
        # Prefer the newly added tab
        self.tabs.setCurrentIndex(self.tabs.count() - 1)
        # Remove placeholder "Document" tab if present and empty
        self._remove_placeholder_document_tab_if_present()
        if media is not None:
            # * Store mapping by title for simplicity; titles are per-session unique
            self._tab_to_media[title] = media

    # * Testing helpers
    def get_editor_at(self, index: int) -> WysiwygEditor | None:
        """Return the tab editor at the given index or None if not present."""
        w = self.tabs.widget(index)
        return w if isinstance(w, WysiwygEditor) else None

    def _save_table_widths(self, editor: WysiwygEditor) -> None:
        """Persist current table column widths to QSettings."""
        with contextlib.suppress(Exception):
            s = QSettings("AskVLM", "AskVLM")
            s.setValue("ui/table_time_width", int(editor.columnWidth(0)))
            s.setValue("ui/table_speaker_width", int(editor.columnWidth(1)))
            s.setValue("ui/table_text_width", int(editor.columnWidth(2)))

    # * Input helpers
    def _gather_inputs(self) -> list[Path]:
        def expand_entry(entry: str) -> list[Path]:
            p = Path(entry)
            if p.is_dir():
                try:
                    # Include all immediate files in the directory (no extension filter)
                    return [c for c in p.iterdir() if c.is_file()]
                except OSError:
                    return []
            return [p] if p.is_file() else []

        # Prefer entries from Input tab
        items = []
        if hasattr(self, "input_list") and self.input_list.rowCount() > 0:
            for i in range(self.input_list.rowCount()):
                it = self.input_list.item(i, 1)
                if it is not None:
                    items.append(it.text())
        collected = [p for s in items for p in expand_entry(s)]

        # Top-row selectors removed; rely entirely on Input tab entries

        # Deduplicate preserving order
        uniq: list[Path] = []
        seen: set[str] = set()
        for p in collected:
            try:
                s = str(p.resolve())
            except OSError:
                s = str(p)
            if s not in seen:
                seen.add(s)
                uniq.append(Path(s))
        return uniq

    # * Input status and icon helpers
    def _get_status_icon(self, status: str) -> QIcon:
        # Cache by status key
        if status in self._status_icon_cache:
            return self._status_icon_cache[status]
        symbol = {
            "": "",
            "error": "❌",
            "fast": "⏩",
            "good": "📄",
            "burned": "🔥",
        }.get(status, "")
        if not symbol:
            icon = QIcon()
            self._status_icon_cache[status] = icon
            return icon
        # Render a small pixmap with the symbol
        pm = QPixmap(16, 16)
        pm.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pm)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, on=True)
            f = QFont()
            # Prefer emoji-capable font on Windows/Linux/macOS for symbol rendering
            try:
                families = set(QFontDatabase.families())
            except Exception:  # noqa: BLE001
                families = set()
            for fam in (
                "Segoe UI Emoji",
                "Segoe UI Symbol",
                "Noto Color Emoji",
                "Apple Color Emoji",
            ):
                if fam in families:
                    f.setFamily(fam)
                    break
            f.setPointSize(10)
            painter.setFont(f)
            painter.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, symbol)
        finally:
            painter.end()
        icon = QIcon(pm)
        self._status_icon_cache[status] = icon
        return icon

    def _get_overlay_icon(self, overlay: str) -> QIcon:
        symbol = {
            "done": "✔️",
            "spin0": "🔃",
            "spin1": "🔁",
            "": "",
        }.get(overlay, "")
        if not symbol:
            return QIcon()
        pm = QPixmap(16, 16)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing, on=True)
            f = QFont()
            try:
                families = set(QFontDatabase.families())
            except Exception:  # noqa: BLE001
                families = set()
            for fam in (
                "Segoe UI Emoji",
                "Segoe UI Symbol",
                "Noto Color Emoji",
                "Apple Color Emoji",
            ):
                if fam in families:
                    f.setFamily(fam)
                    break
            f.setPointSize(10)
            p.setFont(f)
            p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, symbol)
        finally:
            p.end()
        return QIcon(pm)

    def _get_composite_text_for_key(self, key: str) -> str:
        base = self._input_status.get(key, "")
        overlay = self._input_overlay.get(key, "")
        base_symbol = {
            "": "",
            "error": "❌",
            "fast": "⏩",
            "good": "📄",
            "burned": "🔥",
        }.get(base, "")
        overlay_symbol = ""
        if overlay == "done":
            overlay_symbol = "✔️"
        elif overlay == "spin":
            overlay_symbol = "🔁" if (self._spinner_phase % 2) else "🔃"
        # Return combined string with thin space between to avoid overlap
        return (
            base_symbol + ("\u2009" + overlay_symbol if overlay_symbol else "")
        ).strip()

    def _on_spinner_tick(self) -> None:
        self._spinner_phase = (self._spinner_phase + 1) % 2
        # Refresh visible composite icons
        for i in range(self.input_list.rowCount()):
            it = self.input_list.item(i, 1)
            if it is None:
                continue
            try:
                key = str(Path(it.text()).resolve())
            except OSError:
                key = str(Path(it.text()))
            if self._input_overlay.get(key) == "spin":
                self._update_item_icon_row(i)

    def _get_input_status(self, path: Path) -> str:
        with contextlib.suppress(OSError):
            key = str(path.resolve())
            return self._input_status.get(key, "")
        return self._input_status.get(key, "")

    def _set_input_status(self, path: Path, status: str) -> None:
        with contextlib.suppress(OSError):
            key = str(path.resolve())
        if "key" not in locals():
            key = str(path)
        prev = self._input_status.get(key, "")
        if prev == status:
            return
        self._input_status[key] = status
        # Update any matching list item icon (composite with overlay)
        for i in range(self.input_list.rowCount()):
            it = self.input_list.item(i, 1)
            if it is None:
                continue
            with contextlib.suppress(OSError):
                if str(Path(it.text()).resolve()) == key:
                    icon_text = self._get_composite_text_for_key(key)
                    icon_item = self.input_list.item(i, 0)
                    if icon_item is None:
                        icon_item = QTableWidgetItem(icon_text)
                        self.input_list.setItem(i, 0, icon_item)
                    else:
                        icon_item.setText(icon_text)
                    continue
            if str(Path(it.text())) == key:
                icon_text = self._get_composite_text_for_key(key)
                icon_item = self.input_list.item(i, 0)
                if icon_item is None:
                    icon_item = QTableWidgetItem(icon_text)
                    self.input_list.setItem(i, 0, icon_item)
                else:
                    icon_item.setText(icon_text)

    def _update_item_icon_row(self, row: int) -> None:
        it = self.input_list.item(row, 1)
        if it is None:
            return
        p = Path(it.text())
        try:
            key = str(p.resolve())
        except OSError:
            key = str(p)
        icon_text = self._get_composite_text_for_key(key)
        icon_item = self.input_list.item(row, 0)
        if icon_item is None:
            icon_item = QTableWidgetItem(icon_text)
            with contextlib.suppress(Exception):
                icon_item.setFlags(icon_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                icon_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.input_list.setItem(row, 0, icon_item)
        else:
            icon_item.setText(icon_text)
            with contextlib.suppress(Exception):
                icon_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

    def _scan_output_statuses(self) -> None:  # noqa: C901, PLR0912
        out_dir_text = self.out_dir_edit.text()
        try:
            out_dir = Path(out_dir_text).resolve()
        except (OSError, ValueError):
            return
        # Build availability maps
        burned_set: set[str] = set()
        fast_set: set[str] = set()
        good_set: set[str] = set()
        for srt in out_dir.glob("*.srt"):
            try:
                txt = srt.read_text(encoding="utf-8")
            except OSError:
                continue
            meta = extract_askvlm_metadata_from_srt(txt) or {}
            if isinstance(meta, dict):
                if str(meta.get("tool", "")) != "AskVLM" or not bool(
                    meta.get("completed", False)
                ):
                    pass
                else:
                    q = str(meta.get("quality", "")).lower()
                    if q == "good":
                        good_set.add(srt.stem)
                    elif q == "fast":
                        fast_set.add(srt.stem)
        for mp4 in out_dir.glob("*_subbed.mp4"):
            burned_set.add(re.sub(r"_subbed$", "", mp4.stem))
        # Assign statuses with precedence: burned > good > fast > error/none
        for i in range(self.input_list.rowCount()):
            it = self.input_list.item(i, 1)
            if it is None:
                continue
            p = Path(it.text())
            stem = p.stem
            new_status = ""
            if stem in burned_set:
                new_status = "burned"
            elif stem in good_set:
                new_status = "good"
            elif stem in fast_set:
                new_status = "fast"
            self._set_input_status(p, new_status)
            # Also refresh composite icon in case overlay exists
            self._update_item_icon_row(i)

    def _input_add_file(self) -> None:
        # Allow selecting multiple files; if a directory is given, add it directly
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Add File(s)",
            str(self.last_input_dir),
            "All Files (*.*)",
        )
        for fn in files:
            p = Path(fn)
            if not p.exists():
                continue
            if p.is_dir():
                # Add directory itself; later expansion happens in _gather_inputs
                self.last_input_dir = p
                row = self.input_list.rowCount()
                self.input_list.insertRow(row)
                # icon cell empty; path in column 1
                self.input_list.setItem(row, 1, QTableWidgetItem(str(p)))
                self._update_item_icon_row(row)
                continue
            if p.is_file():
                self.last_input_dir = p.parent
                row = self.input_list.rowCount()
                self.input_list.insertRow(row)
                self.input_list.setItem(row, 1, QTableWidgetItem(str(p)))
                self._update_item_icon_row(row)
        # After adding, rescan statuses
        self._scan_output_statuses()

    # (Removed _input_add_folder)

    def _input_remove_selected(self) -> None:
        sel_rows = sorted(
            {i.row() for i in self.input_list.selectedIndexes()}, reverse=True
        )
        for row in sel_rows:
            it = self.input_list.item(row, 1)
            if it is not None:
                with contextlib.suppress(OSError):
                    key = str(Path(it.text()).resolve())
                    self._input_status.pop(key, None)
                    self._input_overlay.pop(key, None)
            self.input_list.removeRow(row)
        # Close tabs whose media is no longer present in Input
        self._close_orphan_tabs()
        self._scan_output_statuses()
        # Refresh preview; if nothing selected, it will be cleared
        self._update_preview_for_selection()

    # (Removed _input_clear)

    def _input_move_up(self) -> None:
        sel = self.input_list.selectedIndexes()
        if not sel:
            return
        row = sel[0].row()
        if row <= 0:
            return
        self._swap_rows(row, row - 1)
        self.input_list.selectRow(row - 1)
        self._scan_output_statuses()

    def _input_move_down(self) -> None:
        sel = self.input_list.selectedIndexes()
        if not sel:
            return
        row = sel[0].row()
        if row >= self.input_list.rowCount() - 1:
            return
        self._swap_rows(row, row + 1)
        self.input_list.selectRow(row + 1)
        self._scan_output_statuses()

    def _swap_rows(self, a: int, b: int) -> None:
        if a == b:
            return
        max_col = max(1, self.input_list.columnCount() - 1)
        for col in range(max_col + 1):
            ia = self.input_list.takeItem(a, col)
            ib = self.input_list.takeItem(b, col)
            if ib is not None:
                self.input_list.setItem(a, col, ib)
            if ia is not None:
                self.input_list.setItem(b, col, ia)

    def _input_reset_status_selected(self) -> None:
        rows = {i.row() for i in self.input_list.selectedIndexes()}
        for row in rows:
            it = self.input_list.item(row, 1)
            if it is None:
                continue
            try:
                key = str(Path(it.text()).resolve())
            except OSError:
                key = str(Path(it.text()))
            # Clear status and overlay
            self._input_status.pop(key, None)
            self._input_overlay.pop(key, None)
            self._update_item_icon_row(row)

    # * Quality toggle
    def _toggle_quality(self) -> None:
        self._quality_mode = "fast" if self._quality_mode == "good" else "good"
        self.btn_quality.setText(
            "Quality: Good" if self._quality_mode == "good" else "Quality: Fast"
        )
        # Informative hint for what will be processed under current statuses
        self.status.showMessage(
            "Quality set to %s" % ("Good" if self._quality_mode == "good" else "Fast")
        )

    def _apply_quality_to_pipeline(self, *, force_reload: bool = False) -> None:
        model = "large-v3" if self._quality_mode == "good" else "small"
        try:
            # Update underlying wrapper and force reload next time
            self.pipeline.whisperx.model_name = model
            if force_reload and isinstance(self.pipeline.whisperx, WhisperXWrapper):
                # Recreate the wrapper to ensure a clean reload.
                wx_old = self.pipeline.whisperx
                self.pipeline.whisperx = WhisperXWrapper(
                    model_name=wx_old.model_name,
                    device=wx_old.device,
                    compute_type=wx_old.compute_type,
                    model_root=wx_old.model_root,
                )
        except Exception:  # noqa: BLE001
            # Fallback: rebuild pipeline with desired model
            self.pipeline = LocalPipeline(
                enable_diarization=bool(self.chk_diar.isChecked()),
                enable_dialog_blocks=bool(self.chk_dialog.isChecked()),
                whisper_model=model,
            )


class PipelineWorker(QObject):
    """Background worker that runs the LocalPipeline over one or more inputs."""

    progress = Signal(float, str)
    log = Signal(str)
    error = Signal(str)
    canceled = Signal()
    finished = Signal(list, str)  # (output_paths, view_text)
    # * Per-file lifecycle signals
    file_started = Signal(str)  # absolute media path
    file_finished = Signal(str, list)  # absolute media path, outputs for this file

    def __init__(
        self,
        pipeline: object,
        inputs: list[Path],
        out_dir: Path,
        options: dict[str, object],
    ) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._inputs = inputs
        self._out_dir = out_dir
        self._opts = options
        self._cancel = False
        self._closing = False
        # Track live LocalPipeline instances per job index to allow best-effort
        # resource release on user cancellation.
        self._live_pipelines: dict[int, object] = {}
        # Serializer for heavy CUDA cleanup operations to prevent race conditions
        self._cleanup_lock = threading.Lock()

    def set_closing(self) -> None:
        """Mark the worker as closing to skip aggressive resource cleanup."""
        self._closing = True

    def request_cancel(self) -> None:
        """Request cancellation of processing."""
        self._cancel = True
        # Emit log for visibility in console runs
        with contextlib.suppress(Exception):
            get_logger(__name__).info("Cancel requested by user")
        # Do not unload live models here to avoid races with in-flight kernels.
        # Unload happens in job's finally block after the model call returns.

    def _report(self, msg: str, frac: float) -> None:
        """Emit progress signal."""
        self.progress.emit(frac, msg)

    def _raise_canceled(self) -> None:
        """Raise a cancellation error to abort processing cleanly."""
        msg = "Canceled"
        raise CancelledByUserError(msg)

    def _export_primary(self, doc: Any, media: Path, fmt: str) -> Path:  # noqa: ANN401
        if fmt.lower() == "none":
            return self._out_dir / f"{media.stem}.skip"
        out_path = self._out_dir / f"{media.stem}.{fmt}"
        if fmt.lower() == "srt":
            lw_obj = self._opts.get("subtitle_max_line_width", 42)
            ml_obj = self._opts.get("subtitle_max_lines", 2)
            lw_val = int(lw_obj) if isinstance(lw_obj, (int, str)) else 42
            ml_val = int(ml_obj) if isinstance(ml_obj, (int, str)) else 2
            rules = SubtitleRules(max_line_chars=lw_val, max_lines=ml_val)
            out_path.write_text(export_srt_with_rules(doc, rules), encoding="utf-8")
            return out_path
        return export_document(doc, fmt, out_path)

    def _maybe_export_srt(
        self,
        doc: Any,  # noqa: ANN401
        media: Path,
        fmt: str,
        outputs: list[str],
    ) -> Path:
        srt_path = self._out_dir / f"{media.stem}.srt"
        save_srt = bool(self._opts.get("save_srt", True))
        if fmt.lower() == "none":
            need_srt = save_srt
        else:
            need_srt = (fmt.lower() != "srt" and save_srt) or (fmt.lower() == "srt")
        if need_srt:
            try:
                lw_obj2 = self._opts.get("subtitle_max_line_width", 42)
                ml_obj2 = self._opts.get("subtitle_max_lines", 2)
                lw_val2 = int(lw_obj2) if isinstance(lw_obj2, (int, str)) else 42
                ml_val2 = int(ml_obj2) if isinstance(ml_obj2, (int, str)) else 2
                rules = SubtitleRules(max_line_chars=lw_val2, max_lines=ml_val2)
                srt_text = export_srt_with_rules(doc, rules)
                # * Append AskVLM metadata so the status scan can detect quality.
                qual = str(self._opts.get("quality", "")).lower() or None
                if qual in {"fast", "good"}:
                    srt_text = append_askvlm_metadata_to_srt(
                        srt_text,
                        tool_name="AskVLM",
                        quality=qual,
                        completed=True,
                    )
                # * If requested via options, stretch gaps via the NoEmpty path.
                if bool(self._opts.get("no_empty", False)):
                    with contextlib.suppress(Exception):
                        srt_text = fill_empty_gaps_in_srt(srt_text)
                srt_path.write_text(srt_text, encoding="utf-8")
                if fmt.lower() != "srt":
                    outputs.append(str(srt_path))
            except Exception as ex:  # noqa: BLE001
                self.log.emit(f"SRT export failed: {ex}")
        return srt_path

    def _cleanup_partial_file(self, media: Path) -> None:
        """Remove streaming partial transcript file for the given media if present."""
        try:
            p = self._out_dir / f"{media.stem}.partial.txt"
            if p.exists():
                with contextlib.suppress(OSError):
                    p.unlink()
        except Exception:  # noqa: BLE001
            return

    def run(self) -> None:  # noqa: C901, PLR0912, PLR0915
        """Execute the processing pipeline with optional parallelism.

        - In 'fast' quality mode, run up to 2 files in parallel (GPU + CPU).
        - Otherwise, run sequentially, but if the next file is 10x shorter than the
          currently running GPU job, schedule it concurrently on CPU.
        """
        try:
            outputs_all: list[str] = []
            view_text_global: str = ""
            files: list[Path] = list(self._inputs)
            total = max(1, len(files))
            if not files:
                self.error.emit("No inputs provided")
                return

            # * Resolve durations for scheduling decisions
            dur_map: dict[Path, float] = {}
            for p in files:
                with contextlib.suppress(Exception):
                    dur_map[p] = max(0.0, get_media_duration_seconds(p))
                dur_map.setdefault(p, 0.0)
            with contextlib.suppress(Exception):
                self.log.emit(
                    "Durations: "
                    + ", ".join(f"{p.name}={dur_map.get(p, 0.0):.2f}s" for p in files)
                )

            # * Options snapshot
            fmt = str(self._opts.get("export_format", "txt"))
            lw_obj = self._opts.get("subtitle_max_line_width", 42)
            ml_obj = self._opts.get("subtitle_max_lines", 2)
            lw_i = int(lw_obj) if isinstance(lw_obj, (int, str)) else 42
            ml_i = int(ml_obj) if isinstance(ml_obj, (int, str)) else 2
            quality = str(self._opts.get("quality", "")).lower()
            gpu_slots = 2 if quality == "fast" else 1
            cpu_slots = 1

            # * Helper to clone pipeline for a specific device
            def _make_pipeline_for_device(device: str) -> Any:  # noqa: ANN401
                """Create an isolated pipeline instance for the target device.

                This avoids sharing a single Whisper model object across parallel
                jobs. In Fast mode, two CUDA jobs run concurrently, each with its
                own faster-whisper model instance (small), which fits typical VRAM
                budgets. Diarization remains heavy and is serialized elsewhere.
                """
                if not isinstance(self._pipeline, LocalPipeline):
                    return self._pipeline
                base = self._pipeline
                # * Reuse the base pipeline in non-Fast mode (sequential GPU) to
                # * avoid unnecessary reloads of large models (good quality).
                if device != "cpu" and gpu_slots <= 1:
                    return base
                # * Otherwise (Fast mode CUDA or any CPU job), clone to ensure
                # * independent model instances per job.
                return LocalPipeline(
                    model_root=base.model_root,
                    whisper_model=base.whisperx.model_name,
                    llm_model=base.llm_model_name,
                    engine=base.engine,
                    enable_diarization=base.enable_diarization,
                    enable_dialog_blocks=base.enable_dialog_blocks,
                    language=base.language,
                    device=device,
                    compute_type=base.compute_type,
                )

            completed_count = 0

            # * Shared progress callback factory for each job (index-based)
            def make_job_cb(
                job_index: int, start_ts: float
            ) -> Callable[[str, float], None]:
                def _cb(msg: str, f: float) -> None:
                    inner = max(0.0, min(1.0, f))
                    frac_overall = min(0.99, (completed_count + inner) / total)
                    elapsed = max(0.0, time.time() - start_ts)
                    inner_safe = max(1e-4, inner if inner > 0 else 0.0001)
                    est_total = elapsed / inner_safe
                    eta = max(0.0, est_total - elapsed)
                    msg2 = (
                        f"[{job_index + 1}/{total}] {msg} "
                        f"(elapsed {_format_eta(elapsed)} • ETA {_format_eta(eta)})"
                    )
                    self._report(msg2, frac_overall)
                    if self._cancel:
                        # Raise cancel via dedicated helper to keep ruff satisfied
                        self._raise_canceled()

                return _cb

            def _job(
                media: Path, device: str, job_index: int
            ) -> tuple[Path, list[str], str]:
                """Run end-to-end processing for one media on the given device."""
                # Emit start
                t_start = time.time()
                try:
                    self.file_started.emit(str(media.resolve()))
                except OSError:
                    self.file_started.emit(str(media))
                local_outputs: list[str] = []
                try:
                    pl = _make_pipeline_for_device(device)
                    # Register live pipeline for potential Cancel-time cleanup
                    self._live_pipelines[job_index] = pl
                    with contextlib.suppress(Exception):
                        self.log.emit(
                            f"Job start: idx={job_index} device={device} "
                            f"media={media.name}"
                        )
                    cb = make_job_cb(job_index, time.time())
                    doc = pl.process(
                        media,
                        self._out_dir,
                        progress=cb,
                        should_cancel=lambda: bool(self._cancel),
                        subtitle_max_line_width=lw_i,
                        subtitle_max_lines=ml_i,
                    )
                    # Export primary and optional SRT
                    out_primary = self._export_primary(doc, media, fmt)
                    local_outputs.append(str(out_primary))
                    self._maybe_export_srt(doc, media, fmt, local_outputs)
                    view_text_local = (
                        doc.get_full_text()
                        if bool(self._opts.get("single_view", False))
                        else ""
                    )
                    with contextlib.suppress(Exception):
                        elapsed = max(0.0, time.time() - t_start)
                        self.log.emit(
                            f"Job done: idx={job_index} device={device} "
                            f"media={media.name} elapsed={_format_eta(elapsed)}"
                        )
                    return media, local_outputs, view_text_local
                finally:
                    # Always attempt to remove partial file for this media
                    self._cleanup_partial_file(media)
                    # Remove from live pipelines registry
                    with contextlib.suppress(Exception):
                        self._live_pipelines.pop(job_index, None)
                    # * Unload Whisper model only for cloned pipelines to keep
                    # * the base pipeline (good quality) resident and stable.
                    try:
                        if isinstance(pl, LocalPipeline) and pl is not self._pipeline:
                            wx = getattr(pl, "whisperx", None)
                            unload_fn = getattr(wx, "unload", None)
                            if callable(unload_fn) and not self._closing:
                                # Force full cleanup (sync + empty_cache) if canceling.
                                # * Skip entirely if closing to let OS reclaim memory
                                # * and avoid potential driver crashes at process exit.
                                with self._cleanup_lock:
                                    unload_fn(safe=not self._cancel)
                    except Exception as _unload_ex:  # noqa: BLE001
                        get_logger(__name__).debug(
                            "Unload cleanup ignored: %s", _unload_ex
                        )

            # * Scheduler
            idx = 0
            running: dict[
                Future[tuple[Path, list[str], str]], tuple[Path, str, int]
            ] = {}

            def _count_running(device: str) -> int:
                return sum(1 for _f, (_m, d, _i) in running.items() if d == device)

            def _gpu_medias_running() -> list[Path]:
                return [m for _f, (m, d, _i) in running.items() if d == "cuda"]

            def _dur_gpu_ref() -> float:
                meds = _gpu_medias_running()
                if not meds:
                    return 0.0
                return max(dur_map.get(m, 0.0) for m in meds)

            def _can_schedule_cpu(next_media: Path) -> bool:
                if _count_running("cuda") <= 0:
                    return False
                if _count_running("cpu") >= cpu_slots:
                    return False
                dur_ref = _dur_gpu_ref()
                dn = dur_map.get(next_media, 0.0)
                return dur_ref > 0.0 and dn > 0.0 and dn <= (dur_ref / 10.0)

            pool = ThreadPoolExecutor(max_workers=(gpu_slots + cpu_slots))
            try:
                was_canceled = False
                # Seed initial GPU jobs
                while idx < len(files) and _count_running("cuda") < gpu_slots:
                    media = files[idx]
                    fut = pool.submit(_job, media, "cuda", idx)
                    running[fut] = (media, "cuda", idx)
                    idx += 1
                    with contextlib.suppress(Exception):
                        self.log.emit(f"Scheduled CUDA: idx={idx} media={media.name}")

                # One-time early CPU scheduling attempt while GPU is running
                if (
                    idx < len(files)
                    and _count_running("cuda") > 0
                    and _count_running("cpu") < cpu_slots
                ):
                    next_media0 = files[idx]
                    if _can_schedule_cpu(next_media0):
                        fut0 = pool.submit(_job, next_media0, "cpu", idx)
                        running[fut0] = (next_media0, "cpu", idx)
                        idx += 1
                        with contextlib.suppress(Exception):
                            self.log.emit(
                                f"Scheduled CPU (early): idx={idx} "
                                f"media={next_media0.name}"
                            )

                while running:
                    if self._cancel:
                        was_canceled = True
                        # Keep draining running jobs without scheduling new ones.
                    done, _pending = wait(
                        list(running.keys()), return_when=FIRST_COMPLETED, timeout=0.5
                    )
                    for fut in done:
                        media, _device, job_index = running.pop(fut)
                        try:
                            m_out, outs, view_text_local = fut.result()
                            outputs_all.extend(outs)
                            if view_text_local and not view_text_global:
                                view_text_global = view_text_local
                            completed_count += 1
                            try:
                                self.file_finished.emit(str(m_out.resolve()), outs)
                            except OSError:
                                self.file_finished.emit(str(m_out), outs)
                            # Update overall progress discretely after completion
                            self._report(
                                f"[{job_index + 1}/{total}] Exported",
                                min(0.99, completed_count / total),
                            )
                        except (CancelledByUserError, CancelledError):
                            was_canceled = True
                            # Do not schedule further tasks; break outer loop soon
                        except Exception as ex:  # noqa: BLE001
                            with contextlib.suppress(Exception):
                                self.log.emit(
                                    f"Job error: idx={job_index} "
                                    f"media={media.name} ex={ex}"
                                )
                            self.log.emit(f"Skipping '{media}': {ex}")
                            completed_count += 1
                            self._report(
                                f"[{job_index + 1}/{total}] Skipped (error)",
                                min(0.99, completed_count / total),
                            )

                    # Fill GPU slots first (skip scheduling after cancel)
                    if not was_canceled and not self._cancel:
                        while idx < len(files) and _count_running("cuda") < gpu_slots:
                            media2 = files[idx]
                            fut2 = pool.submit(_job, media2, "cuda", idx)
                            running[fut2] = (media2, "cuda", idx)
                            idx += 1
                            with contextlib.suppress(Exception):
                                self.log.emit(
                                    f"Scheduled CUDA: idx={idx} media={media2.name}"
                                )

                    # Then optionally schedule one CPU job based on heuristic
                    if idx < len(files) and not was_canceled and not self._cancel:
                        next_media = files[idx]
                        if _can_schedule_cpu(next_media):
                            fut3 = pool.submit(_job, next_media, "cpu", idx)
                            running[fut3] = (next_media, "cpu", idx)
                            idx += 1
                            with contextlib.suppress(Exception):
                                self.log.emit(
                                    f"Scheduled CPU: idx={idx} media={next_media.name}"
                                )
            finally:
                # Always shut down the pool after draining queued work.
                with contextlib.suppress(Exception):
                    pool.shutdown(wait=True, cancel_futures=True)

            # Finalize
            if was_canceled or self._cancel:
                # After cancel and drain, free the base Whisper model.
                with contextlib.suppress(Exception):
                    if isinstance(self._pipeline, LocalPipeline):
                        wxb = getattr(self._pipeline, "whisperx", None)
                        unload_b = getattr(wxb, "unload", None)
                        if callable(unload_b) and not self._closing:
                            # Force full cleanup (sync + empty_cache) on cancel.
                            # * Skip entirely if closing to let OS reclaim memory
                            # * and avoid potential driver crashes at process exit.
                            with self._cleanup_lock:
                                unload_b(safe=False)
                self._report("Canceled", min(0.99, completed_count / max(1, total)))
                self.canceled.emit()
                return
            self._report("Completed", 1.0)
            self.log.emit("Processing completed successfully")
            if not outputs_all:
                self.error.emit("No valid inputs were processed")
                return
            self.finished.emit(outputs_all, view_text_global)
        except Exception as e:  # noqa: BLE001
            self.error.emit(str(e))


def main() -> int:
    """Start the Qt application and show the main window."""
    # * Load project-local environment variables before any env-backed GUI settings.
    load_env_file(Path(__file__).resolve().parents[1] / ".env")
    get_logger(__name__).info("Application starting")
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    get_logger(__name__).info("MainWindow shown successfully")

    # Handle Ctrl+C (SIGINT): trigger Cancel instead of raising KeyboardInterrupt
    def _on_sigint(_sig: int, _frame: object) -> None:
        with contextlib.suppress(Exception):
            get_logger(__name__).info("SIGINT received -> requesting Cancel")
            w.request_cancel()

    with contextlib.suppress(Exception):
        signal.signal(signal.SIGINT, _on_sigint)

    try:
        return app.exec()
    except KeyboardInterrupt:
        # Graceful shutdown on Ctrl+C from console
        with contextlib.suppress(Exception):
            get_logger(__name__).info("KeyboardInterrupt: graceful shutdown")
            w.request_cancel()
            w.await_worker_shutdown(10000)
        return 130


if __name__ == "__main__":  # pragma: no cover - manual run path
    raise SystemExit(main())
