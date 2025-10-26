from __future__ import annotations

import contextlib
import os
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from PySide6.QtCore import QByteArray, QObject, QSettings, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QFontDatabase, QImage
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QCompleter,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.ffmpeg import (
    extract_frame_to_file,
    get_media_duration_seconds,
    start_burn_process,
)
from core.pipelines import LocalPipeline
from core.whisperx_wrapper import WhisperXWrapper
from gui.speaker_sidebar import SpeakerSidebar
from gui.subtitle_preview import SubtitlePreview
from gui.wysiwyg_editor import TableRow, WysiwygEditor
from utils.exporters import SubtitleRules, export_document, export_srt_with_rules
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
            with contextlib.suppress(OSError):
                self._proc.terminate()

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
    from subprocess import Popen

    from PySide6.QtGui import QCloseEvent

# * Constants for boolean UI states
_UI_CHECKED = True
_UI_UNCHECKED = False


class CancelledByUserError(Exception):
    """Raised internally to abort processing when user requests cancel."""


# * Main window for Artemonim's Speech Kit GUI
class MainWindow(QMainWindow):
    """Application main window with Quick Transcribe controls and text viewer."""

    # * Constants
    MAX_SPEAKER_LEN = 64

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
        self.setWindowTitle("Artemonim's Speech Kit")
        self.resize(800, 600)

        # * Menu bar setup
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")

        open_action = QAction("Open...", self)
        open_action.triggered.connect(self.choose_file)
        file_menu.addAction(open_action)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # * Central widget and main vertical layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # (Removed legacy top input row; input management lives in the Input tab)

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
        self.btn_cancel.setEnabled(_UI_UNCHECKED)
        opts_row.addWidget(self.btn_start)
        opts_row.addWidget(self.btn_cancel)
        layout.addLayout(opts_row)

        # * Subtitle rules row (Phase 1.81)
        rules_row = QHBoxLayout()
        rules_row.addWidget(QLabel("Line length:"))
        self.spin_line_len = QSpinBox()
        self.spin_line_len.setRange(20, 120)
        self.spin_line_len.setValue(42)
        rules_row.addWidget(self.spin_line_len)
        rules_row.addWidget(QLabel("Lines:"))
        self.spin_max_lines = QSpinBox()
        self.spin_max_lines.setRange(1, 3)
        self.spin_max_lines.setValue(2)
        rules_row.addWidget(self.spin_max_lines)
        rules_row.addStretch(1)
        layout.addLayout(rules_row)

        # * Burn-in row
        burn_row = QHBoxLayout()
        self.btn_burn = QPushButton("Burn")
        self.chk_normalize = QCheckBox("Normalization")
        self.chk_normalize.setChecked(_UI_CHECKED)
        burn_row.addWidget(self.btn_burn)
        burn_row.addWidget(self.chk_normalize)
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
            self.font_combo.setEditable(_UI_CHECKED)
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
        self.input_list = QListWidget()
        self.input_list.setSelectionMode(
            self.input_list.SelectionMode.ExtendedSelection
        )
        in_layout.addWidget(self.input_list, 1)
        in_controls = QHBoxLayout()
        self.btn_in_add_file = QPushButton("Add File(s)…")
        self.btn_in_remove = QPushButton("Remove")
        self.btn_in_up = QPushButton("Up")
        self.btn_in_down = QPushButton("Down")
        for b in (
            self.btn_in_add_file,
            self.btn_in_remove,
            self.btn_in_up,
            self.btn_in_down,
        ):
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
        self._has_transcript: bool = False
        # * Preview mapping and state
        self._last_inputs: list[Path] = []
        self._tab_to_media: dict[str, Path] = {}

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
        self.btn_in_up.clicked.connect(self._log_wrap(self._input_move_up, "Input Up"))
        self.btn_in_down.clicked.connect(
            self._log_wrap(self._input_move_down, "Input Down")
        )
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

        # * Ensure at least one empty tab is visible
        self._clear_tabs()
        self._add_tab("Document", "", None)

        # * Connect editor selection changes to preview updates
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # * Load persisted settings
        self._load_settings()
        # Burn is disabled until transcript exists
        self.btn_burn.setEnabled(_UI_UNCHECKED)
        # Initialize preview font
        self.preview.set_font_family(self.font_combo.currentText())
        self._font_px: int | None = None
        # Apply initial quality
        self._apply_quality_to_pipeline()

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

    def start_processing(self) -> None:
        """Start pipeline processing in a background thread (QThread)."""
        # Validate input
        inputs = self._gather_inputs()
        if not inputs:
            QMessageBox.information(
                self,
                "No input",
                "Please add inputs in the Input tab.",
            )
            return
        # * Remember inputs for preview/tab mapping
        self._last_inputs = inputs
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
        self._worker = PipelineWorker(self.pipeline, inputs, out_dir, opts)
        self._worker.moveToThread(self._thread)

        # Connect signals
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.on_progress)
        self._worker.log.connect(self.on_log)
        self._worker.error.connect(self.on_error)
        self._worker.canceled.connect(self.on_canceled)
        self._worker.finished.connect(self.on_finished)
        # Ensure cleanup
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    def request_cancel(self) -> None:
        """Request cancellation from the worker (best effort)."""
        if self._worker is not None:
            self._worker.request_cancel()
            self.status.showMessage("Cancel requested…")
            self.btn_cancel.setEnabled(_UI_UNCHECKED)
        if isinstance(self._burn_worker, _InlineBurnWorker):
            self._burn_worker.request_cancel()

    def on_progress(self, frac: float, msg: str) -> None:
        """Update progress bar and status message."""
        # Show progress percentage, elapsed and ETA when encoded in msg
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

    def on_error(self, message: str) -> None:
        """Handle processing error."""
        self._set_controls_enabled(enabled=True)
        self.progress.setValue(0)
        # Also print to console for visibility when launched from terminal
        with contextlib.suppress(Exception):
            pass
        if self._can_show_modal():
            QMessageBox.critical(self, "Processing error", message)
        self.status.showMessage("Error")

    def on_canceled(self) -> None:
        """Handle processing cancellation."""
        self._set_controls_enabled(enabled=True)
        self.progress.setValue(0)
        self.status.showMessage("Canceled")

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
            self._add_tab(p.stem, content, self._find_input_media_by_stem(p.stem))
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
            self._add_tab(p.stem, content, self._find_input_media_by_stem(p.stem))

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
            # No selection: clear preview text only
            self.preview.set_text_lines([])
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
        items = [
            Path(self.input_list.item(i).text()) for i in range(self.input_list.count())
        ]
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
        for i in range(self.input_list.count()):
            p = Path(self.input_list.item(i).text())
            if p.is_file() and p.stem == stem:
                return p
        return None

    def start_burn(self) -> None:
        """Start burn-in process for selected inputs using burn settings."""
        if not self.input_path or self.input_mode not in {"file", "folder"}:
            QMessageBox.information(self, "No input", "Please choose a file or folder.")
            return
        if not self._has_transcript:
            QMessageBox.information(self, "No transcript", "Please transcribe first.")
            return
        out_dir = Path(self.out_dir_edit.text()).resolve()
        normalize = self.chk_normalize.isChecked()
        font_name = self.font_combo.currentText().strip() or None
        # Build inputs list (videos only)
        if self.input_mode == "file":
            inputs = (
                [self.input_path]
                if self.input_path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}
                else []
            )
        else:
            patterns = ["*.mp4", "*.mov", "*.mkv", "*.avi"]
            inputs = []
            for pat in patterns:
                inputs.extend(Path(self.input_path).glob(pat))
        if not inputs:
            QMessageBox.information(self, "No video", "No video files found to burn.")
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
        # Trigger global cancel
        with contextlib.suppress(Exception):
            self.request_cancel()
            # Give threads time to stop
            if self._thread is not None and self._thread.isRunning():
                self._thread.quit()
                self._thread.wait(2000)
            if self._burn_thread is not None and self._burn_thread.isRunning():
                self._burn_thread.quit()
                self._burn_thread.wait(2000)
        event.accept()

    # * Settings persistence
    def _load_settings(self) -> None:
        s = QSettings("Artemonim", "SpeechKit")
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
        s = QSettings("Artemonim", "SpeechKit")
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
        # Phase 1.81
        s.setValue("subs/max_line_chars", int(self.spin_line_len.value()))
        s.setValue("subs/max_lines", int(self.spin_max_lines.value()))

    # * Tabs helpers
    def _clear_tabs(self) -> None:
        while self.tabs.count() > 0:
            self.tabs.removeTab(0)

    def _add_tab(self, title: str, text: str, media: Path | None = None) -> None:  # noqa: PLR0915, C901
        editor = WysiwygEditor()
        # Parse plain text into table rows: "[hh:mm:ss.mmm --> hh:mm:ss.mmm] speaker: text"
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
            s = QSettings("Artemonim", "SpeechKit")
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
        # Persist column widths on resize changes
        editor.horizontalHeader().sectionResized.connect(
            lambda _i, _o, _n, ed=editor: self._save_table_widths(ed)
        )
        # * Register tab and map to media (for preview frame extraction)
        self.tabs.addTab(editor, title)
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
            s = QSettings("Artemonim", "SpeechKit")
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
        items = (
            [self.input_list.item(i).text() for i in range(self.input_list.count())]
            if hasattr(self, "input_list") and self.input_list.count() > 0
            else []
        )
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
                self.input_list.addItem(QListWidgetItem(str(p)))
                continue
            if p.is_file():
                self.last_input_dir = p.parent
                self.input_list.addItem(QListWidgetItem(str(p)))

    # (Removed _input_add_folder)

    def _input_remove_selected(self) -> None:
        for item in self.input_list.selectedItems():
            row = self.input_list.row(item)
            self.input_list.takeItem(row)

    # (Removed _input_clear)

    def _input_move_up(self) -> None:
        sel = self.input_list.selectedItems()
        if not sel:
            return
        row = self.input_list.row(sel[0])
        if row <= 0:
            return
        item = self.input_list.takeItem(row)
        self.input_list.insertItem(row - 1, item)
        self.input_list.setCurrentItem(item)

    def _input_move_down(self) -> None:
        sel = self.input_list.selectedItems()
        if not sel:
            return
        row = self.input_list.row(sel[0])
        if row >= self.input_list.count() - 1:
            return
        item = self.input_list.takeItem(row)
        self.input_list.insertItem(row + 1, item)
        self.input_list.setCurrentItem(item)

    # * Quality toggle
    def _toggle_quality(self) -> None:
        self._quality_mode = "fast" if self._quality_mode == "good" else "good"
        self.btn_quality.setText(
            "Quality: Good" if self._quality_mode == "good" else "Quality: Fast"
        )

    def _apply_quality_to_pipeline(self, *, force_reload: bool = False) -> None:
        model = "large-v3" if self._quality_mode == "good" else "small"
        try:
            # Update underlying wrapper and force reload next time
            self.pipeline.whisperx.model_name = model
            if force_reload and isinstance(self.pipeline.whisperx, WhisperXWrapper):
                # Recreate wrapper to avoid private member access and ensure clean reload
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

    def __init__(
        self,
        pipeline: LocalPipeline,
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

    def request_cancel(self) -> None:
        """Request cancellation of processing."""
        self._cancel = True

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
                srt_path.write_text(srt_text, encoding="utf-8")
                if fmt.lower() != "srt":
                    outputs.append(str(srt_path))
            except Exception as ex:  # noqa: BLE001
                self.log.emit(f"SRT export failed: {ex}")
        return srt_path

    def run(self) -> None:  # noqa: C901
        """Execute the processing pipeline."""
        try:
            outputs: list[str] = []
            view_text: str = ""
            total = max(1, len(self._inputs))

            def make_cb(
                prefix: str, base: float, start_ts: float
            ) -> Callable[[str, float], None]:
                def _cb(msg: str, f: float) -> None:
                    frac_overall = min(0.99, base + max(0.0, min(1.0, f)) / total)
                    elapsed = max(0.0, time.time() - start_ts)
                    inner = max(1e-4, min(0.9999, f if f > 0 else 0.0001))
                    est_total = elapsed / inner
                    eta = max(0.0, est_total - elapsed)
                    msg2 = f"{msg} (elapsed {_format_eta(elapsed)} • ETA {_format_eta(eta)})"
                    self._report(prefix + msg2, frac_overall)
                    if self._cancel:
                        self._raise_canceled()

                return _cb

            for idx, media in enumerate(self._inputs):
                if self._cancel:
                    self.canceled.emit()
                    return
                prefix = f"[{idx + 1}/{total}] " if total > 1 else ""
                base = idx / total
                cb = make_cb(prefix, base, time.time())

                try:
                    lw = self._opts.get("subtitle_max_line_width", 42)
                    ml = self._opts.get("subtitle_max_lines", 2)
                    lw_i = int(lw) if isinstance(lw, (int, str)) else 42
                    ml_i = int(ml) if isinstance(ml, (int, str)) else 2
                    doc = self._pipeline.process(
                        media,
                        self._out_dir,
                        progress=cb,
                        should_cancel=lambda: bool(self._cancel),
                        subtitle_max_line_width=lw_i,
                        subtitle_max_lines=ml_i,
                    )
                    fmt = str(self._opts.get("export_format", "txt"))
                    out_primary = self._export_primary(doc, media, fmt)
                    outputs.append(str(out_primary))
                    self._maybe_export_srt(doc, media, fmt, outputs)
                    if bool(self._opts.get("single_view", False)):
                        view_text = doc.get_full_text()
                    self._report(prefix + "Exported", min(0.99, (idx + 1) / total))
                except CancelledByUserError:
                    # Treat cancel as a normal early-exit path
                    self.canceled.emit()
                    return
                except Exception as ex:  # noqa: BLE001
                    self.log.emit(f"Skipping '{media}': {ex}")
                    self._report(
                        prefix + "Skipped (error)", min(0.99, (idx + 1) / total)
                    )
                    continue

            self._report("Completed", 1.0)
            self.log.emit("Processing completed successfully")
            if not outputs:
                # Emit error directly without raising
                self.error.emit("No valid inputs were processed")
                return
            self.finished.emit(outputs, view_text)
        except Exception as e:  # noqa: BLE001
            self.error.emit(str(e))


def main() -> int:
    """Start the Qt application and show the main window."""
    get_logger(__name__).info("Application starting")
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    get_logger(__name__).info("MainWindow shown successfully")
    return app.exec()


if __name__ == "__main__":  # pragma: no cover - manual run path
    raise SystemExit(main())
