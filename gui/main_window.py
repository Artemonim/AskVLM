from __future__ import annotations

import contextlib
import os
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from PySide6.QtCore import QObject, QSettings, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QFontDatabase
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QCompleter,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.audio_io import cleanup_intermediate_audio
from core.ffmpeg import get_media_duration_seconds, start_burn_process
from core.pipelines import LocalPipeline
from gui.speaker_sidebar import SpeakerSidebar
from gui.wysiwyg_editor import TableRow, WysiwygEditor
from utils.exporters import export_document
from utils.logging import get_logger

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

        # * Input selection row (file or folder)
        input_row = QHBoxLayout()
        self.btn_choose_file = QPushButton("Choose File…")
        self.btn_choose_folder = QPushButton("Choose Folder…")
        self.lbl_input = QLabel("No input selected")
        self.lbl_input.setMinimumWidth(300)
        input_row.addWidget(self.btn_choose_file)
        input_row.addWidget(self.btn_choose_folder)
        input_row.addWidget(self.lbl_input, 1)
        layout.addLayout(input_row)

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
        # Start/Cancel moved to options row
        self.btn_start = QPushButton("Start")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(_UI_UNCHECKED)
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
        burn_row.addStretch(1)
        layout.addLayout(burn_row)

        # * Speaker sidebar + tabbed viewers
        splitter = QSplitter()
        self.sidebar = SpeakerSidebar()
        splitter.addWidget(self.sidebar)
        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        # * Status bar with progress
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status.addPermanentWidget(self.progress)
        self.status.showMessage("Ready")

        # * Runtime/session state
        self.pipeline = LocalPipeline()
        self.input_mode: Literal["file", "folder"] | None = None
        self.input_path: Path | None = None
        self.last_input_dir: Path = Path.cwd()
        self.last_output_dir: Path = Path(self.out_dir_edit.text()).resolve()
        self._thread: QThread | None = None
        self._worker: PipelineWorker | None = None
        self._burn_thread: QThread | None = None
        self._burn_worker: BurnWorker | None = None
        self._has_transcript: bool = False

        # * Wire up signals
        self.btn_choose_file.clicked.connect(
            self._log_wrap(self.choose_file, "Choose File")
        )
        self.btn_choose_folder.clicked.connect(
            self._log_wrap(self.choose_folder, "Choose Folder")
        )
        self.out_dir_btn.clicked.connect(
            self._log_wrap(self.choose_output_dir, "Choose Output Dir")
        )
        self.btn_start.clicked.connect(self._log_wrap(self.start_processing, "Start"))
        self.btn_cancel.clicked.connect(self._log_wrap(self.request_cancel, "Cancel"))
        self.btn_open_out.clicked.connect(
            self._log_wrap(self.open_output_folder, "Open Output Folder")
        )
        self.btn_burn.clicked.connect(self._log_wrap(self.start_burn, "Burn"))

        # * Ensure at least one empty tab is visible
        self._clear_tabs()
        self._add_tab("Document", "")

        # * Load persisted settings
        self._load_settings()
        # Burn is disabled until transcript exists
        self.btn_burn.setEnabled(_UI_UNCHECKED)

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
            self.lbl_input.setText(str(p))
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
            self.lbl_input.setText(str(p))
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
        self.btn_choose_file.setEnabled(enabled)
        self.btn_choose_folder.setEnabled(enabled)
        self.out_dir_btn.setEnabled(enabled)
        self.out_dir_edit.setEnabled(enabled)
        self.chk_diar.setEnabled(enabled)
        self.chk_dialog.setEnabled(enabled)
        self.chk_save_srt.setEnabled(enabled)
        self.format_combo.setEnabled(enabled)
        self.btn_start.setEnabled(enabled)
        self.btn_cancel.setEnabled(not enabled)
        if self._burn_worker is None:
            self.btn_burn.setEnabled(enabled)
            self.chk_normalize.setEnabled(enabled)
            self.font_combo.setEnabled(enabled)

    def start_processing(self) -> None:
        """Start pipeline processing in a background thread (QThread)."""
        # Validate input
        if not self.input_path or self.input_mode not in {"file", "folder"}:
            QMessageBox.information(self, "No input", "Please choose a file or folder.")
            return
        out_dir = Path(self.out_dir_edit.text()).resolve()
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, "Output error", f"Cannot create output dir: {e}")
            return

        # Build inputs list
        if self.input_mode == "file":
            inputs = [self.input_path]
        else:
            # Accept common media extensions
            patterns = ["*.wav", "*.mp3", "*.mp4", "*.avi", "*.mkv"]
            inputs = []
            for pat in patterns:
                inputs.extend(self.input_path.glob(pat))
            if not inputs:
                QMessageBox.information(
                    self, "No media", "No media files found in folder."
                )
                return

        # Configure worker options
        opts = {
            "enable_diarization": self.chk_diar.isChecked(),
            "enable_dialog_blocks": self.chk_dialog.isChecked(),
            "export_format": str(self.format_combo.currentText()),
            "single_view": self.input_mode == "file",
            "save_srt": self.chk_save_srt.isChecked(),
        }

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
        if self._burn_worker is not None:
            self._burn_worker.request_cancel()

    def on_progress(self, frac: float, msg: str) -> None:
        """Update progress bar and status message."""
        self.progress.setValue(int(max(0.0, min(1.0, frac)) * 100))
        if msg:
            self.status.showMessage(msg)

    def on_log(self, line: str) -> None:
        """Handle log message from worker (currently unused)."""
        # Minimal Phase 1.7: surface critical warnings in GUI
        if not line:
            return
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
            if view_text:
                self._add_tab("Document", view_text)
            else:
                # * Populate one tab per output artifact (show raw text for any format)
                for out_str in _outputs:
                    try:
                        p = Path(out_str)
                        content = p.read_text(encoding="utf-8")
                    except OSError:
                        content = ""
                    self._add_tab(p.stem, content)
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

    def open_output_folder(self) -> None:
        """Open the output directory in the system file manager."""
        out_dir = Path(self.out_dir_edit.text()).resolve()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(out_dir)))

    def _log_wrap(self, func: Callable[..., Any], _name: str) -> Callable[..., Any]:
        def _wrapped(*args: object, **kwargs: object) -> object:
            return func(*args, **kwargs)

        return _wrapped

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
        self._burn_worker = BurnWorker(
            inputs, out_dir, normalize=normalize, font_name=font_name
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
        self.chk_normalize.setChecked(bool(s.value("burn/normalize", 1, type=bool)))
        font = str(s.value("burn/font", ""))
        if font:
            fidx = self.font_combo.findText(font)
            if fidx >= 0:
                self.font_combo.setCurrentIndex(fidx)

    def _save_settings(self) -> None:
        s = QSettings("Artemonim", "SpeechKit")
        s.setValue("opts/diar", self.chk_diar.isChecked())
        s.setValue("opts/dialog", self.chk_dialog.isChecked())
        s.setValue("opts/save_srt", self.chk_save_srt.isChecked())
        s.setValue("opts/format", self.format_combo.currentText())
        s.setValue("paths/out_dir", self.out_dir_edit.text())
        s.setValue("burn/normalize", self.chk_normalize.isChecked())
        s.setValue("burn/font", self.font_combo.currentText())

    # * Tabs helpers
    def _clear_tabs(self) -> None:
        while self.tabs.count() > 0:
            self.tabs.removeTab(0)

    def _add_tab(self, title: str, text: str) -> None:
        editor = WysiwygEditor()
        # Parse plain text into table rows: "[hh:mm:ss.mmm --> hh:mm:ss.mmm] speaker: text"
        # Regex for leading time range
        time_pat = re.compile(
            r"^(\d\d:\d\d:\d\d[,.]\d{3})\s*(?:-->|→|-+>?)\s*(\d\d:\d\d:\d\d[,.]\d{3})"
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
        self.tabs.addTab(editor, title)

    # * Testing helpers
    def get_editor_at(self, index: int) -> WysiwygEditor | None:
        """Return the tab editor at the given index or None if not present."""
        w = self.tabs.widget(index)
        return w if isinstance(w, WysiwygEditor) else None


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

    def run(self) -> None:
        """Execute the processing pipeline."""
        try:
            outputs: list[str] = []
            view_text: str = ""
            total = max(1, len(self._inputs))
            for idx, media in enumerate(self._inputs):
                if self._cancel:
                    self.canceled.emit()
                    return
                prefix = f"[{idx + 1}/{total}] " if total > 1 else ""
                cb = self._make_progress_cb(idx, total, prefix)
                self._apply_pipeline_options()
                try:
                    doc = self._pipeline.process(media, self._out_dir, progress=cb)
                except CancelledByUserError:
                    # Best-effort cleanup of intermediate audio
                    cleanup_intermediate_audio(media, self._out_dir)
                    self.canceled.emit()
                    return
                fmt = str(self._opts.get("export_format", "txt"))
                out_primary = self._export_primary(doc, media, fmt)
                outputs.append(str(out_primary))
                self._maybe_export_srt(doc, media, fmt, outputs)
                # Burn removed from pipeline; handled by separate BurnWorker
                if bool(self._opts.get("single_view", False)):
                    view_text = doc.get_full_text()
                self._report(prefix + "Exported", min(0.99, (idx + 1) / total))
            self._report("Completed", 1.0)
            self.log.emit("Processing completed successfully")
            self.finished.emit(outputs, view_text)
        except Exception as e:  # noqa: BLE001
            self.error.emit(str(e))

    # * Helpers to keep run() simple
    def _make_progress_cb(
        self, idx: int, total: int, prefix: str
    ) -> Callable[[str, float], None]:
        base = idx / total

        def cb(msg: str, f: float) -> None:
            self._report(prefix + msg, min(0.99, base + f / total))
            if self._cancel:
                raise CancelledByUserError

        return cb

    def _apply_pipeline_options(self) -> None:
        self._pipeline.enable_diarization = bool(
            self._opts.get("enable_diarization", False)
        )
        self._pipeline.enable_dialog_blocks = bool(
            self._opts.get("enable_dialog_blocks", False)
        )

    def _export_primary(self, doc: Any, media: Path, fmt: str) -> Path:  # noqa: ANN401
        if fmt.lower() == "none":
            # Skip exporting any primary text artifact
            return self._out_dir / f"{media.stem}.skip"
        out_path = self._out_dir / f"{media.stem}.{fmt}"
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
        # When fmt is 'none', use only the save_srt switch
        if fmt.lower() == "none":
            need_srt = save_srt
        else:
            need_srt = (fmt.lower() != "srt" and save_srt) or (fmt.lower() == "srt")
        if need_srt:
            try:
                export_document(doc, "srt", srt_path)
                if fmt.lower() != "srt":
                    outputs.append(str(srt_path))
            except Exception as ex:  # noqa: BLE001
                self.log.emit(f"SRT export failed: {ex}")
        return srt_path


class BurnWorker(QObject):
    """Background worker for cancellable ffmpeg burn-in operations."""

    progress = Signal(float, str)
    log = Signal(str)
    error = Signal(str)
    canceled = Signal()
    finished = Signal(list)  # output video paths

    def __init__(
        self,
        inputs: list[Path],
        out_dir: Path,
        *,
        normalize: bool,
        font_name: str | None,
    ) -> None:
        super().__init__()
        self._inputs = inputs
        self._out_dir = out_dir
        self._normalize = normalize
        self._font_name = font_name
        self._cancel = False
        self._proc: Popen[bytes] | None = None
        self._progress_file: Path | None = None

    def request_cancel(self) -> None:
        """Signal cancellation and attempt to terminate ffmpeg process."""
        self._cancel = True
        proc = self._proc
        if proc is not None:
            try:
                proc.terminate()
            except OSError as exc:
                get_logger(__name__).debug("Burn process terminate failed: %s", exc)

    def _parse_progress(self) -> float:
        """Parse ffmpeg -progress file; return out_time_ms if found else -1."""
        pf = self._progress_file
        if pf is None or not pf.exists():
            return -1.0
        last = -1.0
        try:
            for line in pf.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("out_time_ms="):
                    val = float(line.split("=", 1)[1])
                    last = val
        except Exception:  # noqa: BLE001
            last = -1.0
        return last

    def _wait_until_finished(
        self,
        proc: Popen[bytes],
        *,
        base_frac: float,
        end_frac: float,
        prefix: str,
        total_duration_s: float,
    ) -> int:
        """Poll process until it finishes or cancellation is requested.

        Computes progress from ffmpeg progress file if available, else interpolates.
        Returns process return code; returns -1 when canceled.
        """
        burn_progress_cap = 0.995
        poll_sleep_s = 0.25
        shown = base_frac
        while True:
            if self._cancel:
                with contextlib.suppress(OSError):
                    proc.terminate()
                return -1
            ret = proc.poll()
            if ret is not None:
                return ret
            # Real progress if possible
            p_ms = self._parse_progress()
            if p_ms >= 0 and total_duration_s > 0:
                ratio = min(1.0, (p_ms / 1000.0) / total_duration_s)
                shown = min(
                    burn_progress_cap, base_frac + ratio * (end_frac - base_frac)
                )
            else:
                # Interpolate progress towards cap
                target = min(
                    burn_progress_cap, end_frac - (end_frac - base_frac) * 0.02
                )
                shown = min(target, shown + 0.02)
            self.progress.emit(shown, prefix + "Burning subtitles")
            time.sleep(poll_sleep_s)

    def run(self) -> None:  # noqa: C901
        """Run burn-in sequentially for all inputs; supports cooperative cancel."""
        try:
            outputs: list[str] = []
            total = max(1, len(self._inputs))
            for idx, media in enumerate(self._inputs):
                if self._cancel:
                    self.canceled.emit()
                    return
                prefix = f"[{idx + 1}/{total}] " if total > 1 else ""
                # Expect SRT with same stem in out_dir
                srt_path = self._out_dir / f"{media.stem}.srt"
                if not srt_path.exists():
                    self.log.emit(f"SRT not found for burn: {srt_path}")
                    continue
                burned_out = self._out_dir / f"{media.stem}_subbed.mp4"
                # Setup ffmpeg progress file and probe duration
                prog_file = self._out_dir / f".{media.stem}.ffprogress"
                with contextlib.suppress(Exception):
                    if prog_file.exists():
                        prog_file.unlink()
                self._progress_file = prog_file
                # Initial progress bump
                self.progress.emit((idx + 0.1) / total, prefix + "Burning subtitles")
                proc = start_burn_process(
                    media,
                    srt_path,
                    burned_out,
                    normalize_audio=self._normalize,
                    font_name=self._font_name,
                    progress_path=prog_file,
                )
                self._proc = proc
                base = idx / total
                end = (idx + 1) / total
                # Total video duration for precise progress
                duration_s = get_media_duration_seconds(media)
                ret = self._wait_until_finished(
                    proc,
                    base_frac=base,
                    end_frac=end,
                    prefix=prefix,
                    total_duration_s=duration_s,
                )
                if ret == -1:
                    # Canceled: remove partial file
                    with contextlib.suppress(OSError):
                        if burned_out.exists():
                            burned_out.unlink()
                    with contextlib.suppress(OSError):
                        if prog_file.exists():
                            prog_file.unlink()
                    self.canceled.emit()
                    return
                if ret != 0:
                    self.log.emit("Burn-in failed")
                    with contextlib.suppress(OSError):
                        if burned_out.exists():
                            burned_out.unlink()
                    with contextlib.suppress(OSError):
                        if prog_file.exists():
                            prog_file.unlink()
                else:
                    outputs.append(str(burned_out))
                self.progress.emit((idx + 1) / total, prefix + "Burned")
            self.finished.emit(outputs)
        except Exception as e:  # noqa: BLE001
            self.error.emit(str(e))


# * Entry point for GUI application


def main() -> None:
    """Start the GUI application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
