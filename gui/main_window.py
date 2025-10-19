from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from PySide6.QtCore import QObject, QThread, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from core.pipelines import LocalPipeline
from gui.wysiwyg_editor import WysiwygEditor
from utils.exporters import export_document

if TYPE_CHECKING:
    from collections.abc import Callable


class CancelledByUserError(Exception):
    """Raised internally to abort processing when user requests cancel."""


# * Main window for Artemonim's Speech Kit GUI
class MainWindow(QMainWindow):
    """Application main window with Quick Transcribe controls and text viewer."""

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
        layout.addLayout(out_row)

        # * Options row: toggles and format
        opts_row = QHBoxLayout()
        self.chk_diar = QCheckBox("Diarization")
        self.chk_diar.setChecked(False)
        self.chk_dialog = QCheckBox("Dialog blocks")
        self.chk_dialog.setChecked(False)
        opts_row.addWidget(self.chk_diar)
        opts_row.addWidget(self.chk_dialog)
        opts_row.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["txt", "srt", "vtt", "json"])
        self.format_combo.setCurrentText("txt")
        opts_row.addWidget(self.format_combo)
        opts_row.addStretch(1)
        layout.addLayout(opts_row)

        # * Action buttons row
        actions_row = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_open_out = QPushButton("Open Output Folder")
        actions_row.addWidget(self.btn_start)
        actions_row.addWidget(self.btn_cancel)
        actions_row.addStretch(1)
        actions_row.addWidget(self.btn_open_out)
        layout.addLayout(actions_row)

        # * Text viewer
        self.editor = WysiwygEditor()
        self.editor.setReadOnly(True)
        layout.addWidget(self.editor, 1)

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
        self.format_combo.setEnabled(enabled)
        self.btn_start.setEnabled(enabled)
        self.btn_cancel.setEnabled(not enabled)

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
        }

        # Spin up worker and thread
        self._set_controls_enabled(False)  # noqa: FBT003
        self.progress.setValue(0)
        self.status.showMessage("Processing…")
        # Ensure previous thread is fully stopped before creating new one
        if self._thread is not None:
            try:
                self._thread.quit()
                self._thread.wait(2000)
            except Exception:  # noqa: BLE001
                pass
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
            self.btn_cancel.setEnabled(False)

    def on_progress(self, frac: float, msg: str) -> None:
        """Update progress bar and status message."""
        self.progress.setValue(int(max(0.0, min(1.0, frac)) * 100))
        if msg:
            self.status.showMessage(msg)

    def on_log(self, line: str) -> None:
        """Handle log message from worker (currently unused)."""
        # Append to editor bottom as lightweight log when not in view mode
        # Keep existing text and add a separator for logs
        # Minimal Phase 1.7: status bar + progress only

    def on_error(self, message: str) -> None:
        """Handle processing error."""
        self._set_controls_enabled(True)  # noqa: FBT003
        self.progress.setValue(0)
        QMessageBox.critical(self, "Processing error", message)
        self.status.showMessage("Error")

    def on_canceled(self) -> None:
        """Handle processing cancellation."""
        self._set_controls_enabled(True)  # noqa: FBT003
        self.progress.setValue(0)
        self.status.showMessage("Canceled")

    def on_finished(self, _outputs: list[str], view_text: str) -> None:
        """Handle processing completion."""
        self._set_controls_enabled(True)  # noqa: FBT003
        self.progress.setValue(100)
        if view_text:
            self.editor.setPlainText(view_text)
        self.status.showMessage("Done")

    def open_output_folder(self) -> None:
        """Open the output directory in the system file manager."""
        out_dir = Path(self.out_dir_edit.text()).resolve()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(out_dir)))

    def _log_wrap(self, func: Callable[..., Any], _name: str) -> Callable[..., Any]:
        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return _wrapped


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

                def cb(
                    msg: str, f: float, base: float = idx / total, prefix: str = prefix
                ) -> None:
                    # Scale progress across files
                    self._report(prefix + msg, min(0.99, base + f / total))
                    if self._cancel:
                        # Interrupt pipeline through callback
                        raise CancelledByUserError  # noqa: TRY301

                # Configure pipeline according to options
                self._pipeline.enable_diarization = bool(
                    self._opts.get("enable_diarization", False)
                )
                self._pipeline.enable_dialog_blocks = bool(
                    self._opts.get("enable_dialog_blocks", False)
                )

                try:
                    doc = self._pipeline.process(media, self._out_dir, progress=cb)
                except CancelledByUserError:
                    self.canceled.emit()
                    return
                fmt = str(self._opts.get("export_format", "txt"))
                out_path = self._out_dir / f"{media.stem}.{fmt}"
                outp = export_document(doc, fmt, out_path)
                outputs.append(str(outp))
                if bool(self._opts.get("single_view", False)):
                    view_text = doc.get_full_text()
                self._report(prefix + "Exported", min(0.99, (idx + 1) / total))
            self._report("Completed", 1.0)
            self.finished.emit(outputs, view_text)
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
