from pathlib import Path
import threading
import sys
from typing import Optional

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QTextEdit,
    QStatusBar,
    QMenu,
    QFileDialog,
)
from PySide6.QtGui import QAction

from core.pipelines import LocalPipeline


# * Main window for Artemonim's Speech Kit GUI
class MainWindow(QMainWindow):
    """Application main window with file menu, processing controls, and text editor."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Artemonim's Speech Kit")
        self.resize(800, 600)

        # * Menu bar setup
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")

        open_action = QAction("Open...", self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # * Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # * Process button
        self.process_button = QPushButton("Process")
        self.process_button.clicked.connect(self.run_pipeline)
        layout.addWidget(self.process_button)

        # * Text editor for displaying results
        self.editor = QTextEdit()
        layout.addWidget(self.editor)

        # * Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready")

        # * Pipeline instance
        self.pipeline = LocalPipeline()
        self.input_path: Optional[Path] = None

    def open_file(self) -> None:
        """Open file dialog to select media file."""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Media File",
            filter="Media Files (*.wav *.mp3 *.mp4 *.avi *.mkv)",
        )
        if file_name:
            self.input_path = Path(file_name)
            self.status.showMessage(f"Selected: {file_name}")

    def run_pipeline(self) -> None:
        """Run processing pipeline in a separate thread to avoid blocking UI."""
        if not self.input_path:
            self.status.showMessage("No file selected.")
            return

        def task() -> None:
            self.status.showMessage("Processing...")
            try:
                if self.input_path is not None:  # * Type narrowing for mypy
                    document = self.pipeline.process(self.input_path)
                    text = document.get_full_text()
                    self.editor.setPlainText(text)
                    self.status.showMessage("Processing complete.")
                else:
                    self.status.showMessage("No file selected.")
            except Exception as e:
                self.status.showMessage(f"Error: {e}")

        thread = threading.Thread(target=task, daemon=True)
        thread.start()


# * Entry point for GUI application


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
