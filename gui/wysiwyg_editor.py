from typing import Optional

from PySide6.QtWidgets import QTextEdit, QWidget


# * WYSIWYG editor for formatted transcription text
class WysiwygEditor(QTextEdit):
    """Rich text editor with future support for speaker metadata and custom blocks."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        # ! Custom block userData for speaker_id, start_ts, end_ts will be added in Phase 2
