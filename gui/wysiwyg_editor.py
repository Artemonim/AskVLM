from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from PySide6.QtWidgets import QWidget


# * WYSIWYG editor for formatted transcription text
@dataclass
class TableRow:
    """Represents one row in the transcript table."""

    start: float
    end: float
    speaker_id: str
    text: str


class WysiwygEditor(QTableWidget):
    """Table-based transcript view with columns: time, speaker, text.

    Provides a combobox per speaker cell to choose an existing or custom speaker.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the table editor with three columns and editing enabled."""
        super().__init__(0, 3, parent)
        self.setHorizontalHeaderLabels(["Time", "Speaker", "Text"])
        # Use EditTrigger/SelectionBehavior enums
        self.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.AnyKeyPressed
        )
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._speakers: list[str] = ["speaker_1"]
        self._on_speaker_changed: Callable[[int, str], None] | None = None
        self.setPlaceholderText = lambda *_: None

    def set_speakers(self, speakers: list[str]) -> None:
        """Set available speakers for comboboxes and refresh cells."""
        self._speakers = speakers or ["speaker_1"]
        # Refresh all speaker cells to update combobox items
        for row in range(self.rowCount()):
            self._setup_speaker_cell(row)

    def set_rows(self, rows: list[TableRow]) -> None:
        """Replace all table rows with provided rows."""
        self.setRowCount(0)
        for r in rows:
            self._append_row(r)

    def set_rows_from_tuples(self, rows: list[tuple[float, float, str, str]]) -> None:
        """Replace rows using raw tuples: (start, end, speaker_id, text)."""
        self.setRowCount(0)
        for s, e, sp, tx in rows:
            self._append_row(TableRow(start=s, end=e, speaker_id=sp, text=tx))

    def _append_row(self, row: TableRow) -> None:
        """Append one `TableRow` to the table and build widgets for its cells."""
        idx = self.rowCount()
        self.insertRow(idx)
        # Time cell
        time_str = self._format_time(row.start, row.end)
        self.setItem(idx, 0, QTableWidgetItem(time_str))
        # Speaker cell as combobox
        speaker_combo = QComboBox()
        # Use keyword to avoid boolean-positional lint complaint
        speaker_combo.setEditable(editable=True)  # type: ignore[call-arg]
        speaker_combo.addItems(self._speakers)
        current = (
            row.speaker_id if row.speaker_id in self._speakers else self._speakers[0]
        )
        speaker_combo.setCurrentText(current)
        speaker_combo.currentTextChanged.connect(
            lambda val, i=idx: self._emit_speaker_changed(i, val)
        )
        self.setCellWidget(idx, 1, speaker_combo)
        # Text cell
        text_item = QTableWidgetItem(row.text)
        # Prefer modern enum API for editability flag
        try:
            text_item.setFlags(text_item.flags() | Qt.ItemFlag.ItemIsEditable)
        except AttributeError:
            text_item.setFlags(text_item.flags())
        self.setItem(idx, 2, text_item)

    def _setup_speaker_cell(self, row: int) -> None:
        """Recreate combobox options for the speaker cell in the given row."""
        widget = self.cellWidget(row, 1)
        if isinstance(widget, QComboBox):
            cur = widget.currentText()
            should_block = True
            widget.blockSignals(should_block)
            widget.clear()
            widget.addItems(self._speakers)
            widget.setCurrentText(cur if cur in self._speakers else self._speakers[0])
            should_unblock = False
            widget.blockSignals(should_unblock)

    def _format_time(self, start: float, end: float) -> str:
        """Format start/end in hh:mm:ss.mmm → hh:mm:ss.mmm."""

        def fmt(t: float) -> str:
            s = int(t)
            ms = round((t - s) * 1000)
            m, s = divmod(s, 60)
            h, m = divmod(m, 60)
            return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

        return f"{fmt(start)} → {fmt(end if end > 0 else start)}"

    def on_speaker_changed(self, cb: Callable[[int, str], None]) -> None:
        """Register handler for when a row's speaker selection changes."""
        self._on_speaker_changed = cb

    def _emit_speaker_changed(self, row: int, value: str) -> None:
        """Emit a callback when user changes a speaker in a row."""
        if self._on_speaker_changed is not None:
            self._on_speaker_changed(row, value)
