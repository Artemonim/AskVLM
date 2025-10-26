from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


@dataclass
class SpeakerStats:
    """Aggregated statistics for a single speaker."""

    name: str
    utterance_count: int
    tabs: list[str]


class SpeakerSidebar(QWidget):
    """Sidebar widget with speaker list, rename/add, and usage counts per tab."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._speakers: list[str] = ["speaker_1"]
        self._tab_usages: dict[str, Counter[str]] = defaultdict(Counter)

        root = QVBoxLayout(self)
        # Table of speakers and stats
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Name", "Utterances", "Tabs"])
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        root.addWidget(self.table)

        # Rename/add controls
        row = QHBoxLayout()
        row.addWidget(QLabel("Rename/Add:"))
        self.name_edit = QLineEdit()
        row.addWidget(self.name_edit, 1)
        self.btn_add = QPushButton("Add")
        self.btn_rename = QPushButton("Rename Selected")
        row.addWidget(self.btn_add)
        row.addWidget(self.btn_rename)
        root.addLayout(row)

        self.btn_add.clicked.connect(self._on_add)
        self.btn_rename.clicked.connect(self._on_rename)

        self._refresh()

    def set_speakers(self, speakers: list[str]) -> None:
        """Replace the known speakers list and refresh UI."""
        self._speakers = list(dict.fromkeys(speakers or ["speaker_1"]))
        self._refresh()

    def record_usage(self, tab_name: str, speaker: str) -> None:
        """Record one utterance of `speaker` under a given tab name."""
        self._tab_usages[tab_name][speaker] += 1

    def reset_usages(self) -> None:
        """Clear usage counters for all tabs."""
        self._tab_usages.clear()

    def get_speakers(self) -> list[str]:
        """Return a copy of current speakers list."""
        return list(self._speakers)

    def _on_add(self) -> None:
        name = self.name_edit.text().strip()
        if name and name not in self._speakers:
            self._speakers.append(name)
            self._refresh()
            self.name_edit.clear()

    def _on_rename(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        new_name = self.name_edit.text().strip()
        if not new_name:
            return
        itm = self.table.item(row, 0)
        old_name = itm.text() if itm is not None else ""
        if new_name == old_name:
            return
        if new_name in self._speakers:
            return
        # Rename in list
        try:
            idx = self._speakers.index(old_name)
            self._speakers[idx] = new_name
        except ValueError:
            pass
        # Rename in stats
        for counter in self._tab_usages.values():
            if old_name in counter:
                counter[new_name] = counter.pop(old_name)
        self._refresh()
        self.name_edit.clear()

    def _refresh(self) -> None:
        # Compile stats per speaker
        stats: list[SpeakerStats] = []
        for s in self._speakers:
            cnt = sum(c.get(s, 0) for c in self._tab_usages.values())
            tabs = [t for t, c in self._tab_usages.items() if c.get(s, 0) > 0]
            stats.append(SpeakerStats(name=s, utterance_count=cnt, tabs=tabs))

        self.table.setRowCount(0)
        for st in stats:
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(st.name))
            self.table.setItem(r, 1, QTableWidgetItem(str(st.utterance_count)))
            self.table.setItem(r, 2, QTableWidgetItem(", ".join(st.tabs)))
