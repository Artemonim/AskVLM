from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QPointF, QSize, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPixmap,
)
from PySide6.QtWidgets import QWidget

if TYPE_CHECKING:
    from collections.abc import Iterable

    from PySide6.QtGui import (
        QImage,
        QPaintEvent,
    )


class SubtitlePreview(QWidget):
    """Lightweight widget that draws a video frame with an overlaid subtitle.

    The widget accepts a background image (video frame) and a list of text lines
    that it renders near the bottom of the frame using a large, legible font
    with a black outline for readability.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._bg: QPixmap | None = None
        self._text_lines: list[str] = []
        self._font_family: str | None = None
        self._font_px_override: int | None = None
        # * Bottom padding as fraction of widget height
        self._bottom_padding_ratio = 0.08
        # * Inter-line spacing multiplier
        self._line_spacing = 1.25
        # * Minimum sensible font size
        self._min_font_px = 14
        # * Maximum sensible font size
        self._max_font_px = 48
        self.setMinimumSize(QSize(320, 180))

    def sizeHint(self) -> QSize:  # noqa: N802
        """Return a 16:9-friendly default size."""
        return QSize(640, 360)

    def clear(self) -> None:
        """Clear background and text."""
        self._bg = None
        self._text_lines = []
        self.update()

    def set_background_image(self, image: QImage | QPixmap | None) -> None:
        """Set background image from QImage/QPixmap; None clears the background."""
        if image is None:
            self._bg = None
        elif isinstance(image, QPixmap):
            self._bg = image
        else:
            self._bg = QPixmap.fromImage(image)
        self.update()

    def set_text_lines(self, lines: Iterable[str]) -> None:
        """Set subtitle text as multiple lines."""
        self._text_lines = [x for x in lines if x is not None]
        self.update()

    def set_font_family(self, family: str | None) -> None:
        """Prefer a specific font family for rendering (optional)."""
        self._font_family = family
        self.update()

    def set_font_size_override(self, pixels: int | None) -> None:
        """Set an absolute font size in pixels; None restores auto-scaling.

        The override ensures the preview matches forced style used for burn-in.
        """
        self._font_px_override = pixels if (pixels or 0) > 0 else None
        self.update()

    @staticmethod
    def layout_text(text: str, max_line_chars: int, max_lines: int) -> list[str]:
        """Split text into lines respecting character and line limits.

        Words do not break across lines; returns at most `max_lines` lines.
        """
        words = [w for w in (text or "").split() if w]
        if not words:
            return []
        lines: list[str] = []
        cur: list[str] = []
        cur_len = 0
        for w in words:
            wlen = len(w)
            if cur and (cur_len + 1 + wlen) > max_line_chars:
                lines.append(" ".join(cur))
                if len(lines) >= max_lines:
                    return lines[:max_lines]
                cur = [w]
                cur_len = wlen
            else:
                if cur:
                    cur_len += 1 + wlen
                else:
                    cur_len = wlen
                cur.append(w)
        if cur:
            lines.append(" ".join(cur))
        return lines[:max_lines]

    def _draw_centered_text_block(
        self, painter: QPainter, width: int, height: int
    ) -> None:
        """Draw `_text_lines` horizontally centered near the bottom of the widget."""
        if not self._text_lines:
            return

        # Compute font size: respect override; otherwise auto-scale similarly to burn
        if self._font_px_override is not None:
            font_px = int(max(10, min(96, self._font_px_override)))
        else:
            # Approximate ffmpeg burn autoscale (~3% of height, clamped)
            font_px = max(20, min(38, int(height * 0.03)))
        font = QFont(self._font_family or "Open Sans")
        font.setPixelSize(font_px)
        painter.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing
        )
        painter.setFont(font)

        metrics = painter.fontMetrics()
        line_h = int(metrics.height() * self._line_spacing)
        block_h = int(len(self._text_lines) * line_h)
        y = int(height * (1.0 - self._bottom_padding_ratio)) - block_h

        # Outline then fill for readability
        outline_color = QColor(0, 0, 0)
        fill_color = QColor(255, 255, 255)
        # Match libass default Outline=2 (constant px)
        stroke_width = 2.0
        for i, line in enumerate(self._text_lines):
            path = QPainterPath()
            w = metrics.horizontalAdvance(line)
            x = (width - w) // 2
            path.addText(QPointF(float(x), float(y + (i + 1) * line_h)), font, line)
            stroker = QPainterPathStroker()
            stroker.setWidth(stroke_width)
            outline = stroker.createStroke(path)
            painter.fillPath(outline, outline_color)
            painter.fillPath(path, fill_color)

    def paintEvent(self, _event: QPaintEvent) -> None:  # noqa: N802
        """Render background (scaled) and text overlay."""
        painter = QPainter(self)
        try:
            # Fill black background
            painter.fillRect(self.rect(), QColor(0, 0, 0))
            if self._bg is not None and not self._bg.isNull():
                # Scale pixmap to fit while preserving aspect ratio
                pm = self._bg.scaled(
                    self.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                # Center the image
                x = (self.width() - pm.width()) // 2
                y = (self.height() - pm.height()) // 2
                painter.drawPixmap(x, y, pm)
            # Draw text over everything
            self._draw_centered_text_block(painter, self.width(), self.height())
        finally:
            painter.end()
