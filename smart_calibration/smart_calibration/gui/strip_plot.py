"""Strip plot of one value per view, with a limit line; clicking a dot selects it."""

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from smart_calibration.gui.video_widget import SELECTED_COLOR, VIEW_STYLES

DOT_RADIUS = 3.5
MARGIN = 8.0
LIMIT_COLOR = QColor(255, 200, 60)


class StripPlot(QWidget):
    """Dots along a horizontal axis from 0, coloured by style; a vertical line at the limit."""

    selected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(64)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._values = np.zeros(0)
        self._styles = []
        self._limit = None
        self._selected = None
        self._unit = ''

    def set_data(self, values, styles, limit, selected, unit='px'):
        """values: one per dot; styles: a VIEW_STYLES key per dot; selected: index or None."""
        self._values = np.asarray(values, np.float64)
        self._styles, self._limit, self._selected, self._unit = list(styles), limit, selected, unit
        self.update()

    def _x_max(self):
        finite = self._values[np.isfinite(self._values)]
        candidates = list(finite) + ([self._limit] if self._limit is not None else [])
        return max(max(candidates, default=1.0) * 1.1, 1e-6)

    def _positions(self):
        """Dot centres in widget pixels, spread vertically so close values stay apart."""
        plot = QRectF(MARGIN, MARGIN, self.width() - 2 * MARGIN, self.height() - 2 * MARGIN - 14)
        x_max = self._x_max()
        rows = 5
        return [QPointF(plot.left() + plot.width() * min(value, x_max) / x_max,
                        plot.top() + plot.height() * ((i * 3) % rows + 0.5) / rows)
                for i, value in enumerate(self._values)], plot

    def mousePressEvent(self, event):
        positions, _ = self._positions()
        if not positions:
            return
        where = event.position()
        distances = [np.hypot(p.x() - where.x(), p.y() - where.y()) for p in positions]
        nearest = int(np.argmin(distances))
        if distances[nearest] <= 3 * DOT_RADIUS:
            self.selected.emit(nearest)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        positions, plot = self._positions()
        x_max = self._x_max()
        painter.setPen(QColor(120, 120, 120))
        painter.drawLine(QPointF(plot.left(), plot.bottom()), QPointF(plot.right(), plot.bottom()))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QColor(180, 180, 180))
        painter.drawText(QRectF(plot.left(), plot.bottom() + 1, 60, 14), Qt.AlignLeft, '0')
        painter.drawText(QRectF(plot.right() - 80, plot.bottom() + 1, 80, 14), Qt.AlignRight,
                         f'{x_max:.2f} {self._unit}')
        if self._limit is not None and np.isfinite(self._limit):
            x = plot.left() + plot.width() * min(self._limit, x_max) / x_max
            pen = QPen(LIMIT_COLOR, 1.5, Qt.DashLine)
            painter.setPen(pen)
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
        order = [i for i in range(len(positions)) if i != self._selected] + (
            [self._selected] if self._selected is not None and self._selected < len(positions)
            else [])
        for i in order:
            color = QColor(VIEW_STYLES[self._styles[i]][0])
            color.setAlpha(255)
            painter.setPen(QPen(SELECTED_COLOR, 2) if i == self._selected else Qt.NoPen)
            painter.setBrush(color)
            radius = DOT_RADIUS + (1.5 if i == self._selected else 0)
            painter.drawEllipse(positions[i], radius, radius)
        painter.end()
