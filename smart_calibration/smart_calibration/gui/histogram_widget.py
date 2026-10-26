"""Small plot of the tone mapping: histograms before / after the contrast curve, and the curve."""

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from smart_calibration.preprocessing import s_curve

BEFORE_COLOR = QColor(150, 150, 150, 170)
AFTER_COLOR = QColor(255, 150, 40)
CURVE_COLOR = QColor(60, 210, 255)


class HistogramWidget(QWidget):
    """
    Tone mapping plot for the side panel.

    Histogram of the detection image before (grey, filled) and after (orange) the S-shaped
    contrast curve, with the curve itself (cyan; input along x, output along y) and its midpoint.
    Bar heights use a square-root scale, so small peaks stay visible next to large ones.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(110)
        self.setMinimumWidth(150)
        self.setToolTip('Put the midpoint between the dark and light squares\' peaks.')
        self._before = None
        self._after = None
        self._strength = 0.0
        self._midpoint = 0.5

    def set_histograms(self, before, after):
        self._before = np.sqrt(np.asarray(before, dtype=np.float64))
        self._after = np.sqrt(np.asarray(after, dtype=np.float64))
        self.update()

    def set_curve(self, strength, midpoint):
        if (strength, midpoint) != (self._strength, self._midpoint):
            self._strength, self._midpoint = strength, midpoint
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        plot = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        painter.fillRect(plot, QColor(25, 25, 25))

        if self._before is not None:
            peak = max(self._before.max(), self._after.max(), 1.0)
            before = self._points(plot, self._before / peak)
            painter.setPen(Qt.NoPen)
            painter.setBrush(BEFORE_COLOR)
            painter.drawPolygon(QPolygonF(
                [QPointF(plot.left(), plot.bottom())] + before
                + [QPointF(plot.right(), plot.bottom())]))
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(AFTER_COLOR, 1.2))
            painter.drawPolyline(QPolygonF(self._points(plot, self._after / peak)))

        x = np.linspace(0.0, 1.0, 101)
        curve = s_curve(x, self._strength, self._midpoint)
        painter.setPen(QPen(CURVE_COLOR, 1.5))
        painter.drawPolyline(QPolygonF([
            QPointF(plot.left() + u * plot.width(), plot.bottom() - v * plot.height())
            for u, v in zip(x, curve)]))
        if self._strength > 0.0:
            dashed = QPen(CURVE_COLOR, 1, Qt.DashLine)
            painter.setPen(dashed)
            mid_x = plot.left() + self._midpoint * plot.width()
            painter.drawLine(QPointF(mid_x, plot.top()), QPointF(mid_x, plot.bottom()))

        painter.setPen(QPen(QColor(90, 90, 90), 1))
        painter.drawRect(plot)
        painter.end()

    @staticmethod
    def _points(plot, heights):
        """One point per bin (0-1 heights), centred on the bin."""
        n = len(heights)
        return [QPointF(plot.left() + (i + 0.5) / n * plot.width(),
                        plot.bottom() - h * plot.height())
                for i, h in enumerate(heights)]
