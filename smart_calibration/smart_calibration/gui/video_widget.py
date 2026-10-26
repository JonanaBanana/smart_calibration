"""Camera image with a vector overlay of the detected chessboard and an uncertainty map."""

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

CORNER_RADIUS_PX = 4.0  # on screen, independent of zoom

# Status box (top left): text colour per line state.
STATUS_COLORS = {
    'pass': QColor(90, 220, 90),
    'fail': QColor(255, 90, 80),
    'warn': QColor(255, 200, 60),
    'unknown': QColor(150, 150, 150),
    'kept': QColor(80, 200, 255),
}
# Value column wide enough for these, so the box keeps its size from frame to frame.
STATUS_VALUE_WIDTHS = ('not found', '00000 px/s', 'kept #000')

# Uncertainty map colours (fixed scale, so "good" looks the same every time): green at
# MAP_GOOD_PX and below, yellow in between (logarithmic), red at MAP_BAD_PX and above.
MAP_GOOD_PX = 0.25
MAP_BAD_PX = 2.0
MAP_ALPHA = 110
MAP_COLORS = np.array([[0, 200, 0], [255, 220, 0], [255, 40, 40]], np.float64)


def uncertainty_colors(values):
    """(rows, cols, 4) uint8 RGBA for an uncertainty map in px; NaN cells transparent."""
    values = np.asarray(values, np.float64)
    valid = np.isfinite(values)
    position = np.zeros_like(values)
    position[valid] = np.clip(np.log(np.maximum(values[valid], 1e-9) / MAP_GOOD_PX)
                              / np.log(MAP_BAD_PX / MAP_GOOD_PX), 0.0, 1.0)
    scaled = position * (len(MAP_COLORS) - 1)
    low = np.minimum(scaled.astype(int), len(MAP_COLORS) - 2)
    fraction = (scaled - low)[..., None]
    rgba = np.zeros(values.shape + (4,), np.uint8)
    rgba[..., :3] = np.round(MAP_COLORS[low] * (1 - fraction) + MAP_COLORS[low + 1] * fraction)
    rgba[..., 3] = np.where(valid, MAP_ALPHA, 0)
    return rgba


def to_qimage(image):
    """Wrap a uint8 (h, w) gray or (h, w, 3) RGB array in a QImage without copying it."""
    height, width = image.shape[:2]
    image_format = QImage.Format_Grayscale8 if image.ndim == 2 else QImage.Format_RGB888
    return QImage(image.data, width, height, image.strides[0], image_format)


def cosmetic_pen(color, width):
    """Pen whose width is in screen pixels, regardless of the painter's scale."""
    pen = QPen(color, width)
    pen.setCosmetic(True)
    return pen


class VideoWidget(QWidget):
    """Shows an image scaled to fit (aspect ratio kept) with the board drawn on top."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._image = None  # numpy array backing _qimage; must stay alive while it is shown
        self._qimage = None
        self._corners = None  # (rows, cols, 2) corner positions, or None
        self._corners_width = None  # width of the image the corner coordinates refer to
        self._message = 'waiting for images'
        self._status = None
        self._origin = None  # (row, col) of the board origin in the corner grid, or None
        self._map = None  # RGBA array backing _map_image
        self._map_image = None

    def set_frame(self, image, corners=None, message=None, corners_width=None, status=None,
                  origin=None):
        """
        Show image, optionally with a board overlay and a status box.

        corners are in the pixel coordinates of an image corners_width pixels wide (default: the
        shown image), so they can refer to the full-resolution camera image while a downscaled
        preview of it is shown. status: lines (label, value, mark, state) for the status box in
        the top left corner, state being a key of STATUS_COLORS. origin: (row, col) in corners
        of the board origin, marked with a ring.
        """
        self._image = np.ascontiguousarray(image)
        self._qimage = to_qimage(self._image)
        self._corners = corners
        self._corners_width = corners_width or self._image.shape[1]
        self._message = message
        self._status = status
        self._origin = origin
        self.update()

    def set_message(self, message):
        """Show a message instead of the status box (over the last image)."""
        self._message = message
        self._status = None
        self.update()

    def set_uncertainty_map(self, values):
        """
        Show an uncertainty map (rows, cols) in px over the image, or none (None).

        Its cells divide the image evenly, so it is stretched over the shown image whatever its
        resolution.
        """
        if values is None:
            self._map, self._map_image = None, None
        else:
            self._map = np.ascontiguousarray(uncertainty_colors(values))
            rows, cols = self._map.shape[:2]
            self._map_image = QImage(self._map.data, cols, rows, self._map.strides[0],
                                     QImage.Format_RGBA8888)
        self.update()

    def image_rect(self):
        """Widget area the image is drawn into."""
        scale = min(self.width() / self._qimage.width(), self.height() / self._qimage.height())
        width, height = self._qimage.width() * scale, self._qimage.height() * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        if self._qimage is not None:
            target = self.image_rect()
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            painter.drawImage(target, self._qimage)
            if self._map_image is not None:
                # Smooth scaling blends the cells into a continuous map.
                painter.drawImage(target, self._map_image)
                self._draw_map_legend(painter, target)
            if self._corners is not None:
                self._draw_board(painter, target)
        top = 10.0
        if self._status:
            top = self._draw_status(painter, top) + 6.0
        if self._message:
            self._draw_message(painter, top)
        painter.end()

    def _draw_board(self, painter, target):
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        scale = target.width() / self._corners_width
        painter.translate(target.topLeft())
        painter.scale(scale, scale)
        # OpenCV puts pixel centres at integer coordinates, Qt at +0.5.
        painter.translate(0.5, 0.5)

        # Corners not used by the calibration node are NaN: they are left out (gaps in the rows).
        rows, cols = self._corners.shape[:2]
        outline = np.array([self._corners[0, 0], self._corners[0, -1],
                            self._corners[-1, -1], self._corners[-1, 0]])
        painter.setBrush(Qt.NoBrush)
        if np.isfinite(outline).all():
            painter.setPen(cosmetic_pen(QColor(0, 255, 0), 2))
            painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in outline]))

        radius = CORNER_RADIUS_PX / scale
        for row in range(rows):
            # One colour per row (like cv2.drawChessboardCorners), so the orientation is visible.
            color = QColor.fromHsvF(0.8 * row / max(rows - 1, 1), 1.0, 1.0)
            painter.setPen(cosmetic_pen(color, 2))
            line = []
            for x, y in self._corners[row]:
                if not (np.isfinite(x) and np.isfinite(y)):
                    if len(line) > 1:
                        painter.drawPolyline(QPolygonF(line))
                    line = []
                    continue
                point = QPointF(x, y)
                line.append(point)
                painter.drawEllipse(point, radius, radius)
            if len(line) > 1:
                painter.drawPolyline(QPolygonF(line))
        if self._origin is not None:
            row, col = self._origin
            if row < rows and col < cols and np.isfinite(self._corners[row, col]).all():
                x, y = self._corners[row, col]
                painter.setPen(cosmetic_pen(QColor(255, 255, 255), 3))
                painter.drawEllipse(QPointF(x, y), 2.5 * radius, 2.5 * radius)
        painter.restore()

    def _draw_map_legend(self, painter, target):
        """Colour scale of the uncertainty map in the bottom right corner of the image."""
        painter.save()
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        low, high = f'{MAP_GOOD_PX:g} px', f'{MAP_BAD_PX:g} px'
        bar_width, padding = 120.0, 6.0
        width = (metrics.horizontalAdvance(low) + metrics.horizontalAdvance(high) + bar_width
                 + 4 * padding)
        height = metrics.height() + 2 * padding
        box = QRectF(target.right() - width - 10, target.bottom() - height - 10, width, height)
        painter.fillRect(box, QColor(0, 0, 0, 170))
        painter.setPen(QColor(220, 220, 220))
        x = box.left() + padding
        painter.drawText(QRectF(x, box.top(), metrics.horizontalAdvance(low), height),
                         Qt.AlignVCenter, low)
        x += metrics.horizontalAdvance(low) + padding
        steps = 24
        colors = uncertainty_colors(MAP_GOOD_PX * (MAP_BAD_PX / MAP_GOOD_PX) ** (
            np.arange(steps) / (steps - 1)))
        for i, (r, g, b, _) in enumerate(colors):
            painter.fillRect(QRectF(x + i * bar_width / steps, box.top() + padding,
                                    bar_width / steps + 0.5, height - 2 * padding),
                             QColor(int(r), int(g), int(b)))
        x += bar_width + padding
        painter.drawText(QRectF(x, box.top(), metrics.horizontalAdvance(high), height),
                         Qt.AlignVCenter, high)
        painter.restore()

    def _draw_status(self, painter, top):
        """Draw the status box at (10, top), in screen pixels; return its bottom edge."""
        painter.save()
        font = painter.font()
        font.setPointSize(11)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        padding, gap, line_height = 8.0, 12.0, metrics.height() + 2.0
        label_width = max(metrics.horizontalAdvance(label) for label, _, _, _ in self._status)
        value_width = max(metrics.horizontalAdvance(text) for text in
                          STATUS_VALUE_WIDTHS + tuple(value for _, value, _, _ in self._status))
        mark_width = metrics.horizontalAdvance('✓')
        box = QRectF(10.0, top, 2 * padding + label_width + value_width + mark_width + 2 * gap,
                     2 * padding + len(self._status) * line_height)
        painter.fillRect(box, QColor(0, 0, 0, 170))

        y = box.top() + padding
        value_x = box.left() + padding + label_width + gap
        mark_x = value_x + value_width + gap
        for label, value, mark, state in self._status:
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(QRectF(box.left() + padding, y, label_width, line_height),
                             Qt.AlignLeft | Qt.AlignVCenter, label)
            painter.setPen(STATUS_COLORS[state])
            painter.drawText(QRectF(value_x, y, value_width, line_height),
                             Qt.AlignLeft | Qt.AlignVCenter, value)
            painter.drawText(QRectF(mark_x, y, mark_width, line_height),
                             Qt.AlignLeft | Qt.AlignVCenter, mark)
            y += line_height
        painter.restore()
        return box.bottom()

    def _draw_message(self, painter, top):
        painter.save()
        font = painter.font()
        font.setPointSize(12)
        painter.setFont(font)
        rect = painter.fontMetrics().boundingRect(self._message).adjusted(-8, -4, 8, 4)
        rect.moveTopLeft(QPointF(10, top).toPoint())
        painter.fillRect(rect, QColor(0, 0, 0, 170))
        painter.setPen(QColor(255, 210, 80))
        painter.drawText(rect, Qt.AlignCenter, self._message)
        painter.restore()
