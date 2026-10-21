"""Camera image with a vector overlay of the detected chessboard."""

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

CORNER_RADIUS_PX = 4.0  # on screen, independent of zoom


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

    def set_frame(self, image, corners=None, message=None, corners_width=None):
        """
        Show image, optionally with a board overlay.

        corners are in the pixel coordinates of an image corners_width pixels wide (default: the
        shown image), so they can refer to the full-resolution camera image while a downscaled
        preview of it is shown.
        """
        self._image = np.ascontiguousarray(image)
        self._qimage = to_qimage(self._image)
        self._corners = corners
        self._corners_width = corners_width or self._image.shape[1]
        self._message = message
        self.update()

    def set_message(self, message):
        self._message = message
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
            if self._corners is not None:
                self._draw_board(painter, target)
        if self._message:
            self._draw_message(painter)
        painter.end()

    def _draw_board(self, painter, target):
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        scale = target.width() / self._corners_width
        painter.translate(target.topLeft())
        painter.scale(scale, scale)
        # OpenCV puts pixel centres at integer coordinates, Qt at +0.5.
        painter.translate(0.5, 0.5)

        rows, cols = self._corners.shape[:2]
        outline = [self._corners[0, 0], self._corners[0, -1],
                   self._corners[-1, -1], self._corners[-1, 0]]
        painter.setPen(cosmetic_pen(QColor(0, 255, 0), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in outline]))

        radius = CORNER_RADIUS_PX / scale
        for row in range(rows):
            # One colour per row (like cv2.drawChessboardCorners), so the orientation is visible.
            color = QColor.fromHsvF(0.8 * row / max(rows - 1, 1), 1.0, 1.0)
            painter.setPen(cosmetic_pen(color, 2))
            points = [QPointF(x, y) for x, y in self._corners[row]]
            painter.drawPolyline(QPolygonF(points))
            for point in points:
                painter.drawEllipse(point, radius, radius)
        painter.restore()

    def _draw_message(self, painter):
        painter.save()
        font = painter.font()
        font.setPointSize(12)
        painter.setFont(font)
        rect = painter.fontMetrics().boundingRect(self._message).adjusted(-8, -4, 8, 4)
        rect.moveTopLeft(self.rect().topLeft() + QPointF(10, 10).toPoint())
        painter.fillRect(rect, QColor(0, 0, 0, 170))
        painter.setPen(QColor(255, 210, 80))
        painter.drawText(rect, Qt.AlignCenter, self._message)
        painter.restore()
