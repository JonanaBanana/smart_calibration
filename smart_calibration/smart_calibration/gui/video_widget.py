"""
Camera image with vector overlays.

The detected chessboard, an uncertainty map, and for the final calibration pass the outlines of
the kept views and the corner errors of one view.
"""

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
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

# Final pass view outlines: (colour, width, dashed) per style.
# Kept views are many and only context: faint. The flagged ones should stand out.
VIEW_STYLES = {
    'kept': (QColor(90, 220, 90, 70), 1.5, False),
    'removed': (QColor(255, 80, 70, 200), 2.5, True),
    'suspect': (QColor(255, 200, 60, 200), 2.5, False),
    'restored': (QColor(80, 200, 255, 160), 2.0, False),
}
SELECTED_COLOR = QColor(255, 255, 255)
# Undistorted view: where the lens model folds over (beyond it nothing valid can be shown).
BOUNDARY_COLOR = QColor(255, 200, 60, 200)
# Corner errors: a line of fixed length (screen pixels) in the error's direction, coloured by its
# size and labelled with it (largest errors first, as many labels as fit without overlapping).
ERROR_LINE_PX = 14.0
DIM_ALPHA = 150


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

    # A click on the image, in the coordinates of the view outlines (see set_views).
    image_clicked = Signal(float, float)

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
        self._dimmed = False
        self._views = None  # (outlines, styles, selected, width), see set_views
        self._arrows = None  # (points, residuals, width), see set_arrows
        self._error_labels_shown = True
        self._boundary = None  # (polygon, width), see set_boundary

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

    def set_dimmed(self, dimmed):
        """Darken the image (as a background for the final pass's view outlines)."""
        self._dimmed = dimmed
        self.update()

    def set_views(self, outlines, styles, selected, width):
        """
        Draw view outlines: (k, 2) polygons in the pixels of an image width wide (or None).

        styles: a VIEW_STYLES key per outline; selected: index of the outline drawn highlighted,
        or None.
        """
        self._views = None if outlines is None else (outlines, styles, selected, width)
        self.update()

    def set_arrows(self, points, residuals, width):
        """
        Draw corner errors at points (N, 2): residuals (N, 2) as direction, colour and label.

        Coordinates in the pixels of an image width wide; points None: none.
        """
        self._arrows = None if points is None else (points, residuals, width)
        self.update()

    def set_boundary(self, polygon, width):
        """
        Draw where the lens model is valid: a closed dashed line, or none (polygon None).

        polygon (k, 2) in the pixels of an image width wide.
        """
        self._boundary = None if polygon is None else (polygon, width)
        self.update()

    def set_error_labels(self, shown):
        """Label the corner errors (see set_arrows) with their size, or not."""
        self._error_labels_shown = shown
        self.update()

    def mousePressEvent(self, event):
        if self._qimage is None or self._views is None or event.button() != Qt.LeftButton:
            return
        target = self.image_rect()
        scale = target.width() / self._views[3]
        position = event.position()
        self.image_clicked.emit((position.x() - target.left()) / scale - 0.5,
                                (position.y() - target.top()) / scale - 0.5)

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
            if self._dimmed:
                painter.fillRect(target, QColor(0, 0, 0, DIM_ALPHA))
            if self._map_image is not None:
                # Smooth scaling blends the cells into a continuous map.
                painter.drawImage(target, self._map_image)
                self._draw_map_legend(painter, target)
            if self._boundary is not None:
                self._draw_boundary(painter, target)
            if self._corners is not None:
                self._draw_board(painter, target)
            if self._views is not None:
                self._draw_views(painter, target)
            if self._arrows is not None:
                self._draw_arrows(painter, target)
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

    def _image_painter(self, painter, target, width):
        """Set painter up for drawing in the pixels of an image width wide (call save first)."""
        painter.setRenderHint(QPainter.Antialiasing)
        scale = target.width() / width
        painter.translate(target.topLeft())
        painter.scale(scale, scale)
        painter.translate(0.5, 0.5)  # OpenCV puts pixel centres at integer coordinates, Qt at +0.5
        return scale

    def _draw_boundary(self, painter, target):
        polygon, width = self._boundary
        painter.save()
        self._image_painter(painter, target, width)
        pen = cosmetic_pen(BOUNDARY_COLOR, 1.5)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in polygon]))
        painter.restore()

    def _draw_views(self, painter, target):
        outlines, styles, selected, width = self._views
        painter.save()
        self._image_painter(painter, target, width)
        painter.setBrush(Qt.NoBrush)
        order = [i for i in range(len(outlines)) if i != selected] + (
            [selected] if selected is not None else [])
        for i in order:  # the selected outline on top
            color, line_width, dashed = VIEW_STYLES[styles[i]]
            if i == selected:
                color, line_width = SELECTED_COLOR, line_width + 1.5
            pen = cosmetic_pen(color, line_width)
            if dashed:
                pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in outlines[i]]))
        painter.restore()

    def _draw_arrows(self, painter, target):
        """Corner errors, in screen pixels: fixed-length lines, colour and label by size."""
        points, residuals, width = self._arrows
        scale = target.width() / width
        # Pixel centres at integers in OpenCV, at +0.5 in Qt.
        screen = (np.asarray(points, np.float64) + 0.5) * scale + [target.left(), target.top()]
        sizes = np.hypot(residuals[:, 0], residuals[:, 1])
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        labels = (self._error_labels(screen, residuals, sizes, metrics)
                  if self._error_labels_shown else {})
        # Coloured like the map: green below MAP_GOOD_PX, red from MAP_BAD_PX.
        colors = uncertainty_colors(sizes)
        for i, ((x, y), (dx, dy), size) in enumerate(zip(screen, residuals, sizes)):
            r, g, b, _ = colors[i]
            color = QColor(int(r), int(g), int(b))
            painter.setPen(QPen(color, 2))
            painter.setBrush(color)
            painter.drawEllipse(QPointF(x, y), 2.5, 2.5)
            if size > 0:
                painter.drawLine(QPointF(x, y), QPointF(x + dx / size * ERROR_LINE_PX,
                                                        y + dy / size * ERROR_LINE_PX))
            if i in labels:
                painter.fillRect(labels[i], QColor(0, 0, 0, 150))
                painter.setPen(color)
                painter.drawText(labels[i], Qt.AlignCenter, f'{size:.2f}')
        painter.restore()

    @staticmethod
    def _error_labels(screen, residuals, sizes, metrics):
        """
        {corner index: label box} in screen pixels: largest errors first, none overlapping.

        Each label sits opposite its line, so the two do not overlap.
        """
        placed = {}
        for i in np.argsort(sizes)[::-1]:
            (x, y), (dx, dy), size = screen[i], residuals[i], sizes[i]
            away = (-dx / size, -dy / size) if size > 0 else (1.0, 0.0)
            box = QRectF(0, 0, metrics.horizontalAdvance(f'{size:.2f}') + 4, metrics.height())
            box.moveCenter(QPointF(x + away[0] * (box.width() / 2 + 3),
                                   y + away[1] * (box.height() / 2 + 3)))
            if not any(box.intersects(other) for other in placed.values()):
                placed[int(i)] = box
        return placed

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
