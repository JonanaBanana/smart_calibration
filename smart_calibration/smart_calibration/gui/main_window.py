"""Main window of the calibration GUI: video panel, side panel and status bar."""

import math
import time

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QSlider, QVBoxLayout,
    QWidget)

from smart_calibration.gui.video_widget import VideoWidget

REFRESH_INTERVAL_MS = 33
# Without a detection for this long, the calibration node is considered gone.
DETECTION_TIMEOUT_S = 1.0

# Gamma slider: logarithmic, so 1.0 is in the middle and 1/GAMMA_LIMIT .. GAMMA_LIMIT at the ends
# (matches the calibration node's parameter range 0.2 - 5.0).
GAMMA_LIMIT = 5.0
GAMMA_STEPS = 100
# Slider moves within this time are sent to the calibration node as one parameter update.
SEND_DELAY_MS = 100
# After sending, wait this long for the node to confirm before showing its value again.
CONFIRM_TIMEOUT_S = 1.0


# Detector flag parameters of the calibration node: (name, checkbox label, tooltip).
DETECTOR_FLAGS = (
    ('detector_normalize', 'normalize',
     'Histogram equalization before detection; can help on dark / low-contrast images.'),
    ('detector_exhaustive', 'exhaustive search',
     'Searches longer for the board: higher detection rate, slower (mostly on frames '
     'without a board).'),
    ('detector_accuracy', 'accuracy (~4x slower)',
     'Upsamples the image 2x inside the detector: more accurate corners, ~4x detection time.'),
)
REMOTE_PARAMETERS = ('gamma',) + tuple(name for name, _, _ in DETECTOR_FLAGS)


def gamma_to_slider(gamma):
    gamma = min(max(gamma, 1.0 / GAMMA_LIMIT), GAMMA_LIMIT)
    return round(GAMMA_STEPS * math.log(gamma, GAMMA_LIMIT))


def slider_to_gamma(position):
    return GAMMA_LIMIT ** (position / GAMMA_STEPS)


def detection_corners(detection):
    """Corner array (rows, cols, 2) of a BoardDetection, or None if no board was found."""
    if not detection.found:
        return None
    return np.asarray(detection.corners, dtype=np.float32).reshape(
        detection.rows, detection.cols, 2)


def form_box(title, rows):
    box = QGroupBox(title)
    form = QFormLayout(box)
    for label, widget in rows:
        form.addRow(label, widget)
    return box


class MainWindow(QMainWindow):

    def __init__(self, node):
        super().__init__()
        self.node = node
        self.setWindowTitle('smart_calibration')

        self.video = VideoWidget()
        # What is displayed, compared by identity (frames hold numpy arrays).
        self._shown_frame = None
        self._shown_detection = None
        self._shown_message = None
        self._pending = {}  # parameter name -> (value sent, time sent), until the node confirms

        # View
        self.raw_toggle = QCheckBox('raw camera (full resolution)')
        self.raw_toggle.setToolTip(
            'Show the camera stream instead of the calibration node\'s preprocessed preview. '
            'Needs the full camera stream: best on the same machine as the camera.')
        self.raw_toggle.toggled.connect(self.node.set_raw_enabled)
        view_box = QGroupBox('View')
        QVBoxLayout(view_box).addWidget(self.raw_toggle)

        # Preprocessing
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(-GAMMA_STEPS, GAMMA_STEPS)
        self.gamma_slider.setEnabled(False)  # until the calibration node's value is known
        self.gamma_slider.valueChanged.connect(self.on_gamma_slider)
        self.gamma_value = QLabel('-')
        self.gamma_value.setMinimumWidth(36)
        gamma_row = QWidget()
        gamma_layout = QHBoxLayout(gamma_row)
        gamma_layout.setContentsMargins(0, 0, 0, 0)
        gamma_layout.addWidget(self.gamma_slider, 1)
        gamma_layout.addWidget(self.gamma_value)
        detection_box = form_box('Detection', [('gamma:', gamma_row)])
        self.flag_boxes = {}
        for name, label, tooltip in DETECTOR_FLAGS:
            box = QCheckBox(label)
            box.setToolTip(tooltip)
            box.setEnabled(False)  # until the calibration node's value is known
            box.toggled.connect(lambda checked, name=name: self.send_parameter(name, checked))
            detection_box.layout().addRow(box)
            self.flag_boxes[name] = box
        self._send_timer = QTimer(self)
        self._send_timer.setSingleShot(True)
        self._send_timer.setInterval(SEND_DELAY_MS)
        self._send_timer.timeout.connect(self.send_gamma)

        # Info
        self.board_found = QLabel('-')
        self.board_grid = QLabel('-')
        board_box = form_box('Board', [('found:', self.board_found), ('grid:', self.board_grid)])
        self.camera_size = QLabel('-')
        self.shown_size = QLabel('-')
        image_box = form_box('Image', [('camera:', self.camera_size),
                                       ('shown:', self.shown_size)])
        topics_box = QGroupBox('Topics')
        topics_layout = QVBoxLayout(topics_box)
        for topic in (node.preview_sub.topic_name, node.detection_sub.topic_name,
                      f'{node.raw_topic} (raw)'):
            label = QLabel(topic)
            label.setWordWrap(True)
            topics_layout.addWidget(label)

        side_panel = QWidget()
        side_panel.setFixedWidth(260)
        side_layout = QVBoxLayout(side_panel)
        for box in (view_box, detection_box, board_box, image_box, topics_box):
            side_layout.addWidget(box)
        side_layout.addStretch(1)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.addWidget(self.video, 1)
        layout.addWidget(side_panel)
        self.setCentralWidget(central)

        self.status = QLabel()
        self.statusBar().addWidget(self.status, 1)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(REFRESH_INTERVAL_MS)

    # ------------------------------------------------------------------ video

    def refresh(self):
        self.node.poll_parameters(REMOTE_PARAMETERS)
        snap = self.node.snapshot()
        live = snap.detection is not None and snap.detection_age_s < DETECTION_TIMEOUT_S

        if self.raw_toggle.isChecked():
            if live and snap.raw_frame is not None:
                # The camera frame the newest available detection was computed on, so the
                # overlay sits exactly on the board.
                self.show_pair(snap.raw_frame, snap.raw_detection)
            elif not live and snap.newest_raw is not None:
                self.show_pair(snap.newest_raw, None,
                               'no detections received - is calibration_node running?')
            elif snap.newest_raw is None:
                self.show_message(f'waiting for camera images on {self.node.raw_topic}')
        elif live and snap.preview is not None:
            self.show_pair(snap.preview, snap.preview_detection)
        elif not live:
            self.show_message(f'waiting for {self.node.target} (previews and detections)')

        self.sync_gamma(snap.parameters.get('gamma'))
        for name, box in self.flag_boxes.items():
            self.sync_checkbox(name, box, snap.parameters.get(name))
        self.update_labels(snap, live)

    def show_pair(self, frame, detection, message=None):
        if frame is self._shown_frame and detection is self._shown_detection:
            return
        corners, corners_width = None, None
        if detection is not None:
            corners, corners_width = detection_corners(detection), detection.image_width
            message = message or (None if detection.found else 'no board detected')
        self.video.set_frame(frame.image, corners, message, corners_width)
        self._shown_frame, self._shown_detection, self._shown_message = frame, detection, message

    def show_message(self, message):
        """Show a message over the last picture (kept for orientation)."""
        if self._shown_frame is None and self._shown_message == message:
            return
        self.video.set_message(message)
        self._shown_frame, self._shown_detection, self._shown_message = None, None, message

    # ------------------------------------------------------------------ remote parameters

    def send_parameter(self, name, value):
        if self.node.set_parameter(name, value):
            self._pending[name] = (value, time.monotonic())

    def confirmed(self, name, value):
        """Return False while a value sent for name is not yet reflected by the node."""
        if name not in self._pending:
            return True
        sent, sent_at = self._pending[name]
        same = math.isclose(value, sent, rel_tol=1e-6) if isinstance(sent, float) else (
            value == sent)
        if not same and time.monotonic() - sent_at < CONFIRM_TIMEOUT_S:
            return False
        del self._pending[name]  # confirmed, or rejected / lost: show the node's value
        return True

    def sync_checkbox(self, name, box, value):
        """Show the calibration node's value of a boolean parameter."""
        box.setEnabled(value is not None)
        if value is None or not self.confirmed(name, value) or box.isChecked() == value:
            return
        box.blockSignals(True)  # not a user change: do not send it back
        box.setChecked(value)
        box.blockSignals(False)

    def on_gamma_slider(self, position):
        self.gamma_value.setText(f'{slider_to_gamma(position):.2f}')
        self._send_timer.start()  # restarts while the slider keeps moving

    def send_gamma(self):
        self.send_parameter('gamma', slider_to_gamma(self.gamma_slider.value()))

    def sync_gamma(self, gamma):
        """Show the calibration node's gamma unless the user is changing it."""
        if gamma is None:
            self.gamma_slider.setEnabled(False)
            self.gamma_value.setText('-')
            return
        self.gamma_slider.setEnabled(True)
        if self.gamma_slider.isSliderDown() or self._send_timer.isActive():
            return
        if not self.confirmed('gamma', gamma):
            return
        position = gamma_to_slider(gamma)
        if position != self.gamma_slider.value():
            self.gamma_slider.blockSignals(True)  # not a user change: do not send it back
            self.gamma_slider.setValue(position)
            self.gamma_slider.blockSignals(False)
        self.gamma_value.setText(f'{gamma:.2f}')

    # ------------------------------------------------------------------ labels

    def update_labels(self, snap, live):
        detection = snap.detection
        if detection is not None:
            self.camera_size.setText(f'{detection.image_width} x {detection.image_height}')
        frame = self._shown_frame
        if frame is not None:
            height, width = frame.image.shape[:2]
            source = 'preview' if frame.encoding == 'preview' else f'raw, {frame.encoding}'
            self.shown_size.setText(f'{width} x {height} ({source})')

        if live:
            self.board_found.setText('yes' if detection.found else 'no')
            self.board_grid.setText(
                f'{detection.cols} x {detection.rows}' if detection.found else '-')
            detection_text = (f'detection {snap.detection_fps:.1f} fps, '
                              f'{detection.detection_time_ms:.0f} ms')
        else:
            self.board_found.setText('-')
            self.board_grid.setText('-')
            detection_text = 'detection: none received'
        if self.raw_toggle.isChecked():
            image_text = f'raw camera {snap.raw_fps:.1f} fps'
        else:
            image_text = f'preview {snap.preview_fps:.1f} fps'
        self.status.setText(f'{image_text}   |   {detection_text}')
