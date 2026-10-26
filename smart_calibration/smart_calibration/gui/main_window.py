"""Main window of the calibration GUI: video panel, side panel and status bar."""

import math
import time

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QVBoxLayout, QWidget)

from smart_calibration.gui.histogram_widget import HistogramWidget
from smart_calibration.gui.parameter_slider import ParameterSlider
from smart_calibration.gui.video_widget import VideoWidget
from smart_calibration_interfaces.msg import BoardDetection

REFRESH_INTERVAL_MS = 33
# Without a detection for this long, the calibration node is considered gone.
DETECTION_TIMEOUT_S = 1.0
# After sending, wait this long for the node to confirm before showing its value again.
CONFIRM_TIMEOUT_S = 1.0
# The status box shows "kept" this long after a view was kept, so it can be noticed.
KEPT_HIGHLIGHT_S = 1.0

# Gamma slider: logarithmic, so 1.0 is in the middle and 1/GAMMA_LIMIT .. GAMMA_LIMIT at the ends
# (matches the calibration node's parameter range 0.2 - 5.0).
GAMMA_LIMIT = 5.0
GAMMA_STEPS = 100
# Contrast curve sliders (calibration node ranges: strength 0 - 10, midpoint 0 - 1).
CONTRAST_STEPS_PER_UNIT = 5
MIDPOINT_STEPS = 100


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
REMOTE_PARAMETERS = ('gamma', 'contrast_strength', 'contrast_midpoint', 'board_mode',
                     'board_cols', 'board_rows') + tuple(name for name, _, _ in DETECTOR_FLAGS)


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


def check_line(label, status, value, failed_state='fail'):
    """Status box line (label, value, mark, state) for one of BoardDetection's checks."""
    if status == BoardDetection.CHECK_PASSED:
        return label, value, '✓', 'pass'
    if status == BoardDetection.CHECK_FAILED:
        return label, value, '✗', failed_state
    return label, '-', '', 'unknown'


def status_lines(detection, kept_label=None):
    """
    Status box lines for a BoardDetection: board found, then its view selection checks.

    kept_label replaces the new-view line (shown for a while after a view was kept).
    """
    if not detection.found:
        return [('board', 'not found', '', 'fail')] + [
            (label, '-', '', 'unknown') for label in ('motion', 'corners', 'new view')]
    lines = [('board', 'found', '', 'pass'),
             check_line('motion', detection.speed_check, f'{detection.board_speed:.0f} px/s'),
             check_line('corners', detection.residual_check, f'{detection.corner_residual:.2f}')]
    if kept_label:
        lines.append(('new view', kept_label, '', 'kept'))
    else:
        # The first view passes without a distance. Too close to a kept view is no error: amber.
        distance = ('first' if math.isnan(detection.view_distance)
                    else f'{detection.view_distance:.3f}')
        lines.append(check_line('new view', detection.distance_check, distance, 'warn'))
    return lines


def uncertainty_map(calibration):
    """Return the (rows, cols) uncertainty map of a CalibrationState, px."""
    return np.asarray(calibration.uncertainty_map, np.float32).reshape(
        calibration.map_rows, calibration.map_cols)


def calibration_texts(calibration):
    """Texts {label name: text} of the Calibration box for a CalibrationState (or None)."""
    if calibration is None:
        return dict(views='-', rms='-', focal='-', centre='-', uncertainty='-', groups='',
                    hints='')
    K, std = calibration.camera_matrix, calibration.std_deviations
    return dict(
        views=f'{calibration.views}' + (' (k3 fixed)' if calibration.k3_fixed else ''),
        rms=f'{calibration.rms:.3f} px',
        focal=f'{K[0]:.1f} / {K[4]:.1f}  ±{max(std[0], std[1]):.1f}',
        centre=f'{K[2]:.1f} / {K[5]:.1f}  ±{max(std[2], std[3]):.1f}',
        uncertainty=f'{calibration.uncertainty:.2f} px',
        groups=(f'focal {calibration.focal_uncertainty:.2f}, centre '
                f'{calibration.centre_uncertainty:.2f}, distortion '
                f'{calibration.distortion_uncertainty:.2f}'),
        hints='\n'.join(calibration.hints))


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
        self._kept_highlight = None  # (label, monotonic time until which it is shown)
        self._pending = {}  # parameter name -> (value sent, time sent), until the node confirms

        # View
        self.raw_toggle = QCheckBox('raw camera (full resolution)')
        self.raw_toggle.setToolTip(
            'Show the camera stream instead of the calibration node\'s preprocessed preview. '
            'Needs the full camera stream: best on the same machine as the camera.')
        # Mouse only: as the first widget it would get the keyboard focus when the window opens,
        # and a stray Space key press (e.g. typing in the terminal as the window appears) would
        # switch it on.
        self.raw_toggle.setFocusPolicy(Qt.NoFocus)
        self.raw_toggle.toggled.connect(self.node.set_raw_enabled)
        self.map_toggle = QCheckBox('uncertainty map')
        self.map_toggle.setToolTip(
            'Colour the image by how uncertain the newest background calibration is there: how '
            'far a point could be projected off, in px (green: small, red: large). Bring the '
            'board to the red regions.')
        self.map_toggle.setFocusPolicy(Qt.NoFocus)
        self.map_toggle.setChecked(True)
        self._shown_map = None  # (CalibrationState, toggle state) shown
        view_box = QGroupBox('View')
        view_layout = QVBoxLayout(view_box)
        view_layout.addWidget(self.raw_toggle)
        view_layout.addWidget(self.map_toggle)

        # Detection: tone mapping sliders, histogram, detector flags
        self.sliders = {
            'gamma': ParameterSlider(
                'gamma', -GAMMA_STEPS, GAMMA_STEPS, gamma_to_slider, slider_to_gamma,
                self.send_parameter),
            'contrast_strength': ParameterSlider(
                'contrast_strength', 0, 10 * CONTRAST_STEPS_PER_UNIT,
                lambda value: round(value * CONTRAST_STEPS_PER_UNIT),
                lambda position: position / CONTRAST_STEPS_PER_UNIT,
                self.send_parameter, '{:.1f}'),
            'contrast_midpoint': ParameterSlider(
                'contrast_midpoint', 0, MIDPOINT_STEPS,
                lambda value: round(value * MIDPOINT_STEPS),
                lambda position: position / MIDPOINT_STEPS, self.send_parameter),
        }
        self.sliders['gamma'].setToolTip('> 1 brightens dark regions, < 1 darkens bright ones.')
        self.sliders['contrast_strength'].setToolTip(
            'Steepness of the S-shaped contrast curve after gamma (0: off).')
        self.sliders['contrast_midpoint'].setToolTip(
            'Grey level the contrast curve keeps: darker gets darker, lighter gets lighter.')
        self.histogram = HistogramWidget()
        self._shown_histogram = None
        detection_box = form_box('Detection', [
            ('gamma:', self.sliders['gamma']),
            ('contrast:', self.sliders['contrast_strength']),
            ('midpoint:', self.sliders['contrast_midpoint'])])
        detection_box.layout().addRow(self.histogram)
        self.flag_boxes = {}
        for name, label, tooltip in DETECTOR_FLAGS:
            box = QCheckBox(label)
            box.setToolTip(tooltip)
            box.setEnabled(False)  # until the calibration node's value is known
            box.toggled.connect(lambda checked, name=name: self.send_parameter(name, checked))
            detection_box.layout().addRow(box)
            self.flag_boxes[name] = box

        # Info (per-frame detection results are in the video's status box)
        self.board_size = QLabel('-')
        self.board_size.setToolTip('The board the calibration node looks for: inner corners '
                                   '(cols x rows) of a plain chessboard, or an oversized board '
                                   'with markers (any visible part of it).')
        self.kept_views = QLabel('-')
        views_box = form_box('Calibration views', [('board:', self.board_size),
                                                   ('kept:', self.kept_views)])
        # Background calibration
        self.calibration_labels = {name: QLabel('-') for name in (
            'views', 'rms', 'focal', 'centre', 'uncertainty', 'groups', 'hints')}
        tooltips = dict(
            views='Views used by the newest background calibration. k3 stays 0 until '
                  'free_k3_after_views views.',
            rms='Reprojection error over all corners.',
            focal='fx / fy in px, ± the larger standard deviation (times uncertainty_scale).',
            centre='Principal point cx / cy in px, ± the larger standard deviation.',
            uncertainty='Largest value of the uncertainty map: how far a point could be projected '
                        'off, px (times uncertainty_scale). It does not see systematic errors, so '
                        'a small value alone does not prove a good calibration.',
            groups='Largest uncertainty from each parameter group alone, px.',
            hints='View types that are missing; new views fix them.')
        for name, label in self.calibration_labels.items():
            label.setToolTip(tooltips[name])
        for name in ('groups', 'hints'):
            self.calibration_labels[name].setWordWrap(True)
        self.calibration_labels['hints'].setStyleSheet('color: rgb(255, 200, 60)')
        labels = self.calibration_labels
        calibration_box = form_box('Calibration', [
            ('views:', labels['views']), ('rms:', labels['rms']), ('focal:', labels['focal']),
            ('centre:', labels['centre']), ('uncertainty:', labels['uncertainty'])])
        calibration_box.layout().addRow(labels['groups'])
        calibration_box.layout().addRow(labels['hints'])

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
        for box in (view_box, detection_box, views_box, calibration_box, image_box,
                    topics_box):
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

        for name, slider in self.sliders.items():
            self.sync_slider(name, slider, snap.parameters.get(name))
        for name, box in self.flag_boxes.items():
            self.sync_checkbox(name, box, snap.parameters.get(name))
        self.update_histogram(snap)
        self.update_calibration(snap.calibration)
        self.update_labels(snap, live)

    def show_pair(self, frame, detection, message=None):
        if frame is self._shown_frame and detection is self._shown_detection:
            return
        corners, corners_width, status, origin = None, None, None, None
        if detection is not None:
            corners, corners_width = detection_corners(detection), detection.image_width
            if detection.found:
                origin = (detection.origin_row, detection.origin_col)
            now = time.monotonic()
            if detection.kept:
                self._kept_highlight = (f'kept #{detection.kept_views}', now + KEPT_HIGHLIGHT_S)
            kept_label = None
            if self._kept_highlight is not None and now < self._kept_highlight[1]:
                kept_label = self._kept_highlight[0]
            status = status_lines(detection, kept_label)
        self.video.set_frame(frame.image, corners, message, corners_width, status, origin)
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

    def sync_slider(self, name, slider, value):
        """Show the calibration node's value unless the user is changing it."""
        if value is None:
            slider.show_remote(None)
        elif not slider.busy() and self.confirmed(name, value):
            slider.show_remote(value)

    def update_histogram(self, snap):
        # The curve follows a slider while it is moved, otherwise the node's value.
        curve = []
        for name in ('contrast_strength', 'contrast_midpoint'):
            slider = self.sliders[name]
            curve.append(slider.value() if slider.busy() else snap.parameters.get(name))
        if None not in curve:
            self.histogram.set_curve(*curve)
        if snap.histogram is not None and snap.histogram is not self._shown_histogram:
            self.histogram.set_histograms(snap.histogram.before_contrast,
                                          snap.histogram.after_contrast)
            self._shown_histogram = snap.histogram

    # ------------------------------------------------------------------ calibration

    def update_calibration(self, calibration):
        shown = (calibration, self.map_toggle.isChecked())
        if self._shown_map is not None and all(
                a is b for a, b in zip(shown, self._shown_map)):
            return
        self._shown_map = shown
        self.video.set_uncertainty_map(
            uncertainty_map(calibration)
            if calibration is not None and self.map_toggle.isChecked() else None)
        for name, text in calibration_texts(calibration).items():
            self.calibration_labels[name].setText(text)
        self.calibration_labels['hints'].setVisible(bool(calibration and calibration.hints))

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

        mode = snap.parameters.get('board_mode')
        cols, rows = snap.parameters.get('board_cols'), snap.parameters.get('board_rows')
        if mode == 'oversized_marker':
            self.board_size.setText('oversized, markers')
        else:
            self.board_size.setText('-' if None in (cols, rows) else f'{cols} x {rows}')
        if live:
            # The node's current count: from the newest detection, not the one on screen.
            self.kept_views.setText(str(detection.kept_views))
            detection_text = (f'detection {snap.detection_fps:.1f} fps, '
                              f'{detection.detection_time_ms:.0f} ms')
        else:
            self.kept_views.setText('-')
            detection_text = 'detection: none received'
        if self.raw_toggle.isChecked():
            image_text = f'raw camera {snap.raw_fps:.1f} fps'
        else:
            image_text = f'preview {snap.preview_fps:.1f} fps'
        self.status.setText(f'{image_text}   |   {detection_text}')
