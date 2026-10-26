"""Main window of the calibration GUI: video panel, side panel and status bar."""

import math
import time

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QPushButton, QSlider, QVBoxLayout, QWidget)

from smart_calibration.gui.histogram_widget import HistogramWidget
from smart_calibration.gui.parameter_slider import ParameterSlider
from smart_calibration.gui.strip_plot import StripPlot
from smart_calibration.gui.undistortion import Undistortion
from smart_calibration.gui.video_widget import VideoWidget
from smart_calibration_interfaces.msg import BoardDetection, FinalCalibration

REFRESH_INTERVAL_MS = 33
RESULT_STATES = (FinalCalibration.STATE_DONE, FinalCalibration.STATE_ACCEPTED)
# Without a detection for this long, the stream counts as stopped.
DETECTION_TIMEOUT_S = 1.0
# How long a sent parameter value waits for the node's confirmation.
CONFIRM_TIMEOUT_S = 1.0
# How long the status box shows "kept".
KEPT_HIGHLIGHT_S = 1.0

# Gamma slider: logarithmic, 1/GAMMA_LIMIT .. GAMMA_LIMIT.
GAMMA_LIMIT = 5.0
GAMMA_STEPS = 100
CONTRAST_STEPS_PER_UNIT = 5
MIDPOINT_STEPS = 100


# Detector flags: (parameter, checkbox label, tooltip).
DETECTOR_FLAGS = (
    ('detector_normalize', 'normalize',
     'Histogram equalization before detection; may help on dark images.'),
    ('detector_exhaustive', 'exhaustive search',
     'Searches longer: finds the board more often, slower.'),
    ('detector_accuracy', 'accuracy',
     'More accurate corners, ~4x slower.'),
)
REMOTE_PARAMETERS = ('gamma', 'contrast_strength', 'contrast_midpoint', 'board_mode',
                     'board_cols', 'board_rows') + tuple(name for name, _, _ in DETECTOR_FLAGS)


def gamma_to_slider(gamma):
    gamma = min(max(gamma, 1.0 / GAMMA_LIMIT), GAMMA_LIMIT)
    return round(GAMMA_STEPS * math.log(gamma, GAMMA_LIMIT))


def slider_to_gamma(position):
    return GAMMA_LIMIT ** (position / GAMMA_STEPS)


def detection_corners(detection):
    """Return the corners (rows, cols, 2) of a BoardDetection, or None."""
    if not detection.found:
        return None
    return np.asarray(detection.corners, dtype=np.float32).reshape(
        detection.rows, detection.cols, 2)


def check_line(label, status, value, failed_state='fail'):
    """Return the status box line (label, value, mark, state) of a check."""
    if status == BoardDetection.CHECK_PASSED:
        return label, value, '✓', 'pass'
    if status == BoardDetection.CHECK_FAILED:
        return label, value, '✗', failed_state
    return label, '-', '', 'unknown'


def status_lines(detection, kept_label=None):
    """Return the status box lines of a BoardDetection; kept_label replaces the new-view line."""
    if not detection.found:
        return [('board', 'not found', '', 'fail')] + [
            (label, '-', '', 'unknown') for label in ('motion', 'corners', 'new view')]
    lines = [('board', 'found', '', 'pass'),
             check_line('motion', detection.speed_check, f'{detection.board_speed:.0f} px/s'),
             check_line('corners', detection.residual_check, f'{detection.corner_residual:.2f}')]
    if kept_label:
        lines.append(('new view', kept_label, '', 'kept'))
    else:
        # Too close to a kept view is no error: amber.
        distance = ('first' if math.isnan(detection.view_distance)
                    else f'{detection.view_distance:.3f}')
        lines.append(check_line('new view', detection.distance_check, distance, 'warn'))
    return lines


def uncertainty_map(calibration):
    """Return the (rows, cols) uncertainty map of a CalibrationState, px."""
    return np.asarray(calibration.uncertainty_map, np.float32).reshape(
        calibration.map_rows, calibration.map_cols)


def board_text(parameters):
    """Describe the board the calibration node looks for, from its parameters."""
    if parameters.get('board_mode') == 'oversized_marker':
        return 'oversized, markers'
    cols, rows = parameters.get('board_cols'), parameters.get('board_rows')
    return '-' if None in (cols, rows) else f'{cols} x {rows}'


def phase_text(final):
    """Describe the calibration phase (final None: collecting)."""
    if final is None:
        return 'collecting views'
    return {FinalCalibration.STATE_RUNNING: f'finishing: {final.stage}',
            FinalCalibration.STATE_DONE: 'finished: check the result',
            FinalCalibration.STATE_FAILED: final.stage,
            FinalCalibration.STATE_ACCEPTED: 'accepted'}[final.state]


# Calibration box rows: (name, label).
CALIBRATION_ROWS = (('board', 'board:'), ('phase', 'phase:'), ('views', 'views:'),
                    ('rms', 'rms:'), ('uncertainty', 'uncertainty:'))


def calibration_texts(calibration, kept_views, final):
    """
    Return the Calibration box texts {name: text}.

    While collecting (final None): the background calibration and the kept view count.
    """
    texts = dict(phase=phase_text(final), views='-', rms='-', uncertainty='-',
                 hints='')
    if final is None:
        shown = calibration
        kept = '-' if kept_views is None else f'{kept_views} kept'
        texts['views'] = kept if calibration is None else (
            f'{kept}, calibrated from {calibration.views}'
            + (' (k3 fixed)' if calibration.k3_fixed else ''))
    elif final.state in RESULT_STATES:
        shown = final.final
        used = f'{shown.views} of {final.total_views} used'
        counts = [f'{len(final.removed)} removed'] + (
            [f'{len(final.suspects)} suspect'] if final.suspects else [])
        texts['views'] = (f'{used} ({", ".join(counts)})' if final.checked
                          else f'{used} (too few for the outlier check)') + (
            ', k3 fixed' if shown.k3_fixed else '')
    else:
        shown = None
        texts['views'] = f'{final.total_views} kept'
    if shown is None:
        return texts
    values = uncertainty_map(shown)
    mean = float(np.nanmean(values)) if np.isfinite(values).any() else math.nan
    texts.update(
        rms=(f'{final.initial.rms:.3f} -> {shown.rms:.3f} px' if final is not None
             and final.removed else f'{shown.rms:.3f} px'),
        uncertainty=f'{shown.uncertainty:.2f} px max, {mean:.2f} px mean',
        hints='\n'.join(f'• {hint}' for hint in shown.hints))
    return texts


def final_outlines(final):
    """Return the view outlines (k, 2) of a FinalCalibration, in full image pixels."""
    points = np.asarray(final.outline_points, np.float32).reshape(-1, 2)
    ends = np.cumsum(np.asarray(final.outline_sizes, np.int64))
    return [points[int(end) - size:int(end)] for end, size in zip(ends, final.outline_sizes)]


def view_styles(final):
    """Return per view 'removed', 'suspect', 'restored' or 'kept'."""
    styles = ['kept'] * final.total_views
    if final.state in RESULT_STATES:
        for index in final.forced:
            styles[index] = 'restored'
        for index in final.suspects:
            styles[index] = 'suspect'
        for index in final.removed:
            styles[index] = 'removed'
    return styles


def polygon_area(polygon):
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def inside(polygon, x, y):
    """Return whether (x, y) lies inside a polygon (k, 2)."""
    x0, y0 = polygon[:, 0], polygon[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    crosses = (y0 > y) != (y1 > y)
    with np.errstate(divide='ignore', invalid='ignore'):
        at = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
    return bool(np.count_nonzero(crosses & (x < at)) % 2)


def view_at(outlines, x, y):
    """Return the index of the smallest outline containing (x, y), or None."""
    hits = [i for i, outline in enumerate(outlines) if inside(outline, x, y)]
    return min(hits, key=lambda i: polygon_area(outlines[i])) if hits else None


def flagged_entries(final):
    """Return (view index, text, style) for each flagged or restored view."""
    entries = []
    for index, z, ratio in zip(final.removed, final.removed_z, final.removed_uncertainty_ratio):
        entries.append((index, f'#{index + 1} removed: {final.leave_one_out[index]:.2f} px, '
                               f'z {z:.1f}', 'removed'))
    for index, z, ratio in zip(final.suspects, final.suspects_z,
                               final.suspects_uncertainty_ratio):
        without = (f'uncertainty x{ratio:.1f}' if math.isfinite(ratio)
                   else 'no valid model for part of the image')
        entries.append((index, f'#{index + 1} suspect, kept: z {z:.1f}; without it: {without}',
                        'suspect'))
    for index in final.forced:
        entries.append((index, f'#{index + 1} restored', 'restored'))
    return entries


def view_description(final, index):
    """Describe the selected view of a FinalCalibration for the side panel."""
    error = final.leave_one_out[index] if index < len(final.leave_one_out) else math.nan
    style = view_styles(final)[index]
    text = {
        'removed': 'removed, the other views predict it badly',
        'suspect': 'suspect, kept: no other view covers its region. Add views there',
        'restored': 'restored by you',
        'kept': 'kept',
    }[style]
    return f'#{index + 1}: {text}. Error {error:.2f} px.'


def current_calibration(calibration, final):
    """Return the final pass's result, else the background calibration."""
    if final is not None:
        return final.final if final.state in RESULT_STATES else None
    return calibration


def undistortion_key(calibration, alpha):
    """Return Undistortion.key() for a CalibrationState."""
    return (tuple(float(v) for v in calibration.camera_matrix),
            tuple(float(v) for v in calibration.distortion),
            (int(calibration.image_width), int(calibration.image_height)), alpha)


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
        # What is displayed, compared by identity.
        self._shown_frame = None
        self._shown_detection = None
        self._shown_message = None
        self._kept_highlight = None  # (label, monotonic time it ends)
        self._pending = {}  # parameter name -> (value sent, time sent)

        # View
        self.raw_toggle = QCheckBox('raw camera (full resolution)')
        self.raw_toggle.setToolTip('Show the camera stream instead of the preprocessed preview.')
        # Mouse only, so a stray Space key does not toggle it.
        self.raw_toggle.setFocusPolicy(Qt.NoFocus)
        self.raw_toggle.toggled.connect(self.node.set_raw_enabled)
        self.map_toggle = QCheckBox('uncertainty map')
        self.map_toggle.setToolTip('Where the calibration is uncertain: bring the board to the '
                                   'red regions.')
        self.map_toggle.setFocusPolicy(Qt.NoFocus)
        self.map_toggle.setChecked(True)
        self._shown_map = None  # (CalibrationState, toggle state) shown
        self.undistort_toggle = QCheckBox('undistorted')
        self.undistort_toggle.setToolTip('Show the image undistorted by the calibration. Dashed '
                                         'line: beyond it the model is not valid yet.')
        self.undistort_toggle.setFocusPolicy(Qt.NoFocus)
        self.undistort_toggle.setEnabled(False)  # until there is a calibration
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(100)
        self.alpha_slider.setFocusPolicy(Qt.NoFocus)
        self.alpha_slider.setToolTip('Left: only valid pixels (cropped). Right: the whole image.')
        self.alpha_row = QWidget()
        alpha_layout = QHBoxLayout(self.alpha_row)
        alpha_layout.setContentsMargins(20, 0, 0, 0)
        alpha_layout.addWidget(QLabel('crop'))
        alpha_layout.addWidget(self.alpha_slider)
        alpha_layout.addWidget(QLabel('all'))
        self.alpha_row.hide()
        self._undistortion = None  # Undistortion applied to everything shown, or None
        view_box = QGroupBox('View')
        view_layout = QVBoxLayout(view_box)
        view_layout.addWidget(self.raw_toggle)
        view_layout.addWidget(self.map_toggle)
        view_layout.addWidget(self.undistort_toggle)
        view_layout.addWidget(self.alpha_row)

        # Detection
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
        self.sliders['contrast_strength'].setToolTip('Steepness of the contrast curve (0: off).')
        self.sliders['contrast_midpoint'].setToolTip(
            'Grey level the curve keeps: darker gets darker, lighter gets lighter.')
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

        # Calibration
        self.calibration_labels = {
            name: QLabel('-') for name in [row for row, _ in CALIBRATION_ROWS] + ['hints']}
        tooltips = dict(
            board='Inner corners (cols x rows), or the oversized board with markers.',
            phase='Collecting views, or the final calibration after Finish.',
            views='Kept views, and how many the shown calibration uses.',
            rms='Reprojection error over all corners.',
            uncertainty='How far a point could be off, px: worst region and image average.',
            hints='View types that are missing; new views fix them.')
        for name, label in self.calibration_labels.items():
            label.setToolTip(tooltips[name])
            label.setWordWrap(True)
        self.calibration_labels['hints'].setStyleSheet('color: rgb(255, 200, 60)')
        labels = self.calibration_labels
        calibration_box = form_box('Calibration', [
            (shown, labels[name]) for name, shown in CALIBRATION_ROWS])
        calibration_box.layout().addRow(labels['hints'])

        # While collecting
        self.finish_button = QPushButton('Finish')
        self.finish_button.setToolTip('Stop collecting, remove clear outliers, calibrate.')
        self.finish_button.setFocusPolicy(Qt.NoFocus)
        self.finish_button.clicked.connect(lambda: self.node.finish())
        calibration_box.layout().addRow(self.finish_button)

        # After Finish
        self.strip = StripPlot()
        self.strip.setToolTip('How badly the other views predict each view. Click a dot to '
                              'see it.')
        self.strip.selected.connect(self.select_view)
        self.flagged_list = QListWidget()
        self.flagged_list.setMaximumHeight(90)
        self.flagged_list.itemClicked.connect(
            lambda item: self.select_view(item.data(Qt.UserRole)))
        self.selected_text = QLabel('Click a view in the image or a dot above.')
        self.selected_text.setWordWrap(True)
        self.map_before = QCheckBox('map before removal')
        self.map_before.setToolTip('Map of the calibration from all views.')
        self.map_before.setFocusPolicy(Qt.NoFocus)
        self.error_labels = QCheckBox('error labels')
        self.error_labels.setToolTip('Label each corner of the selected view with its error.')
        self.error_labels.setFocusPolicy(Qt.NoFocus)
        self.error_labels.setChecked(True)
        self.error_labels.toggled.connect(self.video.set_error_labels)
        self.fold_warning = QLabel('Not valid in the image corners: resume and add views there.')
        self.fold_warning.setWordWrap(True)
        self.fold_warning.setStyleSheet('color: rgb(255, 200, 60)')
        self.fold_warning.hide()
        self.restore_button = QPushButton('Restore')
        self.restore_button.setToolTip('Put the selected view back.')
        self.restore_button.clicked.connect(self.restore_selected)
        self.accept_button = QPushButton('Accept')
        self.accept_button.setToolTip('Save this calibration (on the calibration node).')
        self.accept_button.clicked.connect(lambda: self.node.accept())
        self.resume_button = QPushButton('Resume')
        self.resume_button.setToolTip('Collect more views (removed views are dropped).')
        self.resume_button.clicked.connect(lambda: self.node.resume())
        self.final_section = QWidget()
        final_layout = QVBoxLayout(self.final_section)
        final_layout.setContentsMargins(0, 0, 0, 0)
        final_layout.addWidget(QLabel('leave-one-out error per view:'))
        final_layout.addWidget(self.strip)
        final_layout.addWidget(self.flagged_list)
        final_layout.addWidget(self.selected_text)
        final_layout.addWidget(self.map_before)
        final_layout.addWidget(self.error_labels)
        final_layout.addWidget(self.fold_warning)
        buttons = QHBoxLayout()
        for button in (self.restore_button, self.accept_button, self.resume_button):
            button.setFocusPolicy(Qt.NoFocus)
            buttons.addWidget(button)
        final_layout.addLayout(buttons)
        self.final_section.hide()
        calibration_box.layout().addRow(self.final_section)

        self._final = None  # newest FinalCalibration
        self._shown_final = None  # FinalCalibration the flagged list shows
        self._shown_texts = None  # inputs of the Calibration box's texts on screen
        self._selected_view = None
        self._canvas_frame = None  # background image of the views canvas
        self._shown_canvas = None  # (FinalCalibration, selected, KeptView) on screen
        self._shown_outlines = None  # view outlines as drawn (undistorted or not)
        self._last_image = None  # the last camera / preview image shown, even if it stopped
        self.video.image_clicked.connect(self.on_image_clicked)

        side_panel = QWidget()
        side_panel.setFixedWidth(260)
        side_layout = QVBoxLayout(side_panel)
        self.detection_box = detection_box
        for box in (view_box, detection_box, calibration_box):
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
        self._final = snap.final
        final_mode = (snap.final is not None
                      and snap.final.state != FinalCalibration.STATE_COLLECTING)
        self.update_undistortion(
            current_calibration(snap.calibration, snap.final if final_mode else None))

        if final_mode:
            self.show_canvas(snap)
        elif self._canvas_frame is not None:
            self.leave_canvas()
        elif self.raw_toggle.isChecked():
            if live and snap.raw_frame is not None:
                # The frame the newest detection was computed on.
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
        final = snap.final if final_mode else None
        self.update_map(snap.calibration, final)
        self.update_calibration(snap, final)
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
        image = frame.image
        if self._undistortion is not None:
            image = self._undistortion.image(image)
            if corners is not None:
                corners = self._undistortion.points(corners)
        self.video.set_frame(image, corners, message, corners_width, status, origin)
        self._shown_frame, self._shown_detection, self._shown_message = frame, detection, message
        self._last_image = frame.image  # background of the views canvas after Finish

    def show_message(self, message):
        """Show a message over the last picture."""
        if self._shown_frame is None and self._shown_message == message:
            return
        self.video.set_message(message)
        self._shown_frame, self._shown_detection, self._shown_message = None, None, message

    # ------------------------------------------------------------------ remote parameters

    def send_parameter(self, name, value):
        if self.node.set_parameter(name, value):
            self._pending[name] = (value, time.monotonic())

    def confirmed(self, name, value):
        """Return False while the node does not reflect a value sent for name yet."""
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
        # The slider's value while it is moved, else the node's.
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

    def update_undistortion(self, calibration):
        """Rebuild the undistortion if the calibration or settings changed, and redraw."""
        self.undistort_toggle.setEnabled(calibration is not None)
        wanted = calibration is not None and self.undistort_toggle.isChecked()
        self.alpha_row.setVisible(self.undistort_toggle.isChecked())
        undistortion = None
        if wanted:
            alpha = self.alpha_slider.value() / 100
            undistortion = self._undistortion
            if undistortion is None or undistortion.key() != undistortion_key(calibration, alpha):
                undistortion = Undistortion(
                    np.asarray(calibration.camera_matrix).reshape(3, 3),
                    calibration.distortion,
                    (calibration.image_width, calibration.image_height), alpha)
        if undistortion is not self._undistortion:
            self._undistortion = undistortion
            self.video.set_boundary(
                undistortion.fold_outline() if undistortion is not None else None,
                undistortion.image_size[0] if undistortion is not None else None)
            self._shown_frame = self._shown_detection = self._shown_message = None
            self._shown_canvas = self._shown_map = None

    def update_map(self, calibration, final):
        """Show the uncertainty map of the background calibration, or of the final pass."""
        mapped = calibration
        if final is not None:
            mapped = None
            if final.state in RESULT_STATES:
                mapped = final.initial if self.map_before.isChecked() else final.final
        shown = (mapped, self.map_toggle.isChecked())
        if self._shown_map is not None and all(
                a is b for a, b in zip(shown, self._shown_map)):
            return
        self._shown_map = shown
        values = None
        if mapped is not None and self.map_toggle.isChecked():
            values = uncertainty_map(mapped)
            if self._undistortion is not None:
                values = self._undistortion.image(values)
        self.video.set_uncertainty_map(values)

    # ------------------------------------------------------------------ final calibration pass

    def show_canvas(self, snap):
        """Show the kept view outlines over the last frame, or the selected view."""
        final = snap.final
        if self._canvas_frame is None:
            self._canvas_frame = (self._last_image if self._last_image is not None
                                  else np.zeros((final.image_height, final.image_width), np.uint8))
        selected = self._selected_view
        if selected is not None and selected >= final.total_views:
            selected = self._selected_view = None
        view = None
        if selected is not None:
            view = snap.views.get((final.run, selected))
            if view is None:
                self.node.request_view(final.run, selected)
        shown = (final, selected, view)
        if self._shown_canvas is not None and all(
                a is b for a, b in zip(shown, self._shown_canvas)):
            return
        self._shown_canvas = shown
        message = final.stage if final.state in (FinalCalibration.STATE_RUNNING,
                                                 FinalCalibration.STATE_FAILED) else None
        undistortion = self._undistortion
        if view is not None:
            points = view.points
            residuals = view.residuals if view.residuals is not None else np.zeros_like(points)
            thumbnail = view.thumbnail
            if undistortion is not None:
                thumbnail = undistortion.image(thumbnail)
                residuals = undistortion.directions(points, residuals)
                points = undistortion.points(points)
            self.video.set_frame(thumbnail, message=message)
            self.video.set_dimmed(False)
            self.video.set_arrows(points, residuals, view.image_width)
        else:
            background = self._canvas_frame
            if undistortion is not None:
                background = undistortion.image(background)
            self.video.set_frame(background, message=message)
            self.video.set_dimmed(True)
            self.video.set_arrows(None, None, None)
        outlines = final_outlines(final)
        if undistortion is not None:
            outlines = [undistortion.points(outline) for outline in outlines]
        self._shown_outlines = outlines  # what a click is tested against
        self.video.set_views(outlines, view_styles(final), selected, final.image_width)
        # Redraw the live view completely once collection resumes.
        self._shown_frame = self._shown_detection = self._shown_message = None

    def leave_canvas(self):
        self._canvas_frame = self._shown_canvas = self._selected_view = None
        self.video.set_views(None, None, None, None)
        self.video.set_arrows(None, None, None)
        self.video.set_dimmed(False)

    def on_image_clicked(self, x, y):
        if self._final is not None and self._shown_outlines is not None:
            self.select_view(view_at(self._shown_outlines, x, y))

    def select_view(self, index):
        self._selected_view = index
        if index is not None and self._final is not None:
            self.node.request_view(self._final.run, index)

    def restore_selected(self):
        final = self._final
        if final is None or self._selected_view not in final.removed:
            return
        self.node.finish(sorted(set(final.forced) | {self._selected_view}))

    def update_calibration(self, snap, final):
        """Update the Calibration box; final: the FinalCalibration after Finish, else None."""
        # The last known count, also when the stream stopped.
        kept = snap.detection.kept_views if snap.detection is not None else None
        inputs = (snap.calibration, kept, final, board_text(snap.parameters))
        if self._shown_texts is None or any(
                a is not b and a != b for a, b in zip(inputs, self._shown_texts)):
            self._shown_texts = inputs
            texts = calibration_texts(*inputs[:3])
            texts['board'] = inputs[3]
            for name, text in texts.items():
                self.calibration_labels[name].setText(text)
            self.calibration_labels['hints'].setVisible(bool(texts['hints']))

        self.detection_box.setVisible(final is None)
        self.finish_button.setVisible(final is None)
        self.finish_button.setEnabled(self.node.finish_available() and bool(kept))
        self.final_section.setVisible(final is not None)
        if final is None:
            self._shown_final = None
            return
        has_result = final.state in RESULT_STATES
        selected = self._selected_view
        self.restore_button.setEnabled(final.state == FinalCalibration.STATE_DONE
                                       and selected in final.removed)
        self.accept_button.setEnabled(final.state == FinalCalibration.STATE_DONE)
        self.map_before.setVisible(has_result and len(final.removed) > 0)
        self.fold_warning.setVisible(has_result and final.final.distortion_folds)
        self.selected_text.setText(
            view_description(final, selected) if selected is not None and has_result
            else 'Click a view in the image or a dot above.')
        self.strip.set_data(final.leave_one_out if has_result else [], view_styles(final),
                            final.limit if has_result and final.checked else None, selected)
        if final is self._shown_final:
            return
        self._shown_final = final
        self.flagged_list.clear()
        if has_result:
            for index, text, style in flagged_entries(final):
                item = QListWidgetItem(text)
                item.setData(Qt.UserRole, index)
                self.flagged_list.addItem(item)
            if not self.flagged_list.count():
                self.flagged_list.addItem('no views flagged')

    # ------------------------------------------------------------------ labels

    def update_labels(self, snap, live):
        """Update the status bar."""
        detection = snap.detection
        camera = f'camera {self.node.raw_topic}' + (
            f' ({detection.image_width} x {detection.image_height})' if detection else '')
        if live:
            detection_text = (f'detection {snap.detection_fps:.1f} fps, '
                              f'{detection.detection_time_ms:.0f} ms')
        else:
            detection_text = 'detection: none received'
        if self.raw_toggle.isChecked():
            image_text = f'showing raw camera {snap.raw_fps:.1f} fps'
        else:
            image_text = f'showing preview {snap.preview_fps:.1f} fps'
        message = f'   |   {snap.service_message}' if snap.service_message else ''
        self.status.setText(f'{camera}   |   {image_text}   |   {detection_text}{message}')
