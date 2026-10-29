from datetime import datetime
import math
import signal
import sys
import threading
import time

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from rcl_interfaces.msg import (
    FloatingPointRange, IntegerRange, ParameterDescriptor, SetParametersResult)
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Header
from std_srvs.srv import Trigger

from smart_calibration.calibration import calibrate_views, final_calibration, summary
from smart_calibration.calibration_file import output_path, write_calibration
from smart_calibration.camera_models import camera_model, CAMERA_MODELS
from smart_calibration.detection import BOARD_MODES, detect_board, sb_flags
from smart_calibration.environment import check_environment
from smart_calibration.preprocessing import (
    FLOAT_MONO_ENCODINGS, preprocess, preprocess_with_histograms, SIXTEEN_BIT_ENCODINGS)
from smart_calibration.qos import image_qos
from smart_calibration.views import Assessment, reference_outline, ViewCollection
from smart_calibration_interfaces.msg import (
    BoardDetection, CalibrationState, FinalCalibration, ToneHistogram)
from smart_calibration_interfaces.srv import Finish, GetView

# Parameters that can be changed while the node runs.
LIVE_PARAMETERS = ('gamma', 'stretch_low_percentile', 'stretch_high_percentile',
                   'contrast_strength', 'contrast_midpoint',
                   'detector_normalize', 'detector_exhaustive', 'detector_accuracy',
                   'max_board_speed_rel', 'max_corner_residual', 'border_margin_px',
                   'min_view_distance', 'calibrate_every_views', 'min_calibration_views',
                   'free_k3_after_views', 'uncertainty_scale', 'outlier_z_limit',
                   'outlier_min_views', 'outlier_max_uncertainty_increase',
                   'outlier_max_fraction', 'output_file')
# Thumbnails of the kept views, for the GUI.
THUMBNAIL_WIDTH = 320
THUMBNAIL_JPEG_QUALITY = 80
# FinalCalibration states with a result.
RESULT_STATES = (FinalCalibration.STATE_DONE, FinalCalibration.STATE_ACCEPTED)


def float_range(low, high, description):
    return ParameterDescriptor(
        description=description,
        floating_point_range=[FloatingPointRange(from_value=low, to_value=high, step=0.0)])


def check_result(value, ok):
    """Return (check status, value) for a BoardDetection; None: not evaluated."""
    status = (BoardDetection.CHECK_NOT_EVALUATED if ok is None
              else BoardDetection.CHECK_PASSED if ok else BoardDetection.CHECK_FAILED)
    return status, math.nan if value is None else value


def fill_checks(detection, assessment):
    """Copy the view checks into a BoardDetection (assessment None: no board)."""
    if assessment is None:
        assessment = Assessment(None, None, None, None, None, None, None)
    detection.speed_check, detection.board_speed = check_result(
        assessment.speed, assessment.speed_ok)
    detection.residual_check, detection.corner_residual = check_result(
        assessment.residual, assessment.residual_ok)
    detection.distance_check, detection.view_distance = check_result(
        assessment.distance, assessment.distance_ok)
    detection.kept = assessment.view is not None


def calibration_message(result, header):
    """Return a CalibrationState message for a CalibrationResult."""
    state = CalibrationState()
    state.header = header
    state.image_width, state.image_height = result.image_size
    state.views, state.points = result.views, result.points
    state.duration_ms = result.duration_s * 1000.0
    state.camera_model = result.camera_model.name
    state.k3_fixed = result.k3_fixed
    state.rms, state.noise = result.rms, result.noise
    state.camera_matrix = result.camera_matrix.ravel().tolist()
    state.distortion = result.distortion.tolist()
    state.std_deviations = result.std_deviations.tolist()
    state.uncertainty_scale = result.uncertainty_scale
    state.uncertainty = result.uncertainty['total']
    state.focal_uncertainty = result.uncertainty['focal']
    state.centre_uncertainty = result.uncertainty['centre']
    state.distortion_uncertainty = result.uncertainty['distortion']
    state.map_rows, state.map_cols = result.uncertainty_map.shape
    state.uncertainty_map = result.uncertainty_map.astype(np.float32).ravel().tolist()
    state.view_rms = result.view_rms.astype(np.float32).tolist()
    state.view_tilt = result.view_tilts[:, 0].astype(np.float32).tolist()
    state.view_tilt_axis = result.view_tilts[:, 1].astype(np.float32).tolist()
    state.hints = list(result.hints)
    state.distortion_folds = result.distortion_folds
    return state


def jpeg(gray, max_width, quality):
    """Return a gray image as JPEG bytes, downscaled to at most max_width."""
    if gray.shape[1] > max_width:
        scale = max_width / gray.shape[1]
        gray = cv2.resize(gray, (max_width, round(gray.shape[0] * scale)),
                          interpolation=cv2.INTER_AREA)
    return cv2.imencode('.jpg', gray, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def compressed_image(data, header=None):
    image = CompressedImage()
    if header is not None:
        image.header = header
    image.format = 'mono8; jpeg compressed mono8'  # image_transport's naming
    image.data = data
    return image


def fill_final(msg, result, image_points, image_size, header):
    """Fill a FinalCalibration's view outlines and result (FinalResult or None)."""
    msg.total_views = len(image_points)
    msg.image_width, msg.image_height = image_size
    for points in image_points:
        hull = cv2.convexHull(points.reshape(-1, 1, 2).astype(np.float32)).reshape(-1, 2)
        msg.outline_sizes.append(len(hull))
        msg.outline_points.extend(hull.ravel().tolist())
    if result is None:
        return
    msg.initial = calibration_message(result.initial, header)
    msg.final = calibration_message(result.final, header)
    msg.checked, msg.z_limit, msg.limit = result.checked, result.z_limit, result.initial_limit
    msg.leave_one_out = result.initial.leave_one_out.astype(np.float32).tolist()
    msg.removed = [view.index for view in result.removed]
    msg.removed_z = [view.z for view in result.removed]
    msg.removed_uncertainty_ratio = [view.uncertainty_ratio for view in result.removed]
    msg.suspects = [view.index for view in result.suspects]
    msg.suspects_z = [view.z for view in result.suspects]
    msg.suspects_uncertainty_ratio = [view.uncertainty_ratio for view in result.suspects]
    msg.forced = list(result.forced)


def without_text(uncertainty_ratio):
    """Describe what removing a flagged view does to the uncertainty."""
    if not math.isfinite(uncertainty_ratio):
        return 'no valid lens model for part of the image'
    return f'uncertainty x{uncertainty_ratio:.1f}'


class Cancelled(Exception):
    """The final pass was cancelled."""


def format_value(value, spec):
    return '-' if value is None else format(value, spec)


class CalibrationNode(Node):

    def __init__(self):
        super().__init__('calibration_node')
        self.get_logger().info(
            f'python {sys.executable}, OpenCV {cv2.__version__}, numpy {np.__version__}')

        self.board_mode = self.declare_parameter('board_mode', 'chessboard', ParameterDescriptor(
            description='chessboard: plain board_cols x board_rows chessboard. '
                        'oversized_marker: radon checkerboard with markers, larger than the '
                        'view.')).value
        if self.board_mode not in BOARD_MODES:
            raise ValueError(f'board_mode must be one of {BOARD_MODES}, got {self.board_mode!r}')
        corner_count = ParameterDescriptor(
            description='Inner corners along this axis (chessboard mode).',
            integer_range=[IntegerRange(from_value=3, to_value=100, step=1)])
        self.board_cols = self.declare_parameter('board_cols', 9, corner_count).value
        self.board_rows = self.declare_parameter('board_rows', 6, corner_count).value
        self.square_size = self.declare_parameter('square_size', 0.025, float_range(
            0.001, 1.0, 'Side length of one square in metres.')).value
        self.camera_model = camera_model(self.declare_parameter(
            'camera_model', 'plumb_bob', ParameterDescriptor(
                description='Lens model: ' + ' or '.join(CAMERA_MODELS)
                            + ' (Kannala-Brandt, for wide-angle and fisheye lenses).')).value)
        # findChessboardCornersSB flags.
        self.declare_parameter('detector_normalize', False, ParameterDescriptor(
            description='Histogram equalization before detection.'))
        self.declare_parameter('detector_exhaustive', True, ParameterDescriptor(
            description='Longer search: finds the board more often, slower.'))
        self.declare_parameter('detector_accuracy', True, ParameterDescriptor(
            description='More accurate corners, ~4x slower.'))

        # Preprocessing.
        self.declare_parameter('gamma', 1.0, float_range(
            0.2, 5.0, 'Gamma after the contrast stretch; > 1 brightens dark regions.'))
        self.declare_parameter('stretch_low_percentile', 0.5, float_range(
            0.0, 50.0, 'Intensity percentile mapped to black.'))
        self.declare_parameter('stretch_high_percentile', 99.5, float_range(
            50.0, 100.0, 'Intensity percentile mapped to white.'))
        self.declare_parameter('contrast_strength', 0.0, float_range(
            0.0, 10.0, 'Steepness of the contrast curve after gamma (0: off).'))
        self.declare_parameter('contrast_midpoint', 0.5, float_range(
            0.0, 1.0, 'Grey level (0-1) the contrast curve keeps.'))
        # View selection.
        self.declare_parameter('max_board_speed_rel', 0.1, float_range(
            0.001, 10.0, 'Largest board speed of a kept view, image widths per second.'))
        self.declare_parameter('max_corner_residual', 0.1, float_range(
            0.0, 10.0, 'Largest corner distance from where its neighbours put it, in '
                       'squares.'))
        self.declare_parameter('border_margin_px', 10.0, float_range(
            0.0, 200.0, 'Corners closer than this to the image border are not used.'))
        self.declare_parameter('min_view_distance', 0.2, float_range(
            0.0, 5.0, 'Smallest outline distance to every kept view, in board sizes.'))
        # Background calibration.
        self.declare_parameter('calibrate_every_views', 3, ParameterDescriptor(
            description='Calibrate each time this many new views were kept.',
            integer_range=[IntegerRange(from_value=1, to_value=1000, step=1)]))
        self.declare_parameter('min_calibration_views', 5, ParameterDescriptor(
            description='First calibration at this many views.',
            integer_range=[IntegerRange(from_value=3, to_value=1000, step=1)]))
        self.declare_parameter('free_k3_after_views', 20, ParameterDescriptor(
            description='k3 (fisheye: k3 and k4) stays 0 below this many views.',
            integer_range=[IntegerRange(from_value=0, to_value=10000, step=1)]))
        self.declare_parameter('uncertainty_scale', 2.0, float_range(
            0.1, 100.0, 'Factor on the estimated uncertainties.'))
        # Final calibration pass.
        self.declare_parameter('outlier_z_limit', 6.0, float_range(
            1.0, 100.0, 'Robust z-score of the leave-one-out error that flags a view.'))
        self.declare_parameter('outlier_min_views', 15, ParameterDescriptor(
            description='No outlier check with fewer views.',
            integer_range=[IntegerRange(from_value=3, to_value=10000, step=1)]))
        self.declare_parameter('outlier_max_uncertainty_increase', 1.5, float_range(
            1.0, 1000.0, 'A flagged view is kept if removing it raises the mean uncertainty '
                         'by more than this factor.'))
        self.declare_parameter('outlier_max_fraction', 0.1, float_range(
            0.0, 0.5, 'Largest fraction of the views removed.'))
        self.declare_parameter(
            'output_file', '~/.ros/smart_calibration/calibration_%Y-%m-%d_%H-%M-%S.yaml',
            ParameterDescriptor(
                description='Where Accept saves the calibration; strftime placeholders and ~ '
                            'work.'))
        self.live = {name: self.get_parameter(name).value for name in LIVE_PARAMETERS}

        self.preview_max_width = self.declare_parameter(
            'preview_max_width', 1280, ParameterDescriptor(
                description='Preview images wider than this are downscaled.',
                integer_range=[IntegerRange(from_value=160, to_value=10000, step=1)])).value
        self.preview_jpeg_quality = self.declare_parameter(
            'preview_jpeg_quality', 80, ParameterDescriptor(
                description='JPEG quality of the preview images.',
                integer_range=[IntegerRange(from_value=10, to_value=100, step=1)])).value

        self.bridge = CvBridge()
        self.image_size = None  # (width, height) of the incoming images
        # Guards the views, thumbnails and final pass state below.
        self._lock = threading.Lock()
        self.views = ViewCollection(self.square_size, reference_outline(
            self.board_mode, self.board_cols, self.board_rows))
        self.thumbnails = []  # JPEG bytes per kept view
        self._state = FinalCalibration.STATE_COLLECTING
        self._stage = ''
        self._run = 0  # increases with every final pass and cancellation
        self._final = None  # FinalResult of the current pass
        self._final_views = None  # (image points per view, image size) of the pass

        self.detection_pub = self.create_publisher(BoardDetection, '~/detection', 10)
        self.preview_pub = self.create_publisher(
            CompressedImage, '~/preview', QoSProfile(depth=2))
        self.histogram_pub = self.create_publisher(
            ToneHistogram, '~/histogram', QoSProfile(depth=2))
        # Transient local: a GUI started later still gets them.
        self.calibration_pub = self.create_publisher(
            CalibrationState, '~/calibration',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.final_pub = self.create_publisher(
            FinalCalibration, '~/final_calibration',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_service(Finish, '~/finish', self.on_finish)
        self.create_service(Trigger, '~/resume', self.on_resume)
        self.create_service(Trigger, '~/accept', self.on_accept)
        self.create_service(GetView, '~/get_view', self.on_get_view)

        # Detection runs in a worker thread on the newest frame; frames arriving meanwhile are
        # dropped.
        self._frame_ready = threading.Condition()
        self._latest_msg = None
        self._stopping = False
        self._worker = threading.Thread(target=self._detection_loop, daemon=True)
        self._worker.start()
        # Calibration runs in its own thread, on the newest request only.
        self._calibration_ready = threading.Condition()
        self._calibration_request = None
        self._requested_views = 0  # view count of the last calibration request
        self._calibrator = threading.Thread(target=self._calibration_loop, daemon=True)
        self._calibrator.start()
        self.publish_final()

        image_topic = self.resolve_topic_name('image')
        qos = image_qos(self, image_topic, self.declare_parameter(
            'image_qos', 'auto', ParameterDescriptor(
                description='Camera subscription reliability: auto (match the publisher), '
                            'reliable or best_effort.')).value)
        # After all declarations: it rejects parameters that are not live.
        self.add_on_set_parameters_callback(self.on_set_parameters)
        self.image_sub = self.create_subscription(Image, 'image', self.image_callback, qos)
        if self.board_mode == 'chessboard':
            board = f'a {self.board_cols}x{self.board_rows} chessboard'
        else:
            board = 'an oversized radon checkerboard with markers'
        self.get_logger().info(
            f'waiting for images on {image_topic} ({qos.reliability.name.lower()}), looking for '
            f'{board} ({self.square_size * 1000:g} mm squares), calibrating with the '
            f'{self.camera_model.name} model, publishing detections on '
            f'{self.detection_pub.topic_name}')

    def on_set_parameters(self, parameters):
        live = dict(self.live)
        for parameter in parameters:
            if parameter.name in LIVE_PARAMETERS:
                live[parameter.name] = parameter.value
            elif parameter.name not in ('use_sim_time',):
                return SetParametersResult(
                    successful=False,
                    reason=f'{parameter.name} is only read at startup; restart the node')
        if live['stretch_low_percentile'] >= live['stretch_high_percentile']:
            return SetParametersResult(
                successful=False,
                reason='stretch_low_percentile must be below stretch_high_percentile')
        self.live = live  # replaced as a whole for the worker thread
        return SetParametersResult(successful=True)

    def image_callback(self, msg):
        with self._frame_ready:
            self._latest_msg = msg
            self._frame_ready.notify()

    def _detection_loop(self):
        while True:
            with self._frame_ready:
                while self._latest_msg is None and not self._stopping:
                    self._frame_ready.wait()
                if self._stopping:
                    return
                msg, self._latest_msg = self._latest_msg, None
            try:
                self.process_frame(msg)
            except Exception as error:  # keep the worker alive
                if not self.context.ok():  # shutting down
                    return
                self.get_logger().error(
                    f'frame processing failed: {error!r}', throttle_duration_sec=5.0)

    def process_frame(self, msg):
        image = self.prepare_image(msg)
        if image is None:
            return
        live = self.live
        tone = (live['stretch_low_percentile'], live['stretch_high_percentile'], live['gamma'],
                live['contrast_strength'], live['contrast_midpoint'])
        # Preview and histograms only while the GUI is watching.
        watched = self.preview_pub.get_subscription_count() > 0
        if watched:
            gray, before, after = preprocess_with_histograms(image, *tone)
            histogram = ToneHistogram(header=msg.header, before_contrast=before.tolist(),
                                      after_contrast=after.tolist())
        else:
            gray = preprocess(image, *tone)

        size = (gray.shape[1], gray.shape[0])
        if size != self.image_size:
            if self.image_size is None:
                self.get_logger().info(
                    f'receiving {size[0]}x{size[1]} images, encoding {msg.encoding!r}')
            else:
                self.get_logger().warn(
                    f'image size changed from {self.image_size[0]}x{self.image_size[1]} '
                    f'to {size[0]}x{size[1]}')
            self.image_size = size

        start = time.perf_counter()
        flags = sb_flags(normalize=live['detector_normalize'],
                         exhaustive=live['detector_exhaustive'],
                         accuracy=live['detector_accuracy'])
        board = detect_board(gray, self.board_mode, self.board_cols, self.board_rows, flags,
                             live['border_margin_px'])
        detection_time_ms = (time.perf_counter() - start) * 1000.0

        detection = BoardDetection()
        detection.header = msg.header
        detection.image_width, detection.image_height = size
        detection.found = board is not None
        if board is not None:
            detection.rows, detection.cols = board.shape
            detection.origin_row, detection.origin_col = board.origin
            corners = board.corners.astype(np.float32)
            corners[~board.valid] = np.nan
            detection.corners = corners.ravel().tolist()
        detection.detection_time_ms = detection_time_ms

        assessment = None
        with self._lock:
            if board is not None:
                stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                # After Finish, frames are still checked but not kept.
                assessment = self.views.consider(
                    board, size, stamp, live['max_board_speed_rel'], live['max_corner_residual'],
                    live['min_view_distance'],
                    keep=self._state == FinalCalibration.STATE_COLLECTING)
                if assessment.view is not None:
                    self.thumbnails.append(jpeg(gray, THUMBNAIL_WIDTH, THUMBNAIL_JPEG_QUALITY))
                    self.get_logger().info(f'kept view {len(self.views)}')
                    self.request_calibration(live, msg.header)
            detection.kept_views = len(self.views)
        fill_checks(detection, assessment)

        # Preview first: the GUI pairs both by timestamp.
        if watched:
            self.preview_pub.publish(self.make_preview(gray, msg.header))
            self.histogram_pub.publish(histogram)
        self.detection_pub.publish(detection)

        checks = ''
        if assessment is not None:
            dropped = int((~board.valid).sum())
            checks = (f', moving {format_value(assessment.speed, ".3f")} image widths/s, corner '
                      'residual '
                      f'{format_value(assessment.residual, ".3f")}'
                      + (f' ({dropped} corners at the border not used)' if dropped else '')
                      + f', {format_value(assessment.distance, ".3f")} from the nearest view'
                      + (', kept' if detection.kept else ''))
        self.get_logger().debug(
            f'{size[0]}x{size[1]} frame from {msg.encoding!r}: board '
            f'{"found" if detection.found else "not found"} in {detection_time_ms:.1f} ms'
            + checks)

    def request_calibration(self, live, header, force=False):
        """
        Queue a background calibration if enough new views were kept (call under _lock).

        force: whenever there are enough views.
        """
        count = len(self.views)
        if count < live['min_calibration_views'] or (
                not force and count - self._requested_views < live['calibrate_every_views']):
            return
        self._requested_views = count
        objects, images, size = self.views.calibration_input()
        self._queue_calibration(('background', objects, images, size, header,
                                 live['free_k3_after_views'], live['uncertainty_scale']))

    def _queue_calibration(self, request):
        """Replace any waiting calibration request."""
        with self._calibration_ready:
            self._calibration_request = request
            self._calibration_ready.notify()

    def _calibration_loop(self):
        while True:
            with self._calibration_ready:
                while self._calibration_request is None and not self._stopping:
                    self._calibration_ready.wait()
                if self._stopping:
                    return
                request, self._calibration_request = self._calibration_request, None
            if request[0] == 'final':
                self.run_final(*request[1:])
            else:
                self.run_background(*request[1:])
            if not self.context.ok():
                return

    def run_background(self, objects, images, size, header, free_k3_after_views, scale):
        try:
            result = calibrate_views(objects, images, size, free_k3_after_views, scale,
                                     model=self.camera_model)
        except (cv2.error, np.linalg.LinAlgError) as error:
            self.get_logger().warn(f'calibration from {len(objects)} views failed: {error}')
            return
        if self.context.ok():
            self.calibration_pub.publish(calibration_message(result, header))
            u = result.uncertainty
            self.get_logger().info(
                f'calibrated from {result.views} views in {result.duration_s * 1000:.0f} ms: '
                f'rms {result.rms:.3f} px, uncertainty {u["total"]:.2f} px (focal '
                f'{u["focal"]:.2f}, centre {u["centre"]:.2f}, distortion '
                f'{u["distortion"]:.2f})' + ''.join(f'; {hint}' for hint in result.hints))

    # ------------------------------------------------------------------ final calibration pass

    def run_final(self, run, objects, images, size, keep, live):
        """Run the final pass number run, unless it is cancelled meanwhile."""
        def progress(text):
            with self._lock:
                if self._run != run:
                    raise Cancelled()
                self._stage = text
            self.publish_final()

        start = time.perf_counter()
        try:
            result = final_calibration(
                objects, images, size, live['free_k3_after_views'], live['uncertainty_scale'],
                live['outlier_z_limit'], live['outlier_min_views'],
                live['outlier_max_uncertainty_increase'], live['outlier_max_fraction'], keep,
                progress, self.camera_model)
        except Cancelled:
            return
        except (cv2.error, np.linalg.LinAlgError) as error:
            with self._lock:
                if self._run != run:
                    return
                self._state, self._stage = FinalCalibration.STATE_FAILED, f'failed: {error}'
            self.get_logger().error(f'final calibration failed: {error}')
            self.publish_final()
            return
        with self._lock:
            if self._run != run:
                return
            self._state, self._stage, self._final = FinalCalibration.STATE_DONE, '', result
        self.publish_final()
        final = result.final
        removed = ', '.join(f'#{v.index + 1} (z {v.z:.1f})' for v in result.removed) or 'none'
        suspects = ', '.join(f'#{v.index + 1} (z {v.z:.1f}, without it: '
                             f'{without_text(v.uncertainty_ratio)})' for v in result.suspects)
        self.get_logger().info(
            f'final calibration in {time.perf_counter() - start:.1f} s: {final.views} of '
            f'{len(objects)} views, rms {final.rms:.3f} px, uncertainty '
            f'{final.uncertainty["total"]:.2f} px; '
            + (f'removed: {removed}' if result.checked else
               'too few views for the outlier check (outlier_min_views)')
            + (f'; suspects kept: {suspects}' if suspects else ''))

    def publish_final(self):
        with self._lock:
            msg = FinalCalibration(state=self._state, stage=self._stage, run=self._run)
            msg.header.stamp = self.get_clock().now().to_msg()
            result = self._final if self._state in RESULT_STATES else None
            if self._final_views is not None:
                fill_final(msg, result, *self._final_views, msg.header)
        self.final_pub.publish(msg)

    def on_finish(self, request, response):
        live = self.live
        with self._lock:
            count = len(self.views)
            needed = live['min_calibration_views']
            if self._state == FinalCalibration.STATE_RUNNING:
                response.message = 'the final calibration is already running'
            elif self._state == FinalCalibration.STATE_ACCEPTED:
                response.message = 'the result is accepted; resume collecting first'
            elif count < needed:
                response.message = f'only {count} views kept, need {needed}'
            else:
                response.accepted = True
                response.message = f'final calibration of {count} views started'
                self._state, self._stage = FinalCalibration.STATE_RUNNING, 'starting'
                self._run += 1
                self._final = None
                objects, images, size = self.views.calibration_input()
                self._final_views = (images, size)
                keep = sorted({i for i in request.keep if i < count})
                self._queue_calibration(('final', self._run, objects, images, size, keep, live))
        if response.accepted:
            self.publish_final()
        self.get_logger().info(f'finish: {response.message}')
        return response

    def on_resume(self, request, response):
        with self._lock:
            if self._state == FinalCalibration.STATE_COLLECTING:
                response.message = 'already collecting'
                return response
            # Removed views are dropped.
            removed = [view.index for view in self._final.removed] if (
                self._final is not None and self._state in RESULT_STATES) else []
            self.views.remove(removed)
            self.thumbnails = [t for i, t in enumerate(self.thumbnails) if i not in removed]
            self._state, self._stage = FinalCalibration.STATE_COLLECTING, ''
            self._run += 1  # cancels a pass still running
            self._final, self._final_views = None, None
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            self.request_calibration(self.live, header, force=True)
            response.success = True
            response.message = (f'collecting again with {len(self.views)} views'
                                + (f' ({len(removed)} removed views dropped)' if removed else ''))
        self.publish_final()
        self.get_logger().info(f'resume: {response.message}')
        return response

    def on_accept(self, request, response):
        """Save the final calibration (output_file) and mark it accepted."""
        with self._lock:
            if self._state != FinalCalibration.STATE_DONE:
                response.message = 'no final calibration result to accept'
                return response
            result = self._final
            final = result.final
            now = datetime.now()
            path = output_path(self.live['output_file'], now)
            try:
                write_calibration(path, final.camera_matrix, final.distortion, final.rms,
                                  final.image_size, final.camera_model.name, now)
            except OSError as error:
                response.message = f'not accepted: cannot save the calibration: {error}'
                self.get_logger().error(response.message)
                return response
            self._state = FinalCalibration.STATE_ACCEPTED
        self.publish_final()
        response.success = True
        response.message = f'calibration saved to {path}' + (
            '. Warning: its distortion model is not valid near the image corners'
            if final.distortion_folds else '')
        self.get_logger().info(f'calibration accepted, saved to {path}\n{summary(result)}')
        return response

    def on_get_view(self, request, response):
        with self._lock:
            if request.index >= len(self.views):
                return response
            view = self.views.views[request.index]
            thumbnail = self.thumbnails[request.index]
            result = self._final if self._state in RESULT_STATES else None
        response.found = True
        response.thumbnail = compressed_image(thumbnail)
        response.image_width, response.image_height = view.image_size
        response.image_points = view.image_points.ravel().tolist()
        if result is not None:
            response.residuals = result.view_residuals(request.index).astype(
                np.float32).ravel().tolist()
        return response

    def make_preview(self, gray, header):
        return compressed_image(jpeg(gray, self.preview_max_width, self.preview_jpeg_quality),
                                header)

    def prepare_image(self, msg):
        """Convert an Image message to a single-channel image at its bit depth, or None."""
        try:
            if msg.encoding in SIXTEEN_BIT_ENCODINGS:
                return self.bridge.imgmsg_to_cv2(msg, 'mono16')
            if msg.encoding in FLOAT_MONO_ENCODINGS:
                return self.bridge.imgmsg_to_cv2(msg, 'passthrough')
            return self.bridge.imgmsg_to_cv2(msg, 'mono8')  # also color, Bayer and YUV
        except (CvBridgeError, ValueError) as error:
            self.get_logger().error(
                f'cannot convert image with encoding {msg.encoding!r}: {error}',
                throttle_duration_sec=5.0)
            return None

    def destroy_node(self):
        with self._frame_ready:
            self._stopping = True
            self._frame_ready.notify()
        with self._calibration_ready:
            self._calibration_ready.notify()
        # The calibration thread is a daemon: a long calibration does not hold up the exit.
        self._worker.join(timeout=2.0)
        self._calibrator.join(timeout=0.5)
        super().destroy_node()


def main(args=None):
    error = check_environment()
    if error:
        rclpy.logging.get_logger('calibration_node').fatal(error)
        sys.exit(1)

    rclpy.init(args=args)
    node = CalibrationNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # ros2 launch forwards a second SIGINT.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
