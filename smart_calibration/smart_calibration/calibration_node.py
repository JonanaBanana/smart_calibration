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

from smart_calibration.calibration import calibrate_views
from smart_calibration.detection import BOARD_MODES, detect_board, sb_flags
from smart_calibration.environment import check_environment
from smart_calibration.image_preprocessing import FLOAT_MONO_ENCODINGS, SIXTEEN_BIT_ENCODINGS
from smart_calibration.preprocessing import preprocess, preprocess_with_histograms
from smart_calibration.qos import image_qos
from smart_calibration.views import Assessment, reference_outline, ViewCollection
from smart_calibration_interfaces.msg import BoardDetection, CalibrationState, ToneHistogram

# Parameters that can be changed while the node runs (e.g. from the GUI's controls).
LIVE_PARAMETERS = ('gamma', 'stretch_low_percentile', 'stretch_high_percentile',
                   'contrast_strength', 'contrast_midpoint',
                   'detector_normalize', 'detector_exhaustive', 'detector_accuracy',
                   'max_board_speed', 'max_corner_residual', 'border_margin_px',
                   'min_view_distance', 'calibrate_every_views', 'min_calibration_views',
                   'free_k3_after_views', 'uncertainty_scale')


def float_range(low, high, description):
    return ParameterDescriptor(
        description=description,
        floating_point_range=[FloatingPointRange(from_value=low, to_value=high, step=0.0)])


def check_result(value, ok):
    """(check status, value) for a BoardDetection; None means not evaluated / not measurable."""
    status = (BoardDetection.CHECK_NOT_EVALUATED if ok is None
              else BoardDetection.CHECK_PASSED if ok else BoardDetection.CHECK_FAILED)
    return status, math.nan if value is None else value


def fill_checks(detection, assessment):
    """Copy the view selection checks into a BoardDetection (assessment None: no board)."""
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
    """Fill a CalibrationState message from a calibration.CalibrationResult."""
    state = CalibrationState()
    state.header = header
    state.image_width, state.image_height = result.image_size
    state.views, state.points = result.views, result.points
    state.duration_ms = result.duration_s * 1000.0
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
    return state


def format_value(value, spec):
    return '-' if value is None else format(value, spec)


class CalibrationNode(Node):

    def __init__(self):
        super().__init__('calibration_node')
        # Logged so it is obvious which interpreter / OpenCV the node runs on (system vs. venv).
        self.get_logger().info(
            f'python {sys.executable}, OpenCV {cv2.__version__}, numpy {np.__version__}')

        self.board_mode = self.declare_parameter('board_mode', 'chessboard', ParameterDescriptor(
            description='chessboard: plain chessboard of board_cols x board_rows inner corners, '
                        'completely visible. oversized_marker: radon checkerboard with three '
                        'marker dots, larger than the view (see detection.py).')).value
        if self.board_mode not in BOARD_MODES:
            raise ValueError(f'board_mode must be one of {BOARD_MODES}, got {self.board_mode!r}')
        corner_count = ParameterDescriptor(
            description='Inner corners of the chessboard along this axis (board_mode '
                        'chessboard only).',
            integer_range=[IntegerRange(from_value=3, to_value=100, step=1)])
        self.board_cols = self.declare_parameter('board_cols', 9, corner_count).value
        self.board_rows = self.declare_parameter('board_rows', 6, corner_count).value
        self.square_size = self.declare_parameter('square_size', 0.025, float_range(
            0.001, 1.0, 'Side length of one chessboard square in metres (measure the printed '
                        'board). Scales the board poses only, not the intrinsics.')).value
        # findChessboardCornersSB flags (see detection.py and explore.md).
        self.declare_parameter('detector_normalize', False, ParameterDescriptor(
            description='Histogram equalization before detection.'))
        self.declare_parameter('detector_exhaustive', True, ParameterDescriptor(
            description='More exhaustive search: higher detection rate, slower.'))
        self.declare_parameter('detector_accuracy', True, ParameterDescriptor(
            description='Internal 2x upsampling: more accurate corners, ~4x slower.'))

        # Preprocessing (see preprocessing.py).
        self.declare_parameter('gamma', 1.0, float_range(
            0.2, 5.0, 'Gamma applied after the contrast stretch; > 1 brightens dark regions.'))
        self.declare_parameter('stretch_low_percentile', 0.5, float_range(
            0.0, 50.0, 'Intensity percentile mapped to black.'))
        self.declare_parameter('stretch_high_percentile', 99.5, float_range(
            50.0, 100.0, 'Intensity percentile mapped to white.'))
        self.declare_parameter('contrast_strength', 0.0, float_range(
            0.0, 10.0, 'Steepness of the S-shaped contrast curve after gamma (0: off).'))
        self.declare_parameter('contrast_midpoint', 0.5, float_range(
            0.0, 1.0, 'Grey level (0-1) the contrast curve keeps; darker gets darker, lighter '
                      'gets lighter. Best between the dark and light squares\' histogram peaks.'))
        # View selection (see views.py).
        self.declare_parameter('max_board_speed', 100.0, float_range(
            1.0, 10000.0, 'Detections are only kept while the board moves at most this many '
                          'pixels per second (mean corner movement), against motion blur.'))
        self.declare_parameter('max_corner_residual', 0.1, float_range(
            0.0, 10.0, 'Detections are only kept if no corner is further than this (in square '
                       'sizes) from where a quadratic surface through its neighbours puts it: '
                       'rejects frames with a misplaced corner or a deformed region.'))
        self.declare_parameter('border_margin_px', 10.0, float_range(
            0.0, 200.0, 'Corners closer than this to the image border (or outside it) are not '
                        'used: the detector may predict or mismatch corners there.'))
        self.declare_parameter('min_view_distance', 0.2, float_range(
            0.0, 5.0, 'A detection is only kept if its board outline differs from every kept '
                      'view by at least this much: mean distance of the four outline corners, '
                      'in board sizes (mean outline side length), so near and far boards are '
                      'judged alike.'))
        # Background calibration (see calibration.py).
        self.declare_parameter('calibrate_every_views', 3, ParameterDescriptor(
            description='Calibrate in the background each time this many new views were kept.',
            integer_range=[IntegerRange(from_value=1, to_value=1000, step=1)]))
        self.declare_parameter('min_calibration_views', 5, ParameterDescriptor(
            description='First background calibration once this many views were kept.',
            integer_range=[IntegerRange(from_value=3, to_value=1000, step=1)]))
        self.declare_parameter('free_k3_after_views', 20, ParameterDescriptor(
            description='Distortion coefficient k3 stays 0 while fewer views than this are '
                        'used: with few views it makes the distortion run wild where no corners '
                        'are.',
            integer_range=[IntegerRange(from_value=0, to_value=10000, step=1)]))
        self.declare_parameter('uncertainty_scale', 2.0, float_range(
            0.1, 100.0, 'Multiplies the calibration\'s estimated uncertainties: they assume '
                        'independent corner errors and come out ~1.5-2.5x too small on synthetic '
                        'data.'))
        # Live parameter values, read once per frame by the worker.
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
        # Detections kept for calibration (touched by the detection worker only).
        self.views = ViewCollection(self.square_size, reference_outline(
            self.board_mode, self.board_cols, self.board_rows))

        self.detection_pub = self.create_publisher(BoardDetection, '~/detection', 10)
        # The image the detector works on, for display (the GUI does not need the raw stream).
        self.preview_pub = self.create_publisher(
            CompressedImage, '~/preview', QoSProfile(depth=2))
        self.histogram_pub = self.create_publisher(
            ToneHistogram, '~/histogram', QoSProfile(depth=2))
        # Transient local: a GUI started later still gets the newest calibration.
        self.calibration_pub = self.create_publisher(
            CalibrationState, '~/calibration',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

        # Detection is slower than the camera, so it runs in a worker thread on the newest frame
        # only; frames that arrive while it is busy are dropped. The image callback never blocks.
        self._frame_ready = threading.Condition()
        self._latest_msg = None
        self._stopping = False
        self._worker = threading.Thread(target=self._detection_loop, daemon=True)
        self._worker.start()
        # Calibration takes up to seconds with many views, so it runs in its own thread, on the
        # newest request only (OpenCV releases the GIL, detection keeps running meanwhile).
        self._calibration_ready = threading.Condition()
        self._calibration_request = None
        self._requested_views = 0  # view count of the last calibration request
        self._calibrator = threading.Thread(target=self._calibration_loop, daemon=True)
        self._calibrator.start()

        image_topic = self.resolve_topic_name('image')
        qos = image_qos(self, image_topic, self.declare_parameter(
            'image_qos', 'auto', ParameterDescriptor(
                description='Camera subscription reliability: auto (match the publisher), '
                            'reliable or best_effort.')).value)
        # Registered after all declarations: rclpy also runs it for declare_parameter(), and it
        # rejects every parameter that is not live.
        self.add_on_set_parameters_callback(self.on_set_parameters)
        self.image_sub = self.create_subscription(Image, 'image', self.image_callback, qos)
        if self.board_mode == 'chessboard':
            board = f'a {self.board_cols}x{self.board_rows} chessboard'
        else:
            board = 'an oversized radon checkerboard with markers'
        self.get_logger().info(
            f'waiting for images on {image_topic} ({qos.reliability.name.lower()}), looking for '
            f'{board} ({self.square_size * 1000:g} mm squares), publishing detections on '
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
        self.live = live  # replaced as a whole, so the worker never sees a half-updated set
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
            except Exception as error:  # keep the worker alive whatever goes wrong
                if not self.context.ok():
                    # Ctrl+C shuts the ROS context down before destroy_node() stops this thread,
                    # so a detection finishing in between cannot be published any more.
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
        # Preview and histograms only while something (the GUI) is watching.
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
                # Intrinsics are only valid for one resolution.
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
        if board is not None:
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            assessment = self.views.consider(
                board, size, stamp, live['max_board_speed'], live['max_corner_residual'],
                live['min_view_distance'])
            if assessment.view is not None:
                self.get_logger().info(f'kept view {len(self.views)}')
                self.request_calibration(live, msg.header)
        fill_checks(detection, assessment)
        detection.kept_views = len(self.views)

        # Preview first: the GUI pairs both by timestamp and shows a pair once it is complete.
        if watched:
            self.preview_pub.publish(self.make_preview(gray, msg.header))
            self.histogram_pub.publish(histogram)
        self.detection_pub.publish(detection)

        checks = ''
        if assessment is not None:
            dropped = int((~board.valid).sum())
            checks = (f', moving {format_value(assessment.speed, ".0f")} px/s, corner residual '
                      f'{format_value(assessment.residual, ".3f")}'
                      + (f' ({dropped} corners at the border not used)' if dropped else '')
                      + f', {format_value(assessment.distance, ".3f")} from the nearest view'
                      + (', kept' if detection.kept else ''))
        self.get_logger().debug(
            f'{size[0]}x{size[1]} frame from {msg.encoding!r}: board '
            f'{"found" if detection.found else "not found"} in {detection_time_ms:.1f} ms'
            + checks)

    def request_calibration(self, live, header):
        """Hand the kept views to the calibration thread if enough new ones were kept."""
        count = len(self.views)
        if (count < live['min_calibration_views']
                or count - self._requested_views < live['calibrate_every_views']):
            return
        self._requested_views = count
        objects, images, size = self.views.calibration_input()
        with self._calibration_ready:
            self._calibration_request = (objects, images, size, header,
                                         live['free_k3_after_views'], live['uncertainty_scale'])
            self._calibration_ready.notify()

    def _calibration_loop(self):
        while True:
            with self._calibration_ready:
                while self._calibration_request is None and not self._stopping:
                    self._calibration_ready.wait()
                if self._stopping:
                    return
                request, self._calibration_request = self._calibration_request, None
            objects, images, size, header, free_k3_after_views, scale = request
            try:
                result = calibrate_views(objects, images, size, free_k3_after_views, scale)
            except (cv2.error, np.linalg.LinAlgError) as error:
                self.get_logger().warn(f'calibration from {len(objects)} views failed: {error}')
                continue
            if not self.context.ok():
                return
            self.calibration_pub.publish(calibration_message(result, header))
            u = result.uncertainty
            self.get_logger().info(
                f'calibrated from {result.views} views in {result.duration_s * 1000:.0f} ms: '
                f'rms {result.rms:.3f} px, uncertainty {u["total"]:.2f} px (focal '
                f'{u["focal"]:.2f}, centre {u["centre"]:.2f}, distortion '
                f'{u["distortion"]:.2f})' + ''.join(f'; {hint}' for hint in result.hints))

    def make_preview(self, gray, header):
        if gray.shape[1] > self.preview_max_width:
            scale = self.preview_max_width / gray.shape[1]
            gray = cv2.resize(gray, (self.preview_max_width, round(gray.shape[0] * scale)),
                              interpolation=cv2.INTER_AREA)
        preview = CompressedImage()
        preview.header = header
        preview.format = 'mono8; jpeg compressed mono8'  # image_transport's naming
        preview.data = cv2.imencode(
            '.jpg', gray, [cv2.IMWRITE_JPEG_QUALITY, self.preview_jpeg_quality])[1].tobytes()
        return preview

    def prepare_image(self, msg):
        """Convert an Image message to a single-channel image at its bit depth, None on failure."""
        try:
            if msg.encoding in SIXTEEN_BIT_ENCODINGS:
                return self.bridge.imgmsg_to_cv2(msg, 'mono16')
            if msg.encoding in FLOAT_MONO_ENCODINGS:
                return self.bridge.imgmsg_to_cv2(msg, 'passthrough')
            # cv_bridge handles 8-bit color, Bayer and YUV encodings.
            return self.bridge.imgmsg_to_cv2(msg, 'mono8')
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
        # Bounded wait: a detection in progress finishes within a fraction of a second. A
        # calibration may take longer; its thread is a daemon and does not hold up the exit.
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
        # On Ctrl+C under ros2 launch the node gets SIGINT twice (terminal + launch forwarding);
        # ignore the second one so it cannot interrupt the cleanup below.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
