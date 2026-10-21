"""
ROS side of the calibration GUI, used from the Qt thread through thread-safe methods.

Receives the calibration node's previews and detections (and, on request, the raw camera
stream), and reads / sets the calibration node's live parameters.
"""

import collections
import threading
import time

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor, ParameterEvent
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.node import Node
from rclpy.parameter import Parameter, parameter_value_to_python
from rclpy.qos import qos_profile_parameter_events, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image

from smart_calibration.image_preprocessing import (
    FLOAT_MONO_ENCODINGS, SIXTEEN_BIT_ENCODINGS, to_mono8)
from smart_calibration.qos import image_qos
from smart_calibration_interfaces.msg import BoardDetection, ToneHistogram

# Buffers for pairing images with detections by timestamp; they must cover the detection
# latency (a detection arrives after its preview / raw frame).
PREVIEW_BUFFER_SIZE = 10
RAW_BUFFER_SIZE = 30
DETECTION_BUFFER_SIZE = 30

Frame = collections.namedtuple('Frame', ['stamp_ns', 'image', 'encoding'])

Snapshot = collections.namedtuple('Snapshot', [
    'detection',          # newest BoardDetection, or None
    'detection_age_s',    # seconds since it arrived, or None
    'preview',            # newest preview Frame whose detection has arrived, or None
    'preview_detection',  # the BoardDetection of that preview
    'raw_frame',          # newest raw camera Frame whose detection has arrived, or None
    'raw_detection',      # the BoardDetection of that raw frame
    'newest_raw',         # newest raw camera Frame, or None
    'parameters',         # {name: value} of the calibration node, as far as known
    'histogram',          # newest ToneHistogram, or None
    'preview_fps',
    'detection_fps',
    'raw_fps',
])


def stamp_to_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def to_display_image(bridge, msg):
    """Convert an Image message to uint8 for display: (h, w) gray or (h, w, 3) RGB."""
    if msg.encoding in SIXTEEN_BIT_ENCODINGS:
        return to_mono8(bridge.imgmsg_to_cv2(msg, 'mono16'))
    if msg.encoding in FLOAT_MONO_ENCODINGS:
        return to_mono8(bridge.imgmsg_to_cv2(msg, 'passthrough'))
    if msg.encoding in ('mono8', '8UC1'):
        return bridge.imgmsg_to_cv2(msg, 'mono8')
    # cv_bridge handles 8-bit color, Bayer and YUV encodings.
    return bridge.imgmsg_to_cv2(msg, 'rgb8')


def newest_pair(frames, detections):
    """Return the newest frame that has a detection with the same stamp, and that detection."""
    for frame in reversed(frames):
        detection = detections.get(frame.stamp_ns)
        if detection is not None:
            return frame, detection
    return None, None


class RateMeter:
    """Events per second over a sliding time window."""

    def __init__(self, window_s=2.0):
        self.window_s = window_s
        self._times = collections.deque()

    def tick(self, now):
        self._times.append(now)
        self._drop_old(now)

    def rate(self, now):
        self._drop_old(now)
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])

    def _drop_old(self, now):
        while self._times and now - self._times[0] > self.window_s:
            self._times.popleft()


class GuiNode(Node):

    def __init__(self):
        super().__init__('calibration_gui')
        self.bridge = CvBridge()
        self._lock = threading.Lock()

        self.target = self.declare_parameter('calibration_node', '/calibration_node', (
            ParameterDescriptor(description='Fully qualified name of the calibration node.')
        )).value
        self.image_qos_mode = self.declare_parameter('image_qos', 'auto', ParameterDescriptor(
            description='Raw camera subscription reliability: auto (match the publisher), '
                        'reliable or best_effort.')).value
        self.raw_topic = self.resolve_topic_name('image')

        self._previews = collections.deque(maxlen=PREVIEW_BUFFER_SIZE)
        self._raw_frames = collections.deque(maxlen=RAW_BUFFER_SIZE)
        self._detections = collections.OrderedDict()  # stamp_ns -> BoardDetection
        self._newest_detection = None
        self._newest_detection_received = None
        self._preview_rate = RateMeter()
        self._detection_rate = RateMeter()
        self._raw_rate = RateMeter()

        # Best effort: newest previews matter, and it connects to any publisher.
        self.preview_sub = self.create_subscription(
            CompressedImage, f'{self.target}/preview', self.preview_callback,
            QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.detection_sub = self.create_subscription(
            BoardDetection, f'{self.target}/detection', self.detection_callback, 10)
        self._histogram = None
        self.histogram_sub = self.create_subscription(
            ToneHistogram, f'{self.target}/histogram', self.histogram_callback,
            QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT))

        # Raw camera stream: subscribed only while the raw view is switched on.
        self._raw_wanted = False
        self._raw_sub = None

        # The calibration node's live parameters: fetched once it is reachable, then kept in
        # sync through parameter events (also covers changes made with `ros2 param set`).
        self.parameters = {}
        self._parameters_requested = False
        self._get_client = self.create_client(GetParameters, f'{self.target}/get_parameters')
        self._set_client = self.create_client(SetParameters, f'{self.target}/set_parameters')
        self.create_subscription(ParameterEvent, '/parameter_events',
                                 self.parameter_event_callback, qos_profile_parameter_events)

        self.get_logger().info(
            f'previews from {self.preview_sub.topic_name}, detections from '
            f'{self.detection_sub.topic_name}, raw camera (on request) from {self.raw_topic}')

    # ------------------------------------------------------------------ images and detections

    def preview_callback(self, msg):
        image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            self.get_logger().error('cannot decode preview image', throttle_duration_sec=5.0)
            return
        now = time.monotonic()
        with self._lock:
            self._previews.append(Frame(stamp_to_ns(msg.header.stamp), image, 'preview'))
            self._preview_rate.tick(now)

    def detection_callback(self, msg):
        now = time.monotonic()
        with self._lock:
            self._detections[stamp_to_ns(msg.header.stamp)] = msg
            while len(self._detections) > DETECTION_BUFFER_SIZE:
                self._detections.popitem(last=False)
            self._newest_detection = msg
            self._newest_detection_received = now
            self._detection_rate.tick(now)

    def histogram_callback(self, msg):
        with self._lock:
            self._histogram = msg

    def raw_callback(self, msg):
        try:
            image = to_display_image(self.bridge, msg)
        except (CvBridgeError, ValueError) as error:
            self.get_logger().error(
                f'cannot display image with encoding {msg.encoding!r}: {error}',
                throttle_duration_sec=5.0)
            return
        now = time.monotonic()
        with self._lock:
            self._raw_frames.append(Frame(stamp_to_ns(msg.header.stamp), image, msg.encoding))
            self._raw_rate.tick(now)

    def set_raw_enabled(self, enabled):
        """Subscribe to or unsubscribe from the raw camera stream."""
        with self._lock:
            if enabled == self._raw_wanted:
                return
            self._raw_wanted = enabled
            sub, self._raw_sub = (None, self._raw_sub) if enabled else (self._raw_sub, None)
            if not enabled:
                self._raw_frames.clear()
        if enabled:
            # Choosing the QoS may wait up to 2 s for discovery; keep that off the Qt thread.
            threading.Thread(target=self._subscribe_raw, daemon=True).start()
        elif sub is not None:
            self.destroy_subscription(sub)

    def _subscribe_raw(self):
        qos = image_qos(self, self.raw_topic, self.image_qos_mode)
        sub = self.create_subscription(Image, self.raw_topic, self.raw_callback, qos)
        with self._lock:
            if self._raw_wanted and self._raw_sub is None:
                self._raw_sub, sub = sub, None
        if sub is not None:  # switched off again in the meantime
            self.destroy_subscription(sub)

    def snapshot(self):
        """Return the newest data for the Qt thread."""
        now = time.monotonic()
        with self._lock:
            preview, preview_detection = newest_pair(self._previews, self._detections)
            raw_frame, raw_detection = newest_pair(self._raw_frames, self._detections)
            age = None
            if self._newest_detection_received is not None:
                age = now - self._newest_detection_received
            return Snapshot(
                self._newest_detection, age, preview, preview_detection, raw_frame,
                raw_detection, self._raw_frames[-1] if self._raw_frames else None,
                dict(self.parameters), self._histogram, self._preview_rate.rate(now),
                self._detection_rate.rate(now), self._raw_rate.rate(now))

    # ------------------------------------------------------------------ parameters

    def poll_parameters(self, names):
        """Fetch the calibration node's parameters once it is reachable; call periodically."""
        with self._lock:
            if self._parameters_requested or not self._get_client.service_is_ready():
                return
            self._parameters_requested = True
        future = self._get_client.call_async(GetParameters.Request(names=list(names)))
        future.add_done_callback(lambda done: self._store_parameters(names, done))

    def _store_parameters(self, names, future):
        response = future.result()
        with self._lock:
            if response is None or len(response.values) != len(names):
                self._parameters_requested = False  # try again on the next poll
                return
            for name, value in zip(names, response.values):
                self.parameters[name] = parameter_value_to_python(value)

    def set_parameter(self, name, value):
        """Ask the calibration node to change a parameter; returns False if it is unreachable."""
        if not self._set_client.service_is_ready():
            return False
        request = SetParameters.Request(
            parameters=[Parameter(name, value=value).to_parameter_msg()])
        future = self._set_client.call_async(request)
        future.add_done_callback(lambda done: self._check_set_result(name, done))
        return True

    def _check_set_result(self, name, future):
        response = future.result()
        if response is not None and response.results and response.results[0].successful:
            return
        reason = response.results[0].reason if response is not None and response.results else ''
        self.get_logger().warn(f'calibration node rejected {name}: {reason}')
        with self._lock:
            self._parameters_requested = False  # re-read the actual value

    def parameter_event_callback(self, msg):
        if msg.node != self.target:
            return
        with self._lock:
            for parameter in list(msg.new_parameters) + list(msg.changed_parameters):
                self.parameters[parameter.name] = parameter_value_to_python(parameter.value)
