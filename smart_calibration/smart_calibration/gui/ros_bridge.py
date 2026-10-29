"""ROS side of the calibration GUI; its methods are called from the Qt thread."""

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
from rclpy.qos import (
    DurabilityPolicy, qos_profile_parameter_events, QoSProfile, ReliabilityPolicy)
from sensor_msgs.msg import CompressedImage, Image
from std_srvs.srv import Trigger

from smart_calibration.preprocessing import (
    FLOAT_MONO_ENCODINGS, SIXTEEN_BIT_ENCODINGS, to_mono8)
from smart_calibration.qos import image_qos
from smart_calibration_interfaces.msg import (
    BoardDetection, CalibrationState, FinalCalibration, ToneHistogram)
from smart_calibration_interfaces.srv import Finish, GetView

# Buffers for pairing images with detections by timestamp; they cover the detection latency.
PREVIEW_BUFFER_SIZE = 10
RAW_BUFFER_SIZE = 30
DETECTION_BUFFER_SIZE = 30
# Polled requests without a reply are sent again after this: the node can drop a reply to a
# client it has not fully discovered yet ("failed to send response (timeout)").
REPLY_TIMEOUT_S = 2.0

Frame = collections.namedtuple('Frame', ['stamp_ns', 'image', 'encoding'])

# A kept view from GetView; points and residuals (or None) in full image pixels.
KeptView = collections.namedtuple('KeptView', ['thumbnail', 'image_width', 'points',
                                               'residuals'])

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
    'calibration',        # newest CalibrationState, or None
    'final',              # newest FinalCalibration, or None
    'views',              # {(run, index): KeptView} fetched so far
    'service_message',    # reply to the last finish / resume / accept call, or None
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
    return bridge.imgmsg_to_cv2(msg, 'rgb8')  # also Bayer and YUV


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

        self.preview_sub = self.create_subscription(
            CompressedImage, f'{self.target}/preview', self.preview_callback,
            QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.detection_sub = self.create_subscription(
            BoardDetection, f'{self.target}/detection', self.detection_callback, 10)
        self._histogram = None
        self.histogram_sub = self.create_subscription(
            ToneHistogram, f'{self.target}/histogram', self.histogram_callback,
            QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT))
        # Transient local: also the calibration published before the GUI started.
        self._calibration = None
        self.calibration_sub = self.create_subscription(
            CalibrationState, f'{self.target}/calibration', self.calibration_callback,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

        # Final calibration pass.
        self._final = None
        self.final_sub = self.create_subscription(
            FinalCalibration, f'{self.target}/final_calibration', self.final_callback,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self._finish_client = self.create_client(Finish, f'{self.target}/finish')
        self._resume_client = self.create_client(Trigger, f'{self.target}/resume')
        self._accept_client = self.create_client(Trigger, f'{self.target}/accept')
        self._view_client = self.create_client(GetView, f'{self.target}/get_view')
        self._views = {}  # (run, index) -> KeptView, or None while requested
        self._view_requests = {}  # (run, index) -> time.monotonic() of the request
        self._service_message = None

        # Raw camera stream, subscribed only while shown.
        self._raw_wanted = False
        self._raw_sub = None

        # The calibration node's parameters: fetched once, then updated from parameter events.
        self.parameters = {}
        self._have_parameters = False
        self._parameters_requested = None  # time.monotonic() of the pending request
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

    def calibration_callback(self, msg):
        with self._lock:
            self._calibration = msg

    def final_callback(self, msg):
        with self._lock:
            if self._final is not None and msg.run != self._final.run:
                self._views.clear()  # they belong to the previous pass
                self._view_requests.clear()
            self._final = msg

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
            # Choosing the QoS waits for discovery: not on the Qt thread.
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
                dict(self.parameters), self._histogram, self._calibration, self._final,
                {key: view for key, view in self._views.items() if view is not None},
                self._service_message,
                self._preview_rate.rate(now),
                self._detection_rate.rate(now), self._raw_rate.rate(now))

    # ------------------------------------------------------------------ parameters

    def poll_parameters(self, names):
        """Fetch the calibration node's parameters once it is reachable; call regularly."""
        now = time.monotonic()
        with self._lock:
            if self._have_parameters or not self._get_client.service_is_ready():
                return
            sent = self._parameters_requested
            if sent is not None and now - sent < REPLY_TIMEOUT_S:
                return
            self._parameters_requested = now
        future = self._get_client.call_async(GetParameters.Request(names=list(names)))
        future.add_done_callback(lambda done: self._store_parameters(names, done))

    def _store_parameters(self, names, future):
        response = future.result()
        with self._lock:
            if response is None or len(response.values) != len(names):
                self._parameters_requested = None  # try again on the next poll
                return
            for name, value in zip(names, response.values):
                self.parameters[name] = parameter_value_to_python(value)
            self._have_parameters = True

    def set_parameter(self, name, value):
        """Ask the calibration node to change a parameter; False if it is unreachable."""
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
            self._have_parameters = False  # re-read the actual value
            self._parameters_requested = None

    def parameter_event_callback(self, msg):
        if msg.node != self.target:
            return
        with self._lock:
            for parameter in list(msg.new_parameters) + list(msg.changed_parameters):
                self.parameters[parameter.name] = parameter_value_to_python(parameter.value)

    # ------------------------------------------------------------------ final calibration pass

    def _call(self, client, request, name):
        """Call a service and keep its reply message; False if unreachable."""
        if not client.service_is_ready():
            with self._lock:
                self._service_message = f'{name}: calibration node not reachable'
            return False
        future = client.call_async(request)
        future.add_done_callback(lambda done: self._store_reply(name, done))
        return True

    def _store_reply(self, name, future):
        response = future.result()
        text = 'no reply' if response is None else response.message
        with self._lock:
            self._service_message = f'{name}: {text}'

    def finish_available(self):
        """Return whether the finish service is reachable."""
        return self._finish_client.service_is_ready()

    def finish(self, keep=()):
        """Stop collecting and run the final pass; keep: view indices it must not remove."""
        return self._call(self._finish_client, Finish.Request(keep=list(keep)), 'finish')

    def resume(self):
        return self._call(self._resume_client, Trigger.Request(), 'resume')

    def accept(self):
        return self._call(self._accept_client, Trigger.Request(), 'accept')

    def request_view(self, run, index):
        """Fetch a kept view of final pass run once; it appears in snapshot().views."""
        key, now = (run, index), time.monotonic()
        with self._lock:
            if not self._view_client.service_is_ready():
                return
            if key in self._views and (self._views[key] is not None or
                                       now - self._view_requests[key] < REPLY_TIMEOUT_S):
                return
            self._views[key] = None
            self._view_requests[key] = now
        future = self._view_client.call_async(GetView.Request(index=index))
        future.add_done_callback(lambda done: self._store_view(run, index, done))

    def _store_view(self, run, index, future):
        response = future.result()
        view = None
        if response is not None and response.found:
            thumbnail = cv2.imdecode(np.frombuffer(response.thumbnail.data, np.uint8),
                                     cv2.IMREAD_GRAYSCALE)
            points = np.asarray(response.image_points, np.float32).reshape(-1, 2)
            residuals = (np.asarray(response.residuals, np.float32).reshape(-1, 2)
                         if len(response.residuals) else None)
            view = KeptView(thumbnail, response.image_width, points, residuals)
        with self._lock:
            if view is None:
                self._views.pop((run, index), None)  # try again on the next request
            elif self._final is not None and self._final.run == run:
                self._views[(run, index)] = view
