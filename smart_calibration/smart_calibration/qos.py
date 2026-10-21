"""QoS selection for camera image subscriptions."""

import time

from rclpy.qos import QoSProfile, ReliabilityPolicy

IMAGE_QOS_MODES = ('auto', 'reliable', 'best_effort')


def wait_for_publishers(node, topic, timeout_s):
    """Return the publisher infos of topic, waiting up to timeout_s for discovery to find any."""
    deadline = time.monotonic() + timeout_s
    while True:
        publishers = node.get_publishers_info_by_topic(topic)
        if publishers or time.monotonic() >= deadline:
            return publishers
        time.sleep(0.1)


def image_qos(node, topic, mode='auto', discovery_timeout_s=2.0):
    """
    Return a QoS profile for subscribing to the camera images on topic.

    Best effort loses many large images in transport (each image is split into many fragments,
    and losing one drops the whole image: ~16 of 30 fps arrived in a 1280x720 mono8 test), but a
    reliable subscription cannot connect to a best-effort publisher. 'auto' therefore matches the
    reliability of the topic's current publishers, like ROS camera_calibration does.
    """
    if mode not in IMAGE_QOS_MODES:
        raise ValueError(f'image_qos must be one of {IMAGE_QOS_MODES}, got {mode!r}')

    reliability = {'reliable': ReliabilityPolicy.RELIABLE,
                   'best_effort': ReliabilityPolicy.BEST_EFFORT}.get(mode)
    if reliability is None:
        # Right after the node starts, discovery may not have found the publishers yet.
        publishers = wait_for_publishers(node, topic, discovery_timeout_s)
        if not publishers:
            reliability = ReliabilityPolicy.BEST_EFFORT  # connects to any publisher
            node.get_logger().warn(
                f'no publisher on {topic} yet: subscribing best effort, which drops many large '
                'images if the camera turns out to publish reliably. Start the camera first or '
                'set image_qos.')
        elif all(p.qos_profile.reliability == ReliabilityPolicy.RELIABLE for p in publishers):
            reliability = ReliabilityPolicy.RELIABLE
        else:
            reliability = ReliabilityPolicy.BEST_EFFORT

    # Shallow queue: only the newest frames matter.
    return QoSProfile(depth=2, reliability=reliability)
