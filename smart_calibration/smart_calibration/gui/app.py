"""Runs the calibration GUI: ROS in a background thread, Qt on the main thread."""

import signal
import sys
import threading

from PySide6.QtWidgets import QApplication
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.signals import SignalHandlerOptions

from smart_calibration.gui.main_window import MainWindow
from smart_calibration.gui.ros_bridge import GuiNode


def run(args=None):
    # rclpy's own SIGINT handler would only shut down ROS; the Qt event loop must stop too.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = GuiNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # Only the program name: ROS arguments (--ros-args ...) are not meant for Qt.
    app = QApplication(sys.argv[:1])
    window = MainWindow(node)
    window.resize(1280, 800)
    window.show()

    # Python runs signal handlers only when Qt calls back into Python, which the window's
    # refresh timer does regularly.
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: app.quit())
    try:
        app.exec()
    finally:
        # Under ros2 launch, Ctrl+C delivers SIGINT twice; keep the second one from interrupting
        # the cleanup below.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
        spin_thread.join(timeout=2.0)
