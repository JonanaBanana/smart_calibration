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
    # Signals are handled below, so that they also stop Qt.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = GuiNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    app = QApplication(sys.argv[:1])
    window = MainWindow(node)
    window.resize(1280, 800)
    window.show()

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: app.quit())
    try:
        app.exec()
    finally:
        # ros2 launch forwards a second SIGINT.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
        spin_thread.join(timeout=2.0)
