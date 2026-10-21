import sys

import rclpy

from smart_calibration.environment import check_environment


def main(args=None):
    error = check_environment()
    if error:
        rclpy.logging.get_logger('calibration_gui').fatal(error)
        sys.exit(1)

    # Imported only after the check: PySide6 exists only in the package venv.
    from smart_calibration.gui.app import run
    run(args)


if __name__ == '__main__':
    main()
