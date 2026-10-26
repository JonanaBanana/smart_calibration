import sys

import rclpy

from smart_calibration.environment import check_environment


def main(args=None):
    error = check_environment()
    if error:
        rclpy.logging.get_logger('calibration_gui').fatal(error)
        sys.exit(1)

    # PySide6 is only installed in the venv.
    from smart_calibration.gui.app import run
    run(args)


if __name__ == '__main__':
    main()
