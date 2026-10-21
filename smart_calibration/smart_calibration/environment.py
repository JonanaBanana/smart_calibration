"""Check that a node runs on the package venv (see README.md)."""

import sys

import cv2

# Oldest OpenCV with the fixes this package relies on (see explore.md §5): calibration std-devs,
# fisheye focal-length initialization, fisheye.solvePnP.
MIN_OPENCV_VERSION = (4, 11)


def check_environment():
    """Return an error message if not running on the package venv, else None."""
    if sys.prefix == sys.base_prefix:
        return (f'not running inside a Python venv (python: {sys.executable}). '
                'Activate the smart_calibration venv before running (see README.md).')
    opencv_version = tuple(int(part) for part in cv2.__version__.split('.')[:2])
    if opencv_version < MIN_OPENCV_VERSION:
        return (f'OpenCV {cv2.__version__} is too old (need >= '
                f'{".".join(map(str, MIN_OPENCV_VERSION))}). '
                'Install requirements.txt into the venv (see README.md).')
    return None
