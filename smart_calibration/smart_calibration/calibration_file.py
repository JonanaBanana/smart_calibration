"""
Calibration files in OpenCV's YAML format (cv2.FileStorage).

    %YAML:1.0
    ---
    M: !!opencv-matrix        camera matrix, 3 x 3
    D: !!opencv-matrix        distortion coefficients, n x 1 (plumb_bob: k1, k2, p1, p2, k3;
                              fisheye: k1, k2, k3, k4)
    reprojection_error: ...   RMS over all corners, px
    width: ...
    height: ...
    distortion_model: plumb_bob or fisheye
    calibration_time: "YYYY-MM-DD hh:mm:ss"
"""

import os
from pathlib import Path

import cv2
import numpy as np


def output_path(pattern, when):
    """Expand strftime placeholders (e.g. %Y-%m-%d) and ~ in a path pattern."""
    return Path(os.path.expanduser(when.strftime(pattern)))


def write_calibration(path, camera_matrix, distortion, reprojection_error, image_size,
                      distortion_model, when):
    """Write a calibration file, creating its folder; raise OSError on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    if not storage.isOpened():
        raise OSError(f'cannot write {path}')
    try:
        storage.write('M', np.asarray(camera_matrix, np.float64).reshape(3, 3))
        storage.write('D', np.asarray(distortion, np.float64).reshape(-1, 1))
        storage.write('reprojection_error', float(reprojection_error))
        storage.write('width', int(image_size[0]))
        storage.write('height', int(image_size[1]))
        storage.write('distortion_model', distortion_model)
        storage.write('calibration_time', when.strftime('%Y-%m-%d %H:%M:%S'))
    finally:
        storage.release()


def read_calibration(path):
    """Read a calibration file into a dict with the keys above."""
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise OSError(f'cannot read {path}')
    try:
        return dict(
            M=storage.getNode('M').mat(), D=storage.getNode('D').mat(),
            reprojection_error=storage.getNode('reprojection_error').real(),
            width=int(storage.getNode('width').real()),
            height=int(storage.getNode('height').real()),
            distortion_model=storage.getNode('distortion_model').string(),
            calibration_time=storage.getNode('calibration_time').string())
    finally:
        storage.release()
