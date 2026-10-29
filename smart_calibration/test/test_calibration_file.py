from datetime import datetime
import os
from pathlib import Path

import numpy as np

from smart_calibration.calibration_file import output_path, read_calibration, write_calibration

WHEN = datetime(2026, 10, 26, 14, 3, 12)


def test_calibration_file_is_opencv_yaml_and_reads_back(tmp_path):
    camera_matrix = np.array([[640.54, 0, 639.29], [0, 640.37, 358.66], [0, 0, 1]])
    distortion = np.array([-0.1994, 0.0395, 0.0003, -0.00059, 0.0011])
    path = tmp_path / 'new_folder' / 'calibration.yaml'
    write_calibration(path, camera_matrix, distortion, 0.1234, (1280, 720), 'plumb_bob', WHEN)
    text = path.read_text()
    assert text.startswith('%YAML:1.0\n---\nM: !!opencv-matrix')
    assert 'D: !!opencv-matrix\n   rows: 5\n   cols: 1\n   dt: d' in text
    assert 'calibration_time: "2026-10-26 14:03:12"' in text
    read = read_calibration(path)
    np.testing.assert_array_equal(read['M'], camera_matrix)
    np.testing.assert_array_equal(read['D'].ravel(), distortion)
    assert read['reprojection_error'] == 0.1234
    assert (read['width'], read['height']) == (1280, 720)
    assert read['distortion_model'] == 'plumb_bob'
    assert read['calibration_time'] == '2026-10-26 14:03:12'


def test_fisheye_calibration_file(tmp_path):
    path = tmp_path / 'fisheye.yaml'
    distortion = np.array([0.05, -0.01, 0.002, -0.0003])
    write_calibration(path, np.eye(3), distortion, 0.2, (1280, 720), 'fisheye', WHEN)
    read = read_calibration(path)
    assert read['D'].shape == (4, 1) and read['distortion_model'] == 'fisheye'
    np.testing.assert_array_equal(read['D'].ravel(), distortion)


def test_output_path_fills_in_the_time_and_home_folder():
    path = output_path('~/calibration/cam_%Y-%m-%d_%H-%M-%S.yaml', WHEN)
    assert path == Path(os.path.expanduser('~'), 'calibration/cam_2026-10-26_14-03-12.yaml')
