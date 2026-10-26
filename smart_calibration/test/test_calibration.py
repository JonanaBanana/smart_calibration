import math

import cv2
import numpy as np

from smart_calibration.calibration import (
    axis_offset, board_tilt, calibrate_views, map_grid, pose_hints)

SIZE = (1280, 720)
CAMERA_MATRIX = np.array([[640.0, 0.0, 639.5], [0.0, 640.0, 359.5], [0.0, 0.0, 1.0]])
DISTORTION = np.array([-0.2, 0.04, 0.0005, -0.0005, 0.0])
# 9 x 6 inner corners, 30 mm squares, centred on the board origin.
BOARD = np.zeros((54, 1, 3), np.float32)
BOARD[:, 0, :2] = (np.mgrid[0:9, 0:6].T.reshape(-1, 2) - [4, 2.5]) * 0.03


def tilt_rotation(tilt_deg, axis_deg):
    """Rotation vector turning the board by tilt_deg about an image-plane axis at axis_deg."""
    axis = np.array([math.cos(math.radians(axis_deg)), math.sin(math.radians(axis_deg)), 0.0])
    return axis * math.radians(tilt_deg)


def views(poses, noise=0.05, seed=0):
    """Object and image points of the board at (tilt, axis, x, y, distance) poses."""
    rng = np.random.default_rng(seed)
    objects, images = [], []
    for tilt, axis, x, y, distance in poses:
        rvec, tvec = tilt_rotation(tilt, axis), np.array([x, y, distance])
        pixels, _ = cv2.projectPoints(BOARD.astype(np.float64), rvec, tvec, CAMERA_MATRIX,
                                      DISTORTION)
        assert ((pixels >= 0) & (pixels < SIZE)).all(), (tilt, axis, x, y, distance)
        objects.append(BOARD)
        images.append((pixels + rng.normal(0.0, noise, pixels.shape)).astype(np.float32))
    return objects, images


# Tilted 20-40 degrees about axes in many directions, at different positions and distances.
VARIED = [(30, a, x, y, d) for a, (x, y, d) in zip(
    range(0, 360, 25), [(-0.2, -0.1, 0.45), (0.0, 0.0, 0.4), (0.2, 0.1, 0.45),
                        (-0.2, 0.1, 0.45), (0.2, -0.1, 0.45), (0.0, -0.1, 0.35),
                        (0.0, 0.1, 0.35), (-0.25, 0.0, 0.5), (0.25, 0.0, 0.5),
                        (0.1, 0.05, 0.3), (-0.1, -0.05, 0.3), (0.0, 0.0, 0.6),
                        (0.15, 0.12, 0.5), (-0.15, -0.12, 0.5), (0.0, 0.0, 0.3)])]


def test_calibration_recovers_the_camera_and_reports_its_uncertainty():
    objects, images = views(VARIED)
    result = calibrate_views(objects, images, SIZE, free_k3_after_views=20, uncertainty_scale=1)
    assert result.views == 15 and result.points == 15 * 54 and result.k3_fixed
    assert np.abs(result.camera_matrix - CAMERA_MATRIX).max() < 2.0
    assert result.rms < 0.1 and abs(result.noise - 0.05) < 0.01
    assert result.std_deviations[8] == 0.0  # k3 fixed
    assert np.all(result.std_deviations[:8] > 0)
    rows, cols = result.uncertainty_map.shape
    assert (rows, cols) == (18, 32)
    total = result.uncertainty['total']
    assert 0 < total == np.nanmax(result.uncertainty_map)
    for group in ('focal', 'centre', 'distortion'):
        assert 0 < result.uncertainty[group] <= total + 1e-9
    assert result.view_rms.shape == (15,) and result.view_tilts.shape == (15, 2)
    np.testing.assert_allclose(result.view_tilts[:, 0], 30.0, atol=1.0)
    assert result.hints == ()


def test_standard_deviations_match_opencv_and_scale():
    objects, images = views(VARIED)
    result = calibrate_views(objects, images, SIZE, free_k3_after_views=0, uncertainty_scale=1)
    assert not result.k3_fixed
    # calibrateCameraExtended's standard deviations are wrong before OpenCV 4.8 (the system
    # OpenCV of ROS Humble is 4.5.4; the venv has 4.11).
    if tuple(int(v) for v in cv2.__version__.split('.')[:2]) >= (4, 8):
        std = cv2.calibrateCameraExtended(objects, images, SIZE, None, None)[5].ravel()[:9]
        np.testing.assert_allclose(result.std_deviations, std, rtol=1e-3)
    doubled = calibrate_views(objects, images, SIZE, free_k3_after_views=0, uncertainty_scale=2)
    np.testing.assert_allclose(doubled.std_deviations, 2 * result.std_deviations, rtol=1e-6)
    assert abs(doubled.uncertainty['total'] - 2 * result.uncertainty['total']) < 1e-6


def test_boards_facing_the_camera_leave_the_focal_length_uncertain():
    flat = [(3, a, x, y, d) for a, x, y, d in (
        (0, -0.2, -0.1, 0.45), (90, 0.2, 0.1, 0.45), (45, -0.2, 0.1, 0.45),
        (135, 0.2, -0.1, 0.45), (0, 0.0, 0.0, 0.35), (90, 0.0, 0.0, 0.6))]
    tilted = VARIED[:6]
    flat_result = calibrate_views(*views(flat), SIZE, 20, 1)
    tilted_result = calibrate_views(*views(tilted), SIZE, 20, 1)
    assert flat_result.uncertainty['focal'] > 3 * tilted_result.uncertainty['focal']
    assert flat_result.hints and 'Tilt the board' in flat_result.hints[0]


def test_board_tilt():
    assert board_tilt(np.zeros(3))[0] == 0.0
    # Turned about the image's vertical axis: left or right edge nearer, tilt axis vertical.
    tilt, axis = board_tilt(tilt_rotation(30, 90))
    assert abs(tilt - 30) < 1e-9 and abs(axis - 90) < 1e-9
    tilt, axis = board_tilt(tilt_rotation(-20, 0))
    assert abs(tilt - 20) < 1e-9 and min(axis, 180 - axis) < 1e-9
    assert abs(board_tilt(tilt_rotation(25, 45))[1] - 45) < 1e-9
    assert axis_offset(3) == 3 and axis_offset(88) == 2 and axis_offset(135) == 45


def test_pose_hints():
    assert 'Tilt the board' in pose_hints([[5, 0], [30, 45]])[0]
    # Only turned about the vertical axis (axis 90): the horizontal tilt is missing; the axes
    # are all aligned with the image axes.
    hints = pose_hints([[30, 90], [25, 85], [30, 95]])
    assert len(hints) == 2 and 'top or bottom' in hints[0] and 'diagonal' in hints[1]
    assert pose_hints([[30, 45], [30, 135], [25, 20]]) == ()


def test_map_grid_divides_the_image_evenly():
    grid = map_grid((1280, 720))
    assert grid.shape == (18, 32, 2)
    np.testing.assert_allclose(grid[0, 0], [19.5, 19.5])
    np.testing.assert_allclose(grid[-1, -1], [1259.5, 699.5])
