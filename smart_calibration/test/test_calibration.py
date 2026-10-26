import math

import cv2
import numpy as np

from smart_calibration.calibration import (
    axis_offset, board_tilt, calibrate, calibrate_views, distortion_folds_in_image,
    final_calibration, FOLD_HINT, map_grid, pose_hints, summary, uncertainty_hints)

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
    assert flat_result.hints and 'Tilt the board more' in flat_result.hints[0]


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
    assert 'Tilt the board more' in pose_hints([[5, 0], [30, 45]])[0]
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


def bent(image_points, amount):
    """Bow the view's rows by up to amount px in the middle (a flexing board)."""
    points = image_points.astype(np.float64).copy()
    u = (BOARD[:, 0, 0] - BOARD[:, 0, 0].min()) / np.ptp(BOARD[:, 0, 0]) * 2 - 1
    points[:, 0, 1] += amount * (1 - u ** 2)
    return points.astype(np.float32)


def test_leave_one_out_errors_match_recalibrating_without_the_view():
    objects, images = views(VARIED[:10])
    result = calibrate_views(objects, images, SIZE, 20, 1, with_leave_one_out=True)
    assert result.leave_one_out.shape == (10,)
    assert calibrate_views(objects, images, SIZE, 20, 1).leave_one_out is None
    for i in (0, 4):
        rest = [k for k in range(10) if k != i]
        _, K, D, _, _ = calibrate([objects[k] for k in rest], [images[k] for k in rest], SIZE,
                                  True)
        _, rvec, tvec = cv2.solvePnP(objects[i], images[i], K, D)
        projected, _ = cv2.projectPoints(objects[i].astype(np.float64), rvec, tvec, K, D)
        exact = np.sqrt(np.mean(np.sum((images[i] - projected)[:, 0] ** 2, axis=1)))
        assert abs(result.leave_one_out[i] / exact - 1) < 0.05


def final(objects, images, **overrides):
    settings = dict(free_k3_after_views=20, uncertainty_scale=1, z_limit=6, min_views=10,
                    max_uncertainty_increase=1.5, max_removed_fraction=0.1)
    settings.update(overrides)
    return final_calibration(objects, images, SIZE, **settings)


def test_final_calibration_removes_a_bent_view_and_keeps_clean_ones():
    objects, images = views(VARIED)
    clean = final(objects, images)
    assert clean.checked and clean.removed == () and clean.kept == tuple(range(15))
    images[6] = bent(images[6], 3.0)
    stages = []
    result = final(objects, images, progress=stages.append)
    assert [view.index for view in result.removed] == [6] and result.suspects == ()
    assert 6 not in result.kept and len(result.kept) == 14
    flagged = result.removed[0]
    assert flagged.z > 6 and flagged.error > result.initial_limit
    assert flagged.uncertainty_ratio <= 1.5 and flagged.residuals.shape == (54, 2)
    assert np.array_equal(result.view_residuals(6), flagged.residuals)
    assert result.view_residuals(7).shape == (54, 2)
    assert result.final.rms < result.initial.rms
    assert stages[0] == 'calibrating all 15 views' and len(stages) == 2


def test_final_calibration_guards():
    objects, images = views(VARIED)
    images[6] = bent(images[6], 3.0)
    # Too few views for the outlier check: calibrated, nothing removed.
    few = final(objects, images, min_views=16)
    assert not few.checked and few.removed == () and few.kept == tuple(range(15))
    # Restored by the user: never removed.
    assert final(objects, images, forced=[6]).removed == ()
    # Removing it would raise the uncertainty too much: kept as a suspect.
    guarded = final(objects, images, max_uncertainty_increase=0.2)
    assert guarded.removed == () and [view.index for view in guarded.suspects] == [6]
    assert 6 in guarded.kept
    # At most this fraction removed (here: none of 15).
    assert final(objects, images, max_removed_fraction=0.05).removed == ()


def test_uncertainty_hints_name_a_weak_focal_length_or_principal_point():
    def hints(focal, centre, distortion):
        return uncertainty_hints(dict(total=max(focal, centre, distortion), focal=focal,
                                      centre=centre, distortion=distortion))
    assert 'focal length' in hints(3.0, 1.0, 2.0)[0]
    assert 'image centre' in hints(1.0, 3.0, 2.0)[0]
    assert hints(1.0, 2.0, 3.0) == ()
    assert hints(math.nan, math.nan, math.nan) == ()


def test_distortion_folding_inside_the_image():
    # k1 -0.3 alone: turns back at r = 1.05, reaching a distorted radius of 0.70, while the image
    # corners are at 1.15.
    assert distortion_folds_in_image(CAMERA_MATRIX, [-0.3, 0, 0, 0, 0], SIZE)
    assert not distortion_folds_in_image(CAMERA_MATRIX, DISTORTION, SIZE)  # never turns back
    # Turns back, but only beyond the image corners (a narrow image).
    narrow = [[640.0, 0.0, 199.5], [0.0, 640.0, 149.5], [0.0, 0.0, 1.0]]
    assert not distortion_folds_in_image(narrow, [-0.3, 0, 0, 0, 0], (400, 300))
    result = calibrate_views(*views(VARIED), SIZE, 20, 1)
    assert not result.distortion_folds and FOLD_HINT not in result.hints


def test_summary_lists_the_result():
    objects, images = views(VARIED)
    images[6] = bent(images[6], 3.0)
    text = summary(final(objects, images))
    lines = text.splitlines()
    assert lines[0] == '  image        1280 x 720, plumb_bob'
    assert lines[1] == '  views        14 used, 1 removed, k3 fixed'
    assert lines[4].split()[:2] == ['fx', 'fy'] and len(lines) == 8
