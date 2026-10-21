import cv2
import numpy as np

from smart_calibration.detection import detect_chessboard
from smart_calibration.views import (
    board_object_points, board_outline, board_speed, grid_irregularity, outline_distance,
    ViewCollection)

from test_detection import render_board

MAX_SPEED = 100.0
MAX_IRREGULARITY = 0.3
MIN_DISTANCE = 0.05
# A flat 9 x 6 grid of corners with 30 px squares.
GRID = (100.0 + 30.0 * np.mgrid[0:9, 0:6].T).astype(np.float32)


def consider(views, corners, image_size, stamp, min_distance=MIN_DISTANCE):
    return views.consider(corners, image_size, stamp, MAX_SPEED, MAX_IRREGULARITY, min_distance)


def consider_still(views, corners, image_size, stamp, min_distance=MIN_DISTANCE):
    """Present a board twice, 0.1 s apart and unmoved; return the assessment of the second one."""
    consider(views, corners, image_size, stamp - 0.1, min_distance)
    return consider(views, corners, image_size, stamp, min_distance)


def test_object_points_are_row_major_on_the_board_plane():
    points = board_object_points(4, 3, 0.03)
    assert points.shape == (12, 1, 3) and points.dtype == np.float32
    np.testing.assert_allclose(points[[0, 1, 4, 11], 0], [
        [0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.0, 0.03, 0.0], [0.09, 0.06, 0.0]])


def test_object_points_are_in_the_order_of_the_detected_corners():
    corners = detect_chessboard(render_board(9, 6), 9, 6)
    board = board_object_points(9, 6, 1.0)[:, 0, :2]
    # Same order: one homography maps every board point onto its detected corner.
    homography, _ = cv2.findHomography(board, corners.reshape(-1, 2))
    mapped = cv2.perspectiveTransform(board[:, None], homography)[:, 0]
    assert np.abs(mapped - corners.reshape(-1, 2)).max() < 0.1


def test_only_views_different_from_the_kept_ones_are_kept():
    views = ViewCollection(9, 6, 0.03)
    corners = detect_chessboard(render_board(9, 6), 9, 6)
    size = (440, 320)  # diagonal ~544 px: MIN_DISTANCE ~27 px
    first = consider_still(views, corners, size, 1.0)
    assert first.view is not None and first.distance is None and first.distance_ok
    close = consider_still(views, corners + 10.0, size, 2.0)  # 14 px away: too close
    assert close.view is None and not close.distance_ok and 0.0 < close.distance < MIN_DISTANCE
    assert consider_still(views, corners + 30.0, size, 3.0).view is not None  # 42 px away
    assert consider_still(views, corners + 20.0, size, 4.0).view is None  # 14 px from the second
    assert len(views) == 2 and views.image_size == size


def test_all_checks_are_evaluated_even_if_one_fails():
    views = ViewCollection(9, 6, 0.03)
    consider_still(views, GRID, (640, 480), 1.0)
    bad = GRID + 50.0
    bad[2, 4] += [0.0, 6.0]  # misplaced corner
    consider(views, bad, (640, 480), 2.0)
    moving = consider(views, bad + 30.0, (640, 480), 2.1)  # 424 px/s
    assert moving.speed_ok is False and moving.irregularity_ok is False
    assert moving.distance is not None and moving.distance_ok and moving.view is None


def test_outline_is_the_four_outermost_corners():
    points = np.arange(6 * 4 * 2, dtype=np.float32).reshape(-1, 1, 2)  # 4 cols, 6 rows
    grid = points.reshape(6, 4, 2)
    np.testing.assert_array_equal(board_outline(points, 4, 6),
                                  [grid[0, 0], grid[0, 3], grid[5, 3], grid[5, 0]])


def test_outline_distance():
    outline = np.array([[100.0, 100], [300, 100], [300, 250], [100, 250]])
    assert abs(outline_distance(outline + [3.0, 4.0], outline) - 5.0) < 1e-9
    # The same board numbered from the opposite end has no distance.
    assert outline_distance(np.roll(outline, 2, axis=0), outline) < 1e-9
    # A tilt that keeps the centre still moves the corners: it counts as different.
    tilted = outline + [[-20.0, 0], [20, 0], [-20, 0], [20, 0]]
    assert abs(outline_distance(tilted, outline) - 20.0) < 1e-9


def test_views_of_another_image_size_are_not_kept():
    views = ViewCollection(9, 6, 0.03)
    assert consider_still(views, GRID, (640, 480), 1.0).view is not None
    other = consider_still(views, GRID + 100.0, (320, 240), 2.0)
    assert other.view is None and other.distance_ok is None
    assert len(views) == 1 and views.image_size == (640, 480)


def test_first_detection_is_not_kept_because_its_speed_is_unknown():
    views = ViewCollection(9, 6, 0.03)
    first = consider(views, GRID, (640, 480), 1.0)
    assert first.view is None and first.speed is None and first.speed_ok is None


def test_only_slow_boards_are_kept():
    views = ViewCollection(9, 6, 0.03)
    consider(views, GRID, (640, 480), 1.0, min_distance=0.0)
    # 5 px in 0.1 s = 50 px/s: kept; then 20 px in 0.1 s = 200 px/s: not kept.
    slow = consider(views, GRID + [3.0, 4.0], (640, 480), 1.1, min_distance=0.0)
    assert slow.view is not None and abs(slow.view.speed - 50.0) < 1e-3
    fast = consider(views, GRID + [15.0, 20.0], (640, 480), 1.2, min_distance=0.0)
    assert fast.view is None and not fast.speed_ok and abs(fast.speed - 200.0) < 1e-3
    assert len(views) == 1


def test_boards_with_a_misplaced_corner_are_not_kept():
    views = ViewCollection(9, 6, 0.03)
    bad = GRID.copy()
    bad[2, 4] += [0.0, 6.0]  # 0.2 squares off
    assessment = consider_still(views, bad, (640, 480), 1.0)
    assert assessment.view is None and assessment.irregularity > MAX_IRREGULARITY
    assert consider_still(views, GRID, (640, 480), 2.0).view is not None


def test_grid_irregularity_ignores_distortion_and_perspective_but_not_outliers():
    cols, rows = 6, 7
    board = board_object_points(cols, rows, 0.1)
    camera = np.array([[600.0, 0, 640], [0, 600, 512], [0, 0, 1]])
    barrel = np.array([-0.4, 0.15, 0, 0, 0])
    for rotation, translation in (([0, 0, 0], [-0.25, -0.3, 2.0]),     # centre, facing
                                  ([0.2, -0.3, 0.3], [0.6, 0.4, 1.4]),  # image corner
                                  ([1.05, 0, 0], [-0.25, -0.3, 1.5])):  # tilted 60 degrees
        points, _ = cv2.projectPoints(board, np.array(rotation, float),
                                      np.array(translation, float), camera, barrel)
        assert grid_irregularity(points, cols, rows) < 0.05
        square = np.linalg.norm(points[1] - points[0])
        points[17] += [0.15 * square, 0.0]
        assert grid_irregularity(points, cols, rows) > MAX_IRREGULARITY
    # Corners on top of each other are not a grid.
    assert grid_irregularity(np.zeros((42, 1, 2)), cols, rows) == float('inf')


def test_board_speed():
    points = np.random.default_rng(0).uniform(0, 500, (54, 1, 2)).astype(np.float32)
    assert abs(board_speed(points + [0.0, 2.0], 1.25, points, 1.0) - 8.0) < 1e-4
    # The same board numbered from the other end does not count as movement.
    assert board_speed(points[::-1], 1.1, points, 1.0) < 1e-4
    # Speed is unknown after a long gap, or for out-of-order stamps.
    assert board_speed(points, 2.0, points, 1.0) is None
    assert board_speed(points, 1.0, points, 1.0) is None


def test_calibration_input_works_for_pinhole_and_fisheye():
    cols, rows, size = 9, 6, (640, 480)
    camera = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])
    views = ViewCollection(cols, rows, 0.03)
    rng = np.random.default_rng(0)
    for i in range(8):
        rotation = rng.uniform(-0.4, 0.4, 3)
        translation = np.array([-0.12, -0.08, 0.5]) + rng.uniform(-0.05, 0.05, 3)
        projected, _ = cv2.projectPoints(views.object_points, rotation, translation, camera,
                                         None)
        consider_still(views, projected.reshape(rows, cols, 2), size, float(i), 0.0)
    assert len(views) == 8

    object_points, image_points, image_size = views.calibration_input()
    rms, matrix, _, _, _ = cv2.calibrateCamera(object_points, image_points, image_size,
                                               None, None)
    assert rms < 1e-3
    np.testing.assert_allclose(matrix, camera, atol=0.5)
    rms, matrix, _, _, _ = cv2.fisheye.calibrate(
        object_points, image_points, image_size, None, None,
        flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC)
    assert rms < 0.1
