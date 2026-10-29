import cv2
import numpy as np

from smart_calibration.detection import detect_chessboard, DetectedBoard
from smart_calibration.views import (
    board_outline, board_speed, common_corners, corner_residuals, image_points, object_points,
    outline_distance, outline_size, reference_outline, relative_outline_distance,
    ViewCollection)

from test_detection import render_board

MAX_SPEED = 0.15  # image widths per second
MAX_RESIDUAL = 0.1
MIN_DISTANCE = 0.1
# A flat 9 x 6 grid of corners with 30 px squares, (rows, cols, 2).
GRID = (100.0 + 30.0 * np.mgrid[0:9, 0:6].T).astype(np.float32)
CHESSBOARD_REFERENCE = reference_outline('chessboard', 9, 6)


def chessboard(corners):
    return DetectedBoard(np.asarray(corners, np.float32), (0, 0), False)


def collection(reference=CHESSBOARD_REFERENCE):
    return ViewCollection(0.03, reference)


def consider(views, board, image_size, stamp, min_distance=MIN_DISTANCE):
    return views.consider(board, image_size, stamp, MAX_SPEED, MAX_RESIDUAL, min_distance)


def consider_still(views, board, image_size, stamp, min_distance=MIN_DISTANCE):
    """Present a board twice, 0.1 s apart and unmoved; return the assessment of the second one."""
    consider(views, board, image_size, stamp - 0.1, min_distance)
    return consider(views, board, image_size, stamp, min_distance)


def test_object_points_are_row_major_board_coordinates_on_the_board_plane():
    points = object_points(chessboard(np.zeros((3, 4, 2))), 0.03)
    assert points.shape == (12, 1, 3) and points.dtype == np.float32
    np.testing.assert_allclose(points[[0, 1, 4, 11], 0], [
        [0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.0, 0.03, 0.0], [0.09, 0.06, 0.0]])


def test_object_points_are_relative_to_the_origin():
    board = DetectedBoard(np.zeros((3, 4, 2), np.float32), (1, 2), True)
    points = object_points(board, 0.1)
    np.testing.assert_allclose(points[0, 0], [-0.2, -0.1, 0.0])   # grid (0, 0)
    np.testing.assert_allclose(points[6, 0], [0.0, 0.0, 0.0])     # grid (1, 2): the origin


def test_object_points_are_in_the_order_of_the_detected_corners():
    corners = detect_chessboard(render_board(9, 6), 9, 6)
    board = object_points(chessboard(corners), 1.0)[:, 0, :2]
    # Same order: one homography maps every board point onto its detected corner.
    homography, _ = cv2.findHomography(board, corners.reshape(-1, 2))
    mapped = cv2.perspectiveTransform(board[:, None], homography)[:, 0]
    assert np.abs(mapped - corners.reshape(-1, 2)).max() < 0.1


def test_only_views_different_from_the_kept_ones_are_kept():
    views = collection()
    corners = detect_chessboard(render_board(9, 6), 9, 6)
    size = (440, 320)
    # 40 px squares: outline 320 x 200 px, mean side 260 px, so MIN_DISTANCE is 26 px.
    first = consider_still(views, chessboard(corners), size, 1.0)
    assert first.view is not None and first.distance is None and first.distance_ok
    close = consider_still(views, chessboard(corners + 10.0), size, 2.0)  # 14 px: too close
    assert close.view is None and not close.distance_ok and 0.0 < close.distance < MIN_DISTANCE
    assert consider_still(views, chessboard(corners + 30.0), size, 3.0).view is not None
    assert consider_still(views, chessboard(corners + 20.0), size, 4.0).view is None
    assert len(views) == 2 and views.image_size == size


def test_all_checks_are_evaluated_even_if_one_fails():
    views = collection()
    consider_still(views, chessboard(GRID), (640, 480), 1.0)
    bad = GRID + 50.0
    bad[2, 4] += [0.0, 6.0]  # misplaced corner
    consider(views, chessboard(bad), (640, 480), 2.0)
    moving = consider(views, chessboard(bad + 30.0), (640, 480), 2.1)  # 424 px/s = 0.66 w/s
    assert moving.speed_ok is False and moving.residual_ok is False
    assert moving.distance is not None and moving.distance_ok and moving.view is None


def test_reference_outline_and_board_outline_of_a_chessboard():
    reference = reference_outline('chessboard', 4, 6)
    np.testing.assert_array_equal(reference, [[0, 0], [3, 0], [3, 5], [0, 5]])
    grid = np.arange(6 * 4 * 2, dtype=np.float32).reshape(6, 4, 2)
    np.testing.assert_array_equal(board_outline(chessboard(grid), reference),
                                  [grid[0, 0], grid[0, 3], grid[5, 3], grid[5, 0]])


def test_board_outline_projects_reference_points_outside_the_visible_part():
    # 20 px squares, origin at grid (1, 1): board coordinate (x, y) is at (50 + 20 x, 40 + 20 y).
    rows, cols = np.mgrid[0:4, 0:5]
    corners = np.stack([30.0 + 20.0 * cols, 20.0 + 20.0 * rows], -1).astype(np.float32)
    board = DetectedBoard(corners, (1, 1), True)
    reference = reference_outline('oversized_marker', 0, 0)  # +-3 squares around the origin
    np.testing.assert_allclose(board_outline(board, reference),
                               [[-10, -20], [110, -20], [110, 100], [-10, 100]], atol=1e-3)


def test_outline_distance():
    outline = np.array([[100.0, 100], [300, 100], [300, 250], [100, 250]])
    assert abs(outline_distance(outline + [3.0, 4.0], outline, True) - 5.0) < 1e-9
    # A plain chessboard numbered from the opposite end has no distance; an oriented one does.
    assert outline_distance(np.roll(outline, 2, axis=0), outline, True) < 1e-9
    assert outline_distance(np.roll(outline, 2, axis=0), outline, False) > 100.0
    # A tilt that keeps the centre still moves the corners: it counts as different.
    tilted = outline + [[-20.0, 0], [20, 0], [-20, 0], [20, 0]]
    assert abs(outline_distance(tilted, outline, True) - 20.0) < 1e-9


def test_relative_outline_distance_is_the_same_near_and_far():
    outline = np.array([[100.0, 100], [300, 100], [300, 250], [100, 250]])
    assert abs(outline_size(outline) - 175.0) < 1e-9
    near = relative_outline_distance(outline + [35.0, 0], outline, True)
    # The same board 4 times further away, moved by the same part of its size.
    far_outline = outline / 4 + 500
    far = relative_outline_distance(far_outline + [35.0 / 4, 0], far_outline, True)
    assert abs(near - 0.2) < 1e-9 and abs(far - 0.2) < 1e-9
    # Symmetric, also between a near and a far view.
    assert abs(relative_outline_distance(outline, far_outline, False)
               - relative_outline_distance(far_outline, outline, False)) < 1e-9


def test_views_of_another_image_size_are_not_kept():
    views = collection()
    assert consider_still(views, chessboard(GRID), (640, 480), 1.0).view is not None
    other = consider_still(views, chessboard(GRID + 100.0), (320, 240), 2.0)
    assert other.view is None and other.distance_ok is None
    assert len(views) == 1 and views.image_size == (640, 480)


def test_first_detection_is_not_kept_because_its_speed_is_unknown():
    first = consider(collection(), chessboard(GRID), (640, 480), 1.0)
    assert first.view is None and first.speed is None and first.speed_ok is None


def test_only_slow_boards_are_kept():
    views = collection()
    consider(views, chessboard(GRID), (640, 480), 1.0, min_distance=0.0)
    # 5 px in 0.1 s on a 640 px wide image = 0.078 w/s: kept; then 20 px = 0.31 w/s: not kept.
    slow = consider(views, chessboard(GRID + [3.0, 4.0]), (640, 480), 1.1, min_distance=0.0)
    assert slow.view is not None and abs(slow.view.speed - 50.0 / 640) < 1e-6
    fast = consider(views, chessboard(GRID + [15.0, 20.0]), (640, 480), 1.2, min_distance=0.0)
    assert fast.view is None and not fast.speed_ok and abs(fast.speed - 200.0 / 640) < 1e-6
    assert len(views) == 1


def test_boards_with_a_misplaced_corner_are_not_kept():
    views = collection()
    bad = GRID.copy()
    bad[2, 4] += [0.0, 6.0]  # 0.2 squares off
    assessment = consider_still(views, chessboard(bad), (640, 480), 1.0)
    assert assessment.view is None and assessment.residual > MAX_RESIDUAL
    assert consider_still(views, chessboard(GRID), (640, 480), 2.0).view is not None


def projected_grid(rotation, translation, cols=6, rows=7):
    """Corners (rows, cols, 2) of a board seen by a wide-angle camera with strong distortion."""
    board = object_points(chessboard(np.zeros((rows, cols, 2))), 0.1)
    camera = np.array([[600.0, 0, 640], [0, 600, 512], [0, 0, 1]])
    barrel = np.array([-0.4, 0.15, 0, 0, 0])
    points, _ = cv2.projectPoints(board, np.array(rotation, float),
                                  np.array(translation, float), camera, barrel)
    return points.reshape(rows, cols, 2)


POSES = (([0, 0, 0], [-0.25, -0.3, 2.0]),     # centre, facing the camera
         ([0.2, -0.3, 0.3], [0.6, 0.4, 1.4]),  # image corner
         ([1.05, 0, 0], [-0.25, -0.3, 1.5]))   # tilted 60 degrees


def test_corner_residuals_ignore_distortion_and_perspective():
    for rotation, translation in POSES:
        residuals = corner_residuals(chessboard(projected_grid(rotation, translation)))
        assert np.isfinite(residuals).all() and residuals.max() < 0.03


def test_corner_residuals_find_a_misplaced_corner_anywhere_in_the_grid():
    # Also at the grid's edge and corner.
    for rotation, translation in POSES:
        for row, col in ((3, 2), (0, 3), (0, 0), (6, 5)):
            grid = projected_grid(rotation, translation)
            square = np.linalg.norm(grid[row, 1] - grid[row, 0])
            grid[row, col] += [0.0, 0.2 * square]  # slid along a square's edge
            residuals = corner_residuals(chessboard(grid))
            assert residuals[row, col] > MAX_RESIDUAL, (rotation, row, col)
            assert np.nanargmax(residuals) == row * 6 + col


def test_corner_residuals_ignore_invalid_corners():
    grid = projected_grid(*POSES[0])
    grid[0, 0] += [40.0, 30.0]  # far off, but invalid: neither checked nor used as a neighbour
    valid = np.ones((7, 6), bool)
    valid[0, 0] = False
    residuals = corner_residuals(DetectedBoard(grid, (0, 0), False, valid))
    assert np.isnan(residuals[0, 0]) and np.nanmax(residuals) < 0.03


def test_invalid_corners_are_not_used_in_views():
    valid = np.ones((6, 9), bool)
    valid[0, 0] = valid[5, 8] = False
    board = DetectedBoard(GRID, (0, 0), False, valid)
    objects, images = object_points(board, 0.03), image_points(board)
    assert objects.shape == (52, 1, 3) and images.shape == (52, 1, 2)
    np.testing.assert_allclose(objects[0, 0], [0.03, 0.0, 0.0])  # grid (0, 1)
    np.testing.assert_allclose(images[0, 0], GRID[0, 1])
    # An invalid outline corner is projected from the valid ones, not taken as detected.
    broken = GRID.copy()
    broken[0, 0] = [-50.0, -50.0]
    outline = board_outline(DetectedBoard(broken, (0, 0), False, valid), CHESSBOARD_REFERENCE)
    np.testing.assert_allclose(outline, [GRID[0, 0], GRID[0, 8], GRID[5, 8], GRID[5, 0]],
                               atol=1e-3)


def test_board_speed_ignores_invalid_corners():
    moved = GRID + [3.0, 4.0]
    moved[2, 2] += 100.0
    valid = np.ones((6, 9), bool)
    valid[2, 2] = False
    speed = board_speed(DetectedBoard(moved, (0, 0), False, valid), 1.1, chessboard(GRID), 1.0)
    assert abs(speed - 50.0) < 1e-3


def test_board_speed_of_a_chessboard():
    corners = np.random.default_rng(0).uniform(0, 500, (6, 9, 2)).astype(np.float32)
    board = chessboard(corners)
    assert abs(board_speed(chessboard(corners + [0.0, 2.0]), 1.25, board, 1.0) - 8.0) < 1e-4
    # The same board numbered from the other end does not count as movement.
    assert board_speed(chessboard(corners[::-1, ::-1]), 1.1, board, 1.0) < 1e-4
    # Speed is unknown after a long gap, or for out-of-order stamps.
    assert board_speed(board, 2.0, board, 1.0) is None
    assert board_speed(board, 1.0, board, 1.0) is None


def test_board_speed_of_an_oversized_board_matches_corners_by_board_coordinates():
    rows, cols = np.mgrid[0:8, 0:10]
    full = np.stack([20.0 * cols, 20.0 * rows], -1).astype(np.float32)  # board coords * 20 px
    # Two views of different parts of the same unmoved board: grid origins differ.
    previous = DetectedBoard(full[0:6, 0:7], (2, 3), True)
    current = DetectedBoard(full[3:8, 2:10], (-1, 1), True)
    points, previous_points = common_corners(current, previous)
    assert len(points) == 3 * 5  # board rows 3-5, cols 2-6
    np.testing.assert_array_equal(points, previous_points)
    assert board_speed(current, 1.1, previous, 1.0) == 0.0
    # Without common corners the speed is unknown.
    apart = DetectedBoard(full[6:8, 7:10], (-4, -4), True)
    assert board_speed(apart, 1.1, previous, 1.0) is None


def test_calibration_input_works_for_pinhole_and_fisheye():
    cols, rows, size = 9, 6, (640, 480)
    camera = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])
    views = collection(reference_outline('chessboard', cols, rows))
    board = object_points(chessboard(np.zeros((rows, cols, 2))), 0.03)
    rng = np.random.default_rng(0)
    for i in range(8):
        rotation = rng.uniform(-0.4, 0.4, 3)
        translation = np.array([-0.12, -0.08, 0.5]) + rng.uniform(-0.05, 0.05, 3)
        projected, _ = cv2.projectPoints(board, rotation, translation, camera, None)
        consider_still(views, chessboard(projected.reshape(rows, cols, 2)), size, float(i), 0.0)
    assert len(views) == 8

    object_points_list, image_points, image_size = views.calibration_input()
    rms, matrix, _, _, _ = cv2.calibrateCamera(object_points_list, image_points, image_size,
                                               None, None)
    assert rms < 1e-3
    np.testing.assert_allclose(matrix, camera, atol=0.5)
    rms, matrix, _, _, _ = cv2.fisheye.calibrate(
        object_points_list, image_points, image_size, None, None,
        flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC)
    assert rms < 0.1


def test_views_are_not_kept_while_collection_is_stopped_and_can_be_removed():
    views = collection()
    consider(views, chessboard(GRID), (640, 480), 1.0)
    stopped = views.consider(chessboard(GRID), (640, 480), 1.1, MAX_SPEED, MAX_RESIDUAL,
                             MIN_DISTANCE, keep=False)
    assert stopped.speed_ok and stopped.distance_ok and stopped.view is None and len(views) == 0
    for i in range(3):
        consider_still(views, chessboard(GRID + 100.0 * i), (640, 480), 2.0 + i)
    first, third = views.views[0], views.views[2]
    views.remove([1])
    assert views.views == [first, third]
