import cv2
import numpy as np

from smart_calibration.detection import (
    border_mask, detect_board, detect_chessboard, detect_marker_board, sb_flags)

SQUARE_PX = 40


def render_board(cols, rows):
    """Chessboard with cols x rows inner corners and a one-square white border."""
    squares_x, squares_y = cols + 1, rows + 1
    image = np.full(((squares_y + 2) * SQUARE_PX, (squares_x + 2) * SQUARE_PX), 255, np.uint8)
    for row in range(squares_y):
        for col in range(squares_x):
            if (row + col) % 2 == 0:
                y, x = (row + 1) * SQUARE_PX, (col + 1) * SQUARE_PX
                image[y:y + SQUARE_PX, x:x + SQUARE_PX] = 0
    return cv2.GaussianBlur(image, (0, 0), 1.0)


def test_detects_board_with_expected_shape_and_positions():
    corners = detect_chessboard(render_board(9, 6), 9, 6)
    assert corners is not None
    assert corners.shape == (6, 9, 2)
    # Inner corners lie on the square boundaries: (2 + i) * SQUARE_PX - 0.5 in OpenCV pixel
    # coordinates. The board may be reported in either of its two symmetric orientations.
    xs = np.sort(np.unique(np.round(corners[..., 0].ravel() + 0.5)))
    ys = np.sort(np.unique(np.round(corners[..., 1].ravel() + 0.5)))
    np.testing.assert_allclose(xs, (2 + np.arange(9)) * SQUARE_PX, atol=1)
    np.testing.assert_allclose(ys, (2 + np.arange(6)) * SQUARE_PX, atol=1)


def test_no_board_in_blank_image():
    assert detect_chessboard(np.full((240, 320), 128, np.uint8), 9, 6) is None


def test_wrong_board_size_is_not_detected():
    assert detect_chessboard(render_board(9, 6), 7, 5) is None


def render_marker_board(markers, squares_x=14, squares_y=10):
    """Oversized board texture (gen_pattern.py colours) with marker dots in the given cells."""
    image = np.full((squares_y * SQUARE_PX, squares_x * SQUARE_PX), 255, np.uint8)
    for y in range(squares_y):
        for x in range(squares_x):
            if x % 2 == y % 2:
                image[y * SQUARE_PX:(y + 1) * SQUARE_PX, x * SQUARE_PX:(x + 1) * SQUARE_PX] = 0
    for x, y in markers:
        centre = (int((x + 0.5) * SQUARE_PX), int((y + 0.5) * SQUARE_PX))
        color = 255 if x % 2 == y % 2 else 0  # white dot in a black cell, black in a white one
        cv2.circle(image, centre, round(0.17 * SQUARE_PX), color, -1, cv2.LINE_AA)
    return image


# L with exactly one white marker cell, (7, 4): the origin is its top-left corner.
ONE_ORIGIN = [(7, 4), (8, 4), (7, 5)]


def board_view(texture, scale, angle_deg, tilt=(0.0, 0.0), size=(960, 720)):
    """Texture seen through a homography around the origin corner; returns (image, homography)."""
    angle = np.radians(angle_deg)
    to_origin = np.array([[1, 0, -7 * SQUARE_PX], [0, 1, -4 * SQUARE_PX], [0, 0, 1]], float)
    rotate = np.array([[scale * np.cos(angle), -scale * np.sin(angle), 0],
                       [scale * np.sin(angle), scale * np.cos(angle), 0], [0, 0, 1]])
    perspective = np.array([[1, 0, 0], [0, 1, 0], [tilt[0], tilt[1], 1]], float)
    to_centre = np.array([[1, 0, size[0] / 2], [0, 1, size[1] / 2], [0, 0, 1]], float)
    homography = to_centre @ perspective @ rotate @ to_origin
    image = cv2.warpPerspective(texture, homography, size, flags=cv2.INTER_AREA,
                                borderValue=128)
    return cv2.GaussianBlur(image, (0, 0), 0.8), homography


def test_marker_board_coordinates_are_the_same_physical_corners_in_every_view():
    texture = render_marker_board(ONE_ORIGIN)
    # In-plane rotations of 30-45 degrees are not detected reliably with CALIB_CB_LARGER.
    for scale, angle, tilt in ((2.5, 0, (0, 0)), (2.5, 90, (0, 0)), (2.5, 180, (0, 0)),
                               (2.0, 20, (0, 0)), (2.3, -15, (0.0005, -0.0003))):
        image, homography = board_view(texture, scale, angle, tilt)
        board = detect_marker_board(image)
        assert board is not None and board.oriented, (scale, angle)
        # The board overflows the view: fewer corners than its 13 x 9 inner corners.
        assert board.corners.shape[0] * board.corners.shape[1] < 13 * 9
        # Board coordinates (x, y) are squares from the top-left corner of cell (7, 4).
        texture_points = (board.board_coordinates() + [7, 4]) * SQUARE_PX - 0.5
        expected = cv2.perspectiveTransform(
            texture_points.reshape(-1, 1, 2).astype(np.float64), homography).reshape(-1, 2)
        assert np.abs(board.corners.reshape(-1, 2) - expected).max() < 0.5, (scale, angle)


def test_marker_board_with_two_origin_cells_is_rejected():
    # L whose corner cell (7, 5) is black: both arm cells are white marker cells.
    image, _ = board_view(render_marker_board([(7, 4), (7, 5), (8, 5)]), 1.6, 0)
    assert detect_marker_board(image) is None


def test_detect_board_modes():
    board = detect_board(render_board(9, 6), 'chessboard', 9, 6)
    assert board.shape == (6, 9) and board.origin == (0, 0) and not board.oriented
    image, _ = board_view(render_marker_board(ONE_ORIGIN), 1.6, 0)
    assert detect_board(image, 'oversized_marker', 9, 6).oriented
    # A plain chessboard has no markers.
    assert detect_board(render_board(9, 6), 'oversized_marker', 9, 6) is None


def test_border_mask():
    # Pixel centres at integers: with 100 px and a 5 px margin, 5.0 ... 94.0 is inside.
    corners = np.array([[[50.0, 50.0], [5.0, 94.0], [50.0, 94.5], [-3.0, 50.0], [4.9, 50.0]]])
    np.testing.assert_array_equal(border_mask(corners, (100, 100), 5.0),
                                  [[True, True, False, False, False]])


def test_detect_board_marks_corners_near_the_border_invalid():
    image = render_board(9, 6)
    board = detect_board(image, 'chessboard', 9, 6, border_margin_px=0.0)
    assert board.valid.all()
    board = detect_board(image, 'chessboard', 9, 6, border_margin_px=100.0)
    expected = border_mask(board.corners, (image.shape[1], image.shape[0]), 100.0)
    assert not expected.all() and expected.any()
    np.testing.assert_array_equal(board.valid, expected)


def test_sb_flags():
    assert sb_flags() == 0
    assert sb_flags(normalize=True, accuracy=True) == (
        cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY)
