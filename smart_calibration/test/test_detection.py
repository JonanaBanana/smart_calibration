import cv2
import numpy as np

from smart_calibration.detection import detect_chessboard, sb_flags

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


def test_sb_flags():
    assert sb_flags() == 0
    assert sb_flags(normalize=True, accuracy=True) == (
        cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY)
