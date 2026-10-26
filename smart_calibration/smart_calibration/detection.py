"""
Chessboard detection on prepared 8-bit grayscale images.

Two board modes (chosen at startup):

- 'chessboard': a plain chessboard with a known number of inner corners, completely visible.
  The corners are numbered from the first detected corner; the detector may number them from
  the opposite end of the board in another frame.
- 'oversized_marker': a radon checkerboard with three marker dots in an L near its middle, larger
  than the camera's view (findChessboardCornersSB with CALIB_CB_LARGER | CALIB_CB_MARKER). Each
  frame shows a different part of it with a different number of corners; the markers fix the
  board's orientation and origin, so every corner gets consistent board coordinates. The L must
  have exactly one white marker cell (black dot): its top-left corner is the origin.
"""

from dataclasses import dataclass, replace

import cv2
import numpy as np

BOARD_MODES = ('chessboard', 'oversized_marker')
# Smallest grid (cols, rows of inner corners) asked for in oversized_marker mode. The marker test
# skips the cells on the edge of the detected grid, so smaller boards are never found with
# markers anyway (measured with OpenCV 4.11 on synthetic views).
MARKER_MIN_PATTERN = (5, 5)
# Value of findChessboardCornersSBWithMeta's meta output at the origin corner: the top-left
# corner of a white cell with a black marker dot.
META_ORIGIN = 4


@dataclass(frozen=True)
class DetectedBoard:
    """
    Detected corners and where they are on the board.

    corners: (rows, cols, 2) float32 pixel positions of the detected grid, row-major, pixel centres
    at integer coordinates (OpenCV convention). origin: (row, col) in that grid of the corner with
    board coordinates (0, 0); the corner at (row, col) has board coordinates (col - origin col,
    row - origin row), in squares. oriented: True if the markers fix the board's orientation, so
    board coordinates mean the same physical corner in every frame. valid: (rows, cols) bool,
    False for corners not to be used (see border_mask); all True if not given.
    """

    corners: np.ndarray
    origin: tuple
    oriented: bool
    valid: np.ndarray = None

    def __post_init__(self):
        if self.valid is None:
            object.__setattr__(self, 'valid', np.ones(self.shape, bool))

    @property
    def shape(self):
        """(rows, cols) of the detected grid."""
        return self.corners.shape[:2]

    def board_coordinates(self):
        """(rows, cols, 2) int array of each corner's board coordinates (x, y) in squares."""
        rows, cols = self.shape
        row_index, col_index = np.mgrid[0:rows, 0:cols]
        return np.stack([col_index - self.origin[1], row_index - self.origin[0]], axis=-1)


def sb_flags(normalize=False, exhaustive=False, accuracy=False):
    """
    Combine findChessboardCornersSB flags.

    See explore.md for what they do and their measured effect: normalize (histogram
    equalization) helps on dark / low-contrast images, accuracy (internal 2x upsampling) lowers
    the corner error by 30-45 % at ~4x the runtime, exhaustive made no difference so far.
    """
    flags = 0
    if normalize:
        flags |= cv2.CALIB_CB_NORMALIZE_IMAGE
    if exhaustive:
        flags |= cv2.CALIB_CB_EXHAUSTIVE
    if accuracy:
        flags |= cv2.CALIB_CB_ACCURACY
    return flags


def detect_chessboard(gray, cols, rows, flags=0):
    """
    Detect a chessboard with cols x rows inner corners.

    Returns a (rows, cols, 2) float32 array of corner positions in pixels, row-major with pixel
    centres at integer coordinates (OpenCV convention), or None if no board was found.
    """
    found, corners = cv2.findChessboardCornersSB(gray, (cols, rows), flags=flags)
    if not found:
        return None
    return corners.reshape(rows, cols, 2)


def detect_marker_board(gray, flags=0):
    """
    Detect an oversized radon checkerboard with markers; return a DetectedBoard or None.

    Also None if the result does not have exactly one origin corner (an L with two white marker
    cells, or none), so board coordinates are never ambiguous.
    """
    found, corners, meta = cv2.findChessboardCornersSBWithMeta(
        gray, MARKER_MIN_PATTERN, flags | cv2.CALIB_CB_LARGER | cv2.CALIB_CB_MARKER)
    if not found:
        return None
    rows, cols = meta.shape
    origins = np.argwhere(meta == META_ORIGIN)
    if len(origins) != 1 or corners.shape[0] != rows * cols:
        return None
    return DetectedBoard(corners.reshape(rows, cols, 2), tuple(int(i) for i in origins[0]), True)


def border_mask(corners, image_size, margin_px):
    """
    (rows, cols) bool: True for corners at least margin_px inside the image.

    With CALIB_CB_LARGER, findChessboardCornersSB adds whole rows / columns of which up to half
    may be predicted rather than found: such corners can lie outside the image, and a black square
    cut off by the image border can be mistaken for a corner. image_size: (width, height); pixel
    centres at integer coordinates.
    """
    width, height = image_size
    x, y = corners[..., 0], corners[..., 1]
    return ((x >= margin_px) & (y >= margin_px)
            & (x <= width - 1 - margin_px) & (y <= height - 1 - margin_px))


def detect_board(gray, mode, cols, rows, flags=0, border_margin_px=0.0):
    """
    Detect the board of the given mode (see BOARD_MODES); return a DetectedBoard or None.

    cols, rows: inner corners of the board in 'chessboard' mode (unused for 'oversized_marker').
    Corners closer than border_margin_px to the image border (or outside it) are marked invalid.
    """
    if mode == 'oversized_marker':
        board = detect_marker_board(gray, flags)
    elif mode == 'chessboard':
        corners = detect_chessboard(gray, cols, rows, flags)
        board = None if corners is None else DetectedBoard(corners, (0, 0), False)
    else:
        raise ValueError(f'unknown board mode {mode!r}, expected one of {BOARD_MODES}')
    if board is None:
        return None
    size = (gray.shape[1], gray.shape[0])
    return replace(board, valid=board.valid & border_mask(board.corners, size, border_margin_px))
