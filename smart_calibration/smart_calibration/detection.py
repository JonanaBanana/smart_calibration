"""
Chessboard detection on 8-bit grayscale images.

Board modes:

- 'chessboard': a plain chessboard with a known number of inner corners, completely visible. It
  may be numbered from either end.
- 'oversized_marker': a radon checkerboard larger than the view, with three marker dots in an L
  that fix its orientation and origin (the top-left corner of the one white marker cell).
"""

from dataclasses import dataclass, replace

import cv2
import numpy as np

BOARD_MODES = ('chessboard', 'oversized_marker')
# Smallest grid (cols, rows) detected in oversized_marker mode; smaller ones carry no markers.
MARKER_MIN_PATTERN = (5, 5)
# findChessboardCornersSBWithMeta's meta value at the origin corner.
META_ORIGIN = 4


@dataclass(frozen=True)
class DetectedBoard:
    """
    Detected corners and their position on the board.

    corners: (rows, cols, 2) float32 pixel positions, row-major. origin: (row, col) of the corner
    with board coordinates (0, 0). oriented: board coordinates name the same physical corner in
    every frame. valid: (rows, cols) bool, corners to use (default all).
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

    normalize: histogram equalization. exhaustive: longer search. accuracy: 2x upsampling inside
    the detector, more accurate corners at ~4x the detection time.
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
    """Return the (rows, cols, 2) corners of a cols x rows chessboard, or None."""
    found, corners = cv2.findChessboardCornersSB(gray, (cols, rows), flags=flags)
    if not found:
        return None
    return corners.reshape(rows, cols, 2)


def detect_marker_board(gray, flags=0):
    """Detect an oversized board with markers; None unless it has exactly one origin."""
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
    Return (rows, cols) bool: True for corners at least margin_px inside the image.

    The detector can predict corners outside the image, or mistake a square cut off by the border
    for a corner. image_size: (width, height).
    """
    width, height = image_size
    x, y = corners[..., 0], corners[..., 1]
    return ((x >= margin_px) & (y >= margin_px)
            & (x <= width - 1 - margin_px) & (y <= height - 1 - margin_px))


def detect_board(gray, mode, cols, rows, flags=0, border_margin_px=0.0):
    """
    Detect the board of the given mode (BOARD_MODES); return a DetectedBoard or None.

    cols, rows: inner corners ('chessboard' only). Corners within border_margin_px of the image
    border are marked invalid.
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
