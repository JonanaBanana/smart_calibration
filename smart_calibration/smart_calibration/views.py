"""
Calibration views: the board detections kept for calibration.

A view pairs the valid detected corners of one frame (image points, pixels) with their positions
on the flat board (object points, metres, z = 0).
"""

from dataclasses import dataclass

import cv2
import numpy as np

# Longest gap between two detections over which the board speed is measured.
MAX_SPEED_GAP_S = 0.5
# Oversized boards: half size, in squares, of the region around the origin used as outline.
MARKER_REFERENCE_HALF_SIZE = 3
# Corner check: neighbours within this many squares, and at least this many of them.
RESIDUAL_REACH = 3
RESIDUAL_MIN_NEIGHBOURS = 8


def object_points(board, square_size):
    """
    Return the valid corners' positions on the board plane, (N, 1, 3) float32, row-major.

    This shape and type are what both cv2.calibrateCamera and cv2.fisheye.calibrate accept.
    """
    coordinates = board.board_coordinates()[board.valid]
    points = np.zeros((len(coordinates), 1, 3), np.float32)
    points[:, 0, :2] = coordinates * square_size
    return points


def image_points(board):
    """Return the valid corners' pixel positions as an (N, 1, 2) float32 array (row-major)."""
    return board.corners[board.valid].reshape(-1, 1, 2).astype(np.float32)


def reference_outline(mode, cols, rows):
    """
    Return the board coordinates (4, 2) of the outline compared between views.

    'chessboard': the four outermost inner corners. 'oversized_marker': a square around the origin.
    """
    if mode == 'oversized_marker':
        half = MARKER_REFERENCE_HALF_SIZE
        return np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    return np.array([[0, 0], [cols - 1, 0], [cols - 1, rows - 1], [0, rows - 1]])


def board_outline(board, reference):
    """
    Return the image positions (4, 2) of the reference board coordinates.

    Points without a valid detected corner are projected with a homography fitted to the corners.
    """
    rows, cols = board.shape
    outline = np.zeros((len(reference), 2))
    homography = None
    for i, (x, y) in enumerate(reference):
        row, col = y + board.origin[0], x + board.origin[1]
        if 0 <= row < rows and 0 <= col < cols and board.valid[row, col]:
            outline[i] = board.corners[row, col]
            continue
        if homography is None:
            homography, _ = cv2.findHomography(
                board.board_coordinates()[board.valid].astype(np.float32),
                board.corners[board.valid].astype(np.float32))
        outline[i] = cv2.perspectiveTransform(
            np.array([[[x, y]]], np.float64), homography)[0, 0]
    return outline


def common_corners(board, other):
    """Return the valid corners of two boards with the same board coordinates, (n, 2) each."""
    (rows, cols), (other_rows, other_cols) = board.shape, other.shape
    (row0, col0), (other_row0, other_col0) = board.origin, other.origin
    x_low, x_high = max(-col0, -other_col0), min(cols - col0, other_cols - other_col0)
    y_low, y_high = max(-row0, -other_row0), min(rows - row0, other_rows - other_row0)
    if x_low >= x_high or y_low >= y_high:
        return np.zeros((0, 2)), np.zeros((0, 2))
    part = (slice(y_low + row0, y_high + row0), slice(x_low + col0, x_high + col0))
    other_part = (slice(y_low + other_row0, y_high + other_row0),
                  slice(x_low + other_col0, x_high + other_col0))
    both = board.valid[part] & other.valid[other_part]
    return board.corners[part][both], other.corners[other_part][both]


def board_speed(board, stamp, previous, previous_stamp):
    """
    Return the mean corner movement in pixels per second between two detections.

    A plain chessboard is also compared with its numbering turned by 180 degrees. None if the
    detections are more than MAX_SPEED_GAP_S apart or have no valid corner in common.
    """
    elapsed = stamp - previous_stamp
    if not 0.0 < elapsed <= MAX_SPEED_GAP_S:
        return None
    if board.oriented:
        points, previous_points = common_corners(board, previous)
        if len(points) == 0:
            return None
        movement = np.linalg.norm(points - previous_points, axis=-1).mean()
    else:
        if board.shape != previous.shape:
            return None
        movements = []
        for corners, valid in ((previous.corners, previous.valid),
                               (previous.corners[::-1, ::-1], previous.valid[::-1, ::-1])):
            both = board.valid & valid
            if both.any():
                movements.append(np.linalg.norm(board.corners[both] - corners[both],
                                                axis=-1).mean())
        if not movements:
            return None
        movement = min(movements)
    return float(movement) / elapsed


def _quadratic_terms(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    return np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])


def corner_residuals(board):
    """
    Return each corner's distance from where its neighbours put it, (rows, cols), in squares.

    A quadratic surface (pixel position over grid position) is fitted to the valid corners within
    RESIDUAL_REACH squares, leaving the corner out. NaN for invalid corners and corners with
    fewer than RESIDUAL_MIN_NEIGHBOURS valid neighbours.
    """
    corners = board.corners.astype(np.float64)
    rows, cols = board.shape
    row_index, col_index = np.mgrid[0:rows, 0:cols]
    residuals = np.full((rows, cols), np.nan)
    at_corner = _quadratic_terms([0, 1, 0], [0, 0, 1])  # the corner and two neighbours
    for row, col in zip(*np.nonzero(board.valid)):
        near = (board.valid & (np.abs(row_index - row) <= RESIDUAL_REACH)
                & (np.abs(col_index - col) <= RESIDUAL_REACH))
        near[row, col] = False
        if near.sum() < RESIDUAL_MIN_NEIGHBOURS:
            continue
        terms = _quadratic_terms(col_index[near] - col, row_index[near] - row)
        coefficients, *_ = np.linalg.lstsq(terms, corners[near], rcond=None)
        predicted, right, below = at_corner @ coefficients
        square = (np.linalg.norm(right - predicted) + np.linalg.norm(below - predicted)) / 2
        if square > 0:
            residuals[row, col] = np.linalg.norm(corners[row, col] - predicted) / square
    return residuals


def outline_distance(outline, other, numbering_may_flip):
    """
    Return the mean distance in pixels between the corresponding corners of two outlines.

    numbering_may_flip: also compare with the outline turned by 180 degrees, take the smaller.
    """
    candidates = (other, np.roll(other, 2, axis=0)) if numbering_may_flip else (other,)
    return float(min(np.linalg.norm(outline - candidate, axis=-1).mean()
                     for candidate in candidates))


def outline_size(outline):
    """Return the mean side length of a board outline, in pixels."""
    return float(np.linalg.norm(outline - np.roll(outline, 1, axis=0), axis=-1).mean())


def relative_outline_distance(outline, other, numbering_may_flip):
    """Return the distance between two outlines in board sizes (mean size of both)."""
    size = (outline_size(outline) + outline_size(other)) / 2
    return outline_distance(outline, other, numbering_may_flip) / size


@dataclass(frozen=True)
class View:
    """One kept detection: (N, 1, 2) image points in pixels, (N, 1, 3) object points in metres."""

    stamp: float            # image header stamp, seconds
    image_size: tuple       # (width, height) of the image the corners were detected in
    image_points: np.ndarray
    object_points: np.ndarray
    speed: float            # board speed since the previous detection, pixels per second
    outline: np.ndarray     # (4, 2) image positions of the reference outline, see board_outline()


@dataclass(frozen=True)
class Assessment:
    """
    The checks of one detection: each value and whether it passed.

    None where a value cannot be measured. view: the kept View, or None.
    """

    speed: float            # board speed, pixels per second (board_speed)
    speed_ok: bool
    residual: float         # largest corner_residuals() value, squares
    residual_ok: bool
    distance: float         # to the nearest kept view, board sizes (relative_outline_distance)
    distance_ok: bool
    view: View


class ViewCollection:
    """The views kept for calibration. Not thread-safe."""

    def __init__(self, square_size, reference):
        """Collect views of a board with this square size; reference: see reference_outline."""
        self.square_size = square_size
        self.reference = np.asarray(reference)
        self.views = []
        self._previous = None  # (stamp, DetectedBoard) of the previous detection

    def __len__(self):
        return len(self.views)

    @property
    def image_size(self):
        """(width, height) of the kept views, None while there are none."""
        return self.views[0].image_size if self.views else None

    def consider(self, board, image_size, stamp, max_speed, max_residual, min_distance,
                 keep=True):
        """
        Check a detection and keep it as a view if it passes; return an Assessment.

        Pass every detection: consecutive ones give the board speed. Kept if the board moved at
        most max_speed px/s, no corner residual exceeds max_residual and the outline is at least
        min_distance board sizes from every kept view. keep=False: check only.
        """
        image_size = tuple(image_size)
        previous, self._previous = self._previous, (stamp, board)

        speed = None
        if previous is not None:
            previous_stamp, previous_board = previous
            speed = board_speed(board, stamp, previous_board, previous_stamp)
        speed_ok = None if speed is None else speed <= max_speed
        residuals = corner_residuals(board)
        residual, residual_ok = None, None
        if np.isfinite(residuals).any():
            residual = float(np.nanmax(residuals))
            residual_ok = residual <= max_residual
        outline = board_outline(board, self.reference)
        distance, distance_ok = None, None
        if not self.views:
            distance_ok = True  # the first view is always new
        elif image_size == self.image_size:
            distance = self.nearest_view_distance(outline, not board.oriented)
            distance_ok = distance >= min_distance

        view = None
        if keep and speed_ok and residual_ok and distance_ok:
            view = View(stamp, image_size, image_points(board),
                        object_points(board, self.square_size), speed, outline)
            self.views.append(view)
        return Assessment(speed, speed_ok, residual, residual_ok, distance, distance_ok, view)

    def nearest_view_distance(self, outline, numbering_may_flip):
        """Return the distance of an outline to the nearest kept view's, in board sizes."""
        return min(relative_outline_distance(outline, kept.outline, numbering_may_flip)
                   for kept in self.views)

    def remove(self, indices):
        """Remove the views at these indices."""
        gone = set(indices)
        self.views = [view for i, view in enumerate(self.views) if i not in gone]

    def calibration_input(self):
        """Return (object_points, image_points, image_size) in the form cv2's calibrate takes."""
        return ([view.object_points for view in self.views],
                [view.image_points for view in self.views],
                self.image_size)
