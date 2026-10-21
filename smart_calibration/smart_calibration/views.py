"""
Calibration views: the board detections kept for calibration, with their board geometry.

A view pairs the detected corners of one frame (image points, pixels) with the corners' positions
on the flat board (object points, metres; z = 0). The object points are the same for every view.
"""

from dataclasses import dataclass

import numpy as np


def board_object_points(cols, rows, square_size):
    """
    Return the inner corners' positions on the board plane as a (rows * cols, 1, 3) float32 array.

    Row-major like the detections: point i = (i % cols, i // cols, 0) * square_size. float32 in
    the (N, 1, 3) shape is the format both calibration functions accept: cv2.calibrateCamera
    rejects float64, cv2.fisheye.calibrate rejects (N, 3) arrays (OpenCV 4.11).
    """
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points = np.zeros((rows * cols, 1, 3), np.float32)
    points[:, 0, :2] = grid * square_size
    return points


# Board speed needs two detections at most this far apart; after a longer gap (board lost,
# frames dropped) the motion in between is unknown.
MAX_SPEED_GAP_S = 0.5


def board_speed(points, stamp, previous_points, previous_stamp):
    """
    Mean corner movement in pixels per second between two detections of the same board.

    The detector may number the corners from the opposite end of the board in the next frame
    (a 180 degree turn of the numbering), so both orders are compared and the smaller movement
    is used. Returns None if the time between the detections is not in (0, MAX_SPEED_GAP_S].
    """
    elapsed = stamp - previous_stamp
    if not 0.0 < elapsed <= MAX_SPEED_GAP_S:
        return None
    movement = min(np.linalg.norm(points - previous, axis=-1).mean()
                   for previous in (previous_points, previous_points[::-1]))
    return float(movement) / elapsed


def _line_irregularity(lines):
    """
    Largest normalised third difference along each line of corners: (n_lines, n >= 4, 2) array.

    The third difference p[i+2] - 3 p[i+1] + 3 p[i] - p[i-1] is zero when four neighbouring
    corners lie on a quadratic curve. Lens distortion and perspective bend a row only slowly from
    square to square, so it stays close to zero even at the image edges; a single misplaced corner
    makes it jump by 1-3x its displacement. Divided by the local square size in pixels.
    """
    third = lines[:, 3:] - 3.0 * lines[:, 2:-1] + 3.0 * lines[:, 1:-2] - lines[:, :-3]
    steps = np.linalg.norm(np.diff(lines, axis=1), axis=-1)
    square = (steps[:, :-2] + steps[:, 1:-1] + steps[:, 2:]) / 3.0
    if square.min() <= 0.0:
        return float('inf')  # corners on top of each other: not a valid grid
    return float((np.linalg.norm(third, axis=-1) / square).max())


def grid_irregularity(points, cols, rows):
    """
    How far the detected corners deviate from a smooth grid, in square sizes.

    Maximum of the normalised third differences along all rows and columns of at least four
    corners (see _line_irregularity); 0 if the board has no such line. A wrongly located corner
    shows up as a large value in both its row and its column.
    """
    grid = points.reshape(rows, cols, 2).astype(np.float64)
    values = [0.0]
    if cols >= 4:
        values.append(_line_irregularity(grid))
    if rows >= 4:
        values.append(_line_irregularity(grid.transpose(1, 0, 2)))
    return max(values)


def board_outline(points, cols, rows):
    """Return the four outermost inner corners (first row's ends, last row's ends) as (4, 2)."""
    grid = points.reshape(rows, cols, 2)
    return np.array([grid[0, 0], grid[0, -1], grid[-1, -1], grid[-1, 0]], np.float64)


def outline_distance(outline, other):
    """
    Mean distance in pixels between the corresponding corners of two board outlines.

    Changes with the board's position, size (distance to the camera), tilt and rotation. A board
    numbered from the opposite end has its outline order shifted by two, so both orders are
    compared and the smaller distance is used.
    """
    return float(min(np.linalg.norm(outline - shifted, axis=-1).mean()
                     for shifted in (other, np.roll(other, 2, axis=0))))


@dataclass(frozen=True)
class View:
    """One kept detection: image_points is (rows * cols, 1, 2) float32, in pixels."""

    stamp: float            # image header stamp, seconds
    image_size: tuple       # (width, height) of the image the corners were detected in
    image_points: np.ndarray
    speed: float            # board speed since the previous detection, pixels per second
    outline: np.ndarray     # (4, 2) float64 outermost inner corners, see board_outline()


@dataclass(frozen=True)
class Assessment:
    """
    The checks of one detection: each measured value and whether it passed.

    A value is None when it cannot be measured (speed: first detection or after a gap; distance:
    no view kept yet, or a different image size); its verdict is then None too, except that the
    first view always passes the distance check. view is the kept View, or None.
    """

    speed: float            # board speed, pixels per second (board_speed)
    speed_ok: bool
    irregularity: float     # grid_irregularity
    irregularity_ok: bool
    distance: float         # to the nearest kept view, fraction of the image diagonal
    distance_ok: bool
    view: View


class ViewCollection:
    """The views kept for calibration, in memory. Used from the detection worker thread only."""

    def __init__(self, cols, rows, square_size):
        self.cols, self.rows = cols, rows
        self.object_points = board_object_points(cols, rows, square_size)
        self.views = []
        self._previous = None  # (stamp, image_points) of the previous detection

    def __len__(self):
        return len(self.views)

    @property
    def image_size(self):
        """(width, height) of the kept views, None while there are none."""
        return self.views[0].image_size if self.views else None

    def consider(self, corners, image_size, stamp, max_speed, max_irregularity, min_distance):
        """
        Run all checks on a detection and keep it as a view if it passes them; return Assessment.

        corners: (rows, cols, 2) array from detect_chessboard, to be called for every detection
        (consecutive detections give the board speed). Kept are detections whose board moved at
        most max_speed pixels per second since the previous detection (motion blur, rolling
        shutter), whose corners form a smooth grid (grid_irregularity at most max_irregularity:
        no misplaced corner), and whose board outline differs from every kept view by at least
        min_distance (nearest_view_distance), in the same image size as the views so far. All
        checks are evaluated even if an earlier one fails, so all values can be shown.
        """
        if corners.shape != (self.rows, self.cols, 2):
            raise ValueError(f'expected corners of shape {(self.rows, self.cols, 2)}, '
                             f'got {corners.shape}')
        image_size = tuple(image_size)
        points = corners.reshape(-1, 1, 2).astype(np.float32)
        previous, self._previous = self._previous, (stamp, points)

        speed = None
        if previous is not None:
            previous_stamp, previous_points = previous
            speed = board_speed(points, stamp, previous_points, previous_stamp)
        speed_ok = None if speed is None else speed <= max_speed
        irregularity = grid_irregularity(points, self.cols, self.rows)
        irregularity_ok = irregularity <= max_irregularity
        outline = board_outline(points, self.cols, self.rows)
        distance, distance_ok = None, None
        if not self.views:
            distance_ok = True  # the first view is always new
        elif image_size == self.image_size:  # intrinsics are only valid for one resolution
            distance = self.nearest_view_distance(outline, image_size)
            distance_ok = distance >= min_distance

        view = None
        if speed_ok and irregularity_ok and distance_ok:
            view = View(stamp, image_size, points, speed, outline)
            self.views.append(view)
        return Assessment(speed, speed_ok, irregularity, irregularity_ok, distance, distance_ok,
                          view)

    def nearest_view_distance(self, outline, image_size):
        """
        Distance of a board outline to the nearest kept view's, as a fraction of the diagonal.

        Mean distance between the outline corners, see outline_distance.
        """
        nearest = min(outline_distance(outline, kept.outline) for kept in self.views)
        return nearest / float(np.hypot(*image_size))

    def calibration_input(self):
        """Return (object_points, image_points, image_size) in the form cv2's calibrate takes."""
        return ([self.object_points] * len(self.views),
                [view.image_points for view in self.views],
                self.image_size)
