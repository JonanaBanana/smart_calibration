"""
Calibration from the kept views, with its uncertainty: what it is unsure about and where.

Pinhole model (plumb_bob: k1, k2, p1, p2, k3). After cv2.calibrateCamera, the covariance of the
intrinsics comes from the Gauss-Newton normal equations over all views (the pose of each view
eliminated), with the point noise estimated from the residuals. It is propagated to the image:
how far the projection of a ray could be off at each point of a grid over the image (the
uncertainty map), in total and per parameter group (focal length, principal point, distortion).

On synthetic data (utility/uncertainty_prototype.py) the map's shape follows the true error, but
its values are ~1.5-2.5x too small: detector errors are correlated, not independent noise, and
with many views the true error stops shrinking (a floor) while the predicted one does not.
uncertainty_scale compensates for the first; the second means a small uncertainty alone does not
prove a good calibration.

The board tilt of each view (from its estimated pose) gives hints about missing view types; they
only inform, new views fix them.
"""

from dataclasses import dataclass
import math
import time

import cv2
import numpy as np

# Estimated parameters, in the order of cv2's camera matrix / distortion vector.
INTRINSICS = ('fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3')
GROUPS = {'focal': (0, 1), 'centre': (2, 3), 'distortion': (4, 5, 6, 7, 8)}
K3 = 8
# Uncertainty map: this many cells across the image (rows follow the aspect ratio).
MAP_COLS = 32
# Hints: a view counts as tilted from this angle between board normal and optical axis on; tilt
# axes closer than AXIS_ALIGNED_DEG to an image axis count as aligned with it.
MIN_TILT_DEG = 15.0
MIN_TILTED_VIEWS = 2
AXIS_ALIGNED_DEG = 10.0

_UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)


@dataclass(frozen=True)
class CalibrationResult:
    """One calibration from the kept views and its uncertainty; pixel values already scaled."""

    image_size: tuple             # (width, height)
    views: int
    points: int
    k3_fixed: bool
    rms: float                    # reprojection error, px
    noise: float                  # point noise estimated from the residuals, px
    camera_matrix: np.ndarray     # (3, 3)
    distortion: np.ndarray        # (5,) k1, k2, p1, p2, k3
    std_deviations: np.ndarray    # (9,) per INTRINSICS, scaled; 0 for a fixed parameter
    uncertainty_scale: float
    uncertainty: dict             # 'total' and each GROUPS key: largest map value, px
    uncertainty_map: np.ndarray   # (rows, cols) px, NaN where the model has no ray
    view_rms: np.ndarray          # (views,) reprojection error per view, px
    view_tilts: np.ndarray        # (views, 2) board_tilt() of each view, degrees
    hints: tuple                  # strings: view types that are missing
    duration_s: float


def calibrate(object_points, image_points, image_size, fix_k3):
    """Run cv2.calibrateCamera (pinhole); return (rms, camera matrix, distortion, rvecs, tvecs)."""
    flags = cv2.CALIB_FIX_K3 if fix_k3 else 0
    rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None, flags=flags)
    return rms, camera_matrix, distortion.ravel()[:5], rvecs, tvecs


def intrinsic_covariance(object_points, image_points, camera_matrix, distortion, rvecs, tvecs,
                         used):
    """
    Covariance (len(used), len(used)) of the estimated intrinsics, point noise, per-view rms.

    used: indices into INTRINSICS of the estimated parameters. Normal equations J^T J of all
    views; each view's 6 pose parameters are eliminated with the Schur complement, so only a
    small matrix is inverted. Noise^2 = squared residuals / degrees of freedom. Without k3 fixed,
    the standard deviations equal cv2.calibrateCameraExtended's.
    """
    columns = [6 + i for i in used]  # projectPoints' jacobian: rvec, tvec, fx, fy, cx, cy, dist
    reduced = np.zeros((len(used), len(used)))
    squared, observations, view_rms = 0.0, 0, []
    for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, jacobian = cv2.projectPoints(
            obj.astype(np.float64), rvec, tvec, camera_matrix, distortion)
        residuals = img.reshape(-1, 2) - projected.reshape(-1, 2)
        view_squared = float(np.sum(residuals ** 2))
        squared += view_squared
        observations += residuals.size
        view_rms.append(math.sqrt(view_squared / len(residuals)))
        a, b = jacobian[:, columns], jacobian[:, :6]
        reduced += a.T @ a - a.T @ b @ np.linalg.solve(b.T @ b, b.T @ a)
    dof = max(observations - len(used) - 6 * len(object_points), 1)
    noise = math.sqrt(squared / dof)
    return noise ** 2 * np.linalg.pinv(reduced), noise, np.array(view_rms)


def map_grid(image_size, cols=MAP_COLS):
    """
    Pixel positions (rows, cols, 2) of the uncertainty map's cell centres.

    The cells divide the image evenly (pixel centres at integers, OpenCV convention).
    """
    width, height = image_size
    rows = max(round(cols * height / width), 1)
    xs = (np.arange(cols) + 0.5) * width / cols - 0.5
    ys = (np.arange(rows) + 0.5) * height / rows - 0.5
    return np.stack(np.meshgrid(xs, ys), -1)


def projection_covariances(camera_matrix, distortion, covariance, used, pixels):
    """
    (N, 2, 2) covariance of where the model projects the ray through each pixel (N, 2).

    The rays come from the model itself (its inverse); NaN where it has none (e.g. beyond the
    image corners of a strongly distorting model). First order: J Sigma J^T, J being the
    projection's derivative by the intrinsics.
    """
    pixels = np.asarray(pixels, np.float64).reshape(-1, 1, 2)
    rays = cv2.undistortPointsIter(pixels, camera_matrix, distortion, None, None,
                                   _UNDISTORT_CRITERIA).reshape(-1, 2)
    points = np.column_stack([rays, np.ones(len(rays))]).reshape(-1, 1, 3)
    back, jacobian = cv2.projectPoints(points, np.zeros(3), np.zeros(3), camera_matrix,
                                       distortion)
    j = jacobian[:, [6 + i for i in used]].reshape(-1, 2, len(used))
    result = j @ covariance @ j.transpose(0, 2, 1)
    result[np.linalg.norm(back.reshape(-1, 2) - pixels.reshape(-1, 2), axis=1) > 1e-3] = np.nan
    return result


def uncertainty_maps(camera_matrix, distortion, covariance, used, grid):
    """
    {'total' and each GROUPS key: (rows, cols) map}: sqrt of the projection covariance's trace.

    That is the RMS length of the 2-D projection error, in px. A group's map takes the other
    parameters as exact.
    """
    covariances = {'total': covariance}
    for group, members in GROUPS.items():
        keep = [k for k, i in enumerate(used) if i in members]
        partial = np.zeros_like(covariance)
        partial[np.ix_(keep, keep)] = covariance[np.ix_(keep, keep)]
        covariances[group] = partial
    pixels = grid.reshape(-1, 2)
    return {name: np.sqrt(np.trace(projection_covariances(
        camera_matrix, distortion, c, used, pixels), axis1=1, axis2=2)).reshape(grid.shape[:2])
        for name, c in covariances.items()}


def board_tilt(rvec):
    """
    (tilt, tilt axis) of a board pose in degrees.

    tilt: angle between the board normal and the optical axis (0: facing the camera). tilt axis:
    direction in the image of the board line that stays at the same depth, in [0, 180); 0 is
    horizontal (top or bottom edge nearer), 90 vertical (left or right edge nearer).
    """
    normal = cv2.Rodrigues(np.asarray(rvec, np.float64))[0][:, 2]
    tilt = math.degrees(math.acos(min(abs(normal[2]), 1.0)))
    axis = (math.degrees(math.atan2(normal[1], normal[0])) + 90.0) % 180.0
    return tilt, axis


def axis_offset(axis):
    """Angle of a tilt axis to the nearest image axis in degrees: 0 aligned, 45 diagonal."""
    off = axis % 90.0
    return min(off, 90.0 - off)


def pose_hints(tilts):
    """
    Hints about missing view types, from (views, 2) board_tilt() values; empty if none.

    Following Sturm & Maybank's singular cases (via Rojtberg & Kuijper 2018): boards facing the
    camera do not separate focal length and distance; tilting about only one image axis leaves
    the focal length and principal point along the other weakly determined; tilt axes parallel to
    an image axis leave the principal point along it weakly determined (on synthetic data much
    less important than the other two, as views at different positions also constrain it).
    """
    tilts = np.asarray(tilts, np.float64).reshape(-1, 2)
    tilted = tilts[tilts[:, 0] >= MIN_TILT_DEG]
    if len(tilted) < MIN_TILTED_VIEWS:
        return (f'Tilt the board: fewer than {MIN_TILTED_VIEWS} views are tilted by '
                f'{MIN_TILT_DEG:.0f} deg or more (focal length).',)
    hints = []
    tilt, axis = np.radians(tilted[:, 0]), np.radians(tilted[:, 1])
    # Tilt about the vertical / horizontal image axis: the tilt's components.
    about_vertical = np.degrees(np.arctan(np.tan(tilt) * np.abs(np.sin(axis))))
    about_horizontal = np.degrees(np.arctan(np.tan(tilt) * np.abs(np.cos(axis))))
    if about_vertical.max() < MIN_TILT_DEG:
        hints.append('Add views with the left or right edge nearer (tilt about the vertical '
                     'axis): fx, cx.')
    if about_horizontal.max() < MIN_TILT_DEG:
        hints.append('Add views with the top or bottom edge nearer (tilt about the horizontal '
                     'axis): fy, cy.')
    if all(axis_offset(a) < AXIS_ALIGNED_DEG for a in tilted[:, 1]):
        hints.append('All tilt axes are close to the image axes: also tilt about a diagonal '
                     '(turn the board in its plane).')
    return tuple(hints)


def calibrate_views(object_points, image_points, image_size, free_k3_after_views,
                    uncertainty_scale):
    """
    Calibrate from the kept views and estimate the uncertainty; return a CalibrationResult.

    k3 stays fixed (0) while there are fewer than free_k3_after_views views: with few views it
    makes the distortion run wild where no corners are. uncertainty_scale multiplies all
    uncertainties (see the module docstring). Raises cv2.error if OpenCV cannot calibrate.
    """
    start = time.perf_counter()
    image_size = tuple(image_size)
    fix_k3 = len(object_points) < free_k3_after_views
    rms, camera_matrix, distortion, rvecs, tvecs = calibrate(
        object_points, image_points, image_size, fix_k3)
    used = [i for i in range(len(INTRINSICS)) if not (fix_k3 and i == K3)]
    covariance, noise, view_rms = intrinsic_covariance(
        object_points, image_points, camera_matrix, distortion, rvecs, tvecs, used)
    covariance = covariance * uncertainty_scale ** 2
    maps = uncertainty_maps(camera_matrix, distortion, covariance, used, map_grid(image_size))
    std = np.zeros(len(INTRINSICS))
    std[used] = np.sqrt(np.diag(covariance))
    tilts = np.array([board_tilt(rvec) for rvec in rvecs])
    return CalibrationResult(
        image_size=image_size, views=len(object_points),
        points=sum(len(points) for points in object_points), k3_fixed=fix_k3, rms=float(rms),
        noise=noise, camera_matrix=camera_matrix, distortion=distortion, std_deviations=std,
        uncertainty_scale=uncertainty_scale,
        uncertainty={name: float(np.nanmax(m)) if np.isfinite(m).any() else math.nan
                     for name, m in maps.items()},
        uncertainty_map=maps['total'], view_rms=view_rms, view_tilts=tilts,
        hints=pose_hints(tilts), duration_s=time.perf_counter() - start)
