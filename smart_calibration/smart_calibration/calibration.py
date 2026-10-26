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

The board tilt of each view (from its estimated pose) gives hints about missing view types, and
so does the parameter group that is least certain (focal length or principal point; distortion,
nearly always the largest, is what the uncertainty map shows); they only inform, new views fix
them.

The final calibration pass (final_calibration) removes clear outliers: views that the other views
predict badly. Each view's leave-one-out error (calibrate without it, fit only its pose, its
reprojection error) is linearized from the full solution; on synthetic data that matched the
exact value from ~8 views on, in milliseconds instead of one calibration per view
(utility/outlier_prototype.py). A high error alone does not make a view bad: a view of a region
no other view covers is also predicted badly, and it is the most valuable one. So a view is only
removed if removing it does not raise the uncertainty much (its mean over the image: the largest
value, at image corners no view reaches, jumps too easily); otherwise it is kept as a suspect (add
views of its region instead). Corruptions that a view's pose absorbs (a board shifted by
whole squares, a zoom, mostly also rolling-shutter shear) cannot be found this way; the latter
has to be prevented by holding the board still.
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
    view_residuals: tuple         # per view (N, 2): detected minus projected corners, px
    # Only if requested (see calibrate_views), else None: per view, the leave-one-out
    # reprojection error (px) and the residuals (N, 2) of that prediction.
    leave_one_out: np.ndarray
    leave_one_out_residuals: tuple
    hints: tuple                  # strings: view types that are missing
    distortion_folds: bool        # see distortion_folds_in_image()
    duration_s: float


def calibrate(object_points, image_points, image_size, fix_k3):
    """Run cv2.calibrateCamera (pinhole); return (rms, camera matrix, distortion, rvecs, tvecs)."""
    flags = cv2.CALIB_FIX_K3 if fix_k3 else 0
    rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None, flags=flags)
    return rms, camera_matrix, distortion.ravel()[:5], rvecs, tvecs


def view_terms(object_points, image_points, camera_matrix, distortion, rvecs, tvecs, used):
    """
    Per view (r, A, B): residuals (2N,), jacobians by the intrinsics (2N, len(used)) and pose.

    used: indices into INTRINSICS of the estimated parameters. r is detected minus projected,
    x and y interleaved like cv2.projectPoints' jacobian rows.
    """
    columns = [6 + i for i in used]  # projectPoints' jacobian: rvec, tvec, fx, fy, cx, cy, dist
    terms = []
    for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, jacobian = cv2.projectPoints(
            obj.astype(np.float64), rvec, tvec, camera_matrix, distortion)
        residuals = (img.reshape(-1, 2) - projected.reshape(-1, 2)).ravel()
        terms.append((residuals, jacobian[:, columns], jacobian[:, :6]))
    return terms


def _without_pose(b, vector):
    """Return vector (2N,) or (2N, k) minus its part that a change of the view's pose explains."""
    return vector - b @ np.linalg.solve(b.T @ b, b.T @ vector)


def intrinsic_covariance(terms, used):
    """
    Covariance (len(used), len(used)) of the estimated intrinsics, and the point noise.

    terms: view_terms(). Normal equations J^T J of all views; each view's 6 pose parameters are
    eliminated with the Schur complement, so only a small matrix is inverted. Noise^2 = squared
    residuals / degrees of freedom. Without k3 fixed, the standard deviations equal
    cv2.calibrateCameraExtended's.
    """
    reduced = sum(a.T @ _without_pose(b, a) for _, a, b in terms)
    squared = sum(float(r @ r) for r, _, _ in terms)
    observations = sum(len(r) for r, _, _ in terms)
    dof = max(observations - len(used) - 6 * len(terms), 1)
    noise = math.sqrt(squared / dof)
    return noise ** 2 * np.linalg.pinv(reduced), noise


def leave_one_out(terms):
    """
    Per view, its residuals (2N,) when the other views calibrate and only its pose is fitted.

    Linearized from the full solution: one Gauss-Newton step with the view's contribution to the
    (pose-reduced) normal equations taken out, then the view's pose re-fitted by projection.
    """
    reduced = [a.T @ _without_pose(b, a) for _, a, b in terms]
    gradients = [a.T @ _without_pose(b, r) for r, a, b in terms]
    total = sum(reduced)
    result = []
    for (r, a, b), view_reduced, gradient in zip(terms, reduced, gradients):
        step = -np.linalg.pinv(total - view_reduced) @ gradient  # intrinsics without this view
        result.append(_without_pose(b, r - a @ step))
    return result


def rms(residuals):
    """RMS length of interleaved 2-D residuals (2N,), px."""
    return math.sqrt(float(residuals @ residuals) / (len(residuals) / 2))


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
        return (f'Tilt the board more ({MIN_TILT_DEG:.0f}+ deg).',)
    hints = []
    tilt, axis = np.radians(tilted[:, 0]), np.radians(tilted[:, 1])
    # Tilt about the vertical / horizontal image axis: the tilt's components.
    about_vertical = np.degrees(np.arctan(np.tan(tilt) * np.abs(np.sin(axis))))
    about_horizontal = np.degrees(np.arctan(np.tan(tilt) * np.abs(np.cos(axis))))
    if about_vertical.max() < MIN_TILT_DEG:
        hints.append('Tilt the left or right edge towards the camera.')
    if about_horizontal.max() < MIN_TILT_DEG:
        hints.append('Tilt the top or bottom edge towards the camera.')
    if all(axis_offset(a) < AXIS_ALIGNED_DEG for a in tilted[:, 1]):
        hints.append('Also tilt about a diagonal.')
    return tuple(hints)


def fold_radius(distortion):
    """
    Radius (normalized image coordinates) where the radial distortion turns back, inf if never.

    Smallest r > 0 where d/dr [r (1 + k1 r^2 + k2 r^4 + k3 r^6)] = 1 + 3 k1 r^2 + 5 k2 r^4 +
    7 k3 r^6 reaches 0. Beyond it rays land inside the image again (the model folds over); the
    tangential terms are left out, they are tiny.
    """
    k1, k2, k3 = (float(distortion[i]) if len(distortion) > i else 0.0 for i in (0, 1, 4))
    roots = np.roots([7 * k3, 5 * k2, 3 * k1, 1.0])  # in r^2; leading zeros are dropped
    positive = [root.real for root in roots if abs(root.imag) < 1e-12 and root.real > 0]
    return float(np.sqrt(min(positive))) if positive else float('inf')


def distortion_folds_in_image(camera_matrix, distortion, image_size):
    """
    Whether the model folds over inside the image: some image pixels have no valid ray.

    Then undistorting (e.g. ROS image_proc, cv2.undistort) shows the image a second time,
    mirrored, near the image corners. Compares the largest distorted radius the model reaches
    (at the fold) with the image corners' distance from the principal point.
    """
    radius = fold_radius(distortion)
    if not math.isfinite(radius):
        return False
    k1, k2, k3 = (float(distortion[i]) if len(distortion) > i else 0.0 for i in (0, 1, 4))
    r2 = radius ** 2
    reach = radius * (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3)
    width, height = image_size
    fx, fy, cx, cy = (camera_matrix[0][0], camera_matrix[1][1], camera_matrix[0][2],
                      camera_matrix[1][2])
    corners = max(math.hypot((x - cx) / fx, (y - cy) / fy)
                  for x in (-0.5, width - 0.5) for y in (-0.5, height - 0.5))
    return reach < corners


# The distortion model folds over inside the image (distortion_folds_in_image).
FOLD_HINT = 'Add views in the image corners.'


def uncertainty_hints(uncertainty):
    """
    Return a hint if the focal length or the principal point is the least certain group.

    uncertainty: CalibrationResult.uncertainty. Distortion is nearly always the largest (at the
    image corners); where to add views for it is what the uncertainty map shows.
    """
    groups = {g: uncertainty[g] for g in GROUPS if math.isfinite(uncertainty[g])}
    if not groups:
        return ()
    weakest = max(groups, key=groups.get)
    if weakest == 'focal':
        return ('Tilt the board more (20-45 deg): focal length is uncertain.',)
    if weakest == 'centre':
        return ('Tilt in other directions: image centre is uncertain.',)
    return ()


def calibrate_views(object_points, image_points, image_size, free_k3_after_views,
                    uncertainty_scale, with_leave_one_out=False):
    """
    Calibrate from the kept views and estimate the uncertainty; return a CalibrationResult.

    k3 stays fixed (0) while there are fewer than free_k3_after_views views: with few views it
    makes the distortion run wild where no corners are. uncertainty_scale multiplies all
    uncertainties (see the module docstring). with_leave_one_out: also the views'
    leave-one-out errors (for the final pass). Raises cv2.error if OpenCV cannot calibrate.
    """
    start = time.perf_counter()
    image_size = tuple(image_size)
    fix_k3 = len(object_points) < free_k3_after_views
    total_rms, camera_matrix, distortion, rvecs, tvecs = calibrate(
        object_points, image_points, image_size, fix_k3)
    used = [i for i in range(len(INTRINSICS)) if not (fix_k3 and i == K3)]
    terms = view_terms(object_points, image_points, camera_matrix, distortion, rvecs, tvecs,
                       used)
    covariance, noise = intrinsic_covariance(terms, used)
    covariance = covariance * uncertainty_scale ** 2
    predicted = leave_one_out(terms) if with_leave_one_out else None
    maps = uncertainty_maps(camera_matrix, distortion, covariance, used, map_grid(image_size))
    uncertainty = {name: float(np.nanmax(m)) if np.isfinite(m).any() else math.nan
                   for name, m in maps.items()}
    std = np.zeros(len(INTRINSICS))
    std[used] = np.sqrt(np.diag(covariance))
    tilts = np.array([board_tilt(rvec) for rvec in rvecs])
    folds = distortion_folds_in_image(camera_matrix, distortion, image_size)
    return CalibrationResult(
        image_size=image_size, views=len(object_points),
        points=sum(len(points) for points in object_points), k3_fixed=fix_k3,
        rms=float(total_rms),
        noise=noise, camera_matrix=camera_matrix, distortion=distortion, std_deviations=std,
        uncertainty_scale=uncertainty_scale,
        uncertainty=uncertainty,
        uncertainty_map=maps['total'], view_rms=np.array([rms(r) for r, _, _ in terms]),
        view_tilts=tilts, view_residuals=tuple(r.reshape(-1, 2) for r, _, _ in terms),
        leave_one_out=None if predicted is None else np.array([rms(r) for r in predicted]),
        leave_one_out_residuals=None if predicted is None else tuple(
            r.reshape(-1, 2) for r in predicted),
        # The tilt hints already say how to fix a weak focal length or principal point.
        hints=((FOLD_HINT,) if folds else ()) + (pose_hints(tilts)
                                                 or uncertainty_hints(uncertainty)),
        distortion_folds=folds,
        duration_s=time.perf_counter() - start)


def robust_z(values):
    """Robust z-scores: (value - median) / (1.4826 MAD), the MAD's scale of a normal spread."""
    values = np.asarray(values, np.float64)
    median = np.median(values)
    return (values - median) / max(1.4826 * np.median(np.abs(values - median)), 1e-12)


def robust_limit(values, z_limit):
    """Return the value whose robust_z() would be z_limit."""
    values = np.asarray(values, np.float64)
    median = np.median(values)
    return median + z_limit * max(1.4826 * np.median(np.abs(values - median)), 1e-12)


def uncertainty_ratio(after, before):
    """
    How much the mean uncertainty over the image grows from before to after.

    Over the cells both maps have; inf if after has none of before's.
    """
    both = np.isfinite(before.uncertainty_map)
    if not both.any():
        return 1.0
    after_values = after.uncertainty_map[both]
    if not np.isfinite(after_values).all():
        return math.inf
    return float(after_values.mean() / before.uncertainty_map[both].mean())


@dataclass(frozen=True)
class FlaggedView:
    """A view the outlier check flagged: removed, or kept as a suspect."""

    index: int                 # into all views
    z: float                   # robust z-score of its leave-one-out error when it was flagged
    error: float               # that leave-one-out error, px
    residuals: np.ndarray      # (N, 2) its corners' leave-one-out residuals, px
    uncertainty_ratio: float   # mean uncertainty over the image without it / with it


@dataclass(frozen=True)
class FinalResult:
    """The final calibration pass over all views (indices into them)."""

    initial: CalibrationResult  # from all views, with leave-one-out errors
    final: CalibrationResult    # from the kept views, in their order
    kept: tuple                 # indices of the views in final
    removed: tuple              # FlaggedView, in removal order
    suspects: tuple             # FlaggedView kept: removing it raised the uncertainty too much
    forced: tuple               # indices the pass was not allowed to remove (restored by the user)
    checked: bool               # False: too few views for the outlier check
    z_limit: float
    initial_limit: float        # leave-one-out error with z = z_limit among all views, px

    def view_residuals(self, index):
        """(N, 2) residuals of a view: in the final calibration, or leave-one-out if removed."""
        for removed in self.removed:
            if removed.index == index:
                return removed.residuals
        return self.final.view_residuals[self.kept.index(index)]


def final_calibration(object_points, image_points, image_size, free_k3_after_views,
                      uncertainty_scale, z_limit, min_views, max_uncertainty_increase,
                      max_removed_fraction, forced=(), progress=None):
    """
    Calibrate, remove clear outliers one by one, calibrate again; return a FinalResult.

    Outlier check (only with at least min_views views): the view with the highest leave-one-out
    error among those not yet checked is a candidate while its robust z-score among the kept
    views exceeds z_limit. It is removed if the calibration without it has at most
    max_uncertainty_increase times the mean uncertainty over the image, else kept as a suspect
    (no other view covers its region). At most max_removed_fraction of the views are removed,
    never one in forced. progress(text) is called before each calibration. Raises cv2.error if
    OpenCV cannot calibrate.
    """
    forced = tuple(sorted(set(forced)))
    count = len(object_points)

    def calibrate_subset(indices):
        return calibrate_views([object_points[i] for i in indices],
                               [image_points[i] for i in indices], image_size,
                               free_k3_after_views, uncertainty_scale, True)

    if progress:
        progress(f'calibrating all {count} views')
    initial = calibrate_subset(range(count))
    limit = robust_limit(initial.leave_one_out, z_limit)
    checked = count >= min_views
    result, kept, removed, suspects = initial, list(range(count)), [], []
    exempt = set(forced)
    while checked and len(removed) < int(max_removed_fraction * count):
        zs = robust_z(result.leave_one_out)
        candidates = [k for k, index in enumerate(kept) if index not in exempt]
        if not candidates:
            break
        worst = max(candidates, key=lambda k: zs[k])
        if zs[worst] <= z_limit:
            break
        index = kept[worst]
        exempt.add(index)
        if progress:
            progress(f'checking view #{index + 1}: calibrating without it')
        trial_kept = kept[:worst] + kept[worst + 1:]
        trial = calibrate_subset(trial_kept)
        flagged = FlaggedView(index, float(zs[worst]), float(result.leave_one_out[worst]),
                              result.leave_one_out_residuals[worst],
                              uncertainty_ratio(trial, result))
        if flagged.uncertainty_ratio > max_uncertainty_increase:
            suspects.append(flagged)
            continue
        removed.append(flagged)
        result, kept = trial, trial_kept
    return FinalResult(initial, result, tuple(kept), tuple(removed), tuple(suspects), forced,
                       checked, z_limit, limit)


def summary(result):
    """Readable multi-line summary of a FinalResult, e.g. for the log when it is saved."""
    final = result.final
    K, D = final.camera_matrix, final.distortion
    values = final.uncertainty_map[np.isfinite(final.uncertainty_map)]
    mean = float(values.mean()) if len(values) else math.nan
    removed = f', {len(result.removed)} removed' if result.removed else ''
    lines = [
        f'  image        {final.image_size[0]} x {final.image_size[1]}, plumb_bob',
        f'  views        {final.views} used{removed}' + (', k3 fixed' if final.k3_fixed else ''),
        f'  rms          {final.rms:.3f} px',
        f'  uncertainty  {mean:.2f} px mean, {final.uncertainty["total"]:.2f} px max',
        f'  fx fy        {K[0, 0]:10.3f} {K[1, 1]:10.3f}',
        f'  cx cy        {K[0, 2]:10.3f} {K[1, 2]:10.3f}',
        f'  k1 k2 k3     {D[0]:10.6f} {D[1]:10.6f} {D[4]:10.6f}',
        f'  p1 p2        {D[2]:10.6f} {D[3]:10.6f}',
    ]
    if final.distortion_folds:
        lines.append('  WARNING      the distortion model is not valid near the image corners')
    return '\n'.join(lines)
