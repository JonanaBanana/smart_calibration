"""
Calibration from the kept views, its uncertainty, and the final pass.

The lens model is a camera_models.CameraModel.

The covariance of the intrinsics comes from the Gauss-Newton normal equations, the point noise
from the residuals. It is propagated to an uncertainty map over the image. It assumes independent
corner errors, so it underestimates the true error (uncertainty_scale) and a small value alone does
not prove a good calibration.

The final pass removes views the other views predict badly (leave-one-out error), unless removing
one raises the uncertainty too much: then no other view covers its region, and it is kept.
"""

from dataclasses import dataclass
import math
import time

import cv2
import numpy as np

from smart_calibration.camera_models import PLUMB_BOB

# Uncertainty map: this many cells across the image (rows follow the aspect ratio).
MAP_COLS = 32
# Hints: smallest tilt that counts, and the tolerance for a tilt axis along an image axis.
MIN_TILT_DEG = 15.0
MIN_TILTED_VIEWS = 2
AXIS_ALIGNED_DEG = 10.0


@dataclass(frozen=True)
class CalibrationResult:
    """One calibration and its uncertainty (scaled by uncertainty_scale)."""

    camera_model: object          # camera_models.CameraModel
    image_size: tuple             # (width, height)
    views: int
    points: int
    k3_fixed: bool                # k3 and higher held at 0 (camera_model.high_order)
    rms: float                    # reprojection error, px
    noise: float                  # point noise estimated from the residuals, px
    camera_matrix: np.ndarray     # (3, 3)
    distortion: np.ndarray        # plumb_bob (5,) k1, k2, p1, p2, k3; fisheye (4,) k1-k4
    std_deviations: np.ndarray    # per camera_model.parameters, scaled; 0 for a fixed parameter
    uncertainty_scale: float
    uncertainty: dict             # 'total' and each group: largest map value, px
    uncertainty_map: np.ndarray   # (rows, cols) px, NaN where the model has no ray
    view_rms: np.ndarray          # (views,) reprojection error per view, px
    view_tilts: np.ndarray        # (views, 2) board_tilt() of each view, degrees
    view_residuals: tuple         # per view (N, 2): detected minus projected corners, px
    # Only with with_leave_one_out, else None: per view error (px) and residuals (N, 2).
    leave_one_out: np.ndarray
    leave_one_out_residuals: tuple
    hints: tuple                  # strings: view types that are missing
    distortion_folds: bool        # CameraModel.folds_in_image()
    duration_s: float


def view_terms(model, object_points, image_points, camera_matrix, distortion, rvecs, tvecs,
               used):
    """
    Return per view (r, A, B): residuals (2N,) and jacobians by the intrinsics and by the pose.

    used: indices into model.parameters of the estimated parameters. r: detected minus projected.
    """
    terms = []
    for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, by_parameters, by_pose = model.project(obj, rvec, tvec, camera_matrix,
                                                          distortion)
        residuals = (img.reshape(-1, 2) - projected).ravel()
        terms.append((residuals, by_parameters[:, used], by_pose))
    return terms


def _without_pose(b, vector):
    """Return vector minus the part a change of the view's pose explains."""
    return vector - b @ np.linalg.solve(b.T @ b, b.T @ vector)


def intrinsic_covariance(terms, used):
    """
    Return the covariance of the estimated intrinsics and the point noise.

    terms: view_terms(). The pose parameters are eliminated (Schur complement).
    """
    reduced = sum(a.T @ _without_pose(b, a) for _, a, b in terms)
    squared = sum(float(r @ r) for r, _, _ in terms)
    observations = sum(len(r) for r, _, _ in terms)
    dof = max(observations - len(used) - 6 * len(terms), 1)
    noise = math.sqrt(squared / dof)
    return noise ** 2 * np.linalg.pinv(reduced), noise


def leave_one_out(terms):
    """
    Return per view its residuals when the other views calibrate and only its pose is fitted.

    Linearized: one Gauss-Newton step from the full solution without the view.
    """
    reduced = [a.T @ _without_pose(b, a) for _, a, b in terms]
    gradients = [a.T @ _without_pose(b, r) for r, a, b in terms]
    total = sum(reduced)
    result = []
    for (r, a, b), view_reduced, gradient in zip(terms, reduced, gradients):
        step = -np.linalg.pinv(total - view_reduced) @ gradient
        result.append(_without_pose(b, r - a @ step))
    return result


def rms(residuals):
    """Return the RMS length of interleaved 2-D residuals."""
    return math.sqrt(float(residuals @ residuals) / (len(residuals) / 2))


def map_grid(image_size, cols=MAP_COLS):
    """Return the pixel positions (rows, cols, 2) of the uncertainty map's cell centres."""
    width, height = image_size
    rows = max(round(cols * height / width), 1)
    xs = (np.arange(cols) + 0.5) * width / cols - 0.5
    ys = (np.arange(rows) + 0.5) * height / rows - 0.5
    return np.stack(np.meshgrid(xs, ys), -1)


def projection_covariances(model, camera_matrix, distortion, covariance, used, pixels):
    """
    Return the (N, 2, 2) covariance of where the model projects the ray through each pixel.

    NaN where the model has no ray for the pixel.
    """
    rays = model.rays(camera_matrix, distortion, pixels)
    result = np.full((len(rays), 2, 2), np.nan)
    valid = np.isfinite(rays).all(axis=1)
    if valid.any():
        j = model.ray_jacobian(camera_matrix, distortion, rays[valid])[:, :, used]
        result[valid] = j @ covariance @ j.transpose(0, 2, 1)
    return result


def uncertainty_maps(model, camera_matrix, distortion, covariance, used, grid):
    """
    Return {'total' and each model.groups key: (rows, cols) map} of the RMS projection error, px.

    A group's map takes the other parameters as exact.
    """
    covariances = {'total': covariance}
    for group, members in model.groups.items():
        keep = [k for k, i in enumerate(used) if i in members]
        partial = np.zeros_like(covariance)
        partial[np.ix_(keep, keep)] = covariance[np.ix_(keep, keep)]
        covariances[group] = partial
    pixels = grid.reshape(-1, 2)
    return {name: np.sqrt(np.trace(projection_covariances(
        model, camera_matrix, distortion, c, used, pixels), axis1=1, axis2=2)).reshape(
            grid.shape[:2])
        for name, c in covariances.items()}


def board_tilt(rvec):
    """
    Return (tilt, tilt axis) of a board pose in degrees.

    tilt: angle between board normal and optical axis. tilt axis: image direction of the board
    line at constant depth, in [0, 180); 0 horizontal, 90 vertical.
    """
    normal = cv2.Rodrigues(np.asarray(rvec, np.float64))[0][:, 2]
    tilt = math.degrees(math.acos(min(abs(normal[2]), 1.0)))
    axis = (math.degrees(math.atan2(normal[1], normal[0])) + 90.0) % 180.0
    return tilt, axis


def axis_offset(axis):
    """Return the angle of a tilt axis to the nearest image axis: 0 aligned, 45 diagonal."""
    off = axis % 90.0
    return min(off, 90.0 - off)


def pose_hints(tilts):
    """
    Return hints about missing view types from (views, 2) board_tilt() values.

    Untilted boards leave the focal length undetermined; tilts about one image axis only leave
    the other axis' focal length and principal point weak (Sturm & Maybank).
    """
    tilts = np.asarray(tilts, np.float64).reshape(-1, 2)
    tilted = tilts[tilts[:, 0] >= MIN_TILT_DEG]
    if len(tilted) < MIN_TILTED_VIEWS:
        return (f'Tilt the board more ({MIN_TILT_DEG:.0f}+ deg).',)
    hints = []
    tilt, axis = np.radians(tilted[:, 0]), np.radians(tilted[:, 1])
    about_vertical = np.degrees(np.arctan(np.tan(tilt) * np.abs(np.sin(axis))))
    about_horizontal = np.degrees(np.arctan(np.tan(tilt) * np.abs(np.cos(axis))))
    if about_vertical.max() < MIN_TILT_DEG:
        hints.append('Tilt the left or right edge towards the camera.')
    if about_horizontal.max() < MIN_TILT_DEG:
        hints.append('Tilt the top or bottom edge towards the camera.')
    if all(axis_offset(a) < AXIS_ALIGNED_DEG for a in tilted[:, 1]):
        hints.append('Also tilt about a diagonal.')
    return tuple(hints)


FOLD_HINT = 'Add views in the image corners.'


def uncertainty_hints(uncertainty):
    """Return a hint if the focal length or the principal point is the least certain group."""
    groups = {g: value for g, value in uncertainty.items()
              if g != 'total' and math.isfinite(value)}
    if not groups:
        return ()
    weakest = max(groups, key=groups.get)
    if weakest == 'focal':
        return ('Tilt the board more (20-45 deg): focal length is uncertain.',)
    if weakest == 'centre':
        return ('Tilt in other directions: image centre is uncertain.',)
    return ()


def calibrate_views(object_points, image_points, image_size, free_k3_after_views,
                    uncertainty_scale, with_leave_one_out=False, model=PLUMB_BOB):
    """
    Calibrate with a camera model and estimate the uncertainty; return a CalibrationResult.

    k3 (and k4) stay 0 below free_k3_after_views views. Raises cv2.error if OpenCV cannot
    calibrate.
    """
    start = time.perf_counter()
    image_size = tuple(image_size)
    fix_k3 = len(object_points) < free_k3_after_views
    total_rms, camera_matrix, distortion, rvecs, tvecs = model.calibrate(
        object_points, image_points, image_size, fix_k3)
    used = [i for i in range(len(model.parameters)) if not (fix_k3 and i in model.high_order)]
    terms = view_terms(model, object_points, image_points, camera_matrix, distortion, rvecs,
                       tvecs, used)
    covariance, noise = intrinsic_covariance(terms, used)
    covariance = covariance * uncertainty_scale ** 2
    predicted = leave_one_out(terms) if with_leave_one_out else None
    maps = uncertainty_maps(model, camera_matrix, distortion, covariance, used,
                            map_grid(image_size))
    uncertainty = {name: float(np.nanmax(m)) if np.isfinite(m).any() else math.nan
                   for name, m in maps.items()}
    std = np.zeros(len(model.parameters))
    std[used] = np.sqrt(np.diag(covariance))
    tilts = np.array([board_tilt(rvec) for rvec in rvecs])
    folds = model.folds_in_image(camera_matrix, distortion, image_size)
    return CalibrationResult(
        camera_model=model, image_size=image_size, views=len(object_points),
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
        hints=((FOLD_HINT,) if folds else ()) + (pose_hints(tilts)
                                                 or uncertainty_hints(uncertainty)),
        distortion_folds=folds,
        duration_s=time.perf_counter() - start)


def robust_z(values):
    """Return robust z-scores: (value - median) / (1.4826 MAD)."""
    values = np.asarray(values, np.float64)
    median = np.median(values)
    return (values - median) / max(1.4826 * np.median(np.abs(values - median)), 1e-12)


def robust_limit(values, z_limit):
    """Return the value whose robust_z() would be z_limit."""
    values = np.asarray(values, np.float64)
    median = np.median(values)
    return median + z_limit * max(1.4826 * np.median(np.abs(values - median)), 1e-12)


def uncertainty_ratio(after, before):
    """Return the ratio of the mean uncertainty over the image, after / before (inf: no ray)."""
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
    forced: tuple               # indices never removed (restored by the user)
    checked: bool               # False: too few views for the outlier check
    z_limit: float
    initial_limit: float        # leave-one-out error with z = z_limit among all views, px

    def view_residuals(self, index):
        """Return a view's residuals (N, 2): final, or leave-one-out if removed."""
        for removed in self.removed:
            if removed.index == index:
                return removed.residuals
        return self.final.view_residuals[self.kept.index(index)]


def final_calibration(object_points, image_points, image_size, free_k3_after_views,
                      uncertainty_scale, z_limit, min_views, max_uncertainty_increase,
                      max_removed_fraction, forced=(), progress=None, model=PLUMB_BOB):
    """
    Calibrate, remove clear outliers one at a time, and return a FinalResult.

    With at least min_views views, the view with the highest leave-one-out z-score above z_limit
    is removed if that raises the mean uncertainty by at most max_uncertainty_increase, else
    kept as a suspect. At most max_removed_fraction of the views; never one in forced.
    progress(text) is called before each calibration. model: the camera model.
    """
    forced = tuple(sorted(set(forced)))
    count = len(object_points)

    def calibrate_subset(indices):
        return calibrate_views([object_points[i] for i in indices],
                               [image_points[i] for i in indices], image_size,
                               free_k3_after_views, uncertainty_scale, True, model)

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
    """Return a multi-line summary of a FinalResult."""
    final = result.final
    model = final.camera_model
    K, D = final.camera_matrix, final.distortion
    values = final.uncertainty_map[np.isfinite(final.uncertainty_map)]
    mean = float(values.mean()) if len(values) else math.nan
    removed = f', {len(result.removed)} removed' if result.removed else ''
    distortion = dict(zip(model.parameters[4:], D))
    radial = [name for name in distortion if name.startswith('k')]
    tangential = [name for name in distortion if name.startswith('p')]
    lines = [
        f'  image        {final.image_size[0]} x {final.image_size[1]}, {model.name}',
        f'  views        {final.views} used{removed}'
        + (f', {model.high_order_text()} fixed' if final.k3_fixed else ''),
        f'  rms          {final.rms:.3f} px',
        f'  uncertainty  {mean:.2f} px mean, {final.uncertainty["total"]:.2f} px max',
        f'  fx fy        {K[0, 0]:10.3f} {K[1, 1]:10.3f}',
        f'  cx cy        {K[0, 2]:10.3f} {K[1, 2]:10.3f}',
    ]
    for names in (radial, tangential):
        if names:
            lines.append(f'  {" ".join(names):<13}'
                         + ' '.join(f'{distortion[name]:10.6f}' for name in names))
    if final.distortion_folds:
        lines.append('  WARNING      the distortion model is not valid near the image corners')
    return '\n'.join(lines)
