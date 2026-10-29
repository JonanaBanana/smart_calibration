"""
Kannala-Brandt (fisheye) projection and calibration.

The projection uses theta = atan2(rho, z), so it also holds for rays beyond 90 degrees.

Calibration:
- Start: principal point at the image centre, an equidistant lens (k = 0) and the focal length
  from a scan over fields of view. Each view's pose comes from planar PnP (IPPE) in a camera
  frame rotated onto the view's mean ray.
- Levenberg-Marquardt over the intrinsics and all poses, the poses eliminated per view (Schur
  complement): first focal length and principal point, then the distortion.

K and D follow OpenCV's fisheye model; skew is 0.
"""

import math

import cv2
import numpy as np

# Parameter indices (jacobian columns): fx, fy, cx, cy, k1, k2, k3, k4.
FOCAL_AND_CENTRE = (0, 1, 2, 3)
K1_K2 = (4, 5)
K3_K4 = (6, 7)
PARAMETERS = 8
# Focal length scan: ray angle of the image corners, degrees.
SCAN_CORNER_DEG = np.geomspace(20.0, 130.0, 25)
# Starting poses use the corners within this angle of the view's mean ray.
PNP_MAX_DEG = 80.0
# Iterations per optimization stage.
MAX_ITERATIONS = 200
# Converged: an undamped step lowers the cost by less than this fraction.
CONVERGED = 1e-10
# Damping at which no step lowers the cost: a minimum.
MAX_DAMPING = 1e16


def distorted_radius(theta, distortion):
    """Return theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)."""
    k1, k2, k3, k4 = (float(v) for v in np.ravel(distortion)[:4])
    t2 = np.square(theta)
    return theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))


def project_points(points, camera_matrix, distortion, with_jacobian=True):
    """
    Return the pixels (N, 2) of camera-frame points (N, 3).

    with_jacobian: also d pixel / d point (N, 2, 3) and d pixel / d parameters (N, 2, 8).
    """
    K = np.asarray(camera_matrix, np.float64)
    D = np.ravel(np.asarray(distortion, np.float64))[:4]
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    rho = np.hypot(x, y)
    n2 = rho * rho + z * z
    theta = np.arctan2(rho, z)
    theta_d = distorted_radius(theta, D)
    # On the optical axis theta_d / rho -> 1 / z.
    axis = rho <= 1e-12 * np.sqrt(n2)
    safe_rho = np.where(axis, 1.0, rho)
    safe_z = np.where(axis, z, 1.0)
    scale = np.where(axis, 1.0 / safe_z, theta_d / safe_rho)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pixels = np.column_stack([fx * scale * x + cx, fy * scale * y + cy])
    if not with_jacobian:
        return pixels
    t2 = theta * theta
    slope = 1 + t2 * (3 * D[0] + t2 * (5 * D[1] + t2 * (7 * D[2] + t2 * 9 * D[3])))
    # d scale = slope / rho * d theta - theta_d / rho^2 * d rho
    dtheta = np.column_stack([z * x / (safe_rho * n2), z * y / (safe_rho * n2), -rho / n2])
    drho = np.column_stack([x / safe_rho, y / safe_rho, np.zeros_like(x)])
    dscale = (slope / safe_rho)[:, None] * dtheta - (theta_d / safe_rho ** 2)[:, None] * drho
    dscale[axis] = np.column_stack([np.zeros(axis.sum()), np.zeros(axis.sum()),
                                    -1 / safe_z[axis] ** 2])
    by_point = np.empty((len(points), 2, 3))
    by_point[:, 0] = fx * x[:, None] * dscale
    by_point[:, 1] = fy * y[:, None] * dscale
    by_point[:, 0, 0] += fx * scale
    by_point[:, 1, 1] += fy * scale
    by_parameters = np.zeros((len(points), 2, PARAMETERS))
    by_parameters[:, 0, 0] = scale * x
    by_parameters[:, 1, 1] = scale * y
    by_parameters[:, 0, 2] = by_parameters[:, 1, 3] = 1.0
    ux, uy = np.where(axis, 0.0, x / safe_rho), np.where(axis, 0.0, y / safe_rho)
    for i, power in enumerate((3, 5, 7, 9)):
        by_parameters[:, 0, 4 + i] = fx * ux * theta ** power
        by_parameters[:, 1, 4 + i] = fy * uy * theta ** power
    return pixels, by_point, by_parameters


def project_view(object_points, rvec, tvec, camera_matrix, distortion, with_jacobian=True):
    """
    Return the pixels (N, 2) of board points (N, 3) in a pose.

    with_jacobian: also the jacobians (2N, 8) by the parameters and (2N, 6) by rvec, tvec,
    rows interleaved x, y as cv2.projectPoints.
    """
    points = np.asarray(object_points, np.float64).reshape(-1, 3)
    R, dR = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    camera = points @ R.T + np.ravel(tvec)
    if not with_jacobian:
        return project_points(camera, camera_matrix, distortion, False)
    pixels, by_point, by_parameters = project_points(camera, camera_matrix, distortion)
    # dR (3 x 9): d R_ij / d r_k at [k, 3 i + j].
    by_rotation = np.einsum('kij,nj->nik', dR.reshape(3, 3, 3), points)  # (N, 3, 3)
    by_pose = np.concatenate([by_point @ by_rotation, by_point], axis=2)
    return pixels, by_parameters.reshape(-1, PARAMETERS), by_pose.reshape(-1, 6)


def _turn_to_axis(direction):
    """Return the rotation (3, 3) turning a unit direction onto the optical axis (0, 0, 1)."""
    helper = np.array([0.0, 1.0, 0.0]) if abs(direction[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(helper, direction)
    x /= np.linalg.norm(x)
    return np.array([x, np.cross(direction, x), direction])


def _equidistant_pose(object_points, image_points, camera_matrix):
    """
    Return the pose (rvec and tvec stacked, (6,)) of a view under an equidistant lens, or None.

    Planar PnP (IPPE) in a camera frame rotated onto the view's mean ray, on the corners within
    PNP_MAX_DEG of it.
    """
    d = (image_points - camera_matrix[:2, 2]) / [camera_matrix[0, 0], camera_matrix[1, 1]]
    theta = np.hypot(d[:, 0], d[:, 1])  # equidistant: the distorted radius is the angle
    phi = np.arctan2(d[:, 1], d[:, 0])
    valid = theta < math.pi
    rays = np.column_stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                            np.cos(theta)])[valid]
    mean = rays.sum(axis=0)
    if len(rays) < 4 or np.linalg.norm(mean) < 1e-9:
        return None
    turn = _turn_to_axis(mean / np.linalg.norm(mean))
    turned = rays @ turn.T
    use = turned[:, 2] > math.cos(math.radians(PNP_MAX_DEG))
    if use.sum() < 4:
        return None
    points = object_points[valid][use]
    ok, rvec, tvec = cv2.solvePnP(points, turned[use, :2] / turned[use, 2:], np.eye(3), None,
                                  flags=cv2.SOLVEPNP_IPPE)
    if not ok or not (np.isfinite(rvec).all() and np.isfinite(tvec).all()):
        return None
    R_turned = cv2.Rodrigues(rvec)[0]
    if np.any((points @ R_turned.T + tvec.ravel())[:, 2] <= 0):
        return None
    R, t = turn.T @ R_turned, turn.T @ tvec.ravel()
    return np.concatenate([cv2.Rodrigues(R)[0].ravel(), t])


def _squared_error(objects, images, poses, camera_matrix, distortion):
    return sum(float(np.sum((p - project_view(P, pose[:3], pose[3:], camera_matrix, distortion,
                                              False)) ** 2))
               for P, p, pose in zip(objects, images, poses) if pose is not None)


def _start(objects, images, image_size):
    """Return (camera matrix, poses) of the best focal length candidate; poses None if failed."""
    width, height = image_size
    centre = (width / 2 - 0.5, height / 2 - 0.5)
    half_diagonal = math.hypot(width / 2, height / 2)
    best = None
    for corner_deg in SCAN_CORNER_DEG:
        f = half_diagonal / math.radians(corner_deg)
        K = np.array([[f, 0.0, centre[0]], [0.0, f, centre[1]], [0.0, 0.0, 1.0]])
        poses = [_equidistant_pose(P, p, K) for P, p in zip(objects, images)]
        corners = sum(len(p) for p, pose in zip(images, poses) if pose is not None)
        if not corners:
            continue
        error = _squared_error(objects, images, poses, K, np.zeros(4)) / corners
        score = (-sum(pose is not None for pose in poses), error)
        if math.isfinite(error) and (best is None or score < best[0]):
            best = (score, K, poses)
    if best is None:
        raise cv2.error('fisheye calibration: no view gives a starting pose')
    return best[1], best[2]


def _values(camera_matrix, distortion):
    K = camera_matrix
    return np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2], *np.ravel(distortion)[:4]])


def _matrices(values):
    K = np.array([[values[0], 0.0, values[2]], [0.0, values[1], values[3]], [0.0, 0.0, 1.0]])
    return K, values[4:].copy()


def levenberg_marquardt(objects, images, camera_matrix, distortion, poses, free):
    """
    Minimize the squared reprojection error over the parameters free and all poses.

    poses: per view rvec and tvec stacked (6,). Returns (camera matrix, distortion, poses,
    squared error). Raises cv2.error if it has not converged after MAX_ITERATIONS.
    """
    free = list(free)
    values = _values(camera_matrix, distortion)
    poses = [np.asarray(pose, np.float64) for pose in poses]

    def linearize(values, poses):
        K, D = _matrices(values)
        terms, error = [], 0.0
        for P, p, pose in zip(objects, images, poses):
            projected, by_parameters, by_pose = project_view(P, pose[:3], pose[3:], K, D)
            r = (p - projected).ravel()
            terms.append((r, by_parameters[:, free], by_pose))
            error += float(r @ r)
        return terms, error

    terms, error = linearize(values, poses)
    damping, growth = 1e-3, 2.0  # relative to diag(J^T J) (Marquardt)
    for _ in range(MAX_ITERATIONS):
        U = sum(a.T @ a for _, a, _ in terms)
        g_a = sum(a.T @ r for r, a, _ in terms)
        V = [b.T @ b for _, _, b in terms]
        W = [a.T @ b for _, a, b in terms]
        g_b = [b.T @ r for r, _, b in terms]
        converged = False
        while True:
            try:
                V_inv = [np.linalg.inv(v + damping * np.diag(np.diag(v))) for v in V]
                S = U + damping * np.diag(np.diag(U)) - sum(
                    w @ vi @ w.T for w, vi in zip(W, V_inv))
                rhs = g_a - sum(w @ vi @ g for w, vi, g in zip(W, V_inv, g_b))
                step = np.linalg.solve(S, rhs)
            except np.linalg.LinAlgError:
                step = None
            if step is not None:
                pose_steps = [vi @ (g - w.T @ step) for w, vi, g in zip(W, V_inv, g_b)]
                new_values = values.copy()
                new_values[free] += step
                new_poses = [pose + d for pose, d in zip(poses, pose_steps)]
                new_terms, new_error = linearize(new_values, new_poses)
                if math.isfinite(new_error) and new_error < error:
                    # Gain ratio (Madsen, Nielsen & Tingleff 2004, eq. 3.14).
                    h = np.concatenate([step, *pose_steps])
                    diagonal = np.concatenate([np.diag(U), *[np.diag(v) for v in V]])
                    predicted = float(h @ (damping * diagonal * h
                                           + np.concatenate([g_a, *g_b])))
                    gain = (error - new_error) / max(predicted, 1e-300)
                    converged = damping < 1.0 and error - new_error < CONVERGED * error
                    damping *= max(1 / 3, 1 - (2 * gain - 1) ** 3)
                    growth = 2.0
                    values, poses, terms, error = new_values, new_poses, new_terms, new_error
                    break
            damping *= growth
            growth *= 2
            if damping > MAX_DAMPING:
                converged = True
                break
        if converged:
            break
    else:
        names = ', '.join(('fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4')[i] for i in free)
        raise cv2.error(f'fisheye calibration: not converged after {MAX_ITERATIONS} '
                        f'iterations (estimating {names})')
    K, D = _matrices(values)
    return K, D, poses, error


def calibrate(object_points, image_points, image_size, fix_k3_k4):
    """
    Calibrate a fisheye camera; return (rms, camera matrix, distortion (4,), rvecs, tvecs).

    Like cv2.fisheye.calibrate with CALIB_FIX_SKEW, plus CALIB_FIX_K3 | CALIB_FIX_K4 with
    fix_k3_k4. Raises cv2.error if a view gets no starting pose or the optimization does not
    converge.
    """
    objects = [np.asarray(P, np.float64).reshape(-1, 3) for P in object_points]
    images = [np.asarray(p, np.float64).reshape(-1, 2) for p in image_points]
    K, poses = _start(objects, images, image_size)
    no_distortion = np.zeros(4)
    # Views without a starting pose: retry with the focal length fitted to the others.
    have = [i for i, pose in enumerate(poses) if pose is not None]
    if len(have) < len(poses):
        K, _, fitted, _ = levenberg_marquardt(
            [objects[i] for i in have], [images[i] for i in have], K, no_distortion,
            [poses[i] for i in have], FOCAL_AND_CENTRE)
        for i, pose in zip(have, fitted):
            poses[i] = pose
        poses = [_equidistant_pose(P, p, K) if pose is None else pose
                 for P, p, pose in zip(objects, images, poses)]
        missing = [i for i, pose in enumerate(poses) if pose is None]
        if missing:
            raise cv2.error('fisheye calibration: no starting pose for view '
                            + ', '.join(f'#{i + 1}' for i in missing))
    K, D, poses, _ = levenberg_marquardt(objects, images, K, no_distortion, poses,
                                         FOCAL_AND_CENTRE)
    free = FOCAL_AND_CENTRE + K1_K2 + (() if fix_k3_k4 else K3_K4)
    K, D, poses, error = levenberg_marquardt(objects, images, K, D, poses, free)
    rms = math.sqrt(error / sum(len(p) for p in images))
    return (rms, K, D, tuple(pose[:3].reshape(3, 1) for pose in poses),
            tuple(pose[3:].reshape(3, 1) for pose in poses))
