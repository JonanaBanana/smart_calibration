"""
Camera models: pinhole with plumb_bob distortion, and fisheye (Kannala-Brandt).

Both map a ray to normalized distorted coordinates (x_d, y_d), then to pixels u = fx x_d + cx,
v = fy y_d + cy (no skew).

- plumb_bob: polynomial in r = tan(theta), theta the angle to the optical axis:
  x_d = x (1 + k1 r^2 + k2 r^4 + k3 r^6) + tangential terms. Rays in front of the camera only.
- fisheye: polynomial in theta: theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 +
  k4 theta^8), OpenCV's fisheye model. Also valid beyond 90 degrees (see fisheye_calibration).

A model folds over where its radius stops growing; pixels beyond the largest radius have no ray.
Real lenses do not fold, so a fold inside the image means the calibration is not valid there.
"""

import math

import cv2
import numpy as np

from smart_calibration import fisheye_calibration

_UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)
_POINTS_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-9)
# Bisection steps for the fisheye ray angle on [0, pi] (below 1e-12 rad).
_BISECTION_STEPS = 60


def _normalized(camera_matrix, pixels):
    """Return the normalized distorted coordinates (N, 2) of pixels (N, 2)."""
    K = np.asarray(camera_matrix, np.float64)
    return (np.asarray(pixels, np.float64).reshape(-1, 2) - K[:2, 2]) / [K[0, 0], K[1, 1]]


def _corner_radius(camera_matrix, image_size):
    """Return the largest normalized distorted radius of the image corners."""
    width, height = image_size
    corners = [(x, y) for x in (-0.5, width - 0.5) for y in (-0.5, height - 0.5)]
    return float(np.hypot(*_normalized(camera_matrix, corners).T).max())


def _smallest_positive_root(coefficients):
    """Return the smallest positive real root of a polynomial (highest power first), or inf."""
    roots = np.roots(coefficients)
    positive = [root.real for root in roots if abs(root.imag) < 1e-12 and root.real > 0]
    return min(positive) if positive else math.inf


class CameraModel:
    """A lens model: its parameters and the operations the calibration and the GUI need."""

    name = ''
    # Estimated parameters: fx, fy, cx, cy, then the distortion vector in OpenCV's order.
    parameters = ()
    # Indices into parameters held at 0 with fix_higher_order: k3 and higher.
    high_order = ()

    @property
    def groups(self):
        """Parameter groups for the uncertainty: {name: indices into parameters}."""
        return {'focal': (0, 1), 'centre': (2, 3),
                'distortion': tuple(range(4, len(self.parameters)))}

    def high_order_text(self):
        """Return e.g. 'k3' or 'k3, k4': the parameters held at 0 with few views."""
        return ', '.join(self.parameters[i] for i in self.high_order)

    def __repr__(self):
        return f'<camera model {self.name}>'


class PlumbBob(CameraModel):
    name = 'plumb_bob'
    parameters = ('fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3')
    high_order = (8,)

    def calibrate(self, object_points, image_points, image_size, fix_high_order):
        """Run cv2.calibrateCamera; return (rms, camera matrix, distortion, rvecs, tvecs)."""
        flags = cv2.CALIB_FIX_K3 if fix_high_order else 0
        rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, tuple(image_size), None, None, flags=flags)
        return rms, camera_matrix, distortion.ravel()[:5], rvecs, tvecs

    def project(self, object_points, rvec, tvec, camera_matrix, distortion):
        """Return the projections (N, 2) and their jacobians by the parameters and the pose."""
        projected, jacobian = cv2.projectPoints(
            np.asarray(object_points, np.float64).reshape(-1, 1, 3), rvec, tvec, camera_matrix,
            distortion)
        # Columns: rvec, tvec, fx, fy, cx, cy, k1, k2, p1, p2, k3.
        return projected.reshape(-1, 2), jacobian[:, 6:15], jacobian[:, :6]

    def rays(self, camera_matrix, distortion, pixels):
        """Return ray directions (N, 3) through pixels (N, 2), z = 1; NaN where there is none."""
        camera_matrix = np.asarray(camera_matrix, np.float64)
        distortion = np.asarray(distortion, np.float64)
        pixels = np.asarray(pixels, np.float64).reshape(-1, 1, 2)
        normalized = cv2.undistortPointsIter(pixels, camera_matrix, distortion, None, None,
                                             _UNDISTORT_CRITERIA).reshape(-1, 2)
        rays = np.column_stack([normalized, np.ones(len(normalized))])
        back, _ = cv2.projectPoints(rays.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                    camera_matrix, distortion)
        # Beyond a fold the iteration ends on a ray that projects elsewhere.
        rays[np.linalg.norm(back.reshape(-1, 2) - pixels.reshape(-1, 2), axis=1) > 1e-3] = np.nan
        return rays

    def ray_jacobian(self, camera_matrix, distortion, rays):
        """Return the jacobian (N, 2, parameters) of the pixels of fixed rays (N, 3), z > 0."""
        rays = np.asarray(rays, np.float64)
        points = (rays / rays[:, 2:]).reshape(-1, 1, 3)
        _, jacobian = cv2.projectPoints(points, np.zeros(3), np.zeros(3),
                                        np.asarray(camera_matrix, np.float64),
                                        np.asarray(distortion, np.float64))
        return jacobian[:, 6:15].reshape(-1, 2, len(self.parameters))

    def fold_radius(self, distortion):
        """
        Return the pinhole radius r = tan(theta) where the model folds over, or inf.

        Smallest r > 0 with d/dr [r (1 + k1 r^2 + k2 r^4 + k3 r^6)] = 0; tangential terms
        ignored.
        """
        k1, k2, k3 = (float(distortion[i]) if len(distortion) > i else 0.0 for i in (0, 1, 4))
        return math.sqrt(_smallest_positive_root([7 * k3, 5 * k2, 3 * k1, 1.0]))  # in r^2

    def folds_in_image(self, camera_matrix, distortion, image_size):
        """Return whether the model folds over inside the image: some pixels have no ray."""
        radius = self.fold_radius(distortion)
        if not math.isfinite(radius):
            return False
        k1, k2, k3 = (float(distortion[i]) if len(distortion) > i else 0.0 for i in (0, 1, 4))
        r2 = radius ** 2
        reach = radius * (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3)
        return bool(reach < _corner_radius(camera_matrix, image_size))

    def new_camera_matrix(self, camera_matrix, distortion, image_size, alpha):
        """Return the camera matrix of the undistorted view (cv2.getOptimalNewCameraMatrix)."""
        new, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion, image_size, alpha,
                                               image_size)
        return new

    def undistort_maps(self, camera_matrix, distortion, new_camera_matrix, size):
        """Return the remap tables (x, y) of an undistorted view of size (w, h)."""
        return cv2.initUndistortRectifyMap(camera_matrix, distortion, None, new_camera_matrix,
                                           size, cv2.CV_32FC1)

    def undistort_points(self, camera_matrix, distortion, pixels, new_camera_matrix):
        """Return where pixels (N, 2) land in the undistorted view."""
        return cv2.undistortPointsIter(
            np.asarray(pixels, np.float64).reshape(-1, 1, 2), camera_matrix, distortion, None,
            new_camera_matrix, _POINTS_CRITERIA).reshape(-1, 2)


class Fisheye(CameraModel):
    name = 'fisheye'
    parameters = ('fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4')
    high_order = (6, 7)

    distorted_radius = staticmethod(fisheye_calibration.distorted_radius)

    def calibrate(self, object_points, image_points, image_size, fix_high_order):
        """Return (rms, camera matrix, distortion, rvecs, tvecs); see fisheye_calibration."""
        return fisheye_calibration.calibrate(object_points, image_points, image_size,
                                             fix_high_order)

    def project(self, object_points, rvec, tvec, camera_matrix, distortion):
        """Return the projections (N, 2) and their jacobians by the parameters and the pose."""
        return fisheye_calibration.project_view(object_points, rvec, tvec, camera_matrix,
                                                distortion)

    def fold_angle(self, distortion):
        """Return the ray angle where the model folds over (d theta_d / d theta = 0), or inf."""
        k1, k2, k3, k4 = (float(v) for v in np.ravel(distortion)[:4])
        return math.sqrt(_smallest_positive_root(
            [9 * k4, 7 * k3, 5 * k2, 3 * k1, 1.0]))  # in theta^2

    def _angle_limit(self, distortion):
        """Return the largest ray angle with a pixel: the fold, at most 180 degrees."""
        return min(self.fold_angle(distortion), math.pi)

    def rays(self, camera_matrix, distortion, pixels):
        """Return unit ray directions (N, 3) through pixels (N, 2); NaN where there is none."""
        normalized = _normalized(camera_matrix, pixels)
        theta_d = np.hypot(normalized[:, 0], normalized[:, 1])
        limit = self._angle_limit(distortion)
        # theta_d is monotonic up to the limit.
        low, high = np.zeros_like(theta_d), np.full_like(theta_d, limit)
        for _ in range(_BISECTION_STEPS):
            middle = (low + high) / 2
            short = self.distorted_radius(middle, distortion) < theta_d
            low, high = np.where(short, middle, low), np.where(short, high, middle)
        theta = (low + high) / 2
        phi = np.arctan2(normalized[:, 1], normalized[:, 0])
        rays = np.column_stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                                np.cos(theta)])
        rays[theta_d > self.distorted_radius(limit, distortion) * (1 + 1e-12)] = np.nan
        return rays

    def ray_jacobian(self, camera_matrix, distortion, rays):
        """Return the jacobian (N, 2, parameters) of the pixels of fixed rays (N, 3)."""
        rays = np.asarray(rays, np.float64)
        theta = np.arctan2(np.hypot(rays[:, 0], rays[:, 1]), rays[:, 2])
        phi = np.arctan2(rays[:, 1], rays[:, 0])
        direction = np.column_stack([np.cos(phi), np.sin(phi)])  # (N, 2)
        theta_d = self.distorted_radius(theta, distortion)
        focal = np.array([camera_matrix[0][0], camera_matrix[1][1]])
        jacobian = np.zeros((len(rays), 2, len(self.parameters)))
        jacobian[:, 0, 0] = theta_d * direction[:, 0]
        jacobian[:, 1, 1] = theta_d * direction[:, 1]
        jacobian[:, 0, 2] = jacobian[:, 1, 3] = 1.0
        for i, power in enumerate((3, 5, 7, 9)):
            jacobian[:, :, 4 + i] = focal * direction * theta[:, None] ** power
        return jacobian

    def fold_radius(self, distortion):
        """Return the pinhole radius r = tan(theta) where the model folds over, or inf."""
        angle = self.fold_angle(distortion)
        return math.tan(angle) if angle < math.pi / 2 else math.inf

    def folds_in_image(self, camera_matrix, distortion, image_size):
        """Return whether the model folds over inside the image: some pixels have no ray."""
        angle = self.fold_angle(distortion)
        if angle >= math.pi:
            return False
        reach = self.distorted_radius(angle, distortion)
        return bool(reach < _corner_radius(camera_matrix, image_size))

    def new_camera_matrix(self, camera_matrix, distortion, image_size, alpha):
        """
        Return the camera matrix of the undistorted view.

        alpha is the balance of cv2.fisheye.estimateNewCameraMatrixForUndistortRectify: 1 shows
        the middle of every image edge, 0 only the region inside them.
        """
        return cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            np.asarray(camera_matrix, np.float64), np.asarray(distortion, np.float64),
            image_size, np.eye(3), balance=alpha)

    def undistort_maps(self, camera_matrix, distortion, new_camera_matrix, size):
        """Return the remap tables (x, y) of an undistorted view of size (w, h)."""
        return cv2.fisheye.initUndistortRectifyMap(
            np.asarray(camera_matrix, np.float64), np.asarray(distortion, np.float64),
            np.eye(3), new_camera_matrix, size, cv2.CV_32FC1)

    def undistort_points(self, camera_matrix, distortion, pixels, new_camera_matrix):
        """Return where pixels (N, 2) land in the undistorted view; NaN at 90 degrees or more."""
        rays = self.rays(camera_matrix, distortion, pixels)
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized = np.where(rays[:, 2:] > 0, rays[:, :2] / rays[:, 2:], np.nan)
        new = np.asarray(new_camera_matrix, np.float64)
        return normalized * [new[0, 0], new[1, 1]] + new[:2, 2]


PLUMB_BOB = PlumbBob()
FISHEYE = Fisheye()
CAMERA_MODELS = {model.name: model for model in (PLUMB_BOB, FISHEYE)}


def camera_model(name):
    """Return the CameraModel called name; raise ValueError for an unknown one."""
    try:
        return CAMERA_MODELS[name]
    except KeyError:
        raise ValueError(f'camera_model must be one of {tuple(CAMERA_MODELS)}, '
                         f'got {name!r}') from None
