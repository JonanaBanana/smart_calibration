import math

import cv2
import numpy as np
import pytest

from smart_calibration.fisheye_calibration import (
    _equidistant_pose, calibrate, distorted_radius, project_points, project_view)

SIZE = (1280, 800)
# A 220 degree lens (diagonal), monotonic beyond 180 degrees.
WIDE_D = np.array([0.05, -0.01, 0.0015, -0.0001])
WIDE_K = np.array([[356.4, 0.0, 641.8], [0.0, 355.7, 397.8], [0.0, 0.0, 1.0]])
# A 260 degree lens: the image corners 130 degrees off the axis.
WIDEST_D = np.array([0.02, -0.003, 0.0, 0.0])
WIDEST_K = np.array([[325.6, 0.0, 641.8], [0.0, 325.0, 397.8], [0.0, 0.0, 1.0]])
# A 60 degree lens following tan(theta): an ordinary lens.
NARROW_D = np.array([0.33333, 0.13352, 0.05211, 0.02919])
NARROW_K = np.array([[1309.8, 0.0, 641.8], [0.0, 1307.2, 397.8], [0.0, 0.0, 1.0]])
# A 140 degree lens.
MODERATE_D = np.array([0.02, -0.005, 0.0, 0.0])
MODERATE_K = np.array([[607.6, 0.0, 641.8], [0.0, 606.4, 397.8], [0.0, 0.0, 1.0]])
# 12 x 9 corners, 50 mm squares.
BOARD = np.zeros((108, 3))
BOARD[:, :2] = np.mgrid[0:9, 0:12].reshape(2, -1).T[:, ::-1] * 0.05


def numeric(function, values, steps):
    """Central differences of function (returning an array) by each of values."""
    columns = []
    for j, step in enumerate(steps):
        up, down = values.copy(), values.copy()
        up[j] += step
        down[j] -= step
        columns.append((function(up) - function(down)).ravel() / (2 * step))
    return np.column_stack(columns)


def wide_views(K, D, count, seed, max_deg=85.0, noise=0.2):
    """
    Return views of the board facing the camera, up to max_deg off the axis.

    Also per view the corners' ray angles, degrees.
    """
    rng = np.random.default_rng(seed)
    objects, images, angles = [], [], []
    while len(objects) < count:
        cos_theta = rng.uniform(math.cos(math.radians(max_deg)), 1.0)
        phi = rng.uniform(0, 2 * math.pi)
        direction = np.array([math.sqrt(1 - cos_theta ** 2) * math.cos(phi),
                              math.sqrt(1 - cos_theta ** 2) * math.sin(phi), cos_theta])
        # Face the camera (board normal along the ray), then tilt up to 45 degrees.
        x = np.cross([0.0, 1.0, 0.0], direction)
        x /= np.linalg.norm(x)
        R = np.column_stack([x, np.cross(direction, x), direction])
        axis = rng.uniform(0, 2 * math.pi)
        R = R @ cv2.Rodrigues(np.array([math.cos(axis), math.sin(axis), 0.0])
                              * math.radians(rng.uniform(0, 45)))[0]
        tvec = direction * rng.uniform(0.35, 1.0) - R @ BOARD.mean(axis=0)
        camera = BOARD @ R.T + tvec
        pixels = project_points(camera, K, D, False)
        rays = camera / np.linalg.norm(camera, axis=1, keepdims=True)
        grazing = np.degrees(np.arccos(np.abs(rays @ R[:, 2])))
        keep = (((pixels >= 10) & (pixels < np.array(SIZE) - 11)).all(axis=1)
                & (grazing < 75))
        if keep.sum() < 25:  # as the detector's 5 x 5 minimum
            continue
        objects.append(BOARD[keep].astype(np.float32).reshape(-1, 1, 3))
        images.append((pixels[keep] + rng.normal(0, noise, (keep.sum(), 2)))
                      .astype(np.float32).reshape(-1, 1, 2))
        angles.append(np.degrees(np.arctan2(np.hypot(camera[keep, 0], camera[keep, 1]),
                                            camera[keep, 2])))
    return objects, images, angles


def test_projection_is_opencvs_in_front_of_the_camera():
    rvec, tvec = np.array([0.4, -0.3, 0.2]), np.array([-0.2, -0.1, 0.4])
    pixels, _, _ = project_view(BOARD, rvec, tvec, WIDE_K, WIDE_D)
    expected, _ = cv2.fisheye.projectPoints(BOARD.reshape(-1, 1, 3), rvec, tvec, WIDE_K, WIDE_D)
    np.testing.assert_allclose(pixels, expected.reshape(-1, 2), atol=1e-9)


def test_projection_beyond_90_degrees_follows_the_model():
    theta = np.radians([95.0, 110.0, 130.0])
    phi = np.radians([20.0, 200.0, -70.0])
    points = np.column_stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                              np.cos(theta)]) * 2.0
    r = distorted_radius(theta, WIDE_D)
    expected = np.column_stack([WIDE_K[0, 0] * r * np.cos(phi) + WIDE_K[0, 2],
                                WIDE_K[1, 1] * r * np.sin(phi) + WIDE_K[1, 2]])
    np.testing.assert_allclose(project_points(points, WIDE_K, WIDE_D, False), expected,
                               atol=1e-9)


def test_jacobians_match_finite_differences():
    # In front of the camera, beyond 90 degrees, and on the optical axis.
    rvec, tvec = np.array([0.4, -0.3, 0.2]), np.array([-0.2, -0.1, 0.4])
    _, by_parameters, by_pose = project_view(BOARD, rvec, tvec, WIDE_K, WIDE_D)
    values = np.array([*WIDE_K[[0, 1, 0, 1], [0, 1, 2, 2]], *WIDE_D])

    def by_values(v):
        K = np.array([[v[0], 0, v[2]], [0, v[1], v[3]], [0, 0, 1]])
        return project_view(BOARD, rvec, tvec, K, v[4:], False)
    np.testing.assert_allclose(
        by_parameters, numeric(by_values, values, 1e-6 * np.maximum(1, np.abs(values))),
        rtol=1e-5, atol=1e-5)
    pose = np.concatenate([rvec, tvec])
    np.testing.assert_allclose(
        by_pose, numeric(lambda v: project_view(BOARD, v[:3], v[3:], WIDE_K, WIDE_D, False),
                         pose, np.full(6, 1e-7)), rtol=1e-4, atol=1e-3)
    points = np.array([[1.0, 0.2, -0.3], [-0.5, 0.8, -0.1], [0.0, 0.0, 0.5]])
    _, by_point, _ = project_points(points, WIDE_K, WIDE_D)
    for point, jacobian in zip(points, by_point):
        expected = numeric(lambda v: project_points(v.reshape(1, 3), WIDE_K, WIDE_D, False),
                           point.copy(), np.full(3, 1e-7))
        np.testing.assert_allclose(jacobian, expected, rtol=1e-5, atol=1e-4)


def test_calibrates_a_lens_wider_than_180_degrees():
    objects, images, angles = wide_views(WIDE_K, WIDE_D, 40, seed=0)
    # Corners beyond 90 degrees, which OpenCV's projection mirrors.
    assert np.concatenate(angles).max() > 95
    rms, K, D, rvecs, tvecs = calibrate(objects, images, SIZE, False)
    assert len(rvecs) == len(tvecs) == 40 and rvecs[0].shape == (3, 1)
    assert rms < 0.35  # 2-D noise of 0.2 px per axis: RMS length 0.28
    np.testing.assert_allclose(K, WIDE_K, atol=0.5)
    theta = np.radians(np.linspace(5, 100, 20))
    np.testing.assert_allclose(K[1, 1] * distorted_radius(theta, D),
                               WIDE_K[1, 1] * distorted_radius(theta, WIDE_D), atol=0.5)


def test_starting_pose_of_a_view_beyond_90_degrees():
    # Equidistant lens: the start's model is exact, and so must be the pose.
    K = WIDEST_K
    for direction_deg in (20.0, 100.0, 125.0):
        angle = math.radians(direction_deg)
        direction = np.array([math.sin(angle), 0.0, math.cos(angle)])
        x = np.cross([0.0, 1.0, 0.0], direction)
        R = np.column_stack([x, np.cross(direction, x), direction])
        tvec = direction * 0.8 - R @ BOARD.mean(axis=0)
        camera = BOARD @ R.T + tvec
        angles = np.degrees(np.arctan2(np.hypot(camera[:, 0], camera[:, 1]), camera[:, 2]))
        pixels = project_points(camera, K, np.zeros(4), False)
        pose = _equidistant_pose(BOARD, pixels, K)
        assert pose is not None, direction_deg
        np.testing.assert_allclose(cv2.Rodrigues(pose[:3])[0], R, atol=1e-6)
        np.testing.assert_allclose(pose[3:], tvec, atol=1e-6)
    assert angles.min() > 90  # the last view lies entirely behind the camera's plane


def test_calibrates_a_260_degree_lens_with_views_in_its_outer_ring():
    objects, images, angles = wide_views(WIDEST_K, WIDEST_D, 40, seed=4, max_deg=115.0)
    # Views entirely beyond 80 degrees, one entirely beyond 90.
    assert sum(a.min() > 80 for a in angles) >= 3 and any(a.min() > 90 for a in angles)
    rms, K, D, _, _ = calibrate(objects, images, SIZE, False)
    assert rms < 0.35
    np.testing.assert_allclose(K, WIDEST_K, atol=0.5)
    theta = np.radians(np.linspace(5, 125, 25))
    np.testing.assert_allclose(K[1, 1] * distorted_radius(theta, D),
                               WIDEST_K[1, 1] * distorted_radius(theta, WIDEST_D), atol=0.5)


def test_calibrates_a_narrow_ordinary_lens():
    objects, images, _ = wide_views(NARROW_K, NARROW_D, 30, seed=5, max_deg=35.0)
    rms, K, D, _, _ = calibrate(objects, images, SIZE, False)
    assert rms < 0.35
    assert abs(K[1, 1] / NARROW_K[1, 1] - 1) < 0.002
    # Within 1 px out to the image corners (30 degrees), which few views reach.
    theta = np.radians(np.linspace(2, 30, 15))
    np.testing.assert_allclose(K[1, 1] * distorted_radius(theta, D),
                               NARROW_K[1, 1] * distorted_radius(theta, NARROW_D), atol=1.0)


def test_not_converging_is_an_error(monkeypatch):
    from smart_calibration import fisheye_calibration
    objects, images, _ = wide_views(WIDE_K, WIDE_D, 10, seed=6)
    monkeypatch.setattr(fisheye_calibration, 'MAX_ITERATIONS', 2)
    with pytest.raises(cv2.error, match='not converged after 2 iterations'):
        calibrate(objects, images, SIZE, True)


def test_few_views_with_k3_and_k4_fixed():
    objects, images, _ = wide_views(WIDE_K, WIDE_D, 5, seed=1)
    rms, K, D, _, _ = calibrate(objects, images, SIZE, True)
    assert D[2] == D[3] == 0.0 and rms < 1.0
    assert abs(K[1, 1] / WIDE_K[1, 1] - 1) < 0.02


def test_agrees_with_opencv_where_opencv_converges():
    objects, images, _ = wide_views(MODERATE_K, MODERATE_D, 30, seed=2, max_deg=60.0)
    rms, K, D, _, _ = calibrate(objects, images, SIZE, False)
    try:
        expected_rms, expected_K, expected_D, _, _ = cv2.fisheye.calibrate(
            [o.astype(np.float64) for o in objects], [i.astype(np.float64) for i in images],
            SIZE, None, None,
            flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW)
    except cv2.error as error:  # e.g. OpenCV 4.5.4: fabs(norm_u1) > 0 in InitExtrinsics
        pytest.skip(f'cv2.fisheye.calibrate {cv2.__version__} fails on this data: '
                    f'{str(error).strip().splitlines()[-1][-80:]}')
    assert abs(rms - expected_rms) < 1e-6
    np.testing.assert_allclose(K, expected_K, atol=1e-3)
    np.testing.assert_allclose(D, expected_D.ravel(), atol=1e-5)


def test_a_view_without_a_starting_pose_is_named():
    objects, images, _ = wide_views(MODERATE_K, MODERATE_D, 6, seed=3, max_deg=60.0)
    objects[4], images[4] = objects[4][:3], images[4][:3]  # too few corners for PnP
    with pytest.raises(cv2.error, match='#5'):
        calibrate(objects, images, SIZE, True)
