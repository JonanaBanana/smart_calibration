import math

import cv2
import numpy as np
import pytest

from smart_calibration.camera_models import camera_model, FISHEYE, PLUMB_BOB

SIZE = (1280, 720)
K = np.array([[400.0, 0.0, 641.0], [0.0, 410.0, 358.0], [0.0, 0.0, 1.0]])
D = np.array([0.05, -0.01, 0.002, -0.0003])


def unit(rays):
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def test_models_by_name():
    assert camera_model('plumb_bob') is PLUMB_BOB and camera_model('fisheye') is FISHEYE
    with pytest.raises(ValueError):
        camera_model('equidistant')
    assert PLUMB_BOB.high_order_text() == 'k3' and FISHEYE.high_order_text() == 'k3, k4'
    assert FISHEYE.groups['distortion'] == (4, 5, 6, 7)


def test_fisheye_rays_invert_the_projection_also_beyond_90_degrees():
    theta = np.radians([0.0, 10.0, 45.0, 80.0, 89.0, 100.0])
    phi = np.radians([0.0, 30.0, 135.0, -60.0, 200.0, 75.0])
    rays = np.column_stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                            np.cos(theta)])
    theta_d = FISHEYE.distorted_radius(theta, D)
    pixels = np.column_stack([K[0, 0] * theta_d * np.cos(phi) + K[0, 2],
                              K[1, 1] * theta_d * np.sin(phi) + K[1, 2]])
    # In front of the camera this is OpenCV's projection.
    projected, _, _ = FISHEYE.project(rays[:5], np.zeros(3), np.zeros(3), K, D)
    np.testing.assert_allclose(projected, pixels[:5], atol=1e-9)
    np.testing.assert_allclose(FISHEYE.rays(K, D, pixels), rays, atol=1e-9)


def test_fisheye_has_no_ray_beyond_its_fold():
    folding = np.array([-0.2, 0.0, 0.0, 0.0])
    angle = FISHEYE.fold_angle(folding)
    assert abs(angle - math.sqrt(1 / 0.6)) < 1e-12  # 1 - 0.6 theta^2 = 0
    reach = FISHEYE.distorted_radius(angle, folding)
    pixels = np.array([[K[0, 2] + K[0, 0] * reach * 0.99, K[1, 2]],
                       [K[0, 2] + K[0, 0] * reach * 1.01, K[1, 2]]])
    rays = FISHEYE.rays(K, folding, pixels)
    assert np.isfinite(rays[0]).all() and np.isnan(rays[1]).all()
    assert FISHEYE.folds_in_image(K, folding, SIZE) is True  # a bool, for the ROS message
    assert FISHEYE.folds_in_image(K, D, SIZE) is False
    # k4 < 0 folds at 136 degrees, beyond the image corners and beyond a pinhole view.
    assert 2.3 < FISHEYE.fold_angle(D) < 2.4 and FISHEYE.fold_radius(D) == math.inf
    assert FISHEYE.fold_angle([0.05, -0.01, 0.002, 0.0]) == math.inf
    # A fold beyond 90 degrees has no place in a pinhole view.
    assert FISHEYE.fold_radius([-0.1, 0.0, 0.0, 0.0]) == math.inf


# A plumb_bob camera with a 90 degree horizontal field of view.
PINHOLE = np.array([[640.0, 0.0, 641.0], [0.0, 650.0, 358.0], [0.0, 0.0, 1.0]])
PINHOLE_DISTORTION = np.array([-0.2, 0.04, 0.0005, -0.0005, 0.01])


@pytest.mark.parametrize('model, K, distortion', [
    (PLUMB_BOB, PINHOLE, PINHOLE_DISTORTION), (FISHEYE, K, D)])
def test_ray_jacobian_matches_finite_differences(model, K, distortion):
    pixels = np.array([[100.0, 80.0], [640.0, 360.0], [1000.0, 500.0], [300.0, 650.0]])
    rays = model.rays(K, distortion, pixels)
    assert np.isfinite(rays).all()
    values = np.concatenate([[K[0, 0], K[1, 1], K[0, 2], K[1, 2]], distortion])

    def project(v):
        matrix = np.array([[v[0], 0, v[2]], [0, v[1], v[3]], [0, 0, 1]])
        return model.project(rays, np.zeros(3), np.zeros(3), matrix, v[4:])[0]
    jacobian = model.ray_jacobian(K, distortion, rays)
    assert jacobian.shape == (4, 2, len(model.parameters))
    for j in range(len(values)):
        step = 1e-6 * max(1.0, abs(values[j]))
        up, down = values.copy(), values.copy()
        up[j] += step
        down[j] -= step
        numeric = (project(up) - project(down)) / (2 * step)
        np.testing.assert_allclose(jacobian[:, :, j], numeric, rtol=1e-5, atol=1e-5)


def test_project_returns_the_pose_jacobian():
    board = np.zeros((6, 3))
    board[:, :2] = np.mgrid[0:3, 0:2].T.reshape(-1, 2) * 0.05
    rvec, tvec = np.array([0.3, -0.2, 0.1]), np.array([-0.05, 0.02, 0.3])
    for model, distortion in ((PLUMB_BOB, PINHOLE_DISTORTION), (FISHEYE, D)):
        _, by_parameters, by_pose = model.project(board, rvec, tvec, K, distortion)
        assert by_parameters.shape == (12, len(model.parameters)) and by_pose.shape == (12, 6)
        pose = np.concatenate([rvec, tvec])
        step = 1e-7
        for j in range(6):
            up, down = pose.copy(), pose.copy()
            up[j] += step
            down[j] -= step
            numeric = (model.project(board, up[:3], up[3:], K, distortion)[0]
                       - model.project(board, down[:3], down[3:], K, distortion)[0]) / (2 * step)
            np.testing.assert_allclose(by_pose[:, j], numeric.ravel(), rtol=1e-4, atol=1e-3)


def test_plumb_bob_rays_project_back_onto_their_pixels():
    pixels = np.array([[100.0, 80.0], [640.0, 360.0], [1270.0, 710.0]])
    rays = PLUMB_BOB.rays(PINHOLE, PINHOLE_DISTORTION, pixels)
    projected, _ = cv2.projectPoints(rays.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), PINHOLE,
                                     PINHOLE_DISTORTION)
    np.testing.assert_allclose(projected.reshape(-1, 2), pixels, atol=1e-6)
    # Beyond the fold of k1 -0.3 there is no ray (it reaches 0.70, the corner is at 1.15).
    assert np.isnan(PLUMB_BOB.rays(PINHOLE, [-0.3, 0, 0, 0, 0], [[0.0, 0.0]])).all()
