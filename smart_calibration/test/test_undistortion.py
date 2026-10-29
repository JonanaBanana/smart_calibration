import cv2
import numpy as np

from smart_calibration.camera_models import FISHEYE, PLUMB_BOB
from smart_calibration.gui.undistortion import scaled_matrix, Undistortion

SIZE = (1280, 720)
CAMERA_MATRIX = np.array([[640.0, 0.0, 639.5], [0.0, 640.0, 359.5], [0.0, 0.0, 1.0]])
DISTORTION = np.array([-0.2, 0.04, 0.0005, -0.0005, 0.0])


def distorted(rays):
    """Pixels (N, 2) where the camera sees normalized rays (N, 2)."""
    points = np.column_stack([rays, np.ones(len(rays))]).reshape(-1, 1, 3)
    return cv2.projectPoints(points, np.zeros(3), np.zeros(3), CAMERA_MATRIX,
                             DISTORTION)[0].reshape(-1, 2)


def test_points_land_where_the_ideal_camera_puts_them():
    undistortion = Undistortion(PLUMB_BOB, CAMERA_MATRIX, DISTORTION, SIZE, 1.0)
    rays = np.array([[0.0, 0.0], [0.6, 0.3], [-0.8, -0.4]])
    new = undistortion.new_camera_matrix
    ideal = rays * [new[0, 0], new[1, 1]] + [new[0, 2], new[1, 2]]
    np.testing.assert_allclose(undistortion.points(distorted(rays)), ideal, atol=1e-3)
    # NaN (unused corners) stays NaN, the shape is kept.
    grid = np.array([[[100.0, 100.0], [np.nan, np.nan]]])
    result = undistortion.points(grid)
    assert result.shape == (1, 2, 2) and np.isnan(result[0, 1]).all()


def test_images_of_any_size_match_the_points():
    undistortion = Undistortion(PLUMB_BOB, CAMERA_MATRIX, DISTORTION, SIZE, 1.0)
    pixel = distorted(np.array([[0.7, 0.35]]))[0]
    target = undistortion.points(pixel[None])[0]
    for scale in (1.0, 0.5):  # full image and a half-size preview
        size = (int(SIZE[0] * scale), int(SIZE[1] * scale))
        image = np.zeros((size[1], size[0]), np.float32)
        cv2.circle(image, tuple(int(round(v)) for v in (pixel + 0.5) * scale - 0.5), 3, 1.0, -1)
        result = undistortion.image(image)
        ys, xs = np.nonzero(result > 0.5)
        found = np.array([xs.mean(), ys.mean()])
        np.testing.assert_allclose(found, (target + 0.5) * scale - 0.5, atol=1.5)
    # Outside the view: NaN for float images, black otherwise.
    assert np.isnan(undistortion.image(np.ones((720, 1280), np.float32))[0, 0])
    assert undistortion.image(np.full((720, 1280), 200, np.uint8))[0, 0] == 0


def test_directions_keep_their_length_and_alpha_crops():
    undistortion = Undistortion(PLUMB_BOB, CAMERA_MATRIX, DISTORTION, SIZE, 1.0)
    vectors = np.array([[0.3, 0.0], [0.0, -0.5], [0.0, 0.0]])
    moved = undistortion.directions(np.array([[200.0, 150.0], [640.0, 360.0], [10.0, 10.0]]),
                                    vectors)
    np.testing.assert_allclose(np.hypot(moved[:, 0], moved[:, 1]), [0.3, 0.5, 0.0], atol=1e-6)
    # Near the centre the distortion hardly turns a direction.
    np.testing.assert_allclose(moved[1], [0.0, -0.5], atol=1e-3)
    cropped = Undistortion(PLUMB_BOB, CAMERA_MATRIX, DISTORTION, SIZE, 0.0)
    assert cropped.new_camera_matrix[0, 0] > undistortion.new_camera_matrix[0, 0]
    assert cropped.key() != undistortion.key()


def test_scaled_matrix_keeps_the_image_edges():
    half = scaled_matrix(CAMERA_MATRIX, SIZE, (640, 360))
    np.testing.assert_allclose(half, [[320, 0, 319.5], [0, 320, 179.5], [0, 0, 1]])


def test_fold_radius():
    # r (1 - 0.2 r^2) stops growing at 1 - 0.6 r^2 = 0.
    assert abs(PLUMB_BOB.fold_radius([-0.2, 0.0, 0.0, 0.0, 0.0]) - np.sqrt(1 / 0.6)) < 1e-9
    assert PLUMB_BOB.fold_radius(DISTORTION) == float('inf')  # k2 = 0.04 keeps it growing
    assert PLUMB_BOB.fold_radius([0.1, 0.0, 0.0, 0.0, 0.0]) == float('inf')


def test_nothing_is_shown_beyond_the_fold():
    folding = np.array([-0.3, 0.0, 0.0, 0.0, 0.0])  # folds at r = 1.05, inside the image
    undistortion = Undistortion(PLUMB_BOB, CAMERA_MATRIX, folding, SIZE, 1.0)
    outline = undistortion.fold_outline()
    assert outline.shape == (180, 2)
    result = undistortion.image(np.full((720, 1280), 200, np.uint8))
    new = undistortion.new_camera_matrix
    radius = np.hypot((np.arange(1280)[None] - new[0, 2]) / new[0, 0],
                      (np.arange(720)[:, None] - new[1, 2]) / new[1, 1])
    assert (result[radius > undistortion.fold_radius * 1.01] == 0).all()
    assert (result[radius < undistortion.fold_radius * 0.5] == 200).all()
    assert Undistortion(PLUMB_BOB, CAMERA_MATRIX, DISTORTION, SIZE, 1.0).fold_outline() is None
    # Zoomed out only until the valid circle fills the view's height.
    assert abs(undistortion.new_camera_matrix[0, 0] * undistortion.fold_radius - 360) < 1e-6


FISHEYE_MATRIX = np.array([[400.0, 0.0, 639.5], [0.0, 400.0, 359.5], [0.0, 0.0, 1.0]])
FISHEYE_DISTORTION = np.array([0.05, -0.01, 0.002, 0.0])


def fisheye_pixels(rays):
    """Pixels (N, 2) where the fisheye camera sees normalized rays (N, 2), z = 1."""
    points = np.column_stack([rays, np.ones(len(rays))]).reshape(-1, 1, 3)
    return cv2.fisheye.projectPoints(points, np.zeros(3), np.zeros(3), FISHEYE_MATRIX,
                                     FISHEYE_DISTORTION)[0].reshape(-1, 2)


def test_fisheye_points_and_images_land_where_the_ideal_camera_puts_them():
    undistortion = Undistortion(FISHEYE, FISHEYE_MATRIX, FISHEYE_DISTORTION, SIZE, 1.0)
    new = undistortion.new_camera_matrix
    rays = np.array([[0.0, 0.0], [1.2, 0.5], [-2.0, -0.3]])  # up to 64 degrees off the axis
    ideal = rays * [new[0, 0], new[1, 1]] + [new[0, 2], new[1, 2]]
    np.testing.assert_allclose(undistortion.points(fisheye_pixels(rays)), ideal, atol=1e-3)
    # The image corners are 105 degrees off the axis: no place in a pinhole view.
    assert np.isnan(undistortion.points(np.array([[0.0, 0.0]]))).all()
    # The image is mapped like the points.
    pixel = fisheye_pixels(np.array([[1.0, 0.4]]))[0]
    target = undistortion.points(pixel[None])[0]
    image = np.zeros((720, 1280), np.float32)
    cv2.circle(image, tuple(int(round(v)) for v in pixel), 3, 1.0, -1)
    ys, xs = np.nonzero(undistortion.image(image) > 0.5)
    np.testing.assert_allclose([xs.mean(), ys.mean()], target, atol=1.5)
    # The left and right edges are 85 degrees off the axis: the zoom stops at its limit.
    assert abs(new[0, 0] / FISHEYE_MATRIX[0, 0] - 0.2) < 1e-9
    top, right = undistortion.points(np.array([[639.5, -0.5], [1279.5, 359.5]]))
    assert 0 < top[1] < 360 and right[0] > 1280


def test_fisheye_fold_is_masked():
    folding = np.array([-0.2, 0.0, 0.0, 0.0])  # folds at 74 degrees, inside the image
    undistortion = Undistortion(FISHEYE, FISHEYE_MATRIX, folding, SIZE, 1.0)
    assert abs(undistortion.fold_radius - np.tan(np.sqrt(1 / 0.6))) < 1e-9
    result = undistortion.image(np.full((720, 1280), 200, np.uint8))
    new = undistortion.new_camera_matrix
    radius = np.hypot((np.arange(1280)[None] - new[0, 2]) / new[0, 0],
                      (np.arange(720)[:, None] - new[1, 2]) / new[1, 1])
    assert (result[radius > undistortion.fold_radius * 1.01] == 0).all()
    assert (result[radius < undistortion.fold_radius * 0.5] == 200).all()
    assert undistortion.fold_outline() is not None
    assert Undistortion(FISHEYE, FISHEYE_MATRIX, FISHEYE_DISTORTION, SIZE, 1.0).fold_outline() \
        is None
