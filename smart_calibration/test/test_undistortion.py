import cv2
import numpy as np

from smart_calibration.gui.undistortion import fold_radius, scaled_matrix, Undistortion

SIZE = (1280, 720)
CAMERA_MATRIX = np.array([[640.0, 0.0, 639.5], [0.0, 640.0, 359.5], [0.0, 0.0, 1.0]])
DISTORTION = np.array([-0.2, 0.04, 0.0005, -0.0005, 0.0])


def distorted(rays):
    """Pixels (N, 2) where the camera sees normalized rays (N, 2)."""
    points = np.column_stack([rays, np.ones(len(rays))]).reshape(-1, 1, 3)
    return cv2.projectPoints(points, np.zeros(3), np.zeros(3), CAMERA_MATRIX,
                             DISTORTION)[0].reshape(-1, 2)


def test_points_land_where_the_ideal_camera_puts_them():
    undistortion = Undistortion(CAMERA_MATRIX, DISTORTION, SIZE, 1.0)
    rays = np.array([[0.0, 0.0], [0.6, 0.3], [-0.8, -0.4]])
    new = undistortion.new_camera_matrix
    ideal = rays * [new[0, 0], new[1, 1]] + [new[0, 2], new[1, 2]]
    np.testing.assert_allclose(undistortion.points(distorted(rays)), ideal, atol=1e-3)
    # NaN (unused corners) stays NaN, the shape is kept.
    grid = np.array([[[100.0, 100.0], [np.nan, np.nan]]])
    result = undistortion.points(grid)
    assert result.shape == (1, 2, 2) and np.isnan(result[0, 1]).all()


def test_images_of_any_size_match_the_points():
    undistortion = Undistortion(CAMERA_MATRIX, DISTORTION, SIZE, 1.0)
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
    undistortion = Undistortion(CAMERA_MATRIX, DISTORTION, SIZE, 1.0)
    vectors = np.array([[0.3, 0.0], [0.0, -0.5], [0.0, 0.0]])
    moved = undistortion.directions(np.array([[200.0, 150.0], [640.0, 360.0], [10.0, 10.0]]),
                                    vectors)
    np.testing.assert_allclose(np.hypot(moved[:, 0], moved[:, 1]), [0.3, 0.5, 0.0], atol=1e-6)
    # Near the centre the distortion hardly turns a direction.
    np.testing.assert_allclose(moved[1], [0.0, -0.5], atol=1e-3)
    cropped = Undistortion(CAMERA_MATRIX, DISTORTION, SIZE, 0.0)
    assert cropped.new_camera_matrix[0, 0] > undistortion.new_camera_matrix[0, 0]
    assert cropped.key() != undistortion.key()


def test_scaled_matrix_keeps_the_image_edges():
    half = scaled_matrix(CAMERA_MATRIX, SIZE, (640, 360))
    np.testing.assert_allclose(half, [[320, 0, 319.5], [0, 320, 179.5], [0, 0, 1]])


def test_fold_radius():
    # r (1 - 0.2 r^2) stops growing at 1 - 0.6 r^2 = 0.
    assert abs(fold_radius([-0.2, 0.0, 0.0, 0.0, 0.0]) - np.sqrt(1 / 0.6)) < 1e-9
    assert fold_radius(DISTORTION) == float('inf')  # k2 = 0.04 keeps it growing
    assert fold_radius([0.1, 0.0, 0.0, 0.0, 0.0]) == float('inf')


def test_nothing_is_shown_beyond_the_fold():
    folding = np.array([-0.3, 0.0, 0.0, 0.0, 0.0])  # folds at r = 1.05, inside the image
    undistortion = Undistortion(CAMERA_MATRIX, folding, SIZE, 1.0)
    outline = undistortion.fold_outline()
    assert outline.shape == (180, 2)
    result = undistortion.image(np.full((720, 1280), 200, np.uint8))
    new = undistortion.new_camera_matrix
    radius = np.hypot((np.arange(1280)[None] - new[0, 2]) / new[0, 0],
                      (np.arange(720)[:, None] - new[1, 2]) / new[1, 1])
    assert (result[radius > undistortion.fold_radius * 1.01] == 0).all()
    assert (result[radius < undistortion.fold_radius * 0.5] == 200).all()
    assert Undistortion(CAMERA_MATRIX, DISTORTION, SIZE, 1.0).fold_outline() is None
    # Zoomed out only until the valid circle fills the view's height.
    assert abs(undistortion.new_camera_matrix[0, 0] * undistortion.fold_radius - 360) < 1e-6
