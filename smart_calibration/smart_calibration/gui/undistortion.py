"""
Undistortion of images of any size and of points in full image pixels.

The undistorted view is a pinhole view. alpha 0 shows only valid pixels, alpha 1 the whole image
(see CameraModel.new_camera_matrix). Beyond the radius where the model folds over the view is
black.
"""

import cv2
import numpy as np

# Zoom of the undistorted view, limited to [1 / MAX_ZOOM, MAX_ZOOM].
MAX_ZOOM = 5.0


def scaled_matrix(camera_matrix, full_size, size):
    """Camera matrix for an image of size (w, h) showing the same view as one of full_size."""
    sx, sy = size[0] / full_size[0], size[1] / full_size[1]
    scaled = np.array(camera_matrix, np.float64)
    scaled[0, 0] *= sx
    scaled[1, 1] *= sy
    # Pixel centres at integers: the image edge is at -0.5.
    scaled[0, 2] = (scaled[0, 2] + 0.5) * sx - 0.5
    scaled[1, 2] = (scaled[1, 2] + 0.5) * sy - 0.5
    return scaled


class Undistortion:
    """Undistorts images and points with one calibration; remap tables cached per image size."""

    def __init__(self, model, camera_matrix, distortion, image_size, alpha):
        """
        model: camera_models.CameraModel; image_size: (width, height) the calibration is for.

        alpha: see the module docstring.
        """
        self.model = model
        self.camera_matrix = np.asarray(camera_matrix, np.float64).reshape(3, 3)
        self.distortion = np.asarray(distortion, np.float64).ravel()
        self.image_size = tuple(int(v) for v in image_size)
        self.alpha = alpha
        new = np.array(model.new_camera_matrix(self.camera_matrix, self.distortion,
                                               self.image_size, alpha), np.float64)
        zoom = new[0, 0] / self.camera_matrix[0, 0]
        if not (np.isfinite(new).all() and zoom > 0):
            new = self.camera_matrix.copy()
        elif not 1 / MAX_ZOOM <= zoom <= MAX_ZOOM:
            new[:2, :2] *= np.clip(zoom, 1 / MAX_ZOOM, MAX_ZOOM) / zoom
        self.fold_radius = model.fold_radius(self.distortion)
        if np.isfinite(self.fold_radius):
            # Zoom out no further than the fold circle filling the view's height.
            least = min(self.image_size) / 2 / self.fold_radius
            if new[0, 0] < least:
                new = new.copy()
                new[:2, :2] *= least / new[0, 0]
        self.new_camera_matrix = new
        self._maps = {}

    def key(self):
        """Return what this undistortion depends on, to tell whether it must be rebuilt."""
        return (self.model.name, tuple(self.camera_matrix.ravel()), tuple(self.distortion),
                self.image_size, self.alpha)

    def image(self, image):
        """Return the undistorted picture: same size and type, black (NaN) outside the view."""
        height, width = image.shape[:2]
        maps = self._maps.get((width, height))
        if maps is None:
            size = (width, height)
            new = scaled_matrix(self.new_camera_matrix, self.image_size, size)
            maps = self.model.undistort_maps(
                scaled_matrix(self.camera_matrix, self.image_size, size), self.distortion, new,
                size)
            if np.isfinite(self.fold_radius):
                xs = (np.arange(width) - new[0, 2]) / new[0, 0]
                ys = (np.arange(height) - new[1, 2]) / new[1, 1]
                beyond = np.hypot(xs[None, :], ys[:, None]) > self.fold_radius
                for table in maps:
                    table[beyond] = -1.0
            self._maps[(width, height)] = maps
        floating = np.issubdtype(image.dtype, np.floating)
        return cv2.remap(image, maps[0], maps[1], cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=float('nan') if floating else 0)

    def fold_outline(self, count=180):
        """Boundary (count, 2) of the valid model in undistorted full image pixels, or None."""
        if not np.isfinite(self.fold_radius):
            return None
        angles = np.linspace(0.0, 2 * np.pi, count, endpoint=False)
        new = self.new_camera_matrix
        return np.column_stack([
            new[0, 2] + new[0, 0] * self.fold_radius * np.cos(angles),
            new[1, 2] + new[1, 1] * self.fold_radius * np.sin(angles)]).astype(np.float32)

    def points(self, points):
        """Undistort positions (..., 2) in full image pixels; NaN stays NaN."""
        points = np.asarray(points, np.float64)
        result = np.full(points.shape, np.nan)
        flat, out = points.reshape(-1, 2), result.reshape(-1, 2)
        finite = np.isfinite(flat).all(axis=1)
        if finite.any():
            out[finite] = self.model.undistort_points(
                self.camera_matrix, self.distortion, flat[finite], self.new_camera_matrix)
        return result.astype(np.float32)

    def directions(self, points, vectors):
        """Turn vectors (N, 2) at points (N, 2) into the undistorted view; lengths are kept."""
        vectors = np.asarray(vectors, np.float64)
        lengths = np.hypot(vectors[:, 0], vectors[:, 1])
        unit = np.divide(vectors, lengths[:, None], out=np.zeros_like(vectors),
                         where=lengths[:, None] > 0)
        start = self.points(points).astype(np.float64)
        end = self.points(np.asarray(points, np.float64) + unit).astype(np.float64)
        moved = end - start
        moved_lengths = np.hypot(moved[:, 0], moved[:, 1])
        return np.divide(moved * lengths[:, None], moved_lengths[:, None],
                         out=np.zeros_like(moved),
                         where=moved_lengths[:, None] > 0).astype(np.float32)
