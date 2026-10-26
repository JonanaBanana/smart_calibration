#!/usr/bin/env python3
"""
Synthetic calibration images with exact ground truth, for testing smart_calibration.

A simulated camera (pinhole with plumb_bob distortion, or fisheye with equidistant distortion)
views a board in random poses: a plain chessboard (completely visible) or an oversized radon
checkerboard with three marker dots (larger than the view). For every supersampled pixel, the ray
through the lens model is followed to the board plane and the board texture looked up there, so
perspective and lens distortion are exact and the image is interpolated only once. Blur,
vignetting and noise are added afterwards. ground_truth.json holds the camera, every board pose
and the true pixel position and board coordinates of every inner corner.

The board drawing follows OpenCV's apps/pattern-tools/generate_pattern.py (Apache License 2.0,
copyright the OpenCV authors, see THIRD_PARTY_NOTICES.md): same layout, colours, rounded outer
squares of the radon checkerboard and marker dots, rasterised directly instead of written as SVG.

Usage (venv active, from the repository root):
    python utility/synthetic_dataset.py --output dataset/synthetic
"""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import cv2
import numpy as np

# Board texture resolution; well above the camera's resolution on the board at the closest
# distances used here.
PX_PER_MM = 6.0
# Subsamples per pixel and axis when drawing curved shapes (rounded squares, marker dots).
SHAPE_SUBSAMPLES = 4
# Board distances (m) for both boards, so they are compared at the same distances. Poses that
# break a board's visibility rule are drawn again: the chessboard must be completely visible,
# so in practice it only appears from ~0.45 m on.
DISTANCE_RANGE_M = (0.3, 2.6)


# ---------------------------------------------------------------------------------- boards

@dataclass(frozen=True)
class BoardSpec:
    """A printed board, laid out like generate_pattern.py: pattern centred on a white page."""

    kind: str               # 'chessboard' or 'oversized_marker' (smart_calibration board_mode)
    squares_x: int
    squares_y: int
    square_mm: float
    margin_mm: float        # white border around the pattern
    markers: tuple = ()     # (x, y) cells with a marker dot (radon checkerboard)
    origin_cell: tuple = None   # white marker cell; its top-left corner is the board origin

    @property
    def page_mm(self):
        return (self.squares_x * self.square_mm + 2 * self.margin_mm,
                self.squares_y * self.square_mm + 2 * self.margin_mm)

    @property
    def grid(self):
        """(rows, cols) of inner corners."""
        return self.squares_y - 1, self.squares_x - 1

    def inner_corners_mm(self):
        """(rows, cols, 2) page positions (x, y) in mm of the inner corners, row-major."""
        rows, cols = self.grid
        j, i = np.mgrid[1:rows + 1, 1:cols + 1]
        return np.stack([self.margin_mm + i * self.square_mm,
                         self.margin_mm + j * self.square_mm], -1).astype(np.float64)

    def board_coordinates(self):
        """
        (rows, cols, 2) board coordinates (x, y) of the inner corners, in squares.

        As smart_calibration.detection.DetectedBoard: grid indices for a plain chessboard,
        relative to the origin corner (top-left corner of origin_cell) for a marker board.
        """
        rows, cols = self.grid
        row, col = np.mgrid[0:rows, 0:cols]
        coordinates = np.stack([col, row], -1)
        if self.origin_cell is not None:
            coordinates = coordinates + 1 - np.array(self.origin_cell)
        return coordinates


# 6 x 7 inner corners, like config/ircam_wide.yaml.
CHESSBOARD = BoardSpec('chessboard', 7, 8, 60.0, 60.0)
# Oversized radon checkerboard; the marker L has exactly one white cell, (7, 4): the origin.
MARKER_BOARD = BoardSpec('oversized_marker', 14, 10, 50.0, 50.0,
                         ((7, 4), (8, 4), (7, 5)), (7, 4))


def _square_weights(n_px, margin_mm, square_mm):
    """Per texture pixel along one axis: first square index it overlaps and the overlap share."""
    edges = (np.arange(n_px + 1) / PX_PER_MM - margin_mm) / square_mm  # in squares
    start, end = edges[:-1], edges[1:]
    first = np.floor(start)
    share = (np.minimum(end, first + 1) - start) / (end - start)
    return first.astype(int), share


def _subsampled_fraction(x0_mm, y0_mm, x1_mm, y1_mm, inside):
    """
    Texture window covering a mm rectangle, and per pixel the fraction of its subsamples inside.

    inside(x_mm, y_mm) -> bool array. Pixel u covers [u, u + 1) / PX_PER_MM mm.
    """
    u0, v0 = math.floor(x0_mm * PX_PER_MM), math.floor(y0_mm * PX_PER_MM)
    u1, v1 = math.ceil(x1_mm * PX_PER_MM), math.ceil(y1_mm * PX_PER_MM)
    offsets = (np.arange(SHAPE_SUBSAMPLES) + 0.5) / SHAPE_SUBSAMPLES
    xs = ((np.arange(u0, u1)[:, None] + offsets).ravel()) / PX_PER_MM
    ys = ((np.arange(v0, v1)[:, None] + offsets).ravel()) / PX_PER_MM
    x, y = np.meshgrid(xs, ys)
    n = SHAPE_SUBSAMPLES
    fraction = inside(x, y).reshape(v1 - v0, n, u1 - u0, n).mean(axis=(1, 3))
    return (slice(v0, v1), slice(u0, u1)), fraction


def draw_board(spec):
    """
    Board texture as float32 (h, w), 0 = black, 1 = white; pixel u covers [u, u+1) / PX_PER_MM mm.

    Squares get their exact pixel coverage; curved shapes (rounded outer squares of the radon
    checkerboard, marker dots) are subsampled. Black squares where x % 2 == y % 2, as
    generate_pattern.py.
    """
    width_mm, height_mm = spec.page_mm
    width, height = math.ceil(width_mm * PX_PER_MM), math.ceil(height_mm * PX_PER_MM)
    sq, margin = spec.square_mm, spec.margin_mm
    assert sq * PX_PER_MM > 2, 'a pixel must not span more than two squares'

    # black[cy + 1, cx + 1]: square (cx, cy) is black; padded with white around the pattern.
    black = np.zeros((spec.squares_y + 2, spec.squares_x + 2))
    cy, cx = np.mgrid[0:spec.squares_y, 0:spec.squares_x]
    black[1:-1, 1:-1] = (cx % 2) == (cy % 2)
    ix, sx = _square_weights(width, margin, sq)
    iy, sy = _square_weights(height, margin, sq)
    ix0, ix1 = np.clip(ix + 1, 0, spec.squares_x + 1), np.clip(ix + 2, 0, spec.squares_x + 1)
    iy0, iy1 = np.clip(iy + 1, 0, spec.squares_y + 1), np.clip(iy + 2, 0, spec.squares_y + 1)
    coverage = (sy[:, None] * (sx * black[iy0][:, ix0] + (1 - sx) * black[iy0][:, ix1])
                + (1 - sy)[:, None] * (sx * black[iy1][:, ix0] + (1 - sx) * black[iy1][:, ix1]))
    texture = (1.0 - coverage).astype(np.float32)

    if spec.kind == 'oversized_marker':
        _round_outer_squares(texture, spec)
        for x, y in spec.markers:
            centre = (margin + (x + 0.5) * sq, margin + (y + 0.5) * sq)
            radius = 0.17 * sq
            window, fraction = _subsampled_fraction(
                centre[0] - radius, centre[1] - radius, centre[0] + radius, centre[1] + radius,
                lambda px, py, c=centre, r=radius: (px - c[0]) ** 2 + (py - c[1]) ** 2 <= r * r)
            color = 1.0 if x % 2 == y % 2 else 0.0  # white dot in a black cell, black in white
            texture[window] = texture[window] * (1 - fraction) + color * fraction
    return texture


def _round_outer_squares(texture, spec):
    """Round the outward corners of the black edge squares (generate_pattern.py's radon board)."""
    sq, margin = spec.square_mm, spec.margin_mm
    for y in range(spec.squares_y):
        for x in range(spec.squares_x):
            if x % 2 != y % 2:
                continue
            # Quadrants (dx, dy) of the square whose corner is rounded, as _get_type().
            quadrants = set()
            if x == 0:
                quadrants |= {(0, 0), (0, 1)}
            if y == 0:
                quadrants |= {(0, 0), (1, 0)}
            if x == spec.squares_x - 1:
                quadrants |= {(1, 0), (1, 1)}
            if y == spec.squares_y - 1:
                quadrants |= {(1, 1), (0, 1)}
            centre = (margin + (x + 0.5) * sq, margin + (y + 0.5) * sq)
            for dx, dy in quadrants:
                x0, y0 = centre[0] - sq / 2 + dx * sq / 2, centre[1] - sq / 2 + dy * sq / 2

                def outside_arc(px, py, x0=x0, y0=y0):
                    in_quadrant = (px >= x0) & (px < x0 + sq / 2) & (py >= y0) & (py < y0 + sq / 2)
                    return in_quadrant & ((px - centre[0]) ** 2 + (py - centre[1]) ** 2
                                          > (sq / 2) ** 2)
                window, fraction = _subsampled_fraction(x0, y0, x0 + sq / 2, y0 + sq / 2,
                                                        outside_arc)
                texture[window] = np.minimum(texture[window] + fraction, 1.0)


def texture_pyramid(texture, levels=6):
    """Mipmap: each level half the size of the previous one (area averaged)."""
    pyramid = [texture]
    for _ in range(levels):
        h, w = pyramid[-1].shape
        if min(h, w) < 64:
            break
        pyramid.append(cv2.resize(pyramid[-1], (w // 2, h // 2), interpolation=cv2.INTER_AREA))
    return pyramid


# ---------------------------------------------------------------------------------- camera

@dataclass(frozen=True)
class Camera:
    """Simulated camera; K and D in OpenCV's conventions for the model."""

    model: str              # 'pinhole' (plumb_bob: k1, k2, p1, p2, k3) or 'fisheye' (k1-k4)
    width: int
    height: int
    K: np.ndarray
    D: np.ndarray


def make_camera(model, width=1280, height=720, fov_deg=90.0):
    """
    Camera with a horizontal field of view of fov_deg and moderate wide-angle distortion.

    pinhole: f such that an ideal pinhole would see fov_deg; the barrel distortion widens the
    actual view (to ~105 deg here). fisheye: f such that the image edge is at fov_deg / 2 off-axis.
    The distortion stays monotonic over the whole image (no fold-over at the corners).
    """
    half = math.radians(fov_deg) / 2
    if model == 'pinhole':
        D = np.array([-0.2, 0.04, 0.0005, -0.0005, 0.0])
        f = (width / 2) / math.tan(half)
    elif model == 'fisheye':
        D = np.array([0.05, -0.01, 0.002, 0.0])
        theta_d = half * (1 + D[0] * half ** 2 + D[1] * half ** 4 + D[2] * half ** 6
                          + D[3] * half ** 8)
        f = (width / 2) / theta_d
    else:
        raise ValueError(f'unknown camera model {model!r}')
    K = np.array([[f, 0, (width - 1) / 2], [0, f, (height - 1) / 2], [0, 0, 1]])
    return Camera(model, width, height, K, D)


def project(camera, points):
    """Pixel positions (N, 2) of camera-frame points (N, 3), pixel centres at integers."""
    points = np.asarray(points, np.float64).reshape(-1, 1, 3)
    zero = np.zeros(3)
    if camera.model == 'pinhole':
        pixels, _ = cv2.projectPoints(points, zero, zero, camera.K, camera.D)
    else:
        pixels, _ = cv2.fisheye.projectPoints(points, zero, zero, camera.K, camera.D)
    return pixels.reshape(-1, 2)


def pixel_rays(camera, pixels):
    """
    Normalised ray directions (x, y), z = 1, through pixel positions (N, 2); NaN where invalid.

    Inverts the lens model iteratively; a ray counts as valid if projecting it back lands within
    1e-3 px of its pixel.
    """
    pixels = np.asarray(pixels, np.float64).reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)
    if camera.model == 'pinhole':
        rays = cv2.undistortPointsIter(pixels, camera.K, camera.D, None, None, criteria)
    else:
        rays = cv2.fisheye.undistortPoints(pixels, camera.K, camera.D, criteria=criteria)
    rays = rays.reshape(-1, 2)
    back = project(camera, np.column_stack([rays, np.ones(len(rays))]))
    rays[np.linalg.norm(back - pixels.reshape(-1, 2), axis=1) > 1e-3] = np.nan
    return rays


def supersampled_rays(camera, supersample):
    """Rays (h * s, w * s, 2) float32 through the centres of the supersampled pixels."""
    s = supersample
    us = (np.arange(camera.width * s) + 0.5) / s - 0.5
    vs = (np.arange(camera.height * s) + 0.5) / s - 0.5
    grid = np.stack(np.meshgrid(us, vs), -1).reshape(-1, 2)
    return pixel_rays(camera, grid).reshape(camera.height * s, camera.width * s, 2).astype(
        np.float32)


# ---------------------------------------------------------------------------------- rendering

def render_board(rays, pyramid, spec, rotation, translation, levels, background):
    """
    Supersampled image of the board seen by rays (see supersampled_rays), float32 0-1.

    rotation, translation: board pose (board frame: x right, y down on the page, origin at the
    page's top-left corner, metres) in the camera frame. levels: (dark, light) intensity of the
    black and white print; background: intensity where the rays miss the page.
    """
    # Camera ray d = (x, y, 1) hits board point (bx, by) where d ~ G (bx, by, 1), G = [r1 r2 t].
    G = np.column_stack([rotation[:, 0], rotation[:, 1], translation])
    Ginv = np.linalg.inv(G).astype(np.float32)
    x, y = rays[..., 0], rays[..., 1]
    w = Ginv[2, 0] * x + Ginv[2, 1] * y + Ginv[2, 2]
    with np.errstate(invalid='ignore', divide='ignore'):
        bx_mm = 1000 * (Ginv[0, 0] * x + Ginv[0, 1] * y + Ginv[0, 2]) / w
        by_mm = 1000 * (Ginv[1, 0] * x + Ginv[1, 1] * y + Ginv[1, 2]) / w
    width_mm, height_mm = spec.page_mm
    on_page = ((w > 0) & (bx_mm >= 0) & (bx_mm < width_mm) & (by_mm >= 0)
               & (by_mm < height_mm))  # w > 0: board in front of the camera (also excludes NaN)

    # Texture coordinates (pixel centres at integers) at the mipmap level whose texels are about
    # as large as a supersampled pixel (25th percentile over the page: sharp near side).
    u = np.where(on_page, bx_mm * PX_PER_MM - 0.5, -1).astype(np.float32)
    v = np.where(on_page, by_mm * PX_PER_MM - 0.5, -1).astype(np.float32)
    footprint = np.hypot(np.diff(u, axis=1), np.diff(v, axis=1))[on_page[:, 1:] & on_page[:, :-1]]
    level = 0
    if footprint.size:
        level = int(np.clip(math.floor(math.log2(max(np.percentile(footprint, 25), 1.0))),
                            0, len(pyramid) - 1))
    scale = 2.0 ** level
    albedo = cv2.remap(pyramid[level], (u + 0.5) / scale - 0.5, (v + 0.5) / scale - 0.5,
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    dark, light = levels
    return np.where(on_page, dark + (light - dark) * albedo, background).astype(np.float32)


def degrade(image, rng, blur_sigma, noise_sigma, vignetting):
    """Lens blur, vignetting (intensity falls by `vignetting` towards the corners) and noise."""
    if blur_sigma > 0:
        image = cv2.GaussianBlur(image, (0, 0), blur_sigma)
    height, width = image.shape
    yy, xx = np.mgrid[0:height, 0:width]
    r2 = ((xx - (width - 1) / 2) ** 2 + (yy - (height - 1) / 2) ** 2) / (
        ((width - 1) / 2) ** 2 + ((height - 1) / 2) ** 2)
    image = image * (1.0 - vignetting * r2) + rng.normal(0.0, noise_sigma, image.shape)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------------- poses

def board_rotation(yaw_deg, pitch_deg, roll_deg):
    """Board orientation: in-plane roll, then tilts around the board's vertical and horizontal."""
    yaw, pitch, roll = (math.radians(a) for a in (yaw_deg, pitch_deg, roll_deg))
    rz = np.array([[math.cos(roll), -math.sin(roll), 0], [math.sin(roll), math.cos(roll), 0],
                   [0, 0, 1]])
    ry = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0],
                   [-math.sin(yaw), 0, math.cos(yaw)]])
    rx = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)],
                   [0, math.sin(pitch), math.cos(pitch)]])
    return ry @ rx @ rz


def place_board(camera, rotation, anchor_mm, pixel, distance):
    """Translation that puts the page point anchor_mm at `distance` metres along pixel's ray."""
    ray = pixel_rays(camera, [pixel])[0]
    direction = np.array([ray[0], ray[1], 1.0])
    anchor = np.array([anchor_mm[0], anchor_mm[1], 0.0]) / 1000
    return distance * direction / np.linalg.norm(direction) - rotation @ anchor


def corner_pixels(camera, spec, rotation, translation):
    """(rows, cols, 2) pixel positions and (rows, cols) visibility of the board's inner corners."""
    rows, cols = spec.grid
    page = spec.inner_corners_mm().reshape(-1, 2) / 1000
    points = (rotation @ np.column_stack([page, np.zeros(len(page))]).T).T + translation
    pixels = project(camera, points).reshape(rows, cols, 2)
    inside = ((points[:, 2] > 0).reshape(rows, cols)
              & (pixels[..., 0] >= 0) & (pixels[..., 0] <= camera.width - 1)
              & (pixels[..., 1] >= 0) & (pixels[..., 1] <= camera.height - 1))
    return pixels, inside


def pattern_border_visible(camera, spec, rotation, translation, border_px):
    """True if the pattern's outline (sampled densely, it bends) lies inside the image."""
    sq, margin = spec.square_mm, spec.margin_mm
    x0, y0 = margin, margin
    x1, y1 = margin + spec.squares_x * sq, margin + spec.squares_y * sq
    t = np.linspace(0, 1, 40)
    outline = np.concatenate([np.column_stack([x0 + (x1 - x0) * t, np.full_like(t, y0)]),
                              np.column_stack([np.full_like(t, x1), y0 + (y1 - y0) * t]),
                              np.column_stack([x1 - (x1 - x0) * t, np.full_like(t, y1)]),
                              np.column_stack([np.full_like(t, x0), y1 - (y1 - y0) * t])])
    points = (rotation @ np.column_stack([outline / 1000, np.zeros(len(outline))]).T).T
    points += translation
    if np.any(points[:, 2] <= 0):
        return False
    pixels = project(camera, points)
    limit = np.array([camera.width - 1, camera.height - 1]) - border_px
    return bool(np.all((pixels >= border_px) & (pixels <= limit)))


def sample_chessboard_pose(rng, camera, spec):
    """A pose with the whole pattern visible; centre anywhere, any roll, tilts up to 45 deg."""
    width_mm, height_mm = spec.page_mm
    for _ in range(500):
        rotation = board_rotation(rng.uniform(-45, 45), rng.uniform(-45, 45),
                                  rng.uniform(-180, 180))
        pixel = (rng.uniform(0.15, 0.85) * camera.width, rng.uniform(0.15, 0.85) * camera.height)
        translation = place_board(camera, rotation, (width_mm / 2, height_mm / 2), pixel,
                                  rng.uniform(*DISTANCE_RANGE_M))
        if not pattern_border_visible(camera, spec, rotation, translation, 4):
            continue
        pixels, _ = corner_pixels(camera, spec, rotation, translation)
        square_px = np.linalg.norm(np.diff(pixels, axis=1), axis=-1).min()
        if square_px >= 8:
            return rotation, translation, []
    raise RuntimeError('no valid chessboard pose found')


def sample_marker_pose(rng, camera, spec):
    """
    A pose of the oversized board with the 7 x 7 corners around the marker L visible.

    Rolls: mostly near level, some near 90 / 180 / 270 deg, some at 20-45 deg (tagged
    'rotated': hard for findChessboardCornersSB with CALIB_CB_LARGER).
    """
    ox, oy = spec.origin_cell
    sq, margin = spec.square_mm, spec.margin_mm
    anchor = (margin + (ox + 1) * sq, margin + (oy + 1) * sq)  # centre of the marker L
    coordinates = spec.board_coordinates()
    around_markers = np.all((coordinates >= -2) & (coordinates <= 4), axis=-1)
    for _ in range(500):
        tags = []
        kind = rng.uniform()
        if kind < 0.7:
            roll = rng.uniform(-12, 12)
        elif kind < 0.85:
            roll = rng.choice([90, 180, 270]) + rng.uniform(-8, 8)
        else:
            roll = rng.choice([-1, 1]) * rng.uniform(20, 45)
            tags.append('rotated')
        rotation = board_rotation(rng.uniform(-35, 35), rng.uniform(-35, 35), roll)
        pixel = (rng.uniform(0.3, 0.7) * camera.width, rng.uniform(0.3, 0.7) * camera.height)
        translation = place_board(camera, rotation, anchor, pixel,
                                  rng.uniform(*DISTANCE_RANGE_M))
        pixels, inside = corner_pixels(camera, spec, rotation, translation)
        if not inside[around_markers].all():
            continue
        if not inside.all():
            tags.append('overflowing')
        return rotation, translation, tags
    raise RuntimeError('no valid marker board pose found')


# ---------------------------------------------------------------------------------- dataset

def sample_appearance(rng):
    """Print, lighting and sensor parameters of one image (IR-like: often low contrast)."""
    dark = rng.uniform(0.1, 0.35)
    light = rng.uniform(max(dark + 0.2, 0.55), 0.9)
    blurred = rng.uniform() < 0.1
    return {
        'levels': (dark, light),
        'background': rng.uniform(0.25, 0.6),
        'blur_sigma': rng.uniform(1.5, 2.5) if blurred else rng.uniform(0.4, 1.2),
        'noise_sigma': rng.uniform(0.003, 0.015),
        'vignetting': rng.uniform(0.0, 0.3),
        'tags': ['blurred'] if blurred else [],
    }


def to_integer_image(image, bit_depth):
    if bit_depth == 8:
        return np.rint(image * 255).astype(np.uint8)
    return np.rint(image * 4095).astype(np.uint16)  # 12-bit sensor in a 16-bit image


def generate(output, model, counts, seed, bit_depth, supersample):
    output = Path(output)
    (output / 'images').mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    camera = make_camera(model)
    print(f'{model} camera {camera.width}x{camera.height}, f = {camera.K[0, 0]:.1f} px; '
          'casting the pixel rays ...')
    rays = supersampled_rays(camera, supersample)
    boards = {spec.kind: (spec, texture_pyramid(draw_board(spec)))
              for spec in (CHESSBOARD, MARKER_BOARD)}
    sampler = {'chessboard': sample_chessboard_pose, 'oversized_marker': sample_marker_pose}

    entries = []
    jobs = ([('chessboard', i) for i in range(counts['chessboard'])]
            + [('oversized_marker', i) for i in range(counts['oversized_marker'])]
            + [(None, i) for i in range(counts['empty'])])
    for kind, index in jobs:
        appearance = sample_appearance(rng)
        entry = {'file': f'images/{kind or "empty"}_{index:03d}.png', 'board': kind,
                 **{key: appearance[key] for key in ('blur_sigma', 'noise_sigma', 'vignetting')}}
        tags = list(appearance['tags'])
        if kind is None:
            image = np.full((camera.height * supersample, camera.width * supersample),
                            appearance['background'], np.float32)
        else:
            spec, pyramid = boards[kind]
            rotation, translation, pose_tags = sampler[kind](rng, camera, spec)
            tags += pose_tags
            image = render_board(rays, pyramid, spec, rotation, translation,
                                 appearance['levels'], appearance['background'])
            pixels, inside = corner_pixels(camera, spec, rotation, translation)
            entry.update({
                'rvec': cv2.Rodrigues(rotation)[0].ravel().tolist(),
                'tvec': translation.tolist(),
                'grid': list(spec.grid),
                'corners': pixels.reshape(-1, 2).tolist(),
                'board_coordinates': spec.board_coordinates().reshape(-1, 2).tolist(),
                'visible': inside.ravel().tolist(),
            })
        entry['tags'] = tags
        image = cv2.resize(image, (camera.width, camera.height), interpolation=cv2.INTER_AREA)
        image = degrade(image, rng, appearance['blur_sigma'], appearance['noise_sigma'],
                        appearance['vignetting'])
        cv2.imwrite(str(output / entry['file']), to_integer_image(image, bit_depth))
        entries.append(entry)
        print(f'{entry["file"]} {" ".join(tags)}')

    truth = {
        'camera': {'model': model, 'width': camera.width, 'height': camera.height,
                   'K': camera.K.tolist(), 'D': camera.D.tolist()},
        'boards': {spec.kind: {'squares_x': spec.squares_x, 'squares_y': spec.squares_y,
                               'square_size': spec.square_mm / 1000,
                               'markers': [list(cell) for cell in spec.markers],
                               'origin_cell': spec.origin_cell}
                   for spec in (CHESSBOARD, MARKER_BOARD)},
        'pose_frame': 'board frame: x right, y down on the page, origin at the page top-left '
                      'corner, metres; rvec / tvec map it into the camera frame',
        'bit_depth': bit_depth,
        'seed': seed,
        'images': entries,
    }
    (output / 'ground_truth.json').write_text(json.dumps(truth, indent=1))
    print(f'{len(entries)} images and ground_truth.json written to {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--output', default='dataset/synthetic')
    parser.add_argument('--model', choices=('pinhole', 'fisheye'), default='pinhole')
    parser.add_argument('--chessboard', type=int, default=100, help='images of the chessboard')
    parser.add_argument('--marker', type=int, default=100,
                        help='images of the oversized marker board')
    parser.add_argument('--empty', type=int, default=10, help='images without a board')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bit-depth', type=int, choices=(8, 16), default=8)
    parser.add_argument('--supersample', type=int, default=3)
    args = parser.parse_args()
    generate(args.output, args.model, {'chessboard': args.chessboard,
                                       'oversized_marker': args.marker, 'empty': args.empty},
             args.seed, args.bit_depth, args.supersample)


if __name__ == '__main__':
    main()
