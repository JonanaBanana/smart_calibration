#!/usr/bin/env python3
"""
Benchmark fisheye calibration: the package's calibrator against cv2.fisheye.calibrate.

Synthetic corners (no images) of a 12 x 9 corner board on a 1280 x 800 sensor:
- Lenses: near-equidistant ones from 140 to 260 degrees diagonal field of view, and the classic
  projections (equisolid, stereographic, orthographic, pinhole) as Kannala-Brandt fits.
- Views as the detector delivers them: a complete 5 x 5 block of detectable corners, then corners
  within border_margin_px of the border dropped. Corners seen more than 75 degrees off the board
  normal count as undetectable; noise grows as the lens squeezes the squares.
- Ground truth is projected with theta = atan2(rho, z), independently of the package.

A run is correct with the focal length within 1 % and an rms below 1 px. The model error is the
largest pixel error against the true lens over the image, within the views' coverage.

Usage (venv active, from the repository root):
    python utility/fisheye_benchmark.py                 # all lenses, all and < 90 deg corners
    python utility/fisheye_benchmark.py --few-views     # 5 to 30 views, k3 and k4 fixed below 20
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from multiprocessing import Pool
import os
from pathlib import Path
import sys
import time

# One math library thread per worker process.
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '1')

import cv2  # noqa: E402
import numpy as np  # noqa: E402

REPOSITORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY / 'smart_calibration'))  # the package source, not install/

from smart_calibration.fisheye_calibration import calibrate as package_calibrate  # noqa: E402

SIZE = (1280, 800)
GRID = (12, 9)           # corners per row, rows
SQUARE = 0.05            # m
BORDER_MARGIN = 10.0     # px, as border_margin_px
MAX_GRAZING_DEG = 75.0   # corners seen more obliquely than this are not detected
MIN_BLOCK = 5            # detector minimum pattern, 5 x 5 corners
REF_SQUARE_PX = 40.0     # noise grows as squares get smaller than this
NOISE = 0.2              # px per axis, at REF_SQUARE_PX and larger
MAX_F_ERROR = 0.01
MAX_RMS = 1.0
FEW_VIEW_COUNTS = (5, 8, 12, 20, 30)
FREE_K3_AFTER_VIEWS = 20


@dataclass(frozen=True)
class Lens:
    name: str
    K: np.ndarray
    D: np.ndarray
    max_view_deg: float   # boards are placed up to this angle off the axis


def theta_d(theta, D):
    t2 = np.square(theta)
    return theta * (1 + t2 * (D[0] + t2 * (D[1] + t2 * (D[2] + t2 * D[3]))))


# Image radius / f of the classic projections, by ray angle.
SHAPES = {'equisolid': lambda t: 2 * np.sin(t / 2), 'stereographic': lambda t: 2 * np.tan(t / 2),
          'orthographic': np.sin, 'pinhole': np.tan}


def shape_distortion(shape, diagonal_fov_deg):
    """Return the Kannala-Brandt coefficients fitted to a classic projection over its field."""
    theta = np.linspace(1e-4, math.radians(diagonal_fov_deg / 2), 400)
    powers = np.column_stack([theta ** 3, theta ** 5, theta ** 7, theta ** 9])
    return np.linalg.lstsq(powers, SHAPES[shape](theta) - theta, rcond=None)[0]


def make_lens(name, diagonal_fov_deg, D, max_view_deg=85.0):
    """Lens whose image corners lie at half the diagonal field of view."""
    D = (shape_distortion(D, diagonal_fov_deg) if isinstance(D, str)
         else np.asarray(D, np.float64))
    corner = math.radians(diagonal_fov_deg / 2)
    fy = math.hypot(SIZE[0] / 2, SIZE[1] / 2) / theta_d(corner, D)
    # Slightly off-centre, slightly non-square pixels, as real cameras.
    K = np.array([[fy * 1.002, 0.0, SIZE[0] / 2 - 0.5 + 2.3],
                  [0.0, fy, SIZE[1] / 2 - 0.5 - 1.7], [0.0, 0.0, 1.0]])
    theta = np.linspace(0, min(corner + math.radians(5), math.pi), 2000)
    assert np.all(np.diff(theta_d(theta, D)) > 0), f'{name} folds within its field of view'
    return Lens(name, K, D, max_view_deg)


LENSES = {lens.name: lens for lens in (
    make_lens('A140', 140, [0.02, -0.005, 0.0, 0.0]),
    make_lens('B185', 185, [0.04, -0.01, 0.002, 0.0]),
    make_lens('C200', 200, [-0.03, 0.005, -0.0005, 0.0]),
    make_lens('D220', 220, [0.05, -0.01, 0.0015, -0.0001]),
    # Boards out to 115 degrees, in the lens's outer ring.
    make_lens('E260', 260, [0.02, -0.003, 0.0, 0.0], max_view_deg=115.0),
    make_lens('EQS180', 180, 'equisolid'),
    make_lens('STG190', 190, 'stereographic'),
    make_lens('ORT160', 160, 'orthographic'),
    # Ordinary lenses: boards within the field of view.
    make_lens('PIN100', 100, 'pinhole', max_view_deg=60.0),
    make_lens('PIN60', 60, 'pinhole', max_view_deg=35.0),
)}


def project(lens, points):
    """Return exact pixels (N, 2) and ray angles (N,) of camera-frame points (N, 3)."""
    theta = np.arctan2(np.hypot(points[:, 0], points[:, 1]), points[:, 2])
    phi = np.arctan2(points[:, 1], points[:, 0])
    r = theta_d(theta, lens.D)
    pixels = np.column_stack([lens.K[0, 0] * r * np.cos(phi) + lens.K[0, 2],
                              lens.K[1, 1] * r * np.sin(phi) + lens.K[1, 2]])
    return pixels, theta


def has_block(cells, size=MIN_BLOCK):
    """Return whether corners at grid cells (N, 2) (row, column) contain a size x size block."""
    grid = np.zeros((GRID[1], GRID[0]), bool)
    grid[cells[:, 0], cells[:, 1]] = True
    return any(grid[r:r + size, c:c + size].all()
               for r in range(GRID[1] - size + 1) for c in range(GRID[0] - size + 1))


@dataclass
class View:
    object_points: np.ndarray   # (N, 3) board frame
    image_points: np.ndarray    # (N, 2) with noise
    theta_deg: np.ndarray       # (N,) true ray angles
    cells: np.ndarray           # (N, 2) grid row, column of each corner


def views(lens, count, seed):
    """Return count random detectable views of the board, facing the camera, tilted."""
    rng = np.random.default_rng(seed)
    cells = np.mgrid[0:GRID[1], 0:GRID[0]].reshape(2, -1).T  # (row, column)
    board = np.zeros((len(cells), 3))
    board[:, 0], board[:, 1] = cells[:, 1] * SQUARE, cells[:, 0] * SQUARE
    result = []
    while len(result) < count:
        # Uniform on the sphere cap, so weighted outwards.
        cos_theta = rng.uniform(math.cos(math.radians(lens.max_view_deg)), 1.0)
        phi = rng.uniform(0, 2 * math.pi)
        s = math.sqrt(1 - cos_theta ** 2)
        direction = np.array([s * math.cos(phi), s * math.sin(phi), cos_theta])
        # Face the camera (board normal along the ray), roll, then tilt up to 45 degrees.
        helper = [0.0, 1.0, 0.0] if abs(direction[1]) < 0.9 else [1.0, 0.0, 0.0]
        x = np.cross(helper, direction)
        x /= np.linalg.norm(x)
        R = np.column_stack([x, np.cross(direction, x), direction])
        R = cv2.Rodrigues(direction * rng.uniform(0, 2 * math.pi))[0] @ R
        tilt_axis = rng.uniform(0, 2 * math.pi)
        R = R @ cv2.Rodrigues(np.array([math.cos(tilt_axis), math.sin(tilt_axis), 0.0])
                              * math.radians(rng.uniform(0, 45)))[0]
        tvec = direction * rng.uniform(0.35, 1.0) - R @ board.mean(axis=0)
        points = board @ R.T + tvec
        pixels, theta = project(lens, points)
        rays = points / np.linalg.norm(points, axis=1, keepdims=True)
        grazing = np.degrees(np.arccos(np.clip(np.abs(rays @ R[:, 2]), 0, 1)))
        in_image = ((pixels >= -0.5) & (pixels < np.array(SIZE) - 0.5)).all(axis=1)
        detectable = in_image & (grazing < MAX_GRAZING_DEG)
        if not has_block(cells[detectable]):
            continue
        inside = ((pixels >= BORDER_MARGIN)
                  & (pixels < np.array(SIZE) - 1 - BORDER_MARGIN)).all(axis=1)
        keep = detectable & inside
        # Local square size from the grid neighbours: smaller squares, noisier corners.
        grid = pixels.reshape(GRID[1], GRID[0], 2)
        dx = np.linalg.norm(np.gradient(grid, axis=1), axis=2)
        dy = np.linalg.norm(np.gradient(grid, axis=0), axis=2)
        local = np.sqrt(dx * dy).ravel()
        sigma = NOISE * np.clip(np.sqrt(REF_SQUARE_PX / np.maximum(local, 1e-6)), 1, 3)
        noisy = pixels + rng.normal(0, 1, pixels.shape) * sigma[:, None]
        result.append(View(board[keep], noisy[keep], np.degrees(theta[keep]), cells[keep]))
    return result


def corner_lists(all_views, max_theta_deg=None):
    """
    Return (object points, image points) lists for calibration.

    max_theta_deg: only corners below this ray angle; a view is then kept only if those still
    contain a complete 5 x 5 block.
    """
    objects, images = [], []
    for view in all_views:
        mask = np.ones(len(view.cells), bool)
        if max_theta_deg is not None:
            mask = view.theta_deg < max_theta_deg
            if not has_block(view.cells[mask]):
                continue
        objects.append(view.object_points[mask].astype(np.float32).reshape(-1, 1, 3))
        images.append(view.image_points[mask].astype(np.float32).reshape(-1, 1, 2))
    return objects, images


def model_error(lens, K, D, max_theta_deg):
    """Return the largest pixel error of (K, D) against the lens, over pixels within coverage."""
    xs, ys = np.meshgrid(np.linspace(0, SIZE[0] - 1, 81), np.linspace(0, SIZE[1] - 1, 51))
    pixels = np.column_stack([xs.ravel(), ys.ravel()])
    normalized = (pixels - lens.K[:2, 2]) / [lens.K[0, 0], lens.K[1, 1]]
    r = np.hypot(normalized[:, 0], normalized[:, 1])
    low, high = np.zeros_like(r), np.full_like(r, math.pi)
    for _ in range(60):  # the true ray angle of each pixel
        middle = (low + high) / 2
        short = theta_d(middle, lens.D) < r
        low, high = np.where(short, middle, low), np.where(short, high, middle)
    theta, phi = (low + high) / 2, np.arctan2(normalized[:, 1], normalized[:, 0])
    rays = np.column_stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                            np.cos(theta)])
    estimate = Lens('estimate', np.asarray(K, np.float64), np.ravel(D).astype(np.float64), 0)
    error = np.linalg.norm(project(estimate, rays)[0] - pixels, axis=1)
    within = error[np.degrees(theta) <= max_theta_deg]
    return float(within.max()) if len(within) else math.nan


def opencv(objects, images, fix_k3_k4):
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    if fix_k3_k4:
        flags |= cv2.fisheye.CALIB_FIX_K3 | cv2.fisheye.CALIB_FIX_K4
    rms, K, D, _, _ = cv2.fisheye.calibrate(
        [o.astype(np.float64) for o in objects], [i.astype(np.float64) for i in images], SIZE,
        None, None, flags=flags)
    return rms, K, D.ravel()


def package(objects, images, fix_k3_k4):
    rms, K, D, _, _ = package_calibrate(objects, images, SIZE, fix_k3_k4)
    return rms, K, D


CALIBRATORS = {'opencv': opencv, 'package': package}


def run(task):
    """Calibrate one synthetic dataset; return a result dict."""
    cv2.setNumThreads(1)
    lens_name, count, seed, corners, calibrator = task
    lens = LENSES[lens_name]
    all_views = views(lens, count, seed)
    objects, images = corner_lists(all_views, 90.0 if corners == '<90' else None)
    coverage = max(float(v.theta_deg.max()) for v in all_views)
    result = dict(lens=lens_name, views=count, seed=seed, corners=corners,
                  calibrator=calibrator, ok=False)
    start = time.perf_counter()
    try:
        rms, K, D = CALIBRATORS[calibrator](objects, images, count < FREE_K3_AFTER_VIEWS)
    except cv2.error as error:
        result.update(error=str(error).strip().splitlines()[-1][-120:],
                      seconds=time.perf_counter() - start)
        return result
    result['seconds'] = time.perf_counter() - start
    if not (math.isfinite(rms) and np.isfinite(K).all() and np.isfinite(D).all()):
        result['error'] = 'non-finite result'
        return result
    f_error = abs(K[1, 1] / lens.K[1, 1] - 1)
    # Wider tolerance on f with few views.
    max_f_error = MAX_F_ERROR if count >= FREE_K3_AFTER_VIEWS else 3 * MAX_F_ERROR
    result.update(rms=float(rms), f_error=float(f_error),
                  c_error=float(np.hypot(*(K[:2, 2] - lens.K[:2, 2]))),
                  model_error=model_error(lens, K, D, coverage),
                  ok=bool(rms < MAX_RMS and f_error < max_f_error))
    return result


def report(results):
    groups = defaultdict(list)
    for r in results:
        groups[(r['lens'], r['views'], r['corners'])].append(r)
    calibrators = sorted({r['calibrator'] for r in results})
    print(f"{'lens':5s} {'views':>5s} {'corners':7s} "
          + ' '.join(f'{c:>10s}' for c in calibrators)
          + '   package: rms, model error med/max (px), seconds med/max')
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2])):
        rows = groups[key]
        counts = []
        for c in calibrators:
            mine = [r for r in rows if r['calibrator'] == c]
            counts.append(f"{sum(r['ok'] for r in mine):>7d}/{len(mine):<2d}")
        ok = [r for r in rows if r['calibrator'] == 'package' and r['ok']]
        seconds = [r['seconds'] for r in rows if r['calibrator'] == 'package']
        detail = (f"{np.median([r['rms'] for r in ok]):.3f}, "
                  f"{np.median([r['model_error'] for r in ok]):.2f}/"
                  f"{max(r['model_error'] for r in ok):.2f}, "
                  f"{np.median(seconds):.2f}/{max(seconds):.2f}") if ok else '-'
        print(f'{key[0]:5s} {key[1]:5d} {key[2]:7s} ' + ' '.join(counts) + '   ' + detail)
    failures = [r for r in results if r['calibrator'] == 'package' and not r['ok']]
    print(f'\npackage failures: {len(failures)}')
    for r in failures:
        print(f"  {r['lens']} {r['views']} views {r['corners']} seed {r['seed']}: "
              + (r.get('error') or f"rms {r['rms']:.3f}, f error {100 * r['f_error']:.2f} %"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--few-views', action='store_true',
                        help=f'views {FEW_VIEW_COUNTS} instead of 89, all corners')
    parser.add_argument('--views', type=int, default=89)
    parser.add_argument('--seeds', type=int, default=20, help='datasets per lens')
    parser.add_argument('--lenses', default=','.join(LENSES))
    parser.add_argument('--calibrators', default=','.join(CALIBRATORS))
    parser.add_argument('--jobs', type=int, default=8, help='parallel processes')
    parser.add_argument('--output', help='also write all results to this JSON file')
    args = parser.parse_args()
    lenses, calibrators = args.lenses.split(','), args.calibrators.split(',')
    if args.few_views:
        tasks = [(lens, count, 1000 + seed, 'all', c) for lens in lenses
                 for count in FEW_VIEW_COUNTS for seed in range(args.seeds) for c in calibrators]
    else:
        tasks = [(lens, args.views, seed, corners, c) for lens in lenses
                 for seed in range(args.seeds) for corners in ('all', '<90') for c in calibrators]
    start = time.perf_counter()
    with Pool(args.jobs) as pool:
        results = pool.map(run, tasks, chunksize=1)
    print(f'{len(results)} calibrations in {time.perf_counter() - start:.0f} s, '
          f'{args.jobs} in parallel\n')
    report(results)
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
