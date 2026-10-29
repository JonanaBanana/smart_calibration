#!/usr/bin/env python3
"""
Prototype: which covariance of the intrinsics predicts a calibration's true error?

Calibrates from growing random view sets and compares, per uncertainty map cell, the predicted
2-D covariance with the true error (the true ray through the cell projected by the estimated
lens). The squared Mahalanobis distance averages 2 if the uncertainty is right; sqrt(mean / 2) is
the scale it needs. Cells within the corners' angular coverage and beyond it are reported
separately.

Covariances, the poses eliminated:
- standard: noise^2 (J^T J)^-1, independent corner errors (calibration.intrinsic_covariance);
- sandwich: cluster-robust with views as the independent units,
  N^-1 (sum_v g_v g_v^T) N^-1 G / (G - 1);
- jackknife: spread of the linearized leave-one-view-out solutions, (G - 1) / G sum_v d_v d_v^T;
- larger: per cell, the larger of standard and jackknife.

Sources:
- a dataset from synthetic_dataset.py (pinhole or fisheye): the package's detection on rendered
  images;
- --corners LENS: fisheye corners with independent noise from fisheye_benchmark.py, as a control
  (every method then needs a scale of about 1).

--compare calibrates all views from the true corners, from the true corners plus independent
noise of the detection's spread, and from the detected corners.

Usage (venv active, from the repository root):
    python utility/synthetic_dataset.py --model fisheye --fov 160 --marker 0 --empty 0 \\
        --chessboard 120 --supersample 2 --output dataset/fisheye160
    python utility/uncertainty_methods_prototype.py dataset/fisheye160 [--drop-outliers]
    python utility/uncertainty_methods_prototype.py dataset/fisheye160 --compare
    python utility/uncertainty_methods_prototype.py --corners E260
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY / 'smart_calibration'))  # the package source, not install/

from smart_calibration.calibration import (  # noqa: E402
    intrinsic_covariance, map_grid, projection_covariances, view_terms)
from smart_calibration.camera_models import FISHEYE, PLUMB_BOB  # noqa: E402

STEPS = (10, 20, 40, 80)
FREE_K3_AFTER_VIEWS = 20
OUTLIER_PX = 1.0
METHODS = ('standard', 'sandwich', 'jackknife', 'larger')
MODELS = {'pinhole': PLUMB_BOB, 'fisheye': FISHEYE}


def nearest_true(entry, image_points):
    """Return the true corner (N, 2) nearest to each detected corner."""
    true = np.array([c for c in entry['corners'] if c is not None], np.float64)
    points = image_points.reshape(-1, 2).astype(np.float64)
    return true[np.linalg.norm(true[None] - points[:, None], axis=2).argmin(axis=1)]


def load_dataset(dataset, drop_outliers):
    """Return (model, K, D, size, views, true corners per view) of a rendered dataset."""
    import uncertainty_prototype
    truth = json.loads((dataset / 'ground_truth.json').read_text())
    camera = truth['camera']
    detected = uncertainty_prototype.detect_views(dataset, truth, dataset / 'views_cache.npz')
    entries = {e['file']: e for e in truth['images']}
    views, exact = [], []
    for file, _, objects, images, _ in detected:
        true = nearest_true(entries[file], images)
        error = np.linalg.norm(images.reshape(-1, 2) - true, axis=1).max()
        if drop_outliers and error > OUTLIER_PX:
            continue
        views.append((objects, images))
        exact.append(true)
    if drop_outliers:
        print(f'dropped {len(detected) - len(views)} views with a corner more than '
              f'{OUTLIER_PX} px from the truth')
    return (MODELS[camera['model']], np.array(camera['K']), np.array(camera['D']),
            (camera['width'], camera['height']), views, exact)


def load_corners(lens_name):
    """Return (model, K, D, size, views) of a fisheye_benchmark.py corner set."""
    import fisheye_benchmark
    lens = fisheye_benchmark.LENSES[lens_name]
    objects, images = fisheye_benchmark.corner_lists(fisheye_benchmark.views(lens, 120, seed=11))
    return FISHEYE, lens.K, lens.D, fisheye_benchmark.SIZE, list(zip(objects, images))


def ray_angles(rays):
    return np.degrees(np.arccos(rays[:, 2] / np.linalg.norm(rays, axis=1)))


def coverage_deg(model, K, D, views):
    """Return the largest true ray angle of the views' corners."""
    corners = np.concatenate([images.reshape(-1, 2) for _, images in views])
    return float(np.nanmax(ray_angles(model.rays(K, D, corners))))


def map_errors(model, K_true, D_true, K, D, size):
    """Return the map cell pixels (N, 2), their true ray angles and the true errors (N, 2)."""
    pixels = map_grid(size).reshape(-1, 2)
    rays = model.rays(K_true, D_true, pixels)
    errors = np.full(pixels.shape, np.nan)
    valid = np.isfinite(rays).all(axis=1)
    errors[valid] = model.project(rays[valid], np.zeros(3), np.zeros(3), K, D)[0] - pixels[valid]
    return pixels, ray_angles(rays), errors


def without_pose(b, vector):
    """Return vector minus the part a change of the view's pose explains."""
    return vector - b @ np.linalg.solve(b.T @ b, b.T @ vector)


def covariances_by_method(terms, used):
    """Return {method: covariance of the intrinsics} (larger excepted) and the noise."""
    standard, noise = intrinsic_covariance(terms, used)
    reduced = [a.T @ without_pose(b, a) for _, a, b in terms]
    gradients = [a.T @ without_pose(b, r) for r, a, b in terms]
    total = sum(reduced)
    inverse = np.linalg.inv(total)
    views = len(terms)
    sandwich = inverse @ sum(np.outer(g, g) for g in gradients) @ inverse * views / (views - 1)
    steps = np.array([np.linalg.solve(total - n, g) for n, g in zip(reduced, gradients)])
    steps -= steps.mean(axis=0)
    jackknife = (views - 1) / views * steps.T @ steps
    return dict(standard=standard, sandwich=sandwich, jackknife=jackknife), noise


def evaluate(model, K_true, D_true, size, views, coverage):
    """
    Calibrate; return {method: per region (covered, beyond) (d^2, predicted, true error)}.

    Also the noise estimate.
    """
    objects, images = [o for o, _ in views], [i for _, i in views]
    fix = len(views) < FREE_K3_AFTER_VIEWS
    _, K, D, rvecs, tvecs = model.calibrate(objects, images, size, fix)
    used = [i for i in range(len(model.parameters)) if not (fix and i in model.high_order)]
    covariances, noise = covariances_by_method(
        view_terms(model, objects, images, K, D, rvecs, tvecs, used), used)
    pixels, angles, errors = map_errors(model, K_true, D_true, K, D, size)
    cells = {method: projection_covariances(model, K, D, covariance, used, pixels)
             for method, covariance in covariances.items()}
    traces = {m: np.trace(c, axis1=1, axis2=2) for m, c in cells.items()}
    cells['larger'] = np.where((traces['jackknife'] > traces['standard'])[:, None, None],
                               cells['jackknife'], cells['standard'])
    result = {}
    for method in METHODS:
        regions = []
        for region in (angles <= coverage, angles > coverage):
            valid = (region & np.isfinite(cells[method][:, 0, 0])
                     & np.isfinite(errors).all(axis=1))
            c, e = cells[method][valid], errors[valid]
            regions.append((np.einsum('ni,nij,nj->n', e, np.linalg.pinv(c), e),
                            np.sqrt(np.trace(c, axis1=1, axis2=2)), np.linalg.norm(e, axis=1)))
        result[method] = regions
    return result, noise


def growth(model, K, D, size, views, runs):
    coverage = coverage_deg(model, K, D, views)
    print(f'{model.name}: {len(views)} views, corners up to {coverage:.0f} deg off the axis')
    print('  per method: predicted rms (covered cells), needed scale covered / beyond coverage')
    print(f'  {"views":>5s} {"noise":>6s} {"true rms":>8s}   '
          + '   '.join(f'{m:>21s}' for m in METHODS))
    rng = np.random.default_rng(0)
    for k in STEPS:
        if k > len(views):
            break
        rows = [evaluate(model, K, D, size, [views[i] for i in rng.permutation(len(views))[:k]],
                         coverage) for _ in range(runs)]
        actual = np.concatenate([r[0]['standard'][0][2] for r in rows])
        columns = []
        for method in METHODS:
            scales = []
            for region in (0, 1):
                d2 = np.concatenate([r[0][method][region][0] for r in rows])
                scales.append(f'{np.sqrt(np.mean(d2) / 2):5.2f}' if len(d2) else '    -')
            predicted = np.concatenate([r[0][method][0][1] for r in rows])
            columns.append(f'{np.sqrt(np.mean(predicted ** 2)):6.3f} px, {scales[0]}/{scales[1]}')
        print(f'  {k:5d} {np.median([r[1] for r in rows]):6.3f} '
              f'{np.sqrt(np.mean(actual ** 2)):8.3f}   '
              + '   '.join(f'{c:>21s}' for c in columns))


def compare(model, K_true, D_true, size, views, exact):
    """Calibrate all views from true, true + independent noise and detected corners."""
    objects = [o for o, _ in views]
    detected = [i.reshape(-1, 2).astype(np.float64) for _, i in views]
    spread = np.concatenate([p - t for p, t in zip(detected, exact)]).std(axis=0).mean()
    rng = np.random.default_rng(0)
    noisy = [t + rng.normal(0, spread, t.shape) for t in exact]
    coverage = coverage_deg(model, K_true, D_true, views)
    print(f'{model.name}: {len(views)} views, detection error {spread:.3f} px per axis')
    for label, images in (('true corners', exact), ('true + independent noise', noisy),
                          ('detected corners', detected)):
        rms, K, D, _, _ = model.calibrate(objects, [i.astype(np.float32) for i in images],
                                          size, False)
        _, angles, errors = map_errors(model, K_true, D_true, K, D, size)
        error = np.linalg.norm(errors, axis=1)[angles <= coverage]
        error = error[np.isfinite(error)]
        print(f'  {label:25s} rms {rms:.3f} px; lens error within coverage: rms '
              f'{np.sqrt(np.mean(error ** 2)):.3f}, max {error.max():.3f} px')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('dataset', nargs='?', help='synthetic_dataset.py dataset')
    parser.add_argument('--corners', metavar='LENS', help='fisheye_benchmark.py lens instead')
    parser.add_argument('--drop-outliers', action='store_true',
                        help=f'drop views with a corner more than {OUTLIER_PX} px off')
    parser.add_argument('--compare', action='store_true',
                        help='true vs noisy vs detected corners, all views')
    parser.add_argument('--runs', type=int, default=20, help='random view sets per step')
    args = parser.parse_args()
    if args.corners:
        growth(*load_corners(args.corners), args.runs)
        return
    if not args.dataset:
        parser.error('give a dataset or --corners LENS')
    model, K, D, size, views, exact = load_dataset(Path(args.dataset), args.drop_outliers)
    if args.compare:
        compare(model, K, D, size, views, exact)
    else:
        growth(model, K, D, size, views, args.runs)


if __name__ == '__main__':
    main()
