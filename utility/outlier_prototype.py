#!/usr/bin/env python3
"""
Prototype: which views should the final calibration pass remove? (pinhole, plumb_bob)

Uses the views of a synthetic dataset that pass the calibration node's checks (cached by
uncertainty_prototype.py) and corrupts a few of them in ways those per-frame checks cannot see:
smooth over the board, or consistent within the frame but not with the other views.

  bend     rows bowed (a board flexing), up to amp px in the middle
  shear    x shifted proportionally to y (rolling shutter while moving), +-amp px at the ends
  scale    zoomed about the image centre (focus or temperature changed), amp px at the furthest
           corner
  patch    the corners within one square of a random corner shifted by amp px (a local glitch)
  origin   object points shifted by one square (wrong marker origin; marker board only)

Three scores per view, each turned into a robust z-score against all views (median, MAD):

  rms      reprojection error in the calibration from all views
  exact    leave-one-out: calibrate without the view, fit only its pose (solvePnP), its
           reprojection error; one calibration per view
  approx   the same, linearized: one Gauss-Newton step from the full solution with the view's
           normal-equation contribution taken out (the Schur-reduced form the covariance in
           smart_calibration.calibration uses), pose re-fitted by projection; milliseconds

A score flags a view when its z-score exceeds --z. A good score flags no view on clean data and
the corrupted views when they matter.

Usage (venv active, from the repository root; run uncertainty_prototype.py once first):
    python utility/outlier_prototype.py dataset/synthetic [--views 40] [--trials 5]
"""

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

REPOSITORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY / 'smart_calibration'))  # the package source, not install/

from evaluate_dataset import mapping_error  # noqa: E402
from smart_calibration.calibration import calibrate  # noqa: E402
from synthetic_dataset import Camera  # noqa: E402
from uncertainty_prototype import detect_views  # noqa: E402

SIZE = (1280, 720)
FREE_K3_AFTER_VIEWS = 20
CORRUPTIONS = ('bend', 'shear', 'scale', 'patch', 'origin')


# ---------------------------------------------------------------------------------- corruption

def corrupt(kind, objects, images, square, amp, rng):
    """Corrupted copies (object_points, image_points) of one view."""
    objects, images = objects.copy(), images.astype(np.float64).copy()
    pixels, board = images[:, 0], objects[:, 0, :2] / square
    if kind == 'bend':
        u = board[:, 0] - board[:, 0].min()
        u = 2 * u / max(u.max(), 1e-9) - 1
        pixels[:, 1] += amp * (1 - u ** 2)
    elif kind == 'shear':
        y = pixels[:, 1]
        pixels[:, 0] += amp * (y - (y.max() + y.min()) / 2) / max((y.max() - y.min()) / 2, 1e-9)
    elif kind == 'scale':
        centre = np.array([(SIZE[0] - 1) / 2, (SIZE[1] - 1) / 2])
        reach = np.linalg.norm(pixels - centre, axis=1).max()
        pixels[:] = centre + (pixels - centre) * (1 + amp / reach)
    elif kind == 'patch':
        centre = board[rng.integers(len(board))]
        near = np.abs(board - centre).max(axis=1) <= 1
        angle = rng.uniform(0, 2 * np.pi)
        pixels[near] += amp * np.array([np.cos(angle), np.sin(angle)])
    elif kind == 'origin':
        objects[:, 0, 0] += square
    return objects, images.astype(np.float32)


# ------------------------------------------------------------------------------------- scores

def view_terms(objects, images, camera_matrix, distortion, rvecs, tvecs, used):
    """Per view: (A intrinsics jacobian, P pose projector, r residual), plus reduced S_j, b_j."""
    terms = []
    for obj, img, rvec, tvec in zip(objects, images, rvecs, tvecs):
        projected, jacobian = cv2.projectPoints(
            obj.astype(np.float64), rvec, tvec, camera_matrix, distortion)
        r = (img.reshape(-1, 2) - projected.reshape(-1, 2)).ravel()
        a, b = jacobian[:, [6 + i for i in used]], jacobian[:, :6]
        p = np.eye(len(r)) - b @ np.linalg.solve(b.T @ b, b.T)
        terms.append((a, p, r, a.T @ p @ a, a.T @ p @ r))
    return terms


def approx_loo(terms):
    """Linearized leave-one-out reprojection RMS of each view, px."""
    total = sum(t[3] for t in terms)
    errors = []
    for a, p, r, s, b in terms:
        step = -np.linalg.solve(total - s, b)  # intrinsics change without this view
        moved = p @ (r - a @ step)             # its residual, pose re-fitted
        errors.append(np.sqrt(np.sum(moved ** 2) / (len(r) / 2)))
    return np.array(errors)


def exact_loo(objects, images, fix_k3):
    """Leave-one-out reprojection RMS of each view: calibrate without it, solvePnP it."""
    errors = []
    for i in range(len(objects)):
        rest = [k for k in range(len(objects)) if k != i]
        _, K, D, _, _ = calibrate([objects[k] for k in rest], [images[k] for k in rest], SIZE,
                                  fix_k3)
        ok, rvec, tvec = cv2.solvePnP(objects[i], images[i], K, D)
        projected, _ = cv2.projectPoints(objects[i].astype(np.float64), rvec, tvec, K, D)
        errors.append(np.sqrt(np.mean(np.sum(
            (images[i].reshape(-1, 2) - projected.reshape(-1, 2)) ** 2, axis=1))))
    return np.array(errors)


def remove_outliers(objects, images, z, max_fraction=0.1):
    """
    Remove the worst view while its approx leave-one-out z-score exceeds z; return kept indices.

    One at a time with a recalibration in between (a bad view also raises the others' errors);
    at most max_fraction of the views.
    """
    keep = list(range(len(objects)))
    while len(keep) > (1 - max_fraction) * len(objects):
        result, _, _ = scores([objects[i] for i in keep], [images[i] for i in keep], False)
        zs = robust_z(result['approx'])
        worst = int(np.argmax(zs))
        if zs[worst] <= z:
            break
        del keep[worst]
    return keep


def robust_z(values):
    median = np.median(values)
    scale = 1.4826 * np.median(np.abs(values - median))
    return (values - median) / max(scale, 1e-12)


def scores(objects, images, with_exact):
    """{'rms', 'approx'[, 'exact']}: per-view scores, and the full calibration (K, D)."""
    fix_k3 = len(objects) < FREE_K3_AFTER_VIEWS
    _, K, D, rvecs, tvecs = calibrate(objects, images, SIZE, fix_k3)
    used = [i for i in range(9) if not (fix_k3 and i == 8)]
    terms = view_terms(objects, images, K, D, rvecs, tvecs, used)
    result = {'rms': np.array([np.sqrt(np.sum(t[2] ** 2) / (len(t[2]) / 2)) for t in terms])}
    start = time.perf_counter()
    result['approx'] = approx_loo(terms)
    result['approx_s'] = time.perf_counter() - start
    if with_exact:
        start = time.perf_counter()
        result['exact'] = exact_loo(objects, images, fix_k3)
        result['exact_s'] = time.perf_counter() - start
    return result, K, D


# --------------------------------------------------------------------------------- experiments

def lens_error(camera, objects, images):
    _, K, D, _, _ = calibrate(objects, images, SIZE, len(objects) < FREE_K3_AFTER_VIEWS)
    (worst, mean), _ = mapping_error(camera, K, D)
    return worst, mean


def clean_experiment(camera, views, n, trials, z, rng):
    """No corruption: how many views each score flags, and exact vs approx agreement."""
    flags = {'rms': [], 'approx': [], 'exact': []}
    for trial in range(trials):
        subset = [views[i] for i in rng.permutation(len(views))[:n]]
        result, _, _ = scores([v[0] for v in subset], [v[1] for v in subset], trial == 0)
        for name in flags:
            if name in result:
                flags[name].append(int(np.sum(robust_z(result[name]) > z)))
        if trial == 0:
            exact, approx = result['exact'], result['approx']
            print(f'    exact vs approx leave-one-out: correlation '
                  f'{np.corrcoef(exact, approx)[0, 1]:.3f}, approx / exact median '
                  f'{np.median(approx / exact):.3f}; time exact {result["exact_s"]:.1f} s, '
                  f'approx {result["approx_s"] * 1000:.0f} ms; max z: rms '
                  f'{robust_z(result["rms"]).max():.1f}, exact {robust_z(exact).max():.1f}, '
                  f'approx {robust_z(approx).max():.1f}')
    print('    views flagged on clean data (per trial): ' + ', '.join(
        f'{name} {values}' for name, values in flags.items() if values))


def corruption_experiment(camera, views, mode, square, n, trials, z, amps, rng):
    """Two corrupted views per trial: z-scores of the corrupted views and false flags."""
    print(f'    {"corruption":>10s} {"amp":>4s}   z of corrupted (min over trials): '
          f'{"rms":>5s} {"approx":>6s}   false flags: {"rms":>3s} {"approx":>6s}   removed '
          f'(iterative): corrupted / good   lens error max px: clean / with / after removal')
    for kind in CORRUPTIONS:
        if kind == 'origin' and mode != 'oversized_marker':
            continue
        for amp in (amps if kind != 'origin' else (0.0,)):
            z_bad = {'rms': [], 'approx': []}
            false = {'rms': 0, 'approx': 0}
            errors, removed_bad, removed_good = [], 0, 0
            for _ in range(trials):
                order = rng.permutation(len(views))[:n]
                objects = [views[i][0] for i in order]
                images = [views[i][1] for i in order]
                clean = lens_error(camera, objects, images)
                bad = rng.choice(n, 2, replace=False)
                for i in bad:
                    objects[i], images[i] = corrupt(kind, objects[i], images[i], square, amp, rng)
                result, _, _ = scores(objects, images, False)
                for name in z_bad:
                    zs = robust_z(result[name])
                    z_bad[name].append(zs[bad].min())
                    false[name] += int(np.sum(np.delete(zs, bad) > z))
                corrupted = lens_error(camera, objects, images)
                keep = remove_outliers(objects, images, z)
                gone = set(range(n)) - set(keep)
                removed_bad += len(gone & set(bad.tolist()))
                removed_good += len(gone - set(bad.tolist()))
                removed = lens_error(camera, [objects[i] for i in keep],
                                     [images[i] for i in keep])
                errors.append((clean[0], corrupted[0], removed[0]))
            e = np.median(errors, axis=0)
            print(f'    {kind:>10s} {amp:4.1f}   {"":33s}{min(z_bad["rms"]):5.1f} '
                  f'{min(z_bad["approx"]):6.1f}   {"":13s}{false["rms"]:3d} {false["approx"]:6d}'
                  f'   {"":21s}{removed_bad:2d} / {removed_good:2d}   {"":19s}'
                  f'{e[0]:.2f} / {e[1]:.2f} / {e[2]:.2f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('dataset', nargs='?', default='dataset/synthetic')
    parser.add_argument('--views', type=int, default=40, help='views per calibration')
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--z', type=float, default=6.0, help='robust z-score that flags a view')
    parser.add_argument('--amps', type=float, nargs='+', default=[0.5, 1.0, 2.0])
    args = parser.parse_args()
    dataset = Path(args.dataset)
    cache = dataset / 'uncertainty_prototype_cache.npz'
    if not cache.exists():
        sys.exit(f'{cache} missing: run utility/uncertainty_prototype.py first')
    truth = json.loads((dataset / 'ground_truth.json').read_text())
    cam = truth['camera']
    camera = Camera(cam['model'], cam['width'], cam['height'], np.array(cam['K']),
                    np.array(cam['D']))
    views = detect_views(dataset, truth, cache)
    rng = np.random.default_rng(0)
    for mode in ('chessboard', 'oversized_marker'):
        selected = [(v[2], v[3]) for v in views if v[1] == mode]
        square = truth['boards'][mode]['square_size']
        print(f'== {mode}: {len(selected)} views, {args.views} per calibration, z > {args.z}')
        print('  clean views:')
        clean_experiment(camera, selected, args.views, args.trials, args.z, rng)
        print('  two corrupted views per calibration:')
        corruption_experiment(camera, selected, mode, square, args.views, args.trials, args.z,
                              args.amps, rng)
        print()


if __name__ == '__main__':
    main()
