#!/usr/bin/env python3
"""
Prototype: does a calibration's own uncertainty predict its true error? (pinhole only)

Calibrates from growing random sets of the detected views of a synthetic dataset and compares
the predicted uncertainty map with the true error map, and the estimated board tilts with the
true ones. If the uncertainty is right, the squared Mahalanobis distance of the true error
averages 2.

Usage (venv active, from the repository root):
    python utility/uncertainty_prototype.py dataset/synthetic [--runs 20]
"""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

REPOSITORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY / 'smart_calibration'))  # the package source, not install/

from evaluate_dataset import corner_check  # noqa: E402
from smart_calibration.detection import detect_board, sb_flags  # noqa: E402
from smart_calibration.preprocessing import preprocess  # noqa: E402
from smart_calibration.views import image_points, object_points  # noqa: E402
from synthetic_dataset import Camera, pixel_rays, project  # noqa: E402

INTRINSICS = ('fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3')
GROUPS = {'focal': (0, 1), 'centre': (2, 3), 'distortion': (4, 5, 6, 7, 8)}


# --------------------------------------------------------------------------------- detection

def detect_views(dataset, truth, cache):
    """
    Return the views [(file, mode, object_points, image_points, true_rvec)] passing the checks.

    Border rule and corner check as the node's defaults; cached in the file cache.
    """
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        return list(data['views'])
    chessboard = truth['boards']['chessboard']
    cols, rows = chessboard['squares_x'] - 1, chessboard['squares_y'] - 1
    flags = sb_flags(exhaustive=True)
    views = []
    for entry in truth['images']:
        mode = entry['board']
        if mode is None:
            continue
        image = cv2.imread(str(dataset / entry['file']), cv2.IMREAD_UNCHANGED)
        board = detect_board(preprocess(image, 0.5, 99.5, 1.0), mode, cols, rows, flags, 10.0)
        if board is None or corner_check(board, 0.1)[1].any():
            continue
        square = truth['boards'][mode]['square_size']
        views.append((entry['file'], mode, object_points(board, square), image_points(board),
                      np.array(entry['rvec'])))
    np.savez(cache, views=np.array(views, dtype=object))
    return views


# ----------------------------------------------------------------------- calibration, covariance

def calibrate(views, size, fix_k3):
    """Pinhole calibration of [(object_points, image_points)]; return (rms, K, D, rvecs, tvecs)."""
    flags = cv2.CALIB_FIX_K3 if fix_k3 else 0
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        [o for o, _ in views], [i for _, i in views], size, None, None, flags=flags)
    return rms, K, D.ravel(), rvecs, tvecs


def intrinsic_covariance(views, K, D, rvecs, tvecs, fix_k3):
    """Return (covariance of the intrinsics, point noise, indices into INTRINSICS used)."""
    used = list(range(8 if fix_k3 else 9))
    columns = [6 + i for i in used]  # projectPoints' jacobian: rvec, tvec, fx, fy, cx, cy, dist
    n = len(used)
    reduced = np.zeros((n, n))
    squared, observations = 0.0, 0
    for (obj, img), rvec, tvec in zip(views, rvecs, tvecs):
        projected, jacobian = cv2.projectPoints(obj.astype(np.float64), rvec, tvec, K, D)
        squared += float(np.sum((img.reshape(-1, 2) - projected.reshape(-1, 2)) ** 2))
        observations += 2 * len(obj)
        a, b = jacobian[:, columns], jacobian[:, :6]
        reduced += a.T @ a - a.T @ b @ np.linalg.solve(b.T @ b, b.T @ a)
    dof = observations - n - 6 * len(views)
    sigma = np.sqrt(squared / dof)
    return sigma ** 2 * np.linalg.inv(reduced), sigma, used


def grid_pixels(width, height, step):
    us, vs = np.meshgrid(np.arange(step / 2, width, step), np.arange(step / 2, height, step))
    return np.column_stack([us.ravel(), vs.ravel()])


def projection_covariances(K, D, covariance, used, pixels, width, height):
    """(N, 2, 2) covariance of where the estimate projects the ray through each pixel."""
    rays = pixel_rays(Camera('pinhole', width, height, K, D), pixels)
    result = np.full((len(pixels), 2, 2), np.nan)
    valid = np.isfinite(rays[:, 0])
    points = np.column_stack([rays[valid], np.ones(valid.sum())]).reshape(-1, 1, 3)
    _, jacobian = cv2.projectPoints(points, np.zeros(3), np.zeros(3), K, D)
    j = jacobian[:, [6 + i for i in used]].reshape(-1, 2, len(used))
    result[valid] = j @ covariance @ j.transpose(0, 2, 1)
    return result


def group_covariance(covariance, used, group):
    """The covariance with only one parameter group's uncertainty (the others taken as exact)."""
    keep = [k for k, i in enumerate(used) if i in GROUPS[group]]
    partial = np.zeros_like(covariance)
    partial[np.ix_(keep, keep)] = covariance[np.ix_(keep, keep)]
    return partial


def true_errors(camera, K, D, pixels):
    """(N, 2) error of the estimated model projecting the true ray through each pixel."""
    rays = pixel_rays(camera, pixels)
    estimated = project(Camera('pinhole', camera.width, camera.height, K, D),
                        np.column_stack([rays, np.ones(len(rays))]))
    return estimated - pixels


# ---------------------------------------------------------------------------------- board tilt

def tilt(rvec):
    """(tilt angle, tilt axis direction in the image) of a board pose, degrees."""
    normal = cv2.Rodrigues(np.asarray(rvec, np.float64))[0][:, 2]
    angle = np.degrees(np.arccos(min(abs(normal[2]), 1.0)))
    axis = (np.degrees(np.arctan2(normal[1], normal[0])) + 90.0) % 180.0
    return angle, axis


def axis_alignment(axis):
    """Angle of a tilt axis to the nearest image axis, degrees (0: aligned, 45: diagonal)."""
    off = axis % 90.0
    return min(off, 90.0 - off)


# --------------------------------------------------------------------------------- experiments

def summarize(camera, views, size, fix_k3, pixels):
    """Calibrate; return a dict of predicted and true values for one view set."""
    rms, K, D, rvecs, tvecs = calibrate(views, size, fix_k3)
    covariance, sigma, used = intrinsic_covariance(views, K, D, rvecs, tvecs, fix_k3)
    covariances = projection_covariances(K, D, covariance, used, pixels, *size)
    errors = true_errors(camera, K, D, pixels)
    valid = np.isfinite(covariances[:, 0, 0]) & np.isfinite(errors[:, 0])
    covariances, errors = covariances[valid], errors[valid]
    predicted = np.sqrt(np.trace(covariances, axis1=1, axis2=2))
    actual = np.linalg.norm(errors, axis=1)
    mahalanobis = np.einsum('ni,nij,nj->n', errors, np.linalg.inv(covariances), errors)
    groups = {}
    for group in GROUPS:
        partial = projection_covariances(K, D, group_covariance(covariance, used, group), used,
                                         pixels, *size)[valid]
        groups[group] = float(np.sqrt(np.trace(partial, axis1=1, axis2=2)).max())
    return dict(rms=rms, sigma=sigma, K=K, D=D, covariance=covariance, used=used,
                predicted=predicted, actual=actual, mahalanobis=mahalanobis, groups=groups,
                rvecs=rvecs)


def check_against_opencv(views, size):
    """Compare our standard deviations with calibrateCameraExtended's (same model)."""
    objects, images = [o for o, _ in views], [i for _, i in views]
    result = cv2.calibrateCameraExtended(objects, images, size, None, None)
    rms, K, D, rvecs, tvecs, std_intrinsics = result[:6]
    covariance, _, used = intrinsic_covariance(views, K, D.ravel(), rvecs, tvecs, False)
    ours = np.sqrt(np.diag(covariance))
    # OpenCV's order: fx, fy, cx, cy, k1, k2, p1, p2, k3, ...
    theirs = std_intrinsics.ravel()[:9]
    print('  std deviations, ours vs calibrateCameraExtended:')
    print('    ' + '  '.join(f'{INTRINSICS[i]} {a:.3g}/{b:.3g}'
                             for i, a, b in zip(used, ours, theirs)))


def growth_experiment(camera, views, size, runs, steps, fix_k3_below, pixels, rng):
    """Random view orders: prediction vs truth after the first k views, for each k in steps."""
    print(f'  {"views":>5s} {"k3":>5s} {"sigma":>6s} {"max pred":>8s} {"max true":>8s} '
          f'{"rms pred":>8s} {"rms true":>8s} {"maha":>6s} {"<2.5s":>6s} '
          f'{"corr":>5s}   max per group: focal / centre / distortion')
    for k in steps:
        if k > len(views):
            break
        rows = []
        for _ in range(runs):
            subset = [views[i] for i in rng.permutation(len(views))[:k]]
            rows.append(summarize(camera, subset, size, k < fix_k3_below, pixels))
        predicted = np.concatenate([r['predicted'] for r in rows])
        actual = np.concatenate([r['actual'] for r in rows])
        mahalanobis = np.concatenate([r['mahalanobis'] for r in rows])
        within = np.mean(actual < 2.5 * predicted)
        # Does the map's shape follow the error's? Correlation over each run's grid.
        correlation = np.mean([np.corrcoef(r['predicted'], r['actual'])[0, 1] for r in rows])
        groups = {g: np.median([r['groups'][g] for r in rows]) for g in GROUPS}
        print(f'  {k:5d} {"fixed" if k < fix_k3_below else "free":>5s} '
              f'{np.median([r["sigma"] for r in rows]):6.3f} '
              f'{np.median([r["predicted"].max() for r in rows]):8.3f} '
              f'{np.median([r["actual"].max() for r in rows]):8.3f} '
              f'{np.sqrt(np.mean(predicted ** 2)):8.3f} {np.sqrt(np.mean(actual ** 2)):8.3f} '
              f'{np.mean(mahalanobis):6.2f} {within:6.1%} {correlation:5.2f}   '
              f'{groups["focal"]:.3f} / {groups["centre"]:.3f} / {groups["distortion"]:.3f}')


def tilt_experiment(views, result):
    """Tilt from the estimated poses against the true poses; spread of the view set."""
    estimated = np.array([tilt(r) for r in result['rvecs']])
    true = np.array([tilt(v[4]) for v in views])
    angle_error = np.abs(estimated[:, 0] - true[:, 0])
    axis_error = np.abs((estimated[:, 1] - true[:, 1] + 90.0) % 180.0 - 90.0)
    tilted = true[:, 0] > 10.0  # the axis direction is meaningless for a flat board
    print(f'  tilt angle from the estimate vs true: max error {angle_error.max():.2f} deg; '
          f'axis direction (boards > 10 deg): max error {axis_error[tilted].max():.2f} deg')
    print(f'  true tilt: median {np.median(true[:, 0]):.0f} deg (5-95 %: '
          f'{np.percentile(true[:, 0], 5):.0f}-{np.percentile(true[:, 0], 95):.0f} deg); '
          f'{np.sum(true[:, 0] < 10)} of {len(true)} views < 10 deg; axis within 10 deg of an '
          f'image axis: {np.sum([axis_alignment(a) < 10 for a in true[tilted, 1]])} of '
          f'{tilted.sum()} tilted views')


def degenerate_experiment(camera, views, size, k, runs, pixels, rng):
    """Calibrate from view sets that are degenerate in one way vs random sets of the same size."""
    true = np.array([tilt(v[4]) for v in views])
    views = [(v[2], v[3]) for v in views]
    sets = {
        'random': np.ones(len(views), bool),
        'flat (< 15 deg)': true[:, 0] < 15,
        'tilted, axis within 15 deg of an image axis':
            (true[:, 0] >= 15) & np.array([axis_alignment(a) < 15 for a in true[:, 1]]),
        'tilted, axis 30-60 deg from the image axes':
            (true[:, 0] >= 15) & np.array([axis_alignment(a) > 30 for a in true[:, 1]]),
    }
    print(f'  {k} views per set; medians over {runs} sets: max over the image, predicted / '
          f'true, px')
    for name, mask in sets.items():
        candidates = np.flatnonzero(mask)
        if len(candidates) < k:
            print(f'    {name}: only {len(candidates)} views, skipped')
            continue
        rows = [summarize(camera, [views[i] for i in rng.choice(candidates, k, replace=False)],
                          size, True, pixels) for _ in range(runs)]
        groups = '  '.join(f'{g} {np.median([r["groups"][g] for r in rows]):.2f}'
                           for g in GROUPS)
        print(f'    {name:45s} ({len(candidates):2d} views): all '
              f'{np.median([r["predicted"].max() for r in rows]):6.2f} / '
              f'{np.median([r["actual"].max() for r in rows]):6.2f}   {groups}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('dataset', nargs='?', default='dataset/synthetic')
    parser.add_argument('--runs', type=int, default=20, help='random view orders per step')
    parser.add_argument('--fix-k3-below', type=int, default=0,
                        help='fix k3 while fewer views than this are used')
    parser.add_argument('--grid-step', type=int, default=40, help='uncertainty map grid, px')
    args = parser.parse_args()
    dataset = Path(args.dataset)
    truth = json.loads((dataset / 'ground_truth.json').read_text())
    cam = truth['camera']
    if cam['model'] != 'pinhole':
        sys.exit('pinhole datasets only')
    camera = Camera(cam['model'], cam['width'], cam['height'], np.array(cam['K']),
                    np.array(cam['D']))
    size = (camera.width, camera.height)
    pixels = grid_pixels(camera.width, camera.height, args.grid_step)
    views = detect_views(dataset, truth, dataset / 'uncertainty_prototype_cache.npz')
    rng = np.random.default_rng(0)

    for mode in ('chessboard', 'oversized_marker'):
        selected = [v for v in views if v[1] == mode]
        pairs = [(v[2], v[3]) for v in selected]
        print(f'== {mode}: {len(selected)} views pass the checks')
        check_against_opencv(pairs, size)
        tilt_experiment(selected, summarize(camera, pairs, size, False, pixels))
        print('  growing random view sets (prediction: sigma from the residuals):')
        growth_experiment(camera, pairs, size, args.runs, (4, 5, 6, 8, 10, 15, 20, 30, 50, 90),
                          args.fix_k3_below, pixels, rng)
        print('  degenerate view sets:')
        degenerate_experiment(camera, selected, size, 6, args.runs, pixels, rng)
        print()


if __name__ == '__main__':
    main()
