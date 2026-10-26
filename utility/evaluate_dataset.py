#!/usr/bin/env python3
"""
Evaluate smart_calibration's detection and calibration on a synthetic dataset.

Runs the package's own preprocessing and board detection (the source in this repository, not an
installed copy) on every image of a dataset made by synthetic_dataset.py and compares with the
ground truth: detection rate (per board type and tag), corner error, origin of marker boards,
false detections on images without a board. Then calibrates from all detections of each board
type and compares with the true camera: how far the estimated lens model maps pixels from the
true one across the image (comparing distortion coefficients directly is misleading: they trade
off against each other).

Usage (venv active, from the repository root):
    python utility/evaluate_dataset.py dataset/synthetic [--exhaustive] [--accuracy]
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

REPOSITORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY / 'smart_calibration'))  # the package source, not install/

from smart_calibration.detection import detect_board, sb_flags  # noqa: E402
from smart_calibration.preprocessing import preprocess  # noqa: E402
from smart_calibration.views import (  # noqa: E402
    corner_residuals, image_points, object_points)
from synthetic_dataset import Camera, pixel_rays, project  # noqa: E402

BOARD_MODES = ('chessboard', 'oversized_marker')


def true_positions(board, entry):
    """
    True pixel position (N, 2) of each detected corner (row-major), or None if they do not match.

    chessboard: the detected grid against the true grid, in both numbering directions (the
    detector may start from either end). marker board: each detected corner's board coordinates
    looked up in the ground truth; a missing coordinate means a wrong origin.
    """
    truth = np.array(entry['corners']).reshape(*entry['grid'], 2)
    corners = board.corners.astype(np.float64)
    if entry['board'] == 'chessboard':
        if corners.shape != truth.shape:
            return None
        flipped = truth[::-1, ::-1]
        better = min((truth, flipped), key=lambda t: np.linalg.norm(corners - t, axis=-1).mean())
        return better.reshape(-1, 2)
    lookup = {tuple(c): p for c, p in zip(map(tuple, entry['board_coordinates']),
                                          truth.reshape(-1, 2))}
    coordinates = [tuple(c) for c in board.board_coordinates().reshape(-1, 2).tolist()]
    if not all(c in lookup for c in coordinates):
        return None
    return np.array([lookup[c] for c in coordinates])


def corner_check(board, max_residual):
    """
    The calibration node's corner check of a detection: (largest residual or None, flagged mask).

    flagged: (rows, cols) corners whose residual exceeds max_residual (see
    smart_calibration.views.corner_residuals); any flagged corner discards the frame.
    """
    residuals = corner_residuals(board)
    largest = float(np.nanmax(residuals)) if np.isfinite(residuals).any() else None
    with np.errstate(invalid='ignore'):
        return largest, residuals > max_residual


def compare(board, entry):
    """Corner errors (N,) of a detection against the ground truth, or None if they do not match."""
    expected = true_positions(board, entry)
    if expected is None:
        return None
    return np.linalg.norm(board.corners.reshape(-1, 2) - expected, axis=-1)


ERROR_COLORS = ((0.5, (0, 200, 0)), (1.0, (0, 220, 255)), (float('inf'), (0, 0, 255)))  # BGR
TRUTH_COLOR = (255, 200, 0)
OUTSIDE_COLOR = (90, 0, 90)
FLAGGED_COLOR = (0, 140, 255)  # orange
DROPPED_COLOR = (170, 170, 170)
INSET_RADIUS, INSET_ZOOM = 24, 6
SHIFT = 4  # sub-pixel drawing: coordinates in 1/16 px


def _point(xy, zoom=1.0, offset=(0.0, 0.0)):
    """Pixel position -> cv2 drawing coordinates with SHIFT bits of sub-pixel precision."""
    return tuple(int(round(((c - o) * zoom + (zoom - 1) / 2) * (1 << SHIFT)))
                 for c, o in zip(xy, offset))


def _draw_marks(canvas, truth, detected, errors, zoom=1.0, offset=(0.0, 0.0), flagged=None,
                dropped=None):
    """
    Ground truth as crosses, detections as circles coloured by error, lines for errors >= 1.

    flagged: per detection, True for an orange ring (corner check failed). dropped: per
    detection, True for a grey x instead of a circle (not used: border rule).
    """
    size = 4 * max(zoom / 2, 1.0)
    if detected is not None and flagged is not None:
        for (x, y) in detected[flagged]:
            cv2.circle(canvas, _point((x, y), zoom, offset), int(2 * size * (1 << SHIFT)),
                       FLAGGED_COLOR, 2, cv2.LINE_AA, SHIFT)
    for x, y in truth:
        for dx, dy in ((size, 0), (0, size)):
            cv2.line(canvas, _point((x - dx / zoom, y - dy / zoom), zoom, offset),
                     _point((x + dx / zoom, y + dy / zoom), zoom, offset), TRUTH_COLOR, 1,
                     cv2.LINE_AA, SHIFT)
    if detected is None:
        return
    if dropped is None:
        dropped = np.zeros(len(detected), bool)
    for (x, y), error, true_xy, gone in zip(detected, errors, truth, dropped):
        if gone:
            for dx, dy in ((size, size), (size, -size)):
                cv2.line(canvas, _point((x - dx / zoom, y - dy / zoom), zoom, offset),
                         _point((x + dx / zoom, y + dy / zoom), zoom, offset), DROPPED_COLOR, 2,
                         cv2.LINE_AA, SHIFT)
            continue
        color = next(c for limit, c in ERROR_COLORS if error < limit)
        cv2.circle(canvas, _point((x, y), zoom, offset), int(size * (1 << SHIFT)), color, 1,
                   cv2.LINE_AA, SHIFT)
        if error >= 1.0:
            cv2.line(canvas, _point((x, y), zoom, offset), _point(true_xy, zoom, offset), color,
                     1, cv2.LINE_AA, SHIFT)


def visualize(gray, entry, board, expected, path, max_residual):
    """
    Save the image with true and detected corners, and a zoomed inset of the worst corner.

    The inset shows the white-framed square around the worst corner, in the image corner
    opposite to it; purple in the inset is outside the image.

    Truth: cyan crosses (all corners of the board inside the image). Detections: circles, green
    < 0.5 px error, yellow < 1 px, red >= 1 px with a line to the true position; magenta if the
    detection does not match the board (wrong grid or origin); grey x for corners the border rule
    drops. Orange rings: corners whose residual exceeds max_residual; the label says whether the
    corner check discards the frame.
    """
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    truth_all = np.array(entry['corners'])[np.array(entry['visible'])]
    detected = None if board is None else board.corners.reshape(-1, 2).astype(np.float64)
    errors = None
    if detected is not None and expected is not None:
        errors = np.linalg.norm(detected - expected, axis=1)
    _draw_marks(canvas, truth_all, None, None)
    if errors is not None:
        residual, flagged = corner_check(board, max_residual)
        flagged, dropped = flagged.ravel(), ~board.valid.ravel()
        verdict = ('frame DISCARDED' if flagged.any() else 'frame kept') + (
            f', {int(dropped.sum())} corners dropped at the border' if dropped.any() else '')
        _draw_marks(canvas, expected, detected, errors, flagged=flagged, dropped=dropped)
        worst = int(np.argmax(errors))
        cx, cy = detected[worst]
        # Padded with purple ("outside the image"): the detector can place corners beyond it.
        r = INSET_RADIUS
        padded = cv2.copyMakeBorder(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), r, r, r, r,
                                    cv2.BORDER_CONSTANT, value=OUTSIDE_COLOR)
        x0 = int(np.clip(round(cx) - r, -r, gray.shape[1] - r))
        y0 = int(np.clip(round(cy) - r, -r, gray.shape[0] - r))
        crop = padded[y0 + r:y0 + 3 * r, x0 + r:x0 + 3 * r]
        inset = cv2.resize(crop, None, fx=INSET_ZOOM, fy=INSET_ZOOM,
                           interpolation=cv2.INTER_NEAREST)
        near = np.all(np.abs(expected - [cx, cy]) < 2 * INSET_RADIUS, axis=1)
        _draw_marks(inset, expected[near], detected[near], errors[near], INSET_ZOOM, (x0, y0),
                    flagged[near], dropped[near])
        cv2.rectangle(inset, (0, 0), (inset.shape[1] - 1, inset.shape[0] - 1), (255, 255, 255))
        # In the image corner opposite the worst corner, so it does not cover it (bottom: above
        # the label).
        height, width = gray.shape
        ix = width - 10 - inset.shape[1] if cx < width / 2 else 10
        iy = height - 34 - inset.shape[0] if cy < height / 2 else 10
        canvas[iy:iy + inset.shape[0], ix:ix + inset.shape[1]] = inset
        cv2.rectangle(canvas, (x0, y0), (x0 + 2 * INSET_RADIUS, y0 + 2 * INSET_RADIUS),
                      (255, 255, 255), 1)
        residual_text = '-' if residual is None else f'{residual:.3f}'
        label = (f'worst corner {errors[worst]:.2f} px (inset: x{INSET_ZOOM}), residual '
                 f'{residual_text}: {verdict}')
    elif detected is not None:
        for x, y in detected:
            cv2.circle(canvas, _point((x, y)), 4 << SHIFT, (255, 0, 255), 1, cv2.LINE_AA, SHIFT)
        label = 'WRONG grid / origin'
    else:
        label = 'not detected'
    cv2.putText(canvas, f'{entry["file"]}: {label} {" ".join(entry["tags"])}',
                (10, gray.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def mapping_error(camera, K, D, step=16):
    """
    Pixel distance between the true and an estimated lens model over the image.

    The true rays through a grid of pixels are projected with the estimated model. Returns
    (max, mean) as is, and (max, mean, angle in degrees) after the best rotation between the true
    and the estimated camera frame: a principal point error looks much like a small rotation of
    the camera, so this separates it from errors in the lens model's shape.
    """
    us, vs = np.meshgrid(np.arange(0, camera.width, step), np.arange(0, camera.height, step))
    pixels = np.column_stack([us.ravel(), vs.ravel()]).astype(np.float64)
    estimated = Camera(camera.model, camera.width, camera.height, K, D)
    true_rays, estimated_rays = pixel_rays(camera, pixels), pixel_rays(estimated, pixels)
    valid = np.isfinite(true_rays[:, 0]) & np.isfinite(estimated_rays[:, 0])
    pixels = pixels[valid]
    true_rays = np.column_stack([true_rays[valid], np.ones(valid.sum())])
    estimated_rays = np.column_stack([estimated_rays[valid], np.ones(valid.sum())])
    raw = np.linalg.norm(project(estimated, true_rays) - pixels, axis=1)

    # Rotation that best maps the true rays onto the estimated ones (Kabsch).
    a = true_rays / np.linalg.norm(true_rays, axis=1)[:, None]
    b = estimated_rays / np.linalg.norm(estimated_rays, axis=1)[:, None]
    U, _, Vt = np.linalg.svd(b.T @ a)
    rotation = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    rotated = np.linalg.norm(project(estimated, true_rays @ rotation.T) - pixels, axis=1)
    angle = np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0)))
    return (raw.max(), raw.mean()), (rotated.max(), rotated.mean(), angle)


def calibrate(camera, views):
    """Calibrate the camera's model from [(object_points, image_points)]; return (rms, K, D)."""
    objects = [o for o, _ in views]
    images = [i.astype(np.float32) for _, i in views]
    size = (camera.width, camera.height)
    if camera.model == 'pinhole':
        rms, K, D, _, _ = cv2.calibrateCamera(objects, images, size, None, None)
    else:
        rms, K, D, _, _ = cv2.fisheye.calibrate(
            [o.astype(np.float64) for o in objects], [i.astype(np.float64) for i in images],
            size, None, None,
            flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW)
    return rms, K, np.asarray(D).ravel()


def evaluate(dataset, args):
    dataset = Path(dataset)
    truth = json.loads((dataset / 'ground_truth.json').read_text())
    cam = truth['camera']
    camera = Camera(cam['model'], cam['width'], cam['height'], np.array(cam['K']),
                    np.array(cam['D']))
    chessboard = truth['boards']['chessboard']
    cols, rows = chessboard['squares_x'] - 1, chessboard['squares_y'] - 1
    flags = sb_flags(args.normalize, args.exhaustive, args.accuracy)
    print(f'{len(truth["images"])} images, {camera.model} camera; flags: normalize '
          f'{args.normalize}, exhaustive {args.exhaustive}, accuracy {args.accuracy}; '
          f'gamma {args.gamma}\n')

    stats = defaultdict(lambda: defaultdict(list))
    views = defaultdict(list)
    for entry in truth['images']:
        image = cv2.imread(str(dataset / entry['file']), cv2.IMREAD_UNCHANGED)
        gray = preprocess(image, 0.5, 99.5, args.gamma)
        kind = entry['board']
        for mode in ([kind] if kind else BOARD_MODES):
            start = time.perf_counter()
            board = detect_board(gray, mode, cols, rows, flags, args.border_margin)
            stats[mode]['time_ms'].append((time.perf_counter() - start) * 1000)
            if kind is None:
                stats[mode]['false'].append(board is not None)
                continue
            truth_grid = np.array(entry['corners']).reshape(*entry['grid'], 2)
            stats[mode]['square_px'].append(float(np.median(
                np.linalg.norm(np.diff(truth_grid, axis=1), axis=-1))))
            found = board is not None
            stats[mode]['found'].append(found)
            for tag in entry['tags'] or ['plain']:
                stats[mode][f'tag {tag}'].append(found)
            expected = true_positions(board, entry) if found else None
            if args.visualize:
                if expected is not None:
                    errors = np.linalg.norm(board.corners.reshape(-1, 2) - expected, axis=1)
                    prefix = f'{errors.max():07.3f}px'
                else:
                    prefix = 'wrong' if found else 'not_detected'
                visualize(gray, entry, board, expected,
                          Path(args.visualize) / f'{prefix}_{Path(entry["file"]).name}',
                          args.max_residual)
            if not found:
                continue
            errors = compare(board, entry)
            if errors is None:
                stats[mode]['mismatch'].append(entry['file'])
                continue
            stats[mode]['rms'].append(float(np.sqrt(np.mean(errors ** 2))))
            stats[mode]['max'].append(float(errors.max()))
            # What the border rule and the corner check do, as in the calibration node: corners
            # > 1 px off that are dropped, in a discarded frame, or still used.
            valid = board.valid.ravel()
            _, flagged = corner_check(board, args.max_residual)
            discarded = bool(flagged.any())
            bad = errors > 1.0
            stats[mode]['dropped'].append(int((~valid).sum()))
            stats[mode]['discarded'].append(discarded)
            stats[mode]['bad'].append(int(bad.sum()))
            stats[mode]['bad dropped'].append(int((bad & ~valid).sum()))
            stats[mode]['bad discarded'].append(int((bad & valid).sum()) if discarded else 0)
            stats[mode]['bad used'].append(0 if discarded else int((bad & valid).sum()))
            if discarded:
                stats[mode]['discarded files'].append(Path(entry['file']).stem)
                continue
            square = truth['boards'][mode]['square_size']
            views[mode].append((object_points(board, square), image_points(board)))

    for mode in BOARD_MODES:
        s = stats[mode]
        print(f'== {mode}')
        if s['square_px']:
            low, median, high = np.percentile(s['square_px'], [5, 50, 95])
            print(f'  square size in the image: median {median:.0f} px (5-95 %: {low:.0f}-'
                  f'{high:.0f} px)')
        if s['found']:
            print(f'  detected {sum(s["found"])} / {len(s["found"])}')
            for key in sorted(k for k in s if k.startswith('tag ')):
                print(f'    {key[4:]:12s} {sum(s[key])} / {len(s[key])}')
        if s['rms']:
            print(f'  corner error: rms mean {np.mean(s["rms"]):.3f} px, median '
                  f'{np.median(s["rms"]):.3f} px; worst corner {np.max(s["max"]):.3f} px')
        if s['discarded']:
            print(f'  border rule (< {args.border_margin} px): {sum(s["dropped"])} corners '
                  f'dropped; corner check (> {args.max_residual}): {sum(s["discarded"])} frames '
                  f'discarded ({", ".join(s["discarded files"]) or "-"})')
            print(f'  corners > 1 px off: {sum(s["bad"])}; dropped {sum(s["bad dropped"])}, in '
                  f'discarded frames {sum(s["bad discarded"])}, still used '
                  f'{sum(s["bad used"])}')
        if s['mismatch']:
            print(f'  WRONG grid / origin: {", ".join(s["mismatch"])}')
        if s['false']:
            print(f'  false detections on images without a board: {sum(s["false"])} / '
                  f'{len(s["false"])}')
        print(f'  detection time: mean {np.mean(s["time_ms"]):.0f} ms, max '
              f'{np.max(s["time_ms"]):.0f} ms')
        if len(views[mode]) >= 5:
            rms, K, D = calibrate(camera, views[mode])
            (worst, mean), (shape_worst, shape_mean, angle) = mapping_error(camera, K, D)
            print(f'  calibration from the {len(views[mode])} frames that pass (valid corners): '
                  f'reprojection rms {rms:.3f} px; '
                  f'fx {K[0, 0]:.2f} fy {K[1, 1]:.2f} cx {K[0, 2]:.2f} cy {K[1, 2]:.2f} (true '
                  f'{camera.K[0, 0]:.2f}, {camera.K[1, 1]:.2f}, {camera.K[0, 2]:.2f}, '
                  f'{camera.K[1, 2]:.2f})')
            print(f'    lens model vs truth over the image: max {worst:.3f} px, mean {mean:.3f} '
                  f'px; allowing a {angle:.3f} deg camera rotation (principal point error): '
                  f'max {shape_worst:.3f} px, mean {shape_mean:.3f} px')
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('dataset', nargs='?', default='dataset/synthetic')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--exhaustive', action='store_true')
    parser.add_argument('--accuracy', action='store_true')
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--max-residual', type=float, default=0.1,
                        help='corner check threshold in squares (node parameter '
                             'max_corner_residual)')
    parser.add_argument('--border-margin', type=float, default=10.0,
                        help='border rule margin in px (node parameter border_margin_px)')
    parser.add_argument('--visualize', metavar='DIR',
                        help='save every board image with true and detected corners to DIR, '
                             'named by its worst corner error')
    args = parser.parse_args()
    if args.visualize:
        Path(args.visualize).mkdir(parents=True, exist_ok=True)
    evaluate(args.dataset, args)


if __name__ == '__main__':
    main()
