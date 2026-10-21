# Exploration: building a custom mono camera calibration package (ROS 2 Humble)

Scope: monocular cameras, two models
- **pinhole** → ROS `distortion_model: plumb_bob` (or `rational_polynomial`)
- **fisheye** → ROS `distortion_model: equidistant` (OpenCV `cv::fisheye`, Kannala–Brandt)

Sources studied (all read locally, so they match exactly what is installed on this machine):
- ROS 2 `camera_calibration` **3.0.9** (Humble binary): `/opt/ros/humble/lib/python3.10/site-packages/camera_calibration/`
  - `calibrator.py` (core algorithms), `camera_calibrator.py` (node + GUI), `nodes/cameracalibrator.py` (CLI/flags), `camera_checker.py`
- OpenCV **4.5.4** (Ubuntu 22.04 / Humble system OpenCV): API docs taken from the doc-comments in
  `/usr/include/opencv4/opencv2/calib3d.hpp` (docs.opencv.org is generated from these; the website blocked automated fetching).
- `image_geometry` 3.2.1 (C++ lib + Python `cameramodels.py`) to see how consumers read the result.

---

## 1. How ROS 2 `camera_calibration` works (mono path)

### 1.1 Architecture

```
cameracalibrator (CLI, optparse)  ──► builds cv flags, boards
        │
OpenCVCalibrationNode (rclpy Node + cv2.imshow GUI)
  ├─ sub 'image' (sensor_msgs/Image), QoS copied from publisher
  ├─ BufferQueue(queue_size=1)  → drops old frames, consumer thread
  ├─ client 'camera/set_camera_info' (COMMIT button)
  └─ MonoCalibrator (calibrator.py)  ← all the actual math
        handle_msg → detect → score view → maybe add to db
        do_calibration → cv2.calibrateCamera / cv2.fisheye.calibrate
        set_alpha → P matrix + undistort maps
        yaml()/ost() → output, do_save → /tmp/calibrationdata.tar.gz
```

The math is in a ROS-agnostic class (`MonoCalibrator`), which is a good pattern to copy: it can be
tested offline from image folders (`cal(images)`, `do_tarfile_calibration`).

### 1.2 Per-frame pipeline (`MonoCalibrator.handle_msg`)

1. **`mkgray`**: convert to mono8. 16-bit images are divided by 256 (loses dynamic range for
   dark 12-bit sensors); float images scaled by max.
2. **`downsample_and_detect`**: image is resized to ~VGA (`scale = sqrt(w*h / 640*480)`).
   - Chessboard: detected on the *downsampled* image, then corners scaled up and refined with
     `cornerSubPix` on the full image using window radius `ceil(scale)` (i.e. tiny, 2–3 px).
   - Circles: detected on the full image.
3. **Detection** (`_get_corners`):
   - `cv2.findChessboardCorners(..., ADAPTIVE_THRESH | NORMALIZE_IMAGE | FAST_CHECK)` (classic detector, **not** `findChessboardCornersSB`).
   - Rejects the whole view if any corner is within **8 px** (`BORDER = 8`) of the image edge →
     the image corners, where distortion is strongest, are systematically under-sampled. Bad for fisheye.
   - Re-orders corners so the board always goes top→bottom (avoids 180° flips between views).
   - `cornerSubPix` with radius = half the minimum inter-corner distance, criteria
     `(EPS|MAX_ITER, 30, 0.1)` → **epsilon 0.1 px is quite coarse**.
   - ⚠️ Bug: the min-distance loop indexes `row*board.n_rows + col` instead of `row*board.n_cols + col`
     (only correct for square boards).
   - ChArUco: `CharucoDetector`/`interpolateCornersCharuco`, needs > 5 corners. Circles: `findCirclesGrid` (sym/asym).
4. **View descriptor** (`get_parameters`) → 4 numbers in [0,1]:
   - `X`, `Y`: mean corner position, with the image "shrunk" by half the board size
   - `Size`: `sqrt(board_area / image_area)` (area of outer quadrilateral)
   - `Skew`: `min(1, 2*|90° − angle(up_left, up_right, down_right)|)` — one corner angle only, so
     it does not distinguish tilt axis/direction.
5. **Sample acceptance** (`is_good_sample`): accept if the L1 distance in (X,Y,Size,Skew) space to
   *every* stored sample is > **0.2** (hard-coded), and optionally if mean corner motion
   ≤ `--max-chessboard-speed` px/frame (motion blur / rolling shutter guard).
6. **"Good enough"** (`compute_goodenough`): progress per param = `(max−min)/range` with ranges
   `[0.7, 0.7, 0.4, 0.5]` (min of Size/Skew forced to 0). CALIBRATE enabled when all bars are full
   **or ≥ 40 samples**. Full gray images are stored for every sample.

### 1.3 Calibration (`cal_fromcorners`)

- Object points from `mk_object_points(boards)` with `use_board_size=False` for mono → board in
  **square units**, not meters. Intrinsics/distortion are unaffected, but rvecs/tvecs are not metric.
- Initial K = identity, no `CALIB_USE_INTRINSIC_GUESS`.
- **Pinhole**: `cv2.calibrateCamera(opts, ipts, size, K0, None, flags=calib_flags)` with default
  criteria (30 iters, DBL_EPSILON). D truncated to 5 (plumb_bob) or 8 (rational_polynomial).
- **Fisheye**: `cv2.fisheye.calibrate(opts64, ipts64, size, K0, None, flags=fisheye_flags)`
  (points must be float64), default criteria (100 iters, DBL_EPSILON).
- **ChArUco**: `cv2.aruco.calibrateCameraCharuco(...)` — ⚠️ **`calib_flags` are not passed**
  (so `-k`, `--fix-aspect-ratio`, etc. are silently ignored), and **fisheye + ChArUco raises
  NotImplemented**.
- The returned RMS reprojection error is **discarded** — never printed or shown. No per-view
  errors, no std-devs, no outlier rejection, no iteration.
- R = I, then `set_alpha(0.0)`:
  - Pinhole: `P[:3,:3] = getOptimalNewCameraMatrix(K, D, size, alpha)`.
  - Fisheye: naive `P = K` with `fx, fy /= (1+alpha)` (comment says
    `estimateNewCameraMatrixForUndistortRectify` "not producing proper results").
  - Builds remap maps for display.

### 1.4 CLI → OpenCV flag mapping (`nodes/cameracalibrator.py`)

| CLI option | Pinhole flag |
|---|---|
| `--fix-principal-point` | `CALIB_FIX_PRINCIPAL_POINT` |
| `--fix-aspect-ratio` | `CALIB_FIX_ASPECT_RATIO` (with K0=I → forces fx = fy) |
| `--zero-tangent-dist` | `CALIB_ZERO_TANGENT_DIST` |
| `-k N` (default 2!) | `N>3` → `CALIB_RATIONAL_MODEL`; `CALIB_FIX_K{N+1..6}` |

Note the default `-k 2` means **k3 is fixed to 0** by default (plumb_bob with only k1, k2, p1, p2).

| CLI option | Fisheye flag |
|---|---|
| `--fisheye-fix-principal-point` | `fisheye.CALIB_FIX_PRINCIPAL_POINT` |
| `--fisheye-fix-skew` | `fisheye.CALIB_FIX_SKEW` |
| `--fisheye-recompute-extrinsicsts` (sic) | `fisheye.CALIB_RECOMPUTE_EXTRINSIC` |
| `--fisheye-check-conditions` | `fisheye.CALIB_CHECK_COND` |
| `--fisheye-k-coefficients N` (default 4) | `fisheye.CALIB_FIX_K{N+1..4}` |

Other: `--disable_calib_cb_fast_check`, `--max-chessboard-speed`, `--queue-size`,
`--no-service-check`, multiple boards via repeated `--size/--square`. Camera model is chosen by a
**GUI trackbar**, not a CLI flag. Default board if none given: 8x6, 0.108 m.

### 1.5 Quality metric shown after calibration

- **Linear error** (`linear_error`): undistort the detected corners, fit a line between the
  first/last corner of each row, RMS of point-to-line distance (px). Measures straightness only
  (not scale/focal correctness), only on the live frame.
- `cameracheck` node: linearity RMS + a reprojection RMS via `solvePnP` on rectified corners.

### 1.6 Output

- `ost.yaml` (camera_info_manager / `camera_calibration_parsers` format): `image_width`,
  `image_height`, `camera_name`, `camera_matrix`, `distortion_model`, `distortion_coefficients`,
  `rectification_matrix` (I for mono), `projection_matrix`.
- `ost.txt` (legacy INI), plus all sample images as PNG, in `/tmp/calibrationdata.tar.gz`.
- COMMIT → `sensor_msgs/srv/SetCameraInfo` on `camera/set_camera_info`.
- Distortion model string: pinhole → `plumb_bob` (D size 5) / `rational_polynomial` (8);
  fisheye → `equidistant` (D size 4).

### 1.7 Weak spots worth improving in our own package

1. RMS error never reported; no per-view error / outlier rejection / re-calibration loop.
2. No parameter uncertainty (`calibrateCameraExtended` std-devs are available but unused).
3. 8 px border rejection + full-board-visibility requirement → poor edge/corner coverage.
4. Detection on downsampled image, small refine window, coarse `cornerSubPix` epsilon (0.1).
5. Classic `findChessboardCorners` instead of the more accurate `findChessboardCornersSB`.
6. Coverage metric is crude (4 scalars, L1 threshold 0.2, single-angle skew); no spatial
   coverage map of where corners have landed in the image.
7. No blur/sharpness check (only optional speed check).
8. Fisheye: no ChArUco support, naive P, no recompute-extrinsic by default, no handling of
   ill-conditioned views.
9. ChArUco ignores calibration flags.
10. Calibration can't be run offline from a bag / folder via ROS interface easily; parameters via
    optparse, not ROS parameters; output path hard-coded to `/tmp`.

---

## 2. OpenCV 4.5.4 calibration API (what we can build on)

### 2.1 Pinhole model (`cv::calibrateCamera`)

Projection: `x' = Xc/Zc, y' = Yc/Zc, r² = x'² + y'²`

```
x'' = x' (1 + k1 r² + k2 r⁴ + k3 r⁶) / (1 + k4 r² + k5 r⁴ + k6 r⁶) + 2 p1 x'y' + p2 (r² + 2x'²) + s1 r² + s2 r⁴
y'' = y' (1 + k1 r² + k2 r⁴ + k3 r⁶) / (1 + k4 r² + k5 r⁴ + k6 r⁶) + p1 (r² + 2y'²) + 2 p2 x'y' + s3 r² + s4 r⁴
u = fx x'' + cx,  v = fy y'' + cy        (optional tilt τx, τy applied before K)
```

D vector length: 4 / **5 (k1 k2 p1 p2 k3 = plumb_bob)** / **8 (+k4 k5 k6 = rational_polynomial)** / 12 (+s1..s4) / 14 (+τx τy).
Only 5 and 8 map to standard ROS models.

> OpenCV docs note: radial distortion must be **monotonic** for real lenses; a non-monotonic
> result is a calibration failure even if it looks fine near the center. The optimizer does
> **not** enforce this → we should check it ourselves (sample r over the image's max radius).

Functions:
- `calibrateCamera(obj, img, size, K, D, flags, criteria)` → RMS, K, D, rvecs, tvecs.
  Default criteria `COUNT+EPS, 30, DBL_EPSILON`.
- `calibrateCameraExtended(...)` (Python name) → additionally **`stdDeviationsIntrinsics`**
  (order fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6, s1..s4, τx, τy), `stdDeviationsExtrinsics`,
  **`perViewErrors`**. Available in 4.5.4 ✅.
- `calibrateCameraRO(..., iFixedPoint, ...)` / `calibrateCameraROExtended` — Strobl & Hirzinger
  **object-releasing** method: also refines the 3D board points (handles imperfectly printed /
  non-flat boards). Requires the same, fully visible board in every view; fix e.g. the top-right
  corner (`iFixedPoint = cols-1`). Returns `newObjPoints`. Slower. Pinhole only.
- Algorithm: Zhang homography init (planar target) → per-view `solvePnP` → global
  Levenberg–Marquardt on total squared reprojection error.

Flags (values from header):

| Flag | Meaning |
|---|---|
| `CALIB_USE_INTRINSIC_GUESS` | start from provided K (needed for 3D rigs; useful for re-runs) |
| `CALIB_FIX_PRINCIPAL_POINT` | keep cx, cy (image center or guess) |
| `CALIB_FIX_ASPECT_RATIO` | keep fx/fy ratio from input K |
| `CALIB_FIX_FOCAL_LENGTH` | keep fx, fy (needs USE_INTRINSIC_GUESS) |
| `CALIB_ZERO_TANGENT_DIST` / `CALIB_FIX_TANGENT_DIST` | p1 = p2 = 0 / keep p1, p2 |
| `CALIB_FIX_K1..K6` | keep coefficient (0 unless guess) |
| `CALIB_RATIONAL_MODEL` | enable k4..k6 (must be explicit) |
| `CALIB_THIN_PRISM_MODEL`, `CALIB_FIX_S1_S2_S3_S4` | s1..s4 |
| `CALIB_TILTED_MODEL`, `CALIB_FIX_TAUX_TAUY` | τx, τy |
| `CALIB_USE_QR`, `CALIB_USE_LU` | faster, less precise solver |

Docs gotcha: for non-square boards pass `patternSize = (cols, rows)`; swapping gives garbage
(cx/cy far off, fx:fy ≈ 10:1, zero distortion).

### 2.2 Fisheye / equidistant model (`cv::fisheye`)

```
a = x/z, b = y/z, r² = a² + b², θ = atan(r)
θd = θ (1 + k1 θ² + k2 θ⁴ + k3 θ⁶ + k4 θ⁸)
x' = (θd / r) a,  y' = (θd / r) b
u = fx (x' + α y') + cx,  v = fy y' + cy          (α = skew)
```

D = (k1, k2, k3, k4) → ROS `equidistant`. Because θ = atan(x/z) it only handles points in front of
the camera (FOV < 180°). Polynomial is in θ, so distortion is well behaved at large angles where the
pinhole r-polynomial blows up.

`fisheye.calibrate(obj, img, size, K, D, rvecs, tvecs, flags, criteria)` → RMS.
Default criteria `COUNT+EPS, 100, DBL_EPSILON`. **No per-view errors or std-devs** → compute
per-view errors ourselves with `fisheye.projectPoints(obj, rvec, tvec, K, D)`.
Input requirement (in practice / per ROS comment): object points `(N,1,3)` float64, image points
`(N,1,2)` float64, one array per view.

Fisheye flags:

| Flag | Meaning |
|---|---|
| `CALIB_USE_INTRINSIC_GUESS` | start from provided K/D |
| `CALIB_RECOMPUTE_EXTRINSIC` | re-solve poses after each intrinsic iteration (**generally recommended**) |
| `CALIB_CHECK_COND` | assert on ill-conditioned per-view Jacobian (raises `cv2.error`) — useful to detect bad views; typical strategy: catch, drop the offending view, retry |
| `CALIB_FIX_SKEW` | α = 0 (**generally recommended**; skew is ~0 on modern sensors) |
| `CALIB_FIX_K1..K4` | keep coefficient at 0 / guess |
| `CALIB_FIX_PRINCIPAL_POINT` | keep cx, cy |
| `CALIB_FIX_FOCAL_LENGTH` | keep f (init `max(w,h)/π` or guess) |

Other fisheye functions:
- `fisheye.projectPoints`, `fisheye.distortPoints` (input assumed with K = I), `fisheye.undistortPoints(pts, K, D, R, P)`.
- `fisheye.initUndistortRectifyMap(K, D, R, P, size, m1type)` + `remap`; `fisheye.undistortImage`.
- `fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, size, R, balance∈[0,1], new_size, fov_scale)`:
  `balance` interpolates new focal between min and max; `fov_scale` divides the focal. ROS
  avoided it ("not proper results") — worth evaluating ourselves; at minimum expose
  `balance`/`fov_scale` and fall back to the naive `P = K / (1+alpha)` approach.

### 2.3 Target detection

- `findChessboardCorners(img, (cols,rows), flags)` — classic quad-based detector.
  Flags: `ADAPTIVE_THRESH`, `NORMALIZE_IMAGE`, `FILTER_QUADS`, `FAST_CHECK`. Needs `cornerSubPix` after.
- **`findChessboardCornersSB(img, (cols,rows), flags)`** (available in 4.5.4 ✅) — sector-based /
  localized Radon transform (Duda 2018). More robust to noise, faster on large images, and returns
  sub-pixel corners **more accurate than `cornerSubPix`**. Flags: `NORMALIZE_IMAGE`, `EXHAUSTIVE`,
  `ACCURACY` (upsamples to reduce aliasing), `LARGER` (board may be larger than the FOV/pattern
  size), `MARKER` (board with marker dots defines a consistent origin).
  `findChessboardCornersSBWithMeta` returns per-corner metadata for `LARGER`/`MARKER` use.
  Docs: board needs a white border ~1 square wide; rounded outer corners help.
- `estimateChessboardSharpness(img, size, corners)` → (avg edge transition width, min, max brightness).
  Docs: sharpness should be **< ~3 px**; good for rejecting blurry/under-exposed views.
- `findCirclesGrid(img, size, flags)` — `SYMMETRIC_GRID`, `ASYMMETRIC_GRID`, `CLUSTERING`
  (robust to perspective, sensitive to clutter). Needs white margin around the grid.
- ChArUco (`cv2.aruco`, contrib, present): in 4.5.4 use the old API —
  `aruco.detectMarkers` → `aruco.interpolateCornersCharuco` → `aruco.calibrateCameraCharuco(Extended)`.
  (The `CharucoDetector` class is ≥ 4.7/4.8.) Partial views are allowed, which helps edge coverage.
  For **fisheye + ChArUco** we can build object points ourselves from `board.chessboardCorners[ids]`
  and call `fisheye.calibrate` — note that fisheye undistortion of marker detections near the edge
  can be unreliable, so corner interpolation quality should be checked.
- `cornerSubPix(img, corners, win, zeroZone, criteria)`; tighter criteria (e.g. eps 1e-3, 30–100 iters)
  than ROS' 0.1.

### 2.4 Post-calibration helpers (pinhole)

- `getOptimalNewCameraMatrix(K, D, size, alpha, newSize, centerPrincipalPoint)` → new K and
  `validPixROI`. `alpha=0` all pixels valid, `alpha=1` keep all source pixels.
- `initUndistortRectifyMap` + `remap`, `undistortPoints` (5 iterations default; `undistortPointsIter` with criteria).
- `projectPoints` (with Jacobian), `solvePnP` (`SOLVEPNP_IPPE` is ideal for planar boards),
  `solvePnPRefineLM`, `drawFrameAxes`, `calibrationMatrixValues` (FOV, focal mm from sensor size).

---

## 3. ROS-side compatibility notes

- `sensor_msgs/distortion_models.hpp` defines `plumb_bob`, `rational_polynomial`, `equidistant`.
- `image_geometry` **C++** 3.2.1 (used by `image_proc` rectify) supports all three
  ("Supported models: PLUMB_BOB, RATIONAL_POLYNOMIAL and EQUIDISTANT") and calls `cv::fisheye::*`
  for equidistant. ✅
- `image_geometry` **Python** (`cameramodels.py`) does **not** look at `distortion_model` — always uses
  the pinhole functions → fisheye results will be wrong in Python consumers. Our package should do
  its own model-aware rectification/validation rather than relying on it.
- Output YAML must match `camera_calibration_parsers` format (see §1.6) so `camera_info_manager`
  (`camera_info_url: file://...`) can load it. Keep `rectification_matrix = I` and
  `projection_matrix = [K'|0]` for mono.
- `set_camera_info` service upload is optional but nice to keep for drivers that support it.
- Subscribe with the publisher's QoS (ROS does this via `get_publishers_info_by_topic`), or use
  sensor-data QoS, to avoid silent no-data with best-effort camera drivers.

---

## 4. Takeaways / proposed direction for `smart_calibration`

Keep the good structure of the ROS package (ROS-agnostic core + thin node), and fix its blind spots.

**Core library (pure Python + OpenCV/NumPy, unit-testable offline)**
- `models/`: `PinholeModel` (plumb_bob 5 / rational 8) and `FisheyeModel` (equidistant 4) behind
  one interface: `calibrate()`, `project()`, `undistort_points()`, `new_camera_matrix(alpha|balance)`,
  `rectify_maps()`, `to_ros_dict()`.
- `detection/`: **chessboard only** (decided, see §6) via `findChessboardCornersSB` (option for
  classic + `cornerSubPix`). Detect at full resolution (or downsample only for a quick "is there a
  board?" check). Configurable border margin (default 0–2 px, not 8).
- `quality/`: per-view sharpness (`estimateChessboardSharpness`), motion between frames, exposure.
- `coverage/`: a spatial grid/heatmap of corner hits over the image (esp. corners/edges for fisheye)
  plus pose diversity (distance, tilt about both axes from `solvePnP` rather than a single 2D angle).
  Accept new views based on *information gain* rather than a fixed L1 threshold.
- `solver/`: iterative calibration loop:
  1. calibrate → per-view RMS (OpenCV extended for pinhole; own reprojection for fisheye)
  2. drop views with error > `k·median` (or top-N%) / ill-conditioned (fisheye `CHECK_COND`)
  3. re-calibrate until stable; report RMS, per-view errors, **std-devs of intrinsics**
  4. sanity checks: monotonic radial distortion, principal point near center, fx≈fy, positive focal
  - optional `calibrateCameraRO` for non-ideal boards (pinhole).
  - sensible defaults: pinhole `-k` 3 (k1,k2,k3,p1,p2); fisheye `RECOMPUTE_EXTRINSIC | FIX_SKEW`.
- `io/`: write ROS-compatible YAML + a JSON/markdown report (errors, std-devs, coverage image,
  flags used); save the raw samples + detected corners so calibration can be re-run with different
  settings without re-collecting.

**ROS 2 node(s)**
- `calibration_node`: subscribes to `image` (+ `camera_info` optional for size), all settings as
  ROS parameters (YAML-configurable), services/actions for `capture`, `calibrate`, `save`, `commit`
  (`SetCameraInfo`), publishes an annotated debug image (works headless / with rqt_image_view)
  instead of only a `cv2.imshow` window.
- `offline` CLI: calibrate from an image folder or rosbag2 using the same core — critical for tuning
  the method repeatedly on the same dataset.
- Validation: hold-out views / a separate recording → reprojection error with fixed intrinsics
  (`solvePnP`), plus line-straightness (ROS' "linear error") for both models.

**Open questions to decide before implementing**
0. Which OpenCV version in the venv: 4.11 (numpy 1.x, ROS-safe) vs 4.14/5.0 (numpy 2) — see §5.
1. Python only, or C++ core (for speed at high resolution)? Python + OpenCV is likely sufficient.
2. ~~Which targets must be supported first?~~ **Decided: chessboard (square cells) only.** No
   ArUco, ChArUco, AprilTag or circle grids for now (§6).
3. Interactive GUI (OpenCV window like ROS) vs. headless node + debug image topic + rqt?
4. Is `rational_polynomial` needed, or strictly plumb_bob + equidistant?
5. Target camera resolutions / FOV (how wide are the fisheye lenses — close to 180°?).

---

## 5. Newer OpenCV in a Python venv (4.5.4 → 4.14 / 5.0)

Researched 2026-10-20. Sources: OpenCV GitHub release list; the wiki change logs
(`OpenCV-Change-Logs`, `OpenCV-Change-Logs-v2.2‐v4.10`, `OpenCV-4-to-5-migration`); the headers at
tags `4.14.0` (`calib3d.hpp`, `objdetect/charuco_detector.hpp`, `objdetect/aruco_board.hpp`) and
`5.0.0` (`calib.hpp`, `geometry/3d.hpp`, `imgproc.hpp`, `stereo.hpp`), compared symbol by symbol
with the local 4.5.4 header; the linked PR descriptions; and PyPI metadata for `opencv-contrib-python`.

### 5.1 Release landscape

| Line | Latest | Date | PyPI wheel |
|---|---|---|---|
| 4.x | **4.14.0** | 2026-07 | `4.14.0.94` |
| 5.x | **5.0.0** | 2026-06 | `5.0.0.93` (current PyPI "latest") |
| Humble system | 4.5.4 | 2021-10 | — |

All recent wheels are `abi3` (cp37+), so they install fine on Humble's Python 3.10.

### 5.2 ⚠️ Blocking constraint: numpy 2 vs. ROS Humble

| Wheel | numpy requirement on Python 3.10 |
|---|---|
| ≤ **4.11.0.86** | `numpy>=1.21.2` (numpy 1.x OK) |
| 4.12.0.88 | `numpy>=2,<2.3` |
| 4.13 / 4.14 / 5.0 | `numpy>=2` |

Humble ships numpy **1.21.5**. What I checked locally:
- `cv_bridge_boost.so` uses the numpy **C API** (`_ARRAY_API`) and was compiled against numpy 1.x,
  so it **will not import under numpy 2**. The stock `camera_calibration` and `image_geometry`
  (Python) both import `cv_bridge`.
- The rosidl message bindings (`sensor_msgs`, …) use numpy only from Python and through
  `numpy.ndarray` type checks, not the C API, so `rclpy` with sensor messages *probably* works
  under numpy 2. **This must be verified in the venv**; it is an assumption, not a test result.

Consequences / options:
- **A — `opencv-contrib-python==4.11.0.86` + `numpy<2`** in a venv created with
  `--system-site-packages`. This is the newest OpenCV that can share a process with `rclpy` and
  `cv_bridge` safely. You lose only the 4.12–4.14 items marked † in §5.3, and each has a workaround.
- **B — 4.14 or 5.0 + numpy 2.** Don't use `cv_bridge`: converting `sensor_msgs/Image` to an
  ndarray yourself takes about 10 lines (`np.frombuffer(msg.data, dtype).reshape(h, w[, c])`
  plus handling of encoding and `step`). This only works if `rclpy` imports cleanly under numpy 2.
  The safest form of B runs the calibration core as a separate offline process or CLI (numpy 2
  venv) that reads images and bags and writes YAML. The ROS node then only collects images.
- In both cases, keep the **core library free of ROS imports** (§4 already recommends this), so
  the OpenCV/numpy choice stays an environment decision rather than a code change.

colcon + venv tips: create the venv with `--system-site-packages`, put a `COLCON_IGNORE` file in
the venv folder, and build/run with the venv activated. For `ament_python` packages, set
`[build_scripts] executable = /usr/bin/env python3` in `setup.cfg` so console scripts use the
venv interpreter rather than `/usr/bin/python3`.

### 5.3 Calibration-relevant changes since 4.5.4

(† = not available in 4.11, see option A)

**Optimizer / `calibrateCamera` (pinhole)**
- 4.8: fixed the std-dev computation in calibration (#22992/#23025), so
  `calibrateCameraExtended` uncertainty values in 4.5.4 are **not trustworthy**. A strong reason to upgrade.
- 4.9: `calibrateCamera` **throws if the problem is under-constrained** (too few views / points for
  the number of free parameters) instead of returning garbage.
- 4.8: fixed pixel-grid generation in `icvGetRectangles`, which improves the accuracy of
  `getOptimalNewCameraMatrix` and `stereoRectify` (#23305). 4.12: further speed-ups to the
  new-camera-matrix estimation (#27001).
- † 4.14: `calibrateCamera` now uses **sparse bundle adjustment with Schur complement by default**,
  with parallel Jacobian accumulation (#28461): O(n) instead of O(n³) in the number of views,
  "without changing outputs" (benchmarked on 500 images against the old engine and Ceres). New
  flag `CALIB_DISABLE_SCHUR_COMPLEMENT` selects the old Bouguet engine. Mostly a **speed** gain,
  which makes a calibrate → reject outliers → recalibrate loop cheap even with hundreds of views.
- 5.0: the 5.0-alpha notes mention a "new Levenberg–Marquardt implementation, faster and more accurate".

**Fisheye**
- 4.10: **new initial focal-length estimate in `fisheye.calibrate`** (#25030). The old
  initialization assumed an FOV that came out above 180° at the image corners, which caused
  numerical failures. This fixes one known cause of fisheye calibration diverging or failing.
- 4.10: **`cv2.fisheye.solvePnP`** (#25028). Lets us compute per-view poses and errors for
  validation, held-out tests and checker-style tools with the fisheye model. 4.5.4 has no fisheye
  PnP, so you have to undistort first and then call the pinhole `solvePnP`.
- † 4.12: `cv2.fisheye.solvePnPRansac` (#26669); optimized fisheye `undistortPoints` (#26988).
- 4.11: `fisheye.distortPoints` with a non-identity projection matrix (#25943).
- † 4.13: fixed-point convergence fix in `undistortPoints` (#27993).
- **Unchanged through 4.14:** the `fisheye.calibrate` signature and flags. It still returns **no
  per-view errors or std-devs**, so we compute those ourselves with `fisheye.projectPoints`.

**Chessboard / circle detection**
- 4.9–4.11: many `findChessboardCorners` / `findChessboardCornersSB` fixes (#25365, #25427,
  #25807, #25991, #26014, …); improved symmetric circle-grid detection (#25258).
- 4.11: SB detector works with the **marker on a black tile** too (#25808). † 4.12: fixed marker
  ordering for `findChessboardCornersSBWithMeta` + `CALIB_CB_LARGER`.
- † 4.14: new `CALIB_CB_PLAIN` flag (no preprocessing, faster, for pre-binarized images); Delaunay
  triangulation in `findCirclesGrid` (#29117).
- 4.9: fixed out-of-image access in `cornerSubPix` (#24527).

**ArUco / ChArUco** (big change, relevant for edge coverage and fisheye). **Out of scope for now**
(chessboard only, §6); kept for reference in case ChArUco is added later.
- 4.7: ArUco/ChArUco moved from contrib into the **main `objdetect` module** as
  `cv2.aruco.ArucoDetector` and `cv2.aruco.CharucoDetector`. Plain `opencv-python` now has it,
  so contrib isn't needed. The old `aruco.calibrateCameraCharuco` / `interpolateCornersCharuco`
  API is gone from main.
- Recommended pipeline (works for **both pinhole and fisheye**, which removes the ROS
  "fisheye + ChArUco not implemented" limitation):
  `CharucoDetector(board, CharucoParameters, DetectorParameters).detectBoard(img)` →
  `charuco_corners, charuco_ids` → `board.matchImagePoints(charuco_corners, charuco_ids)` →
  `obj_pts, img_pts` → `cv2.calibrateCamera(...)` **or** `cv2.fisheye.calibrate(...)`.
- `CharucoParameters`: `minMarkers` (default 2), `tryRefineMarkers`, `checkMarkers`, and optional
  `cameraMatrix`/`distCoeffs`. When these are given, corner interpolation uses an approximate pose
  instead of a local homography, so after a first calibration you can re-detect with the
  intrinsics for better corners.
- `CharucoBoard.checkCharucoCornersCollinear(ids)` rejects degenerate partial views (collinear
  corners make the homography/PnP singular).
- **Board-layout gotcha:** since 4.6.0, boards with an **even number of rows** are generated with a
  different start colour. Boards printed with older OpenCV (or older ROS docs) need
  `board.setLegacyPattern(True)` (4.8+), or detection fails or the IDs come out wrong.
- 4.8 (#24355): more accurate ArUco corner refinement with a dynamic window. 4.12: multi-dictionary
  detection, fixed detection near the image edge (#26968), and the ChArUco board consistency check
  can be turned off.
- † 4.14: **fixed a systematic ~0.5 px offset in `CharucoDetector` sub-pixel refinement**
  (#28380): corners were shifted −0.5 px before `cornerSubPix` and shifted back afterwards. Up to
  and including 4.13, run `cv2.cornerSubPix` yourself on `charuco_corners` or add 0.5 px back.
  Otherwise the principal point is biased by about 0.5 px.

**Python binding quality**
- 4.13/4.14: `distCoeffs`, `cameraMatrix`, `R`, `D` are marked **optional** in calib3d functions,
  so you can pass `None`, and the type stubs are much better (useful with mypy/pyright).

**Tools (not shipped in the wheels, but useful references)**
- 4.13: the `interactive-calibration` app gained extended calibration flags and camera backend
  selection; the pattern generator moved to `apps/` (`generate_pattern.py`, ChArUco fixes, ArUco
  index offset). You can use it to print boards: `python generate_pattern.py -t charuco_board ...`.

### 5.4 OpenCV 5.0 specifics

- **Modules split:** `calib3d` became `calib` (calibrateCamera, fisheye.calibrate, multiview),
  `geometry` (solvePnP, projectPoints, undistortPoints, getOptimalNewCameraMatrix, fisheye
  project/undistortPoints/solvePnP) and `stereo`. The fisheye `initUndistortRectifyMap` and
  `undistortImage` moved to `imgproc`. **Python is unaffected**: everything stays at `cv2.<name>`
  and `cv2.fisheye.<name>` per the migration guide.
- **Unified flags:** fisheye flags were merged into `cv::CALIB_*` with **new numeric values**
  (e.g. `CALIB_RECOMPUTE_EXTRINSIC = 1<<23`, `CALIB_CHECK_COND = 1<<24`, `CALIB_FIX_SKEW = 1<<25`,
  `CALIB_DISABLE_SCHUR_COMPLEMENT = 1<<18`, against 1<<1, 1<<2, 1<<3 and 1<<23 in 4.x). C++ keeps
  `fisheye::CALIB_*` as `using` aliases. **Never hard-code flag integers; always use the named
  constants.** Whether `cv2.fisheye.CALIB_*` still exists in the 5.0 Python bindings needs a check
  in the venv. If not, use `cv2.CALIB_*`.
- New `CameraModel` enum (`CALIB_MODEL_PINHOLE`, `CALIB_MODEL_FISHEYE`), used by the new
  **`calibrateMultiview` / `calibrateMultiviewExtended`** (several cameras at once, pinhole or
  fisheye, supports partial observations via invalid points and a detection mask, returns
  per-frame errors) and **`registerCameras`** (extrinsics between two cameras). Both are
  multi-camera features. For mono they don't matter, but the multiview tutorial and sample show
  good practice for detection masks and per-frame errors.
- 1D-array semantics: vector-backed `Mat`s are 1D in 5.x. In Python, the shapes of some outputs
  (e.g. `perViewErrors`, `distCoeffs`) may change. Always `np.ravel()` or `reshape()` explicitly.
- C++17 required (irrelevant for Python).

### 5.5 What this changes in the plan (§4)

- ~~Use the modern ChArUco pipeline.~~ Superseded: chessboard only (§6). Use
  `findChessboardCornersSB` (benefits from the 4.9–4.11 detector fixes, all in 4.11).
- Rely on `calibrateCameraExtended` **std-devs** (fixed in 4.8) for uncertainty-based stopping:
  "collect views until σ(fx, fy, cx, cy) < threshold".
- Use `fisheye.solvePnP` (4.10+) for fisheye validation (held-out reprojection error) and
  per-view error computation.
- Catch `cv2.error` from under-constrained `calibrateCamera` (4.9+) and report it as "need more /
  more varied views".
- Write a thin `flags.py` that builds flags from named constants and works on both 4.x and 5.x.
- Version floor: **≥ 4.10** (fisheye init fix + fisheye PnP). **4.11.0.86** is the practical pin
  if `cv_bridge` is needed in-process; **4.14.0.94** if the numpy 2 path (option B) is validated.
  Prefer 4.14 over 5.0 for now: 5.0 is weeks old, its ChangeLog is still "TBD", and it has no
  mono-calibration advantage over 4.14 that we need.

### 5.6 Things to verify once the venv exists

**Verified 2026-10-20 for the chosen 4.11 path** (`requirements.txt`, venv with
`--system-site-packages`, ROS sourced): `cv2` 4.11.0 loads from the venv (not the system
4.5.4), numpy stays at the system 1.21.5, a `cv_bridge` mono8 round-trip and `sensor_msgs` work in
the same process, and on synthetic pinhole renders of a 9×6 board `findChessboardCornersSB`,
`estimateChessboardSharpness`, `calibrateCameraExtended` (fx 601.6 vs 600 true, RMS 0.08 px,
`perViewErrors` shape (N, 1)), `fisheye.calibrate` and `fisheye.solvePnP` all work. Items 1 and 3
below only matter for the numpy 2 / 5.0 path.

1. With numpy 2: `python -c "import rclpy, sensor_msgs.msg; ..."` round-trips an `Image` /
   `CameraInfo` message. Also confirm that `import cv_bridge` fails, as expected.
2. Which `cv2` gets imported (`cv2.__file__`) when ROS's `PYTHONPATH` is sourced; the system
   `python3-opencv` must not shadow the venv one.
3. In 5.0 only: whether `cv2.fisheye.CALIB_FIX_SKEW` exists and what its value is.
4. The output shapes of `calibrateCameraExtended`, and the `findChessboardCornersSBWithMeta`
   output (corners + meta) in the chosen version.

---

## 6. Decision: square-cell chessboard only

Scope for now: a standard checkerboard with square cells. ArUco, ChArUco, AprilTag, circle grids
and other targets are out of scope (§5.3 ChArUco notes kept for reference only).

### 6.1 Effect on the OpenCV 4.11 vs 4.14 choice

With ChArUco gone, the only real accuracy fix in 4.12–4.14 (the ~0.5 px `CharucoDetector` offset)
no longer applies. Without 4.14 we lose only:
- the faster Schur-complement `calibrateCamera` (speed only, same results),
- the `undistortPoints` convergence fix (4.13; matters only with strong pinhole edge distortion),
- the `findChessboardCornersSBWithMeta` + `CALIB_CB_LARGER` ordering fix (4.12; matters only if
  we use `LARGER`, see 6.3),
- `CALIB_CB_PLAIN` and fisheye `solvePnPRansac` (not needed).

So **`opencv-contrib-python==4.11.0.86` (or plain `opencv-python==4.11.0.86`, since contrib isn't
needed without ArUco) + `numpy<2`** is a sound default. That keeps `cv_bridge`/`rclpy` working
in-process.

### 6.2 Board design recommendations

- **Use an inner-corner count with one odd and one even dimension** (e.g. 9×6 or 7×10 inner
  corners). An N×N grid is 90°-rotation symmetric, and even×even or odd×odd grids are
  180°-symmetric, so corner ordering can flip between views. Intrinsics are unaffected (each view
  has its own pose), but per-view poses, coverage/tilt statistics and debugging become
  inconsistent. ROS needs special re-ordering code for exactly this case (§1.2).
- Leave a **white border of about one square** around the grid, which `findChessboardCornersSB`
  needs for reliable detection. Rounded outer corners help the SB detector (per OpenCV docs).
- Rigid and flat (glass, aluminium composite or similar), with the square size measured after
  printing. If flatness or print scale is doubtful, `calibrateCameraRO` (pinhole) can absorb small
  board errors.
- Matte surface, to avoid specular highlights that break corner detection.

### 6.3 Chessboard-specific weaknesses and how to compensate

- **The whole grid must be visible** (both detectors). That makes the image corners hard to cover,
  which is the critical area for fisheye and high-order distortion. Mitigations:
  - A board with fewer/larger squares, or a smaller board for corner shots, so it fits near the edges.
  - Coverage feedback (§4 `coverage/`) that tells the user exactly which image regions are still empty.
  - Optional later: `findChessboardCornersSB` with `CALIB_CB_LARGER` allows a board **larger than
    `patternSize`** that is only partly visible. Object points then come from the detected grid
    size in `meta`. Without markers the grid origin isn't consistent across views, which is fine
    for intrinsics because each view has its own pose. (On 4.11, check the ordering bug fixed in 4.12.)
- Detection flags: start with `CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_EXHAUSTIVE | CALIB_CB_ACCURACY`
  and keep the classic `findChessboardCorners` + `cornerSubPix` (tight criteria, e.g. eps 1e-3) as a
  fallback option.
- Reject blurry views with `estimateChessboardSharpness` (docs: transition width should be < ~3 px).

---

## 7. Decision: separate Qt GUI node (2026-10-21)

- **Architecture:** the calibration node stays headless; a separate **GUI node** provides the
  visual feedback and manual inputs, so the GUI can run on a laptop while the camera and
  calibration node run on a robot or drone, connected over ROS.
  - Calibration node → GUI: detection results (corners, board found, coverage, quality),
    calibration state and progress.
  - GUI → calibration node: services for button actions (calibrate, save, reset, ...) and ROS
    parameters (declared with ranges) for sliders.
  - The GUI subscribes to the camera topic itself and draws the overlay locally from the small
    detection messages, rather than the calibration node publishing annotated full-size images.
- **Toolkit:** **PySide6 6.11.2** (Qt 6). All visualization in Qt, so OpenCV is switched to
  **`opencv-python-headless==4.11.0.86`** (the regular wheel bundles its own Qt, which conflicts
  with PySide6).
- **Verified** in a throwaway venv (opencv-python-headless 4.11.0.86, numpy 1.26.4, PySide6 6.11.2,
  ROS Humble sourced): `rclpy` spinning in a background thread with Qt on the main thread,
  `cv_bridge` → `QImage` display (41 frames in 2 s, offscreen), and Qt's `xcb` platform plugin
  loads on this desktop. The wheel needs glibc ≥ 2.34 (Ubuntu 22.04: 2.35) and `libxcb-cursor0`.
- **Interfaces (decided):** custom messages/services live in a separate `ament_cmake` package,
  `smart_calibration_interfaces`, next to the Python package in the same repository (option A).
  - Tested alternatives: a package nested inside another package's folder is not discovered by
    colcon; a single `ament_cmake` package with interfaces plus Python nodes works only if the
    Python module is named differently from the package (the generated `<pkg>.msg` Python package
    collides with a same-named module at CMake configure time).
  - Repository layout: `smart_calibration/` (ament_python) and `smart_calibration_interfaces/`
    (ament_cmake) as subfolders; `.venv`, `requirements.txt`, `explore.md`, `README.md` at the
    repository root. `.venv` needs a `COLCON_IGNORE` file, since colcon now searches the root.

---

## 8. Field result: ircam_wide camera (2026-10-21)

- First tests with the real camera (`/ircam_wide/decoded`) and a 6x7 inner-corner board: with
  default settings the board was found only occasionally.
- Measures that helped, found by the user with the live GUI controls:
  - `detector_exhaustive` + `detector_accuracy` on, and preprocessing `gamma` ≈ 0.45 →
    **~80-90 % detection rate** on this camera / dataset.
  - `detector_exhaustive` alone already helped but lowered the frame rate to ~15 fps.
  - Gamma < 1 darkens bright tones, which suggests the board's light squares are close to
    saturation in this camera.
- In the synthetic test, `accuracy` took detection time from ~55 ms to ~178 ms per 1280x720
  frame (3.2x), consistent with the benchmark (§ flags). It also lowers the corner error, which
  matters for calibration quality, so the slower detection rate is a reasonable trade.
- The tuned values are stored in `smart_calibration/config/ircam_wide.yaml` (selected with the
  `params_file` launch argument); gamma, stretch percentiles and detector flags are live
  parameters (GUI controls / `ros2 param set`).
