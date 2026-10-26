# smart_calibration

Interactive monocular camera calibration for ROS 2 Humble. A calibration node detects a
chessboard in the camera stream, keeps only good and new views, and calibrates in the background
while a GUI shows where the calibration is still uncertain. When enough views are collected, a
final pass removes outlier views, and the result is saved as an OpenCV YAML file.

- **Board modes:** a plain chessboard, or an oversized radon checkerboard with markers that only
  needs to be partly visible
- **View checks:** motion, misplaced corners and duplicates are rejected before a view is kept
- **Live guidance:** an uncertainty map over the image, plus hints on missing view types (tilts,
  image corners)
- **Final pass:** outlier views are removed automatically, and you can inspect each view's corner
  errors
- **Undistorted view:** check the calibration on the live stream
- **Tunable preprocessing** for low-contrast cameras (e.g. thermal): contrast stretch, gamma and
  an S-curve, with a live histogram

Camera model: pinhole with `plumb_bob` distortion (k1, k2, p1, p2, k3); fisheye is planned.

## Installation

The nodes run in a Python venv with OpenCV 4.11, numpy 1.26 and PySide6
([requirements.txt](requirements.txt) explains the pins). ROS's own Python packages (`rclpy`,
`cv_bridge`) come from sourcing ROS. Qt also needs the `libxcb-cursor0` apt package.

```bash
cd ~/ros2_ws/src
git clone https://github.com/JonanaBanana/smart_calibration.git
cd smart_calibration
python3 -m venv .venv --prompt smart_calibration
touch .venv/COLCON_IGNORE
source .venv/bin/activate && pip install -r requirements.txt
deactivate
```

Build with the venv **deactivated** and **without** `--symlink-install`:

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select smart_calibration_interfaces smart_calibration
```

The message generators need the system Python, so the build stops with an error if a venv is
active. The installed executables run on the first `python3` on `PATH`, which is the venv's once
it's activated; symlink builds would ignore that. Rebuild after every change, including edits to
the config files.

## Usage

In every terminal that runs a node:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
source ~/ros2_ws/src/smart_calibration/.venv/bin/activate
```

```bash
# node and GUI on one machine
ros2 launch smart_calibration calibration_with_gui.launch.py image_topic:=/my_camera/image

# or separately, e.g. the node on a robot and the GUI on a laptop (same ROS_DOMAIN_ID)
ros2 launch smart_calibration calibration.launch.py image_topic:=/my_camera/image
ros2 launch smart_calibration gui.launch.py
```

| Launch argument | Default | |
|---|---|---|
| `image_topic` | `/ircam_wide/decoded` | camera image topic (the GUI only uses it for the raw view) |
| `params_file` | installed `config/ircam_wide.yaml` | node parameters (not for `gui.launch.py`) |
| `log_level` | `info` | `debug` logs one line per frame (not for `gui.launch.py`) |
| `calibration_node` | `/calibration_node` | node the GUI connects to (`gui.launch.py` only) |

### Calibrating

1. **Collect views.** Hold the board still in different places, distances and tilts. The status
   box at the top left shows the checks for each frame: motion, corners, new view. "kept #N" means
   the view was kept.
2. **Follow the guidance.** From 5 views on, the node calibrates in the background. The
   uncertainty map shows where the calibration is still weak: bring the board to the red regions.
   The amber hints name missing view types, e.g. "Tilt the top or bottom edge towards the camera".
3. **Finish.** The node stops collecting and removes clear outliers: views the other views predict
   badly. The video then shows all kept views; click one to see its corner errors.
   - **Restore:** puts a removed view back.
   - **Resume:** collects more views.
   - **Suspect** (amber): a flagged view that is the only one of its region. Add views there.
4. **Accept** saves the calibration to `output_file`, on the machine running the node.

Check **undistorted** in the View box to see the stream undistorted by the current calibration.
A dashed amber line marks where the distortion model is not valid yet; views near the image
corners move it out.

### Output

An OpenCV YAML file, readable with `cv2.FileStorage`:

```yaml
%YAML:1.0
---
M: !!opencv-matrix        # camera matrix, 3 x 3
   ...
D: !!opencv-matrix        # k1, k2, p1, p2, k3 (5 x 1)
   ...
reprojection_error: 0.11
width: 1280
height: 720
distortion_model: plumb_bob
calibration_time: "2026-10-26 15:02:11"
```

## Boards

- **`chessboard`:** a plain chessboard, always completely visible. Set `board_cols` and
  `board_rows` to its **inner** corners (squares − 1). Leave a white border of about one square.
- **`oversized_marker`:** a radon checkerboard larger than the camera's view. Three marker dots
  in an L near its middle fix its orientation and origin. Exactly one of the three must be a black
  dot in a white cell; that cell's top-left corner is the origin. Any visible part with at least
  5 × 5 corners and the markers is detected, so the image corners are easy to cover.

Generate a board with OpenCV's pattern generator, e.g. 14 × 10 squares of 50 mm with markers in
cells (7, 4), (8, 4) and (7, 5):

```bash
python utility/generate_pattern.py -o radon.svg -T radon_checkerboard -c 14 -r 10 -s 50 \
    -u mm -w 800 -h 600 -m 7 4 8 4 7 5
```

Print it flat and rigid, and measure the printed square size for `square_size`.

## Parameters

Set in the parameter file ([config/ircam_wide.yaml](smart_calibration/config/ircam_wide.yaml),
one per camera). Except for the startup ones, they can be changed while running, from the GUI or
with `ros2 param set /calibration_node <name> <value>`. Such changes are not written back to the
file.

| Parameter | Meaning |
|---|---|
| `board_mode`, `board_cols`, `board_rows`, `square_size` | board, see [Boards](#boards); `square_size` in metres (startup) |
| `stretch_low_percentile`, `stretch_high_percentile` | intensity percentiles mapped to black and white |
| `gamma` | after the stretch; > 1 brightens dark regions |
| `contrast_strength`, `contrast_midpoint` | S-curve after gamma: grey levels below the midpoint get darker, above it lighter; strength 0 is off. Put the midpoint between the dark and light squares' peaks in the GUI histogram |
| `detector_normalize`, `detector_exhaustive`, `detector_accuracy` | detector flags: histogram equalization; longer search (finds more boards, slower); more accurate corners (~4× slower) |
| `max_board_speed` | largest board speed of a kept view, px/s |
| `max_corner_residual` | largest distance of a corner from where its neighbours put it, in squares; one bad corner rejects the frame |
| `min_view_distance` | smallest difference of the board outline to every kept view, in board sizes |
| `border_margin_px` | corners closer to the image border are not used |
| `min_calibration_views`, `calibrate_every_views` | first background calibration at this many views, then after every this many new views |
| `free_k3_after_views` | k3 stays 0 below this many views |
| `uncertainty_scale` | factor on the estimated uncertainty, which assumes independent corner errors and comes out too small |
| `outlier_z_limit` | final pass: robust z-score of a view's leave-one-out error that flags it |
| `outlier_min_views` | final pass: no outlier check with fewer views |
| `outlier_max_uncertainty_increase` | final pass: a flagged view is kept as a suspect if removing it raises the mean uncertainty by more than this factor |
| `outlier_max_fraction` | final pass: at most this fraction of the views is removed |
| `output_file` | where Accept saves; strftime placeholders (`%Y`, `%m`, ...) and `~` work |
| `image_qos` | `auto` (match the camera), `reliable` or `best_effort` (startup) |
| `preview_max_width`, `preview_jpeg_quality` | preview images sent to the GUI (startup) |

## ROS interface

All names are relative to the node (`/calibration_node/...`). The types are in
`smart_calibration_interfaces`.

| Name | Type | |
|---|---|---|
| `image` (input) | `sensor_msgs/Image` | camera stream: mono, colour, Bayer, YUV, 16-bit or float |
| `~/detection` | `BoardDetection` | corners and check results of every processed frame |
| `~/preview`, `~/histogram` | `CompressedImage`, `ToneHistogram` | preprocessed image and its histograms, while the GUI watches |
| `~/calibration` | `CalibrationState` | newest background calibration, uncertainty map, hints |
| `~/final_calibration` | `FinalCalibration` | state and result of the final pass |
| `~/finish` | `Finish` (service) | stop collecting and run the final pass; `keep` = views not to remove |
| `~/accept`, `~/resume` | `std_srvs/Trigger` | save the result; collect more views |
| `~/get_view` | `GetView` (service) | a kept view's thumbnail and corner errors |

## Development

```bash
# tests (system Python, with ROS sourced), from smart_calibration/
python3 -m pytest test/

# synthetic test images with exact ground truth, and an evaluation of detection and calibration
# (venv active, from the repository root)
python utility/synthetic_dataset.py --output dataset/synthetic
python utility/evaluate_dataset.py dataset/synthetic --exhaustive --visualize dataset/synthetic/visualized
```

`utility/uncertainty_prototype.py` and `utility/outlier_prototype.py` are the experiments behind
the uncertainty scale and the outlier rules.

```
smart_calibration/                 repository root
├── smart_calibration/             ament_python package: node, GUI, config, launch files, tests
├── smart_calibration_interfaces/  messages and services
├── utility/                       pattern generator, synthetic dataset, evaluation, prototypes
├── requirements.txt               venv packages
└── THIRD_PARTY_NOTICES.md
```

## License

Apache-2.0. `utility/generate_pattern.py` and `utility/svgfig.py` come from OpenCV; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
