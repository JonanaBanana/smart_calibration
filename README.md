# smart_calibration

ROS 2 (Humble) packages for tunable monocular camera calibration with a square-cell chessboard,
for pinhole (`plumb_bob`) and fisheye (`equidistant`) camera models.
Background research and design notes are in [explore.md](explore.md).

## Repository layout

```
smart_calibration/                 repository root
├── smart_calibration/             ament_python package: calibration node (and later the GUI)
├── smart_calibration_interfaces/  ament_cmake package: message / service definitions
├── .venv/                         Python venv shared by the nodes (not in git)
├── requirements.txt
└── explore.md                     research and design notes
```

## Installation

### 1. Python dependencies (venv at the repository root)

```bash
cd ~/ros2_ws/src/smart_calibration
python3 -m venv .venv --prompt smart_calibration
touch .venv/COLCON_IGNORE   # keep colcon from searching the venv for packages
# `&&`: stop if activation fails, so pip can never install outside the venv
source .venv/bin/activate && pip install -r requirements.txt
python -c "import cv2; print(cv2.__version__, cv2.__file__)"   # expect 4.11.0, inside .venv/
deactivate
```

The venv is isolated (no system site-packages). ROS Python packages (`rclpy`, `cv_bridge`, ...)
come from sourcing ROS, not from the venv. See [requirements.txt](requirements.txt) for why
OpenCV 4.11 (headless) / numpy 1.26 / PySide6 6.11 are pinned. Qt 6 also needs the
`libxcb-cursor0` apt package.

If an existing venv still has the non-headless `opencv-python`, remove it first; both wheels
install the same `cv2` module and must not be installed together:

```bash
source .venv/bin/activate && pip uninstall -y opencv-python && pip install -r requirements.txt
```

### 2. Build

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select smart_calibration_interfaces smart_calibration
source install/setup.bash
```

Build with the venv **deactivated**: the message code generators run on the first `python3` on
`PATH` and need system Python packages the venv does not have (the interfaces package stops with
an error otherwise). Activate the venv only for running the nodes.

Build **without** `--symlink-install`: `smart_calibration/setup.cfg` gives the installed
executables a `#!/usr/bin/env python3` shebang, and colcon ignores that setting for symlink builds.
Rebuild after every code change (files are copied into `install/`).

## Running

Activate the venv first, in every terminal that runs something from these packages:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
source ~/ros2_ws/src/smart_calibration/.venv/bin/activate

ros2 launch smart_calibration calibration_with_gui.launch.py # both on this machine
# or separately, e.g. the GUI on a laptop:
ros2 launch smart_calibration calibration.launch.py          # calibration node (on the robot)
ros2 launch smart_calibration gui.launch.py                  # GUI (on the robot or a laptop)
# or
ros2 run smart_calibration calibration_node
ros2 run smart_calibration calibration_gui
```

The calibration node and the GUI talk over ROS only, so the GUI can run on another machine in the
same ROS network (same `ROS_DOMAIN_ID`), e.g. a laptop connected to a drone. Both need the venv.

Launch arguments:

| Launch file | Argument | Default |
|---|---|---|
| `calibration.launch.py`, `calibration_with_gui.launch.py` | `image_topic` | `/ircam_wide/decoded` |
| | `params_file` | `config/ircam_wide.yaml` (installed copy) |
| `gui.launch.py` | `image_topic` (raw view only) | `/ircam_wide/decoded` |
| | `calibration_node` | `/calibration_node` |

calibration_node's settings live in a parameter file, one per camera / board
([config/ircam_wide.yaml](smart_calibration/config/ircam_wide.yaml) is the tuned example):

| Parameter | Live | Meaning |
|---|---|---|
| `board_cols`, `board_rows` | no | inner corners of the chessboard (squares - 1) |
| `stretch_low_percentile`, `stretch_high_percentile` | yes | contrast stretch: these intensity percentiles become black / white |
| `gamma` | yes | applied after the stretch; > 1 brightens dark regions, < 1 darkens bright ones |
| `detector_normalize`, `detector_exhaustive`, `detector_accuracy` | yes | findChessboardCornersSB flags (see explore.md) |
| `image_qos` | no | `auto` matches the camera publisher's reliability (best effort loses many large images; reliable cannot connect to a best-effort publisher), or `reliable` / `best_effort` |
| `preview_max_width`, `preview_jpeg_quality` | no | preview images sent to the GUI |

Live parameters can be changed while running, from the GUI (gamma slider, detector checkboxes)
or with `ros2 param set /calibration_node <name> <value>`; the others are rejected while running.
Changes made while running are not written back to the file. Since the installed copy is used,
rebuild after editing a file under `config/`, or pass your own file with `params_file:=<path>`.

The executables run on whatever `python3` is first on `PATH`, which is the venv's interpreter
when it is activated. Without the venv, the node exits with an error instead of silently running
on the system Python and its OpenCV 4.5.4.

The camera topic is set with `image_topic:=<topic>` (default `/ircam_wide/decoded`), or with
`--ros-args -r image:=<topic>` for `ros2 run`.

What it does so far:

- **calibration_node** converts each camera frame to 8-bit grayscale (8-bit color/Bayer/YUV via
  cv_bridge; 16-bit and float images scaled by their maximum so 10/12-bit sensors keep their
  contrast), detects the chessboard with `findChessboardCornersSB` in a worker thread (always on
  the newest frame; frames arriving while it is busy are dropped), and publishes a
  `smart_calibration_interfaces/BoardDetection` per processed frame on `~/detection`.
  Run with `--log-level calibration_node:=debug` for a line per frame.
- **calibration_gui** (PySide6) shows the camera image with the detected corners drawn on top,
  board / image info and camera / detection rates. It shows the frame each detection was computed
  on (matched by timestamp), so the overlay sits exactly on the board and the video lags by one
  detection. Without detections for 1 s it shows the live image without overlay.
