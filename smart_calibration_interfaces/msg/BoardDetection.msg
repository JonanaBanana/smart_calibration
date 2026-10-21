# Result of one chessboard detection on a camera frame, published by calibration_node.
# The GUI pairs it with the frame it received from the camera topic itself via header.stamp.

# Copied from the source sensor_msgs/Image.
std_msgs/Header header

# Size of the source image that the corner coordinates refer to.
uint32 image_width
uint32 image_height

# False: no board in this frame; cols, rows and corners are then 0 / empty.
bool found

# Detected grid of inner corners: cols corners per row, rows rows.
uint32 cols
uint32 rows

# Corner positions in source-image pixels, row-major, alternating x and y:
# x0, y0, x1, y1, ... (length 2 * cols * rows). Pixel centres are at integer coordinates
# (OpenCV convention).
float32[] corners

# Time spent on detection for this frame, for real-time feedback.
float32 detection_time_ms

# View selection checks, evaluated on every found board (all of them, even if one fails).
# A frame is kept as a calibration view only if all three passed.
uint8 CHECK_NOT_EVALUATED = 0  # no board, or the value cannot be measured (value is NaN)
uint8 CHECK_PASSED = 1
uint8 CHECK_FAILED = 2

# Motion: board speed since the previous detection in pixels per second (mean corner movement),
# against max_board_speed. Not evaluated for the first detection and after a gap.
uint8 speed_check
float32 board_speed

# Corners: grid irregularity in square sizes (largest deviation of a corner from the smooth grid of
# its neighbours), against max_grid_irregularity.
uint8 irregularity_check
float32 grid_irregularity

# New view: distance of the board outline to the nearest kept view, as a fraction of the image
# diagonal, against min_view_distance. The first view passes with a NaN distance.
uint8 distance_check
float32 view_distance

# True if this frame was kept as a calibration view.
bool kept

# Number of views kept for calibration so far, including this frame if it was kept.
uint32 kept_views
