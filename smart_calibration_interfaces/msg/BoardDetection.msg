# Result of one chessboard detection on a camera frame, published by calibration_node.
# The GUI pairs it with the frame it received from the camera topic itself via header.stamp.

# Copied from the source sensor_msgs/Image.
std_msgs/Header header

# Size of the source image that the corner coordinates refer to.
uint32 image_width
uint32 image_height

# False: no board in this frame; cols, rows and corners are then 0 / empty.
bool found

# Detected grid of inner corners: cols corners per row, rows rows. With board_mode
# oversized_marker the board can be partly visible, so this can vary per frame.
uint32 cols
uint32 rows

# Row and column in the detected grid of the board origin, the corner with board coordinates
# (0, 0): the corner at (row, col) has board coordinates (col - origin_col, row - origin_row),
# in squares. Oversized board: the corner given by the markers, the same physical corner in
# every frame. Plain chessboard: 0, 0 (the first detected corner).
uint32 origin_row
uint32 origin_col

# Corner positions in source-image pixels, row-major, alternating x and y:
# x0, y0, x1, y1, ... (length 2 * cols * rows). Pixel centres are at integer coordinates
# (OpenCV convention). NaN for corners that are not used: outside the image or closer than
# border_margin_px to its border (the detector may predict or mismatch corners there).
float32[] corners

# Time spent on detection for this frame, for real-time feedback.
float32 detection_time_ms

# View selection checks, evaluated on every found board (all of them, even if one fails).
# A frame is kept as a calibration view only if all three passed.
uint8 CHECK_NOT_EVALUATED = 0  # no board, or the value cannot be measured (value is NaN)
uint8 CHECK_PASSED = 1
uint8 CHECK_FAILED = 2

# Motion: board speed since the previous detection in image widths per second (mean corner
# movement), against max_board_speed_rel. Not evaluated for the first detection and after a gap.
uint8 speed_check
float32 board_speed

# Corners: largest distance of a corner from where its neighbours put it, in square sizes,
# against max_corner_residual (a misplaced corner or a deformed region fails the frame). Not
# evaluated if no corner has enough valid neighbours.
uint8 residual_check
float32 corner_residual

# New view: distance of the board outline to the nearest kept view, in board sizes (mean side
# length of the two outlines), against min_view_distance. The first view passes with a NaN distance.
uint8 distance_check
float32 view_distance

# True if this frame was kept as a calibration view.
bool kept

# Number of views kept for calibration so far, including this frame if it was kept.
uint32 kept_views
