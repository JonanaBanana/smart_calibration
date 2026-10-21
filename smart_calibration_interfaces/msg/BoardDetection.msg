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
