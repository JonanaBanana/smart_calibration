# Background calibration from the kept views and its uncertainty, published by calibration_node
# every few kept views (see calibration.py).

# stamp: image stamp of the newest view used.
std_msgs/Header header

# Size of the images the views were detected in.
uint32 image_width
uint32 image_height

# Views and corners used, and how long the calibration took.
uint32 views
uint32 points
float32 duration_ms

# Lens model (camera_model parameter): plumb_bob, or fisheye (Kannala-Brandt, cv2.fisheye).
string camera_model

# k3 (fisheye: k3 and k4) is held at 0 (fix_higher_order parameter).
bool k3_fixed

# Reprojection error (RMS over all corners) and the point noise estimated from it, px.
float64 rms
float64 noise

# Camera matrix, row-major (fx 0 cx; 0 fy cy; 0 0 1), and distortion in OpenCV's order:
# plumb_bob k1, k2, p1, p2, k3; fisheye k1, k2, k3, k4.
float64[9] camera_matrix
float64[] distortion

# Standard deviation of fx, fy, cx, cy and each distortion coefficient (0 if fixed).
float64[] std_deviations

# Largest uncertainty over the image, px: from all parameters, and from each parameter group
# alone (the others taken as exact).
float32 uncertainty
float32 focal_uncertainty
float32 centre_uncertainty
float32 distortion_uncertainty

# Uncertainty map: map_rows x map_cols cells dividing the image evenly, row-major. Each value is
# how far a point at the cell centre could be projected off, px (RMS length of the 2-D error).
# NaN where the model has no ray.
uint32 map_cols
uint32 map_rows
float32[] uncertainty_map

# Per view, in the order kept: reprojection error (px), board tilt (degrees between the board
# normal and the optical axis) and tilt axis direction in the image (degrees, 0 horizontal).
float32[] view_rms
float32[] view_tilt
float32[] view_tilt_axis

# Missing view types (inform only; new views fix them).
string[] hints

# The distortion model folds over inside the image: some pixels have no valid ray, and undistorting
# with it shows the image a second time, mirrored, near the image corners.
bool distortion_folds
