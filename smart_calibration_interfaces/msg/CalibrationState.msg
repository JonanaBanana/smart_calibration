# Background calibration from the kept views and its uncertainty, published by calibration_node
# every few kept views (see calibration.py). Pinhole model (plumb_bob).

# stamp: image stamp of the newest view used.
std_msgs/Header header

# Size of the images the views were detected in.
uint32 image_width
uint32 image_height

# Views and corners used, and how long the calibration took.
uint32 views
uint32 points
float32 duration_ms

# k3 is held at 0 while there are fewer than free_k3_after_views views.
bool k3_fixed

# Reprojection error (RMS over all corners) and the point noise estimated from it, px.
float64 rms
float64 noise

# Camera matrix, row-major (fx 0 cx; 0 fy cy; 0 0 1), and distortion k1, k2, p1, p2, k3.
float64[9] camera_matrix
float64[5] distortion

# Standard deviation of fx, fy, cx, cy, k1, k2, p1, p2, k3 (0 if fixed), times uncertainty_scale.
float64[9] std_deviations

# All uncertainties below are multiplied by this (the raw ones underestimate the true error).
float32 uncertainty_scale

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
