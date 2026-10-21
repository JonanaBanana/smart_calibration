# Intensity histograms of calibration_node's tone mapping for one frame, published together with
# each preview (same header) while the preview has subscribers.

# Copied from the source sensor_msgs/Image.
std_msgs/Header header

# 256 bins over the 0-1 intensity range (pixel counts).
# Before the S-shaped contrast curve, i.e. after the contrast stretch and gamma:
uint32[256] before_contrast
# Of the final detection image:
uint32[256] after_contrast
