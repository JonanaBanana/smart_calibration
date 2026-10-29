# State of the final calibration pass (the GUI's Finish button), published by calibration_node
# whenever it changes; transient local, so a GUI started later gets the current state.

std_msgs/Header header

uint8 STATE_COLLECTING = 0  # no final pass: views are being collected
uint8 STATE_RUNNING = 1     # final pass in progress; stage says what it is doing
uint8 STATE_DONE = 2        # final pass finished: results below
uint8 STATE_FAILED = 3      # final pass failed; stage says why
uint8 STATE_ACCEPTED = 4    # results accepted by the user
uint8 state
string stage

# Increases with every final pass (cached views in the GUI are stale after a change).
uint32 run

# The views the pass started from (all kept views; indices below are 0-based into them, the GUI
# shows them as #1, #2, ...). Outline of view i, for display: the convex hull of its corners,
# outline_sizes[i] points, x and y interleaved in outline_points.
uint32 total_views
uint32 image_width
uint32 image_height
uint32[] outline_sizes
float32[] outline_points

# Calibrations from all views and from the kept ones.
CalibrationState initial
CalibrationState final

# Outlier check: a view whose leave-one-out error (how badly the other views predict it) has a
# robust z-score above z_limit is removed, unless removing it raised the mean coverage
# uncertainty over the image by more than outlier_max_uncertainty_increase (no other view covers
# its region): then it is a suspect.
bool checked                           # false: too few views for the check (outlier_min_views)
float32 z_limit
float32 limit                          # leave-one-out error with z = z_limit among all views, px
float32[] leave_one_out                # per view, from the calibration with all views, px
uint32[] removed                       # in removal order
float32[] removed_z
# Mean coverage uncertainty over the image (from independent corner errors, so it depends on where
# the views are) without the view / with it; inf if without it the lens model has no valid ray
# for part of the image (it folds over where no other view has corners).
float32[] removed_uncertainty_ratio
uint32[] suspects
float32[] suspects_z
float32[] suspects_uncertainty_ratio
uint32[] forced                        # restored by the user: never removed
