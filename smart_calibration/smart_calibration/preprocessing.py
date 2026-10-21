"""Preprocessing of single-channel camera images into the 8-bit image used for detection."""

import numpy as np

# For large images, percentiles are estimated on every n-th pixel in both directions, which is
# plenty for an intensity distribution and keeps the cost at a fraction of a millisecond.
PERCENTILE_SUBSAMPLE = 4
SUBSAMPLE_MIN_PIXELS = 100_000


def stretch_limits(image, low_percentile, high_percentile):
    """Return the intensities at the given percentiles (0-100) of a single-channel image."""
    sample = image
    if image.size >= SUBSAMPLE_MIN_PIXELS:
        sample = image[::PERCENTILE_SUBSAMPLE, ::PERCENTILE_SUBSAMPLE]
    if sample.dtype.kind == 'f':
        sample = sample[np.isfinite(sample)]
        if sample.size == 0:
            return 0.0, 0.0
    low, high = np.percentile(sample, (low_percentile, high_percentile))
    return float(low), float(high)


def tone_curve(values, low, high, gamma):
    """
    Map intensities to 0-255: linear stretch of [low, high] to [0, 1], then gamma.

    gamma > 1 brightens dark regions, gamma < 1 darkens bright ones (output = input^(1/gamma)).
    Both steps are monotonic per pixel, so they do not move chessboard corners (saddle points).
    """
    if high <= low:
        return np.zeros(np.shape(values), dtype=np.uint8)
    normalized = np.clip((np.asarray(values, dtype=np.float32) - low) / (high - low), 0.0, 1.0)
    return np.rint(255.0 * normalized ** (1.0 / gamma)).astype(np.uint8)


def preprocess(image, low_percentile=0.5, high_percentile=99.5, gamma=1.0):
    """
    Convert a single-channel image (uint8, uint16 or float) to the uint8 detection image.

    Stretches the contrast between the given intensity percentiles (robust against hot pixels,
    unlike scaling by the maximum), then applies gamma. Integer images use one lookup table for
    both steps; NaN/inf in float images map to 0.
    """
    if image.ndim != 2:
        raise ValueError(f'expected a single-channel image, got shape {image.shape}')
    low, high = stretch_limits(image, low_percentile, high_percentile)

    if image.dtype in (np.uint8, np.uint16):
        lut = tone_curve(np.arange(np.iinfo(image.dtype).max + 1), low, high, gamma)
        return lut[image]

    image = np.nan_to_num(image.astype(np.float32), nan=low, posinf=low, neginf=low)
    return tone_curve(image, low, high, gamma)
