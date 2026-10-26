"""
Conversion of single-channel camera images to 8 bit, for detection and for display.

preprocess: contrast stretch between two intensity percentiles, gamma, then an optional S-shaped
contrast curve. All steps are monotonic, so they do not move chessboard corners.
"""

import numpy as np

# Encodings converted at their own bit depth: cv_bridge's mono8 conversion would leave
# 10/12-bit sensors nearly black.
SIXTEEN_BIT_ENCODINGS = {'mono16', '16UC1', 'bgr16', 'rgb16', 'bgra16', 'rgba16'}
FLOAT_MONO_ENCODINGS = {'32FC1', '64FC1'}

# Percentiles of large images are estimated on every n-th pixel in both directions.
PERCENTILE_SUBSAMPLE = 4
SUBSAMPLE_MIN_PIXELS = 100_000
HISTOGRAM_BINS = 256


def to_mono8(image):
    """Convert a single-channel image to uint8 scaled by its maximum; NaN, inf, < 0 map to 0."""
    if image.ndim != 2:
        raise ValueError(f'expected a single-channel image, got shape {image.shape}')
    if image.dtype == np.uint8:
        return image

    image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    max_value = float(image.max())
    if max_value <= 0.0:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.clip(np.rint(image * (255.0 / max_value)), 0, 255).astype(np.uint8)


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


def gamma_curve(values, low, high, gamma):
    """Map [low, high] linearly to [0, 1], then apply gamma (> 1 brightens dark regions)."""
    if high <= low:
        return np.zeros(np.shape(values), dtype=np.float32)
    normalized = np.clip((np.asarray(values, dtype=np.float32) - low) / (high - low), 0.0, 1.0)
    return normalized ** (1.0 / gamma)


def lower_half_s(t, strength):
    """Lower half of a tanh S-curve, scaled to map [0, 1] onto [0, 1]."""
    half_range = np.tanh(strength / 4.0)
    return (np.tanh(strength * (t - 1.0) / 4.0) + half_range) / half_range


def s_curve(values, strength, midpoint):
    """
    S-shaped contrast curve on 0-1 values that keeps 0, midpoint and 1 fixed.

    Values below the midpoint get darker, above it lighter; strength sets the steepness (0: off).
    """
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    if strength <= 0.0:
        return values.astype(np.float32)
    midpoint = min(max(midpoint, 0.01), 0.99)

    below = midpoint * lower_half_s(values / midpoint, strength)
    above = 1.0 - (1.0 - midpoint) * lower_half_s((1.0 - values) / (1.0 - midpoint), strength)
    return np.where(values < midpoint, below, above).astype(np.float32)


def to_uint8(values):
    return np.rint(255.0 * np.asarray(values)).astype(np.uint8)


def tone_curve(values, low, high, gamma, contrast_strength=0.0, contrast_midpoint=0.5):
    """Map intensities to 0-255: stretch, gamma, then the S-shaped contrast curve."""
    return to_uint8(s_curve(gamma_curve(values, low, high, gamma),
                            contrast_strength, contrast_midpoint))


def preprocess_with_histograms(image, low_percentile=0.5, high_percentile=99.5, gamma=1.0,
                               contrast_strength=0.0, contrast_midpoint=0.5):
    """
    Convert a single-channel image (uint8, uint16 or float) to the uint8 detection image.

    Returns (image, histogram before the contrast curve, histogram after it), 256 bins each.
    """
    if image.ndim != 2:
        raise ValueError(f'expected a single-channel image, got shape {image.shape}')
    low, high = stretch_limits(image, low_percentile, high_percentile)

    if image.dtype in (np.uint8, np.uint16):
        levels = np.arange(np.iinfo(image.dtype).max + 1)
        before = to_uint8(gamma_curve(levels, low, high, gamma))
        lut = tone_curve(levels, low, high, gamma, contrast_strength, contrast_midpoint)
        counts = np.bincount(image.ravel(), minlength=levels.size)
        return (lut[image],
                np.bincount(before, weights=counts, minlength=HISTOGRAM_BINS).astype(np.int64),
                np.bincount(lut, weights=counts, minlength=HISTOGRAM_BINS).astype(np.int64))

    image = np.nan_to_num(image.astype(np.float32), nan=low, posinf=low, neginf=low)
    before = gamma_curve(image, low, high, gamma)
    out = to_uint8(s_curve(before, contrast_strength, contrast_midpoint))
    return (out,
            np.bincount(to_uint8(before).ravel(), minlength=HISTOGRAM_BINS),
            np.bincount(out.ravel(), minlength=HISTOGRAM_BINS))


def preprocess(image, low_percentile=0.5, high_percentile=99.5, gamma=1.0,
               contrast_strength=0.0, contrast_midpoint=0.5):
    """Convert a single-channel image to the uint8 detection image."""
    if image.ndim != 2:
        raise ValueError(f'expected a single-channel image, got shape {image.shape}')
    low, high = stretch_limits(image, low_percentile, high_percentile)
    if image.dtype in (np.uint8, np.uint16):
        levels = np.arange(np.iinfo(image.dtype).max + 1)
        return tone_curve(levels, low, high, gamma, contrast_strength, contrast_midpoint)[image]
    image = np.nan_to_num(image.astype(np.float32), nan=low, posinf=low, neginf=low)
    return tone_curve(image, low, high, gamma, contrast_strength, contrast_midpoint)
