"""
Preprocessing of single-channel camera images into the 8-bit image used for detection.

Tone mapping: contrast stretch between two intensity percentiles, then gamma, then an optional
S-shaped contrast curve. All steps are monotonic per pixel, so they do not move chessboard
corners (saddle points of the intensity); a very strong S-curve approaches a hard threshold,
though, which removes the grey transitions that sub-pixel corner positions are computed from.
"""

import numpy as np

# For large images, percentiles are estimated on every n-th pixel in both directions, which is
# plenty for an intensity distribution and keeps the cost at a fraction of a millisecond.
PERCENTILE_SUBSAMPLE = 4
SUBSAMPLE_MIN_PIXELS = 100_000
HISTOGRAM_BINS = 256


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
    """
    Map intensities to 0-1: linear stretch of [low, high] to [0, 1], then gamma.

    gamma > 1 brightens dark regions, gamma < 1 darkens bright ones (output = input^(1/gamma)).
    """
    if high <= low:
        return np.zeros(np.shape(values), dtype=np.float32)
    normalized = np.clip((np.asarray(values, dtype=np.float32) - low) / (high - low), 0.0, 1.0)
    return normalized ** (1.0 / gamma)


def lower_half_s(t, strength):
    """
    Lower half of a symmetric S-curve, scaled to map [0, 1] onto [0, 1].

    The S-curve is a sigmoid around 0.5 (tanh form of the logistic), rescaled to keep 0 and 1;
    its lower half (convex, slope strength / (4 tanh(strength / 4)) at t = 1) is returned.
    """
    half_range = np.tanh(strength / 4.0)
    return (np.tanh(strength * (t - 1.0) / 4.0) + half_range) / half_range


def s_curve(values, strength, midpoint):
    """
    S-shaped contrast curve on 0-1 values that keeps 0, midpoint and 1 fixed.

    Values below midpoint get darker, values above it lighter; strength sets how steep the
    curve is (0: unchanged). Below the midpoint, the lower half of a symmetric S-curve is
    scaled onto [0, midpoint]; above it, the mirrored half onto [midpoint, 1]. Both sides are
    scaled equally in x and y, so the slope at the midpoint depends on strength only, and the
    curve for midpoint m is the mirror image of the one for 1 - m.
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

    Returns (image, histogram_before, histogram_after): 256-bin histograms (over 0-1) of the
    image before the contrast curve (after stretch and gamma) and of the final image. Integer
    images use one lookup table, and their histograms are derived from the input histogram, so
    no extra pass over the image is needed. NaN/inf in float images map to 0.
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
    """Convert a single-channel image to the uint8 detection image (see above)."""
    if image.ndim != 2:
        raise ValueError(f'expected a single-channel image, got shape {image.shape}')
    low, high = stretch_limits(image, low_percentile, high_percentile)
    if image.dtype in (np.uint8, np.uint16):
        levels = np.arange(np.iinfo(image.dtype).max + 1)
        return tone_curve(levels, low, high, gamma, contrast_strength, contrast_midpoint)[image]
    image = np.nan_to_num(image.astype(np.float32), nan=low, posinf=low, neginf=low)
    return tone_curve(image, low, high, gamma, contrast_strength, contrast_midpoint)
