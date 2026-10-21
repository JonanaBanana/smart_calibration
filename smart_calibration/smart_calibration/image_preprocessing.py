"""Conversion of camera images into the 8-bit grayscale format used for target detection."""

import numpy as np

# Encodings deeper than 8 bit. cv_bridge's own mono8 conversion of these divides by the type's
# range, which leaves 10/12-bit sensors nearly black, so they go through to_mono8 instead:
# 16-bit (mono or color) via cv_bridge's mono16 conversion, float as-is.
SIXTEEN_BIT_ENCODINGS = {'mono16', '16UC1', 'bgr16', 'rgb16', 'bgra16', 'rgba16'}
FLOAT_MONO_ENCODINGS = {'32FC1', '64FC1'}


def to_mono8(image):
    """
    Convert a single-channel image of any numeric type to uint8, scaled by its maximum value.

    Scaling by the actual maximum instead of the type's full range (e.g. dividing 16-bit images
    by 256) keeps the contrast of 10/12-bit sensors that publish 16-bit images. NaN/inf values
    (float depth-like images) and negative values map to 0.
    """
    if image.ndim != 2:
        raise ValueError(f'expected a single-channel image, got shape {image.shape}')
    if image.dtype == np.uint8:
        return image

    image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    max_value = float(image.max())
    if max_value <= 0.0:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.clip(np.rint(image * (255.0 / max_value)), 0, 255).astype(np.uint8)
