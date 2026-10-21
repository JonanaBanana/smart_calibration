import numpy as np
import pytest

from smart_calibration.preprocessing import preprocess, tone_curve


def test_full_range_stretch_maps_min_and_max_to_0_and_255():
    image = np.array([[100, 150, 200]], dtype=np.uint8)
    np.testing.assert_array_equal(preprocess(image, 0, 100), [[0, 128, 255]])


def test_12bit_in_uint16_with_hot_pixel_keeps_contrast():
    rng = np.random.default_rng(0)
    image = rng.integers(1000, 3000, (64, 64)).astype(np.uint16)
    image[0, 0] = 65535  # a single hot pixel must not darken the rest
    out = preprocess(image, 0.5, 99.5)
    assert out.dtype == np.uint8
    assert out[1:, 1:].max() >= 250 and out[1:, 1:].min() <= 5


def test_gamma_above_one_brightens_mid_tones():
    image = np.array([[0, 128, 255]], dtype=np.uint8)
    neutral = preprocess(image, 0, 100, gamma=1.0)
    bright = preprocess(image, 0, 100, gamma=2.0)
    assert bright[0, 1] > neutral[0, 1]
    assert bright[0, 0] == 0 and bright[0, 2] == 255


def test_tone_curve_is_monotonic():
    out = tone_curve(np.arange(256), 20.0, 230.0, 0.6)
    assert np.all(np.diff(out.astype(int)) >= 0)


def test_float_image_with_nan():
    image = np.array([[np.nan, 0.0, 0.5, 1.0]], dtype=np.float32)
    out = preprocess(image, 0, 100)
    np.testing.assert_array_equal(out, [[0, 0, 128, 255]])


def test_constant_image_gives_zeros():
    out = preprocess(np.full((4, 4), 7, np.uint16), 0, 100)
    np.testing.assert_array_equal(out, 0)


def test_multi_channel_is_rejected():
    with pytest.raises(ValueError):
        preprocess(np.zeros((2, 2, 3), np.uint8))
