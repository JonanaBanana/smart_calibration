import numpy as np
import pytest

from smart_calibration.preprocessing import (
    preprocess, preprocess_with_histograms, s_curve, tone_curve)


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


def test_s_curve_keeps_ends_and_midpoint_and_increases_contrast():
    x = np.linspace(0, 1, 101)
    y = s_curve(x, 10.0, 0.4)
    np.testing.assert_allclose([y[0], y[-1]], [0.0, 1.0], atol=1e-6)
    assert np.all(np.diff(y) > 0)                    # monotonic
    assert y[20] < x[20] and y[80] > x[80]           # darker below, lighter above the midpoint
    for midpoint in (0.2, 0.4, 0.5, 0.7):              # midpoint stays exactly in place
        assert abs(float(s_curve(midpoint, 10.0, midpoint)) - midpoint) < 1e-5


def test_s_curve_strength_zero_is_identity():
    x = np.linspace(0, 1, 11)
    np.testing.assert_allclose(s_curve(x, 0.0, 0.3), x)


def test_histograms_match_image_and_pixel_count():
    rng = np.random.default_rng(1)
    for image in (rng.integers(0, 256, (50, 60)).astype(np.uint8),
                  rng.integers(0, 4096, (50, 60)).astype(np.uint16),
                  rng.random((50, 60)).astype(np.float32)):
        out, before, after = preprocess_with_histograms(image, 1, 99, 0.6, 8.0, 0.45)
        np.testing.assert_array_equal(out, preprocess(image, 1, 99, 0.6, 8.0, 0.45))
        assert before.shape == after.shape == (256,)
        assert before.sum() == after.sum() == image.size
        np.testing.assert_array_equal(after, np.bincount(out.ravel(), minlength=256))


def test_s_curve_is_mirror_symmetric_in_the_midpoint():
    x = np.linspace(0, 1, 201)
    for midpoint in (0.05, 0.2, 0.35, 0.8, 0.95):
        np.testing.assert_allclose(s_curve(1 - x, 8.0, 1 - midpoint),
                                   1 - s_curve(x, 8.0, midpoint), atol=1e-5)


def test_s_curve_does_not_collapse_near_the_ends():
    # Regression: with a power-function warp, midpoints close to 1 sent all lower values to 0.
    x = np.linspace(0, 1, 1001)
    for midpoint in (0.02, 0.5, 0.85, 0.95, 0.99):
        y = s_curve(x, 12.0, midpoint)
        assert np.all(np.diff(y) > 0), midpoint            # strictly increasing: nothing merged
        assert abs(float(s_curve(midpoint, 12.0, midpoint)) - midpoint) < 1e-5


def test_s_curve_slope_at_midpoint_depends_on_strength_only():
    step = 1e-4
    for strength in (2.0, 8.0, 20.0):
        expected = strength / (4.0 * np.tanh(strength / 4.0))
        for midpoint in (0.1, 0.5, 0.85):
            low, high = s_curve([midpoint - step, midpoint + step], strength, midpoint)
            assert abs((high - low) / (2 * step) - expected) < 1e-2 * expected
