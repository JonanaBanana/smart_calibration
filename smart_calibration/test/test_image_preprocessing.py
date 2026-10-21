import numpy as np
import pytest

from smart_calibration.image_preprocessing import to_mono8


def test_uint8_is_returned_unchanged():
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert to_mono8(image) is image


def test_12bit_in_uint16_uses_full_8bit_range():
    image = np.array([[0, 2048, 4095]], dtype=np.uint16)
    np.testing.assert_array_equal(to_mono8(image), [[0, 128, 255]])


def test_float_with_nan_and_negative_values():
    image = np.array([[np.nan, -1.0, 0.5, 1.0, np.inf]], dtype=np.float32)
    np.testing.assert_array_equal(to_mono8(image), [[0, 0, 128, 255, 0]])


def test_all_zero_image():
    image = np.zeros((2, 2), dtype=np.uint16)
    out = to_mono8(image)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, 0)


def test_multi_channel_is_rejected():
    with pytest.raises(ValueError):
        to_mono8(np.zeros((2, 2, 3), dtype=np.uint16))
