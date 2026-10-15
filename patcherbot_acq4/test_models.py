"""Sensor dynamic range is part of the detector/policy input contract."""

import numpy as np
import pytest

from patcherbot_acq4.models import _camera_uint8


def test_16bit_sensor_requires_calibration_not_silent_wraparound():
    raw = np.array([[0, 1000, 2000, 4000, 65535]], dtype=np.uint16)
    with pytest.raises(ValueError, match='inputRange'):
        _camera_uint8(raw, None)
    np.testing.assert_array_equal(
        _camera_uint8(raw, (1000, 4000)),
        np.array([[0, 0, 85, 255, 255]], dtype=np.uint8),
    )
    calibrated = np.array([[11, 240]], dtype=np.uint8)
    assert _camera_uint8(calibrated, (1000, 4000)) is calibrated
