"""Tests for sensitivity analysis math functions in sensaug/sensitivity_analysis.py.

These tests cover objective_function, pchip_interpolator, and level math
without requiring full dataset loading or MMSeg runner models.
"""

import numpy as np
import pytest

pytest.importorskip("mmengine")
pytestmark = pytest.mark.requires_mmseg

from sensaug.sensitivity_analysis import objective_function, pchip_interpolator



def test_objective_function_at_zero_level():
    # At level=0, miou drop is 0, kid drop is 0
    # objective = (0 / max_miou) + 2*(0/max_level) - (0 / max_kid) = 0
    val = objective_function(
        level=0.0,
        max_level=1.0,
        miou=0.0,
        max_miou=0.5,
        kid=0.0,
        max_kid=1.0,
    )
    assert np.isclose(val, 0.0)


def test_objective_function_at_max_level():
    # At max level, miou drop = max_miou, kid drop = max_kid
    # objective = 1.0 + 2.0 - 1.0 = 2.0
    val = objective_function(
        level=1.0,
        max_level=1.0,
        miou=0.5,
        max_miou=0.5,
        kid=1.0,
        max_kid=1.0,
    )
    assert np.isclose(val, 2.0)


def test_pchip_interpolator_monotonic_points():
    points = np.array([
        [0.0, 0.0],
        [0.5, 0.8],
        [1.0, 2.0],
    ])
    targets = np.array([0.4, 1.2])

    values, errors = pchip_interpolator(points, targets)

    assert len(values) == 2
    assert len(errors) == 2
    # Interpolated level x for y=0.4 should be between 0.0 and 0.5
    assert 0.0 < values[0] < 0.5
    # Interpolated level x for y=1.2 should be between 0.5 and 1.0
    assert 0.5 < values[1] < 1.0
    # Errors should be non-negative
    assert np.all(errors >= 0.0)


def test_pchip_interpolator_exact_point_match():
    points = np.array([
        [0.0, 0.0],
        [0.5, 1.0],
        [1.0, 2.0],
    ])
    targets = np.array([1.0])

    values, errors = pchip_interpolator(points, targets)

    assert np.isclose(values[0], 0.5)
    assert np.isclose(errors[0], 0.0)
