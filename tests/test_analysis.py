"""Tests for correlation and analysis math functions in sensaug/hooks/grad_sens_analysis.py.

These tests use synthetic gradient matrices to test correlation, shared factor
loadings, closure null calculations, and Benjamini-Hochberg FDR corrections
without needing any deep learning dependencies.
"""

import numpy as np
import pytest

pytest.importorskip("statsmodels")

from sensaug.hooks.grad_sens_analysis import (

    closure_null,
    correlate,
    fdr_correct,
    shared_factor_loadings,
    shared_image_factor,
)


def test_correlate_identity_for_identical_rows():
    # 2 ops x 100 images; row 0 and row 1 are identical
    rng = np.random.default_rng(42)
    row = rng.normal(size=100)
    d_grad = np.vstack([row, row])

    r, dropped = correlate(d_grad)
    assert len(dropped) == 0
    assert np.isclose(r[0, 1], 1.0)
    assert np.isclose(r[1, 0], 1.0)


def test_correlate_drops_zero_variance_op():
    rng = np.random.default_rng(42)
    op1 = rng.normal(size=50)
    op2 = np.zeros(50)  # zero variance op
    d_grad = np.vstack([op1, op2])

    r, dropped = correlate(d_grad)
    assert list(dropped) == [1]
    assert np.isnan(r[0, 1])
    assert np.isnan(r[1, 1])


def test_shared_image_factor_computation():
    # 3 ops, 4 images
    d_grad = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [-1.0, -2.0, -3.0, -4.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    shared = shared_image_factor(d_grad)
    # mean abs per column: col0=(1+1+0)/3 = 2/3, col1=(2+2+0)/3 = 4/3, etc.
    expected = np.array([2/3, 4/3, 6/3, 8/3])
    assert np.allclose(shared, expected)


def test_shared_factor_loadings():
    # High loading scenario: op values scale directly with shared difficulty
    shared = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    d_grad = np.array([
        [2.0, 4.0, 6.0, 8.0, 10.0],  # perfectly correlated
        [-2.0, -4.0, -6.0, -8.0, -10.0],  # perfectly anti-correlated
    ])
    loadings = shared_factor_loadings(d_grad, shared)
    assert np.isclose(loadings[0], 1.0)
    assert np.isclose(loadings[1], -1.0)


def test_closure_null():
    assert closure_null(1) == 0.0
    assert np.isclose(closure_null(14), -1.0 / 13.0)
    assert np.isclose(closure_null(2), -1.0)


def test_fdr_correct_benjamini_hochberg():
    # p-values: some strong signals, some noise
    p_values = np.array([[0.001, 0.002], [0.4, 0.9]])
    q_matrix, is_sig = fdr_correct(p_values, alpha=0.05)

    assert q_matrix.shape == (2, 2)
    assert is_sig[0, 0] is np.bool_(True)
    assert is_sig[0, 1] is np.bool_(True)
    assert is_sig[1, 1] is np.bool_(False)
