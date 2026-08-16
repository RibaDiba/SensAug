"""Tests for the correlation-matrix render layer.

Deliberately free of `pytest.importorskip("mmengine")`, unlike
test_grad_sens_analysis.py -- sensaug.hooks.corr_viz imports only numpy and
matplotlib, so these run on a laptop. That is the whole reason the render layer
is its own module: iterating on what the heatmap looks like should not cost a
SLURM round-trip.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensaug.hooks.corr_viz import (  # noqa: E402
    CORR_CMAP,
    MIDPOINT,
    dropped_ops,
    markdown_report,
    render_matrix,
)

NAMES = [f"op_{i}" for i in range(8)]


def _r(seed=0, n_ops=8, n_images=300):
    rng = np.random.default_rng(seed)
    return np.corrcoef(rng.normal(size=(n_ops, n_images)))


def _drop(r, index):
    r = r.copy()
    r[index, :] = np.nan
    r[:, index] = np.nan
    return r


# --- the colour map ----------------------------------------------------------


def test_midpoint_is_neutral_gray_not_white():
    """A white midpoint is indistinguishable from the figure surface, which is
    exactly the confusion the NaN masking exists to remove."""
    r, g, b = CORR_CMAP(0.5)[:3]
    assert max(r, g, b) - min(r, g, b) < 0.03, "midpoint must be neutral, not a hue"
    assert 0.88 < r < 0.98, f"midpoint should be ~{MIDPOINT}, got {(r, g, b)}"


def test_the_two_arms_are_lightness_symmetric():
    """An r of -0.6 and an r of +0.6 must read as equally strong. Interpolating
    the poles in sRGB does not give this; interpolating in OKLab does."""
    luminance = lambda c: 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    for offset in (0.1, 0.2, 0.3, 0.4):
        neg = luminance(CORR_CMAP(0.5 - offset)[:3])
        pos = luminance(CORR_CMAP(0.5 + offset)[:3])
        assert abs(neg - pos) < 0.06, f"arms differ at ±{offset}: {neg:.3f} vs {pos:.3f}"


def test_the_arms_are_two_hues_not_a_rainbow():
    """Diverging means two poles and a neutral between them. A ramp that wanders
    through extra hues encodes magnitude order that is not in the data."""
    hue_sign = lambda c: np.sign(c[0] - c[2])  # red-vs-blue dominance
    assert hue_sign(CORR_CMAP(0.0)[:3]) < 0, "negative pole should be blue-dominant"
    assert hue_sign(CORR_CMAP(1.0)[:3]) > 0, "positive pole should be red-dominant"
    # Dominance flips exactly once, at the midpoint.
    signs = [hue_sign(CORR_CMAP(t)[:3]) for t in np.linspace(0, 1, 64)]
    flips = sum(1 for a, b in zip(signs, signs[1:]) if a != b and a != 0 and b != 0)
    assert flips == 1, f"colour ramp crosses hues {flips} times, expected 1"


# --- the figure --------------------------------------------------------------


def test_render_returns_an_rgb_image():
    img = render_matrix(_r(), NAMES, "R")
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.dtype == np.uint8


def test_a_dropped_op_is_not_painted_like_an_uncorrelated_one():
    """The defect this module was written to fix.

    correlate() NaNs a dropped op's row precisely so it "can never be mistaken for
    an uncorrelated one"; rendering NaN as 0.0 hands that guarantee straight back.
    """
    r = _r()
    r[2, 5] = r[5, 2] = 0.0  # a genuine, measured zero
    dropped = _drop(r, 3)

    zero_rgb = np.array(CORR_CMAP(0.5)[:3])
    bad_rgb = np.array(CORR_CMAP.get_bad()[:3])
    assert not np.allclose(zero_rgb, bad_rgb, atol=0.015), (
        "a measured r=0 and an unmeasured cell render the same colour"
    )
    # And the figure actually differs from one where the row was genuinely zero.
    zeroed = r.copy()
    zeroed[3, :] = 0.0
    zeroed[:, 3] = 0.0
    assert not np.array_equal(
        render_matrix(dropped, NAMES, "R"), render_matrix(zeroed, NAMES, "R")
    )


def test_a_dropped_op_is_labelled_as_dropped():
    assert dropped_ops(_drop(_r(), 3)).tolist() == [3]
    assert dropped_ops(_r()).tolist() == []


def test_an_all_nan_matrix_still_renders():
    """Every op dropped is a legitimate (if useless) window -- it must not crash
    the emission, since this runs inside after_train_iter."""
    img = render_matrix(np.full((8, 8), np.nan), NAMES, "R")
    assert img.ndim == 3


def test_colour_scale_is_fixed_not_autoscaled():
    """Two checkpoints with different structure must not render identically.

    An autoscaled matrix normalises its own range away, so a window where every
    cell sits near zero looks exactly like one with real structure -- which
    destroys the checkpoint-to-checkpoint comparison the panel exists for.
    """
    weak = _r() * 0.05
    np.fill_diagonal(weak, 1.0)
    strong = _r()
    assert not np.array_equal(
        render_matrix(weak, NAMES, "R"), render_matrix(strong, NAMES, "R")
    )


def test_marking_changes_the_figure_and_absent_marks_are_allowed():
    r = _r()
    mark = np.zeros_like(r, dtype=bool)
    mark[1, 4] = mark[4, 1] = True
    assert not np.array_equal(
        render_matrix(r, NAMES, "R", mark=mark), render_matrix(r, NAMES, "R")
    )


def test_the_warning_banner_does_not_cover_the_title():
    """The banner is placed in figure coordinates, so without reserved space it
    lands on top of the title instead of above it."""
    r = _r()
    plain = render_matrix(r, NAMES, "R", subtitle="sub")
    warned = render_matrix(r, NAMES, "R", subtitle="sub", warn="do not act on this")
    assert warned.shape[0] > 0
    assert not np.array_equal(plain, warned)
    # The banner's row band must not be blank -- i.e. something was drawn there.
    top_band = warned[: int(warned.shape[0] * 0.04)]
    assert top_band.std() > 0


def test_render_does_not_leak_figures():
    """This runs once per emission for the life of a training run; a leaked
    figure per call is a slow memory leak inside after_train_iter."""
    import matplotlib.pyplot as plt

    before = len(plt.get_fignums())
    for _ in range(5):
        render_matrix(_r(), NAMES, "R")
    assert len(plt.get_fignums()) == before


# --- the markdown report -----------------------------------------------------


def test_report_lists_pairs_strongest_first():
    r = _r()
    r[0, 1] = r[1, 0] = 0.91
    r[2, 3] = r[3, 2] = -0.77
    text = markdown_report(NAMES, r, 0.5, 100, 500, 500, top_n=2)
    assert "`op_0` / `op_1`" in text
    assert text.index("`op_0` / `op_1`") < text.index("`op_2` / `op_3`")
    assert "+0.910" in text and "-0.770" in text


def test_report_survives_dropped_ops_and_missing_statistics():
    r = _drop(_r(), 3)
    text = markdown_report(
        NAMES, r, 0.25, 50, 500, 500,
        dropped_names=["op_3"],
        magnitude_source="sa_snapshot", magnitude_mode="mode",
        max_shared_loading=0.42,
    )
    assert "op_3" in text
    # No bootstrap was passed, so those columns are absent rather than faked.
    assert "q (BH-FDR)" not in text and "95% CI" not in text


def test_report_renders_a_nan_q_as_a_dash_not_the_word_nan():
    r = _r()
    q = np.full_like(r, np.nan)
    ci = np.full_like(r, np.nan)
    survives = np.zeros_like(r, dtype=bool)
    text = markdown_report(
        NAMES, r, 0.5, 100, 500, 500, ci_lo=ci, ci_hi=ci, q=q, survives=survives, top_n=3
    )
    assert "nan" not in text.lower()
    assert "—" in text


@pytest.mark.parametrize("n_ops", [2, 3, 14])
def test_render_handles_the_real_op_counts(n_ops):
    names = [f"op_{i}" for i in range(n_ops)]
    img = render_matrix(_r(n_ops=n_ops), names, "R", subtitle="x")
    assert img.ndim == 3 and img.shape[2] == 3
