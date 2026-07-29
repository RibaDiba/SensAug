"""Tests for the SA -> probe magnitude handoff.

Unlike the other test modules here, this one needs NOTHING from the OpenMMLab
stack (sensaug.corr_magnitudes is deliberately dependency-free), so it runs on a
laptop as well as on a compute node.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensaug.corr_magnitudes import (
    LEVEL_JITTER_STD,
    MODE_FIXED,
    MODE_MODAL,
    MODE_SAMPLED_INDEPENDENT,
    MODE_SAMPLED_SHARED,
    conditional_levels,
    modal_magnitude,
    resolve_magnitudes,
    sample_magnitudes,
)

OPS = ["lighter_R", "blur", "noise"]


def make_pdf():
    """A joint pdf shaped like RobustValLoop.generate_pdf_new's output: values sum
    to 1 across ALL ops together, plus the synthetic ("none", 0) entry."""
    return {
        ("lighter_R", 0.1): 0.20,
        ("lighter_R", 0.5): 0.10,
        ("lighter_R", 0.9): 0.05,
        ("blur", 0.2): 0.05,
        ("blur", 0.8): 0.30,
        ("none", 0): 0.30,
    }


# --- conditional_levels -------------------------------------------------------


def test_renormalizes_per_op_not_globally():
    """The pdf is a JOINT over (op, level). Using an op's raw values as its level
    distribution would leave them summing to well under 1, so every draw would be
    silently biased toward whichever level np.random.choice normalizes into."""
    snapshot = conditional_levels(make_pdf())

    for op, entry in snapshot.items():
        assert np.isclose(sum(entry["probs"]), 1.0), op

    # lighter_R's raw mass is 0.35; its levels must be renormalized by that, not
    # by the global 1.0.
    assert np.allclose(snapshot["lighter_R"]["probs"], [0.20, 0.10, 0.05] / np.float64(0.35))


def test_drops_the_none_entry():
    """("none", 0) is training's probability of applying no augmentation at all.
    Carried through it would become a magnitude-0 probe for a nonexistent op."""
    assert "none" not in conditional_levels(make_pdf())


def test_levels_are_sorted():
    pdf = {("blur", 0.9): 0.3, ("blur", 0.1): 0.3, ("blur", 0.5): 0.4}
    assert conditional_levels(pdf)["blur"]["levels"] == [0.1, 0.5, 0.9]


def test_op_names_filter():
    snapshot = conditional_levels(make_pdf(), op_names={"blur"})
    assert set(snapshot) == {"blur"}


def test_all_zero_mass_op_becomes_uniform():
    """Renormalizing by a zero total would be NaN, and NaN magnitudes propagate
    into every gradient for that op without any error being raised."""
    entry = conditional_levels({("blur", 0.2): 0.0, ("blur", 0.8): 0.0})["blur"]
    assert np.allclose(entry["probs"], [0.5, 0.5])


# --- modal_magnitude ----------------------------------------------------------


def test_modal_picks_the_highest_probability_level():
    snapshot = conditional_levels(make_pdf())
    assert modal_magnitude(snapshot["lighter_R"]) == 0.1  # mass 0.20, the largest
    assert modal_magnitude(snapshot["blur"]) == 0.8  # mass 0.30, the largest


# --- resolve_magnitudes -------------------------------------------------------


def test_modal_mode_is_constant_across_the_batch():
    """The default mode must not vary within a batch: a per-image magnitude adds a
    second source of variance to every row of D_grad, and R would then mix
    'which images are hard' with 'which magnitudes were drawn'."""
    got = resolve_magnitudes(
        conditional_levels(make_pdf()), OPS, 8, MODE_MODAL, fallback=0.5
    )
    assert np.all(got["lighter_R"] == 0.1)
    assert np.all(got["blur"] == 0.8)
    assert got["lighter_R"].shape == (8,)


def test_ops_missing_from_the_snapshot_fall_back():
    """`noise` is absent from the pdf. It must still be probed -- dropping it would
    silently shrink R from 14 ops to 13 with nothing in the log to say so."""
    got = resolve_magnitudes(
        conditional_levels(make_pdf()), OPS, 4, MODE_MODAL, fallback=0.5
    )
    assert np.all(got["noise"] == 0.5)
    assert set(got) == set(OPS)


def test_fixed_mode_ignores_the_snapshot():
    """The regression guard: `fixed` must reproduce the pre-snapshot behaviour
    exactly, so an old run can still be reproduced."""
    got = resolve_magnitudes(
        conditional_levels(make_pdf()), OPS, 4, MODE_FIXED, fallback=0.5
    )
    assert all(np.all(v == 0.5) for v in got.values())


def test_empty_snapshot_falls_back_everywhere():
    """Before the first post-warmup SA round, and for the whole run when SA is
    disabled, there is no snapshot at all."""
    got = resolve_magnitudes({}, OPS, 4, MODE_MODAL, fallback=0.5)
    assert all(np.all(v == 0.5) for v in got.values())


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown magnitude mode"):
        resolve_magnitudes({}, OPS, 4, "argmax", fallback=0.5)


def test_sampled_shared_uses_common_random_numbers():
    """The defining property of `sampled_shared`: image i is at the same PERCENTILE
    of every op's level distribution. Independent draws per op would inject
    row-uncorrelated noise and attenuate every cell of R toward zero."""
    # Two ops with identical distributions must then receive identical magnitudes.
    pdf = {
        ("a", 0.1): 0.25, ("a", 0.9): 0.25,
        ("b", 0.1): 0.25, ("b", 0.9): 0.25,
    }
    snapshot = conditional_levels(pdf)
    got = resolve_magnitudes(
        snapshot, ["a", "b"], 32, MODE_SAMPLED_SHARED, 0.5,
        rng=np.random.default_rng(0),
    )
    assert np.array_equal(got["a"], got["b"])
    # ...and it must actually vary across images, or the shared draw has collapsed.
    assert got["a"].std() > 0


def test_sampled_independent_decouples_identical_ops():
    """The contrast case. Same two identical distributions, independent draws ->
    the magnitudes must differ, which is exactly the noise that attenuates R."""
    pdf = {
        ("a", 0.1): 0.25, ("a", 0.9): 0.25,
        ("b", 0.1): 0.25, ("b", 0.9): 0.25,
    }
    got = resolve_magnitudes(
        conditional_levels(pdf), ["a", "b"], 64, MODE_SAMPLED_INDEPENDENT, 0.5,
        rng=np.random.default_rng(0),
    )
    assert not np.array_equal(got["a"], got["b"])


def test_sampled_magnitudes_stay_in_range():
    """The jitter can push a level past a rail; the diff ops' magnitude contract is
    [0, 1] and a magnitude outside it is not meaningful for any of them."""
    entry = {"levels": [0.0, 1.0], "probs": [0.5, 0.5]}
    drawn = sample_magnitudes(entry, 2000, rng=np.random.default_rng(0))
    assert drawn.min() >= 0.0 and drawn.max() <= 1.0


def test_sampling_reproduces_the_training_jitter():
    """Magnitudes should be the ones training actually applies, which means the
    N(0, 0.1) jitter RandomTrainTransformNew adds -- not just the pdf's support."""
    entry = {"levels": [0.5], "probs": [1.0]}
    drawn = sample_magnitudes(entry, 4000, rng=np.random.default_rng(0))
    assert drawn.std() == pytest.approx(LEVEL_JITTER_STD, rel=0.1)
    # Without the jitter it would be exactly 0.5 everywhere.
    assert sample_magnitudes(entry, 10, rng=np.random.default_rng(0), jitter_std=0).tolist() == [0.5] * 10


def test_shared_quantiles_respect_each_ops_own_distribution():
    """Shared quantile does NOT mean shared magnitude: ops with differently shaped
    pdfs must still get their own levels out of the same percentile."""
    snapshot = conditional_levels(
        {("a", 0.1): 0.5, ("a", 0.2): 0.5, ("b", 0.7): 0.5, ("b", 0.8): 0.5}
    )
    got = resolve_magnitudes(
        snapshot, ["a", "b"], 16, MODE_SAMPLED_SHARED, 0.5,
        rng=np.random.default_rng(1),
    )
    assert got["a"].max() < 0.5 < got["b"].min()
