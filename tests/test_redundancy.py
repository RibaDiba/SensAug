"""Tests for sensaug/redundancy.py -- red(a) and the max-entropy reweighting.

Deliberately free of mmseg, torch and the registries: this is the half of the
mechanism that must be checkable without a GPU, a checkpoint, or a training run,
and every claim below is independent of whether R means anything. If the
correlation measurement turns out to be confounded, these still hold; they are
about the transformation, not about the input.

No importorskip guard, unlike the rest of the suite -- if this file cannot run,
numpy is missing.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensaug.redundancy import (
    NONE_KEY,
    MODES,
    compute_red,
    ramp_lambda,
    reweight,
    summarise,
    within_op_pairs,
)

NAMES_14 = [
    "lighter_R", "darker_R", "lighter_G", "darker_G", "lighter_B", "darker_B",
    "lighter_H", "darker_H", "lighter_S", "darker_S", "lighter_V", "darker_V",
    "blur", "noise",
]  # fmt: skip

NAMES_32 = NAMES_14 + [
    "rotate_pos", "rotate_neg", "shear_x_pos", "shear_x_neg",
    "shear_y_pos", "shear_y_neg", "translate_x_pos", "translate_x_neg",
    "translate_y_pos", "translate_y_neg", "brightness_pos", "brightness_neg",
    "contrast_pos", "contrast_neg", "sharpness_pos", "sharpness_neg",
    "color_pos", "color_neg",
]  # fmt: skip


def _random_r(names, seed=0):
    """A symmetric correlation-shaped matrix with a unit diagonal."""
    rng = np.random.default_rng(seed)
    n = len(names)
    m = rng.normal(0, 0.3, size=(n, n))
    m = (m + m.T) / 2
    np.fill_diagonal(m, 1.0)
    return np.clip(m, -1.0, 1.0)


def _pdf(names, levels=(0.2, 0.5, 0.8), none_mass=None):
    """A generate_pdf_new-shaped pdf: (op, level) keys plus ("none", 0), summing
    to 1, with 1/(A+1) reserved for none exactly as loops.py does."""
    n = len(names)
    none_mass = 1.0 / (n + 1) if none_mass is None else none_mass
    per_entry = (1.0 - none_mass) / (n * len(levels))
    pdf = {(name, level): per_entry for name in names for level in levels}
    pdf[NONE_KEY] = none_mass
    return pdf


# --- within-op pair derivation ------------------------------------------------


def test_within_op_pairs_finds_six_in_the_legacy_vocabulary():
    """The RGB/HSV lighter/darker pairs, and nothing else -- blur and noise are
    unsigned and have no partner."""
    pairs = within_op_pairs(NAMES_14)
    assert len(pairs) == 6
    for i, j in pairs:
        assert NAMES_14[i].startswith("lighter_")
        assert NAMES_14[j].startswith("darker_")


def test_within_op_pairs_finds_fifteen_in_the_merged_vocabulary():
    """6 lighter/darker + 5 geometric _pos/_neg + 4 photometric _pos/_neg. 15 of
    the 496 off-diagonal cells, which is why this is a footnote rather than a
    headline: it moves the raw row-sum mean by about a tenth and the standard
    deviation barely at all."""
    assert len(within_op_pairs(NAMES_32)) == 15


def test_within_op_pairs_ignores_a_partnerless_name():
    """A vocabulary subset (--photometric-only, say) can contain one half of a pair
    without the other. Pairing it against a missing index would raise."""
    assert within_op_pairs(["lighter_R", "blur", "color_pos"]) == []


# --- compute_red --------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_red_is_standardized_in_every_mode(mode):
    """The premise that makes one lambda portable across runs and checkpoints. The
    raw row sums are not comparable -- on the logged matrices their standard
    deviations span 0.24 to 3.69 -- so lambda tuned on one would be meaningless on
    the next."""
    score = compute_red(_random_r(NAMES_32), NAMES_32, mode=mode)

    assert score.std.mean() == pytest.approx(0.0, abs=1e-9)
    assert score.std.std() == pytest.approx(1.0, rel=1e-6)


def test_red_ignores_the_diagonal():
    """r[a, a] is 1 by construction and identical for every op, so including it
    would add a constant -- harmless after standardization, but only by accident."""
    r = _random_r(NAMES_14)
    with_unit_diagonal = compute_red(r, NAMES_14)

    r_scaled = r.copy()
    np.fill_diagonal(r_scaled, 0.5)
    assert np.allclose(with_unit_diagonal.raw, compute_red(r_scaled, NAMES_14).raw)


def test_red_excludes_within_op_cells_when_asked():
    """An op's two directions must not count as evidence that the op is redundant
    with the bank."""
    r = _random_r(NAMES_32)
    masked = compute_red(r, NAMES_32, mask_within_op=True)
    unmasked = compute_red(r, NAMES_32, mask_within_op=False)

    assert not np.allclose(masked.raw, unmasked.raw)
    # blur and noise have no partner, so their rows are untouched either way.
    blur = NAMES_32.index("blur")
    assert masked.raw[blur] == pytest.approx(unmasked.raw[blur])


def test_red_applies_the_fdr_survivor_mask():
    """Cells that did not survive multiplicity correction at 496 simultaneous
    tests contribute nothing."""
    r = _random_r(NAMES_14)
    survivors = np.zeros_like(r, dtype=bool)
    # lighter_R/lighter_G, deliberately NOT a within-op pair -- picking one would
    # test nothing, since the within-op mask zeroes it before the survivor mask
    # is ever consulted.
    survivors[0, 2] = survivors[2, 0] = True

    score = compute_red(r, NAMES_14, survivor_mask=survivors)

    assert score.raw[0] > 0 and score.raw[2] > 0
    assert score.raw[[1, *range(3, 14)]].sum() == pytest.approx(0.0)


def test_the_within_op_mask_is_applied_before_the_survivor_mask():
    """Order matters and the composition is AND, not OR: a within-op cell that
    survives FDR must still be excluded. Surviving multiplicity correction says the
    correlation is real, which is exactly what a parameterization artefact looks
    like."""
    r = _random_r(NAMES_32)
    survivors = np.ones_like(r, dtype=bool)

    both = compute_red(r, NAMES_32, survivor_mask=survivors, mask_within_op=True)
    fdr_only = compute_red(r, NAMES_32, survivor_mask=survivors, mask_within_op=False)

    assert not np.allclose(both.raw, fdr_only.raw)
    assert np.allclose(both.raw, compute_red(r, NAMES_32, mask_within_op=True).raw)


def test_nan_cells_contribute_zero_rather_than_poisoning_the_row():
    """correlate() leaves a dropped op's row and column NaN by design, and the json
    log writes them as null. A single NaN propagating through the row sum would take
    out every op that op touches, i.e. all of them."""
    r = _random_r(NAMES_14)
    r[3, :] = np.nan
    r[:, 3] = np.nan

    score = compute_red(r, NAMES_14)

    assert np.isfinite(score.raw).all()
    assert np.isfinite(score.std).all()
    assert score.raw[3] == pytest.approx(0.0)


def test_a_fully_nan_row_is_reported_as_dropped():
    """Distinguishable from an op that genuinely correlates with nothing -- both
    score 0, but only one of them was measured."""
    r = _random_r(NAMES_14)
    r[5, :] = np.nan
    r[:, 5] = np.nan

    assert compute_red(r, NAMES_14).dropped == ["darker_B"]
    assert compute_red(_random_r(NAMES_14), NAMES_14).dropped == []


def test_squared_and_abs_are_sign_blind_but_signed_is_not():
    """Why `squared` is the default: it cannot be flipped by a sign-convention
    audit landing the other way, and it does not depend on how closure resolves."""
    r = _random_r(NAMES_14)
    flipped = -r.copy()
    np.fill_diagonal(flipped, 1.0)

    assert np.allclose(
        compute_red(r, NAMES_14, mode="squared").raw,
        compute_red(flipped, NAMES_14, mode="squared").raw,
    )
    assert np.allclose(
        compute_red(r, NAMES_14, mode="abs").raw,
        compute_red(flipped, NAMES_14, mode="abs").raw,
    )
    assert not np.allclose(
        compute_red(r, NAMES_14, mode="signed").raw,
        compute_red(flipped, NAMES_14, mode="signed").raw,
    )


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown red mode"):
        compute_red(_random_r(NAMES_14), NAMES_14, mode="cubed")


def test_mismatched_names_and_matrix_raise():
    """The one error that would otherwise be silent and catastrophic: scoring a
    32x32 matrix against 14 names would mislabel every op."""
    with pytest.raises(ValueError, match="14 names"):
        compute_red(_random_r(NAMES_32), NAMES_14)


# --- the degenerate guards ----------------------------------------------------


def test_an_all_zero_matrix_reports_a_reason_rather_than_a_silent_no_op():
    """The failure mode most likely to waste a week: reweighting runs, changes
    nothing, and the run looks like a real experimental arm."""
    score = compute_red(np.eye(len(NAMES_14)), NAMES_14)

    assert not score.usable
    assert "no cell survived" in score.reason


def test_zero_survivors_reports_a_reason():
    score = compute_red(
        _random_r(NAMES_14),
        NAMES_14,
        survivor_mask=np.zeros((len(NAMES_14),) * 2, dtype=bool),
    )
    assert not score.usable
    assert "FDR" in score.reason


def test_a_uniform_matrix_reports_zero_variance():
    """Every op equally redundant. Standardization would divide by ~0 and amplify
    float noise into a ranking that means nothing."""
    r = np.full((len(NAMES_14),) * 2, 0.4)
    np.fill_diagonal(r, 1.0)

    score = compute_red(r, NAMES_14, mask_within_op=False)

    assert not score.usable
    assert "standard deviation" in score.reason


def test_a_healthy_matrix_reports_no_reason():
    assert compute_red(_random_r(NAMES_32), NAMES_32).usable


# --- reweight: the seven claims from the design -------------------------------


def test_lambda_zero_is_bit_identical():
    """THE test. The lambda=0 arm is the control every other arm is compared
    against, so 'close to the baseline' is not good enough -- it has to BE the
    baseline. Exact equality, not approx."""
    pdf = _pdf(NAMES_32)
    red = compute_red(_random_r(NAMES_32), NAMES_32).as_dict()

    result = reweight(pdf, red, 0.0)

    assert result.pdf == pdf
    assert not result.applied


@pytest.mark.parametrize("lam", [0.1, 0.25, 0.5, 1.0, 10.0, 100.0])
def test_every_entry_stays_strictly_positive(lam):
    """Soft down-weighting only. exp() is strictly positive so this is structural
    rather than a clamp, but an overflow or an underflow to exactly 0 would break it
    in practice, which is what the 100.0 case is really probing."""
    pdf = _pdf(NAMES_32)
    rng = np.random.default_rng(7)
    adversarial = dict(zip(NAMES_32, rng.normal(0, 3.0, size=len(NAMES_32))))

    result = reweight(pdf, adversarial, lam)

    assert all(v > 0 for v in result.pdf.values()), f"an entry hit zero at lambda={lam}"
    assert all(np.isfinite(v) for v in result.pdf.values())


@pytest.mark.parametrize("lam", [0.1, 0.25, 0.5, 1.0, 5.0])
def test_the_result_is_still_a_distribution(lam):
    """RandomTrainTransformNew raises if the probabilities do not sum to 1 within
    1e-6, and it raises inside a dataloader worker where the traceback is close to
    unreadable."""
    pdf = _pdf(NAMES_32)
    red = compute_red(_random_r(NAMES_32), NAMES_32).as_dict()

    total = sum(reweight(pdf, red, lam).pdf.values())

    assert total == pytest.approx(1.0, abs=1e-9)


def test_the_ratio_between_two_ops_decreases_monotonically_in_lambda():
    """What 'lambda controls the strength' has to mean. Non-monotonicity would make
    the sweep in section 7.3 uninterpretable."""
    pdf = _pdf(NAMES_32)
    red = compute_red(_random_r(NAMES_32), NAMES_32).as_dict()
    more, less = max(red, key=red.get), min(red, key=red.get)

    ratios = []
    for lam in (0.1, 0.25, 0.5, 1.0, 2.0):
        out = reweight(pdf, red, lam).pdf
        ratios.append(out[(more, 0.5)] / out[(less, 0.5)])

    assert all(b < a for a, b in zip(ratios, ratios[1:])), ratios


def test_adding_a_constant_to_red_changes_nothing():
    """Encodes the standardization argument as an executable claim: exp(-lambda*c)
    is a constant factor that cancels in the normaliser, so only deviations from the
    mean can affect the output."""
    pdf = _pdf(NAMES_32)
    red = compute_red(_random_r(NAMES_32), NAMES_32).as_dict()
    shifted = {name: value + 5.0 for name, value in red.items()}

    base = reweight(pdf, red, 0.5).pdf
    moved = reweight(pdf, shifted, 0.5).pdf

    for key in base:
        assert base[key] == pytest.approx(moved[key], rel=1e-12)


def test_an_exact_duplicate_pair_is_downweighted_under_squared():
    """The behaviour the mechanism exists for. Two ops that are the same
    augmentation under two names should not jointly hold twice the sampling mass of
    a unique one."""
    names = ["a", "b", "c", "d", "e", "f"]
    r = np.eye(6) * 1.0
    r[0, 1] = r[1, 0] = 1.0  # exact alias

    red = compute_red(r, names, mask_within_op=False)
    out = reweight(_pdf(names), red.as_dict(), 0.5).pdf

    assert out[("a", 0.5)] < out[("c", 0.5)]
    assert out[("b", 0.5)] < out[("c", 0.5)]


def test_a_signed_anti_alias_is_protected_rather_than_downweighted():
    """Documented, not discovered. Under `signed` with within-op masking off, a
    perfectly ANTI-correlated pair receives the strongest protection in the matrix
    -- which is backwards, and is the reason `squared` is the default and the reason
    within-op pairs are excluded."""
    names = ["a", "b", "c", "d", "e", "f"]
    r = np.eye(6) * 1.0
    r[0, 1] = r[1, 0] = -1.0

    red = compute_red(r, names, mode="signed", mask_within_op=False)
    out = reweight(_pdf(names), red.as_dict(), 0.5).pdf

    assert out[("a", 0.5)] > out[("c", 0.5)], (
        "the anti-correlated pair was not protected -- if this now fails, the "
        "signed variant's known weakness has changed and the docs need revisiting"
    )


def test_the_same_inputs_give_the_same_output():
    pdf = _pdf(NAMES_32)
    red = compute_red(_random_r(NAMES_32, seed=3), NAMES_32).as_dict()

    assert reweight(pdf, red, 0.4).pdf == reweight(pdf, red, 0.4).pdf


# --- reweight: this repo's pdf shape ------------------------------------------


@pytest.mark.parametrize("lam", [0.0, 0.25, 1.0, 5.0])
def test_the_none_mass_is_invariant_in_lambda(lam):
    """Lever 3 changes WHICH augmentation is sampled, not HOW OFTEN augmentation
    happens. Letting exp(-lambda*red) touch ("none", 0) would fold a change in the
    augmentation rate into the same knob, and no experiment could separate them
    afterwards."""
    pdf = _pdf(NAMES_32)
    red = compute_red(_random_r(NAMES_32), NAMES_32).as_dict()

    out = reweight(pdf, red, lam).pdf

    assert out[NONE_KEY] == pdf[NONE_KEY]


def test_all_levels_of_one_op_are_scaled_by_the_same_factor():
    """red(a) is per-op. The beta-binomial shape generate_pdf_new gives an op's
    levels encodes which magnitudes hurt most and is a separate mechanism; only the
    mass allotted between ops may move."""
    levels = (0.2, 0.5, 0.8)
    pdf = _pdf(NAMES_32, levels=levels)
    red = compute_red(_random_r(NAMES_32), NAMES_32).as_dict()

    out = reweight(pdf, red, 0.5).pdf

    for name in NAMES_32:
        ratios = [out[(name, lv)] / pdf[(name, lv)] for lv in levels]
        assert ratios == pytest.approx([ratios[0]] * len(levels), rel=1e-12)


def test_ops_missing_from_red_are_left_alone_rather_than_raising():
    """The vocabulary in the pdf and the vocabulary R was measured over need not
    coincide -- an op can be dropped upstream, or simply absent. Scoring 0 is the
    standardized mean, i.e. no adjustment in either direction."""
    pdf = _pdf(NAMES_32)
    partial = {"blur": 2.0, "noise": -2.0}

    out = reweight(pdf, partial, 0.5).pdf

    assert out[("blur", 0.5)] < pdf[("blur", 0.5)]
    assert out[("noise", 0.5)] > pdf[("noise", 0.5)]
    assert sum(out.values()) == pytest.approx(1.0)


def test_a_red_key_absent_from_the_pdf_is_ignored():
    """The reverse mismatch: --no-inv-aug pops ops out of the pdf, and R still
    carries them."""
    pdf = _pdf(["blur", "noise"])
    red = {"blur": 1.0, "noise": -1.0, "lighter_R": 5.0}

    result = reweight(pdf, red, 0.5)

    assert result.applied
    assert set(result.pdf) == set(pdf)


def test_no_score_at_all_returns_the_pdf_with_a_reason():
    pdf = _pdf(NAMES_14)
    for empty in (None, {}):
        result = reweight(pdf, empty, 0.5)
        assert result.pdf == pdf
        assert not result.applied
        assert result.reason


def test_a_score_matching_nothing_in_the_pdf_returns_a_reason():
    """Exactly the disjoint-vocabulary failure: a "new"-keyed pdf against a
    diff-keyed score matches on zero ops. It has to say so rather than quietly
    behaving like lambda=0."""
    result = reweight(_pdf(NAMES_14), {"BrightnessTransform": 1.0}, 0.5)

    assert not result.applied
    assert "no pdf entry has a redundancy score" in result.reason


def test_the_reported_spread_matches_the_pdf_it_returns():
    """The number the training log prints, and the one the offline calibration
    picks lambda against -- it must describe the distribution actually produced."""
    pdf = _pdf(NAMES_32)
    red = compute_red(_random_r(NAMES_32), NAMES_32).as_dict()

    result = reweight(pdf, red, 0.5)
    free = [v for k, v in result.pdf.items() if k != NONE_KEY]

    assert result.spread == pytest.approx(max(free) / min(free))


# --- lambda ramping -----------------------------------------------------------


def test_the_linear_ramp_starts_at_zero_and_reaches_the_target():
    """Early R describes a model that barely discriminates between augmentations,
    so full strength from step 0 acts hardest on the least trustworthy
    measurement."""
    assert ramp_lambda(0.5, 0.0) == 0.0
    assert ramp_lambda(0.5, 0.5) == pytest.approx(0.25)
    assert ramp_lambda(0.5, 1.0) == pytest.approx(0.5)


def test_the_ramp_clamps_out_of_range_progress():
    assert ramp_lambda(0.5, 1.4) == pytest.approx(0.5)
    assert ramp_lambda(0.5, -0.2) == 0.0


def test_the_constant_ramp_is_the_ablation_arm():
    assert ramp_lambda(0.5, 0.1, mode="constant") == 0.5


def test_unknown_ramp_raises():
    with pytest.raises(ValueError, match="unknown lambda ramp"):
        ramp_lambda(0.5, 0.5, mode="cosine")


# --- the logged summary -------------------------------------------------------


def test_summarise_names_the_extremes_of_a_usable_score():
    score = compute_red(_random_r(NAMES_32), NAMES_32)
    line = summarise(score)

    assert "most redundant" in line and "least" in line
    assert str(len(NAMES_32)) in line


def test_summarise_reports_the_reason_when_the_score_is_unusable():
    line = summarise(compute_red(np.eye(len(NAMES_14)), NAMES_14))
    assert "unusable" in line
