"""Tests for the redundancy down-weighting registry in loops/grad_corr_loop.py.

`--corr-downweight-method` picks the function that turns the published redundancy
score into the reweighted training pdf. The registry is module-level and takes no
runner, so everything below runs without a GPU, a checkpoint, or a training loop
-- but the module pulls in the mmseg/mmengine registries to declare the loop, so
unlike test_redundancy.py this file needs the `sensaug` env.

Two claims carry the weight here:

* `soft-weighting` is BIT-IDENTICAL to the pre-registry behaviour, so every result
  measured before this flag existed remains comparable to one measured after it;
* an unnamed or unknown method is an error, never a silent fallback -- `none`
  included. A run whose arm cannot be recovered afterwards is worse than a run
  that refused to start.

The contract tests below are parametrized over the WHOLE registry rather than
listing arms, so a method added to `DOWNWEIGHT_METHODS` is held to them the moment
it is added, without anyone remembering to come back here.
"""

import os
import sys

import numpy as np
import pytest

pytest.importorskip("mmengine")
pytest.importorskip("mmseg")
pytestmark = pytest.mark.requires_mmseg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensaug.loops.grad_corr_loop import (
    DOWNWEIGHT_METHODS,
    HARD_PRUNING_METHODS,
    resolve_downweight_method,
)
from sensaug.redundancy import NONE_KEY, compute_red, reweight

NAMES = [
    "lighter_R", "darker_R", "lighter_G", "darker_G", "lighter_B", "darker_B",
    "lighter_H", "darker_H", "lighter_S", "darker_S", "lighter_V", "darker_V",
    "blur", "noise",
]  # fmt: skip


def _random_r(seed=0):
    """A symmetric correlation-shaped matrix with a unit diagonal."""
    rng = np.random.default_rng(seed)
    n = len(NAMES)
    m = rng.normal(0, 0.3, size=(n, n))
    m = (m + m.T) / 2
    np.fill_diagonal(m, 1.0)
    return np.clip(m, -1.0, 1.0)


def _pdf(levels=(0.2, 0.5, 0.8)):
    """A generate_pdf_new-shaped pdf: (op, level) keys plus ("none", 0)."""
    n = len(NAMES)
    none_mass = 1.0 / (n + 1)
    per_entry = (1.0 - none_mass) / (n * len(levels))
    pdf = {(name, level): per_entry for name in NAMES for level in levels}
    pdf[NONE_KEY] = none_mass
    return pdf


def _published(seed=0, r=None):
    """What PerturbationSensitivityAnalysisHookWithGradients.prune_augmentations
    puts on `runner.corr_redundancy`. Methods are handed this record whole, not
    just its "red" field, so one built on the structure of R rather than on a
    per-op summary can be added without changing the contract -- which is exactly
    what "r"/"names"/"mask_within_op" below are for, and what 'mRMR' reads."""
    r = _random_r(seed) if r is None else np.asarray(r, dtype=np.float64)
    score = compute_red(r, NAMES)
    return {
        "iter": 8000,
        "checkpoint": 0.4,
        "mode": score.mode,
        "fdr_gated": False,
        "red": score.as_dict(),
        "raw": {n: float(v) for n, v in zip(score.names, score.raw)},
        "dropped": score.dropped,
        "names": list(NAMES),
        "mask_within_op": True,
        "r": r,
        "survives": None,
    }


def _ops_with_mass(pdf):
    """The op names still reachable by sampling. Hard pruning is only visible at
    this granularity -- an op is removed by zeroing all of its levels at once."""
    return {op for (op, _level), p in pdf.items() if p > 0}


# --- the registry -------------------------------------------------------------


def test_every_registered_method_is_callable():
    assert DOWNWEIGHT_METHODS, "the registry must not be empty"
    for name, fn in DOWNWEIGHT_METHODS.items():
        assert callable(fn), f"{name} is not callable"


def test_resolve_returns_the_registered_callable():
    for name, fn in DOWNWEIGHT_METHODS.items():
        assert resolve_downweight_method(name) is fn


def test_resolve_rejects_an_unknown_name_and_names_the_valid_ones():
    """The error has to be actionable at 3am on a compute node: a bare KeyError
    would not say what IS accepted."""
    with pytest.raises(ValueError) as excinfo:
        resolve_downweight_method("maxent")
    message = str(excinfo.value)
    assert "maxent" in message
    for name in DOWNWEIGHT_METHODS:
        assert name in message


def test_resolve_rejects_the_old_exp_name():
    """"exp" was the registry's only entry for one commit before it was renamed to
    "soft-weighting". Aliasing it would let an old launch command keep working
    while the logs and the flag disagree about what the arm is called."""
    with pytest.raises(ValueError):
        resolve_downweight_method("exp")


def test_resolve_rejects_none_rather_than_defaulting():
    """There is no default arm -- not even "none". `None` reaching here means a
    config was built without stating the method, and quietly picking one would
    make the run's arm unrecoverable from the logs afterwards. Note this is the
    Python `None`, not the "none" METHOD, which is a deliberate choice someone
    made and is perfectly valid."""
    with pytest.raises(ValueError):
        resolve_downweight_method(None)


# --- the 'none' arm -----------------------------------------------------------


@pytest.mark.parametrize("lam", [0.1, 0.5, 1.0])
def test_none_ignores_lambda_entirely(lam):
    """Not "weakly affected by lambda" -- unaffected. 'none' is the arm every
    other method is read against, so if a lambda could leak into it the
    comparison would have no fixed end."""
    pdf = _pdf()
    assert DOWNWEIGHT_METHODS["none"](pdf, _published(), lam).pdf == pdf


def test_none_reports_itself_rather_than_looking_like_a_failure():
    """`applied=False` is shared with "wanted to reweight and could not", so the
    reason has to distinguish them -- otherwise a deliberate 'none' run reads in
    the log exactly like a soft-weighting run whose score was withheld."""
    result = DOWNWEIGHT_METHODS["none"](_pdf(), _published(), 0.5)

    assert not result.applied
    assert "none" in result.reason
    assert result.spread == 1.0


def test_none_does_not_care_that_no_score_was_published():
    """It never reads one. Before the first R emission, at the first emission and
    after it, 'none' does the same thing."""
    pdf = _pdf()
    assert DOWNWEIGHT_METHODS["none"](pdf, {}, 0.5).pdf == pdf


# --- the 'soft-weighting' arm -------------------------------------------------


@pytest.mark.parametrize("lam", [0.1, 0.25, 0.5, 1.0])
def test_soft_weighting_reproduces_reweight_exactly(lam):
    """The regression that pins "this arm is bit-identical to the behaviour before
    the flag existed". Not approx -- exactly: it is a thin adapter over
    redundancy.reweight, and any drift between them silently invalidates
    comparisons against every previously logged run."""
    published = _published()
    pdf = _pdf()

    through_registry = DOWNWEIGHT_METHODS["soft-weighting"](pdf, published, lam)
    direct = reweight(pdf, published["red"], lam)

    assert through_registry.pdf == direct.pdf
    assert through_registry.applied == direct.applied
    assert through_registry.spread == pytest.approx(direct.spread)


def test_soft_weighting_actually_moves_the_pdf():
    """Guards the test above from passing because both sides no-op'd."""
    result = DOWNWEIGHT_METHODS["soft-weighting"](_pdf(), _published(), 0.5)

    assert result.applied
    assert result.spread > 1.0


# --- the 'mRMR' arm -----------------------------------------------------------


def _shaped_pdf(shape=(0.2, 0.5, 0.3)):
    """A pdf with a non-flat shape over an op's magnitude levels.

    generate_pdf_new gives an op's levels a beta-binomial shape, and the flat
    `_pdf` above cannot tell "the shape was preserved" from "the shape was
    flattened by the reweighting"."""
    n = len(NAMES)
    none_mass = 1.0 / (n + 1)
    pdf = {
        (name, level): (1.0 - none_mass) * w / n
        for name in NAMES
        for level, w in zip((0.2, 0.5, 0.8), shape)
    }
    pdf[NONE_KEY] = none_mass
    return pdf


def _r_with_redundant(ops, rho=0.9):
    """R in which `ops` correlate strongly with everything and nothing else
    correlates with anything. The ranking has exactly one defensible answer."""
    r = np.zeros((len(NAMES), len(NAMES)))
    for op in ops:
        i = NAMES.index(op)
        r[i, :] = rho
        r[:, i] = rho
    np.fill_diagonal(r, 1.0)
    return r


def test_mrmr_drives_pruned_ops_to_exactly_zero():
    """The point of the arm. Not "very small" -- 0.0, so the op is unreachable by
    sampling rather than merely rare."""
    result = DOWNWEIGHT_METHODS["mRMR"](_pdf(), _published(), 1.0)

    assert result.applied
    zeroed = [k for k, v in result.pdf.items() if v == 0.0]
    assert zeroed, "mRMR at lambda=1 must remove something"
    assert NONE_KEY not in zeroed


def test_mrmr_prunes_whole_ops_not_individual_levels():
    """Redundancy is a property of the op, so every magnitude level of a pruned op
    goes at once. A partially-pruned op would be a magnitude decision wearing a
    redundancy decision's name."""
    pdf = _pdf()
    result = DOWNWEIGHT_METHODS["mRMR"](pdf, _published(), 1.0)

    by_op = {}
    for (op, _level), p in result.pdf.items():
        by_op.setdefault(op, []).append(p)
    for op, probs in by_op.items():
        assert all(p > 0 for p in probs) or all(p == 0.0 for p in probs), (
            f"{op} was pruned at only some of its levels: {probs}"
        )


def test_mrmr_preserves_the_level_shape_of_a_surviving_op():
    """Only the mass allotted BETWEEN ops moves. A survivor's levels keep their
    relative weights exactly, the same invariant redundancy.reweight holds."""
    pdf = _shaped_pdf()
    result = DOWNWEIGHT_METHODS["mRMR"](pdf, _published(), 1.0)

    survivor = next(iter(_ops_with_mass(result.pdf) - {"none"}))
    before = [pdf[(survivor, level)] for level in (0.2, 0.5, 0.8)]
    after = [result.pdf[(survivor, level)] for level in (0.2, 0.5, 0.8)]
    scale = after[0] / before[0]
    assert after == pytest.approx([b * scale for b in before])
    assert scale > 1.0, "the pruned mass has to go somewhere"


def test_mrmr_prunes_the_ops_that_are_redundant_with_everything():
    """The ranking is not just "some ops got removed": with a flat relevance term
    and an R where only blur/noise correlate with anything, minimum-redundancy
    selection has one right answer."""
    published = _published(r=_r_with_redundant(["blur", "noise"]))
    result = DOWNWEIGHT_METHODS["mRMR"](_pdf(), published, 1.0)

    assert result.applied
    survivors = _ops_with_mass(result.pdf)
    assert "blur" not in survivors
    assert "noise" not in survivors


@pytest.mark.parametrize("lam", [0.25, 0.5, 1.0, 2.0])
def test_mrmr_keeps_fewer_ops_as_lambda_grows(lam):
    """lambda still reads as "how hard redundancy pushes"; on this arm it pushes
    ops out of the bank. ceil(A/(1+lambda)) of the A measured ops survive."""
    result = DOWNWEIGHT_METHODS["mRMR"](_pdf(), _published(), lam)
    expected = int(np.ceil(len(NAMES) / (1.0 + lam)))

    assert len(_ops_with_mass(result.pdf) - {"none"}) == expected


def test_mrmr_budget_is_monotone_in_lambda():
    counts = [
        len(_ops_with_mass(DOWNWEIGHT_METHODS["mRMR"](_pdf(), _published(), lam).pdf))
        for lam in (0.25, 0.5, 1.0, 2.0)
    ]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] > counts[-1]


def test_mrmr_declines_when_only_the_row_sums_were_published():
    """It ranks ops against each other, so the per-op summary is not enough. A
    record from before "r" was published has to decline, not silently rank on
    something else."""
    published = _published()
    del published["r"]

    result = DOWNWEIGHT_METHODS["mRMR"](_pdf(), published, 0.5)

    assert not result.applied
    assert "pairwise" in result.reason
    assert result.pdf == _pdf()


def test_mrmr_never_prunes_an_op_with_no_usable_row_in_r():
    """Absence of evidence buys immunity. Deleting an augmentation on the strength
    of a measurement that does not exist is the one failure here that cannot be
    argued for, so an op R dropped upstream survives however hard lambda pushes."""
    r = _random_r()
    dropped = NAMES.index("noise")
    r[dropped, :] = np.nan
    r[:, dropped] = np.nan

    result = DOWNWEIGHT_METHODS["mRMR"](_pdf(), _published(r=r), 3.0)

    assert result.applied
    assert "noise" in _ops_with_mass(result.pdf)


def test_mrmr_is_deterministic():
    """Every rank runs this independently off an identical published record (see
    the DDP note in CLAUDE.md), so a ranking that broke ties non-deterministically
    would give each rank a different augmentation bank and nothing would say so."""
    published = _published()
    first = DOWNWEIGHT_METHODS["mRMR"](_pdf(), published, 1.0)
    second = DOWNWEIGHT_METHODS["mRMR"](_pdf(), published, 1.0)

    assert first.pdf == second.pdf


def test_mrmr_survives_the_json_round_trip():
    """`corr_redundancy_log.txt` and any offline consumer read R back with every
    non-finite cell already turned into `None` by _jsonable, which numpy refuses to
    coerce. Tolerated on the way in rather than crashing a live round."""
    r = _random_r()
    r[0, 1] = r[1, 0] = np.nan
    published = _published(r=r)
    published["r"] = [
        [None if not np.isfinite(v) else float(v) for v in row] for row in r
    ]

    result = DOWNWEIGHT_METHODS["mRMR"](_pdf(), published, 1.0)

    assert result.applied


# --- contract shared by every method -----------------------------------------


@pytest.mark.parametrize("name", sorted(DOWNWEIGHT_METHODS))
def test_every_method_preserves_normalisation_and_the_none_mass(name):
    """Down-weighting redistributes which augmentation is sampled, never how often
    augmentation happens: ("none", 0) is held fixed and the pdf still sums to 1."""
    pdf = _pdf()
    result = DOWNWEIGHT_METHODS[name](pdf, _published(), 0.5)

    assert sum(result.pdf.values()) == pytest.approx(1.0)
    assert result.pdf[NONE_KEY] == pytest.approx(pdf[NONE_KEY])
    assert set(result.pdf) == set(pdf)


@pytest.mark.parametrize(
    "name", sorted(set(DOWNWEIGHT_METHODS) - set(HARD_PRUNING_METHODS))
)
def test_every_soft_method_is_soft_never_zero(name):
    """Down-weighting is soft by construction unless a method opts out.

    Driving an entry to exactly 0 is deletion, and the correlation sizes actually
    observed here (mean |r| of 0.11-0.22) do not imply it -- so a method that
    deletes has to say so by joining HARD_PRUNING_METHODS, and everything else is
    held to softness automatically the moment it is registered. The opt-out is
    deliberately a line someone writes rather than a test someone forgets."""
    result = DOWNWEIGHT_METHODS[name](_pdf(), _published(), 1.0)

    assert all(v > 0 for v in result.pdf.values())


@pytest.mark.parametrize("name", sorted(HARD_PRUNING_METHODS))
def test_hard_pruning_methods_are_registered(name):
    """The opt-out set names methods, not aspirations. A stale entry here would
    silently exempt nothing from the softness contract while looking like it
    exempted something."""
    assert name in DOWNWEIGHT_METHODS


@pytest.mark.parametrize("name", sorted(DOWNWEIGHT_METHODS))
def test_every_method_leaves_the_pdf_alone_at_lambda_zero(name):
    """`_apply_redundancy_reweighting` short-circuits before dispatch at lambda=0,
    so this is belt-and-braces -- but a method that MOVED the pdf at lambda=0 would
    make --corr-lambda=0 quietly not a control arm, which is the one failure here
    that cannot be spotted in the logs. ('none' ignoring lambda is fine; acting
    without one is not.)"""
    pdf = _pdf()
    assert DOWNWEIGHT_METHODS[name](pdf, _published(), 0.0).pdf == pdf


@pytest.mark.parametrize("name", sorted(DOWNWEIGHT_METHODS))
def test_every_method_declines_with_a_reason_when_no_score_is_published(name):
    """Before the first R emission there is nothing to reweight by. A method must
    say so through `applied`/`reason` rather than raise -- the SA round that calls
    it is mid-training."""
    result = DOWNWEIGHT_METHODS[name](_pdf(), {"red": {}}, 0.5)

    assert not result.applied
    assert result.reason
