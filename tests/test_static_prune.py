"""Tests for the system-wide effects of `--pruned-augmentations`.

Pruning an op used to reach exactly two places -- the SA curve's source and the
GPU op bank -- which was enough to stop TRAINING on it and nothing else. These
cover the three consequences of that, and the guard that replaced a fourth:

* `--hold-none-prob` keeps the synthetic `("none", 0)` entry fixed as the bank
  shrinks, so a prune changes WHICH augmentation is sampled and not how often one
  is. Off by default, and the off path is asserted against the pre-flag
  arithmetic written out longhand -- a run launched before this flag existed has
  to stay comparable to one launched after it.

  Two things these turned up that the flag does NOT fix, both pre-existing and
  both pinned below rather than quietly assumed away: `generate_pdf_new` leaks
  ~13% of its perturbation mass to ("none", 0) through a truncated beta-binomial,
  so its P(none) is ~16% and not the 1/(N+1) the code reads as; and
  `generate_pdf_new_weighted_aug` has a rate drift of its own, from that same
  truncation being a function of the op count.
* `--no-inv-aug`'s ops are excluded at the SA curve's source rather than
  evaluated and then discarded.
* `--corr-skip-pruned-eval` is rejected outright on `--aug-type=grad_corr`.

`RobustValLoop.__init__` wants a live mmengine runner, so the pdf tests build the
loop through `__new__` and set only the attributes the generators actually read.
That is deliberate: it keeps the assertions pointed at the pdf arithmetic rather
than at a fixture's fidelity.
"""

import os
import sys

import pytest

pytest.importorskip("mmengine")
pytest.importorskip("mmseg")
pytestmark = pytest.mark.requires_mmseg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensaug.loops.sensaug_loop import RobustValLoop  # noqa: E402
from sensaug.redundancy import NONE_KEY  # noqa: E402

# Four levels per op, values irrelevant to every assertion here: the generators
# only ever sort by mIoU, and none of these tests is about the ORDER the
# beta-binomial assigns.
LEVELS = {0.2: 0.7, 0.4: 0.6, 0.6: 0.5, 0.8: 0.4}


def _loop(
    n_ops,
    *,
    hold_none_prob=False,
    n_static_excluded=0,
    uniform=False,
    remove_H=False,
):
    """A RobustValLoop with just enough state for the pdf generators to run."""
    loop = RobustValLoop.__new__(RobustValLoop)
    loop.sa_curve = {f"op{i}": dict(LEVELS) for i in range(n_ops)}
    loop.descending_MA = False
    loop.remove_H = remove_H
    loop._remove_H_names = ["lighter_H", "darker_H"]
    loop.hold_none_prob = hold_none_prob
    loop._n_static_excluded = n_static_excluded
    loop.uniform = uniform
    loop.pdf_dict = None
    # The generators call these three; none of them is what is under test.
    loop.test_perturbed_new = lambda: (
        {op: dict(levels) for op, levels in loop.sa_curve.items()},
        {},
    )
    loop._apply_redundancy_reweighting = lambda pdf: pdf
    return loop


def _none_prob(loop, uniform=False):
    pdf, _ = loop.generate_uniform_pdf() if uniform else loop.generate_pdf_new()
    return pdf[NONE_KEY]


# --- the flag is off by default, and off is the old arithmetic exactly --------


@pytest.mark.parametrize("n_ops", [1, 5, 24, 30, 32])
def test_flag_off_uniform_pdf_is_the_old_closed_form_exactly(n_ops):
    """generate_uniform_pdf's entries, off the SURVIVING count, bit-identical to
    the `1 / ((N + 1) * num_levels)` this replaced.

    Exact equality on purpose. The restructure had to introduce a `denom` that
    cancels when the flag is off, and a staged `(total / N) / levels` version of
    it differs from the original by an ULP -- close enough to pass a tolerance
    and not close enough to say a control arm did not move.
    """
    loop = _loop(n_ops, hold_none_prob=False, n_static_excluded=6)
    pdf, _ = loop.generate_uniform_pdf()

    expected = 1.0 / ((n_ops + 1) * len(LEVELS))
    for key, prob in pdf.items():
        if key != NONE_KEY:
            assert prob == expected
    # P(none) is 1 - sum(N * levels terms), so its last ULP depends on how many
    # terms were summed. Equal to within that, and no closer, is all there is.
    assert pdf[NONE_KEY] == pytest.approx(1.0 - n_ops / float(n_ops + 1), abs=1e-12)


@pytest.mark.parametrize("n_ops", [5, 24, 30])
def test_generate_pdf_new_leaks_mass_to_none_and_that_is_pre_existing(n_ops):
    """P(none) in generate_pdf_new is ~16%, NOT 1/(N+1) ~ 3%.

    `lambda_bb(i) = betabinom.pmf(i, len(levels), 0.75, 1.0)` is evaluated for
    i in 0..len(levels)-1, but that pmf's support is 0..len(levels) -- so each
    op's block sums to ~0.87 of its allotment and the remainder falls through to
    ("none", 0). Long-standing behaviour, untouched here and pinned so a change
    to it is deliberate. It does not affect --hold-none-prob, which holds the
    ALLOTMENT fixed and so holds whatever fraction of it leaks fixed too.
    """
    from scipy.stats import betabinom

    leak = sum(betabinom.pmf(i, len(LEVELS), 0.75, 1.0) for i in range(len(LEVELS)))
    loop = _loop(n_ops, hold_none_prob=False, n_static_excluded=6)
    pdf, _ = loop.generate_pdf_new()

    assert pdf[NONE_KEY] == pytest.approx(
        1.0 - (n_ops / float(n_ops + 1)) * leak, abs=1e-12
    )
    assert pdf[NONE_KEY] > 1.0 / (n_ops + 1)  # the leak, stated as an inequality


@pytest.mark.parametrize("n_ops", [1, 5, 24, 32])
@pytest.mark.parametrize("uniform", [False, True])
def test_flag_off_ignores_n_static_excluded_entirely(n_ops, uniform):
    """The count is carried on the loop unconditionally; only the flag reads it."""
    pdf_a, _ = (
        _loop(n_ops, n_static_excluded=0).generate_uniform_pdf()
        if uniform
        else _loop(n_ops, n_static_excluded=0).generate_pdf_new()
    )
    pdf_b, _ = (
        _loop(n_ops, n_static_excluded=9).generate_uniform_pdf()
        if uniform
        else _loop(n_ops, n_static_excluded=9).generate_pdf_new()
    )
    assert pdf_a == pdf_b


# --- with the flag on, the rate stops moving ---------------------------------


@pytest.mark.parametrize("uniform", [False, True])
def test_hold_none_prob_matches_the_unpruned_control(uniform):
    """The comparison this exists for: ours-control (30 ops) vs ours-pruned (24).

    Without the flag the pruned arm trains on clean images ~0.8pp more often,
    which is a second difference sitting inside a two-arm comparison.
    """
    control = _none_prob(_loop(30), uniform)
    pruned_off = _none_prob(_loop(24, n_static_excluded=6), uniform)
    pruned_on = _none_prob(
        _loop(24, hold_none_prob=True, n_static_excluded=6), uniform
    )

    assert pruned_off > control  # the drift being fixed
    # Sub-percentage-point and easy to overlook, which is exactly why it needs a
    # flag rather than a note. Not pinned to a literal: the size depends on the
    # generator (generate_pdf_new leaks part of its mass to none, see above).
    assert 0.004 < pruned_off - control < 0.010
    assert pruned_on == pytest.approx(control, abs=1e-12)


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("n_pruned", [1, 6, 12])
def test_hold_none_prob_is_invariant_to_how_many_ops_were_pruned(uniform, n_pruned):
    """Whatever the prune's size, P(none) lands where the unpruned run had it."""
    total = 30
    control = _none_prob(_loop(total), uniform)
    held = _none_prob(
        _loop(
            total - n_pruned, hold_none_prob=True, n_static_excluded=n_pruned
        ),
        uniform,
    )
    assert held == pytest.approx(control, abs=1e-12)


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("hold", [False, True])
@pytest.mark.parametrize("n_ops", [1, 7, 24])
def test_pdf_still_sums_to_one(uniform, hold, n_ops):
    """Non-negotiable: GpuAugSegDataPreProcessor.set_train_pdf rejects anything else."""
    loop = _loop(n_ops, hold_none_prob=hold, n_static_excluded=6)
    pdf, _ = loop.generate_uniform_pdf() if uniform else loop.generate_pdf_new()
    assert sum(pdf.values()) == pytest.approx(1.0, abs=1e-12)
    assert all(p >= 0.0 for p in pdf.values())


@pytest.mark.parametrize("uniform", [False, True])
def test_hold_none_prob_moves_mass_between_ops_not_out_of_them(uniform):
    """The pruned ops' mass goes to the survivors, which is the whole point.

    Each surviving op ends up with strictly MORE mass than it had in the
    unpruned run -- a redistribution, not a reduction.
    """
    control, _ = (
        _loop(30).generate_uniform_pdf() if uniform else _loop(30).generate_pdf_new()
    )
    pruned_loop = _loop(24, hold_none_prob=True, n_static_excluded=6)
    pruned, _ = (
        pruned_loop.generate_uniform_pdf()
        if uniform
        else pruned_loop.generate_pdf_new()
    )

    def by_op(pdf):
        out = {}
        for (op, _lvl), p in pdf.items():
            if op != NONE_KEY[0]:
                out[op] = out.get(op, 0.0) + p
        return out

    control_ops, pruned_ops = by_op(control), by_op(pruned)
    assert set(pruned_ops) < set(control_ops)
    assert sum(pruned_ops.values()) == pytest.approx(
        sum(control_ops.values()), abs=1e-12
    )
    for op, mass in pruned_ops.items():
        assert mass > control_ops[op]


def test_weighted_augs_generator_has_its_own_rate_drift():
    """--weighted-augs is NOT immune, contrary to what the `* 0.95` suggests.

    Its `lambda_bb` is a pmf over `len(all_mious) = N * levels`, so the same
    truncation leak documented above is itself a function of N -- P(none) moves
    from ~6.4% at 10 ops to ~5.5% at 30. That is a different mechanism from the
    `1/(N+1)` drift --hold-none-prob fixes, and of the same order, so the flag is
    deliberately NOT wired into this generator: correcting it means choosing how
    to renormalize a truncated pmf, which is a modelling decision rather than a
    bug fix. Pinned here so the gap is on the record rather than assumed away.
    """
    rates = []
    for n_ops in (10, 24, 30):
        loop = _loop(n_ops)
        pdf, _ = loop.generate_pdf_new_weighted_aug()
        assert sum(pdf.values()) == pytest.approx(1.0, abs=1e-12)
        rates.append(pdf[NONE_KEY])

    assert rates == sorted(rates, reverse=True)  # more ops -> less "none"
    assert 0.005 < max(rates) - min(rates) < 0.02
    # And --hold-none-prob does not touch it, in either direction.
    held = _loop(24, hold_none_prob=True, n_static_excluded=6)
    assert held.generate_pdf_new_weighted_aug()[0][NONE_KEY] == pytest.approx(
        rates[1], abs=1e-12
    )


# --- the denominator helper itself -------------------------------------------


@pytest.mark.parametrize(
    "hold,excluded,surviving,expected",
    [
        (False, 6, 24, 24),
        (True, 6, 24, 30),
        (True, 0, 30, 30),
        (False, 0, 30, 30),
    ],
)
def test_none_prob_denominator(hold, excluded, surviving, expected):
    loop = _loop(1, hold_none_prob=hold, n_static_excluded=excluded)
    assert loop._none_prob_denominator(surviving) == expected


# --- --no-inv-aug is excluded at the source, not evaluated then discarded -----


def test_remove_H_names_resolve_per_vocabulary():
    """The names differ by vocabulary, and both call sites now read one attribute
    rather than each re-deriving the branch."""
    for pset, expected in (
        ("diff32", ["lighter_H", "darker_H"]),
        ("non-diff32", ["lighter_H", "darker_H"]),
        ("legacy20", ["PosterizeTransform", "SolarizeTransform"]),
    ):
        loop = RobustValLoop.__new__(RobustValLoop)
        loop.perturbation_set = pset
        names = (
            ["lighter_H", "darker_H"]
            if loop._is_corr_vocabulary
            else ["PosterizeTransform", "SolarizeTransform"]
        )
        assert names == expected


def test_remove_H_perturbations_is_idempotent_on_an_already_filtered_curve():
    """It still guards the load_sa_curve() path, where the curve comes off disk
    unfiltered -- so it has to be a no-op when update_sa_curve already excluded
    the same names rather than raising on their absence."""
    loop = _loop(2, remove_H=True)
    record = {"op0": dict(LEVELS), "op1": dict(LEVELS)}
    loop._remove_H_perturbations(record)
    assert set(record) == {"op0", "op1"}

    record["lighter_H"] = dict(LEVELS)
    loop._remove_H_perturbations(record)
    assert "lighter_H" not in record


# --- --corr-skip-pruned-eval is rejected on grad_corr ------------------------


class _Args:
    def __init__(self, **kw):
        self.corr_skip_pruned_eval = False
        self.aug_type = "ours"
        self.__dict__.update(kw)


@pytest.mark.parametrize("aug_type", ["ours", "none", "random", "default"])
def test_skip_pruned_eval_is_not_rejected_off_grad_corr(aug_type):
    """Inert there, not wrong -- it stays a warning, so an otherwise fine run
    is never refused over a flag that does nothing."""
    train = pytest.importorskip("train")
    assert (
        train.reject_skip_pruned_eval(
            _Args(corr_skip_pruned_eval=True, aug_type=aug_type)
        )
        is None
    )


def test_skip_pruned_eval_is_rejected_on_grad_corr():
    train = pytest.importorskip("train")
    msg = train.reject_skip_pruned_eval(
        _Args(corr_skip_pruned_eval=True, aug_type="grad_corr")
    )
    assert msg is not None
    # The message has to say what to do instead, not only what was refused.
    assert "--pruned-augmentations" in msg


def test_skip_pruned_eval_unset_is_never_rejected():
    train = pytest.importorskip("train")
    assert train.reject_skip_pruned_eval(_Args(aug_type="grad_corr")) is None
