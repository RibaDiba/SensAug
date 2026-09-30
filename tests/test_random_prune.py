"""The `null` random-pruning arm: does it draw what it says, and reproducibly?

The arm itself is deliberately thin -- it picks N op names and hands them to the
`--pruned-augmentations` path that tests/test_static_prune.py already covers end
to end. So what is worth pinning here is only the part that is new, and all of it
is about the draw being trustworthy rather than merely random:

* **Determinism across ranks.** Every rank runs train.py independently and
  torch.distributed is not initialized at argparse time, so the draw has to be a
  pure function of a seed every rank already agrees on. A draw that depended on
  process-local state (the global numpy RNG, PYTHONHASHSEED, dict ordering) would
  give each GPU a different bank, averaged into one gradient update, with nothing
  in the logs saying so.
* **The count means what it says.** N dropped has to be N ops fewer than the
  control trains on, or the comparison against an mRMR arm's N is off by however
  many the pool already excluded.
* **Every refusal has a message.** The arm is launched from an sbatch; a bad
  combination has to cost exit 2 rather than a compute node and a queue.
"""

import os
import sys

import pytest

pytest.importorskip("mmengine")
pytest.importorskip("mmseg")
pytestmark = pytest.mark.requires_mmseg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensaug.dataset.augmentations import LEGACY20_OPS  # noqa: E402
from sensaug.dataset.differentiable_augmentations_aa import (  # noqa: E402
    DIFF32_OPS,
    GEOMETRIC_OP_KEYS,
)


class _Args:
    """Just the attributes the random-prune helpers read off `args`."""

    def __init__(self, **kw):
        self.aug_type = "ours"
        self.no_inv_aug = False
        self.geometric_only = False
        self.photometric_only = False
        self.pruned_augmentations = []
        self.random_prune_method = "null"
        self.random_prune_count = 6
        self.random_prune_seed = 0
        self.random_prune_ops = []
        self.random_prune_pool = []
        self.random_prune_seed_used = None
        self.random_prune_seed_source = None
        self.__dict__.update(kw)


@pytest.fixture
def train():
    return pytest.importorskip("train")


# --------------------------------------------------------------------------
# The pool
# --------------------------------------------------------------------------


def test_pool_is_the_whole_diff32_bank_by_default(train):
    assert train.random_prune_pool(_Args()) == sorted(DIFF32_OPS)


def test_random_arm_draws_from_legacy20_instead(train):
    assert train.random_prune_pool(_Args(aug_type="random")) == sorted(LEGACY20_OPS)


@pytest.mark.parametrize("aug_type", ["none", "autoaugment", "augmix", "idbh", "vip"])
def test_pool_is_empty_on_an_arm_that_samples_from_no_registry(train, aug_type):
    assert train.random_prune_pool(_Args(aug_type=aug_type)) == []


def test_no_inv_aug_ops_are_not_eligible(train):
    """--no-inv-aug already removed these at the SA curve's source.

    The Nexus sbatch passes it on every run, so drawing one would spend a prune
    on an op that was already gone -- N-1 real prunes against an mRMR arm's N.
    """
    pool = train.random_prune_pool(_Args(no_inv_aug=True))
    assert "lighter_H" not in pool and "darker_H" not in pool
    assert len(pool) == len(DIFF32_OPS) - 2


def test_no_inv_aug_ops_are_the_legacy20_pair_on_the_legacy20_arm(train):
    pool = train.random_prune_pool(_Args(aug_type="random", no_inv_aug=True))
    assert "PosterizeTransform" not in pool and "SolarizeTransform" not in pool
    assert len(pool) == len(LEGACY20_OPS) - 2


def test_pool_respects_geometric_and_photometric_only(train):
    geo = train.random_prune_pool(_Args(geometric_only=True))
    photo = train.random_prune_pool(_Args(photometric_only=True))
    assert set(geo) == set(GEOMETRIC_OP_KEYS)
    assert set(geo) & set(photo) == set()
    assert set(geo) | set(photo) == set(DIFF32_OPS)


def test_pool_excludes_what_is_already_pruned(train):
    """So a random draw COMPOSES with an explicit list instead of overlapping it.

    An overlap would silently shrink the draw's own count.
    """
    already = ["blur", "rotate_neg"]
    pool = train.random_prune_pool(_Args(), already_pruned=already)
    assert set(already).isdisjoint(pool)
    assert len(pool) == len(DIFF32_OPS) - 2


def test_pool_is_sorted(train):
    """The draw must not depend on a registry's dict iteration order."""
    pool = train.random_prune_pool(_Args())
    assert pool == sorted(pool)


# --------------------------------------------------------------------------
# The draw
# --------------------------------------------------------------------------


def test_same_seed_gives_the_same_draw(train):
    pool = train.random_prune_pool(_Args())
    assert train.draw_random_prune(pool, 6, 17) == train.draw_random_prune(pool, 6, 17)


def test_different_seeds_give_different_draws(train):
    pool = train.random_prune_pool(_Args())
    draws = {tuple(train.draw_random_prune(pool, 6, s)) for s in range(8)}
    assert len(draws) > 1


def test_draw_is_independent_of_the_global_numpy_rng(train):
    """The DDP-safety property, and the one worth stating loudest.

    train.py pins the global RNG to 0 via set_manual_seed, and the GPU
    augmentation sampler then draws every per-image op from it. If the prune
    consumed from that stream, the two arms would differ by more than their
    banks -- and if it *depended* on that stream, any rank whose state had
    diverged would prune differently.
    """
    import numpy as np

    pool = train.random_prune_pool(_Args())
    np.random.seed(123)
    first = train.draw_random_prune(pool, 6, 42)
    np.random.seed(999)
    [np.random.random() for _ in range(50)]
    second = train.draw_random_prune(pool, 6, 42)
    assert first == second


def test_draw_returns_exactly_count_distinct_ops_from_the_pool(train):
    pool = train.random_prune_pool(_Args(no_inv_aug=True))
    for count in (1, 6, 15, len(pool) - 1):
        drawn = train.draw_random_prune(pool, count, 3)
        assert len(drawn) == len(set(drawn)) == count
        assert set(drawn) <= set(pool)
        assert drawn == sorted(drawn)


# --------------------------------------------------------------------------
# The seed
# --------------------------------------------------------------------------


def test_explicit_seed_wins_over_every_env_var(train):
    env = {"SLURM_JOB_ID": "11809659", "TORCHELASTIC_RUN_ID": "abc", "WORLD_SIZE": "4"}
    assert train.resolve_random_prune_seed(7, env=env) == (7, "--random-prune-seed")


def test_seed_falls_back_to_the_slurm_job_id(train):
    seed, source = train.resolve_random_prune_seed(
        None, env={"SLURM_JOB_ID": "11809659", "WORLD_SIZE": "4"}
    )
    assert source == "SLURM_JOB_ID"
    assert seed == 11809659


def test_seed_falls_back_to_torchelastic_run_id(train):
    env = {"TORCHELASTIC_RUN_ID": "a-uuid-shaped-thing", "WORLD_SIZE": "4"}
    seed, source = train.resolve_random_prune_seed(None, env=env)
    assert source == "TORCHELASTIC_RUN_ID"
    assert 0 <= seed < 2**32
    # Stable across processes: crc32, not the PYTHONHASHSEED-salted hash().
    assert seed == train.resolve_random_prune_seed(None, env=env)[0]


def test_a_non_numeric_run_id_still_yields_a_seed_in_range(train):
    for text in ("abc", "11809659_4", "", "x" * 300):
        env = {"TORCHELASTIC_RUN_ID": text or "z", "WORLD_SIZE": "4"}
        seed, _ = train.resolve_random_prune_seed(None, env=env)
        assert 0 <= seed < 2**32


@pytest.mark.parametrize("env", [{}, {"WORLD_SIZE": "1"}, {"WORLD_SIZE": ""}])
def test_single_process_runs_may_generate_a_seed(train, env):
    seed, source = train.resolve_random_prune_seed(None, env=dict(env))
    assert source == "urandom"
    assert 0 <= seed < 2**32


def test_multi_rank_with_nothing_to_derive_from_is_refused(train):
    """Rather than silently giving each rank a different bank."""
    with pytest.raises(ValueError, match="--random-prune-seed"):
        train.resolve_random_prune_seed(None, env={"WORLD_SIZE": "4"})


# --------------------------------------------------------------------------
# The refusals
# --------------------------------------------------------------------------


def test_no_arm_no_flags_no_complaint(train):
    args = _Args(random_prune_method="none", random_prune_count=None, random_prune_seed=None)
    assert train.reject_random_prune(args) is None


@pytest.mark.parametrize(
    "kw, needle",
    [
        ({"random_prune_count": 6, "random_prune_seed": None}, "--random-prune-count"),
        ({"random_prune_count": None, "random_prune_seed": 3}, "--random-prune-seed"),
    ],
)
def test_a_random_prune_flag_without_the_arm_is_refused(train, kw, needle):
    args = _Args(random_prune_method="none", **kw)
    msg = train.reject_random_prune(args)
    assert msg is not None and needle in msg


def test_the_arm_without_a_count_is_refused(train):
    msg = train.reject_random_prune(_Args(random_prune_count=None))
    assert msg is not None and "--random-prune-count" in msg


@pytest.mark.parametrize("count", [0, -1])
def test_a_non_positive_count_is_refused(train, count):
    msg = train.reject_random_prune(_Args(random_prune_count=count))
    assert msg is not None and ">= 1" in msg


def test_the_bounds_check_is_skipped_until_a_pool_is_supplied(train):
    """So the cheap checks can run before the vocabulary is even resolved."""
    assert train.reject_random_prune(_Args(random_prune_count=999)) is None


def test_an_arm_with_no_eligible_ops_is_refused(train):
    args = _Args(aug_type="none")
    pool = train.random_prune_pool(args)
    msg = train.reject_random_prune(args, pool=pool)
    assert msg is not None and "nothing to draw from" in msg


def test_pruning_the_entire_pool_is_refused(train):
    """>= not >: it would leave a pdf of nothing but ("none", 0), and
    CollectGradientHook refuses a static prune naming every op anyway."""
    args = _Args(random_prune_count=len(DIFF32_OPS))
    pool = train.random_prune_pool(args)
    msg = train.reject_random_prune(args, pool=pool)
    assert msg is not None and "would leave no augmentations" in msg


def test_leaving_one_op_standing_is_allowed(train):
    args = _Args(random_prune_count=len(DIFF32_OPS) - 1)
    pool = train.random_prune_pool(args)
    assert train.reject_random_prune(args, pool=pool) is None


def test_the_bounds_message_names_the_flags_that_shrank_the_pool(train):
    args = _Args(
        random_prune_count=31,
        no_inv_aug=True,
        pruned_augmentations=["blur"],
    )
    pool = train.random_prune_pool(args, args.pruned_augmentations)
    msg = train.reject_random_prune(args, pool=pool)
    assert msg is not None
    assert "--no-inv-aug" in msg and "blur" in msg and "29" in msg


# --------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------


def test_the_record_partitions_the_pool_into_dropped_and_kept(train):
    args = _Args(no_inv_aug=True, random_prune_count=6)
    args.random_prune_pool = train.random_prune_pool(args)
    args.random_prune_ops = train.draw_random_prune(args.random_prune_pool, 6, 0)
    args.random_prune_seed_used = 0
    args.random_prune_seed_source = "--random-prune-seed"
    args.pruned_augmentations = sorted(args.random_prune_ops)

    record = train.random_prune_record(args)
    assert record["seed"] == 0
    assert record["count"] == len(record["dropped"]) == 6
    assert record["perturbation_set"] == "diff32"
    assert set(record["dropped"]).isdisjoint(record["kept"])
    assert sorted(record["dropped"] + record["kept"]) == sorted(record["pool"])
    assert record["pool_size"] == len(DIFF32_OPS) - 2


def test_log_random_prune_writes_nothing_on_the_none_arm(train, tmp_path):
    train.log_random_prune(_Args(random_prune_method="none"), str(tmp_path))
    assert list(tmp_path.iterdir()) == []
