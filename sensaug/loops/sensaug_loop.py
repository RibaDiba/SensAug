"""The sensitivity-analysis pipeline: `RobustValLoop`.

This is the loop `--aug-type=ours` runs. Every `round_interval` iterations it
re-evaluates how badly the model does under each perturbation, turns that into a
sampling pdf over (perturbation, magnitude) pairs, and rebuilds the train
dataloader to sample augmentations from it. The SA *curve* itself is recomputed
every `SA_CURVE_CADENCE`-th round (6, from `sensaug/round_schedule.py`), so its
effective cadence is `6 x round_interval` -- and that is the grid the gradient
cross-correlation pipeline's sweeps are scheduled onto.

`--aug-type=grad_corr` runs `GradCorrValLoop` (`grad_corr_loop.py`), which is a
strict superset of this: same SA machinery, plus Lever 3's redundancy
down-weighting of the pdf it produces. The one extension point between them is
`_apply_redundancy_reweighting`, a no-op here.

The gradient cross-correlation measurement itself lives entirely in
`sensaug/hooks/` and runs off its own clock -- the two pipelines share no clock
and no hook point.
"""

import os
import json
from pprint import pprint
from copy import deepcopy

from scipy.stats import betabinom

from mmengine.logging import print_log
from mmengine.runner import ValLoop
from mmseg.registry import LOOPS
from mmengine.dist import is_main_process

# Check Pytorch installation
import torch

# Local imports
# from sensaug.dataset.augmentations import *
from sensaug.sensitivity_analysis import *  # noqa: F401,F403
from sensaug.runner_utils import *  # noqa: F401,F403
from sensaug.corr_magnitudes import conditional_levels, modal_magnitude
# The SA-curve cadence is defined once, in the module that also builds the
# correlation pipeline's firing schedule out of it. The two must agree: those
# sweeps exist to land on the rounds this `%` selects.
from sensaug.round_schedule import SA_CURVE_CADENCE
from sensaug.dataset.differentiable_augmentations_aa import DIFF32_OPS
from sensaug.dataset.augmentations import LEGACY20_OPS

__all__ = [
    "dict_mean",
    "RobustValLoop",
    "RobustBaselineValLoop",
]


def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


@LOOPS.register_module()
class RobustValLoop(ValLoop):
    """Custom loop for val."""

    def __init__(
        self,
        runner,
        dataloader,
        evaluator,
        ratio=0.5,
        sa_curve_path=None,
        uniform=False,
        descending_MA=False,
        remove_H=False,
        warmup_rounds: int = 4,
        random_aug: bool = False,
        geometric_only: bool = False,
        photometric_only: bool = False,
        weighted_augs: bool = False,
        perturbation_set: str = "legacy20",
        corr_skip_pruned_eval: bool = False,
        pruned_augmentations: list = None,
        hold_none_prob: bool = False,
        fp16: bool = False,
    ) -> None:
        super().__init__(runner, dataloader, evaluator, fp16)

        assert sa_curve_path is not None, (
            "sa curve path must be provided for evaluation"
        )

        self.ratio = ratio
        self.kid = None
        self.sa_curve = None
        self.pdf_dict = None
        self.remove_H = remove_H
        self.uniform = uniform
        self.warmup_rounds: int = warmup_rounds
        self.descending_MA = descending_MA
        self.static_sa_curve_path = sa_curve_path
        self.sa_log_path = os.path.join(runner.cfg.work_dir, "sa_curve_log.txt")
        self.perturb_metrics_path = os.path.join(
            runner.cfg.work_dir, "perturb_eval.txt"
        )
        self.n_rounds = runner.iter // runner.cfg.train_cfg.val_interval
        self.random_aug = random_aug
        self.geometric_only = geometric_only
        self.photometric_only = photometric_only
        self.weighted_augs = weighted_augs

        # Only meaningful under GradCorrValLoop + a HARD_PRUNING_METHODS arm
        # (mRMR): skip re-evaluating an op nobody is currently training on,
        # except on the rounds the SA curve itself refreshes. Inert here --
        # `runner.corr_pruned_ops` is never populated unless the down-weighting
        # loop is the one running -- so this flag is safe to carry on the base
        # class rather than duplicating test_perturbed_new for the subclass.
        self.corr_skip_pruned_eval = corr_skip_pruned_eval
        # (op -> (miou_by_level, mean_metrics)) as of the last round it was
        # actually measured. What a skip round reads back instead of a fresh
        # eval; see test_perturbed_new.
        self._last_measured = {}
        # Set for real in run(), right where the SA-curve recompute gate is
        # already evaluated -- test_perturbed_new reads this directly (same
        # object, no lag). CollectGradientHook needs the equivalent signal too,
        # but reads it as a STATIC full_recheck_iters schedule fixed at
        # config-build time instead (sensaug/round_schedule.py), not a live
        # attribute here: a runtime flag would lag by one round on that hook,
        # since after_train_iter hooks run before this round's own run() call
        # on a shared firing iteration.
        self._sa_curve_just_recomputed = False

        # Which augmentation vocabulary SA measures. "legacy20" is the historical
        # LEGACY20_OPS set; "non-diff32" and "diff32" are both keyed by the 32 op
        # names the gradient cross-correlation pipeline differentiates, so that the
        # SA curve and the matrix R describe the SAME augmentations (they
        # previously shared no names at all, so no SA magnitude could be looked up
        # for any op in R).
        #
        # "diff32" is what every compared arm (ours/default/grad_corr) uses: the
        # differentiable ops themselves, applied batched on GPU by
        # sensaug.dataset.gpu_augment. That makes the function SA probes, the
        # function training applies, and the function the gradient probe
        # differentiates literally the same one -- previously they were at best two
        # implementations sharing a name, and 8 of the 32 were not even calibrated
        # against each other.
        self.perturbation_set = perturbation_set
        # Static, whole-run exclusion list (--pruned-augmentations). Applied at
        # the SA curve's source in update_sa_curve -- test_perturbed_new and
        # every pdf generator below it only ever iterate self.sa_curve, so
        # excluding a pruned op there is sufficient to keep it out of the
        # round-eval sweep AND every pdf built from it.
        self.pruned_augmentations = list(pruned_augmentations or [])

        # The color/photometric ops --no-inv-aug drops, resolved ONCE here
        # rather than inside _remove_H_perturbations, because two call sites now
        # need them: that method (which still guards the load_sa_curve() path,
        # where the curve comes off disk unfiltered) and update_sa_curve's
        # `exclude=`, which keeps them out of the round-eval in the first place.
        # Previously they were only ever popped from miou_record AFTER
        # test_perturbed_new had already paid for a full 5-level eval on each.
        self._remove_H_names = (
            ["lighter_H", "darker_H"]
            if self._is_corr_vocabulary
            else ["PosterizeTransform", "SolarizeTransform"]
        )

        # How many ops --pruned-augmentations removed that WOULD otherwise have
        # reached the pdf: in this run's vocabulary, and not already gone via
        # remove_H. This is the only quantity --hold-none-prob needs; see
        # _none_prob_denominator for what it is for.
        self.hold_none_prob = hold_none_prob
        _vocab = set(DIFF32_OPS) if self._is_corr_vocabulary else set(LEGACY20_OPS)
        _also_removed = set(self._remove_H_names) if self.remove_H else set()
        self._n_static_excluded = len(
            {n for n in self.pruned_augmentations if n in _vocab} - _also_removed
        )

        self.corr_magnitudes_path = os.path.join(
            runner.cfg.work_dir, "corr_magnitudes.json"
        )

        with open(sa_curve_path, "r") as f:
            sa_curve_str = f.read()
            self.eval_sa_curve = json.loads(sa_curve_str)

        if is_main_process():
            print(
                f"Starting from round {self.n_rounds}; \
                Uniform sampling: {self.uniform}; \
                Descending MA: {self.descending_MA}; \
                Remove color augs: {self.remove_H}; \
                Warmup rounds: {self.warmup_rounds}"
            )

    def load_sa_curve(self):
        assert self.static_sa_curve_path is not None, "No SA curve path provided"
        path = self.static_sa_curve_path
        with open(path, "r") as f:
            sa_curve_str = f.read()
            self.sa_curve = json.loads(sa_curve_str)

    def _apply_redundancy_reweighting(self, pdf_dict: dict) -> dict:
        """The SA arm's pdf is used exactly as generated.

        This is the single extension point GradCorrValLoop overrides to apply
        Lever 3. It is a no-op here rather than a `if self.corr_lambda` branch
        because nothing publishes `runner.corr_redundancy` unless the gradient
        cross-correlation hooks are registered, and only `--aug-type=grad_corr`
        registers them -- so on this arm there is never anything to reweight by.

        Called by all three pdf generators, so the override composes with
        `--uniform` and `--weighted-augs` rather than silently applying to only one
        of them.
        """
        return pdf_dict

    @property
    def _is_corr_vocabulary(self) -> bool:
        """Whether this vocabulary's op names are the ones R is indexed by.

        True for both "non-diff32" and "diff32" -- they share all 32 keys and differ
        only in how each key is applied. Anything handed to the gradient probe or
        read back from it keys off this, never off the implementation.
        """
        return self.perturbation_set in ("non-diff32", "diff32")

    def update_sa_curve(self):
        # if get_rank() == 0:
        print_log("Running sensitivity analysis...", logger="current")
        val_dataloader_cfg = deepcopy(self.runner.cfg.val_dataloader)
        self.sa_curve = adaptive_sensitivity_analysis_new(  # noqa: F405
            val_dataloader_cfg,
            self.runner,
            num_levels=5,
            tolerance=0.05,
            perturbation_set=self.perturbation_set,
            # Both exclusions at the SOURCE. test_perturbed_new and every pdf
            # generator below it only ever iterate self.sa_curve, so excluding
            # here keeps an op out of the round-eval sweep AND out of every pdf
            # built from it. The remove_H half used to be applied instead by
            # _remove_H_perturbations, which pops from miou_record only after a
            # full 5-level eval has already been run on each of those ops.
            exclude=self.pruned_augmentations
            + (self._remove_H_names if self.remove_H else []),
        )

        assert self.sa_curve is not None, "SA curve is None after broadcasting"

        if is_main_process():
            print("New SA curve computed")
            print(self.sa_curve)

            # SA curve history logging
            sa_curve_str = json.dumps(self.sa_curve)
            with open(self.sa_log_path, "a") as f:
                f.write(sa_curve_str + "\n")

    def test_perturbed_new(self):
        # dict to keep track of MIOU performance of all types and levels
        miou_record = {}
        metrics_record = {}
        final_metrics = {}

        # Currently-pruned ops (mRMR, via GradCorrValLoop) are skipped on every
        # round EXCEPT the ones where the SA curve itself just refreshed -- that's
        # the natural point to re-measure everything, since the curve's own
        # relevance numbers just changed too. `corr_pruned_ops` is only ever
        # populated by GradCorrValLoop, so this is empty (a full sweep every
        # round, today's behaviour) unless both the flag and a hard-pruning
        # method are active.
        skip_ops = (
            frozenset()
            if not self.corr_skip_pruned_eval or self._sa_curve_just_recomputed
            else getattr(self.runner, "corr_pruned_ops", frozenset())
        )
        stale_ops = []

        # iterate through perturbation levels and test their MIOU performance
        for p_type, levels in self.sa_curve.items():
            if p_type in skip_ops and p_type in self._last_measured:
                # Not training on this op right now -- reuse its last real
                # reading instead of spending a full eval pass re-measuring an
                # op nobody is sampling. Flagged in final_metrics["_stale_ops"]
                # below, not silently blended in: this is exactly the data a
                # "did pruning hurt robustness" comparison reads.
                miou_record[p_type], metrics_record[p_type] = self._last_measured[
                    p_type
                ]
                stale_ops.append(p_type)
                continue

            miou_record[p_type] = {}
            metrics_record[p_type] = []

            for _, level in enumerate(levels):
                # evaluate on current SA curve to choose PDF
                apply_perturbations_dataloader(  # noqa: F405
                    self.runner,
                    train=False,
                    perturb_levels={p_type: level},
                    # "non-diff32" and "diff32" share all 32 keys; without this the
                    # round-eval resolves every one of them onto the per-image
                    # torch wrappers regardless of which vocabulary is in play.
                    perturbation_set=self.perturbation_set,
                )
                metrics = self._run_eval(ratio=self.ratio)
                miou = metrics["mIoU"]
                miou_record[p_type][level] = miou

                metrics_eval = metrics
                metrics_record[p_type].append(metrics_eval)

            metrics_record[p_type] = dict_mean(
                metrics_record[p_type]
            )  # average metric for p type across all levels

            # Cache the fresh reading so a later skip round has something real
            # to carry forward -- see the skip branch above.
            self._last_measured[p_type] = (
                miou_record[p_type],
                metrics_record[p_type],
            )

        for p_type, mean_dict in metrics_record.items():
            for metric, value in mean_dict.items():
                new_key = p_type.replace("_", "") + f"_{metric}"
                final_metrics[new_key] = value

        if stale_ops:
            # Self-documenting: a reader of perturb_eval.txt must be able to
            # tell a carried-forward number from a fresh one.
            final_metrics["_stale_ops"] = sorted(stale_ops)

        return miou_record, final_metrics

    def test_perturbed(self):
        # dict to keep track of MIOU performance of all types and levels
        miou_record = {}
        metrics_record = {}
        final_metrics = {}

        # iterate through perturbation levels and test their MIOU performance
        for p_type, levels in self.sa_curve.items():
            miou_record[p_type] = {}
            metrics_record[p_type] = []

            for i, level in enumerate(levels):
                # evaluate on current SA curve to choose PDF
                apply_perturbations_dataloader(  # noqa: F405
                    self.runner, train=False, perturb_levels={p_type: level}
                )
                metrics = self._run_eval(ratio=self.ratio)
                miou = metrics["mIoU"]
                miou_record[p_type][level] = miou

                metrics_eval = metrics
                metrics_record[p_type].append(metrics_eval)

            metrics_record[p_type] = dict_mean(
                metrics_record[p_type]
            )  # average metric for p type across all levels

        for p_type, mean_dict in metrics_record.items():
            for metric, value in mean_dict.items():
                new_key = p_type.replace("_", "") + f"_{metric}"
                final_metrics[new_key] = value

        return miou_record, final_metrics

    def publish_corr_magnitudes(self):
        """Hand this round's per-op magnitude distributions to the gradient
        cross-correlation probe.

        Only meaningful for an R-keyed vocabulary ("non-diff32" or "diff32"): the probe
        differentiates DIFF32_OPS, so a snapshot keyed by
        LEGACY20_OPS names would match nothing and every op would silently
        fall back to the fixed reference magnitude. Skipped outright rather than
        published-and-ignored, so `runner.corr_magnitudes` is never a misleading
        non-empty dict.

        Under "diff32" the magnitude is exact: SA derives the level on the very op
        the probe then applies it to. Under "non-diff32" the names still match but
        the SCALES do not for 8 of the 32 -- the SA level is derived on the CPU op
        and applied to the differentiable one, and those 8 were never calibrated
        against each other (blur is the starkest: cv2's kernel-size-derived
        implicit sigma against sigma itself). There, op identity transfers but the
        magnitude is approximate.

        Published onto the runner (the same handoff channel CollectGradientHook
        already uses for `runner.aug_grad_buffer`) rather than pushed: the probe
        fires on its own clock and simply reads whatever is current, which is what
        makes "use the latest distribution" work when it fires less often than
        this loop.

        Lives on the SA base rather than on GradCorrValLoop even though only the
        grad_corr arm has a live probe: `corr_magnitudes.json` is also the seed
        file the OFFLINE pipeline reads. scripts/compute_grad_corr.py auto-detects
        <work-dir>/corr_magnitudes.json when --magnitudes-path is not given, so an
        `ours` run's snapshot is what lets you recompute R against that run's
        checkpoint later. Publishing only under grad_corr would leave that path
        silently falling back to the fixed 0.5 reference magnitude. Nothing on the
        `ours` code path reads it back -- both consumers are inside
        CollectGradientHook, which only --aug-type=grad_corr registers.

        Rank-consistency: every rank runs this loop and mmengine all-reduces the
        evaluator metrics that produce the pdf, so all ranks derive the same
        snapshot. Only rank 0 writes the file.
        """
        if not self._is_corr_vocabulary or not self.pdf_dict:
            return

        snapshot = conditional_levels(
            self.pdf_dict, op_names=set(DIFF32_OPS)
        )
        self.runner.corr_magnitudes = snapshot

        if not is_main_process():
            return

        record = {
            "iter": int(self.runner.iter),
            "round": int(self.n_rounds),
            "perturbation_set": self.perturbation_set,
            # The modal level is recorded because it is the probe's default
            # magnitude; the full distribution is recorded because the sampled
            # modes need it and because R must be recomputable offline.
            "magnitudes": {
                op: dict(entry, mode=modal_magnitude(entry))
                for op, entry in snapshot.items()
            },
        }
        records = []
        if os.path.exists(self.corr_magnitudes_path):
            with open(self.corr_magnitudes_path) as f:
                records = json.load(f)
        records.append(record)
        # Whole-file rewrite via a temp file: a plain truncate+write interrupted
        # midway would destroy every previous round's snapshot, not just this one.
        tmp = self.corr_magnitudes_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(records, f, indent=2, allow_nan=False)
        os.replace(tmp, self.corr_magnitudes_path)

        print_log(
            f"[corr-magnitudes] round {self.n_rounds} (iter {self.runner.iter}): "
            f"published {len(snapshot)} ops, modal magnitudes "
            + ", ".join(
                f"{op}={modal_magnitude(entry):.3f}"
                for op, entry in sorted(snapshot.items())
            ),
            logger="current",
        )

    def _none_prob_denominator(self, n_surviving: int) -> int:
        """Op count the ``("none", 0)`` mass is computed against.

        Default: the surviving count, exactly as it always has been -- returning
        `n_surviving` unchanged reproduces the old arithmetic bit-for-bit, which
        is what keeps a run launched before this flag existed comparable with one
        launched after.

        Under ``--hold-none-prob``: the count this run WOULD have had without
        ``--pruned-augmentations``, so the static prune redistributes mass
        between ops without changing how often any augmentation fires at all.
        That is the same invariant `_downweight_mrmr` already holds via its
        `free_mass / survivor_mass` rescale, and the one CLAUDE.md states for
        Lever 3 ("changes which augmentation is sampled, never how often").
        Without it, pruning 6 of 30 ops raises P(no augmentation) by ~0.7pp -- a
        second difference between arms, on top of the one being measured.

        Note this fixes the ALLOTMENT, not P(none) directly. `generate_pdf_new`
        leaks ~13% of each op's allotment back into ("none", 0) through a
        truncated beta-binomial (see its call site and
        tests/test_static_prune.py), so its real P(none) is ~16% rather than the
        `1/(N+1)` its arithmetic reads as. Holding the allotment fixed holds the
        leaked fraction fixed too, so the drift is removed either way -- but
        "P(none) == 1/(N+1)" is not a claim to make about that generator.

        `remove_H` and the vocabulary choice deliberately still set the bank
        size: they are configured identically across the arms being compared, so
        they are not a within-experiment intervention the way the prune is.
        """
        return n_surviving + (self._n_static_excluded if self.hold_none_prob else 0)

    def _remove_H_perturbations(self, miou_record):
        """Drop the color/photometric ops from the training pdf (`--no-inv-aug`).

        The op names are vocabulary-specific. For "new" this is Posterize/Solarize,
        as it has always been. For the R-keyed vocabularies those names do not exist
        ("non-diff32"/"diff32" have no Posterize/Solarize at all, since neither has a
        differentiable counterpart), so it is the hue ops themselves -- which is
        also what the flag's name (remove_H) says. Popping nothing at all would
        silently ignore --no-inv-aug for those whole vocabularies.

        Kept even though `update_sa_curve` now excludes the same names at the
        source, because the two guard DIFFERENT paths: that exclusion only
        applies to a curve this loop computed, while `load_sa_curve()` reads one
        off disk that nothing ever filtered. On the `update_sa_curve` path this
        is a no-op -- `dict.pop(name, None)` on names already absent -- so the
        overlap costs nothing and the disk path stays covered.
        """
        if not self.remove_H:
            return
        for name in self._remove_H_names:
            miou_record.pop(name, None)

    def generate_uniform_pdf(self):
        assert self.sa_curve is not None, "SA curve is None in RobustValLoop"
        miou_record, final_metrics = self.test_perturbed_new()

        self._remove_H_perturbations(miou_record)

        # process miou_record into a probability density function
        pdf_dict = {}
        # Two different counts, and conflating them is the bug this splits apart.
        # `n_surviving` is what the perturbation mass is shared BETWEEN; `denom`
        # is what P(none) is computed FROM. They are equal unless
        # --hold-none-prob is on, in which case denom also counts the statically
        # pruned ops and P(none) stops drifting with the size of the bank -- see
        # _none_prob_denominator. With the flag off this is bit-identical to the
        # old `1 / ((N + 1) * num_levels)`.
        n_surviving = len(miou_record.keys())
        denom = self._none_prob_denominator(n_surviving)

        for perturbation, levels in sorted(miou_record.items()):
            num_levels = len(levels.keys())
            for level, _ in levels.items():
                # One fused division rather than (total / n_surviving) / levels:
                # with the flag off `denom == n_surviving` cancels and this is
                # bit-identical to the original `1.0 / ((N + 1) * num_levels)`,
                # where the staged form differs from it by an ULP. That matters
                # only because "the control arm's pdf did not move" has to be
                # checkable by equality rather than by tolerance.
                pdf_dict[(perturbation, level)] = denom / float(
                    (denom + 1) * n_surviving * num_levels
                )

        pdf_dict_perturb_prob = sum(list(pdf_dict.values()))

        # add a "none" perturbation to pdf
        pdf_dict[("none", 0)] = 1.0 - pdf_dict_perturb_prob
        pdf_dict = self._apply_redundancy_reweighting(pdf_dict)
        self.pdf_dict = pdf_dict

        return pdf_dict, final_metrics

    def generate_pdf_new_weighted_aug(self):
        """same as generate_pdf_new, but we dont give uniformity to augmentation classes---they are all weighted by MA."""

        assert self.sa_curve is not None, "SA curve is None in RobustValLoop"
        miou_record, final_metrics = self.test_perturbed_new()

        self._remove_H_perturbations(miou_record)

        # process miou_record into a probability density function
        pdf_dict = {}
        reverse = not self.descending_MA

        all_mious = {}
        for perturbation, levels in sorted(miou_record.items()):
            for i, (level, miou) in enumerate(
                sorted(levels.items(), key=lambda x: x[1], reverse=reverse)
            ):
                all_mious[(perturbation, level)] = miou

        for i, ((perturbation, level), miou) in enumerate(
            sorted(all_mious.items(), key=lambda x: x[1], reverse=reverse)
        ):

            def lambda_bb(x):
                return betabinom.pmf(x, len(all_mious), 0.75, 1.0)

            pdf_dict[(perturbation, level)] = lambda_bb(i) * 0.95

        pdf_dict_perturb_prob = sum(list(pdf_dict.values()))

        # add a "none" perturbation.
        #
        # No _none_prob_denominator call here, and this generator is NOT immune
        # to the drift that flag exists for -- it just drifts for a different
        # reason, which the flag does not address.
        #
        # lambda_bb is a pmf over `len(all_mious)` = (ops x levels) but is only
        # evaluated at i = 0 .. len(all_mious)-1, one short of that pmf's
        # support, so the block sums to slightly under 0.95 and the shortfall
        # falls through to ("none", 0). How short depends on len(all_mious),
        # hence on the op count: P(none) runs ~6.4% at 10 ops and ~5.5% at 30.
        # Same order as the 1/(N+1) drift, different mechanism.
        #
        # Left alone deliberately. Correcting it means deciding how to
        # renormalize a truncated pmf, which changes the SHAPE of the ranking
        # this generator exists to impose rather than just its scale -- a
        # modelling decision, not a bug fix, and one no current arm needs
        # (--weighted-augs is not used by any launch script here). Pinned in
        # tests/test_static_prune.py so it stays on the record.
        pdf_dict[("none", 0)] = 1.0 - pdf_dict_perturb_prob

        if is_main_process():
            print("Perturbation PDF computed:")
            print(
                json.dumps(
                    {str(k): v for k, v in pdf_dict.items()}, indent=4, sort_keys=True
                )
            )

        pdf_dict = self._apply_redundancy_reweighting(pdf_dict)
        self.pdf_dict = pdf_dict

        return pdf_dict, final_metrics

    def generate_pdf_new(self):
        assert self.sa_curve is not None, "SA curve is None in RobustValLoop"
        miou_record, final_metrics = self.test_perturbed_new()

        self._remove_H_perturbations(miou_record)

        # process miou_record into a probability density function
        pdf_dict = {}
        # `num_perturbations` shares the mass out between the ops that actually
        # survived; `denom` is what P(none) = 1/(denom+1) is computed from. Equal
        # unless --hold-none-prob is on -- see _none_prob_denominator -- so with
        # the flag off this is the original arithmetic unchanged.
        num_perturbations = len(miou_record.keys())
        denom = self._none_prob_denominator(num_perturbations)
        perturbation_total_prob = denom / float(denom + 1)
        perturbation_uniform_prob = (
            perturbation_total_prob / num_perturbations if num_perturbations else 0.0
        )

        for perturbation, levels in sorted(miou_record.items()):

            def lambda_bb(x):
                return betabinom.pmf(x, len(levels), 0.75, 1.0)

            # iterate through order of decreasing miou
            reverse = not self.descending_MA
            for i, (level, _) in enumerate(
                sorted(levels.items(), key=lambda x: x[1], reverse=reverse)
            ):
                pdf_dict[(perturbation, level)] = (
                    lambda_bb(i) * perturbation_uniform_prob
                )

        pdf_dict_perturb_prob = sum(list(pdf_dict.values()))

        # add a "none" perturbation
        pdf_dict[("none", 0)] = 1.0 - pdf_dict_perturb_prob

        if is_main_process():
            print("Perturbation PDF computed:")
            print(
                json.dumps(
                    {str(k): v for k, v in pdf_dict.items()}, indent=4, sort_keys=True
                )
            )

        pdf_dict = self._apply_redundancy_reweighting(pdf_dict)
        self.pdf_dict = pdf_dict

        return pdf_dict, final_metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook("before_val_iter", batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        with torch.autocast(device_type="cuda", enabled=self.fp16):
            outputs = self.runner.model.val_step(data_batch)

        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            "after_val_iter", batch_idx=idx, data_batch=data_batch, outputs=outputs
        )

    def _run_eval(self, ratio):
        max_iter = int(len(self.dataloader) * ratio)
        n_samples = 0

        dataloader_iter = iter(self.dataloader)
        idx = 0
        while idx < max_iter:
            data_batch = next(dataloader_iter)
            self.run_iter(idx, data_batch)
            n_samples += len(data_batch["inputs"])
            idx += 1

        # compute metrics
        metrics = self.evaluator.evaluate(n_samples)
        return metrics

    def run(self) -> dict:
        """Launch test."""

        self.runner.call_hook("before_val")
        self.runner.call_hook("before_val_epoch")
        self.runner.model.eval()

        # perturb_metrics = self._run_eval(ratio=1.0)
        perturb_metrics = {}

        if self.n_rounds >= self.warmup_rounds:
            if self.random_aug:
                # Every set is trainable now. Under "diff32" this installs a policy
                # on the model's data preprocessor; under the two CPU sets it
                # rebuilds the dataloader as before. The guard that used to sit here
                # excluded the one measurement-only vocabulary, which no longer
                # exists -- the GPU ops are applied where they were designed to be.
                apply_random_alpha_training_augmentations(  # noqa: F405
                    self.runner,
                    geometric_only=self.geometric_only,
                    photometric_only=self.photometric_only,
                    perturbation_set=self.perturbation_set,
                )

            else:
                self._sa_curve_just_recomputed = (
                    self.n_rounds - self.warmup_rounds
                ) % SA_CURVE_CADENCE == 0
                if self._sa_curve_just_recomputed:
                    self.update_sa_curve()

                if self.sa_curve is not None:
                    _, perturb_metrics = (
                        self.generate_uniform_pdf()
                        if self.uniform
                        else (
                            self.generate_pdf_new_weighted_aug()
                            if self.weighted_augs
                            else self.generate_pdf_new()
                        )
                    )  # update pdf to sample from
                    self.publish_corr_magnitudes()
                    # This is what puts the pdf -- and so the redundancy
                    # reweighting applied to it above -- in front of training.
                    # Under "diff32" it is a preprocessor state update, so the pdf
                    # can change every round without rebuilding a dataloader.
                    apply_random_perturbations_train_dataloader_new(  # noqa: F405
                        self.runner,
                        pdf_dict=self.pdf_dict,
                        perturbation_set=self.perturbation_set,
                    )  # type: ignore

        # full clean evaluation last
        apply_perturbations_dataloader(  # noqa: F405
            self.runner, train=False, perturb_levels={}
        )
        metrics = self._run_eval(ratio=1.0)

        perturb_metrics.update(metrics)

        if is_main_process():
            print("Evaluation results with perturbations: ")
            pprint(perturb_metrics)

            # perturb eval logging
            metrics_str = json.dumps(perturb_metrics)
            with open(self.perturb_metrics_path, "a") as f:
                f.write(metrics_str + "\n")

            print("Train Dataloader config:")
            pprint(self.runner.train_dataloader.dataset.pipeline)

            print("Test Dataloader config:")
            pprint(self.runner.test_dataloader.dataset.pipeline)

        # we log the clean evaluation officially to monitor progress
        self.runner.call_hook("after_val_epoch", metrics=perturb_metrics)
        self.runner.call_hook("after_val")
        self.n_rounds += 1

        return perturb_metrics


@LOOPS.register_module()
class RobustBaselineValLoop(RobustValLoop):
    """Custom loop for val."""

    def __init__(
        self,
        runner,
        dataloader,
        evaluator,
        ratio=0.5,
        sa_curve_path=None,
        fp16: bool = False,
    ) -> None:
        super().__init__(
            runner,
            dataloader,
            evaluator,
            ratio=ratio,
            sa_curve_path=sa_curve_path,
            fp16=fp16,
        )

        self.load_sa_curve()

    def run(self) -> dict:
        """Launch test."""

        self.runner.call_hook("before_val")
        self.runner.call_hook("before_val_epoch")
        self.runner.model.eval()

        _, perturb_metrics = self.test_perturbed()

        # full clean evaluation last
        apply_perturbations_dataloader(  # noqa: F405
            self.runner, train=False, perturb_levels={}
        )
        metrics = self._run_eval(ratio=1.0)

        perturb_metrics.update(metrics)

        if is_main_process():
            print("Evaluation results with perturbations: ")
            pprint(perturb_metrics)

            # perturb eval logging
            metrics_str = json.dumps(perturb_metrics)
            with open(self.perturb_metrics_path, "a") as f:
                f.write(metrics_str + "\n")

            print("Train Dataloader config:")
            pprint(self.runner.train_dataloader.dataset.pipeline)

            print("Test Dataloader config:")
            pprint(self.runner.test_dataloader.dataset.pipeline)

        # we log the clean evaluation officially to monitor progress
        self.runner.call_hook("after_val_epoch", metrics=perturb_metrics)
        self.runner.call_hook("after_val")
        self.n_rounds += 1

        return perturb_metrics
