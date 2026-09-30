# Changelog

Debug log for getting a `train_nexus_gamma.sbatch` job running on the UMD Nexus
`gamma` partition, under `amodak`'s own checkout at `/gammascratch/amodak/SensAug`.
Newest entries first.

## 2026-08-24

### The gradient cross-correlation pipeline now fires on the SA rounds, from `configs/rounds.yaml`

Not a bug fix — a scheduling change. R was emitted every `corr_interval` iterations,
defaulting to `max_iters // 4`: four matrices at 25 / 50 / 75 / 100% of training,
placed with no reference to the thing they exist to feed. Lever 3 reads
`runner.corr_redundancy` from inside the val loop, which rebuilds the pdf every
`round_interval` off an SA curve recomputed only every 6th post-warmup round. So an R
could be measured mid-curve and sit unread for several rounds, and the last one — at
`max_iters`, emitted by the "final iteration always fires" rule — could never be read
at all, because training ends before another pdf is built.

The default is now the round grid: at 20 rounds with 4 warmup, rounds **3, 4, 10, 16**.

| rounds | what |
|---|---|
| 0–2 | nothing |
| 3 | one control probe (baseline, does not feed the pdf) |
| 4 | compute, used for rounds 4–9 |
| 10 | compute, used for rounds 10–15 |
| 16 | compute, used for rounds 16–18 |
| 19 | nothing (training's over) |

4 / 10 / 16 are the SA-curve recompute rounds, so each matrix stays current for exactly
the rounds that curve governs. Round `r`'s sweep fires at iteration
`(r+1) × round_interval`, which `IterBasedTrainLoop` reaches one hook point *before*
round `r`'s val loop — so round `r`'s pdf is reweighted by R measured on round `r`'s own
model. Round 3 is the last warmup round: R measured before any pdf existed, recorded
with `"role": "control"` in `corr_matrix_log.json` and published to nothing.

New `sensaug/round_schedule.py` (pure integers + yaml, no torch/mmseg) holds the
arithmetic and parses the new `configs/rounds.yaml` (`--rounds-config`), which is
deliberately *not* part of the cluster configs — the round grid is an experiment
parameter, and a copy in each of `della.yaml` / `nexus.yaml` is a copy that drifts.
`n_rounds` / `warmup_rounds` derive the schedule; `corr_rounds:` / `control_rounds:`
pin it literally. The emission count follows the grid rather than being fixed at four:
`--round_interval=2000` on an 80k run gives 40 rounds and 7 emissions.

Escape hatches, in precedence order: `--corr-sync-sa` (every round), `--corr-interval N`
(the old fixed clock, including its final-iteration rule). `SA_CURVE_CADENCE = 6` moved
out of the val loop's `% 6` literal into `round_schedule.py`, so the cadence and the
schedule built from it cannot disagree; the value is unchanged.

Tests: `tests/test_round_schedule.py` (25, pure), plus schedule-gate and control-probe
cases in `tests/test_grad_hook.py` and `tests/test_grad_sens_analysis.py`.

## 2026-08-22

### `NameError: DIFFERENTIABLE_PERTURBATIONS` in `publish_corr_magnitudes()` (jobs `grad-corr-none-1` `7305081`, `sensaug-baseline-2` `7305045`, `sensaug-baseline-3` `7305080`)

All three killed at the very last step of a full-length (40k-iter) run — checkpoints,
`sa_curve_log.txt`, `perturb_eval.txt`, and complete per-class mIoU tables for every
round all made it to disk, then the final round-eval died:

```
NameError: name 'DIFFERENTIABLE_PERTURBATIONS' is not defined
```

**Cause:** `sensaug/loops/sensaug_loop.py` (`RobustValLoop.publish_corr_magnitudes()`,
line 306) referenced `DIFFERENTIABLE_PERTURBATIONS` bare, counting on one of the file's
two wildcard imports (`sensaug.sensitivity_analysis import *`, `sensaug.runner_utils
import *`) to bring it into scope. Neither module re-exports it — it's only ever
defined in `sensaug/dataset/differentiable_augmentations.py`. This path is gated behind
`self._is_corr_vocabulary and self.pdf_dict` (`sensaug_loop.py:303`), both only true
once the run is deep enough to have a real pdf under `"non-diff32"`/`"diff32"`, which is
why every affected run had already finished all 40k iterations before hitting it —
nothing exercised this line at low iteration counts or in `tests/`.

Fixing the missing import naively (`from sensaug.dataset.differentiable_augmentations
import DIFFERENTIABLE_PERTURBATIONS`) would have "fixed" the crash while silently
breaking the snapshot: `DIFFERENTIABLE_PERTURBATIONS` is only 14 ops by design (see
`differentiable_augmentations_aa.py`'s comment on it) — the full 32-op vocabulary R is
actually indexed by, and the one `CollectGradientHook`/
`PerturbationSensitivityAnalysisHookWithGradients` use throughout `sensaug/hooks/`, is
`DIFF32_OPS` (`differentiable_augmentations_aa.py`). Importing the 14-op name would
have quietly dropped 18 of 32 ops from every published `corr_magnitudes.json` snapshot
going forward, with no error or log line to flag it.

**Fix:** `sensaug_loop.py` now imports `DIFF32_OPS` from
`sensaug.dataset.differentiable_augmentations_aa` and `publish_corr_magnitudes()` keys
`conditional_levels()` off `set(DIFF32_OPS)` (32 ops), matching every other R-consuming
call site. Also dropped the stale `# noqa: F405` (was suppressing a wildcard-import
lint warning that no longer applies now that the name is a real import) and corrected
the docstring's own reference to the old 14-op name. Verified: `py_compile` on the
file, a standalone import of `DIFF32_OPS` at module load time, and the existing
`tests/test_corr_magnitudes.py` (16 tests, all pass — `conditional_levels()` itself is
untouched, only the op-name set passed into it changed).

## 2026-08-21

### `GpuAugSegDataPreProcessor._apply()` shadowed `nn.Module._apply()` (job `sensaug-baseline-1`, `7305023`)

`--aug-type=grad_corr` crashed inside `Runner.wrap_model()` → `model.to(device)`:

```
TypeError: GpuAugSegDataPreProcessor._apply() missing 2 required positional arguments: 'labels' and 'specs'
```

**Cause:** `sensaug/dataset/gpu_augment.py` (new in `cc1e403`, 2026-08-16 — see commit
for the full rationale: it replaced the old CPU per-image pipeline-transform path with
a GPU-batched `SegDataPreProcessor` subclass, to fix the SA-round-eval walltime problem
elsewhere in this changelog) named its batch-apply method `_apply`. `SegDataPreProcessor`
extends `nn.Module`, which reserves `_apply(self, fn)` for the internal recursive walk
`.to()`/`.cuda()`/`.half()` use to push a conversion function through every submodule.
The old CPU transforms were never `nn.Module` submodules, so this name was never live
before; the new class is a real submodule (`model.data_preprocessor`), so
`model.to(device)` calls `module._apply(fn)` on it directly and collides with the
3-argument override.

No test in `tests/` references `gpu_augment.py` or `GpuAugSegDataPreProcessor`, and the
one test that touches `.data_preprocessor` (`test_grad_hook.py`) uses a hand-rolled fake
that never calls `nn.Module.to()` — so this could only surface at a real training launch,
which is what this job was the first of.

**Fix:** renamed the method to `_apply_ops` (`gpu_augment.py:220` and its one call site
at `gpu_augment.py:290` — verified no other callers repo-wide). Pure rename, no signature
or behavior change. Verified with a standalone `model.to('cpu')` smoke test.

### `tensorboard`/`future` missing (job `sensaug-test-2`, `7304801`)

Past `pkg_resources` (below), `Runner.__init__` now reached
`self.visualizer.add_config()`, which mmengine's TensorBoard vis-backend needs, and
crashed:

```
ModuleNotFoundError: No module named 'tensorboard'
ImportError: Please run "pip install future tensorboard" to install the dependencies
to use torch.utils.tensorboard
```

Neither package is in `requirements.txt`, but mmengine's TensorBoard backend (used
for the tensorboard logs mentioned in the README) needs both.

**Fix:** `pip install future tensorboard` into `sensaug`.

### `pkg_resources` missing at runtime (job `sensaug-test-2`, `7304793`)

`torchrun` launched and reached `Runner.from_cfg()`, but every rank crashed inside
`mmengine`'s `collect_env()` → `torch.utils.cpp_extension` → `from pkg_resources
import packaging`:

```
ModuleNotFoundError: No module named 'pkg_resources'
```

**Cause:** the `sensaug` conda env has `setuptools==84.0.0`, which no longer bundles
`pkg_resources` — the same underlying break that caused the earlier `mmcv` source-build
failure, resurfacing here because `torch`'s C++/CUDA extension loader still imports it
unconditionally.

**Fix:** pin `setuptools<81` in the `sensaug` env, which still bundles `pkg_resources`.

### `IndexError: pop from empty list` in `build_config()` (job `sensaug-test-1`, `7304383`)

`configs/nexus.yaml`'s `mmconfig_path` pointed at
`/fs/nexus-scratch/lyzheng/sensaug/sensaug/custom_configs/mmseg` — Laura's personal
scratch checkout, which `amodak` has no read permission on. `glob.glob()` on a
directory it can't read returns `[]` silently rather than raising, so both the
dataset-specific and any-`.py`-fallback globs in `build_config()` (`train.py`) came
back empty, and `mm_configs.pop(0)` threw on the empty list.

**Fix:** repointed `mmconfig_path` at `amodak`'s own checkout:
`/gammascratch/amodak/SensAug/sensaug/custom_configs/mmseg`.

Same job also surfaced that `data_root: /fs/nexus-projects/robustness_datasets/segmentation`
does not exist on Nexus at all (not a permissions issue — the path is simply absent).
**Fix:** repointed `data_root` at `/gammascratch/amodak/SensAug/data` (94GB free) and
populated `data/cityscapes/` there via `csDownload` + `csCreateTrainIdLabelImgs`
(mirrors the same `data_root/cityscapes/{gtFine,leftImg8bit}` relative layout Della uses).

### `torchrun: command not found` (job `sensaug_gamma`, `7304083`)

The `sensaug` conda env existed but was created with nothing installed
(`conda create -n sensaug python=3.10` only — no torch, no mmcv, no mmengine).
`conda activate sensaug` succeeded silently, so the job ran for 3 seconds before dying
on `torchrun: command not found` (exit 127).

Also found while diagnosing: `conda` itself is not on PATH on gamma compute nodes.
The job had only worked before by inheriting miniforge's bin dir from the submitting
shell's PATH via SLURM's `--export=ALL` default — a fresh shell would fail differently.

**Fix:**
- Installed the full stack into `sensaug` (torch 2.0.0/cu117, mmcv 2.1.0 via the
  openmmlab prebuilt-wheel index, mmengine, mmsegmentation, mmpretrain,
  `requirements.txt`, `pip install -e .`), matching the versions pinned in `Dockerfile`.
- Hardened `job_scripts/train_nexus_gamma.sbatch`: explicit `module load miniforge3`
  instead of relying on inherited PATH, a `torchrun`-on-PATH pre-flight check with an
  actionable error message, and a `nvidia-smi`-returns-nothing guard so a failed GPU
  probe can't silently pass `--nproc_per_node=0` to torchrun.
