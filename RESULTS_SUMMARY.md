# SensAug results summary — handoff snapshot

Generated 2026-08-24. All runs: segformer (mit-b0) / Cityscapes / 4-GPU / 160k iters.
Config launch args pulled from job stdout logs (`job_scripts/logs/<job>/*.out`); metrics
pulled from each run's mmengine `.log` file, last `Iter(val)` line.

## ⚠️ Read this first: Lever 3 (redundancy down-weighting) has never actually fired

Every `--aug-type=grad_corr` run so far — completed **and** the 3 currently running —
was launched with `--corr-lambda` unset, so it defaulted to `0.0`. Per the pdf math
(`sensaug/redundancy.py`), **λ=0 short-circuits: the reweighting is a no-op** regardless
of which `--corr-downweight-method` is named. Concretely:

- `grad_corr_grad-corr-soft-1` (completed) and `grad-corr-soft-2` (running) —
  `corr_downweight_method='soft-weighting'`, `corr_lambda=0.0` → bit-identical to no
  downweighting.
- `grad-corr-hard-1` (running, **mRMR**) — `corr_lambda=0.0` → per CLAUDE.md,
  `ceil(A/(1+λ))` survive the prune budget; at λ=0 that's `ceil(A/1)=A`, i.e. **nothing
  gets pruned**. The hard-pruning arm is running as a plain control.

So of the ~4 GPU-days of `grad_corr` compute spent so far (2 completed + 2 in flight),
**none has tested Lever 3 with a real budget.** `job_scripts/train_nexus_gamma.sbatch`
never passes `--corr-lambda` at all — there's no env var for it. If the paper needs an
actual mRMR/soft-weighting result, the next run needs e.g. `--corr-lambda=0.25` added
to the `torchrun` args in that script (see CLAUDE.md's λ-portability table for what
each value does).

## Completed runs

| Run | aug-type | downweight | λ | descending_MA | clean mIoU | mAcc | aAcc | mean perturbed mIoU (32 ops) | best ckpt |
|---|---|---|---|---|---|---|---|---|---|
| `random_random-baseline-1` | random | — | — | — | **75.47** | 83.16 | 95.81 | n/a — plain `ValLoop`, no perturbation sweep logged | iter_160000 |
| `grad_corr_grad-corr-none-highestmiou-2` | grad_corr | none | 0.0 | True | 75.26 | 82.79 | 94.73 | 70.31 | iter_160000 |
| `ours_sensaug-1` | ours | — | — | True | 74.89 | 82.43 | 94.75 | 68.96 | iter_128000 |
| `grad_corr_grad-corr-soft-1` | grad_corr | soft-weighting | 0.0 (no-op, see above) | True | 74.67 | 82.36 | 94.69 | 69.11 | iter_144000 |

All checkpoints live under `experiments/<name>/`.

### Caveats on these numbers

- **`random-baseline-1`'s aAcc (95.81) doesn't match the other three (~94.7)** and it has
  no perturbation-robustness breakdown in its log — it went through a plain `ValLoop`,
  not `RobustValLoop`/`GradCorrValLoop`. Its clean-mIoU lead over the others may not be
  apples-to-apples; don't cite it against the others without checking the eval config
  actually matches.
- **`ours_sensaug-1`'s post-training corruption-robustness sweep (ImageNet-C style, via
  `test_robust.py`/`test.py`) never finished** — the SLURM job hit its time limit mid-sweep
  after training itself completed cleanly at iter_160000. Only the standard 32-op perturbed
  val curve above exists for it; no ACDC/IDD/Dark Zurich/Nighttime numbers.
- **All four runs used `descending_MA=True`** (or n/a) — the "prioritize ops the model
  already handles well" SA direction. The `descending_MA=False` ("worst-mIoU", severe-first)
  counterparts (`*-lowestmiou-2`, `ours_sensaug-2-`) were deleted from `experiments/` earlier
  this session per user request, so there's currently no severe-first arm to compare against.
- **Perturbation-set naming**: CLAUDE.md documents the three vocabularies as `new`/`diff`/
  `aligned`. The code was renamed in commit `cc1e403` ("rename the three perturbation sets")
  and now uses `legacy20`/`non-diff32`/`diff32` — e.g. the launch configs above show
  `perturbation_set='diff32'`. **CLAUDE.md is stale on this point; trust the code.**

## Currently running (started 2026-08-24 ~13:18–13:20, no checkpoints yet)

| Job | ID | aug-type | downweight | λ | Node |
|---|---|---|---|---|---|
| `grad-corr-hard-1` | 7327440 | grad_corr | mRMR | 0.0 (no-op, see warning above) | gammagpu05 |
| `grad-corr-soft-2` | 7327446 | grad_corr | soft-weighting | 0.0 (no-op) | gammagpu06 |
| `ours-none-3` | 7327447 | ours | — | — | gammagpu06 |

None have produced results yet. Given the λ=0 issue above, `grad-corr-hard-1` and
`grad-corr-soft-2` are likely to land within noise of `grad-corr-none-highestmiou-2`
rather than showing any pruning/tilting effect — worth deciding whether to let them
finish as an extra "none" replicate or kill and relaunch with `--corr-lambda` set.

## Deleted this session (for reference — no longer on disk)

- `grad_corr_grad-corr-none-highestmiou-1`, `grad_corr_grad-corr-none-lowestmiou-1` —
  cancelled mid-training (stopped at iter 88000/160000), superseded by the `-2` reruns.
- `grad_corr_grad-corr-none-lowestmiou-2`, `ours_sensaug-2-` — completed, `descending_MA=False`
  ("worst-mIoU") arms, deleted for cleanup at user's request. If a severe-first comparison
  is needed for the paper, these need to be rerun.
