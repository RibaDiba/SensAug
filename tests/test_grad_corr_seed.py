"""Tests for the seeded two-stage mRMR sweep (`job_scripts/grad_corr_seed.{sh,sbatch}`).

The thing these exist to protect is the **handoff**: stage 1 measures R, lets mRMR
hard-prune the training pdf, and prints which ops it dropped; stage 2 has to retrain
with *exactly* those ops statically excluded. That path crosses three format
boundaries -- an mmengine log line, a text file, and an argv list -- so a silent
loss is entirely plausible and would not show up as a failure anywhere: stage 2
would just train on a bank that was never pruned and quietly answer a different
question.

So the central assertion, repeated in several shapes below, is set equality
between the ops named on the log line and the ops in stage 2's
`--pruned-augmentations` argv. Not a count, not a prefix -- the set.

These drive the real bash scripts through `subprocess` with stub `torchrun` /
`sbatch` / `nvidia-smi` on PATH, rather than re-implementing the scrape in Python.
Re-implementing it would pass happily while the shipped `sed` did something else,
which is the exact class of bug at issue: the real log lines carry a NESTED
parenthetical --

    ... (budget 24 of 29 measured, 1 exempt (no usable row in R)): darker_B, ...

-- and a non-greedy match would leave `no usable row in R): darker_B` as the first
"op". `test_scraped_ops_are_real_augmentation_names` is what pins that down: every
scraped token has to be a key in the actual `DIFF32_OPS` registry, so garbage from
a mis-parse cannot pass as an op name.

Sample log lines below are copied verbatim from real finished runs under
`archive/` (`grad_corr_grad-corr-hard-{1,ade20k-1}_segformer_*`).
"""

import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

SBATCH_SCRIPT = os.path.join(REPO_ROOT, "job_scripts", "grad_corr_seed.sbatch")
SH_SCRIPT = os.path.join(REPO_ROOT, "job_scripts", "grad_corr_seed.sh")

# Verbatim from archive/grad_corr_grad-corr-hard-ade20k-1_segformer_ade20k_4gpu_gradcorr.
ADE20K_OPS = ["darker_B", "darker_R", "darker_S", "lighter_R", "sharpness_neg"]
# Verbatim from archive/grad_corr_grad-corr-hard-1_segformer_cityscapes_4gpu_gradcorr.
CITYSCAPES_OPS = ["color_neg", "contrast_neg", "lighter_V", "sharpness_neg", "sharpness_pos"]


def _log_line(ops, lam="0.25", n_exempt=1):
    """One `[redundancy] mRMR pruned` line in the exact shape the hook emits."""
    return (
        "2026/08/26 07:54:07 - mmengine - INFO - [redundancy] mRMR pruned "
        f"{len(ops)}/30 ops at lambda={lam} (budget {30 - len(ops)} of 29 measured, "
        f"{n_exempt} exempt (no usable row in R)): " + ", ".join(ops)
    )


# --------------------------------------------------------------------------- stubs


STUB_TORCHRUN = """#!/bin/bash
# Record argv one entry per line, so args containing spaces stay distinguishable.
n=0
while [ -e "${STUB_DIR}/torchrun.${n}.argv" ]; do n=$((n+1)); done
printf '%s\\n' "$@" > "${STUB_DIR}/torchrun.${n}.argv"
exit 0
"""

STUB_SBATCH = """#!/bin/bash
n=0
while [ -e "${STUB_DIR}/sbatch.${n}.argv" ]; do n=$((n+1)); done
printf '%s\\n' "$@" > "${STUB_DIR}/sbatch.${n}.argv"
echo "Submitted batch job 999${n}"
exit 0
"""

STUB_NVIDIA_SMI = """#!/bin/bash
for i in 0 1 2 3; do echo "GPU $i: NVIDIA RTX A6000 (UUID: GPU-stub-$i)"; done
exit 0
"""

STUB_NOOP = """#!/bin/bash
exit 0
"""


@pytest.fixture
def harness(tmp_path):
    """A tmp work_dir plus a stub bin/ that shadows torchrun, sbatch and friends."""
    stub_dir = tmp_path / "stubs"
    bin_dir = tmp_path / "bin"
    work_dir = tmp_path / "experiments"
    log_base = tmp_path / "logs"
    for d in (stub_dir, bin_dir, work_dir, log_base):
        d.mkdir(parents=True, exist_ok=True)

    for name, body in (
        ("torchrun", STUB_TORCHRUN),
        ("sbatch", STUB_SBATCH),
        ("nvidia-smi", STUB_NVIDIA_SMI),
        # `module` is normally a shell function and `conda` a real binary; neither
        # exists in a bare test shell. Stub both so the scripts' setup lines are
        # exercised rather than merely surviving a "command not found".
        ("module", STUB_NOOP),
        ("conda", STUB_NOOP),
    ):
        p = bin_dir / name
        p.write_text(body)
        p.chmod(0o755)

    env = dict(os.environ)
    env.update(
        # A MINIMAL PATH, not the inherited one: if a real torchrun or conda were
        # reachable, a test could launch actual multi-GPU training on whatever node
        # the suite runs on. Everything the scripts need beyond the stubs (awk, sed,
        # grep, ls) lives in /usr/bin:/bin.
        PATH=f"{bin_dir}:/usr/bin:/bin",
        STUB_DIR=str(stub_dir),
        REPO_DIR=REPO_ROOT,
        SELF_PATH=SBATCH_SCRIPT,
        WORK_DIR=str(work_dir),
        LOG_BASE=str(log_base),
        SLURM_JOB_ID="1234",
        SKIP_EVAL="1",  # keep the assertions on train + handoff, not the eval legs
    )
    # Inherited values would leak into every run; these must come from the test.
    for stale in ("STAGE", "PRUNE_MODE", "NO_STAGE2", "REQUIRE_PRUNE",
                  "PRUNED_AUGS_FILE", "ALLOW_RESUME", "DATASETS", "GPU_TYPE"):
        env.pop(stale, None)

    class Harness:
        def __init__(self):
            self.env = env
            self.work_dir = work_dir
            self.stub_dir = stub_dir

        def seed_log(self, exp_name, lines):
            """Write the mmengine log stage 1 will scrape, where it looks for it."""
            d = work_dir / exp_name / "20260101_000000"
            d.mkdir(parents=True, exist_ok=True)
            (d / "run.log").write_text("\n".join(lines) + "\n")

        def run_stage(self, stage, seed=1, run_iter=1, dataset="loveda",
                      backbone="segformer", **extra_env):
            e = dict(self.env, STAGE=str(stage), **{k: str(v) for k, v in extra_env.items()})
            return subprocess.run(
                ["bash", SBATCH_SCRIPT, str(seed), str(run_iter), dataset, backbone],
                capture_output=True, text=True, env=e, cwd=REPO_ROOT,
            )

        def argv(self, kind, n=0):
            p = self.stub_dir / f"{kind}.{n}.argv"
            if not p.exists():
                return None
            return p.read_text().splitlines()

    return Harness()


def _flag_values(argv, flag):
    """Values following `flag` up to the next `--option` (mimics nargs='+')."""
    if flag not in argv:
        return None
    out = []
    for tok in argv[argv.index(flag) + 1:]:
        if tok.startswith("--"):
            break
        out.append(tok)
    return out


# --------------------------------------------------------------- the handoff


def test_pruned_ops_reach_stage_two_argv(harness):
    """The whole point: every op mRMR pruned in stage 1 is excluded in stage 2."""
    stage1 = "seeded_1_loveda_segformer-1_gradcorr"
    harness.seed_log(stage1, [_log_line(ADE20K_OPS)])

    r1 = harness.run_stage(1)
    assert r1.returncode == 0, r1.stderr

    # 1. log line -> file
    ops_file = harness.work_dir / stage1 / "mrmr_pruned_ops.txt"
    assert ops_file.exists(), "stage 1 wrote no mrmr_pruned_ops.txt"
    assert ops_file.read_text().split() == ADE20K_OPS

    # 2. stage 2 was queued, gated on this job, under the derived name
    sb = harness.argv("sbatch")
    assert sb is not None, "stage 1 never submitted stage 2"
    assert "--dependency=afterok:1234" in sb
    assert sb[sb.index("-J") + 1] == "seeded_1_loveda_segformer-1_pruned_ours"
    assert sb[-4:] == ["1", "1", "loveda", "segformer"]

    # 3. file -> stage 2's train.py argv. Set equality, not a count or a prefix.
    r2 = harness.run_stage(2)
    assert r2.returncode == 0, r2.stderr
    argv = harness.argv("torchrun", 1)  # 0 is stage 1's own train call
    assert argv is not None, "stage 2 never invoked torchrun"

    transferred = _flag_values(argv, "--pruned-augmentations")
    assert transferred is not None, "stage 2 ran without --pruned-augmentations"
    assert set(transferred) == set(ADE20K_OPS)
    assert len(transferred) == len(ADE20K_OPS), "an op was duplicated in transit"
    assert "--aug-type=ours" in argv
    assert "--exp_name=seeded_1_loveda_segformer-1_pruned_ours" in argv


def test_scraped_ops_are_real_augmentation_names(harness):
    """Guards the nested `(...)` in the log line: a mis-parse yields non-ops.

    `1 exempt (no usable row in R)` sits between the budget and the op list, so a
    non-greedy match leaks `no usable row in R): darker_B` into the first slot.
    Checking membership in the live registry is what makes that impossible to
    miss -- a count or a prefix check would pass.
    """
    from sensaug.dataset.differentiable_augmentations_aa import DIFF32_OPS

    stage1 = "seeded_1_loveda_segformer-1_gradcorr"
    harness.seed_log(stage1, [_log_line(CITYSCAPES_OPS)])
    assert harness.run_stage(1).returncode == 0

    scraped = (harness.work_dir / stage1 / "mrmr_pruned_ops.txt").read_text().split()
    assert scraped, "nothing scraped"
    unknown = [op for op in scraped if op not in DIFF32_OPS]
    assert not unknown, f"scrape produced non-ops (log-line parse is wrong): {unknown}"
    assert set(scraped) == set(CITYSCAPES_OPS)


def test_last_line_wins_because_the_prune_is_not_latched(harness):
    """mRMR re-derives its prune every round; the final round is the decision.

    With `--corr-lambda-ramp linear` (the default) later rounds prune harder, so
    scraping anything but the last line would retrain against a budget the run had
    already moved past.
    """
    stage1 = "seeded_1_loveda_segformer-1_gradcorr"
    early, late = ["blur"], ["blur", "noise", "darker_R"]
    harness.seed_log(stage1, [
        _log_line(early, lam="0.10"),
        _log_line(["blur", "noise"], lam="0.19"),
        _log_line(late, lam="0.25"),
    ])
    assert harness.run_stage(1).returncode == 0

    ops = (harness.work_dir / stage1 / "mrmr_pruned_ops.txt").read_text().split()
    assert set(ops) == set(late)


def test_union_mode_collects_every_round(harness):
    """PRUNE_MODE=union is the 'anything ever pruned' variant; it must lose nothing."""
    stage1 = "seeded_1_loveda_segformer-1_gradcorr"
    rounds = [["blur"], ["noise", "darker_R"], ["blur", "lighter_V"]]
    harness.seed_log(stage1, [_log_line(r) for r in rounds])
    assert harness.run_stage(1, PRUNE_MODE="union").returncode == 0

    ops = (harness.work_dir / stage1 / "mrmr_pruned_ops.txt").read_text().split()
    assert set(ops) == {"blur", "noise", "darker_R", "lighter_V"}
    assert len(ops) == 4, "union should de-duplicate, not repeat"


def test_large_prune_set_transfers_whole(harness):
    """A wide prune is where word-splitting or truncation would first show up."""
    from sensaug.dataset.differentiable_augmentations_aa import DIFF32_OPS

    many = sorted(DIFF32_OPS)[:12]
    stage1 = "seeded_2_ade20k_segformer-3_gradcorr"
    harness.seed_log(stage1, [_log_line(many)])
    assert harness.run_stage(1, seed=2, run_iter=3, dataset="ade20k").returncode == 0
    assert harness.run_stage(2, seed=2, run_iter=3, dataset="ade20k").returncode == 0

    transferred = _flag_values(harness.argv("torchrun", 1), "--pruned-augmentations")
    assert set(transferred) == set(many)
    assert len(transferred) == 12


def test_no_prune_means_no_flag_not_an_empty_flag(harness):
    """An empty prune must omit the flag; `--pruned-augmentations` with no values
    would make train.py's nargs='+' swallow the next flag."""
    stage1 = "seeded_1_loveda_segformer-1_gradcorr"
    harness.seed_log(stage1, ["2026/08/26 - mmengine - INFO - nothing to see here"])
    assert harness.run_stage(1).returncode == 0
    assert harness.run_stage(2).returncode == 0

    argv = harness.argv("torchrun", 1)
    assert "--pruned-augmentations" not in argv
    assert "--aug-type=ours" in argv


def test_require_prune_refuses_to_chain_an_empty_prune(harness):
    stage1 = "seeded_1_loveda_segformer-1_gradcorr"
    harness.seed_log(stage1, ["2026/08/26 - mmengine - INFO - no prune here"])
    r = harness.run_stage(1, REQUIRE_PRUNE="1")
    assert r.returncode != 0
    assert harness.argv("sbatch") is None, "stage 2 was queued despite REQUIRE_PRUNE"


# ------------------------------------------------------ arms, names, pinning


def test_stage_one_is_grad_corr_mrmr_over_the_full_bank(harness):
    """Stage 1 must not carry a static prune of its own: mRMR has to rank the whole
    30-op bank, or stage 2 inherits a cut that was never measured."""
    harness.seed_log("seeded_1_loveda_segformer-1_gradcorr", [_log_line(ADE20K_OPS)])
    assert harness.run_stage(1).returncode == 0

    argv = harness.argv("torchrun", 0)
    assert "--aug-type=grad_corr" in argv
    assert "--corr-downweight-method=mRMR" in argv
    assert "--corr-lambda=0.25" in argv
    assert "--pruned-augmentations" not in argv, (
        "stage 1 pruned ops before measuring R"
    )
    # The flags every Nexus run shares, so the two arms stay comparable.
    for flag in ("--no-inv-aug", "--auto-scale-lr", "--descending-MA"):
        assert flag in argv


def test_no_control_arm_is_ever_launched(harness):
    """This sweep is grad_corr -> pruned ours and nothing else."""
    harness.seed_log("seeded_1_loveda_segformer-1_gradcorr", [_log_line(ADE20K_OPS)])
    assert harness.run_stage(1).returncode == 0
    assert harness.run_stage(2).returncode == 0

    stage1_argv, stage2_argv = harness.argv("torchrun", 0), harness.argv("torchrun", 1)
    for argv in (stage1_argv, stage2_argv):
        for banned in ("--no-corr-sa", "--random-prune-method=null",
                       "--corr-downweight-method=none", "--corr-lambda=0",
                       "--corr-lambda=0.0", "--uniform", "--random-aug"):
            assert banned not in argv, f"control-arm flag {banned} leaked in"
    # Exactly two jobs' worth of training, and only one chained submission.
    assert harness.argv("torchrun", 2) is None
    assert harness.argv("sbatch", 1) is None


def test_exp_names_match_the_scheme_and_need_no_train_py_suffix(harness):
    """train.py appends `_gradcorr`/`_ours` to names lacking them (train.py:1637-1646);
    baking the suffixes in is what keeps -J, the log dir and the work_dir equal."""
    harness.seed_log("seeded_7_pascal_voc12_segformer-2_gradcorr", [_log_line(ADE20K_OPS)])
    assert harness.run_stage(1, seed=7, run_iter=2, dataset="pascal_voc12").returncode == 0

    argv = harness.argv("torchrun", 0)
    assert "--exp_name=seeded_7_pascal_voc12_segformer-2_gradcorr" in argv
    # Idempotent guard in train.py: the suffix is already present, so nothing is added.
    exp = [a for a in argv if a.startswith("--exp_name=")][0].split("=", 1)[1]
    assert "gradcorr" in exp and exp.count("gradcorr") == 1


def test_stage_two_is_pinned_to_the_same_gpu_model(harness):
    """Seeds are label-only, so hardware is the one variable that could masquerade
    as run-to-run spread -- both legs must land on the same GPU model."""
    harness.seed_log("seeded_1_loveda_segformer-1_gradcorr", [_log_line(ADE20K_OPS)])
    assert harness.run_stage(1, GPU_TYPE="l40s", N_GPUS="4").returncode == 0

    sb = harness.argv("sbatch")
    assert "--gres=gpu:l40s:4" in sb, (
        "stage 2 did not re-pass --gres; it would fall back to the #SBATCH default"
    )


@pytest.mark.parametrize("bad", ["abc", "-1", "1.5", ""])
def test_seed_and_run_iteration_must_be_integers(harness, bad):
    """A non-numeric arg would flow straight into the directory name."""
    assert harness.run_stage(1, seed=bad).returncode != 0
    assert harness.run_stage(1, run_iter=bad).returncode != 0


# ------------------------------------------------------- environment setup


def test_runs_without_module_in_the_environment(harness):
    """Regression for jobs 7554655/7554656, which died at 1s with exit 127.

    `module` is a shell FUNCTION from /etc/profile.d/modules.sh. An interactive
    login shell exports it, so a hand-submitted job inherits it through
    --export=ALL and `module load miniforge3` works. A job submitted from a
    non-interactive shell inherits nothing, and the original script's bare
    `module load` then cascaded: no module -> no conda -> no torchrun -> 127.

    So: delete the `module` stub and the run must still reach the handoff.

    MODULE_INIT_FILES="" keeps this hermetic. Sourcing the real
    /etc/profile.d/modules.sh would let `module load miniforge3` prepend the real
    miniforge bin AHEAD of the stub dir, so `conda activate sensaug` and then
    torchrun would both resolve to the real thing and the test would launch
    training for real -- which is exactly what happened the first time this test
    was written.
    """
    (harness.stub_dir.parent / "bin" / "module").unlink()

    harness.seed_log("seeded_1_loveda_segformer-1_gradcorr", [_log_line(ADE20K_OPS)])
    r = harness.run_stage(1, MODULE_INIT_FILES="")

    assert r.returncode == 0, f"a missing `module` still breaks the run:\n{r.stderr}"
    assert harness.argv("torchrun", 0) is not None, "train never launched"
    assert harness.argv("sbatch") is not None, "stage 2 never queued"


def test_missing_torchrun_fails_loudly_before_training(harness):
    """A bare exit 127 out of torchrun is what made the original failure
    unreadable. Check the environment and say so instead."""
    bin_dir = harness.stub_dir.parent / "bin"
    (bin_dir / "torchrun").unlink()
    # Minimal PATH: guarantees the test can never find a REAL torchrun and
    # accidentally launch training.
    env = dict(harness.env, PATH=f"{bin_dir}:/usr/bin:/bin", STAGE="1", MODULE_INIT_FILES="")

    r = subprocess.run(
        ["bash", SBATCH_SCRIPT, "1", "1", "loveda", "segformer"],
        capture_output=True, text=True, env=env, cwd=REPO_ROOT,
    )
    assert r.returncode != 0
    assert "torchrun not on PATH" in r.stderr
    assert harness.argv("sbatch") is None, "stage 2 queued despite a broken environment"


# ------------------------------------------------------------- the submitter


def _dry_run(args, **extra_env):
    env = dict(os.environ, DRY_RUN="1")
    env.update({k: str(v) for k, v in extra_env.items()})
    env.pop("DATASETS", None) if "DATASETS" not in extra_env else None
    return subprocess.run(["bash", SH_SCRIPT, *args],
                          capture_output=True, text=True, env=env, cwd=REPO_ROOT)


def test_submitter_emits_one_stage_one_job_per_seed_and_dataset():
    r = _dry_run(["2", "1"], DATASETS="cityscapes loveda")
    assert r.returncode == 0, r.stderr
    lines = [l for l in r.stdout.splitlines() if l.startswith("sbatch ")]
    assert len(lines) == 4
    for seed in ("1", "2"):
        for ds in ("cityscapes", "loveda"):
            name = f"seeded_{seed}_{ds}_segformer-1_gradcorr"
            assert any(f"-J {name} " in l for l in lines), f"missing {name}"
    # Only stage 1 is submitted here; stage 2 chains itself.
    assert not any("pruned_ours" in l for l in lines)


def test_submitter_never_offers_the_untrainable_datasets():
    """acdc and idd pass a folder-exists check and then die -- acdc on
    FileNotFoundError for data/acdc/leftImg8bit/val, idd with no base config."""
    r = _dry_run(["1", "1"])
    assert r.returncode == 0, r.stderr
    assert "acdc" not in r.stdout
    assert "idd" not in r.stdout


def test_submitter_pins_the_gpu_on_every_job():
    r = _dry_run(["1", "1"], DATASETS="loveda")
    lines = [l for l in r.stdout.splitlines() if l.startswith("sbatch ")]
    assert lines and all("--gres=gpu:rtxa6000:4" in l for l in lines)

    r = _dry_run(["1", "1", "segformer", "rtxa5000"], DATASETS="loveda")
    lines = [l for l in r.stdout.splitlines() if l.startswith("sbatch ")]
    assert lines and all("--gres=gpu:rtxa5000:4" in l for l in lines)


def test_submitter_rejects_bad_arguments():
    assert _dry_run(["0", "1"]).returncode != 0          # n_seeds must be positive
    assert _dry_run(["x", "1"]).returncode != 0
    assert _dry_run(["1", "y"]).returncode != 0
    assert _dry_run(["1"]).returncode != 0               # run_iteration required
    assert _dry_run(["1", "1", "not_a_backbone"]).returncode != 0
    assert _dry_run(["1", "1", "--nope"]).returncode != 0
