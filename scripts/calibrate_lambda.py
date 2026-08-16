#!/usr/bin/env python
"""Calibrate the redundancy down-weighting strength (lambda) against R matrices
that have already been logged. No training, no GPU, no checkpoint.

sensaug/redundancy.py standardizes red(a) before exponentiating, which is what is
supposed to make one lambda mean the same thing across runs, checkpoints and
vocabulary sizes. This script is how that claim gets checked: it reads the
corr_matrix_log.json files an --aug-type=grad_corr run already wrote, builds red(a) at each
checkpoint, and reports the max/min spread the reweighting would induce.

The number to watch is the spread column. It should be near-constant DOWN each
column -- same lambda, same spread, whichever run and whichever checkpoint. If it
is not, the standardization is not doing its job and nothing downstream of it can
be trusted; that is the script's real purpose, more than picking a value.

Usage:
    python scripts/calibrate_lambda.py
    python scripts/calibrate_lambda.py --logs 'experiments/*/corr_matrix_log.json'
    python scripts/calibrate_lambda.py --mode signed --top 8
"""

import argparse
import glob
import json
import os
import sys

# Repo root, so `sensaug` resolves whether this is run as
# `python scripts/calibrate_lambda.py` from the repo root or from elsewhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sensaug.redundancy import MODES, compute_red, reweight

DEFAULT_LAMBDAS = (0.1, 0.25, 0.5, 1.0, 2.0)


def load_records(pattern):
    """Every (label, record) pair from the matching corr_matrix_log.json files.

    The log is a single JSON array rather than JSONL, and every record carries its
    own `names`, so the axis order is self-describing on disk and matrices of
    different vocabulary sizes can be compared side by side.
    """
    out = []
    for path in sorted(glob.glob(pattern)):
        label = os.path.basename(os.path.dirname(path))
        try:
            with open(path) as f:
                records = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  ! skipping {path}: {exc}", file=sys.stderr)
            continue
        for record in records:
            out.append((label, record))
    return out


def matrix_of(record):
    """The matrix the pipeline would actually act on, NaN-restored.

    _jsonable writes NaN as null (json.dump is called with allow_nan=False), so the
    dropped rows and columns come back as None and have to be put back as NaN --
    compute_red distinguishes "dropped upstream" from "correlates with nothing".
    """
    rows = record.get("R_scalenorm") or record["R_raw"]
    return np.array(
        [[np.nan if v is None else v for v in row] for row in rows], dtype=np.float64
    )


def uniform_pdf(names):
    """A stand-in for the training pdf: one entry per op plus the reserved
    ("none", 0) mass, shaped the way generate_pdf_new shapes it.

    Uniform on purpose. The spread this reports is then the spread the reweighting
    itself induces, not one inherited from whatever the SA curve happened to
    produce that round.
    """
    none_mass = 1.0 / (len(names) + 1)
    per_op = (1.0 - none_mass) / len(names)
    pdf = {(name, 0.5): per_op for name in names}
    pdf[("none", 0)] = none_mass
    return pdf


def main():
    parser = argparse.ArgumentParser(
        description="Sweep lambda against logged correlation matrices."
    )
    parser.add_argument(
        "--logs",
        default="experiments/*/corr_matrix_log.json",
        help="glob for corr_matrix_log.json files (default: %(default)s)",
    )
    parser.add_argument(
        "--mode",
        default="squared",
        choices=list(MODES) + ["all"],
        help="red(a) reduction; 'all' sweeps every mode (default: %(default)s)",
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=list(DEFAULT_LAMBDAS),
        help="lambda values to sweep (default: %(default)s)",
    )
    parser.add_argument(
        "--target-spread",
        type=float,
        default=5.0,
        help="max(pdf)/min(pdf) to recommend a lambda for (default: %(default)s)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="how many extreme ops to name per checkpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--no-mask-within-op",
        action="store_true",
        help="keep the lighter/darker and _pos/_neg cells in red(a)",
    )
    args = parser.parse_args()

    records = load_records(args.logs)
    if not records:
        print(f"no records matched {args.logs!r}", file=sys.stderr)
        return 1

    modes = list(MODES) if args.mode == "all" else [args.mode]
    mask_within_op = not args.no_mask_within_op

    for mode in modes:
        print(f"\n{'=' * 78}")
        print(f"red mode: {mode}   within-op cells: {'masked' if mask_within_op else 'kept'}")
        print("=" * 78)

        header = f"{'run':<30}{'ckpt':>6}{'A':>4}  " + "".join(
            f"{'l=' + format(lam, 'g'):>10}" for lam in args.lambdas
        )
        print(header)
        print("-" * len(header))

        spreads = {lam: [] for lam in args.lambdas}
        unusable = []

        for label, record in records:
            names = record["names"]
            score = compute_red(
                matrix_of(record),
                names,
                mode=mode,
                mask_within_op=mask_within_op,
            )

            tag = label.replace("_pspnet_cityscapes", "").replace("grad_corr_", "")[:29]
            row = f"{tag:<30}{record['checkpoint']:>6.2f}{len(names):>4}  "

            if not score.usable:
                unusable.append((tag, record["checkpoint"], score.reason))
                print(row + "  (unusable)")
                continue

            pdf = uniform_pdf(names)
            red = score.as_dict()
            for lam in args.lambdas:
                result = reweight(pdf, red, lam)
                spreads[lam].append(result.spread)
                row += f"{result.spread:>9.1f}x"
            print(row)

        if not any(spreads.values()):
            print("\nnothing usable to calibrate against.")
            continue

        print("-" * len(header))
        for stat, fn in (("min", min), ("max", max)):
            line = f"{stat + ' across all':<30}{'':>6}{'':>4}  " + "".join(
                f"{fn(spreads[lam]):>9.1f}x" for lam in args.lambdas
            )
            print(line)

        # The portability claim, as a number: if standardization works, the spread
        # at a given lambda barely moves between checkpoints of different runs and
        # different vocabulary sizes.
        print()
        for lam in args.lambdas:
            values = np.array(spreads[lam])
            drift = values.max() / values.min()
            verdict = "portable" if drift < 2.0 else "NOT PORTABLE"
            print(
                f"  lambda={lam:<5g} spread {values.min():6.1f}x - {values.max():6.1f}x  "
                f"(varies {drift:.1f}x across checkpoints -> {verdict})"
            )

        within = [
            lam for lam in args.lambdas if max(spreads[lam]) <= args.target_spread
        ]
        if within:
            print(
                f"\n  -> largest lambda holding every checkpoint under "
                f"{args.target_spread:g}x: {max(within):g}"
            )
        else:
            print(
                f"\n  -> no swept lambda holds every checkpoint under "
                f"{args.target_spread:g}x; try smaller values"
            )

        if unusable:
            print("\n  unusable checkpoints:")
            for tag, checkpoint, reason in unusable:
                print(f"    {tag} @ {checkpoint:.2f}: {reason}")

        # Which ops the down-weighting would actually bite, at the last checkpoint
        # of each run -- the one closest to what a trained model looks like.
        print("\n  most / least redundant ops (final checkpoint of each run):")
        seen = {}
        for label, record in records:
            seen[label] = record
        for label, record in seen.items():
            score = compute_red(
                matrix_of(record),
                record["names"],
                mode=mode,
                mask_within_op=mask_within_op,
            )
            tag = label.replace("_pspnet_cityscapes", "").replace("grad_corr_", "")[:29]
            if not score.usable:
                print(f"    {tag:<30} unusable: {score.reason}")
                continue
            order = np.argsort(score.std)
            most = ", ".join(
                f"{score.names[i]}" for i in order[::-1][: args.top]
            )
            least = ", ".join(f"{score.names[i]}" for i in order[: args.top])
            print(f"    {tag:<30} ckpt {record['checkpoint']:.2f}")
            print(f"      most:  {most}")
            print(f"      least: {least}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
