"""Run the four E-b experiments. Each one writes its own card under ml/eval/out/e-b/."""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse

from ml.eval import e03_operator_holdout, e05_twins, e06_loco, e15_cross_domain

JOBS = {
    "e15": e15_cross_domain.run,
    "e05": e05_twins.run,
    "e03": e03_operator_holdout.run,
    "e06": e06_loco.run,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="E-b retrain experiments")
    parser.add_argument("--only", nargs="*", choices=sorted(JOBS))
    args = parser.parse_args(argv)
    # E15 trains the shared full model that E5 reuses, so it goes first unless the caller chose one.
    order = ["e15", "e05", "e03", "e06"]
    chosen = args.only or order
    for name in order:
        if name in chosen:
            JOBS[name]()


if __name__ == "__main__":
    main()
