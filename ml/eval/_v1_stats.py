"""Seed-level means and 95% Student-t intervals for the V1 simulations.

The interval is across seeds. It is the uncertainty of the Monte Carlo mean.
It is not a confidence interval for real learners.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np
from scipy.stats import t as student_t


def mean_interval(values):
    """Mean and 95% t interval of seed-level numbers. ``None`` seeds are dropped.

    Returns ``None`` when every seed is missing (the conditioning set was empty).
    A single seed returns a point interval.
    """
    clean = [float(v) for v in values if v is not None]
    n = len(clean)
    if n == 0:
        return None
    arr = np.asarray(clean, dtype=float)
    mean = float(arr.mean())
    if n < 2:
        return {"mean": mean, "lo": mean, "hi": mean, "n_seeds": n}
    se = float(arr.std(ddof=1) / np.sqrt(n))
    half = float(student_t.ppf(0.975, n - 1)) * se
    return {"mean": mean, "lo": mean - half, "hi": mean + half, "n_seeds": n}


def _fmt_num(value, digits):
    text = f"{value:.{digits}f}"
    # A t interval on an exact 0 can be -0.0, which formats as "-0.000".
    if text.startswith("-") and float(text) == 0.0:
        return text[1:]
    return text


def fmt_stat(stat, digits=3):
    """``0.123 [0.120, 0.126]``, or an em dash when the cell is undefined."""
    if stat is None:
        return "—"
    return (
        f"{_fmt_num(stat['mean'], digits)} "
        f"[{_fmt_num(stat['lo'], digits)}, {_fmt_num(stat['hi'], digits)}]"
    )
