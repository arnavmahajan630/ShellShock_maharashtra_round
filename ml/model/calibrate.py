"""Temperature scaling and calibration measures (ml_plan/03 §5.4), package M1.

One scalar T is fitted on out-of-fold logits by minimising the negative log-likelihood of
softmax(logits / T), bounds [0.5, 5]. ECE uses 10 equal-width confidence bins.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

T_BOUNDS = (0.5, 5.0)
N_BINS = 10
_EPS = 1e-12


def softmax(logits, T=1.0):
    z = np.asarray(logits, dtype=np.float64) / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(probs, Y):
    """Mean cross-entropy against soft targets Y (rows sum to 1)."""
    return float(-(Y * np.log(np.clip(probs, _EPS, 1.0))).sum(axis=1).mean())


def fit_temperature(logits, Y, bounds=T_BOUNDS):
    result = minimize_scalar(lambda T: nll(softmax(logits, T), Y), bounds=bounds, method="bounded",
                             options={"xatol": 1e-4})
    return float(result.x)


def is_correct(probs, Y):
    """A prediction counts as correct when it is one of the row's labels (soft rows have two)."""
    pred = probs.argmax(axis=1)
    return Y[np.arange(len(pred)), pred] > 0


def reliability(probs, correct, n_bins=N_BINS):
    """Per confidence bin: count, mean confidence, accuracy."""
    conf = probs.max(axis=1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        inside = which == b
        count = int(inside.sum())
        rows.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "count": count,
                     "confidence": float(conf[inside].mean()) if count else None,
                     "accuracy": float(correct[inside].mean()) if count else None})
    return rows


def ece(probs, correct, n_bins=N_BINS):
    """Expected calibration error: sum over bins of share × |accuracy − confidence|."""
    total = len(correct)
    return float(sum(row["count"] / total * abs(row["accuracy"] - row["confidence"])
                     for row in reliability(probs, correct, n_bins) if row["count"]))


def brier(probs, Y):
    return float(((probs - Y) ** 2).sum(axis=1).mean())


def calibrate(logits, Y):
    """Fit T on OOF logits and report ECE / NLL / Brier before and after (E7 reads this)."""
    T = fit_temperature(logits, Y)
    before, after = softmax(logits, 1.0), softmax(logits, T)
    return {
        "temperature": T,
        "ece_before": ece(before, is_correct(before, Y)), "ece_after": ece(after, is_correct(after, Y)),
        "nll_before": nll(before, Y), "nll_after": nll(after, Y),
        "brier_before": brier(before, Y), "brier_after": brier(after, Y),
        "reliability_before": reliability(before, is_correct(before, Y)),
        "reliability_after": reliability(after, is_correct(after, Y)),
        "n": int(len(Y)),
    }
