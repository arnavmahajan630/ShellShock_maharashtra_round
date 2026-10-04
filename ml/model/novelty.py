"""Novelty score and thresholds (ml_plan/03 §4.8), package M1.

knn_dist  mean Euclidean distance to the 5 nearest training rows in standardised A+B+R space
          (NaN -> column median, zero-variance columns dropped).
tau_d     99th percentile of out-of-fold distances.
tau_p     smallest confidence threshold at which accepted out-of-fold predictions are >= 90% correct.
novel     knn_dist > tau_d  or  p_other >= 0.5  or  (p_max < tau_p and the top two are not a known twin pair).

`exclude` drops named columns from the space; E6 "LOCO strict" passes CLASS_DEFINING_FEATURES[k].
"""
from __future__ import annotations

import base64

import numpy as np

from ml.contracts.classes import LABELS, twin_set_of
from ml.contracts.feature_names import FEATURES, GROUPS
from ml.contracts.params import NOVEL_P_OTHER

K_NEIGHBOURS = 5
MAX_REFERENCE_ROWS = 4000
TAU_D_PERCENTILE = 99.0
TARGET_PRECISION = 0.90
TAU_P_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


class Novelty:
    """The kNN reference space. Built with `fit`, stored in meta.json with `to_meta`."""

    def __init__(self, columns, median, mean, std, reference, k=K_NEIGHBOURS):
        self.columns = np.asarray(columns, dtype=np.int64)      # indices into the feature list
        self.median = np.asarray(median, dtype=np.float32)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.reference = np.asarray(reference, dtype=np.float32)
        self.k = k
        self._ref_sq = (self.reference.astype(np.float64) ** 2).sum(axis=1)

    @classmethod
    def fit(cls, X, features=FEATURES, exclude=(), max_rows=MAX_REFERENCE_ROWS, seed=42, k=K_NEIGHBOURS):
        space = set(GROUPS["A"]) | set(GROUPS["B"]) | set(GROUPS["R"])
        candidates = np.array([i for i, name in enumerate(features) if name in space and name not in set(exclude)])
        sub = np.asarray(X, dtype=np.float64)[:, candidates]
        with np.errstate(all="ignore"):
            median = np.nanmedian(sub, axis=0)
        median = np.where(np.isnan(median), 0.0, median)        # a column that is always missing
        filled = np.where(np.isnan(sub), median, sub)
        std = filled.std(axis=0)
        keep = std > 0
        columns, median = candidates[keep], median[keep]
        mean, std = filled[:, keep].mean(axis=0), std[keep]
        Z = (filled[:, keep] - mean) / std
        if len(Z) > max_rows:
            Z = Z[np.sort(np.random.default_rng(seed).choice(len(Z), size=max_rows, replace=False))]
        return cls(columns, median, mean, std, Z.astype(np.float32), k)

    def transform(self, X):
        sub = np.atleast_2d(np.asarray(X, dtype=np.float64))[:, self.columns]
        filled = np.where(np.isnan(sub), self.median, sub)
        return (filled - self.mean) / self.std

    def distance(self, X, exclude_self=False):
        """Mean distance to the k nearest reference rows. `exclude_self` skips one zero-distance
        match, for scoring rows that are themselves in the reference."""
        Z = self.transform(X)
        d2 = (Z ** 2).sum(axis=1)[:, None] + self._ref_sq[None, :] - 2.0 * Z @ self.reference.astype(np.float64).T
        d = np.sqrt(np.clip(d2, 0.0, None))
        k = min(self.k + (1 if exclude_self else 0), d.shape[1])
        nearest = np.sort(np.partition(d, k - 1, axis=1)[:, :k], axis=1)
        if exclude_self:
            nearest = nearest[:, 1:]
        return nearest.mean(axis=1)

    def to_meta(self):
        return {"k": self.k, "columns": self.columns.tolist(), "median": self.median.tolist(),
                "mean": self.mean.tolist(), "std": self.std.tolist(),
                "reference": {"dtype": "float32", "shape": list(self.reference.shape),
                              "b64": base64.b64encode(np.ascontiguousarray(self.reference).tobytes()).decode("ascii")}}

    @classmethod
    def from_meta(cls, meta):
        ref = meta["reference"]
        reference = np.frombuffer(base64.b64decode(ref["b64"]), dtype=ref["dtype"]).reshape(ref["shape"])
        return cls(meta["columns"], meta["median"], meta["mean"], meta["std"], reference, meta["k"])


def oof_distances(X, folds, features=FEATURES, exclude=(), seed=42):
    """Distance of every row to the training rows of the folds it was held out from."""
    out = np.full(len(X), np.nan)
    for train_idx, valid_idx in folds:
        out[valid_idx] = Novelty.fit(X[train_idx], features, exclude=exclude, seed=seed).distance(X[valid_idx])
    return out


def choose_tau_d(distances, percentile=TAU_D_PERCENTILE):
    return float(np.nanpercentile(distances, percentile))


def choose_tau_p(probs, correct, target=TARGET_PRECISION, grid=TAU_P_GRID):
    """Smallest threshold on p_max at which the accepted predictions reach the target precision.
    If none does, the largest grid value is returned (almost everything abstains)."""
    p_max = probs.max(axis=1)
    for tau in grid:
        accepted = p_max >= tau
        if accepted.any() and correct[accepted].mean() >= target:
            return float(tau)
    return float(grid[-1])


def assess(knn_dist, probs, tau_d, tau_p, labels=LABELS):
    """The `novelty` object of a diagnosis (03 §11.1) for one probability vector."""
    probs = np.asarray(probs, dtype=np.float64)
    order = np.argsort(-probs)
    top1, top2 = labels[order[0]], labels[order[1]]
    p_max, p_other = float(probs[order[0]]), float(probs[labels.index("OTHER")])
    twins = twin_set_of(top1, top2) is not None
    abstain = bool(knn_dist > tau_d or p_other >= NOVEL_P_OTHER or (p_max < tau_p and not twins))
    return {"knn_dist": round(float(knn_dist), 4), "tau_d": round(float(tau_d), 4), "p_max": round(p_max, 4),
            "tau_p": round(float(tau_p), 4), "abstain": abstain}
