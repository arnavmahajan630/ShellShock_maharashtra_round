"""Structural class masking (ml_plan/03 §4.1b), package M1.

A class is impossible unless one of its precondition features is present in the code
(`MASK_PRECONDITIONS` in ml/contracts/feature_names.py). After calibration its probability is
set to 0 and the rest is renormalised. The same function is used on out-of-fold predictions
during training and at serving time, so both see identical masking.

A precondition that is NaN (the code did not parse) is treated as unknown: no masking.
If masking would remove all probability mass the row is left unmasked.
"""
from __future__ import annotations

import numpy as np

from ml.contracts.classes import LABELS
from ml.contracts.feature_names import FEATURES, MASK_PRECONDITIONS


def allowed_classes(X, features=FEATURES, labels=LABELS, preconditions=MASK_PRECONDITIONS):
    """Boolean (n, n_labels): False where the class's structure is absent from the code."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    allowed = np.ones((X.shape[0], len(labels)), dtype=bool)
    for cls, needed in preconditions.items():
        cols = X[:, [features.index(name) for name in needed]]
        present = (np.nan_to_num(cols, nan=0.0) != 0).any(axis=1) | np.isnan(cols).any(axis=1)
        allowed[:, labels.index(cls)] = present
    return allowed


def apply_mask(P, allowed):
    """Zero the disallowed classes and renormalise each row."""
    P = np.atleast_2d(np.asarray(P, dtype=np.float64))
    masked = np.where(allowed, P, 0.0)
    total = masked.sum(axis=1, keepdims=True)
    empty = total[:, 0] <= 0
    masked[empty], total[empty] = P[empty], P[empty].sum(axis=1, keepdims=True)
    return masked / total


def mask_proba(P, X, features=FEATURES, labels=LABELS):
    return apply_mask(P, allowed_classes(X, features, labels))


def masked_out(x, features=FEATURES, labels=LABELS):
    """Names of the classes masking removes for one feature row (for logs and tests)."""
    allowed = allowed_classes(x, features, labels)[0]
    return [name for name, ok in zip(labels, allowed) if not ok]
