"""Diagnoser confusion for E14 (ml_plan/03 §9.3b).

The default matrix is the E1 out-of-fold confusion: calibrated, masked argmax
on TRAIN, from ``ml/eval/_ea_cache/oof.npz`` and the E1 card's model. Rows are
the dataset label. Columns are that argmax. This is the noise that corrupts
``fail_k``. It is not the soft-label plot in ``cards/E01_confusion.png``, which
moves an acceptable alternate onto the diagonal.

``stand_in_confusion`` stays for comparison. Pass ``confusion="stand-in"``.
"""
from __future__ import annotations

import json
import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import numpy as np

from ml.contracts.classes import LABELS, MISCONCEPTIONS, TWIN_SETS

DIAGONAL = 0.72
TWIN_MASS = 0.18
STAND_IN_SOURCE = "stand-in (comparison flag; not the E1 matrix)"
E1_SOURCE = "E1 out-of-fold confusion on TRAIN (calibrated, masked argmax | dataset label)"
# Old name. The default matrix is the E1 one; this string is only the stand-in.
SOURCE = STAND_IN_SOURCE

_E1 = None


def _partners(class_id):
    found = []
    allowed = set(MISCONCEPTIONS)
    for info in TWIN_SETS.values():
        members = [m for m in info["members"] if m in allowed]
        if class_id not in members:
            continue
        for other in members:
            if other != class_id and other not in found:
                found.append(other)
    return found


def stand_in_confusion():
    """Row-stochastic matrix indexed by ``MISCONCEPTIONS`` order.

    ``matrix[i, j]`` is P(diagnoser says j | true class i) given that the
    attempt failed. Rows sum to 1. The draw never turns a failure into a pass.
    """
    index = {cls: i for i, cls in enumerate(MISCONCEPTIONS)}
    n = len(MISCONCEPTIONS)
    matrix = np.zeros((n, n), dtype=float)
    for cls, i in index.items():
        matrix[i, i] = DIAGONAL
        partners = _partners(cls)
        if partners:
            share = TWIN_MASS / len(partners)
            for other in partners:
                matrix[i, index[other]] += share
            leftover = 1.0 - DIAGONAL - TWIN_MASS
        else:
            leftover = 1.0 - DIAGONAL
        others = [j for j in range(n) if matrix[i, j] == 0.0]
        if others and leftover > 0.0:
            matrix[i, others] += leftover / len(others)
        total = matrix[i].sum()
        if total <= 0.0:
            matrix[i, i] = 1.0
        else:
            matrix[i] /= total
    return matrix


def _normalise(matrix):
    out = np.asarray(matrix, dtype=float).copy()
    totals = out.sum(axis=1, keepdims=True)
    for i in range(out.shape[0]):
        if totals[i, 0] <= 0.0:
            out[i, :] = 0.0
            if i < out.shape[1]:
                out[i, i] = 1.0
            else:
                out[i, 0] = 1.0
    out /= out.sum(axis=1, keepdims=True)
    return out


def e1_oof_confusion():
    """Row-stochastic E1 matrix. Reads E-a's cache. Does not train and does not write it.

    Raises if the cached model hash does not match the current TRAIN rows, so a
    mismatch cannot start E-a's training run.
    """
    global _E1
    if _E1 is not None:
        return _E1["matrix"]
    from ml.eval import _ea_common as common
    from ml.model.calibrate import softmax
    from ml.model.mask import allowed_classes, apply_mask

    info_path = common.CACHE / "model_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    data = common.train_bundle()
    if info.get("data_hash") != data.data_hash() or not (common.CACHE / "oof.npz").exists():
        raise RuntimeError("E1 OOF cache does not match TRAIN; refusing to retrain")
    meta_path = common.CACHE / info["version"] / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if list(meta["classes"]) != list(LABELS):
        raise RuntimeError("E1 class order is not LABELS")
    logits = np.load(common.CACHE / "oof.npz")["logits"]
    if len(logits) != len(data.y):
        raise RuntimeError("E1 OOF logits and TRAIN labels differ in length")
    probs = apply_mask(
        softmax(logits, float(meta["temperature"])),
        allowed_classes(data.X, data.features, data.labels),
    )
    pred = probs.argmax(axis=1)
    n = len(LABELS)
    counts = np.zeros((n, n), dtype=float)
    for true_i, pred_i in zip(data.y, pred):
        counts[int(true_i), int(pred_i)] += 1.0
    matrix = _normalise(counts)
    support = counts.sum(axis=1)
    misc = [LABELS.index(cls) for cls in MISCONCEPTIONS]
    _E1 = {
        "matrix": matrix,
        "n": int(len(pred)),
        "model_version": info["version"],
        "temperature": float(meta["temperature"]),
        "empty_rows": [LABELS[i] for i in range(n) if support[i] <= 0],
        "mean_diagonal_misconceptions": float(np.mean(matrix[misc, misc])),
    }
    return matrix


def e1_oof_meta():
    """Counts behind ``e1_oof_confusion``. Loads the matrix if needed."""
    e1_oof_confusion()
    return {key: value for key, value in _E1.items() if key != "matrix"}


def labels_for(matrix):
    """Column order. The stand-in is the 17 misconceptions. E1 is all 19 labels."""
    width = int(np.asarray(matrix).shape[1])
    if width == len(MISCONCEPTIONS):
        return list(MISCONCEPTIONS)
    if width == len(LABELS):
        return list(LABELS)
    raise ValueError(f"confusion matrix has {width} columns")


def corrupt(rng, true_class, matrix, labels=None):
    """Draw the class the diagnoser reports. ``true_class`` must be a row label."""
    if labels is None:
        labels = labels_for(matrix)
    i = labels.index(true_class)
    j = int(rng.choice(len(labels), p=matrix[i]))
    return labels[j]
