"""Operator-holdout splits for E3 (03 §9.1 OPHOLD, §9.2 E3).

A base operator is an `op_id` with no '+'. Two-bug ids are components joined by '+'.
The held-out operator is the lexicographically last base operator of that class on TRAIN
with at least MIN_OP_ROWS rows. Training also drops every row whose ast_hash appears on
the held-out rows, so a soft-label twin of the same code cannot stay in the fit.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from ml.contracts.classes import LABELS
from ml.eval._eb_data import bank_train_mask

MIN_OP_ROWS = 8


def components(op_id):
    text = str(op_id or "")
    if not text:
        return ()
    return tuple(part for part in text.split("+") if part)


def base_operator_counts(bundle, class_name, train_mask):
    """Counts of single operators on TRAIN rows whose primary label is `class_name`."""
    counts = Counter()
    labels = bundle.label
    ops = bundle.op_id
    for keep, label, op in zip(train_mask, labels, ops):
        if not keep or str(label) != class_name:
            continue
        parts = components(op)
        if len(parts) == 1:
            counts[parts[0]] += 1
    return counts


def choose_operator(counts, min_rows=MIN_OP_ROWS):
    """Lexicographically last base operator that has enough rows. None if the class has one."""
    if len(counts) < 2:
        return None
    eligible = sorted(op for op, n in counts.items() if n >= min_rows)
    if not eligible:
        eligible = sorted(counts)
    return eligible[-1]


def plan_jobs(bundle, min_rows=MIN_OP_ROWS):
    """One holdout job per class with at least two base operators, plus a skip list."""
    train0 = bank_train_mask(bundle)
    jobs, skipped = [], []
    for class_name in LABELS:
        counts = base_operator_counts(bundle, class_name, train0)
        if len(counts) < 2:
            skipped.append({
                "class": class_name,
                "reason": f"{len(counts)} base operator(s) on TRAIN; E3 needs at least 2",
                "operators": dict(counts),
            })
            continue
        held = choose_operator(counts, min_rows)
        test = train0 & (bundle.label == class_name) & (bundle.op_id == held)
        test_index = np.flatnonzero(test)
        test_hashes = set(bundle.ast_hash[test_index].tolist())
        leak_op = np.array([held in components(op) for op in bundle.op_id], dtype=bool)
        leak_hash = np.array([str(h) in test_hashes and str(h) != "" for h in bundle.ast_hash], dtype=bool)
        drop = train0 & (leak_op | leak_hash)
        train_index = np.flatnonzero(train0 & ~drop)
        _assert_clean(bundle, train_index, test_index, held)
        jobs.append({
            "class": class_name,
            "op_id": held,
            "n_base_ops": int(len(counts)),
            "operators": dict(counts),
            "n_test": int(len(test_index)),
            "n_train": int(len(train_index)),
            "dropped_for_hash": int((train0 & leak_hash & ~leak_op).sum()),
            "train_index": train_index,
            "test_index": test_index,
        })
    return jobs, skipped


def _assert_clean(bundle, train_index, test_index, held):
    if len(set(train_index.tolist()) & set(test_index.tolist())):
        raise RuntimeError(f"{held}: a test row is still in training")
    test_hashes = set(bundle.ast_hash[test_index].tolist())
    train_hashes = set(bundle.ast_hash[train_index].tolist())
    overlap = test_hashes & train_hashes - {""}
    if overlap:
        raise RuntimeError(f"{held}: ast_hash leakage ({len(overlap)})")
    for op in bundle.op_id[train_index]:
        if held in components(op):
            raise RuntimeError(f"{held}: operator still in training ({op})")
