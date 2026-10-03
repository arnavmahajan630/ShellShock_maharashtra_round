"""Data loading for the diagnoser, kept apart from training (package M1).

Two sources give the same `TrainData`:
  * `load_npz(path)`      a ready matrix (now: tests/fixtures/features_synth.npz, a made-up matrix)
  * `load_dataset(path)`  `ml/data/dataset.jsonl` (03 §3.6): rows are filtered per 03 §5.3 and the
                          matrix is built with `ml.features.extract.extract`, which needs the
                          interpreter (package A1) and the problem files (B1, B2). A built matrix
                          is cached next to the dataset.

`train.py` only sees `TrainData`, so nothing there changes when the real dataset arrives.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ml.contracts.classes import LABELS
from ml.contracts.feature_names import FEATURES, GROUPS, TRACE_DEPENDENT_GROUPS

FEATURE_DROPOUT = 0.15          # share of training rows with groups B and R set to NaN (03 §4.6)
DEFAULT_DATASET = Path("ml/data/dataset.jsonl")
DEFAULT_PROBLEM_DIRS = (Path("ml/problems/main"), Path("ml/problems/dsa"))


@dataclass
class TrainData:
    X: np.ndarray                               # (n, n_features) float32, NaN = missing
    Y: np.ndarray                               # (n, n_labels) soft labels, rows sum to 1
    y: np.ndarray                               # (n,) index of the row's primary label
    groups: np.ndarray                          # (n,) problem id per row (CV groups)
    features: list = field(default_factory=lambda: list(FEATURES))
    labels: list = field(default_factory=lambda: list(LABELS))
    domain: np.ndarray | None = None            # (n,) "main" / "dsa": CV stratification (03 §5.3)
    codes: list | None = None                   # source text, only for the TF-IDF baselines
    ids: list | None = None

    def __len__(self):
        return len(self.y)

    def subset(self, index):
        index = np.asarray(index)
        return TrainData(
            X=self.X[index], Y=self.Y[index], y=self.y[index], groups=self.groups[index],
            features=self.features, labels=self.labels,
            domain=None if self.domain is None else self.domain[index],
            codes=None if self.codes is None else [self.codes[i] for i in index],
            ids=None if self.ids is None else [self.ids[i] for i in index])

    def data_hash(self):
        digest = hashlib.sha256()
        for part in (np.ascontiguousarray(self.X, dtype=np.float32), np.ascontiguousarray(self.Y, dtype=np.float64),
                     np.asarray([str(g) for g in self.groups])):
            digest.update(part.tobytes())
        return digest.hexdigest()[:16]


def one_hot(y, n_labels):
    Y = np.zeros((len(y), n_labels), dtype=np.float64)
    Y[np.arange(len(y)), y] = 1.0
    return Y


def trace_dependent_columns(features=FEATURES):
    names = {name for group in TRACE_DEPENDENT_GROUPS for name in GROUPS[group]}
    return np.array([i for i, name in enumerate(features) if name in names])


def load_npz(path):
    """A matrix saved with X, y, groups, features, labels (the shape of features_synth.npz).
    Optional arrays: Y (soft labels), domain. Feature dropout is assumed to be in the file already."""
    data = np.load(path, allow_pickle=False)
    features, labels = [str(f) for f in data["features"]], [str(k) for k in data["labels"]]
    if features != FEATURES or labels != LABELS:
        raise ValueError(f"{path}: columns or labels differ from ml/contracts")
    y = data["y"].astype(np.int64)
    Y = data["Y"].astype(np.float64) if "Y" in data.files else one_hot(y, len(labels))
    domain = np.array([str(d) for d in data["domain"]]) if "domain" in data.files else None
    return TrainData(X=data["X"].astype(np.float32), Y=Y, y=y, groups=data["groups"], domain=domain,
                     features=features, labels=labels)


# ---------------------------------------------------------------- dataset.jsonl

def read_rows(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def training_rows(rows):
    """03 §5.3 step 1: drop held-out problems, the passes-by-luck pool and U rows.
    R and X rows are evaluation-only as well (03 §9.1) and are dropped too."""
    kept = []
    for row in rows:
        if row.get("split") == "holdout_problem":
            continue
        if (row.get("verified") or {}).get("passes_by_luck"):
            continue
        if row.get("source") not in ("A", "E", "AMB"):
            continue
        kept.append(row)
    return kept


def soft_label_vector(row, labels=LABELS):
    """Soft label of a dataset row: `soft_label` if present, 0.5/0.5 for a two-bug row, else one-hot."""
    vec = np.zeros(len(labels), dtype=np.float64)
    soft = row.get("soft_label")
    if not soft and row.get("is_two_bug") and len(row.get("labels_all") or []) >= 2:
        soft = {name: 1.0 for name in row["labels_all"]}
    if soft:
        for name, p in soft.items():
            vec[labels.index(name)] = p
        return vec / vec.sum()
    vec[labels.index(row["label"])] = 1.0
    return vec


def load_problems(dirs=DEFAULT_PROBLEM_DIRS):
    problems = {}
    for folder in dirs:
        for path in sorted(Path(folder).glob("*.json")):
            problem = json.loads(path.read_text(encoding="utf-8"))
            problems[problem["problem_id"]] = problem
    return problems


def domain_of(problem):
    return "dsa" if problem.get("sector") else "main"


def dropout_rows(n, rate=FEATURE_DROPOUT, seed=42):
    """Which rows lose their trace features (fixed by the seed, independent of the label)."""
    return np.random.default_rng(seed).random(n) < rate


def build_matrix(rows, problems, *, trace_fn, run_tests_fn, dropout=FEATURE_DROPOUT, seed=42, progress=None):
    """Feature matrix for dataset rows.

    trace_fn(problem, code) -> trace dict and run_tests_fn(problem, code) -> RunResult are the
    backend (normally `ml.runner.trace` and `ml.runner.run_tests`). Dropout rows are extracted
    with no trace at all, so they look exactly like an AST-only request at serving time
    (groups B and R NaN, main loop chosen by the static rule).
    """
    from ml.features.extract import extract
    from ml.features.relation_feats import make_run_reference

    drop = dropout_rows(len(rows), dropout, seed)
    reference = {}
    X = np.full((len(rows), len(FEATURES)), np.nan, dtype=np.float32)
    for i, row in enumerate(rows):
        problem = problems[row["problem_id"]]
        if drop[i]:
            X[i], _ = extract(problem, row["code"])
        else:
            pid = problem["problem_id"]
            if pid not in reference:
                ref_code = problem["correct_variants"][0]
                reference[pid] = (trace_fn(problem, ref_code), make_run_reference(problem, run_tests_fn))
            ref_trace, run_reference = reference[pid]
            X[i], _ = extract(problem, row["code"], trace_fn(problem, row["code"]), ref_trace, run_reference,
                              run_result=run_tests_fn(problem, row["code"]))
        if progress and (i + 1) % 500 == 0:
            progress(i + 1, len(rows))
    return X


def load_dataset(path=DEFAULT_DATASET, *, problems=None, trace_fn=None, run_tests_fn=None, cache=True,
                 dropout=FEATURE_DROPOUT, seed=42):
    """`dataset.jsonl` -> TrainData. Needs the interpreter unless a cached matrix exists."""
    path = Path(path)
    rows = training_rows(read_rows(path))
    if not rows:
        raise ValueError(f"{path}: no training rows")
    problems = problems if problems is not None else load_problems()
    labels = list(LABELS)
    Y = np.stack([soft_label_vector(row, labels) for row in rows])
    y = np.array([labels.index(row["label"]) for row in rows], dtype=np.int64)
    groups = np.array([row["problem_id"] for row in rows])
    domain = np.array([domain_of(problems[row["problem_id"]]) for row in rows])
    key = hashlib.sha256(json.dumps([[r["id"], r["code"]] for r in rows] + [FEATURES, dropout, seed]).encode()).hexdigest()[:12]
    cache_path = path.with_name(f"features_{key}.npz")
    if cache and cache_path.exists():
        X = np.load(cache_path)["X"]
    else:
        if trace_fn is None or run_tests_fn is None:
            from ml import runner
            trace_fn, run_tests_fn = runner.trace, runner.run_tests
        X = build_matrix(rows, problems, trace_fn=trace_fn, run_tests_fn=run_tests_fn, dropout=dropout, seed=seed)
        if cache:
            np.savez_compressed(cache_path, X=X)
    return TrainData(X=X, Y=Y, y=y, groups=groups, domain=domain, codes=[row["code"] for row in rows],
                     ids=[row["id"] for row in rows])


def load(path):
    """Pick the loader from the file type."""
    path = Path(path)
    return load_npz(path) if path.suffix == ".npz" else load_dataset(path)


# ---------------------------------------------------------------- soft labels as weighted duplicates

def class_weights(Y):
    """Balanced weights from the soft-label mass per class: N / (classes present × mass_k)."""
    mass = Y.sum(axis=0)
    present = mass > 0
    weights = np.zeros(len(mass))
    weights[present] = mass.sum() / (present.sum() * mass[present])
    return weights


def expand_soft(X, Y, weights=None):
    """One copy of a row per label with soft_p > 0, weight = soft_p × class weight (03 §3.5.5).
    This is cross-entropy with soft targets. Returns (X, y, sample_weight, source row index)."""
    weights = class_weights(Y) if weights is None else weights
    rows, cols = np.nonzero(Y > 0)
    return X[rows], cols.astype(np.int64), Y[rows, cols] * weights[cols], rows
