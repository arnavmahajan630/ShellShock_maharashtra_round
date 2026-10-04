"""Feature matrix and splits for E-b. Reads the dataset; writes only under ml/eval/out/e-b/.

Training rows follow 03 §5.3 (sources A/E/AMB, not holdout problems). The matrix is cached
here, not next to ml/data/, so this package does not write a file it does not own.
Dropout rows are extracted with no trace, the same path as ml.model.data.build_matrix.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ml.contracts.classes import LABELS
from ml.contracts.feature_names import FEATURES
from ml.model.data import (
    FEATURE_DROPOUT,
    TrainData,
    domain_of,
    load_problems,
    read_rows,
    soft_label_vector,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ml" / "eval" / "out" / "e-b"
CACHE = OUT / "cache"
DATASET = ROOT / "ml" / "data" / "dataset.jsonl"
RLLM = ROOT / "ml" / "data" / "realistic_llm.jsonl"
PROBLEM_DIRS = (ROOT / "ml" / "problems" / "main", ROOT / "ml" / "problems" / "dsa")

SEED = 42
N_BOOT = 1000
STAMP = "eb-1"
FEATURE_STAMP = "f1"
HOLDOUT_PROBLEMS = ("P04", "P09", "P13", "Q04", "Q09", "Q16")


class Bundle:
    """One row per dataset line, then every R-llm line. `y` is -1 when the label is not a model class."""

    def __init__(self, **arrays):
        self.__dict__.update(arrays)
        self.features = list(FEATURES)

    def __len__(self):
        return int(len(self.y))


def bank_train_mask(bundle):
    """TRAIN: A/E/AMB rows of the 28 training problems (03 §9.1)."""
    return (bundle.kind == "bank") & (bundle.split == "train") & (bundle.y >= 0)


def holdout_mask(bundle):
    """HOLDOUT-P: the six problems named in 03 §9.1."""
    return (bundle.kind == "bank") & (bundle.split == "holdout_problem") & (bundle.y >= 0)


def rllm_mask(bundle):
    """R-llm rows whose label is one of the 19 classes. GATE rows stay out."""
    return (bundle.kind == "rllm") & (bundle.y >= 0)


def fingerprint():
    digest = hashlib.sha256()
    digest.update(FEATURE_STAMP.encode())
    digest.update(",".join(FEATURES).encode())
    for path in (DATASET, RLLM):
        st = path.stat()
        digest.update(f"{path.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    for folder in PROBLEM_DIRS:
        for path in sorted(folder.glob("*.json")):
            st = path.stat()
            digest.update(f"{path.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    return digest.hexdigest()[:16]


def _u(values):
    return np.asarray(["" if v is None else str(v) for v in values], dtype="U256")


def _extract_pair(problem, code, refs):
    """Full serving row, plus the no-trace row used for feature dropout. Returns (full, ast, tests_passed)."""
    from ml.features.extract import extract
    from ml.features.relation_feats import make_run_reference
    from ml import runner

    ast, _ = extract(problem, code)
    pid = problem["problem_id"]
    if pid not in refs:
        ref_code = problem["correct_variants"][0]
        refs[pid] = (runner.trace(problem, ref_code), make_run_reference(problem, runner.run_tests))
    ref_trace, run_reference = refs[pid]
    trace = runner.trace(problem, code)
    run_result = runner.run_tests(problem, code)
    full, _ = extract(problem, code, trace, ref_trace, run_reference, run_result=run_result)
    tests = run_result.get("tests") or {}
    passed = bool(tests.get("total")) and tests.get("passed") == tests.get("total")
    return full, ast, passed


def _accumulate(rows, kind, problems, refs, into):
    fails = 0
    for row in rows:
        problem = problems.get(row["problem_id"])
        label = row.get("label")
        if problem is None or label is None:
            fails += 1
            continue
        try:
            full, ast, passed = _extract_pair(problem, row["code"], refs)
        except Exception:
            fails += 1
            full = np.full(len(FEATURES), np.nan)
            ast = full.copy()
            passed = False
        if label in LABELS:
            target = soft_label_vector(row)
            y = LABELS.index(label)
        else:
            target = np.zeros(len(LABELS))
            y = -1
        into["X_full"].append(np.asarray(full, dtype=np.float32))
        into["X_ast"].append(np.asarray(ast, dtype=np.float32))
        into["Y"].append(target)
        into["y"].append(y)
        into["tests_passed"].append(passed)
        into["id"].append(row.get("id"))
        into["problem_id"].append(row["problem_id"])
        into["label"].append(label)
        into["op_id"].append(row.get("op_id") or "")
        into["source"].append(row.get("source") or "")
        into["split"].append(row.get("split") or "")
        into["domain"].append(domain_of(problem))
        into["ast_hash"].append(row.get("ast_hash") or "")
        into["kind"].append(kind)
        n = len(into["y"])
        if n % 500 == 0:
            print(f"  features {n}", flush=True)
    return fails


def build_bundle():
    print("building feature cache (interpreter, one process)", flush=True)
    started = time.perf_counter()
    problems = load_problems(PROBLEM_DIRS)
    bank = read_rows(DATASET)
    rllm = read_rows(RLLM) if RLLM.exists() else []
    into = {key: [] for key in (
        "X_full", "X_ast", "Y", "y", "tests_passed", "id", "problem_id", "label",
        "op_id", "source", "split", "domain", "ast_hash", "kind")}
    refs = {}
    fails = _accumulate(bank, "bank", problems, refs, into)
    fails += _accumulate(rllm, "rllm", problems, refs, into)
    bundle = Bundle(
        X_full=np.stack(into["X_full"]).astype(np.float32),
        X_ast=np.stack(into["X_ast"]).astype(np.float32),
        Y=np.stack(into["Y"]).astype(np.float64),
        y=np.asarray(into["y"], dtype=np.int64),
        tests_passed=np.asarray(into["tests_passed"], dtype=np.uint8),
        id=_u(into["id"]),
        problem_id=_u(into["problem_id"]),
        label=_u(into["label"]),
        op_id=_u(into["op_id"]),
        source=_u(into["source"]),
        split=_u(into["split"]),
        domain=_u(into["domain"]),
        ast_hash=_u(into["ast_hash"]),
        kind=_u(into["kind"]),
        n_fail=int(fails),
        fingerprint=fingerprint(),
    )
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE / "features.npz",
        X_full=bundle.X_full, X_ast=bundle.X_ast, Y=bundle.Y, y=bundle.y,
        tests_passed=bundle.tests_passed, id=bundle.id, problem_id=bundle.problem_id,
        label=bundle.label, op_id=bundle.op_id, source=bundle.source, split=bundle.split,
        domain=bundle.domain, ast_hash=bundle.ast_hash, kind=bundle.kind,
    )
    (CACHE / "meta.json").write_text(json.dumps({
        "fingerprint": bundle.fingerprint, "n": len(bundle), "n_fail": bundle.n_fail,
        "seconds": round(time.perf_counter() - started, 1),
    }, indent=1), encoding="utf-8", newline="\n")
    print(f"  cached {len(bundle)} rows, {bundle.n_fail} failed, {time.perf_counter() - started:.1f}s", flush=True)
    return bundle


def _from_npz(path, meta):
    data = np.load(path, allow_pickle=False)
    return Bundle(
        X_full=data["X_full"], X_ast=data["X_ast"], Y=data["Y"], y=data["y"],
        tests_passed=data["tests_passed"], id=data["id"], problem_id=data["problem_id"],
        label=data["label"], op_id=data["op_id"], source=data["source"], split=data["split"],
        domain=data["domain"], ast_hash=data["ast_hash"], kind=data["kind"],
        n_fail=int(meta.get("n_fail", 0)), fingerprint=meta.get("fingerprint", ""),
    )


def load_bundle():
    """Cached matrix, rebuilt when the dataset, problems, or feature list change."""
    meta_path = CACHE / "meta.json"
    npz = CACHE / "features.npz"
    current = fingerprint()
    if meta_path.exists() and npz.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == current:
            return _from_npz(npz, meta)
    return build_bundle()


def matrix_for_training(bundle, index, seed=SEED):
    """Copy of the full rows with 15% replaced by the no-trace extraction (03 §4.6)."""
    index = np.asarray(index, dtype=np.int64)
    X = np.array(bundle.X_full[index], dtype=np.float32, copy=True)
    drop = np.random.default_rng(seed).random(len(index)) < FEATURE_DROPOUT
    if drop.any():
        X[drop] = bundle.X_ast[index[drop]]
    return X


def make_train_data(bundle, index, drop_features=(), seed=SEED):
    """TrainData for one retrain. `drop_features` removes columns (E6 strict)."""
    index = np.asarray(index, dtype=np.int64)
    if len(index) == 0:
        raise ValueError("no training rows")
    if np.any(bundle.y[index] < 0):
        raise ValueError("training index contains a non-class label")
    X = matrix_for_training(bundle, index, seed)
    features = list(bundle.features)
    if drop_features:
        blocked = set(drop_features)
        keep = [i for i, name in enumerate(features) if name not in blocked]
        X = X[:, keep]
        features = [features[i] for i in keep]
    return TrainData(
        X=X, Y=bundle.Y[index], y=bundle.y[index], groups=np.asarray(bundle.problem_id[index]),
        features=features, labels=list(LABELS), domain=np.asarray(bundle.domain[index]),
        ids=[str(i) for i in bundle.id[index]],
    )


def columns_for(model_features):
    """Indices of a saved model's columns inside the full FEATURES vector."""
    return [FEATURES.index(name) for name in model_features]
