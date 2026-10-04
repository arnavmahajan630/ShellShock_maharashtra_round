"""Shared helpers for package E-a (E1, E2, E4, E7, E8, E9, E11, E12, E13). Private to E-a.

    .venv\\Scripts\\python -m ml.eval._ea_common        # build every cache once (train the model)

What lives here
  * feature matrices for TRAIN, HOLDOUT-P and the R-llm set, cached under ml/eval/_ea_cache/
  * the diagnoser trained on TRAIN (03 §5.3) with its out-of-fold logits, cached the same way
  * scoring helpers: macro-F1 (soft-label aware, same rule as ml/model/train.py), bootstrap CIs,
    per-class tables, confusion plots
  * `write_card`, which writes ml/eval/cards/<ID>.json (+ PNGs next to it)

The model trained here is NOT written to ml/artifacts/ (that is I1's / M1's place); it stays in
ml/eval/_ea_cache/. At most 4 threads are used everywhere.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from ml.contracts.classes import LABELS, MAIN_CLASSES, DSA_CLASSES, MISCONCEPTIONS, TWIN_SETS
from ml.contracts.feature_names import FEATURES, GROUPS

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "ml" / "eval" / "_ea_cache"
CARDS = ROOT / "ml" / "eval" / "cards"
DATASET = ROOT / "ml" / "data" / "dataset.jsonl"
REALISTIC = ROOT / "ml" / "data" / "realistic_llm.jsonl"
REALISTIC_META = ROOT / "ml" / "data" / "realistic_llm_meta.json"
PROBLEM_DIRS = (ROOT / "ml" / "problems" / "main", ROOT / "ml" / "problems" / "dsa")

N_BOOT = 1000
SEED = 42
M_IDX = [LABELS.index(k) for k in MAIN_CLASSES]
D_IDX = [LABELS.index(k) for k in DSA_CLASSES]

R_LLM_NAME = "R-llm (58 LLM-written programs, NOT hand-written, NOT real students)"
R_BLIND_STATUS = ("R-blind (40 hand-written by FE) is still open: no file exists, so nothing was run on it. "
                  "The headline of 03 §0.3 is therefore not measured.")


def log(*args):
    print(*args, flush=True)


# ------------------------------------------------------------------ rows and problems

_PROBLEMS = None


def problems():
    global _PROBLEMS
    if _PROBLEMS is None:
        from ml.model.data import load_problems
        _PROBLEMS = load_problems(PROBLEM_DIRS)
    return _PROBLEMS


def domain_of(problem_id):
    from ml.model.data import domain_of as _d
    return _d(problems()[problem_id])


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_rows():
    """(train rows, holdout-P rows) from dataset.jsonl, using the same filter as ml/model/data.py."""
    from ml.model.data import training_rows
    rows = read_jsonl(DATASET)
    train = training_rows(rows)
    hold = [r for r in rows if r.get("split") == "holdout_problem" and r.get("source") in ("A", "E", "AMB")
            and not (r.get("verified") or {}).get("passes_by_luck")]
    return train, hold


def realistic_rows():
    return read_jsonl(REALISTIC)


# ------------------------------------------------------------------ feature extraction

class Extractor:
    """Feature rows for (problem, code). Reference traces are cached per problem."""

    def __init__(self):
        from ml import runner
        from ml.features.extract import extract
        from ml.features.relation_feats import make_run_reference
        self.runner, self.extract, self.make_ref = runner, extract, make_run_reference
        self.reference = {}
        self.failures = 0

    def _ref(self, problem):
        pid = problem["problem_id"]
        if pid not in self.reference:
            code = problem["correct_variants"][0]
            self.reference[pid] = (self.runner.trace(problem, code), self.make_ref(problem, self.runner.run_tests))
        return self.reference[pid]

    def row(self, problem_id, code, static=False):
        """static=True: no trace at all (an AST-only request). Failures give an all-NaN row."""
        problem = problems()[problem_id]
        try:
            if static:
                return np.asarray(self.extract(problem, code)[0], dtype=np.float32)
            ref_trace, run_ref = self._ref(problem)
            return np.asarray(self.extract(problem, code, self.runner.trace(problem, code), ref_trace, run_ref,
                                           run_result=self.runner.run_tests(problem, code))[0], dtype=np.float32)
        except Exception:
            self.failures += 1
            return np.full(len(FEATURES), np.nan, dtype=np.float32)


_EXTRACTOR = None


def extractor():
    global _EXTRACTOR
    if _EXTRACTOR is None:
        _EXTRACTOR = Extractor()
    return _EXTRACTOR


def matrix(name, rows, *, static=False, dropout=0.0):
    """Cached feature matrix for dataset-shaped rows. `dropout` follows ml/model/data.py (seed 42)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps([[r["id"], r["code"]] for r in rows] + [FEATURES, static, dropout]).encode()
                         ).hexdigest()[:12]
    path = CACHE / f"X_{name}_{key}.npz"
    if path.exists():
        return np.load(path)["X"]
    from ml.model.data import dropout_rows
    drop = dropout_rows(len(rows), dropout, SEED) if dropout > 0 else np.zeros(len(rows), dtype=bool)
    ex = extractor()
    started = time.time()
    X = np.full((len(rows), len(FEATURES)), np.nan, dtype=np.float32)
    for i, r in enumerate(rows):
        X[i] = ex.row(r["problem_id"], r["code"], static=bool(static or drop[i]))
        if (i + 1) % 1000 == 0:
            log(f"  {name}: {i + 1}/{len(rows)}  {time.time() - started:.0f} s")
    np.savez_compressed(path, X=X)
    return X


# ------------------------------------------------------------------ labelled bundles

class Bundle:
    """Rows + features + labels of one evaluation slice."""

    def __init__(self, name, rows, X, Xs=None):
        from ml.model.data import soft_label_vector
        self.name, self.rows, self.X, self.Xs = name, rows, X, Xs
        self.ids = [r["id"] for r in rows]
        self.codes = [r["code"] for r in rows]
        self.groups = np.array([r["problem_id"] for r in rows])
        self.domain = np.array([domain_of(r["problem_id"]) for r in rows])
        Y = np.zeros((len(rows), len(LABELS)))
        y = np.full(len(rows), -1, dtype=np.int64)
        for i, r in enumerate(rows):
            if r["label"] in LABELS:
                Y[i] = soft_label_vector(r)
                y[i] = LABELS.index(r["label"])
        self.Y, self.y = Y, y
        self.scored = y >= 0               # R-llm junk ("GATE") rows belong to the gate, not the model

    def __len__(self):
        return len(self.rows)

    def sub(self, mask):
        mask = np.asarray(mask)
        idx = np.nonzero(mask)[0] if mask.dtype == bool else mask
        out = Bundle.__new__(Bundle)
        out.name = self.name
        out.rows = [self.rows[i] for i in idx]
        out.X = self.X[idx]
        out.Xs = None if self.Xs is None else self.Xs[idx]
        for field in ("ids", "codes"):
            setattr(out, field, [getattr(self, field)[i] for i in idx])
        for field in ("groups", "domain", "Y", "y", "scored"):
            setattr(out, field, getattr(self, field)[idx])
        return out


_BUNDLES = {}


def holdout_bundle():
    if "hold" not in _BUNDLES:
        _, hold = split_rows()
        _BUNDLES["hold"] = Bundle("HOLDOUT-P", hold, matrix("hold", hold), matrix("hold_static", hold, static=True))
    return _BUNDLES["hold"]


def realistic_bundle():
    if "real" not in _BUNDLES:
        rows = realistic_rows()
        b = Bundle("R-llm", rows, matrix("real", rows), matrix("real_static", rows, static=True))
        _BUNDLES["real"] = b
    return _BUNDLES["real"]


def train_bundle():
    """TRAIN rows as an ml.model.data.TrainData (15% feature dropout, exactly like `make train`)
    plus the static-extraction matrix `Xs` of the same rows (used by E9 and E13)."""
    if "train" not in _BUNDLES:
        from ml.model.data import TrainData, soft_label_vector, one_hot
        train, _ = split_rows()
        X = matrix("train", train, dropout=0.15)
        Xs = matrix("train_static", train, static=True)
        Y = np.stack([soft_label_vector(r) for r in train])
        y = np.array([LABELS.index(r["label"]) for r in train], dtype=np.int64)
        groups = np.array([r["problem_id"] for r in train])
        domain = np.array([domain_of(r["problem_id"]) for r in train])
        data = TrainData(X=X, Y=Y, y=y, groups=groups, domain=domain, codes=[r["code"] for r in train],
                         ids=[r["id"] for r in train])
        data.rows = train
        data.Xs = Xs
        _BUNDLES["train"] = data
    return _BUNDLES["train"]


# ------------------------------------------------------------------ the model (trained once)

_MODEL = None


def model_bundle():
    """dict(model=Diagnoser, oof_logits, folds, meta). Trains on TRAIN with the 12-config grid if needed."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    from ml.model.predict import Diagnoser
    from ml.model.train import GRID, make_folds, train
    data = train_bundle()
    info_path = CACHE / "model_info.json"
    folds = make_folds(data.groups, data.domain)
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info["data_hash"] == data.data_hash() and (CACHE / info["version"]).exists() and (CACHE / "oof.npz").exists():
            model = Diagnoser.load(CACHE / info["version"])
            oof = np.load(CACHE / "oof.npz")["logits"]
            _MODEL = dict(model=model, oof_logits=oof, folds=folds, meta=model.meta, data=data)
            return _MODEL
    log("training the diagnoser on TRAIN (12-config grid, 4 threads) ...")
    summary = train(data, CACHE, grid=GRID, log=log)
    np.savez_compressed(CACHE / "oof.npz", logits=summary["oof_logits"], distances=summary["oof_distances"])
    info_path.write_text(json.dumps({"version": summary["model_version"], "data_hash": data.data_hash()}),
                         encoding="utf-8")
    model = Diagnoser.load(summary["path"])
    _MODEL = dict(model=model, oof_logits=summary["oof_logits"], folds=folds, meta=model.meta, data=data)
    return _MODEL


def oof_proba():
    """Calibrated and masked out-of-fold probabilities of TRAIN (the ones E1 and E7 use)."""
    from ml.model.calibrate import softmax
    from ml.model.mask import allowed_classes, apply_mask
    mb = model_bundle()
    data, T = mb["data"], mb["model"].temperature
    return apply_mask(softmax(mb["oof_logits"], T), allowed_classes(data.X, data.features, data.labels))


# ------------------------------------------------------------------ scoring

def truth_vector(Y, y, pred):
    """A prediction that names any label of a soft-labelled row counts as that row's label."""
    hit = Y[np.arange(len(pred)), pred] > 0
    return np.where(hit, pred, y), hit


def macro_f1(Y, y, pred, label_set=None):
    from sklearn.metrics import f1_score
    truth, _ = truth_vector(Y, y, pred)
    labels = np.unique(y) if label_set is None else np.array(sorted(set(np.unique(y)) & set(label_set)))
    if len(labels) == 0:
        return float("nan")
    return float(f1_score(truth, pred, labels=labels, average="macro", zero_division=0))


def bootstrap_ci(stat, n, groups=None, n_boot=N_BOOT, seed=SEED):
    """95% percentile interval of stat(index). Cluster bootstrap by `groups` when given."""
    rng = np.random.default_rng(seed)
    if groups is not None:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        members = {g: np.nonzero(groups == g)[0] for g in uniq}
    values = []
    for _ in range(n_boot):
        if groups is None:
            idx = rng.integers(0, n, n)
        else:
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([members[g] for g in pick])
        v = stat(idx)
        if v == v:
            values.append(v)
    if not values:
        return [float("nan"), float("nan")]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def evaluate(Y, y, pred, groups=None, domain=None, cluster=False, n_boot=N_BOOT):
    """The standard block of numbers: macro-F1 + 95% CI, accuracy, per class, per problem, per domain."""
    from sklearn.metrics import precision_recall_fscore_support
    n = len(pred)
    truth, hit = truth_vector(Y, y, pred)
    f1 = macro_f1(Y, y, pred)
    ci = bootstrap_ci(lambda i: macro_f1(Y[i], y[i], pred[i]), n, groups if cluster else None, n_boot)
    p, r, f, s = precision_recall_fscore_support(truth, pred, labels=list(range(len(LABELS))), zero_division=0)
    per_class = {LABELS[k]: {"precision": float(p[k]), "recall": float(r[k]), "f1": float(f[k]), "support": int(s[k]),
                             "predicted": int((pred == k).sum())}
                 for k in range(len(LABELS)) if s[k] > 0 or (pred == k).any()}
    out = {"n": int(n), "macro_f1": f1, "macro_f1_ci95": ci, "ci_kind": "cluster bootstrap by problem" if cluster else "row bootstrap",
           "n_boot": n_boot, "acc": float(hit.mean()), "per_class": per_class}
    if groups is not None:
        out["per_problem_acc"] = {str(g): {"acc": float(hit[groups == g].mean()), "n": int((groups == g).sum())}
                                  for g in np.unique(groups)}
    if domain is not None:
        out["by_domain"] = {}
        for d in ("main", "dsa"):
            m = domain == d
            if m.any():
                out["by_domain"][d] = {"n": int(m.sum()), "macro_f1": macro_f1(Y[m], y[m], pred[m]),
                                       "acc": float(hit[m].mean())}
    out["by_class_family"] = {
        "M": {"n": int(np.isin(y, M_IDX).sum()), "macro_f1": macro_f1(Y, y, pred, M_IDX),
              "recall": float(hit[np.isin(y, M_IDX)].mean()) if np.isin(y, M_IDX).any() else None},
        "D": {"n": int(np.isin(y, D_IDX).sum()), "macro_f1": macro_f1(Y, y, pred, D_IDX),
              "recall": float(hit[np.isin(y, D_IDX)].mean()) if np.isin(y, D_IDX).any() else None}}
    return out


def fold_scores(Y, y, pred, folds):
    return [macro_f1(Y[va], y[va], pred[va]) for _, va in folds]


def cohen_kappa(a, b):
    a, b = list(a), list(b)
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


# ------------------------------------------------------------------ plots and cards

def confusion_plot(Y, y, pred, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    truth, _ = truth_vector(Y, y, pred)
    k = len(LABELS)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(truth, pred):
        cm[t, p] += 1
    rows = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, rows, out=np.zeros(cm.shape), where=rows > 0)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(k), LABELS, rotation=90, fontsize=7)
    ax.set_yticks(range(k), LABELS, fontsize=7)
    for i in range(k):
        for j in range(k):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true (colour = share of the true class)")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return cm


def write_card(card, name):
    """ml/eval/cards/<name>.json. `card` follows 03 §9.4 (id, title, slice, n, metrics, per_class, plots, caveat)
    plus extra keys."""
    CARDS.mkdir(parents=True, exist_ok=True)
    path = CARDS / f"{name}.json"
    path.write_text(json.dumps(card, indent=1, ensure_ascii=False, default=_json_default), encoding="utf-8")
    log(f"wrote {path}")
    return path


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def plot_path(name):
    CARDS.mkdir(parents=True, exist_ok=True)
    return CARDS / name


def model_version():
    return model_bundle()["model"].model_version


def predict(bundle, which="X"):
    """Calibrated, masked probabilities of the shipped-style model on a bundle (full or static features)."""
    mb = model_bundle()
    return mb["model"].proba(getattr(bundle, which))


if __name__ == "__main__":
    t = time.time()
    tb = train_bundle()
    log(f"TRAIN {len(tb)} rows, {len(set(tb.groups))} problems, nan share {np.isnan(tb.X).mean():.3f}; extractor failures {extractor().failures}")
    hb = holdout_bundle()
    rb = realistic_bundle()
    log(f"HOLDOUT-P {len(hb)} rows; R-llm {len(rb)} rows ({int(rb.scored.sum())} scored); extractor failures {extractor().failures}")
    mb = model_bundle()
    log(f"model {mb['model'].model_version}; {time.time() - t:.0f} s")
