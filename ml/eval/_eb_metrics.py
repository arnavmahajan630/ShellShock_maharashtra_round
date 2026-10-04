"""Intervals, card files, and plots shared by the E-b experiments."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier

from ml.eval._eb_data import N_BOOT, OUT, SEED

CARD_KEYS = ("id", "title", "slice", "n", "metrics", "caveat")


def clean(obj):
    """JSON-safe copy. Floats stay at 6 decimals so a card does not grow fake precision."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        if not np.isfinite(value):
            return None
        return round(value, 6)
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), indent=1), encoding="utf-8", newline="\n")


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def cluster_mean_ci(values, clusters, n_boot=N_BOOT, seed=SEED):
    """95% cluster bootstrap interval of the mean. Clusters are problem ids (03 §9.2)."""
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    n = int(len(values))
    if n == 0:
        return {"estimate": None, "lo": None, "hi": None, "n": 0, "n_clusters": 0}
    estimate = float(values.mean())
    uniq, inverse = np.unique(clusters, return_inverse=True)
    groups = [np.flatnonzero(inverse == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    n_g = len(groups)
    for draw in range(n_boot):
        chosen = rng.integers(0, n_g, size=n_g)
        stats[draw] = values[np.concatenate([groups[i] for i in chosen])].mean()
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return {"estimate": estimate, "lo": float(lo), "hi": float(hi), "n": n, "n_clusters": int(n_g)}


def safe_auc(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    ok = np.isfinite(scores)
    y_true, scores = y_true[ok], scores[ok]
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def auc_ci(y_true, scores, clusters, n_boot=N_BOOT, seed=SEED):
    """95% cluster bootstrap of AUROC. Draws with one class are dropped."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    clusters = np.asarray(clusters)
    point = safe_auc(y_true, scores)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    base = {"estimate": point, "lo": None, "hi": None, "n": int(len(y_true)), "n_pos": n_pos, "n_neg": n_neg}
    if point is None:
        return base
    uniq, inverse = np.unique(clusters, return_inverse=True)
    groups = [np.flatnonzero(inverse == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    valid = []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in chosen])
        score = safe_auc(y_true[idx], scores[idx])
        if score is not None:
            valid.append(score)
    if len(valid) >= 100:
        lo, hi = np.quantile(valid, [0.025, 0.975])
        base["lo"], base["hi"] = float(lo), float(hi)
    return base


def macro_f1(y_true, y_pred, labels):
    """Macro-F1 over labels that occur in `y_true`."""
    y_true = np.asarray(y_true)
    present = [label for label in labels if np.any(y_true == label)]
    if not present:
        return None
    return float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))


def macro_recall(y_true, y_pred, labels):
    """Unweighted mean of per-class recall, classes with no true rows omitted."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    recalls = []
    for label in labels:
        mask = y_true == label
        if mask.any():
            recalls.append(float((y_pred[mask] == label).mean()))
    if not recalls:
        return None
    return float(np.mean(recalls))


def per_class_recall(y_true, y_pred, labels, names):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {}
    for name, label in zip(names, labels):
        mask = y_true == label
        if mask.any():
            out[name] = {"recall": float((y_pred[mask] == label).mean()), "n": int(mask.sum())}
    return out


def metric_ci(y_true, y_pred, clusters, metric, n_boot=N_BOOT, seed=SEED):
    """95% cluster bootstrap of a metric(y_true, y_pred) -> float."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    clusters = np.asarray(clusters)
    point = metric(y_true, y_pred)
    if point is None or len(y_true) == 0:
        return {"estimate": point, "lo": None, "hi": None, "n": int(len(y_true)), "n_clusters": 0}
    uniq, inverse = np.unique(clusters, return_inverse=True)
    groups = [np.flatnonzero(inverse == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in chosen])
        value = metric(y_true[idx], y_pred[idx])
        if value is not None:
            stats.append(value)
    lo = hi = None
    if len(stats) >= 100:
        lo, hi = (float(v) for v in np.quantile(stats, [0.025, 0.975]))
    return {"estimate": float(point), "lo": lo, "hi": hi, "n": int(len(y_true)),
            "n_clusters": int(len(groups))}


def depth1_auc(X, y, groups, seed=SEED, n_boot=N_BOOT):
    """Grouped out-of-fold AUROC of a depth-1 tree. Returns the interval and the in-sample split."""
    X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    n_splits = min(5, len(np.unique(groups)))
    oof = np.full(len(y), np.nan)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, valid_idx in splitter.split(X, y, groups):
        clf = DecisionTreeClassifier(max_depth=1, random_state=seed)
        clf.fit(X[train_idx], y[train_idx])
        oof[valid_idx] = _positive_proba(clf, X[valid_idx])
    fitted = DecisionTreeClassifier(max_depth=1, random_state=seed).fit(X, y)
    feature = int(fitted.tree_.feature[0])
    threshold = None if feature < 0 else float(fitted.tree_.threshold[0])
    return auc_ci(y, oof, groups, n_boot=n_boot, seed=seed), {"feature_index": feature, "threshold": threshold}


def _positive_proba(clf, X):
    proba = clf.predict_proba(X)
    if 1 not in clf.classes_:
        return np.zeros(len(X))
    return proba[:, list(clf.classes_).index(1)]


def new_card(experiment_id, title, slice_name, n, metrics, caveat, per_class=None, plots=None, extra=None):
    card = {
        "id": experiment_id,
        "title": title,
        "slice": slice_name,
        "n": int(n),
        "metrics": metrics,
        "per_class": per_class or {},
        "plots": list(plots or []),
        "caveat": caveat,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        card.update(extra)
    missing = [key for key in CARD_KEYS if key not in card]
    if missing:
        raise ValueError(f"card missing {missing}")
    return card


def write_card(card, filename):
    path = OUT / filename
    write_json(path, card)
    print(f"wrote {path}", flush=True)
    return path


def save_fig(fig, filename):
    import matplotlib.pyplot as plt
    path = OUT / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return filename


def bar_with_intervals(names, estimates, los, his, ylabel, title, filename, color="#3c6e71"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    est = np.asarray(estimates, dtype=float)
    lo = np.asarray([e if v is None else v for e, v in zip(est, los)], dtype=float)
    hi = np.asarray([e if v is None else v for e, v in zip(est, his)], dtype=float)
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(names) + 2), 4.6))
    x = np.arange(len(names))
    ax.bar(x, est, color=color, zorder=2)
    ax.errorbar(x, est, yerr=np.vstack([np.maximum(est - lo, 0), np.maximum(hi - est, 0)]),
                fmt="none", ecolor="#1d1a1a", capsize=3, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(np.nanmean(est), color="#c44536", linestyle="--", linewidth=1, label="unweighted mean")
    ax.legend(frameon=False)
    return save_fig(fig, filename)
