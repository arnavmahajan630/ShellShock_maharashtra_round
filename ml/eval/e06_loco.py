"""E6 — leave-one-class-out novelty (03 §4.8, §9.2).

Strict retrains without class k and without CLASS_DEFINING_FEATURES[k]. Naive keeps those
features. The score is the kNN distance. Fixer k is not run. U1/U2 are scored only if a
slice file exists.
"""
from __future__ import annotations

import numpy as np

from ml.contracts.classes import DSA_CLASSES, LABELS, MAIN_CLASSES, MISCONCEPTIONS
from ml.contracts.feature_names import CLASS_DEFINING_FEATURES, MASK_PRECONDITIONS
from ml.eval._eb_data import (
    OUT, ROOT, STAMP, bank_train_mask, columns_for, holdout_mask, load_bundle, make_train_data,
)
from ml.eval._eb_metrics import auc_ci, new_card, read_json, save_fig, write_card, write_json
from ml.eval._eb_train import fit_diagnoser, load_model
from ml.model.novelty import assess

COVERAGES = np.round(np.arange(0.1, 1.01, 0.1), 2)


def _u_files():
    folder = ROOT / "ml" / "data"
    found = []
    for path in folder.glob("*.jsonl"):
        name = path.name.lower()
        if "u1" in name or "u2" in name or name.startswith("unseen"):
            found.append(path.name)
    return found


def _indices(bundle, class_name):
    class_index = LABELS.index(class_name)
    train = np.flatnonzero(bank_train_mask(bundle) & (bundle.Y[:, class_index] <= 0))
    pos = np.flatnonzero((bundle.kind == "bank") & (bundle.label == class_name) & (bundle.y >= 0))
    neg = np.flatnonzero(
        holdout_mask(bundle) & (bundle.label != class_name) & (bundle.Y[:, class_index] <= 0))
    return train, pos, neg


def _risk_curve(is_pos, dist):
    order = np.argsort(dist, kind="mergesort")
    n = len(dist)
    curve = []
    for coverage in COVERAGES:
        take = max(1, int(round(float(coverage) * n)))
        curve.append(float(is_pos[order[:take]].mean()))
    return curve


def _domain_auc(is_pos, dist, domain, problems, want):
    mask = domain == want
    if int(mask.sum()) < 2:
        return None
    return auc_ci(is_pos[mask], dist[mask], problems[mask], seed=42)


def _strict_columns(class_name):
    """Features to remove, and defining features that must stay as NaN so masking can still find them.

    `allowed_classes` looks every mask precondition up by name. D08's list overlaps
    CLASS_DEFINING_FEATURES, so those columns are blanked instead of deleted.
    """
    drop = set(CLASS_DEFINING_FEATURES[class_name])
    protected = {name for names in MASK_PRECONDITIONS.values() for name in names}
    return sorted(drop - protected), sorted(drop & protected)


def _blank(X, features, names):
    if not names:
        return X
    X = np.array(X, dtype=np.float32, copy=True)
    for name in names:
        if name in features:
            X[:, list(features).index(name)] = np.nan
    return X


def _run_mode(bundle, class_name, mode):
    dest = OUT / "e06" / mode / class_name
    cached = read_json(dest / "result.json")
    if cached and cached.get("stamp") == STAMP:
        print(f"[E6 {mode} {class_name}] reuse", flush=True)
        return cached
    train_index, pos, neg = _indices(bundle, class_name)
    remove, blank = _strict_columns(class_name) if mode == "strict" else ([], [])
    data = make_train_data(bundle, train_index, drop_features=remove)
    data.X = _blank(data.X, data.features, blank)
    summary = fit_diagnoser(data, dest, f"E6 {mode} {class_name}")
    model = load_model(summary)
    index = np.concatenate([pos, neg])
    X = _blank(bundle.X_full[index][:, columns_for(model.features)], model.features, blank)
    dist = model.knn_distance(X)
    probs = model.proba(X)
    flags = np.array([
        assess(float(dist[i]), probs[i], model.tau_d, model.tau_p, model.labels)["abstain"]
        for i in range(len(index))
    ])
    is_pos = np.zeros(len(index), dtype=int)
    is_pos[:len(pos)] = 1
    problems = np.asarray(bundle.problem_id[index])
    domain = np.asarray(bundle.domain[index])
    precision = None
    if flags.any():
        precision = float(is_pos[flags].mean())
    result = {
        "stamp": STAMP,
        "class": class_name,
        "mode": mode,
        "model_version": summary["model_version"],
        "seconds": summary["seconds"],
        "n_train": int(len(train_index)),
        "n_pos": int(len(pos)),
        "n_neg": int(len(neg)),
        "dropped_features": list(remove),
        "blanked_features": list(blank),
        "auroc": auc_ci(is_pos, dist, problems, seed=42),
        "auroc_main": _domain_auc(is_pos, dist, domain, problems, "main"),
        "auroc_dsa": _domain_auc(is_pos, dist, domain, problems, "dsa"),
        "abstain_precision": precision,
        "abstain_rate_on_class": float(flags[:len(pos)].mean()) if len(pos) else None,
        "abstain_rate_on_holdout": float(flags[len(pos):].mean()) if len(neg) else None,
        "risk": _risk_curve(is_pos, dist),
    }
    write_json(dest / "result.json", result)
    return result


def _mean_interval(rows, key):
    values = []
    for row in rows:
        item = row.get(key)
        if isinstance(item, dict):
            item = item.get("estimate")
        if item is not None:
            values.append(float(item))
    if not values:
        return {"estimate": None, "lo": None, "hi": None, "n_classes": 0}
    rng = np.random.default_rng(42)
    stats = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(1000)]
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return {"estimate": float(np.mean(values)), "lo": float(lo), "hi": float(hi), "n_classes": len(values)}


def _group_mean(rows, key, classes):
    chosen = [row for row in rows if row["class"] in classes]
    return _mean_interval(chosen, key)


def _curves(rows):
    if not rows:
        return [None] * len(COVERAGES)
    return np.mean([row["risk"] for row in rows], axis=0).tolist()


def _plots(strict, naive):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [row["class"] for row in strict]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(names))
    width = 0.38
    s_est = [row["auroc"]["estimate"] or 0 for row in strict]
    n_est = [row["auroc"]["estimate"] or 0 for row in naive]
    s_err = _yerr(strict)
    n_err = _yerr(naive)
    ax.bar(x - width / 2, s_est, width, yerr=s_err, label="strict", color="#3c6e71", capsize=2)
    ax.bar(x + width / 2, n_est, width, yerr=n_err, label="naive", color="#c44536", capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUROC of kNN distance")
    ax.set_title("E6  leave-one-class-out novelty")
    ax.legend(frameon=False)
    bars = save_fig(fig, "e06_loco.png")
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.plot(COVERAGES, _curves(strict), marker="o", color="#3c6e71", label="strict")
    ax.plot(COVERAGES, _curves(naive), marker="o", color="#c44536", label="naive")
    ax.set_xlabel("coverage (lowest distance first)")
    ax.set_ylabel("share of accepted rows from the held-out class")
    ax.set_ylim(0, 1)
    ax.set_title("E6  risk–coverage of the novelty score")
    ax.legend(frameon=False)
    risk = save_fig(fig, "e06_risk_coverage.png")
    return [bars, risk]


def _yerr(rows):
    est, lo, hi = [], [], []
    for row in rows:
        e = row["auroc"]["estimate"] if row["auroc"]["estimate"] is not None else 0.0
        est.append(e)
        lo.append(e if row["auroc"]["lo"] is None else row["auroc"]["lo"])
        hi.append(e if row["auroc"]["hi"] is None else row["auroc"]["hi"])
    est = np.asarray(est, dtype=float)
    return np.vstack([np.maximum(est - np.asarray(lo), 0), np.maximum(np.asarray(hi) - est, 0)])


def run():
    bundle = load_bundle()
    u_found = _u_files()
    strict, naive = [], []
    for class_name in MISCONCEPTIONS:
        strict.append(_run_mode(bundle, class_name, "strict"))
        naive.append(_run_mode(bundle, class_name, "naive"))
    per_class = {}
    for left, right in zip(strict, naive):
        per_class[left["class"]] = {
            "n_pos": left["n_pos"], "n_neg": left["n_neg"],
            "strict_auroc": left["auroc"], "naive_auroc": right["auroc"],
            "strict_auroc_main": left["auroc_main"], "strict_auroc_dsa": left["auroc_dsa"],
            "naive_auroc_main": right["auroc_main"], "naive_auroc_dsa": right["auroc_dsa"],
            "strict_abstain_precision": left["abstain_precision"],
            "naive_abstain_precision": right["abstain_precision"],
            "strict_abstain_rate_on_class": left["abstain_rate_on_class"],
            "naive_abstain_rate_on_class": right["abstain_rate_on_class"],
        }
    plots = _plots(strict, naive)
    card = new_card(
        "E6",
        "Unseen misconceptions",
        "LOCO: class k removed from TRAIN. Positives are all bank rows of k. "
        "Negatives are seen-class HOLDOUT-P rows. Score is kNN distance in A+B+R.",
        n=int(sum(row["n_pos"] for row in strict)),
        metrics={
            "strict_auroc_mean": _mean_interval(strict, "auroc"),
            "naive_auroc_mean": _mean_interval(naive, "auroc"),
            "strict_auroc_M": _group_mean(strict, "auroc", set(MAIN_CLASSES)),
            "strict_auroc_D": _group_mean(strict, "auroc", set(DSA_CLASSES)),
            "naive_auroc_M": _group_mean(naive, "auroc", set(MAIN_CLASSES)),
            "naive_auroc_D": _group_mean(naive, "auroc", set(DSA_CLASSES)),
            "strict_auroc_main_problems": _mean_interval(strict, "auroc_main"),
            "strict_auroc_dsa_problems": _mean_interval(strict, "auroc_dsa"),
            "naive_auroc_main_problems": _mean_interval(naive, "auroc_main"),
            "naive_auroc_dsa_problems": _mean_interval(naive, "auroc_dsa"),
            "strict_abstain_precision_mean": _mean_interval(strict, "abstain_precision"),
            "naive_abstain_precision_mean": _mean_interval(naive, "abstain_precision"),
            "strict_abstain_rate_on_class_mean": _mean_interval(strict, "abstain_rate_on_class"),
            "naive_abstain_rate_on_class_mean": _mean_interval(naive, "abstain_rate_on_class"),
            "risk_coverage_strict": {"coverage": COVERAGES.tolist(), "risk": _curves(strict)},
            "risk_coverage_naive": {"coverage": COVERAGES.tolist(), "risk": _curves(naive)},
        },
        caveat=(
            "Strict drops CLASS_DEFINING_FEATURES[k] from the retrained model and therefore "
            "from the novelty space. Naive keeps them; that AUROC is optimistic when k's "
            "signature is near zero on every class the model saw. Fixer k is not run. "
            "tau_d and tau_p come from the retrained model's own out-of-fold pass and are "
            "used only for the abstain flag. The AUROC uses the distance itself. "
            "U1/U2 were not scored: no unseen-class item file is in ml/data. "
            "Default LightGBM config, no grid. The mean interval resamples classes. "
            "Card n counts class-k rows once per class."
        ),
        per_class=per_class,
        plots=plots,
        extra={
            "skipped": [{
                "part": "U1/U2",
                "reason": "no U1/U2 jsonl in ml/data; abstain rate on that slice was not invented",
                "files": u_found,
            }],
            "n_classes": len(MISCONCEPTIONS),
        },
    )
    write_card(card, "e06_card.json")
    return card


if __name__ == "__main__":
    run()
