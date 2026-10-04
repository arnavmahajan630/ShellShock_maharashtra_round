"""E15 — cross-domain transfer (03 §9.2).

(a) Train on TRAIN-MAIN only, test M-class rows on DSA problems.
(b) Full TRAIN model, metrics split by domain (grouped out-of-fold, so a row's problem
    was not in that fold's fit).
(c) Depth-1 stump on the five t_* features: can task meta tell D-classes from M-classes?
"""
from __future__ import annotations

import numpy as np

from ml.contracts.classes import DSA_CLASSES, LABELS, MAIN_CLASSES
from ml.contracts.feature_names import FEATURES, GROUP_C
from ml.eval._eb_data import OUT, bank_train_mask, holdout_mask, load_bundle
from ml.eval._eb_metrics import (
    cluster_mean_ci, depth1_auc, macro_f1, macro_recall, metric_ci, new_card,
    per_class_recall, write_card,
)
from ml.eval._eb_train import ensure_full, ensure_split, load_model, load_oof


def _m_ids():
    return [LABELS.index(name) for name in MAIN_CLASSES]


def _predict(model, bundle, index):
    index = np.asarray(index, dtype=np.int64)
    if len(index) == 0:
        empty_i = np.array([], dtype=np.int64)
        empty_f = np.array([], dtype=float)
        return empty_i, empty_i, np.array([]), empty_f
    probs = model.proba(bundle.X_full[index])
    pred = probs.argmax(axis=1)
    y = bundle.y[index]
    problems = np.asarray(bundle.problem_id[index])
    soft = (bundle.Y[index, pred] > 0).astype(float)
    return pred, y, problems, soft


def _recall_f1(y, pred, problems, seed):
    labels = _m_ids()
    return {
        "macro_recall": metric_ci(y, pred, problems, lambda a, b: macro_recall(a, b, labels), seed=seed),
        "macro_f1": metric_ci(y, pred, problems, lambda a, b: macro_f1(a, b, labels), seed=seed + 1),
        "primary_accuracy": cluster_mean_ci((pred == y).astype(float), problems, seed=seed + 2),
    }


def _oof_rows(oof, index):
    place = {int(i): n for n, i in enumerate(oof["index"])}
    rows = [place[int(i)] for i in index]
    return rows


def _domain_f1(oof, domain, seed):
    mask = np.asarray(oof["domain"]) == domain
    y = np.asarray(oof["y"])[mask]
    pred = np.asarray(oof["probs"])[mask].argmax(axis=1)
    problems = np.asarray(oof["problem"])[mask]
    labels = list(range(len(LABELS)))
    return {
        "macro_f1": metric_ci(y, pred, problems, lambda a, b: macro_f1(a, b, labels), seed=seed),
        "accuracy": cluster_mean_ci((pred == y).astype(float), problems, seed=seed + 1),
        "n": int(mask.sum()),
    }


def _plot(bars):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [item[0] for item in bars]
    est = np.asarray([item[1]["estimate"] or 0 for item in bars], dtype=float)
    lo = np.asarray([
        item[1]["estimate"] if item[1].get("lo") is None else item[1]["lo"] for item in bars
    ], dtype=float)
    hi = np.asarray([
        item[1]["estimate"] if item[1].get("hi") is None else item[1]["hi"] for item in bars
    ], dtype=float)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(len(names))
    ax.bar(x, est, color="#3c6e71", zorder=2)
    ax.errorbar(x, est, yerr=np.vstack([np.maximum(est - lo, 0), np.maximum(hi - est, 0)]),
                fmt="none", ecolor="#1d1a1a", capsize=3, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("E15  cross-domain transfer")
    fig.tight_layout()
    path = OUT / "e15_cross_domain.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return "e15_cross_domain.png"


def run():
    bundle = load_bundle()
    full = ensure_full(bundle)
    main_index = np.flatnonzero(bank_train_mask(bundle) & (bundle.domain == "main"))
    main = ensure_split(bundle, "E15 TRAIN-MAIN", main_index, "e15_main", save_oof=True)
    m_ids = _m_ids()
    dsa_m = np.flatnonzero(
        bank_train_mask(bundle) & (bundle.domain == "dsa") & np.isin(bundle.y, m_ids))
    main_model = load_model(main)
    pred, y, problems, soft = _predict(main_model, bundle, dsa_m)
    trained_main = _recall_f1(y, pred, problems, seed=42)
    trained_main["soft_accuracy"] = cluster_mean_ci(soft, problems, seed=45)
    trained_main["per_class"] = per_class_recall(y, pred, m_ids, MAIN_CLASSES)

    oof = load_oof("full")
    rows = _oof_rows(oof, dsa_m)
    oof_pred = np.asarray(oof["probs"])[rows].argmax(axis=1)
    oof_y = np.asarray(oof["y"])[rows]
    oof_problems = np.asarray(oof["problem"])[rows]
    full_same_rows = _recall_f1(oof_y, oof_pred, oof_problems, seed=52)
    by_domain = {"main": _domain_f1(oof, "main", seed=60), "dsa": _domain_f1(oof, "dsa", seed=70)}

    hold_dsa_m = np.flatnonzero(
        holdout_mask(bundle) & (bundle.domain == "dsa") & np.isin(bundle.y, m_ids))
    full_model = load_model(full)
    hold_main_pred, hold_y, hold_problems, _ = _predict(main_model, bundle, hold_dsa_m)
    hold_full_pred, _, _, _ = _predict(full_model, bundle, hold_dsa_m)
    holdout = {
        "trained_main": _recall_f1(hold_y, hold_main_pred, hold_problems, seed=80),
        "full_model": _recall_f1(hold_y, hold_full_pred, hold_problems, seed=90),
        "n": int(len(hold_dsa_m)),
    }

    class_ids = [LABELS.index(name) for name in MAIN_CLASSES + DSA_CLASSES]
    shortcut_index = np.flatnonzero(bank_train_mask(bundle) & np.isin(bundle.y, class_ids))
    d_ids = {LABELS.index(name) for name in DSA_CLASSES}
    y_bin = np.array([1 if int(v) in d_ids else 0 for v in bundle.y[shortcut_index]])
    cols = [FEATURES.index(name) for name in GROUP_C]
    X = bundle.X_full[shortcut_index][:, cols]
    auc, split = depth1_auc(X, y_bin, bundle.problem_id[shortcut_index], seed=42)
    feature = None if split["feature_index"] < 0 else GROUP_C[split["feature_index"]]
    shortcut = {"auc": auc, "stump_feature": feature, "stump_threshold": split["threshold"],
                "n": int(len(shortcut_index)), "n_pos_D": int(y_bin.sum()),
                "n_neg_M": int((y_bin == 0).sum())}

    bars = [
        ("M recall\nmain-only", trained_main["macro_recall"]),
        ("M F1\nmain-only", trained_main["macro_f1"]),
        ("M recall\nfull OOF", full_same_rows["macro_recall"]),
        ("M F1\nfull OOF", full_same_rows["macro_f1"]),
        ("macro-F1\nmain OOF", by_domain["main"]["macro_f1"]),
        ("macro-F1\nDSA OOF", by_domain["dsa"]["macro_f1"]),
        ("shortcut\nAUC", shortcut["auc"]),
    ]
    plot = _plot(bars)
    card = new_card(
        "E15",
        "Cross-domain transfer",
        "TRAIN-MAIN is the 13 main problems. Test (a) is M-class rows on the 15 DSA training "
        "problems. (b) is grouped out-of-fold of the model trained on all of TRAIN. "
        "(c) is a depth-1 stump on t_* only, grouped out-of-fold, D-class vs M-class.",
        n=int(len(dsa_m)),
        metrics={
            "m_recall_on_dsa_trained_main_only": trained_main["macro_recall"],
            "m_f1_on_dsa_trained_main_only": trained_main["macro_f1"],
            "m_primary_accuracy_trained_main_only": trained_main["primary_accuracy"],
            "m_soft_accuracy_trained_main_only": trained_main["soft_accuracy"],
            "m_recall_on_dsa_full_oof": full_same_rows["macro_recall"],
            "m_f1_on_dsa_full_oof": full_same_rows["macro_f1"],
            "full_model_oof_by_domain": by_domain,
            "holdout_dsa_m_class": holdout,
            "shortcut_auc": shortcut["auc"],
        },
        caveat=(
            "Same generator on both domains, so this is transfer across problem surfaces, not "
            "across authors. The main-only test uses full serving features. The full-model "
            "numbers on those same rows are out-of-fold, and 15% of training rows had trace "
            "features dropped, so that control is slightly harder. HOLDOUT-P DSA M-class rows "
            "were in neither fit and are scored with full features for both models. The stump "
            "sees only the five coarse task features. Default LightGBM config, no grid. "
            "A shortcut AUC above 0.8 is the red flag in 03 §11's risk table."
        ),
        per_class=trained_main["per_class"],
        plots=[plot],
        extra={
            "full_model_version": full["model_version"],
            "main_model_version": main["model_version"],
            "n_train_main": int(len(main_index)),
            "n_dsa_m_train_problems": int(len(dsa_m)),
            "shortcut": {"feature": feature, "threshold": split["threshold"],
                         "n": shortcut["n"], "n_D": shortcut["n_pos_D"], "n_M": shortcut["n_neg_M"]},
        },
    )
    write_card(card, "e15_card.json")
    return card


if __name__ == "__main__":
    run()
