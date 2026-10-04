"""E2 — unseen problems (03 §9.2).

The model trained on TRAIN (28 problems) is tested on HOLDOUT-P: P04, P09, P13 (main) and Q04, Q09, Q16 (DSA).
These problems were never used for fitting, calibration, thresholds or the grid.

    .venv\\Scripts\\python -m ml.eval.e02_unseen_problems
"""
from __future__ import annotations

import numpy as np

from ml.eval import _ea_common as C


def run():
    mb = C.model_bundle()
    b = C.holdout_bundle()
    pred = mb["model"].proba(b.X).argmax(axis=1)
    ev = C.evaluate(b.Y, b.y, pred, b.groups, b.domain, cluster=True)
    dsa_d = (b.domain == "dsa") & np.isin(b.y, C.D_IDX)
    d_only = C.macro_f1(b.Y[dsa_d], b.y[dsa_d], pred[dsa_d])
    d_ci = C.bootstrap_ci(lambda i: C.macro_f1(b.Y[dsa_d][i], b.y[dsa_d][i], pred[dsa_d][i]), int(dsa_d.sum()),
                          b.groups[dsa_d])
    train = C.train_bundle()
    overlap = {r["ast_hash"] for r in train.rows} & {r["ast_hash"] for r in b.rows}
    plot = C.plot_path("E02_confusion.png")
    C.confusion_plot(b.Y, b.y, pred, plot, f"E2 HOLDOUT-P confusion, n={len(b)}")
    present = sorted(set(LABELS_PRESENT(b)))
    card = {
        "id": "E2", "title": "Unseen problems",
        "slice": "HOLDOUT-P (P04, P09, P13, Q04, Q09, Q16)", "n": int(len(b)), "model_version": mb["model"].model_version,
        "metrics": {
            "macro_f1": ev["macro_f1"], "macro_f1_ci95": ev["macro_f1_ci95"], "ci_kind": ev["ci_kind"], "n_boot": ev["n_boot"],
            "n_clusters": 6, "acc": ev["acc"],
            "main_macro_f1": ev["by_domain"]["main"]["macro_f1"], "main_n": ev["by_domain"]["main"]["n"],
            "dsa_macro_f1": ev["by_domain"]["dsa"]["macro_f1"], "dsa_n": ev["by_domain"]["dsa"]["n"],
            "d_classes_on_dsa_macro_f1": d_only, "d_classes_on_dsa_ci95_cluster": d_ci, "d_classes_on_dsa_n": int(dsa_d.sum()),
            "m_classes_macro_f1": ev["by_class_family"]["M"]["macro_f1"], "d_classes_macro_f1": ev["by_class_family"]["D"]["macro_f1"],
        },
        "targets": {"macro_f1": ">= 0.70 (red flag 1.00)", "d_classes_dsa": ">= 0.65 (red flag 1.00)"},
        "red_flag": {"macro_f1_eq_1": bool(ev["macro_f1"] >= 0.9999), "d_on_dsa_eq_1": bool(d_only >= 0.9999)},
        "per_class": ev["per_class"], "per_problem_acc": ev["per_problem_acc"],
        "by_domain": ev["by_domain"], "by_class_family": ev["by_class_family"],
        "classes_present": present,
        "leakage_check": {"ast_hash_overlap_with_train": len(overlap)},
        "plots": [plot.name],
        "caveat": ("Synthetic rows from the same generator (same operators) as the training data, on 6 problems only: "
                   "the cluster bootstrap resamples 6 clusters, so the interval is wide and slightly optimistic about "
                   "its own width. Not realistic code (that is E4)."),
    }
    return card


def LABELS_PRESENT(b):
    from ml.contracts.classes import LABELS
    return [LABELS[i] for i in np.unique(b.y)]


def main():
    C.write_card(run(), "E02")


if __name__ == "__main__":
    main()
