"""E1 — grouped cross-validation by problem (03 §9.2).

5-fold StratifiedGroupKFold by problem, stratified by domain, on TRAIN (28 problems). The numbers come from
the out-of-fold predictions of the training run (calibrated, masked) — the same predictions E7 and E8 use.

    .venv\\Scripts\\python -m ml.eval.e01_grouped_cv
"""
from __future__ import annotations

import numpy as np

from ml.eval import _ea_common as C


def run():
    mb = C.model_bundle()
    data = mb["data"]
    P = C.oof_proba()
    pred = P.argmax(axis=1)
    ev = C.evaluate(data.Y, data.y, pred, data.groups, data.domain, cluster=True)
    folds = C.fold_scores(data.Y, data.y, pred, mb["folds"])
    dsa_d = (data.domain == "dsa") & np.isin(data.y, C.D_IDX)
    d_only = C.macro_f1(data.Y[dsa_d], data.y[dsa_d], pred[dsa_d])
    d_only_ci = C.bootstrap_ci(lambda i: C.macro_f1(data.Y[dsa_d][i], data.y[dsa_d][i], pred[dsa_d][i]),
                               int(dsa_d.sum()), data.groups[dsa_d])
    hashes = {r["ast_hash"] for r in data.rows}
    plot = C.plot_path("E01_confusion.png")
    C.confusion_plot(data.Y, data.y, pred, plot, f"E1 out-of-fold confusion, TRAIN n={len(data)} (calibrated, masked)")
    cv_meta = mb["meta"]["cv"]
    card = {
        "id": "E1", "title": "Grouped CV by problem",
        "slice": f"TRAIN ({len(set(data.groups))} problems, {len(mb['folds'])} folds, out-of-fold)", "n": int(len(data)),
        "model_version": mb["model"].model_version,
        "metrics": {
            "macro_f1": float(np.mean(folds)), "macro_f1_std": float(np.std(folds)), "macro_f1_fold_scores": folds,
            "macro_f1_pooled_oof": ev["macro_f1"], "macro_f1_pooled_oof_ci95_cluster": ev["macro_f1_ci95"],
            "acc": ev["acc"], "n_boot": ev["n_boot"],
            "d_classes_on_dsa_problems_macro_f1": d_only, "d_classes_on_dsa_problems_ci95_cluster": d_only_ci,
            "d_classes_on_dsa_problems_n": int(dsa_d.sum()),
            "trainer_reported_cv_macro_f1": cv_meta["macro_f1_mean"], "trainer_reported_cv_macro_f1_std": cv_meta["macro_f1_std"],
        },
        "targets": {"macro_f1": ">= 0.80 (red flag >= 0.98)", "d_classes_dsa": ">= 0.75 (red flag >= 0.98)"},
        "red_flag": {"macro_f1_ge_0.98": bool(np.mean(folds) >= 0.98), "d_on_dsa_ge_0.98": bool(d_only >= 0.98)},
        "per_class": ev["per_class"], "per_problem_acc": ev["per_problem_acc"],
        "by_domain": ev["by_domain"], "by_class_family": ev["by_class_family"],
        "effective_n": {"rows": int(len(data)), "unique_ast_hash": len(hashes)},
        "plots": [plot.name],
        "caveat": ("Synthetic rows from the same generator as the training rows, so this is optimistic (03 §9.2). "
                   "Folds hold out whole problems but the mutation operators are shared across folds. "
                   "macro_f1 is the mean of the per-fold macro-F1 over the classes present in each fold, computed from the "
                   "final OOF logits with the fitted temperature and masking; pooled_oof is one macro-F1 over all rows with a "
                   "cluster bootstrap by problem (1,000 resamples)."),
    }
    return card


def main():
    C.write_card(run(), "E01")


if __name__ == "__main__":
    main()
