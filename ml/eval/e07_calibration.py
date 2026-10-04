"""E7 — calibration (03 §5.4, §9.2).

Out-of-fold probabilities of TRAIN before and after the fitted temperature: ECE (10 equal-width bins), NLL, Brier,
reliability diagram. Also ECE of the final model, with no refit of T, on HOLDOUT-P and on R-llm.

    .venv\\Scripts\\python -m ml.eval.e07_calibration
"""
from __future__ import annotations

import numpy as np

from ml.contracts.classes import LABELS
from ml.eval import _ea_common as C
from ml.model import calibrate as CAL
from ml.model.mask import allowed_classes, apply_mask


def _masked(logits, T, X, features):
    return apply_mask(CAL.softmax(logits, T), allowed_classes(X, features, LABELS))


def _block(P, Y):
    correct = CAL.is_correct(P, Y)
    return {"ece": CAL.ece(P, correct), "nll": CAL.nll(P, Y), "brier": CAL.brier(P, Y),
            "acc": float(correct.mean()), "mean_confidence": float(P.max(axis=1).mean())}


def run():
    mb = C.model_bundle()
    data, model = mb["data"], mb["model"]
    logits, T = mb["oof_logits"], model.temperature
    cal = CAL.calibrate(logits, data.Y)            # the trainer's numbers (unmasked), re-derived here
    before_m, after_m = _masked(logits, 1.0, data.X, data.features), _masked(logits, T, data.X, data.features)
    ece_ci = C.bootstrap_ci(lambda i: CAL.ece(after_m[i], CAL.is_correct(after_m[i], data.Y[i])), len(data), data.groups)
    ece_ci_unmasked = C.bootstrap_ci(
        lambda i: CAL.ece(CAL.softmax(logits[i], T), CAL.is_correct(CAL.softmax(logits[i], T), data.Y[i])),
        len(data), data.groups)

    # per predicted class: is the confidence honest for that class? (n is small for several)
    pred = after_m.argmax(axis=1)
    correct = CAL.is_correct(after_m, data.Y)
    conf = after_m.max(axis=1)
    per_class = {}
    for k, name in enumerate(LABELS):
        m = pred == k
        if m.sum():
            per_class[name] = {"n_predicted": int(m.sum()), "acc": float(correct[m].mean()),
                               "mean_confidence": float(conf[m].mean()),
                               "gap": float(conf[m].mean() - correct[m].mean())}
    flagged = {k: v for k, v in per_class.items() if v["n_predicted"] >= 30 and abs(v["gap"]) > 0.10}

    other = {}
    hb, rb = C.holdout_bundle(), C.realistic_bundle()
    for name, b in (("HOLDOUT-P", hb), ("R-llm", rb)):
        m = b.scored
        P = model.proba(b.X[m])
        other[name] = dict(_block(P, b.Y[m]), n=int(m.sum()),
                           ece_ci95_row_bootstrap=C.bootstrap_ci(
                               lambda i, P=P, Y=b.Y[m]: CAL.ece(P[i], CAL.is_correct(P[i], Y[i])), int(m.sum()),
                               b.groups[m] if name == "HOLDOUT-P" else None))

    # Informational: the temperature that minimises ECE instead of NLL (NOT what the trainer fits or ships).
    grid = np.round(np.arange(0.5, 5.0001, 0.05), 2)
    ece_by_T = [CAL.ece(_masked(logits, t, data.X, data.features), CAL.is_correct(_masked(logits, t, data.X, data.features), data.Y))
                for t in grid]
    T_ece = float(grid[int(np.argmin(ece_by_T))])
    P_ece = _masked(logits, T_ece, data.X, data.features)
    alt = dict(_block(P_ece, data.Y), temperature=T_ece)
    alt_other = {}
    for name, b in (("HOLDOUT-P", hb), ("R-llm", rb)):
        m = b.scored
        Pa = apply_mask(CAL.softmax(model.logits(b.X[m]), T_ece), allowed_classes(b.X[m], model.features, model.labels))
        alt_other[name] = dict(_block(Pa, b.Y[m]), n=int(m.sum()))
    # NLL is dominated by the few confident mistakes on whole unseen problems
    nll_row = -(data.Y * np.log(np.clip(before_m, 1e-12, 1))).sum(axis=1)
    top_share = float(np.sort(nll_row)[::-1][: max(1, len(nll_row) // 20)].sum() / nll_row.sum())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, P, title in ((axes[0], before_m, "before temperature (T=1)"), (axes[1], after_m, f"after temperature (T={T:.3f})")):
        rel = CAL.reliability(P, CAL.is_correct(P, data.Y))
        xs = [r["confidence"] for r in rel if r["count"]]
        ys = [r["accuracy"] for r in rel if r["count"]]
        sizes = [r["count"] for r in rel if r["count"]]
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.scatter(xs, ys, s=[20 + 300 * s / max(sizes) for s in sizes], alpha=0.7)
        ax.plot(xs, ys, lw=1)
        ax.set_title(f"{title}\nECE {CAL.ece(P, CAL.is_correct(P, data.Y)):.3f}", fontsize=9)
        ax.set_xlabel("mean confidence")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("accuracy")
    fig.suptitle(f"E7 reliability, TRAIN out-of-fold, n={len(data)} (masked); dot size = bin count", fontsize=9)
    fig.tight_layout()
    plot = C.plot_path("E07_reliability.png")
    fig.savefig(plot, dpi=130)
    plt.close(fig)

    card = {
        "id": "E7", "title": "Calibration", "slice": f"TRAIN out-of-fold ({len(data)} rows, {len(set(data.groups))} problems)",
        "n": int(len(data)), "model_version": model.model_version,
        "metrics": {
            "temperature": T,
            "ece_before": _block(before_m, data.Y)["ece"], "ece_after": _block(after_m, data.Y)["ece"],
            "ece_after_ci95_cluster": ece_ci,
            "nll_before": _block(before_m, data.Y)["nll"], "nll_after": _block(after_m, data.Y)["nll"],
            "brier_before": _block(before_m, data.Y)["brier"], "brier_after": _block(after_m, data.Y)["brier"],
            "unmasked": {"ece_before": cal["ece_before"], "ece_after": cal["ece_after"], "ece_after_ci95_cluster": ece_ci_unmasked,
                         "nll_before": cal["nll_before"], "nll_after": cal["nll_after"],
                         "brier_before": cal["brier_before"], "brier_after": cal["brier_after"]},
            "n_boot": C.N_BOOT,
        },
        "target": "ECE after temperature <= 0.08",
        "target_met": bool(_block(after_m, data.Y)["ece"] <= 0.08),
        "reliability_after": CAL.reliability(after_m, CAL.is_correct(after_m, data.Y)),
        "reliability_before": CAL.reliability(before_m, CAL.is_correct(before_m, data.Y)),
        "per_predicted_class": per_class, "per_class_flagged_gap_gt_0.10_n_ge_30": flagged,
        "final_model_no_refit": other,
        "finding": ("Temperature scaling by NLL makes ECE WORSE here: T is pushed up to the value that limits the damage of the few "
                    f"confident mistakes on whole unseen problems (the worst 5% of rows carry {top_share:.0%} of the NLL at T=1), "
                    "which makes the many correct predictions under-confident. Not changed (ml/model/ is M1's); see notes/E-a.md."),
        "informational_ece_optimal_T": {"oof": alt, "final_model_no_refit": alt_other,
                                        "note": "T chosen to minimise ECE on the same OOF rows, in-sample; not what the trainer fits or ships"},
        "plots": [plot.name],
        "caveat": ("Numbers without a 'masked' label are masked (structural masking applied after temperature, as served); "
                   "'unmasked' is what ml/model/calibrate.py reports. T is fitted on these same OOF logits, so 'after' is "
                   "in-sample for one parameter. ECE with 10 bins on ~5,800 correlated synthetic rows is optimistic for real code; "
                   "R-llm numbers (n=52 scored, LLM-written) are noisy and not a calibration claim."),
    }
    return card


def main():
    C.write_card(run(), "E07")


if __name__ == "__main__":
    main()
