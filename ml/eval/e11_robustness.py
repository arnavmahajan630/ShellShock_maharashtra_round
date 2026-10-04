"""E11 — robustness to semantics-preserving rewrites (03 §9.2, target >= 0.90 unchanged).

Each program of R-llm (52 scored, an LLM-written stand-in) and of HOLDOUT-P (938 rows) is rewritten five ways and
once with all five together: rename variables, reformat (pycparser reprint), comments, a dead variable, for<->while.
A rewrite counts only if the problem's tests give the same pass/fail vector before and after. Reported: the share of
predictions (top-1) that stay the same, and what happens to accuracy. Also the false-alarm rate on near-miss CORRECT
programs (correct code that looks like a bug).

    .venv\\Scripts\\python -m ml.eval.e11_robustness
"""
from __future__ import annotations

import numpy as np

from ml import runner
from ml.contracts.classes import LABELS, MISCONCEPTIONS
from ml.eval import _ea_common as C
from ml.eval import _ea_perturb as PT

CORRECT = LABELS.index("CORRECT")


def _vector(problem, code):
    try:
        res = runner.run_tests(problem, code)
        return tuple(bool(t["pass"]) for t in res["tests"]["results"])
    except Exception:
        return None


def _perturb_bundle(b, model, kinds, extractor):
    P0 = model.proba(b.X)
    pred0 = P0.argmax(axis=1)
    orig_vec = {}
    results = {k: {"applicable": 0, "preserved": 0, "unchanged": 0, "feat_identical": 0, "rows": [], "flips": [],
                   "acc_before": [], "acc_after": []} for k in kinds}
    for i in range(len(b)):
        if not b.scored[i]:
            continue
        problem = C.problems()[b.rows[i]["problem_id"]]
        orig_vec[i] = _vector(problem, b.codes[i])
        for name, fn in kinds.items():
            new = PT.safe(fn, b.codes[i])
            if new is None:
                continue
            results[name]["applicable"] += 1
            if orig_vec[i] is None or _vector(problem, new) != orig_vec[i]:
                continue
            results[name]["preserved"] += 1
            x = extractor.row(problem["problem_id"], new)
            p = model.proba(x)[0]
            results[name]["feat_identical"] += bool(np.array_equal(x, b.X[i], equal_nan=True))
            same = int(p.argmax()) == int(pred0[i])
            results[name]["unchanged"] += same
            results[name]["rows"].append((i, same))
            results[name]["acc_before"].append(float(b.Y[i, pred0[i]] > 0))
            results[name]["acc_after"].append(float(b.Y[i, int(p.argmax())] > 0))
            if not same and len(results[name]["flips"]) < 12:
                results[name]["flips"].append({"id": b.ids[i], "from": LABELS[pred0[i]], "to": LABELS[int(p.argmax())],
                                               "true": LABELS[b.y[i]]})
    return results, pred0


def _summarise(res, groups):
    out = {}
    for name, r in res.items():
        n = r["preserved"]
        rows = np.array([i for i, _ in r["rows"]], dtype=int)
        same = np.array([s for _, s in r["rows"]], dtype=float)
        if n:
            g = groups[rows]
            ci = C.bootstrap_ci(lambda idx, same=same: float(same[idx].mean()), n, g)
        else:
            ci = [None, None]
        out[name] = {"applicable": r["applicable"], "behaviour_preserved": n,
                     "not_preserved_dropped": r["applicable"] - n,
                     "feature_vector_identical_share": (r["feat_identical"] / n) if n else None,
                     "unchanged_share": (r["unchanged"] / n) if n else None, "unchanged_ci95": ci,
                     "acc_before": float(np.mean(r["acc_before"])) if n else None,
                     "acc_after": float(np.mean(r["acc_after"])) if n else None, "example_flips": r["flips"]}
    plan = [k for k in res if k in PT.KINDS]
    pooled_n = sum(res[k]["preserved"] for k in plan)
    pooled_same = sum(res[k]["unchanged"] for k in plan)
    pooled_feat = sum(res[k]["feat_identical"] for k in plan)
    out["_pooled_single_rewrites"] = {"n": pooled_n, "unchanged_share": (pooled_same / pooled_n) if pooled_n else None,
                                      "feature_vector_identical_share": (pooled_feat / pooled_n) if pooled_n else None}
    stress = [k for k in res if k in PT.STRESS]
    sn = sum(res[k]["preserved"] for k in stress)
    out["_pooled_stress_rewrites"] = {"n": sn, "unchanged_share": (sum(res[k]["unchanged"] for k in stress) / sn) if sn else None,
                                      "feature_vector_identical_share": (sum(res[k]["feat_identical"] for k in stress) / sn) if sn else None}
    return out


def _false_alarm(P, y_is_correct_mask, label):
    pred = P.argmax(axis=1)
    if not len(pred):
        return {"n": 0}
    mis_conf = np.array([P[i, [LABELS.index(k) for k in MISCONCEPTIONS]].max() for i in range(len(pred))])
    return {"n": int(len(pred)), "top1_not_correct": float((pred != CORRECT).mean()),
            "top1_misconception_p_ge_0.5": float(((pred != CORRECT) & (mis_conf >= 0.5)).mean()), "slice": label}


def run():
    mb = C.model_bundle()
    model, data = mb["model"], mb["data"]
    hold, real = C.holdout_bundle(), C.realistic_bundle()
    ex = C.extractor()
    kinds = dict(PT.KINDS, combined=PT.combined, **PT.STRESS)
    C.log("rewriting R-llm ...")
    res_r, pred_r = _perturb_bundle(real, model, kinds, ex)
    C.log("rewriting HOLDOUT-P (938 rows x 6 rewrites) ...")
    res_h, pred_h = _perturb_bundle(hold, model, kinds, ex)
    sum_r, sum_h = _summarise(res_r, real.groups), _summarise(res_h, hold.groups)

    # false alarms on near-miss CORRECT code
    P_hold = model.proba(hold.X)
    nm_h = np.array([r["label"] == "CORRECT" and str(r.get("op_id") or "").startswith("nm_") for r in hold.rows])
    plain_h = np.array([r["label"] == "CORRECT" and not str(r.get("op_id") or "").startswith("nm_") for r in hold.rows])
    oof = C.oof_proba()
    nm_t = np.array([r["label"] == "CORRECT" and str(r.get("op_id") or "").startswith("nm_") for r in data.rows])
    plain_t = np.array([r["label"] == "CORRECT" and not str(r.get("op_id") or "").startswith("nm_") for r in data.rows])
    P_real = model.proba(real.X)
    real_correct = np.array([r["label"] == "CORRECT" for r in real.rows])
    fa = {
        "near_miss_CORRECT_HOLDOUT-P": _false_alarm(P_hold[nm_h], None, "HOLDOUT-P near-miss CORRECT (unseen problems)"),
        "plain_CORRECT_HOLDOUT-P": _false_alarm(P_hold[plain_h], None, "HOLDOUT-P other CORRECT"),
        "near_miss_CORRECT_TRAIN_oof": _false_alarm(oof[nm_t], None, "TRAIN near-miss CORRECT, out-of-fold"),
        "plain_CORRECT_TRAIN_oof": _false_alarm(oof[plain_t], None, "TRAIN other CORRECT, out-of-fold"),
        "correct_but_unusual_R-llm": _false_alarm(P_real[real_correct], None, "R-llm correct-but-unusual (LLM-written stand-in)"),
    }
    ci_nm = C.bootstrap_ci(lambda i: float((P_hold[nm_h][i].argmax(axis=1) != CORRECT).mean()), int(nm_h.sum()), hold.groups[nm_h])
    fa["near_miss_CORRECT_HOLDOUT-P"]["top1_not_correct_ci95_cluster"] = ci_nm

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(kinds)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for j, (lab, s) in enumerate((("R-llm (stand-in)", sum_r), ("HOLDOUT-P", sum_h))):
        ax.bar(np.arange(len(names)) + (j - 0.5) * 0.38, [(s[k]["unchanged_share"] or 0) for k in names], 0.38, label=lab)
    ax.axhline(0.9, color="k", ls="--", lw=1)
    ax.set_xticks(range(len(names)), names, fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("top-1 unchanged")
    ax.legend(fontsize=7)
    ax.set_title("E11 prediction invariance (dashed = 0.90 target)", fontsize=9)
    fig.tight_layout()
    plot = C.plot_path("E11_invariance.png")
    fig.savefig(plot, dpi=130)
    plt.close(fig)

    card = {
        "id": "E11", "title": "Robustness to semantics-preserving rewrites",
        "slice": "R-llm (52 scored, LLM-written stand-in) and HOLDOUT-P (938)", "n": int(real.scored.sum() + len(hold)),
        "model_version": model.model_version,
        "metrics": {
            "invariance_R-llm_pooled": sum_r["_pooled_single_rewrites"]["unchanged_share"],
            "invariance_HOLDOUT-P_pooled": sum_h["_pooled_single_rewrites"]["unchanged_share"],
            "feature_vector_identical_R-llm_pooled": sum_r["_pooled_single_rewrites"]["feature_vector_identical_share"],
            "feature_vector_identical_HOLDOUT-P_pooled": sum_h["_pooled_single_rewrites"]["feature_vector_identical_share"],
            "stress_invariance_R-llm": sum_r["_pooled_stress_rewrites"]["unchanged_share"],
            "stress_invariance_HOLDOUT-P": sum_h["_pooled_stress_rewrites"]["unchanged_share"],
            "invariance_R-llm_combined": sum_r["combined"]["unchanged_share"],
            "invariance_HOLDOUT-P_combined": sum_h["combined"]["unchanged_share"],
            "near_miss_false_alarm_HOLDOUT-P": fa["near_miss_CORRECT_HOLDOUT-P"].get("top1_not_correct"),
            "near_miss_false_alarm_TRAIN_oof": fa["near_miss_CORRECT_TRAIN_oof"].get("top1_not_correct"),
        },
        "target": "invariance >= 0.90 unchanged",
        "by_rewrite": {"R-llm": sum_r, "HOLDOUT-P": sum_h},
        "false_alarm": fa,
        "data_status": {"R-blind": C.R_BLIND_STATUS, "R-llm": C.R_LLM_NAME},
        "plots": [plot.name],
        "finding": ("Invariance is high mostly because the feature extractors already ignore these edits: the plan's five rewrites leave "
                    f"the feature vector bit-identical in {sum_h['_pooled_single_rewrites']['feature_vector_identical_share']:.0%} of HOLDOUT-P cases "
                    f"and {sum_r['_pooled_single_rewrites']['feature_vector_identical_share']:.1%} of R-llm cases, and the two stress rewrites "
                    f"(flip_compare `i<n` -> `n>i`, incr_form `i++` -> `i += 1`, not in the plan) in {sum_h['_pooled_stress_rewrites']['feature_vector_identical_share']:.0%} "
                    f"and {sum_r['_pooled_stress_rewrites']['feature_vector_identical_share']:.0%}. HOLDOUT-P programs come from a generator that reprints "
                    "with pycparser, so they are already in canonical form and most rewrites change nothing structural there; R-llm "
                    "(messier, LLM-written) is the more informative slice. Where features did change on R-llm, the top-1 did not."),
        "caveat": ("Rewrites are applied to the program text; behaviour preservation is checked by running the problem's tests "
                   "before and after (only same pass/fail vectors are kept; dropped counts are listed). 'Unchanged' compares "
                   "the model's top-1 before and after, whether or not it was right. HOLDOUT-P rows already contain generator "
                   "style augmentations (renames, comments, ...) so they are less surprising than fresh rewrites, and its intervals "
                   "are a cluster bootstrap over 6 problems. R-llm is LLM-written, 52 programs. for<->while changes the loop "
                   "structure the AST features read, so a drop there is a real weakness of the features, not a bug in the rewrite. "
                   "Near-miss CORRECT = generator rows with op_id nm_* (correct code that looks like a bug); 'false alarm' = top-1 "
                   "is not CORRECT."),
    }
    return card


def main():
    C.write_card(run(), "E11")


if __name__ == "__main__":
    main()
