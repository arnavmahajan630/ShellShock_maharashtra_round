"""E4 — realistic code (03 §9.2; the plan's headline).

R-blind (40 hand-written programs by FE) does NOT exist yet: reported as missing, nothing is run on it.
R-team (hand-written) is replaced by the 58 LLM-written programs of ml/data/realistic_llm.jsonl (source `R-llm`,
notes/T1.md). Every number here is on that LLM-written stand-in. It is not hand-written, not blind-authored, not
real student code, and the second rater is an LLM too.

    .venv\\Scripts\\python -m ml.eval.e04_realistic
"""
from __future__ import annotations

import json

import numpy as np

from ml.contracts.classes import LABELS, MISCONCEPTIONS
from ml.eval import _ea_common as C


def _gate_report(b):
    try:
        from server.app import gate
    except Exception as exc:                                    # pragma: no cover
        return {"status": f"gate not importable: {type(exc).__name__}"}
    out = {"junk_rows": 0, "junk_caught": 0, "normal_rows": 0, "normal_wrongly_gated": 0, "junk_detail": {},
           "normal_wrongly_gated_ids": []}
    for r in b.rows:
        res = gate.check(C.problems()[r["problem_id"]], r["code"])
        code = res.get("code")
        if r["label"] == "GATE":
            out["junk_rows"] += 1
            out["junk_caught"] += code != "G0"
            out["junk_detail"][r["id"]] = code
        else:
            out["normal_rows"] += 1
            if code != "G0":
                out["normal_wrongly_gated"] += 1
                out["normal_wrongly_gated_ids"].append([r["id"], code])
    return out


def _fixer_coverage(b, P, pred):
    """Share of misconception-labelled programs for which a verified fix exists (03 §4.4, §7.3)."""
    from ml.learner.fixer import probe
    wrong = [i for i in range(len(b)) if b.scored[i] and b.rows[i]["label"] in MISCONCEPTIONS]
    true_hit = top3_hit = 0
    detail = {}
    for i in wrong:
        problem, code = C.problems()[b.rows[i]["problem_id"]], b.codes[i]
        true_classes = [LABELS[j] for j in np.nonzero(b.Y[i])[0]]
        th = any(probe(problem, code, k)["hit"] for k in true_classes)
        top3 = [LABELS[j] for j in np.argsort(-P[i])[:3] if LABELS[j] in MISCONCEPTIONS]
        t3 = any(probe(problem, code, k)["hit"] for k in top3)
        true_hit += th
        top3_hit += t3
        detail[b.ids[i]] = {"true": true_classes, "fix_true_class": bool(th), "top3": top3, "fix_top3": bool(t3)}
    n = len(wrong)
    return {"n_misconception_programs": n, "fix_found_for_true_class": true_hit / n if n else None,
            "fix_found_in_top3": top3_hit / n if n else None, "detail": detail}


def _passing(b):
    from ml import runner
    out = {}
    for i, r in enumerate(b.rows):
        res = runner.run_tests(C.problems()[r["problem_id"]], r["code"])
        out[r["id"]] = res["tests"]["passed"] == res["tests"]["total"]
    return out


def run():
    mb = C.model_bundle()
    model = mb["model"]
    b = C.realistic_bundle()
    P = model.proba(b.X)
    pred_all = P.argmax(axis=1)
    m = b.scored
    sub = b.sub(m)
    pred = pred_all[m]
    ev = C.evaluate(sub.Y, sub.y, pred, groups=None, domain=sub.domain, cluster=False)
    plot = C.plot_path("E04_confusion.png")
    C.confusion_plot(sub.Y, sub.y, pred, plot, f"E4 R-llm (LLM-written stand-in) confusion, n={len(sub)}")

    meta = json.loads(C.REALISTIC_META.read_text(encoding="utf-8"))
    kinds = {}
    items = meta.get("kept") or {}
    if isinstance(items, dict):
        for k, v in items.items():
            kinds[k] = v.get("kind") if isinstance(v, dict) else None
    # second rater agreement (an LLM, from the code alone), see notes/T1.md
    pairs = [(r["label"], r["rater2_label"]) for r in b.rows if r.get("rater2_label") is not None]
    agree = sum(a == c for a, c in pairs) / len(pairs)
    kappa = C.cohen_kappa([a for a, _ in pairs], [c for _, c in pairs])
    pairs_model = [(r["label"], r["rater2_label"]) for r in sub.rows if r.get("rater2_label") is not None]

    cover = _fixer_coverage(b, P, pred_all)
    gate = _gate_report(b)
    passing = _passing(b)
    mis = [i for i in range(len(b)) if m[i] and b.rows[i]["label"] in MISCONCEPTIONS]
    luck = [b.ids[i] for i in mis if passing[b.ids[i]]]

    # a second view: the text-only sanity check — majority class and "predict CORRECT/OTHER only"
    majority = np.bincount(sub.y).argmax()
    maj_f1 = C.macro_f1(sub.Y, sub.y, np.full(len(sub), majority))

    wrong_rows = [{"id": sub.ids[i], "problem": str(sub.groups[i]), "true": LABELS[sub.y[i]], "pred": LABELS[pred[i]],
                   "conf": float(P[np.nonzero(m)[0][i]].max())}
                  for i in range(len(sub)) if sub.Y[i, pred[i]] == 0]

    card = {
        "id": "E4", "title": "Realistic code (LLM-written stand-in)",
        "slice": "R-llm: 58 LLM-written programs (source R-llm), 52 scored by the model + 6 junk rows for the gate",
        "n": int(len(sub)), "model_version": model.model_version,
        "data_status": {
            "R-blind": {"status": "missing", "note": C.R_BLIND_STATUS},
            "R-team": {"status": "replaced", "note": "replaced by R-llm: 58 programs written by deepseek-v4-pro role-playing a student; "
                       "run-checked against reference solutions; NOT hand-written, NOT real students."},
            "R-llm": {"status": "used", "label": C.R_LLM_NAME},
        },
        "metrics": {
            "macro_f1": ev["macro_f1"], "macro_f1_ci95": ev["macro_f1_ci95"], "ci_kind": "row bootstrap, 1,000 resamples",
            "acc": ev["acc"], "n_scored": int(len(sub)), "n_wrong": len(wrong_rows),
            "main_macro_f1": ev["by_domain"].get("main", {}).get("macro_f1"), "main_n": ev["by_domain"].get("main", {}).get("n"),
            "dsa_macro_f1": ev["by_domain"].get("dsa", {}).get("macro_f1"), "dsa_n": ev["by_domain"].get("dsa", {}).get("n"),
            "m_classes_macro_f1": ev["by_class_family"]["M"]["macro_f1"], "m_n": ev["by_class_family"]["M"]["n"],
            "d_classes_macro_f1": ev["by_class_family"]["D"]["macro_f1"], "d_n": ev["by_class_family"]["D"]["n"],
            "majority_class_macro_f1": maj_f1,
            "fixer_coverage_true_class": cover["fix_found_for_true_class"],
            "fixer_coverage_top3": cover["fix_found_in_top3"], "fixer_coverage_n": cover["n_misconception_programs"],
            "label_rater2_agreement": agree, "label_rater2_kappa": kappa, "label_rater2_n": len(pairs),
            "label_rater2_is": "deepseek-flash reading the code alone (an LLM, not a person)",
            "misconception_programs_that_pass_all_tests": len(luck),
        },
        "target": "macro-F1 >= 0.60 on R-blind (not measured); this card is the LLM stand-in only",
        "per_class": ev["per_class"], "by_domain": ev["by_domain"], "by_class_family": ev["by_class_family"],
        "gate": gate,
        "fixer_coverage": {k: v for k, v in cover.items() if k != "detail"},
        "fixer_coverage_detail": cover["detail"],
        "passes_all_tests_but_labelled_misconception": luck,
        "wrong_predictions": wrong_rows,
        "plots": [plot.name],
        "caveat": ("LLM-WRITTEN STAND-IN. 58 programs written by deepseek-v4-pro playing a student with a given belief; "
                   "not hand-written, not blind to the operators in the sense of 03 §3.4 (the author model saw only the belief text), "
                   "not real learners. n=52 scored over ~19 classes means 1-2 items per class: per-class numbers are anecdotes "
                   "and the interval is wide. The run check shows a program is wrong, not that it is wrong for the intended reason. "
                   "Labels were not adjudicated by a person. R-blind (the headline) is missing. Do not quote this as R-blind or as a human study."),
    }
    return card


def main():
    C.write_card(run(), "E04")


if __name__ == "__main__":
    main()
