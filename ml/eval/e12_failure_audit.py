"""E12 — failure audit (03 §9.2; becomes a table in the Lab Report).

Every wrong prediction made with confidence >= 0.6 on E2 (HOLDOUT-P) and E4 (R-llm, LLM-written stand-in). Confidence
is the shipped probability (temperature + masking). Because the fitted temperature (2.45, see E7) flattens the
probabilities, wrong predictions whose un-flattened (T=1) confidence is >= 0.6 are listed as well and marked.

Each row: id, code, true, pred, conf, top-3 contributors for the predicted class (SHAP-style, LightGBM pred_contrib), top-3
for the true class, which defining features of the two classes fired, and a written "why". The "why" is written by
E-a after reading the code and the contributors (ml/eval/_ea_manual_why.json, keyed by row id); items without a manual
entry carry an automatic sentence marked `why_source: auto`.

    .venv\\Scripts\\python -m ml.eval.e12_failure_audit
"""
from __future__ import annotations

import json

import numpy as np

from ml.contracts.classes import CLASS_INFO, LABELS, TWIN_SETS
from ml.contracts.feature_names import CLASS_DEFINING_FEATURES, FEATURES
from ml.eval import _ea_common as C
from ml.model.calibrate import softmax
from ml.model.mask import allowed_classes, apply_mask

MANUAL = C.ROOT / "ml" / "eval" / "_ea_manual_why.json"
THRESHOLD = 0.6


def _top_contrib(model, x, cls, k=3):
    contrib = model.contributions(x)[LABELS.index(cls)][:-1]
    order = np.argsort(-contrib)[:k]
    return [{"feature": FEATURES[j], "value": None if np.isnan(x[j]) else float(x[j]), "contribution": float(contrib[j])}
            for j in order if contrib[j] > 0]


def _defining(x, cls):
    names = CLASS_DEFINING_FEATURES.get(cls, [])
    fired = [n for n in names if not np.isnan(x[FEATURES.index(n)]) and x[FEATURES.index(n)] != 0]
    return {"defining_features": list(names), "fired": fired}


def _auto_why(true, pred, tc, pc, df_true, df_pred):
    twin = [s for s, i in TWIN_SETS.items() if true in i["members"] and pred in i["members"]]
    parts = [f"Predicted {pred}; the label is {true}."]
    if twin:
        parts.append(f"{true} and {pred} share twin set {twin[0]}.")
    if pc:
        parts.append("Strongest evidence for the prediction: " + ", ".join(f"{c['feature']}={c['value']:g}" for c in pc[:2] if c["value"] is not None) + ".")
    parts.append(f"{true}'s defining features that fired: {df_true['fired'] or 'none'}.")
    return " ".join(parts)


def _collect(name, b, model):
    X = b.X
    P_cal = model.proba(X)
    P_raw = apply_mask(softmax(model.logits(X), 1.0), allowed_classes(X, model.features, model.labels))
    items = []
    for i in range(len(b)):
        if not b.scored[i]:
            continue
        pred = int(P_cal[i].argmax())
        if b.Y[i, pred] > 0:
            continue
        pred_raw = int(P_raw[i].argmax())
        conf_cal, conf_raw = float(P_cal[i].max()), float(P_raw[i].max())
        if conf_cal < THRESHOLD and conf_raw < THRESHOLD:
            continue
        true = LABELS[int(b.y[i])]
        pcls = LABELS[pred]
        pc, tc = _top_contrib(model, X[i], pcls), _top_contrib(model, X[i], true)
        dft, dfp = _defining(X[i], true), _defining(X[i], pcls)
        items.append({
            "set": name, "id": b.ids[i], "problem": str(b.groups[i]), "code": b.codes[i], "true": true, "pred": pcls,
            "conf": conf_cal, "conf_T1": conf_raw, "p_true": float(P_cal[i, int(b.y[i])]),
            "selected_by": "calibrated>=0.6" if conf_cal >= THRESHOLD else "T=1 confidence>=0.6 only",
            "top3_for_pred": pc, "top3_for_true": tc, "true_class_signature": dft, "pred_class_signature": dfp,
            "twin_set": next((s for s, t in TWIN_SETS.items() if true in t["members"] and pcls in t["members"]), None),
            "op_id": b.rows[i].get("op_id"), "why": _auto_why(true, pcls, pc, tc, dft, dfp), "why_source": "auto"})
    return items


def run():
    mb = C.model_bundle()
    model = mb["model"]
    hold, real = C.holdout_bundle(), C.realistic_bundle()
    raw_items = _collect("E2 HOLDOUT-P", hold, model) + _collect("E4 R-llm (LLM-written stand-in)", real, model)
    # augmented copies of one mutant share (problem, normalised AST hash, labels): show each cluster once
    hashes = {r["id"]: r.get("ast_hash") for b in (hold, real) for r in b.rows}
    clusters = {}
    for it in raw_items:
        key = (it["set"], it["problem"], it["true"], it["pred"], hashes.get(it["id"]) if it["set"].startswith("E2") else it["id"])
        if key in clusters:
            clusters[key]["cluster_size"] += 1
            clusters[key]["cluster_ids"].append(it["id"])
        else:
            clusters[key] = dict(it, cluster_size=1, cluster_ids=[it["id"]])
    items = list(clusters.values())
    manual = json.loads(MANUAL.read_text(encoding="utf-8")) if MANUAL.exists() else {}
    for it in items:
        if it["id"] in manual:
            it["why"], it["why_source"] = manual[it["id"]], "manual (E-a, after reading the code)"
    from collections import Counter
    pair = Counter((it["set"], it["true"], it["pred"]) for it in items)
    by_set = {}
    for name, b in (("E2 HOLDOUT-P", hold), ("E4 R-llm", real)):
        m = b.scored
        P = model.proba(b.X[m])
        wrong = b.Y[m][np.arange(m.sum()), P.argmax(axis=1)] == 0
        by_set[name] = {"n": int(m.sum()), "wrong": int(wrong.sum()),
                        "wrong_conf_ge_0.6_calibrated": int((wrong & (P.max(axis=1) >= THRESHOLD)).sum())}
    card = {
        "id": "E12", "title": "Failure audit", "slice": "wrong predictions with confidence >= 0.6 on E2 (HOLDOUT-P) and E4 (R-llm stand-in)",
        "n": len(items), "model_version": model.model_version,
        "metrics": {"audited": len(items), "by_set": by_set,
                    "selected_by_calibrated": sum(it["selected_by"] == "calibrated>=0.6" for it in items),
                    "manual_whys": sum(it["why_source"].startswith("manual") for it in items),
                    "most_common_confusions": [[list(k), v] for k, v in pair.most_common(10)]},
        "table": items,
        "data_status": {"R-blind": C.R_BLIND_STATUS, "R-llm": C.R_LLM_NAME},
        "caveat": ("Confidence is the shipped one (T=2.45 flattens it; E7). 'Manual' whys were written by E-a after reading the code "
                   "and the SHAP-style contributors; they are an explanation of the model's mistake, not a re-labelling of the data. "
                   "Contributors are LightGBM pred_contrib on the model's raw score (before temperature and masking)."),
    }
    return card


def main():
    card = run()
    C.write_card(card, "E12")
    for it in card["table"]:
        print(it["set"], it["id"], it["problem"], it["true"], "->", it["pred"], f"conf {it['conf']:.2f} T1 {it['conf_T1']:.2f}",
              f"x{it['cluster_size']}", it["why_source"][:6], it.get("op_id"))
        if "--code" in __import__("sys").argv:
            print(it["code"])
            print("  PRED:", [(c["feature"], c["value"], round(c["contribution"], 2)) for c in it["top3_for_pred"]])
            print("  TRUE:", [(c["feature"], c["value"], round(c["contribution"], 2)) for c in it["top3_for_true"]],
                  "fired", it["true_class_signature"]["fired"], "of", it["true_class_signature"]["defining_features"])


if __name__ == "__main__":
    main()
