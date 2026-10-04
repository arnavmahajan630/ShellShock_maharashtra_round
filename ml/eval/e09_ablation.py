"""E9 — feature ablation (03 §4.4, §9.2).

Rows: A, A+B, A+B+R, A+B+R+C (the shipped feature set), AST-only (no trace at all: static extraction, groups A and
C), the shipped model fed AST-only input, and one OFFLINE row that adds group F (fix-probe features, 34 numbers).
Every row uses the shipped hyper-parameters; macro-F1 is reported on E1 (grouped CV on TRAIN), on E2 (HOLDOUT-P) and on
E4 (R-llm, an LLM-written stand-in, 52 scored programs).

Ablated columns are set to NaN (LightGBM ignores an all-NaN column); the structural-mask preconditions are all
group-A features, so masking is identical in every row.

Group F costs 17 fixers per program. It is computed in a seeded random order under a time budget (--f-budget seconds,
default 1500); the with-F and without-F models are then compared on the same rows.

    .venv\\Scripts\\python -m ml.eval.e09_ablation [--f-budget 1500]
"""
from __future__ import annotations

import json
import math
import sys
import time

import numpy as np

from ml.contracts.classes import LABELS, MISCONCEPTIONS
from ml.contracts.feature_names import FEATURES, GROUPS, OFFLINE_FEATURES
from ml.eval import _ea_common as C
from ml.model import train as T
from ml.model.calibrate import softmax
from ml.model.data import TrainData
from ml.model.mask import allowed_classes, apply_mask

COLS = {g: [FEATURES.index(n) for n in GROUPS[g]] for g in ("A", "B", "R", "C")}


def _keep(X, groups):
    out = np.full_like(X, np.nan)
    for g in groups:
        out[:, COLS[g]] = X[:, COLS[g]]
    return out


def _predict(booster, X, features):
    probs = apply_mask(softmax(T.raw_logits(booster, X), 1.0), allowed_classes(X, features, LABELS))
    return probs.argmax(axis=1)


def _score_eval(booster, features, sets):
    out = {}
    for name, (X, b) in sets.items():
        pred = _predict(booster, X, features)
        ev = C.evaluate(b.Y, b.y, pred, b.groups if name == "E2" else None, None, cluster=(name == "E2"))
        out[name] = {"macro_f1": ev["macro_f1"], "ci95": ev["macro_f1_ci95"], "n": ev["n"], "acc": ev["acc"]}
    return out


def _row(label, Xtr, data, folds, params, features, sets):
    td = TrainData(X=Xtr, Y=data.Y, y=data.y, groups=data.groups, features=list(features), labels=data.labels,
                   domain=data.domain)
    t = time.time()
    cv = T.cross_validate(params, td, folds, refit=True)
    pred = np.zeros(len(td), dtype=np.int64)
    fold_f1 = []
    for _, va in folds:
        pred[va] = apply_mask(softmax(cv["oof_logits"][va], 1.0), allowed_classes(td.X[va], td.features, td.labels)).argmax(axis=1)
        fold_f1.append(C.macro_f1(td.Y[va], td.y[va], pred[va]))
    rounds = max(int(math.ceil(float(np.mean(cv["rounds"])))), 1)
    booster, _ = T.fit_model(params, td, np.arange(len(td)), rounds)
    res = {"row": label, "E1": {"macro_f1": float(np.mean(fold_f1)), "std": float(np.std(fold_f1)), "n": len(td)},
           "n_feature_columns_used": int((~np.isnan(Xtr).all(axis=0)).sum())}
    res.update(_score_eval(booster, features, sets))
    C.log(f"  {label:34s} E1 {res['E1']['macro_f1']:.3f}  E2 {res['E2']['macro_f1']:.3f}  E4 {res['E4']['macro_f1']:.3f}  ({time.time() - t:.0f} s)")
    return res


# ------------------------------------------------------------------ group F

def _f_vector(problem, code):
    from ml.features.fix_feats import fix_features
    d = fix_features(problem, code, MISCONCEPTIONS)
    return [float(d[n]) for n in OFFLINE_FEATURES]


def f_matrix(name, rows, budget_s, order_seed=0):
    """Cached group-F values; rows are processed in a seeded random order until the time budget is spent.
    Returns (F, done_mask)."""
    path = C.CACHE / f"F_{name}.npz"
    n = len(rows)
    F = np.full((n, len(OFFLINE_FEATURES)), np.nan, dtype=np.float32)
    done = np.zeros(n, dtype=bool)
    sig = json.dumps([r["id"] for r in rows])
    import hashlib
    sig = hashlib.sha256(sig.encode()).hexdigest()[:12]
    if path.exists():
        z = np.load(path)
        if z["sig"].item() == sig:
            F, done = z["F"], z["done"]
    order = np.random.default_rng(order_seed).permutation(n)
    started = time.time()
    pending = [i for i in order if not done[i]]
    for k, i in enumerate(pending):
        if budget_s is not None and time.time() - started > budget_s:
            break
        try:
            F[i] = _f_vector(C.problems()[rows[i]["problem_id"]], rows[i]["code"])
            done[i] = True
        except Exception:
            F[i] = np.nan
        if (k + 1) % 250 == 0:
            np.savez_compressed(path, F=F, done=done, sig=np.array(sig))
            C.log(f"  F {name}: {int(done.sum())}/{n}  {time.time() - started:.0f} s")
    np.savez_compressed(path, F=F, done=done, sig=np.array(sig))
    return F, done


def run(f_budget=1500):
    mb = C.model_bundle()
    data, model = mb["data"], mb["model"]
    hold, real = C.holdout_bundle(), C.realistic_bundle()
    rsub = real.sub(real.scored)
    params = mb["meta"]["chosen"]
    folds = mb["folds"]
    sets = {"E2": (hold.X, hold), "E4": (rsub.X, rsub)}
    sets_static = {"E2": (hold.Xs, hold), "E4": (rsub.Xs, rsub)}
    rows = []
    C.log("ablation rows (shipped hyper-parameters) ...")
    for label, groups in (("A", ["A"]), ("A+B", ["A", "B"]), ("A+B+R", ["A", "B", "R"]), ("A+B+R+C (shipped set)", ["A", "B", "R", "C"])):
        rows.append(_row(label, _keep(data.X, groups), data, folds, params, FEATURES,
                         {k: (_keep(X, groups), b) for k, (X, b) in sets.items()}))
    # AST-only: the training rows are extracted with no trace at all (static main loop), groups A and C
    rows.append(_row("AST-only (no trace; A+C, static)", _keep(data.Xs, ["A", "C"]), data, folds, params, FEATURES,
                     {k: (_keep(X, ["A", "C"]), b) for k, (X, b) in sets_static.items()}))
    # shipped model (trained with 15% dropout rows) fed an AST-only request
    shipped = {}
    for k, (X, b) in sets_static.items():
        ev = C.evaluate(b.Y, b.y, model.proba(X).argmax(axis=1), b.groups if k == "E2" else None, None, cluster=(k == "E2"))
        shipped[k] = {"macro_f1": ev["macro_f1"], "ci95": ev["macro_f1_ci95"], "n": ev["n"], "acc": ev["acc"]}
    rows.append({"row": "shipped model, AST-only input (B and R NaN)", "E1": None, **shipped})
    C.log(f"  shipped model on AST-only input: E2 {shipped['E2']['macro_f1']:.3f}  E4 {shipped['E4']['macro_f1']:.3f}")

    # group F (offline) on the same rows with and without F
    C.log(f"group F: {len(OFFLINE_FEATURES)} numbers per program, budget {f_budget} s for TRAIN ...")
    F_tr, done = f_matrix("train", data.rows, f_budget)
    F_ho, done_ho = f_matrix("hold", hold.rows, None)
    F_re, done_re = f_matrix("real", real.rows, None)
    S = np.nonzero(done)[0]
    f_rows = []
    f_info = {"train_rows_with_F": int(len(S)), "train_rows_total": int(len(data)), "budget_s": f_budget,
              "holdout_rows_with_F": int(done_ho.sum()), "r_rows_with_F": int(done_re.sum())}
    if len(S) >= 600 and len(set(data.groups[S])) >= 10:
        sub = data.subset(S)
        sub.rows = [data.rows[i] for i in S]
        folds_s = T.make_folds(sub.groups, sub.domain)
        rm = real.scored & done_re
        rsub_f = real.sub(np.nonzero(rm)[0])
        ho_m = np.nonzero(done_ho)[0]
        hold_f = hold.sub(ho_m)
        F_real = F_re[np.nonzero(rm)[0]]
        sets_f_without = {"E2": (hold_f.X, hold_f), "E4": (rsub_f.X, rsub_f)}
        sets_f_with = {"E2": (np.hstack([hold_f.X, F_ho[ho_m]]), hold_f), "E4": (np.hstack([rsub_f.X, F_real]), rsub_f)}
        C.log(f"  comparing with and without F on {len(S)} TRAIN rows")
        f_rows.append(_row("A+B+R+C  (same rows as the F row)", sub.X, sub, folds_s, params, FEATURES, sets_f_without))
        f_rows.append(_row("A+B+R+C + F (OFFLINE, not shipped)", np.hstack([sub.X, F_tr[S]]), sub, folds_s, params,
                           list(FEATURES) + list(OFFLINE_FEATURES), sets_f_with))
    else:
        f_info["note"] = "too few rows got F inside the budget; the F row was not run"

    allrows = rows + f_rows
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.6))
    width = 0.27
    for j, exp in enumerate(("E1", "E2", "E4")):
        vals = [(r.get(exp) or {}).get("macro_f1") for r in allrows]
        xs = np.arange(len(allrows)) + (j - 1) * width
        ax.bar(xs, [v if v is not None else 0 for v in vals], width,
               label={"E1": "E1 grouped CV", "E2": "E2 unseen problems", "E4": "E4 R-llm (stand-in)"}[exp])
    ax.set_xticks(np.arange(len(allrows)), [r["row"].replace(" (", "\n(").replace("  ", "\n") for r in allrows], fontsize=6, rotation=0)
    ax.set_ylabel("macro-F1")
    ax.legend(fontsize=7)
    ax.set_title("E9 feature ablation", fontsize=9)
    fig.tight_layout()
    plot = C.plot_path("E09_ablation.png")
    fig.savefig(plot, dpi=130)
    plt.close(fig)

    card = {
        "id": "E9", "title": "Feature ablation", "slice": "E1 folds on TRAIN; E2 = HOLDOUT-P; E4 = R-llm (LLM-written stand-in, 52 scored)",
        "n": int(len(data)), "model_version": model.model_version,
        "metrics": {r["row"]: {e: r.get(e) for e in ("E1", "E2", "E4")} for r in allrows},
        "rows": allrows, "group_F": f_info,
        "shipped_row": "A+B+R+C (shipped set); the shipped model is the row WITHOUT F",
        "data_status": {"R-blind": C.R_BLIND_STATUS, "R-llm": C.R_LLM_NAME},
        "plots": [plot.name],
        "caveat": ("The plan asks for E1 and R; E2 and E4 (R-llm stand-in) are both given. Rows are retrained with the shipped "
                   "hyper-parameters and no new grid, so the A+B+R+C row can differ slightly from E1/E2. The F row is OFFLINE: on "
                   "synthetic data the class-k fixer is the inverse of the operator that built the row, so f_fix_k is close to a "
                   "label (03 §4.4); a high E1 for it is the generator memorised, not a better diagnoser. F was computed for all 17 "
                   "classes, not only the top-3 that serving would use. The with/without-F comparison uses only the TRAIN rows that "
                   "got F inside the time budget. E4 has 52 programs; differences of a few points are inside the interval."),
    }
    return card


def main():
    budget = 1500
    if "--f-budget" in sys.argv:
        budget = int(sys.argv[sys.argv.index("--f-budget") + 1])
    C.write_card(run(budget), "E09")


if __name__ == "__main__":
    main()
