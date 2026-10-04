"""E13 — real students (Strong): the ITSP slice made by package X1 (03 §3.4 protocol X, §9.2).

What can be run: AST-only. ITSP programs are whole C programs (main, scanf, malloc, char**): the interpreter cannot run
them, so there are no trace features and no test run; "AST-only vs full" has only the AST-only side. Because ITSP has
no problem bank entry, a pseudo-problem is built per program: the function that contains the first differing line
between the buggy and the corrected file is the entry, and the corrected file is the reference.

What the labels are: the 396 rows of ml/data/external/itsp_slice.jsonl carry rule-based candidate labels (19 non-OTHER,
377 OTHER = "no rule matched"). The human hand-check (06 §5, 40-80 items) has NOT been done, so every number below is
against unverified candidate labels. The ITSP code itself is not written to the card (no redistribution).

    .venv\\Scripts\\python -m ml.eval.e13_real_students
"""
from __future__ import annotations

import difflib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from pycparser import c_ast, c_generator, c_parser

from ml.c_interp.preprocess import preprocess
from ml.contracts.classes import LABELS
from ml.eval import _ea_common as C
from ml.model.data import soft_label_vector

SLICE = C.ROOT / "ml" / "data" / "external" / "itsp_slice.jsonl"
ITSP_DIR = C.ROOT / "ml" / "data" / "external" / "ITSP" / "dataset"
GEN = c_generator.CGenerator()


def _funcs(code):
    ast = c_parser.CParser().parse(preprocess(code))
    fns = [e for e in ast.ext if isinstance(e, c_ast.FuncDef)]
    return fns


def _pseudo_problem(row):
    """(problem dict, note) for one slice row, or (None, reason)."""
    lab, prob, pid = row["author"].split(":")[1].split("/")
    base = ITSP_DIR / lab / prob / pid
    buggy_path, fixed_path = Path(f"{base}_buggy.c"), Path(f"{base}_correct.c")
    if not fixed_path.exists():
        return None, "no corrected file on disk"
    buggy = row["code"]
    fixed = fixed_path.read_text(encoding="utf-8", errors="replace")
    b_lines, f_lines = buggy.splitlines(), fixed.splitlines()
    first = None
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, [l.strip() for l in b_lines], [l.strip() for l in f_lines]).get_opcodes():
        if tag != "equal":
            first = i1 + 1
            break
    try:
        fns = _funcs(buggy)
    except Exception as exc:
        return None, f"parse: {type(exc).__name__}"
    if not fns:
        return None, "no function"
    starts = [fn.coord.line for fn in fns]
    pick = fns[-1]
    # the first function whose start is not after the first changed line (code lines may shift by removed #includes)
    for fn, start in zip(fns, starts):
        if first is not None and start <= first + 0:
            pick = fn
    sig = GEN.visit(pick.decl).strip().rstrip(";")
    return {"problem_id": f"ITSP-{prob}", "signature": sig, "correct_variants": [fixed], "tests": [], "display_test": 0}, sig


def run():
    mb = C.model_bundle()
    model = mb["model"]
    rows = C.read_jsonl(SLICE) if SLICE.exists() else []
    if not rows:
        return {"id": "E13", "title": "Real students (ITSP slice)", "status": "not run",
                "reason": f"{SLICE} does not exist; run `python -m ml.external.itsp` (package X1) first", "n": 0}
    from ml.features.extract import extract
    X = np.full((len(rows), len(C.FEATURES)), np.nan, dtype=np.float32)
    notes = Counter()
    ok = np.zeros(len(rows), dtype=bool)
    sigs = {}
    for i, r in enumerate(rows):
        prob, note = _pseudo_problem(r)
        if prob is None:
            notes[note] += 1
            continue
        try:
            X[i] = extract(prob, r["code"])[0]
            ok[i] = not np.isnan(X[i][:10]).all()
            sigs[r["id"]] = note
        except Exception as exc:
            notes[f"extract: {type(exc).__name__}"] += 1
    P = model.proba(X)
    pred = P.argmax(axis=1)
    Y = np.stack([soft_label_vector(r) for r in rows])
    y = np.array([LABELS.index(r["label"]) for r in rows])
    groups = np.array([r["problem_id"] for r in rows])
    sel = ok
    n_ok = int(sel.sum())
    ev_all = C.evaluate(Y[sel], y[sel], pred[sel], groups[sel], None, cluster=True)
    rule = sel & (y != LABELS.index("OTHER"))
    ev_rule = C.evaluate(Y[rule], y[rule], pred[rule], None, None, cluster=False) if rule.sum() else None
    dist = Counter(LABELS[p] for p in pred[sel])
    on_other = Counter(LABELS[p] for p, yy in zip(pred[sel], y[sel]) if LABELS[yy] == "OTHER")
    per_label = {}
    for k in sorted({LABELS[v] for v in y[rule]}):
        m = rule & (y == LABELS.index(k))
        per_label[k] = {"n": int(m.sum()), "recall_top1_hits_label": float((Y[m][np.arange(m.sum()), pred[m]] > 0).mean()),
                        "predicted_as": dict(Counter(LABELS[p] for p in pred[m]))}
    card = {
        "id": "E13", "title": "Real students (ITSP slice) — AST-only, candidate labels",
        "slice": "X: ITSP buggy/fixed pairs, one hunk <= 3 lines (X1)", "n": n_ok, "model_version": model.model_version,
        "status": "partially run: AST-only; labels are UNVERIFIED rule candidates (human hand-check not done)",
        "metrics": {
            "rows_in_slice": len(rows), "rows_with_ast_features": n_ok, "rows_skipped": int(len(rows) - n_ok),
            "skipped_reasons": dict(notes),
            "macro_f1_vs_candidate_labels_all": ev_all["macro_f1"], "macro_f1_ci95_cluster_by_problem": ev_all["macro_f1_ci95"],
            "acc_vs_candidate_labels_all": ev_all["acc"],
            "n_rule_labelled_non_other": int(rule.sum()),
            "macro_f1_on_rule_labelled_only": ev_rule["macro_f1"] if ev_rule else None,
            "macro_f1_on_rule_labelled_ci95_row_bootstrap": ev_rule["macro_f1_ci95"] if ev_rule else None,
            "acc_on_rule_labelled_only": ev_rule["acc"] if ev_rule else None,
            "full_features_macro_f1": None,
            "full_features_status": "not available: ITSP programs use main/scanf/malloc/char** (outside the frozen subset), so the interpreter cannot trace them",
        },
        "per_candidate_label": per_label,
        "prediction_distribution_all": dict(dist.most_common()),
        "prediction_distribution_on_auto_OTHER": dict(on_other.most_common()),
        "finding": (f"{dist.get('M10', 0) / max(n_ok, 1):.0%} of the programs are predicted M10 (printed instead of returned) and "
                    f"{sum('printf' in r['code'] for r in rows) / len(rows):.0%} of them call printf: whole-program I/O (prompts, results) looks like "
                    "M10's signature a_printf_in_nonvoid, which is a distribution shift from the bank's single-function problems. "
                    "Agreement with the candidate labels is low (see metrics); the cause cannot be separated from label noise until the hand-check is done."),
        "status_x1": "ml/data/external/itsp_slice.jsonl (gitignored) holds code from ITSP; code is not copied into this card",
        "caveat": ("The ITSP labels are the output of regex rules on the diff (notes/X1.md): 377 of 396 are 'OTHER' = no rule matched, which "
                   "includes real classes the rules do not reach, so agreement with them is not accuracy. Only 19 rows carry a class, "
                   "unchecked by a person; per-label counts are 1-7. The entry function is the function containing the first changed "
                   "line (a pseudo-problem, since ITSP has no problem-bank entry); without tests there are no trace or relation "
                   "features, and the shipped model, trained with feature dropout, is run on AST (A) + task (C) features only. "
                   "Real student programs here are much larger than the bank's single-function problems (loops over several functions, "
                   "pointers, I/O). Treat this card as a feasibility check, not a real-student result."),
    }
    return card


def main():
    C.write_card(run(), "E13")


if __name__ == "__main__":
    main()
