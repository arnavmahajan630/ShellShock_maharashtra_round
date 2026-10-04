"""Rule-based stand-in for the Conditions-planet diagnoser.

The trained model (`ml/model/`) isn't ready yet. This module fills the same contract
(`ml/contracts/schemas.Diagnosis`) with deterministic pattern-matching instead of a
classifier — it reads the *exact* trace events the real interpreter already emits
(`assign_in_cond`, `empty_body`, `uninit_read`, `missing_return` — see
`ml/contracts/subset.py` EVENT_FIELDS) plus one static code pattern (chained comparison),
so swapping this for a real model later needs no change to the response shape.

    diagnose(problem, code) -> Diagnosis dict
    reference_fix(problem)  -> the hand-authored correct solution for `problem`, verified

Shared helpers (posterior/card builders, trap items, intervention-modality table) live in
`server/app/diagnosis_common.py` — `loops_rules.py` uses the same ones.
"""
import re

from ml.c_interp import harness
from server.app.diagnosis_common import FLOOR, card, posterior, to_internal_problem

_CHAINED_CMP_RE = re.compile(r"[A-Za-z_]\w*|\d+")


def _looks_chained(code):
    """True if a relational comparison's result feeds straight into another relational op,
    e.g. `lo < x < hi` (parses in C as `(lo<x) < hi`, never what the author intended)."""
    bare = re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', "", code)
    return bool(re.search(r"[\w)\]]\s*[<>]=?\s*[\w(]+\s*[<>]=?\s*[\w(]", bare))


_EVIDENCE_TEXT = {
    "M06": "`=` inside `if (...)` assigns a value; it does not compare (line {line}).",
    "M07": "A `;` right after `if (...)` gives it an empty body (line {line}).",
    "M05": "`{var}` is read on line {line} before it is given a value.",
    "M10": "`printf` shows a value but the function still falls off the end without `return`.",
}


def diagnose(public_problem, code):
    """Returns a full `Diagnosis` dict (schemas.Diagnosis)."""
    problem = to_internal_problem(public_problem)
    run = harness.run_tests(problem, code)
    trace = harness.trace(problem, code)
    events = trace.get("events") or []
    passed, total = run["tests"]["passed"], run["tests"]["total"]

    def result(status, top, evidence, abstain=False):
        p_max = top[0]["p"] if top else 0.1
        return {
            "status": status,
            "posterior": posterior(
                primary=top[0]["id"] if top else None, p_primary=top[0]["p"] if top else 0.0,
                secondary=top[1]["id"] if len(top) > 1 else None, p_secondary=top[1]["p"] if len(top) > 1 else 0.0,
                correct=0.95 if status == "correct" else FLOOR,
            ),
            "top": top,
            "twin_set": "T3" if top and top[0]["id"] in ("M06", "M07") else None,
            "two_bug": False,
            "novelty": {"knn_dist": 1.0, "tau_d": 3.2, "p_max": p_max, "tau_p": 0.55, "abstain": abstain},
            "evidence": evidence,
            "next_probe": None,
            "probes_asked": [],
            "model_version": "rules-v1",
        }

    if total and passed == total:
        return result("correct", [card("CORRECT", 0.95)], [])

    if _looks_chained(code):
        return result(
            "novel", [],
            [{"type": "CODE", "text": "A chained comparison like `lo < x < hi` doesn't mean what it looks like in C — it compares the result of the first `<` (0 or 1) against `hi`."}],
            abstain=True,
        )

    for event in events:
        if event["type"] == "assign_in_cond":
            return result("confident", [card("M06", 0.88), card("M07", 0.05)],
                           [{"type": "RUN", "text": _EVIDENCE_TEXT["M06"].format(line=event["line"]), "line": event["line"]}])
        if event["type"] == "empty_body" and event.get("kind") == "if":
            return result("confident", [card("M07", 0.88), card("M06", 0.05)],
                           [{"type": "RUN", "text": _EVIDENCE_TEXT["M07"].format(line=event["line"]), "line": event["line"]}])
        if event["type"] == "uninit_read":
            return result("confident", [card("M05", 0.85)],
                           [{"type": "RUN", "text": _EVIDENCE_TEXT["M05"].format(var=event.get("var", "a variable"), line=event["line"]), "line": event["line"]}])

    if any(e["type"] == "missing_return" for e in events) and trace.get("printed"):
        line = next(e["line"] for e in events if e["type"] == "missing_return")
        return result("confident", [card("M10", 0.82)],
                       [{"type": "RUN", "text": _EVIDENCE_TEXT["M10"].format(line=line), "line": line}])

    return result("novel", [], [{"type": "RUN", "text": "This code runs but the bug doesn't match a known pattern yet."}], abstain=True)


# ---------------------------------------------------------------- reference fixes (per problem)

_REFERENCE_FIXES = {
    "P11": "int door_open(int code) {\n    if (code == 42) {\n        return 1;\n    }\n    return 0;\n}",
    "P12": "int shield_mode(int energy) {\n    if (energy < 30) {\n        return 0;\n    }\n    if (energy < 70) {\n        return 1;\n    }\n    return 2;\n}",
    "P16": "int in_range(int x, int lo, int hi) {\n    if (x > lo && x < hi) {\n        return 1;\n    }\n    return 0;\n}",
    "P17": "int max_of_three(int a, int b, int c) {\n    int m = a;\n    if (b > m) {\n        m = b;\n    }\n    if (c > m) {\n        m = c;\n    }\n    return m;\n}",
}


def reference_fix(public_problem):
    """The hand-authored correct solution for this problem, verified against its own tests."""
    code = _REFERENCE_FIXES.get(public_problem["problem_id"])
    if code is None:
        return None
    problem = to_internal_problem(public_problem)
    run = harness.run_tests(problem, code)
    return {"code": code, "verified": run["tests"]["passed"] == run["tests"]["total"]}
