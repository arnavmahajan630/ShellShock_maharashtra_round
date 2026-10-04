"""Rule-based stand-in for the Arrays-planet diagnoser (Lost Fleet).

Fills the ml/contracts/schemas.Diagnosis contract with deterministic pattern-matching
over the real interpreter's trace events and test output.

Problems handled:
- P08 last_beacon (targets M08, M10)
- P07 avg_fuel (targets M04, M08, M03, M05)
- P10 sum_first_k (targets M01, M08, M03)
- P09 max_shield (targets M08, M01, M05, M06)
"""
import re

from ml.c_interp import harness
from server.app.diagnosis_common import FLOOR, card, posterior, to_internal_problem

ACCUMULATOR_VARS = {"P07": "total", "P10": "total"}

_EVIDENCE_TEXT = {
    "M08": "Reads index {idx} — arrays are indexed 0..n-1. Index {idx} is beyond the valid range (line {line}).",
    "M04": "Integer division drops the decimal: got {got} instead of {expected}. Cast to (float) before dividing: `(float)total / n`.",
    "M01": "The loop runs one extra pass (line {line}). Check if `<=` should be `<`.",
    "M03": "`{var}` is reassigned (not `+=`) on every pass (line {line}) — each iteration overwrites the previous value.",
    "M05": "`{var}` is read on line {line} before it was given an initial value.",
    "M06": "`=` inside `if (...)` assigns a value; it does not compare (line {line}).",
    "M07": "A `;` right after the header gives it an empty body (line {line}).",
    "M10": "`printf` shows a value on screen but the function returns without handing it back (line {line}).",
}


def _loop_body(code):
    match = re.search(r"\b(?:for|while)\s*\([^)]*\)\s*\{", code)
    if not match:
        return ""
    start = match.end() - 1
    depth = 0
    for i in range(start, len(code)):
        if code[i] == "{":
            depth += 1
        elif code[i] == "}":
            depth -= 1
            if depth == 0:
                return code[start + 1 : i]
    return code[start + 1 :]


def _accumulator_reset_line(code, var):
    if not var:
        return None
    body = _loop_body(code)
    if not body:
        return None
    m = re.search(rf"\b{re.escape(var)}\b\s*=(?!=)", body)
    if not m:
        return None
    return code[: code.index(body) + m.start()].count("\n") + 1


def _is_integer_division(run):
    for r in run["tests"]["results"]:
        if r["pass"]:
            continue
        exp = r["expected"].get("returned")
        got = r["got"].get("returned")
        if isinstance(exp, float) and isinstance(got, float):
            if abs(got - int(exp)) < 1e-4 and abs(got - exp) > 1e-4:
                return exp, got
    return None


def _le_loop_line(code):
    m = re.search(r"for\s*\([^;]*;\s*[^;]*<=\s*([A-Za-z_]\w*|\d+)\s*;", code)
    if m:
        return code[: m.start()].count("\n") + 1
    return None


def _one_based_loop_line(code):
    m = re.search(r"for\s*\(\s*(?:int\s+)?([A-Za-z_]\w*)\s*=\s*1\s*;", code)
    if m:
        return code[: m.start()].count("\n") + 1
    return None


def _result(status, top, evidence, abstain=False, next_probe=None, twin_set=None):
    p_max = top[0]["p"] if top else 0.1
    return {
        "status": status,
        "posterior": posterior(
            primary=top[0]["id"] if top else None,
            p_primary=top[0]["p"] if top else 0.0,
            secondary=top[1]["id"] if len(top) > 1 else None,
            p_secondary=top[1]["p"] if len(top) > 1 else 0.0,
            correct=0.95 if status == "correct" else FLOOR,
        ),
        "top": top,
        "twin_set": twin_set,
        "two_bug": False,
        "novelty": {"knn_dist": 1.0, "tau_d": 3.2, "p_max": p_max, "tau_p": 0.55, "abstain": abstain},
        "evidence": evidence,
        "next_probe": next_probe,
        "probes_asked": [],
        "model_version": "rules-v1",
    }


def diagnose(public_problem, code):
    problem = to_internal_problem(public_problem)
    run = harness.run_tests(problem, code)
    trace = harness.trace(problem, code)
    events = trace.get("events") or []
    passed, total = run["tests"]["passed"], run["tests"]["total"]
    problem_id = public_problem["problem_id"]

    if total and passed == total:
        return _result("correct", [card("CORRECT", 0.95)], [])

    # Check interpreter trace events (syntax/semantic runtime events)
    for event in events:
        if event["type"] == "assign_in_cond":
            return _result(
                "confident",
                [card("M06", 0.88)],
                [{"type": "RUN", "text": _EVIDENCE_TEXT["M06"].format(line=event["line"]), "line": event["line"]}],
            )
        if event["type"] == "empty_body" and event.get("kind") in ("for", "while", "if"):
            return _result(
                "confident",
                [card("M07", 0.88)],
                [{"type": "RUN", "text": _EVIDENCE_TEXT["M07"].format(line=event["line"]), "line": event["line"]}],
            )
        if event["type"] == "uninit_read":
            return _result(
                "confident",
                [card("M05", 0.88)],
                [
                    {
                        "type": "RUN",
                        "text": _EVIDENCE_TEXT["M05"].format(var=event.get("var", "a variable"), line=event["line"]),
                        "line": event["line"],
                    }
                ],
            )

    # Check for loop bound <= in sum_first_k P10 (M01 Boundary Drift)
    if problem_id == "P10":
        le_line = _le_loop_line(code)
        if le_line:
            return _result(
                "confident",
                [card("M01", 0.90), card("M08", 0.06)],
                [{"type": "CODE", "text": _EVIDENCE_TEXT["M01"].format(line=le_line), "line": le_line}],
            )

    # Check for OOB read/write (Index Origin Fault M08 or Boundary Drift M01)
    boundary = next((e for e in events if e["type"] in ("oob_read", "oob_write")), None)
    if boundary:
        idx, size, line = boundary.get("idx"), boundary.get("size"), boundary.get("line", 1)
        if problem_id == "P08":
            return _result(
                "confident",
                [card("M08", 0.92)],
                [{"type": "RUN", "text": _EVIDENCE_TEXT["M08"].format(idx=idx, line=line), "line": line}],
            )
        return _result(
            "confident",
            [card("M08", 0.88), card("M01", 0.08)],
            [{"type": "RUN", "text": _EVIDENCE_TEXT["M08"].format(idx=idx, line=line), "line": line}],
        )

    # Check for M04 integer division (specifically on avg_fuel P07)
    if problem_id == "P07":
        div_mismatch = _is_integer_division(run)
        if div_mismatch or ("/ n" in code and "(float)" not in code and "float" not in code.split("return")[-1]):
            exp, got = div_mismatch if div_mismatch else (5.5, 5.0)
            return _result(
                "confident",
                [card("M04", 0.92)],
                [{"type": "RUN", "text": _EVIDENCE_TEXT["M04"].format(got=got, expected=exp)}],
            )

    # Check for accumulator reset M03
    acc_var = ACCUMULATOR_VARS.get(problem_id)
    if not acc_var:
        for possible_var in ("total", "sum", "ans"):
            if _accumulator_reset_line(code, possible_var):
                acc_var = possible_var
                break
    acc_line = _accumulator_reset_line(code, acc_var) if acc_var else None
    if acc_line:
        return _result(
            "confident",
            [card("M03", 0.86)],
            [{"type": "CODE", "text": _EVIDENCE_TEXT["M03"].format(var=acc_var, line=acc_line), "line": acc_line}],
        )

    # Check for 1-based indexing in loop header: for (i = 1; i <= n; i++) (excluding max search)
    if problem_id != "P09":
        one_based_line = _one_based_loop_line(code)
        if one_based_line:
            return _result(
                "confident",
                [card("M08", 0.85)],
                [
                    {
                        "type": "CODE",
                        "text": "Loop starts at index 1 instead of 0 — arrays are 0-indexed.",
                        "line": one_based_line,
                    }
                ],
            )

    # Missing return but printed M10
    if any(e["type"] == "missing_return" for e in events) and trace.get("printed"):
        line = next(e["line"] for e in events if e["type"] == "missing_return")
        return _result(
            "confident",
            [card("M10", 0.85)],
            [{"type": "RUN", "text": _EVIDENCE_TEXT["M10"].format(line=line), "line": line}],
        )

    return _result(
        "novel", [], [{"type": "RUN", "text": "This code runs but the bug doesn't match a known pattern yet."}], abstain=True
    )


# ---------------------------------------------------------------- reference fixes (per problem)

_REFERENCE_FIXES = {
    "P08": "int last_beacon(int ids[], int n) {\n    return ids[n - 1];\n}",
    "P07": "float avg_fuel(int tanks[], int n) {\n    int total = 0;\n    for (int i = 0; i < n; i++) {\n        total += tanks[i];\n    }\n    return (float)total / n;\n}",
    "P10": "int sum_first_k(int a[], int n, int k) {\n    int total = 0;\n    for (int i = 0; i < k; i++) {\n        total += a[i];\n    }\n    return total;\n}",
    "P09": "int max_shield(int s[], int n) {\n    int m = s[0];\n    for (int i = 1; i < n; i++) {\n        if (s[i] > m) {\n            m = s[i];\n        }\n    }\n    return m;\n}",
}


def reference_fix(public_problem):
    code = _REFERENCE_FIXES.get(public_problem["problem_id"])
    if code is None:
        return None
    problem = to_internal_problem(public_problem)
    run = harness.run_tests(problem, code)
    return {"code": code, "verified": run["tests"]["passed"] == run["tests"]["total"]}
