"""Rule-based stand-in for the Variables & State planet diagnoser (Planet Neo Core).

Fills the ml/contracts/schemas.Diagnosis contract with deterministic pattern-matching
over the real interpreter's trace events and test output.

Problems handled:
- P13 sync_ratio (targets M04 Fraction Shear / Integer Division, M10 Silent Messenger)
- P14 signal_diff (targets M10 Silent Messenger, M05 Static Signal, M06 Sensor Overwrite)
- P18 state_balance (targets M05 Static Signal / Uninitialized Variable, M06 Sensor Overwrite)
- P19 coordinate_nodes (targets M06 Sensor Overwrite / Assign in Condition, M05 Static Signal)
"""
import re

from ml.c_interp import harness
from server.app.diagnosis_common import FLOOR, card, posterior, to_internal_problem

_EVIDENCE_TEXT = {
    "M05": "`{var}` is read on line {line} before it was given an initial value.",
    "M04": "Integer division drops the decimal: got {got} instead of {expected}. Cast to (float) before dividing: `(float)synced / total * 100`.",
    "M06": "`=` inside `if (...)` assigns a value; it does not compare (line {line}).",
    "M07": "A `;` right after `if (...)` gives it an empty body (line {line}).",
    "M10": "`printf` shows a value on screen but the function returns without handing it back (line {line}).",
}


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


def _first_fail(run):
    for r in run["tests"]["results"]:
        if not r["pass"]:
            return r
    return None


def _is_int_division(code, run, events):
    # Direct trace event
    intdiv_event = next((e for e in events if e.get("type") == "intdiv"), None)
    if intdiv_event:
        return intdiv_event

    fail = _first_fail(run)
    if fail:
        exp = fail["expected"].get("returned")
        got = fail["got"].get("returned")
        if isinstance(exp, (int, float)) and isinstance(got, (int, float)):
            if abs(float(got) - float(exp)) > 1e-4 and "/ total" in code and "(float)" not in code:
                return {"type": "intdiv", "line": 2, "exp": exp, "got": got}
    return None


def diagnose(public_problem, code):
    problem = to_internal_problem(public_problem)
    run = harness.run_tests(problem, code)
    trace = harness.trace(problem, code)
    events = trace.get("events") or []
    passed, total = run["tests"]["passed"], run["tests"]["total"]
    problem_id = public_problem.get("problem_id")

    if total and passed == total:
        return _result("correct", [card("CORRECT", 0.95)], [])

    # 1. Interpreter trace events: Uninitialized Variable Read (M05)
    uninit = next((e for e in events if e["type"] == "uninit_read"), None)
    if uninit:
        var_name = uninit.get("var", "variable")
        line = uninit.get("line", 1)
        return _result(
            "confident",
            [card("M05", 0.88)],
            [{"type": "RUN", "text": _EVIDENCE_TEXT["M05"].format(var=var_name, line=line), "line": line}],
            twin_set="T4",
        )

    # 2. Interpreter trace events: Assignment in Condition (M06)
    assign_cond = next((e for e in events if e["type"] == "assign_in_cond"), None)
    if assign_cond:
        line = assign_cond.get("line", 1)
        return _result(
            "confident",
            [card("M06", 0.88), card("M07", 0.05)],
            [{"type": "RUN", "text": _EVIDENCE_TEXT["M06"].format(line=line), "line": line}],
            twin_set="T3",
        )

    # 3. Interpreter trace events: Empty Body / Stray Semicolon (M07)
    empty_body = next((e for e in events if e["type"] == "empty_body" and e.get("kind") in ("if", "for", "while")), None)
    if empty_body:
        line = empty_body.get("line", 1)
        return _result(
            "confident",
            [card("M07", 0.88), card("M06", 0.05)],
            [{"type": "RUN", "text": _EVIDENCE_TEXT["M07"].format(line=line), "line": line}],
            twin_set="T3",
        )

    # 4. Integer Division on P13 sync_ratio (M04 Fraction Shear)
    if problem_id == "P13":
        intdiv_info = _is_int_division(code, run, events)
        if intdiv_info:
            fail = _first_fail(run)
            exp = fail["expected"].get("returned") if fail else 80.0
            got = fail["got"].get("returned") if fail else 0.0
            line = intdiv_info.get("line", 2)
            return _result(
                "confident",
                [card("M04", 0.92)],
                [{"type": "RUN", "text": _EVIDENCE_TEXT["M04"].format(got=got, expected=exp), "line": line}],
            )

    # 5. Missing Return with Output / Silent Messenger (M10)
    has_missing_return = any(e["type"] == "missing_return" for e in events)
    if (has_missing_return and trace.get("printed")) or ("printf" in code and "return" not in code):
        mr = next((e for e in events if e["type"] == "missing_return"), None)
        line = mr.get("line") if mr else 1
        return _result(
            "confident",
            [card("M10", 0.85)],
            [{"type": "RUN", "text": _EVIDENCE_TEXT["M10"].format(line=line), "line": line}],
        )

    # Fallback static pattern checks
    # Static check for assign in if: `if (x = ...)`
    m_assign = re.search(r"if\s*\([^=!<>\n]*=[^=]", code)
    if m_assign:
        line = code[: m_assign.start()].count("\n") + 1
        return _result(
            "confident",
            [card("M06", 0.88), card("M07", 0.05)],
            [{"type": "CODE", "text": _EVIDENCE_TEXT["M06"].format(line=line), "line": line}],
            twin_set="T3",
        )

    return _result(
        "novel", [], [{"type": "RUN", "text": "This code runs but the bug doesn't match a known pattern yet."}], abstain=True
    )


# ---------------------------------------------------------------- reference fixes (per problem)

_REFERENCE_FIXES = {
    "P13": "float sync_ratio(int synced, int total) {\n    return (float)synced / total * 100;\n}",
    "P14": "int signal_diff(int a, int b) {\n    if (a > b) {\n        return a - b;\n    }\n    return b - a;\n}",
    "P18": "int state_balance(int initial, int delta) {\n    int balance = initial;\n    balance += delta;\n    return balance;\n}",
    "P19": "int coordinate_nodes(int primary, int replica) {\n    if (primary > 0 && replica > 0) {\n        return primary + replica;\n    }\n    if (primary > 0) {\n        return primary;\n    }\n    if (replica > 0) {\n        return replica;\n    }\n    return 0;\n}",
}


def reference_fix(public_problem):
    code = _REFERENCE_FIXES.get(public_problem["problem_id"])
    if code is None:
        return None
    problem = to_internal_problem(public_problem)
    run = harness.run_tests(problem, code)
    return {"code": code, "verified": run["tests"]["passed"] == run["tests"]["total"]}
