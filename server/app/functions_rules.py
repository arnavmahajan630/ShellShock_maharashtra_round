"""Rule-based stand-in for the Functions & Modular Systems diagnoser (Module Deck).

Fills the ml/contracts/schemas.Diagnosis contract with deterministic pattern-matching
over the real interpreter's trace events and test output.

Problems handled:
- P20 clamp_range (targets M10 Silent Messenger / Print not Return, M06 Sensor Overwrite)
- P21 boost_shield (targets M09 Copy Module / Pass-by-Value, M10 Silent Messenger)
- P22 compose_pipeline (targets M05 Static Signal, M10 Silent Messenger)
- P23 integrate_subsystem (targets M06 Sensor Overwrite / Assign in Condition, M10, M05)
"""
import re

from ml.c_interp import harness
from server.app.diagnosis_common import FLOOR, card, posterior, to_internal_problem

_EVIDENCE_TEXT = {
    "M09": "Parameters in C are passed by value. Modifying `{param}` does not change the caller's variable — return the new value instead.",
    "M10": "`printf` shows a value on screen but the function returns without handing it back (line {line}).",
    "M05": "`{var}` is read on line {line} before it was given an initial value.",
    "M06": "`=` inside `if (...)` assigns a value; it does not compare (line {line}).",
    "M07": "A `;` right after `if (...)` gives it an empty body (line {line}).",
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


def _is_pass_by_value(code, run, trace):
    # Check if a helper function was called and caller returns its argument unchanged
    results = run["tests"].get("results") or []
    has_failing_unchanged = False
    for r in results:
        if not r["pass"]:
            args = r.get("args") or []
            got = r.get("got", {}).get("returned")
            if len(args) > 0 and got == args[0]:
                has_failing_unchanged = True
                break

    # Look for helper function that takes scalar and is void or mutates parameter
    has_helper = bool(re.search(r"void\s+[a-zA-Z_]\w*\s*\([^)]*\)", code)) or bool(
        re.search(r"\b[a-zA-Z_]\w*\s*\([^)]*\)\s*;", code)
    )

    if has_failing_unchanged and (has_helper or "void" in code):
        return True

    # Check effects in trace for helper call that returns void followed by caller returning input
    steps = trace.get("steps") or []
    saw_void_ret = False
    for s in steps:
        effects = s.get("effects") or []
        for eff in effects:
            if eff.startswith("ret:") and eff.endswith(":"):
                saw_void_ret = True
    if saw_void_ret and has_failing_unchanged:
        return True

    return False


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

    # 4. Pass by Value / Copy Module (M09) on functions
    if _is_pass_by_value(code, run, trace) or (problem_id == "P21" and ("void" in code or "add_power" in code)):
        # Extract mutated param name if possible
        m_param = re.search(r"void\s+\w+\s*\([^,)]*\b(\w+)\b", code)
        param_name = m_param.group(1) if m_param else "parameter"
        return _result(
            "confident",
            [card("M09", 0.90), card("M10", 0.05)],
            [{"type": "CODE", "text": _EVIDENCE_TEXT["M09"].format(param=param_name)}],
            twin_set="T5",
        )

    # 5. Missing Return with Output / Silent Messenger (M10)
    has_missing_return = any(e["type"] == "missing_return" for e in events)
    if (has_missing_return and trace.get("printed")) or ("printf" in code and "return" not in code):
        mr = next((e for e in events if e["type"] == "missing_return"), None)
        line = mr.get("line") if mr else 1
        return _result(
            "confident",
            [card("M10", 0.88), card("M09", 0.04)],
            [{"type": "RUN", "text": _EVIDENCE_TEXT["M10"].format(line=line), "line": line}],
            twin_set="T5",
        )

    # Static fallback check for assign in if: `if (mode = 1)`
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
    "P20": "int clamp_range(int val, int lo, int hi) {\n    if (val < lo) {\n        return lo;\n    }\n    if (val > hi) {\n        return hi;\n    }\n    return val;\n}",
    "P21": "int boost_shield(int shield, int boost) {\n    int total = shield + boost;\n    if (total > 100) {\n        return 100;\n    }\n    return total;\n}",
    "P22": "int step_sq(int n) {\n    return n * n;\n}\n\nint step_scale(int n) {\n    return n * 2;\n}\n\nint compose_pipeline(int x) {\n    return step_scale(step_sq(x)) + 1;\n}",
    "P23": "int integrate_subsystem(int mode, int a, int b) {\n    if (mode == 1) {\n        return a + b;\n    }\n    if (mode == 2) {\n        if (a > b) {\n            return a;\n        }\n        return b;\n    }\n    return -1;\n}",
}


def reference_fix(public_problem):
    code = _REFERENCE_FIXES.get(public_problem["problem_id"])
    if code is None:
        return None
    problem = to_internal_problem(public_problem)
    run = harness.run_tests(problem, code)
    return {"code": code, "verified": run["tests"]["passed"] == run["tests"]["total"]}
