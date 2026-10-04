"""Rule-based stand-in for the Loops-planet diagnoser.

Same approach as `conditions_rules.py` (trained model not ready; deterministic pattern
matching over the real interpreter's trace, same `Diagnosis` contract). Two things Loops
needs that Conditions didn't:

- `step_cap_hit` (ml/contracts/subset.py) -> M02 STALLED_THRUSTER (loop never progresses).
- M03 ACCUMULATOR_RESET has no dedicated interpreter event, so it's detected by checking
  whether the submitted code's loop body plainly reassigns (not `+=`) the problem's known
  accumulator variable (`ACCUMULATOR_VARS`) — we already know that name since we hand-author
  a reference solution per problem anyway.

P03 `total_energy` is the flagship "hard twin": `for(i=0;i<=n;i++) sum+=a[i];` is genuinely
code-identical whether the learner believes the loop bound is wrong (M01) or arrays start at
1 (M08) — `diagnose` returns real `status="ambiguous"` for it (reusing the exact copy already
scripted in `server/fixtures/attempt.json`), and `resolve_probe` answers the follow-up.
"""
import re

from ml.c_interp import harness
from server.app.diagnosis_common import FLOOR, card, posterior, to_internal_problem

ACCUMULATOR_VARS = {"P03": "total", "P06": "result"}

_EVIDENCE_TEXT = {
    "M02": "The loop never changes the value it tests — it runs until the step cap, not until the work is done.",
    "M03": "`{var}` is reassigned (not `+=`) on every pass (line {line}) — each iteration overwrites the last.",
    "M06": "`=` inside `if (...)` assigns a value; it does not compare (line {line}).",
    "M07": "A `;` right after the loop/if header gives it an empty body (line {line}).",
    "M05": "`{var}` is read on line {line} before it is given a value.",
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


def _off_by_one(run):
    diffs = []
    for r in run["tests"]["results"]:
        if r["pass"]:
            continue
        for key, expected in (r["expected"] or {}).items():
            got = r["got"].get(key)
            if isinstance(expected, (int, float)) and isinstance(got, (int, float)) and not isinstance(expected, bool):
                diffs.append(abs(got - expected))
    return bool(diffs) and all(d == 1 for d in diffs)


def _result(status, top, evidence, abstain=False, next_probe=None, twin_set=None):
    p_max = top[0]["p"] if top else 0.1
    return {
        "status": status,
        "posterior": posterior(
            primary=top[0]["id"] if top else None, p_primary=top[0]["p"] if top else 0.0,
            secondary=top[1]["id"] if len(top) > 1 else None, p_secondary=top[1]["p"] if len(top) > 1 else 0.0,
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


_T1_PROBE = {
    "probe_id": "P_T1_a",
    "prompt": "Index of the last valid cell?",
    "code": "int a[5];",
    "options": ["4", "5", "depends on values"],
    "eig_bits": 0.71,
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

    if any(e["type"] == "step_cap_hit" for e in events):
        return _result("confident", [card("M02", 0.86)], [{"type": "RUN", "text": _EVIDENCE_TEXT["M02"]}])

    boundary = next((e for e in events if e["type"] in ("oob_read", "oob_write") and e.get("idx") == e.get("size")), None)
    if boundary:
        if problem_id == "P03":
            return _result(
                "ambiguous",
                [card("M01", 0.46), card("M08", 0.44)],
                [
                    {"type": "CODE", "text": "Loop condition uses `<=` (line 3).", "line": 3},
                    {"type": "RUN", "text": f"Reads `cells[{boundary['idx']}]`, one cell past the end.", "line": boundary["line"]},
                    {"type": "RUN", "text": "This code is identical for both explanations. Asking one question."},
                ],
                next_probe=_T1_PROBE,
                twin_set="T1",
            )
        return _result("confident", [card("M01", 0.86)],
                        [{"type": "RUN", "text": f"Reads one cell past the end (index {boundary['idx']}, line {boundary['line']}).", "line": boundary["line"]}])

    accum_line = _accumulator_reset_line(code, ACCUMULATOR_VARS.get(problem_id))
    if accum_line:
        return _result("confident", [card("M03", 0.84)],
                        [{"type": "CODE", "text": _EVIDENCE_TEXT["M03"].format(var=ACCUMULATOR_VARS[problem_id], line=accum_line), "line": accum_line}])

    for event in events:
        if event["type"] == "assign_in_cond":
            return _result("confident", [card("M06", 0.85)],
                            [{"type": "RUN", "text": _EVIDENCE_TEXT["M06"].format(line=event["line"]), "line": event["line"]}])
        if event["type"] == "empty_body" and event.get("kind") in ("for", "while"):
            return _result("confident", [card("M07", 0.85)],
                            [{"type": "RUN", "text": _EVIDENCE_TEXT["M07"].format(line=event["line"]), "line": event["line"]}])
        if event["type"] == "uninit_read":
            return _result("confident", [card("M05", 0.85)],
                            [{"type": "RUN", "text": _EVIDENCE_TEXT["M05"].format(var=event.get("var", "a variable"), line=event["line"]), "line": event["line"]}])

    if _off_by_one(run):
        return _result("confident", [card("M01", 0.75)], [{"type": "RUN", "text": "Off by exactly one pass of the loop."}])

    return _result("novel", [], [{"type": "RUN", "text": "This code runs but the bug doesn't match a known pattern yet."}], abstain=True)


def resolve_probe(public_problem, code, probe_id, answer):
    """Only the T1 (P03) probe is real. Returns a `Diagnosis` dict."""
    if answer == "4":
        return _result("confident", [card("M01", 0.90)],
                        [{"type": "PROBE", "text": "Correctly placed the last valid index at 4 — the array indexing itself is understood. The bound is what's wrong."}])
    if answer == "5":
        return _result("confident", [card("M08", 0.90)],
                        [{"type": "PROBE", "text": "Placed the last valid index at 5 — arrays are believed to run 1..n, not 0..n-1."}])
    return _result("confident", [card("M01", 0.6)],
                    [{"type": "PROBE", "text": "Unsure answer — leaning on the more common explanation."}])


# ---------------------------------------------------------------- reference fixes (per problem)

_REFERENCE_FIXES = {
    "P01": "void fire_shots(int n) {\n    for (int i = 0; i < n; i++) {\n        fire();\n    }\n}",
    "P03": "int total_energy(int cells[], int n) {\n    int total = 0;\n    for (int i = 0; i < n; i++) {\n        total += cells[i];\n    }\n    return total;\n}",
    "P05": "int charge_steps(int level, int target) {\n    int steps = 0;\n    while (level < target) {\n        level += 7;\n        steps++;\n    }\n    return steps;\n}",
    "P06": "int power_up(int base, int k) {\n    int result = 1;\n    for (int i = 0; i < k; i++) {\n        result = result * base;\n    }\n    return result;\n}",
}


def reference_fix(public_problem):
    code = _REFERENCE_FIXES.get(public_problem["problem_id"])
    if code is None:
        return None
    problem = to_internal_problem(public_problem)
    run = harness.run_tests(problem, code)
    return {"code": code, "verified": run["tests"]["passed"] == run["tests"]["total"]}
