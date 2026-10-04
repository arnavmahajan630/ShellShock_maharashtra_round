"""Live route for the Conditions planet: POST /intervene.

Builds the trace-timeline modality (the only one Conditions misconceptions need) from the
player's own last attempt, re-run fresh here (stateless, like /lab/diagnose — no server-side
attempt storage yet). Counterexample comes from the real first failing test; the fix panel is
a hand-authored, freshly-verified reference solution (`conditions_rules.reference_fix`).
"""
from fastapi import APIRouter, Request

from ml.c_interp import harness
from ml.contracts.classes import CLASS_INFO
from server.app import conditions_rules
from server.app.routes.attempt import _find_problem

router = APIRouter()

_COPY = {
    "M06": ["`=` inside a condition stores a value; `==` compares two values.", "The `if` becomes true whenever the assigned value isn't 0 — every time, here."],
    "M07": ["A `;` right after `if (...)` ends the statement there — the `{ }` block after it always runs on its own.", "The condition never actually gates anything."],
    "M05": ["A local variable has no starting value until you give it one.", "Reading it before that first assignment is garbage, not 0."],
    "M10": ["`printf` only shows a value on screen — it doesn't hand anything back to whoever called the function.", "Without `return`, the caller gets an unpredictable value."],
}


@router.post("/intervene")
async def intervene(request: Request):
    body = await request.json()
    class_id = body.get("class")
    problem_id = body.get("problem_id")
    code = body.get("code") or ""
    modality = "trace_timeline"

    problem = _find_problem(problem_id)
    timeline, counterexample = [], None
    if problem is not None:
        internal = conditions_rules.to_internal_problem(problem)
        run = harness.run_tests(internal, code)
        trace = harness.trace(internal, code)
        for step in (trace.get("steps") or []):
            if step.get("events"):
                event = step["events"][0]
                timeline.append({
                    "step": step["i"], "line": step["line"], "vars": step["vars"],
                    "effect": event["type"], "flag": event["type"],
                })
            if len(timeline) >= 6:
                break
        first_fail = next((r for r in run["tests"]["results"] if not r["pass"]), None)
        if first_fail is not None:
            counterexample = {
                "input": first_fail["args"], "intended": first_fail["expected"],
                "yours": first_fail["got"], "effect_diff": "wrong result on this input",
            }

    fix = conditions_rules.reference_fix(problem) if problem is not None else None
    info = CLASS_INFO.get(class_id, {})

    return {
        "class": class_id,
        "modality": modality,
        "next_modalities": [],
        "copy": _COPY.get(class_id, [info.get("subtitle", ""), info.get("belief", "")]),
        "question": None,
        "timeline": timeline,
        "counterexample": counterexample,
        "fix": {"kind": "reference", "code": fix["code"], "changed_lines": [], "rule": info.get("subtitle", ""), "verified": fix["verified"]} if fix else None,
        "model_version": "rules-v1",
        "latency_ms": 1.0,
    }
