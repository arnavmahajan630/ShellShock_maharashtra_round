"""Live route for the Conditions and Loops planets: POST /intervene.

Builds whichever D06 modality panel the diagnosed class actually calls for
(`server/app/diagnosis_common.CLASS_MODALITY`) from the player's own last attempt, re-run
fresh here (stateless, like /lab/diagnose — no server-side attempt storage yet). Counterexample
comes from the real first failing test; the fix panel is a hand-authored, freshly-verified
reference solution.
"""
from fastapi import APIRouter, Request

from ml.c_interp import harness
from ml.contracts.classes import CLASS_INFO
from server.app import conditions_rules
from server.app.diagnosis_common import CLASS_MODALITY
from server.app.routes.attempt import RULES_BY_ID, _find_problem

router = APIRouter()

_COPY = {
    "M01": ["The loop's stopping condition includes one extra (or one too few) pass.", "Check whether the last value of the loop variable is still inside the valid range."],
    "M02": ["The loop's test never becomes false because the variable it checks never changes inside the loop body.", "Something inside the loop needs to move the variable toward the exit."],
    "M03": ["Reassigning the accumulator with `=` replaces its value instead of building on it.", "Use `+=` (or the matching combine operator) so each pass adds to the last."],
    "M05": ["A local variable has no starting value until you give it one.", "Reading it before that first assignment is garbage, not 0."],
    "M06": ["`=` inside a condition stores a value; `==` compares two values.", "The condition becomes true whenever the assigned value isn't 0 — every time, here."],
    "M07": ["A `;` right after the loop/if header ends the statement there — the `{ }` block after it runs on its own.", "The condition never actually gates anything."],
    "M08": ["Arrays are indexed 0 through n−1, not 1 through n.", "The last valid cell of an n-element array is cell n−1, not cell n."],
    "M10": ["`printf` only shows a value on screen — it doesn't hand anything back to whoever called the function.", "Without `return`, the caller gets an unpredictable value."],
}


def _trace_timeline(trace):
    timeline = []
    for step in trace.get("steps") or []:
        if step.get("events"):
            event = step["events"][0]
            timeline.append({"step": step["i"], "line": step["line"], "vars": step["vars"], "effect": event["type"], "flag": event["type"]})
        if len(timeline) >= 6:
            break
    return timeline


def _value_meter(problem, rules, first_fail):
    """exact (reference solution's value on the failing input) vs shown (the player's own)."""
    if first_fail is None:
        return None
    fix = rules.reference_fix(problem)
    exact = first_fail["expected"].get("returned") if fix is None else None
    if fix is not None:
        internal = rules.to_internal_problem(problem)
        ref_run = harness.run_tests(internal, fix["code"])
        matching = next((r for r in ref_run["tests"]["results"] if r["args"] == first_fail["args"]), None)
        if matching is not None:
            exact = matching["got"].get("returned")
    return {"exact": exact, "shown": first_fail["got"].get("returned")}


def _memory_strip(first_fail, boundary_idx):
    if first_fail is None or not first_fail["args"]:
        return None
    array_arg = next((a for a in first_fail["args"] if isinstance(a, list)), None)
    if array_arg is None:
        return None
    reads = list(range(0, (boundary_idx if boundary_idx is not None else len(array_arg)) + 1))
    return {"array": "cells", "values": array_arg, "reads": reads}


@router.post("/intervene")
async def intervene(request: Request):
    body = await request.json()
    class_id = body.get("class")
    problem_id = body.get("problem_id")
    code = body.get("code") or ""
    modality = CLASS_MODALITY.get(class_id, "trace_timeline")

    rules = RULES_BY_ID.get(problem_id, conditions_rules)
    problem = _find_problem(problem_id)

    timeline, counterexample, value_meter, memory_strip = [], None, None, None
    if problem is not None:
        internal = rules.to_internal_problem(problem)
        run = harness.run_tests(internal, code)
        trace = harness.trace(internal, code)
        first_fail = next((r for r in run["tests"]["results"] if not r["pass"]), None)
        if first_fail is not None:
            counterexample = {
                "input": first_fail["args"], "intended": first_fail["expected"],
                "yours": first_fail["got"], "effect_diff": "wrong result on this input",
            }

        if modality == "value_meter":
            value_meter = _value_meter(problem, rules, first_fail)
        elif modality == "memory_strip":
            boundary = next((e for e in (trace.get("events") or []) if e["type"] in ("oob_read", "oob_write")), None)
            memory_strip = _memory_strip(first_fail, boundary.get("idx") if boundary else None)
        else:
            timeline = _trace_timeline(trace)

    fix = rules.reference_fix(problem) if problem is not None else None
    info = CLASS_INFO.get(class_id, {})

    return {
        "class": class_id,
        "modality": modality,
        "next_modalities": [],
        "copy": _COPY.get(class_id, [info.get("subtitle", ""), info.get("belief", "")]),
        "question": None,
        "timeline": timeline,
        "memory_strip": memory_strip,
        "value_meter": value_meter,
        "counterexample": counterexample,
        "fix": {"kind": "reference", "code": fix["code"], "changed_lines": [], "rule": info.get("subtitle", ""), "verified": fix["verified"]} if fix else None,
        "model_version": "rules-v1",
        "latency_ms": 1.0,
    }
