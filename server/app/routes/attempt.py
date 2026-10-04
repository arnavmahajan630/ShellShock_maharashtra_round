"""Live routes for the Conditions and Loops planets: POST /run, POST /attempt,
POST /lab/diagnose.

Real interpreter + real gate, rule-based diagnosis (`server/app/conditions_rules.py`,
`server/app/loops_rules.py` — the trained model isn't ready). Any other problem_id (Q17, the
GATE demo id, ...) falls through to the exact same fixture this path used to answer,
unchanged, so nothing already working breaks.
"""
import json
import time
from pathlib import Path

from fastapi import APIRouter, Request

from ml.c_interp import harness
from server.app import arrays_rules, conditions_rules, functions_rules, gate as gate_module, loops_rules, variables_rules

router = APIRouter()

FIXTURES = Path(__file__).resolve().parent.parent.parent / "fixtures"
CONDITIONS_IDS = {"P11", "P12", "P16", "P17"}
LOOPS_IDS = {"P01", "P03", "P05", "P06"}
ARRAYS_IDS = {"P08", "P07", "P10", "P09"}
VARIABLES_IDS = {"P13", "P14", "P18", "P19"}
FUNCTIONS_IDS = {"P20", "P21", "P22", "P23"}
RULES_BY_ID = {
    **{pid: conditions_rules for pid in CONDITIONS_IDS},
    **{pid: loops_rules for pid in LOOPS_IDS},
    **{pid: arrays_rules for pid in ARRAYS_IDS},
    **{pid: variables_rules for pid in VARIABLES_IDS},
    **{pid: functions_rules for pid in FUNCTIONS_IDS},
}


def _problems_catalog():
    return json.loads((FIXTURES / "problems.json").read_text(encoding="utf-8"))


def _find_problem(problem_id):
    return next((p for p in _problems_catalog() if p["problem_id"] == problem_id), None)


def _fixture_case(filename, by_value):
    data = json.loads((FIXTURES / filename).read_text(encoding="utf-8"))
    return data["cases"].get(str(by_value), data["cases"][data["default"]])


def _run_and_diagnose(problem, code, rules, want_diagnosis):
    gate_result = gate_module.check(problem, code)
    if gate_result["code"] != "G0":
        return {"gate": gate_result, "trace": None, "tests": None, "diagnosis": None}

    internal = rules.to_internal_problem(problem)
    run = harness.run_tests(internal, code)
    trace = harness.trace(internal, code)
    diagnosis = rules.diagnose(problem, code) if want_diagnosis else None
    return {"gate": gate_result, "trace": trace, "tests": run["tests"], "diagnosis": diagnosis}


async def _problem_id_and_code(request: Request):
    body = await request.json()
    return body.get("problem_id"), body.get("code") or ""


@router.post("/run")
async def run(request: Request):
    problem_id, code = await _problem_id_and_code(request)
    rules = RULES_BY_ID.get(problem_id)
    problem = _find_problem(problem_id) if rules else None
    if problem is None:
        return _fixture_case("run.json", problem_id)
    out = _run_and_diagnose(problem, code, rules, want_diagnosis=False)
    return {**out, "model_version": "rules-v1", "latency_ms": 1.0}


@router.post("/attempt")
async def attempt(request: Request):
    body = await request.json()
    problem_id, code = body.get("problem_id"), body.get("code") or ""
    rules = RULES_BY_ID.get(problem_id)
    problem = _find_problem(problem_id) if rules else None
    if problem is None:
        return _fixture_case("attempt.json", problem_id)
    out = _run_and_diagnose(problem, code, rules, want_diagnosis=True)
    return {
        "attempt_id": f"at_{problem_id}_{int(time.time() * 1000)}",
        **out,
        "model_version": "rules-v1",
        "latency_ms": 1.0,
    }


@router.post("/lab/diagnose")
async def lab_diagnose(request: Request):
    body = await request.json()
    problem_id, code = body.get("problem_id"), body.get("code") or ""
    rules = RULES_BY_ID.get(problem_id)
    problem = _find_problem(problem_id) if rules else None
    if problem is None:
        return _fixture_case("lab_diagnose.json", problem_id)
    out = _run_and_diagnose(problem, code, rules, want_diagnosis=True)
    return {**out, "model_version": "rules-v1", "latency_ms": 1.0}
