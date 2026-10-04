"""Live route: POST /probe/answer — real for the first time.

Conditions' twins (M06/M07) are code-separable, so its diagnoser never actually emits
`status: "ambiguous"` — this route only does real work for the Loops P03 hard twin
(M01 Boundary Drift vs M08 Index Origin Fault). Any other problem_id falls through to the
existing generic fixture (a single fixed response), unchanged.
"""
import json
from pathlib import Path

from fastapi import APIRouter, Request

from server.app import loops_rules
from server.app.routes.attempt import _find_problem

router = APIRouter()

FIXTURES = Path(__file__).resolve().parent.parent.parent / "fixtures"
RESOLVABLE_IDS = {"P03"}


@router.post("/probe/answer")
async def probe_answer(request: Request):
    body = await request.json()
    problem_id = body.get("problem_id")
    code = body.get("code") or ""
    probe_id = body.get("probe_id")
    answer = body.get("answer")

    if problem_id not in RESOLVABLE_IDS:
        return json.loads((FIXTURES / "probe_answer.json").read_text(encoding="utf-8"))

    problem = _find_problem(problem_id)
    diagnosis = loops_rules.resolve_probe(problem, code, probe_id, answer)
    return {"diagnosis": diagnosis, "model_version": "rules-v1", "latency_ms": 1.0}
