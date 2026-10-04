"""Live route: GET /trap-items/{class_id} — the no-execution trap question for D07.

Same convention the fixtures already use for `predict_item` (the correct answer is sent
in plain alongside the question; see server/fixtures/problem.json) — grading happens in
the client, no separate grading round-trip needed.
"""
from fastapi import APIRouter, HTTPException

from server.app.conditions_rules import TRAP_ITEMS

router = APIRouter()


@router.get("/trap-items/{class_id}")
def trap_item(class_id: str):
    item = TRAP_ITEMS.get(class_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"no trap item for class {class_id}")
    return {"class": class_id, **item}
