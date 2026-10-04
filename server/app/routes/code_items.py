"""Code-item routes (package S3): GET /code-items, POST /code-items/debug.

Items come from `ml/data/code_items.json` (C4). The list is `PublicCodeItem` only.
`fix` (including snippet answers under `fix.answers`) is never sent on the list.
`POST /code-items/debug` does return `fix`: that is the reveal after the learner picks a line.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from fastapi import APIRouter, Body, Query
from fastapi.responses import JSONResponse

from ml.contracts.schemas import PublicCodeItem
from ml.learner.knowledge import update_p
from ml.learner.state_machine import next_state
from server.app.routes import exam as exam_store
from server.app.store import StoreError, get_store

log = logging.getLogger("relearn.code_items")
router = APIRouter()

ROOT = Path(__file__).resolve().parents[3]
ITEMS_PATH = ROOT / "ml" / "data" / "code_items.json"
MODEL_VERSION = "code-items"
_HIDDEN = ("planted", "bug_lines", "fix", "exposes", "op_id", "explain")

_cache = {"mtime": None, "items": []}


def _error(status, message):
    return JSONResponse(status_code=status, content={"error": message})


def _ms(started):
    return round((time.perf_counter() - started) * 1000.0, 3)


def code_items():
    mtime = ITEMS_PATH.stat().st_mtime if ITEMS_PATH.exists() else None
    if _cache["mtime"] != mtime:
        _cache["items"] = json.loads(ITEMS_PATH.read_text(encoding="utf-8")) if mtime else []
        _cache["mtime"] = mtime
    return _cache["items"]


def code_item(item_id):
    return next((item for item in code_items() if item.get("item_id") == item_id), None)


def public_code_item(item):
    """Fields the list may show. `fix` stays off this object."""
    data = {
        "item_id": item["item_id"],
        "type": item["type"],
        "problem_id": item["problem_id"],
    }
    if item.get("starter") is not None:
        data["starter"] = item["starter"]
    if item.get("holes"):
        data["holes"] = item["holes"]
    if item.get("code") is not None:
        data["code"] = item["code"]
    if item.get("type") == "debug_line":
        data["allow_no_bug"] = bool(item.get("allow_no_bug", True))
    checked = PublicCodeItem.model_validate(data).model_dump()
    for key in _HIDDEN:
        checked.pop(key, None)
    if item.get("type") != "debug_line":
        checked.pop("allow_no_bug", None)
    return {key: value for key, value in checked.items() if value is not None or key == "allow_no_bug"}


@router.get("/code-items")
def list_code_items(problem_id: str = "", type_: str = Query("", alias="type")):
    rows = []
    for item in code_items():
        if problem_id and item.get("problem_id") != problem_id:
            continue
        if type_ and item.get("type") != type_:
            continue
        rows.append(public_code_item(item))
    return rows


@router.post("/code-items/debug")
def debug_answer(body: dict = Body(...)):
    started = time.perf_counter()
    learner_id = body.get("learner_id") if isinstance(body, dict) else None
    item_id = body.get("item_id") if isinstance(body, dict) else None
    if not isinstance(learner_id, str) or not learner_id:
        return _error(422, "learner_id is required")
    if not isinstance(item_id, str) or not item_id:
        return _error(422, "item_id is required")
    if "line" not in body:
        line = None
    else:
        line = body.get("line")
    if line is not None and (isinstance(line, bool) or not isinstance(line, int)):
        return _error(422, "line must be an integer or null")
    item = code_item(item_id)
    if item is None:
        return _error(404, f"unknown code item {item_id}")
    if item.get("type") != "debug_line":
        return _error(422, "item is not a debug_line")
    try:
        learner = get_store().get_learner(learner_id)
    except StoreError as exc:
        return _error(exc.status, str(exc))
    bug_lines = [int(number) for number in item.get("bug_lines") or []]
    planted = item.get("planted")
    no_bug = planted is None or not bug_lines
    correct = (line is None) if no_bug else (line is not None and int(line) in bug_lines)
    updates = []
    if planted is not None:
        cls = planted
        entry = learner["misconceptions"][cls]
        before = float(entry["p_active"])
        after = update_p(before, "debug_line", planted=planted, correct=correct)
        new_state = next_state(entry["state"], after, {"after_item": True})
        updates.append({"class": cls, "p_before": before, "p_after": after, "state": new_state})

    def work(conn):
        exam_store.ensure(conn)
        for update in updates:
            exam_store.write_knowledge(
                conn, learner_id, update["class"], update["p_after"], update["state"], item_id,
                f"debug {item_id}: line {line} {'correct' if correct else 'wrong'}",
                evidence_type="QUIZ",
            )

    if updates:
        try:
            get_store()._run(work)
        except StoreError as exc:
            return _error(exc.status, str(exc))
        except Exception as exc:
            log.exception("debug answer failed")
            return _error(500, f"debug answer failed ({type(exc).__name__})")
    public_updates = [
        {"class": update["class"], "p_before": update["p_before"], "p_after": update["p_after"]}
        for update in updates
    ]
    return {
        "correct": correct,
        "bug_lines": bug_lines,
        "explain": item.get("explain") or "",
        "fix": item.get("fix"),
        "updates": public_updates,
        "model_version": MODEL_VERSION,
        "latency_ms": _ms(started),
    }
