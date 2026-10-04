"""Quiz routes (package S3): GET /quiz/next, POST /quiz/answer.

Selection and scoring are D3's (`ml.quiz.select`). Asked items and the collision context
for `/reason` are stored in `quiz_progress` (created by `routes/exam.py` on the S1 connection).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from fastapi import APIRouter, Body, Query
from fastapi.responses import JSONResponse

from ml.bayes.posterior import prior as bayes_prior
from ml.bayes.posterior import update as bayes_update
from ml.contracts.classes import LABELS
from ml.learner.state_machine import next_state
from ml.quiz.select import next_item, score_answer
from server.app.routes import exam as exam_store
from server.app.store import StoreError, get_store

log = logging.getLogger("relearn.quiz")
router = APIRouter()

ROOT = Path(__file__).resolve().parents[3]
QUIZ_PATH = ROOT / "ml" / "data" / "quiz_items.json"
MODEL_VERSION = "quiz"

_cache = {"mtime": None, "items": []}


def _error(status, message):
    return JSONResponse(status_code=status, content={"error": message})


def _ms(started):
    return round((time.perf_counter() - started) * 1000.0, 3)


def quiz_items():
    mtime = QUIZ_PATH.stat().st_mtime if QUIZ_PATH.exists() else None
    if _cache["mtime"] != mtime:
        _cache["items"] = json.loads(QUIZ_PATH.read_text(encoding="utf-8")) if mtime else []
        _cache["mtime"] = mtime
    return _cache["items"]


def quiz_item(item_id):
    return next((item for item in quiz_items() if item.get("item_id") == item_id), None)


def _learner_state(learner_id, asked):
    learner = get_store().get_learner(learner_id)
    return {
        "learner_id": learner_id,
        "asked": list(asked),
        "misconceptions": learner["misconceptions"],
        "p_active": {cls: entry["p_active"] for cls, entry in learner["misconceptions"].items()},
        "states": {cls: entry["state"] for cls, entry in learner["misconceptions"].items()},
    }


def _uniform():
    share = 1.0 / len(LABELS)
    return {label: share for label in LABELS}


def _option_posterior(learner, item, answer):
    """Option-alone posterior (E17). `/reason` multiplies by the sentence when the reader is sure."""
    p_active = {cls: entry["p_active"] for cls, entry in learner["misconceptions"].items()}
    posterior = bayes_prior(_uniform(), p_active)
    return bayes_update(
        posterior, str(answer), correct=item.get("correct"), belief=item.get("belief"), options=item.get("options"),
    )


@router.get("/quiz/next")
def quiz_next(learner_id: str = "", type_: str = Query("", alias="type"), concept: str = ""):
    started = time.perf_counter()
    asked = []
    state = {"learner_id": learner_id or "quiz", "asked": [], "p_active": {}}
    if learner_id:
        try:
            progress = get_store()._run(lambda conn: (exam_store.ensure(conn), exam_store.load_quiz(conn, learner_id))[1])
            asked = progress["asked"]
            state = _learner_state(learner_id, asked)
        except StoreError as exc:
            return _error(exc.status, str(exc))
        except Exception as exc:
            log.exception("quiz next failed")
            return _error(500, f"quiz next failed ({exc.__class__.__name__})")
    try:
        item, reason = next_item(
            state, quiz_items(), type=type_ or None, concept=concept or None,
        )
    except ValueError:
        return _error(404, "no quiz item matches the filters")
    except Exception as exc:
        log.exception("quiz next failed")
        return _error(500, f"quiz next failed ({type(exc).__name__})")
    return {"item": item, "reason": reason, "model_version": MODEL_VERSION, "latency_ms": _ms(started)}


@router.post("/quiz/answer")
def quiz_answer(body: dict = Body(...)):
    started = time.perf_counter()
    learner_id = body.get("learner_id") if isinstance(body, dict) else None
    item_id = body.get("item_id") if isinstance(body, dict) else None
    if not isinstance(learner_id, str) or not learner_id:
        return _error(422, "learner_id is required")
    if not isinstance(item_id, str) or not item_id:
        return _error(422, "item_id is required")
    if "answer" not in body:
        return _error(422, "answer is required")
    item = quiz_item(item_id)
    if item is None:
        return _error(404, f"unknown quiz item {item_id}")
    if item.get("type") == "reasoning":
        return _error(422, "reasoning answers go to POST /reason")
    try:
        learner = get_store().get_learner(learner_id)
        progress = get_store()._run(lambda conn: (exam_store.ensure(conn), exam_store.load_quiz(conn, learner_id))[1])
        scored = score_answer(_learner_state(learner_id, progress["asked"]), item, body.get("answer"))
    except StoreError as exc:
        return _error(exc.status, str(exc))
    except ValueError as exc:
        return _error(422, str(exc))
    except Exception as exc:
        log.exception("quiz answer failed")
        return _error(500, f"quiz answer failed ({type(exc).__name__})")

    asked = list(progress["asked"])
    if item_id not in asked:
        asked.append(item_id)
    pending = dict(progress["pending"])
    if scored["ask_reason"]:
        try:
            posterior = _option_posterior(learner, item, body.get("answer"))
        except ValueError:
            posterior = None
        pending[item_id] = {
            "item_id": item_id,
            "code": item.get("code") or "",
            "classes": list(item.get("classes") or []),
            "answer": str(body.get("answer")),
            "posterior": posterior,
            "settled": False,
        }
    else:
        pending.pop(item_id, None)

    updates = scored["updates"]

    def work(conn):
        exam_store.ensure(conn)
        for update in updates:
            cls = update["class"]
            entry_state = learner["misconceptions"][cls]["state"]
            new_state = next_state(entry_state, update["p_after"], {"after_item": True})
            exam_store.write_knowledge(
                conn, learner_id, cls, update["p_after"], new_state, item_id,
                f"quiz {item_id}: {'correct' if scored['correct'] else 'wrong'}",
                evidence_type="QUIZ",
            )
        exam_store.save_quiz(conn, learner_id, asked, pending)

    try:
        get_store()._run(work)
    except StoreError as exc:
        return _error(exc.status, str(exc))
    except Exception as exc:
        log.exception("quiz answer failed")
        return _error(500, f"quiz answer failed ({type(exc).__name__})")
    return {**scored, "model_version": MODEL_VERSION, "latency_ms": _ms(started)}
