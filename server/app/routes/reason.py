"""Reason routes (package S3): POST /reason, POST /lab/reason.

The reader is `ml.text.serve.read(code, text, reader=None)`. `reader=None` keeps T2's load
order (tfidf, then the fine-tuned model in ml/artifacts/reason_9d7b8962/ if tfidf cannot load,
then frozen). An unsure reader does not change the answer (05 §4). E17 measured that this
rule costs about 0.19 accuracy on collision pairs; the threshold is not changed here.
"""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

from ml.bayes.posterior import apply_text
from ml.contracts.classes import MISCONCEPTIONS, REASON_LABELS
from ml.contracts.params import TEXT_MATCH_P
from ml.learner.knowledge import update_p
from ml.learner.state_machine import next_state
from ml.text.serve import model_version, read, top_classes
from server.app.routes import exam as exam_store
from server.app.routes.quiz import quiz_item
from server.app.store import StoreError, get_store

log = logging.getLogger("relearn.reason")
router = APIRouter()


def _error(status, message):
    return JSONResponse(status_code=status, content={"error": message})


def _ms(started):
    return round((time.perf_counter() - started) * 1000.0, 3)


def _attempt_code(attempt_id):
    try:
        from server.app.routes import attempt as attempt_routes
        entry = attempt_routes._recall(attempt_id)
    except Exception:
        entry = None
    if entry:
        analysis = entry.get("analysis")
        code = getattr(analysis, "code", None)
        if isinstance(code, str):
            return code
    row = get_store().get_attempt(attempt_id)
    if row and isinstance(row.get("code"), str):
        return row["code"]
    return None


def _context(learner_id, kind, ref_id):
    """(code, keep, pending dict or None)."""
    if kind == "quiz":
        item = quiz_item(ref_id)
        code = (item or {}).get("code") or ""
        keep = tuple((item or {}).get("classes") or [])
        progress = get_store()._run(lambda conn: (exam_store.ensure(conn), exam_store.load_quiz(conn, learner_id))[1])
        pending = (progress.get("pending") or {}).get(ref_id)
        return code, keep, pending
    if kind == "attempt":
        return _attempt_code(ref_id), (), None
    raise StoreError("ref.kind must be quiz or attempt", 422)


def _p_text(result, label):
    if label not in REASON_LABELS:
        return 0.0
    probs = result.get("probs")
    if probs is None:
        return 0.0
    return float(probs[REASON_LABELS.index(label)])


@router.post("/reason")
def reason(body: dict = Body(...)):
    started = time.perf_counter()
    learner_id = body.get("learner_id") if isinstance(body, dict) else None
    ref = body.get("ref") if isinstance(body, dict) else None
    text = body.get("text") if isinstance(body, dict) else None
    if not isinstance(learner_id, str) or not learner_id:
        return _error(422, "learner_id is required")
    if not isinstance(ref, dict) or ref.get("kind") not in ("quiz", "attempt") or not isinstance(ref.get("id"), str):
        return _error(422, "ref must be {kind: quiz|attempt, id}")
    if not isinstance(text, str):
        return _error(422, "text is required")
    try:
        get_store().get_learner(learner_id)
        code, keep, pending = _context(learner_id, ref["kind"], ref["id"])
    except StoreError as exc:
        return _error(exc.status, str(exc))
    except Exception as exc:
        log.exception("reason failed")
        return _error(500, f"reason failed ({type(exc).__name__})")
    result = read(code, text, reader=None, keep=keep)
    status = result["status"]
    top = top_classes(result)
    updates = []
    # Unsure: the sentence is not evidence. Do not touch P(A) or the option posterior.
    if status != "unsure" and top and not (pending or {}).get("settled"):
        label = top[0]["id"]
        p_text = _p_text(result, label)
        if status == "matched" and label in MISCONCEPTIONS and p_text >= TEXT_MATCH_P:
            try:
                entry = get_store().get_entry(learner_id, label)
            except StoreError as exc:
                return _error(exc.status, str(exc))
            before = float(entry["p_active"])
            after = update_p(before, "why", status="matched", matched=True, p_text=p_text)
            new_state = next_state(entry["state"], after)
            updates.append({"class": label, "p_before": before, "p_after": after, "state": new_state})
        if pending and isinstance(pending.get("posterior"), dict):
            pending["posterior"] = apply_text(pending["posterior"], _text_map(result), unsure=False)
        if pending is not None and (updates or status != "unsure"):
            pending["settled"] = True

    def work(conn):
        exam_store.ensure(conn)
        for update in updates:
            exam_store.write_knowledge(
                conn, learner_id, update["class"], update["p_after"], update["state"], ref["id"],
                f"reason {ref['kind']} {ref['id']}: {status} {update['class']}",
                evidence_type="REASON",
            )
        if ref["kind"] == "quiz" and pending is not None:
            progress = exam_store.load_quiz(conn, learner_id)
            progress["pending"][ref["id"]] = pending
            exam_store.save_quiz(conn, learner_id, progress["asked"], progress["pending"])

    if updates or (ref["kind"] == "quiz" and pending is not None and pending.get("settled")):
        try:
            get_store()._run(work)
        except StoreError as exc:
            return _error(exc.status, str(exc))
        except Exception as exc:
            log.exception("reason failed")
            return _error(500, f"reason failed ({type(exc).__name__})")
    public_updates = [
        {"class": update["class"], "p_before": update["p_before"], "p_after": update["p_after"]}
        for update in updates
    ]
    return {
        "status": status,
        "top": top,
        "reader": result["reader"],
        "updates": public_updates,
        "model_version": model_version(result),
        "latency_ms": _ms(started),
    }


def _text_map(result):
    """Reader labels → Bayes labels. CORRECT_REASON is the posterior's CORRECT (E17)."""
    probs = result.get("probs")
    if probs is None:
        return {}
    out = {}
    for index, label in enumerate(REASON_LABELS):
        key = "CORRECT" if label == "CORRECT_REASON" else label
        out[key] = float(probs[index])
    return out


@router.post("/lab/reason")
def lab_reason(body: dict | None = Body(default=None)):
    """Stateless. Nothing is stored and nothing is updated."""
    started = time.perf_counter()
    if not isinstance(body, dict):
        body = {}
    text = body.get("text") if isinstance(body.get("text"), str) else ""
    code = body.get("code") if isinstance(body.get("code"), str) else None
    result = read(code, text, reader=None)
    return {
        "status": result["status"],
        "top": top_classes(result),
        "reader": result["reader"],
        "updates": [],
        "model_version": model_version(result),
        "latency_ms": _ms(started),
    }
