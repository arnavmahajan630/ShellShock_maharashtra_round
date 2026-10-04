"""Live API routes for the Black Hole: Deep Space Trials (rule-based, five fixed problems).

Served under /trials/. The /exam/ paths belong to the adaptive exam in routes/exam.py.

Endpoints:
- POST /trials/start: begins a trials session with 5 canonical DSA items
- POST /trials/run: executes candidate code against sample tests
- POST /trials/answer: logs submission, runs diagnosis, advances to next trial
- POST /trials/finish: compiles debrief report with score & revealed misconceptions
- GET /trials/{exam_id}/report: fetches debrief report
- GET /trials/list: lists the 5 trials with metadata
"""
import json
import time
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Request

from ml.c_interp import harness
from server.app import dsa_rules

router = APIRouter()

FIXTURES_PATH = Path(__file__).resolve().parent.parent.parent / "fixtures" / "dsa_trials.json"

_SESSIONS: Dict[str, Dict[str, Any]] = {}


def _get_trials() -> list:
    return json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))


def _find_trial(item_id: str) -> Dict[str, Any] | None:
    trials = _get_trials()
    return next((t for t in trials if t["problem_id"] == item_id), None)


@router.get("/trials/list")
async def get_trials():
    return _get_trials()


@router.post("/trials/start")
async def start_exam(request: Request):
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    learner_id = body.get("learner_id", "pilot")
    trials = _get_trials()
    exam_id = f"ex_{int(time.time() * 1000)}"

    _SESSIONS[exam_id] = {
        "exam_id": exam_id,
        "learner_id": learner_id,
        "current_index": 0,
        "answers": [],
        "created_at": time.time(),
        "report": None,
    }

    return {
        "exam_id": exam_id,
        "total_trials": len(trials),
        "current_index": 0,
        "item": trials[0],
        "trials": trials,
        "time_limit_s": 1500,
    }


@router.post("/trials/run")
async def run_trial(request: Request):
    body = await request.json()
    item_id = body.get("item_id")
    code = body.get("code") or ""
    trial = _find_trial(item_id)
    if not trial:
        return {"error": f"Unknown trial '{item_id}'"}

    internal = {
        "problem_id": item_id,
        "signature": trial["signature"],
        "tests": trial.get("sample_tests", []),
        "display_test": 0,
        "forbid": [],
    }

    run = harness.run_tests(internal, code)
    trace = harness.trace(internal, code)
    diag = dsa_rules.diagnose_trial(trial, code)

    return {
        "tests": run["tests"],
        "trace": trace,
        "diagnosis": diag,
    }


@router.post("/trials/answer")
async def answer_trial(request: Request):
    body = await request.json()
    exam_id = body.get("exam_id")
    item_id = body.get("item_id")
    code = body.get("code") or ""
    predict_answer = body.get("predict_answer")

    trials = _get_trials()
    trial = _find_trial(item_id)
    if not trial:
        return {"error": f"Unknown trial '{item_id}'"}

    # Diagnostic & test verification
    diag = dsa_rules.diagnose_trial(trial, code)

    sess = _SESSIONS.get(exam_id)
    if not sess:
        sess = {
            "exam_id": exam_id or f"ex_{int(time.time() * 1000)}",
            "current_index": 0,
            "answers": [],
            "report": None,
        }
        _SESSIONS[sess["exam_id"]] = sess

    answer_entry = {
        "problem_id": item_id,
        "code": code,
        "predict_answer": predict_answer,
        "diagnosis": diag,
        "pass": diag["is_correct"],
    }
    # Update existing answer for this problem_id or append
    existing_idx = next((i for i, a in enumerate(sess["answers"]) if a["problem_id"] == item_id), None)
    if existing_idx is not None:
        sess["answers"][existing_idx] = answer_entry
    else:
        sess["answers"].append(answer_entry)

    current_idx = next((i for i, t in enumerate(trials) if t["problem_id"] == item_id), 0)
    next_idx = current_idx + 1

    if next_idx < len(trials):
        next_item = trials[next_idx]
        finished = False
    else:
        next_item = None
        finished = True

    return {
        "logged": True,
        "next_item": next_item,
        "finished": finished,
        "progress": {"k": next_idx if not finished else len(trials), "n": len(trials)},
        "diagnosis": diag,
    }


@router.post("/trials/finish")
async def finish_exam(request: Request):
    body = await request.json()
    exam_id = body.get("exam_id")
    trials = _get_trials()
    sess = _SESSIONS.get(exam_id)

    answers = sess.get("answers", []) if sess else []
    # If some trials weren't answered, fill defaults
    answered_ids = {a["problem_id"] for a in answers}
    for t in trials:
        if t["problem_id"] not in answered_ids:
            answers.append({
                "problem_id": t["problem_id"],
                "code": "",
                "predict_answer": None,
                "diagnosis": dsa_rules.diagnose_trial(t, ""),
                "pass": False,
            })

    report = dsa_rules.compile_debrief(exam_id, answers, trials)
    if sess:
        sess["report"] = report

    return {"report": report}


@router.get("/trials/{exam_id}/report")
async def get_exam_report(exam_id: str):
    sess = _SESSIONS.get(exam_id)
    if sess and sess.get("report"):
        return {"report": sess["report"]}
    trials = _get_trials()
    report = dsa_rules.compile_debrief(exam_id, [], trials)
    return {"report": report}
