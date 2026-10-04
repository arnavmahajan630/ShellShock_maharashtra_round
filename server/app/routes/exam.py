"""Exam routes (package S3): POST /exam/start, /exam/answer, /exam/finish, /exam/probe, GET /exam/{id}/report.

The selector, observation model and debrief are D3's (`ml.exam`). This file is the HTTP layer.
Exam rows live in an `exams` table (03 §11.3). S1's store has no exams table and this package
cannot edit `store.py`, so the table is created here on the store's own connection.

Knowledge P(A) for classes the exam actually tested is written back through that same connection
(`Store._read_entry` / `_write_entry`). `reassess` is not used: it would apply a second, different
update, and a `predict_output` reassess on an ACTIVE class starts an intervention.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

from ml.bayes.eig import load_probes
from ml.bayes.likelihood import likelihood
from ml.contracts.classes import MISCONCEPTIONS, PLANETS
from ml.contracts.params import EXAM_LENGTH, EXAM_TIME_LIMIT_S
from ml.contracts.schemas import ExamReport
from ml.exam.report import build_report
from ml.exam.run import _apply, _elo, _serve, _session
from ml.exam.select import load_pool, public_exam_item, select_next
from ml.learner.knowledge import likelihood_ratio
from ml.learner.state_machine import next_state
from server.app import pipeline
from server.app.store import StoreError, get_store

log = logging.getLogger("relearn.exam")
router = APIRouter()

ROOT = Path(__file__).resolve().parents[3]
_SET_KEYS = ("covered", "used", "tested")

EXAM_SCHEMA = """
CREATE TABLE IF NOT EXISTS exams (
    exam_id     TEXT PRIMARY KEY,
    learner_id  TEXT NOT NULL,
    state_json  TEXT NOT NULL,
    log_json    TEXT NOT NULL DEFAULT '[]',
    report_json TEXT
);
CREATE TABLE IF NOT EXISTS quiz_progress (
    learner_id   TEXT PRIMARY KEY,
    asked_json   TEXT NOT NULL DEFAULT '[]',
    pending_json TEXT NOT NULL DEFAULT '{}'
);
"""

_pool_cache = None


def _error(status, message):
    return JSONResponse(status_code=status, content={"error": message})


def _ms(started):
    return round((time.perf_counter() - started) * 1000.0, 3)


def _plain(value):
    """JSON-safe copy. Numpy scalars show up inside a diagnosis snapshot."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, set):
        return sorted((_plain(item) for item in value), key=str)
    if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return _plain(value.item())
        except (ValueError, AttributeError):
            pass
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    return value


def model_version():
    """Folder name of the newest diagnoser artifact. Does not load the model."""
    folders = [path for path in (ROOT / "ml" / "artifacts").glob("diagnoser_*") if path.is_dir()]
    if not folders:
        return "exam"
    return max(folders, key=lambda path: path.stat().st_mtime).name


def pool():
    global _pool_cache
    if _pool_cache is None:
        _pool_cache = load_pool(strings=True)
    return _pool_cache


def ensure(conn):
    conn.executescript(EXAM_SCHEMA)


def freeze(session):
    data = _plain(session)
    for key in _SET_KEYS:
        data[key] = sorted(data.get(key) or [])
    return data


def thaw(data):
    data = dict(data)
    for key in _SET_KEYS:
        data[key] = set(data.get(key) or [])
    return data


def save_exam(conn, exam_id, learner_id, session, report=None):
    report_json = json.dumps(report) if report is not None else None
    conn.execute(
        "INSERT INTO exams (exam_id, learner_id, state_json, log_json, report_json) VALUES (?,?,?,?,?) "
        "ON CONFLICT(exam_id) DO UPDATE SET state_json=excluded.state_json, log_json=excluded.log_json, "
        "report_json=COALESCE(excluded.report_json, exams.report_json)",
        (exam_id, learner_id, json.dumps(freeze(session)), json.dumps(session.get("administered") or []), report_json),
    )


def load_exam(conn, exam_id):
    row = conn.execute("SELECT * FROM exams WHERE exam_id=?", (exam_id,)).fetchone()
    if row is None:
        raise StoreError(f"unknown exam {exam_id}", 404)
    report = json.loads(row["report_json"]) if row["report_json"] else None
    return {"learner_id": row["learner_id"], "session": thaw(json.loads(row["state_json"])), "report": report}


def load_quiz(conn, learner_id):
    row = conn.execute("SELECT asked_json, pending_json FROM quiz_progress WHERE learner_id=?", (learner_id,)).fetchone()
    if row is None:
        return {"asked": [], "pending": {}}
    return {"asked": json.loads(row["asked_json"] or "[]"), "pending": json.loads(row["pending_json"] or "{}")}


def save_quiz(conn, learner_id, asked, pending):
    conn.execute(
        "INSERT INTO quiz_progress (learner_id, asked_json, pending_json) VALUES (?,?,?) "
        "ON CONFLICT(learner_id) DO UPDATE SET asked_json=excluded.asked_json, pending_json=excluded.pending_json",
        (learner_id, json.dumps(list(asked)), json.dumps(pending)),
    )


def write_knowledge(conn, learner_id, cls, p_active, state, item_id, text, evidence_type="EXAM"):
    """Set one class to an already-computed P and state. Does not run a second update."""
    store = get_store()
    entry = store._read_entry(conn, learner_id, cls)
    before = float(entry["p_active"])
    entry["p_active"] = float(p_active)
    entry["state"] = state
    entry["times_seen"] = int(entry["times_seen"]) + 1
    entry["last_tested"] = item_id
    entry["evidence"].append({"type": evidence_type, "text": text})
    store._write_entry(conn, learner_id, cls, entry)
    store._event(conn, learner_id, "s3", {
        "class": cls, "item_id": item_id, "p_before": before, "p_after": float(p_active), "state": state,
    })
    return before


def _sync_exam_classes(conn, learner_id, session, before_counts, item_id, outcome):
    for cls in MISCONCEPTIONS:
        if len(session["evidence"].get(cls) or []) <= before_counts.get(cls, 0):
            continue
        write_knowledge(
            conn, learner_id, cls, session["p"][cls], session["states"][cls], item_id,
            f"{item_id}: {outcome}", evidence_type="EXAM",
        )


def _evidence_counts(session):
    return {cls: len(session["evidence"].get(cls) or []) for cls in MISCONCEPTIONS}


def _learner_snapshot(learner_id):
    learner = get_store().get_learner(learner_id)
    states, p_active, stable_since = {}, {}, {}
    for cls, entry in learner["misconceptions"].items():
        states[cls] = entry["state"]
        p_active[cls] = entry["p_active"]
        if entry["state"] == "STABLE" and entry.get("last_tested"):
            stable_since[cls] = entry["last_tested"]
    done = set()
    for key, node in (learner.get("nodes") or {}).items():
        if isinstance(node, dict) and node.get("status") == "done" and ":" in str(key):
            done.add(str(key).split(":", 1)[0])
    return p_active, states, stable_since, set(PLANETS) <= done


def _mode(body):
    demo = bool(body.get("demo"))
    length = body.get("length", 10)
    if not isinstance(length, int) or isinstance(length, bool):
        raise StoreError("length must be an integer", 422)
    # Only demo (2 coding + 1 trace) and full (5 + 5) exist. `demo: true` or length <= 3 selects demo.
    if demo or length <= sum(EXAM_LENGTH["demo"]):
        return "demo"
    return "full"


def _public(item):
    data = public_exam_item(item)
    return {key: value for key, value in data.items() if value is not None}


def _progress(session):
    n = int(session["n_coding_target"]) + int(session["n_trace_target"])
    answered = len(session["administered"])
    k = answered + 1 if session.get("pending") else answered
    return {"k": k, "n": n}


def _diagnoser(learner_id, problem, code):
    """Gate, then the S2 pipeline with probes not applied. No class is invented if the model is down."""
    try:
        analysis = pipeline.analyse(problem, code, learner_id=learner_id)
    except pipeline.ModelUnavailable:
        return _tests_only(problem, code)
    except Exception:
        log.exception("exam diagnose failed")
        return {"status": "fail", "top": [], "posterior": {}, "latent": None}
    if analysis.gate["code"] != "G0":
        return {"status": "gate", "top": [], "posterior": {}, "latent": None, "twin_set": None}
    try:
        return _plain(pipeline.diagnose(analysis))
    except Exception:
        log.exception("exam diagnose failed")
        return {"status": "fail", "top": [], "posterior": {}, "latent": None}


def _tests_only(problem, code):
    try:
        from ml import runner
        result = runner.run_tests(problem, code)
        tests = result.get("tests") or {}
        passed = tests.get("total") and tests.get("passed") == tests.get("total")
    except Exception:
        log.exception("exam tests-only fallback failed")
        passed = False
    return {"status": "correct" if passed else "fail", "top": [], "posterior": {}, "latent": None}


def _report(session):
    session["time_used_s"] = int(session.get("time_used_ms") or 0) // 1000
    report = build_report(session).model_dump(by_alias=True)
    report = _plain(report)
    ExamReport.model_validate(report)
    return report


def _open_next(session):
    picked = select_next(pool(), session, session["n_coding_target"], session["n_trace_target"])
    if picked is None:
        session["pending"] = None
        return
    item, reason, eig = picked
    session["pending"] = {"item": item, "reason": reason, "eig": float(eig)}


# ---------------------------------------------------------------- routes

@router.post("/exam/start")
def exam_start(body: dict = Body(...)):
    started = time.perf_counter()
    learner_id = body.get("learner_id") if isinstance(body, dict) else None
    if not isinstance(learner_id, str) or not learner_id:
        return _error(422, "learner_id is required")
    try:
        mode = _mode(body)
        p_active, states, stable_since, planets_complete = _learner_snapshot(learner_id)
    except StoreError as exc:
        return _error(exc.status, str(exc))
    n_coding, n_trace = EXAM_LENGTH[mode]
    exam_id = f"ex_{uuid.uuid4().hex[:8]}"
    session = _session(
        learner_id=learner_id, exam_id=exam_id, p_active=p_active, states=states,
        planets_complete=planets_complete, strings=True, time_used_s=0, stable_since=stable_since,
    )
    session["mode"] = mode
    session["n_coding_target"] = n_coding
    session["n_trace_target"] = n_trace
    session["time_used_ms"] = 0
    session["finished"] = False
    session["probes_asked"] = []
    try:
        _open_next(session)
    except Exception as exc:
        log.exception("exam select failed")
        return _error(500, f"exam select failed ({type(exc).__name__})")
    if session.get("pending") is None:
        return _error(500, "exam select failed (ValueError)")
    try:
        get_store()._run(lambda conn: (ensure(conn), save_exam(conn, exam_id, learner_id, session)))
    except StoreError as exc:
        return _error(exc.status, str(exc))
    except Exception as exc:
        log.exception("exam start failed")
        return _error(500, f"exam start failed ({type(exc).__name__})")
    return {
        "exam_id": exam_id,
        "item": _public(session["pending"]["item"]),
        "progress": _progress(session),
        "time_limit_s": EXAM_TIME_LIMIT_S,
        "model_version": model_version(),
        "latency_ms": _ms(started),
    }


@router.post("/exam/answer")
def exam_answer(body: dict = Body(...)):
    started = time.perf_counter()
    exam_id = body.get("exam_id") if isinstance(body, dict) else None
    item_id = body.get("item_id") if isinstance(body, dict) else None
    if not isinstance(exam_id, str) or not exam_id:
        return _error(422, "exam_id is required")
    if not isinstance(item_id, str) or not item_id:
        return _error(422, "item_id is required")
    store = get_store()
    try:
        record = store._run(lambda conn: (ensure(conn), load_exam(conn, exam_id))[1])
    except StoreError as exc:
        return _error(exc.status, str(exc))
    session = record["session"]
    if session.get("finished"):
        return _error(422, "exam is already finished")
    pending = session.get("pending")
    if not pending:
        return _error(422, "exam has no item waiting")
    if pending["item"]["item_id"] != item_id:
        return _error(422, f"expected item {pending['item']['item_id']}")
    item = pending["item"]
    reason, eig = pending["reason"], pending["eig"]
    skipped = bool(body.get("skipped"))
    if not skipped and item["kind"] == "trace":
        answer = body.get("answer")
        options = [str(option) for option in item.get("options") or []]
        if answer is None or str(answer) not in options:
            return _error(422, "answer is not one of the options")
    diagnosis = None
    if not skipped and item["kind"] == "coding":
        code = body.get("code") if isinstance(body.get("code"), str) else ""
        diagnosis = _diagnoser(session["learner_id"], item.get("problem") or {}, code)
    ms = body.get("ms", 0)
    if isinstance(ms, bool) or not isinstance(ms, int):
        ms = 0
    before = _evidence_counts(session)
    try:
        if skipped:
            _elo(session, item, False)
            _serve(session, item)
            session["administered"].append({
                "item_id": item["item_id"], "sector": item["sector"], "passed": False,
                "eig_bits": float(eig), "reason": reason, "outcome": "skip",
            })
            outcome = "skip"
        elif item["kind"] == "trace":
            _apply(session, item, {"answer": str(body.get("answer"))}, None, reason, eig)
            outcome = session["administered"][-1]["outcome"]
        else:
            held = diagnosis
            _apply(session, item, {"code": body.get("code") or ""}, lambda _problem, _code: held, reason, eig)
            outcome = session["administered"][-1]["outcome"]
    except Exception as exc:
        log.exception("exam answer failed")
        return _error(500, f"exam answer failed ({type(exc).__name__})")
    session["time_used_ms"] = int(session.get("time_used_ms") or 0) + max(int(ms), 0)
    if body.get("sample_runs") is not None:
        session["administered"][-1]["sample_runs"] = body.get("sample_runs")
    try:
        _open_next(session)
    except Exception as exc:
        log.exception("exam select failed")
        return _error(500, f"exam select failed ({type(exc).__name__})")
    next_item = _public(session["pending"]["item"]) if session.get("pending") else None
    progress = _progress(session)

    def work(conn):
        ensure(conn)
        if not skipped:
            _sync_exam_classes(conn, session["learner_id"], session, before, item_id, outcome)
        save_exam(conn, exam_id, session["learner_id"], session)

    try:
        store._run(work)
    except StoreError as exc:
        return _error(exc.status, str(exc))
    except Exception as exc:
        log.exception("exam answer failed")
        return _error(500, f"exam answer failed ({type(exc).__name__})")
    return {
        "logged": True,
        "next_item": next_item,
        "progress": progress,
        "model_version": model_version(),
        "latency_ms": _ms(started),
    }


@router.post("/exam/finish")
def exam_finish(body: dict = Body(...)):
    started = time.perf_counter()
    exam_id = body.get("exam_id") if isinstance(body, dict) else None
    if not isinstance(exam_id, str) or not exam_id:
        return _error(422, "exam_id is required")
    store = get_store()
    try:
        record = store._run(lambda conn: (ensure(conn), load_exam(conn, exam_id))[1])
    except StoreError as exc:
        return _error(exc.status, str(exc))
    session = record["session"]
    if session.get("finished") and record.get("report"):
        report = record["report"]
    else:
        session["finished"] = True
        session["pending"] = None
        try:
            report = _report(session)
        except Exception as exc:
            log.exception("exam report failed")
            return _error(500, f"exam report failed ({type(exc).__name__})")

        def work(conn):
            ensure(conn)
            save_exam(conn, exam_id, session["learner_id"], session, report)

        try:
            store._run(work)
        except Exception as exc:
            log.exception("exam finish failed")
            return _error(500, f"exam finish failed ({type(exc).__name__})")
    return {"report": report, "model_version": model_version(), "latency_ms": _ms(started)}


@router.post("/exam/probe")
def exam_probe(body: dict = Body(...)):
    """A deferred probe. Likelihood separates the belief classes; an unsure threshold is not involved."""
    started = time.perf_counter()
    exam_id = body.get("exam_id") if isinstance(body, dict) else None
    probe_id = body.get("probe_id") if isinstance(body, dict) else None
    answer = body.get("answer") if isinstance(body, dict) else None
    if not isinstance(exam_id, str) or not exam_id:
        return _error(422, "exam_id is required")
    if not isinstance(probe_id, str) or not probe_id:
        return _error(422, "probe_id is required")
    if answer is None:
        return _error(422, "answer is required")
    probe = next((row for row in load_probes() if row["probe_id"] == probe_id), None)
    if probe is None:
        return _error(422, f"unknown probe {probe_id}")
    options = [str(option) for option in probe.get("options") or []]
    if str(answer) not in options:
        return _error(422, "answer is not one of the options")
    store = get_store()
    try:
        record = store._run(lambda conn: (ensure(conn), load_exam(conn, exam_id))[1])
    except StoreError as exc:
        return _error(exc.status, str(exc))
    session = record["session"]
    if probe_id in (session.get("probes_asked") or []):
        try:
            report = record["report"] or _report(session)
        except Exception as exc:
            log.exception("exam report failed")
            return _error(500, f"exam report failed ({type(exc).__name__})")
        return {"report": report, "model_version": model_version(), "latency_ms": _ms(started)}
    table = likelihood(probe["options"], probe["correct"], probe.get("belief"))
    answer_s = str(answer)
    like_not = table["CORRECT"][answer_s]
    changed = []
    for cls in probe.get("belief") or {}:
        if cls not in MISCONCEPTIONS or cls not in session["p"]:
            continue
        old = float(session["p"][cls])
        new_p = likelihood_ratio(old, table[cls][answer_s], like_not)
        new_state = next_state(session["states"][cls], new_p)
        session["p"][cls] = new_p
        session["states"][cls] = new_state
        session["tested"].add(cls)
        session["evidence"][cls].append({"item_id": probe_id, "answer": answer_s, "p_answer": table[cls][answer_s]})
        changed.append((cls, new_p, new_state))
    session.setdefault("probes_asked", []).append(probe_id)
    try:
        report = _report(session)
    except Exception as exc:
        log.exception("exam report failed")
        return _error(500, f"exam report failed ({type(exc).__name__})")

    def work(conn):
        ensure(conn)
        for cls, new_p, new_state in changed:
            write_knowledge(
                conn, session["learner_id"], cls, new_p, new_state, probe_id,
                f"probe {probe_id}: answered {answer_s}", evidence_type="PROBE",
            )
        save_exam(conn, exam_id, session["learner_id"], session, report)

    try:
        store._run(work)
    except StoreError as exc:
        return _error(exc.status, str(exc))
    except Exception as exc:
        log.exception("exam probe failed")
        return _error(500, f"exam probe failed ({type(exc).__name__})")
    return {"report": report, "model_version": model_version(), "latency_ms": _ms(started)}


@router.get("/exam/{exam_id}/report")
def exam_report(exam_id: str):
    started = time.perf_counter()
    store = get_store()
    try:
        record = store._run(lambda conn: (ensure(conn), load_exam(conn, exam_id))[1])
    except StoreError as exc:
        return _error(exc.status, str(exc))
    report = record.get("report")
    if report is None:
        try:
            report = _report(record["session"])
        except Exception as exc:
            log.exception("exam report failed")
            return _error(500, f"exam report failed ({type(exc).__name__})")
    return {"report": report, "model_version": model_version(), "latency_ms": _ms(started)}
