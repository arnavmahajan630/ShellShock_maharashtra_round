"""Diagnosis routes (package S2): POST /run, /attempt, /probe/answer, /lab/diagnose.

The real work is in `server/app/pipeline.py` (gate, interpreter, features, model, Bayes layer,
decision). This file is the HTTP layer and three fallbacks, tried in this order:

1. The problem is in the ML problem bank (`ml/problems/main`, `ml/problems/dsa`) and a diagnoser
   artifact exists -> the pipeline.
2. The problem is one of the older Arnav-built planet problems (P18-P23, and the bank ids when
   there is no artifact) -> the rule-based path that used to live in this file
   (`server/app/*_rules.py`, which this package does not own).
3. Anything else (the `GATE` demo id, unknown ids) -> the fixture for that id, unchanged.

`POST /probe/answer` also exists in `routes/probe.py` (not owned by S2). Routers load in
alphabetical order and the first registered path wins, so this file answers; probe.py is left
alone and is only reached for nothing. See notes/S2.md.
"""
import json
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse, Response

from ml.c_interp import harness
from server.app import (arrays_rules, conditions_rules, functions_rules, gate as gate_module, loops_rules, pipeline,
                        store as store_module, variables_rules)

log = logging.getLogger("relearn.attempt")
router = APIRouter()

FIXTURES = Path(__file__).resolve().parent.parent.parent / "fixtures"
MAX_ATTEMPTS_KEPT = 1000

# ---------------------------------------------------------------- the older rule-based path

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
    """Public problem from server/fixtures/problems.json. routes/probe.py imports this name."""
    return next((p for p in _problems_catalog() if p["problem_id"] == problem_id), None)


def _fixture_case(filename, by_value):
    data = json.loads((FIXTURES / filename).read_text(encoding="utf-8"))
    return data["cases"].get(str(by_value), data["cases"][data["default"]])


def _legacy_run_and_diagnose(problem, code, rules, want_diagnosis):
    gate_result = gate_module.check(problem, code)
    if gate_result["code"] != "G0":
        return {"gate": gate_result, "trace": None, "tests": None, "diagnosis": None}
    internal = rules.to_internal_problem(problem)
    run_result = harness.run_tests(internal, code)
    trace = harness.trace(internal, code)
    diagnosis = rules.diagnose(problem, code) if want_diagnosis else None
    return {"gate": gate_result, "trace": trace, "tests": run_result["tests"], "diagnosis": diagnosis}


def _legacy(problem_id, code, want_diagnosis):
    """The rule-based answer for `problem_id`, or None when the older path does not know it."""
    rules = RULES_BY_ID.get(problem_id)
    problem = _find_problem(problem_id) if rules else None
    if problem is None:
        return None
    return {**_legacy_run_and_diagnose(problem, code, rules, want_diagnosis), "model_version": "rules-v1"}


# ---------------------------------------------------------------- which path

def _live_problem(problem_id):
    """The bank problem when the pipeline can serve it, else None.

    The ML pipeline's own problem bank (`ml/problems/main/`) numbers its problems P01, P02, ...
    independently of the older planet fixtures (`server/fixtures/problems.json`) — a handful of
    ids collide with a *different* problem under the same id (P13 is "sync_ratio" on the planet
    but "fuel_percent" in the pipeline bank; P14 is "signal_diff" vs "distance"). Trusting the id
    alone there means the gate checks the submission against the wrong signature and every
    attempt fails before diagnosis ever runs. Cross-check the signature against the fixture's
    own declared one before trusting the pipeline's answer for this id.
    """
    problem = pipeline.get_problem(problem_id) if problem_id else None
    if problem is None:
        return None
    fixture_problem = _find_problem(problem_id)
    if fixture_problem is not None and fixture_problem.get("signature") != problem.get("signature"):
        return None
    try:
        pipeline._model()
    except pipeline.ModelUnavailable:
        return None
    return problem


def _plain(value):
    """json.dumps fallback for numpy scalars and arrays."""
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def _json(body):
    """A trace can hold thousands of steps; json.dumps is several times faster than FastAPI's encoder."""
    return Response(content=json.dumps(body, default=_plain, separators=(",", ":")), media_type="application/json")


def _error(status, message):
    return JSONResponse(status_code=status, content={"error": message})


def _failed(where, exc):
    log.exception("%s failed", where)
    return _error(500, f"{where} failed ({type(exc).__name__})")


# ---------------------------------------------------------------- attempts kept for /probe/answer

_attempts = OrderedDict()           # attempt_id -> {"analysis", "answers", "problem_id", "code_item"}
_attempts_lock = threading.Lock()


def _remember(attempt_id, entry):
    with _attempts_lock:
        _attempts[attempt_id] = entry
        _attempts.move_to_end(attempt_id)
        while len(_attempts) > MAX_ATTEMPTS_KEPT:
            _attempts.popitem(last=False)


def _recall(attempt_id):
    with _attempts_lock:
        entry = _attempts.get(attempt_id)
        if entry is not None:
            _attempts.move_to_end(attempt_id)
        return entry


def _new_attempt_id(problem_id):
    return f"at_{problem_id}_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------- saving for a known learner (package K1)

def _known_store(learner_id):
    """The store when `learner_id` is a learner in it, else None. Never creates the database."""
    if not isinstance(learner_id, str) or not learner_id:
        return None
    path = store_module.db_path()
    if path != ":memory:" and not os.path.exists(path):
        return None
    store = store_module.get_store()
    return store if store.exists(learner_id) else None


def _save_model_attempt(learner_id, attempt_id, problem, analysis, diagnosis):
    """Record a pipeline outcome and apply the 03 8.2 code-task update. True when a P(A) moved.

    The learner prior of this diagnosis was read before this write, so an attempt never feeds
    itself. Saving must not break the answer: any failure is logged and the attempt goes out.
    """
    try:
        store = _known_store(learner_id)
        if store is None or analysis.tests is None or diagnosis is None:
            return False
        top = (diagnosis.get("top") or [{}])[0]
        latent = (diagnosis.get("latent") or {}).get("class")
        target = latent if analysis.passed else top.get("id")
        topic = problem.get("planet") or problem.get("sector")
        out = store.apply_attempt_result(
            learner_id, attempt_id, problem["problem_id"], analysis.tests["passed"], analysis.tests["total"],
            diagnosis["status"], top=top.get("id"), top_p=top.get("p"),
            exposure=(problem.get("exposure") or {}).get(target), latent=latent, family=problem.get("family"),
            code=analysis.code, node=f"{topic}:{problem['problem_id']}")
        return bool(out["updates"])
    except Exception:                                   # noqa: BLE001
        log.exception("saving the attempt failed")
        return False


def _save_rule_attempt(learner_id, attempt_id, problem_id, code, legacy):
    """Rule-based problems: the attempt and a cleared node are recorded. No P(A) moves, because
    the finding does not come from the model."""
    try:
        store = _known_store(learner_id)
        tests = legacy.get("tests")
        if store is None or not tests:
            return
        status = (legacy.get("diagnosis") or {}).get("status") or "rules"
        store.record_attempt(learner_id, attempt_id, problem_id, tests["passed"], tests["total"], status, code=code)
        if tests["total"] > 0 and tests["passed"] == tests["total"]:
            planet = (_find_problem(problem_id) or {}).get("planet")
            store.set_node(learner_id, f"{planet}:{problem_id}", "done", 3)
    except Exception:                                   # noqa: BLE001
        log.exception("saving the attempt failed")


# ---------------------------------------------------------------- routes

@router.post("/run")
def run(body: dict = Body(...)):
    problem_id, code = body.get("problem_id"), body.get("code") or ""
    problem = _live_problem(problem_id)
    if problem is None:
        started = time.perf_counter()
        legacy = _legacy(problem_id, code, want_diagnosis=False)
        if legacy is None:
            return _fixture_case("run.json", problem_id)
        return {**legacy, "latency_ms": pipeline._ms(started)}
    try:
        return _json(pipeline.run(problem, code, sample_only=bool(body.get("sample_only"))))
    except Exception as exc:
        return _failed("run", exc)


@router.post("/attempt")
def attempt(body: dict = Body(...)):
    problem_id, code = body.get("problem_id"), body.get("code") or ""
    problem = _live_problem(problem_id)
    if problem is None:
        started = time.perf_counter()
        legacy = _legacy(problem_id, code, want_diagnosis=True)
        if legacy is None:
            return _fixture_case("attempt.json", problem_id)
        attempt_id = _new_attempt_id(problem_id)
        _save_rule_attempt(body.get("learner_id"), attempt_id, problem_id, code, legacy)
        return {"attempt_id": attempt_id, **legacy, "latency_ms": pipeline._ms(started)}

    item = pipeline.get_code_item(body.get("code_item_id"))
    if body.get("code_item_id") and (item is None or item.get("problem_id") != problem_id):
        item = None                                     # unknown or mismatched id: a normal attempt
    try:
        started = time.perf_counter()
        analysis, response = pipeline.attempt(problem, code, learner_id=body.get("learner_id"),
                                              prediction=body.get("prediction"), events=body.get("events"))
        attempt_id = _new_attempt_id(problem_id)
        response = {"attempt_id": attempt_id, **response}
        if item is not None and analysis.gate["code"] == "G0":
            response["code_item"] = pipeline.code_item_rule(item, analysis.code, response["diagnosis"], analysis.passed)
        response["latency_ms"] = pipeline._ms(started)
    except pipeline.PipelineError as exc:
        return _error(422, str(exc))
    except Exception as exc:
        return _failed("attempt", exc)
    saved = _save_model_attempt(body.get("learner_id"), attempt_id, problem, analysis, response["diagnosis"])
    _remember(attempt_id, {"analysis": analysis, "answers": [], "problem_id": problem_id,
                           "learner_id": body.get("learner_id"), "saved": saved})
    return _json(response)


@router.post("/probe/answer")
def probe_answer(body: dict = Body(...)):
    problem_id, code = body.get("problem_id"), body.get("code") or ""
    probe_id, answer = body.get("probe_id"), body.get("answer")
    entry = _recall(body.get("attempt_id"))
    problem_id = entry["problem_id"] if entry else problem_id

    problem = _live_problem(problem_id)
    if problem is None:
        return json.loads((FIXTURES / "probe_answer.json").read_text(encoding="utf-8"))
    if probe_id is None or answer is None:
        return _error(422, "probe_id and answer are required")
    started = time.perf_counter()
    try:
        if entry is not None:
            analysis, earlier = entry["analysis"], list(entry["answers"])
        elif code:                                      # the app also sends problem_id and code: stateless replay
            analysis = pipeline.analyse(problem, code, learner_id=body.get("learner_id"))
            earlier = [{"probe_id": a["probe_id"], "answer": a["answer"]} for a in body.get("probe_answers") or []]
        else:
            return _error(404, "unknown attempt_id; send problem_id and code to answer without one")
        answers = earlier + [{"probe_id": probe_id, "answer": answer}]
        diagnosis = pipeline.diagnose(analysis, answers)
    except pipeline.PipelineError as exc:
        return _error(422, str(exc))
    except Exception as exc:
        return _failed("probe/answer", exc)
    if entry is not None:
        with _attempts_lock:
            entry["answers"] = answers
        if not entry.get("saved"):                      # the probe may be what made the diagnosis confident
            entry["saved"] = _save_model_attempt(entry.get("learner_id"), body.get("attempt_id"), problem, analysis,
                                                 diagnosis)
    return _json({"diagnosis": diagnosis, "model_version": diagnosis["model_version"],
                  "latency_ms": pipeline._ms(started)})


@router.post("/lab/diagnose")
def lab_diagnose(body: dict = Body(...)):
    """Stateless (no learner prior, nothing stored). Probe answers are applied in order."""
    problem_id, code = body.get("problem_id"), body.get("code") or ""
    problem = _live_problem(problem_id)
    if problem is None:
        started = time.perf_counter()
        legacy = _legacy(problem_id, code, want_diagnosis=True)
        if legacy is None:
            return _fixture_case("lab_diagnose.json", problem_id)
        return {**legacy, "latency_ms": pipeline._ms(started)}
    try:
        started = time.perf_counter()
        analysis = pipeline.analyse(problem, code, prediction=body.get("prediction"))
        answers = [{"probe_id": a["probe_id"], "answer": a["answer"]} for a in body.get("probe_answers") or []]
        diagnosis = pipeline.diagnose(analysis, answers) if analysis.gate["code"] == "G0" else None
        return _json({"gate": analysis.gate, "trace": analysis.trace, "tests": analysis.tests, "diagnosis": diagnosis,
                      "model_version": pipeline.model_version(), "latency_ms": pipeline._ms(started)})
    except (pipeline.PipelineError, KeyError, TypeError) as exc:
        return _error(422, f"bad request: {exc}")
    except Exception as exc:
        return _failed("lab/diagnose", exc)


# ---------------------------------------------------------------- warm-up

if not os.environ.get("RELEARN_NO_WARM"):
    threading.Thread(target=pipeline.warm, name="relearn-warm", daemon=True).start()
