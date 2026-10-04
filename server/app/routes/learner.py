"""Learner routes (package S1): POST /learner, POST /learner/{id}/seed, GET /learner/{id},
POST /reassess, POST /intervene, POST /learner/{id}/intervene, POST /learner/{id}/attempt-result.

The state lives in SQLite (`server/app/store.py`). Knowledge updates are D2's
(`ml.learner.knowledge`, `ml.learner.state_machine`); the intervention package is R2's
(`ml.learner.interventions.build_package`). This file is the HTTP layer.

About `POST /intervene`: `routes/intervene.py` (the earlier stateless build) registers the same
path, and `main.py` loads routers alphabetically, so it answers first and the version here is
never reached until that file is removed. The learner-aware version is also served at
`POST /learner/{learner_id}/intervene`, which has no conflict. See notes/S1.md.
"""
import json
import logging
import os
import time
from pathlib import Path

from fastapi import APIRouter, Body, Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from ml.contracts.classes import MISCONCEPTIONS
from server.app import pipeline, store as store_module
from server.app.store import StoreError, get_store

log = logging.getLogger("relearn.learner")
router = APIRouter()

MODEL_VERSION = "store-v1"
FIXTURES = Path(__file__).resolve().parent.parent.parent / "fixtures"


# ---------------------------------------------------------------- helpers

def _ms(started):
    return round((time.perf_counter() - started) * 1000.0, 3)


def _envelope(body, started):
    return {**body, "model_version": MODEL_VERSION, "latency_ms": _ms(started)}


def _error(status, message):
    return JSONResponse(status_code=status, content={"error": message})


def _plain(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def _json(body):
    return Response(content=json.dumps(body, default=_plain, separators=(",", ":")), media_type="application/json")


def _failed(where, exc):
    log.exception("%s failed", where)
    return _error(500, f"{where} failed ({type(exc).__name__})")


def _store_error(exc):
    return _error(exc.status, str(exc))


def _lookup(problem_id):
    return pipeline.get_problem(problem_id) if problem_id else None


def _bank():
    return list(pipeline._problems().values())


# ---------------------------------------------------------------- the learner prior for S2 (pipeline hook)

def _learner_prior(learner_id):
    """pipeline.learner_prior_fn: {class: P(active)} for a learner with history, else None.

    Never creates the database: a lookup against a store that does not exist yet answers None.
    """
    path = store_module.db_path()
    if path != ":memory:" and not os.path.exists(path):
        return None
    return get_store().prior(learner_id)


if getattr(pipeline, "learner_prior_fn", None) is None:
    pipeline.learner_prior_fn = _learner_prior


# ---------------------------------------------------------------- learner

@router.post("/learner")
def create_learner(body: dict = Body(...)):
    started = time.perf_counter()
    callsign = body.get("callsign") if isinstance(body, dict) else None
    if not isinstance(callsign, str) or not callsign.strip():
        return _error(422, "callsign is required")
    try:
        learner_id = get_store().create_learner(callsign.strip()[:40])
    except StoreError as exc:
        return _store_error(exc)
    except Exception as exc:                                    # noqa: BLE001
        return _failed("create learner", exc)
    return _envelope({"learner_id": learner_id}, started)


@router.post("/learner/{learner_id}/seed")
def seed_learner(learner_id: str):
    """Reset `learner_id` to the demo learner. An unknown id is created (so `demo-learner` works)."""
    started = time.perf_counter()
    try:
        learner = get_store().seed(learner_id)
    except StoreError as exc:
        return _store_error(exc)
    except Exception as exc:                                    # noqa: BLE001
        return _failed("seed", exc)
    return _envelope(learner, started)


@router.get("/learner/{learner_id}")
def get_learner(learner_id: str):
    started = time.perf_counter()
    try:
        learner = get_store().get_learner(learner_id)
    except StoreError as exc:
        return _store_error(exc)
    except Exception as exc:                                    # noqa: BLE001
        return _failed("get learner", exc)
    return _envelope(learner, started)


@router.post("/learner/{learner_id}/attempt-result")
def attempt_result(learner_id: str, body: dict = Body(...)):
    """Not in the contract. Records one /attempt outcome and applies the 03 8.2 code-task update.

    Body: {attempt_id, problem_id, passed, total, status, top?, top_p?, latent?, exposure?, code?, node?}.
    `exposure` defaults to the problem's authored e_ik for `top` (or `latent`) when the bank has it.
    """
    started = time.perf_counter()
    for key in ("attempt_id", "problem_id", "status"):
        if not isinstance(body.get(key), str):
            return _error(422, f"{key} is required")
    for key in ("passed", "total"):
        if not isinstance(body.get(key), int) or isinstance(body.get(key), bool):
            return _error(422, f"{key} must be an integer")
    problem = _lookup(body["problem_id"]) or {}
    target = body.get("top") if body["passed"] < body["total"] else body.get("latent")
    exposure = body.get("exposure")
    if exposure is None and target:
        exposure = (problem.get("exposure") or {}).get(target)
    try:
        out = get_store().apply_attempt_result(
            learner_id, body["attempt_id"], body["problem_id"], body["passed"], body["total"], body["status"],
            top=body.get("top"), top_p=body.get("top_p"), exposure=exposure, latent=body.get("latent"),
            family=problem.get("family"), code=body.get("code"), node=body.get("node"))
    except StoreError as exc:
        return _store_error(exc)
    except Exception as exc:                                    # noqa: BLE001
        return _failed("attempt result", exc)
    return _envelope(out, started)


# ---------------------------------------------------------------- reassess

@router.post("/reassess")
def reassess(body: dict = Body(...)):
    """{learner_id, class, item_id, item_type, result} -> {state, p_active, conditions[], resolved_level, next_item}."""
    started = time.perf_counter()
    for key in ("learner_id", "class", "item_id", "item_type"):
        if not isinstance(body.get(key), str) or not body[key]:
            return _error(422, f"{key} is required")
    result = body.get("result")
    if not isinstance(result, dict):
        return _error(422, "result must be an object")
    try:
        out = get_store().reassess(body["learner_id"], body["class"], body["item_id"], body["item_type"], result,
                                   problem_lookup=_lookup, problem_bank=_bank())
    except StoreError as exc:
        return _store_error(exc)
    except Exception as exc:                                    # noqa: BLE001
        return _failed("reassess", exc)
    return _envelope(out, started)


# ---------------------------------------------------------------- intervene

def _find_attempt(store, learner_id, attempt_id):
    """(problem_id, code) of an attempt: this store first, then S2's in-memory attempts."""
    if not attempt_id:
        return None, None
    row = store.get_attempt(attempt_id, learner_id)
    if row and row.get("code") is not None:
        return row["problem_id"], row["code"]
    try:
        from server.app.routes import attempt as attempt_routes
        entry = attempt_routes._recall(attempt_id)
    except Exception:                                           # noqa: BLE001  S2's module may be mid-rewrite
        entry = None
    if entry is not None:
        analysis = entry.get("analysis")
        return entry.get("problem_id"), getattr(analysis, "code", None)
    return (row["problem_id"] if row else None), None


async def _intervene(request, path_learner_id=None):
    started = time.perf_counter()
    try:
        body = await request.json()
    except ValueError:
        return _error(422, "body must be JSON")
    if not isinstance(body, dict):
        return _error(422, "body must be an object")
    cls = body.get("class")
    if not isinstance(cls, str) or not cls:
        return _error(422, "class is required")
    learner_id = path_learner_id or body.get("learner_id")
    modality = body.get("modality")
    problem_id, code = body.get("problem_id"), body.get("code")
    store = get_store()
    tracked = bool(learner_id) and cls in MISCONCEPTIONS
    if learner_id and not store.exists(learner_id):
        return _error(404, f"unknown learner {learner_id}")
    if code is None or not problem_id:
        found_problem, found_code = _find_attempt(store, learner_id, body.get("attempt_id"))
        problem_id = problem_id or found_problem
        code = found_code if code is None else code
    if not problem_id or code is None:
        return _error(422, "send a known attempt_id, or problem_id and code")

    problem = _lookup(problem_id)
    try:
        if problem is None:
            package = await _legacy(request)
            if package is None:
                return _error(404, f"unknown problem {problem_id}")
        else:
            from ml.learner import interventions
            tried = store.tried_modalities(learner_id, cls) if tracked else ()
            package = await run_in_threadpool(interventions.build_package, problem, code, cls, tuple(tried), modality)
    except StoreError as exc:
        return _store_error(exc)
    except Exception as exc:                                    # noqa: BLE001
        if type(exc).__name__ == "BackendUnavailable":
            return _error(503, "interpreter unavailable")
        return _failed("intervene", exc)

    package = dict(package)
    if tracked:
        entry = store.start_intervention(learner_id, cls, package.get("modality"), problem_id,
                                         family=(problem or {}).get("family"))
        package["learner_state"] = {"state": entry["state"], "p_active": entry["p_active"],
                                    "interventions": entry["interventions"]}
    package.setdefault("model_version", MODEL_VERSION)
    package["latency_ms"] = _ms(started)
    return _json(package)


async def _legacy(request):
    """Problems outside the ML bank (P18-P23): the earlier stateless route, if it still loads."""
    try:
        body = await request.json()
        catalog = json.loads((FIXTURES / "problems.json").read_text(encoding="utf-8"))
        if body.get("problem_id") not in {p["problem_id"] for p in catalog}:
            return None                                         # the earlier route would answer with an empty shell
        from server.app.routes import intervene as legacy
    except Exception:                                           # noqa: BLE001
        return None
    return await legacy.intervene(request)


@router.post("/intervene")
async def intervene(request: Request):
    """{learner_id?, attempt_id?, class, modality?, problem_id?, code?} -> intervention package (03 11.2)."""
    return await _intervene(request)


@router.post("/learner/{learner_id}/intervene")
async def intervene_for_learner(learner_id: str, request: Request):
    """Same as POST /intervene with the learner in the path. Always reachable (no other router has it)."""
    return await _intervene(request, learner_id)
