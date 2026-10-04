"""Star chart route (package K1): GET /learner/{learner_id}/graph.

The chart itself is `ml.learner.concept_graph.build_view` (the map of C in
`ml/data/concept_graph.json`, laid under the learner's stored attempts, nodes and knowledge).
This file is the HTTP layer. Nothing is written here: the chart is rebuilt from the store on
every call, so it is as persistent as the store is.
"""
import logging
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ml.learner import concept_graph
from server.app.store import StoreError, get_store

log = logging.getLogger("relearn.graph")
router = APIRouter()


def _error(status, message):
    return JSONResponse(status_code=status, content={"error": message})


@router.get("/learner/{learner_id}/graph")
def learner_graph(learner_id: str):
    started = time.perf_counter()
    try:
        learner = get_store().get_learner(learner_id)
    except StoreError as exc:
        return _error(exc.status, str(exc))
    try:
        graph = concept_graph.load_graph()
        view = concept_graph.build_view(graph, concept_graph.load_problems(), learner)
    except Exception as exc:                                    # noqa: BLE001
        log.exception("graph failed")
        return _error(500, f"graph failed ({type(exc).__name__})")
    return {**view, "model_version": graph.get("version", "chart-v1"),
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3)}
