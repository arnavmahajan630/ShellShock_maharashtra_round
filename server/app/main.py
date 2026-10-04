"""Re:Learn API.

Run from the repo root:
    .venv\\Scripts\\python -m uvicorn server.app.main:app --host 0.0.0.0 --port 8000

Every endpoint in ml_plan/03 §11 and ml_plan/05 §5 is listed in FIXTURE_ROUTES. A module in
server/app/routes/ that defines `router` is loaded automatically and takes over its paths;
any path without a live router answers from server/fixtures/. Nobody needs to edit this file
to add an endpoint: add a routes module.
"""
import importlib
import json
import pkgutil
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from ml.contracts.feature_names import FEATURES
from server.app import routes

FIXTURES = Path(__file__).parent.parent / "fixtures"

# (method, path, fixture file)
FIXTURE_ROUTES = [
    ("GET", "/problems", "problems.json"),
    ("GET", "/problems/{problem_id}", "problem.json"),
    ("POST", "/learner", "learner_create.json"),
    ("POST", "/learner/{learner_id}/seed", "learner.json"),
    ("GET", "/learner/{learner_id}", "learner.json"),
    ("POST", "/run", "run.json"),
    ("POST", "/attempt", "attempt.json"),
    ("POST", "/probe/answer", "probe_answer.json"),
    ("POST", "/intervene", "intervene.json"),
    ("POST", "/reassess", "reassess.json"),
    ("POST", "/lab/diagnose", "lab_diagnose.json"),
    ("GET", "/metrics", "metrics.json"),
    ("POST", "/exam/start", "exam_start.json"),
    ("POST", "/exam/answer", "exam_answer.json"),
    ("POST", "/exam/finish", "exam_report.json"),
    ("POST", "/exam/probe", "exam_probe.json"),
    ("GET", "/exam/{exam_id}/report", "exam_report.json"),
    ("GET", "/quiz/next", "quiz_next.json"),
    ("POST", "/quiz/answer", "quiz_answer.json"),
    ("GET", "/code-items", "code_items.json"),
    ("POST", "/code-items/debug", "code_items_debug.json"),
    ("POST", "/reason", "reason.json"),
    ("POST", "/lab/reason", "lab_reason.json"),
]

app = FastAPI(title="Re:Learn")
# The plan says localhost only; the phone reaches the laptop by its network address, so all origins are allowed.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


_ROUTER_ROUTES = set()        # (method, path) of every route that came from a module in routes/


def _live_routes():
    direct = {(method, route.path) for route in app.routes for method in getattr(route, "methods", None) or []}
    return direct | _ROUTER_ROUTES


def _load_routers():
    loaded = []
    for module_info in pkgutil.iter_modules(routes.__path__):
        module = importlib.import_module(f"{routes.__name__}.{module_info.name}")
        if hasattr(module, "router"):
            app.include_router(module.router)
            loaded.append(module_info.name)
            # Read the paths from the router itself: newer FastAPI wraps an included router,
            # so its routes no longer show up one by one in app.routes.
            _ROUTER_ROUTES.update((method, route.path) for route in module.router.routes
                                  for method in getattr(route, "methods", None) or [])
    return loaded


def _fixture_handler(filename):
    async def handler(request: Request):
        fixture = json.loads((FIXTURES / filename).read_text(encoding="utf-8"))
        if not (isinstance(fixture, dict) and "_by" in fixture):
            return fixture
        # The response depends on one request field, looked up in the path, the query, then the body.
        field = fixture["_by"]
        value = request.path_params.get(field, request.query_params.get(field))
        if value is None and request.method == "POST":
            try:
                value = (await request.json()).get(field)
            except (ValueError, AttributeError):
                value = None
        return fixture["cases"].get(str(value), fixture["cases"][fixture["default"]])
    return handler


LIVE_ROUTERS = _load_routers()
FIXTURE_PATHS = []
for _method, _path, _file in FIXTURE_ROUTES:
    if (_method, _path) not in _live_routes():
        app.add_api_route(_path, _fixture_handler(_file), methods=[_method], tags=["fixture"])
        FIXTURE_PATHS.append(f"{_method} {_path}")


if ("GET", "/health") not in _live_routes():
    @app.get("/health")
    def health():
        return {"ok": True, "model_version": "fixtures", "latency_ms": 0.0, "n_features": len(FEATURES), "db": "none"}


@app.get("/_routes", tags=["dev"])
def which_routes_are_live():
    """Development helper: which endpoints are real and which still answer from fixtures."""
    return {"live_routers": LIVE_ROUTERS, "fixture_routes": FIXTURE_PATHS}
