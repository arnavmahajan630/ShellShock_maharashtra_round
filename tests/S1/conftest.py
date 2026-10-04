import os

# The attempt route starts a warm-up thread when imported; S1 does not need the model.
os.environ.setdefault("RELEARN_NO_WARM", "1")

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Every test gets its own database file. Nothing touches server/data/relearn.db."""
    path = tmp_path / "relearn_test.db"
    monkeypatch.setenv("RELEARN_DB", str(path))
    return path


@pytest.fixture(scope="session")
def full_app():
    from server.app.main import app
    return app


@pytest.fixture
def full(full_app):
    """The real app as main.py assembles it (all routers, fixtures for the rest)."""
    return TestClient(full_app)


@pytest.fixture
def client():
    """Only the learner router: what S1 serves, with no other router in the way."""
    from server.app.routes import learner
    app = FastAPI()
    app.include_router(learner.router)
    return TestClient(app)


P03_LE = """int total_energy(int cells[], int n) {
    int total = 0;
    for (int i = 0; i <= n; i++) {
        total += cells[i];
    }
    return total;
}"""
