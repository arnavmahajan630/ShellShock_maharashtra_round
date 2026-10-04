import os

# The attempt route starts a warm-up thread when imported; the tests warm explicitly instead.
os.environ.setdefault("RELEARN_NO_WARM", "1")

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Every test gets its own database file. Nothing touches server/data/relearn.db."""
    path = tmp_path / "relearn_test.db"
    monkeypatch.setenv("RELEARN_DB", str(path))
    return path


@pytest.fixture(scope="session")
def full_app():
    from server.app import pipeline
    try:
        pipeline._model()
    except pipeline.ModelUnavailable as exc:
        pytest.skip(f"no diagnoser artifact ({exc}); run `python -m ml.model.train`")
    from server.app.main import app
    return app


@pytest.fixture
def full(full_app):
    """The real app as main.py assembles it."""
    return TestClient(full_app)
