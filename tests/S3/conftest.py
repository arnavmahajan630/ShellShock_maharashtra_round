import os

os.environ.setdefault("RELEARN_NO_WARM", "1")

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Each test gets its own database. Nothing writes server/data/relearn.db."""
    path = tmp_path / "relearn_s3.db"
    monkeypatch.setenv("RELEARN_DB", str(path))
    return path


@pytest.fixture(scope="session")
def client():
    from server.app.main import app
    return TestClient(app)
