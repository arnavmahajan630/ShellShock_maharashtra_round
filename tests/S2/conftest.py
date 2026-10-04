import os

# The route module starts a warm-up thread when it is imported; the tests warm explicitly instead.
os.environ.setdefault("RELEARN_NO_WARM", "1")

import pytest
from fastapi.testclient import TestClient

from server.app import pipeline


@pytest.fixture(scope="session")
def client():
    try:
        pipeline._model()
    except pipeline.ModelUnavailable as exc:
        pytest.skip(f"no diagnoser artifact ({exc}); run `python -m ml.model.train`")
    from server.app.main import app
    pipeline.warm()
    return TestClient(app)


P03_LE = """int total_energy(int cells[], int n) {
    int total = 0;
    for (int i = 0; i <= n; i++) {
        total += cells[i];
    }
    return total;
}"""

P03_OK = """int total_energy(int cells[], int n) {
    int total = 0;
    for (int i = 0; i < n; i++) {
        total += cells[i];
    }
    return total;
}"""

P11_SEMI = """int door_open(int code) {
    int open = 0;
    if (code == 42);
        open = 1;
    return open;
}"""
