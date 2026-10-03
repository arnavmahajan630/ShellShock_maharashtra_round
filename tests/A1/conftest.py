import pytest

from ml.c_interp import interp


@pytest.fixture(autouse=True)
def strict_interpreter(monkeypatch):
    """An exception inside the interpreter is a bug: let it fail the test instead of
    being reported as status runtime_error."""
    monkeypatch.setattr(interp, "STRICT", True)
