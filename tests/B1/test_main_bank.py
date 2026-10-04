"""Structural rules for the 16 main problems. Execution is `python -m ml.problems.check main`."""
from ml.problems.check import load_problems, structure_errors


def test_main_structure():
    errors = structure_errors("main", load_problems("main"))
    assert errors == []
