"""Structural rules for the 18 DSA problems. Execution is `python -m ml.problems.check dsa`."""
from ml.problems.check import load_problems, structure_errors


def test_dsa_structure():
    errors = structure_errors("dsa", load_problems("dsa"))
    assert errors == []
