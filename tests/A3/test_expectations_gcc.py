"""Checks this suite's expected values against a real C compiler (the A2 gcc backend).

These tests do not touch the interpreter, so they run before package A1 is merged. They are
what makes the expectations in cases.py more than a reading of the spec.
"""
import pytest

from ml import runner
from tests.A3.cases import CASES
from tests.A3.helpers import make_problem, needs_gcc, passes

pytestmark = [needs_gcc, pytest.mark.gcc_crosscheck]


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_expected_values_are_what_gcc_produces(case):
    """The spec row is in case.spec: the values in cases.py are what gcc returns for the same program."""
    problem = make_problem(case.signature, case.tests)
    result = runner.run_tests(problem, case.code, backend="gcc")
    assert result["status"] == "ok", result
    assert passes(result) == [True] * len(case.tests), [r["got"] for r in result["tests"]["results"]]
    assert not any("unchecked" in r["got"] for r in result["tests"]["results"])


def test_case_ids_are_unique():
    assert len({c.id for c in CASES}) == len(CASES)
