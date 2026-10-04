"""Performance budget of ml_plan/03 §2.5: a problem's whole test suite in under 30 ms."""
import time

import pytest

from ml.c_interp import harness
from tests.fixtures import build

from .util import load_problem

BUDGET_MS = 30.0
PROBLEMS = [load_problem("P03_total_energy"), load_problem("P11_door_open"), load_problem("Q17_factorial")]


def best_ms(fn, repeats=7):
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - start) * 1000)
    return best


@pytest.mark.parametrize("prob", PROBLEMS, ids=lambda p: p["problem_id"])
def test_suite_runs_inside_the_budget(prob):
    for variant in prob["correct_variants"]:
        def cold():
            harness._cache.clear()                      # includes parsing
            harness.run_tests(prob, variant)
        assert best_ms(cold) < BUDGET_MS
        assert best_ms(lambda: harness.run_tests(prob, variant)) < BUDGET_MS
        assert best_ms(lambda: harness.trace(prob, variant)) < BUDGET_MS


def test_worst_case_suite_every_test_hits_the_step_cap():
    prob = PROBLEMS[0]
    harness.run_tests(prob, build.P03_NO_UPDATE)
    assert best_ms(lambda: harness.run_tests(prob, build.P03_NO_UPDATE)) < 150      # 5 x 5,000 statements
    assert best_ms(lambda: harness.trace(prob, build.P03_NO_UPDATE)) < 150


def test_sort_of_eight_and_recursion_of_twelve():
    sort = ("void bubble_sort(int a[], int n) {\n    for (int i = 0; i < n - 1; i++) {\n"
            "        for (int j = 0; j < n - 1 - i; j++) {\n            if (a[j] > a[j + 1]) {\n"
            "                int t = a[j];\n                a[j] = a[j + 1];\n                a[j + 1] = t;\n"
            "            }\n        }\n    }\n}")
    sort_prob = {"name": "bubble_sort", "signature": "void bubble_sort(int a[], int n)", "display_test": 0,
                 "tests": [{"args": [[8, 7, 6, 5, 4, 3, 2, 1], 8], "expect": {"array0": [1, 2, 3, 4, 5, 6, 7, 8]}}] * 6}
    assert harness.run_tests(sort_prob, sort)["tests"]["passed"] == 6
    assert best_ms(lambda: harness.run_tests(sort_prob, sort)) < BUDGET_MS
    fib = "int fib(int n) {\n    if (n < 2) {\n        return n;\n    }\n    return fib(n - 1) + fib(n - 2);\n}"
    fib_prob = {"name": "fib", "signature": "int fib(int n)", "display_test": 0,
                "tests": [{"args": [12], "expect": {"returned": 144}}] * 5}
    assert harness.run_tests(fib_prob, fib)["tests"]["passed"] == 5
    assert best_ms(lambda: harness.run_tests(fib_prob, fib)) < BUDGET_MS
