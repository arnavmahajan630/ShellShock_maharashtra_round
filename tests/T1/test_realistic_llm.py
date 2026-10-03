"""Checks the reference solutions and the run check used for the LLM-written test programs.

No DeepSeek calls. Run from the repo root:  .venv\\Scripts\\python -m pytest tests/T1
"""
import pytest

from ml import runner
from ml.contracts.classes import MISCONCEPTIONS
from ml.text import gen_realistic_llm as G

pytestmark = pytest.mark.skipif(not runner.available("interp"), reason="needs the interpreter")

# Outputs worked out by hand for each reference on its inputs (floats checked separately).
KNOWN = {
    "P01": [0, 1, 3, 5], "P03": [12, 5, 10, 9, 8], "P05": [1, 3, 0, 7, 1], "P06": [8, 1, 3, 1024, 49],
    "P08": [9, 4, 5, 3], "P10": [3, 18, 9, 0], "P11": [1, 0, 0, 0, 0], "P12": [0, 0, 1, 1, 1, 2, 2, 0],
    "P14": [5, 5, 0, 6, 9], "Q01": [1, 0, 3, -1, 0, 2], "Q02": [3, 1, 0, 2, 1, 1],
    "Q03": [3, 0, 4, -1, 1, 0, 1, -1, -1], "Q08": [1, 0, 0, 1, 1, 0, 0], "Q09": [4, 1, 8, 4, 15, -5],
    "Q12": [1, 1, 1, 0, 0, 1, 1, 0], "Q14": [2, 0, 5, 3, 0], "Q15": [1, 0, 1, 0, 1, 1, 0],
    "Q16": [0, 5, 7, 6, 18, 1], "Q17": [1, 1, 6, 120, 3628800], "Q18": [6, 5, 16, 9, 11, 0],
    "Q06": [[1, 2, 3], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 4, 5, 8], [7]],
    "Q07": [[1, 2, 3], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 4, 5, 8], [7]],
    "Q10": [[3, 2, 1], [4, 3, 2, 1], [5], [2, 1], [5, 6, 7, 8, 9]], "Q11": [[2, 3, 1], [6, 7, 8, 5], [4], [2, 1]],
}
KNOWN_FLOAT = {"P07": [1.5, 11 / 3, 10.0, 2.5, 5.0], "P13": [75.0, 100 / 3, 100.0, 0.0, 87.5]}


def test_every_planned_problem_has_a_reference():
    assert set(G.PROBLEMS) == set(G.REFERENCE)
    assert set(G.CLASS_PROBLEMS) == set(MISCONCEPTIONS)
    assert {i["problem_id"] for i in G.plan()} <= set(G.REFERENCE)


@pytest.mark.parametrize("problem_id", sorted(G.REFERENCE))
def test_reference_gives_the_hand_worked_outputs(problem_id):
    status, outputs = G.run_outputs(problem_id, G.REFERENCE[problem_id][0])
    assert status == "ok" and all(s == "ok" for s, _ in outputs)
    values = [v for _, v in outputs]
    if problem_id in KNOWN_FLOAT:
        assert all(abs(a - b) < 1e-3 for a, b in zip(values, KNOWN_FLOAT[problem_id]))
    else:
        assert values == KNOWN[problem_id]


def test_run_check_tells_wrong_from_right():
    harmless = ("void reverse(int a[], int n) { int i = 0; int j = n - 1; int t; "
                "while (i <= j) { t = a[i]; a[i] = a[j]; a[j] = t; i = i + 1; j = j - 1; } }")
    assert G.run_check("misconception", "Q10", harmless)[0] == "wrong"         # correct code offered as a bug
    assert G.run_check("correct_unusual", "Q10", harmless) is None
    twice = G.REFERENCE["Q10"][0].replace("i < n / 2", "i < n")
    assert G.run_check("misconception", "Q10", twice) is None                   # reverses twice: really wrong
    assert G.run_check("correct_unusual", "Q10", twice)[0] == "correct"
    stuck = "int charge_steps(int level, int target) { int s = 0; while (level < target) { s++; } return s; }"
    assert G.run_check("misconception", "P05", stuck) is None                   # never finishes: counts as wrong
    assert G.run_check("other", "P03", "def total_energy(cells, n): return sum(cells)")[0] == "broken"
    assert G.run_check("junk", "P03", "def total_energy(cells, n): return sum(cells)") is None
