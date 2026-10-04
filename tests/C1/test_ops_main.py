"""C1: every main-game operator yields a mutant that parses and fails a test.

Run from the repo root:  .venv\\Scripts\\python -m pytest tests/C1 -q
"""
import fnmatch
import json
from pathlib import Path

import pytest

from ml.c_interp.harness import check
from ml.generate.ops_main import all_ops, apply, sites
from ml.generate.predicates_main import holds
from ml.runner import run_tests

BANK = Path(__file__).resolve().parents[2] / "ml" / "problems" / "main"

# Operators the catalogue describes but no main-bank correct variant contains.
# The snippets are the same shape as the "Applies when" column.
SNIPPETS = {
    "m04_int_literal": (
        "float half(int n) {\n    return n / 2.0;\n}",
        {
            "problem_id": "SNIP_m04",
            "signature": "float half(int n)",
            "forbid": [],
            "tests": [
                {"args": [3], "expect": {"returned": 1.5}},
                {"args": [4], "expect": {"returned": 2.0}},
            ],
        },
    ),
    "m06_while_assign": (
        "int steps_until(int level, int target) {\n"
        "    int steps = 0;\n"
        "    while (level != target) {\n"
        "        level = level + 1;\n"
        "        steps++;\n"
        "    }\n"
        "    return steps;\n}",
        {
            "problem_id": "SNIP_m06",
            "signature": "int steps_until(int level, int target)",
            "forbid": [],
            "tests": [
                {"args": [1, 4], "expect": {"returned": 3}},
                {"args": [0, 2], "expect": {"returned": 2}},
            ],
        },
    ),
    "m02_update_in_branch": (
        "int count_hot(int t[], int n, int limit) {\n"
        "    int count = 0;\n"
        "    int i = 0;\n"
        "    while (i < n) {\n"
        "        if (t[i] > limit) {\n"
        "            count++;\n"
        "        }\n"
        "        i++;\n"
        "    }\n"
        "    return count;\n}",
        {
            "problem_id": "SNIP_m02",
            "signature": "int count_hot(int t[], int n, int limit)",
            "forbid": [],
            "tests": [
                {"args": [[30, 90], 2, 50], "expect": {"returned": 1}},
                {"args": [[10, 20], 2, 50], "expect": {"returned": 0}},
            ],
        },
    ),
}

EXPECTED = [
    "m01_le", "m01_nminus1", "m01_start1_count", "m01_countdown_ge0", "m01_while_le",
    "m08_index_n", "m08_onebased", "m08_iplus1", "m08_first1",
    "amb_le_array", "amb_start1_array",
    "m02_drop_update", "m02_reverse", "m02_update_in_branch", "m02_wrong_var",
    "m03_decl_in_loop", "m03_assign_in_loop",
    "m04_drop_cast", "m04_int_literal", "m04_cast_late",
    "m05_drop_init_acc", "m05_drop_init_counter", "m05_drop_init_max",
    "m06_if_assign", "m06_while_assign",
    "m07_if_semi", "m07_for_semi", "m07_while_semi",
    "m10_printf_noreturn", "m10_printf_return0",
    "oth_swap_operands", "oth_wrong_op", "oth_wrong_const", "oth_wrong_var",
    "oth_flip_branch_rel", "oth_drop_stmt",
    "u1_chained", "u2_or_chain",
]

DSA_SURFACE = {
    "amb_pair_bound", "m08_mirror", "m01_half_bound", "m02_pointer_stuck", "m10_print_bool",
}


def _problems():
    found = []
    for path in sorted(BANK.glob("*.json")):
        found.append(json.loads(path.read_text(encoding="utf-8")))
    return found


PROBLEMS = _problems()


def _allowed(op, problem):
    return any(fnmatch.fnmatch(op.op_id, pattern) for pattern in problem["allowed_ops"])


def _parses(code, problem):
    return check(code, problem)["status"] == "ok"


def _fails(problem, code):
    result = run_tests(problem, code, backend="interp")
    tests = result["tests"]
    return tests["passed"] < tests["total"], result


def _exercise(op, code, problem):
    """First site whose mutant parses and fails a test, or None."""
    found = sites(op, code)
    if not found:
        return None
    if not _parses(code, problem):
        return None
    original_fails, _ = _fails(problem, code)
    if original_fails:
        return None
    for site in found:
        mutant = apply(op, code, site)
        if not _parses(mutant, problem):
            continue
        failed, result = _fails(problem, mutant)
        if failed:
            return {"mutant": mutant, "problem": problem, "result": result, "site": site}
    return None


def _bank_hit(op):
    for problem in PROBLEMS:
        if not _allowed(op, problem):
            continue
        for code in problem["correct_variants"]:
            hit = _exercise(op, code, problem)
            if hit is not None:
                hit["source"] = problem["problem_id"]
                return hit
    return None


def _snippet_hit(op):
    if op.op_id not in SNIPPETS:
        return None
    code, problem = SNIPPETS[op.op_id]
    hit = _exercise(op, code, problem)
    if hit is not None:
        hit["source"] = "snippet"
    return hit


def test_catalogue_is_complete():
    ops = all_ops()
    assert [op.op_id for op in ops] == EXPECTED
    assert DSA_SURFACE.isdisjoint(op.op_id for op in ops)
    assert {op.op_id for op in ops if op.test_only} == {"u1_chained", "u2_or_chain"}
    labels = {op.op_id: op.labels for op in ops}
    assert labels["amb_le_array"] == ("M01", "M08")
    assert labels["amb_start1_array"] == ("M01", "M08")
    assert labels["m01_le"] == ("M01",)
    assert labels["oth_drop_stmt"] == ("OTHER",)
    assert all(op.test_only is False for op in ops if op.op_id not in ("u1_chained", "u2_or_chain"))


@pytest.mark.parametrize("op", [op for op in all_ops() if not op.test_only], ids=lambda op: op.op_id)
def test_operator_fails_a_test(op):
    hit = _bank_hit(op)
    if hit is None:
        hit = _snippet_hit(op)
    assert hit is not None, f"{op.op_id} had no parsing mutant that fails a test"
    mutant = hit["mutant"]
    if op.op_id.startswith("oth_"):
        assert holds(op.op_id, mutant) is False
    else:
        assert holds(op.op_id, mutant) is True, mutant


def test_u1_chained_edit_parses():
    op = next(op for op in all_ops() if op.op_id == "u1_chained")
    code = "int in_range(int x, int lo, int hi) {\n    if (lo < x && x < hi) {\n        return 1;\n    }\n    return 0;\n}"
    assert op.test_only and op.labels == ("U1",)
    assert sites(op, code)
    mutant = apply(op, code, 0)
    assert "&&" not in mutant
    assert check(mutant)["status"] == "ok"
    assert holds("u1_chained", mutant) is True
    assert holds("m06_if_assign", mutant) is False
    assert holds("m07_if_semi", mutant) is False


def test_u2_or_chain_edit_parses():
    op = next(op for op in all_ops() if op.op_id == "u2_or_chain")
    code = "int pick(int c) {\n    if (c == 42 || c == 7) {\n        return 1;\n    }\n    return 0;\n}"
    assert op.test_only and op.labels == ("U2",)
    assert sites(op, code)
    mutant = apply(op, code, 0)
    assert "c == 7" not in mutant
    assert check(mutant)["status"] == "ok"
    assert holds("u2_or_chain", mutant) is True
    assert holds("m06_if_assign", mutant) is False


def test_sites_empty_when_pattern_is_absent():
    op = next(op for op in all_ops() if op.op_id == "m08_index_n")
    code = "int fire_shots(int n) {\n    int i = 0;\n    while (i < n) {\n        i++;\n    }\n    return i;\n}"
    assert sites(op, code) == []


def test_sites_do_not_leak():
    op = next(op for op in all_ops() if op.op_id == "m01_nminus1")
    code = PROBLEMS[0]["correct_variants"][0]  # P01 for-loop, one `<`
    assert sites(op, code)
    once = apply(op, code, 0)
    again = apply(op, code, 0)
    assert once == again
    assert "n - 1" in once or "n-1" in once.replace(" ", "")
