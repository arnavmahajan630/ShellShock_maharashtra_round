"""Every C2 operator edits a real solution into a program that fails a test.

The site comes from ``ml/problems/dsa`` (a problem whose ``allowed_ops`` match).
``d08_array_eq`` has no such site in the bank; a snippet of the same shape covers it.
"""
import fnmatch

from pycparser import c_parser

from ml.c_interp.preprocess import preprocess
from ml.generate.ops_dsa import all_ops, apply, sites
from ml.generate.predicates_dsa import holds
from ml.problems.check import load_problems
from ml.runner import run_tests

_DSA = load_problems("dsa")

# Two char arrays compared cell by cell. No Q-problem has this shape (Q15 compares
# one string with itself), so d08_array_eq is checked on this snippet instead.
_ARRAY_EQ = r"""
int same_text(char s[], char t[]) {
    int i = 0;
    while (s[i] != '\0') {
        if (s[i] != t[i]) return 0;
        i++;
    }
    return 1;
}
"""
_ARRAY_EQ_PROBLEM = {
    "signature": "int same_text(char s[], char t[])",
    "forbid": [],
    "tests": [
        {"args": ["ab", "ab"], "expect": {"returned": 1}},
        {"args": ["ab", "cd"], "expect": {"returned": 0}},
    ],
}


def _allowed(problem, op_id):
    return any(fnmatch.fnmatch(op_id, pattern) for pattern in problem["allowed_ops"])


def _odd_n(problem):
    for test in problem["tests"]:
        for arg in test["args"]:
            if isinstance(arg, int) and arg % 2 == 1:
                return True
    return False


def _parses(code):
    c_parser.CParser().parse(preprocess(code))


def _fails(problem, code):
    result = run_tests(problem, code, backend="interp")
    assert result["status"] != "parse_error", code
    return any(not row["pass"] for row in result["tests"]["results"])


def _bank_hit(op):
    """First allowed variant whose mutant parses, fails a test, and satisfies ``holds``."""
    for problem in _DSA:
        if not _allowed(problem, op.op_id):
            continue
        if op.op_id == "d05_unreachable_base" and not _odd_n(problem):
            continue
        for variant in problem["correct_variants"]:
            found = sites(op, variant)
            if not found:
                continue
            for site in found:
                mutant = apply(op, variant, site)
                _parses(mutant)
                if not _fails(problem, mutant):
                    continue
                if not holds(op.op_id, mutant) or holds(op.op_id, variant):
                    continue
                return problem["problem_id"], site
    return None


def test_catalogue_shape():
    ops = all_ops()
    ids = [op.op_id for op in ops]
    assert ids[-2:] == ["d08_str_literal", "d08_array_eq"]
    assert len(ids) == len(set(ids)) == 20
    assert all(op.test_only is False for op in ops)
    amb = next(op for op in ops if op.op_id == "amb_pair_bound")
    assert amb.labels == ("M01", "M08")
    assert next(op for op in ops if op.op_id == "d03_drop_temp").labels == ("D03",)


def test_holds_is_false_on_correct_variants():
    bad = []
    for op in all_ops():
        for problem in _DSA:
            for index, variant in enumerate(problem["correct_variants"]):
                if holds(op.op_id, variant):
                    bad.append(f"{op.op_id} on {problem['problem_id']} cv{index + 1}")
    assert bad == []


def test_every_op_fails_a_test():
    missed = []
    for op in all_ops():
        if op.op_id == "d08_array_eq":
            continue
        if _bank_hit(op) is None:
            missed.append(op.op_id)
    assert missed == []


def test_d08_array_eq_snippet_when_bank_has_no_site():
    op = next(op for op in all_ops() if op.op_id == "d08_array_eq")
    assert _bank_hit(op) is None
    for problem in _DSA:
        if not _allowed(problem, op.op_id):
            continue
        for variant in problem["correct_variants"]:
            assert sites(op, variant) == []
    found = sites(op, _ARRAY_EQ)
    assert found
    mutant = apply(op, _ARRAY_EQ, found[0])
    _parses(mutant)
    assert _fails(_ARRAY_EQ_PROBLEM, mutant)
    assert holds(op.op_id, mutant)
    assert not holds(op.op_id, _ARRAY_EQ)
    assert "s == t" in mutant or "s==t" in mutant.replace(" ", "")


def test_apply_reparses_so_sites_do_not_leak():
    op = next(item for item in all_ops() if item.op_id == "d05_drop_base")
    variant = next(problem for problem in _DSA if problem["problem_id"] == "Q17")["correct_variants"][0]
    assert "factorial" in variant
    first = apply(op, variant, 0)
    second = apply(op, variant, 0)
    assert first == second
    assert sites(op, variant) == sites(op, variant)
