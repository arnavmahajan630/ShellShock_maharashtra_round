"""The supported subset of 03 §2.1, run through the interpreter.

Each program is in cases.py with the values gcc produces for it (test_expectations_gcc.py
checks that), so a failure here means the interpreter and real C disagree on a program whose
behaviour C defines. Also: what the interpreter does with code outside the subset (03 §2.1
"Rejected", §2.5 preprocessing).
"""
import pytest

from tests.A3.cases import CASES
from tests.A3.helpers import (WORLD_EFFECTS, close, events, irun, itrace, make_problem, needs_interp,
                              passes, src)

pytestmark = needs_interp


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_interpreter_matches_c(case):
    """03 §2.1 (the row is in case.spec and in the failure message): same result as gcc,
    through run_tests (pass/fail) and through trace (per-test values)."""
    problem = make_problem(case.signature, case.tests)
    result = irun(problem, case.code)
    assert result["status"] == "ok", f"{case.spec}: {result}"
    assert passes(result) == [True] * len(case.tests), f"{case.spec}: {result['tests']['results']}"
    assert result["tests"]["passed"] == result["tests"]["total"] == len(case.tests)

    trace = itrace(problem, case.code)
    assert trace["status"] == "ok" and len(trace["per_test"]) == len(case.tests)
    for index, (_args, expect) in enumerate(case.tests):
        per_test = trace["per_test"][index]
        assert per_test["status"] == "ok"
        if "returned" in expect:
            assert close(per_test["returned"], expect["returned"]), (case.spec, index, per_test["returned"])
        if "printed" in expect:
            assert per_test["printed"] == expect["printed"], (case.spec, index)
        for name in WORLD_EFFECTS:
            if name in expect:
                assert per_test["effects_count"].get(name, 0) == expect[name], (case.spec, index, name)
        if "max_depth_le" in expect:
            assert per_test["max_depth"] <= expect["max_depth_le"], (case.spec, index)
    # 03 §2.5: these programs have no undefined behaviour, so none of these events may appear.
    for kind in ("uninit_read", "oob_read", "oob_write", "overflow", "div_zero", "step_cap_hit", "depth_cap_hit"):
        assert events(trace, kind) == [], (case.spec, kind)


def test_expect_keys_decide_pass_and_fail():
    """03 §2.3: a test passes only if every key in expect matches (returned, printed, effect counts)."""
    code = src("""
        int f(int n) {
            fire();
            printf("%d", n);
            return n + 1;
        }""")
    tests = [([1], {"returned": 2, "printed": "1", "fire": 1}),
             ([1], {"returned": 3, "printed": "1", "fire": 1}),
             ([1], {"returned": 2, "printed": "one", "fire": 1}),
             ([1], {"returned": 2, "printed": "1", "fire": 2}),
             ([1], {"returned": 2})]
    assert passes(irun(make_problem("int f(int n)", tests), code)) == [True, False, False, False, True]


def test_float_expectations_use_the_tolerance():
    """03 §2.3: floats compare within ±1e-3 (subset.FLOAT_TOLERANCE)."""
    code = src("""
        float f(int a, int b) {
            return (float) a / b;
        }""")
    tests = [([1, 3], {"returned": 0.3333}), ([1, 3], {"returned": 0.34}), ([2, 1], {"returned": 2}),
             ([1, 3], {"returned": 0.33})]
    assert passes(irun(make_problem("float f(int a, int b)", tests), code)) == [True, False, True, False]


def test_max_depth_le_expectation():
    """03 §2.3: expect.max_depth_le is an upper bound on the recursion depth of that test."""
    code = src("""
        int sum_to(int n) {
            if (n == 0) return 0;
            return n + sum_to(n - 1);
        }""")
    tests = [([5], {"returned": 15, "max_depth_le": 6}), ([5], {"returned": 15, "max_depth_le": 5})]
    assert passes(irun(make_problem("int sum_to(int n)", tests, sector="recursion"), code)) == [True, False]


def test_world_builtin_effects_in_steps():
    """03 §2.1 world builtins + subset.EFFECT_FIELDS: fire, launch, door_open, door_closed, scan:<x>."""
    code = src("""
        void f(int n) {
            open_door();
            fire();
            scan(n);
            scan(n + 1);
            close_door();
            launch();
        }""")
    trace = itrace(make_problem("void f(int n)", [([4], {})]), code)
    world = [e for step in trace["steps"] for e in step["effects"] if not e.startswith(("call", "ret"))]
    assert world == ["door_open", "fire", "scan:4", "scan:5", "door_closed", "launch"]
    counts = trace["effects_count"]
    assert (counts["fire"], counts["launch"], counts["door_open"], counts["door_closed"], counts["scan"]) == (1, 1, 1, 1, 2)


# ---------------------------------------------------------------- 03 §2.5 preprocessing

def test_comments_keep_line_numbers():
    """03 §2.5: comments become spaces and the newlines stay, so events keep the learner's line numbers."""
    code = src("""
        // first line is a comment
        int f(int a, int b) {
            /* a block comment
               over three lines
               that mentions a / 0 */
            int q = a / b;   // intdiv on line 6
            return q;
        }""")
    trace = itrace(make_problem("int f(int a, int b)", [([7, 2], {"returned": 3})]), code)
    assert trace["returned"] == 3
    (event,) = events(trace, "intdiv")
    assert event["line"] == 6


def test_smart_quotes_are_normalised():
    """03 §2.5 / gate G8: smart quotes are turned into plain quotes before parsing."""
    code = "int f(int n) {\n    char c = ‘a’;\n    printf(“%d!”, n);\n    return c;\n}"
    trace = itrace(make_problem("int f(int n)", [([7], {"returned": 97, "printed": "7!"})]), code)
    assert trace["status"] == "ok" and trace["printed"] == "7!" and trace["returned"] == 97


def test_other_preprocessor_lines_are_rejected():
    """03 §2.5: `#` lines other than #include and #define NAME literal are rejected, not run."""
    code = "#ifdef DEBUG\n#endif\nint f(int n) {\n    return n;\n}"
    result = irun(make_problem("int f(int n)", [([1], {"returned": 1})]), code)
    assert result["status"] in ("unsupported", "parse_error") and result["tests"]["passed"] == 0


def test_syntax_error_is_parse_error():
    """03 §2.4: a file that does not parse has status parse_error, from run_tests and from trace."""
    code = "int f(int n) {\n    int t = n +;\n    return t;\n}"
    problem = make_problem("int f(int n)", [([1], {"returned": 1})])
    result = irun(problem, code)
    assert result["status"] == "parse_error" and result["tests"]["passed"] == 0
    assert itrace(problem, code)["status"] == "parse_error"


# ---------------------------------------------------------------- 03 §2.1 rejected constructs

REJECTED = {
    "pointer": "int f(int n) {\n    int *p;\n    return n;\n}",
    "address_of": "void set(int *p) {\n    *p = 1;\n}\nint f(int n) {\n    set(&n);\n    return n;\n}",
    "struct": "struct point { int x; int y; };\nint f(int n) {\n    struct point p;\n    p.x = n;\n    return p.x;\n}",
    "malloc": "int f(int n) {\n    int *a = malloc(n * sizeof(int));\n    return n;\n}",
    "string_h": "int f(int n) {\n    char a[] = \"x\";\n    char b[] = \"x\";\n    return strcmp(a, b) + n;\n}",
    "scanf": "int f(int n) {\n    scanf(\"%d\", &n);\n    return n;\n}",
    "goto": "int f(int n) {\n    goto end;\nend:\n    return n;\n}",
    "multi_dim_array": "int f(int n) {\n    int g[2][2] = {{1, 2}, {3, 4}};\n    return g[1][1] + n;\n}",
    "switch": "int f(int n) {\n    switch (n) {\n    case 1:\n        return 10;\n    default:\n        return 0;\n    }\n}",
}


@pytest.mark.parametrize("construct", sorted(REJECTED))
def test_rejected_constructs_are_unsupported(construct):
    """03 §2.1 "Rejected" / subset.REJECTED: the interpreter does not run these; status unsupported."""
    problem = make_problem("int f(int n)", [([1], {"returned": 1})])
    result = irun(problem, REJECTED[construct])
    assert result["status"] == "unsupported", construct
    assert result["tests"]["passed"] == 0
    assert itrace(problem, REJECTED[construct])["status"] == "unsupported"


def test_rejected_table_matches_the_contract():
    from ml.contracts.subset import REJECTED as CONTRACT
    assert set(REJECTED) == set(CONTRACT)
