"""String tests: char arrays, string literals, strlen (03 §2.2 rows 16, 17, 21; §2.5 v3 extras 1, 2, 6).

Kept in one file and marked `strings` so they can be dropped if strings are cut (06 §2):
    python -m pytest tests/A3 -m "not strings"        or        RELEARN_SKIP_STRINGS=1
"""
import pytest

from ml import runner
from ml.contracts.subset import GARBAGE
from tests.A3.cases import STRING_CASES
from tests.A3.helpers import (close, covers, effects, events, irun, itrace, make_problem, needs_gcc,
                              needs_interp, passes, skip_if_strings_cut, src)

pytestmark = [pytest.mark.strings, skip_if_strings_cut]


# ---------------------------------------------------------------- expected values, checked by gcc

@needs_gcc
@pytest.mark.gcc_crosscheck
@pytest.mark.parametrize("case", STRING_CASES, ids=[c.id for c in STRING_CASES])
def test_expected_string_values_are_what_gcc_produces(case):
    """The spec row is in case.spec: the values in cases.py are what gcc returns for the same program."""
    result = runner.run_tests(make_problem(case.signature, case.tests, sector="strings"), case.code, backend="gcc")
    assert result["status"] == "ok", result
    assert passes(result) == [True] * len(case.tests), [r["got"] for r in result["tests"]["results"]]


# ---------------------------------------------------------------- the interpreter

@needs_interp
@pytest.mark.parametrize("case", STRING_CASES, ids=[c.id for c in STRING_CASES])
def test_interpreter_matches_c_on_strings(case):
    """03 §2.1 strings (the row is in case.spec): same result as gcc."""
    problem = make_problem(case.signature, case.tests, sector="strings")
    result = irun(problem, case.code)
    assert result["status"] == "ok", f"{case.spec}: {result}"
    assert passes(result) == [True] * len(case.tests), f"{case.spec}: {result['tests']['results']}"
    trace = itrace(problem, case.code)
    for index, (_args, expect) in enumerate(case.tests):
        if "returned" in expect:
            assert close(trace["per_test"][index]["returned"], expect["returned"]), (case.spec, index)
        if "printed" in expect:
            assert trace["per_test"][index]["printed"] == expect["printed"], (case.spec, index)
    for kind in ("uninit_read", "oob_read", "oob_write", "str_literal_compare"):
        assert events(trace, kind) == [], (case.spec, kind)


@needs_interp
@covers("2.2/16", "2.5/x1")
def test_row16_string_literal_init_and_terminator():
    """03 §2.2 row 16 / §2.5 v3 extra 1: `char s[] = "level";` is the char codes plus a 0 terminator:
    six cells, so s[5] is 0 and in bounds, and s[6] is out of bounds."""
    code = src("""
        int f(int k) {
            char s[] = "level";
            return s[k];
        }""")
    tests = [([0], {"returned": 108}), ([4], {"returned": 108}), ([5], {"returned": 0}), ([6], {})]
    trace = itrace(make_problem("int f(int k)", tests, sector="strings"), code)
    assert [pt["returned"] for pt in trace["per_test"]] == [108, 108, 0, GARBAGE]     # gcc: string_literal_cells_are_char_codes in cases.py
    (event,) = events(trace, "oob_read")
    assert (event["arr"], event["idx"], event["size"], event["test"]) == ("s", 6, 6, 3)


@needs_interp
@covers("2.2/16")
def test_row16_scanning_to_the_terminator():
    """03 §2.2 row 16: a loop that scans to '\\0' stops after the last character."""
    code = src("""
        int f(int n) {
            char s[] = "level";
            int i = 0;
            while (s[i] != '\\0') {
                i++;
            }
            return i + n;
        }""")
    trace = itrace(make_problem("int f(int n)", [([0], {"returned": 5})], sector="strings"), code)
    assert trace["returned"] == 5 and trace["loop_iters"] == {"L4": 5}                # gcc agrees: cases.py
    assert events(trace, "oob_read") == []


@needs_interp
@covers("2.2/17", "2.5/x2")
def test_row17_char_cell_compared_with_string_literal():
    """03 §2.2 row 17 / §2.5 v3 extra 2: `s[i] == "a"` is false (address comparison, as real C would);
    event str_literal_compare. So a vowel counter written with "a" counts nothing (D08)."""
    code = src("""
        int count_a(char s[]) {
            int c = 0;
            for (int i = 0; s[i] != '\\0'; i++) {
                if (s[i] == "a") {
                    c++;
                }
            }
            return c;
        }""")
    problem = make_problem("int count_a(char s[])", [(["banana"], {"returned": 3}), (["xyz"], {"returned": 0})], sector="strings")
    trace = itrace(problem, code)
    assert trace["status"] == "ok" and [pt["returned"] for pt in trace["per_test"]] == [0, 0]
    found = events(trace, "str_literal_compare")
    assert found and {e["line"] for e in found} == {4}
    assert {e["test"] for e in found} == {0, 1}
    assert passes(irun(problem, code)) == [False, True]                              # passes by luck on "xyz"


@needs_interp
@covers("2.2/17")
def test_row17_not_equal_to_a_string_literal_is_true():
    """03 §2.2 row 17: `!=` with a string literal is true; trap item D08: `if (c == "a") fire(); else open_door();`."""
    code = src("""
        int f(int n) {
            char c = 'a';
            int r = 0;
            if (c == "a") fire(); else open_door();
            if (c != "a") r = 1;
            return r;
        }""")
    trace = itrace(make_problem("int f(int n)", [([0], {})], sector="strings"), code)
    assert trace["returned"] == 1
    assert trace["effects_count"]["fire"] == 0 and trace["effects_count"].get("door_open") == 1
    assert [e["line"] for e in events(trace, "str_literal_compare")] == [4, 5]


@needs_interp
@covers("2.2/17")
def test_row17_char_literal_compare_is_ordinary():
    """03 §2.2 row 17: only string literals are special; `s[i] == 'a'` compares char codes, no event."""
    code = src("""
        int count_a(char s[]) {
            int c = 0;
            for (int i = 0; s[i] != '\\0'; i++) {
                if (s[i] == 'a') {
                    c++;
                }
            }
            return c;
        }""")
    trace = itrace(make_problem("int count_a(char s[])", [(["banana"], {"returned": 3})], sector="strings"), code)
    assert trace["returned"] == 3 and events(trace, "str_literal_compare") == []


@needs_interp
@covers("2.2/21", "2.5/x6")
def test_row21_strlen_is_not_executed_when_forbidden():
    """03 §2.2 row 21 / §2.5 v3 extra 6: in a problem whose `forbid` lists strlen, a call to strlen is
    not executed (the gate reports G4b). Here: status unsupported, nothing passes."""
    code = src("""
        int str_length(char s[]) {
            return strlen(s);
        }""")
    tests = [(["level"], {"returned": 5}), ([""], {"returned": 0})]
    forbidden = make_problem("int str_length(char s[])", tests, forbid=["strlen"], sector="strings")
    result = irun(forbidden, code)
    assert result["status"] == "unsupported" and result["tests"]["passed"] == 0
    assert itrace(forbidden, code)["status"] == "unsupported"


@needs_interp
@covers("2.2/21")
def test_row21_strlen_works_when_not_forbidden():
    """03 §2.1: strlen is a builtin unless the problem forbids it; a hand-written loop is always allowed."""
    tests = [(["level"], {"returned": 5}), ([""], {"returned": 0})]
    allowed = make_problem("int str_length(char s[])", tests, sector="strings")
    assert passes(irun(allowed, "int str_length(char s[]) {\n    return strlen(s);\n}")) == [True, True]     # gcc agrees: cases.py
    forbidden = make_problem("int str_length(char s[])", tests, forbid=["strlen"], sector="strings")
    own = "int str_length(char s[]) {\n    int i = 0;\n    while (s[i] != '\\0') {\n        i++;\n    }\n    return i;\n}"
    result = irun(forbidden, own)
    assert result["status"] == "ok" and passes(result) == [True, True]


@needs_interp
def test_string_argument_reads_are_read_cell_effects():
    """03 §2.4: SignalTiles replays read_cell on char[] params; the terminator cell is cell len(s)."""
    code = "int str_length(char s[]) {\n    int i = 0;\n    while (s[i] != '\\0') {\n        i++;\n    }\n    return i;\n}"
    trace = itrace(make_problem("int str_length(char s[])", [(["abc"], {"returned": 3})], sector="strings"), code)
    assert [e for e in effects(trace) if e.startswith("read_")] == ["read_cell:0", "read_cell:1", "read_cell:2", "read_cell:3"]
    assert trace["effects_count"]["read_void"] == 0


@needs_interp
def test_string_argument_has_exactly_one_terminator_cell():
    """03 §2.3: a Python str arg becomes char[] with a terminator: reading one past it is out of bounds."""
    code = "int f(char s[], int k) {\n    return s[k];\n}"
    tests = [(["ab", 0], {}), (["ab", 2], {}), (["ab", 3], {})]
    trace = itrace(make_problem("int f(char s[], int k)", tests, sector="strings"), code)
    assert [pt["returned"] for pt in trace["per_test"]] == [97, 0, GARBAGE]
    (event,) = events(trace, "oob_read")
    assert (event["idx"], event["size"], event["test"]) == (3, 3, 2)
