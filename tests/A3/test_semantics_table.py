"""One test (or more) per row of the table in 03 §2.2, plus the §2.5 "v3 extra unit tests".

String rows (16, 17 with a literal, 21 and extras 1, 2, 6) are in test_strings.py.
Where C defines the returned value, the same program is also in cases.py, where gcc checks it.
"""
import pytest

from ml.contracts.subset import DEPTH_CAP, GARBAGE, MAX_RECORDED_STEPS
from tests.A3.helpers import (covers, effects, events, irun, itrace, make_problem, needs_interp,
                              passes, src)

pytestmark = needs_interp

INT_MAX = 2147483647
INT_MIN = -2147483648


# ---------------------------------------------------------------- row 1

@covers("2.2/01")
def test_row01_uninit_read_returns_garbage():
    """03 §2.2 row 1: read of an uninitialised local returns GARBAGE; event uninit_read {var}."""
    code = src("""
        int f(int n) {
            int x;
            return x;
        }""")
    trace = itrace(make_problem("int f(int n)", [([1], {})]), code)
    assert trace["status"] == "ok" and trace["returned"] == GARBAGE == -858993460
    (event,) = events(trace, "uninit_read")
    assert event["var"] == "x" and event["line"] == 3 and event["test"] == 0


@covers("2.2/01")
def test_row01_uninit_accumulator():
    """03 §2.2 row 1 (M05 shape): `int total;` then `total += cells[i]` starts from GARBAGE."""
    code = src("""
        int total_energy(int cells[], int n) {
            int total;
            for (int i = 0; i < n; i++) {
                total += cells[i];
            }
            return total;
        }""")
    trace = itrace(make_problem("int total_energy(int cells[], int n)", [([[2, 4, 6], 3], {"returned": 12})]), code)
    assert trace["returned"] == GARBAGE + 12
    found = events(trace, "uninit_read")
    assert found and found[0]["var"] == "total" and found[0]["line"] == 4


@covers("2.2/01")
def test_row01_no_event_when_initialised():
    """03 §2.2 row 1: no uninit_read once the variable has a value (declared, then assigned, then read)."""
    code = src("""
        int f(int n) {
            int x;
            x = n;
            return x;
        }""")
    trace = itrace(make_problem("int f(int n)", [([4], {})]), code)
    assert trace["returned"] == 4 and events(trace, "uninit_read") == []


# ---------------------------------------------------------------- row 2

@covers("2.2/02")
def test_row02_oob_read_of_array_param():
    """03 §2.2 row 2: array read out of bounds returns GARBAGE; event oob_read {arr, idx, size}."""
    code = src("""
        int f(int a[], int n) {
            return a[n];
        }""")
    trace = itrace(make_problem("int f(int a[], int n)", [([[2, 4, 6], 3], {})]), code)
    assert trace["status"] == "ok" and trace["returned"] == GARBAGE
    (event,) = events(trace, "oob_read")
    assert (event["arr"], event["idx"], event["size"], event["line"]) == ("a", 3, 3, 2)


@covers("2.2/02")
def test_row02_oob_read_negative_index():
    """03 §2.2 row 2: a negative index is out of bounds too."""
    code = src("""
        int f(int a[], int n) {
            return a[n - 4];
        }""")
    trace = itrace(make_problem("int f(int a[], int n)", [([[2, 4, 6], 3], {})]), code)
    assert trace["returned"] == GARBAGE
    (event,) = events(trace, "oob_read")
    assert (event["idx"], event["size"]) == (-1, 3)


@covers("2.2/02")
def test_row02_oob_read_of_local_array():
    """03 §2.2 row 2: also for a local array; an in-bounds read gives no event."""
    code = src("""
        int f(int k) {
            int b[2] = {1, 2};
            return b[k];
        }""")
    trace = itrace(make_problem("int f(int k)", [([5], {}), ([1], {})]), code)
    assert [pt["returned"] for pt in trace["per_test"]] == [GARBAGE, 2]
    (event,) = events(trace, "oob_read")
    assert (event["arr"], event["idx"], event["size"], event["line"], event["test"]) == ("b", 5, 2, 3, 0)


@covers("2.2/02")
def test_row02_off_by_one_sum_matches_the_fixture_numbers():
    """03 §2.2 row 2 (T1 shape): `i <= n` on P03 returns GARBAGE + 12 with one oob_read per test."""
    code = src("""
        int total_energy(int cells[], int n) {
            int total = 0;
            for (int i = 0; i <= n; i++) {
                total += cells[i];
            }
            return total;
        }""")
    tests = [([[2, 4, 6], 3], {"returned": 12}), ([[5], 1], {"returned": 5})]
    trace = itrace(make_problem("int total_energy(int cells[], int n)", tests), code)
    assert [pt["returned"] for pt in trace["per_test"]] == [GARBAGE + 12, GARBAGE + 5]
    assert [(e["idx"], e["size"], e["test"]) for e in events(trace, "oob_read")] == [(3, 3, 0), (1, 1, 1)]
    assert passes(irun(make_problem("int total_energy(int cells[], int n)", tests), code)) == [False, False]


# ---------------------------------------------------------------- row 3

@covers("2.2/03")
def test_row03_oob_write_is_ignored():
    """03 §2.2 row 3: array write out of bounds is ignored; event oob_write {arr, idx, size}."""
    code = src("""
        void f(int a[], int n) {
            a[n] = 99;
            a[0] = 5;
        }""")
    problem = make_problem("void f(int a[], int n)", [([[1, 2, 3], 3], {"array0": [5, 2, 3]})])
    trace = itrace(problem, code)
    assert trace["status"] == "ok"
    (event,) = events(trace, "oob_write")
    assert (event["arr"], event["idx"], event["size"], event["line"]) == ("a", 3, 3, 2)
    assert passes(irun(problem, code)) == [True]            # the array still has exactly 3 cells: [5, 2, 3]
    assert "write_cell:0:5" in effects(trace)
    assert trace["effects_count"]["write_cell"] == 1        # the ignored write is not a write_cell


@covers("2.2/03")
def test_row03_oob_write_to_local_array():
    """03 §2.2 row 3: also for a local array; execution continues."""
    code = src("""
        int f(int n) {
            int b[2] = {1, 2};
            b[2] = 7;
            b[-1] = 7;
            return b[0] + b[1];
        }""")
    trace = itrace(make_problem("int f(int n)", [([0], {})]), code)
    assert trace["status"] == "ok" and trace["returned"] == 3
    assert [(e["arr"], e["idx"], e["size"], e["line"]) for e in events(trace, "oob_write")] == [("b", 2, 2, 3), ("b", -1, 2, 4)]


# ---------------------------------------------------------------- row 4

@covers("2.2/04")
def test_row04_intdiv_truncates_toward_zero_with_event():
    """03 §2.2 row 4: int / int truncates toward zero; event intdiv {remainder_nonzero, into_float}."""
    code = src("""
        int f(int a, int b) {
            int q = a / b;
            return q;
        }""")
    tests = [([7, 2], {"returned": 3}), ([6, 3], {"returned": 2}), ([-7, 2], {"returned": -3}), ([7, -2], {"returned": -3})]
    trace = itrace(make_problem("int f(int a, int b)", tests), code)
    assert [pt["returned"] for pt in trace["per_test"]] == [3, 2, -3, -3]         # gcc agrees: cases.py
    found = events(trace, "intdiv")
    assert [e["test"] for e in found] == [0, 1, 2, 3]
    assert [e["remainder_nonzero"] for e in found] == [True, False, True, True]
    assert [e["into_float"] for e in found] == [False] * 4
    assert {e["line"] for e in found} == {2}


@covers("2.2/04")
@pytest.mark.parametrize("body", [
    "float r = a / b;\n    return r;",
    "float r;\n    r = a / b;\n    return r;",
    "return a / b;",
], ids=["float_decl", "float_assign", "float_return"])
def test_row04_intdiv_into_float(body):
    """03 §2.2 row 4: into_float is true when the int / int result is assigned to / returned as float."""
    code = "float f(int a, int b) {\n    " + body + "\n}"
    trace = itrace(make_problem("float f(int a, int b)", [([7, 2], {"returned": 3.0})]), code)
    assert trace["returned"] == pytest.approx(3.0)                                # gcc agrees: cases.py
    (event,) = events(trace, "intdiv")
    assert event["into_float"] is True and event["remainder_nonzero"] is True


@covers("2.2/04")
def test_row04_no_intdiv_event_for_float_division():
    """03 §2.2 row 4: the event is for int / int only; (float) a / b is ordinary division."""
    code = src("""
        float f(int a, int b) {
            return (float) a / b;
        }""")
    trace = itrace(make_problem("float f(int a, int b)", [([7, 2], {"returned": 3.5})]), code)
    assert trace["returned"] == pytest.approx(3.5) and events(trace, "intdiv") == []


# ---------------------------------------------------------------- row 5

@covers("2.2/05")
@pytest.mark.parametrize("op", ["/", "%"])
def test_row05_division_by_zero_halts(op):
    """03 §2.2 row 5: division by zero halts with status runtime_error; event div_zero.
    (gcc: both `/` and `%` by zero trap, see tests/A2.)"""
    code = "int f(int a, int b) {\n    int q = 1;\n    q = a " + op + " b;\n    fire();\n    return q;\n}"
    problem = make_problem("int f(int a, int b)", [([7, 0], {"returned": 0})])
    trace = itrace(problem, code)
    assert trace["status"] == "runtime_error"
    assert trace["per_test"][0]["status"] == "runtime_error"
    (event,) = events(trace, "div_zero")
    assert event["line"] == 3
    assert trace["effects_count"]["fire"] == 0              # halted: the statement after it never ran
    result = irun(problem, code)
    assert result["status"] == "runtime_error" and result["tests"]["passed"] == 0


@covers("2.2/05")
def test_row05_division_by_zero_in_one_test_only():
    """03 §2.2 row 5: the halt ends that test; the other tests still run and are counted."""
    code = src("""
        int f(int a, int b) {
            return a / b;
        }""")
    problem = make_problem("int f(int a, int b)", [([6, 3], {"returned": 2}), ([6, 0], {"returned": 0}), ([9, 3], {"returned": 3})])
    trace = itrace(problem, code)
    assert [pt["status"] for pt in trace["per_test"]] == ["ok", "runtime_error", "ok"]
    assert [e["test"] for e in events(trace, "div_zero")] == [1]
    assert passes(irun(problem, code)) == [True, False, True]


# ---------------------------------------------------------------- row 6

@covers("2.2/06")
def test_row06_assignment_in_condition():
    """03 §2.2 row 6: `if (code = 42)` evaluates normally (assigns, then tests the value);
    event assign_in_cond {var, value}."""
    code = src("""
        int door_open(int code) {
            if (code = 42) {
                return 1;
            }
            return 0;
        }""")
    tests = [([42], {"returned": 1}), ([7], {"returned": 0}), ([0], {"returned": 0})]
    trace = itrace(make_problem("int door_open(int code)", tests), code)
    assert [pt["returned"] for pt in trace["per_test"]] == [1, 1, 1]              # gcc agrees: cases.py
    found = events(trace, "assign_in_cond")
    assert [(e["var"], e["value"], e["line"], e["test"]) for e in found] == [("code", 42, 2, 0), ("code", 42, 2, 1), ("code", 42, 2, 2)]
    assert passes(irun(make_problem("int door_open(int code)", tests), code)) == [True, False, False]


@covers("2.2/06")
def test_row06_assignment_of_zero_takes_the_else_branch():
    """03 §2.2 row 6: `if (x = 0)` is false, so the else branch runs (trap item M06)."""
    code = src("""
        int f(int x) {
            if (x = 0) {
                fire();
            } else {
                open_door();
            }
            return x;
        }""")
    trace = itrace(make_problem("int f(int x)", [([5], {})]), code)
    assert trace["returned"] == 0                                                 # gcc agrees: cases.py
    assert trace["effects_count"]["fire"] == 0 and trace["effects_count"].get("door_open") == 1
    (event,) = events(trace, "assign_in_cond")
    assert (event["var"], event["value"]) == ("x", 0)


@covers("2.2/06")
def test_row06_no_event_for_a_comparison():
    """03 §2.2 row 6: `==` in a condition gives no assign_in_cond."""
    code = src("""
        int door_open(int code) {
            if (code == 42) {
                return 1;
            }
            return 0;
        }""")
    trace = itrace(make_problem("int door_open(int code)", [([42], {}), ([7], {})]), code)
    assert events(trace, "assign_in_cond") == []


# ---------------------------------------------------------------- row 7

@covers("2.2/07")
def test_row07_empty_if_body():
    """03 §2.2 row 7: `if (...);` executes nothing, the next statement always runs; empty_body {kind: if}."""
    code = src("""
        void f(int shield) {
            if (shield < 0);
            fire();
        }""")
    problem = make_problem("void f(int shield)", [([10], {"fire": 1}), ([-10], {"fire": 1})])
    trace = itrace(problem, code)
    assert [pt["effects_count"].get("fire", 0) for pt in trace["per_test"]] == [1, 1]   # gcc agrees: cases.py
    found = events(trace, "empty_body")
    assert found and {(e["kind"], e["line"]) for e in found} == {("if", 2)}
    assert passes(irun(problem, code)) == [True, True]


@covers("2.2/07")
def test_row07_empty_for_body():
    """03 §2.2 row 7: `for (...);` loops over nothing; empty_body {kind: for}."""
    code = src("""
        int f(int n) {
            int i;
            for (i = 0; i < n; i++);
            fire();
            return i;
        }""")
    trace = itrace(make_problem("int f(int n)", [([3], {})]), code)
    assert trace["returned"] == 3 and trace["effects_count"]["fire"] == 1         # gcc agrees: cases.py
    found = events(trace, "empty_body")
    assert found and {(e["kind"], e["line"]) for e in found} == {("for", 3)}
    assert trace["loop_iters"]["L3"] == 3


@covers("2.2/07")
def test_row07_empty_while_body():
    """03 §2.2 row 7: `while (...);`; empty_body {kind: while}."""
    code = src("""
        int f(int n) {
            int i = 0;
            while (i++ < n);
            return i;
        }""")
    trace = itrace(make_problem("int f(int n)", [([3], {})]), code)
    assert trace["returned"] == 4                                                 # gcc agrees: cases.py
    found = events(trace, "empty_body")
    assert found and {(e["kind"], e["line"]) for e in found} == {("while", 3)}


@covers("2.2/07")
def test_row07_empty_while_body_never_ends():
    """03 §2.2 row 7 + row 10 (M07/M02 twin): `while (i < n);` spins until the step cap."""
    code = src("""
        int f(int n) {
            int i = 0;
            while (i < n);
            {
                i++;
            }
            return i;
        }""")
    trace = itrace(make_problem("int f(int n)", [([3], {})]), code)
    assert trace["status"] == "timeout"
    assert events(trace, "empty_body") and events(trace, "step_cap_hit")


# ---------------------------------------------------------------- row 8

@covers("2.2/08")
def test_row08_loop_iteration_counters():
    """03 §2.2 row 8: each loop iteration adds 1 to loop_iters["L<line>"], for for / while / do-while."""
    code = src("""
        int f(int n) {
            int c = 0;
            for (int i = 0; i < n; i++) {
                c++;
            }
            int j = 0;
            while (j < n) {
                j++;
            }
            do {
                c++;
            } while (c < 0);
            return c;
        }""")
    trace = itrace(make_problem("int f(int n)", [([3], {}), ([0], {})]), code)
    first, second = trace["per_test"]
    assert first["loop_iters"] == {"L3": 3, "L7": 3, "L10": 1}
    assert {k: v for k, v in second["loop_iters"].items() if v} == {"L10": 1}
    assert {k: v for k, v in trace["loop_iters"].items() if v} == {"L3": 3, "L7": 3, "L10": 2}     # sums over tests


@covers("2.2/08")
def test_row08_nested_loops_count_separately():
    """03 §2.2 row 8: the inner loop's counter adds up over every pass of the outer loop."""
    code = src("""
        int f(int n) {
            int c = 0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < 2; j++) {
                    c++;
                }
            }
            return c;
        }""")
    trace = itrace(make_problem("int f(int n)", [([3], {})]), code)
    assert trace["loop_iters"] == {"L3": 3, "L4": 6} and trace["returned"] == 6


# ---------------------------------------------------------------- row 9

@covers("2.2/09")
def test_row09_branch_counters():
    """03 §2.2 row 9: each `if` counts into branch["B<line>"] = {true: n, false: m}."""
    code = src("""
        int f(int a[], int n, int limit) {
            int c = 0;
            for (int i = 0; i < n; i++) {
                if (a[i] > limit) {
                    c++;
                } else if (a[i] == limit) {
                    c += 10;
                }
            }
            return c;
        }""")
    trace = itrace(make_problem("int f(int a[], int n, int limit)", [([[1, 5, 9, 3], 4, 3], {}), ([[7], 1, 0], {})]), code)
    first, second = trace["per_test"]
    assert first["returned"] == 12
    assert (first["branch"]["B4"].get("true", 0), first["branch"]["B4"].get("false", 0)) == (2, 2)
    assert (first["branch"]["B6"].get("true", 0), first["branch"]["B6"].get("false", 0)) == (1, 1)
    assert (second["branch"]["B4"].get("true", 0), second["branch"]["B4"].get("false", 0)) == (1, 0)
    assert second["branch"].get("B6", {}).get("true", 0) == 0 and second["branch"].get("B6", {}).get("false", 0) == 0
    assert trace["branch"]["B4"] == {"true": 3, "false": 2}                       # sums over tests
    assert trace["branch"]["B6"] == {"true": 1, "false": 1}


# ---------------------------------------------------------------- row 10

@covers("2.2/10")
def test_row10_step_cap():
    """03 §2.2 row 10: more than 5,000 executed statements halts with status timeout; event step_cap_hit.
    Recorded steps stop at 2,000 with truncated = true (03 §2.4)."""
    code = src("""
        int total_energy(int cells[], int n) {
            int total = 0;
            int i = 0;
            while (i < n) {
                total += cells[i];
            }
            return total;
        }""")
    problem = make_problem("int total_energy(int cells[], int n)", [([[2, 4, 6], 3], {"returned": 12}), ([[5], 1], {"returned": 5})])
    trace = itrace(problem, code)
    assert trace["status"] == "timeout"
    assert [pt["status"] for pt in trace["per_test"]] == ["timeout", "timeout"]
    assert [(e["test"], e["line"] in (4, 5)) for e in events(trace, "step_cap_hit")] == [(0, True), (1, True)]
    assert trace["truncated"] is True and len(trace["steps"]) == MAX_RECORDED_STEPS == 2000
    assert 1000 <= trace["per_test"][0]["loop_iters"]["L4"] <= 5000
    result = irun(problem, code)
    assert result["status"] == "timeout" and result["tests"]["passed"] == 0


@covers("2.2/10")
def test_row10_long_but_finite_loop_is_not_a_timeout():
    """03 §2.2 row 10: the cap is per test, and a loop of a few thousand statements is fine.
    Four tests of about 2,100 statements each: more than 5,000 in total, under 5,000 each."""
    code = src("""
        int f(int n) {
            int c = 0;
            for (int i = 0; i < n; i++) {
                c += 2;
            }
            return c;
        }""")
    problem = make_problem("int f(int n)", [([700], {"returned": 1400})] * 4)
    trace = itrace(problem, code)
    assert trace["status"] == "ok" and events(trace, "step_cap_hit") == []
    assert [pt["returned"] for pt in trace["per_test"]] == [1400] * 4
    assert trace["truncated"] is True and len(trace["steps"]) == MAX_RECORDED_STEPS     # 2,100 steps > 2,000 kept
    assert passes(irun(problem, code)) == [True] * 4


@covers("2.2/10")
def test_row10_finite_loop_over_the_cap_times_out():
    """03 §2.2 row 10: a loop that would end, but only after more than 5,000 statements, is cut."""
    code = src("""
        int f(int n) {
            int c = 0;
            for (int i = 0; i < n; i++) {
                c += 1;
            }
            return c;
        }""")
    trace = itrace(make_problem("int f(int n)", [([6000], {"returned": 6000})]), code)
    assert trace["status"] == "timeout" and len(events(trace, "step_cap_hit")) == 1


# ---------------------------------------------------------------- row 11

@covers("2.2/11")
def test_row11_missing_return():
    """03 §2.2 row 11: a non-void function that falls off the end returns GARBAGE; event missing_return."""
    code = src("""
        int f(int n) {
            int t = n * 2;
            printf("%d", t);
        }""")
    trace = itrace(make_problem("int f(int n)", [([4], {"returned": 8})]), code)
    assert trace["status"] == "ok" and trace["returned"] == GARBAGE and trace["printed"] == "8"
    assert len(events(trace, "missing_return")) == 1


@covers("2.2/11")
def test_row11_missing_return_on_one_path_only():
    """03 §2.2 row 11: only the path that falls off the end gets the event."""
    code = src("""
        int f(int n) {
            if (n > 0) {
                return 1;
            }
        }""")
    trace = itrace(make_problem("int f(int n)", [([5], {}), ([0], {})]), code)
    assert [pt["returned"] for pt in trace["per_test"]] == [1, GARBAGE]
    assert [e["test"] for e in events(trace, "missing_return")] == [1]


@covers("2.2/11")
def test_row11_void_function_may_fall_off_the_end():
    """03 §2.2 row 11: the row is about non-void functions; a void function ending is normal."""
    code = src("""
        void f(int n) {
            fire();
        }""")
    trace = itrace(make_problem("void f(int n)", [([1], {"fire": 1})]), code)
    assert trace["status"] == "ok" and events(trace, "missing_return") == []


# ---------------------------------------------------------------- row 12

@covers("2.2/12")
@pytest.mark.parametrize("expr,arg,expected", [
    ("n * 2", 2000000000, -294967296),
    ("n + 1", INT_MAX, INT_MIN),
    ("n - 1", INT_MIN, INT_MAX),
    ("n * n", 65536, 0),
], ids=["mul", "add", "sub", "square"])
def test_row12_int_overflow_wraps(expr, arg, expected):
    """03 §2.2 row 12: int overflow wraps to 32 bits; event overflow.
    (Not compared with gcc: signed overflow is undefined in C, 03 §2.5.)"""
    code = "int f(int n) {\n    int x = " + expr + ";\n    return x;\n}"
    trace = itrace(make_problem("int f(int n)", [([arg], {})]), code)
    assert trace["status"] == "ok" and trace["returned"] == expected
    found = events(trace, "overflow")
    assert found and found[0]["line"] == 2


@covers("2.2/12")
def test_row12_factorial_13_wraps():
    """03 §2.2 row 12: 13! does not fit in 32 bits: 6227020800 wraps to 1932053504."""
    code = src("""
        int factorial(int n) {
            if (n <= 1) return 1;
            return n * factorial(n - 1);
        }""")
    trace = itrace(make_problem("int factorial(int n)", [([13], {}), ([12], {})], sector="recursion"), code)
    assert [pt["returned"] for pt in trace["per_test"]] == [1932053504, 479001600]
    assert {e["test"] for e in events(trace, "overflow")} == {0}


@covers("2.2/12")
def test_row12_no_event_inside_the_range():
    """03 §2.2 row 12: INT_MAX itself is not an overflow."""
    code = src("""
        int f(int n) {
            return n + 1;
        }""")
    trace = itrace(make_problem("int f(int n)", [([INT_MAX - 1], {})]), code)
    assert trace["returned"] == INT_MAX and events(trace, "overflow") == []


# ---------------------------------------------------------------- row 13

@covers("2.2/13")
def test_row13_read_cell_effects():
    """03 §2.2 row 13: each read of an array parameter is a world effect read_cell {i}."""
    code = src("""
        int total_energy(int cells[], int n) {
            int total = 0;
            for (int i = 0; i < n; i++) {
                total += cells[i];
            }
            return total;
        }""")
    tests = [([[2, 4, 6], 3], {"returned": 12}), ([[5], 1], {"returned": 5}), ([[1, 2, 3, 4], 4], {"returned": 10})]
    trace = itrace(make_problem("int total_energy(int cells[], int n)", tests), code)
    assert [e for e in effects(trace) if e.startswith("read_")] == ["read_cell:0", "read_cell:1", "read_cell:2"]
    assert [pt["effects_count"]["read_cell"] for pt in trace["per_test"]] == [3, 1, 4]
    assert trace["effects_count"]["read_cell"] == 8 and trace["effects_count"]["read_void"] == 0
    reading = [s for s in trace["steps"] if "read_cell:1" in s["effects"]]
    assert len(reading) == 1 and reading[0]["line"] == 4


@covers("2.2/13")
def test_row13_read_void_when_out_of_bounds():
    """03 §2.2 row 13: an out-of-bounds read of an array parameter is read_void {i} (and oob_read)."""
    code = src("""
        int total_energy(int cells[], int n) {
            int total = 0;
            for (int i = 0; i <= n; i++) {
                total += cells[i];
            }
            return total;
        }""")
    trace = itrace(make_problem("int total_energy(int cells[], int n)", [([[2, 4, 6], 3], {"returned": 12})]), code)
    assert [e for e in effects(trace) if e.startswith("read_")] == ["read_cell:0", "read_cell:1", "read_cell:2", "read_void:3"]
    assert trace["effects_count"]["read_cell"] == 3 and trace["effects_count"]["read_void"] == 1
    void_step = [s for s in trace["steps"] if "read_void:3" in s["effects"]][0]
    assert [e["type"] for e in void_step["events"]] == ["oob_read"] and void_step["line"] == 4


# ---------------------------------------------------------------- row 14

@covers("2.2/14")
def test_row14_array_param_write():
    """03 §2.2 row 14: a write to an array parameter is performed and is a world effect write_cell {i, v}."""
    code = src("""
        void f(int a[], int n) {
            a[0] = 7;
            a[n - 1] = a[0] + 1;
        }""")
    problem = make_problem("void f(int a[], int n)", [([[1, 2, 3], 3], {"array0": [7, 2, 8]})])
    trace = itrace(problem, code)
    assert [e for e in effects(trace) if e.startswith("write_cell")] == ["write_cell:0:7", "write_cell:2:8"]
    assert trace["effects_count"]["write_cell"] == 2
    assert passes(irun(problem, code)) == [True]                                  # gcc agrees: tests/A2


@covers("2.2/14", "2.5/x3")
def test_row14_in_place_sort_is_visible_to_the_caller():
    """03 §2.2 row 14 / §2.5 v3 extra 3: an in-place sort changes the caller's array (expect.array0)."""
    good = src("""
        void bubble_sort(int a[], int n) {
            for (int i = 0; i < n - 1; i++) {
                for (int j = 0; j < n - 1 - i; j++) {
                    if (a[j] > a[j + 1]) {
                        int t = a[j];
                        a[j] = a[j + 1];
                        a[j + 1] = t;
                    }
                }
            }
        }""")
    tests = [([[3, 1, 2], 3], {"array0": [1, 2, 3]}), ([[4, 3, 2, 1], 4], {"array0": [1, 2, 3, 4]}),
             ([[1, 2, 3], 3], {"array0": [1, 2, 3]})]
    problem = make_problem("void bubble_sort(int a[], int n)", tests, sector="sorting")
    result = irun(problem, good)
    assert result["status"] == "ok" and passes(result) == [True, True, True]      # gcc agrees: cases.py
    assert passes(irun(problem, "void bubble_sort(int a[], int n) {\n}")) == [False, False, True]
    lost = good.replace("int t = a[j];\n", "").replace("a[j + 1] = t;", "a[j + 1] = a[j];")
    assert passes(irun(problem, lost)) == [False, False, True]                    # D03: [3,1,2] becomes [1,1,2]


@covers("2.2/14")
def test_row14_arrays_are_shared_between_functions():
    """03 §2.2 row 14 / §2.1: arrays are passed by reference, scalars by value."""
    code = src("""
        void zero_first(int a[], int k) {
            a[0] = 0;
            k = 99;
        }
        int f(int a[], int n) {
            zero_first(a, n);
            return a[0] + a[1] + n;
        }""")
    problem = make_problem("int f(int a[], int n)", [([[5, 6], 2], {"returned": 8, "array0": [0, 6]})])
    assert passes(irun(problem, code)) == [True]                                  # gcc agrees: cases.py


# ---------------------------------------------------------------- row 15

@covers("2.2/15")
def test_row15_compare_effect():
    """03 §2.2 row 15: a relational compare of two array cells is a world effect compare {i, j}."""
    code = src("""
        int f(int a[], int n) {
            int c = 0;
            for (int j = 0; j < n - 1; j++) {
                if (a[j] > a[j + 1]) {
                    c++;
                }
            }
            return c;
        }""")
    trace = itrace(make_problem("int f(int a[], int n)", [([[3, 1, 2], 3], {}), ([[1, 2], 2], {})]), code)
    assert [e for e in effects(trace) if e.startswith("compare")] == ["compare:0:1", "compare:1:2"]
    assert [pt["effects_count"]["compare"] for pt in trace["per_test"]] == [2, 1]
    assert trace["effects_count"]["compare"] == 3 and trace["per_test"][0]["returned"] == 1


@covers("2.2/15")
def test_row15_no_compare_effect_for_cell_against_scalar():
    """03 §2.2 row 15: the effect is for two array cells; a cell against a scalar is not a compare."""
    code = src("""
        int f(int a[], int n, int x) {
            for (int i = 0; i < n; i++) {
                if (a[i] > x) {
                    return i;
                }
            }
            return -1;
        }""")
    trace = itrace(make_problem("int f(int a[], int n, int x)", [([[1, 5], 2, 3], {})]), code)
    assert trace["returned"] == 1 and trace["effects_count"]["compare"] == 0


# ---------------------------------------------------------------- row 17 (array names; the string-literal half is in test_strings.py)

@covers("2.2/17")
def test_row17_comparing_two_array_names():
    """03 §2.2 row 17: `a == b` on two array names is false, `a != b` is true; event array_compare."""
    code = src("""
        int f(int n) {
            int a[3] = {1, 2, 3};
            int b[3] = {1, 2, 3};
            int r = 0;
            if (a == b) r += 1;
            if (a != b) r += 10;
            return r + n;
        }""")
    trace = itrace(make_problem("int f(int n)", [([0], {})]), code)
    assert trace["returned"] == 10                                                # gcc agrees: cases.py
    assert [e["line"] for e in events(trace, "array_compare")] == [5, 6]


# ---------------------------------------------------------------- row 18

@covers("2.2/18")
def test_row18_call_and_ret_effects_in_recursion():
    """03 §2.2 row 18: each call pushes a frame (effect call {fn, depth, args}), each return pops one
    (effect ret {fn, depth, value}); counter max_depth."""
    code = src("""
        int factorial(int n) {
            if (n <= 1) {
                return 1;
            }
            return n * factorial(n - 1);
        }""")
    tests = [([3], {"returned": 6}), ([0], {"returned": 1}), ([5], {"returned": 120})]
    trace = itrace(make_problem("int factorial(int n)", tests, sector="recursion"), code)
    assert [e for e in effects(trace) if e.startswith(("call", "ret"))] == [
        "call:factorial:1:3", "call:factorial:2:2", "call:factorial:3:1",
        "ret:factorial:3:1", "ret:factorial:2:2", "ret:factorial:1:6"]
    assert [pt["max_depth"] for pt in trace["per_test"]] == [3, 1, 5]
    assert trace["max_depth"] == 5                                                # maximum over tests
    assert [pt["effects_count"]["call"] for pt in trace["per_test"]] == [3, 1, 5]
    assert trace["effects_count"]["call"] == 9                                    # sum over tests
    assert trace["effects_count"].get("ret", 9) == 9


@covers("2.2/18")
def test_row18_helper_function_calls():
    """03 §2.2 row 18: calls to helper functions; args are joined by ",", depth counts from the entry call."""
    code = src("""
        int add(int x, int y) {
            return x + y;
        }
        int f(int n) {
            int a = add(n, 1);
            return add(a, a);
        }""")
    trace = itrace(make_problem("int f(int n)", [([2], {"returned": 6})]), code)
    seen = [e for e in effects(trace) if e.startswith(("call", "ret"))]
    assert seen.count("call:f:1:2") == 1 and seen[-1] == "ret:f:1:6"
    assert [e for e in seen if ":add:" in e] == ["call:add:2:2,1", "ret:add:2:3", "call:add:2:3,3", "ret:add:2:6"]
    assert trace["max_depth"] == 2 and trace["effects_count"]["call"] == 3


@covers("2.2/18")
def test_row18_world_builtins_are_not_calls():
    """03 §2.2 row 18: call/ret and max_depth are about user functions, not fire() or printf()."""
    code = src("""
        void f(int n) {
            fire();
            printf("%d", n);
        }""")
    trace = itrace(make_problem("void f(int n)", [([1], {})]), code)
    assert trace["effects_count"]["call"] == 1 and trace["max_depth"] == 1
    assert [e for e in effects(trace) if e.startswith("call")] == ["call:f:1:1"]


# ---------------------------------------------------------------- row 19

@covers("2.2/19", "2.5/x4")
def test_row19_depth_cap_on_missing_base_case():
    """03 §2.2 row 19 / §2.5 v3 extra 4: recursion deeper than 100 halts with status timeout;
    event depth_cap_hit {fn, last_args}."""
    code = src("""
        int factorial(int n) {
            return n * factorial(n - 1);
        }""")
    problem = make_problem("int factorial(int n)", [([3], {"returned": 6}), ([0], {"returned": 1})], sector="recursion")
    trace = itrace(problem, code)
    assert trace["status"] == "timeout"
    assert [pt["status"] for pt in trace["per_test"]] == ["timeout", "timeout"]
    found = events(trace, "depth_cap_hit")
    assert [(e["fn"], e["test"], e["line"]) for e in found] == [("factorial", 0, 2), ("factorial", 1, 2)]
    assert all("last_args" in e for e in found)
    assert events(trace, "step_cap_hit") == []              # it is the depth cap that stops it, not the step cap
    assert DEPTH_CAP <= trace["max_depth"] <= DEPTH_CAP + 1
    result = irun(problem, code)
    assert result["status"] == "timeout" and result["tests"]["passed"] == 0


@covers("2.2/19")
def test_row19_depth_100_is_allowed_101_is_not():
    """03 §2.2 row 19: "deeper than 100" — a chain of exactly 100 frames finishes, 101 frames does not.
    (The gcc backend applies the same cap, by construction.)"""
    code = src("""
        int sum_to(int n) {
            if (n == 0) return 0;
            return n + sum_to(n - 1);
        }""")
    problem = make_problem("int sum_to(int n)", [([99], {"returned": 4950}), ([100], {"returned": 5050}), ([40], {"returned": 820})],
                           sector="recursion")
    trace = itrace(problem, code)
    assert [pt["status"] for pt in trace["per_test"]] == ["ok", "timeout", "ok"]
    assert [pt["max_depth"] for pt in trace["per_test"]][0] == 100 and trace["per_test"][2]["max_depth"] == 41
    assert [e["test"] for e in events(trace, "depth_cap_hit")] == [1]
    assert passes(irun(problem, code)) == [True, False, True]


# ---------------------------------------------------------------- row 20

@covers("2.2/20", "2.5/x5")
def test_row20_discarded_call_value():
    """03 §2.2 row 20 / §2.5 v3 extra 5: a statement that is only a call to a non-void user function
    throws the result away; event discarded_call_value {fn, line}. Execution is normal."""
    code = src("""
        int factorial(int n) {
            if (n <= 1) {
                return 1;
            }
            factorial(n - 1);
            return n;
        }""")
    trace = itrace(make_problem("int factorial(int n)", [([3], {"returned": 6})], sector="recursion"), code)
    assert trace["status"] == "ok" and trace["returned"] == 3
    found = events(trace, "discarded_call_value")
    assert found and {(e["fn"], e["line"]) for e in found} == {("factorial", 5)}
    assert trace["max_depth"] == 3


@covers("2.2/20")
def test_row20_no_event_for_void_functions_or_builtins():
    """03 §2.2 row 20: the event is only for non-void user functions."""
    code = src("""
        void beep(int n) {
            fire();
        }
        int f(int n) {
            beep(n);
            fire();
            printf("x");
            scan(n);
            return n;
        }""")
    trace = itrace(make_problem("int f(int n)", [([1], {})]), code)
    assert trace["status"] == "ok" and events(trace, "discarded_call_value") == []


@covers("2.2/20")
def test_row20_no_event_when_the_value_is_used():
    """03 §2.2 row 20: using the result (assign or return) gives no event."""
    code = src("""
        int twice(int n) {
            return n * 2;
        }
        int f(int n) {
            int a = twice(n);
            return twice(a);
        }""")
    trace = itrace(make_problem("int f(int n)", [([1], {})]), code)
    assert trace["returned"] == 4 and events(trace, "discarded_call_value") == []
