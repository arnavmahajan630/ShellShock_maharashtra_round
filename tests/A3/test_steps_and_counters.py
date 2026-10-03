"""Step, effect, event and per-test counter rules.

Sources: the "Step rules" block in ml/contracts/subset.py, decisions 2-5 in notes/W0.md,
and 03 §2.3 / §2.4 (display test, aggregation over tests, per_test, the 2,000-step cap).
"""
import copy
import time

from ml.contracts.subset import EFFECTS_COUNT_KEYS, GARBAGE
from tests.A3.helpers import effects, events, irun, itrace, make_problem, needs_interp, passes, src

pytestmark = needs_interp

P03_FOR = src("""
    int total_energy(int cells[], int n) {
        int total = 0;
        for (int i = 0; i < n; i++) {
            total += cells[i];
        }
        return total;
    }""")
P03_SIG = "int total_energy(int cells[], int n)"


def test_steps_one_per_statement_and_for_header():
    """subset.py step rules: one step per executed statement; a for-header yields a step for its
    init, each test and each update."""
    trace = itrace(make_problem(P03_SIG, [([[2, 4], 2], {"returned": 6})]), P03_FOR)
    #                                   decl init test body update test body update test return
    assert [s["line"] for s in trace["steps"]] == [2, 3, 3, 4, 3, 3, 4, 3, 3, 6]
    assert trace["truncated"] is False and trace["returned"] == 6


def test_step_vars_are_scalars_of_the_frame_after_the_step():
    """subset.py step rules: "vars" holds the scalar locals and parameters of the current frame
    after the step; arrays are not in "vars"."""
    trace = itrace(make_problem(P03_SIG, [([[2, 4], 2], {"returned": 6})]), P03_FOR)
    steps = trace["steps"]
    assert all("cells" not in s["vars"] for s in steps)
    assert all(s["vars"]["n"] == 2 for s in steps)
    assert steps[0]["vars"]["total"] == 0 and steps[0]["vars"].get("i") is None    # i has no value yet
    assert steps[1]["vars"]["i"] == 0                                              # after the for-init
    assert (steps[3]["vars"]["total"], steps[3]["vars"]["i"]) == (2, 0)            # after total += cells[0]
    assert steps[4]["vars"]["i"] == 1                                              # after i++
    assert (steps[6]["vars"]["total"], steps[6]["vars"]["i"]) == (6, 1)
    assert steps[-1]["vars"]["total"] == 6


def test_variable_without_a_value_is_null():
    """subset.py step rules: a variable with no value yet is null."""
    code = src("""
        int f(int n) {
            int x;
            int y = n;
            x = y + 1;
            return x;
        }""")
    trace = itrace(make_problem("int f(int n)", [([4], {})]), code)
    before = [s for s in trace["steps"] if s["line"] == 3]      # x is declared and has no value yet
    after = [s for s in trace["steps"] if s["line"] == 4]
    assert before and "x" in before[0]["vars"] and before[0]["vars"]["x"] is None
    assert before[0]["vars"]["y"] == 4
    assert after and after[0]["vars"]["x"] == 5
    assert events(trace, "uninit_read") == []


def test_local_arrays_are_not_in_vars():
    """subset.py step rules: arrays are not in "vars" (local arrays included)."""
    code = src("""
        int f(int n) {
            int b[3] = {1, 2, 3};
            int k = b[n];
            return k;
        }""")
    trace = itrace(make_problem("int f(int n)", [([1], {})]), code)
    assert trace["returned"] == 2 and all("b" not in s["vars"] for s in trace["steps"])
    assert trace["steps"][-1]["vars"] == {"n": 1, "k": 2}


def test_call_effect_on_first_callee_step_ret_on_its_return_step():
    """subset.py step rules: a step is recorded when its statement finishes; a "call" effect sits on
    the first step inside the called function and a "ret" effect on that function's return step, so
    the caller's own step comes after the callee's steps."""
    code = src("""
        int twice(int x) {
            return x * 2;
        }
        int f(int n) {
            int y = twice(n);
            return y + 1;
        }""")
    trace = itrace(make_problem("int f(int n)", [([5], {"returned": 11})]), code)
    steps = trace["steps"]
    assert [s["line"] for s in steps] == [2, 5, 6]          # callee's step first, then the caller's own
    assert "call:twice:2:5" in steps[0]["effects"] and "ret:twice:2:10" in steps[0]["effects"]
    assert steps[0]["vars"] == {"x": 5}                     # the callee's frame
    assert (steps[1]["vars"]["n"], steps[1]["vars"]["y"]) == (5, 10)
    assert "ret:f:1:11" in steps[2]["effects"]
    assert "call:f:1:5" in steps[0]["effects"] + steps[1]["effects"]


def test_vars_follow_the_current_frame_in_recursion():
    """subset.py step rules: "vars" is the current frame; in recursion each frame has its own n."""
    code = src("""
        int factorial(int n) {
            if (n <= 1) {
                return 1;
            }
            return n * factorial(n - 1);
        }""")
    trace = itrace(make_problem("int factorial(int n)", [([3], {"returned": 6})], sector="recursion"), code)
    assert all(set(s["vars"]) == {"n"} for s in trace["steps"])
    by_effect = {e: s["vars"]["n"] for s in trace["steps"] for e in s["effects"]}
    assert by_effect == {"call:factorial:1:3": 3, "call:factorial:2:2": 2, "call:factorial:3:1": 1,
                         "ret:factorial:3:1": 1, "ret:factorial:2:2": 2, "ret:factorial:1:6": 3}


def test_array_argument_in_call_effect():
    """notes/W0.md decision 2: effects are "<name>:<field>:<field>"; a call's args are joined by ","."""
    trace = itrace(make_problem(P03_SIG, [([[2, 4], 2], {"returned": 6})]), P03_FOR)
    calls = [e for e in effects(trace) if e.startswith("call:")]
    assert len(calls) == 1 and calls[0].startswith("call:total_energy:1:") and calls[0].endswith(",2")
    assert effects(trace)[-1] == "ret:total_energy:1:6"


def test_events_list_every_test_in_order_step_events_only_the_display_test():
    """subset.py step rules: trace["events"] lists every occurrence over all tests, in order, each with
    "test"; steps[].events carry no "test" and belong to the recorded test."""
    code = src("""
        int f(int a, int b) {
            int q = a / b;
            return q;
        }""")
    problem = make_problem("int f(int a, int b)", [([7, 2], {}), ([8, 2], {}), ([9, 2], {})], display_test=1)
    trace = itrace(problem, code)
    assert [(e["type"], e["test"], e["remainder_nonzero"]) for e in trace["events"]] == [
        ("intdiv", 0, True), ("intdiv", 1, False), ("intdiv", 2, True)]
    step_events = [e for s in trace["steps"] for e in s["events"]]
    assert len(step_events) == 1 and step_events[0]["type"] == "intdiv" and "test" not in step_events[0]
    assert step_events[0]["remainder_nonzero"] is False     # it is test 1 that was recorded
    holder = [s for s in trace["steps"] if s["events"]][0]
    assert holder["line"] == step_events[0]["line"] == 2


def test_steps_are_recorded_for_the_display_test():
    """03 §2.4: steps are recorded for one display test (problem.display_test); returned and printed
    on the trace are that test's."""
    code = src("""
        int f(int n) {
            printf("%d;", n);
            return n * 10;
        }""")
    problem = make_problem("int f(int n)", [([1], {}), ([2], {}), ([3], {})], display_test=2)
    trace = itrace(problem, code)
    assert all(s["vars"]["n"] == 3 for s in trace["steps"]) and trace["steps"]
    assert trace["returned"] == 30 and trace["printed"] == "3;"
    assert [pt["returned"] for pt in trace["per_test"]] == [10, 20, 30]
    assert [pt["printed"] for pt in trace["per_test"]] == ["1;", "2;", "3;"]


def test_trace_with_test_index_records_that_test():
    """ml/runner.py: trace(problem, code, test_index=k) records steps for test k; counters and events
    still cover all tests."""
    problem = make_problem(P03_SIG, [([[2, 4, 6], 3], {}), ([[5], 1], {}), ([[1, 2, 3, 4], 4], {})])
    default = itrace(problem, P03_FOR)
    chosen = itrace(problem, P03_FOR, test_index=2)
    assert default["steps"][0]["vars"]["n"] == 3 and chosen["steps"][0]["vars"]["n"] == 4
    assert chosen["returned"] == 10
    assert len(chosen["per_test"]) == 3 and chosen["loop_iters"] == default["loop_iters"] == {"L3": 8}
    assert [e for e in effects(chosen) if e.startswith("read")] == ["read_cell:0", "read_cell:1", "read_cell:2", "read_cell:3"]


def test_trace_counters_are_sums_and_max_depth_is_the_maximum():
    """subset.py step rules / notes/W0.md decision 5: loop_iters, branch and effects_count on the
    trace are sums over tests; max_depth is the maximum; per_test holds each test's own counters."""
    code = src("""
        int count_big(int a[], int n, int limit) {
            int c = 0;
            for (int i = 0; i < n; i++) {
                if (a[i] > limit) {
                    c++;
                    fire();
                }
            }
            return c;
        }""")
    tests = [([[1, 5, 9], 3, 4], {"returned": 2}), ([[], 0, 4], {"returned": 0}), ([[7, 8], 2, 0], {"returned": 2})]
    trace = itrace(make_problem("int count_big(int a[], int n, int limit)", tests), code)
    per_test = trace["per_test"]
    assert len(per_test) == 3
    assert [pt["status"] for pt in per_test] == ["ok"] * 3 and [pt["returned"] for pt in per_test] == [2, 0, 2]
    assert [pt["loop_iters"].get("L3", 0) for pt in per_test] == [3, 0, 2]
    assert [(pt["branch"].get("B4", {}).get("true", 0), pt["branch"].get("B4", {}).get("false", 0)) for pt in per_test] == [(2, 1), (0, 0), (2, 0)]
    assert [pt["effects_count"].get("fire", 0) for pt in per_test] == [2, 0, 2]
    assert [pt["effects_count"].get("read_cell", 0) for pt in per_test] == [3, 0, 2]
    assert [pt["max_depth"] for pt in per_test] == [1, 1, 1]
    assert trace["loop_iters"] == {"L3": 5} and trace["branch"] == {"B4": {"true": 4, "false": 1}}
    assert trace["effects_count"]["fire"] == 4 and trace["effects_count"]["read_cell"] == 5
    assert trace["effects_count"]["call"] == 3 and trace["max_depth"] == 1
    for key, total in trace["effects_count"].items():
        assert sum(pt["effects_count"].get(key, 0) for pt in per_test) == total, key


def test_effects_count_always_has_the_contract_keys():
    """03 §2.4 / subset.EFFECTS_COUNT_KEYS: these keys are present even when the count is 0."""
    trace = itrace(make_problem("int f(int n)", [([1], {})]), "int f(int n) {\n    return n;\n}")
    assert set(EFFECTS_COUNT_KEYS) <= set(trace["effects_count"])
    assert {k: trace["effects_count"][k] for k in EFFECTS_COUNT_KEYS} == {
        "fire": 0, "read_cell": 0, "read_void": 0, "write_cell": 0, "compare": 0, "call": 1}


def test_per_test_loop_counts_for_the_off_by_one_twin():
    """03 §2.4: features that say "on every test" (b_iter_delta_const_pm1) read per_test, so each
    test's own loop count must be there: `i <= n` runs n + 1 times on every test."""
    code = P03_FOR.replace("i < n", "i <= n")
    tests = [([[2, 4, 6], 3], {}), ([[5], 1], {}), ([[1, 2, 3, 4], 4], {}), ([[0, 0, 9], 3], {}), ([[7, 1], 2], {})]
    trace = itrace(make_problem(P03_SIG, tests), code)
    assert [pt["loop_iters"]["L3"] for pt in trace["per_test"]] == [4, 2, 5, 4, 3]
    assert trace["loop_iters"] == {"L3": 18}
    assert [pt["returned"] for pt in trace["per_test"]] == [GARBAGE + 12, GARBAGE + 5, GARBAGE + 10, GARBAGE + 9, GARBAGE + 8]
    assert [pt["effects_count"].get("read_void", 0) for pt in trace["per_test"]] == [1] * 5
    assert trace["effects_count"]["read_cell"] == 13 and trace["effects_count"]["read_void"] == 5


def test_per_test_branch_counts_for_always_true():
    """03 §2.4: b_branch_always / b_branch_never read per_test: `if (code = 42)` is true on every test."""
    code = "int door_open(int code) {\n    if (code = 42) {\n        return 1;\n    }\n    return 0;\n}"
    trace = itrace(make_problem("int door_open(int code)", [([42], {}), ([7], {}), ([0], {})]), code)
    assert [(pt["branch"]["B2"].get("true", 0), pt["branch"]["B2"].get("false", 0)) for pt in trace["per_test"]] == [(1, 0)] * 3
    assert trace["branch"] == {"B2": {"true": 3, "false": 0}}


def test_per_test_printed_and_void_return():
    """03 §2.4 per_test: printed is each test's own output; a void entry function has returned = null."""
    code = "void countdown(int n) {\n    while (n > 0) {\n        printf(\"%d\\n\", n);\n        n--;\n    }\n    launch();\n}"
    trace = itrace(make_problem("void countdown(int n)", [([2], {}), ([0], {}), ([3], {})]), code)
    assert [pt["printed"] for pt in trace["per_test"]] == ["2\n1\n", "", "3\n2\n1\n"]
    assert [pt["returned"] for pt in trace["per_test"]] == [None, None, None]
    assert [pt["effects_count"].get("launch", 0) for pt in trace["per_test"]] == [1, 1, 1]
    assert trace["effects_count"]["launch"] == 3


def test_each_test_gets_fresh_arguments():
    """03 §2.3: every test is called with its own args; an in-place change does not leak into the
    next test or into the problem dict."""
    code = "void f(int a[], int n) {\n    a[0] = a[0] + 1;\n}"
    problem = make_problem("void f(int a[], int n)", [([[1, 9], 2], {"array0": [2, 9]}), ([[1, 9], 2], {"array0": [2, 9]})])
    before = copy.deepcopy(problem)
    assert passes(irun(problem, code)) == [True, True]
    itrace(problem, code)
    assert passes(irun(problem, code)) == [True, True]
    assert problem == before


def test_sample_only_runs_only_the_sample_tests():
    """03 §2.3: each problem marks 2 tests as sample; run with sample_only runs only those."""
    problem = make_problem(P03_SIG, [([[2, 4, 6], 3], {"returned": 12}), ([[5], 1], {"returned": 5}),
                                     ([[1, 2, 3, 4], 4], {"returned": 10})])
    problem["tests"][0]["sample"] = problem["tests"][2]["sample"] = True
    result = irun(problem, P03_FOR, sample_only=True)
    assert result["tests"]["total"] == 2 and result["tests"]["passed"] == 2
    assert [r["args"] for r in result["tests"]["results"]] == [[[2, 4, 6], 3], [[1, 2, 3, 4], 4]]
    assert irun(problem, P03_FOR)["tests"]["total"] == 3


def test_event_lines_point_at_the_statement():
    """03 §2.2 header "Event (with line)" / §2.5: every event carries the line of the construct."""
    code = src("""
        int f(int a[], int n) {
            int x;
            int y = a[n];
            int z = n / 2;
            if (n = 3);
            return x;
        }""")
    trace = itrace(make_problem("int f(int a[], int n)", [([[1, 2, 3], 3], {})]), code)
    lines = {e["type"]: e["line"] for e in trace["events"]}
    assert lines["oob_read"] == 3 and lines["intdiv"] == 4 and lines["assign_in_cond"] == 5
    assert lines["empty_body"] == 5 and lines["uninit_read"] == 6


def test_performance_budget():
    """03 §2.5 performance budget: a problem's whole test suite in under 30 ms (best of 7 runs)."""
    tests = [([[2, 4, 6], 3], {"returned": 12}), ([[5], 1], {"returned": 5}), ([[1, 2, 3, 4], 4], {"returned": 10}),
             ([[0, 0, 9], 3], {"returned": 9}), ([[7, 1], 2], {"returned": 8})]
    problem = make_problem(P03_SIG, tests)
    best = float("inf")
    for _ in range(7):
        started = time.perf_counter()
        result = irun(problem, P03_FOR)
        best = min(best, time.perf_counter() - started)
    assert result["tests"]["passed"] == 5
    assert best < 0.030, f"best of 7 runs: {best * 1000:.1f} ms"
