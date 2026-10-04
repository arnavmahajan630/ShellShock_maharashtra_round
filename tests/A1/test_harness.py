"""harness.py: run_tests and trace shapes (ml_plan/03 §2.3, §2.4; notes/W0.md 2-5, 7)."""
import copy
import json

import pytest

from ml import runner
from ml.c_interp import harness
from ml.contracts import schemas as S
from ml.contracts.subset import EFFECTS_COUNT_KEYS, GARBAGE, MAX_RECORDED_STEPS
from tests.fixtures import build

from .util import check_trace, effects, events, load_problem, load_trace, problem, run, trace

P03, P11, Q17 = load_problem("P03_total_energy"), load_problem("P11_door_open"), load_problem("Q17_factorial")


# ---------------------------------------------------------------- the three sample problems

@pytest.mark.parametrize("prob", [P03, P11, Q17], ids=lambda p: p["problem_id"])
def test_correct_variants_pass_every_test(prob):
    for variant in prob["correct_variants"]:
        result = harness.run_tests(prob, variant)
        S.RunResult.model_validate(result)
        assert result["status"] == "ok" and result["backend"] == "interp"
        assert result["tests"]["passed"] == result["tests"]["total"] == len(prob["tests"])
        out = harness.trace(prob, variant)
        check_trace(out)
        assert out["status"] == "ok" and out["events"] == []
        assert out["returned"] == prob["tests"][prob["display_test"]]["expect"]["returned"]


@pytest.mark.parametrize("prob", [P03, P11, Q17], ids=lambda p: p["problem_id"])
def test_starter_fails_without_crashing(prob):
    result = harness.run_tests(prob, prob["starter"])
    assert result["status"] == "ok" and result["tests"]["passed"] == 0
    assert all(r["got"]["returned"] == GARBAGE for r in result["tests"]["results"])


def test_runner_uses_the_interpreter():
    assert runner.available("interp")
    assert runner.run_tests(P03, P03["correct_variants"][0])["backend"] == "interp"
    assert runner.run_tests(P03, P03["correct_variants"][0], backend="interp", sample_only=True)["tests"]["total"] == 2
    assert runner.trace(Q17, build.Q17_OK, test_index=3)["returned"] == 120


# ---------------------------------------------------------------- agreement with the sample traces

SAMPLES = [("p03_le_oob_read", P03, build.P03_LE), ("p03_no_update_step_cap", P03, build.P03_NO_UPDATE),
           ("q17_recursion_ok", Q17, build.Q17_OK)]


def _without_zero_sides(per_test):
    """The hand-written samples drop a zero `true` / `false`; the interpreter keeps both keys."""
    out = copy.deepcopy(per_test)
    for entry in out:
        entry["branch"] = {k: {side: n for side, n in v.items() if n} for k, v in entry["branch"].items()}
    return out


@pytest.mark.parametrize("stem,prob,code", SAMPLES, ids=[s[0] for s in SAMPLES])
def test_agrees_with_sample_trace(stem, prob, code):
    sample = load_trace(stem)
    out = harness.trace(prob, code)
    check_trace(out)
    for key in ("status", "returned", "printed", "events", "loop_iters", "branch", "effects_count", "max_depth"):
        assert out[key] == sample[key], key
    assert _without_zero_sides(out["per_test"]) == sample["per_test"]
    shared = min(len(out["steps"]), len(sample["steps"]))
    assert out["steps"][:shared] == sample["steps"][:shared]


def test_sample_trace_step_counts():
    assert len(harness.trace(P03, build.P03_LE)["steps"]) == 16
    assert len(harness.trace(Q17, build.Q17_OK)["steps"]) == 6
    capped = harness.trace(P03, build.P03_NO_UPDATE)
    assert len(capped["steps"]) == MAX_RECORDED_STEPS and capped["truncated"] is True


def test_p11_semi_matches_the_hand_simulation():
    out = harness.trace(P11, build.P11_SEMI)
    check_trace(out)
    assert out["status"] == "ok" and out["returned"] == 1                  # display_test is 1 (code 7)
    assert [t["returned"] for t in out["per_test"]] == [1, 1, 1, 1, 1]
    assert out["branch"] == {"B2": {"true": 1, "false": 4}}
    assert [(e["type"], e["kind"], e["line"], e["test"]) for e in out["events"]] == \
        [("empty_body", "if", 2, t) for t in range(5)]
    assert out["steps"] == [
        {"i": 0, "line": 2, "vars": {"code": 7}, "events": [{"type": "empty_body", "kind": "if", "line": 2}],
         "effects": ["call:door_open:1:7"]},
        {"i": 1, "line": 3, "vars": {"code": 7}, "events": [], "effects": ["ret:door_open:1:1"]},
    ]
    assert harness.run_tests(P11, build.P11_SEMI)["tests"]["passed"] == 1


def test_q06_swap_without_temp_matches_the_hand_simulation_up_to_the_end_step():
    recorder_trace, outcomes = build.run_all(build.Q06, build.q06_no_temp)
    full = dict(build.Q06, family="bubble_sort", split="train", correct_variants=[], allowed_ops=[])
    out = harness.trace(full, build.Q06_NO_TEMP)
    check_trace(out)
    # the interpreter adds one step at the closing brace of a void function (the `ret` effect)
    assert out["steps"][:-1] == recorder_trace["steps"]
    assert out["steps"][-1]["line"] == 10 and out["steps"][-1]["effects"] == ["ret:bubble_sort:1:"]
    for key in ("loop_iters", "branch", "events", "max_depth"):
        assert out[key] == recorder_trace[key]
    assert out["effects_count"] == dict(recorder_trace["effects_count"], ret=4)
    result = harness.run_tests(full, build.Q06_NO_TEMP)
    assert [r["got"]["array0"] for r in result["tests"]["results"]] == [o[2][0] for o in outcomes]
    assert result["tests"] == build.tests_object(build.Q06, outcomes)


# ---------------------------------------------------------------- run_tests

def test_run_result_shape_and_got_keys():
    result = harness.run_tests(P03, build.P03_LE)
    S.RunResult.model_validate(result)
    assert set(result) == {"status", "backend", "tests"}
    assert result["tests"]["passed"] == 0 and result["tests"]["total"] == 5
    first = result["tests"]["results"][0]
    assert first == {"args": [[2, 4, 6], 3], "expected": {"returned": 12}, "got": {"returned": GARBAGE + 12},
                     "pass": False}
    json.dumps(result, allow_nan=False)


def test_sample_only_runs_the_two_sample_tests():
    result = harness.run_tests(P03, P03["correct_variants"][0], sample_only=True)
    assert result["tests"]["total"] == 2 and result["tests"]["passed"] == 2
    assert [r["args"] for r in result["tests"]["results"]] == [t["args"] for t in P03["tests"] if t.get("sample")]


def test_status_is_the_first_one_that_is_not_ok():
    code = "int f(int n) {\n    while (n == 1) {\n    }\n    return 10 / n;\n}"
    result = run(code, "int f(int n)", [([5], {"returned": 2}), ([0], {"returned": 0}), ([1], {"returned": 10})])
    assert result["status"] == "runtime_error"
    assert [r["pass"] for r in result["tests"]["results"]] == [True, False, False]
    assert result["tests"]["results"][1]["got"] == {"returned": None, "status": "runtime_error"}
    assert result["tests"]["results"][2]["got"] == {"returned": None, "status": "timeout"}


def test_expect_printed_and_effects():
    code = "void countdown(int n) {\n    for (int i = n; i >= 1; i--) {\n        printf(\"%d\\n\", i);\n    }\n    launch();\n}"
    tests = [([3], {"printed": "3\n2\n1\n", "launch": 1}), ([0], {"printed": "", "launch": 1, "fire": 0}),
             ([2], {"printed": "2 1", "launch": 1}), ([2], {"printed": "2\n1\n", "launch": 2})]
    result = run(code, "void countdown(int n)", tests)
    assert [r["pass"] for r in result["tests"]["results"]] == [True, True, False, False]
    assert result["tests"]["results"][3]["got"] == {"printed": "2\n1\n", "launch": 1}


def test_expect_fire_count():
    code = "void fire_shots(int n) {\n    for (int i = 0; i <= n; i++) {\n        fire();\n    }\n}"
    result = run(code, "void fire_shots(int n)", [([3], {"fire": 3})])
    assert result["tests"]["results"][0] == {"args": [3], "expected": {"fire": 3}, "got": {"fire": 4}, "pass": False}


def test_expect_float_within_tolerance():
    code = "float avg(int t[], int n) {\n    int s = 0;\n    for (int i = 0; i < n; i++) {\n        s += t[i];\n    }\n    return (float)s / n;\n}"
    tests = [([[1, 2, 2], 3], {"returned": 1.6667}), ([[1, 2, 2], 3], {"returned": 1.67}),
             ([[2, 4], 2], {"returned": 3}), ([[2, 4], 2], {"returned": 3.0005})]
    result = run(code, "float avg(int t[], int n)", tests)
    assert [r["pass"] for r in result["tests"]["results"]] == [True, False, True, True]
    truncating = code.replace("(float)s / n", "s / n")
    assert run(truncating, "float avg(int t[], int n)", tests[:1])["tests"]["results"][0]["got"] == {"returned": 1.0}


def test_expect_array0_and_max_depth_le():
    code = ("void reverse(int a[], int n) {\n    for (int i = 0; i < n / 2; i++) {\n        int t = a[i];\n"
            "        a[i] = a[n - 1 - i];\n        a[n - 1 - i] = t;\n    }\n}")
    result = run(code, "void reverse(int a[], int n)", [([[1, 2, 3], 3], {"array0": [3, 2, 1]}),
                                                          ([[1, 2], 2], {"array0": [1, 2]})])
    assert [r["pass"] for r in result["tests"]["results"]] == [True, False]
    assert result["tests"]["results"][1]["got"] == {"array0": [2, 1]}

    deep = "int sum_to(int n) {\n    if (n == 0) {\n        return 0;\n    }\n    return n + sum_to(n - 1);\n}"
    result = run(deep, "int sum_to(int n)", [([4], {"returned": 10, "max_depth_le": 5}),
                                              ([4], {"returned": 10, "max_depth_le": 4})])
    assert [r["pass"] for r in result["tests"]["results"]] == [True, False]
    assert result["tests"]["results"][1]["got"] == {"returned": 10, "max_depth_le": 5}


def test_argument_kinds():
    code = ("float f(int a[], float w[], char s[], int n, float x, char c) {\n"
            "    return a[0] + w[1] + s[0] + n + x + c;\n}")
    sig = "float f(int a[], float w[], char s[], int n, float x, char c)"
    result = run(code, sig, [([[1], [0.5, 0.25], "A", 2, 1.5, "B"], {"returned": 1 + 0.25 + 65 + 2 + 1.5 + 66})])
    assert result["tests"]["passed"] == 1
    out = trace(code, sig, [[1], [0.5, 0.25], "A", 2, 1.5, "B"])
    assert effects(out)[0] == "call:f:1:a,w,s,2,1.5,66"
    assert out["steps"][0]["vars"] == {"n": 2, "x": 1.5, "c": 66}


def test_problem_args_are_not_modified():
    prob = problem("void f(int a[], int n)", [([[3, 1], 2], {"array0": [0, 0]})])
    before = copy.deepcopy(prob)
    harness.run_tests(prob, "void f(int a[], int n) {\n    a[0] = 0;\n    a[1] = 0;\n}")
    harness.trace(prob, "void f(int a[], int n) {\n    a[0] = 0;\n    a[1] = 0;\n}")
    assert prob == before


# ---------------------------------------------------------------- trace

def test_trace_shape_for_code_that_cannot_run():
    for code, status in (("int total_energy(int cells[], int n) {\n    return 1\n}", "parse_error"),
                         ("int total_energy(int *cells, int n) {\n    return 1;\n}", "unsupported")):
        out = harness.trace(P03, code)
        S.Trace.model_validate(out)
        assert out["status"] == status and out["steps"] == [] and out["events"] == [] and out["returned"] is None
        assert out["effects_count"] == dict.fromkeys(EFFECTS_COUNT_KEYS, 0)
        assert [t["status"] for t in out["per_test"]] == [status] * 5
        result = harness.run_tests(P03, code)
        S.RunResult.model_validate(result)
        assert result["status"] == status and result["tests"]["passed"] == 0 and result["tests"]["total"] == 5
        assert all(r["got"] == {} and r["pass"] is False for r in result["tests"]["results"])


def test_trace_steps_come_from_display_test_or_test_index():
    code = P03["correct_variants"][0]
    assert harness.trace(P03, code)["steps"][0]["vars"]["n"] == 3
    shown = harness.trace(P03, code, test_index=2)
    assert shown["steps"][0]["effects"] == ["call:total_energy:1:cells,4"] and shown["returned"] == 10
    assert harness.trace(P11, P11["correct_variants"][0])["steps"][0]["vars"] == {"code": 7}      # display_test 1
    # counters and events are the same whichever test is shown
    for key in ("loop_iters", "branch", "events", "effects_count", "max_depth", "per_test"):
        assert shown[key] == harness.trace(P03, code)[key]


def test_trace_counters_are_sums_and_per_test_is_separate():
    out = harness.trace(P03, P03["correct_variants"][0])
    assert out["loop_iters"] == {"L3": 3 + 1 + 4 + 3 + 2}
    assert [t["loop_iters"] for t in out["per_test"]] == [{"L3": 3}, {"L3": 1}, {"L3": 4}, {"L3": 3}, {"L3": 2}]
    assert out["effects_count"] == {"fire": 0, "read_cell": 13, "read_void": 0, "write_cell": 0, "compare": 0,
                                    "call": 5, "ret": 5}
    assert out["per_test"][1] == {"status": "ok", "returned": 5, "printed": "", "loop_iters": {"L3": 1},
                                  "branch": {}, "effects_count": {"read_cell": 1, "call": 1, "ret": 1},
                                  "max_depth": 1}
    assert [t["returned"] for t in out["per_test"]] == [12, 5, 10, 9, 8]


def test_trace_events_list_every_occurrence_with_its_test():
    out = harness.trace(P03, build.P03_LE)
    assert [(e["test"], e["idx"], e["size"]) for e in out["events"]] == [(0, 3, 3), (1, 1, 1), (2, 4, 4), (3, 3, 3),
                                                                        (4, 2, 2)]
    step_events = [e for s in out["steps"] for e in s["events"]]
    assert step_events == [{"type": "oob_read", "line": 4, "arr": "cells", "idx": 3, "size": 3}]


def test_trace_printed_and_returned_come_from_the_shown_test():
    code = "int f(int n) {\n    printf(\"%d;\", n);\n    return n * 2;\n}"
    prob = problem("int f(int n)", [[1], [2], [3]], display_test=2)
    out = harness.trace(prob, code)
    assert (out["returned"], out["printed"]) == (6, "3;")
    assert [(t["returned"], t["printed"]) for t in out["per_test"]] == [(2, "1;"), (4, "2;"), (6, "3;")]
    assert harness.trace(prob, code, test_index=0)["printed"] == "1;"


def test_same_source_is_parsed_once():
    code = "int f(int n) {\n    return n + 123456;\n}"
    first, _ = harness.compile_source(code)
    again, _ = harness.compile_source(code)
    assert first is again
    other, _ = harness.compile_source(code, forbid=["strlen"])
    assert other is not first


def test_repeated_runs_give_identical_results():
    first = harness.trace(P03, build.P03_LE)
    harness.trace(Q17, build.Q17_OK)
    assert harness.trace(P03, build.P03_LE) == first
    assert events(first, "oob_read")
