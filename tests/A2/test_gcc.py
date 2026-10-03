"""Package A2: the gcc stand-in (ml/oracle/gcc.py), through ml.runner where possible."""
import glob
import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from ml import runner
from ml.contracts import schemas as S
from ml.contracts.subset import GARBAGE
from ml.oracle import gcc

PROBLEMS = {p["problem_id"]: p for p in (json.loads(f.read_text(encoding="utf-8")) for f in
            sorted((Path(__file__).parent.parent / "fixtures" / "problems").glob("*.json")))}
VARIANTS = [(pid, k) for pid, p in PROBLEMS.items() for k in range(len(p["correct_variants"]))]


def run(problem, code, **kwargs):
    result = runner.run_tests(problem, code, backend="gcc", **kwargs)
    S.RunResult.model_validate(result)
    assert result["backend"] == "gcc"
    return result


def adhoc(signature, tests, **extra):
    return {"signature": signature, "tests": [{"args": a, "expect": e} for a, e in tests], **extra}


def failed(result):
    return [i for i, r in enumerate(result["tests"]["results"]) if not r["pass"]]


def test_runner_sees_the_backend():
    assert runner.available("gcc")
    assert set(PROBLEMS) == {"P03", "P11", "Q17"}


@pytest.mark.parametrize("pid,k", VARIANTS, ids=[f"{p}-cv{k}" for p, k in VARIANTS])
def test_correct_variants_pass(pid, k):
    result = run(PROBLEMS[pid], PROBLEMS[pid]["correct_variants"][k])
    assert result["status"] == "ok"
    assert result["tests"]["passed"] == result["tests"]["total"] == len(PROBLEMS[pid]["tests"])
    assert all(r["pass"] for r in result["tests"]["results"])


def test_wrong_program_fails_the_right_tests():
    code = ("int total_energy(int cells[], int n) {\n    int total = 0;\n"
            "    for (int i = 0; i < n; i++) {\n        total = cells[i];\n    }\n    return total;\n}")
    result = run(PROBLEMS["P03"], code)
    assert result["status"] == "ok"
    assert failed(result) == [0, 2, 4]                  # tests 1 and 3 pass by luck
    assert result["tests"]["passed"] == 2
    assert result["tests"]["results"][0]["got"] == {"returned": 6}
    assert result["tests"]["results"][0]["expected"] == {"returned": 12}


def test_assign_in_condition_fails_the_right_tests():
    result = run(PROBLEMS["P11"], "int door_open(int code) {\n    if (code = 42) return 1;\n    return 0;\n}")
    assert result["status"] == "ok" and failed(result) == [1, 2, 3, 4]


def test_endless_loop_times_out():
    code = ("int total_energy(int cells[], int n) {\n    int total = 0;\n    int i = 0;\n"
            "    while (i < n) {\n        total += cells[i];\n    }\n    return total;\n}")
    started = time.perf_counter()
    result = run(PROBLEMS["P03"], code)
    assert time.perf_counter() - started < gcc.TEST_TIMEOUT_S + 5
    assert result["status"] == "timeout"
    assert result["tests"]["passed"] == 0
    assert all(r["got"]["error"] == "time_limit" for r in result["tests"]["results"])


def test_endless_printing_is_cut_off():
    problem = adhoc("void spam(int n)", [([1], {"printed": "x"})])
    result = run(problem, 'void spam(int n) {\n    while (n > 0) {\n        printf("x");\n    }\n}')
    assert result["status"] == "timeout"
    assert result["tests"]["results"][0]["got"]["error"] == "output_limit"


def test_endless_recursion_hits_the_depth_cap():
    result = run(PROBLEMS["Q17"], "int factorial(int n) {\n    return n * factorial(n - 1);\n}")
    assert result["status"] == "timeout" and result["tests"]["passed"] == 0
    assert result["tests"]["results"][0]["got"]["error"] == "depth_cap_hit"


def test_compile_error_is_parse_error_with_the_learners_line():
    result = run(PROBLEMS["P03"], "int total_energy(int cells[], int n) {\n    int t = 0;\n    return t +;\n}")
    assert result["status"] == "parse_error"
    assert result["tests"]["passed"] == 0 and result["tests"]["total"] == 5
    assert result["tests"]["results"][0]["got"]["error"].startswith("line 3:")


def test_missing_entry_function_is_parse_error():
    assert run(PROBLEMS["P03"], "int something_else(int n) {\n    return n;\n}")["status"] == "parse_error"


def test_a_crash_in_one_test_keeps_the_other_results():
    code = "int door_open(int code) {\n    int z = 10 / (code - 7);\n    return code == 42;\n}"
    result = run(PROBLEMS["P11"], code)
    assert result["status"] == "runtime_error"
    assert failed(result) == [1] and result["tests"]["passed"] == 4
    assert result["tests"]["results"][1]["got"]["error"] == "div_zero"


def test_out_of_bounds_read_gives_garbage_like_the_interpreter():
    code = ("int total_energy(int cells[], int n) {\n    int total = 0;\n"
            "    for (int i = 0; i <= n; i++) total += cells[i];\n    return total;\n}")
    result = run(PROBLEMS["P03"], code)
    assert result["status"] == "ok" and result["tests"]["passed"] == 0
    assert result["tests"]["results"][0]["got"]["returned"] == GARBAGE + 12     # as in p03_le_oob_read.json


def test_sample_only_runs_the_two_sample_tests():
    result = run(PROBLEMS["P03"], PROBLEMS["P03"]["correct_variants"][0], sample_only=True)
    assert result["tests"]["total"] == 2 and result["tests"]["passed"] == 2
    assert [r["args"] for r in result["tests"]["results"]] == [[[2, 4, 6], 3], [[5], 1]]


def test_string_argument_becomes_a_char_array_with_terminator():
    problem = adhoc("int str_length(char s[])", [(["level"], {"returned": 5}), ([""], {"returned": 0}),
                                                 (["a b"], {"returned": 3})])
    own = "int str_length(char s[]) {\n    int i = 0;\n    while (s[i] != '\\0') i++;\n    return i;\n}"
    assert run(problem, own)["tests"]["passed"] == 3
    assert run(problem, "int str_length(char s[]) {\n    return strlen(s);\n}")["tests"]["passed"] == 3


def test_forbidden_builtin_is_unsupported():
    problem = adhoc("int str_length(char s[])", [(["level"], {"returned": 5})], forbid=["strlen"])
    result = run(problem, "int str_length(char s[]) {\n    return strlen(s);\n}")
    assert result["status"] == "unsupported" and result["tests"]["passed"] == 0
    ok = run(problem, "int str_length(char s[]) {\n    int i = 0; // no strlen( here\n    while (s[i]) i++;\n    return i;\n}")
    assert ok["status"] == "ok" and ok["tests"]["passed"] == 1


def test_array0_shows_an_in_place_sort():
    problem = adhoc("void bubble_sort(int a[], int n)", [([[3, 1, 2], 3], {"array0": [1, 2, 3]}),
                                                         ([[4, 3, 2, 1], 4], {"array0": [1, 2, 3, 4]})])
    good = ("void bubble_sort(int a[], int n) {\n    for (int i = 0; i < n - 1; i++)\n"
            "        for (int j = 0; j < n - 1 - i; j++)\n            if (a[j] > a[j + 1]) {\n"
            "                int t = a[j]; a[j] = a[j + 1]; a[j + 1] = t;\n            }\n}")
    assert run(problem, good)["tests"]["passed"] == 2
    lost = good.replace("int t = a[j]; a[j] = a[j + 1]; a[j + 1] = t;", "a[j] = a[j + 1]; a[j + 1] = a[j];")
    result = run(problem, lost)
    assert result["status"] == "ok" and result["tests"]["passed"] == 0
    assert result["tests"]["results"][0]["got"] == {"array0": [1, 1, 2]}


def test_array0_of_an_empty_array_is_an_empty_list():
    problem = adhoc("void reverse(int a[], int n)", [([[], 0], {"array0": []})])
    result = run(problem, "void reverse(int a[], int n) {\n    int i = 0;\n    while (i < n - 1 - i) i++;\n}")
    assert result["tests"]["passed"] == 1 and result["tests"]["results"][0]["got"] == {"array0": []}
    assert gcc.observe({"signature": "int f(int n)"}, "int f(int n) { return n; }", [[1]])["runs"][0]["array0"] is None


def test_array0_of_a_string_argument():
    problem = adhoc("void shout(char s[])", [(["abc"], {"array0": "Abc"})])
    assert run(problem, "void shout(char s[]) {\n    s[0] = s[0] - 32;\n}")["tests"]["passed"] == 1


def test_world_effect_counts():
    problem = adhoc("void fire_shots(int n)", [([3], {"fire": 3}), ([0], {"fire": 0})])
    good = "void fire_shots(int n) {\n    for (int i = 0; i < n; i++) {\n        fire();\n    }\n}"
    assert run(problem, good)["tests"]["passed"] == 2
    off = run(problem, good.replace("i < n", "i <= n"))
    assert off["tests"]["passed"] == 0 and off["tests"]["results"][0]["got"] == {"fire": 4}
    door = adhoc("void work(int n)", [([2], {"door_open": 1, "door_closed": 2, "launch": 1, "scan": 2, "printed": "2\n1\n"})])
    code = ('void work(int n) {\n    open_door();\n    while (n > 0) {\n        printf("%d\\n", n);\n'
            "        close_door();\n        scan(n);\n        n--;\n    }\n    launch();\n}")
    result = run(door, code)
    assert result["tests"]["passed"] == 1, result
    assert result["tests"]["results"][0]["got"]["printed"] == "2\n1\n"


def test_printf_formats():
    problem = adhoc("void show(int n, float x, char c, char s[])",
                    [([7, 2.5, "q", "hi"], {"printed": "7 7 2.500000 2.50 q hi 100% 2.500000 7\n"})])
    code = ('void show(int n, float x, char c, char s[]) {\n'
            '    printf("%d %i %f %.2f %c %s 100%% %lf %ld\\n", n, n, x, x, c, s, x, (long)n);\n}')
    result = run(problem, code)
    assert result["tests"]["passed"] == 1, result["tests"]["results"][0]["got"]


def test_float_results_use_the_tolerance():
    problem = adhoc("float avg_fuel(int tanks[], int n)", [([[1, 2], 2], {"returned": 1.5}),
                                                           ([[1, 1, 2], 3], {"returned": 1.3333}),
                                                           ([[1, 1, 2], 3], {"returned": 1.34})])
    code = ("float avg_fuel(int tanks[], int n) {\n    float total = 0;\n"
            "    for (int i = 0; i < n; i++) total += tanks[i];\n    return total / n;\n}")
    result = run(problem, code)
    assert failed(result) == [2]
    intdiv = code.replace("float total = 0;", "int total = 0;")
    assert failed(run(problem, intdiv)) == [0, 1, 2]


def test_max_depth_le():
    problem = adhoc("int sum_to(int n)", [([5], {"returned": 15, "max_depth_le": 6}),
                                          ([5], {"returned": 15, "max_depth_le": 5})])
    result = run(problem, "int sum_to(int n) {\n    if (n == 0) return 0;\n    return n + sum_to(n - 1);\n}")
    assert failed(result) == [1]
    assert result["tests"]["results"][1]["got"] == {"returned": 15, "max_depth": 6}


def test_keys_gcc_cannot_measure_are_reported_not_judged():
    problem = adhoc("int first(int a[], int n)", [([[4, 5], 2], {"returned": 4, "read_cell": 1})])
    result = run(problem, "int first(int a[], int n) {\n    return a[0];\n}")
    assert result["tests"]["passed"] == 1
    assert result["tests"]["results"][0]["got"] == {"returned": 4, "unchecked": ["read_cell"]}


def test_comments_includes_defines_helpers_and_a_learner_main():
    code = ("#include <stdio.h>\n#define LIMIT 3\n/* helper named like a windows macro */\n"
            "int max(int a, int b) { return a > b ? a : b; }   // fine\n"
            "int small(int a[], int n) {\n    int best = a[0];\n"
            "    for (int i = 1; i < n && i < LIMIT; i++) best = max(best, a[i]);\n    return best;\n}\n"
            "int main() {\n    int a[3] = {1, 2, 3};\n    printf(\"%d\", small(a, 3));\n    return 0;\n}\n")
    problem = adhoc("int small(int a[], int n)", [([[1, 9, 3, 50], 4], {"returned": 9, "printed": ""})])
    result = run(problem, code)
    assert result["status"] == "ok" and result["tests"]["passed"] == 1, result


def test_observe_reports_everything_measured():
    seen = gcc.observe({"signature": "int twice(int a[], int n)"},
                       'int twice(int a[], int n) {\n    a[0] = 9;\n    fire();\n    printf("hi");\n    return 2 * n;\n}',
                       [[[1, 2], 2]])
    assert seen["status"] == "ok"
    assert seen["runs"][0] == {"status": "ok", "error": None, "returned": 4, "printed": "hi", "array0": [9, 2],
                               "effects": {"fire": 1, "launch": 0, "door_open": 0, "door_closed": 0, "scan": 0},
                               "max_depth": 1, "calls": 1}


def test_bad_problem_signature_is_the_authors_error():
    with pytest.raises(ValueError):
        gcc.run_tests(adhoc("int f(int a, int b)", [([1], {"returned": 1})]), "int f(int a, int b) { return a; }")


def test_no_temp_folders_left_behind():
    run(PROBLEMS["P11"], "int door_open(int code) {\n    while (1) { }\n    return 0;\n}")
    assert glob.glob(os.path.join(tempfile.gettempdir(), "relearn_gcc_*", "run_*")) == []
