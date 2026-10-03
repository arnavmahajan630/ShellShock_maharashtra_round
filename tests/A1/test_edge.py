"""Corners that are easy to get wrong: evaluation order, typing, robustness, threads."""
import threading

from ml.c_interp import harness, interp
from ml.contracts.subset import DEPTH_CAP, GARBAGE
from tests.fixtures import build

from .util import effects, events, load_problem, problem, run, trace


def test_declaration_that_reads_itself_is_an_uninitialised_read():
    out = trace("int f(int n) {\n    int x = x + 1;\n    return x;\n}", "int f(int n)", [0])
    assert out["returned"] == GARBAGE + 1 and [e["var"] for e in events(out, "uninit_read")] == ["x"]


def test_every_read_of_an_unset_variable_fires():
    code = "int f(int n) {\n    int x;\n    int t = 0;\n    for (int i = 0; i < n; i++) {\n        t = x;\n    }\n    return t;\n}"
    assert len(events(trace(code, "int f(int n)", [3]), "uninit_read")) == 3


def test_declaration_inside_a_loop_is_unset_again_each_pass():
    code = ("int f(int n) {\n    int last = 0;\n    for (int i = 0; i < n; i++) {\n        int t;\n"
            "        t += i;\n        last = t;\n    }\n    return last;\n}")
    out = trace(code, "int f(int n)", [3])
    assert out["returned"] == GARBAGE + 2 and len(events(out, "uninit_read")) == 3


def test_accumulator_reset_in_loop_returns_last_element():
    code = ("int total_energy(int cells[], int n) {\n    int total = 0;\n    for (int i = 0; i < n; i++) {\n"
            "        total = 0;\n        total += cells[i];\n    }\n    return total;\n}")
    result = harness.run_tests(load_problem("P03_total_energy"), code)
    assert [r["got"]["returned"] for r in result["tests"]["results"]] == [6, 5, 4, 9, 1]
    assert [r["pass"] for r in result["tests"]["results"]] == [False, True, False, True, False]   # the adversarial test


def test_compound_assignment_on_a_cell_reads_then_writes_once():
    code = "void f(int a[], int n) {\n    int i = 0;\n    a[i++] += 5;\n    a[i] -= a[0];\n}"
    out = trace(code, "void f(int a[], int n)", [[1, 20], 2])
    assert [e for e in effects(out) if "cell" in e] == ["read_cell:0", "write_cell:0:6", "read_cell:1", "read_cell:0",
                                                        "write_cell:1:14"]
    assert run(code, "void f(int a[], int n)", [([[1, 20], 2], {"array0": [6, 14]})])["tests"]["passed"] == 1


def test_out_of_bounds_compound_assignment_reads_garbage_and_drops_the_write():
    out = trace("void f(int a[], int n) {\n    a[n] += 1;\n}", "void f(int a[], int n)", [[1], 1])
    assert [e["type"] for e in out["events"]] == ["oob_read", "oob_write"]
    assert [e for e in effects(out) if "cell" in e or "void" in e] == ["read_void:1"]


def test_int_and_float_mix_follows_c_typing():
    code = ("float f(int a, int b) {\n    int half = a / b;\n    float exact = a / (float)b;\n    int back = exact;\n"
            "    int scaled = a * 1.5;\n    return half + exact + back + scaled + (a > 2.5) + 7 / 2 * 2.0;\n}")
    out = trace(code, "float f(int a, int b)", [7, 2])
    assert out["returned"] == 3 + 3.5 + 3 + 10 + 1 + 6.0
    assert [e["into_float"] for e in events(out, "intdiv")] == [False, True]


def test_float_parameter_and_array():
    code = ("float f(float w[], int n, float scale) {\n    float t = 0;\n    for (int i = 0; i < n; i++) {\n"
            "        t += w[i] * scale;\n    }\n    w[0] = 9;\n    return t / n;\n}")
    result = run(code, "float f(float w[], int n, float scale)",
                 [([[1, 2.5], 2, 2], {"returned": 3.5, "array0": [9.0, 2.5]})])
    assert result["tests"]["passed"] == 1 and result["tests"]["results"][0]["got"]["returned"] == 3.5


def test_wrong_argument_shapes_are_parse_errors():
    cases = [("int g(int a[]) {\n    return a[0];\n}\nint f(int n) {\n    return g(n);\n}", 5),
             ("int g(int x) {\n    return x;\n}\nint f(int n) {\n    int a[2] = {1, 2};\n    return g(a);\n}", 6),
             ("int g(int x) {\n    return x;\n}\nint f(int n) {\n    return g(n, n);\n}", 5),
             ("void g(int x) {\n}\nint f(int n) {\n    return g(n) + 1;\n}", 4),
             ("int f(int n) {\n    int a[2] = {1, 2};\n    a = n;\n    return 0;\n}", 3),
             ("int f(int n) {\n    int n2 = fire(1);\n    return 0;\n}", 2)]
    for code, line in cases:
        report = harness.check(code, problem("int f(int n)", [[1]]))
        assert (report["status"], report["line"]) == ("parse_error", line), report


def test_unknown_printf_format_is_unsupported():
    report = harness.check("int f(int n) {\n    printf(\"%p\", n);\n    return 0;\n}", problem("int f(int n)", [[1]]))
    assert report["status"] == "unsupported" and report["line"] == 2 and report["construct"] == "printf"


def test_printf_mismatches_do_not_crash():
    code = ("int f(int n) {\n    char s[] = \"ab\";\n    printf(\"%d|%d|%s|%c\", n);\n"
            "    printf(\"|%f|%s|%d\", n, n, s);\n    return 0;\n}")
    out = trace(code, "int f(int n)", [5])
    assert out["status"] == "ok" and out["printed"] == f"5|{GARBAGE}|(null)|?|0.000000|(null)|{GARBAGE}"


def test_recursion_at_the_depth_cap_through_nested_blocks():
    code = ("int f(int n, int acc) {\n    int r = 0;\n    for (int i = 0; i < 1; i++) {\n        if (n > 1) {\n"
            "            while (r == 0) {\n                { r = (n > 0 ? f(n - 1, acc + (n % 2 == 0 ? 1 : 2)) : 0) + 1; }\n"
            "            }\n        } else {\n            r = acc;\n        }\n    }\n    return r;\n}")
    out = trace(code, "int f(int n, int acc)", [DEPTH_CAP, 0], [DEPTH_CAP + 1, 0])
    assert [t["status"] for t in out["per_test"]] == ["ok", "timeout"]
    assert out["per_test"][0]["returned"] == 50 + 49 * 2 + 99 and out["per_test"][0]["max_depth"] == DEPTH_CAP


def test_problem_without_signature_or_expect_uses_the_name():
    prob = {"name": "f", "tests": [{"args": [4]}], "display_test": 0}
    assert harness.trace(prob, "int f(int n) {\n    return n + 1;\n}")["returned"] == 5
    result = harness.run_tests(prob, "int f(int n) {\n    return n + 1;\n}")
    assert result["tests"] == {"passed": 1, "total": 1, "results": [{"args": [4], "expected": {}, "got": {}, "pass": True}]}


def test_main_and_helpers_around_the_entry_function_are_fine():
    code = ("#include <stdio.h>\nint door_open(int code) {\n    return code == 42;\n}\n"
            "int main() {\n    printf(\"%d\", door_open(42));\n    return 0;\n}")
    assert harness.run_tests(load_problem("P11_door_open"), code)["tests"]["passed"] == 5


def test_an_interpreter_bug_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(interp, "STRICT", False)

    def broken(self, code, forbid=()):
        raise KeyError("boom")

    monkeypatch.setattr(interp.Program, "__init__", broken)
    out = harness.trace(problem("int f(int n)", [[1]]), "int f(int n) {\n    return 77001;\n}")
    assert out["status"] == "unsupported" and out["steps"] == []


def test_threads_sharing_one_parsed_program_do_not_mix_state():
    prob = load_problem("P03_total_energy")
    codes = [build.P03_LE, prob["correct_variants"][0], build.P03_NO_UPDATE]
    expected = [harness.trace(prob, code) for code in codes]
    failures = []

    def worker(k):
        for _ in range(8):
            for code, want in zip(codes, expected):
                if harness.trace(prob, code) != want:
                    failures.append(k)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert failures == []
