"""One test (or more) per row of the table in plans/03 §2.2, in table order."""
from ml.contracts.subset import DEPTH_CAP, GARBAGE, STEP_CAP

from .util import effects, events, lines, run, trace


# ---- row 1: read of uninitialised local -> GARBAGE + uninit_read {var}

def test_row01_uninit_read():
    out = trace("int f(int n) {\n    int total;\n    total += n;\n    return total;\n}", "int f(int n)", [3])
    assert out["status"] == "ok" and out["returned"] == GARBAGE + 3
    assert events(out, "uninit_read") == [{"type": "uninit_read", "line": 3, "var": "total", "test": 0}]
    assert out["steps"][0]["vars"] == {"n": 3, "total": None}          # no value yet -> null


def test_row01_uninit_float_and_array_cell():
    out = trace("float f(int n) {\n    float x;\n    return x;\n}", "float f(int n)", [1])
    assert out["returned"] == float(GARBAGE) and events(out, "uninit_read")[0]["var"] == "x"
    out = trace("int f(int n) {\n    int a[3];\n    a[0] = 1;\n    return a[0] + a[2];\n}", "int f(int n)", [1])
    assert out["returned"] == GARBAGE + 1
    assert [(e["var"], e["line"]) for e in events(out, "uninit_read")] == [("a", 4)]


def test_row01_initialised_and_parameters_do_not_fire():
    out = trace("int f(int n) {\n    int t = 0;\n    t += n;\n    return t;\n}", "int f(int n)", [3])
    assert out["returned"] == 3 and out["events"] == []


# ---- row 2: array read out of bounds -> GARBAGE + oob_read {arr, idx, size}

def test_row02_oob_read():
    out = trace("int f(int a[], int n) {\n    return a[n];\n}", "int f(int a[], int n)", [[1, 2, 3], 3])
    assert out["returned"] == GARBAGE
    assert events(out, "oob_read") == [{"type": "oob_read", "line": 2, "arr": "a", "idx": 3, "size": 3, "test": 0}]


def test_row02_oob_read_negative_index_and_local_array():
    out = trace("int f(int a[], int n) {\n    int b[2] = {7, 8};\n    return a[-1] + b[2];\n}",
                "int f(int a[], int n)", [[1], 1])
    assert [(e["arr"], e["idx"], e["size"]) for e in events(out, "oob_read")] == [("a", -1, 1), ("b", 2, 2)]


# ---- row 3: array write out of bounds -> ignored + oob_write {arr, idx, size}

def test_row03_oob_write_is_ignored():
    code = "void f(int a[], int n) {\n    a[n] = 99;\n    a[-1] = 98;\n    a[0] = 5;\n}"
    out = trace(code, "void f(int a[], int n)", [[1, 2], 2])
    assert [(e["line"], e["arr"], e["idx"], e["size"]) for e in events(out, "oob_write")] == \
        [(2, "a", 2, 2), (3, "a", -1, 2)]
    assert effects(out).count("write_cell:0:5") == 1 and out["effects_count"]["write_cell"] == 1
    assert run(code, "void f(int a[], int n)", [([[1, 2], 2], {"array0": [5, 2]})])["tests"]["passed"] == 1


# ---- row 4: int / int -> truncation toward zero + intdiv {remainder_nonzero, into_float}

def test_row04_intdiv_truncates_toward_zero():
    code = "int f(int a, int b) {\n    return a / b;\n}"
    out = trace(code, "int f(int a, int b)", [7, 2], [-7, 2], [7, -2], [6, 3])
    assert [t["returned"] for t in out["per_test"]] == [3, -3, -3, 2]
    assert [(e["remainder_nonzero"], e["into_float"]) for e in events(out, "intdiv")] == \
        [(True, False), (True, False), (True, False), (False, False)]
    assert all(e["line"] == 2 for e in events(out, "intdiv"))


def test_row04_into_float_when_assigned_or_returned_as_float():
    code = ("float f(int sum, int n) {\n"
            "    float avg = sum / n;\n"            # assigned to a float
            "    int k = sum / n;\n"                # stays an int
            "    float late = (float)(sum / n);\n"  # the cast comes too late
            "    float fine = (float)sum / n;\n"    # float division: no event
            "    avg = sum / n * 1.0;\n"            # the int result is widened by the multiply
            "    return sum / n;\n"                 # returned as float
            "}")
    out = trace(code, "float f(int sum, int n)", [7, 2])
    assert [(e["line"], e["into_float"]) for e in events(out, "intdiv")] == \
        [(2, True), (3, False), (4, True), (6, True), (7, True)]
    assert out["returned"] == 3.0


def test_row04_modulo_follows_c():
    out = trace("int f(int a, int b) {\n    return a % b;\n}", "int f(int a, int b)", [7, 3], [-7, 3], [7, -3])
    assert [t["returned"] for t in out["per_test"]] == [1, -1, 1]
    assert events(out, "intdiv") == []


# ---- row 5: division by zero -> halt, runtime_error, div_zero

def test_row05_div_zero_halts():
    out = trace("int f(int a, int b) {\n    int x = 1;\n    x = a / b;\n    return x;\n}", "int f(int a, int b)", [4, 0])
    assert out["status"] == "runtime_error" and out["returned"] is None
    assert events(out, "div_zero") == [{"type": "div_zero", "line": 3, "test": 0}]
    assert out["steps"][-1]["line"] == 3 and out["steps"][-1]["events"][0]["type"] == "div_zero"
    assert "ret" not in out["effects_count"]


def test_row05_modulo_and_float_division_by_zero():
    assert trace("int f(int a) {\n    return a % 0;\n}", "int f(int a)", [4])["status"] == "runtime_error"
    out = trace("float f(float a) {\n    return a / 0.0;\n}", "float f(float a)", [4.0])
    assert out["status"] == "runtime_error" and events(out, "div_zero")


def test_row05_other_tests_still_run():
    out = trace("int f(int a, int b) {\n    return a / b;\n}", "int f(int a, int b)", [4, 2], [4, 0], [9, 3])
    assert [t["status"] for t in out["per_test"]] == ["ok", "runtime_error", "ok"]
    assert out["status"] == "runtime_error" and out["returned"] == 2
    assert [e["test"] for e in events(out, "div_zero")] == [1]


# ---- row 6: condition is an assignment -> evaluates normally + assign_in_cond {var, value}

def test_row06_assign_in_if_condition():
    code = "int f(int code) {\n    if (code = 42) {\n        return 1;\n    }\n    return 0;\n}"
    out = trace(code, "int f(int code)", [7], [42], [0])
    assert [t["returned"] for t in out["per_test"]] == [1, 1, 1]
    assert events(out, "assign_in_cond")[0] == {"type": "assign_in_cond", "line": 2, "var": "code", "value": 42,
                                                 "test": 0}
    assert out["branch"] == {"B2": {"true": 3, "false": 0}}
    assert out["steps"][0]["vars"] == {"code": 42}


def test_row06_assign_zero_in_while_condition():
    code = "int f(int k) {\n    int n = 0;\n    while (k = 0) {\n        n++;\n    }\n    return n;\n}"
    out = trace(code, "int f(int k)", [5])
    assert out["returned"] == 0 and out["loop_iters"] == {}
    assert [(e["var"], e["value"]) for e in events(out, "assign_in_cond")] == [("k", 0)]


def test_row06_comparison_and_nested_assignment_do_not_fire():
    code = "int f(int x) {\n    int y;\n    if ((y = x) == 5) {\n        return y;\n    }\n    return 0;\n}"
    out = trace(code, "int f(int x)", [5])
    assert out["returned"] == 5 and events(out, "assign_in_cond") == []


# ---- row 7: body is an EmptyStatement -> executes nothing + empty_body {kind}

def test_row07_empty_body_if():
    code = "int f(int code) {\n    if (code == 42); {\n        return 1;\n    }\n    return 0;\n}"
    out = trace(code, "int f(int code)", [7])
    assert out["returned"] == 1
    assert events(out, "empty_body") == [{"type": "empty_body", "line": 2, "kind": "if", "test": 0}]
    assert out["branch"] == {"B2": {"true": 0, "false": 1}}


def test_row07_empty_body_for():
    code = "int f(int n) {\n    int t = 0;\n    for (int i = 0; i < n; i++); {\n        t += 1;\n    }\n    return t;\n}"
    out = trace(code, "int f(int n)", [4])
    assert out["returned"] == 1 and out["loop_iters"] == {"L3": 4}
    assert [(e["line"], e["kind"]) for e in events(out, "empty_body")] == [(3, "for")]


def test_row07_empty_body_while():
    code = "int f(int n) {\n    int i = 0;\n    while (i < n); {\n        i++;\n    }\n    return i;\n}"
    out = trace(code, "int f(int n)", [3], [0])
    assert [t["status"] for t in out["per_test"]] == ["timeout", "ok"]
    assert [(e["kind"], e["test"]) for e in events(out, "empty_body")] == [("while", 0), ("while", 1)]
    assert out["per_test"][1]["returned"] == 1


def test_row07_braces_are_not_an_empty_body():
    out = trace("int f(int n) {\n    if (n > 0) {\n    }\n    return n;\n}", "int f(int n)", [3])
    assert events(out, "empty_body") == []


# ---- row 8: each loop iteration -> loop_iters["L<line>"] += 1

def test_row08_loop_counters():
    code = ("int f(int n) {\n    int t = 0;\n    for (int i = 0; i < n; i++) {\n"
            "        int j = 0;\n        while (j < 2) {\n            j++;\n            t++;\n        }\n    }\n"
            "    do {\n        t++;\n    } while (t < 0);\n    return t;\n}")
    out = trace(code, "int f(int n)", [3], [0])
    assert out["per_test"][0]["loop_iters"] == {"L3": 3, "L5": 6, "L10": 1}
    assert out["per_test"][1]["loop_iters"] == {"L10": 1}
    assert out["loop_iters"] == {"L3": 3, "L5": 6, "L10": 2}
    assert [t["returned"] for t in out["per_test"]] == [7, 1]


# ---- row 9: each if -> branch["B<line>"] = {true: n, false: m}

def test_row09_branch_counters():
    code = ("int f(int a[], int n) {\n    int c = 0;\n    for (int i = 0; i < n; i++) {\n"
            "        if (a[i] > 2) {\n            c++;\n        } else if (a[i] == 0) {\n            c--;\n        }\n"
            "    }\n    return c;\n}")
    out = trace(code, "int f(int a[], int n)", [[1, 5, 0, 9], 4], [[3], 1])
    assert out["per_test"][0]["branch"] == {"B4": {"true": 2, "false": 2}, "B6": {"true": 1, "false": 1}}
    assert out["per_test"][1]["branch"] == {"B4": {"true": 1, "false": 0}}
    assert out["branch"] == {"B4": {"true": 3, "false": 2}, "B6": {"true": 1, "false": 1}}


# ---- row 10: more than 5,000 executed statements -> halt, timeout, step_cap_hit

def test_row10_step_cap():
    code = "int f(int n) {\n    int i = 0;\n    while (i < n) {\n        n = n;\n    }\n    return i;\n}"
    out = trace(code, "int f(int n)", [1])
    assert out["status"] == "timeout" and out["returned"] is None
    # 1 declaration, then (test, body) pairs: statement 5,001 is the body of pass 2,500
    assert events(out, "step_cap_hit") == [{"type": "step_cap_hit", "line": 4, "test": 0}]
    assert out["loop_iters"] == {"L3": STEP_CAP // 2}
    assert out["truncated"] is True and len(out["steps"]) == 2000
    assert out["per_test"][0]["status"] == "timeout"


def test_row10_exactly_5000_statements_is_allowed():
    # 2 declarations + k * (test, body) + final test + return = 2k + 4 statements
    code = ("int f(int n) {\n    int i = 0;\n    int j = 0;\n    while (i < n) {\n        i++;\n    }\n"
            "    return i + j;\n}")
    k_ok = (STEP_CAP - 4) // 2
    out = trace(code, "int f(int n)", [k_ok], [k_ok + 1])
    assert 2 * k_ok + 4 == STEP_CAP
    assert [t["status"] for t in out["per_test"]] == ["ok", "timeout"]
    assert out["per_test"][0]["returned"] == k_ok and out["truncated"] is True


def test_row10_for_without_condition_and_empty_body_still_stops():
    out = trace("int f(int n) {\n    for (;;);\n    return 1;\n}", "int f(int n)", [1])
    assert out["status"] == "timeout" and events(out, "step_cap_hit")[0]["line"] == 2


# ---- row 11: non-void function falls off the end -> GARBAGE + missing_return

def test_row11_missing_return():
    code = "int f(int n) {\n    int r = n * 2;\n    printf(\"%d\", r);\n}"
    out = trace(code, "int f(int n)", [4])
    assert out["status"] == "ok" and out["returned"] == GARBAGE and out["printed"] == "8"
    assert events(out, "missing_return") == [{"type": "missing_return", "line": 4, "test": 0}]
    assert effects(out)[-1] == f"ret:f:1:{GARBAGE}"


def test_row11_missing_return_on_one_path_only():
    code = "int f(int n) {\n    if (n > 0) {\n        return 1;\n    }\n}"
    out = trace(code, "int f(int n)", [1], [0])
    assert [t["returned"] for t in out["per_test"]] == [1, GARBAGE]
    assert [e["test"] for e in events(out, "missing_return")] == [1]


def test_row11_void_function_end_is_not_an_event():
    out = trace("void f(int n) {\n    fire();\n}", "void f(int n)", [1])
    assert out["events"] == [] and out["returned"] is None
    assert out["steps"][-1]["line"] == 3 and out["steps"][-1]["effects"] == ["ret:f:1:"]


# ---- row 12: int overflow -> wrap to 32 bits + overflow

def test_row12_overflow_wraps():
    code = "int f(int n) {\n    int x = 2147483647;\n    x = x + n;\n    return x;\n}"
    out = trace(code, "int f(int n)", [1])
    assert out["returned"] == -2147483648
    assert events(out, "overflow") == [{"type": "overflow", "line": 3, "test": 0}]


def test_row12_overflow_in_multiply_increment_and_compound():
    out = trace("int f(int n) {\n    return n * n;\n}", "int f(int n)", [65536], [46341], [46340])
    assert [t["returned"] for t in out["per_test"]] == [0, -2147479015, 2147395600]
    assert [e["test"] for e in events(out, "overflow")] == [0, 1]
    out = trace("int f(int n) {\n    n++;\n    return n;\n}", "int f(int n)", [2147483647])
    assert out["returned"] == -2147483648 and len(events(out, "overflow")) == 1
    out = trace("int f(int n) {\n    n -= 10;\n    return n;\n}", "int f(int n)", [-2147483640])
    assert out["returned"] == 2147483646 and len(events(out, "overflow")) == 1


def test_row12_factorial_13_wraps_like_32_bit_c():
    code = "int f(int n) {\n    int r = 1;\n    for (int i = 2; i <= n; i++) {\n        r *= i;\n    }\n    return r;\n}"
    out = trace(code, "int f(int n)", [13], [12])
    assert [t["returned"] for t in out["per_test"]] == [1932053504, 479001600]


# ---- row 13: array param read -> read_cell {i}, or read_void {i} when out of bounds

def test_row13_read_cell_and_read_void():
    code = "int f(int cells[], int n) {\n    int t = 0;\n    for (int i = 0; i <= n; i++) {\n        t += cells[i];\n    }\n    return t;\n}"
    out = trace(code, "int f(int cells[], int n)", [[2, 4], 2])
    assert [e for e in effects(out) if e.startswith("read")] == ["read_cell:0", "read_cell:1", "read_void:2"]
    assert out["effects_count"]["read_cell"] == 2 and out["effects_count"]["read_void"] == 1
    assert out["returned"] == GARBAGE + 6


def test_row13_local_arrays_make_no_world_effects():
    code = "int f(int n) {\n    int a[3] = {1, 2, 3};\n    a[1] = 9;\n    return a[0] + a[1] + a[5];\n}"
    out = trace(code, "int f(int n)", [1])
    assert out["effects_count"]["read_cell"] == 0 and out["effects_count"]["write_cell"] == 0
    assert out["effects_count"]["read_void"] == 0 and len(events(out, "oob_read")) == 1


# ---- row 14: array param write -> performed, shared with the caller + write_cell {i, v}

def test_row14_write_cell_is_shared_with_the_caller():
    code = ("void set(int a[], int i, int v) {\n    a[i] = v;\n}\n"
            "int f(int a[], int n) {\n    set(a, 0, 7);\n    a[1] += 5;\n    a[2]++;\n    return a[0] + a[1] + a[2];\n}")
    out = trace(code, "int f(int a[], int n)", [[1, 2, 3], 3])
    assert out["returned"] == 7 + 7 + 4
    assert [e for e in effects(out) if e.startswith("write")] == ["write_cell:0:7", "write_cell:1:7", "write_cell:2:4"]
    assert out["effects_count"]["write_cell"] == 3
    assert run(code, "int f(int a[], int n)", [([[1, 2, 3], 3], {"array0": [7, 7, 4]})])["tests"]["passed"] == 1


def test_row14_scalars_are_passed_by_value():
    code = "void bump(int x) {\n    x = x + 10;\n}\nint f(int n) {\n    bump(n);\n    return n;\n}"
    assert trace(code, "int f(int n)", [1])["returned"] == 1


# ---- row 15: relational compare of two array cells -> compare {i, j}

def test_row15_compare_effect():
    code = "int f(int a[], int n) {\n    int c = 0;\n    for (int j = 0; j < n - 1; j++) {\n        if (a[j] > a[j + 1]) {\n            c++;\n        }\n    }\n    return c;\n}"
    out = trace(code, "int f(int a[], int n)", [[3, 1, 2], 3])
    step = next(s for s in out["steps"] if s["line"] == 4)
    assert step["effects"] == ["read_cell:0", "read_cell:1", "compare:0:1"]
    assert out["effects_count"]["compare"] == 2 and out["returned"] == 1


def test_row15_cell_against_a_scalar_is_not_a_compare():
    out = trace("int f(int a[], int x) {\n    return a[0] == x;\n}", "int f(int a[], int x)", [[3], 3])
    assert out["effects_count"]["compare"] == 0 and out["returned"] == 1


# ---- row 16: string literal initialising char s[] -> char codes + 0 terminator

def test_row16_string_literal_initialiser():
    code = ("int f(int n) {\n    char s[] = \"level\";\n    int len = 0;\n"
            "    while (s[len] != '\\0') {\n        len++;\n    }\n    return len * 1000 + s[0] + s[5];\n}")
    out = trace(code, "int f(int n)", [0])
    assert out["returned"] == 5 * 1000 + ord("l") + 0 and out["events"] == []


# ---- row 17: == / != with a string literal or two array names -> false / true + event

def test_row17_string_literal_compare():
    code = "int f(char s[]) {\n    int v = 0;\n    if (s[0] == \"a\") {\n        v += 1;\n    }\n    if (s[0] != \"a\") {\n        v += 2;\n    }\n    return v;\n}"
    out = trace(code, "int f(char s[])", ["abc"])
    assert out["returned"] == 2
    assert events(out, "str_literal_compare") == [{"type": "str_literal_compare", "line": 3, "test": 0},
                                                   {"type": "str_literal_compare", "line": 6, "test": 0}]
    assert out["branch"] == {"B3": {"true": 0, "false": 1}, "B6": {"true": 1, "false": 0}}


def test_row17_array_compare():
    code = "int f(char s[], char t[]) {\n    if (s == t) {\n        return 1;\n    }\n    return s != t;\n}"
    out = trace(code, "int f(char s[], char t[])", ["abc", "abc"])
    assert out["returned"] == 1                       # s != t is true: two different arrays
    assert [(e["type"], e["line"]) for e in out["events"]] == [("array_compare", 2), ("array_compare", 5)]
    assert out["branch"] == {"B2": {"true": 0, "false": 1}}


# ---- row 18: function call / return -> call {fn, depth, args}, ret {fn, depth, value}, max_depth

def test_row18_call_and_ret_effects():
    code = "int factorial(int n) {\n    if (n <= 1) {\n        return 1;\n    }\n    return n * factorial(n - 1);\n}"
    out = trace(code, "int factorial(int n)", [3], [1])
    assert effects(out) == ["call:factorial:1:3", "call:factorial:2:2", "call:factorial:3:1",
                            "ret:factorial:3:1", "ret:factorial:2:2", "ret:factorial:1:6"]
    assert lines(out) == [2, 2, 2, 3, 5, 5]
    assert [s["vars"] for s in out["steps"]] == [{"n": 3}, {"n": 2}, {"n": 1}, {"n": 1}, {"n": 2}, {"n": 3}]
    assert out["max_depth"] == 3 and [t["max_depth"] for t in out["per_test"]] == [3, 1]
    assert out["effects_count"]["call"] == 4 and out["effects_count"]["ret"] == 4


def test_row18_helper_call_args_and_array_names():
    code = ("int add(int a[], int i, float w) {\n    return a[i];\n}\n"
            "int f(int cells[], int n) {\n    int t = add(cells, 1, 0.5);\n    return t;\n}")
    out = trace(code, "int f(int cells[], int n)", [[4, 5], 2])
    assert effects(out) == ["call:f:1:cells,2", "call:add:2:a,1,0.5", "read_cell:1", "ret:add:2:5", "ret:f:1:5"]
    assert out["max_depth"] == 2


# ---- row 19: recursion deeper than 100 -> halt, timeout, depth_cap_hit {fn, last_args}

def test_row19_depth_cap():
    code = "int f(int n) {\n    return n + f(n + 1);\n}"
    out = trace(code, "int f(int n)", [1])
    assert out["status"] == "timeout" and out["returned"] is None
    assert events(out, "depth_cap_hit") == [{"type": "depth_cap_hit", "line": 2, "fn": "f",
                                             "last_args": [DEPTH_CAP + 1], "test": 0}]
    assert out["max_depth"] == DEPTH_CAP and out["effects_count"]["call"] == DEPTH_CAP
    assert effects(out) == [f"call:f:{d}:{d}" for d in range(1, DEPTH_CAP + 1)]


def test_row19_depth_100_itself_is_allowed():
    code = "int f(int n) {\n    if (n <= 1) {\n        return 1;\n    }\n    return 1 + f(n - 1);\n}"
    out = trace(code, "int f(int n)", [DEPTH_CAP], [DEPTH_CAP + 1])
    assert [t["status"] for t in out["per_test"]] == ["ok", "timeout"]
    assert out["per_test"][0]["returned"] == DEPTH_CAP and out["per_test"][0]["max_depth"] == DEPTH_CAP


# ---- row 20: call result unused -> discarded_call_value {fn, line}

def test_row20_discarded_call_value():
    code = "int f(int n) {\n    if (n <= 1) {\n        return 1;\n    }\n    f(n - 1);\n    return n;\n}"
    out = trace(code, "int f(int n)", [3])
    assert out["returned"] == 3
    assert events(out, "discarded_call_value") == [{"type": "discarded_call_value", "line": 5, "fn": "f", "test": 0}] * 2


def test_row20_used_values_and_void_calls_do_not_fire():
    code = ("void log_it(int n) {\n    scan(n);\n}\nint twice(int n) {\n    return 2 * n;\n}\n"
            "int f(int n) {\n    int x = twice(n);\n    log_it(x);\n    x = twice(x);\n    fire();\n    return x;\n}")
    out = trace(code, "int f(int n)", [3])
    assert out["returned"] == 12 and events(out, "discarded_call_value") == []


# ---- row 21: strlen in a problem whose forbid lists it -> not executed (gate G4b)

def test_row21_strlen_forbidden():
    code = "int str_length(char s[]) {\n    return strlen(s);\n}"
    allowed = trace(code, "int str_length(char s[])", ["hello"])
    assert allowed["status"] == "ok" and allowed["returned"] == 5
    refused = trace(code, "int str_length(char s[])", ["hello"], forbid=["strlen"])
    assert refused["status"] == "unsupported" and refused["steps"] == [] and refused["returned"] is None
    result = run(code, "int str_length(char s[])", [(["hello"], {"returned": 5})], forbid=["strlen"])
    assert result["status"] == "unsupported" and result["tests"]["passed"] == 0
