"""Step rules written in ml/contracts/subset.py and notes/W0.md (decisions 2-5)."""
from ml.contracts.subset import MAX_RECORDED_STEPS

from .util import effects, lines, trace


def test_one_step_per_finished_statement():
    code = "int f(int n) {\n    int a = 1;\n    int b;\n    b = a + n;\n    a++;\n    return a + b;\n}"
    out = trace(code, "int f(int n)", [5])
    assert lines(out) == [2, 3, 4, 5, 6]
    assert [s["vars"] for s in out["steps"]] == [
        {"n": 5, "a": 1, "b": None}, {"n": 5, "a": 1, "b": None}, {"n": 5, "a": 1, "b": 6},
        {"n": 5, "a": 2, "b": 6}, {"n": 5, "a": 2, "b": 6}]


def test_for_header_gives_init_test_and_update_steps():
    code = "int f(int n) {\n    int t = 0;\n    for (int i = 0; i < n; i++) {\n        t += i;\n    }\n    return t;\n}"
    out = trace(code, "int f(int n)", [2])
    #            decl init test body upd test body upd test ret
    assert lines(out) == [2, 3, 3, 4, 3, 3, 4, 3, 3, 6]
    assert [s["vars"]["i"] for s in out["steps"]] == [None, 0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert out["steps"][0]["vars"] == {"n": 2, "t": 0, "i": None}          # declared later: null


def test_while_and_if_conditions_are_steps():
    code = ("int f(int n) {\n    int i = 0;\n    while (i < n) {\n        if (i == 1) {\n            n--;\n        }\n"
            "        i++;\n    }\n    return i;\n}")
    out = trace(code, "int f(int n)", [3])
    assert lines(out) == [2, 3, 4, 7, 3, 4, 5, 7, 3, 9]


def test_do_while_condition_step_is_on_the_while_line():
    code = "int f(int n) {\n    do {\n        n--;\n    }\n    while (n > 0);\n    return n;\n}"
    out = trace(code, "int f(int n)", [2])
    assert lines(out) == [3, 5, 3, 5, 6] and out["loop_iters"] == {"L2": 2}


def test_one_declaration_with_several_names_is_one_step():
    code = "int f(int n) {\n    int low = 0, high = n - 1, mid;\n    int a; int b;\n    return low + high;\n}"
    out = trace(code, "int f(int n)", [5])
    assert lines(out) == [2, 3, 3, 4]
    assert out["steps"][0]["vars"] == {"n": 5, "low": 0, "high": 4, "mid": None, "a": None, "b": None}


def test_vars_hold_scalars_of_the_current_frame_only():
    code = ("int helper(int k) {\n    int twice = k * 2;\n    return twice;\n}\n"
            "int f(int a[], int n) {\n    int local[2] = {1, 2};\n    int x = helper(n);\n    return x + local[0];\n}")
    out = trace(code, "int f(int a[], int n)", [[9], 4])
    assert [(s["line"], s["vars"]) for s in out["steps"]] == [
        (6, {"n": 4, "x": None}),                      # arrays `a` and `local` are not in vars
        (2, {"k": 4, "twice": 8}),
        (3, {"k": 4, "twice": 8}),
        (7, {"n": 4, "x": 8}),
        (8, {"n": 4, "x": 8}),
    ]


def test_call_effect_on_first_step_of_callee_and_ret_on_its_return_step():
    code = ("int helper(int k) {\n    int twice = k * 2;\n    return twice;\n}\n"
            "int f(int n) {\n    int x = helper(n);\n    return x;\n}")
    out = trace(code, "int f(int n)", [4])
    assert [(s["line"], s["effects"]) for s in out["steps"]] == [
        (6, ["call:f:1:4"]),                 # the caller's effects so far are closed before the call
        (2, ["call:helper:2:4"]),
        (3, ["ret:helper:2:8"]),
        (6, []),                             # the caller's own step comes after the callee's steps
        (7, ["ret:f:1:8"]),
    ]


def test_effects_stay_in_the_order_they_happened():
    code = ("int pick(int a[], int i) {\n    return a[i];\n}\n"
            "int f(int a[], int n) {\n    return a[0] + pick(a, 1) + a[2];\n}")
    out = trace(code, "int f(int a[], int n)", [[5, 6, 7], 3])
    assert effects(out) == ["call:f:1:a,3", "read_cell:0", "call:pick:2:a,1", "read_cell:1", "ret:pick:2:6",
                            "read_cell:2", "ret:f:1:18"]
    assert out["returned"] == 18


def test_one_line_recursion_keeps_calls_before_returns():
    code = "int f(int n) {\n    return n <= 1 ? 1 : n * f(n - 1);\n}"
    out = trace(code, "int f(int n)", [3])
    assert effects(out) == ["call:f:1:3", "call:f:2:2", "call:f:3:1", "ret:f:3:1", "ret:f:2:2", "ret:f:1:6"]
    assert [s["vars"]["n"] for s in out["steps"]] == [3, 2, 1, 2, 3]


def test_void_function_gets_a_closing_step_with_its_ret():
    code = "void helper(int a[], int i) {\n    a[i] = 0;\n}\nvoid f(int a[], int n) {\n    helper(a, 0);\n}"
    out = trace(code, "void f(int a[], int n)", [[4, 5], 2])
    assert [(s["line"], s["effects"]) for s in out["steps"]] == [
        (5, ["call:f:1:a,2"]), (2, ["call:helper:2:a,0", "write_cell:0:0"]), (3, ["ret:helper:2:"]), (5, []),
        (6, ["ret:f:1:"])]
    assert out["effects_count"]["call"] == out["effects_count"]["ret"] == 2


def test_empty_function_body_still_shows_call_and_ret():
    out = trace("void f(int n) {\n}", "void f(int n)", [1])
    assert out["steps"] == [{"i": 0, "line": 2, "vars": {"n": 1}, "events": [], "effects": ["call:f:1:1", "ret:f:1:"]}]


def test_halt_records_a_last_step_with_the_event():
    out = trace("int f(int n) {\n    int x = 3;\n    x = x / n;\n    return x;\n}", "int f(int n)", [0])
    assert out["steps"][-1] == {"i": 1, "line": 3, "vars": {"n": 0, "x": 3},
                                "events": [{"type": "div_zero", "line": 3}], "effects": []}


def test_shadowed_name_shows_the_live_variable():
    code = ("int f(int n) {\n    int x = 1;\n    for (int i = 0; i < 1; i++) {\n        int x = 50;\n        x++;\n    }\n"
            "    x++;\n    return x;\n}")
    out = trace(code, "int f(int n)", [0])
    by_line = {}
    for step in out["steps"]:
        by_line.setdefault(step["line"], step["vars"]["x"])
    assert by_line[2] == 1 and by_line[4] == 50 and by_line[5] == 51 and by_line[7] == 2 and out["returned"] == 2


def test_two_loops_with_the_same_counter_name():
    code = ("int f(int n) {\n    int t = 0;\n    for (int i = 0; i < 2; i++) {\n        t += i;\n    }\n"
            "    for (int i = 10; i < 11; i++) {\n        t += i;\n    }\n    return t;\n}")
    out = trace(code, "int f(int n)", [0])
    assert [s["vars"]["i"] for s in out["steps"]] == [None, 0, 0, 0, 1, 1, 1, 2, 2, 10, 10, 10, 11, 11, 11]
    assert out["returned"] == 11


def test_recorded_steps_are_capped():
    code = "int f(int n) {\n    int t = 0;\n    for (int i = 0; i < n; i++) {\n        t += i;\n    }\n    return t;\n}"
    out = trace(code, "int f(int n)", [1000], [3])
    assert out["status"] == "ok" and out["returned"] == 499500
    assert len(out["steps"]) == MAX_RECORDED_STEPS and out["truncated"] is True
    assert out["loop_iters"] == {"L3": 1003}                              # counters are never truncated
    short = trace(code, "int f(int n)", [3], [1000])
    assert short["truncated"] is False and len(short["steps"]) == 13


def test_float_values_in_vars_and_effects():
    code = "float f(float x) {\n    float y = x / 4;\n    return y;\n}"
    out = trace(code, "float f(float x)", [1.0])
    assert out["steps"][0]["vars"] == {"x": 1.0, "y": 0.25}
    assert effects(out) == ["call:f:1:1", "ret:f:1:0.25"] and out["returned"] == 0.25
