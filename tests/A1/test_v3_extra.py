"""The 'v3 extra unit tests' list of ml_plan/03 §2.5, plus the rest of the string support."""
from ml.contracts.subset import DEPTH_CAP

from .util import effects, events, run, trace


# ---- string literal init + terminator

def test_string_literal_init_has_terminator():
    code = ("int f(int n) {\n    char s[] = \"hi\";\n    char t[6] = \"ab\";\n"
            "    return s[2] + t[2] + t[5] + sizeof(s) * 100 + sizeof(t) * 10000;\n}")
    out = trace(code, "int f(int n)", [0])
    assert out["returned"] == 0 + 3 * 100 + 6 * 10000 and out["events"] == []


def test_str_argument_becomes_char_array_with_terminator():
    code = "int f(char s[]) {\n    int i = 0;\n    while (s[i] != '\\0') {\n        i++;\n    }\n    return i;\n}"
    out = trace(code, "int f(char s[])", ["level"], [""])
    assert [t["returned"] for t in out["per_test"]] == [5, 0]
    assert out["per_test"][0]["effects_count"]["read_cell"] == 6        # five letters and the terminator
    assert effects(out)[0] == "call:f:1:s" and events(out, "oob_read") == []


def test_reading_past_the_terminator_is_out_of_bounds():
    out = trace("int f(char s[]) {\n    return s[4];\n}", "int f(char s[])", ["abc"])
    assert [(e["idx"], e["size"]) for e in events(out, "oob_read")] == [(4, 4)]


def test_printf_percent_s_and_percent_c():
    code = ("void f(char s[]) {\n    char t[] = \"ship\";\n"
            "    printf(\"%s-%s-%c%c\\n\", s, t, s[0], t[3]);\n    printf(\"%s\", \"ok\");\n}")
    assert trace(code, "void f(char s[])", ["core"])["printed"] == "core-ship-cp\nok"


def test_char_arrays_can_be_edited_in_place():
    code = ("void f(char s[]) {\n    int n = 0;\n    while (s[n] != '\\0') {\n        n++;\n    }\n"
            "    for (int i = 0; i < n / 2; i++) {\n        char c = s[i];\n        s[i] = s[n - 1 - i];\n"
            "        s[n - 1 - i] = c;\n    }\n}")
    result = run(code, "void f(char s[])", [(["abcd"], {"array0": "dcba"}), (["x"], {"array0": "x"})])
    assert result["tests"]["passed"] == 2 and result["tests"]["results"][0]["got"] == {"array0": "dcba"}


# ---- `s[i] == "a"` -> false + event

def test_cell_compared_with_string_literal_is_false():
    code = ("int count_a(char s[]) {\n    int c = 0;\n    for (int i = 0; s[i] != '\\0'; i++) {\n"
            "        if (s[i] == \"a\") {\n            c++;\n        }\n    }\n    return c;\n}")
    out = trace(code, "int count_a(char s[])", ["banana"])
    assert out["returned"] == 0
    assert len(events(out, "str_literal_compare")) == 6 and events(out, "str_literal_compare")[0]["line"] == 4
    assert out["branch"]["B4"] == {"true": 0, "false": 6}
    fixed = trace(code.replace('"a"', "'a'"), "int count_a(char s[])", ["banana"])
    assert fixed["returned"] == 3 and fixed["events"] == []


def test_whole_array_compared_with_string_literal():
    out = trace("int f(char s[]) {\n    return s == \"abc\";\n}", "int f(char s[])", ["abc"])
    assert out["returned"] == 0 and [e["type"] for e in out["events"]] == ["str_literal_compare"]


# ---- in-place sort visible to the caller

BUBBLE = ("void bubble_sort(int a[], int n) {\n    for (int i = 0; i < n - 1; i++) {\n"
          "        for (int j = 0; j < n - 1 - i; j++) {\n            if (a[j] > a[j + 1]) {\n"
          "                int t = a[j];\n                a[j] = a[j + 1];\n                a[j + 1] = t;\n"
          "            }\n        }\n    }\n}")


def test_in_place_sort_is_visible_to_the_harness():
    tests = [([[3, 1, 2], 3], {"array0": [1, 2, 3]}), ([[4, 3, 2, 1], 4], {"array0": [1, 2, 3, 4]}),
             ([[5], 1], {"array0": [5]})]
    result = run(BUBBLE, "void bubble_sort(int a[], int n)", tests)
    assert result["tests"]["passed"] == 3
    assert result["tests"]["results"][0]["got"] == {"array0": [1, 2, 3]}
    assert tests[0][0][0] == [3, 1, 2]                 # the problem's own args are not touched


def test_in_place_sort_is_visible_to_a_calling_function():
    code = BUBBLE + ("\nint f(int n) {\n    int v[4] = {9, 7, 8, 1};\n    bubble_sort(v, 4);\n"
                     "    return v[0] * 1000 + v[1] * 100 + v[2] * 10 + v[3];\n}")
    assert trace(code, "int f(int n)", [0])["returned"] == 1789


def test_swap_without_temp_duplicates_a_value():
    code = BUBBLE.replace("int t = a[j];\n", "").replace("a[j + 1] = t;", "a[j + 1] = a[j];")
    result = run(code, "void bubble_sort(int a[], int n)", [([[3, 1, 2], 3], {"array0": [1, 2, 3]})])
    assert result["tests"]["passed"] == 0 and result["tests"]["results"][0]["got"] == {"array0": [1, 1, 2]}
    out = trace(code, "void bubble_sort(int a[], int n)", [[3, 1, 2], 3])
    swap = [s["effects"] for s in out["steps"] if s["line"] in (5, 6)][:2]
    assert swap == [["read_cell:1", "write_cell:0:1"], ["read_cell:0", "write_cell:1:1"]]


# ---- depth_cap_hit on missing base case

def test_depth_cap_hit_on_missing_base_case():
    code = "int factorial(int n) {\n    return n * factorial(n - 1);\n}"
    out = trace(code, "int factorial(int n)", [3], [0])
    assert out["status"] == "timeout" and [t["status"] for t in out["per_test"]] == ["timeout", "timeout"]
    hits = events(out, "depth_cap_hit")
    assert [(e["fn"], e["last_args"], e["test"]) for e in hits] == [("factorial", [3 - DEPTH_CAP], 0),
                                                                    ("factorial", [-DEPTH_CAP], 1)]
    assert out["max_depth"] == DEPTH_CAP and "ret" not in out["effects_count"]


def test_same_argument_recursion_shows_constant_args():
    out = trace("int f(int n) {\n    if (n == 0) {\n        return 0;\n    }\n    return 1 + f(n);\n}",
                "int f(int n)", [4])
    calls = [e for e in effects(out) if e.startswith("call")]
    assert out["status"] == "timeout" and len(calls) == DEPTH_CAP
    assert {e.split(":")[3] for e in calls} == {"4"} and events(out, "depth_cap_hit")[0]["last_args"] == [4]


# ---- discarded_call_value

def test_discarded_call_value_top_frame_only():
    code = "int factorial(int n) {\n    if (n <= 1) {\n        return 1;\n    }\n    factorial(n - 1);\n    return n;\n}"
    out = trace(code, "int factorial(int n)", [5], [1])
    assert [t["returned"] for t in out["per_test"]] == [5, 1]
    assert [(e["fn"], e["line"], e["test"]) for e in events(out, "discarded_call_value")] == [("factorial", 5, 0)] * 4


# ---- strlen forbid

def test_strlen_forbid():
    code = "int str_length(char s[]) {\n    int n = strlen(s);\n    return n;\n}"
    assert run(code, "int str_length(char s[])", [(["abc"], {"returned": 3})])["tests"]["passed"] == 1
    refused = run(code, "int str_length(char s[])", [(["abc"], {"returned": 3})], forbid=["strlen"])
    assert refused["status"] == "unsupported" and refused["tests"] == {
        "passed": 0, "total": 1, "results": [{"args": ["abc"], "expected": {"returned": 3}, "got": {}, "pass": False}]}


def test_strlen_on_literal_and_missing_terminator():
    out = trace("int f(int n) {\n    char s[2] = \"hi\";\n    return strlen(\"four\") * 10 + strlen(s);\n}",
                "int f(int n)", [0])
    assert out["returned"] == 42 and [e["type"] for e in out["events"]] == ["oob_read"]
