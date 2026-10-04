"""The frozen C subset of ml_plan/03 §2.1: what runs, and what is rejected (gate G4)."""
import pytest

from ml.c_interp import harness
from ml.contracts.subset import REJECTED

from .util import effects, events, problem, trace

P = problem("int f(int n)", [[1]])


def value(body, n=1, signature="int f(int n)"):
    """Run `signature { body }` on n; no events may fire unless the test asks for them."""
    out = trace(signature + " {\n" + body + "\n}", signature, [n])
    assert out["status"] == "ok", out
    return out["returned"]


# ---------------------------------------------------------------- supported

def test_types_int_float_double_char():
    assert value("int a = 7; float b = 2; double c = 0.5; char d = 'A'; return a + b + c + d;") == 74
    assert value("float x = 7; return x / 2;", signature="float f(int n)") == 3.5
    assert value("double x = 1e2; return x + .5;", signature="double f(int n)") == 100.5
    assert value("char c = 'a'; c = c + 1; return c;") == ord("b")
    assert value(r"return '\n' + '\0' + '\\' + '\t';") == 10 + 0 + 92 + 9
    assert value("long a = 5L; unsigned int b = 3u; short c = 2; return a * b + c + 0x10 + 010;") == 15 + 2 + 16 + 8


def test_float_to_int_conversion_truncates_toward_zero():
    assert value("int a = 3.9; int b = -3.9; return a * 10 + b;") == 27
    assert value("float x = 7.9; return (int)x + (int)-7.9;") == 0
    assert value("float x = n; x = x / 2; return x * 4;", n=3) == 6


def test_array_declarations_and_initialisers():
    assert value("int a[5]; a[0] = 4; a[4] = 6; return a[0] + a[4];") == 10
    assert value("int a[] = {1, 2, 3}; return a[0] + a[1] * a[2];") == 7
    assert value("int a[5] = {1, 2}; return a[1] + a[4];") == 2                 # the rest are zero
    assert value("float w[2] = {1, 2.5}; return w[0] + w[1];", signature="float f(int n)") == 3.5
    assert value("int a[n]; a[n - 1] = 8; return a[n - 1];", n=4) == 8
    assert value("int a[3] = {0}; int i = 1; a[i]++; a[i] += 4; ++a[i]; a[i] *= 2; return a[i]--;") == 12


def test_arithmetic_and_compound_assignment():
    assert value("return 2 + 3 * 4 - 10 / 4 - 10 % 4;") == 2 + 12 - 2 - 2
    assert value("int x = 10; x += 5; x -= 3; x *= 4; x /= 5; x %= 7; return x;") == 2
    assert value("return -n + +n - (-5);", n=3) == 5
    assert value("float x = 1; x += 1; x *= 2.5; x -= 1; x /= 2; return x;", signature="float f(int n)") == 2.0


def test_increment_and_decrement():
    assert value("int i = 5; int a = i++; int b = ++i; int c = i--; int d = --i; return a*1000 + b*100 + c*10 + d;") == \
        5 * 1000 + 7 * 100 + 7 * 10 + 5
    assert value("float x = 1.5; x++; ++x; x--; return x;", signature="float f(int n)") == 2.5


def test_relational_and_logical_operators():
    assert value("return (1 < 2) + (2 <= 2) * 2 + (3 > 4) * 4 + (4 >= 5) * 8 + (5 == 5) * 16 + (5 != 5) * 32;") == 19
    assert value("return (n && 0) + (n || 0) * 2 + !n * 4 + !0 * 8;") == 10
    assert value("return 0 < n < 10;", n=50) == 1                                # (0 < 50) < 10
    assert value("return n == 42 || 7;", n=0) == 1


def test_short_circuit():
    code = ("int hit(int x) {\n    fire();\n    return x;\n}\n"
            "int f(int n) {\n    int a = n && hit(1);\n    int b = n || hit(1);\n    return a * 10 + b;\n}")
    zero = trace(code, "int f(int n)", [0])
    assert zero["returned"] == 1 and zero["effects_count"]["fire"] == 1         # only `||` evaluates hit()
    one = trace(code, "int f(int n)", [5])
    assert one["returned"] == 11 and one["effects_count"]["fire"] == 1          # only `&&` evaluates hit()
    safe = trace("int f(int n) {\n    if (n != 0 && 10 / n > 1) {\n        return 1;\n    }\n    return 0;\n}",
                 "int f(int n)", [0])
    assert safe["status"] == "ok" and safe["returned"] == 0


def test_ternary_and_casts():
    assert value("return n > 2 ? 10 : 20;", n=3) == 10
    assert value("return n > 2 ? 1 : 2.5;", n=3, signature="float f(int n)") == 1.0
    assert value("return (float)n / 2;", n=3, signature="float f(int n)") == 1.5
    assert value("return (double)n / 2;", n=3, signature="double f(int n)") == 1.5
    assert value("return (int)(n * 1.5);", n=3) == 4


def test_if_else_chain():
    body = "if (n < 30) { return 0; } else if (n < 70) { return 1; } else { return 2; }"
    assert [value(body, n=n) for n in (10, 30, 69, 70)] == [0, 1, 1, 2]
    assert value("int r = 0; if (n) r = 1; else r = 2; return r;", n=0) == 2


def test_for_while_do_while_break_continue():
    assert value("int t = 0; for (int i = 0; i < n; i++) { t += i; } return t;", n=5) == 10
    assert value("int t = 0; int i; for (i = n; i > 0; i--) t += i; return t + i;", n=4) == 10
    assert value("int t = 0, i = 0, j = n; for (i = 0, j = n; i < j; i++, j--) t++; return t;", n=6) == 3
    assert value("int i = 0; while (i * i < n) i++; return i;", n=17) == 5
    assert value("int i = 0; do { i += 3; } while (i < n); return i;", n=0) == 3
    assert value("int t = 0; for (int i = 0; i < 10; i++) { if (i == n) break; if (i % 2) continue; t += i; } return t;",
                 n=7) == 0 + 2 + 4 + 6
    assert value("int t = 0; while (1) { t++; if (t >= n) break; } return t;", n=4) == 4
    assert value("int t = 0; for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) { if (j == 1) continue; t++; } return t;") == 6


def test_user_functions_and_recursion():
    code = ("int square(int x) {\n    return x * x;\n}\n"
            "float half(int x) {\n    return x / 2.0;\n}\n"
            "int fib(int n) {\n    if (n < 2) return n;\n    return fib(n - 1) + fib(n - 2);\n}\n"
            "int is_even(int n);\n"
            "int is_odd(int n) {\n    if (n == 0) return 0;\n    return is_even(n - 1);\n}\n"
            "int is_even(int n) {\n    if (n == 0) return 1;\n    return is_odd(n - 1);\n}\n"
            "int f(int n) {\n    return square(n) + (int)(half(n) * 10) + fib(n) * 1000 + is_even(n) * 100000;\n}")
    out = trace(code, "int f(int n)", [6])
    assert out["returned"] == 36 + 30 + 8 * 1000 + 100000 and out["max_depth"] == 8


def test_block_scope_and_shadowing():
    assert value("int x = 1; { int x = 2; x++; } return x;") == 1
    assert value("int t = 0; for (int i = 0; i < 2; i++) { int x = 10; x += i; t += x; } return t;") == 21
    assert value("int i = 9; for (int i = 0; i < 3; i++) { } return i;") == 9


def test_global_variables():
    code = "int base = 10;\nint table[3] = {1, 2, 3};\nint counter;\nint f(int n) {\n    counter++;\n    base += n;\n    return base + table[2] + counter;\n}"
    out = trace(code, "int f(int n)", [5], [5])
    assert [t["returned"] for t in out["per_test"]] == [19, 19]          # globals start fresh in every test
    assert out["steps"][0]["vars"] == {"n": 5}


def test_world_builtins_append_effects():
    code = ("void f(int n) {\n    for (int i = 0; i < n; i++) {\n        fire();\n    }\n"
            "    launch();\n    open_door();\n    close_door();\n    scan(n * 2);\n}")
    out = trace(code, "void f(int n)", [3])
    assert [e for e in effects(out) if not e.startswith(("call", "ret"))] == \
        ["fire", "fire", "fire", "launch", "door_open", "door_closed", "scan:6"]
    assert out["effects_count"] == {"fire": 3, "read_cell": 0, "read_void": 0, "write_cell": 0, "compare": 0,
                                    "call": 1, "launch": 1, "door_open": 1, "door_closed": 1, "scan": 1, "ret": 1}


@pytest.mark.parametrize("call,expected", [
    ('printf("%d %i", n, -n)', "42 -42"),
    ('printf("%f", n / 8.0)', "5.250000"),
    ('printf("%.2f|%.0f|%.3f", 3.14159, 2.6, n * 0.5)', "3.14|3|21.000"),
    ('printf("%c%c", 65, \'b\')', "Ab"),
    ('printf("%s and %s", "one", "two")', "one and two"),
    ('printf("100%%")', "100%"),
    ('printf("%lf", 1.5)', "1.500000"),
    ('printf("%ld", 123456L)', "123456"),
    ('printf("%5d|%-5d|%05d", n, n, n)', "   42|42   |00042"),
    ('printf("a\\tb\\n")', "a\tb\n"),
    ('printf("no args")', "no args"),
])
def test_printf_formats(call, expected):
    out = trace("int f(int n) {\n    " + call + ";\n    return 0;\n}", "int f(int n)", [42])
    assert out["status"] == "ok" and out["printed"] == expected


def test_printf_output_accumulates_in_order():
    out = trace("void f(int n) {\n    for (int i = n; i > 0; i--) {\n        printf(\"%d\\n\", i);\n    }\n}",
                "void f(int n)", [3])
    assert out["printed"] == "3\n2\n1\n" and out["per_test"][0]["printed"] == "3\n2\n1\n"


def test_define_include_and_comments():
    code = ("#include <stdio.h>\n#include \"ship.h\"\n#define MAX 4\n#define STEP (MAX - 2)\n"
            "// counts up to MAX\nint f(int n) {\n    int t = 0; /* running\n    total */\n"
            "    for (int i = 0; i < MAX; i += STEP) {   // MAX stays in comments\n        t += i;\n    }\n"
            "    printf(\"MAX // not a comment\");\n    return t;\n}")
    out = trace(code, "int f(int n)", [0])
    assert out["status"] == "ok" and out["returned"] == 2 and out["printed"] == "MAX // not a comment"
    assert [s["line"] for s in out["steps"]][:2] == [7, 9]                 # lines are the learner's own


def test_smart_quotes_are_normalised():
    code = "int f(int n) {\n    printf(“%c”, ‘x’);\n    return n – 1;\n}"
    out = trace(code, "int f(int n)", [5])
    assert out["status"] == "ok" and out["printed"] == "x" and out["returned"] == 4


# ---------------------------------------------------------------- rejected (gate G4)

REJECTED_SOURCES = {
    "pointer": ["int f(int n) {\n    int *p;\n    return 0;\n}", "int f(char *s) {\n    return 0;\n}",
                "int f(int n) {\n    int a[2];\n    return *a;\n}"],
    "address_of": ["void g(int x) {\n}\nint f(int n) {\n    g(&n);\n    return 0;\n}"],
    "struct": ["struct P {\n    int x;\n};\nint f(int n) {\n    return 0;\n}"],
    "malloc": ["int f(int n) {\n    malloc(4);\n    return 0;\n}", "int f(int n) {\n    free(n);\n    return 0;\n}"],
    "string_h": ["int f(int n) {\n    return strcmp(\"a\", \"b\");\n}",
                 "int f(int n) {\n    char s[4];\n    strcpy(s, \"ab\");\n    return 0;\n}"],
    "scanf": ["int f(int n) {\n    scanf(\"%d\", n);\n    return n;\n}"],
    "goto": ["int f(int n) {\n    goto end;\nend:\n    return 0;\n}"],
    "multi_dim_array": ["int f(int n) {\n    int g[2][2];\n    return 0;\n}"],
    "switch": ["int f(int n) {\n    switch (n) {\n    case 1:\n        return 1;\n    }\n    return 0;\n}"],
}


def test_rejected_table_is_covered():
    assert set(REJECTED_SOURCES) == set(REJECTED)


@pytest.mark.parametrize("construct,code", [(k, c) for k, codes in REJECTED_SOURCES.items() for c in codes])
def test_rejected_constructs(construct, code):
    report = harness.check(code, P)
    assert report["status"] == "unsupported" and report["construct"] == construct and report["line"] >= 1
    out = harness.trace(P, code)
    assert out["status"] == "unsupported" and out["steps"] == [] and out["per_test"][0]["status"] == "unsupported"
    result = harness.run_tests(P, code)
    assert result["status"] == "unsupported" and result["tests"]["passed"] == 0 and result["tests"]["total"] == 1


def test_rejected_line_is_reported():
    assert harness.check("int f(int n) {\n    int x = 1;\n    int *p;\n    return x;\n}", P)["line"] == 3


@pytest.mark.parametrize("code", [
    "#pragma once\nint f(int n) {\n    return 1;\n}",
    "#ifdef X\n#endif\nint f(int n) {\n    return 1;\n}",
    "#define SQ(x) ((x) * (x))\nint f(int n) {\n    return SQ(n);\n}",
])
def test_other_hash_lines_are_rejected(code):
    report = harness.check(code, P)
    assert report["status"] == "unsupported" and report["line"] == 1


@pytest.mark.parametrize("code,line", [
    ("int f(int n) {\n    return n\n}", 3),
    ("int f(int n) {\n    int x = ;\n    return x;\n}", 2),
    ("int f(int n) {\n    return m;\n}", 2),
    ("int f(int n) {\n    return g(n);\n}", 2),
    ("int f(int n) {\n    return n;\n", 2),
    ("def f(n):\n    return n", 1),
    ("int f(int n) {\n    break;\n    return n;\n}", 2),
])
def test_parse_errors(code, line):
    report = harness.check(code, P)
    assert report["status"] == "parse_error" and report["line"] == line and report["reason"]
    assert harness.trace(P, code)["status"] == "parse_error"
    assert harness.run_tests(P, code)["status"] == "parse_error"


def test_entry_function_missing_or_wrong_shape():
    assert harness.check("int g(int n) {\n    return n;\n}", P)["status"] == "parse_error"
    assert harness.check("int f(int n, int m) {\n    return n;\n}", P)["status"] == "parse_error"
    assert harness.check("int f(int a[]) {\n    return 1;\n}", P)["status"] == "parse_error"
    assert harness.check("", P)["status"] == "parse_error"
    assert harness.check("int f(int n) {\n    return n;\n}", P) == {"status": "ok"}
    assert harness.check("int helper(int n) {\n    return n;\n}") == {"status": "ok"}
