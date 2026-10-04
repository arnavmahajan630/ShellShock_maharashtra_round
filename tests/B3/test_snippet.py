"""ml/items/snippet.py: running the fragment an item shows, on the interpreter and on gcc."""
import pytest

from ml.items import snippet
from ml.oracle import gcc

try:
    gcc.gcc_path()
    HAVE_GCC = True
except gcc.GccUnavailable:
    HAVE_GCC = False
needs_gcc = pytest.mark.skipif(not HAVE_GCC, reason="gcc is not installed")

FUNCTION = "int f(int n) {\n    if (n == 0) return 0;\n    return n + f(n - 1);\n}"
LOOP = "int s = 0;\nfor (int i = 0; i < 4; i++) {\n    s = s + i;\n}"


def test_split_keeps_definitions_apart_from_statements():
    definitions, statements = snippet.split(FUNCTION + "\nint y = f(3);")
    assert definitions == FUNCTION
    assert statements == "\nint y = f(3);"


def test_split_with_only_statements_or_only_definitions():
    assert snippet.split("int x = 3;\nx++;") == ("", "int x = 3;\nx++;")
    assert snippet.split(FUNCTION) == (FUNCTION, "")


def test_split_is_not_fooled_by_braces_in_text():
    code = 'void say() {\n    printf("}{");\n}\nsay();'
    assert snippet.split(code)[1] == "\nsay();"


def test_define_lines_stay_outside_the_wrapper():
    code = "#define SIZE 3\nint twice(int x) { return x * 2; }\nint y = twice(SIZE);"
    assert snippet.interp_answer(code, 'printf("%d", y);')["text"] == "6"


def test_build_keeps_line_numbers():
    program = snippet.build(LOOP, 'printf("%d", s);')
    assert program.split("\n")[2] == LOOP.split("\n")[2]
    with_function = FUNCTION + "\nint y = f(3);"
    assert snippet.build(with_function).split("\n")[4].endswith("int y = f(3);")


@pytest.mark.parametrize("code, tail, kind, says, expected", [
    ("for (int i = 0; i <= 3; i++) fire();", "", "count:fire", None, "4"),
    ('printf("%d", 2 + 3);', 'printf("9");', "printed", None, "5"),
    ('printf("%d", 2 + 3);', 'printf("9");', "tail", None, "9"),
    ("int x = 0;\nif (x = 0) fire(); else open_door();", "", "effect", None, "door opens"),
    ("int x = 9;", "", "effect", None, "nothing happens"),
    ("int x = 2;\nif (x > 10);\nfire();", "", "effect", {"fire": "yes"}, "yes"),
    ("int i = 0;\nwhile (i < 3) { fire(); }", "", "count:fire", None, "never stops"),
    ("int h(int n) { return h(n - 1); }", 'printf("%d", h(2));', "tail", None, "never stops"),
    ("int a = 7, b = 2;\nfloat r = a / b;", 'printf("%.1f", r);', "tail", None, "3.0"),
])
def test_answer_kinds_on_the_interpreter(code, tail, kind, says, expected):
    assert snippet.interp_answer(code, tail, kind, says)["text"] == expected


def test_reads_that_c_leaves_undefined_are_named_and_kept_from_gcc():
    unset = snippet.answer("int t;\nt = t + 5;", 'printf("%d", t);')
    assert unset == {"interp": "unpredictable", "gcc": None, "agree": True, "error": None}
    outside = snippet.answer("int a[4] = {3, 5, 7, 9};", 'printf("%d", a[4]);')
    assert outside["interp"] == "outside the array" and outside["gcc"] is None


def test_state_query_reads_the_value_after_the_nth_run_of_a_line():
    for hit, value in [(1, "0"), (2, "1"), (3, "3"), (4, "6")]:
        query = {"var": "s", "after_line": 3, "hit": hit}
        assert snippet.interp_answer(LOOP, kind="state", state_query=query)["text"] == value
    never = {"var": "s", "after_line": 3, "hit": 9}
    assert snippet.interp_answer(LOOP, kind="state", state_query=never)["text"] is None


def test_code_that_cannot_run_gives_no_answer_and_a_reason():
    got = snippet.answer("int x = ;")
    assert got["interp"] is None and not got["agree"] and got["error"]
    pointer = snippet.interp_answer("int x = 1;\nint *p = &x;")
    assert pointer["text"] is None and pointer["error"]


@needs_gcc
@pytest.mark.parametrize("code, tail, kind, query, expected", [
    ("for (int i = 0; i <= 3; i++) fire();", "", "count:fire", None, "4"),
    ("int i = 0;\nwhile (i < 3) { fire(); }", "", "count:fire", None, "never stops"),
    ("int a[2] = {4, 7};\na[0] = a[1];\na[1] = a[0];", 'printf("{%d, %d}", a[0], a[1]);', "tail", None, "{7, 7}"),
    (LOOP, "", "state", {"var": "s", "after_line": 3, "hit": 3}, "3"),
    ("float half = 0;\nfor (int i = 1; i < 4; i++) {\n    half = i / 2.0;\n}", "", "state",
     {"var": "half", "after_line": 3, "hit": 3}, "1.5"),
])
def test_gcc_agrees_with_the_interpreter(code, tail, kind, query, expected):
    got = snippet.answer(code, tail, kind, None, query)
    assert got == {"interp": expected, "gcc": expected, "agree": True, "error": None}


@needs_gcc
def test_a_state_query_on_a_loop_header_is_caught_by_the_gcc_cross_check():
    got = snippet.answer(LOOP, kind="state", state_query={"var": "i", "after_line": 2, "hit": 3})
    assert not got["agree"]
