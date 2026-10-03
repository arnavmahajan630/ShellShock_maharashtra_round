"""Package G1: the serving-time gate of 03 §3.7.1 (server/app/gate.py).

For every code G1 … G8 there is at least one input that must get that code ("positive") and
one near miss that must not ("negative"), on the sample problems P03, P11 and Q17. 03 §13 K6
asks for "8 gate fixtures" without listing them; GATE_FIXTURES below is that list.
"""
import json
import time
from pathlib import Path

import pytest

from ml.contracts import schemas as S
from ml.contracts.subset import GATE_MESSAGES, MAX_BYTES, MAX_LINES, REJECTED
from server.app.gate import check, gate, normalise, strip_comments

PROBLEMS = {p["problem_id"]: p for p in (json.loads(f.read_text(encoding="utf-8")) for f in
            sorted((Path(__file__).parent.parent / "fixtures" / "problems").glob("*.json")))}
P03, P11, Q17 = PROBLEMS["P03"], PROBLEMS["P11"], PROBLEMS["Q17"]
# No sample problem forbids a builtin, so Q13 (03 §3.3.2: "Q13 forbids strlen") is sketched here.
Q13 = dict(Q17, problem_id="Q13", name="str_length", sector="strings", family="string_scan",
           signature="int str_length(char s[])", starter="int str_length(char s[]) {\n    // TODO\n}",
           forbid=["strlen"])

GOOD_P03 = P03["correct_variants"][0]
HEAD = "int total_energy(int cells[], int n) {\n"


def run(problem, code):
    result = check(problem, code)
    S.Gate.model_validate(result)
    assert set(result) == {"code", "message"}
    return result


def code_of(problem, code):
    return run(problem, code)["code"]


# ---------------------------------------------------------------- 03 §13 K6: the gate fixtures

GATE_FIXTURES = [
    ("empty",              P03, "   \n",                                                         "G1"),
    ("unchanged_starter",  P03, P03["starter"],                                                  "G2"),
    ("parse_error",        P03, HEAD + "    int total = 0\n    return total;\n}",               "G3a"),
    ("python",             P03, "def total_energy(cells, n):\n    return sum(cells)",            "G3b"),
    ("cpp",                P03, "#include <iostream>\n" + HEAD + "    std::cout << n;\n    return n;\n}", "G3c"),
    ("pointer",            P03, "int total_energy(int *cells, int n) {\n    return cells[0] + n;\n}", "G4"),
    ("strlen_in_q13",      Q13, "int str_length(char s[]) {\n    return strlen(s);\n}",          "G4b"),
    ("wrong_signature",    P03, "int total(int cells[], int n) {\n    return cells[0] + n;\n}",  "G5"),
    ("loop_only_recursion", Q17, "int factorial(int n) {\n    int r = 1;\n    for (int i = 2; i <= n; i++) r *= i;\n    return r;\n}", "G5b"),
    ("constant_answer",    P03, HEAD + "    return 12;\n}",                                      "G6"),
    ("too_large",          P03, GOOD_P03 + "\n" * (MAX_LINES + 1),                               "G7"),
    ("smart_quotes",       P03, GOOD_P03.replace("return total;", "printf(“%d”, total);\n    return total;"), "G0"),
    ("correct",            P03, GOOD_P03,                                                        "G0"),
]


@pytest.mark.parametrize("name,problem,code,expected", GATE_FIXTURES, ids=[f[0] for f in GATE_FIXTURES])
def test_gate_fixtures(name, problem, code, expected):
    assert code_of(problem, code) == expected


def test_every_code_has_a_fixture():
    assert {f[3] for f in GATE_FIXTURES} == set(GATE_MESSAGES)


def test_messages_come_from_the_contract():
    assert run(P03, "")["message"] == GATE_MESSAGES["G1"]
    assert run(P03, P03["starter"])["message"] == GATE_MESSAGES["G2"]
    assert run(P03, "def f():\n    pass")["message"] == GATE_MESSAGES["G3b"]
    assert run(P03, HEAD + "    cout << n;\n    return n;\n}")["message"] == GATE_MESSAGES["G3c"]
    assert run(Q13, "int str_length(char s[]) {\n    return strlen(s);\n}")["message"] == GATE_MESSAGES["G4b"]
    assert run(P03, "int f(int n) {\n    return n;\n}")["message"] == "Mission needs `int total_energy(int cells[], int n)`."
    assert run(Q17, "int factorial(int n) {\n    return n;\n}")["message"] == GATE_MESSAGES["G5b"]
    assert run(P11, "int door_open(int code) {\n    return 1;\n}")["message"] == GATE_MESSAGES["G6"]
    assert run(P03, GOOD_P03 + "\n" * 130)["message"] == GATE_MESSAGES["G7"]
    assert run(P03, GOOD_P03) == {"code": "G0", "message": ""}
    assert gate is check


@pytest.mark.parametrize("pid,k", [(pid, k) for pid, p in PROBLEMS.items() for k in range(len(p["correct_variants"]))])
def test_correct_variants_pass_the_gate(pid, k):
    """Negative case for every code at once: all nine correct variants are G0."""
    assert run(PROBLEMS[pid], PROBLEMS[pid]["correct_variants"][k]) == {"code": "G0", "message": ""}


# ---------------------------------------------------------------- G1

@pytest.mark.parametrize("code", ["", " ", "\n\n\t  \n", "// only a comment", "/* a block\n   comment */\n// and a line",
                                  None, " ​\n"])
def test_g1_empty_or_comments_only(code):
    assert code_of(P03, code) == "G1"


def test_g1_negative_one_statement_is_not_empty():
    assert code_of(P03, "// plan\nint total_energy(int cells[], int n) {\n    return cells[0] + n;\n}") == "G0"
    assert code_of(P03, ";") != "G1"


# ---------------------------------------------------------------- G2

@pytest.mark.parametrize("pid", sorted(PROBLEMS))
def test_g2_starter_unchanged(pid):
    assert code_of(PROBLEMS[pid], PROBLEMS[pid]["starter"]) == "G2"


@pytest.mark.parametrize("code", [
    "int total_energy(int cells[],int n){\n\n\n}\n",                             # whitespace only
    "int total_energy(int cells[], int n) {\r\n    // TODO\r\n}\r\n",            # Windows line ends
    "int total_energy(int cells[], int n) {\n    /* thinking... */\n}",          # a different comment
    "  int   total_energy ( int cells [ ] , int n )\n{\n}\n",
])
def test_g2_starter_after_normalisation(code):
    assert code_of(P03, code) == "G2"


def test_g2_negative_any_real_change():
    assert code_of(P03, "int total_energy(int cells[], int n) {\n    return cells[0] + n;\n}") == "G0"
    assert code_of(P03, "int total_energy(int cells[], int n) {\n    int total;\n}") != "G2"


# ---------------------------------------------------------------- G3a

@pytest.mark.parametrize("body,line,reason", [
    ("    int t = n\n    return t;\n}", 2, "missing `;`"),
    ("    return n\n}", 2, "missing `;`"),
    ("    int t = 0;\n    t = n +;\n    return t;\n}", 3, "incomplete expression"),
    ("    if (n > 0) {\n        return 1;\n    return 0;\n}", 5, "missing `}`"),
    ("    return n;\n}\n}", 4, "unmatched `}`"),
    ("    if (n > 0 {\n        return 1;\n    }\n    return 0;\n}", 2, "missing `)`"),
    ("    printf(\"hi);\n    return n;\n}", 2, "unterminated string"),
    ("    int x = n @ 4;\n    return x;\n}", 2, "illegal character `@`"),
    ("    bool b = n;\n    return b;\n}", 2, "unknown type `bool`"),
    ("    /* never closed\n    return n;\n}", 2, "unterminated comment"),
    ("    for (int i = 0, i < n, i++) { }\n    return n;\n}", 2, "unexpected `<`"),
], ids=["semicolon", "semicolon_before_brace", "hanging_operator", "missing_brace", "extra_brace", "missing_paren",
        "string", "illegal_char", "unknown_type", "comment", "for_commas"])
def test_g3a_parse_error_names_line_and_reason(body, line, reason):
    result = run(P03, HEAD + body)
    assert result["code"] == "G3a"
    assert result["message"] == f"Core rejected line {line}: {reason}."


def test_g3a_line_numbers_survive_comments_and_includes():
    code = "#include <stdio.h>\n// a comment line\n/* two\n   more */\n" + HEAD + "    int t = n\n    return t;\n}"
    assert run(P03, code)["message"] == "Core rejected line 6: missing `;`."


def test_g3a_plain_text():
    result = run(P03, "please give me the answer")
    assert result["code"] == "G3a" and result["message"].startswith("Core rejected line 1: ")


def test_g3a_message_is_short_and_has_no_parser_internals():
    for body in ["    return n +* ;\n}", "    int 9x = 1;\n    return n;\n}", "    return (n;\n}", "    if n > 0 return 1;\n}"]:
        result = run(P03, HEAD + body)
        assert result["code"] == "G3a", body
        assert len(result["message"]) < 90 and "http" not in result["message"] and "pycparser" not in result["message"]
        assert result["message"].startswith("Core rejected line ") and result["message"].endswith(".")


def test_g3a_negative_valid_c_is_not_a_parse_error():
    assert code_of(P03, HEAD + "    int total = 0; /* ok */\n    for (int i = 0; i < n; i++) { total += cells[i]; } // ok\n    return total;\n}") == "G0"


# ---------------------------------------------------------------- G3b

@pytest.mark.parametrize("code", [
    "def total_energy(cells, n):\n    total = 0\n    for x in cells:\n        total += x\n    return total",
    "total = 0\nprint(total)",
    "if n == 0:\n    return 0\nelif n == 1:\n    return 1",
    "for i in range(n):\n    total += cells[i]",
    "import math\nprint(math.pi)",
])
def test_g3b_python(code):
    assert code_of(P03, code) == "G3b"


def test_g3b_negative_c_with_python_words():
    """A C program that defines print(), or mentions def in a comment or string, is not Python."""
    code = ("void print(int x) { printf(\"%d\", x); }\n" + HEAD
            + "    // def not python: elif\n    print(n);\n    printf(\"def f(x): elif\");\n    return cells[0] + n;\n}")
    assert code_of(P03, code) == "G0"
    broken_c = HEAD + "    print(n)\n    return cells[0];\n}"          # a C slip, not Python
    assert code_of(P03, broken_c) == "G3a"


# ---------------------------------------------------------------- G3c

@pytest.mark.parametrize("code", [
    HEAD + "    cout << n;\n    return n;\n}",
    "#include <iostream>\n" + HEAD + "    return cells[0] + n;\n}",
    "using namespace std;\n" + HEAD + "    return cells[0] + n;\n}",
    HEAD + "    std::cout << cells[0] << std::endl;\n    return n;\n}",
    HEAD + "    int x;\n    cin >> x;\n    return x + n;\n}",
    "#include <bits/stdc++.h>\n" + HEAD + "    return cells[0] + n;\n}",
])
def test_g3c_cpp(code):
    assert code_of(P03, code) == "G3c"


def test_g3c_negative_cpp_words_in_comments_and_strings():
    code = HEAD + "    // cout << n; std::endl\n    printf(\"std::cout\");\n    return cells[0] + n;\n}"
    assert code_of(P03, code) == "G0"
    assert code_of(P03, "#include <stdio.h>\n#include <string.h>\n" + GOOD_P03) == "G0"


# ---------------------------------------------------------------- G4

REJECTED_CODE = {
    "pointer": "int total_energy(int *cells, int n) {\n    return n;\n}",
    "address_of": HEAD + "    int same = (&n == 0);\n    return same + n;\n}",
    "struct": "struct cell { int e; };\n" + HEAD + "    struct cell c;\n    c.e = n;\n    return c.e;\n}",
    "malloc": HEAD + "    int *copy = malloc(n * 4);\n    return n;\n}",
    "string_h": HEAD + "    char a[] = \"x\";\n    return strcmp(a, a) + n;\n}",
    "scanf": HEAD + "    scanf(\"%d\", &n);\n    return n;\n}",
    "goto": HEAD + "    goto end;\nend:\n    return n;\n}",
    "multi_dim_array": HEAD + "    int g[2][2];\n    g[0][0] = n;\n    return g[0][0];\n}",
    "switch": HEAD + "    switch (n) {\n    case 1:\n        return 1;\n    default:\n        return cells[0];\n    }\n}",
}


def test_g4_table_covers_the_contract():
    assert set(REJECTED_CODE) == set(REJECTED)


@pytest.mark.parametrize("construct", sorted(REJECTED_CODE))
def test_g4_unsupported_construct_is_named(construct):
    result = run(P03, REJECTED_CODE[construct])
    assert result["code"] == "G4"
    assert result["message"] == GATE_MESSAGES["G4"].format(construct=REJECTED[construct])


@pytest.mark.parametrize("code,word", [
    (HEAD + "    return *cells + n;\n}", "pointers"),
    (HEAD + "    char *s = \"hi\";\n    return n;\n}", "pointers"),
    (HEAD + "    return abs(cells[0]) + n;\n}", "abs()"),
    (HEAD + "    return sizeof(cells) + n;\n}", "sizeof"),
    ("typedef int cell;\n" + GOOD_P03, "typedef"),
    ("#ifdef DEBUG\n#endif\n" + GOOD_P03, "#ifdef"),
    ("#define SQ(x) ((x) * (x))\n" + HEAD + "    return SQ(n) + cells[0];\n}", "#define with arguments"),
    (HEAD + "    int g[2][2] = {{1, 2}, {3, 4}};\n    return g[1][1] + n;\n}", "multi-dimensional arrays"),
], ids=["deref", "char_pointer", "unknown_function", "sizeof", "typedef", "ifdef", "macro_with_args", "2d_init"])
def test_g4_other_unsupported(code, word):
    result = run(P03, code)
    assert result["code"] == "G4" and f"`{word}`" in result["message"]


def test_g4_negative_everything_in_the_subset_passes():
    code = ("#include <stdio.h>\n#define START 0\n"
            "int add(int a, int b) { return a + b; }\n" + HEAD +
            "    int total = START;\n    float half = 0.5;\n    char mark = 'x';\n    int seen[3] = {0, 0, 0};\n"
            "    for (int i = START; i < n; i++) {\n        total = add(total, cells[i]);\n        seen[i % 3]++;\n    }\n"
            "    int j = 0;\n    while (j < n && !(cells[j] < 0)) { j++; }\n    do { j--; } while (j > 0);\n"
            "    printf(\"%d %f %c\\n\", total, half * 2, mark);\n    fire();\n    scan(total);\n"
            "    return (int) (total * 1.0) + (n > 0 ? 0 : 0);\n}")
    assert code_of(P03, code) == "G0"
    assert code_of(dict(Q13, forbid=[]), "int str_length(char s[]) {\n    char t[] = \"ok\";\n    return strlen(s) + strlen(t) - 2;\n}") == "G0"


# ---------------------------------------------------------------- G4b

def test_g4b_forbidden_builtin():
    result = run(Q13, "int str_length(char s[]) {\n    int n = strlen(s);\n    return n;\n}")
    assert result == {"code": "G4b", "message": "Trial rules: count it yourself, no `strlen`."}


def test_g4b_negative():
    own = "int str_length(char s[]) {\n    int i = 0; // strlen(s) is not allowed here\n    while (s[i] != '\\0') {\n        i++;\n    }\n    printf(\"strlen(s)\");\n    return i;\n}"
    assert code_of(Q13, own) == "G0"                                    # only in a comment and a string
    assert code_of(dict(Q13, forbid=[]), "int str_length(char s[]) {\n    return strlen(s);\n}") == "G0"


# ---------------------------------------------------------------- G5

@pytest.mark.parametrize("code", [
    "int sum_cells(int cells[], int n) {\n    return cells[0] + n;\n}",                 # other name
    "int total_energy(int cells[]) {\n    return cells[0];\n}",                         # one parameter missing
    "int total_energy(int cells[], int n, int extra) {\n    return cells[0] + n + extra;\n}",
    "int total_energy(int cells[], int n);",                                            # prototype only
    "int total_energy(void) {\n    return 0;\n}",
    "int main() {\n    return 0;\n}",
])
def test_g5_entry_signature_missing_or_wrong_arity(code):
    result = run(P03, code)
    assert result == {"code": "G5", "message": "Mission needs `int total_energy(int cells[], int n)`."}


def test_g5_negative_other_names_for_parameters_and_helpers():
    code = "int helper(int x) { return x; }\nint total_energy(int c[], int count) {\n    return helper(c[0]) + count;\n}\nint main() { return 0; }"
    assert code_of(P03, code) == "G0"


# ---------------------------------------------------------------- G5b

@pytest.mark.parametrize("code", [
    "int factorial(int n) {\n    int r = 1;\n    for (int i = 2; i <= n; i++) {\n        r *= i;\n    }\n    return r;\n}",
    "int helper(int n) {\n    int r = 1;\n    while (n > 1) { r *= n; n--; }\n    return r;\n}\nint factorial(int n) {\n    return helper(n);\n}",
    "int loop(int n) {\n    if (n <= 1) return 1;\n    return n * loop(n - 1);\n}\nint factorial(int n) {\n    int r = 1;\n    for (int i = 2; i <= n; i++) r *= i;\n    return r;\n}",
], ids=["loop", "loop_in_helper", "recursive_helper_never_called"])
def test_g5b_recursion_problem_without_a_self_call(code):
    assert run(Q17, code) == {"code": "G5b", "message": GATE_MESSAGES["G5b"]}


@pytest.mark.parametrize("code", [
    "int factorial(int n) {\n    return n * factorial(n - 1);\n}",                                    # no base case: still recursion
    "int go(int n, int acc) {\n    if (n <= 1) return acc;\n    return go(n - 1, acc * n);\n}\nint factorial(int n) {\n    return go(n, 1);\n}",
    "int b(int n);\nint a(int n) {\n    if (n <= 1) return 1;\n    return n * b(n - 1);\n}\nint b(int n) {\n    return a(n);\n}\nint factorial(int n) {\n    return a(n);\n}",
    "int factorial(int n) {\n    if (n <= 1) return 1;\n    factorial(n - 1);\n    return n;\n}",
], ids=["no_base_case", "recursive_helper", "mutual_recursion", "discarded_value"])
def test_g5b_negative_any_recursion_reachable_from_the_entry(code):
    assert code_of(Q17, code) == "G0"


def test_g5b_only_applies_to_recursion_problems():
    loop = "int door_open(int code) {\n    for (int i = 0; i < 1; i++) { }\n    return code == 42;\n}"
    assert code_of(P11, loop) == "G0"
    assert P11.get("sector") is None and Q17["sector"] == "recursion"


# ---------------------------------------------------------------- G6

@pytest.mark.parametrize("problem,code", [
    (P03, HEAD + "    return 12;\n}"),
    (P11, "int door_open(int code) {\n    int open = 1;\n    return open;\n}"),
    (P11, "int door_open(int code) {\n    printf(\"1\");\n}"),
    (P11, "int door_open(int code) {\n    code = 42;\n    return 1;\n}"),          # writes the parameter, never reads it
    (P03, HEAD + "    int total = 2 + 4 + 6;\n    printf(\"%d\", total);\n    return total;\n}"),
], ids=["literal", "variable", "printf", "parameter_overwritten", "hard_coded_sum"])
def test_g6_constant_answer(problem, code):
    assert run(problem, code) == {"code": "G6", "message": GATE_MESSAGES["G6"]}


@pytest.mark.parametrize("problem,code", [
    (P11, "int door_open(int code) {\n    if (code == 42) return 1;\n    return 0;\n}"),
    (P11, "int door_open(int code) {\n    if (code = 42) {\n        return 1;\n    }\n    return 0;\n}"),   # M06: must be diagnosed
    (P03, HEAD + "    return cells[0];\n}"),                                         # reads the array parameter
    (P03, HEAD + "    int total = 0;\n    for (int i = 0; i < n; i++);\n    return total;\n}"),   # reads n
    (P11, "int door_open(int code) {\n    int x;\n}"),                              # no return, no print: not G6
    (P11, "int check(int c) { return c == 42; }\nint door_open(int code) {\n    return check(code);\n}"),
], ids=["correct", "assign_in_condition", "reads_array", "reads_n", "no_output", "passes_parameter_on"])
def test_g6_negative(problem, code):
    assert code_of(problem, code) == "G0"


# ---------------------------------------------------------------- G7

def test_g7_too_many_lines_or_bytes():
    assert code_of(P03, GOOD_P03 + "\n" * (MAX_LINES + 1)) == "G7"
    assert code_of(P03, GOOD_P03 + "\n// " + "x" * MAX_BYTES) == "G7"
    assert code_of(P03, "\n".join(f"int v{i};" for i in range(MAX_LINES + 1))) == "G7"


def test_g7_negative_at_the_limits():
    lines = len(GOOD_P03.splitlines())
    padded = GOOD_P03 + "\n" + "\n".join("// pad" for _ in range(MAX_LINES - lines))
    assert len(padded.splitlines()) == MAX_LINES == 120 and code_of(P03, padded) == "G0"
    filler = "// " + "x" * (MAX_BYTES - len(GOOD_P03.encode("utf-8")) - 4)
    exact = GOOD_P03 + "\n" + filler
    assert len(exact.encode("utf-8")) == MAX_BYTES == 4096 and code_of(P03, exact) == "G0"
    assert code_of(P03, exact + "x") == "G7"


# ---------------------------------------------------------------- G8

def test_g8_smart_quotes_are_fixed_silently():
    code = (HEAD + "    int total = 0;\n    char mark = ‘x’;\n    for (int i = 0; i < n; i++) total += cells[i];\n"
            "    printf(“%d %c”, total, mark);\n    return total;\n}")
    assert run(P03, code) == {"code": "G0", "message": ""}
    assert normalise(code) == code.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")


def test_g8_other_non_ascii_punctuation():
    code = (HEAD + "    int total = 0;\n    for (int i = n – 1; i ≥ 0; i−−) total += cells[i]；\n"
            "    return total;\n}")
    assert code_of(P03, code) == "G0"
    assert normalise("a –−；（）≤≠​") == "a --;()<=!="
    assert normalise("x\r\ny\rz") == "x\ny\nz" and len(normalise(code).split("\n")) == len(code.split("\n"))


def test_g8_non_ascii_inside_comments_and_strings_is_fine():
    code = HEAD + "    int total = 0; // कुल jod\n    for (int i = 0; i < n; i++) total += cells[i];\n    printf(\"₹%d\", total);\n    return total;\n}"
    assert code_of(P03, code) == "G0"


def test_g8_negative_real_errors_are_still_reported():
    code = HEAD + "    printf(“%d, n);\n    return n;\n}"                 # the closing quote is missing
    assert run(P03, code)["message"] == "Core rejected line 2: unterminated string."
    assert code_of(P03, HEAD + "    int café = cells[0];\n    return café + n;\n}") == "G3a"


# ---------------------------------------------------------------- what the gate must let through

@pytest.mark.parametrize("problem,code", [
    (P03, HEAD + "    int total = 0;\n    for (int i = 0; i <= n; i++) {\n        total += cells[i];\n    }\n    return total;\n}"),
    (P03, HEAD + "    int total = 0;\n    int i = 0;\n    while (i < n) {\n        total += cells[i];\n    }\n    return total;\n}"),
    (P03, HEAD + "    int total;\n    for (int i = 0; i < n; i++) total += cells[i];\n    return total;\n}"),
    (P03, HEAD + "    int total = 0;\n    for (int i = 0; i < n; i++);\n    {\n        total += cells[0];\n    }\n    return total;\n}"),
    (P03, HEAD + "    int total = 0;\n    for (int i = 0; i < n; i++) total += cells[i];\n    printf(\"%d\", total);\n}"),
    (P03, HEAD + "    return cells[0] / (n - n);\n}"),
    (P11, "int door_open(int code) {\n    if (code == 42);\n    {\n        return 1;\n    }\n    return 0;\n}"),
    (Q17, "int factorial(int n) {\n    if (n <= 1) {\n        return 1;\n    }\n    factorial(n - 1);\n    return n;\n}"),
], ids=["off_by_one", "endless_loop", "uninitialised", "for_semicolon", "print_not_return", "div_by_zero",
        "if_semicolon", "discarded_recursion"])
def test_misconception_programs_are_not_gated(problem, code):
    """03 §3.7.1 last row: a time-out or runtime error is not a gate, and neither is a wrong
    program. These must reach the interpreter and the model."""
    assert run(problem, code) == {"code": "G0", "message": ""}


# ---------------------------------------------------------------- helpers and robustness

def test_strip_comments_keeps_lines_and_strings():
    code = "int a; // one\n/* two\n   three */ int b;\nchar s[] = \"// not a comment /* nor this */\";\n"
    stripped = strip_comments(code)
    assert stripped.count("\n") == code.count("\n") and len(stripped) == len(code)
    assert "one" not in stripped and "three" not in stripped and "int b;" in stripped
    assert '"// not a comment /* nor this */"' in stripped


def test_order_of_checks():
    """Size before parsing; C++ before parse errors; a forbidden builtin before other unsupported things."""
    assert code_of(P03, "def f():\n" + "    pass\n" * 130) == "G7"
    assert code_of(P03, HEAD + "    cout << n\n    return n;\n}") == "G3c"
    assert code_of(Q13, "int str_length(char *s) {\n    return strlen(s);\n}") == "G4b"
    assert code_of(Q17, "int factorial(int n) {\n    return 120;\n}") == "G5b"


@pytest.mark.parametrize("code", ["\x00\x01\x02", "#", "}{", "((((((((", "int", ";;;;", "\"", "'", "/*", "*/", "#define",
                                  "int total_energy(int cells[], int n) {\n    return " + "(" * 1500 + "n" + ")" * 1500 + ";\n}"])
def test_never_raises(code):
    result = run(P03, code)
    assert result["code"] != "G0"


def test_problem_fields_are_optional():
    assert check({}, "int f(int n) {\n    return n;\n}") == {"code": "G0", "message": ""}
    assert check(None, "") == {"code": "G1", "message": GATE_MESSAGES["G1"]}


def test_gate_is_fast():
    started = time.perf_counter()
    for _ in range(50):
        check(P03, GOOD_P03)
    assert (time.perf_counter() - started) / 50 < 0.02
