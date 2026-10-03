"""Package X1: the ITSP pairing, diff and labelling rules (03 §3.4 "X (ITSP) protocol", steps 2-3).

No network and no download are needed: every test builds its own buggy/correct pair.
"""
import json

import pytest

from ml.contracts.classes import LABELS
from ml.contracts.schemas import DatasetRow
from ml.external import itsp

HEADER_BAD = '/*numPass=1, numTotal=4\nVerdict:WRONG_ANSWER, Visibility:1, Input:"1", ExpOutput:"1\n", Output:"2"\n*/\n'
HEADER_OK = '/*numPass=4, numTotal=4\nVerdict:ACCEPTED, Visibility:1, Input:"1", ExpOutput:"1\n", Output:"1"\n*/\n'


def program(*body):
    return "#include <stdio.h>\nint main() {\n    int n, i, sum = 0;\n    scanf(\"%d\", &n);\n" + \
           "".join(f"    {line}\n" for line in body) + "    return 0;\n}\n"


def label_of(buggy, correct):
    hunk = itsp.single_hunk(buggy, correct)
    assert isinstance(hunk, dict), hunk
    return itsp.classify(hunk, buggy, correct)


# ---------------------------------------------------------------- step 2: pairing and diffing

def test_header_comment_is_removed_and_read():
    code, passed, total = itsp.split_header(HEADER_BAD + "int main() { return 0; }\n")
    assert code == "int main() { return 0; }\n" and (passed, total) == (1, 4)
    assert itsp.split_header("int main() { return 0; }") == ("int main() { return 0; }", None, None)


def test_whitespace_only_changes_are_not_a_difference():
    buggy = program("for (i = 0; i < n; i++)", "    sum += i;")
    correct = program("for(i=0;i<n;i++)", "", "sum+=i;").replace("    ", "\t")
    assert itsp.single_hunk(buggy, correct) == "identical"


def test_one_small_hunk_is_kept_with_its_lines():
    buggy = program("for (i = 0; i <= n; i++)", "    sum += i;", 'printf("%d", sum);')
    correct = program("for (i = 0; i < n; i++)", "    sum += i;", 'printf("%d", sum);')
    hunk = itsp.single_hunk(buggy, correct)
    assert hunk["removed"] == ["for (i = 0; i <= n; i++)"] and hunk["added"] == ["for (i = 0; i < n; i++)"]
    assert hunk["line"] == 5


FILLER = [f"sum += {k};" for k in range(8)]         # more than 2 x 3 context lines: keeps two changes apart


def test_two_hunks_are_dropped():
    buggy = program("for (i = 0; i <= n; i++)", "    sum += i;", *FILLER, 'printf("%d", sum);')
    correct = program("for (i = 0; i < n; i++)", "    sum += i;", *FILLER, 'printf("%d\\n", sum);')
    assert itsp.single_hunk(buggy, correct) == "hunks>1"


def test_changes_within_three_lines_of_context_are_one_hunk():
    """As in difflib.unified_diff, which the protocol names: nearby changes share a hunk."""
    buggy = program("for (i = 0; i <= n; i++)", "    sum += i;", 'printf("%d", sum);')
    correct = program("for (i = 0; i < n; i++)", "    sum += i;", 'printf("%d\\n", sum);')
    hunk = itsp.single_hunk(buggy, correct)
    assert hunk["removed"] == ["for (i = 0; i <= n; i++)", 'printf("%d", sum);'] and hunk["between"] == ["sum += i;"]
    assert itsp.classify(hunk, buggy, correct) == ("OTHER", "no_rule")       # two different edits: no single rule


def test_a_hunk_of_more_than_three_lines_is_dropped():
    buggy = program("sum = 1;", "sum = 2;", "sum = 3;", "sum = 4;")
    correct = program("sum = 5;", "sum = 6;", "sum = 7;", "sum = 8;")
    assert itsp.single_hunk(buggy, correct) == "hunk>3"
    assert isinstance(itsp.single_hunk(buggy, correct, max_lines=4), dict)
    three = itsp.single_hunk(program("sum = 1;", "sum = 2;", "sum = 3;"), program("sum = 5;", "sum = 6;", "sum = 7;"))
    assert len(three["removed"]) == len(three["added"]) == 3


def test_a_pure_insertion_is_one_hunk():
    hunk = itsp.single_hunk(program("while (i < n) {", "    sum += i;", "}"), program("while (i < n) {", "    sum += i;", "    i++;", "}"))
    assert hunk["removed"] == [] and hunk["added"] == ["i++;"]


# ---------------------------------------------------------------- step 3: one test per rule, in the order of 03 §3.4

RECURSIVE = "int fact(int n) {\n{base}    {step}\n}\nint main() {\n    int n;\n    scanf(\"%d\", &n);\n    printf(\"%d\", fact(n));\n    return 0;\n}\n"

RULES = [
    # (rule of 03 §3.4, buggy, correct, expected label, rule name)
    ("`<=` <-> `<` in a loop header", program("for (i = 0; i <= n; i++)", "    sum += i;"),
     program("for (i = 0; i < n; i++)", "    sum += i;"), "M01", "le_lt_in_loop_header"),
    ("`<=` <-> `<` in a while header, other direction", program("while (i < n) {", "    sum += i; i++;", "}"),
     program("while (i <= n) {", "    sum += i; i++;", "}"), "M01", "le_lt_in_loop_header"),
    ("`=` -> `==` in a condition", program("if (n = 5)", '    printf("five");'),
     program("if (n == 5)", '    printf("five");'), "M06", "assign_to_equals_in_condition"),
    ("`=` -> `==` twice, with a debug printf deleted", program('printf("%d", n);', "if ((n = 1) && (i = 1)) {", "    sum = 1;", "}"),
     program("if ((n == 1) && (i == 1)) {", "    sum = 1;", "}"), "M06", "assign_to_equals_in_condition"),
    ("removed `;` after `)`", program("for (i = 0; i < n; i++);", "    sum += i;"),
     program("for (i = 0; i < n; i++)", "    sum += i;"), "M07", "semicolon_after_header_removed"),
    ("removed `;` after an if", program("if (n > 3);", "    sum = 1;"), program("if (n > 3)", "    sum = 1;"),
     "M07", "semicolon_after_header_removed"),
    ("added `= 0` on a declaration", program("int total;", "for (i = 0; i < n; i++) total += i;"),
     program("int total = 0;", "for (i = 0; i < n; i++) total += i;"), "M05", "zero_initialiser_added"),
    ("`(float)` added", program("float avg = sum / n;"), program("float avg = (float) sum / n;"),
     "M04", "float_cast_or_point_zero_added"),
    ("`.0` added", program("float half = n / 2;"), program("float half = n / 2.0;"), "M04", "float_cast_or_point_zero_added"),
    ("accumulator init moved out of the loop", program("for (i = 0; i < n; i++) {", "sum = 0;", "sum += i;", "}"),
     program("sum = 0;", "for (i = 0; i < n; i++) {", "sum += i;", "}"), "M03", "accumulator_init_moved_out_of_loop"),
    ("added `i++`", program("while (i < n) {", "    sum += i;", "}"), program("while (i < n) {", "    sum += i;", "    i++;", "}"),
     "M02", "increment_added"),
    ("added `i = i + 1` on the same line", program("while (i < n) { sum += i; }"), program("while (i < n) { sum += i; i = i + 1; }"),
     "M02", "increment_added"),
    ("`printf` -> `return`", "int twice(int n) {\n    printf(\"%d\", n * 2);\n}\nint main() { return 0; }\n",
     "int twice(int n) {\n    return n * 2;\n}\nint main() { return 0; }\n", "M10", "printf_to_return"),
    ("temp variable introduced around a swap", program("int a[5];", "a[0] = a[1];", "a[1] = a[0];"),
     program("int a[5];", "int t = a[0];", "a[0] = a[1];", "a[1] = t;"), "D03", "temp_variable_for_swap_added"),
    ("`mid` -> `mid + 1`", program("int low = 0, high = n, mid = 0;", "if (sum < n) low = mid;"),
     program("int low = 0, high = n, mid = 0;", "if (sum < n) low = mid + 1;"), "D02", "mid_to_mid_plus_minus_one"),
    ("base case added in a recursive function", RECURSIVE.replace("{base}", "").replace("{step}", "return n * fact(n - 1);"),
     RECURSIVE.replace("{base}", "    if (n <= 1) return 1;\n").replace("{step}", "return n * fact(n - 1);"), "D05", "base_case_added"),
    ("`return` added before a recursive call", RECURSIVE.replace("{base}", "    if (n <= 1) return 1;\n").replace("{step}", "fact(n - 1);"),
     RECURSIVE.replace("{base}", "    if (n <= 1) return 1;\n").replace("{step}", "return fact(n - 1);"), "D07", "return_added_before_recursive_call"),
    ('`"x"` -> `\'x\'`', program("char c = 'a';", 'if (c == "a") sum++;'), program("char c = 'a';", "if (c == 'a') sum++;"),
     "D08", "string_literal_to_char_literal"),
    ("`else return` removed from a loop", "int find(int a[], int n, int x) {\n    for (int i = 0; i < n; i++) {\n        if (a[i] == x) return i;\n        else return -1;\n    }\n    return -1;\n}\n",
     "int find(int a[], int n, int x) {\n    for (int i = 0; i < n; i++) {\n        if (a[i] == x) return i;\n    }\n    return -1;\n}\n", "D01", "else_return_removed"),
    ("else -> OTHER", program('printf("Valid");'), program('printf("Not Valid");'), "OTHER", "no_rule"),
    ("a relational change outside a loop header is not the twin", program("if (n >= 0) sum = 1;"), program("if (n > 0) sum = 1;"),
     "OTHER", "no_rule"),
    ("a whole loop added is not `added i++`", program("sum = n;"), program("sum = n;", "for (i = 0; i < n; i++)", '    printf("%d", i);'),
     "OTHER", "no_rule"),
    ("an added `if ... return` outside recursion is not a base case", program("sum = n;"), program("if (n == 1)", "    return 0;", "sum = n;"),
     "OTHER", "no_rule"),
]


@pytest.mark.parametrize("rule,buggy,correct,label,name", RULES, ids=[r[0] for r in RULES])
def test_rule(rule, buggy, correct, label, name):
    assert label_of(buggy, correct) == (label, name), rule


def test_every_class_named_in_the_protocol_has_a_rule_test():
    """03 §3.4 step 3 names these classes (M01/M08 share one rule)."""
    assert {r[3] for r in RULES} == {"M01", "M06", "M07", "M05", "M04", "M03", "M02", "M10", "D03", "D02", "D05", "D07",
                                     "D08", "D01", "OTHER"}
    assert {r[3] for r in RULES} <= set(LABELS)


# ---------------------------------------------------------------- the slice file

@pytest.fixture
def fake_itsp(tmp_path):
    lab = tmp_path / "ITSP" / "dataset" / "Lab-5" / "2864"
    lab.mkdir(parents=True)
    pairs = {
        "100": (program("for (i = 0; i <= n; i++)", "    sum += i;"), program("for (i = 0; i < n; i++)", "    sum += i;")),
        "101": (program("if (n = 5)", "    sum = 1;"), program("if (n == 5)", "    sum = 1;")),
        "102": (program('printf("a");'), program('printf("b");')),
        "103": (program("sum = 1;", *FILLER, "sum = 2;"), program("sum = 3;", *FILLER, "sum = 4;")),                 # two hunks
        "104": (program("sum = 1;"), program("sum  =  1;")),                                                          # identical
    }
    for name, (buggy, correct) in pairs.items():
        (lab / f"{name}_buggy.c").write_text(HEADER_BAD + buggy, encoding="utf-8")
        (lab / f"{name}_correct.c").write_text(HEADER_OK + correct, encoding="utf-8")
    (lab / "Main.c").write_text("int main() { return 0; }\n", encoding="utf-8")
    (lab / "999_buggy.c").write_text("int main() { return 1; }\n", encoding="utf-8")       # no partner: not a pair
    return tmp_path


def test_build_rows_have_the_dataset_row_shape(fake_itsp):
    rows, review, dropped = itsp.build(fake_itsp / "ITSP")
    assert len(itsp.find_pairs(fake_itsp / "ITSP")) == 5
    assert [r["label"] for r in rows] == ["M01", "M06", "OTHER"] and len(review) == 3
    assert dict(dropped) == {"hunks>1": 1, "identical": 1}
    for row in rows:
        DatasetRow.model_validate(row)
        assert row["source"] == "X" and row["split"] == "holdout_problem" and row["rater2_label"] is None
        assert row["problem_id"] == "ITSP-2864" and row["family"] == "itsp_lab5"
        assert row["code"].startswith("#include <stdio.h>") and "numPass" not in row["code"]
        assert row["verified"]["tests_failed"] == 3 and row["verified"]["tests_total"] == 4
        assert row["ast_hash"].startswith("h_") and row["label"] in LABELS
    twin = rows[0]
    assert twin["soft_label"] == {"M01": 0.5, "M08": 0.5} and twin["labels_all"] == ["M01", "M08"]
    assert twin["ambiguous_group"] == twin["ast_hash"] and twin["author"] == "itsp:Lab-5/2864/100"
    assert twin["verified"]["predicate"] == ["itsp_rule:le_lt_in_loop_header"]
    assert rows[2]["soft_label"] is None and rows[2]["labels_all"] == ["OTHER"]
    assert [r["id"] for r in rows] == ["X-000001", "X-000002", "X-000003"]


def test_main_writes_the_slice_and_prints_counts(fake_itsp, tmp_path, capsys):
    out = tmp_path / "out" / "itsp_slice.jsonl"
    assert itsp.main(["--no-fetch", "--target", str(fake_itsp), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "buggy/correct pairs found: 5" in printed and "kept (one hunk of at most 3 lines): 3" in printed
    assert "M01    1" in printed and "M06    1" in printed and "OTHER  1" in printed and "NOT hand-checked" in printed
    rows = itsp.load(out)
    assert len(rows) == 3 and all(r["rater2_label"] is None for r in rows)
    assert (fake_itsp / "itsp_slice_review.tsv").read_text(encoding="utf-8").count("\n") == 4
    assert json.loads((fake_itsp / "itsp_slice.jsonl").read_text(encoding="utf-8").splitlines()[0])["id"] == "X-000001"


def test_main_without_data_says_so_and_writes_nothing(tmp_path, capsys):
    out = tmp_path / "slice.jsonl"
    assert itsp.main(["--no-fetch", "--target", str(tmp_path), "--out", str(out)]) == 1
    assert "does not exist" in capsys.readouterr().out and not out.exists()


def test_the_real_slice_if_it_has_been_built():
    """Runs only where `python -m ml.external.itsp` has been run (the slice is not committed)."""
    if not itsp.SLICE_PATH.exists():
        pytest.skip("ml/data/itsp_slice.jsonl has not been built here")
    rows = itsp.load()
    assert rows and all(r["source"] == "X" and r["rater2_label"] is None and r["label"] in LABELS for r in rows)
    assert len({r["id"] for r in rows}) == len(rows)
