"""F1: group-A features against hand-checked values.

Run from the repo root:  .venv\\Scripts\\python -m pytest tests/F1 -q
"""
import json
import math
from pathlib import Path

import pytest

from ml.contracts.feature_names import CLASS_DEFINING_FEATURES, GROUP_A, MASK_PRECONDITIONS
from ml.features.ast_feats import ast_features, parse, strip_source
from tests.F1.expected import EXPECTED
from tests.F1.snippets import SNIPPETS

FIXTURES = Path(__file__).parent.parent / "fixtures"


def feats_of(name, **kwargs):
    return ast_features(SNIPPETS[name], **kwargs)


# ---------------------------------------------------------------- every snippet, every feature

def test_enough_snippets():
    assert len(EXPECTED) >= 20
    assert set(EXPECTED) == set(SNIPPETS) - {"does_not_parse"}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_exact_values(name):
    feats, meta = feats_of(name)
    assert list(feats) == GROUP_A                      # exactly the contract names, in order
    assert meta["parse_ok"]
    expected = {key: 0.0 for key in GROUP_A}
    expected.update({key: float(value) for key, value in EXPECTED[name].items()})
    wrong = {key: (feats[key], expected[key]) for key in GROUP_A if feats[key] != expected[key]}
    assert not wrong, f"(got, expected) {wrong}"


def test_expected_only_uses_contract_names():
    for values in EXPECTED.values():
        assert set(values) <= set(GROUP_A)


def test_every_group_a_feature_is_exercised():
    """Each feature is non-zero in at least one snippet, so no column is untested."""
    seen = set()
    for values in EXPECTED.values():
        seen |= set(values)
    never = set(GROUP_A) - seen
    # Shapes no snippet here produces; each is covered by its own test below.
    assert never == {"a_main_cond_op_eq", "a_init_form_n", "a_outer_cond_op_le",
                     "a_outer_cond_op_gt", "a_outer_cond_op_ge", "a_outer_cond_op_ne", "a_outer_cond_op_eq",
                     "a_outer_cond_op_other", "a_inner_cond_op_le", "a_inner_cond_op_gt", "a_inner_cond_op_ge",
                     "a_inner_cond_op_ne", "a_inner_cond_op_eq", "a_inner_cond_op_other",
                     "a_outer_bound_n_plus_1", "a_outer_bound_other", "a_rec_arg_form_other"}


def one(code, **kwargs):
    return ast_features(code, **kwargs)[0]


def test_remaining_one_hot_values():
    f = one("void f(int n) { int i = n; while (i > 0) { fire(); i--; } }")
    assert f["a_main_cond_op_gt"] == 1 and f["a_init_form_n"] == 1 and f["a_bound_form_const"] == 1
    f = one("void f(int n) { int i = 0; while (0 == i) { fire(); i++; } }")
    assert f["a_main_cond_op_eq"] == 1 and f["a_bound_form_const"] == 1
    f = one("void f(int n) { int i = 0; while (n > i) { fire(); i++; } }")     # read from the right
    assert f["a_main_cond_op_lt"] == 1 and f["a_bound_form_n"] == 1
    nested = """
void f(int a[], int n) {
    int i, j;
    for (i = n; i %s; i--) {
        for (j = 0; j %s; j++) {
            a[j] = 0;
        }
    }
}"""
    cases = [("> 0", "<= i", "gt", "le", "const"), (">= n + 1", "!= i", "ge", "ne", "n_plus_1"),
             ("!= 0", "> i", "ne", "gt", "const"), ("== n", ">= i", "eq", "ge", "n"),
             ("<= n * 2", "== i", "le", "eq", "other")]
    for outer, inner, outer_op, inner_op, bound in cases:
        f = one(nested % (outer, inner))
        assert f[f"a_outer_cond_op_{outer_op}"] == 1 and f[f"a_inner_cond_op_{inner_op}"] == 1, (outer, inner)
        assert f[f"a_outer_bound_{bound}"] == 1, outer
        assert sum(f[f"a_outer_cond_op_{op}"] for op in ("lt", "le", "gt", "ge", "ne", "eq", "other")) == 1
    f = one("void f(int a[], int n) { int i, j; for (i = 0; a[i]; i++) { for (j = 0; ; j++) { a[j] = 0; } } }")
    assert f["a_outer_cond_op_other"] == 1 and f["a_inner_cond_op_other"] == 1
    f = one("int f(int n) { if (n <= 1) return 1; return f(n / 2) + 1; }")
    assert f["a_rec_arg_form_other"] == 1 and f["a_base_case_static_reachable"] == 1


# ---------------------------------------------------------------- one-hot groups

@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_one_hot_groups(name):
    feats, _ = feats_of(name)
    has_loop = feats["a_n_loops"] > 0
    for prefix, values in (("a_main_cond_op_", ["lt", "le", "gt", "ge", "ne", "eq", "other"]),
                           ("a_bound_form_", ["n", "n_minus_1", "n_plus_1", "const", "other"]),
                           ("a_init_form_", ["0", "1", "n", "n_minus_1", "other"])):
        assert sum(feats[prefix + v] for v in values) == (1 if has_loop else 0)
    nested = feats["a_max_nesting"] >= 2
    for prefix in ("a_outer_cond_op_", "a_inner_cond_op_"):
        assert sum(feats[prefix + v] for v in ["lt", "le", "gt", "ge", "ne", "eq", "other"]) == (1 if nested else 0)
    assert sum(feats["a_outer_bound_" + v] for v in ["n", "n_minus_1", "n_plus_1", "const", "other"]) == (
        1 if nested else 0)


# ---------------------------------------------------------------- near-miss CORRECT forms (03 §3.5.3)

NEAR_MISS = {
    # snippet -> class-signature features that must stay 0
    "count_le_n_minus_1_correct": ["a_array_loop_le_n", "a_array_loop_start1", "a_bound_form_n"],
    "sum_one_based_correct": ["a_array_loop_start1", "a_array_loop_le_n", "a_index_i", "a_index_n",
                              "a_index_i_plus_1", "a_index_const1"],
    "sum_countdown_correct": ["a_array_loop_le_n", "a_array_loop_start1", "a_decl_noinit_read_first",
                              "a_index_n", "a_update_in_branch"],
    "search_flag_break_correct": ["a_return_in_loop_else", "a_assign_in_loop_else", "a_init_inside_loop"],
    "debug_printf_correct": ["a_nonvoid_missing_return", "a_return_const_only"],
    "cast_denominator_correct": ["a_int_div_into_float", "a_cast_wraps_division"],
    "cast_numerator_correct": ["a_int_div_into_float", "a_cast_wraps_division"],
    "float_literal_correct": ["a_int_div_into_float", "a_cast_wraps_division"],
    "decl_then_assign_correct": ["a_decl_noinit_read_first", "a_init_inside_loop"],
    "if_yoda_correct": ["a_assign_in_cond", "a_empty_body_if"],
    "assign_inside_compare_correct": ["a_assign_in_cond"],
    "bubble_correct": ["a_swap_no_temp", "a_single_loop_with_swap", "a_outer_bound_const1", "a_init_inside_loop"],
    "selection_sort_correct": ["a_swap_no_temp", "a_single_loop_with_swap", "a_init_inside_loop", "a_index_n"],
    "bsearch_correct_overflow_safe": ["a_mid_assign_no_offset", "a_return_in_loop_else", "a_assign_in_loop_else",
                                      "a_index_n", "a_update_in_branch"],
    "two_pointer_correct": ["a_update_in_branch", "a_return_in_loop_else", "a_index_n"],
    "reverse_correct": ["a_swap_no_temp", "a_index_mirror_n_minus_i"],
    "palindrome_correct": ["a_index_mirror_n_minus_i", "a_str_literal_compare", "a_array_name_compare",
                           "a_return_in_loop_else"],
    "fact_correct": ["a_rec_call_discarded", "a_rec_arg_form_same", "a_rec_arg_form_plus1"],
    "fact_accumulator_correct": ["a_rec_call_discarded", "a_rec_arg_form_same", "a_rec_arg_form_plus1"],
    "fib_two_bases_correct": ["a_rec_call_discarded"],
    "void_recursion_correct": ["a_rec_call_discarded"],
    "array_sum_rec_guarded_correct": ["a_rec_call_discarded", "a_index_n"],
    "str_length_correct": ["a_str_literal_compare", "a_array_name_compare", "a_empty_body_while"],
    "alias_last_index_correct": ["a_index_n"],
    "nested_counter_reset_correct": ["a_init_inside_loop"],
    "rotate_left_correct": ["a_swap_no_temp", "a_swap_with_temp", "a_single_loop_with_swap"],
}


@pytest.mark.parametrize("name", sorted(NEAR_MISS))
def test_near_miss_correct_forms_stay_dark(name):
    feats, _ = feats_of(name)
    lit = [f for f in NEAR_MISS[name] if feats[f] != 0]
    assert not lit, f"{name} lights {lit}"


def test_correct_recursion_keeps_base_case_features():
    for name in ("fact_correct", "fact_accumulator_correct", "fib_two_bases_correct", "void_recursion_correct",
                 "array_sum_rec_guarded_correct", "sum_digits_correct"):
        feats, _ = feats_of(name)
        assert feats["a_base_case_present"] == 1 and feats["a_base_case_static_reachable"] == 1, name


# ---------------------------------------------------------------- buggy form vs its correct twin

CONTRASTS = [
    # (buggy, correct, feature that must be 1 in the buggy one and 0 in the correct one)
    ("sum_one_based_m08", "sum_one_based_correct", "a_array_loop_start1"),
    ("sum_for_le_twin", "sum_for_lt", "a_array_loop_le_n"),
    ("last_index_n", "last_index_n_minus_1_correct", "a_index_n"),
    ("max_first_const1", "max_first_const0_correct", "a_index_const1"),
    ("index_i_plus_1", "sum_for_lt", "a_index_i_plus_1"),
    ("for_semicolon", "sum_for_lt", "a_empty_body_for"),
    ("if_semicolon", "if_yoda_correct", "a_empty_body_if"),
    ("if_assign", "if_yoda_correct", "a_assign_in_cond"),
    ("uninit_accumulator", "decl_then_assign_correct", "a_decl_noinit_read_first"),
    ("reset_decl_in_loop", "sum_for_lt", "a_init_inside_loop"),
    ("reset_assign_in_loop", "sum_for_lt", "a_init_inside_loop"),
    ("int_div_returned_as_float", "cast_numerator_correct", "a_int_div_into_float"),
    ("cast_after_division", "float_literal_correct", "a_cast_wraps_division"),
    ("printf_no_return", "debug_printf_correct", "a_nonvoid_missing_return"),
    ("printf_return_zero", "debug_printf_correct", "a_return_const_only"),
    ("bubble_no_temp", "bubble_correct", "a_swap_no_temp"),
    ("bubble_outer_once", "bubble_correct", "a_outer_bound_const1"),
    ("bubble_single_pass", "bubble_correct", "a_single_loop_with_swap"),
    ("search_else_return", "search_flag_break_correct", "a_return_in_loop_else"),
    ("search_flag_reset", "search_flag_break_correct", "a_assign_in_loop_else"),
    ("bsearch_low_mid", "bsearch_correct_overflow_safe", "a_mid_assign_no_offset"),
    ("reverse_mirror_m08_full_bound", "reverse_correct", "a_index_mirror_n_minus_i"),
    ("fact_same_arg", "fact_correct", "a_rec_arg_form_same"),
    ("fact_grow_arg", "fact_correct", "a_rec_arg_form_plus1"),
    ("fact_discarded", "fact_correct", "a_rec_call_discarded"),
    ("vowels_string_literal", "palindrome_correct", "a_str_literal_compare"),
    ("array_name_compare", "palindrome_correct", "a_array_name_compare"),
    ("two_pointer_stuck", "two_pointer_correct", "a_update_in_branch"),
    ("while_update_in_branch", "sum_for_lt", "a_update_in_branch"),
]


@pytest.mark.parametrize("buggy,correct,feature", CONTRASTS, ids=lambda v: str(v))
def test_buggy_vs_correct(buggy, correct, feature):
    assert feats_of(buggy)[0][feature] == 1
    assert feats_of(correct)[0][feature] == 0


def test_progress_structure():
    no_update = feats_of("sum_while_no_update")[0]
    assert no_update["a_update_present"] == 0 and no_update["a_update_dir"] == 0
    reverse = feats_of("while_reverse_update")[0]
    assert reverse["a_update_dir"] == -1 and reverse["a_main_cond_op_lt"] == 1
    wrong_var = feats_of("while_wrong_var")[0]
    assert wrong_var["a_update_present"] == 1 and wrong_var["a_update_var_is_cond_var"] == 0
    assert wrong_var["a_update_dir"] == 0
    semi = feats_of("while_semicolon")[0]              # the block after `while (...);` is not the body
    assert semi["a_update_present"] == 0 and semi["a_empty_body_while"] == 1


def test_base_case_features_per_recursion_bug():
    no_base = feats_of("fact_no_base")[0]
    assert no_base["a_base_case_present"] == 0 and no_base["a_base_case_static_reachable"] == 0
    unreachable = feats_of("fact_unreachable_base")[0]
    assert unreachable["a_base_case_present"] == 1 and unreachable["a_base_case_static_reachable"] == 0
    assert unreachable["a_rec_arg_form_minus2"] == 1
    for name in ("fact_same_arg", "fact_grow_arg"):
        assert feats_of(name)[0]["a_base_case_static_reachable"] == 0


def test_masking_preconditions_follow_the_code():
    """The features in MASK_PRECONDITIONS are present exactly where the structure exists."""
    plain = feats_of("sum_for_lt")[0]
    for needed in MASK_PRECONDITIONS.values():
        assert all(plain[f] == 0 for f in needed)
    assert feats_of("bsearch_correct_overflow_safe")[0]["a_mid_like_var"] == 1
    assert feats_of("bubble_correct")[0]["a_has_array_write"] == 1
    assert feats_of("fact_correct")[0]["a_has_recursion"] == 1
    assert feats_of("str_length_correct")[0]["a_char_array_param"] == 1


def test_class_defining_features_dark_on_plain_correct_code():
    """A plain correct loop lights no group-A signature feature except progress-structure ones."""
    feats = feats_of("sum_for_lt")[0]
    allowed = {"a_update_present", "a_update_dir", "a_update_var_is_cond_var"}     # M02: 1 means healthy
    for cls, names in CLASS_DEFINING_FEATURES.items():
        for name in names:
            if name.startswith("a_") and name not in allowed:
                assert feats[name] == 0, (cls, name)


# ---------------------------------------------------------------- main loop: trace rule and static fallback

def test_main_loop_static_rule_is_first_outermost_loop():
    feats, meta = feats_of("bubble_correct")
    assert meta["loops"] == {"main": 2, "outer": 2, "inner": 3, "all": [2, 3], "main_var": "i",
                             "inline": [], "main_cond_vars": ["i", "n"], "selected_by": "static"}
    assert feats["a_bound_form_n_minus_1"] == 1


def test_main_loop_from_trace_iteration_counts():
    trace = {"status": "ok", "loop_iters": {"L2": 5, "L3": 9},
             "per_test": [{"status": "ok", "loop_iters": {"L2": 2, "L3": 3}},
                          {"status": "ok", "loop_iters": {"L2": 3, "L3": 6}}]}
    feats, meta = feats_of("bubble_correct", trace=trace, display_test=0)
    assert meta["loops"]["main"] == 3 and meta["loops"]["selected_by"] == "trace"
    assert meta["loops"]["main_var"] == "j" and meta["loops"]["outer"] == 2
    assert feats["a_bound_form_other"] == 1 and feats["a_bound_form_n_minus_1"] == 0   # j < n - 1 - i
    # The twin `j < n` only shows in the main-loop columns when the inner loop is the main loop.
    twin, _ = feats_of("bubble_pair_bound_twin", trace=trace, display_test=0)
    assert twin["a_bound_form_n"] == 1


def test_main_loop_tie_goes_to_the_outermost():
    trace = {"status": "ok", "per_test": [{"status": "ok", "loop_iters": {"L2": 1, "L3": 1}}]}
    _, meta = feats_of("bubble_correct", trace=trace)
    assert meta["loops"]["main"] == 2


def test_main_loop_on_fixture_trace():
    trace = json.loads((FIXTURES / "traces" / "p03_le_oob_read.json").read_text(encoding="utf-8"))
    problem = json.loads((FIXTURES / "problems" / "P03_total_energy.json").read_text(encoding="utf-8"))
    feats, meta = ast_features(SNIPPETS["sum_for_le_twin"], entry="total_energy", trace=trace,
                               display_test=problem["display_test"])
    assert meta["loops"]["main"] == 3 and meta["entry"] == "total_energy"
    assert feats["a_main_cond_op_le"] == 1


def test_reference_variants_of_fixture_problems():
    """All correct variants parse; none lights a stray-semicolon, assignment or uninitialised flag."""
    for path in sorted((FIXTURES / "problems").glob("*.json")):
        problem = json.loads(path.read_text(encoding="utf-8"))
        for code in problem["correct_variants"]:
            feats, meta = ast_features(code)
            assert meta["parse_ok"], path.name
            for name in ("a_empty_body_if", "a_empty_body_for", "a_empty_body_while", "a_assign_in_cond",
                         "a_decl_noinit_read_first", "a_init_inside_loop", "a_nonvoid_missing_return",
                         "a_rec_call_discarded", "a_index_n", "a_array_loop_le_n"):
                assert feats[name] == 0, (path.name, name)


# ---------------------------------------------------------------- evidence metadata

def test_lines_for_evidence():
    _, meta = feats_of("sum_for_le_twin")
    assert meta["lines"]["a_main_cond_op_le"] == 3 and meta["lines"]["a_index_i"] == 4
    assert meta["values"]["a_main_cond_op_le"] == {"cond": "i <= n", "line": 3}
    _, meta = feats_of("if_semicolon")
    assert meta["lines"]["a_empty_body_if"] == 2
    _, meta = feats_of("uninit_accumulator")
    assert meta["values"]["a_decl_noinit_read_first"] == {"var": "total", "decl_line": 2, "line": 4}
    _, meta = feats_of("bsearch_low_mid")
    assert meta["values"]["a_mid_assign_no_offset"] == {"var": "low", "mid": "mid", "line": 9}
    _, meta = feats_of("fact_discarded")
    assert meta["values"]["a_rec_call_discarded"] == {"fn": "factorial", "arg": "n - 1", "line": 3}
    _, meta = feats_of("vowels_string_literal")
    assert meta["values"]["a_str_literal_compare"] == {"expr": 's[i] == "a"', "line": 4}
    _, meta = feats_of("last_index_n")
    assert meta["values"]["a_index_n"] == {"expr": "ids[n]", "arr": "ids", "line": 2}
    _, meta = feats_of("cast_after_division")
    assert meta["values"]["a_cast_wraps_division"]["expr"] == "(float)((fuel * 100) / cap)"
    _, meta = feats_of("sum_while_no_update")           # value is 0, the place is still reported
    assert meta["lines"]["a_update_present"] == 4 and meta["values"]["a_update_present"]["var"] == "i"
    _, meta = feats_of("fact_no_base")
    assert meta["values"]["a_base_case_present"] == {"line": 1, "fn": "factorial"}


def test_every_fired_flag_has_a_line():
    for name in EXPECTED:
        feats, meta = feats_of(name)
        for key, value in feats.items():
            if value and key not in ("a_n_loops", "a_max_nesting", "a_n_nested_loops", "a_n_self_calls",
                                     "a_update_dir"):
                assert key in meta["lines"], (name, key)


# ---------------------------------------------------------------- source clean-up and failure

def test_comments_and_preprocessor_lines_keep_line_numbers():
    code = SNIPPETS["comments_include_define"]
    cleaned = strip_source(code)
    assert cleaned.count("\n") == code.count("\n")
    assert "//" not in cleaned and "/*" not in cleaned and "#" not in cleaned
    assert '"' not in cleaned                           # the quoted word sat inside a comment
    feats, meta = ast_features(code)
    assert meta["lines"]["a_main_cond_op_le"] == 6      # the `for` is on line 6 of the learner file
    assert feats["a_bound_form_const"] == 1             # LIMIT replaced by 3


def test_comment_markers_inside_strings_survive():
    code = 'int f(int n) {\n    printf("// not a comment /* really */");\n    return n;\n}'
    assert strip_source(code) == code
    assert parse(code) is not None


def test_define_is_not_applied_inside_literals():
    code = '#define N 3\nint f(int n) {\n    printf("N");\n    return n + N;\n}'
    cleaned = strip_source(code)
    assert 'printf("N")' in cleaned and "n + 3" in cleaned


def test_smart_quotes_are_normalised():
    assert parse("int f(int n) { printf(“%d”, n); return n; }") is not None


def test_unparseable_code_gives_nan_not_zero():
    feats, meta = feats_of("does_not_parse")
    assert list(feats) == GROUP_A and not meta["parse_ok"]
    assert all(math.isnan(v) for v in feats.values())


def test_learner_main_is_ignored_but_used_when_alone():
    feats, meta = feats_of("with_main_ignored")
    assert meta["entry"] == "door_open" and feats["a_n_loops"] == 0 and feats["a_printf_in_nonvoid"] == 0
    feats, meta = ast_features("int main() { int i; for (i = 0; i < 3; i++) { fire(); } return 0; }")
    assert meta["entry"] == "main" and feats["a_n_loops"] == 1


def test_entry_is_named_function_when_given():
    code = SNIPPETS["fact_accumulator_correct"]
    assert ast_features(code)[1]["entry"] == "factorial"
    feats, meta = ast_features(code, entry="go")
    assert meta["entry"] == "go" and feats["a_reads_all_params"] == 1


def test_inline_loop_bodies_are_reported_for_the_trace_features():
    """F2 counts steps per line; a body on the header line must not be mistaken for header steps."""
    one_line = "int f(int a[], int n) {\n    int t = 0;\n    for (int i = 0; i < n; i++) t += a[i];\n    return t;\n}"
    assert ast_features(one_line)[1]["loops"]["inline"] == [3]
    braces_same_line = "int f(int a[], int n) {\n    int t = 0;\n    for (int i = 0; i < n; i++) { t += a[i]; }\n    return t;\n}"
    assert ast_features(braces_same_line)[1]["loops"]["inline"] == [3]
    assert feats_of("sum_for_lt")[1]["loops"]["inline"] == []
    assert feats_of("for_semicolon")[1]["loops"]["inline"] == []          # an empty body is not an inline body
