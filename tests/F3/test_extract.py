"""F3: the assembled feature row and the evidence templates.

Run from the repo root:  .venv\\Scripts\\python -m pytest tests/F3 -q
"""
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from ml.contracts import schemas as S
from ml.contracts.feature_names import (
    CLASS_DEFINING_FEATURES, FEATURES, GROUP_A, GROUP_B, GROUP_C, GROUP_R, GROUPS, OFFLINE_FEATURES,
)
from ml.contracts.subset import GARBAGE
from ml.features import extract as X
from ml.features.extract import (
    OPTIONAL_FIELDS, SENTENCE_KEYS, TEMPLATE_FIELDS, extract, load_templates, parse_signature, placeholders,
    render_evidence, sentence_key, task_features,
)
from tests.F1.snippets import SNIPPETS
from tests.F2.helpers import P03, P11, Q17, load, make_trace
from tests.fixtures import build as B

TEMPLATES = load_templates()
P03_REF_TRACE, _ = B.run_all(P03, lambda rec, c, n: B.p03_for(rec, c, n, rel_le=False))
Q17_TRACE = load("traces/q17_recursion_ok.json")


def total_energy(args):
    return sum(args[0][:args[1]])


def group(row, names):
    start = FEATURES.index(names[0])
    return row[start:start + len(names)]


# ---------------------------------------------------------------- the row

def test_row_length_and_order():
    row, meta = extract(P03, B.P03_LE, load("traces/p03_le_oob_read.json"), P03_REF_TRACE, total_energy)
    assert isinstance(row, np.ndarray) and row.dtype == np.float64 and row.shape == (len(FEATURES),)
    assert list(meta["features"]) == FEATURES
    assert [meta["features"][name] for name in FEATURES] == pytest.approx(row.tolist(), nan_ok=True)
    assert FEATURES == GROUP_A + GROUP_B + GROUP_R + GROUP_C and not set(OFFLINE_FEATURES) & set(meta["features"])


def test_exact_row_for_the_hard_twin_fixture():
    """`i <= n` over cells (P03): every non-zero column, by hand (see tests/F1 and tests/F2 for the parts)."""
    row, meta = extract(P03, B.P03_LE, load("traces/p03_le_oob_read.json"), P03_REF_TRACE, total_energy)
    expected = {name: 0.0 for name in FEATURES}
    expected.update({
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_le": 1, "a_bound_form_n": 1, "a_init_form_0": 1,
        "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1, "a_index_i": 1,
        "a_array_loop_le_n": 1, "a_reads_all_params": 1,
        "b_oob_read": 1, "b_oob_read_idx_eq_n": 1, "b_iter_delta_mean": 1, "b_iter_delta_const_pm1": 1,
        "b_returned_garbage": 1, "b_max_depth_ratio": 1, "b_base_return_executed": 1,
        "r_is_garbage": 1,
        "t_has_array_param": 1,
    })
    nan = {"b_sorted_frac", "b_n_outer_passes_ratio"}
    for name, value in zip(FEATURES, row):
        if name in nan:
            assert math.isnan(value), name
        else:
            assert value == expected[name], name
    assert meta["tests"] == {"passed": 0, "total": 5} and meta["has_trace"] and meta["entry"] == "total_energy"


def test_no_trace_gives_nan_for_groups_b_and_r_only():
    row, meta = extract(P03, B.P03_LE)
    assert row.shape == (len(FEATURES),) and not meta["has_trace"] and meta["tests"] is None
    assert np.isnan(group(row, GROUP_B)).all() and np.isnan(group(row, GROUP_R)).all()
    assert not np.isnan(group(row, GROUP_A)).any() and not np.isnan(group(row, GROUP_C)).any()
    assert meta["features"]["a_main_cond_op_le"] == 1 and meta["loops"]["selected_by"] == "static"
    S.DatasetRow                               # the row is plain floats: usable as one line of a matrix
    assert np.isfinite(row[~np.isnan(row)]).all()


def test_trace_without_reference_or_runner():
    trace = load("traces/p03_le_oob_read.json")
    row, meta = extract(P03, B.P03_LE, trace)
    feats = meta["features"]
    assert feats["b_oob_read_idx_eq_n"] == 1 and math.isnan(feats["b_iter_delta_mean"])
    assert math.isnan(feats["r_eq_ref_drop_first"]) and feats["r_is_garbage"] == 1
    row2, _ = extract(P03, B.P03_LE, trace, P03_REF_TRACE)          # reference trace, still no runner
    assert not math.isnan(row2[FEATURES.index("b_iter_delta_mean")])
    assert math.isnan(row2[FEATURES.index("r_eq_ref_last_only")])


def test_unparseable_code_still_gives_a_full_row():
    row, meta = extract(P11, SNIPPETS["does_not_parse"])
    assert row.shape == (len(FEATURES),) and not meta["parse_ok"]
    assert np.isnan(group(row, GROUP_A)).all() and not np.isnan(group(row, GROUP_C)).any()


def test_all_three_fixture_traces_assemble():
    cases = [(P03, B.P03_LE, "traces/p03_le_oob_read.json", P03_REF_TRACE, total_energy),
             (P03, B.P03_NO_UPDATE, "traces/p03_no_update_step_cap.json", P03_REF_TRACE, total_energy),
             (Q17, Q17["correct_variants"][0], "traces/q17_recursion_ok.json", Q17_TRACE,
              lambda args: math.factorial(args[0]))]
    for problem, code, path, reference, runner in cases:
        row, meta = extract(problem, code, load(path), reference, runner)
        assert row.shape == (len(FEATURES),) and meta["parse_ok"]
        assert not np.isnan(group(row, GROUP_R)).any(), path
    stuck = extract(P03, B.P03_NO_UPDATE, load("traces/p03_no_update_step_cap.json"), P03_REF_TRACE, total_energy)[1]
    assert stuck["features"]["a_update_present"] == 0 and stuck["features"]["b_step_cap"] == 1
    assert stuck["features"]["b_window_frozen"] == 1 and stuck["loops"]["main"] == 4
    ok = extract(Q17, Q17["correct_variants"][0], Q17_TRACE, Q17_TRACE, lambda args: math.factorial(args[0]))[1]
    assert ok["features"]["b_pass_frac"] == 1 and ok["features"]["a_has_recursion"] == 1
    assert ok["tests"] == {"passed": 5, "total": 5}


def test_reference_recursion_switches_the_top_frame_relation():
    """`factorial(n - 1); return n;` on Q17: the relation is on because the reference recurses."""
    code = SNIPPETS["fact_discarded"]
    steps = [{"line": 2, "vars": {"n": a}, "effects": [f"call:factorial:{d + 1}:{a}"]} for d, a in enumerate([3, 2, 1])]
    steps += [{"line": 2, "vars": {}, "effects": ["ret:factorial:3:1"]},
              {"line": 4, "vars": {}, "effects": ["ret:factorial:2:2"]},
              {"line": 4, "vars": {}, "effects": ["ret:factorial:1:3"]}]
    events = [{"type": "discarded_call_value", "fn": "factorial", "line": 3, "test": 0}]
    trace = make_trace([{"returned": r, "max_depth": d} for r, d in ((3, 3), (1, 1), (1, 1), (5, 5), (10, 10))],
                       events=events, steps=steps)
    row, meta = extract(Q17, code, trace, Q17_TRACE, lambda args: math.factorial(args[0]))
    feats = meta["features"]
    assert feats["a_rec_call_discarded"] == 1 and feats["b_discarded_call_value"] == 1
    assert feats["r_eq_top_frame_only"] == 1.0 and feats["b_pass_frac"] == pytest.approx(0.4)
    assert meta["values"]["r_eq_top_frame_only"]["fn"] == "factorial"
    item = render_evidence("r_eq_top_frame_only", meta)
    assert item == {"type": "RUN", "feature": "r_eq_top_frame_only", "line": 3,
                    "text": "Only the last gate's value came back; the result of `factorial(n - 1)` was thrown away (line 3)."}
    # the same learner output on a problem whose reference does not recurse: relation off
    sum_like = dict(P03, tests=[{"args": [[2, 4, 6], 3], "expect": {"returned": 12}}], display_test=0)
    row, meta = extract(sum_like, SNIPPETS["sum_for_lt"], make_trace([{"returned": 6}]),
                        make_trace([{"returned": 12, "loop_iters": {"L3": 3}}]), total_energy)
    assert meta["features"]["r_eq_top_frame_only"] == 0 and meta["features"]["r_eq_ref_last_only"] == 1


def test_static_main_loop_option():
    """A training row can be built with the same main-loop rule an AST-only request gets."""
    problem = dict(B.Q06, family="bubble_sort", split="train", allowed_ops=[],
                   correct_variants=[SNIPPETS["bubble_correct"]])
    trace = make_trace([{"loop_iters": {"L2": 2, "L3": 3}}, {"loop_iters": {"L2": 1, "L3": 1}},
                        {"loop_iters": {"L2": 2, "L3": 3}}, {"loop_iters": {"L2": 3, "L3": 6}}])
    code = SNIPPETS["bubble_pair_bound_twin"]
    by_trace = extract(problem, code, trace, trace)[1]
    assert by_trace["loops"]["main"] == 3 and by_trace["features"]["a_bound_form_n"] == 1
    static = extract(problem, code, trace, trace, static_main_loop=True)[1]
    assert static["loops"]["main"] == 2 and static["features"]["a_bound_form_n_minus_1"] == 1
    assert static["features"]["b_n_outer_passes_ratio"] == 1.0          # group B still uses the trace
    assert extract(problem, code)[1]["features"]["a_bound_form_n_minus_1"] == 1


def test_run_result_supplies_final_arrays():
    problem = dict(B.Q06, family="bubble_sort", split="train", allowed_ops=[],
                   correct_variants=[SNIPPETS["bubble_correct"]])
    trace, outcomes = B.run_all(B.Q06, B.q06_no_temp)
    reference = make_trace([{"loop_iters": {"L2": 2, "L3": 3}}, {"loop_iters": {"L2": 1, "L3": 1}},
                            {"loop_iters": {"L2": 2, "L3": 3}}, {"loop_iters": {"L2": 3, "L3": 6}}])
    run_result = {"status": "ok", "backend": "interp", "tests": B.tests_object(B.Q06, outcomes)}
    row, meta = extract(problem, B.Q06_NO_TEMP, trace, reference, lambda args: {"array0": sorted(args[0])},
                        run_result=run_result)
    feats = meta["features"]
    assert feats["a_swap_no_temp"] == 1 and feats["b_multiset_changed"] == 1 and feats["b_pass_frac"] == 0.25
    assert feats["t_mutates_array_arg"] == 1 and feats["t_is_void"] == 1 and meta["tests"] == {"passed": 1, "total": 4}
    assert render_evidence("b_multiset_changed", meta)["text"] == "After your swap, crate 1 appears twice and 3 is gone."
    assert render_evidence("a_swap_no_temp", meta)["text"].startswith("`a[j] = a[j + 1];` then `a[j + 1] = a[j];` (line 5)")


def test_extract_does_not_use_problem_identity():
    """Renaming the problem, its family and its sector changes nothing (03 §4.7)."""
    trace = load("traces/p03_le_oob_read.json")
    base, _ = extract(P03, B.P03_LE, trace, P03_REF_TRACE, total_energy)
    other = dict(P03, problem_id="Q99", name="x", family="zzz", planet=None, sector="sorting", world="w",
                 prompt="", exposure={}, allowed_ops=[], difficulty=3)
    renamed, _ = extract(other, B.P03_LE, trace, P03_REF_TRACE, total_energy)
    assert np.array_equal(base, renamed, equal_nan=True)


# ---------------------------------------------------------------- group C

def test_parse_signature():
    assert parse_signature("int total_energy(int cells[], int n)") == {
        "name": "total_energy", "returns": "int", "params": [("int", "cells", True), ("int", "n", False)]}
    assert parse_signature("void fire_shots(int n)")["returns"] == "void"
    assert parse_signature("float avg_fuel(int tanks[], int n)")["returns"] == "float"
    assert parse_signature("int str_length(char s[])")["params"] == [("char", "s", True)]
    assert parse_signature("int f(void)")["params"] == [] and parse_signature(None)["name"] is None


@pytest.mark.parametrize("signature,tests,expected", [
    ("int total_energy(int cells[], int n)", [{"args": [[1], 1], "expect": {"returned": 1}}], (1, 0, 0, 0, 0)),
    ("int door_open(int code)", [{"args": [1], "expect": {"returned": 0}}], (0, 0, 0, 0, 0)),
    ("float avg_fuel(int tanks[], int n)", [{"args": [[1], 1], "expect": {"returned": 1.0}}], (1, 1, 0, 0, 0)),
    ("double ratio(int a, int b)", [], (0, 1, 0, 0, 0)),
    ("void fire_shots(int n)", [{"args": [2], "expect": {"fire": 2}}], (0, 0, 1, 0, 0)),
    ("void bubble_sort(int a[], int n)", [{"args": [[2, 1], 2], "expect": {"array0": [1, 2]}}], (1, 0, 1, 0, 1)),
    ("void reverse(int a[], int n)", [], (1, 0, 1, 0, 1)),
    ("int is_palindrome(char s[])", [{"args": ["aba"], "expect": {"returned": 1}}], (1, 0, 0, 1, 0)),
])
def test_task_features(signature, tests, expected):
    feats = task_features({"signature": signature, "tests": tests})
    assert list(feats) == GROUP_C
    assert tuple(int(v) for v in feats.values()) == expected


def test_task_features_of_the_fixture_problems():
    assert list(task_features(P03).values()) == [1, 0, 0, 0, 0]
    assert list(task_features(P11).values()) == [0, 0, 0, 0, 0]
    assert list(task_features(Q17).values()) == [0, 0, 0, 0, 0]


# ---------------------------------------------------------------- templates: static checks

def sentences(entry):
    """(key, text) for every sentence of a template entry, fallbacks and variants included."""
    out = []
    for key in SENTENCE_KEYS:
        if key in entry:
            out.append((key, entry[key]))
        if f"{key}_fallback" in entry:
            out.append((f"{key}_fallback", entry[f"{key}_fallback"]))
    out += [("variant", v["text"]) for v in entry.get("variants", [])]
    return out


def test_template_file_shape():
    raw = json.loads(Path(X.__file__).with_name("evidence_templates.json").read_text(encoding="utf-8"))
    assert "_doc" in raw and "f_fix" in raw
    assert 100 <= len(TEMPLATES) - 1 <= len(FEATURES)                   # "about 100 entries"
    allowed = {"type", "neutral", "variants", "requires"} | set(SENTENCE_KEYS) | {f"{k}_fallback" for k in SENTENCE_KEYS}
    for name, entry in TEMPLATES.items():
        assert set(entry) <= allowed, name
        assert any(key in entry for key in SENTENCE_KEYS), name
        for key in SENTENCE_KEYS:
            if f"{key}_fallback" in entry:
                assert key in entry, name
        for _, text in sentences(entry):
            assert text.strip() and text[-1] in ".?" and "{}" not in text, (name, text)


def test_every_template_key_is_a_real_feature():
    keys = set(TEMPLATES) - {"f_fix"}
    assert keys <= set(FEATURES)
    assert not any(k.startswith("t_") for k in keys)                    # task meta is never shown as evidence
    for name in keys:
        assert TEMPLATES[name]["type"] == ("CODE" if name in GROUPS["A"] else "RUN"), name
    for entry in TEMPLATES.values():
        for variant in entry.get("variants", []):
            assert set(variant["if"]) <= set(FEATURES)
        assert set(entry.get("requires", [])) <= set(FEATURES)


def test_every_class_defining_feature_has_a_sentence():
    """What the model is expected to lean on for a class can always be said in words."""
    for cls, names in CLASS_DEFINING_FEATURES.items():
        for name in names:
            assert name in TEMPLATES, (cls, name)


def test_plan_examples_are_present():
    """The sentences 03 §5.6 gives as examples."""
    assert TEMPLATES["a_main_cond_op_le"]["text"] == "Loop condition uses `<=` (line {line})."
    assert TEMPLATES["b_iter_delta_const_pm1"]["text"] == "Loop ran {actual} times; the mission needed {expected}."
    assert TEMPLATES["b_oob_read_idx_eq_n"]["text"] == "Reads `{arr}[{n}]`, one cell past the end (line {line})."
    assert TEMPLATES["r_eq_ref_last_only"]["text"] == "Your total equals only the last cell's value."
    assert TEMPLATES["f_fix"]["text"] == "Changing only {fix_desc} makes every test pass."
    assert TEMPLATES["b_return_first_iter"]["text"] == "The search stopped after checking only the first beacon."


def test_every_placeholder_is_declared_by_extract():
    assert set(TEMPLATE_FIELDS) == set(FEATURES)
    for name, entry in TEMPLATES.items():
        if name == "f_fix":
            assert placeholders(entry["text"]) == {"fix_desc"}
            continue
        optional = OPTIONAL_FIELDS.get(name, set())
        for key, text in sentences(entry):
            used = placeholders(text)
            assert used <= TEMPLATE_FIELDS[name] | optional, (name, key, used - TEMPLATE_FIELDS[name])
        for key in SENTENCE_KEYS:
            if key in entry and placeholders(entry[key]) & optional:
                fallback = entry.get(f"{key}_fallback")
                assert fallback and not placeholders(fallback) & optional, (name, key)


# ---------------------------------------------------------------- templates: filled from real extract output

def corpus():
    """(label, meta) for many programs: every F1 snippet without a trace, plus traced cases."""
    out = []
    for name, code in SNIPPETS.items():
        if name != "does_not_parse":
            out.append((name, extract({"signature": "", "tests": []}, code)[1]))
    out.append(("p03_le", extract(P03, B.P03_LE, load("traces/p03_le_oob_read.json"), P03_REF_TRACE, total_energy)[1]))
    out.append(("p03_stuck", extract(P03, B.P03_NO_UPDATE, load("traces/p03_no_update_step_cap.json"),
                                     P03_REF_TRACE, total_energy)[1]))
    out.append(("q17_ok", extract(Q17, Q17["correct_variants"][0], Q17_TRACE, Q17_TRACE,
                                  lambda args: math.factorial(args[0]))[1]))
    # accumulator reset, printing instead of returning, uninitialised read, integer division
    out.append(("p03_reset", extract(P03, SNIPPETS["reset_decl_in_loop"],
                                     make_trace([{"returned": r, "loop_iters": {"L3": n}} for r, n in
                                                 ((6, 3), (5, 1), (4, 4), (9, 3), (1, 2))]),
                                     P03_REF_TRACE, total_energy)[1]))
    events = [{"type": "missing_return", "line": 6, "test": t} for t in range(5)]
    events += [{"type": "uninit_read", "var": "total", "line": 4, "test": 0},
               {"type": "intdiv", "remainder_nonzero": True, "into_float": True, "line": 4, "test": 0},
               {"type": "assign_in_cond", "var": "n", "value": 5, "line": 3, "test": 0},
               {"type": "empty_body", "kind": "for", "line": 3, "test": 0},
               {"type": "oob_read", "arr": "cells", "idx": -1, "size": 3, "line": 4, "test": 0},
               {"type": "oob_write", "arr": "cells", "idx": 3, "size": 3, "line": 4, "test": 0},
               {"type": "str_literal_compare", "line": 4, "test": 0}, {"type": "array_compare", "line": 4, "test": 0}]
    printed = make_trace([{"returned": GARBAGE, "printed": str(s), "loop_iters": {"L3": n + 1},
                           "branch": {"B4": {"true": n}}}
                          for s, n in ((12, 3), (5, 1), (10, 4), (9, 3), (8, 2))], events=events)
    reference = json.loads(json.dumps(P03_REF_TRACE))
    for pt in reference["per_test"]:
        pt["branch"] = {"B4": {"true": 1, "false": 1}}
    out.append(("p03_everything", extract(P03, SNIPPETS["printf_no_return"], printed, reference, total_energy)[1]))
    events = [{"type": "depth_cap_hit", "fn": "factorial", "last_args": [3], "line": 2, "test": 0}]
    steps = [{"line": 2, "vars": {"n": 3}, "effects": [f"call:factorial:{d}:{3 + d}"]} for d in range(1, 6)]
    deep = make_trace([{"status": "timeout", "max_depth": 100}] * 5, events=events, steps=steps)
    out.append(("q17_no_base", extract(Q17, SNIPPETS["fact_no_base"], deep, Q17_TRACE,
                                       lambda args: math.factorial(args[0]))[1]))
    out += more_cases()
    return out


NESTED = """
void f(int a[], int n) {
    int i, j;
    for (i = n; i %s; i--) {
        for (j = 0; j %s; j++) {
            a[j] = 0;
        }
    }
}"""


def traced(signature, tests, code, learner, reference, runner, arrays=None, events=(), steps=()):
    """extract() on a hand-built case: per-test learner / reference records and a Python reference."""
    problem = {"signature": signature, "tests": tests, "display_test": 0, "correct_variants": [code]}
    run_result = None
    if arrays:
        results = [{"args": t["args"], "expected": t["expect"], "got": {"array0": a},
                    "pass": a == t["expect"]["array0"]} for t, a in zip(tests, arrays)]
        run_result = {"status": "ok", "backend": "interp",
                      "tests": {"passed": sum(r["pass"] for r in results), "total": len(results), "results": results}}
    return extract(problem, code, make_trace(learner, events=events, steps=steps), make_trace(reference), runner,
                   run_result=run_result)[1]


def more_cases():
    out = []
    blank = {"signature": "", "tests": []}
    for outer, inner in (("> 0", "<= i"), (">= n + 1", "!= i"), ("!= 0", "> i"), ("== n", ">= i"), ("<= n * 2", "== i")):
        out.append((f"nested {outer}", extract(blank, NESTED % (outer, inner))[1]))
    out.append(("nested other", extract(blank, "void f(int a[], int n) { int i, j; for (i = 0; a[i]; i++) "
                                               "{ for (j = 0; ; j++) { a[j] = 0; } } }")[1]))
    out.append(("eq loop", extract(blank, "void f(int n) { int i = n; while (0 == i) { fire(); i++; } }")[1]))
    out.append(("half step", extract(blank, "int f(int n) { if (n <= 1) return 1; return f(n / 2) + 1; }")[1]))

    # in-place problems: swap without temp, array left unchanged, shift that loses a value, single pass
    trace, outcomes = B.run_all(B.Q06, B.q06_no_temp)
    sort_ref = [{"loop_iters": {"L2": 2, "L3": 3}}, {"loop_iters": {"L2": 1, "L3": 1}},
                {"loop_iters": {"L2": 2, "L3": 3}}, {"loop_iters": {"L2": 3, "L3": 6}}]
    problem = dict(B.Q06, correct_variants=[SNIPPETS["bubble_correct"]])
    out.append(("q06_no_temp", extract(problem, B.Q06_NO_TEMP, trace, make_trace(sort_ref),
                                       lambda args: {"array0": sorted(args[0])},
                                       run_result={"status": "ok", "backend": "interp",
                                                   "tests": B.tests_object(B.Q06, outcomes)})[1]))

    def sort_runner(args):
        return {"array0": sorted(args[0])}

    def one_pass(args):
        a = list(args[0])
        for j in range(len(a) - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
        return a

    sort_runner.one_pass = one_pass
    out.append(("single pass", traced("void bubble_sort(int a[], int n)", B.Q06["tests"], SNIPPETS["bubble_single_pass"],
                                      [{"loop_iters": {"L3": n}} for n in (2, 1, 2, 3)], sort_ref, sort_runner,
                                      arrays=[[1, 2, 3], [1, 2], [1, 2, 3], [3, 2, 1, 4]])))
    reverse = [{"args": [[1, 2, 3], 3], "expect": {"array0": [3, 2, 1]}},
               {"args": [[4, 7, 8, 9], 4], "expect": {"array0": [9, 8, 7, 4]}}]
    out.append(("reversed twice", traced("void reverse(int a[], int n)", reverse, SNIPPETS["reverse_correct"],
                                         [{"loop_iters": {"L2": 3}}, {"loop_iters": {"L2": 4}}],
                                         [{"loop_iters": {"L2": 1}}, {"loop_iters": {"L2": 2}}],
                                         lambda args: {"array0": args[0][::-1]}, arrays=[[1, 2, 3], [4, 7, 8, 9]])))
    rotate = [{"args": [[1, 2, 3], 3], "expect": {"array0": [2, 3, 1]}},
              {"args": [[4, 7, 8, 9], 4], "expect": {"array0": [7, 8, 9, 4]}}]
    out.append(("lost first", traced("void rotate_left(int a[], int n)", rotate, SNIPPETS["rotate_left_correct"],
                                     [{"loop_iters": {"L3": 2}}, {"loop_iters": {"L3": 3}}],
                                     [{"loop_iters": {"L3": 2}}, {"loop_iters": {"L3": 3}}],
                                     lambda args: {"array0": args[0][1:] + args[0][:1]},
                                     arrays=[[2, 3, 3], [7, 8, 9, 9]])))

    # searching: gives up after the first element
    search = [{"args": [[4, 8, 6], 3, 6], "expect": {"returned": 2}}, {"args": [[4, 8, 6], 3, 4], "expect": {"returned": 0}},
              {"args": [[4, 8, 6, 1], 4, 9], "expect": {"returned": -1}}, {"args": [[5, 7], 2, 7], "expect": {"returned": 1}}]
    out.append(("early exit", traced(
        "int linear_search(int a[], int n, int x)", search, SNIPPETS["search_else_return"],
        [{"returned": r, "loop_iters": {"L2": 1}, "branch": {"B3": {"false": 1}}} for r in (-1, -1, -1, -1)],
        [{"returned": r, "loop_iters": {"L2": n}, "branch": {"B3": {"true": 1, "false": n - 1}}}
         for r, n in ((2, 3), (0, 1), (-1, 4), (1, 2))],
        lambda args: next((i for i in range(args[1]) if args[0][i] == args[2]), -1))))

    # scalars: truncated average, a zero, an off-by-one, one shot too many, a crash
    avg = [{"args": [[1, 2], 2], "expect": {"returned": 1.5}}, {"args": [[5, 4], 2], "expect": {"returned": 4.5}},
           {"args": [[3, 3], 2], "expect": {"returned": 3.0}}, {"args": [[1, 1, 2], 3], "expect": {"returned": 1.3333}}]
    events = [{"type": "div_zero", "line": 6, "test": 2}]
    out.append(("floor", traced(
        "float avg_fuel(int tanks[], int n)", avg, SNIPPETS["int_div_returned_as_float"],
        [{"returned": 1.0, "loop_iters": {"L3": 2}}, {"returned": 0, "loop_iters": {"L3": 2}},
         {"status": "runtime_error", "loop_iters": {"L3": 2}}, {"returned": 2.3333, "loop_iters": {"L3": 3}}],
        [{"returned": 1.5, "loop_iters": {"L3": 2}}, {"returned": 4.5, "loop_iters": {"L3": 2}},
         {"returned": 3.0, "loop_iters": {"L3": 2}}, {"returned": 1.3333, "loop_iters": {"L3": 3}}],
        lambda args: sum(args[0][:args[1]]) / args[1] if args[1] else None, events=events)))
    fire = [{"args": [3], "expect": {"fire": 3}}, {"args": [1], "expect": {"fire": 1}}]
    out.append(("one more", traced("void fire_shots(int n)", fire, SNIPPETS["count_n_plus_1"],
                                   [{"effects_count": {"fire": 4}, "loop_iters": {"L2": 4}},
                                    {"effects_count": {"fire": 2}, "loop_iters": {"L2": 2}}],
                                   [{"effects_count": {"fire": 3}, "loop_iters": {"L2": 3}},
                                    {"effects_count": {"fire": 1}, "loop_iters": {"L2": 1}}],
                                   lambda args: {"fire": args[0]})))

    # stray semicolon after the for: the block below runs once
    steps = [{"line": 2}, {"line": 4}] + [{"line": 4}] * 7 + [{"line": 6}, {"line": 8}]
    out.append(("body once", traced(
        "int total_energy(int cells[], int n)", P03["tests"], SNIPPETS["for_semicolon"],
        [{"returned": GARBAGE, "loop_iters": {"L4": n}} for n in (3, 1, 4, 3, 2)],
        [{"returned": r, "loop_iters": {"L4": n}} for r, n in ((12, 3), (5, 1), (10, 4), (9, 3), (8, 2))],
        total_energy, steps=steps)))

    # recursion: value thrown away; same argument forever
    steps = [{"line": 2, "vars": {"n": a}, "effects": [f"call:factorial:{d + 1}:{a}"]} for d, a in enumerate([3, 2, 1])]
    steps += [{"line": 2, "effects": ["ret:factorial:3:1"]}, {"line": 4, "effects": ["ret:factorial:2:2"]},
              {"line": 4, "effects": ["ret:factorial:1:3"]}]
    events = [{"type": "discarded_call_value", "fn": "factorial", "line": 3, "test": 0}]
    discarded = make_trace([{"returned": r, "max_depth": d} for r, d in ((3, 3), (1, 1), (1, 1), (5, 5), (10, 10))],
                           events=events, steps=steps)
    out.append(("discarded", extract(Q17, SNIPPETS["fact_discarded"], discarded, Q17_TRACE,
                                     lambda args: math.factorial(args[0]))[1]))
    steps = [{"line": 3, "vars": {"n": 3}, "effects": [f"call:factorial:{d}:3"]} for d in range(1, 5)]
    events = [{"type": "depth_cap_hit", "fn": "factorial", "last_args": [3], "line": 3, "test": 0}]
    same = make_trace([{"status": "timeout", "max_depth": 100}] * 5, events=events, steps=steps)
    out.append(("same arg", extract(Q17, SNIPPETS["fact_same_arg"], same, Q17_TRACE,
                                    lambda args: math.factorial(args[0]))[1]))
    return out


CORPUS = corpus()


def test_every_applicable_sentence_can_be_filled():
    """Wherever a feature has a sentence for its value, `extract` supplied every placeholder."""
    filled = set()
    for label, meta in CORPUS:
        for name, value in meta["features"].items():
            entry = TEMPLATES.get(name)
            if entry is None or sentence_key(entry, value, meta["features"]) is None:
                continue
            item = render_evidence(name, meta)
            assert item is not None, (label, name, meta["values"].get(name))
            assert not re.search(r"[{}]", item["text"]), (label, item["text"])
            assert item["type"] in ("CODE", "RUN") and item["feature"] == name
            S.EvidenceItem.model_validate(dict(item, weight=0.5))
            filled.add(name)
    # the corpus reaches every sentence except the fixer's, which is not a model feature
    assert set(TEMPLATES) - filled == {"f_fix"}


def test_declared_fields_match_what_extract_returns():
    """TEMPLATE_FIELDS is not wishful: extract never returns a placeholder it does not declare."""
    for label, meta in CORPUS:
        for name, vals in meta["values"].items():
            extra = set(vals) - TEMPLATE_FIELDS[name] - OPTIONAL_FIELDS.get(name, set()) - {"line", "test"}
            assert not extra, (label, name, extra)


def test_no_sentence_for_missing_or_neutral_values():
    no_trace = extract(P03, B.P03_LE)[1]
    assert render_evidence("b_oob_read_idx_eq_n", no_trace) is None             # NaN: nothing ran
    assert render_evidence("t_has_array_param", no_trace) is None               # no template
    assert render_evidence("a_empty_body_for", no_trace) is None                # value 0, no `zero` sentence
    traced = extract(P03, B.P03_LE, load("traces/p03_le_oob_read.json"), P03_REF_TRACE, total_energy)[1]
    assert render_evidence("b_max_depth_ratio", traced) is None                 # ratio 1 says nothing
    assert render_evidence("b_base_return_executed", traced) is None


def test_zero_sentences_and_variants():
    stuck = extract(P03, B.P03_NO_UPDATE)[1]
    assert render_evidence("a_update_present", stuck)["text"] == "Nothing in the loop changes `i` (line 4)."
    no_base = dict(CORPUS)["q17_no_base"]
    assert render_evidence("a_base_case_present", no_base)["text"] == \
        "`factorial` has no base case: every path calls itself again."
    assert render_evidence("b_depth_cap", no_base)["text"] == \
        "Warp gates opened 100 deep and never closed: no base case."
    assert render_evidence("b_rec_arg_growing", no_base)["text"] == \
        "Each gate was called with a bigger value (4 up to 8); it moves away from the base."
    with_base = json.loads(json.dumps(no_base))
    with_base["features"]["a_base_case_present"] = 1.0
    assert render_evidence("b_depth_cap", with_base)["text"] == "Warp gates opened 100 deep and never closed."


def test_fallback_when_a_placeholder_is_empty():
    meta = extract(P03, B.P03_LE)[1]
    meta = json.loads(json.dumps(meta))
    meta["values"]["a_init_form_0"]["var"] = ""
    assert render_evidence("a_init_form_0", meta)["text"] == "Loop starts at 0 (line 3)."
    meta["values"]["a_main_cond_op_le"].pop("line")
    meta["lines"].pop("a_main_cond_op_le")
    assert render_evidence("a_main_cond_op_le", meta) is None                   # no fallback: skipped, never half-filled


def test_plan_sentences_come_out_as_written():
    meta = extract(P03, B.P03_LE, load("traces/p03_le_oob_read.json"), P03_REF_TRACE, total_energy)[1]
    assert render_evidence("a_main_cond_op_le", meta) == {
        "type": "CODE", "text": "Loop condition uses `<=` (line 3).", "line": 3, "feature": "a_main_cond_op_le"}
    assert render_evidence("b_oob_read_idx_eq_n", meta) == {
        "type": "RUN", "text": "Reads `cells[3]`, one cell past the end (line 4).", "line": 4,
        "feature": "b_oob_read_idx_eq_n"}
    assert render_evidence("b_iter_delta_const_pm1", meta)["text"] == "Loop ran 4 times; the mission needed 3."
