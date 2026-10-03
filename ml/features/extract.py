"""Feature assembly: one ordered row for `FEATURES` plus evidence metadata (package F3).

    row, meta = extract(problem, code, trace=None, reference_trace=None, run_reference=None)

row    numpy float64 vector, one value per name in `ml.contracts.feature_names.FEATURES`
       (groups A, B, R, C in that order). NaN = not available.
meta   {"features": {name: value}, "lines": {name: line}, "values": {name: {placeholder: value}},
        "parse_ok", "entry", "loops", "has_trace", "tests": {"passed", "total"} | None}
       `lines` and `values` are what `ml/features/evidence_templates.json` is filled from.

Arguments
    problem          problem dict (schemas.Problem). Only `signature`, `tests`, `display_test`,
                     `correct_variants[0]` and `one_pass` are read. No id, family or sector.
    code             the learner's C source.
    trace            learner trace dict (schemas.Trace) or None. None → groups B and R are NaN.
    reference_trace  trace of `correct_variants[0]` on the same tests, or None.
    run_reference    callable(args) -> output for the reference on changed inputs (group R);
                     see `relation_feats`. Built by `relation_feats.make_run_reference`.
    run_result       (keyword) the learner's `run_tests` result. Needed for the final arrays of
                     in-place problems on tests other than the display test (see notes/F2.md).
    static_main_loop (keyword) choose the main loop without the trace even when one is given,
                     so a training row can be made identical to what an AST-only request gets.

Nothing here runs code: traces and the reference runner are passed in by the caller.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from ml.contracts.feature_names import FEATURES, GROUP_A, GROUP_B, GROUP_C, GROUP_R
from ml.features.ast_feats import ast_features
from ml.features.relation_feats import relation_features
from ml.features.trace_feats import collect_outputs, trace_features

_SIGNATURE = re.compile(r"^\s*([A-Za-z_][\w\s]*?)\s*\b([A-Za-z_]\w*)\s*\((.*)\)\s*;?\s*$", re.S)
_COUNT_FEATURES = ["a_n_loops", "a_max_nesting", "a_n_nested_loops", "a_n_self_calls"]


def parse_signature(signature):
    """`int total_energy(int cells[], int n)` -> {"name", "returns", "params": [(type, name, is_array)]}."""
    m = _SIGNATURE.match(signature or "")
    if not m:
        return {"name": None, "returns": "int", "params": []}
    returns, name, inside = m.group(1).split(), m.group(2), m.group(3).strip()
    params = []
    if inside and inside != "void":
        for part in inside.split(","):
            is_array = "[" in part or "*" in part
            words = re.sub(r"\[[^\]]*\]|\*", " ", part).split()
            if words:
                params.append((" ".join(words[:-1]) or "int", words[-1], is_array))
    return {"name": name, "returns": " ".join(returns), "params": params}


def task_features(problem):
    """Group C (03 §4.5): coarse task meta from the signature and the kind of expected output."""
    sig = parse_signature((problem or {}).get("signature"))
    returns = sig["returns"]
    arrays = [p for p in sig["params"] if p[2]]
    tests = (problem or {}).get("tests") or []
    if tests:
        mutates = any("array0" in (t.get("expect") or {}) for t in tests)
    else:                                       # no tests to look at: a void function over an array
        mutates = "void" in returns.split() and bool(arrays)
    return {
        "t_has_array_param": float(bool(arrays)),
        "t_returns_float": float(any(w in ("float", "double") for w in returns.split())),
        "t_is_void": float("void" in returns.split()),
        "t_has_char_array_param": float(any("char" in p[0].split() for p in arrays)),
        "t_mutates_array_arg": float(mutates),
    }


_REFERENCE_CACHE = {}


def _reference_info(code, entry, reference_trace, display):
    """Main / outer loop of the reference and whether it recurses (cached per code and counts)."""
    counts = None
    if reference_trace is not None:
        per_test = reference_trace.get("per_test") or []
        counts = per_test[display].get("loop_iters", {}) if display < len(per_test) else {}
    key = (code, entry, json.dumps(counts, sort_keys=True))
    if key not in _REFERENCE_CACHE:
        if len(_REFERENCE_CACHE) > 512:
            _REFERENCE_CACHE.clear()
        feats, meta = ast_features(code, entry=entry, trace=reference_trace, display_test=display)
        recursive = None if not meta["parse_ok"] else bool(feats["a_has_recursion"])
        _REFERENCE_CACHE[key] = (meta["loops"], recursive)
    return _REFERENCE_CACHE[key]


def extract(problem, code, trace=None, reference_trace=None, run_reference=None, *,
            run_result=None, static_main_loop=False):
    problem = problem or {}
    display = problem.get("display_test", 0)
    entry = parse_signature(problem.get("signature"))["name"]

    a_feats, a_meta = ast_features(code, entry=entry, trace=None if static_main_loop else trace,
                                   display_test=display)

    ref_loops, recursive = None, None
    variants = problem.get("correct_variants") or []
    if variants:
        ref_loops, recursive = _reference_info(variants[0], entry, reference_trace, display)

    b_feats, b_meta = trace_features(trace, reference_trace, problem,
                                     loops=a_meta["loops"] if a_meta["parse_ok"] else None,
                                     ref_loops=ref_loops if reference_trace is not None else None,
                                     run_result=run_result)
    outputs = collect_outputs(problem, trace, run_result)
    r_feats, r_meta = relation_features(problem, outputs, run_reference, recursive=recursive)
    c_feats = task_features(problem)

    features = {}
    for group, values in ((GROUP_A, a_feats), (GROUP_B, b_feats), (GROUP_R, r_feats), (GROUP_C, c_feats)):
        for name in group:
            features[name] = float(values[name])
    row = np.array([features[name] for name in FEATURES], dtype=np.float64)

    lines, values = {}, {}
    for part in (a_meta, b_meta, r_meta):
        lines.update(part["lines"])
        for name, vals in part["values"].items():
            values[name] = dict(vals)
    for name in _COUNT_FEATURES:                # counts carry their own number
        if not math.isnan(features[name]):
            values.setdefault(name, {})["count"] = int(features[name])
    # The "only the top frame came back" sentence names the thrown-away call when the code shows it.
    discarded = values.get("a_rec_call_discarded")
    if discarded and "r_eq_top_frame_only" in values:
        values["r_eq_top_frame_only"].update(fn=discarded["fn"], arg=discarded["arg"], line=discarded["line"])
        lines["r_eq_top_frame_only"] = discarded["line"]

    tests = None
    if run_result and (run_result.get("tests") or {}).get("total"):
        tests = {"passed": run_result["tests"]["passed"], "total": run_result["tests"]["total"]}
    elif outputs:
        known = [o["passed"] for o in outputs if o["passed"] is not None]
        if known:
            tests = {"passed": int(sum(known)), "total": len(known)}

    meta = {"features": features, "lines": lines, "values": values, "parse_ok": a_meta["parse_ok"],
            "entry": a_meta["entry"], "loops": a_meta["loops"], "has_trace": trace is not None, "tests": tests}
    return row, meta


# ---------------------------------------------------------------- what the templates may use

def _fields(names, *fields):
    return {name: set(fields) for name in names}


# Placeholders `extract` puts in meta["values"][feature] whenever the feature's sentence applies
# (value non-zero, or zero for the sentences that describe an absence). `line` is listed only
# where it is always known. tests/F3 checks every template against this table.
TEMPLATE_FIELDS = {}
TEMPLATE_FIELDS.update(_fields([f for f in GROUP_A if f.startswith("a_main_cond_op_")], "cond", "line"))
TEMPLATE_FIELDS.update(_fields([f for f in GROUP_A if f.startswith("a_bound_form_")], "bound", "var", "line"))
TEMPLATE_FIELDS.update(_fields([f for f in GROUP_A if f.startswith("a_init_form_")], "init", "var", "line"))
TEMPLATE_FIELDS.update(_fields(["a_update_present", "a_update_dir", "a_update_var_is_cond_var",
                                "a_update_in_branch"], "var", "line"))
TEMPLATE_FIELDS.update(_fields([f for f in GROUP_A if f.startswith(("a_outer_cond_op_", "a_inner_cond_op_"))], "line"))
TEMPLATE_FIELDS.update(_fields([f for f in GROUP_A if f.startswith("a_outer_bound_") and f != "a_outer_bound_const1"],
                               "bound", "line"))
TEMPLATE_FIELDS.update(_fields(["a_outer_bound_const1", "a_empty_body_if", "a_empty_body_for", "a_empty_body_while",
                                "a_single_loop_with_swap"], "line"))
TEMPLATE_FIELDS.update(_fields(["a_half_bound"], "bound", "line"))
TEMPLATE_FIELDS.update(_fields(["a_array_loop_start1"], "var", "arr", "line"))
TEMPLATE_FIELDS.update(_fields(["a_array_loop_le_n"], "var", "arr", "bound", "line"))
TEMPLATE_FIELDS.update(_fields(["a_assign_in_cond"], "expr", "var", "value", "line"))
TEMPLATE_FIELDS.update(_fields(["a_decl_noinit_read_first"], "var", "decl_line", "line"))
TEMPLATE_FIELDS.update(_fields(["a_init_inside_loop"], "var", "line"))
TEMPLATE_FIELDS.update(_fields(["a_int_div_into_float", "a_cast_wraps_division", "a_float_literal_in_div",
                                "a_str_literal_compare", "a_array_name_compare", "a_uses_terminator"], "expr", "line"))
TEMPLATE_FIELDS.update(_fields(["a_index_i", "a_index_i_plus_1", "a_index_i_minus_1", "a_index_n", "a_index_n_minus_1",
                                "a_index_const0", "a_index_const1", "a_index_mirror_n_minus_i",
                                "a_index_mirror_n_minus_1_minus_i", "a_has_array_write"], "expr", "arr", "line"))
TEMPLATE_FIELDS.update(_fields(["a_index_pair_plus1"], "expr", "arr", "var", "line"))
TEMPLATE_FIELDS.update(_fields(["a_printf_in_nonvoid", "a_nonvoid_missing_return", "a_char_array_param",
                                "a_has_recursion", "a_base_case_present", "a_base_case_static_reachable"], "fn", "line"))
TEMPLATE_FIELDS.update(_fields(["a_return_const_only"], "const", "line"))
TEMPLATE_FIELDS.update(_fields(["a_reads_all_params"], "line", "param"))        # `param` only when the value is 0
TEMPLATE_FIELDS.update(_fields(["a_return_in_loop_else"], "ret", "line"))
TEMPLATE_FIELDS.update(_fields(["a_assign_in_loop_else"], "var", "line"))
TEMPLATE_FIELDS.update(_fields(["a_mid_assign_no_offset"], "var", "mid", "line"))
TEMPLATE_FIELDS.update(_fields(["a_mid_like_var"], "var", "expr", "line"))
TEMPLATE_FIELDS.update(_fields(["a_swap_no_temp"], "x", "y", "line"))
TEMPLATE_FIELDS.update(_fields(["a_swap_with_temp"], "x", "y", "temp", "line"))
TEMPLATE_FIELDS.update(_fields([f for f in GROUP_A if f.startswith("a_rec_arg_form_")], "fn", "arg", "param", "line"))
TEMPLATE_FIELDS.update(_fields(["a_rec_call_discarded"], "fn", "arg", "line"))
TEMPLATE_FIELDS.update(_fields(_COUNT_FEATURES, "count"))

TEMPLATE_FIELDS.update(_fields(["b_pass_frac"], "passed", "total"))
TEMPLATE_FIELDS.update(_fields(["b_status_timeout", "b_status_runtime_error", "b_return_first_iter"]))
TEMPLATE_FIELDS.update(_fields(["b_step_cap", "b_intdiv_trunc_nonzero", "b_intdiv_into_float", "b_missing_return",
                                "b_str_literal_compare", "b_array_compare", "b_branch_always", "b_branch_never"],
                               "line"))
TEMPLATE_FIELDS.update(_fields(["b_uninit_read"], "var", "line"))
TEMPLATE_FIELDS.update(_fields(["b_oob_read", "b_oob_write"], "arr", "idx", "size", "line"))
TEMPLATE_FIELDS.update(_fields(["b_oob_read_idx_eq_n"], "arr", "n", "line"))
TEMPLATE_FIELDS.update(_fields(["b_oob_read_idx_neg"], "arr", "idx", "line"))
TEMPLATE_FIELDS.update(_fields(["b_iter_delta_mean"], "delta", "fewer", "actual", "expected"))
TEMPLATE_FIELDS.update(_fields(["b_iter_delta_const_pm1"], "actual", "expected"))
TEMPLATE_FIELDS.update(_fields(["b_body_once_vs_many"], "expected"))
TEMPLATE_FIELDS.update(_fields(["b_empty_body_exec"], "kind", "line"))
TEMPLATE_FIELDS.update(_fields(["b_assign_in_cond_rt"], "var", "value", "line"))
TEMPLATE_FIELDS.update(_fields(["b_returned_garbage"], "returned"))
TEMPLATE_FIELDS.update(_fields(["b_printed_nonempty"], "printed"))
TEMPLATE_FIELDS.update(_fields(["b_printed_eq_ref_return"], "printed", "returned"))
TEMPLATE_FIELDS.update(_fields(["b_depth_cap"], "fn", "arg", "depth", "line"))
TEMPLATE_FIELDS.update(_fields(["b_max_depth_ratio"], "depth", "expected"))
TEMPLATE_FIELDS.update(_fields(["b_rec_arg_constant"], "fn", "arg"))
TEMPLATE_FIELDS.update(_fields(["b_rec_arg_growing"], "fn", "first", "last"))
TEMPLATE_FIELDS.update(_fields(["b_base_return_executed"], "fn"))
TEMPLATE_FIELDS.update(_fields(["b_discarded_call_value"], "fn", "line"))
TEMPLATE_FIELDS.update(_fields(["b_window_frozen"], "vars", "line"))
TEMPLATE_FIELDS.update(_fields(["b_multiset_changed"], "v", "w"))
TEMPLATE_FIELDS.update(_fields(["b_sorted_frac"], "pct"))
TEMPLATE_FIELDS.update(_fields(["b_n_outer_passes_ratio"], "actual", "expected"))

TEMPLATE_FIELDS.update(_fields(GROUP_R, "count", "failing"))
TEMPLATE_FIELDS["r_eq_ref_last_only"] |= {"value"}
TEMPLATE_FIELDS["r_eq_ref_first_only"] |= {"value"}
TEMPLATE_FIELDS["r_eq_floor_ref"] |= {"returned", "expected"}
TEMPLATE_FIELDS["r_eq_zero"] |= {"expected"}
TEMPLATE_FIELDS["r_off_by_value_1"] |= {"returned", "expected"}
TEMPLATE_FIELDS["r_is_garbage"] |= {"returned"}
TEMPLATE_FIELDS["r_eq_top_frame_only"] |= {"returned", "expected"}
TEMPLATE_FIELDS["r_eq_shift_without_wrap"] |= {"lost"}
TEMPLATE_FIELDS.update(_fields(GROUP_C))

# Placeholders that are present only sometimes; a template that uses one needs a "fallback" text.
OPTIONAL_FIELDS = {
    "r_eq_top_frame_only": {"fn", "arg", "line"},      # copied from a_rec_call_discarded when the code has it
    "a_reads_all_params": {"param"},
}


# ---------------------------------------------------------------- filling the sentence templates

_TEMPLATES_PATH = Path(__file__).with_name("evidence_templates.json")
_PLACEHOLDER = re.compile(r"\{([a-z_0-9]+)\}")
SENTENCE_KEYS = ("text", "zero", "positive", "negative")
_templates = None


def load_templates():
    """The sentence templates, without the `_doc` entry."""
    global _templates
    if _templates is None:
        data = json.loads(_TEMPLATES_PATH.read_text(encoding="utf-8"))
        _templates = {key: value for key, value in data.items() if not key.startswith("_")}
    return _templates


def placeholders(text):
    return set(_PLACEHOLDER.findall(text))


def _show(value):
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:.4g}"
    return str(value)


def _fill(text, fields):
    needed = placeholders(text)
    if any(fields.get(name) is None or fields.get(name) == "" for name in needed):
        return None
    return _PLACEHOLDER.sub(lambda m: _show(fields[m.group(1)]), text)


def sentence_key(entry, value, features=None):
    """Which sentence of a template entry applies to this feature value, or None.

    `features` (all feature values) is needed for entries with `requires`: a list of other
    features that must be non-zero, e.g. "no base case" is only said about recursive code.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    for name in entry.get("requires", []):
        other = (features or {}).get(name)
        if other is None or other == 0 or (isinstance(other, float) and math.isnan(other)):
            return None
    if "neutral" in entry and value == entry["neutral"]:
        return None                             # e.g. a depth ratio of 1 says nothing
    if value < 0 and "negative" in entry:
        return "negative"
    if value > 0 and "positive" in entry:
        return "positive"
    if value != 0 and "text" in entry:
        return "text"
    if value == 0 and "zero" in entry:
        return "zero"
    return None


def render_evidence(feature, meta, templates=None):
    """One evidence item for `feature` from the `meta` that `extract` returned, or None when
    the feature has no sentence for its current value or a placeholder cannot be filled.

    Returns {"type", "text", "line", "feature"}; the caller adds the weight.
    """
    entry = (templates or load_templates()).get(feature)
    if entry is None:
        return None
    key = sentence_key(entry, meta["features"].get(feature), meta["features"])
    if key is None:
        return None
    fields = dict(meta["values"].get(feature) or {})
    line = meta["lines"].get(feature)
    if line is not None:
        fields.setdefault("line", line)
    candidates = []
    if key == "text":
        for variant in entry.get("variants", []):
            if all(meta["features"].get(name) == wanted for name, wanted in variant["if"].items()):
                candidates.append(variant["text"])
    candidates.append(entry[key])
    if f"{key}_fallback" in entry:
        candidates.append(entry[f"{key}_fallback"])
    for text in candidates:
        filled = _fill(text, fields)
        if filled is not None:
            return {"type": entry["type"], "text": filled, "line": line, "feature": feature}
    return None
