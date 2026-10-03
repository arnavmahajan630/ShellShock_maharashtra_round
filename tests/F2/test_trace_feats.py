"""F2: group-B features on the fixture traces and on hand-built traces.

Run from the repo root:  .venv\\Scripts\\python -m pytest tests/F2 -q

Every expected value below was worked out by hand from the trace contents.
"""
import math

import pytest

from ml.contracts.feature_names import GROUP_B
from ml.contracts.subset import GARBAGE
from ml.features.ast_feats import ast_features
from ml.features.trace_feats import (
    check_passed, collect_outputs, is_garbage, loops_from_trace, trace_features,
)
from tests.F2.helpers import P03, P11, Q17, exact_b, load, make_trace
from tests.fixtures import build as B

P03_REF = P03["correct_variants"][0]
# The reference's own trace: W0's simulation of the correct `i < n` loop (same code as correct_variants[0]).
P03_REF_TRACE, _ = B.run_all(P03, lambda rec, c, n: B.p03_for(rec, c, n, rel_le=False))


def loops_of(code, trace, problem):
    return ast_features(code, trace=trace, display_test=problem["display_test"])[1]["loops"]


P03_REF_LOOPS = loops_of(P03_REF, P03_REF_TRACE, P03)


# ---------------------------------------------------------------- the three fixture traces

def test_fixture_p03_le_oob_read():
    """`i <= n` over cells: one extra pass on every test, reads cells[n], returns garbage."""
    trace = load("traces/p03_le_oob_read.json")
    loops = loops_of(B.P03_LE, trace, P03)
    assert loops["main"] == 3
    feats, meta = trace_features(trace, P03_REF_TRACE, P03, loops=loops, ref_loops=P03_REF_LOOPS)
    exact_b(feats,
            b_oob_read=1, b_oob_read_idx_eq_n=1,
            b_iter_delta_mean=1.0,             # 4-3, 2-1, 5-4, 4-3, 3-2
            b_iter_delta_const_pm1=1,
            b_returned_garbage=1,              # 12 + GARBAGE on the display test
            b_max_depth_ratio=1.0,
            b_base_return_executed=1,          # the only frame returned
            b_sorted_frac=None, b_n_outer_passes_ratio=None)   # not an in-place problem, no nested reference loop
    assert meta["values"]["b_oob_read_idx_eq_n"] == {"arr": "cells", "n": 3, "line": 4}
    assert meta["values"]["b_iter_delta_const_pm1"] == {"actual": 4, "expected": 3, "line": 3}
    assert meta["values"]["b_pass_frac"] == {"passed": 0, "total": 5}


def test_fixture_p03_no_update_step_cap():
    """`while (i < n)` without `i++`: every test hits the step cap."""
    trace = load("traces/p03_no_update_step_cap.json")
    loops = loops_of(B.P03_NO_UPDATE, trace, P03)
    assert loops["main"] == 4 and loops["main_cond_vars"] == ["i", "n"]
    feats, meta = trace_features(trace, P03_REF_TRACE, P03, loops=loops, ref_loops=P03_REF_LOOPS)
    exact_b(feats,
            b_status_timeout=1, b_step_cap=1,
            b_iter_delta_mean=2499 - 13 / 5,   # 2499 passes per test against 3, 1, 4, 3, 2
            b_window_frozen=1,                 # i and n identical on consecutive passes
            b_max_depth_ratio=1.0,
            b_sorted_frac=None, b_n_outer_passes_ratio=None)
    assert meta["values"]["b_window_frozen"] == {"vars": "n = 3, i = 0", "line": 4}
    assert meta["lines"]["b_step_cap"] == 4


def test_fixture_q17_recursion_ok():
    """The correct factorial against itself: nothing fires."""
    trace = load("traces/q17_recursion_ok.json")
    code = Q17["correct_variants"][0]
    loops = loops_of(code, trace, Q17)
    assert loops["main"] is None
    feats, _ = trace_features(trace, trace, Q17, loops=loops, ref_loops=loops)
    exact_b(feats, b_pass_frac=1.0, b_max_depth_ratio=1.0, b_base_return_executed=1,
            b_sorted_frac=None, b_n_outer_passes_ratio=None)


def test_fixture_traces_match_their_simulations():
    """The reference trace used above comes from the same simulation that wrote the fixtures."""
    le, _ = B.run_all(P03, lambda rec, c, n: B.p03_for(rec, c, n, rel_le=True))
    assert le == load("traces/p03_le_oob_read.json")
    assert [pt["loop_iters"] for pt in P03_REF_TRACE["per_test"]] == [{"L3": 3}, {"L3": 1}, {"L3": 4}, {"L3": 3}, {"L3": 2}]


# ---------------------------------------------------------------- no trace, no reference

def test_no_trace_is_all_nan():
    feats, meta = trace_features(None, P03_REF_TRACE, P03)
    assert list(feats) == GROUP_B and all(math.isnan(v) for v in feats.values())
    assert meta == {"lines": {}, "values": {}}
    assert collect_outputs(P03, None) is None


def test_no_reference_leaves_reference_features_nan():
    trace = load("traces/p03_le_oob_read.json")
    feats, _ = trace_features(trace, None, P03, loops=loops_of(B.P03_LE, trace, P03))
    needs_reference = {"b_iter_delta_mean", "b_iter_delta_const_pm1", "b_body_once_vs_many", "b_branch_always",
                       "b_branch_never", "b_max_depth_ratio", "b_return_first_iter", "b_n_outer_passes_ratio"}
    for name in GROUP_B:
        if name in needs_reference or name == "b_sorted_frac":
            assert math.isnan(feats[name]), name
        else:
            assert not math.isnan(feats[name]), name
    assert feats["b_oob_read_idx_eq_n"] == 1 and feats["b_pass_frac"] == 0


def test_loops_fall_back_to_the_trace():
    trace = load("traces/p03_le_oob_read.json")
    assert loops_from_trace(trace, 0)["main"] == 3
    feats, _ = trace_features(trace, P03_REF_TRACE, P03)          # no AST information at all
    assert feats["b_iter_delta_const_pm1"] == 1 and feats["b_iter_delta_mean"] == 1.0
    nested = make_trace([{"loop_iters": {"L2": 2, "L3": 3}}])
    assert loops_from_trace(nested) == {"main": 3, "outer": 2, "inner": None, "all": [2, 3], "main_var": None,
                                        "main_cond_vars": [], "inline": [], "selected_by": "trace_only"}


# ---------------------------------------------------------------- events the fixtures do not contain

def p03_trace(returned, events=(), printed=None, **extra):
    """A P03-shaped trace with the reference's iteration counts and the given return values."""
    iters = [3, 1, 4, 3, 2]
    per_test = [{"returned": r, "printed": (printed[i] if printed else ""), "loop_iters": {"L3": iters[i]},
                 "effects_count": {"read_cell": iters[i], "call": 1, "ret": 1}} for i, r in enumerate(returned)]
    return make_trace(per_test, events=events, **extra)


P03_LOOPS = {"main": 3, "outer": None, "inner": None, "all": [3], "inline": [], "main_var": "i",
             "main_cond_vars": ["i", "n"], "selected_by": "trace"}


def b_feats(trace, problem=P03, reference=P03_REF_TRACE, loops=P03_LOOPS, ref_loops=P03_REF_LOOPS, **kwargs):
    return trace_features(trace, reference, problem, loops=loops, ref_loops=ref_loops, **kwargs)


def test_uninitialised_read():
    """`int total;` then `total += cells[i]`: garbage plus the real sum comes back."""
    sums = [12, 5, 10, 9, 8]
    events = [{"type": "uninit_read", "var": "total", "line": 4, "test": t} for t in range(5)]
    feats, meta = b_feats(p03_trace([GARBAGE + s for s in sums], events))
    exact_b(feats, b_uninit_read=1, b_returned_garbage=1, b_max_depth_ratio=1.0, b_base_return_executed=1,
            b_sorted_frac=None, b_n_outer_passes_ratio=None)
    assert meta["values"]["b_uninit_read"] == {"var": "total", "line": 4}
    assert meta["values"]["b_returned_garbage"] == {"returned": GARBAGE + 12}


def test_integer_division_flags_are_separate():
    both = [{"type": "intdiv", "remainder_nonzero": True, "into_float": True, "line": 6, "test": 0}]
    feats, _ = b_feats(p03_trace([12, 5, 10, 9, 8], both))
    assert feats["b_intdiv_trunc_nonzero"] == 1 and feats["b_intdiv_into_float"] == 1
    exact_only = [{"type": "intdiv", "remainder_nonzero": False, "into_float": True, "line": 6, "test": 0}]
    feats, _ = b_feats(p03_trace([12, 5, 10, 9, 8], exact_only))
    assert feats["b_intdiv_trunc_nonzero"] == 0 and feats["b_intdiv_into_float"] == 1
    index_only = [{"type": "intdiv", "remainder_nonzero": True, "into_float": False, "line": 6, "test": 0}]
    feats, meta = b_feats(p03_trace([12, 5, 10, 9, 8], index_only))
    assert feats["b_intdiv_trunc_nonzero"] == 1 and feats["b_intdiv_into_float"] == 0
    assert feats["b_pass_frac"] == 1.0 and meta["lines"]["b_intdiv_trunc_nonzero"] == 6


P11_REF_TRACE = make_trace([{"returned": 1, "branch": {"B2": {"true": 1}}}] +
                           [{"returned": 0, "branch": {"B2": {"false": 1}}}] * 4)
P11_LOOPS = {"main": None, "outer": None, "inner": None, "all": [], "inline": [], "main_var": None,
             "main_cond_vars": [], "selected_by": "none"}


def test_assignment_in_condition_makes_the_branch_constant():
    """`if (code = 42)` is true on all five tests; the reference's branch is true once."""
    events = [{"type": "assign_in_cond", "var": "code", "value": 42, "line": 2, "test": t} for t in range(5)]
    trace = make_trace([{"returned": 1, "branch": {"B2": {"true": 1}}}] * 5, events=events)
    feats, meta = b_feats(trace, P11, P11_REF_TRACE, P11_LOOPS, P11_LOOPS)
    exact_b(feats, b_pass_frac=0.2, b_assign_in_cond_rt=1, b_branch_always=1, b_max_depth_ratio=1.0,
            b_base_return_executed=1, b_sorted_frac=None, b_n_outer_passes_ratio=None)
    assert meta["values"]["b_assign_in_cond_rt"] == {"var": "code", "value": 42, "line": 2}
    assert meta["lines"]["b_branch_always"] == 2


def test_branch_never_and_reference_must_vary():
    never = make_trace([{"returned": 0, "branch": {"B2": {"false": 1}}}] * 5)
    feats, _ = b_feats(never, P11, P11_REF_TRACE, P11_LOOPS, P11_LOOPS)
    assert feats["b_branch_never"] == 1 and feats["b_branch_always"] == 0
    constant_reference = make_trace([{"returned": 0, "branch": {"B2": {"false": 1}}}] * 5)
    feats, _ = b_feats(never, P11, constant_reference, P11_LOOPS, P11_LOOPS)
    assert feats["b_branch_never"] == 0            # the reference's branch does not vary either


def test_stray_semicolon_after_if():
    """W0's simulation of `if (code == 42); { return 1; }`: the event fires, the branch still varies."""
    trace, outcomes = B.run_all(P11, B.p11_semi)
    feats, meta = b_feats(trace, P11, P11_REF_TRACE, P11_LOOPS, P11_LOOPS)
    exact_b(feats, b_pass_frac=0.2, b_empty_body_exec=1, b_max_depth_ratio=1.0, b_base_return_executed=1,
            b_sorted_frac=None, b_n_outer_passes_ratio=None)
    assert meta["values"]["b_empty_body_exec"] == {"kind": "if", "line": 2}
    run_result = {"status": "ok", "backend": "interp", "tests": B.tests_object(P11, outcomes)}
    assert b_feats(trace, P11, P11_REF_TRACE, P11_LOOPS, P11_LOOPS, run_result=run_result)[0]["b_pass_frac"] == 0.2


def test_print_instead_of_return():
    """`printf("%d", total);` and no return: prints the right value, returns garbage."""
    sums = [12, 5, 10, 9, 8]
    events = [{"type": "missing_return", "line": 6, "test": t} for t in range(5)]
    trace = p03_trace([GARBAGE] * 5, events, printed=[str(s) for s in sums])
    feats, meta = b_feats(trace)
    exact_b(feats, b_missing_return=1, b_returned_garbage=1, b_printed_nonempty=1, b_printed_eq_ref_return=1.0,
            b_max_depth_ratio=1.0, b_base_return_executed=1, b_sorted_frac=None, b_n_outer_passes_ratio=None)
    assert meta["values"]["b_printed_eq_ref_return"] == {"printed": "12", "returned": GARBAGE}


def test_debug_printf_with_correct_return_is_not_the_m10_signal():
    trace = p03_trace([12, 5, 10, 9, 8], printed=["i=0\ni=1\ni=2\n", "i=0\n", "i=0\n", "i=0\n", "i=0\n"])
    feats, _ = b_feats(trace)
    assert feats["b_printed_nonempty"] == 1 and feats["b_printed_eq_ref_return"] == 0
    assert feats["b_pass_frac"] == 1.0 and feats["b_returned_garbage"] == 0


def test_printed_equals_expected_without_a_reference_trace():
    trace = p03_trace([0] * 5, printed=["12", "5", "10", "oops", ""])
    feats, _ = trace_features(trace, None, P03, loops=P03_LOOPS)
    assert feats["b_printed_eq_ref_return"] == pytest.approx(3 / 5)


Q17_REF = load("traces/q17_recursion_ok.json")


def rec_trace(arg_chain, status="timeout", events=(), returned=None, rets=(), max_depth=None):
    """A factorial-shaped display test: one call per element of `arg_chain`, then the given returns."""
    steps = [{"line": 2, "vars": {"n": a}, "effects": [f"call:factorial:{d + 1}:{a}"]} for d, a in enumerate(arg_chain)]
    steps += [{"line": 3, "vars": {}, "effects": [f"ret:factorial:{d}:{v}"]} for d, v in rets]
    depth = max_depth or len(arg_chain)
    per_test = [{"status": status, "returned": returned, "max_depth": depth,
                 "effects_count": {"call": len(arg_chain)}}] + [{"status": status, "max_depth": depth}] * 4
    return make_trace(per_test, events=events, steps=steps)


def test_depth_cap_without_base_case():
    """`return n * factorial(n - 1);` with no base: the argument shrinks forever."""
    events = [{"type": "depth_cap_hit", "fn": "factorial", "last_args": [-97], "line": 2, "test": 0}]
    trace = rec_trace([3, 2, 1, 0, -1, -2], events=events, max_depth=100)
    feats, meta = b_feats(trace, Q17, Q17_REF, P11_LOOPS, P11_LOOPS)
    exact_b(feats, b_status_timeout=1, b_depth_cap=1, b_max_depth_ratio=10.0,     # 100 against the reference's 10
            b_sorted_frac=None, b_n_outer_passes_ratio=None)
    assert meta["values"]["b_depth_cap"] == {"fn": "factorial", "arg": "-97", "depth": 100, "line": 2}


def test_recursive_argument_constant_or_growing():
    events = [{"type": "depth_cap_hit", "fn": "factorial", "last_args": [3], "line": 2, "test": 0}]
    same, meta = b_feats(rec_trace([3, 3, 3, 3], events=events, max_depth=100), Q17, Q17_REF, P11_LOOPS, P11_LOOPS)
    assert same["b_rec_arg_constant"] == 1 and same["b_rec_arg_growing"] == 0 and same["b_base_return_executed"] == 0
    assert meta["values"]["b_rec_arg_constant"] == {"fn": "factorial", "arg": "3"}
    grow, meta = b_feats(rec_trace([3, 4, 5, 6], events=events, max_depth=100), Q17, Q17_REF, P11_LOOPS, P11_LOOPS)
    assert grow["b_rec_arg_constant"] == 0 and grow["b_rec_arg_growing"] == 1
    assert meta["values"]["b_rec_arg_growing"] == {"fn": "factorial", "first": "3", "last": "6"}
    shrink, _ = b_feats(Q17_REF, Q17, Q17_REF, P11_LOOPS, P11_LOOPS)
    assert shrink["b_rec_arg_constant"] == 0 and shrink["b_rec_arg_growing"] == 0


def test_discarded_call_value():
    """`factorial(n - 1); return n;`: recursion ends normally, the value is dropped."""
    events = [{"type": "discarded_call_value", "fn": "factorial", "line": 5, "test": 0}]
    trace = rec_trace([3, 2, 1], status="ok", events=events, returned=3, rets=[(3, 1), (2, 2), (1, 3)])
    feats, meta = b_feats(trace, Q17, Q17_REF, P11_LOOPS, P11_LOOPS)
    assert feats["b_discarded_call_value"] == 1 and feats["b_base_return_executed"] == 1
    assert feats["b_depth_cap"] == 0 and feats["b_max_depth_ratio"] == pytest.approx(0.3)
    assert meta["values"]["b_discarded_call_value"] == {"fn": "factorial", "line": 5}


def test_base_return_needs_a_leaf_frame():
    """Returns that only happen after a deeper call are not base-case returns."""
    trace = rec_trace([3, 2], status="timeout", rets=[(1, 6)])
    assert b_feats(trace, Q17, Q17_REF, P11_LOOPS, P11_LOOPS)[0]["b_base_return_executed"] == 0
    void_ok = make_trace([{"status": "ok"}], steps=[{"line": 2, "effects": ["call:bubble_sort:1:a,3"]}])
    assert b_feats(void_ok, Q17, Q17_REF, P11_LOOPS, P11_LOOPS)[0]["b_base_return_executed"] == 1


BSEARCH_LOOPS = {"main": 4, "outer": None, "inner": None, "all": [4], "inline": [], "main_var": "low",
                 "main_cond_vars": ["low", "high"], "selected_by": "trace"}


def bsearch_steps(windows):
    steps = []
    for low, high in windows:
        steps.append({"line": 4, "vars": {"low": low, "high": high, "mid": None}})
        steps.append({"line": 5, "vars": {"low": low, "high": high, "mid": (low + high) // 2}})
        steps.append({"line": 9, "vars": {"low": low, "high": high, "mid": (low + high) // 2}})
    return steps


def test_window_frozen_on_low_equals_mid():
    """`low = mid`: the window goes 0..7, 3..7, 5..7, 6..7 and then stays 6..7."""
    stuck = make_trace([{"status": "timeout", "loop_iters": {"L4": 2499}}],
                       events=[{"type": "step_cap_hit", "line": 4, "test": 0}],
                       steps=bsearch_steps([(0, 7), (3, 7), (5, 7), (6, 7), (6, 7), (6, 7)]), truncated=True)
    reference = make_trace([{"returned": 7, "loop_iters": {"L4": 4}}])
    feats, meta = trace_features(stuck, reference, None, loops=BSEARCH_LOOPS, ref_loops=BSEARCH_LOOPS)
    assert feats["b_window_frozen"] == 1 and feats["b_step_cap"] == 1
    assert meta["values"]["b_window_frozen"] == {"vars": "low = 6, high = 7", "line": 4}
    moving = make_trace([{"returned": 7, "loop_iters": {"L4": 4}}],
                        steps=bsearch_steps([(0, 7), (4, 7), (6, 7), (7, 7)]))
    assert trace_features(moving, reference, None, loops=BSEARCH_LOOPS, ref_loops=BSEARCH_LOOPS)[0]["b_window_frozen"] == 0


def test_for_header_steps_are_one_pass_not_two():
    """A for-header records its update and its test on the same line; that is not a frozen window."""
    trace = load("traces/p03_le_oob_read.json")
    feats, _ = b_feats(trace, loops=loops_of(B.P03_LE, trace, P03))
    assert feats["b_window_frozen"] == 0


# ---------------------------------------------------------------- arrays changed in place

Q06 = dict(B.Q06, family="bubble_sort", split="train", correct_variants=[], allowed_ops=[])
Q06_LOOPS = {"main": 3, "outer": 2, "inner": 3, "all": [2, 3], "inline": [], "main_var": "j",
             "main_cond_vars": ["j", "n", "i"], "selected_by": "trace"}
# A correct bubble sort on Q06's tests ([3,1,2], [2,1], [1,2,3], [4,3,2,1]): outer n-1 passes, inner n(n-1)/2.
Q06_REF_TRACE = make_trace([{"loop_iters": {"L2": 2, "L3": 3}}, {"loop_iters": {"L2": 1, "L3": 1}},
                            {"loop_iters": {"L2": 2, "L3": 3}}, {"loop_iters": {"L2": 3, "L3": 6}}])


def test_swap_without_temp_changes_the_multiset():
    """W0's simulation of `a[j] = a[j+1]; a[j+1] = a[j];`. Final arrays: [1,1,2], [1,1], [1,2,3], [1,1,1,1]."""
    trace, outcomes = B.run_all(Q06, B.q06_no_temp)
    assert [o[2][0] for o in outcomes] == [[1, 1, 2], [1, 1], [1, 2, 3], [1, 1, 1, 1]]
    run_result = {"status": "ok", "backend": "interp", "tests": B.tests_object(Q06, outcomes)}
    feats, meta = trace_features(trace, Q06_REF_TRACE, Q06, loops=Q06_LOOPS, ref_loops=Q06_LOOPS,
                                 run_result=run_result)
    exact_b(feats, b_pass_frac=0.25, b_multiset_changed=1, b_sorted_frac=1.0, b_n_outer_passes_ratio=1.0,
            b_max_depth_ratio=1.0, b_base_return_executed=1)
    assert meta["values"]["b_multiset_changed"] == {"v": 1, "w": 3}     # 1 appears twice, 3 is gone
    assert meta["values"]["b_n_outer_passes_ratio"] == {"actual": 8, "expected": 8, "line": 2}


def test_final_array_of_the_display_test_is_rebuilt_from_write_effects():
    """Without a run result only the display test's array is known (per_test has no array0)."""
    trace, _ = B.run_all(Q06, B.q06_no_temp)
    outputs = collect_outputs(Q06, trace)
    assert [o["array0"] for o in outputs] == [[1, 1, 2], None, None, None]
    assert [o["passed"] for o in outputs] == [False, None, None, None]
    feats, _ = trace_features(trace, Q06_REF_TRACE, Q06, loops=Q06_LOOPS, ref_loops=Q06_LOOPS)
    assert feats["b_pass_frac"] == 0.0 and feats["b_multiset_changed"] == 1 and feats["b_sorted_frac"] == 1.0


def test_single_pass_sort():
    """One bubble pass and no outer loop: 4 passes in total against the reference's 8."""
    finals = [[1, 2, 3], [1, 2], [1, 2, 3], [3, 2, 1, 4]]
    trace = make_trace([{"loop_iters": {"L2": n}} for n in (2, 1, 2, 3)])
    results = [{"args": t["args"], "expected": t["expect"], "got": {"array0": f}, "pass": f == t["expect"]["array0"]}
               for t, f in zip(Q06["tests"], finals)]
    run_result = {"status": "ok", "backend": "interp", "tests": {"passed": 3, "total": 4, "results": results}}
    single = {"main": 2, "outer": None, "inner": None, "all": [2], "inline": [], "main_var": "j",
              "main_cond_vars": ["j", "n"], "selected_by": "trace"}
    feats, _ = trace_features(trace, Q06_REF_TRACE, Q06, loops=single, ref_loops=Q06_LOOPS, run_result=run_result)
    assert feats["b_n_outer_passes_ratio"] == 0.5
    assert feats["b_sorted_frac"] == pytest.approx((1 + 1 + 1 + 1 / 3) / 4)     # [3,2,1,4]: one pair of three in order
    assert feats["b_multiset_changed"] == 0 and feats["b_pass_frac"] == 0.75


# ---------------------------------------------------------------- loop behaviour against the reference

SEARCH = {"display_test": 0, "tests": [
    {"args": [[4, 8, 6], 3, 6], "expect": {"returned": 2}},
    {"args": [[4, 8, 6], 3, 4], "expect": {"returned": 0}},
    {"args": [[4, 8, 6, 1], 4, 9], "expect": {"returned": -1}},
    {"args": [[5], 1, 7], "expect": {"returned": -1}},
]}
SEARCH_REF = make_trace([{"returned": 2, "loop_iters": {"L2": 3}}, {"returned": 0, "loop_iters": {"L2": 1}},
                         {"returned": -1, "loop_iters": {"L2": 4}}, {"returned": -1, "loop_iters": {"L2": 1}}])
SEARCH_LOOPS = {"main": 2, "outer": None, "inner": None, "all": [2], "inline": [], "main_var": "i",
                "main_cond_vars": ["i", "n"], "selected_by": "trace"}


def test_return_during_the_first_iteration():
    """`else return -1;` inside the loop: one pass on every test; the reference needed 3 and 4 on two of them."""
    early = make_trace([{"returned": -1, "loop_iters": {"L2": 1}}, {"returned": 0, "loop_iters": {"L2": 1}},
                        {"returned": -1, "loop_iters": {"L2": 1}}, {"returned": -1, "loop_iters": {"L2": 1}}])
    feats, _ = trace_features(early, SEARCH_REF, SEARCH, loops=SEARCH_LOOPS, ref_loops=SEARCH_LOOPS)
    assert feats["b_return_first_iter"] == 1.0          # both eligible tests (reference >= 2 passes)
    assert feats["b_pass_frac"] == 0.75                 # only the first test gives a wrong answer
    assert feats["b_iter_delta_mean"] == pytest.approx((-2 + 0 - 3 + 0) / 4)
    assert feats["b_iter_delta_const_pm1"] == 0
    half = make_trace([{"returned": 2, "loop_iters": {"L2": 3}}, {"returned": 0, "loop_iters": {"L2": 1}},
                       {"returned": -1, "loop_iters": {"L2": 1}}, {"returned": -1, "loop_iters": {"L2": 1}}])
    assert trace_features(half, SEARCH_REF, SEARCH, loops=SEARCH_LOOPS,
                          ref_loops=SEARCH_LOOPS)[0]["b_return_first_iter"] == 0.5
    assert trace_features(SEARCH_REF, SEARCH_REF, SEARCH, loops=SEARCH_LOOPS,
                          ref_loops=SEARCH_LOOPS)[0]["b_return_first_iter"] == 0


def test_iteration_delta_minus_one_on_every_test():
    """`i < n - 1`: one pass fewer everywhere."""
    trace = p03_trace([6, 0, 6, 0, 7])
    for pt, n in zip(trace["per_test"], [2, 0, 3, 2, 1]):
        pt["loop_iters"] = {"L3": n} if n else {}
    feats, meta = b_feats(trace)
    assert feats["b_iter_delta_const_pm1"] == 1 and feats["b_iter_delta_mean"] == -1.0
    assert meta["values"]["b_iter_delta_const_pm1"] == {"actual": 2, "expected": 3, "line": 3}


def test_delta_must_be_the_same_on_every_test():
    trace = p03_trace([12, 5, 10, 9, 8])
    trace["per_test"][2]["loop_iters"] = {"L3": 5}          # +1 on one test only
    feats, _ = b_feats(trace)
    assert feats["b_iter_delta_const_pm1"] == 0 and feats["b_iter_delta_mean"] == pytest.approx(0.2)


def semicolon_for_trace():
    """`for (i = 0; i < n; i++);` on line 3, then the block on line 5 once, on [2, 4, 6]."""
    steps = [{"line": 2, "vars": {"total": 0}}, {"line": 3, "vars": {"i": 0}}]
    for i in range(3):
        steps += [{"line": 3, "vars": {"i": i}}, {"line": 3, "vars": {"i": i + 1}}]
    steps += [{"line": 3, "vars": {"i": 3}}, {"line": 5, "vars": {"i": 3}}, {"line": 7, "vars": {"i": 3}}]
    events = [{"type": "empty_body", "kind": "for", "line": 3, "test": t} for t in range(5)]
    events += [{"type": "oob_read", "arr": "cells", "idx": n, "size": n, "line": 5, "test": t}
               for t, n in enumerate([3, 1, 4, 3, 2])]
    per_test = [{"returned": GARBAGE, "loop_iters": {"L3": n}} for n in (3, 1, 4, 3, 2)]
    return make_trace(per_test, events=events, steps=steps)


def test_body_ran_once_where_the_reference_looped():
    feats, meta = b_feats(semicolon_for_trace())
    exact_b(feats, b_oob_read=1, b_oob_read_idx_eq_n=1, b_body_once_vs_many=1, b_empty_body_exec=1,
            b_returned_garbage=1, b_max_depth_ratio=1.0, b_base_return_executed=1, b_sorted_frac=None,
            b_n_outer_passes_ratio=None)
    assert meta["values"]["b_body_once_vs_many"] == {"expected": 3}


def test_body_on_the_header_line_is_not_body_once():
    """`for (...) total += cells[i];` on one line: every body step shares the header line."""
    steps = [{"line": 2, "vars": {}}, {"line": 3, "vars": {}}]
    for _ in range(3):
        steps += [{"line": 3, "vars": {}}] * 3              # test, body, update
    steps += [{"line": 3, "vars": {}}, {"line": 4, "vars": {}}]
    trace = make_trace([{"returned": 12, "loop_iters": {"L3": 3}}] +
                       [{"returned": r, "loop_iters": {"L3": n}} for r, n in ((5, 1), (10, 4), (9, 3), (8, 2))],
                       steps=steps)
    inline = dict(P03_LOOPS, inline=[3])
    assert b_feats(trace, loops=inline)[0]["b_body_once_vs_many"] == 0          # the AST says the body is inline
    assert trace_features(trace, P03_REF_TRACE, P03)[0]["b_body_once_vs_many"] == 0   # trace-only step count
    assert b_feats(trace)[0]["b_body_once_vs_many"] == 1    # told (wrongly) that line 3 is header only


def test_out_of_bounds_shapes():
    events = [{"type": "oob_read", "arr": "a", "idx": -1, "size": 3, "line": 4, "test": 0},
              {"type": "oob_write", "arr": "a", "idx": 3, "size": 3, "line": 5, "test": 0}]
    feats, meta = b_feats(p03_trace([12, 5, 10, 9, 8], events))
    assert feats["b_oob_read"] == 1 and feats["b_oob_read_idx_neg"] == 1 and feats["b_oob_read_idx_eq_n"] == 0
    assert feats["b_oob_write"] == 1
    assert meta["values"]["b_oob_write"] == {"arr": "a", "idx": 3, "size": 3, "line": 5}
    past = [{"type": "oob_read", "arr": "a", "idx": 5, "size": 3, "line": 4, "test": 0}]
    feats, _ = b_feats(p03_trace([12, 5, 10, 9, 8], past))
    assert feats["b_oob_read"] == 1 and feats["b_oob_read_idx_eq_n"] == 0 and feats["b_oob_read_idx_neg"] == 0


def test_string_compare_events_and_runtime_error():
    events = [{"type": "str_literal_compare", "line": 4, "test": 0}, {"type": "array_compare", "line": 6, "test": 1},
              {"type": "div_zero", "line": 7, "test": 2}]
    trace = p03_trace([12, 5, None, 9, 8], events)
    trace["per_test"][2]["status"] = trace["status"] = "runtime_error"
    feats, meta = b_feats(trace)
    assert feats["b_str_literal_compare"] == 1 and feats["b_array_compare"] == 1
    assert feats["b_status_runtime_error"] == 1 and feats["b_status_timeout"] == 0
    assert feats["b_pass_frac"] == 0.8 and meta["lines"]["b_array_compare"] == 6


# ---------------------------------------------------------------- helpers

def test_check_passed():
    ok = {"status": "ok", "returned": 2.5004, "printed": "", "effects_count": {"fire": 3}, "max_depth": 4,
          "array0": [1, 2]}
    assert check_passed({"returned": 2.5}, ok) is True                 # floats within 1e-3
    assert check_passed({"returned": 2.6}, ok) is False
    assert check_passed({"fire": 3, "max_depth_le": 30}, ok) is True
    assert check_passed({"fire": 3, "max_depth_le": 3}, ok) is False
    assert check_passed({"array0": [1, 2]}, ok) is True
    assert check_passed({"array0": [1, 2]}, dict(ok, array0=None)) is None     # final array not known
    assert check_passed({"returned": 2.5}, dict(ok, status="timeout")) is False


def test_is_garbage():
    assert is_garbage(GARBAGE) and is_garbage(GARBAGE + 12) and is_garbage(GARBAGE - 40)
    assert not is_garbage(0) and not is_garbage(None) and not is_garbage(-1) and not is_garbage("x")
