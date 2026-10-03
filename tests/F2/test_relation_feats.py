"""F2: group-R features with a pure-Python reference function.

The learner outputs are written by hand per test; the reference is ordinary Python, passed in
as `run_reference`, so no C backend is involved. Expected fractions are worked out by hand.
"""
import math

import pytest

from ml.contracts.feature_names import GROUP_R
from ml.contracts.subset import GARBAGE
from ml.features.relation_feats import NEEDS_REFERENCE, make_run_reference, relation_features
from ml.features.trace_feats import collect_outputs
from tests.F2.helpers import P03, Q17, exact_r, load, make_trace
from tests.fixtures import build as B


def outputs_for(problem, returned=None, arrays=None, effects=None):
    """Learner outputs per test, with pass / fail decided from the problem's `expect`."""
    per_test = []
    for i in range(len(problem["tests"])):
        pt = {"returned": returned[i] if returned else None}
        if effects:
            pt["effects_count"] = effects[i]
        per_test.append(pt)
    trace = make_trace(per_test)
    run_result = None
    if arrays:
        results = [{"args": t["args"], "expected": t["expect"], "got": {"array0": a}, "pass": a == t["expect"]["array0"]}
                   for t, a in zip(problem["tests"], arrays)]
        run_result = {"status": "ok", "backend": "interp",
                      "tests": {"passed": sum(r["pass"] for r in results), "total": len(results), "results": results}}
    return collect_outputs(problem, trace, run_result)


def total_energy(args):
    cells, n = args
    return sum(cells[:n])


# P03 tests: [2,4,6] -> 12, [5] -> 5, [1,2,3,4] -> 10, [0,0,9] -> 9, [7,1] -> 8


def test_accumulator_reset_returns_the_last_cell():
    """`total = 0;` inside the loop: returns the last element. Tests [5] and [0,0,9] pass by luck."""
    outputs = outputs_for(P03, returned=[6, 5, 4, 9, 1])
    assert [o["passed"] for o in outputs] == [False, True, False, True, False]
    feats, meta = relation_features(P03, outputs, total_energy, recursive=False)
    exact_r(feats,
            r_eq_ref_last_only=1.0,        # ref([6]) = 6, ref([4]) = 4, ref([1]) = 1
            r_eq_last_check_only=1.0,      # same single-element runs
            r_eq_ref_drop_first=1 / 3,     # only [7,1]: ref([1]) = 1
            r_eq_ref_drop_last=1 / 3,      # only [2,4,6]: ref([2,4]) = 6, a coincidence of the values
            r_eq_ref_n_minus_1=1 / 3,      # same run as drop_last
            r_eq_first_check_only=1 / 3)   # only [2,4,6]: ref([2,4]) = 6
    assert meta["values"]["r_eq_ref_last_only"] == {"value": 6, "test": 0, "count": 3, "failing": 3}


def test_top_frame_relation_is_off_for_a_non_recursive_reference():
    outputs = outputs_for(P03, returned=[6, 5, 4, 9, 1])
    unknown, _ = relation_features(P03, outputs, total_energy)             # not told: the relation is tested
    assert unknown["r_eq_top_frame_only"] == 1.0                           # the last cell is also "a[n-1]"
    assert relation_features(P03, outputs, total_energy, recursive=False)[0]["r_eq_top_frame_only"] == 0


def test_first_cell_skipped():
    """`for (i = 1; i < n; i++)`: sums a[1:]. [5] gives 0."""
    outputs = outputs_for(P03, returned=[10, 0, 9, 9, 1])
    assert [o["passed"] for o in outputs] == [False, False, False, True, False]
    feats, _ = relation_features(P03, outputs, total_energy, recursive=False)
    exact_r(feats,
            r_eq_ref_drop_first=3 / 4,     # the three arrays with two or more cells; [5] has nothing to drop
            r_eq_ref_last_only=1 / 4,      # [7,1]: ref([1]) = 1
            r_eq_last_check_only=2 / 4,    # [7,1] again, and [2,4,6] through its last two cells: ref([4,6]) = 10
            r_eq_ref_n_minus_1=1 / 4,      # [5] with n = 0: ref gives 0, and so did the learner
            r_eq_zero=1 / 4,               # [5] -> 0
            r_off_by_value_1=1 / 4)        # [1,2,3,4]: 9 against 10


def test_last_cell_skipped_matches_n_minus_1():
    """`i < n - 1`: sums a[:-1]."""
    outputs = outputs_for(P03, returned=[6, 0, 6, 0, 7])
    feats, _ = relation_features(P03, outputs, total_energy, recursive=False)
    assert feats["r_eq_ref_drop_last"] == pytest.approx(4 / 5)      # [5] has nothing to drop
    assert feats["r_eq_ref_n_minus_1"] == 1.0                       # ref([5], 0) = 0 as well
    assert feats["r_eq_ref_n_plus_1"] == 0                          # never tried with an array: n + 1 reads past the end
    assert feats["r_eq_ref_first_only"] == pytest.approx(2 / 5)     # [7,1]: ref([7]) = 7; [0,0,9]: ref([0]) = 0
    assert feats["r_eq_zero"] == pytest.approx(2 / 5)


def test_garbage_output():
    outputs = outputs_for(P03, returned=[GARBAGE + s for s in (12, 5, 10, 9, 8)])
    feats, meta = relation_features(P03, outputs, total_energy, recursive=False)
    exact_r(feats, r_is_garbage=1.0)
    assert meta["values"]["r_is_garbage"]["returned"] == GARBAGE + 12


def test_timeouts_hold_no_relation():
    trace = load("traces/p03_no_update_step_cap.json")
    outputs = collect_outputs(P03, trace)
    assert [o["passed"] for o in outputs] == [False] * 5
    exact_r(relation_features(P03, outputs, total_energy, recursive=False)[0])


def test_fixture_p03_le_is_garbage_only():
    outputs = collect_outputs(P03, load("traces/p03_le_oob_read.json"))
    exact_r(relation_features(P03, outputs, total_energy, recursive=False)[0], r_is_garbage=1.0)


def test_all_tests_pass_gives_zeros():
    outputs = collect_outputs(Q17, load("traces/q17_recursion_ok.json"))
    feats, meta = relation_features(Q17, outputs, lambda args: math.factorial(args[0]), recursive=True)
    exact_r(feats)
    assert meta["values"] == {}


# ---------------------------------------------------------------- no trace, no reference

def test_no_outputs_is_all_nan():
    feats, _ = relation_features(P03, None, total_energy)
    assert list(feats) == GROUP_R and all(math.isnan(v) for v in feats.values())


def test_without_run_reference_only_counterfactual_features_are_nan():
    outputs = outputs_for(P03, returned=[0, 5, 11, 9, 8])
    feats, _ = relation_features(P03, outputs, None, recursive=False)
    for name in GROUP_R:
        assert math.isnan(feats[name]) == (name in NEEDS_REFERENCE), name
    assert feats["r_eq_zero"] == 0.5 and feats["r_off_by_value_1"] == 0.5


def test_reference_that_raises_or_returns_none_is_a_false_relation():
    outputs = outputs_for(P03, returned=[6, 5, 4, 9, 1])

    def broken(args):
        if len(args[0]) == 1:
            raise ValueError("cannot run")
        return None

    feats, _ = relation_features(P03, outputs, broken, recursive=False)
    exact_r(feats)


def test_reference_is_called_once_per_distinct_input():
    seen = []

    def counting(args):
        seen.append(repr(args))
        return total_energy(args)

    relation_features(P03, outputs_for(P03, returned=[6, 5, 4, 9, 1]), counting, recursive=False)
    assert len(seen) == len(set(seen))


# ---------------------------------------------------------------- scalars: n ± 1, floor, zero

FIRE = {"display_test": 0, "tests": [{"args": [3], "expect": {"fire": 3}}, {"args": [1], "expect": {"fire": 1}},
                                      {"args": [0], "expect": {"fire": 0}}, {"args": [5], "expect": {"fire": 5}}]}


def fire_shots(args):
    return {"fire": max(args[0], 0)}


def test_one_shot_too_many_and_too_few():
    """Effect counts are outputs too: `i <= n` fires n + 1 times."""
    more = outputs_for(FIRE, effects=[{"fire": 4}, {"fire": 2}, {"fire": 1}, {"fire": 6}])
    exact_r(relation_features(FIRE, more, fire_shots, recursive=False)[0], r_eq_ref_n_plus_1=1.0)
    fewer = outputs_for(FIRE, effects=[{"fire": 2}, {"fire": 0}, {"fire": 0}, {"fire": 4}])
    feats, _ = relation_features(FIRE, fewer, fire_shots, recursive=False)
    assert [o["passed"] for o in fewer] == [False, False, True, False]
    exact_r(feats, r_eq_ref_n_minus_1=1.0)


AVG = {"display_test": 0, "tests": [{"args": [[1, 2], 2], "expect": {"returned": 1.5}},
                                     {"args": [[2, 4], 2], "expect": {"returned": 3.0}},
                                     {"args": [[1, 2, 4], 3], "expect": {"returned": 2.3333}},
                                     {"args": [[7], 1], "expect": {"returned": 7.0}}]}


def avg_fuel(args):
    tanks, n = args
    return sum(tanks[:n]) / n if n else None


def test_floor_of_the_reference():
    """`sum / n` with ints: 1.5 -> 1, 2.33 -> 2; the two exact averages still pass."""
    outputs = outputs_for(AVG, returned=[1.0, 3.0, 2.0, 7.0])
    assert [o["passed"] for o in outputs] == [False, True, False, True]
    feats, meta = relation_features(AVG, outputs, avg_fuel, recursive=False)
    exact_r(feats, r_eq_floor_ref=1.0,
            r_eq_ref_first_only=0.5,       # [1,2]: ref([1]) = 1.0
            r_eq_first_check_only=0.5,
            r_eq_ref_drop_last=0.5,        # [1,2]: ref([1]) again
            r_eq_ref_n_minus_1=0.5,
            r_eq_ref_last_only=0.0)
    assert meta["values"]["r_eq_floor_ref"] == {"returned": 1.0, "expected": 1.5, "test": 0, "count": 2, "failing": 2}


# ---------------------------------------------------------------- searching

SEARCH = {"display_test": 0, "tests": [
    {"args": [[4, 8, 6], 3, 6], "expect": {"returned": 2}},
    {"args": [[4, 8, 6], 3, 4], "expect": {"returned": 0}},
    {"args": [[4, 8, 6, 1], 4, 9], "expect": {"returned": -1}},
    {"args": [[5, 7], 2, 7], "expect": {"returned": 1}},
]}


def linear_search(args):
    a, n, x = args
    return next((i for i in range(n) if a[i] == x), -1)


def test_search_gives_up_after_the_first_element():
    """`else return -1;`: the answer is decided by a[0] alone."""
    outputs = outputs_for(SEARCH, returned=[-1, 0, -1, -1])
    assert [o["passed"] for o in outputs] == [False, True, True, False]
    feats, _ = relation_features(SEARCH, outputs, linear_search, recursive=False)
    exact_r(feats, r_eq_ref_first_only=1.0, r_eq_first_check_only=1.0,
            r_eq_ref_drop_last=1.0,        # 6 and 7 are the last elements, so dropping them also gives -1
            r_eq_ref_n_minus_1=1.0)


def test_search_flag_overwritten_keeps_only_the_last_check():
    """`else idx = -1;` on every miss: only a match in the last cell survives, reported at its real index."""
    outputs = outputs_for(SEARCH, returned=[2, -1, -1, 1])
    assert [o["passed"] for o in outputs] == [True, False, True, True]
    feats, _ = relation_features(SEARCH, outputs, linear_search, recursive=False)
    # ref([6], 1, 4) = -1: the last cell alone decides
    assert feats["r_eq_last_check_only"] == 1.0 and feats["r_eq_ref_last_only"] == 1.0
    assert feats["r_eq_ref_drop_first"] == 1.0 and feats["r_eq_first_check_only"] == 0


def test_last_check_index_is_shifted_back():
    """A learner that only ever reports the last cell: index n-1 when it matches."""
    problem = {"display_test": 0, "tests": [{"args": [[6, 4, 6], 3, 6], "expect": {"returned": 0}}]}
    outputs = outputs_for(problem, returned=[2])
    feats, _ = relation_features(problem, outputs, linear_search, recursive=False)
    assert feats["r_eq_last_check_only"] == 1.0        # ref([6], 1, 6) = 0, shifted by 2
    assert feats["r_eq_ref_last_only"] == 0            # the unshifted value does not match


# ---------------------------------------------------------------- recursion

def test_only_the_top_frame_contributes():
    """`factorial(n - 1); return n;`: 3, 5 and 10 come back as themselves; 0 and 1 return 1 and pass."""
    outputs = outputs_for(Q17, returned=[3, 1, 1, 5, 10])
    assert [o["passed"] for o in outputs] == [False, True, True, False, False]
    feats, meta = relation_features(Q17, outputs, lambda args: math.factorial(args[0]), recursive=True)
    exact_r(feats, r_eq_top_frame_only=1.0)
    assert meta["values"]["r_eq_top_frame_only"] == {"returned": 3, "expected": 6, "test": 0, "count": 3, "failing": 3}


def test_top_frame_for_digits_and_arrays():
    digits = {"display_test": 0, "tests": [{"args": [123], "expect": {"returned": 6}},
                                           {"args": [7], "expect": {"returned": 7}},
                                           {"args": [40], "expect": {"returned": 4}}]}
    outputs = outputs_for(digits, returned=[3, 7, 0])       # n % 10 only
    feats, _ = relation_features(digits, outputs, lambda args: sum(map(int, str(args[0]))), recursive=True)
    assert feats["r_eq_top_frame_only"] == 1.0 and feats["r_eq_zero"] == 0.5
    arr = {"display_test": 0, "tests": [{"args": [[2, 4, 6], 3], "expect": {"returned": 12}},
                                        {"args": [[1, 9], 2], "expect": {"returned": 10}}]}
    outputs = outputs_for(arr, returned=[6, 9])             # a[n - 1] only
    assert relation_features(arr, outputs, total_energy, recursive=True)[0]["r_eq_top_frame_only"] == 1.0


# ---------------------------------------------------------------- arrays changed in place

Q06 = dict(B.Q06)       # tests: [3,1,2], [2,1], [1,2,3], [4,3,2,1]


def bubble_sort(args):
    return {"array0": sorted(args[0][:args[1]])}


def one_pass(args):
    a, n = list(args[0]), args[1]
    for j in range(n - 1):
        if a[j] > a[j + 1]:
            a[j], a[j + 1] = a[j + 1], a[j]
    return a


bubble_sort.one_pass = one_pass


def test_single_pass_equals_the_reference_helper():
    """One bubble pass: [3,1,2] -> [1,2,3] (passes), [4,3,2,1] -> [3,2,1,4] (fails)."""
    outputs = outputs_for(Q06, arrays=[[1, 2, 3], [1, 2], [1, 2, 3], [3, 2, 1, 4]])
    assert [o["passed"] for o in outputs] == [True, True, True, False]
    feats, _ = relation_features(Q06, outputs, bubble_sort, recursive=False)
    exact_r(feats, r_eq_one_pass=1.0)
    wrong = outputs_for(Q06, arrays=[[1, 2, 3], [1, 2], [1, 2, 3], [4, 3, 2, 1]])
    exact_r(relation_features(Q06, wrong, bubble_sort, recursive=False)[0],
            r_eq_reversed_twice=1.0)        # array left as it was


def test_one_pass_without_a_helper():
    outputs = outputs_for(Q06, arrays=[[1, 2, 3], [1, 2], [1, 2, 3], [3, 2, 1, 4]])
    plain = lambda args: bubble_sort(args)                                   # noqa: E731  (no .one_pass)
    assert relation_features(Q06, outputs, plain, recursive=False)[0]["r_eq_one_pass"] == 0
    with_helper_declared = dict(Q06, one_pass="void bubble_sort(int a[], int n) { }")
    assert math.isnan(relation_features(with_helper_declared, outputs, plain, recursive=False)[0]["r_eq_one_pass"])


REVERSE = {"display_test": 0, "tests": [{"args": [[1, 2, 3], 3], "expect": {"array0": [3, 2, 1]}},
                                         {"args": [[5, 5], 2], "expect": {"array0": [5, 5]}},
                                         {"args": [[4, 7, 8, 9], 4], "expect": {"array0": [9, 8, 7, 4]}}]}


def test_reversed_twice_leaves_the_array_unchanged():
    outputs = outputs_for(REVERSE, arrays=[[1, 2, 3], [5, 5], [4, 7, 8, 9]])
    assert [o["passed"] for o in outputs] == [False, True, False]
    feats, _ = relation_features(REVERSE, outputs, lambda args: {"array0": args[0][::-1]}, recursive=False)
    exact_r(feats, r_eq_reversed_twice=1.0)


ROTATE = {"display_test": 0, "tests": [{"args": [[1, 2, 3], 3], "expect": {"array0": [2, 3, 1]}},
                                        {"args": [[4, 7, 8, 9], 4], "expect": {"array0": [7, 8, 9, 4]}}]}


def test_shift_without_wrap_loses_the_first_value():
    rotate = lambda args: {"array0": args[0][1:] + args[0][:1]}             # noqa: E731
    kept_last = outputs_for(ROTATE, arrays=[[2, 3, 3], [7, 8, 9, 9]])       # nothing written to a[n-1]
    feats, meta = relation_features(ROTATE, kept_last, rotate, recursive=False)
    exact_r(feats, r_eq_shift_without_wrap=1.0)
    assert meta["values"]["r_eq_shift_without_wrap"]["lost"] == 1
    overwritten = outputs_for(ROTATE, arrays=[[2, 3, 2], [7, 8, 9, 7]])     # a[n-1] = a[0] after the shift
    exact_r(relation_features(ROTATE, overwritten, rotate, recursive=False)[0], r_eq_shift_without_wrap=1.0)


# ---------------------------------------------------------------- building run_reference from a backend

def test_make_run_reference_wraps_a_run_tests_backend():
    calls = []

    def fake_run_tests(problem, code):
        calls.append((code, problem["tests"][0]["args"]))
        (test,) = problem["tests"]
        assert set(test["expect"]) == {"returned"} and problem["display_test"] == 0
        return {"status": "ok", "backend": "gcc", "tests": {"passed": 0, "total": 1, "results": [
            {"args": test["args"], "expected": test["expect"], "got": {"returned": total_energy(test["args"])},
             "pass": False}]}}

    run_reference = make_run_reference(P03, fake_run_tests)
    assert run_reference([[2, 4], 2]) == {"returned": 6}
    assert run_reference([[2, 4], 2]) == {"returned": 6} and len(calls) == 1       # cached
    assert calls[0][0] == P03["correct_variants"][0]
    assert not hasattr(run_reference, "one_pass")
    outputs = outputs_for(P03, returned=[6, 5, 4, 9, 1])
    assert relation_features(P03, outputs, run_reference, recursive=False)[0]["r_eq_ref_last_only"] == 1.0

    with_helper = make_run_reference(dict(P03, one_pass="/* helper */"), fake_run_tests)
    assert with_helper.one_pass([[1], 1]) == {"returned": 1} and calls[-1][0] == "/* helper */"

    def failing_backend(problem, code):
        raise RuntimeError("backend unavailable")

    assert make_run_reference(P03, failing_backend)([[1], 1]) is None
