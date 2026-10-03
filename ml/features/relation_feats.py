"""Group R: output-relation features (plans/03 §4.3), package F2.

    feats, meta = relation_features(problem, outputs, run_reference, recursive=...)

For every failing test the *reference* solution is run on a modified input and the learner's
output is compared with what the reference gives there. Each feature is the fraction of
failing tests on which its relation holds (0 when no test fails).

`outputs` is `trace_feats.collect_outputs(problem, trace, run_result)`: one dict per test with
the learner's `returned`, `printed`, `array0`, `effects_count` and `passed`. None → every
feature is NaN (no trace).

`run_reference(args) -> output` is supplied by the caller, so no backend is touched here.
It runs `correct_variants[0]` on `args` and returns either a dict with the keys a test's
`expect` uses (`returned`, `array0`, `printed`, effect names), or a bare value (a list is
taken as `array0`, anything else as `returned`). It may return None or raise for an input
the reference cannot handle; the relation is then false. If the callable has an attribute
`one_pass` (same signature), that is the reference's single-pass helper for `r_eq_one_pass`.
`make_run_reference(problem, run_tests)` builds both from a `run_tests(problem, code)` backend.

When `run_reference` is None the features that need it are NaN and the rest are computed.

Input changes (problem-agnostic: they only look at the shapes of the arguments):
* the sequence is the first list or string argument; its size parameter is an int argument
  after it whose value equals the sequence length;
* drop_first / drop_last / first_only / last_only replace the sequence (and the size) and
  need at least two elements;
* n_minus_1: with a sequence, the size parameter − 1 on the same sequence; without one, any
  single int argument − 1. n_plus_1: only without a sequence (size + 1 would read past the end);
* first_check_only: the reference on the first element, the first two, or the two ends;
  last_check_only: on the last element or the last two (an index result is shifted back);
* top_frame_only (only when the reference is recursive): the result equals an int argument,
  the last element of the sequence, or the last decimal digit of a single int argument.
"""
from __future__ import annotations

import json
import math

from ml.contracts.feature_names import GROUP_R
from ml.contracts.subset import FLOAT_TOLERANCE
from ml.features.trace_feats import is_garbage, is_number, output_value, values_equal

NAN = math.nan
NEEDS_REFERENCE = ["r_eq_ref_drop_first", "r_eq_ref_drop_last", "r_eq_ref_last_only", "r_eq_ref_first_only",
                   "r_eq_ref_n_minus_1", "r_eq_ref_n_plus_1", "r_eq_one_pass",
                   "r_eq_first_check_only", "r_eq_last_check_only"]
_SKIP_KEYS = {"max_depth_le"}


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _shape(args):
    """(index of the sequence argument or None, index of its size argument or None)."""
    seq = next((i for i, a in enumerate(args) if isinstance(a, (list, str))), None)
    size = None
    if seq is not None:
        size = next((i for i, a in enumerate(args) if i > seq and _is_int(a) and a == len(args[seq])), None)
    return seq, size


def _with_seq(args, seq, size, new):
    out = list(args)
    out[seq] = new
    if size is not None:
        out[size] = len(new)
    return out


def _normalise(out):
    if out is None or isinstance(out, dict):
        return out
    if isinstance(out, list):
        return {"array0": out}
    return {"returned": out}


def _same(expect, learner, other):
    """Learner output equals `other` on every output key the test checks."""
    if learner is None or other is None:
        return False
    keys = [k for k in expect if k not in _SKIP_KEYS]
    if not keys:
        return False
    return all(values_equal(output_value(learner, k), output_value(other, k)) for k in keys)


def relation_features(problem, outputs, run_reference=None, *, recursive=None):
    """Group-R features. `recursive`: True/False when known whether the reference recurses
    (False switches `r_eq_top_frame_only` off); None = not known, the relation is still tested."""
    feats = {name: NAN for name in GROUP_R}
    meta = {"lines": {}, "values": {}}
    if outputs is None:
        return feats, meta
    tests = (problem or {}).get("tests") or []
    failing = [i for i, o in enumerate(outputs) if i < len(tests) and o.get("passed") is False]
    counts = {name: 0 for name in GROUP_R}
    examples = {}
    cache = {}
    one_pass = getattr(run_reference, "one_pass", None)

    def call(fn, args):
        key = (id(fn), json.dumps(args, sort_keys=True))
        if key not in cache:
            try:
                cache[key] = _normalise(fn(json.loads(json.dumps(args))))     # a fresh copy: arrays are mutable
            except Exception:
                cache[key] = None
        return cache[key]

    def hold(name, i, **vals):
        counts[name] += 1
        examples.setdefault(name, dict(vals, test=i))

    for i in failing:
        args, expect, out = tests[i]["args"], tests[i].get("expect") or {}, outputs[i]
        seq, size = _shape(args)
        items = args[seq] if seq is not None else None
        returned, wanted = out.get("returned"), expect.get("returned")
        array, wanted_array = out.get("array0"), expect.get("array0")

        # ---- counterfactual reference outputs
        if run_reference is not None:
            def ref_equals(new_args):
                return _same(expect, out, call(run_reference, new_args))

            if items is not None and len(items) >= 2:
                if ref_equals(_with_seq(args, seq, size, items[1:])):
                    hold("r_eq_ref_drop_first", i)
                if ref_equals(_with_seq(args, seq, size, items[:-1])):
                    hold("r_eq_ref_drop_last", i)
                if ref_equals(_with_seq(args, seq, size, items[-1:])):
                    hold("r_eq_ref_last_only", i, value=items[-1])
                if ref_equals(_with_seq(args, seq, size, items[:1])):
                    hold("r_eq_ref_first_only", i, value=items[0])

                firsts = [items[:1]] + ([items[:2]] if len(items) >= 3 else [])
                if len(items) >= 3:
                    ends = items[:1] + items[-1:]
                    if ends not in firsts:
                        firsts.append(ends)
                if any(ref_equals(_with_seq(args, seq, size, part)) for part in firsts):
                    hold("r_eq_first_check_only", i)

                lasts = [items[-1:]] + ([items[-2:]] if len(items) >= 3 else [])
                for part in lasts:
                    ref_out = call(run_reference, _with_seq(args, seq, size, part))
                    shift = len(items) - len(part)
                    ref_ret = output_value(ref_out, "returned") if ref_out else None
                    shifted = (_is_int(ref_ret) and 0 <= ref_ret < len(part) and is_number(returned)
                               and returned == ref_ret + shift and list(expect) == ["returned"])
                    if _same(expect, out, ref_out) or shifted:
                        hold("r_eq_last_check_only", i)
                        break

            if seq is not None:
                if size is not None and args[size] >= 1:
                    smaller = list(args)
                    smaller[size] -= 1
                    if ref_equals(smaller):
                        hold("r_eq_ref_n_minus_1", i)
            else:
                ints = [k for k, a in enumerate(args) if _is_int(a)]
                for delta, name in ((-1, "r_eq_ref_n_minus_1"), (1, "r_eq_ref_n_plus_1")):
                    for k in ints:
                        changed = list(args)
                        changed[k] += delta
                        if ref_equals(changed):
                            hold(name, i)
                            break

            if one_pass is not None and array is not None:
                helper_out = call(one_pass, args)
                if helper_out and values_equal(array, output_value(helper_out, "array0")):
                    hold("r_eq_one_pass", i)

        # ---- relations that need only the expected output
        if is_number(returned) and is_number(wanted):
            if isinstance(wanted, float) and abs(wanted - math.trunc(wanted)) > FLOAT_TOLERANCE \
                    and abs(returned - math.trunc(wanted)) <= FLOAT_TOLERANCE:
                hold("r_eq_floor_ref", i, returned=returned, expected=wanted)
            if returned == 0 and wanted != 0:
                hold("r_eq_zero", i, expected=wanted)
            if abs(abs(returned - wanted) - 1) <= FLOAT_TOLERANCE:
                hold("r_off_by_value_1", i, returned=returned, expected=wanted)
        if is_garbage(returned) or (isinstance(array, list) and any(is_garbage(v) for v in array)):
            hold("r_is_garbage", i, returned=returned if is_garbage(returned) else next(v for v in array if is_garbage(v)))

        if recursive is not False and is_number(returned):
            own = []
            if seq is None:
                ints = [a for a in args if _is_int(a)]
                own += ints
                if len(ints) == 1 and ints[0] >= 10:
                    own.append(ints[0] % 10)
            elif isinstance(items, list) and items:
                last = args[size] - 1 if size is not None else len(items) - 1
                if 0 <= last < len(items) and is_number(items[last]):
                    own.append(items[last])
            if any(values_equal(returned, v) for v in own):
                hold("r_eq_top_frame_only", i, returned=returned, expected=wanted)

        if isinstance(array, list) and isinstance(items, list) and wanted_array is not None:
            if values_equal(array, items) and not values_equal(wanted_array, items):
                hold("r_eq_reversed_twice", i)
            if len(items) >= 2 and (values_equal(array, items[1:] + items[-1:])
                                    or values_equal(array, items[1:] + items[1:2])):
                hold("r_eq_shift_without_wrap", i, lost=items[0])

    for name in GROUP_R:
        if run_reference is None and name in NEEDS_REFERENCE:
            continue                            # stays NaN
        if name == "r_eq_one_pass" and one_pass is None and (problem or {}).get("one_pass"):
            continue                            # the problem has a helper but nobody can run it
        feats[name] = counts[name] / len(failing) if failing else 0.0
        if counts[name]:
            meta["values"][name] = dict(examples[name], count=counts[name], failing=len(failing))
    return feats, meta


def make_run_reference(problem, run_tests, code=None):
    """Build `run_reference` from a backend.

    run_tests(problem, code) -> schemas.RunResult, e.g. `ml.runner.run_tests`. The reference
    (`correct_variants[0]`, or `code`) is run on one-test copies of the problem; the copy keeps
    the first test's `expect` keys so the backend reports the same output keys in `got`.
    If the problem has a `one_pass` helper it is attached as `.one_pass`; the helper must
    define the problem's entry function (same signature), since that is what a backend calls.
    """
    tests = problem.get("tests") or []
    expect = tests[0].get("expect", {}) if tests else {}

    def runner_for(source):
        memo = {}

        def run(args):
            key = json.dumps(args, sort_keys=True)
            if key not in memo:
                probe = dict(problem, tests=[{"args": args, "expect": expect}], display_test=0)
                try:
                    memo[key] = run_tests(probe, source)["tests"]["results"][0]["got"]
                except Exception:
                    memo[key] = None
            return memo[key]

        return run

    run_reference = runner_for(code or problem["correct_variants"][0])
    if problem.get("one_pass"):
        run_reference.one_pass = runner_for(problem["one_pass"])
    return run_reference
