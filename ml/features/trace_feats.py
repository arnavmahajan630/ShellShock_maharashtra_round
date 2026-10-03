"""Group B: execution / trace features (plans/03 §4.2), package F2.

    feats, meta = trace_features(trace, reference_trace, problem,
                                 loops=..., ref_loops=..., run_result=...)

`trace` and `reference_trace` are trace dicts (schemas.Trace); the reference is the trace of
`correct_variants[0]` on the same tests. `problem` is the problem dict (tests, display_test).
`loops` / `ref_loops` are `meta["loops"]` from `ast_features` for the learner's and the
reference's code: they say which `L<line>` counter is the main loop and which is the outer
loop. `run_result` is the learner's `run_tests` result (schemas.RunResult), used for pass /
fail and for the final array of in-place problems.

Nothing here calls a backend. Every feature is NaN when `trace` is None (03 §4.6).

Value conventions (the contract leaves them open):
* event features are flags: 1 when the event happened on any test;
* `b_iter_delta_mean` is the plain mean of (learner − reference) main-loop iterations, not clipped;
* `b_iter_delta_const_pm1` needs the same ±1 on every test, read from `per_test`;
* `b_return_first_iter` and `b_printed_eq_ref_return` are fractions of the tests they apply to;
* `b_body_once_vs_many`, `b_window_frozen`, `b_rec_arg_*`, `b_base_return_executed` read the
  recorded steps, so they describe the display test only;
* `b_sorted_frac` and `b_n_outer_passes_ratio` are NaN when they do not apply (the problem
  does not change its array / the reference has no nested loop);
* a feature that needs the reference trace is NaN when `reference_trace` is None.
"""
from __future__ import annotations

import math
from collections import Counter

from ml.contracts.feature_names import GROUP_B
from ml.contracts.subset import FLOAT_TOLERANCE, GARBAGE

NAN = math.nan
GARBAGE_BAND = 1_000_000        # |value - GARBAGE| within this is "garbage plus a little arithmetic"
_NOT_OUTPUT_KEYS = {"max_depth_le"}


# ---------------------------------------------------------------- outputs and pass / fail

def is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def is_garbage(v):
    """True for the interpreter's GARBAGE value or something computed from it by small steps."""
    return is_number(v) and abs(v - GARBAGE) <= GARBAGE_BAND


def values_equal(a, b):
    if a is None or b is None:
        return False
    if is_number(a) and is_number(b):
        return abs(a - b) <= FLOAT_TOLERANCE
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(values_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, str) and isinstance(b, str):
        return a.strip() == b.strip()
    return a == b


def output_value(out, key):
    """The learner's (or a reference run's) value for one `expect` key."""
    if out is None:
        return None
    if key in ("returned", "printed", "array0"):
        return out.get(key)
    if key in out:
        return out[key]
    return (out.get("effects_count") or {}).get(key, 0 if "effects_count" in out else None)


def check_passed(expect, out):
    """True / False, or None when a needed value (the final array) is not known."""
    if out is None:
        return None
    if out.get("status", "ok") != "ok":
        return False
    for key, wanted in expect.items():
        if key == "max_depth_le":
            if out.get("max_depth") is not None and out["max_depth"] > wanted:
                return False
            continue
        got = output_value(out, key)
        if got is None and key == "array0":
            return None
        if not values_equal(got, wanted):
            return False
    return True




def _replay_array(args, steps):
    """Final state of argument 0 after the `write_cell:i:v` effects of the recorded steps."""
    if not args or not isinstance(args[0], list):
        return None
    cells = list(args[0])
    for step in steps:
        for effect in step.get("effects", []):
            name, *fields = effect.split(":")
            if name == "write_cell" and len(fields) == 2:
                try:
                    i = int(fields[0])
                    v = float(fields[1]) if any(c in fields[1] for c in ".eE") else int(fields[1])
                except ValueError:
                    return None
                if 0 <= i < len(cells):
                    cells[i] = v
    return cells


def collect_outputs(problem, trace, run_result=None):
    """One dict per test: status, returned, printed, array0, effects_count, max_depth, passed.

    `array0` (final state of argument 0) is not part of `per_test` in the trace contract, so it
    is taken from `run_result["tests"]["results"][i]["got"]` when given, from a `per_test[i]
    ["array0"]` field if the interpreter adds one, and otherwise rebuilt from the `write_cell`
    effects of the recorded steps, which only covers the display test.
    """
    if trace is None:
        return None
    tests = (problem or {}).get("tests") or []
    display = (problem or {}).get("display_test", 0)
    results = ((run_result or {}).get("tests") or {}).get("results") or []
    outputs = []
    for i, pt in enumerate(trace.get("per_test") or []):
        out = {"status": pt.get("status", "ok"), "returned": pt.get("returned"), "printed": pt.get("printed", ""),
               "effects_count": pt.get("effects_count") or {}, "max_depth": pt.get("max_depth"),
               "array0": pt.get("array0")}
        got = results[i].get("got", {}) if i < len(results) else {}
        if out["array0"] is None and "array0" in got:
            out["array0"] = got["array0"]
        if out["array0"] is None and i == display and i < len(tests) and not trace.get("truncated") \
                and out["status"] == "ok":
            out["array0"] = _replay_array(tests[i].get("args"), trace.get("steps") or [])
        if i < len(results) and "pass" in results[i]:
            out["passed"] = bool(results[i]["pass"])
        elif i < len(tests):
            out["passed"] = check_passed(tests[i].get("expect") or {}, out)
        else:
            out["passed"] = None
        outputs.append(out)
    return outputs


# ---------------------------------------------------------------- loops

def loops_from_trace(trace, display_test=0):
    """Fallback when no AST information is given: main = most iterations on the display test
    (ties to the smallest line), outer = the smallest loop line when there are two or more."""
    per_test = trace.get("per_test") or []
    counts = per_test[display_test].get("loop_iters", {}) if display_test < len(per_test) else {}
    keys = sorted({k for pt in per_test for k in pt.get("loop_iters", {})} | set(trace.get("loop_iters") or {}),
                  key=lambda k: int(k[1:]))
    lines = [int(k[1:]) for k in keys]
    main = min(lines, key=lambda ln: (-counts.get(f"L{ln}", 0), ln)) if lines else None
    return {"main": main, "outer": lines[0] if len(lines) >= 2 else None, "inner": None, "all": lines,
            "main_var": None, "main_cond_vars": [], "inline": [], "selected_by": "trace_only"}


def _iters(pt, line):
    return (pt.get("loop_iters") or {}).get(f"L{line}", 0) if line is not None else 0


# ---------------------------------------------------------------- call frames in the recorded steps

def _calls(steps):
    """(consecutive-frame argument pairs, leaf return seen, any ret seen, entry function name)."""
    args_at, called_deeper, fn_at = {}, {}, {}
    pairs, leaf_return, any_ret, entry = [], False, False, None
    for step in steps:
        for effect in step.get("effects", []):
            name, *fields = effect.split(":")
            if name == "call" and len(fields) >= 2:
                fn, depth, args = fields[0], _to_int(fields[1]), ":".join(fields[2:])
                if depth is None:
                    continue
                entry = entry or fn
                if depth - 1 in args_at and fn_at.get(depth - 1) == fn:
                    pairs.append((args_at[depth - 1], args))
                    called_deeper[depth - 1] = True
                args_at[depth], fn_at[depth], called_deeper[depth] = args, fn, False
            elif name == "ret" and len(fields) >= 2:
                depth = _to_int(fields[1])
                any_ret = True
                if depth is not None and not called_deeper.get(depth, False):
                    leaf_return = True
    return pairs, leaf_return, any_ret, entry


def _to_int(text):
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _numbers(args):
    out = []
    for part in args.split(","):
        try:
            out.append(float(part))
        except ValueError:
            out.append(None)
    return out


# ---------------------------------------------------------------- the feature function

def trace_features(trace, reference_trace=None, problem=None, *, loops=None, ref_loops=None, run_result=None):
    feats = {name: NAN for name in GROUP_B}
    meta = {"lines": {}, "values": {}}
    if trace is None:
        return feats, meta

    def put(name, x, line=None, **vals):
        feats[name] = float(x)
        if line is not None:
            meta["lines"][name] = line
            vals["line"] = line
        if vals:
            meta["values"][name] = vals

    problem = problem or {}
    display = problem.get("display_test", 0)
    tests = problem.get("tests") or []
    per_test = trace.get("per_test") or []
    ref_pt = (reference_trace or {}).get("per_test") or []
    events = trace.get("events") or []
    steps = trace.get("steps") or []
    loops = loops or loops_from_trace(trace, display)
    if reference_trace is not None:
        ref_loops = ref_loops or loops_from_trace(reference_trace, display)
    outputs = collect_outputs(problem, trace, run_result)

    # ---- pass fraction and status
    passed = (run_result or {}).get("tests") or {}
    if "passed" in passed and passed.get("total"):
        put("b_pass_frac", passed["passed"] / passed["total"], passed=passed["passed"], total=passed["total"])
    else:
        known = [o["passed"] for o in outputs if o["passed"] is not None]
        if known:
            put("b_pass_frac", sum(known) / len(known), passed=sum(known), total=len(known))
    statuses = [pt.get("status") for pt in per_test] + [trace.get("status")]
    put("b_status_timeout", "timeout" in statuses)
    put("b_status_runtime_error", "runtime_error" in statuses)

    # ---- events
    def first(kind, test=lambda e: True):
        return next((e for e in events if e.get("type") == kind and test(e)), None)

    def flag(name, event, **vals):
        if event is None:
            put(name, 0)
        else:
            put(name, 1, event.get("line"), **vals)

    e = first("step_cap_hit")
    flag("b_step_cap", e)
    e = first("uninit_read")
    flag("b_uninit_read", e, var=(e or {}).get("var", ""))
    e = first("oob_read")
    flag("b_oob_read", e, arr=(e or {}).get("arr", ""), idx=(e or {}).get("idx"), size=(e or {}).get("size"))
    e = first("oob_read", lambda ev: ev.get("idx") == ev.get("size"))
    flag("b_oob_read_idx_eq_n", e, arr=(e or {}).get("arr", ""), n=(e or {}).get("idx"))
    e = first("oob_read", lambda ev: is_number(ev.get("idx")) and ev["idx"] < 0)
    flag("b_oob_read_idx_neg", e, arr=(e or {}).get("arr", ""), idx=(e or {}).get("idx"))
    e = first("oob_write")
    flag("b_oob_write", e, arr=(e or {}).get("arr", ""), idx=(e or {}).get("idx"), size=(e or {}).get("size"))
    flag("b_intdiv_trunc_nonzero", first("intdiv", lambda ev: ev.get("remainder_nonzero")))
    flag("b_intdiv_into_float", first("intdiv", lambda ev: ev.get("into_float")))
    e = first("empty_body")
    flag("b_empty_body_exec", e, kind=(e or {}).get("kind", ""))
    e = first("assign_in_cond")
    flag("b_assign_in_cond_rt", e, var=(e or {}).get("var", ""), value=(e or {}).get("value"))
    flag("b_missing_return", first("missing_return"))
    e = first("depth_cap_hit")
    flag("b_depth_cap", e, fn=(e or {}).get("fn", ""), arg=_show_args((e or {}).get("last_args")),
         depth=trace.get("max_depth", 0))
    e = first("discarded_call_value")
    flag("b_discarded_call_value", e, fn=(e or {}).get("fn", ""))
    flag("b_str_literal_compare", first("str_literal_compare"))
    flag("b_array_compare", first("array_compare"))

    # ---- iteration counts against the reference
    main, ref_main = loops.get("main"), (ref_loops or {}).get("main")
    if reference_trace is not None and per_test and ref_pt:
        n = min(len(per_test), len(ref_pt))
        deltas = [_iters(per_test[i], main) - _iters(ref_pt[i], ref_main) for i in range(n)]
        shown = min(display, n - 1)
        counts = {"actual": _iters(per_test[shown], main), "expected": _iters(ref_pt[shown], ref_main)}
        mean = sum(deltas) / n
        put("b_iter_delta_mean", mean, main, delta=round(mean, 2), fewer=round(abs(mean), 2), **counts)
        const = all(d == 1 for d in deltas) or all(d == -1 for d in deltas)
        put("b_iter_delta_const_pm1", const, main, **counts)
        eligible = [i for i in range(n) if _iters(ref_pt[i], ref_main) >= 2]
        early = [i for i in eligible if per_test[i].get("status") == "ok" and _iters(per_test[i], main) == 1]
        put("b_return_first_iter", len(early) / len(eligible) if eligible else 0, main)

    # ---- body ran once where the reference looped (display test, recorded steps)
    if reference_trace is not None and display < len(ref_pt):
        headers = set(loops.get("all") or []) - set(loops.get("inline") or [])
        per_line = Counter(s["line"] for s in steps if s["line"] not in headers)
        learner_once = bool(per_line) and max(per_line.values()) <= 1 and not _inline_body(steps, per_test, display, loops)
        ref_many = _iters(ref_pt[display], ref_main) >= 2
        put("b_body_once_vs_many", ref_many and learner_once, expected=_iters(ref_pt[display], ref_main))

    # ---- branches: constant for the learner while some reference branch varies
    if reference_trace is not None:
        def totals(pts):
            total = {}
            for pt in pts:
                for key, counts_ in (pt.get("branch") or {}).items():
                    slot = total.setdefault(key, {"true": 0, "false": 0})
                    slot["true"] += counts_.get("true", 0)
                    slot["false"] += counts_.get("false", 0)
            return total

        ref_varies = any(c["true"] > 0 and c["false"] > 0 for c in totals(ref_pt).values())
        mine = totals(per_test)
        always = [k for k, c in mine.items() if c["true"] >= 2 and c["false"] == 0]
        never = [k for k, c in mine.items() if c["false"] >= 2 and c["true"] == 0]
        put("b_branch_always", ref_varies and bool(always), int(always[0][1:]) if ref_varies and always else None)
        put("b_branch_never", ref_varies and bool(never), int(never[0][1:]) if ref_varies and never else None)

    # ---- returned / printed
    garbage = next((o["returned"] for o in outputs if is_garbage(o["returned"])), None)
    put("b_returned_garbage", garbage is not None, returned=garbage)
    printed = next((o["printed"] for o in outputs if o["printed"]), "")
    put("b_printed_nonempty", bool(printed), printed=printed.strip())
    wanted = []
    for i in range(len(outputs)):
        if i < len(ref_pt) and ref_pt[i].get("returned") is not None:
            wanted.append(ref_pt[i]["returned"])
        elif i < len(tests) and (tests[i].get("expect") or {}).get("returned") is not None:
            wanted.append(tests[i]["expect"]["returned"])
        else:
            wanted.append(None)
    usable = [i for i, w in enumerate(wanted) if w is not None]
    if usable:
        hits = [i for i in usable if _printed_equals(outputs[i]["printed"], wanted[i])]
        vals = {}
        if hits:
            vals = {"printed": outputs[hits[0]]["printed"].strip(), "returned": outputs[hits[0]]["returned"]}
        put("b_printed_eq_ref_return", len(hits) / len(usable), **vals)
    else:                                       # nothing is returned in this problem (void)
        put("b_printed_eq_ref_return", 0)

    # ---- recursion
    if reference_trace is not None:
        ref_depth = max(reference_trace.get("max_depth") or 0, 1)
        put("b_max_depth_ratio", (trace.get("max_depth") or 0) / ref_depth,
            depth=trace.get("max_depth") or 0, expected=ref_depth)
    pairs, leaf_return, any_ret, entry = _calls(steps)
    same = next((a for a, b in pairs if a == b), None)
    put("b_rec_arg_constant", same is not None, fn=entry or "", arg=same if same is not None else "")
    growing = None
    if len(pairs) >= 2:
        rows = [(_numbers(a), _numbers(b)) for a, b in pairs]
        width = min(len(a) for a, _ in rows)
        for pos in range(width):
            if all(a[pos] is not None and len(b) > pos and b[pos] is not None and b[pos] > a[pos] for a, b in rows):
                growing = (pairs[0][0], pairs[-1][1])
                break
    put("b_rec_arg_growing", growing is not None, fn=entry or "",
        first=growing[0] if growing else "", last=growing[1] if growing else "")
    display_ok = display < len(per_test) and per_test[display].get("status") == "ok"
    put("b_base_return_executed", leaf_return or (display_ok and not any_ret), fn=entry or "")

    # ---- window frozen: the main loop's condition variables repeat on two consecutive passes
    frozen = None
    if main is not None:
        names = loops.get("main_cond_vars") or None
        snaps, in_header = [], False
        for step in steps:
            if step["line"] == main:
                snap = {k: v for k, v in step.get("vars", {}).items() if names is None or k in names}
                if in_header:
                    snaps[-1] = snap
                else:
                    snaps.append(snap)
                in_header = True
            else:
                in_header = False
        for a, b in zip(snaps, snaps[1:]):
            if a and a == b:
                frozen = a
                break
    put("b_window_frozen", frozen is not None, main if frozen is not None else None,
        vars=", ".join(f"{k} = {v}" for k, v in (frozen or {}).items()))

    # ---- arrays changed in place
    mutating = any("array0" in (t.get("expect") or {}) for t in tests)
    if not mutating:
        put("b_multiset_changed", 0)
    else:
        pairs_ = [(tests[i]["args"][0], o["array0"]) for i, o in enumerate(outputs)
                  if i < len(tests) and o["array0"] is not None and isinstance(tests[i]["args"][0], list)]
        if pairs_:
            changed = next(((a, b) for a, b in pairs_ if Counter(map(_key, a)) != Counter(map(_key, b))), None)
            vals = {}
            if changed:
                before, after = Counter(map(_key, changed[0])), Counter(map(_key, changed[1]))
                extra, lost = list((after - before).elements()), list((before - after).elements())
                vals = {"v": extra[0] if extra else "", "w": lost[0] if lost else ""}
            put("b_multiset_changed", changed is not None, **vals)
            fracs = []
            for _, after in pairs_:
                if len(after) >= 2:
                    ordered = sum(1 for x, y in zip(after, after[1:]) if is_number(x) and is_number(y) and x <= y)
                    fracs.append(ordered / (len(after) - 1))
            if fracs:
                put("b_sorted_frac", sum(fracs) / len(fracs), pct=round(100 * sum(fracs) / len(fracs)))

    # ---- outer passes against the reference's outer loop
    ref_outer = (ref_loops or {}).get("outer")
    if reference_trace is not None and ref_outer is not None and ref_pt:
        ref_total = sum(_iters(pt, ref_outer) for pt in ref_pt)
        if loops.get("outer") is not None:
            mine_total = sum(_iters(pt, loops["outer"]) for pt in per_test)
        elif main is not None:                  # a single loop is one pass per test in which it ran
            mine_total = sum(1 for pt in per_test if _iters(pt, main) >= 1)
        else:
            mine_total = 0
        if ref_total:
            put("b_n_outer_passes_ratio", mine_total / ref_total, loops.get("outer") or main,
                actual=mine_total, expected=ref_total)
    return feats, meta


def _key(v):
    return round(v, 6) if isinstance(v, float) else v


def _show_args(args):
    if isinstance(args, (list, tuple)):
        return ", ".join(str(a) for a in args)
    return "" if args is None else str(args)


def _printed_equals(printed, wanted):
    text = (printed or "").strip()
    if not text:
        return False
    if is_number(wanted):
        try:
            return abs(float(text) - wanted) <= FLOAT_TOLERANCE
        except ValueError:
            return False
    return text == str(wanted).strip()


def _inline_body(steps, per_test, display, loops):
    """Without AST information: a loop header line with more steps than init + tests + updates
    means the body sits on the header line (`for (...) total += a[i];`)."""
    if "inline" in loops and loops.get("selected_by") != "trace_only":
        return False                            # the AST already said which headers hold a body
    if display >= len(per_test):
        return False
    per_line = Counter(s["line"] for s in steps)
    for line in loops.get("all") or []:
        k = _iters(per_test[display], line)
        if k >= 2 and per_line.get(line, 0) > 2 * k + 2:
            return True
    return False
