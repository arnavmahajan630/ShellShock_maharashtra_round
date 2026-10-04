"""Counterexample search (ml_plan/03 §7.2). Package R1.

Enumerate a small pool (n in {1, 2, 3}, the listed arrays, scalars taken from the
problem's tests, and the DSA pools for sorts, searches, strings and recursion).
Return the smallest input on which the learner's output differs from the reference.

Belief answers are recognised only for M04, M10 and D01 (see notes/R1.md). For
every other class this returns the smallest differing input.
"""
from __future__ import annotations

import copy
import re

from ml.runner import run_tests, trace

_ARRAYS = ([3], [3, 5], [2, 4, 6], [1, 2, 3, 4])
_SORTS = ([2, 1], [3, 1, 2], [4, 3, 2, 1])
_STRINGS = ("a", "ab", "aba", "abca")
_NS = (1, 2, 3)
_REC_NS = (0, 1, 2, 3, 10)

# Classes whose belief answer we can recognise on an input. Anything else is
# "unknown": the search keeps the smallest differing input and does not pretend
# to know what the misconception predicted.
_RECOGNISED = {"M04", "M10", "D01"}

_EFFECT_LABEL = {
    "read_void": "void read",
    "read_cell": "cell read",
    "write_cell": "cell write",
    "fire": "fire",
    "compare": "compare",
    "call": "call",
    "ret": "return",
}


def search(problem, code, cls=None):
    """``{input, intended, yours, effect_diff}`` or None if every pool input agrees.

    ``cls`` selects the belief preference when the class is one of M04, M10, D01.
    """
    reference = (problem.get("correct_variants") or [None])[0]
    if not reference:
        return None
    pool = _pool(problem)
    rows = []
    for args in pool:
        intended_got, intended_trace = _once(problem, reference, args)
        yours_got, yours_trace = _once(problem, code, args)
        if _snapshot(intended_got) == _snapshot(yours_got):
            continue
        agrees = _belief_agrees(cls, problem, args, intended_got, yours_got)
        rows.append((agrees, _size(args), args, intended_got, yours_got, intended_trace, yours_trace))
    if not rows:
        return None
    # True (belief matches the intended output) first, then unknown, then a
    # known mismatch. Inside each band, the smallest input.
    rows.sort(key=lambda row: ({True: 0, None: 1, False: 2}[row[0]], row[1], _dump(row[2])))
    _, _, args, intended_got, yours_got, intended_trace, yours_trace = rows[0]
    return {
        "input": _named(problem, args),
        "intended": _shown(intended_got),
        "yours": _shown(yours_got),
        "effect_diff": _effect_diff(intended_trace, yours_trace),
    }


def belief_recognised(cls):
    """True when ``search`` can tell whether the class's belief matches the intended output."""
    return cls in _RECOGNISED


# ---------------------------------------------------------------- pool

def _pool(problem):
    tests = [list(test["args"]) for test in problem.get("tests") or []]
    found = []
    seen = set()

    def add(args):
        key = _dump(args)
        if key in seen:
            return
        seen.add(key)
        found.append(list(args))

    for args in tests:
        add(args)
    if not tests:
        return found

    sample = tests[0]
    array_at = [i for i, value in enumerate(sample) if isinstance(value, list)]
    string_at = [i for i, value in enumerate(sample) if isinstance(value, str)]
    int_at = [i for i, value in enumerate(sample) if _is_int(value)]
    size_at = _size_index(sample, array_at)
    arrays = list(_SORTS) + list(_ARRAYS) if _is_sort(problem) else list(_ARRAYS)
    if _is_recursion(problem) and not array_at and not string_at:
        for n in _REC_NS:
            args = list(sample)
            if int_at:
                args[int_at[0]] = n
            add(args)
    if string_at or _is_string(problem):
        slot = string_at[0] if string_at else None
        if slot is not None:
            for text in _STRINGS:
                args = list(sample)
                args[slot] = text
                add(args)
    if array_at:
        slot = array_at[0]
        extras = [i for i in int_at if i != size_at]
        for array in arrays:
            args = list(sample)
            args[slot] = list(array)
            if size_at is not None:
                args[size_at] = len(array)
            add(args)
            if _is_search(problem) and extras:
                target_at = extras[0]
                choices = [array[0]]
                if len(array) > 1:
                    choices.append(array[1])
                choices.append(array[-1])
                choices.append(max(array) + 1)
                for target in choices:
                    varied = list(args)
                    varied[target_at] = target
                    add(varied)
        for n in _NS:
            if size_at is None:
                break
            args = list(sample)
            # Keep the template array when it is long enough; otherwise a pool array of length n.
            if isinstance(args[slot], list) and len(args[slot]) >= n:
                args[slot] = list(args[slot][:n])
            else:
                args[slot] = list(arrays[min(n, len(arrays)) - 1])[:n] or [3]
            args[size_at] = n
            add(args)
    elif int_at and not _is_recursion(problem):
        for n in _NS:
            args = list(sample)
            args[int_at[0]] = n
            add(args)
    found.sort(key=lambda args: (_size(args), _dump(args)))
    return found[:32]


def _size_index(args, array_at):
    if not array_at:
        return None
    length = len(args[array_at[0]])
    for index, value in enumerate(args):
        if index > array_at[0] and _is_int(value) and value == length:
            return index
    return None


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_sort(problem):
    family = problem.get("family") or ""
    return problem.get("sector") == "sorting" or "sort" in family


def _is_search(problem):
    family = problem.get("family") or ""
    return problem.get("sector") == "searching" or "search" in family


def _is_string(problem):
    family = problem.get("family") or ""
    return problem.get("sector") == "strings" or "string" in family or "char" in (problem.get("signature") or "")


def _is_recursion(problem):
    family = problem.get("family") or ""
    return problem.get("sector") == "recursion" or family.startswith("rec") or problem.get("problem_id") in {"Q16", "Q17", "Q18"}


def _size(args):
    elements = 0
    scalars = 0
    for value in args:
        if isinstance(value, list):
            elements += len(value)
        elif isinstance(value, str):
            elements += len(value)
        elif _is_int(value):
            scalars += abs(value)
        elif isinstance(value, float):
            scalars += abs(int(value))
    return (elements, scalars)


def _dump(args):
    return repr(args)


# ---------------------------------------------------------------- one input

def _once(problem, code, args):
    mini = copy.deepcopy(problem)
    mini["problem_id"] = str(problem.get("problem_id")) + "#cex"
    mini["tests"] = [{"args": list(args), "expect": _expect(problem, args)}]
    mini["display_test"] = 0
    result = run_tests(mini, code)
    row = result["tests"]["results"][0]
    walked = trace(mini, code, 0)
    return row["got"], walked


def _expect(problem, args):
    """Keys the harness fills in. The values are dummies; the caller reads ``got``."""
    keys = set()
    for test in problem.get("tests") or []:
        keys.update((test.get("expect") or {}).keys())
    expect = {}
    if "returned" in keys or "array0" not in keys:
        expect["returned"] = 0
    if "printed" in keys:
        expect["printed"] = ""
    if "array0" in keys or any(isinstance(value, list) for value in args):
        expect["array0"] = []
    for key in keys:
        if key in ("returned", "printed", "array0", "max_depth_le"):
            continue
        expect[key] = 0
    if not expect:
        expect["returned"] = 0
    return expect


def _snapshot(got):
    array = got.get("array0")
    if isinstance(array, list):
        array = tuple(array)
    return (got.get("status"), got.get("returned"), got.get("printed"), array)


def _shown(got):
    """The value a counterexample panel shows: the return, else the array, else the text."""
    if got.get("status") not in (None, "ok") and got.get("returned") is None and not got.get("array0"):
        return got.get("status")
    if "array0" in got and got.get("returned") is None:
        return got.get("array0")
    if got.get("returned") is not None:
        return got.get("returned")
    if got.get("printed"):
        return got.get("printed")
    return got.get("array0")


def _named(problem, args):
    names = _param_names(problem.get("signature") or "")
    if len(names) != len(args):
        return {f"arg{i}": value for i, value in enumerate(args)}
    return dict(zip(names, args))


def _param_names(signature):
    match = re.search(r"\((.*)\)", signature)
    if not match or not match.group(1).strip():
        return []
    names = []
    for part in match.group(1).split(","):
        token = part.strip().replace("[", " ").replace("]", " ").split()
        names.append(token[-1] if token else "arg")
    return names


# ---------------------------------------------------------------- effects and belief

def _effect_diff(reference, learner):
    ref_counts = (reference or {}).get("effects_count") or {}
    got_counts = (learner or {}).get("effects_count") or {}
    keys = list(dict.fromkeys(list(_EFFECT_LABEL) + list(ref_counts) + list(got_counts)))
    parts = []
    for key in keys:
        delta = got_counts.get(key, 0) - ref_counts.get(key, 0)
        if not delta:
            continue
        label = _EFFECT_LABEL.get(key, key.replace("_", " "))
        word = "extra" if delta > 0 else "missing"
        n = abs(delta)
        if n != 1 and not label.endswith("s"):
            label += "s"
        parts.append(f"{n} {word} {label}")
    if not parts:
        ref_events = _event_types(reference)
        got_events = _event_types(learner)
        for kind in ("oob_read", "uninit_read", "oob_write", "step_cap_hit", "depth_cap_hit"):
            delta = got_events.get(kind, 0) - ref_events.get(kind, 0)
            if not delta:
                continue
            label = {
                "oob_read": "void read",
                "uninit_read": "uninitialised read",
                "oob_write": "out-of-range write",
                "step_cap_hit": "step cap",
                "depth_cap_hit": "depth cap",
            }[kind]
            word = "extra" if delta > 0 else "missing"
            n = abs(delta)
            if n != 1 and not label.endswith("s"):
                label += "s"
            parts.append(f"{n} {word} {label}")
    if not parts:
        return "same effects"
    return ", ".join(parts)


def _event_types(walked):
    counts = {}
    for event in (walked or {}).get("events") or []:
        kind = event.get("type")
        if kind:
            counts[kind] = counts.get(kind, 0) + 1
    return counts


def _belief_agrees(cls, problem, args, intended_got, yours_got):
    """True / False when the class's belief answer can be compared with the intended output.

    None means this class's belief is not computed (the caller then keeps the
    smallest differing input).
    """
    if cls not in _RECOGNISED:
        return None
    intended = _shown(intended_got)
    if cls == "M04":
        return _m04_agrees(intended, _shown(yours_got))
    if cls == "M10":
        return _m10_agrees(yours_got, intended)
    if cls == "D01":
        return _d01_agrees(args, intended)
    return None


def _m04_agrees(intended, yours):
    """The belief is the exact quotient. It matches the intended output when that
    output is a non-integral float and the learner did not produce it."""
    if isinstance(intended, float) and not isinstance(intended, bool):
        if isinstance(yours, (int, float)) and abs(intended - float(yours)) > 1e-6:
            return True
        return False
    return None


def _m10_agrees(yours_got, intended):
    """The belief is that the printed text is the value handed back."""
    printed = (yours_got.get("printed") or "").strip()
    if not printed:
        return False
    if printed in ("yes", "no"):
        believed = 1 if printed == "yes" else 0
    else:
        token = printed.split()[-1]
        try:
            believed = float(token) if "." in token else int(token)
        except ValueError:
            return None
    return _close(believed, intended)


def _d01_agrees(args, intended):
    """The belief predicts a hit. Prefer the plan's counterexample: the target sits at index 1."""
    array = next((value for value in args if isinstance(value, list)), None)
    if not array or len(array) < 2:
        return None
    target = None
    for value in args:
        if _is_int(value) and value != len(array):
            target = value
    if target is None:
        return None
    if target not in array:
        return False
    # A present target is what the "keep looking" belief gets right, and index 1
    # is the input the policy asks for. Index 0 hides the early-exit bug.
    return array[1] == target and _hit(intended)


def _hit(intended):
    """A found flag (1) or the index 1. Both are the value 1 on these problems."""
    return intended is True or intended == 1


def _close(left, right):
    if isinstance(right, float) or isinstance(left, float):
        try:
            return abs(float(left) - float(right)) <= 1e-6
        except (TypeError, ValueError):
            return False
    return left == right
