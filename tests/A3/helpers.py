"""Shared helpers for the interpreter spec tests (package A3).

The interpreter (package A1) is reached only through ml.runner:
    runner.run_tests(problem, code, backend="interp")   -> schemas.RunResult
    runner.trace(problem, code, test_index=None)        -> schemas.Trace
"""
import os
import re
import shutil
import textwrap

import pytest

from ml import runner
from ml.contracts import schemas as S
from ml.contracts.subset import EFFECT_FIELDS, EVENT_FIELDS, FLOAT_TOLERANCE

INTERP = runner.available("interp")
GCC = runner.available("gcc") and bool(os.environ.get("RELEARN_GCC") or shutil.which("gcc"))
STRINGS_CUT = os.environ.get("RELEARN_SKIP_STRINGS", "") not in ("", "0")

needs_interp = pytest.mark.skipif(
    not INTERP, reason="interpreter (package A1) is not built: ml.runner.available('interp') is False")
needs_gcc = pytest.mark.skipif(not GCC, reason="gcc backend (package A2) or gcc itself is missing")
skip_if_strings_cut = pytest.mark.skipif(STRINGS_CUT, reason="strings are cut (RELEARN_SKIP_STRINGS is set)")

WORLD_EFFECTS = ["fire", "launch", "door_open", "door_closed", "scan"]

# ---------------------------------------------------------------- which spec row a test covers

SPEC = {
    "2.2/01": "Read of uninitialised local -> GARBAGE, uninit_read {var}",
    "2.2/02": "Array read out of bounds -> GARBAGE, oob_read {arr, idx, size}",
    "2.2/03": "Array write out of bounds -> ignored, oob_write {arr, idx, size}",
    "2.2/04": "int / int -> truncation toward zero, intdiv {remainder_nonzero, into_float}",
    "2.2/05": "Division by zero -> halt, status runtime_error, div_zero",
    "2.2/06": "Condition expression is an assignment -> evaluates normally, assign_in_cond {var, value}",
    "2.2/07": "Body is EmptyStatement -> executes nothing, empty_body {kind}",
    "2.2/08": "Each loop iteration -> loop_iters['L<line>'] += 1",
    "2.2/09": "Each if -> branch['B<line>'] = {true: n, false: m}",
    "2.2/10": "> 5,000 executed statements -> halt, status timeout, step_cap_hit",
    "2.2/11": "Non-void function falls off the end -> GARBAGE, missing_return",
    "2.2/12": "int overflow -> wrap to 32-bit, overflow",
    "2.2/13": "Array param read -> effect read_cell {i} or read_void {i}",
    "2.2/14": "Array param write -> performed, shared with the caller, effect write_cell {i, v}",
    "2.2/15": "Relational compare of two array cells -> effect compare {i, j}",
    "2.2/16": "String literal initialising char s[] -> char codes + 0 terminator",
    "2.2/17": "== / != with a string literal or two array names -> false / true, str_literal_compare / array_compare",
    "2.2/18": "Function call / return -> effects call {fn, depth, args}, ret {fn, depth, value}; max_depth",
    "2.2/19": "Recursion deeper than 100 -> halt, status timeout, depth_cap_hit {fn, last_args}",
    "2.2/20": "Recursive call result unused -> discarded_call_value {fn, line}",
    "2.2/21": "strlen in a problem whose forbid lists it -> not executed",
    "2.5/x1": "v3 extra: string literal init + terminator",
    "2.5/x2": "v3 extra: s[i] == \"a\" -> false + event",
    "2.5/x3": "v3 extra: in-place sort visible to the caller",
    "2.5/x4": "v3 extra: depth_cap_hit on missing base case",
    "2.5/x5": "v3 extra: discarded_call_value",
    "2.5/x6": "v3 extra: strlen forbid",
}
COVERED = {}


def covers(*ids):
    """Mark a test with the 03 spec rows it covers (checked by test_spec_coverage.py)."""
    def mark(fn):
        for spec_id in ids:
            assert spec_id in SPEC, spec_id
            COVERED.setdefault(spec_id, []).append(fn.__name__)
        fn.spec_rows = ids
        return fn
    return mark


# ---------------------------------------------------------------- building programs and problems

def src(text):
    """Dedent a C snippet so that its first line is line 1."""
    return textwrap.dedent(text).strip("\n")


def make_problem(signature, tests, display_test=0, forbid=(), sector=None):
    """A full schemas.Problem dict around one function. tests: [(args, expect), ...]."""
    name = re.match(r"^[\w\s]*?(\w+)\s*\(", signature).group(1)
    problem = {
        "problem_id": "T00", "name": name, "world": "test_bay", "family": "spec_test", "split": "train",
        "difficulty": 1, "prompt": "Spec test.", "signature": signature,
        "starter": signature + " {\n    // TODO\n}", "correct_variants": [],
        "tests": [{"args": args, "expect": expect} for args, expect in tests],
        "display_test": display_test, "predict_item": None, "allowed_ops": [], "markers": [],
        "forbid": list(forbid), "exposure": {},
    }
    problem["sector" if sector else "planet"] = sector or "loops"
    S.Problem.model_validate(problem)
    return problem


# ---------------------------------------------------------------- calling the interpreter

def check_trace_shape(trace):
    """Every trace must fit schemas.Trace, subset.EVENT_FIELDS and subset.EFFECT_FIELDS."""
    S.Trace.model_validate(trace)
    assert [s["i"] for s in trace["steps"]] == list(range(len(trace["steps"])))
    for event in trace["events"]:
        assert event["type"] in EVENT_FIELDS, event
        assert set(EVENT_FIELDS[event["type"]]) <= set(event), event
        assert "test" in event, f"trace.events entries carry the test index (subset.py step rules): {event}"
    for step in trace["steps"]:
        for event in step["events"]:
            assert event["type"] in EVENT_FIELDS and "test" not in event, event
        for effect in step["effects"]:
            name, *fields = effect.split(":")
            assert name in EFFECT_FIELDS and len(fields) == len(EFFECT_FIELDS[name]), effect


def itrace(problem, code, test_index=None):
    trace = runner.trace(problem, code, test_index=test_index)
    check_trace_shape(trace)
    return trace


def irun(problem, code, sample_only=False):
    result = runner.run_tests(problem, code, backend="interp", sample_only=sample_only)
    S.RunResult.model_validate(result)
    assert result["backend"] == "interp"
    return result


def events(trace, kind):
    return [e for e in trace["events"] if e["type"] == kind]


def effects(trace):
    """Effects of the recorded (display) test, in order."""
    return [effect for step in trace["steps"] for effect in step["effects"]]


def passes(result):
    return [r["pass"] for r in result["tests"]["results"]]


def close(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and abs(float(a) - float(b)) <= FLOAT_TOLERANCE
    return a == b
