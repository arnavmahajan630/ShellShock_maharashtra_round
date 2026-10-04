"""Runs a problem's tests on learner code with the interpreter (ml_plan/03 §2.3, §2.4).

    run_tests(problem, code, sample_only=False) -> schemas.RunResult shape
    trace(problem, code, test_index=None)       -> schemas.Trace shape
    check(code, problem=None)                   -> why a file cannot run (for the gate), or ok

`problem` is a problem dict (schemas.Problem). The entry function is the one named in
`problem["signature"]`. Parsed files are cached by source text.
"""
import re
import threading
from collections import OrderedDict

from ml.contracts.subset import EFFECTS_COUNT_KEYS, FLOAT_TOLERANCE

from . import interp
from .interp import Program
from .preprocess import SourceError

BACKEND = "interp"
_CACHE_SIZE = 256
_cache = OrderedDict()
_cache_lock = threading.Lock()


def compile_source(code, forbid=()):
    """(Program, None) or (None, SourceError). Cached: the same text is parsed once."""
    key = (code, tuple(sorted(forbid or ())))
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit
    try:
        result = (Program(code, forbid), None)
    except SourceError as error:
        result = (None, error)
    except Exception as exc:            # a gap in the interpreter: report it, do not take the caller down
        if interp.STRICT:
            raise
        result = (None, SourceError("unsupported", 1, f"this code could not be prepared ({type(exc).__name__})"))
    with _cache_lock:
        _cache[key] = result
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return result


def entry_name(problem):
    """Name of the function the tests call."""
    m = re.match(r"[^(]*?([A-Za-z_]\w*)\s*\(", problem.get("signature") or "")
    return m.group(1) if m else problem["name"]


def _prepare(problem, code):
    """(Program, entry, None), or (None, entry, SourceError) when the file cannot run."""
    entry = entry_name(problem)
    program, error = compile_source(code, problem.get("forbid") or ())
    tests = problem.get("tests") or []
    if error is None and tests:
        try:
            program.bind(entry, tests[0]["args"])
        except SourceError as bad:
            program, error = None, bad
    return program, entry, error


def check(code, problem=None):
    """{"status": "ok"} or {"status": "parse_error" | "unsupported", "line", "reason", "construct"}.

    `construct` is a key of subset.REJECTED, "strlen" for a forbidden strlen, the entry
    function's name when it is missing or has the wrong parameters, or None.
    """
    if problem is None:
        _, error = compile_source(code)
    else:
        _, _, error = _prepare(problem, code)
    if error is None:
        return {"status": "ok"}
    return {"status": error.status, "line": error.line, "reason": error.reason, "construct": error.construct}


# ---------------------------------------------------------------- judging one test

def _same(expected, actual):
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(_same(e, a) for e, a in zip(expected, actual))
    number = (int, float)
    if isinstance(expected, number) and isinstance(actual, number):
        if isinstance(expected, float) or isinstance(actual, float):
            return abs(expected - actual) <= FLOAT_TOLERANCE
        return int(expected) == int(actual)
    return expected == actual


def _text(cells):
    out = []
    for v in cells:
        if not v:
            break
        out.append(chr(int(v) & 0xFF))
    return "".join(out)


def _judge(test, result):
    """(got, passed): `got` has the same keys as the test's `expect`."""
    got, passed = {}, result["status"] == "ok"
    for key, expected in (test.get("expect") or {}).items():
        if key == "returned":
            actual = result["returned"]
            fits = _same(expected, actual)
        elif key == "printed":
            actual = result["printed"]
            fits = expected == actual
        elif key == "array0":
            cells = result["arrays"][0] if result["arrays"] else None
            if cells is None:
                actual, fits = None, False
            elif isinstance(expected, str):
                actual = _text(cells)
                fits = expected == actual
            else:
                actual = cells
                fits = _same(expected, actual)
        elif key == "max_depth_le":
            actual = result["max_depth"]
            fits = actual <= expected
        else:                                           # an effect name: fire, launch, door_open, ...
            actual = result["effects_count"].get(key, 0)
            fits = _same(expected, actual)
        got[key] = actual
        passed = passed and fits
    if result["status"] != "ok":
        got["status"] = result["status"]
    return got, passed


def _first_bad(statuses):
    return next((s for s in statuses if s != "ok"), "ok")


# ---------------------------------------------------------------- public API

def run_tests(problem, code, sample_only=False):
    """Run the problem's tests (only the two sample tests with sample_only) on `code`."""
    tests = [t for t in problem["tests"] if not sample_only or t.get("sample")]
    program, entry, error = _prepare(problem, code)
    results, statuses = [], []
    for index, test in enumerate(tests):
        expected = test.get("expect") or {}
        if error is not None:
            results.append({"args": test["args"], "expected": expected, "got": {}, "pass": False})
            continue
        result = program.run_test(entry, test["args"], index)
        got, passed = _judge(test, result)
        statuses.append(result["status"])
        results.append({"args": test["args"], "expected": expected, "got": got, "pass": passed})
    return {
        "status": error.status if error is not None else _first_bad(statuses),
        "backend": BACKEND,
        "tests": {"passed": sum(r["pass"] for r in results), "total": len(results), "results": results},
    }


def trace(problem, code, test_index=None):
    """Steps of one test (the problem's display_test, or `test_index`) plus counters and
    events over all tests. Counters on the trace are sums, max_depth the maximum; per_test
    holds each test's own numbers."""
    tests = problem.get("tests") or []
    shown = problem.get("display_test", 0) if test_index is None else test_index
    out = {
        "status": "ok", "returned": None, "printed": "", "steps": [],
        "loop_iters": {}, "branch": {}, "events": [],
        "effects_count": dict.fromkeys(EFFECTS_COUNT_KEYS, 0),
        "max_depth": 0, "truncated": False, "per_test": [],
    }
    program, entry, error = _prepare(problem, code)
    if error is not None:
        out["status"] = error.status
        out["per_test"] = [{"status": error.status, "returned": None, "printed": "", "loop_iters": {},
                            "branch": {}, "effects_count": {}, "max_depth": 0} for _ in tests]
        return out
    if tests:
        shown = min(max(shown, 0), len(tests) - 1)
    loops, branch, effects = out["loop_iters"], out["branch"], out["effects_count"]
    for index, test in enumerate(tests):
        result = program.run_test(entry, test["args"], index, record=index == shown)
        for key, n in result["loop_iters"].items():
            loops[key] = loops.get(key, 0) + n
        for key, counts in result["branch"].items():
            total = branch.setdefault(key, {"true": 0, "false": 0})
            total["true"] += counts["true"]
            total["false"] += counts["false"]
        for key, n in result["effects_count"].items():
            effects[key] = effects.get(key, 0) + n
        out["max_depth"] = max(out["max_depth"], result["max_depth"])
        out["events"] += result["events"]
        if index == shown:
            out["returned"], out["printed"] = result["returned"], result["printed"]
            out["steps"], out["truncated"] = result["steps"], result["truncated"]
        out["per_test"].append({key: result[key] for key in (
            "status", "returned", "printed", "loop_iters", "branch", "effects_count", "max_depth")})
    out["status"] = _first_bad(t["status"] for t in out["per_test"])
    return out
