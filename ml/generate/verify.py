"""Keep / drop rules for one candidate (03 §3.5.4).

A class row must fail at least one test and ``holds(op_id)`` must be true.
OTHER must fail at least one test and no class predicate may hold.
CORRECT must pass every test. A timeout is a failed test, not a crash.
Runs are cached by the reprinted syntax tree so comment and spacing copies
are not executed again.
"""
from __future__ import annotations

from ml.generate.ambiguity import execution_key
from ml.generate.registry import class_op_ids, predicate_holds
from ml.runner import run_tests, trace

_PARSE = frozenset({"parse_error", "unsupported"})


def _odd_scalar(problem):
    for test in problem.get("tests") or []:
        for arg in test.get("args") or []:
            if isinstance(arg, bool):
                continue
            if isinstance(arg, int) and arg % 2 == 1:
                return True
    return False


class Checker:
    def __init__(self):
        self._class_ids = class_op_ids()
        self.runs = {}
        self.holds_cache = {}
        self.firing = {}
        self.judged = {}
        self.traces = {}
        self.odd = {}

    def key(self, code):
        return execution_key(code)

    def execute(self, problem, code, key):
        cache_key = (problem["problem_id"], key)
        hit = self.runs.get(cache_key)
        if hit is None:
            hit = run_tests(problem, code, backend="interp")
            self.runs[cache_key] = hit
        return hit

    def _holds(self, op_id, key):
        cache_key = (op_id, key)
        hit = self.holds_cache.get(cache_key)
        if hit is None:
            hit = predicate_holds(op_id, key)
            self.holds_cache[cache_key] = hit
        return hit

    def _firing(self, key):
        hit = self.firing.get(key)
        if hit is None:
            hit = [op_id for op_id in self._class_ids if self._holds(op_id, key)]
            self.firing[key] = hit
        return hit

    def _odd(self, problem):
        pid = problem["problem_id"]
        if pid not in self.odd:
            self.odd[pid] = _odd_scalar(problem)
        return self.odd[pid]

    def judge(self, problem, code, mode, op_id=None):
        """Return ``(decision, info)``. ``decision`` is ``keep``, ``luck`` or a drop reason."""
        key = self.key(code)
        if key is None:
            return "parse", None
        cache_key = (problem["problem_id"], key, mode, op_id)
        hit = self.judged.get(cache_key)
        if hit is not None:
            return hit
        try:
            result = self.execute(problem, code, key)
        except Exception:
            decision = ("crash", None)
            self.judged[cache_key] = decision
            return decision
        status = result.get("status") or ""
        tests = result.get("tests") or {}
        total = int(tests.get("total") or 0)
        passed = int(tests.get("passed") or 0)
        failed = total - passed
        fail_set = frozenset(
            index for index, row in enumerate(tests.get("results") or []) if not row.get("pass")
        )
        info = {
            "tests_failed": failed,
            "tests_total": total,
            "status": status,
            "fail_set": fail_set,
            "key": key,
        }
        if status in _PARSE:
            decision = ("parse", info)
        elif mode == "correct":
            decision = ("keep", info) if failed == 0 and status == "ok" else ("fail_tests", info)
        elif failed == 0:
            decision = ("luck", info)
        elif mode == "other":
            fired = self._firing(key)
            info = dict(info)
            info["fired"] = fired
            decision = ("other_class", info) if fired else ("keep", info)
        else:
            if op_id == "d05_unreachable_base" and not self._odd(problem):
                decision = ("odd_n", info)
            elif not self._holds(op_id, key):
                decision = ("predicate", info)
            else:
                decision = ("keep", info)
        self.judged[cache_key] = decision
        return decision

    def summarize(self, problem, code, ref_iters):
        key = self.key(code)
        if key is None:
            return {"status": "parse_error", "events": [], "loop_iters_delta": None}
        cache_key = (problem["problem_id"], key)
        hit = self.traces.get(cache_key)
        if hit is not None:
            return _with_delta(hit, ref_iters)
        try:
            recorded = trace(problem, code)
        except Exception:
            recorded = {"status": "runtime_error", "events": [], "loop_iters": {}}
        events = []
        seen = set()
        for event in recorded.get("events") or []:
            kind = event.get("type")
            if kind and kind not in seen:
                seen.add(kind)
                events.append(kind)
        iters = sum((recorded.get("loop_iters") or {}).values())
        hit = {"status": recorded.get("status") or "ok", "events": events, "iters": iters}
        self.traces[cache_key] = hit
        return _with_delta(hit, ref_iters)


def _with_delta(hit, ref_iters):
    delta = None if ref_iters is None else float(hit["iters"] - ref_iters)
    return {"status": hit["status"], "events": list(hit["events"]), "loop_iters_delta": delta}


def reference_iters(checker, problem):
    variants = problem.get("correct_variants") or []
    if not variants:
        return None
    summary = checker.summarize(problem, variants[0], None)
    # summarize with ref None stores the trace; read the cached iter count.
    key = checker.key(variants[0])
    cached = checker.traces.get((problem["problem_id"], key))
    if cached is None:
        return None
    return cached["iters"]
