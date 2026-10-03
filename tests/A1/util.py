"""Helpers shared by the A1 tests."""
import json
import re
from pathlib import Path

from ml.c_interp import harness
from ml.contracts import schemas as S
from ml.contracts.subset import EFFECT_FIELDS, EVENT_FIELDS

FIXTURES = Path(__file__).parent.parent / "fixtures"


def load_problem(stem):
    return json.loads((FIXTURES / "problems" / f"{stem}.json").read_text(encoding="utf-8"))


def load_trace(stem):
    return json.loads((FIXTURES / "traces" / f"{stem}.json").read_text(encoding="utf-8"))


def problem(signature, tests, **extra):
    """A minimal problem dict: tests are (args, expect) pairs or bare args lists."""
    name = re.match(r"[^(]*?(\w+)\s*\(", signature).group(1)
    rows = []
    for test in tests:
        if isinstance(test, tuple):
            rows.append({"args": list(test[0]), "expect": dict(test[1])})
        else:
            rows.append({"args": list(test), "expect": {}})
    out = {"problem_id": "T00", "name": name, "signature": signature, "tests": rows,
           "display_test": 0, "forbid": []}
    out.update(extra)
    return out


def trace(code, signature, *arg_lists, **extra):
    """Trace `code` with one test per args list; the result is checked against the contract."""
    out = harness.trace(problem(signature, arg_lists, **extra), code)
    check_trace(out)
    return out


def run(code, signature, tests, **extra):
    out = harness.run_tests(problem(signature, tests, **extra), code)
    S.RunResult.model_validate(out)
    return out


def check_trace(out):
    """The same checks tests/test_contracts.py makes on the sample traces."""
    model = S.Trace.model_validate(out)
    json.dumps(out, allow_nan=False)
    assert [s.i for s in model.steps] == list(range(len(model.steps)))
    for event in model.events:
        assert event.type in EVENT_FIELDS
        assert set(EVENT_FIELDS[event.type]) <= set(event.model_extra)
        assert "test" in event.model_extra
    for step in model.steps:
        for event in step.events:
            assert event.type in EVENT_FIELDS and "test" not in event.model_extra
        for effect in step.effects:
            name, *fields = effect.split(":")
            assert name in EFFECT_FIELDS and len(fields) == len(EFFECT_FIELDS[name]), effect
    for key, total in model.loop_iters.items():
        assert sum(pt.loop_iters.get(key, 0) for pt in model.per_test) == total
    for key, total in model.effects_count.items():
        assert sum(pt.effects_count.get(key, 0) for pt in model.per_test) == total
    for key, total in model.branch.items():
        for side in ("true", "false"):
            assert sum(pt.branch.get(key, {}).get(side, 0) for pt in model.per_test) == total[side]
    if model.per_test:
        assert max(pt.max_depth for pt in model.per_test) == model.max_depth


def events(out, etype):
    return [e for e in out["events"] if e["type"] == etype]


def effects(out):
    return [effect for step in out["steps"] for effect in step["effects"]]


def lines(out):
    return [step["line"] for step in out["steps"]]
