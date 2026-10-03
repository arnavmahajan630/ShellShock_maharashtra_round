"""Shared helpers for the F2 tests: fixture loading and hand-built traces in the contract shape."""
import json
import math
from pathlib import Path

from ml.contracts import schemas as S
from ml.contracts.feature_names import GROUP_B, GROUP_R
from ml.contracts.subset import EFFECTS_COUNT_KEYS

FIXTURES = Path(__file__).parent.parent / "fixtures"


def load(relative):
    return json.loads((FIXTURES / relative).read_text(encoding="utf-8"))


P03 = load("problems/P03_total_energy.json")
P11 = load("problems/P11_door_open.json")
Q17 = load("problems/Q17_factorial.json")


def make_trace(per_test, events=(), steps=(), truncated=False):
    """A trace dict from per-test records: sums the counters the way the interpreter does and
    checks the result against schemas.Trace, so hand-built traces cannot drift from the contract."""
    full = []
    for pt in per_test:
        full.append({"status": "ok", "returned": None, "printed": "", "loop_iters": {}, "branch": {},
                     "effects_count": {}, "max_depth": 1, **pt})
    loop_iters, branch, effects = {}, {}, {k: 0 for k in EFFECTS_COUNT_KEYS}
    for pt in full:
        for key, n in pt["loop_iters"].items():
            loop_iters[key] = loop_iters.get(key, 0) + n
        for key, counts in pt["branch"].items():
            slot = branch.setdefault(key, {"true": 0, "false": 0})
            for side, n in counts.items():
                slot[side] += n
        for key, n in pt["effects_count"].items():
            effects[key] = effects.get(key, 0) + n
    status = next((pt["status"] for pt in full if pt["status"] != "ok"), "ok")
    trace = {"status": status, "returned": full[0]["returned"], "printed": full[0]["printed"],
             "steps": [{"i": i, "events": [], "effects": [], "vars": {}, **s} for i, s in enumerate(steps)],
             "loop_iters": loop_iters, "branch": branch, "events": list(events), "effects_count": effects,
             "max_depth": max(pt["max_depth"] for pt in full), "truncated": truncated, "per_test": full}
    S.Trace.model_validate(trace)
    return trace


def exact(feats, names, **nonzero):
    """Assert a full group: the named values, NaN where given as None, 0 everywhere else."""
    assert list(feats) == names
    wrong = {}
    for name in names:
        want = nonzero.get(name, 0.0)
        got = feats[name]
        if want is None:
            ok = math.isnan(got)
        else:
            ok = not math.isnan(got) and abs(got - want) < 1e-9
        if not ok:
            wrong[name] = (got, want)
    assert not wrong, f"(got, expected) {wrong}"
    assert set(nonzero) <= set(names)


def exact_b(feats, **nonzero):
    exact(feats, GROUP_B, **nonzero)


def exact_r(feats, **nonzero):
    exact(feats, GROUP_R, **nonzero)
