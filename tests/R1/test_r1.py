"""R1: the fixer repairs held-out mutants, and the counterexample and fix probes fire.

One mutant per operator (first site on the first variant that has one). Holdout
problems are P04, P09, P13, Q04, Q09, Q16. An operator with no site there is
measured on a train problem and is not part of the 80% rate.

    .venv\\Scripts\\python -m pytest tests/R1 -q
"""
import fnmatch
import json
from pathlib import Path

from ml.contracts.classes import MISCONCEPTIONS
from ml.features.fix_feats import fix_features
from ml.generate.ops_dsa import all_ops as dsa_ops
from ml.generate.ops_dsa import apply as dsa_apply
from ml.generate.ops_dsa import sites as dsa_sites
from ml.generate.ops_main import all_ops as main_ops
from ml.generate.ops_main import apply as main_apply
from ml.generate.ops_main import sites as main_sites
from ml.learner.counterexample import search
from ml.learner.fixer import repair
from ml.runner import run_tests

ROOT = Path(__file__).resolve().parents[2]
HOLDOUT = {"P04", "P09", "P13", "Q04", "Q09", "Q16"}


def _load():
    problems = []
    for folder in ("main", "dsa"):
        for path in sorted((ROOT / "ml" / "problems" / folder).glob("*.json")):
            problems.append(json.loads(path.read_text(encoding="utf-8")))
    return problems


def _operators():
    found = []
    for op in main_ops():
        if op.test_only or not any(label in MISCONCEPTIONS for label in op.labels):
            continue
        found.append((op, main_sites, main_apply))
    for op in dsa_ops():
        if op.test_only or not any(label in MISCONCEPTIONS for label in op.labels):
            continue
        found.append((op, dsa_sites, dsa_apply))
    return found


def _allowed(op, problem):
    return any(fnmatch.fnmatch(op.op_id, pattern) for pattern in problem.get("allowed_ops") or [])


def _first_mutant(op, sites, apply, problems):
    for problem in problems:
        for code in problem["correct_variants"]:
            try:
                found = sites(op, code)
            except Exception:
                continue
            if not found:
                continue
            return problem, apply(op, code, found[0])
    return None, None


def _label(op):
    for label in op.labels:
        if label in MISCONCEPTIONS:
            return label
    return op.labels[0]


def collect():
    """One mutant per operator. Holdout sites are the rate; the rest are train."""
    problems = _load()
    holdout = [p for p in problems if p["split"] == "holdout_problem" or p["problem_id"] in HOLDOUT]
    train = [p for p in problems if p not in holdout]
    rows = []
    no_site = []
    for op, sites, apply in _operators():
        problem, mutant = _first_mutant(op, sites, apply, holdout)
        band = "holdout"
        if problem is None:
            problem, mutant = _first_mutant(op, sites, apply, train)
            band = "train"
        if problem is None:
            no_site.append(op.op_id)
            continue
        cls = _label(op)
        fix = repair(problem, mutant, cls)
        passed = False
        if fix["kind"] == "minimal":
            result = run_tests(problem, fix["code"])
            tests = result["tests"]
            passed = tests["total"] > 0 and tests["passed"] == tests["total"]
        rows.append({
            "op": op.op_id,
            "class": cls,
            "problem": problem["problem_id"],
            "band": band,
            "kind": fix["kind"],
            "rule": fix["rule"],
            "ok": fix["kind"] == "minimal" and passed,
            "allowed": _allowed(op, problem),
        })
    return rows, no_site


def _rate(rows, band):
    chosen = [row for row in rows if row["band"] == band]
    if not chosen:
        return 0.0, 0, 0
    ok = sum(row["ok"] for row in chosen)
    return ok / len(chosen), ok, len(chosen)


def _report(rows, no_site):
    hold_rate, hold_ok, hold_n = _rate(rows, "holdout")
    train_rate, train_ok, train_n = _rate(rows, "train")
    lines = [
        f"holdout repair rate: {hold_rate:.1%} ({hold_ok}/{hold_n})",
        f"train repair rate (operators with no holdout site): {train_rate:.1%} ({train_ok}/{train_n})",
    ]
    missed = [row for row in rows if row["band"] == "holdout" and not row["ok"]]
    if missed:
        lines.append("holdout misses:")
        for row in missed:
            lines.append(f"  {row['op']} {row['class']} {row['problem']} -> {row['kind']} {row['rule']}")
    train_miss = [row for row in rows if row["band"] == "train" and not row["ok"]]
    if train_miss:
        lines.append("train misses (not in the 80% rate):")
        for row in train_miss:
            lines.append(f"  {row['op']} {row['class']} {row['problem']} -> {row['kind']} {row['rule']}")
    if no_site:
        lines.append("no site on any problem: " + ", ".join(no_site))
    uncovered = sorted({cls for cls in MISCONCEPTIONS if not any(row["class"] == cls and row["band"] == "holdout" for row in rows)})
    if uncovered:
        lines.append("classes with no holdout mutant: " + ", ".join(uncovered))
    return "\n".join(lines), hold_rate


def test_holdout_repair_rate(capsys):
    rows, no_site = collect()
    text, rate = _report(rows, no_site)
    print(text)
    with capsys.disabled():
        print(text)
    assert rows, "no mutants were generated"
    assert rate >= 0.80


def test_counterexample_differs():
    problems = {p["problem_id"]: p for p in _load()}
    problem = problems["P04"]
    op, sites, apply = next(item for item in _operators() if item[0].op_id == "m03_assign_in_loop")
    code = problem["correct_variants"][0]
    mutant = apply(op, code, sites(op, code)[0])
    found = search(problem, mutant, "M03")
    assert found is not None
    assert found["yours"] != found["intended"]
    assert "input" in found and "effect_diff" in found


def test_fix_feature_for_known_class():
    problems = {p["problem_id"]: p for p in _load()}
    problem = problems["P04"]
    op, sites, apply = next(item for item in _operators() if item[0].op_id == "m03_assign_in_loop")
    code = problem["correct_variants"][0]
    mutant = apply(op, code, sites(op, code)[0])
    feats = fix_features(problem, mutant, ["M03"])
    assert feats["f_fix_M03"] == 1
    assert feats["f_fixgain_M03"] > 0
    assert "f_fix_M01" not in feats
    assert "f_fixgain_M01" not in feats
