"""The three sample problems (tests/fixtures/problems): all correct variants pass on the
interpreter (03 §2.5 "all correct variants of all problems pass their tests"), wrong programs
fail the same tests as on gcc (03 §2.5 "gcc differential"), and the hand-written W0 traces
agree with the interpreter on everything the spec fixes.
"""
import json
from pathlib import Path

import pytest

from ml import runner
from ml.contracts.subset import EFFECTS_COUNT_KEYS
from tests.A3.helpers import events, irun, itrace, needs_gcc, needs_interp, passes

pytestmark = needs_interp

FIXTURES = Path(__file__).parent.parent / "fixtures"
PROBLEMS = {p["problem_id"]: p for p in (json.loads(f.read_text(encoding="utf-8"))
                                         for f in sorted((FIXTURES / "problems").glob("*.json")))}
VARIANTS = [(pid, k) for pid, p in PROBLEMS.items() for k in range(len(p["correct_variants"]))]
UB_EVENTS = ("uninit_read", "oob_read", "oob_write", "overflow", "div_zero", "step_cap_hit", "depth_cap_hit",
             "missing_return")


@pytest.mark.parametrize("pid,k", VARIANTS, ids=[f"{p}-cv{k}" for p, k in VARIANTS])
def test_correct_variants_pass(pid, k):
    """03 §2.5 unit tests: every correct variant of every problem passes all its tests."""
    problem = PROBLEMS[pid]
    result = irun(problem, problem["correct_variants"][k])
    assert result["status"] == "ok"
    assert result["tests"]["passed"] == result["tests"]["total"] == len(problem["tests"])
    trace = itrace(problem, problem["correct_variants"][k])
    assert trace["status"] == "ok" and trace["truncated"] is False
    assert len(trace["per_test"]) == len(problem["tests"])
    assert [pt["returned"] for pt in trace["per_test"]] == [t["expect"]["returned"] for t in problem["tests"]]
    assert trace["returned"] == problem["tests"][problem["display_test"]]["expect"]["returned"]
    for kind in UB_EVENTS:
        assert events(trace, kind) == [], kind


@pytest.mark.parametrize("pid", sorted(PROBLEMS))
def test_sample_only_on_sample_problems(pid):
    """03 §2.3: sample_only runs the 2 tests marked sample."""
    result = irun(PROBLEMS[pid], PROBLEMS[pid]["correct_variants"][0], sample_only=True)
    assert result["tests"]["total"] == 2 and result["tests"]["passed"] == 2


def test_q17_recursion_depth_per_test():
    """03 §2.3 max_depth_le / §2.2 row 18: factorial(n) recurses to depth max(n, 1)."""
    trace = itrace(PROBLEMS["Q17"], PROBLEMS["Q17"]["correct_variants"][0])
    assert [pt["max_depth"] for pt in trace["per_test"]] == [3, 1, 1, 5, 10]
    assert trace["max_depth"] == 10 and trace["effects_count"]["call"] == 20


# Wrong programs whose behaviour C defines (no out-of-bounds, uninitialised or overflowing code).
MUTANTS = [
    ("P03", "overwrite_instead_of_add", "int total_energy(int cells[], int n) {\n    int total = 0;\n    for (int i = 0; i < n; i++) {\n        total = cells[i];\n    }\n    return total;\n}"),
    ("P03", "starts_at_one", "int total_energy(int cells[], int n) {\n    int total = 0;\n    for (int i = 1; i < n; i++) {\n        total += cells[i];\n    }\n    return total;\n}"),
    ("P03", "stops_one_early", "int total_energy(int cells[], int n) {\n    int total = 0;\n    for (int i = 0; i < n - 1; i++) {\n        total += cells[i];\n    }\n    return total;\n}"),
    ("P03", "returns_inside_loop", "int total_energy(int cells[], int n) {\n    int total = 0;\n    for (int i = 0; i < n; i++) {\n        total += cells[i];\n        return total;\n    }\n    return total;\n}"),
    ("P03", "reset_inside_loop", "int total_energy(int cells[], int n) {\n    int total = 0;\n    for (int i = 0; i < n; i++) {\n        total = 0;\n        total += cells[i];\n    }\n    return total;\n}"),
    ("P03", "for_semicolon", "int total_energy(int cells[], int n) {\n    int total = 0;\n    int i;\n    for (i = 0; i < n - 1; i++);\n    {\n        total += cells[i];\n    }\n    return total;\n}"),
    ("P11", "assign_in_condition", "int door_open(int code) {\n    if (code = 42) {\n        return 1;\n    }\n    return 0;\n}"),
    ("P11", "if_semicolon", "int door_open(int code) {\n    if (code == 42);\n    {\n        return 1;\n    }\n    return 0;\n}"),
    ("P11", "not_equal", "int door_open(int code) {\n    if (code != 42) {\n        return 1;\n    }\n    return 0;\n}"),
    ("P11", "prints_instead_of_returning", "int door_open(int code) {\n    if (code == 42) {\n        printf(\"1\");\n        return 0;\n    }\n    return 0;\n}"),
    ("Q17", "base_case_returns_zero", "int factorial(int n) {\n    if (n <= 1) {\n        return 0;\n    }\n    return n * factorial(n - 1);\n}"),
    ("Q17", "adds_instead_of_multiplying", "int factorial(int n) {\n    if (n <= 1) {\n        return 1;\n    }\n    return n + factorial(n - 1);\n}"),
    ("Q17", "discards_the_recursive_value", "int factorial(int n) {\n    if (n <= 1) {\n        return 1;\n    }\n    factorial(n - 1);\n    return n;\n}"),
    ("Q17", "no_base_case", "int factorial(int n) {\n    return n * factorial(n - 1);\n}"),
    ("Q17", "loop_instead_of_recursion", "int factorial(int n) {\n    int r = 1;\n    for (int i = 2; i <= n; i++) {\n        r *= i;\n    }\n    return r;\n}"),
]


@needs_gcc
@pytest.mark.parametrize("pid,name,code", MUTANTS, ids=[f"{m[0]}-{m[1]}" for m in MUTANTS])
def test_wrong_programs_fail_the_same_tests_as_on_gcc(pid, name, code):
    """03 §2.5 gcc differential: for programs without undefined behaviour the interpreter and gcc
    agree on the status and on which tests pass."""
    by_gcc = runner.run_tests(PROBLEMS[pid], code, backend="gcc")
    by_interp = irun(PROBLEMS[pid], code)
    assert by_interp["status"] == by_gcc["status"], name
    assert passes(by_interp) == passes(by_gcc), name
    assert by_interp["tests"]["passed"] == by_gcc["tests"]["passed"]


# ---------------------------------------------------------------- the hand-written W0 traces

P03_LE = "int total_energy(int cells[], int n) {\n    int total = 0;\n    for (int i = 0; i <= n; i++) {\n        total += cells[i];\n    }\n    return total;\n}"
P03_NO_UPDATE = "int total_energy(int cells[], int n) {\n    int total = 0;\n    int i = 0;\n    while (i < n) {\n        total += cells[i];\n    }\n    return total;\n}"


def fixture(name):
    return json.loads((FIXTURES / "traces" / f"{name}.json").read_text(encoding="utf-8"))


def summary(trace):
    """The parts of a trace that follow from 03 alone, whatever the exact step accounting."""
    return {
        "status": trace["status"], "returned": trace["returned"], "printed": trace["printed"],
        "n_events": len(trace["events"]), "loop_iters": trace["loop_iters"],
        "branch": {k: v for k, v in trace["branch"].items()},
        "effects_count": {k: trace["effects_count"].get(k, 0) for k in EFFECTS_COUNT_KEYS},
        "max_depth": trace["max_depth"],
        "per_test": [{"status": p["status"], "returned": p["returned"], "max_depth": p["max_depth"],
                      "loop_iters": {k: v for k, v in p["loop_iters"].items() if v}} for p in trace["per_test"]],
    }


@pytest.mark.parametrize("name,pid,code", [("p03_le_oob_read", "P03", P03_LE), ("q17_recursion_ok", "Q17", None)])
def test_w0_fixture_traces_counters_and_events(name, pid, code):
    """notes/W0.md "Still faked": the sample traces were simulated by hand; the interpreter must
    produce the same status, values, events and counters for the same program (03 §2.2, §2.4)."""
    code = code or PROBLEMS[pid]["correct_variants"][0]
    got, want = itrace(PROBLEMS[pid], code), fixture(name)
    assert summary(got) == summary(want)
    for got_event, want_event in zip(got["events"], want["events"]):      # the fixture's fields; extras are allowed
        assert {k: got_event.get(k) for k in want_event} == want_event


@pytest.mark.parametrize("name,pid,code", [("p03_le_oob_read", "P03", P03_LE), ("q17_recursion_ok", "Q17", None)])
def test_w0_fixture_traces_steps(name, pid, code):
    """notes/W0.md: if the interpreter's steps differ from the hand-written ones, the interpreter
    and 03 win and the fixtures are rebuilt. So a difference here is reported as xfail, not failure."""
    code = code or PROBLEMS[pid]["correct_variants"][0]
    got, want = itrace(PROBLEMS[pid], code)["steps"], fixture(name)["steps"]
    if got != want:
        first = next((i for i, (g, w) in enumerate(zip(got, want)) if g != w), min(len(got), len(want)))
        pytest.xfail(f"{name}: steps differ from the W0 fixture from step {first} "
                     f"({len(got)} steps vs {len(want)}); rebuild the fixture per notes/W0.md")


def test_w0_step_cap_fixture():
    """notes/W0.md: p03_no_update_step_cap is cut to 60 steps to stay small, so only its status,
    events and first steps are compared."""
    want = fixture("p03_no_update_step_cap")
    got = itrace(PROBLEMS["P03"], P03_NO_UPDATE)
    assert got["status"] == want["status"] == "timeout" and got["returned"] is None
    assert [(e["type"], e["test"]) for e in got["events"]] == [(e["type"], e["test"]) for e in want["events"]]
    assert [p["status"] for p in got["per_test"]] == ["timeout"] * 5
    assert got["truncated"] is True
    if got["steps"][:60] != want["steps"]:
        pytest.xfail("first 60 steps differ from the W0 fixture; rebuild the fixture per notes/W0.md")
