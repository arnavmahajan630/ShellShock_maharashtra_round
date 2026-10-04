"""R2: timeline, call-stack, window and memory builders, and the intervention policy.

    .venv\\Scripts\\python -m pytest tests/R2

Three groups:
  * policy: the modality order of 03 §7.1 and "never repeat a tried modality";
  * fixture traces (tests/fixtures/traces/): builders on traces that were not made by the
    interpreter, compared with server/fixtures/intervene.json (03 §11.2);
  * live traces: hand-written mutants of the real problems, then one mutant per generator
    operator, through ml.runner (the interpreter).
"""
import fnmatch
import json
from pathlib import Path

import pytest

from ml.contracts import schemas
from ml.contracts.classes import MISCONCEPTIONS
from ml.generate.ops_dsa import all_ops as dsa_ops
from ml.generate.ops_dsa import apply as dsa_apply
from ml.generate.ops_dsa import sites as dsa_sites
from ml.generate.ops_main import all_ops as main_ops
from ml.generate.ops_main import apply as main_apply
from ml.generate.ops_main import sites as main_sites
from ml.learner import interventions as iv
from ml.learner import timeline as tl
from tests.fixtures import build as B

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures"
TIMELINE_FLAGS = {None, "extra", "missing", "void_read", "static", "reset", "overwrite", "empty_body", "early_return"}
FRAME_FLAGS = {None, "no_base", "same_arg", "growing_arg", "dropped_value"}
WRITE_FLAGS = {None, "duplicate", "lost", "one_pass_end"}


def _problem(pid):
    folder = "main" if pid.startswith("P") else "dsa"
    for path in (ROOT / "ml" / "problems" / folder).glob(f"{pid}_*.json"):
        return json.loads(path.read_text(encoding="utf-8"))
    raise KeyError(pid)


def _variant(pid):
    return _problem(pid)["correct_variants"][0]


def _fixture_trace(name):
    return json.loads((FIX / "traces" / name).read_text(encoding="utf-8"))


def _check_package(pk, cls):
    """What every package must satisfy (03 §11.2)."""
    model = schemas.Intervention.model_validate(pk)
    assert model.cls == cls
    json.dumps(pk)
    for key in ("class", "modality", "next_modalities", "copy", "question", "timeline", "counterexample", "fix"):
        assert key in pk, key
    assert pk["modality"] in iv.MODALITIES
    assert pk["modality"] not in pk["next_modalities"]
    assert len(pk["copy"]) >= 1
    assert pk["question"] and "prompt" in pk["question"] and "answer" in pk["question"]
    assert len(pk["timeline"]) <= tl.MAX_TIMELINE_ROWS
    for row in pk["timeline"]:
        assert set(row) == {"step", "line", "vars", "effect", "flag"}
        assert row["flag"] in TIMELINE_FLAGS
    panel = iv.PANEL_OF.get(cls)
    if panel:
        assert panel in pk, f"{cls} should carry {panel}"
    if "call_stack" in pk:
        frames = [f for f in pk["call_stack"] if isinstance(f, dict)]
        assert len(frames) <= tl.MAX_FRAMES
        assert all(f["flag"] in FRAME_FLAGS for f in frames)
        rest = [f for f in pk["call_stack"] if isinstance(f, str)]
        assert len(rest) <= 1 and all(f.startswith("…+") for f in rest)
    if "window_strip" in pk:
        assert all(r["flag"] in (None, "frozen") for r in pk["window_strip"] if isinstance(r, dict))
    if "memory_strip" in pk:
        strip = pk["memory_strip"]
        assert {"array", "values", "reads"} <= set(strip)
        assert all(w["flag"] in WRITE_FLAGS for w in strip.get("writes", []))
    if pk["fix"] is not None:
        schemas.Fix.model_validate(pk["fix"])
    return model


# ---------------------------------------------------------------- policy (03 §7.1)

EXPECTED_POLICY = {
    "M01": ["trace_timeline", "counterexample", "minimal_fix"],
    "M02": ["trace_timeline", "counterexample", "minimal_fix"],
    "M03": ["trace_timeline", "minimal_fix"],
    "M04": ["value_meter", "counterexample"],
    "M05": ["trace_timeline", "minimal_fix"],
    "M06": ["trace_timeline", "minimal_fix"],
    "M07": ["trace_timeline", "counterexample", "minimal_fix"],
    "M08": ["memory_strip", "counterexample", "minimal_fix"],
    "M10": ["trace_timeline", "minimal_fix"],
    "D01": ["trace_timeline", "counterexample", "minimal_fix"],
    "D02": ["window_strip", "minimal_fix"],
    "D03": ["memory_strip", "counterexample", "minimal_fix"],
    "D04": ["memory_strip", "counterexample"],
    "D05": ["call_stack", "counterexample", "minimal_fix"],
    "D06": ["call_stack", "counterexample", "minimal_fix"],
    "D07": ["call_stack", "minimal_fix"],
    "D08": ["value_meter", "minimal_fix"],
}


def test_policy_table_matches_the_plan():
    assert set(EXPECTED_POLICY) == set(MISCONCEPTIONS)
    for cls, order in EXPECTED_POLICY.items():
        assert iv.policy(cls) == order, cls
    assert iv.policy("M07", "if") == ["trace_timeline", "minimal_fix"]
    assert iv.policy("M07", "for") == iv.policy("M07", "while") == EXPECTED_POLICY["M07"]
    assert iv.policy("OTHER") == ["trace_timeline", "minimal_fix"]
    assert iv.policy("U1") == ["trace_timeline", "minimal_fix"]


def test_policy_returns_copies():
    iv.policy("M01").append("x")
    assert iv.policy("M01") == EXPECTED_POLICY["M01"]


@pytest.mark.parametrize("cls", MISCONCEPTIONS + ["OTHER"])
def test_next_modality_is_first_untried_and_never_repeats(cls):
    order = iv.policy(cls)
    tried = []
    seen = []
    for _ in range(len(order)):
        modality, rest, exhausted = iv.choose_modality(cls, tried)
        assert not exhausted
        assert modality not in tried
        assert rest == [m for m in order if m not in tried and m != modality]
        seen.append(modality)
        tried.append(modality)
    assert seen == order


def test_exhausted_modalities_fall_back_to_fix_then_repeat_flagged():
    modality, rest, exhausted = iv.choose_modality("M04", ["value_meter", "counterexample"])
    assert (modality, rest, exhausted) == ("minimal_fix", [], False)
    modality, rest, exhausted = iv.choose_modality("M04", ["value_meter", "counterexample", "minimal_fix"])
    assert modality == "value_meter" and exhausted is True
    modality, rest, exhausted = iv.choose_modality("M08", ["memory_strip", "counterexample", "minimal_fix"])
    assert exhausted is True


def test_requested_modality_wins_and_unknown_is_ignored():
    modality, rest, exhausted = iv.choose_modality("M08", ["memory_strip"], requested="minimal_fix")
    assert modality == "minimal_fix" and rest == ["counterexample"] and not exhausted
    modality, _, _ = iv.choose_modality("M08", [], requested="nonsense")
    assert modality == "memory_strip"


# ---------------------------------------------------------------- fixture traces

@pytest.fixture(scope="module")
def p03():
    return json.loads((FIX / "problems" / "P03_total_energy.json").read_text(encoding="utf-8"))


def test_timeline_on_fixture_trace_equals_the_server_fixture(p03):
    """03 §11.2 example: M08 on P03 with `i <= n`. The fixture was written by hand."""
    server = json.loads((ROOT / "server" / "fixtures" / "intervene.json").read_text(encoding="utf-8"))
    trace = _fixture_trace("p03_le_oob_read.json")
    ctx = tl.context_from_traces(p03, B.P03_LE, trace, ref=None, test=0)
    assert tl.build_timeline(ctx) == server["timeline"]
    assert tl.build_memory_strip(ctx) == server["memory_strip"]
    assert tl.question_for("M08", ctx)["prompt"] == server["question"]["prompt"]


def test_timeline_on_fixture_trace_with_reference_flags_the_extra_pass(p03):
    trace = _fixture_trace("p03_le_oob_read.json")
    ref = tl.trace_of(p03, p03["correct_variants"][0], None)
    ctx = tl.context_from_traces(p03, B.P03_LE, trace, ref=ref, test=0)
    rows = tl.build_timeline(ctx)
    assert [r["flag"] for r in rows] == [None, None, None, "void_read"]
    assert tl.question_for("M08", ctx)["answer"] == 2
    assert tl.question_for("M01", ctx) == {"prompt": "At which value of i should the loop stop?", "answer": 3}


def test_step_cap_fixture_trace_gives_a_capped_timeline(p03):
    trace = _fixture_trace("p03_no_update_step_cap.json")
    assert trace["truncated"] is True
    ref = tl.trace_of(p03, p03["correct_variants"][0], None)
    ctx = tl.context_from_traces(p03, B.P03_NO_UPDATE, trace, ref=ref, test=0)
    rows = tl.build_timeline(ctx)
    assert 3 <= len(rows) <= tl.MAX_TIMELINE_ROWS
    assert rows[-1]["flag"] == "extra"
    assert all(r["vars"]["i"] == 0 for r in rows)
    # Same trace, no source: falls back to one row per changed step and still stays small.
    flat = tl.build_timeline(tl.context_from_traces({**p03, "correct_variants": []}, None, trace, ref=None, test=0))
    assert 1 <= len(flat) <= tl.MAX_TIMELINE_ROWS


def test_call_stack_on_the_recursion_fixture():
    q17 = json.loads((FIX / "problems" / "Q17_factorial.json").read_text(encoding="utf-8"))
    trace = _fixture_trace("q17_recursion_ok.json")
    ctx = tl.context_from_traces(q17, B.Q17_OK, trace, ref=trace, test=0)
    frames = tl.build_call_stack(ctx)
    assert frames == [
        {"depth": 1, "fn": "factorial", "args": [3], "returned": 6, "flag": None},
        {"depth": 2, "fn": "factorial", "args": [2], "returned": 2, "flag": None},
        {"depth": 3, "fn": "factorial", "args": [1], "returned": 1, "flag": None},
    ]
    assert tl.question_for("D05", ctx) == {"prompt": "At which argument should the calls stop?", "answer": 1}


def test_call_stack_is_capped_with_a_remainder_marker():
    steps = [{"i": i, "line": 2, "vars": {"n": 100 - i}, "events": [],
              "effects": [f"call:f:{i + 1}:{100 - i}"]} for i in range(30)]
    trace = {"status": "timeout", "steps": steps, "events": [{"type": "depth_cap_hit", "line": 2, "fn": "f"}],
             "per_test": [{}]}
    ctx = tl.context_from_traces({"signature": "int f(int n)", "tests": [{"args": [100]}]}, None, trace, ref=None)
    frames = tl.build_call_stack(ctx)
    assert len(frames) == tl.MAX_FRAMES + 1
    assert frames[-1] == "…+18"
    assert all(isinstance(f, dict) for f in frames[:-1])


def test_effect_parsing():
    assert tl.parse_effect("call:factorial:2:3") == ("call", ["factorial", "2", "3"])
    assert tl.parse_effect("call:total_energy:1:cells,3") == ("call", ["total_energy", "1", "cells,3"])
    assert tl.parse_effect("ret:f:2:") == ("ret", ["f", "2", ""])
    assert tl.parse_effect("write_cell:1:7") == ("write_cell", ["1", "7"])
    assert tl.parse_effect("fire") == ("fire", [])


# ---------------------------------------------------------------- live traces: hand-written mutants

def _mutants():
    v = _variant
    return {
        "M01_le": ("P03", "M01", v("P03").replace("i < n", "i <= n")),
        "M08_le": ("P03", "M08", v("P03").replace("i < n", "i <= n")),
        "M08_start1": ("P08", "M08", v("P08").replace("ids[n - 1]", "ids[n]")),
        "M03": ("P03", "M03", v("P03").replace("total += cells[i];", "total = 0;\n        total += cells[i];")),
        "M05": ("P03", "M05", v("P03").replace("int total = 0;", "int total;")),
        "M02": ("P01", "M02", v("P01").replace("i++", "")),
        "M06": ("P11", "M06", v("P11").replace("code == 42", "code = 42")),
        "M07_if": ("P11", "M07", v("P11").replace("code == 42)", "code == 42);")),
        "M07_for": ("P01", "M07", v("P01").replace("i++)", "i++);")),
        "M04": ("P13", "M04", v("P13").replace("(float)fuel / cap * 100", "fuel / cap * 100")),
        "M10": ("P14", "M10", "int distance(int a, int b) {\n    if (a >= b) {\n        printf(\"%d\", a - b);\n"
                              "    } else {\n        printf(\"%d\", b - a);\n    }\n    return 0;\n}"),
        "D01": ("Q01", "D01", "int linear_search(int a[], int n, int x) {\n    for (int i = 0; i < n; i++) {\n"
                              "        if (a[i] == x) {\n            return i;\n        } else {\n            return -1;\n"
                              "        }\n    }\n    return -1;\n}"),
        "D02": ("Q03", "D02", v("Q03").replace("low = mid + 1", "low = mid")),
        "D03": ("Q10", "D03", v("Q10").replace("int t = a[i];\n        a[i] = a[n - 1 - i];\n"
                                               "        a[n - 1 - i] = t;",
                                               "a[i] = a[n - 1 - i];\n        a[n - 1 - i] = a[i];")),
        "D04": ("Q06", "D04", "void bubble_sort(int a[], int n) {\n    for (int j = 0; j < n - 1; j++) {\n"
                              "        if (a[j] > a[j + 1]) {\n            int t = a[j];\n            a[j] = a[j + 1];\n"
                              "            a[j + 1] = t;\n        }\n    }\n}"),
        "D05": ("Q17", "D05", "int factorial(int n) {\n    return n * factorial(n - 1);\n}"),
        "D06": ("Q17", "D06", "int factorial(int n) {\n    if (n <= 1) {\n        return 1;\n    }\n"
                              "    return n * factorial(n);\n}"),
        "D06_grow": ("Q17", "D06", "int factorial(int n) {\n    if (n <= 1) {\n        return 1;\n    }\n"
                                   "    return n * factorial(n + 1);\n}"),
        "D07": ("Q17", "D07", "int factorial(int n) {\n    if (n <= 1) {\n        return 1;\n    }\n"
                              "    factorial(n - 1);\n    return n;\n}"),
        "D08": ("Q15", "D08", v("Q15").replace("int n = 0;", "int n = 0;\n    if (s == \"aba\") return 1;")),
    }


@pytest.fixture(scope="module")
def packages():
    out = {}
    for name, (pid, cls, code) in _mutants().items():
        out[name] = (iv.build_package(_problem(pid), code, cls), cls)
    return out


@pytest.mark.parametrize("name", list(_mutants()))
def test_every_hand_mutant_gives_a_valid_package(packages, name):
    pk, cls = packages[name]
    _check_package(pk, cls)
    assert pk["fix"] is not None
    assert pk["modality"] == iv.policy(cls, "if" if name == "M07_if" else None)[0]


def test_m08_package_has_the_shape_of_the_server_fixture(packages):
    pk, _ = packages["M08_le"]
    server = json.loads((ROOT / "server" / "fixtures" / "intervene.json").read_text(encoding="utf-8"))
    assert {k for k in pk if k != "shown_test"} == set(server) - {"model_version", "latency_ms"}
    assert pk["timeline"] == server["timeline"]
    assert pk["memory_strip"] == server["memory_strip"]
    assert pk["next_modalities"] == server["next_modalities"]
    assert pk["question"]["answer"] == server["question"]["answer"] == 2
    assert pk["copy"] == server["copy"]
    assert pk["counterexample"]["effect_diff"] == "1 extra void read"
    assert pk["fix"]["kind"] == "minimal" and pk["fix"]["verified"] is True
    assert pk["fix"]["rule"] == "<= to <"
    # The fix is reprinted code; only the edited line is marked.
    assert len(pk["fix"]["changed_lines"]) == 1
    assert "i <= n" not in pk["fix"]["code"]
    # The question for the fix modality uses the line in the learner's own code.
    other = iv.build_package(_problem("P03"), _mutants()["M08_le"][2], "M08", tried=["memory_strip", "counterexample"])
    assert other["modality"] == "minimal_fix"
    assert other["question"]["answer"] == 3


def test_timeline_flags_by_class(packages):
    flags = lambda name: [r["flag"] for r in packages[name][0]["timeline"]]
    assert flags("M01_le")[-1] == "void_read"
    assert flags("M02")[:3] == [None, None, None] and flags("M02")[-1] == "extra"
    assert flags("M03")[0] is None and flags("M03")[1:] == ["reset", "reset"]
    assert flags("M05")[0] == "static"
    assert "overwrite" in flags("M06")
    assert "empty_body" in flags("M07_if")
    assert flags("D01") == ["early_return", "missing"]


def test_timeline_is_capped_head_and_tail(packages):
    rows = packages["M02"][0]["timeline"]
    assert len(rows) == tl.MAX_TIMELINE_ROWS
    steps = [r["step"] for r in rows]
    assert steps == sorted(steps)
    assert steps[-1] > 1000          # the tail is the end of the stuck run, not the start


def test_questions_have_answers_from_the_reference(packages):
    q = lambda name: packages[name][0]["question"]
    assert q("M01_le") == {"prompt": "At which value of i should the loop stop?", "answer": 3}
    assert q("M02")["answer"] == 3
    assert q("M03") == {"prompt": "What should total be when the loop is done?", "answer": 12}
    assert q("M06")["answer"] == 7
    assert q("M07_if")["answer"] == 3
    assert q("M04")["answer"] == 50.0
    assert q("M10")["answer"] == 3
    assert q("D01")["answer"] == 2
    assert q("D02")["answer"] == [0, 1]
    assert q("D03")["answer"] == [4, 3, 2, 1]
    assert q("D04")["answer"] == 3
    assert q("D05")["answer"] == 1 and q("D06")["answer"] == 1
    assert q("D07")["answer"] == 6
    assert q("D08")["answer"] == "No, it compares where the text lives"


def test_window_strip_flags_the_frozen_window(packages):
    rows = [r for r in packages["D02"][0]["window_strip"] if isinstance(r, dict)]
    assert len(rows) == tl.MAX_WINDOW_ROWS
    assert rows[0]["flag"] is None and rows[0]["cmp"] == ">"
    assert (rows[0]["low"], rows[0]["mid"], rows[0]["high"]) == (0, 2, 4)
    assert all(r["flag"] == "frozen" for r in rows[2:])
    assert packages["D02"][0]["window_strip"][-1].startswith("…+")


def test_call_stack_flags(packages):
    d05 = [f for f in packages["D05"][0]["call_stack"] if isinstance(f, dict)]
    assert [f["flag"] for f in d05].count("no_base") == 1
    assert d05[3]["flag"] == "no_base" and d05[3]["depth"] == 4        # the reference never goes past depth 3
    assert packages["D05"][0]["call_stack"][-1] == "…+88"
    d06 = [f for f in packages["D06"][0]["call_stack"] if isinstance(f, dict)]
    assert d06[0]["flag"] is None and all(f["flag"] == "same_arg" for f in d06[1:])
    grow = [f for f in packages["D06_grow"][0]["call_stack"] if isinstance(f, dict)]
    assert grow[1]["flag"] == "growing_arg"
    d07 = packages["D07"][0]["call_stack"]
    assert [f["flag"] for f in d07] == ["dropped_value", "dropped_value", None]
    assert [f["returned"] for f in d07] == [3, 2, 1]


def test_memory_strip_flags(packages):
    d03 = packages["D03"][0]["memory_strip"]
    assert d03["final"] != d03["expected"] == [4, 3, 2, 1]
    flags = [w["flag"] for w in d03["writes"]]
    assert "lost" in flags and "duplicate" in flags
    d04 = packages["D04"][0]["memory_strip"]
    assert d04["passes"] == {"yours": 1, "needed": 3}
    assert d04["writes"][-1]["flag"] == "one_pass_end"
    m08 = packages["M08_le"][0]["memory_strip"]
    assert m08["reads"] == [0, 1, 2, 3] and "writes" not in m08


def test_value_meters(packages):
    m04 = packages["M04"][0]["value_meter"]
    assert m04["kind"] == "division" and m04["exact"] == 50.0 and m04["yours"] == 0.0 and m04["lost"] == 50.0
    d08 = packages["D08"][0]["value_meter"]
    assert d08["kind"] == "char_vs_string" and d08["char"] == {"text": "'a'", "code": 97}


def test_m10_package_shows_printed_against_returned(packages):
    out = packages["M10"][0]["output"]
    assert out == {"printed": "3", "returned": 0, "intended": 3}


def test_m07_for_package_uses_the_loop_row_of_the_policy(packages):
    pk = packages["M07_for"][0]
    assert pk["next_modalities"] == ["counterexample", "minimal_fix"]


def test_m07_if_package_has_only_a_fix_after_the_timeline(packages):
    assert packages["M07_if"][0]["next_modalities"] == ["minimal_fix"]


def test_tried_modalities_are_skipped_in_a_package():
    problem = _problem("P03")
    code = _mutants()["M08_le"][2]
    pk = iv.build_package(problem, code, "M08", tried=["memory_strip"])
    assert pk["modality"] == "counterexample"
    assert pk["next_modalities"] == ["minimal_fix"]
    assert pk["question"]["prompt"].startswith("What should this return for")
    assert "memory_strip" in pk          # the class panel is always there
    pk = iv.build_package(problem, code, "M08", tried=["memory_strip", "counterexample", "minimal_fix"])
    assert pk["exhausted"] is True and pk["modality"] == "memory_strip"


def test_requested_panel_is_added_to_the_package():
    problem = _problem("P03")
    pk = iv.build_package(problem, _mutants()["M08_le"][2], "M08", modality="call_stack")
    assert pk["modality"] == "call_stack" and "call_stack" in pk and "memory_strip" in pk


def test_other_class_gets_a_generic_package():
    problem = _problem("P03")
    pk = iv.build_package(problem, _mutants()["M01_le"][2], "OTHER")
    _check_package(pk, "OTHER")
    assert pk["modality"] == "trace_timeline" and pk["next_modalities"] == ["minimal_fix"]
    assert pk["copy"] == iv.COPY_OTHER


def test_passing_code_still_builds_a_package():
    """A latent class on code that passes every test (03 §5.5): nothing differs, so the counterexample shows."""
    problem = _problem("P03")
    pk = iv.build_package(problem, _variant("P03"), "M01")
    _check_package(pk, "M01")
    assert pk["counterexample"] is None
    assert pk["shown_test"]["source"] == "none"
    assert pk["fix"]["kind"] == "reference"


def test_passes_by_luck_uses_the_counterexample_input():
    """Every test passes but a counterexample is known: the timeline is traced on it."""
    problem = _problem("P03")
    code = _mutants()["M03"][2]
    # Keep only the test the reset bug passes by luck on.
    lucky = {**problem, "tests": [t for t in problem["tests"] if t["args"][0] == [0, 0, 9]], "display_test": 0}
    cex = {"input": {"cells": [2, 4], "n": 2}, "intended": 6, "yours": 4, "effect_diff": "same effects"}
    ctx = tl.make_context(lucky, code, cex)
    assert ctx.source == "counterexample" and ctx.args == [[2, 4], 2]
    rows = tl.build_timeline(ctx)
    assert [r["flag"] for r in rows] == [None, "reset"]


def test_builders_survive_a_parse_error():
    problem = _problem("P03")
    pk = iv.build_package(problem, "int total_energy(int cells[], int n) { return ; ", "M01")
    json.dumps(pk)
    assert pk["timeline"] == [] or isinstance(pk["timeline"], list)


# ---------------------------------------------------------------- live traces: one mutant per operator

def _operators():
    found = []
    for op in main_ops():
        if not op.test_only and any(label in MISCONCEPTIONS for label in op.labels):
            found.append((op, main_sites, main_apply))
    for op in dsa_ops():
        if not op.test_only and any(label in MISCONCEPTIONS for label in op.labels):
            found.append((op, dsa_sites, dsa_apply))
    return found


def _all_problems():
    out = []
    for folder in ("main", "dsa"):
        for path in sorted((ROOT / "ml" / "problems" / folder).glob("*.json")):
            out.append(json.loads(path.read_text(encoding="utf-8")))
    return out


def _allowed(op, problem):
    return any(fnmatch.fnmatch(op.op_id, pattern) for pattern in problem.get("allowed_ops") or [])


def _label(op):
    for label in op.labels:
        if label in MISCONCEPTIONS:
            return label
    return op.labels[0]


def _mutant_cases():
    problems = _all_problems()
    cases = []
    for op, sites, apply in _operators():
        for problem in problems:
            if not _allowed(op, problem):
                continue
            found = None
            for code in problem["correct_variants"]:
                try:
                    where = sites(op, code)
                except Exception:
                    continue
                if where:
                    found = apply(op, code, where[0])
                    break
            if found:
                cases.append((op.op_id, _label(op), problem, found))
                break
    return cases


_CASES = _mutant_cases()


def test_there_are_operator_cases():
    assert len(_CASES) >= 30
    assert {cls for _, cls, _, _ in _CASES} >= {"M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08",
                                                "D01", "D02", "D03", "D05", "D06", "D07"}


@pytest.mark.parametrize("op_id,cls,problem,code", _CASES, ids=[c[0] for c in _CASES])
def test_every_operator_mutant_gives_a_valid_package(op_id, cls, problem, code):
    pk = iv.build_package(problem, code, cls)
    _check_package(pk, cls)
    assert pk["fix"] is not None
    assert pk["timeline"], "a mutant that runs has a timeline"
    assert pk["shown_test"]["index"] >= 0
    if cls != "M07":
        assert pk["modality"] == iv.policy(cls)[0]
