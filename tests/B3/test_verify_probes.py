"""Package B3: the probe, trap and exam-item banks and their verifier."""
import copy

import pytest

from ml.bayes import verify_probes as V
from ml.contracts.classes import MISCONCEPTIONS, SECTORS, TWIN_SETS
from ml.oracle import gcc

try:
    gcc.gcc_path()
    HAVE_GCC = True
except gcc.GccUnavailable:
    HAVE_GCC = False

BANKS, CHECKS = V.load()


def changed(file, item_id, **fields):
    banks = copy.deepcopy(BANKS)
    key = V.BANKS[file][1]
    next(item for item in banks[file] if item[key] == item_id).update(fields)
    return banks


def test_counts_match_the_plan():
    assert len(BANKS["probes.json"]) == 14
    assert [t["target"] for t in BANKS["items.json"]] == MISCONCEPTIONS
    per_sector = {s: sum(1 for i in BANKS["exam_items.json"] if i["sector"] == s) for s in SECTORS}
    assert all(n >= 5 for n in per_sector.values()), per_sector
    assert 22 <= len(BANKS["exam_items.json"]) <= 30


def test_structure_is_sound():
    assert V.structure_problems(BANKS, CHECKS) == []


def test_every_twin_set_has_two_probes_that_split_it():
    assert "T1" in V.core_twin_sets() and len(V.core_twin_sets()) == 8
    for set_id in V.core_twin_sets():
        a, b = TWIN_SETS[set_id]["members"]
        probes = [p for p in BANKS["probes.json"] if set_id in p["twin_sets"]]
        assert len(probes) >= 2, set_id
        assert all(p["belief"][a] != p["belief"][b] for p in probes), set_id


def test_every_class_has_an_exam_item_that_exposes_it():
    covered = {cls for item in BANKS["exam_items.json"] for cls in item["belief"]}
    assert covered == set(MISCONCEPTIONS)


def test_every_answer_holds_on_the_interpreter():
    problems, summary = V.run_problems(BANKS, CHECKS, backends=("interp",))
    assert problems == []
    assert summary["run"] == summary["items"] - len(summary["manual"])
    assert [item_id for item_id, _ in summary["manual"]] == ["P_T1_a"]
    assert summary["belief_answers_run"] >= 50


@pytest.mark.skipif(not HAVE_GCC, reason="gcc is not installed")
def test_every_answer_holds_on_gcc_where_c_defines_it():
    problems, summary = V.run_problems(BANKS, CHECKS)
    assert problems == []
    assert summary["on_gcc"] >= 95 and summary["interp_only"] >= 5


def test_a_wrong_correct_answer_is_caught():
    banks = changed("probes.json", "P_T1_b", correct="3")
    problems, _ = V.run_problems(banks, CHECKS, backends=("interp",))
    assert any("P_T1_b: correct" in line and "'4'" in line for line in problems)


def test_a_wrong_belief_answer_is_caught():
    banks = changed("items.json", "trap_D07", belief={"D07": "1"})
    problems, _ = V.run_problems(banks, CHECKS, backends=("interp",))
    assert any("trap_D07: belief of D07" in line and "'24'" in line for line in problems)


@pytest.mark.parametrize("file, item_id, fields, expect", [
    ("probes.json", "P_T1_b", {"options": ["3", "4", "4"]}, "different options"),
    ("probes.json", "P_T1_b", {"correct": "9"}, "is not an option"),
    ("probes.json", "P_T1_b", {"belief": {"M01": "4", "M08": "4"}}, "cannot split T1"),
    ("probes.json", "P_T1_b", {"belief": {"M01": "3"}}, "both members of T1"),
    ("items.json", "trap_M01", {"target": "M02", "belief": {"M02": "4"}}, "traps: 0 for M01"),
    ("items.json", "trap_M01", {"belief": {"M01": "5"}}, "that is wrong"),
    ("exam_items.json", "xt_rec_1", {"sector": "graphs"}, "unknown sector"),
    ("exam_items.json", "xt_rec_1", {"belief": {}}, "1 to 3 classes"),
    ("exam_items.json", "xt_rec_1", {"belief": {"M99": "3"}}, "unknown class"),
    ("exam_items.json", "xt_rec_1", {"extra": 1}, "does not fit ExamTraceItem"),
])
def test_structure_rules_catch(file, item_id, fields, expect):
    problems = V.structure_problems(changed(file, item_id, **fields), CHECKS)
    assert any(expect in line for line in problems), problems


def test_exam_items_may_not_reuse_trap_code():
    trap = next(t for t in BANKS["items.json"] if t["target"] == "D05")
    problems = V.structure_problems(changed("exam_items.json", "xt_rec_1", code=trap["code"]), CHECKS)
    assert any("same code as trap_D05" in line for line in problems)


def test_an_item_without_a_check_is_a_problem():
    checks = {k: v for k, v in CHECKS.items() if k != "trap_M04"}
    assert any("trap_M04: no entry" in line for line in V.structure_problems(BANKS, checks))
