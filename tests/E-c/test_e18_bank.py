"""E18 (item-bank soundness), package E-c.

Run from the repo root:  .venv\\Scripts\\python -m pytest tests/E-c
"""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml.eval import e18_bank as e18

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def quiz():
    # interpreter only: keeps the test fast; the card itself is made with both backends
    return e18.check_quiz(("interp",))


def test_quiz_bank_passes_the_verifier_on_the_interpreter(quiz):
    assert quiz["verify_passed"], quiz["problems"][:5]
    assert quiz["items"] == len(json.loads((ROOT / "ml" / "data" / "quiz_items.json").read_text(encoding="utf-8")))
    assert quiz["items_with_a_problem"] == 0
    assert quiz["pct_verified_interpreter"] == 100.0


def test_targets_and_collisions(quiz):
    assert quiz["per_class_type_short"] == {}
    assert quiz["collision_items"] >= quiz["collision_target"]
    assert len(quiz["per_class_type"]) == 17


def test_position_test_accounts_for_three_and_four_option_items(quiz):
    d = quiz["descriptive"]
    assert sum(d["correct_option_position"].values()) == d["choice_items"]
    assert d["correct_option_position_chi2_p"] is None or 0.0 <= d["correct_option_position_chi2_p"] <= 1.0
    assert sum(d["options_per_item"].values()) == d["choice_items"]


def test_pct():
    assert e18.pct(1, 3) == 33.3
    assert e18.pct(0, 0) is None


def test_a_bank_with_a_wrong_answer_fails(tmp_path, monkeypatch):
    items, checks = e18.vq.load()
    broken = [dict(i) for i in items]
    target = next(i for i in broken if i["type"] == "predict_output")
    wrong = next(o for o in target["options"] if o != target["correct"] and o not in target["belief"].values())
    target["correct"] = wrong
    monkeypatch.setattr(e18.vq, "load", lambda folder=None: (broken, checks))
    result = e18.check_quiz(("interp",))
    assert not result["verify_passed"]
    assert any(target["item_id"] in line for line in result["problems"])


def test_code_items_reported_as_not_built_when_files_are_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(e18, "CODE_ITEMS", tmp_path / "code_items.json")
    monkeypatch.setattr(e18.importlib.util, "find_spec", lambda name: None)
    out = e18.check_code_items()
    assert out["status"] == "not built" and out["checked"] == 0
    assert "C4" in out["why"]


def test_code_items_checked_when_present(monkeypatch, tmp_path):
    path = tmp_path / "code_items.json"
    path.write_text(json.dumps([
        {"item_id": "fb_1", "type": "fix_bug", "planted": "M05", "problem_id": "P03"},
        {"item_id": "dl_1", "type": "debug_line", "planted": "M03", "problem_id": "P03"},
        {"item_id": "dl_2", "type": "debug_line", "planted": None, "problem_id": "P03"},
        {"item_id": "cs_1", "type": "complete_snippet", "problem_id": "P03"},
    ]), encoding="utf-8")
    monkeypatch.setattr(e18, "CODE_ITEMS", path)
    monkeypatch.setattr(e18.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(e18.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout="items complete_snippet=1 fix_bug=1 debug_line=2\nCHECK PASSED: 4 items\n", stderr=""))
    out = e18.check_code_items()
    assert out["status"] == "checked" and out["passed"] and out["checked"] == 4
    assert out["check_line"] == "CHECK PASSED: 4 items"
    assert out["debug_line_no_bug_share"] == 0.5
    assert out["planted_per_class_and_type"]["M05"]["fix_bug"] == 1
    assert out["versus_targets"]["complete_snippet"]["met"] is True
    assert out["counts_meet_targets"] is False


def test_card_on_disk_matches_the_bank():
    path = ROOT / "ml" / "eval" / "e18_card.json"
    if not path.exists():
        pytest.skip("run python -m ml.eval.e18_bank first")
    card = json.loads(path.read_text(encoding="utf-8"))
    items = json.loads((ROOT / "ml" / "data" / "quiz_items.json").read_text(encoding="utf-8"))
    assert card["id"] == "E18"
    assert card["n"] == len(items) == card["quiz"]["items"] == 115
    assert card["quiz"]["seconds"] == 27.7
    assert card["metrics"]["pct_verified_gcc"] == 90.4
    assert card["metrics"]["verify_passed"] == (not card["quiz"]["problems"])
    assert "Nobody has skimmed" in card["caveat"] or "skimmed" in card["caveat"]
    code = card["code_items"]
    assert code["status"] == "checked" and code["passed"] is True
    assert code["check_line"] == "CHECK PASSED: 143 items"
    assert code["checked"] == 143
    assert code["by_type"] == {"complete_snippet": 67, "fix_bug": 34, "debug_line": 42}
    assert code["counts_meet_targets"] is True
    assert code["versus_targets"]["debug_line"]["no_bug"] == 8
    assert code["versus_targets"]["complete_snippet"]["with_1"] == ["P13"]
    md = (ROOT / "ml" / "eval" / "e18_card.md").read_text(encoding="utf-8")
    assert "CHECK PASSED: 143 items" in md
    assert "All three 05 §3.2 targets met." in md
    assert "could not run" not in md
    assert "**Not built.**" not in md
