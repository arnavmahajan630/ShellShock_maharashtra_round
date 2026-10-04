"""Package E-a: helpers, rewrites and the cards that the experiments write.

The helper tests are fast. The card tests need the experiments to have been run first:
    .venv\\Scripts\\python -m ml.eval._ea_common
    .venv\\Scripts\\python -m ml.eval.e01_grouped_cv   (and e02 e04 e07 e08 e09 e11 e12 e13)
"""
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

from ml.contracts.classes import LABELS

ROOT = Path(__file__).resolve().parents[2]
CARDS = ROOT / "ml" / "eval" / "cards"
EXPECTED = {"E1": "E01", "E2": "E02", "E4": "E04", "E7": "E07", "E8": "E08", "E9": "E09",
            "E11": "E11", "E12": "E12", "E13": "E13"}


# ---------------------------------------------------------------- helpers

def test_macro_f1_counts_either_label_of_a_soft_row():
    from ml.eval._ea_common import macro_f1, truth_vector
    from ml.model.train import macro_f1 as m1_macro_f1
    i1, i8, ic = LABELS.index("M01"), LABELS.index("M08"), LABELS.index("CORRECT")
    Y = np.zeros((3, len(LABELS)))
    Y[0, [i1, i8]] = 0.5          # a T1 twin row
    Y[1, ic] = 1
    Y[2, ic] = 1
    y = np.array([i1, ic, ic])
    for pred in ([i1, ic, ic], [i8, ic, ic], [ic, ic, ic]):
        pred = np.array(pred)
        assert macro_f1(Y, y, pred) == pytest.approx(m1_macro_f1(Y, y, pred))   # same rule as the trainer
    _, hit = truth_vector(Y, y, np.array([i8, ic, ic]))
    assert hit.all()                                                           # naming M08 is right for a T1 row
    assert macro_f1(Y, y, np.array([i1, ic, ic])) == pytest.approx(1.0)
    assert macro_f1(Y, y, np.array([ic, ic, ic])) < 1.0


def test_cohen_kappa_extremes():
    from ml.eval._ea_common import cohen_kappa
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == pytest.approx(1.0)
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)


def test_bootstrap_interval_brackets_the_point_estimate():
    from ml.eval._ea_common import bootstrap_ci
    x = np.array([1.0] * 30 + [0.0] * 20)
    lo, hi = bootstrap_ci(lambda i: float(x[i].mean()), len(x), n_boot=300)
    assert lo < 0.6 < hi
    groups = np.repeat(np.arange(10), 5)
    lo, hi = bootstrap_ci(lambda i: float(x[i].mean()), len(x), groups, n_boot=300)
    assert lo <= hi


# ---------------------------------------------------------------- rewrites

SAMPLE = ("int total_energy(int cells[], int n) {\n"
          "    // sum\n    int total = 0;\n"
          "    for (int i = 0; i < n; i++) {\n        total += cells[i];\n    }\n    return total;\n}")


def _passes(code):
    from ml import runner
    problem = json.loads((ROOT / "ml" / "problems" / "main" / "P03_total_energy.json").read_text(encoding="utf-8"))
    res = runner.run_tests(problem, code)
    return tuple(t["pass"] for t in res["tests"]["results"])


@pytest.mark.parametrize("kind", ["rename", "reformat", "comments", "dead_variable", "for_while", "flip_compare", "incr_form", "combined"])
def test_rewrite_keeps_behaviour(kind):
    from ml.eval import _ea_perturb as PT
    fn = dict(PT.KINDS, **PT.STRESS, combined=PT.combined)[kind]
    new = PT.safe(fn, SAMPLE)
    assert new is not None and new != SAMPLE
    assert _passes(new) == _passes(SAMPLE)


def test_for_while_really_changes_the_loop():
    from ml.eval import _ea_perturb as PT
    out = PT.for_while(SAMPLE)
    assert "while" in out and "for (" not in out


def test_dead_variable_ignores_a_brace_in_a_comment():
    from ml.eval import _ea_perturb as PT
    code = "// helper { not code\nint f(int n) {\n    return n;\n}\n"
    out = PT.dead_variable(code)
    assert out.index("unused_zz") > out.index("int f(int n) {")


def test_rename_leaves_function_names_alone():
    from ml.eval import _ea_perturb as PT
    out = PT.rename(SAMPLE)
    assert "total_energy" in out and "cells" not in out


# ---------------------------------------------------------------- cards

def _card(eid):
    path = CARDS / f"{EXPECTED[eid]}.json"
    assert path.exists(), f"{path} missing: run the experiment"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("eid", list(EXPECTED))
def test_card_is_written_and_has_the_shape_of_03_9_4(eid):
    card = _card(eid)
    assert card["id"] == eid
    assert card.get("title") and ("caveat" in card)
    assert "n" in card


@pytest.mark.parametrize("eid", ["E4", "E8", "E9", "E11", "E12"])
def test_cards_that_use_r_say_llm_stand_in_and_r_blind_missing(eid):
    text = json.dumps(_card(eid))
    assert "LLM" in text
    assert re.search(r"R-blind", text) and re.search(r"missing|still open|no file", text)
    assert not re.search(r"hand-written (R|set)\b(?! by)", text) or "NOT hand-written" in text


def test_e4_does_not_call_the_llm_set_hand_written():
    card = _card("E4")
    assert card["data_status"]["R-blind"]["status"] == "missing"
    assert "NOT hand-written" in card["caveat"] or "not hand-written" in card["caveat"]


def test_e8_has_every_baseline_row_and_a_deepseek_status():
    card = _card("E8")
    for k in ("majority", "rules", "tfidf_lr", "tfidf_lgbm", "diagnoser", "deepseek_zero_shot"):
        assert k in card["table"]
    ds = card["table"]["deepseek_zero_shot"]
    assert ds["E4"] is not None or "not run" in ds.get("status", "")
    if card["deepseek"].get("status") == "ran":
        assert card["deepseek"]["calls"] <= 52 and card["deepseek"]["failed"] == 0


def test_e12_every_row_has_a_why():
    card = _card("E12")
    assert card["table"], "no audited rows"
    for row in card["table"]:
        assert row["why"] and row["why_source"]
        assert row["conf"] >= 0.6 or row["conf_T1"] >= 0.6
        assert row["true"] != row["pred"]


def test_e13_says_labels_are_unverified():
    card = _card("E13")
    assert "UNVERIFIED" in card["status"] or card.get("status") == "not run"


def test_no_api_key_in_any_card_or_note():
    env = ROOT / ".env"
    keys = []
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("DEEPSEEK_API_KEY=") and len(line.split("=", 1)[1].strip()) > 8:
                keys.append(line.split("=", 1)[1].strip())
    files = list(CARDS.glob("*.json")) + [ROOT / "notes" / "E-a.md"]
    for f in files:
        if f.exists():
            body = f.read_text(encoding="utf-8")
            assert not any(k in body for k in keys), f
            assert "Bearer " not in body
