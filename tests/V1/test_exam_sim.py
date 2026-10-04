"""E14 exam simulation. Learners in these tests are simulated, not real."""
import json

import numpy as np
import pytest

from ml.eval._v1_confusion import e1_oof_confusion, stand_in_confusion
from ml.eval.e14_exam import CARD_PATH, render_table
from ml.eval.sim_exam import (
    N_ITEMS,
    POLICIES,
    draw_response,
    draw_truth,
    run_exam_episode,
    run_exam_sim,
)
from ml.exam.select import fixed_blueprint, load_pool

CARD = CARD_PATH

_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        _POOL = load_pool(strings=True)
    return _POOL


def test_confusion_rows_sum_to_one_and_prefer_the_true_class():
    matrix = stand_in_confusion()
    assert matrix.shape == (17, 17)
    for i, row in enumerate(matrix):
        assert row.sum() == pytest.approx(1.0)
        assert int(np.argmax(row)) == i


def test_e1_confusion_is_the_nineteen_label_matrix():
    matrix = e1_oof_confusion()
    assert matrix.shape == (19, 19)
    for row in matrix:
        assert row.sum() == pytest.approx(1.0)
        assert (row >= 0).all()
    stand = stand_in_confusion()
    assert stand.shape == (17, 17)
    assert not np.allclose(matrix[:17, :17], stand)


def test_stand_in_flag_is_not_the_default():
    pool = _pool()
    card = run_exam_sim(
        n_learners=4, n_seeds=1, seed=9, pool=pool, confusion="stand-in",
    )
    assert card["diagnoser_noise"]["source"].startswith("stand-in")
    assert card["population"] == "simulated"


def test_one_simulated_exam_stays_inside_five_plus_five():
    pool = _pool()
    by_id = {item["item_id"]: item for item in pool}
    blueprint = [by_id[item_id] for item_id in fixed_blueprint(strings=True)]
    matrix = stand_in_confusion()
    rng = np.random.default_rng(0)
    truth = draw_truth(rng)
    params = draw_response(rng)
    for policy in POLICIES:
        out = run_exam_episode(rng, truth, params, policy, pool, blueprint, matrix, 0)
        assert out["simulated"] is True
        assert len(out["curve"]) == N_ITEMS
        assert out["n_administered"] == N_ITEMS
        assert 0.0 <= out["final_f1"] <= 1.0
        assert out["brier"] >= 0.0


def test_fixed_blueprint_is_ten_items():
    assert len(fixed_blueprint(strings=True)) == 10


def test_repeat_exam_sim_matches():
    pool = _pool()
    matrix = stand_in_confusion()
    kwargs = dict(n_learners=8, n_seeds=2, seed=3, pool=pool, matrix=matrix)
    a = run_exam_sim(**kwargs)
    b = run_exam_sim(**kwargs)
    assert a["rows"] == b["rows"]
    assert a["population"] == "simulated"


def test_table_names_simulated_learners():
    pool = _pool()
    card = run_exam_sim(n_learners=6, n_seeds=2, seed=5, pool=pool, matrix=stand_in_confusion())
    text = render_table(card)
    assert "simulated" in text.lower()
    assert "not real" in text.lower()
    assert "| adaptive (simulated) |" in text
    assert len(card["curves"]["adaptive"]) == 10


def test_shipped_e14_card_has_intervals():
    assert CARD.is_file(), "run python -m ml.eval.e14_exam to write the card"
    card = json.loads(CARD.read_text(encoding="utf-8"))
    assert card["id"] == "E14"
    assert card["population"] == "simulated"
    assert card["n_seeds"] == 20
    assert card["learners_per_seed"] == 2000
    assert "simulated" in card["caveat"].lower()
    assert card["diagnoser_noise"]["source"].startswith("E1")
    assert "simulated" in card["stand_in_comparison"]["population"]
    assert set(card["stand_in_comparison"]["profile_f1"]) == {"adaptive", "fixed", "random"}
    policies = []
    for row in card["rows"]:
        assert row["simulated"] is True
        policies.append(row["policy"])
        for key in ("profile_f1", "brier", "stable_recheck", "items_to_first_finding"):
            stat = row[key]
            assert stat is not None
            assert stat["lo"] <= stat["mean"] <= stat["hi"]
            assert stat["n_seeds"] == 20
            if key != "items_to_first_finding":
                assert 0.0 <= stat["mean"] <= 1.0
    assert policies == ["adaptive", "fixed", "random"]
    for policy in policies:
        assert len(card["curves"][policy]) == 10
        for point in card["curves"][policy]:
            stat = point["profile_f1"]
            assert stat["lo"] <= stat["mean"] <= stat["hi"]
            assert point["simulated"] is True
    text = (CARD.parent / "e14_table.md").read_text(encoding="utf-8")
    assert "simulated" in text.lower()
