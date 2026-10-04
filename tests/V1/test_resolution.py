"""E10 resolution simulation. Learners in these tests are simulated, not real."""
import json

from ml.contracts.params import RELAPSE_TO_TREATING_P, STABLE_P
from ml.eval.e10_resolution import CARD_PATH, render_table
from ml.eval.sim_learners import (
    E10_SEED,
    MAX_FOLLOWUPS,
    enter_probation,
    run_episode,
    run_resolution,
)

CARD = CARD_PATH


def test_probation_starts_inside_the_open_band():
    p, state = enter_probation()
    assert state == "PROBATION"
    assert STABLE_P < p < RELAPSE_TO_TREATING_P


def test_ours_masters_a_simulated_learner_who_is_always_correct():
    # s_sim = 0 would be inactive-perfect, but forced answers skip the draw.
    out = run_episode(
        None, "truly_fixed", "ours",
        params={"g_sim": {}, "s_sim": {}, "p_b_sim": 0.6},
        flipped=False,
        forced=[True, True, True],
    )
    assert out["simulated"] is True
    assert out["final_state"] == "MASTERED"
    assert out["declared_stable"] is True
    assert out["truly_active"] is False
    assert out["items"] == 3


def test_naive_resolves_a_simulated_pattern_copier_on_the_first_correct():
    out = run_episode(
        None, "pattern_copier", "naive",
        params={"g_sim": {}, "s_sim": {}, "p_b_sim": 0.6},
        flipped=False,
        forced=[True],
    )
    assert out["declared_stable"] is True
    assert out["truly_active"] is True
    assert out["items"] == 1


def test_budget_never_exceeds_four_followups_plus_a_ghost():
    import numpy as np
    rng = np.random.default_rng(0)
    for learner_type in ("truly_fixed", "pattern_copier", "lucky_guesser", "forgetful", "slow"):
        for policy in ("naive", "4-check", "ours"):
            out = run_episode(rng, learner_type, policy)
            assert out["items"] <= MAX_FOLLOWUPS + 1
            assert out["simulated"] is True


def test_pattern_copier_false_resolve_ours_at_most_half_naive():
    card = run_resolution(n_learners=250, n_seeds=4, seed=E10_SEED)
    ours = next(
        row for row in card["rows"]
        if row["type"] == "pattern_copier" and row["policy"] == "ours"
    )
    naive = next(
        row for row in card["rows"]
        if row["type"] == "pattern_copier" and row["policy"] == "naive"
    )
    assert ours["false_resolve"]["mean"] <= 0.5 * naive["false_resolve"]["mean"]
    assert card["population"] == "simulated"
    assert "simulated" in card["caveat"].lower()


def test_repeat_run_matches():
    a = run_resolution(n_learners=30, n_seeds=2, seed=7)
    b = run_resolution(n_learners=30, n_seeds=2, seed=7)
    assert a["rows"] == b["rows"]


def test_table_names_simulated_learners():
    card = run_resolution(n_learners=10, n_seeds=2, seed=1)
    text = render_table(card)
    assert "simulated" in text.lower()
    assert "not real" in text.lower()
    assert "| naive | truly_fixed (simulated) |" in text


def test_shipped_e10_card_has_intervals():
    assert CARD.is_file(), "run python -m ml.eval.e10_resolution to write the card"
    card = json.loads(CARD.read_text(encoding="utf-8"))
    assert card["id"] == "E10"
    assert card["population"] == "simulated"
    assert card["n_seeds"] == 20
    assert card["learners_per_type_per_seed"] == 2000
    assert "simulated" in card["caveat"].lower()
    assert card["acceptance"]["met"] is True
    seen = set()
    for row in card["rows"]:
        assert row["simulated"] is True
        seen.add((row["policy"], row["type"]))
        for key in ("false_resolve", "false_not_yet", "items_to_decision"):
            stat = row[key]
            if stat is None:
                continue
            assert stat["lo"] <= stat["mean"] <= stat["hi"]
            assert stat["n_seeds"] == 20
    assert len(seen) == 15
    text = (CARD.parent / "e10_table.md").read_text(encoding="utf-8")
    assert "simulated" in text.lower()
    assert "—" in text
