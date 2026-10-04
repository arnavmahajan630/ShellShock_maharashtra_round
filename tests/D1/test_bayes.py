"""D1: likelihood table (03 §6.2), posterior (03 §6.1, 05 §4), probe choice (03 §6.4)."""
import math

import pytest

from ml.bayes import (
    apply_text, expected_information_gain, likelihood, load_probes, make_best_probe,
    map_free_answer, prediction_options, prior, select_probe, update,
)
from ml.contracts import schemas as S
from ml.contracts.classes import LABELS
from ml.contracts.params import (
    EIG_MIN_BITS, MAX_PROBES, P_BELIEF, P_OTHER_CORRECT, POPULATION_PRIOR, POSTERIOR_STOP,
    PRIOR_FLOOR, PRIOR_GAMMA, PRIOR_NEUTRAL, PROB_FLOOR, Q_CORRECT, SLIP, TEXT_BETA, TEXT_EPSILON,
)
from ml.model.decide import decide

OPTIONS = ["4", "5", "depends"]
CORRECT = "4"
BELIEF = {"M08": "5", "M01": "4"}          # M01's belief is the right answer: not diagnostic


def uniform():
    return {label: 1.0 for label in LABELS}


def split(a, b, share=0.42, third="M04", third_share=0.04):
    """Model distribution: most of the mass on two classes, every label still positive."""
    rest_labels = [label for label in LABELS if label not in (a, b, third)]
    rest = (1.0 - 2 * share - third_share) / len(rest_labels)
    assert rest > 0
    out = {label: rest for label in LABELS}
    out[a] = share
    out[b] = share
    out[third] = third_share
    return out


def bank():
    return {probe["probe_id"]: probe for probe in load_probes()}


def apply_probe(posterior, probe, answer):
    return update(posterior, answer, options=probe["options"], correct=probe["correct"],
                  belief=probe["belief"])


def after_probes(p0, answers):
    """§6.1 order: learner prior (no history), then each probe answer."""
    post = prior(p0)
    probes = bank()
    for probe_id, answer in answers:
        post = apply_probe(post, probes[probe_id], answer)
    return post


# ---------------------------------------------------------------- likelihood (03 §6.2)

def test_every_row_sums_to_one_on_a_probe_and_on_a_free_prediction():
    table = likelihood(OPTIONS, CORRECT, BELIEF)
    assert set(table) == set(LABELS)
    for row in table.values():
        assert sum(row.values()) == pytest.approx(1.0)
        assert set(row) == set(OPTIONS)

    free = likelihood(None, "4", {"M01": "3", "M08": "3"})
    assert list(free["M01"]) == ["4", "3", "other"]
    for row in free.values():
        assert sum(row.values()) == pytest.approx(1.0)


def test_diagnostic_correct_and_other_use_the_three_rows():
    table = likelihood(OPTIONS, CORRECT, BELIEF)
    diagnostic = table["M08"]
    assert diagnostic["5"] == pytest.approx(P_BELIEF)
    assert diagnostic["4"] == pytest.approx((1.0 - P_BELIEF) * Q_CORRECT)
    remainder = 1.0 - P_BELIEF - (1.0 - P_BELIEF) * Q_CORRECT
    assert diagnostic["depends"] == pytest.approx(remainder)

    # Belief defined but equal to the correct answer, or missing entirely: not diagnostic.
    for label in ("M01", "M02", "CORRECT"):
        assert table[label]["4"] == pytest.approx(1.0 - SLIP)
        assert table[label]["5"] == pytest.approx(SLIP / 2)

    assert table["OTHER"]["4"] == pytest.approx(P_OTHER_CORRECT)
    assert table["OTHER"]["5"] == pytest.approx((1.0 - P_OTHER_CORRECT) / 2)
    assert table["M08"] != table["CORRECT"] != table["OTHER"]


def test_probabilities_below_the_floor_are_lifted_then_renormalised():
    options = [str(i) for i in range(10)]
    table = likelihood(options, "0", {})
    raw_other = SLIP / 9
    assert raw_other < PROB_FLOOR
    total = (1.0 - SLIP) + 9 * PROB_FLOOR
    assert table["CORRECT"]["0"] == pytest.approx((1.0 - SLIP) / total)
    assert table["CORRECT"]["1"] == pytest.approx(PROB_FLOOR / total)
    assert table["CORRECT"]["1"] > raw_other
    assert sum(table["CORRECT"].values()) == pytest.approx(1.0)


def test_the_same_table_scores_a_probe_a_predict_item_and_an_exam_item():
    """03 §6.5 and §6.6 are this function, not a second table."""
    shared = {"options": ["3", "4", "2"], "correct": "4", "belief": {"M01": "3", "M08": "3"}}
    probe = likelihood(shared["options"], shared["correct"], shared["belief"])
    predict_item = {"item_id": "PR_P03", "code": "int i;", "question": "How many cells are read?", **shared}
    exam_item = {"item_id": "xt_demo", "kind": "trace", "sector": "arrays", "code": "int i;",
                 "question": "How many cells are read?", **shared}
    assert likelihood(predict_item["options"], predict_item["correct"], predict_item["belief"]) == probe
    assert likelihood(exam_item["options"], exam_item["correct"], exam_item["belief"]) == probe
    assert probe["M01"]["3"] == pytest.approx(probe["M08"]["3"]) == pytest.approx(P_BELIEF)


def test_free_numeric_predictions_map_onto_correct_belief_and_other():
    belief = {"M08": "5", "M01": 5, "M02": "4"}
    assert prediction_options(4, belief) == ["4", "5", "other"]
    assert map_free_answer(5, 4, belief) == "5"
    assert map_free_answer(4, 4, belief) == "4"
    assert map_free_answer(9, 4, belief) == "other"

    post = update(uniform(), 9, correct=4, belief={"M01": "3", "M08": "3"})
    assert post["M01"] == pytest.approx(post["M08"])
    assert post["M01"] > post["M02"]          # both believe 3, so "other" is their remainder bucket
    assert sum(post.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------- posterior (03 §6.1, 05 §4)

def test_learner_prior_nudges_and_does_not_dominate():
    post = prior(uniform(), {"M01": 1.0, "M04": 0.0, "CORRECT": 0.01, "OTHER": 0.01})
    pop = POPULATION_PRIOR ** PRIOR_GAMMA
    assert PRIOR_GAMMA == pytest.approx(0.3)
    assert post["M01"] / post["M02"] == pytest.approx((PRIOR_NEUTRAL ** PRIOR_GAMMA) / pop)
    assert post["M04"] / post["M02"] == pytest.approx((PRIOR_FLOOR ** PRIOR_GAMMA) / pop)
    assert post["CORRECT"] / post["M02"] == pytest.approx(post["M01"] / post["M02"])
    assert post["M02"] < post["M01"] < 0.25
    assert sum(post.values()) == pytest.approx(1.0)
    # History does not move CORRECT or OTHER, and a missing misconception is the population prior.
    assert prior(uniform(), {"CORRECT": 0.01})["CORRECT"] == pytest.approx(prior(uniform())["CORRECT"])


def test_masked_classes_stay_at_zero():
    p0 = {label: 0.0 for label in LABELS}
    p0["M01"], p0["M03"] = 0.7, 0.3
    post = prior(p0, masked={"M03"})
    assert post["M03"] == 0.0
    assert post["M01"] == pytest.approx(1.0)
    assert sum(post.values()) == pytest.approx(1.0)

    answered = update(post, "5", options=OPTIONS, correct=CORRECT, belief={"M03": "5", "M08": "5"})
    assert answered["M03"] == 0.0
    spoken = apply_text(answered, {"M03": 0.99, "M01": 0.01})
    assert spoken["M03"] == 0.0
    assert sum(spoken.values()) == pytest.approx(1.0)


def test_responses_are_conditionally_independent():
    start = prior(split("M01", "M08"))
    probes = bank()
    first = apply_probe(apply_probe(start, probes["P_T1_a"], "5"), probes["P_T1_b"], "3")
    second = apply_probe(apply_probe(start, probes["P_T1_b"], "3"), probes["P_T1_a"], "5")
    for label in LABELS:
        assert first[label] == pytest.approx(second[label])


def test_text_update_uses_the_sentence_factor_and_is_skipped_when_unsure():
    post = {label: 0.0 for label in LABELS}
    post["M01"] = 0.5
    post["M08"] = 0.5
    assert apply_text(post, {"M01": 0.9}, unsure=True) == post

    changed = apply_text(post, {"M01": 0.8})
    up = (TEXT_EPSILON + 0.8) ** TEXT_BETA
    down = (TEXT_EPSILON + 0.0) ** TEXT_BETA
    assert changed["M01"] == pytest.approx((0.5 * up) / (0.5 * up + 0.5 * down))
    assert changed["M01"] > post["M01"]
    assert changed["M08"] < post["M08"]
    assert changed["M02"] == 0.0


# ---------------------------------------------------------------- hard twins (03 §6.3, §6.4)

def test_t1_belief_answers_push_that_class_past_the_stop():
    """M08 says 5 then 4; M01 says 4 then 3. Both probes are applied either way."""
    m08 = after_probes(split("M01", "M08"), [("P_T1_a", "5"), ("P_T1_b", "4")])
    m01 = after_probes(split("M01", "M08"), [("P_T1_a", "4"), ("P_T1_b", "3")])
    assert m08["M08"] >= POSTERIOR_STOP
    assert m08["M08"] > m08["M01"]
    assert m01["M01"] >= POSTERIOR_STOP
    assert m01["M01"] > m01["M08"]
    assert min(m08.values()) >= 0.0 and min(m01.values()) >= 0.0


def test_t7_belief_answers_push_that_class_past_the_stop():
    probes = bank()
    d05_answers = [(pid, probes[pid]["belief"]["D05"]) for pid in ("P_T7_a", "P_T7_b")]
    d06_answers = [(pid, probes[pid]["belief"]["D06"]) for pid in ("P_T7_a", "P_T7_b")]
    d05 = after_probes(split("D05", "D06"), d05_answers)
    d06 = after_probes(split("D05", "D06"), d06_answers)
    assert d05_answers == [("P_T7_a", "0"), ("P_T7_b", "never stops")]
    assert d06_answers == [("P_T7_a", "never stops"), ("P_T7_b", "1")]
    assert d05["D05"] >= POSTERIOR_STOP and d05["D05"] > d05["D06"]
    assert d06["D06"] >= POSTERIOR_STOP and d06["D06"] > d06["D05"]


# ---------------------------------------------------------------- probe selection (03 §6.4)

def test_information_gain_matches_the_definition():
    post = {label: 0.0 for label in LABELS}
    post["M01"] = 0.5
    post["M08"] = 0.5
    score = expected_information_gain(post, OPTIONS, CORRECT, BELIEF)

    m01 = {"4": 1.0 - SLIP, "5": SLIP / 2, "depends": SLIP / 2}
    m08_rest = 1.0 - P_BELIEF - (1.0 - P_BELIEF) * Q_CORRECT
    m08 = {"4": (1.0 - P_BELIEF) * Q_CORRECT, "5": P_BELIEF, "depends": m08_rest}

    def h(pairs):
        return -sum(p * math.log2(p) for p in pairs if p > 0.0)

    expected = h((0.5, 0.5))
    for option in OPTIONS:
        p_answer = 0.5 * m01[option] + 0.5 * m08[option]
        expected -= p_answer * h((0.5 * m01[option] / p_answer, 0.5 * m08[option] / p_answer))
    assert score == pytest.approx(expected)
    assert score >= EIG_MIN_BITS


def test_select_probe_asks_the_t1_probe_with_the_most_gain():
    post = prior(split("M01", "M08"))
    probes = load_probes()
    chosen, score = select_probe(post, [], probes)
    scores = {
        probe["probe_id"]: expected_information_gain(
            post, probe["options"], probe["correct"], probe["belief"])
        for probe in probes if probe["probe_id"] in ("P_T1_a", "P_T1_b")
    }
    assert chosen["probe_id"] == max(scores, key=scores.get)
    assert score == pytest.approx(scores[chosen["probe_id"]])
    assert score >= EIG_MIN_BITS
    assert set(chosen) == {"probe_id", "prompt", "code", "options"}
    S.PublicProbe.model_validate(chosen)


def test_select_probe_stops_once_the_top_class_passes_the_threshold():
    post = {label: (1.0 - POSTERIOR_STOP) / (len(LABELS) - 1) for label in LABELS}
    post["M01"] = POSTERIOR_STOP
    assert max(post.values()) >= POSTERIOR_STOP
    assert select_probe(post, [], load_probes()) is None

    asked = ["P_T1_a", "P_T7_a"]
    assert len(asked) >= MAX_PROBES
    assert select_probe(prior(split("M01", "M08")), asked) is None


def test_an_asked_probe_is_never_proposed_again():
    post = prior(split("M01", "M08"))
    only_a = [probe for probe in load_probes() if probe["probe_id"] == "P_T1_a"]
    assert select_probe(post, ["P_T1_a"], only_a) is None

    chosen, _score = select_probe(post, ["P_T1_a"])
    assert chosen["probe_id"] == "P_T1_b"

    # P_T2_a is one object that serves T2 and T6. Asking it removes it for both.
    t6 = prior(split("D02", "M02", third="M04"))
    again, _score = select_probe(t6, ["P_T2_a"])
    assert again["probe_id"] != "P_T2_a"


def test_fallback_is_only_used_when_nothing_touches_the_top_three():
    post = {label: 0.0 for label in LABELS}
    post["M01"], post["M08"], post["OTHER"] = 0.35, 0.35, 0.30
    separating = {
        "probe_id": "P_T1_a", "twin_sets": ["T1"], "prompt": "p", "code": "int a[5];",
        "options": OPTIONS, "correct": CORRECT, "belief": {},
    }
    # No belief key is in the top 3, so this probe is not a candidate. The rule table still
    # names P_T1_a, and OTHER vs the slip row carries enough bits to clear the minimum.
    chosen, score = select_probe(post, [], [separating])
    assert chosen["probe_id"] == "P_T1_a"
    assert score >= EIG_MIN_BITS
    assert "belief" not in chosen and "correct" not in chosen

    assert select_probe(post, ["P_T1_a"], [separating]) is None

    weak = {
        "probe_id": "P_WEAK", "twin_sets": ["T1"], "prompt": "p", "code": "int x;",
        "options": ["4"], "correct": "4", "belief": {"M01": "4"},
    }
    # The weak probe touches the top 3 and has no information. Do not fall back to P_T1_a.
    assert select_probe(post, [], [weak, separating]) is None


def test_make_best_probe_is_what_decide_calls():
    asked = []
    best = make_best_probe(asked)
    post = prior(split("M01", "M08"))
    diagnosis = decide(post, best_probe=best, probes_asked=asked)
    assert diagnosis["status"] == "ambiguous"
    probe = diagnosis["next_probe"]
    assert probe["probe_id"] in ("P_T1_a", "P_T1_b")
    assert "belief" not in probe and "correct" not in probe
    assert probe["eig_bits"] >= EIG_MIN_BITS
    S.PublicProbe.model_validate(probe)

    asked.append(probe["probe_id"])
    second = best(post)
    assert second[0]["probe_id"] != probe["probe_id"]


def test_probe_verifier_still_imports():
    from ml.bayes import verify_probes
    banks, _checks = verify_probes.load()
    assert len(banks["probes.json"]) == 14
