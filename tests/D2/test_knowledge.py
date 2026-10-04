"""D2: P(A) updates, one test per 03 §8.2 row and per 05 §4 knowledge row."""
import math

import pytest

from ml.contracts.params import (
    DIAGNOSIS_ACTIVE_P,
    FAIL_SIGNATURE_IF_ACTIVE,
    FAIL_SIGNATURE_IF_NOT,
    FORGET_RATE,
    HINT_GUESS_BONUS,
    HINT_GUESS_CAP,
    ITEM_GUESS_SLIP,
    LATENT_EXPOSURE_FACTOR,
    LEARN_RATE,
    P_BELIEF,
    Q_CORRECT,
    TEXT_IF_ACTIVE,
    TEXT_IF_NOT,
    TEXT_MATCH_P,
)
from ml.learner.knowledge import (
    ITEM_RESPONSE_KIND,
    forget,
    guess_with_hint,
    item_response,
    likelihood_ratio,
    resolve_kind,
    update_p,
)

P = 0.4


def ratio(p, if_active, if_not):
    return p * if_active / (p * if_active + (1.0 - p) * if_not)


def row_update(p, row, correct, hint=False):
    g, s = ITEM_GUESS_SLIP[row]
    if hint:
        g = min(HINT_GUESS_CAP, g + HINT_GUESS_BONUS)
    if correct:
        return ratio(p, g, 1.0 - s)
    return ratio(p, 1.0 - g, s)


# ---------------------------------------------------------------- 03 §8.2

def test_likelihood_ratio_is_shared_by_signature_and_text():
    signature = update_p(P, "signature_failure", exposure=0.7)
    text = update_p(P, "reasoning", matched=True, p_text=TEXT_MATCH_P)
    assert signature == pytest.approx(likelihood_ratio(P, 0.7, FAIL_SIGNATURE_IF_NOT))
    assert text == pytest.approx(likelihood_ratio(P, TEXT_IF_ACTIVE, TEXT_IF_NOT))
    assert signature == pytest.approx(ratio(P, 0.7, FAIL_SIGNATURE_IF_NOT))
    assert text == pytest.approx(ratio(P, TEXT_IF_ACTIVE, TEXT_IF_NOT))


def test_zero_denominator_leaves_p_unchanged():
    assert likelihood_ratio(1.0, 0.0, 0.0) == 1.0
    assert likelihood_ratio(0.4, 0.0, 0.0) == 0.4
    assert likelihood_ratio(0.0, 0.0, 0.0) == 0.0
    assert item_response(1.0, 0.0, 1.0, correct=True) == 1.0
    assert item_response(0.4, 1.0, 0.0, correct=False) == 0.4


def test_signature_failure_uses_authored_exposure():
    e = 0.7
    assert update_p(P, "signature_failure", exposure=e) == pytest.approx(
        ratio(P, e, FAIL_SIGNATURE_IF_NOT))
    assert update_p(P, "signature_failure", e_ik=e) == pytest.approx(
        ratio(P, e, FAIL_SIGNATURE_IF_NOT))


def test_signature_failure_defaults_when_the_problem_lists_none():
    expected = ratio(P, FAIL_SIGNATURE_IF_ACTIVE, FAIL_SIGNATURE_IF_NOT)
    assert update_p(P, "signature_failure") == pytest.approx(expected)
    assert update_p(P, "signature_failure", exposure=None) == pytest.approx(expected)
    assert update_p(P, "signature_failure", exposure=0.0) == pytest.approx(
        ratio(P, 0.0, FAIL_SIGNATURE_IF_NOT))


def test_signature_failure_is_not_max_with_a_posterior():
    """notes/W0.md decision 19: one ratio, no max(P, posterior) write."""
    p = 0.05
    updated = update_p(p, "signature_failure")
    posterior = 0.99
    assert updated == pytest.approx(ratio(p, FAIL_SIGNATURE_IF_ACTIVE, FAIL_SIGNATURE_IF_NOT))
    assert updated != pytest.approx(max(p, posterior))
    assert updated < posterior


def test_latent_pass_scales_exposure():
    e = 0.8
    assert update_p(P, "latent_pass", exposure=e) == pytest.approx(
        ratio(P, e * LATENT_EXPOSURE_FACTOR, FAIL_SIGNATURE_IF_NOT))
    assert update_p(P, "latent_pass") == pytest.approx(
        ratio(P, FAIL_SIGNATURE_IF_ACTIVE * LATENT_EXPOSURE_FACTOR, FAIL_SIGNATURE_IF_NOT))


@pytest.mark.parametrize("kind", ["same_family_code", "transfer_code"])
def test_pass_with_no_latent_uses_the_callers_row(kind):
    assert update_p(P, kind, correct=True) == pytest.approx(row_update(P, kind, True))


def test_intervention_multiplies_by_one_minus_learn_rate():
    p = 0.8
    assert update_p(p, "intervention") == pytest.approx(p * (1.0 - LEARN_RATE))


@pytest.mark.parametrize("kind", [
    "same_family_code", "transfer_code", "trap", "belief_mcq", "ghost", "exam_code",
])
@pytest.mark.parametrize("correct", [True, False])
def test_item_response_rows(kind, correct):
    assert ITEM_RESPONSE_KIND[kind] == kind
    assert update_p(P, kind, correct=correct) == pytest.approx(row_update(P, kind, correct))


def test_belief_mcq_uses_the_stored_guess_without_rederiving():
    g, s = ITEM_GUESS_SLIP["belief_mcq"]
    assert g == (1.0 - P_BELIEF) * Q_CORRECT
    assert update_p(P, "belief_mcq", correct=True) == pytest.approx(ratio(P, g, 1.0 - s))
    assert update_p(P, "exam_trace", correct=False) == pytest.approx(row_update(P, "belief_mcq", False))


def test_hint_raises_guess_for_that_item_only_and_caps_it():
    g, s = ITEM_GUESS_SLIP["trap"]
    bumped = min(HINT_GUESS_CAP, g + HINT_GUESS_BONUS)
    helped = update_p(P, "trap", correct=True, hint=True)
    plain = update_p(P, "trap", correct=True)
    assert helped == pytest.approx(ratio(P, bumped, 1.0 - s))
    assert plain == pytest.approx(row_update(P, "trap", True))
    assert helped != pytest.approx(plain)
    assert guess_with_hint(HINT_GUESS_CAP) == HINT_GUESS_CAP
    over = HINT_GUESS_CAP - HINT_GUESS_BONUS / 2.0
    assert guess_with_hint(over) == HINT_GUESS_CAP
    assert update_p(P, "trap", correct=True) == pytest.approx(plain)


def test_forgetting_compounds_once_per_level():
    p = 0.2
    once = p + FORGET_RATE * (1.0 - p)
    twice = once + FORGET_RATE * (1.0 - once)
    assert update_p(p, "forgetting", levels=1) == pytest.approx(once)
    assert update_p(p, "forgetting", levels=2) == pytest.approx(twice)
    assert forget(p, levels=0) == pytest.approx(p)
    assert twice == pytest.approx(update_p(p, "forgetting", event={"levels": 2}))


def test_correct_trap_lowers_p_and_a_wrong_trap_raises_it():
    assert update_p(P, "trap", correct=True) < P
    assert update_p(P, "trap", correct=False) > P


# ---------------------------------------------------------------- 05 §4 knowledge rows

@pytest.mark.parametrize("kind", ["mcq", "predict_output", "next_state"])
@pytest.mark.parametrize("correct", [True, False])
def test_quiz_items_use_the_belief_mcq_row(kind, correct):
    assert update_p(P, kind, correct=correct) == pytest.approx(row_update(P, "belief_mcq", correct))


def test_predict_output_whose_belief_contains_the_class_is_a_trap():
    belief = {"M08": "9"}
    assert resolve_kind("predict_output", class_id="M08", belief=belief) == "predict_output_trap"
    assert ITEM_RESPONSE_KIND["predict_output_trap"] == "trap"
    caught = update_p(P, "predict_output", class_id="M08", belief=belief, correct=False)
    assert caught == pytest.approx(update_p(P, "trap", correct=False))
    assert update_p(P, "predict_output_trap", correct=True) == pytest.approx(
        update_p(P, "trap", correct=True))


def test_predict_output_for_a_class_absent_from_belief_stays_belief_mcq():
    belief = {"M08": "9"}
    assert resolve_kind("predict_output", class_id="M01", belief=belief) == "predict_output"
    got = update_p(P, "predict_output", class_id="M01", belief=belief, correct=True)
    assert got == pytest.approx(update_p(P, "belief_mcq", correct=True))
    assert got != pytest.approx(update_p(P, "trap", correct=True))


def test_mcq_with_a_belief_map_does_not_become_a_trap():
    belief = {"M01": "4"}
    assert resolve_kind("mcq", class_id="M01", belief=belief) == "mcq"
    assert update_p(P, "mcq", class_id="M01", belief=belief, correct=False) == pytest.approx(
        update_p(P, "belief_mcq", correct=False))


def test_debug_line_correct_and_wrong_use_its_guess_and_slip():
    g, s = ITEM_GUESS_SLIP["debug_line"]
    assert update_p(P, "debug_line", planted="M03", correct=True) == pytest.approx(ratio(P, g, 1.0 - s))
    assert update_p(P, "debug_line", planted="M03", correct=False) == pytest.approx(ratio(P, 1.0 - g, s))


def test_debug_line_with_no_planted_bug_does_not_update():
    assert update_p(P, "debug_line", planted=None, correct=True) == pytest.approx(P)
    assert update_p(P, "debug_line", planted=None, correct=False) == pytest.approx(P)


def test_fix_bug_all_tests_pass_is_a_correct_item_response():
    g, s = ITEM_GUESS_SLIP["fix_bug"]
    assert update_p(P, "fix_bug", planted="M05", tests_pass=True) == pytest.approx(ratio(P, g, 1.0 - s))
    assert update_p(P, "fix_bug", planted="M05", class_id="M01", tests_pass=True) == pytest.approx(P)


def test_fix_bug_planted_bug_still_there_is_wrong_not_the_signature_rule():
    g, s = ITEM_GUESS_SLIP["fix_bug"]
    e = 0.7
    got = update_p(P, "fix_bug", planted="M05", top1="M05", tests_pass=False,
                   posterior=0.99, exposure=e)
    assert got == pytest.approx(ratio(P, 1.0 - g, s))
    assert got != pytest.approx(ratio(P, e, FAIL_SIGNATURE_IF_NOT))


def test_fix_bug_a_different_bug_updates_j_and_not_k():
    e = 0.6
    p_j = 0.25
    p_k = 0.55
    updated_j = update_p(p_j, "fix_bug", planted="M05", class_id="M01", top1="M01",
                         tests_pass=False, posterior=DIAGNOSIS_ACTIVE_P, exposure=e)
    assert updated_j == pytest.approx(ratio(p_j, e, FAIL_SIGNATURE_IF_NOT))
    assert update_p(p_k, "fix_bug", planted="M05", class_id="M05", top1="M01",
                    tests_pass=False, posterior=DIAGNOSIS_ACTIVE_P, exposure=e) == pytest.approx(p_k)
    below = math.nextafter(DIAGNOSIS_ACTIVE_P, 0.0)
    assert update_p(p_j, "fix_bug", planted="M05", class_id="M01", top1="M01",
                    tests_pass=False, posterior=below, exposure=e) == pytest.approx(p_j)


def test_complete_snippet_pass_uses_the_same_family_row():
    g, s = ITEM_GUESS_SLIP["same_family_code"]
    assert update_p(P, "complete_snippet", passed=True) == pytest.approx(ratio(P, g, 1.0 - s))
    assert update_p(P, "complete_snippet", passed=False, top1="M01",
                    posterior=DIAGNOSIS_ACTIVE_P) == pytest.approx(P)


@pytest.mark.parametrize("kind", ["reasoning", "why"])
def test_text_match_at_or_above_the_threshold_uses_the_text_ratio(kind):
    expected = ratio(P, TEXT_IF_ACTIVE, TEXT_IF_NOT)
    assert update_p(P, kind, matched=True, p_text=TEXT_MATCH_P) == pytest.approx(expected)
    assert update_p(P, kind, status="matched", p_text=TEXT_MATCH_P + 0.1) == pytest.approx(expected)


def test_text_below_the_threshold_or_unsure_or_unmatched_does_not_update():
    below = math.nextafter(TEXT_MATCH_P, 0.0)
    assert update_p(P, "reasoning", matched=True, p_text=below) == pytest.approx(P)
    assert update_p(P, "why", matched=True, unsure=True, p_text=TEXT_MATCH_P) == pytest.approx(P)
    assert update_p(P, "reasoning", status="unsure", p_text=1.0) == pytest.approx(P)
    assert update_p(P, "reasoning", matched=False, p_text=1.0) == pytest.approx(P)
    assert update_p(P, "why", status="correct_reasoning", p_text=1.0) == pytest.approx(P)


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError):
        update_p(P, "not_an_item", correct=True)
