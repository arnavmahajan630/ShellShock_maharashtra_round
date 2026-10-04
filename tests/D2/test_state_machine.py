"""D2: every arrow in 03 §8.4, and a near miss of each threshold."""
import math

import pytest

from ml.contracts.classes import MISCONCEPTIONS
from ml.contracts.params import (
    DIAGNOSIS_ACTIVE_P,
    EXAM_MASTERY_EXPOSURE,
    FAIL_SIGNATURE_IF_NOT,
    LEARN_RATE,
    MASTERED_P,
    RELAPSE_TO_TREATING_P,
    STABLE_P,
)
from ml.contracts.schemas import Condition
from ml.learner.knowledge import likelihood_ratio, update_p
from ml.learner.state_machine import (
    P_ACTIVE_LABEL,
    TRANSFER_FAMILIES,
    different_family,
    is_transfer,
    next_state,
    reassess_conditions,
)

JUST_UNDER = {
    "diagnosis": math.nextafter(DIAGNOSIS_ACTIVE_P, 0.0),
    "stable": math.nextafter(STABLE_P, 0.0),
    "relapse": math.nextafter(RELAPSE_TO_TREATING_P, 1.0),
    "mastered": math.nextafter(MASTERED_P, 0.0),
    "exposure": math.nextafter(EXAM_MASTERY_EXPOSURE, 0.0),
}


def test_transfer_families_match_the_plan_table():
    assert set(TRANSFER_FAMILIES) == set(MISCONCEPTIONS)
    assert TRANSFER_FAMILIES["M01"] == ("count_loop", "countdown_loop", "prefix_loop")
    assert TRANSFER_FAMILIES["M02"] == ("while_progress", "count_loop")
    assert TRANSFER_FAMILIES["M03"] == ("array_accumulate", "accumulate_product", "array_count_if")
    assert TRANSFER_FAMILIES["M05"] == TRANSFER_FAMILIES["M03"]
    assert TRANSFER_FAMILIES["M04"] == ("average", "ratio")
    assert TRANSFER_FAMILIES["M06"] == ("equality_check", "branch_bands", "while_progress")
    assert TRANSFER_FAMILIES["M07"] == TRANSFER_FAMILIES["M06"]
    assert TRANSFER_FAMILIES["M08"] == ("index_access", "array_accumulate", "array_max")
    assert TRANSFER_FAMILIES["M10"] == ("return_value", "index_access", "equality_check", "pairwise_check")
    assert TRANSFER_FAMILIES["D01"] == ("linear_search", "pairwise_check", "string_two_pointer", "flag_search")
    assert TRANSFER_FAMILIES["D02"] == ("binary_search", "guess_halving")
    assert TRANSFER_FAMILIES["D03"] == ("bubble_sort", "selection_sort", "two_pointer_swap", "shift")
    assert TRANSFER_FAMILIES["D04"] == ("bubble_sort", "selection_sort")
    assert TRANSFER_FAMILIES["D05"] == ("rec_product", "rec_digits", "rec_array")
    assert TRANSFER_FAMILIES["D06"] == TRANSFER_FAMILIES["D05"] == TRANSFER_FAMILIES["D07"]
    assert TRANSFER_FAMILIES["D08"] == ("string_count", "string_two_pointer")


def test_different_family_is_inequality_and_transfer_also_needs_the_list():
    assert different_family("count_loop", "countdown_loop")
    assert not different_family("count_loop", "count_loop")
    assert is_transfer("M01", "count_loop", "prefix_loop")
    assert not is_transfer("M01", "count_loop", "count_loop")
    assert different_family("count_loop", "pairwise_check")
    assert not is_transfer("M01", "count_loop", "pairwise_check")
    assert is_transfer("D08", "string_count", "string_two_pointer")
    assert is_transfer("M04", "somewhere_else", "ratio")


# ---------------------------------------------------------------- transitions

def test_unseen_to_active_when_updated_p_reaches_the_threshold():
    assert next_state("UNSEEN", DIAGNOSIS_ACTIVE_P) == "ACTIVE"
    assert next_state("UNSEEN", DIAGNOSIS_ACTIVE_P, exam_update=True) == "ACTIVE"
    assert next_state("UNSEEN", DIAGNOSIS_ACTIVE_P, type="exam_update") == "ACTIVE"
    assert next_state("UNSEEN", DIAGNOSIS_ACTIVE_P, type="diagnosis") == "ACTIVE"


def test_active_intervention_start_to_treating():
    assert next_state("ACTIVE", 0.8, intervention="start") == "TREATING"
    assert next_state("ACTIVE", 0.8, {"type": "intervention_start"}) == "TREATING"


def test_treating_intervention_done_to_probation():
    """The learning-rate multiply is the knowledge update; this call only changes state."""
    p = update_p(0.8, "intervention")
    assert p == pytest.approx(0.8 * (1.0 - LEARN_RATE))
    assert next_state("TREATING", p, intervention="done") == "PROBATION"
    assert next_state("TREATING", 0.9, intervention="done") == "PROBATION"


def test_probation_to_stable_only_when_all_three_hold():
    low = JUST_UNDER["stable"]
    assert low < STABLE_P
    assert next_state("PROBATION", low, trap_passed=True, transfer_passed=True) == "STABLE"
    wired = is_transfer("M01", "count_loop", "countdown_loop")
    assert next_state("PROBATION", low, trap_passed=True, transfer_passed=wired) == "STABLE"


def test_probation_above_relapse_goes_to_treating_not_active():
    high = JUST_UNDER["relapse"]
    assert high > RELAPSE_TO_TREATING_P
    assert next_state("PROBATION", high) == "TREATING"
    assert next_state("PROBATION", high, type="diagnosis") == "TREATING"
    assert next_state("PROBATION", high, after_item=True) != "ACTIVE"


def test_stable_ghost_return_to_mastered():
    assert next_state("STABLE", JUST_UNDER["mastered"], ghost_passed=True) == "MASTERED"


def test_stable_exam_item_to_mastered():
    low = JUST_UNDER["mastered"]
    assert next_state("STABLE", low, exam_passed=True, exposure=EXAM_MASTERY_EXPOSURE) == "MASTERED"
    assert next_state("STABLE", low, exam_passed=True, e_ik=EXAM_MASTERY_EXPOSURE) == "MASTERED"
    assert next_state("STABLE", low, exam_update=True, exam_passed=True,
                      exposure=EXAM_MASTERY_EXPOSURE) == "MASTERED"


def test_stable_and_mastered_high_p_to_relapsed():
    assert next_state("STABLE", DIAGNOSIS_ACTIVE_P) == "RELAPSED"
    assert next_state("MASTERED", DIAGNOSIS_ACTIVE_P) == "RELAPSED"
    assert next_state("STABLE", DIAGNOSIS_ACTIVE_P, type="diagnosis") == "RELAPSED"
    assert next_state("MASTERED", DIAGNOSIS_ACTIVE_P, exam_update=True) == "RELAPSED"


def test_stable_and_mastered_ghost_failure_to_relapsed():
    assert next_state("STABLE", 0.0, ghost_failed=True) == "RELAPSED"
    assert next_state("MASTERED", JUST_UNDER["diagnosis"], ghost_failed=True) == "RELAPSED"


def test_relapsed_intervention_start_to_treating():
    assert next_state("RELAPSED", 0.9, intervention="start") == "TREATING"


def test_high_p_on_a_pass_does_not_master():
    assert next_state("STABLE", DIAGNOSIS_ACTIVE_P, ghost_passed=True) == "RELAPSED"
    assert next_state("STABLE", DIAGNOSIS_ACTIVE_P, exam_passed=True,
                      exposure=EXAM_MASTERY_EXPOSURE) == "RELAPSED"


def test_diagnosis_does_not_invent_arrows():
    assert next_state("ACTIVE", DIAGNOSIS_ACTIVE_P) == "ACTIVE"
    assert next_state("TREATING", DIAGNOSIS_ACTIVE_P) == "TREATING"
    assert next_state("RELAPSED", DIAGNOSIS_ACTIVE_P) == "RELAPSED"
    assert next_state("ACTIVE", 0.0) == "ACTIVE"
    assert next_state("TREATING", 0.0) == "TREATING"
    assert next_state("RELAPSED", 0.0) == "RELAPSED"
    assert next_state("UNSEEN", 0.1, ghost_failed=True) == "UNSEEN"
    assert next_state("PROBATION", 0.2, ghost_failed=True) == "PROBATION"
    assert next_state("STABLE", 0.05, intervention="start") == "STABLE"
    assert next_state("ACTIVE", 0.4, intervention="done") == "ACTIVE"
    assert next_state("MASTERED", JUST_UNDER["mastered"], ghost_passed=True) == "MASTERED"


def test_state_reads_updated_p_and_ignores_a_model_posterior():
    p0 = 0.2
    assert p0 < DIAGNOSIS_ACTIVE_P
    assert next_state("UNSEEN", p0, posterior=0.99) == "UNSEEN"
    p1 = update_p(p0, "signature_failure", exposure=0.95)
    assert p1 == pytest.approx(likelihood_ratio(p0, 0.95, FAIL_SIGNATURE_IF_NOT))
    assert p1 >= DIAGNOSIS_ACTIVE_P
    assert next_state("UNSEEN", p1, posterior=0.01) == "ACTIVE"
    assert next_state("STABLE", p1, posterior=0.01) == "RELAPSED"


# ---------------------------------------------------------------- near misses

def test_near_miss_unseen_stays_unseen():
    assert next_state("UNSEEN", JUST_UNDER["diagnosis"]) == "UNSEEN"
    assert next_state("UNSEEN", JUST_UNDER["diagnosis"], exam_update=True) == "UNSEEN"


def test_near_miss_stable_and_mastered_do_not_relapse_on_p():
    assert next_state("STABLE", JUST_UNDER["diagnosis"]) == "STABLE"
    assert next_state("MASTERED", JUST_UNDER["diagnosis"]) == "MASTERED"


def test_near_miss_probation_does_not_reach_stable():
    low = JUST_UNDER["stable"]
    assert next_state("PROBATION", STABLE_P, trap_passed=True, transfer_passed=True) == "PROBATION"
    assert next_state("PROBATION", low, trap_passed=False, transfer_passed=True) == "PROBATION"
    assert next_state("PROBATION", low, trap_passed=True, transfer_passed=False) == "PROBATION"
    assert next_state("PROBATION", low, trap_passed=True,
                      transfer_passed=is_transfer("M01", "count_loop", "count_loop")) == "PROBATION"


def test_near_miss_probation_at_the_relapse_threshold_stays():
    assert next_state("PROBATION", RELAPSE_TO_TREATING_P, after_item=True) == "PROBATION"
    assert next_state("PROBATION", JUST_UNDER["relapse"], intervention="done") == "PROBATION"
    assert next_state("PROBATION", JUST_UNDER["relapse"], ghost_failed=True) == "TREATING"


def test_near_miss_ghost_mastery_needs_p_under_the_threshold():
    assert next_state("STABLE", MASTERED_P, ghost_passed=True) == "STABLE"


def test_near_miss_exam_mastery_needs_exposure_and_p():
    low = JUST_UNDER["mastered"]
    assert next_state("STABLE", low, exam_passed=True, exposure=JUST_UNDER["exposure"]) == "STABLE"
    assert next_state("STABLE", MASTERED_P, exam_passed=True, exposure=EXAM_MASTERY_EXPOSURE) == "STABLE"
    assert next_state("STABLE", low, exam_passed=False, exposure=EXAM_MASTERY_EXPOSURE) == "STABLE"
    assert next_state("MASTERED", low, exam_passed=True, exposure=EXAM_MASTERY_EXPOSURE) == "MASTERED"


# ---------------------------------------------------------------- reassess

def test_reassess_conditions_match_the_plan_shape():
    rows = reassess_conditions(
        0.31, trap_passed=False, transfer_passed=True,
        trap_detail="predicted 9, actual: outside the array",
        transfer_detail="P09 max_shield",
    )
    assert rows == [
        {"id": "p_active", "label": "Misconception probability < 0.15", "met": False, "detail": "0.31"},
        {"id": "trap", "label": "Trap item passed", "met": False,
         "detail": "predicted 9, actual: outside the array"},
        {"id": "transfer", "label": "Different-family transfer passed", "met": True, "detail": "P09 max_shield"},
    ]
    assert rows[0]["label"] == P_ACTIVE_LABEL
    for row in rows:
        Condition.model_validate(row)
    assert (0.31 < STABLE_P) is False
    under = reassess_conditions(JUST_UNDER["stable"], trap_passed=True, transfer_passed=False)
    assert under[0]["met"] is True
    at = reassess_conditions(STABLE_P, trap_passed=True, transfer_passed=True)
    assert at[0]["met"] is False
    assert at[0]["detail"] == f"{float(STABLE_P):.2f}"


def test_unknown_state_is_rejected():
    with pytest.raises(ValueError):
        next_state("DONE", 0.1)
