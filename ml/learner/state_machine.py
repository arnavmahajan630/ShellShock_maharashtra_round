"""State machine for one (learner, misconception) (ml_plan/03 §8.4).

``next_state`` reads the P(A) produced by the knowledge update. It does not copy a model
posterior. One call applies one event. Keyword arguments override keys in ``event``.

    intervention="start"          ACTIVE or RELAPSED → TREATING
    intervention="done"           TREATING → PROBATION
    exam_update=True              accepted so a caller can mark an exam update. UNSEEN goes
                                  to ACTIVE when the updated P is at least DIAGNOSIS_ACTIVE_P,
                                  the same destination as any other update that crosses it
                                  (03 §8.4, the v3 "NEW" arrow).
    trap_passed, transfer_passed  PROBATION → STABLE only when P < STABLE_P and both are set
    after_item                    PROBATION → TREATING when P > RELAPSE_TO_TREATING_P.
                                  Defaults to True, except on an intervention event, which is
                                  not an item. That arrow is not a move to ACTIVE.
    ghost_passed                  STABLE → MASTERED when P < MASTERED_P
    ghost_failed                  STABLE or MASTERED → RELAPSED
    exam_passed, exposure         STABLE → MASTERED when exposure >= EXAM_MASTERY_EXPOSURE
                                  and P < MASTERED_P. ``e_ik`` is read as ``exposure``.

Updated P >= DIAGNOSIS_ACTIVE_P sends UNSEEN to ACTIVE, and STABLE or MASTERED to RELAPSED.
No other state grows an arrow from a high P. In particular PROBATION does not become ACTIVE.

``reassess_conditions`` is the ``/reassess`` checklist in ml_plan/03 §8.4. The probability
label is the plan's sentence; the comparison uses ``STABLE_P``.

Transfer families are the table in ml_plan/03 §8.3. A family is different when it is not the
family where the misconception was found. ``is_transfer`` also requires the new family to
be in this class's list.
"""
from __future__ import annotations

from ml.contracts.classes import STATES
from ml.contracts.params import (
    DIAGNOSIS_ACTIVE_P,
    EXAM_MASTERY_EXPOSURE,
    MASTERED_P,
    RELAPSE_TO_TREATING_P,
    STABLE_P,
)

_ACCUMULATE = ("array_accumulate", "accumulate_product", "array_count_if")
_BRANCH = ("equality_check", "branch_bands", "while_progress")
_RECURSION = ("rec_product", "rec_digits", "rec_array")

# 03 §8.3, copied as data. Main-game classes are not given extra DSA families here.
TRANSFER_FAMILIES = {
    "M01": ("count_loop", "countdown_loop", "prefix_loop"),
    "M02": ("while_progress", "count_loop"),
    "M03": _ACCUMULATE,
    "M04": ("average", "ratio"),
    "M05": _ACCUMULATE,
    "M06": _BRANCH,
    "M07": _BRANCH,
    "M08": ("index_access", "array_accumulate", "array_max"),
    "M10": ("return_value", "index_access", "equality_check", "pairwise_check"),
    "D01": ("linear_search", "pairwise_check", "string_two_pointer", "flag_search"),
    "D02": ("binary_search", "guess_halving"),
    "D03": ("bubble_sort", "selection_sort", "two_pointer_swap", "shift"),
    "D04": ("bubble_sort", "selection_sort"),
    "D05": _RECURSION,
    "D06": _RECURSION,
    "D07": _RECURSION,
    "D08": ("string_count", "string_two_pointer"),
}

P_ACTIVE_LABEL = "Misconception probability < 0.15"
TRAP_LABEL = "Trap item passed"
TRANSFER_LABEL = "Different-family transfer passed"


def _merge(event, kwargs):
    if event is None:
        merged = {}
    elif isinstance(event, dict):
        merged = dict(event)
    else:
        raise TypeError("event must be a dict")
    merged.update(kwargs)
    return merged


def different_family(found_family, problem_family):
    """True when the new problem is not in the family where the misconception was found."""
    return problem_family != found_family


def is_transfer(class_id, found_family, problem_family):
    """True when ``problem_family`` is a listed transfer family and not where it was found."""
    return (problem_family in TRANSFER_FAMILIES[class_id]
            and different_family(found_family, problem_family))


def reassess_conditions(p_active, *, trap_passed, transfer_passed, trap_detail="", transfer_detail=""):
    """The three ``/reassess`` conditions (03 §8.4). ``detail`` for P is the probability as text."""
    p = float(p_active)
    return [
        {"id": "p_active", "label": P_ACTIVE_LABEL, "met": p < STABLE_P, "detail": f"{p:.2f}"},
        {"id": "trap", "label": TRAP_LABEL, "met": bool(trap_passed), "detail": trap_detail},
        {"id": "transfer", "label": TRANSFER_LABEL, "met": bool(transfer_passed), "detail": transfer_detail},
    ]


def _intervention(ev):
    kind = ev.get("intervention")
    if ev.get("type") == "intervention_start":
        return "start"
    if ev.get("type") == "intervention_done":
        return "done"
    return kind


def next_state(state, p_active, event=None, **kwargs):
    """Return the state after one event. ``p_active`` is the updated probability."""
    if state not in STATES:
        raise ValueError(f"unknown state {state!r}")
    ev = _merge(event, kwargs)
    p = float(p_active)
    step = _intervention(ev)

    if step == "start" and state in ("ACTIVE", "RELAPSED"):
        return "TREATING"
    if step == "done" and state == "TREATING":
        return "PROBATION"

    if state in ("STABLE", "MASTERED") and (p >= DIAGNOSIS_ACTIVE_P or ev.get("ghost_failed")):
        return "RELAPSED"
    if state == "UNSEEN" and p >= DIAGNOSIS_ACTIVE_P:
        return "ACTIVE"

    if state == "PROBATION":
        after_item = ev.get("after_item")
        if after_item is None:
            after_item = step is None
        if after_item and p > RELAPSE_TO_TREATING_P:
            return "TREATING"
        if p < STABLE_P and ev.get("trap_passed") and ev.get("transfer_passed"):
            return "STABLE"
        return "PROBATION"

    if state == "STABLE":
        if ev.get("ghost_passed") and p < MASTERED_P:
            return "MASTERED"
        exposure = ev.get("exposure", ev.get("e_ik"))
        if (ev.get("exam_passed") and exposure is not None
                and float(exposure) >= EXAM_MASTERY_EXPOSURE and p < MASTERED_P):
            return "MASTERED"

    return state
