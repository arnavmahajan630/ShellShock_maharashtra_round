"""Likelihood tables (ml_plan/03 §6.2).

P(answer = a | class) for one item. Probes, mission predict items, MCQ / predict_output /
next_state (ml_plan/05 §4) and exam trace items (03 §6.5, §6.6) all use this table.
Numbers come from ml.contracts.params. Nothing here is fitted.
"""
from __future__ import annotations

import math

from ml.contracts.classes import LABELS
from ml.contracts.params import P_BELIEF, P_OTHER_CORRECT, PROB_FLOOR, Q_CORRECT, SLIP

FREE_OTHER = "other"   # 03 §6.2: free numeric predictions add this bucket


def prediction_options(correct, belief=None):
    """Options for a free numeric prediction: correct, each distinct belief value, then "other"."""
    return _dedupe([correct, *(belief or {}).values(), FREE_OTHER])


def map_free_answer(answer, correct, belief=None):
    """Map a free prediction onto `prediction_options`. Anything unrecognised is "other"."""
    text = str(answer)
    known = {str(correct), *(str(value) for value in (belief or {}).values())}
    return text if text in known else FREE_OTHER


def likelihood(options, correct, belief=None):
    """P(answer = a | class) for every label. {class: {option: probability}}, each row sums to 1.

    `options` is omitted for a free numeric prediction; the option set is then
    `prediction_options`. A misconception is diagnostic only when its belief is defined,
    different from the correct answer, and one of the options. Everyone else answers like
    CORRECT (slip `s`), except OTHER.
    """
    belief = {str(key): str(value) for key, value in (belief or {}).items()}
    correct = str(correct)
    if options:
        names = _dedupe(options)
    else:
        names = prediction_options(correct, belief)
    if correct not in names:
        raise ValueError(f"correct answer {correct!r} is not an option")

    table = {}
    for label in LABELS:
        if label == "OTHER":
            raw = _on_correct(names, correct, P_OTHER_CORRECT)
        elif label == "CORRECT":
            raw = _on_correct(names, correct, 1.0 - SLIP)
        else:
            believed = belief.get(label)
            if believed is not None and believed != correct and believed in names:
                raw = _diagnostic(names, correct, believed)
            else:
                raw = _on_correct(names, correct, 1.0 - SLIP)
        table[label] = _floor_renormalise(raw)
    return table


def _dedupe(values):
    seen = []
    for value in values:
        text = str(value)
        if text not in seen:
            seen.append(text)
    return seen


def _on_correct(options, correct, on_correct):
    """`on_correct` on the right answer; the rest spread evenly over the other options."""
    others = [option for option in options if option != correct]
    share = (1.0 - on_correct) / len(others) if others else 0.0
    return {option: on_correct if option == correct else share for option in options}


def _diagnostic(options, correct, belief_answer):
    """p_b on the belief answer, (1 - p_b) * q on the correct answer, remainder spread evenly."""
    on_belief = P_BELIEF
    on_correct = (1.0 - P_BELIEF) * Q_CORRECT
    others = [option for option in options if option != belief_answer and option != correct]
    share = (1.0 - on_belief - on_correct) / len(others) if others else 0.0
    raw = {}
    for option in options:
        if option == belief_answer:
            raw[option] = on_belief
        elif option == correct:
            raw[option] = on_correct
        else:
            raw[option] = share
    return raw


def _floor_renormalise(raw):
    lifted = {option: max(value, PROB_FLOOR) for option, value in raw.items()}
    total = math.fsum(lifted.values())
    return {option: value / total for option, value in lifted.items()}
