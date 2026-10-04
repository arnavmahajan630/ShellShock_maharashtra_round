"""Posterior over the 19 labels (ml_plan/03 §6.1, ml_plan/05 §4).

prior:   p(k) ∝ p0(k) · π_L(k)^PRIOR_GAMMA. Masked classes are removed (mass 0).
update:  p(k) ← p(k) · P(answer | k), then renormalise. One call per response;
         responses are conditionally independent given the class.
apply_text: a reasoning or "why?" follow-up. Skipped when the reader is unsure.
            The knowledge-model half of ml_plan/05 §4 is package D2.
"""
from __future__ import annotations

import math

from ml.contracts.classes import LABELS
from ml.contracts.params import (
    POPULATION_PRIOR, PRIOR_FLOOR, PRIOR_GAMMA, PRIOR_NEUTRAL, TEXT_BETA, TEXT_EPSILON,
)
from ml.bayes.likelihood import likelihood, map_free_answer


def prior(p0, learner=None, masked=None):
    """Model probabilities times the learner prior, with `masked` classes removed.

    π_L(k) is the learner's P(active) for a misconception, floored at PRIOR_FLOOR.
    A misconception with no entry uses POPULATION_PRIOR. π_L(CORRECT) and π_L(OTHER)
    are PRIOR_NEUTRAL whatever `learner` says.
    """
    blocked = set(masked or ())
    weights = {}
    for label in LABELS:
        if label in blocked:
            weights[label] = 0.0
            continue
        base = max(float(p0.get(label, 0.0)), 0.0)
        weights[label] = base * (_pi(label, learner) ** PRIOR_GAMMA)
    return _normalise(weights)


def update(posterior, answer, *, correct, belief=None, options=None):
    """Multiply by P(answer | k) and renormalise.

    Pass `options` for an MCQ, probe, predict item or exam trace item. Omit them for a
    free numeric prediction; the answer is mapped onto {correct, belief values, "other"}.
    """
    belief = belief or {}
    if options:
        names = [str(option) for option in options]
        given = str(answer)
        if given not in names:
            raise ValueError(f"answer {answer!r} is not an option")
    else:
        names = None
        given = map_free_answer(answer, correct, belief)
    row = likelihood(names, correct, belief)
    weights = {}
    for label in LABELS:
        mass = max(float(posterior.get(label, 0.0)), 0.0)
        weights[label] = mass * row[label][given]
    return _normalise(weights)


def apply_text(posterior, p_text, unsure=False):
    """p(k) ← p(k) · (TEXT_EPSILON + P_text(k))^TEXT_BETA, then renormalise.

    `unsure=True` returns the posterior unchanged. Classes the reader does not mention
    get P_text = 0. A class with no posterior mass stays at 0.
    """
    current = {label: float(posterior.get(label, 0.0)) for label in LABELS}
    if unsure:
        return current
    text = p_text or {}
    weights = {}
    for label in LABELS:
        support = max(float(text.get(label, 0.0)), 0.0)
        weights[label] = current[label] * (TEXT_EPSILON + support) ** TEXT_BETA
    return _normalise(weights)


def _pi(label, learner):
    if label in ("CORRECT", "OTHER"):
        return PRIOR_NEUTRAL
    if not learner or label not in learner:
        return max(POPULATION_PRIOR, PRIOR_FLOOR)
    return max(float(learner[label]), PRIOR_FLOOR)


def _normalise(weights):
    total = math.fsum(weights.values())
    if total <= 0.0:
        raise ValueError("no probability mass")
    return {key: value / total for key, value in weights.items()}
