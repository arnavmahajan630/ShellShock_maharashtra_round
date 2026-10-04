"""Probe selection by expected information gain (ml_plan/03 §6.4).

candidates  probes whose belief map touches any of the top-3 classes, not already asked
EIG         H(p) − Σ_a P(a) H(p(· | a)), in bits, with P(a) = Σ_k p(k) P(a | k)
ask         the argmax, and only when its EIG is at least EIG_MIN_BITS
stop        max p ≥ POSTERIOR_STOP, or len(asked) ≥ MAX_PROBES, or nothing clears the minimum
fallback    only when the bank has no candidate touching the top-3: the rule table
            {twin set → P_<set>_a}. Skipped if that probe was already asked or its EIG
            is below the minimum. Not used when candidates exist but all fall short.

`make_best_probe(asked)` is the callable `decide` expects:
(posterior) -> ({probe_id, prompt, code, options}, eig_bits) or None.
`decide` adds eig_bits to the probe dict itself.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from ml.contracts.classes import LABELS, twin_set_of
from ml.contracts.params import EIG_MIN_BITS, MAX_PROBES, POSTERIOR_STOP
from ml.bayes.likelihood import likelihood

PROBES_PATH = Path(__file__).resolve().parents[1] / "data" / "probes.json"
TOP_N = 3


def load_probes(path=None):
    """The probe bank. One JSON object per probe; `twin_sets` may name two sets."""
    file = PROBES_PATH if path is None else Path(path)
    return json.loads(file.read_text(encoding="utf-8"))


def expected_information_gain(posterior, options, correct, belief=None):
    """EIG of one item, in bits, under the current posterior."""
    table = likelihood(options, correct, belief)
    names = list(table[LABELS[0]].keys())
    base = [max(float(posterior.get(label, 0.0)), 0.0) for label in LABELS]
    expected = 0.0
    for option in names:
        joint = [base[i] * table[label][option] for i, label in enumerate(LABELS)]
        p_answer = math.fsum(joint)
        if p_answer <= 0.0:
            continue
        conditional = [value / p_answer for value in joint]
        expected += p_answer * _entropy(conditional)
    return _entropy(base) - expected


def select_probe(posterior, asked=(), probes=None):
    """The next probe as (public dict, eig_bits), or None when probing should stop.

    A probe already in `asked` is never proposed. `asked` holds probe ids.
    """
    bank = load_probes() if probes is None else probes
    asked_list = list(asked)
    if len(asked_list) >= MAX_PROBES:
        return None
    if _max_p(posterior) >= POSTERIOR_STOP:
        return None

    asked_ids = set(asked_list)
    order = _ranked(posterior)
    top3 = set(order[:TOP_N])
    candidates = [
        probe for probe in bank
        if probe["probe_id"] not in asked_ids and top3.intersection(probe.get("belief") or {})
    ]
    if candidates:
        probe, score = _argmax(posterior, candidates)
        if score >= EIG_MIN_BITS:
            return _public(probe), score
        return None

    fallback = _fallback_probe(order[0], order[1], bank, asked_ids)
    if fallback is None:
        return None
    score = expected_information_gain(
        posterior, fallback["options"], fallback["correct"], fallback.get("belief"))
    if score < EIG_MIN_BITS:
        return None
    return _public(fallback), score


def make_best_probe(asked, probes=None):
    """callable(posterior) -> (public probe, eig_bits) or None, for `decide`."""
    bank = load_probes() if probes is None else list(probes)

    def best_probe(posterior):
        return select_probe(posterior, asked, bank)

    return best_probe


def _fallback_probe(top1, top2, bank, asked_ids):
    """Rule table {twin set → P_<set>_a}, or None if there is no set or it was asked."""
    set_id = twin_set_of(top1, top2)
    if set_id is None:
        return None
    probe_id = f"P_{set_id}_a"
    if probe_id in asked_ids:
        return None
    for probe in bank:
        if probe["probe_id"] == probe_id:
            return probe
    return None


def _argmax(posterior, probes):
    chosen = None
    chosen_score = None
    for probe in probes:
        score = expected_information_gain(
            posterior, probe["options"], probe["correct"], probe.get("belief"))
        if chosen is None or score > chosen_score:
            chosen, chosen_score = probe, score
    return chosen, chosen_score


def _public(probe):
    return {
        "probe_id": probe["probe_id"],
        "prompt": probe["prompt"],
        "code": probe["code"],
        "options": list(probe["options"]),
    }


def _ranked(posterior):
    return sorted(LABELS, key=lambda label: (-float(posterior.get(label, 0.0)), LABELS.index(label)))


def _max_p(posterior):
    return max(float(posterior.get(label, 0.0)) for label in LABELS)


def _entropy(probs):
    total = 0.0
    for p in probs:
        if p > 0.0:
            total -= p * math.log2(p)
    return total
