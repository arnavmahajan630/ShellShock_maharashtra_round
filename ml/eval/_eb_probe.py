"""Simulated hard-twin probing for E5.

The engine's likelihood stays at params.P_BELIEF (0.6). The simulated learner's true p_b
is swept from 0.4 to 0.9 (03 §6.2, §9.2 E5). q and s stay at the engine's values.
The learner draws from that distribution before the probability floor; the floor is only
inside the engine's update.
"""
from __future__ import annotations

import hashlib

import numpy as np

from ml.bayes.eig import load_probes, make_best_probe
from ml.bayes.posterior import update
from ml.contracts.classes import LABELS
from ml.contracts.params import MAX_PROBES, Q_CORRECT, SLIP
from ml.eval._eb_data import SEED
from ml.model.decide import decide

P_B_GRID = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
N_REPS = 20


def answer_distribution(options, correct, belief, true_label, p_b):
    """P(answer) for a learner whose misconception is `true_label`, at this true p_b."""
    names = [str(option) for option in options]
    correct = str(correct)
    belief = {str(key): str(value) for key, value in (belief or {}).items()}
    believed = belief.get(str(true_label))
    if believed is not None and believed != correct and believed in names:
        raw = {}
        on_correct = (1.0 - p_b) * Q_CORRECT
        others = [option for option in names if option != believed and option != correct]
        share = (1.0 - p_b - on_correct) / len(others) if others else 0.0
        for option in names:
            if option == believed:
                raw[option] = p_b
            elif option == correct:
                raw[option] = on_correct
            else:
                raw[option] = share
    else:
        on_correct = 1.0 - SLIP
        others = [option for option in names if option != correct]
        share = (1.0 - on_correct) / len(others) if others else 0.0
        raw = {option: on_correct if option == correct else share for option in names}
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("answer distribution has no mass")
    return {option: value / total for option, value in raw.items()}


def argmax_label(posterior):
    """Same tie break as decide.ranked: higher probability, then earlier LABELS entry."""
    return min(LABELS, key=lambda name: (-float(posterior.get(name, 0.0)), LABELS.index(name)))


def rng_for(row_id, p_b, rep, seed=SEED):
    digest = hashlib.sha256(f"{seed}|{row_id}|{p_b:.1f}|{rep}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _draw(rng, dist):
    options = list(dist)
    probs = np.asarray([dist[option] for option in options], dtype=np.float64)
    probs /= probs.sum()
    return str(rng.choice(options, p=probs))


def probe_once(posterior, true_label, p_b, rng, probes, by_id, novelty, tests_passed):
    """Follow decide()'s probe policy. Returns (pred_label, n_asked, first_status)."""
    asked = []
    post = dict(posterior)
    first_status = None
    for _ in range(MAX_PROBES + 1):
        diagnosis = decide(
            post, tests_passed=bool(tests_passed), novelty=novelty,
            best_probe=make_best_probe(asked, probes), probes_asked=asked,
        )
        if first_status is None:
            first_status = diagnosis["status"]
        probe = diagnosis.get("next_probe")
        if diagnosis["status"] != "ambiguous" or not probe:
            break
        full = by_id[probe["probe_id"]]
        dist = answer_distribution(full["options"], full["correct"], full.get("belief"), true_label, p_b)
        answer = _draw(rng, dist)
        post = update(post, answer, correct=full["correct"], belief=full.get("belief"), options=full["options"])
        asked.append(probe["probe_id"])
    return argmax_label(post), len(asked), first_status


def load_probe_bank():
    probes = load_probes()
    return probes, {probe["probe_id"]: probe for probe in probes}
