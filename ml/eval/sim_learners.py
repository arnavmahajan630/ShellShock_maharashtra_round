"""Resolution simulation on simulated learners (ml_plan/03 §9.3, E10).

These learners are simulations. They are not real students. Each type has a
hidden truth the policy does not see. ``g_sim``, ``s_sim`` and ``p_b_sim`` are
drawn uniformly from ±50% around the engine's hand-set values, once per
learner, so the engine is not scored only at its own assumptions.

Policies share an item budget of at most 4 follow-ups plus 1 ghost return.
``naive`` and ``4-check`` do not use the ghost. ``ours`` is the §8 knowledge
update and state machine.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np

from ml.contracts.params import (
    FAIL_SIGNATURE_IF_ACTIVE,
    ITEM_GUESS_SLIP,
    P_BELIEF,
    POPULATION_PRIOR,
    Q_CORRECT,
    STABLE_P,
)
from ml.eval._v1_stats import mean_interval
from ml.learner.knowledge import update_p
from ml.learner.state_machine import next_state

# 03 §9.3. Names are the plan's names.
LEARNER_TYPES = (
    "truly_fixed",
    "pattern_copier",
    "lucky_guesser",
    "forgetful",
    "slow",
)
POLICIES = ("naive", "4-check", "ours")

MAX_FOLLOWUPS = 4
FORGET_BEFORE_GHOST = 0.6
PARAM_LO = 0.5
PARAM_HI = 1.5
N_CHOICE_OPTIONS = 3

# Stated active-learner rates (03 §9.3). Not the engine's g.
PATTERN_CODE_PASS = {
    "same_family_code": 0.9,
    "transfer_code": 0.5,
    "ghost": 0.5,
}
LUCKY_CODE_PASS = 0.35

E10_SEED = 1010
N_LEARNERS = 2000
N_SEEDS = 20

SIMULATED_POPULATION = "simulated learners (not real students)"
CAVEAT = (
    "Simulated learners embody our assumptions about learner behaviour. "
    "The table shows how the policies behave if learners act like this, "
    "not that real learners do."
)

FOLLOWUP_ORDER = ("same_family_code", "transfer_code", "trap", "belief_mcq")


def _clip(value):
    if value < 1e-6:
        return 1e-6
    if value > 1.0 - 1e-6:
        return 1.0 - 1e-6
    return float(value)


def draw_params(rng):
    """``g_sim``, ``s_sim``, ``p_b_sim`` in ±50% of the engine values (03 §9.3)."""
    u_g = float(rng.uniform(PARAM_LO, PARAM_HI))
    u_s = float(rng.uniform(PARAM_LO, PARAM_HI))
    u_b = float(rng.uniform(PARAM_LO, PARAM_HI))
    g_sim = {}
    s_sim = {}
    for kind, (g, s) in ITEM_GUESS_SLIP.items():
        g_sim[kind] = _clip(float(g) * u_g)
        s_sim[kind] = _clip(float(s) * u_s)
    return {
        "g_sim": g_sim,
        "s_sim": s_sim,
        "p_b_sim": _clip(P_BELIEF * u_b),
    }


def enter_probation():
    """Diagnosis, then one completed intervention. Hidden from the item rules.

    Starts at the population prior, applies one signature failure (the learner
    was caught), then the learning step. The state machine does the arrows.
    """
    p = update_p(POPULATION_PRIOR, "signature_failure", exposure=FAIL_SIGNATURE_IF_ACTIVE)
    state = next_state("UNSEEN", p)
    state = next_state(state, p, intervention="start")
    p = update_p(p, "intervention")
    state = next_state(state, p, intervention="done")
    return p, state


def answer_correct(rng, behaviour, kind, params):
    """One simulated response. ``True`` means the item is scored correct."""
    if behaviour == "inactive":
        return bool(rng.random() < (1.0 - params["s_sim"][kind]))
    if behaviour == "slow_active":
        # Still active after the first intervention: the engine's g, shifted.
        return bool(rng.random() < params["g_sim"][kind])
    if behaviour == "pattern_copier":
        if kind in PATTERN_CODE_PASS:
            return bool(rng.random() < PATTERN_CODE_PASS[kind])
        # Trap or concept MCQ: belief answer with p_b_sim. Of the rest, the
        # engine's q lands on the correct option (03 §6.2).
        p_correct = (1.0 - params["p_b_sim"]) * Q_CORRECT
        return bool(rng.random() < p_correct)
    if behaviour == "lucky_guesser":
        if kind in ("trap", "belief_mcq"):
            return bool(rng.random() < (1.0 / N_CHOICE_OPTIONS))
        return bool(rng.random() < LUCKY_CODE_PASS)
    raise ValueError(f"unknown behaviour {behaviour!r}")


def _behaviour(learner_type, *, ghost, flipped, second_intervention):
    if learner_type == "truly_fixed":
        return "inactive"
    if learner_type == "pattern_copier":
        return "pattern_copier"
    if learner_type == "lucky_guesser":
        return "lucky_guesser"
    if learner_type == "forgetful":
        # Inactive on the follow-ups. Before the ghost, active with probability
        # FORGET_BEFORE_GHOST, and then answers as a pattern-copier (03 §9.3).
        if ghost and flipped:
            return "pattern_copier"
        return "inactive"
    if learner_type == "slow":
        if second_intervention:
            return "inactive"
        return "slow_active"
    raise ValueError(f"unknown learner type {learner_type!r}")


def _end_active(learner_type, *, flipped, second_intervention):
    """Hidden truth at the end of the episode (03 §9.3, Truth column)."""
    if learner_type == "truly_fixed":
        return False
    if learner_type in ("pattern_copier", "lucky_guesser"):
        return True
    if learner_type == "forgetful":
        return bool(flipped)
    if learner_type == "slow":
        # Active after the first intervention, inactive after the second.
        return not second_intervention
    raise ValueError(f"unknown learner type {learner_type!r}")


class _Responder:
    """Draws answers for one episode. A list of bools overrides the draw (tests)."""

    def __init__(self, rng, learner_type, params, flipped, forced):
        self.rng = rng
        self.learner_type = learner_type
        self.params = params
        self.flipped = flipped
        self.second = False
        self.forced = list(forced) if forced is not None else None

    def correct(self, kind, *, ghost):
        if self.forced is not None:
            if not self.forced:
                raise RuntimeError("forced answers ran out")
            return bool(self.forced.pop(0))
        behaviour = _behaviour(
            self.learner_type,
            ghost=ghost,
            flipped=self.flipped,
            second_intervention=self.second,
        )
        return answer_correct(self.rng, behaviour, kind, self.params)


def _finish(responder, *, declared, items, state):
    return {
        "declared_stable": bool(declared),
        "truly_active": _end_active(
            responder.learner_type,
            flipped=responder.flipped,
            second_intervention=responder.second,
        ),
        "items": int(items),
        "final_state": state,
        "simulated": True,
    }


def _naive(responder):
    """Resolved after one correct follow-up. No model, no trap, no ghost."""
    items = 0
    for kind in FOLLOWUP_ORDER:
        if items >= MAX_FOLLOWUPS:
            break
        items += 1
        if responder.correct(kind, ghost=False):
            return _finish(responder, declared=True, items=items, state="STABLE")
    return _finish(responder, declared=False, items=items, state="PROBATION")


def _four_check(responder):
    """New task passed, and the concept MCQ passed.

    The model check ``p_k < 0.25`` is treated as true whenever the new task's
    tests passed. That is the collapse 03 §9.3 attributes to the 4-check.
    """
    items = 0
    coding_passed = False
    mcq_passed = False
    tried_transfer = False
    tried_family = False
    while items < MAX_FOLLOWUPS and not (coding_passed and mcq_passed):
        if not coding_passed:
            if not tried_transfer:
                kind = "transfer_code"
                tried_transfer = True
            elif not tried_family:
                kind = "same_family_code"
                tried_family = True
            else:
                break
        else:
            kind = "belief_mcq"
        items += 1
        if responder.correct(kind, ghost=False):
            if kind == "belief_mcq":
                mcq_passed = True
            else:
                coding_passed = True
    model_p = 0.0 if coding_passed else 1.0
    declared = bool(coding_passed and model_p < 0.25 and mcq_passed)
    state = "STABLE" if declared else "PROBATION"
    return _finish(responder, declared=declared, items=items, state=state)


def _ours_kind(p, transfer_passed, trap_passed, n_this_cycle):
    if not transfer_passed:
        return "transfer_code"
    if not trap_passed:
        return "trap"
    if p >= STABLE_P:
        if n_this_cycle % 2 == 0:
            return "belief_mcq"
        return "same_family_code"
    return None


def _ours(responder):
    """§8: knowledge updates and the state machine, inside the same budget."""
    p, state = enter_probation()
    items = 0
    transfer_passed = False
    trap_passed = False
    n_this_cycle = 0
    while items < MAX_FOLLOWUPS and state not in ("STABLE", "MASTERED"):
        if state == "TREATING":
            p = update_p(p, "intervention")
            state = next_state(state, p, intervention="done")
            transfer_passed = False
            trap_passed = False
            n_this_cycle = 0
            responder.second = True
            continue
        if state != "PROBATION":
            break
        kind = _ours_kind(p, transfer_passed, trap_passed, n_this_cycle)
        if kind is None:
            break
        correct = responder.correct(kind, ghost=False)
        p = update_p(p, kind, correct=correct)
        items += 1
        n_this_cycle += 1
        if kind == "transfer_code" and correct:
            transfer_passed = True
        if kind == "trap" and correct:
            trap_passed = True
        state = next_state(
            state,
            p,
            trap_passed=trap_passed,
            transfer_passed=transfer_passed,
            after_item=True,
        )
    if state == "STABLE":
        correct = responder.correct("ghost", ghost=True)
        p = update_p(p, "ghost", correct=correct)
        items += 1
        state = next_state(
            state,
            p,
            ghost_passed=bool(correct),
            ghost_failed=not correct,
        )
    declared = state in ("STABLE", "MASTERED")
    return _finish(responder, declared=declared, items=items, state=state)


_POLICY_FN = {
    "naive": _naive,
    "4-check": _four_check,
    "ours": _ours,
}


def run_episode(rng, learner_type, policy, params=None, *, flipped=None, forced=None):
    """One simulated learner, one policy. ``forced`` is a list of correct/wrong bits."""
    if learner_type not in LEARNER_TYPES:
        raise ValueError(f"unknown learner type {learner_type!r}")
    if policy not in _POLICY_FN:
        raise ValueError(f"unknown policy {policy!r}")
    if params is None:
        params = draw_params(rng)
    if flipped is None:
        flipped = bool(rng.random() < FORGET_BEFORE_GHOST) if learner_type == "forgetful" else False
    responder = _Responder(rng, learner_type, params, flipped, forced)
    result = _POLICY_FN[policy](responder)
    if result["items"] > MAX_FOLLOWUPS + 1:
        raise RuntimeError(f"budget exceeded: {result['items']}")
    return result


def _empty_counts():
    return {"n": 0, "n_active": 0, "n_inactive": 0, "false_resolve": 0, "false_not_yet": 0, "items": 0}


def _rate(numer, denom):
    if denom == 0:
        return None
    return numer / denom


def run_seed(rng, n_learners, types=LEARNER_TYPES, policies=POLICIES):
    """One seed: ``n_learners`` of each type, every policy, shared draws per learner."""
    counts = {(t, p): _empty_counts() for t in types for p in policies}
    for _ in range(n_learners):
        for learner_type in types:
            params = draw_params(rng)
            flipped = (
                bool(rng.random() < FORGET_BEFORE_GHOST) if learner_type == "forgetful" else False
            )
            for policy in policies:
                out = run_episode(
                    rng, learner_type, policy, params, flipped=flipped,
                )
                bucket = counts[(learner_type, policy)]
                bucket["n"] += 1
                bucket["items"] += out["items"]
                if out["truly_active"]:
                    bucket["n_active"] += 1
                    if out["declared_stable"]:
                        bucket["false_resolve"] += 1
                else:
                    bucket["n_inactive"] += 1
                    if not out["declared_stable"]:
                        bucket["false_not_yet"] += 1
    summary = {}
    for key, bucket in counts.items():
        summary[key] = {
            "false_resolve": _rate(bucket["false_resolve"], bucket["n_active"]),
            "false_not_yet": _rate(bucket["false_not_yet"], bucket["n_inactive"]),
            "items_to_decision": _rate(bucket["items"], bucket["n"]),
            "n": bucket["n"],
            "n_active": bucket["n_active"],
            "n_inactive": bucket["n_inactive"],
        }
    return summary


def run_resolution(*, n_learners=N_LEARNERS, n_seeds=N_SEEDS, seed=E10_SEED,
                   types=LEARNER_TYPES, policies=POLICIES):
    """E10 card body. Mean ± 95% interval across seeds. Does not write a file."""
    children = np.random.SeedSequence(seed).spawn(n_seeds)
    per_seed = []
    for child in children:
        per_seed.append(run_seed(np.random.default_rng(child), n_learners, types, policies))

    rows = []
    for learner_type in types:
        for policy in policies:
            cells = [seed_row[(learner_type, policy)] for seed_row in per_seed]
            rows.append({
                "policy": policy,
                "type": learner_type,
                "simulated": True,
                "learner_population": SIMULATED_POPULATION,
                "false_resolve": mean_interval([c["false_resolve"] for c in cells]),
                "false_not_yet": mean_interval([c["false_not_yet"] for c in cells]),
                "items_to_decision": mean_interval([c["items_to_decision"] for c in cells]),
                "n_per_seed": n_learners,
            })

    def _mean(learner_type, policy, field):
        for row in rows:
            if row["type"] == learner_type and row["policy"] == policy:
                stat = row[field]
                return None if stat is None else stat["mean"]
        return None

    ours = _mean("pattern_copier", "ours", "false_resolve")
    naive = _mean("pattern_copier", "naive", "false_resolve")
    ratio = None if ours is None or naive in (None, 0.0) else ours / naive
    acceptance = {
        "claim": "false-resolve on simulated pattern-copiers: ours <= 1/2 × naive",
        "population": SIMULATED_POPULATION,
        "ours": ours,
        "naive": naive,
        "ratio": ratio,
        "met": bool(ratio is not None and ratio <= 0.5),
    }
    return {
        "id": "E10",
        "title": "Resolution policies on simulated learners",
        "population": "simulated",
        "learner_population": SIMULATED_POPULATION,
        "slice": "simulated learners × {naive, 4-check, ours}; not real students",
        "caveat": CAVEAT,
        "seed": seed,
        "n_seeds": n_seeds,
        "learners_per_type_per_seed": n_learners,
        "n_per_cell": n_learners * n_seeds,
        "budget": {"followups": MAX_FOLLOWUPS, "ghost": 1},
        "parameter_draw": "uniform ±50% around the engine g, s, and p_b, per learner",
        "rows": rows,
        "acceptance": acceptance,
    }
