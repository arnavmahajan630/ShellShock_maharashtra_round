"""Exam simulation on simulated learners (ml_plan/03 §9.3b, E14).

These learners are simulations. They are not real students. Hidden truth is
drawn independently per class. The engine never sees it. Responses use
exposures and a belief rate shifted away from the engine's hand-set numbers.
Item choice for the adaptive arm is ``select_next``. Knowledge updates go
through ``update_p`` and ``next_state``, the same functions the exam runner uses.

Diagnoser noise defaults to the E1 out-of-fold confusion in ``_v1_confusion``.
``confusion="stand-in"`` keeps the old comparison matrix. Learners are simulated.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np

from ml.contracts.classes import DSA_CLASSES, LABELS, MAIN_CLASSES, MISCONCEPTIONS, SECTORS
from ml.contracts.params import (
    DIAGNOSIS_ACTIVE_P,
    ELO_K,
    ELO_START,
    EXAM_DSA_PRIOR,
    EXAM_LENGTH,
    EXAM_MIN_EXPOSURE,
    POPULATION_PRIOR,
    SLIP,
    STABLE_P,
)
from ml.eval._v1_confusion import (
    E1_SOURCE,
    STAND_IN_SOURCE,
    corrupt,
    e1_oof_confusion,
    e1_oof_meta,
    labels_for,
    stand_in_confusion,
)
from ml.eval._v1_stats import mean_interval
from ml.exam.select import _allowed, fixed_blueprint, load_pool, pass_probability, select_next
from ml.learner.knowledge import update_p
from ml.learner.state_machine import next_state

# 03 §9.3b.
P_ACTIVE_MAIN = 0.15
P_ACTIVE_DSA = 0.25
P_CARRY_STABLE = 0.30
# Entering P for a main-game STABLE class. 03 says STABLE means P < 0.15 and
# does not name the value at exam start. 0.12 sits in that band, above the
# mastery line, so a recheck can still move the state.
STABLE_ENTER_P = 0.12

# "skill noise" is named in §9.3b and not given a number. This band is the stand-in.
SKILL_LO = 0.85
SKILL_HI = 1.0
PB_LO = 0.4
PB_HI = 0.9
EXPOSURE_LO = 0.5
EXPOSURE_HI = 1.5

POLICIES = ("adaptive", "fixed", "random")
N_CODING, N_TRACE = EXAM_LENGTH["full"]
N_ITEMS = N_CODING + N_TRACE

E14_SEED = 1414
N_LEARNERS = 2000
N_SEEDS = 20

SIMULATED_POPULATION = "simulated learners (not real students)"
CAVEAT = (
    "Simulated learners follow our assumptions; this shows the selection logic "
    "works as designed, not that it is optimal for real learners."
)


def draw_truth(rng):
    """Hidden profile. STABLE main classes are truly inactive (03 §9.3b)."""
    active = {}
    for cls in MISCONCEPTIONS:
        rate = P_ACTIVE_DSA if cls in DSA_CLASSES else P_ACTIVE_MAIN
        active[cls] = bool(rng.random() < rate)
    stable = []
    if float(rng.random()) < P_CARRY_STABLE:
        inactive_main = [cls for cls in MAIN_CLASSES if not active[cls]]
        if inactive_main:
            k = int(rng.integers(1, 4))
            k = min(k, len(inactive_main))
            picked = rng.choice(inactive_main, size=k, replace=False)
            stable = [str(cls) for cls in np.atleast_1d(picked)]
    return {"active": active, "stable": stable}


def draw_response(rng):
    """Learner-side noise, not the numbers the engine updates with."""
    slip = float(SLIP) * float(rng.uniform(EXPOSURE_LO, EXPOSURE_HI))
    if slip < 1e-6:
        slip = 1e-6
    if slip > 1.0 - 1e-6:
        slip = 1.0 - 1e-6
    return {
        "e_scale": float(rng.uniform(EXPOSURE_LO, EXPOSURE_HI)),
        "p_b_sim": float(rng.uniform(PB_LO, PB_HI)),
        "skill": float(rng.uniform(SKILL_LO, SKILL_HI)),
        "slip": slip,
    }


def make_session(learner_index, policy, stable):
    """Engine state at exam start. Truth is not written into P or the state."""
    sectors = list(SECTORS)
    stable_set = set(stable)
    p = {}
    states = {}
    since = {}
    for cls in MISCONCEPTIONS:
        if cls in stable_set:
            states[cls] = "STABLE"
            p[cls] = STABLE_ENTER_P
            since[cls] = "main game"
        elif cls in DSA_CLASSES:
            states[cls] = "UNSEEN"
            p[cls] = EXAM_DSA_PRIOR
        else:
            states[cls] = "UNSEEN"
            p[cls] = POPULATION_PRIOR
    return {
        "learner_id": f"sim-{learner_index}",
        "exam_id": policy,
        "p": p,
        "states": states,
        "theta": {sector: float(ELO_START) for sector in sectors},
        "sectors": sectors,
        "covered": set(),
        "used": set(),
        "coding_n": {sector: 0 for sector in sectors},
        "n_coding": 0,
        "n_trace": 0,
        "last": None,
        "tested": set(),
        "stable_since": since,
    }


def _targets(item, fail_k):
    keys = []
    for cls, raw in item["exposure"].items():
        if cls in MISCONCEPTIONS and float(raw) >= EXAM_MIN_EXPOSURE:
            keys.append(cls)
    if fail_k in MISCONCEPTIONS and fail_k not in keys:
        keys.append(fail_k)
    return keys


def _move(session, item, cls, new_p, passed):
    state = session["states"][cls]
    exposure = item["exposure"].get(cls)
    exposed = exposure is not None and float(exposure) >= EXAM_MIN_EXPOSURE
    ghost_failed = (not passed) and state in ("STABLE", "MASTERED") and exposed
    new_state = next_state(
        state,
        new_p,
        exam_update=True,
        exam_passed=bool(passed) and exposed,
        exposure=exposure,
        ghost_failed=ghost_failed,
    )
    session["p"][cls] = new_p
    session["states"][cls] = new_state
    session["tested"].add(cls)


def apply_coding(session, item, passed, fail_k):
    """Same branches as ``ml.exam.run`` for a coding item. No latent class."""
    for cls in _targets(item, fail_k):
        p = session["p"][cls]
        exposure = item["exposure"].get(cls)
        if (not passed) and fail_k == cls:
            new_p = update_p(p, "signature_failure", exposure=exposure)
        else:
            new_p = update_p(p, "exam_code", correct=passed)
        _move(session, item, cls, new_p, passed)


def apply_trace(session, item, passed):
    """Trace items use the exam-trace row, which is the belief-MCQ guess and slip."""
    for cls, raw in item["exposure"].items():
        if cls not in MISCONCEPTIONS or float(raw) < EXAM_MIN_EXPOSURE:
            continue
        new_p = update_p(
            session["p"][cls],
            "exam_trace",
            correct=passed,
            belief=item.get("belief"),
            class_id=cls,
        )
        _move(session, item, cls, new_p, passed)


def _serve(session, item, passed):
    sector = item["sector"]
    expected = pass_probability(item["difficulty"], session["theta"][sector])
    outcome = 1.0 if passed else 0.0
    session["theta"][sector] = session["theta"][sector] + ELO_K * (outcome - expected)
    session["used"].add(item["item_id"])
    session["covered"].add(sector)
    session["last"] = sector
    if item["kind"] == "coding":
        session["n_coding"] += 1
        session["coding_n"][sector] = session["coding_n"].get(sector, 0) + 1
    else:
        session["n_trace"] += 1


def coding_outcome(rng, item, truth, params, matrix, labels=None):
    """Fail with a signature at rate ``e_ik_sim`` when k is active; else pass × skill.

    ``e_ik_sim`` is the authored exposure times a draw in ±50%. The engine's
    update still uses the authored exposure.
    """
    risks = []
    scale = params["e_scale"]
    for cls, raw in item["exposure"].items():
        if not truth["active"].get(cls):
            continue
        e = float(raw) * scale
        if e < 0.0:
            e = 0.0
        elif e > 1.0:
            e = 1.0
        if e > 0.0:
            risks.append((cls, e))
    max_e = max((e for _, e in risks), default=0.0)
    p_pass = (1.0 - max_e) * params["skill"]
    if p_pass < 0.0:
        p_pass = 0.0
    elif p_pass > 1.0:
        p_pass = 1.0
    if float(rng.random()) < p_pass:
        return True, None
    if not risks:
        return False, None
    weights = np.asarray([e for _, e in risks], dtype=float)
    weights /= weights.sum()
    idx = int(rng.choice(len(risks), p=weights))
    true_k = risks[idx][0]
    drawn = corrupt(rng, true_k, matrix, labels)
    # CORRECT or OTHER on a failed attempt is not a signature. The tests still failed.
    if drawn not in MISCONCEPTIONS:
        return False, None
    return False, drawn


def trace_correct(rng, item, truth, params):
    """Belief answer with ``p_b_sim`` when an active class is in the belief map.

    The belief option is not the correct one (the bank's exposure rule).
    Otherwise the learner is correct with probability ``1 - slip``.
    """
    belief = item.get("belief") or {}
    active_belief = any(truth["active"].get(cls) for cls in belief if cls in MISCONCEPTIONS)
    if active_belief:
        return bool(rng.random() >= params["p_b_sim"])
    return bool(rng.random() < (1.0 - params["slip"]))


def _pick(policy, pool, session, blueprint, rng):
    if policy == "fixed":
        n = session["n_coding"] + session["n_trace"]
        if n >= len(blueprint):
            return None
        return blueprint[n]
    if policy == "adaptive":
        picked = select_next(pool, session, N_CODING, N_TRACE)
        if picked is None:
            return None
        return picked[0]
    if policy == "random":
        # Same composition constraints as the adaptive rule, uniform among the
        # items that still fit. ``_allowed`` is the engine's constraint check.
        candidates = [
            item for item in pool
            if _allowed(
                item, session, N_CODING, N_TRACE, session["sectors"],
                require_consecutive=True, require_coverage=True,
            )
        ]
        if not candidates:
            return None
        return candidates[int(rng.integers(0, len(candidates)))]
    raise ValueError(f"unknown exam policy {policy!r}")


def profile_f1(p_map, truth):
    """Set F1 of classes with P(A) >= 0.5 against the hidden active set."""
    tp = fp = fn = 0
    for cls in MISCONCEPTIONS:
        predicted = p_map[cls] >= DIAGNOSIS_ACTIVE_P
        actual = truth["active"][cls]
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif actual and not predicted:
            fn += 1
    if tp == 0:
        return 1.0 if fp == 0 and fn == 0 else 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2.0 * precision * recall / (precision + recall)


def brier_score(p_map, truth):
    """Mean squared error of P(A_k) against the hidden 0/1 truth, over 17 classes."""
    total = 0.0
    for cls in MISCONCEPTIONS:
        y = 1.0 if truth["active"][cls] else 0.0
        diff = p_map[cls] - y
        total += diff * diff
    return total / len(MISCONCEPTIONS)


def _found(p_map, truth):
    for cls in MISCONCEPTIONS:
        if truth["active"][cls] and p_map[cls] >= DIAGNOSIS_ACTIVE_P:
            return True
    return False


def run_exam_episode(rng, truth, params, policy, pool, blueprint, matrix, learner_index, labels=None):
    """One simulated learner, one policy, up to 5 coding + 5 trace."""
    session = make_session(learner_index, policy, truth["stable"])
    has_active = any(truth["active"].values())
    curve = []
    first = None
    while session["n_coding"] < N_CODING or session["n_trace"] < N_TRACE:
        item = _pick(policy, pool, session, blueprint, rng)
        if item is None:
            break
        if item["kind"] == "coding":
            passed, fail_k = coding_outcome(rng, item, truth, params, matrix, labels)
            apply_coding(session, item, passed, fail_k)
        else:
            passed = trace_correct(rng, item, truth, params)
            apply_trace(session, item, passed)
        _serve(session, item, passed)
        curve.append(profile_f1(session["p"], truth))
        if has_active and first is None and _found(session["p"], truth):
            first = len(curve)
        if len(curve) >= N_ITEMS:
            break
    if not curve:
        curve.append(profile_f1(session["p"], truth))
    while len(curve) < N_ITEMS:
        curve.append(curve[-1])
    re_hit = sum(1 for cls in truth["stable"] if cls in session["tested"])
    if has_active:
        items_to = first if first is not None else N_ITEMS
    else:
        items_to = None
    return {
        "curve": curve,
        "final_f1": curve[-1],
        "brier": brier_score(session["p"], truth),
        "recheck_hit": re_hit,
        "recheck_n": len(truth["stable"]),
        "items_to": items_to,
        "n_administered": session["n_coding"] + session["n_trace"],
        "simulated": True,
    }


def run_seed(rng, n_learners, pool, blueprint, matrix, policies=POLICIES, labels=None):
    """One seed. Truth is drawn once per learner and reused across policies."""
    acc = {
        policy: {
            "curve": np.zeros(N_ITEMS, dtype=float),
            "f1": 0.0,
            "brier": 0.0,
            "re_hit": 0,
            "re_n": 0,
            "find_sum": 0.0,
            "find_n": 0,
            "n": 0,
            "administered": 0,
        }
        for policy in policies
    }
    for index in range(n_learners):
        truth = draw_truth(rng)
        params = draw_response(rng)
        for policy in policies:
            out = run_exam_episode(
                rng, truth, params, policy, pool, blueprint, matrix, index, labels,
            )
            bucket = acc[policy]
            bucket["curve"] += np.asarray(out["curve"], dtype=float)
            bucket["f1"] += out["final_f1"]
            bucket["brier"] += out["brier"]
            bucket["re_hit"] += out["recheck_hit"]
            bucket["re_n"] += out["recheck_n"]
            bucket["administered"] += out["n_administered"]
            bucket["n"] += 1
            if out["items_to"] is not None:
                bucket["find_sum"] += out["items_to"]
                bucket["find_n"] += 1
    summary = {}
    for policy, bucket in acc.items():
        n = bucket["n"]
        summary[policy] = {
            "curve": (bucket["curve"] / n).tolist(),
            "final_f1": bucket["f1"] / n,
            "brier": bucket["brier"] / n,
            "stable_recheck": (bucket["re_hit"] / bucket["re_n"]) if bucket["re_n"] else None,
            "items_to_first_finding": (
                bucket["find_sum"] / bucket["find_n"] if bucket["find_n"] else None
            ),
            "mean_items": bucket["administered"] / n,
        }
    return summary


def _confusion(matrix, confusion):
    """``e1`` is the default. ``stand-in`` is the comparison flag. An explicit matrix wins."""
    if confusion not in ("e1", "stand-in"):
        raise ValueError(f"unknown confusion {confusion!r}")
    if matrix is not None:
        labels = labels_for(matrix)
        if labels == list(LABELS):
            source, note = E1_SOURCE, "Caller supplied the matrix. Same one for every policy."
        else:
            source, note = STAND_IN_SOURCE, "Caller supplied the stand-in. Same one for every policy."
        return matrix, labels, source, note, None
    if confusion == "stand-in":
        return (
            stand_in_confusion(),
            list(MISCONCEPTIONS),
            STAND_IN_SOURCE,
            "Comparison arm. Not the E1 measurement. Same matrix for every policy.",
            None,
        )
    matrix = e1_oof_confusion()
    meta = e1_oof_meta()
    note = (
        "P(argmax | dataset label) on TRAIN out-of-fold, after the shipped temperature and mask. "
        "A draw of CORRECT or OTHER on a failed attempt is a failure with no signature. "
        "Same matrix for every policy. The learners are simulated."
    )
    return matrix, list(LABELS), E1_SOURCE, note, meta


def run_exam_sim(*, n_learners=N_LEARNERS, n_seeds=N_SEEDS, seed=E14_SEED, policies=POLICIES,
                 pool=None, matrix=None, confusion="e1"):
    """E14 card body. Mean ± 95% interval across seeds. Does not write a file.

    ``confusion="stand-in"`` uses the pre-E1 comparison matrix.
    """
    if pool is None:
        pool = load_pool(strings=True)
    by_id = {item["item_id"]: item for item in pool}
    blueprint_ids = fixed_blueprint(strings=True)
    blueprint = [by_id[item_id] for item_id in blueprint_ids]
    matrix, labels, source, note, meta = _confusion(matrix, confusion)

    children = np.random.SeedSequence(seed).spawn(n_seeds)
    per_seed = []
    for child in children:
        per_seed.append(run_seed(
            np.random.default_rng(child), n_learners, pool, blueprint, matrix, policies, labels,
        ))

    def col(policy, field):
        return mean_interval([row[policy][field] for row in per_seed])

    curves = {}
    rows = []
    for policy in policies:
        series = []
        for item_i in range(N_ITEMS):
            series.append(mean_interval([row[policy]["curve"][item_i] for row in per_seed]))
        curves[policy] = series
        rows.append({
            "policy": policy,
            "simulated": True,
            "learner_population": SIMULATED_POPULATION,
            "profile_f1": col(policy, "final_f1"),
            "brier": col(policy, "brier"),
            "stable_recheck": col(policy, "stable_recheck"),
            "items_to_first_finding": col(policy, "items_to_first_finding"),
            "mean_items": col(policy, "mean_items"),
        })

    def _m(policy, field):
        for row in rows:
            if row["policy"] == policy:
                stat = row[field]
                return None if stat is None else stat["mean"]
        return None

    adaptive = _m("adaptive", "profile_f1")
    fixed = _m("fixed", "profile_f1")
    gain = None if adaptive is None or fixed in (None, 0.0) else (adaptive - fixed) / fixed
    acceptance = {
        "claim": "profile-recovery F1 on simulated learners: adaptive > fixed by >= 10% relative",
        "population": SIMULATED_POPULATION,
        "adaptive": adaptive,
        "fixed": fixed,
        "relative_gain": gain,
        "met": bool(gain is not None and gain >= 0.10),
    }
    return {
        "id": "E14",
        "title": "Adaptive exam versus fixed and random on simulated learners",
        "population": "simulated",
        "learner_population": SIMULATED_POPULATION,
        "slice": "simulated learners × {adaptive, fixed blueprint, random}; not real students",
        "caveat": CAVEAT,
        "seed": seed,
        "n_seeds": n_seeds,
        "learners_per_seed": n_learners,
        "n": n_learners * n_seeds,
        "length": {"coding": N_CODING, "trace": N_TRACE},
        "blueprint": blueprint_ids,
        "diagnoser_noise": {
            "source": source,
            "note": note,
            "e1": meta,
            "simulated_learners": True,
        },
        "skill_noise": {"low": SKILL_LO, "high": SKILL_HI, "note": "03 §9.3b names skill noise and does not give a range."},
        "stable_enter_p": STABLE_ENTER_P,
        "items_to_first_finding_censor": (
            "Learners with no hidden active class are left out. "
            "A learner who is never found counts as 10 items."
        ),
        "curves": {
            policy: [
                {"item": i + 1, "profile_f1": stat, "simulated": True}
                for i, stat in enumerate(curves[policy])
            ]
            for policy in policies
        },
        "rows": rows,
        "acceptance": acceptance,
    }


# Re-export so a reader can see the entering band is under the state machine's line.
assert STABLE_ENTER_P < STABLE_P
