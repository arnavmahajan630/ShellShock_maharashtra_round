"""Scripted exam runner (ml_plan/03 §8.5.3, §8.5.5).

The diagnoser is injected. This module does not call the real model, and it
does not print an item, a diagnosis, or a probability. The only return value
is the debrief from ``report.build_report``.

Knowledge moves only through ``update_p``. State moves only through ``next_state``.
A trace answer is scored with ``likelihood``; the P(A) update is the ``exam_trace``
row, which is the belief-MCQ guess and slip.
"""
from __future__ import annotations

import json

from ml.bayes.likelihood import likelihood
from ml.contracts.classes import DSA_CLASSES, MISCONCEPTIONS, PLANETS, SECTORS, TWIN_SETS
from ml.contracts.params import (
    DIAGNOSIS_ACTIVE_P,
    ELO_K,
    ELO_PLANETS_BONUS,
    ELO_START,
    EXAM_DSA_PRIOR,
    EXAM_LENGTH,
    EXAM_MIN_EXPOSURE,
    POPULATION_PRIOR,
)
from ml.exam.report import build_report
from ml.exam.select import (
    fixed_blueprint,
    item_eig,
    load_pool,
    pass_probability,
    select_next,
)
from ml.learner.knowledge import update_p
from ml.learner.state_machine import next_state


def run_exam(
    *,
    learner_id,
    exam_id,
    answers,
    diagnoser,
    p_active=None,
    states=None,
    mode="demo",
    adaptive=True,
    planets_complete=False,
    completed_planets=None,
    strings=True,
    time_used_s=0,
    stable_since=None,
):
    """Run one scripted exam and return its ``ExamReport``.

    ``answers`` is a callable ``(item) -> answer``, a list consumed in order, or
    a dict keyed by item id. A coding answer is source text (or ``{"code": ...}``).
    A trace answer is the chosen option (or ``{"answer": ...}``). The callable is
    the script, so it may read ``correct`` on a trace item. ``None`` ends the exam;
    that item is unobserved.

    ``diagnoser(problem, code)`` returns ``status``, ``top``, ``posterior``, and
    ``latent``. ``status == "correct"`` or a latent class means the tests passed.
    """
    if mode not in EXAM_LENGTH:
        raise ValueError(f"unknown exam mode {mode!r}")
    n_coding, n_trace = EXAM_LENGTH[mode]
    session = _session(
        learner_id=learner_id,
        exam_id=exam_id,
        p_active=p_active,
        states=states,
        planets_complete=planets_complete or _all_planets(completed_planets),
        strings=strings,
        time_used_s=time_used_s,
        stable_since=stable_since,
    )
    pool = load_pool(strings=strings)
    planned = None
    if not adaptive:
        planned = _plan_blueprint(pool, fixed_blueprint(strings=strings), n_coding, n_trace)

    index = 0
    while session["n_coding"] < n_coding or session["n_trace"] < n_trace:
        if planned is not None:
            if index >= len(planned):
                break
            item = planned[index]
            reason = "fixed blueprint"
            eig = item_eig(session["p"], item)
        else:
            picked = _select(pool, session, n_coding, n_trace)
            if picked is None:
                break
            item, reason, eig = picked
        answer = _next_answer(answers, item, index)
        if answer is None:
            break
        _apply(session, item, answer, diagnoser, reason, eig)
        index += 1
    return build_report(session)


def _select(pool, session, n_coding, n_trace):
    return select_next(pool, session, n_coding, n_trace)


def _session(*, learner_id, exam_id, p_active, states, planets_complete, strings, time_used_s, stable_since):
    given_p = dict(p_active or {})
    given_states = dict(states or {})
    sectors = [sector for sector in SECTORS if strings or sector != "strings"]
    p = {}
    state = {}
    for cls in MISCONCEPTIONS:
        state[cls] = given_states.get(cls, "UNSEEN")
        p[cls] = _starting_p(cls, given_p, state[cls])
    bonus = ELO_PLANETS_BONUS if planets_complete else 0
    theta = {sector: float(ELO_START + bonus) for sector in sectors}
    return {
        "learner_id": learner_id,
        "exam_id": exam_id,
        "p": p,
        "p0": dict(p),
        "states": state,
        "state0": dict(state),
        "theta": dict(theta),
        "theta0": dict(theta),
        "sectors": sectors,
        "covered": set(),
        "used": set(),
        "coding_n": {sector: 0 for sector in sectors},
        "n_coding": 0,
        "n_trace": 0,
        "last": None,
        "tested": set(),
        "ambiguous": {},
        "evidence": {cls: [] for cls in MISCONCEPTIONS},
        "administered": [],
        "stable_since": dict(stable_since or {}),
        "time_used_s": int(time_used_s),
        "strings": strings,
    }


def _starting_p(cls, given_p, state):
    if cls in given_p:
        return float(given_p[cls])
    if cls in DSA_CLASSES and state == "UNSEEN":
        return EXAM_DSA_PRIOR
    return POPULATION_PRIOR


def _all_planets(completed):
    if not completed:
        return False
    return set(PLANETS) <= set(completed)


def _plan_blueprint(pool, ids, n_coding, n_trace):
    by_id = {item["item_id"]: item for item in pool}
    missing = [item_id for item_id in ids if item_id not in by_id]
    if missing:
        raise KeyError(f"blueprint item not in the pool: {missing[0]}")
    picked = []
    coding = trace = 0
    for item_id in ids:
        item = by_id[item_id]
        if item["kind"] == "coding" and coding < n_coding:
            picked.append(item)
            coding += 1
        elif item["kind"] == "trace" and trace < n_trace:
            picked.append(item)
            trace += 1
        if coding >= n_coding and trace >= n_trace:
            break
    return picked


def _next_answer(answers, item, index):
    if answers is None:
        return None
    if callable(answers):
        return answers(item)
    if isinstance(answers, dict):
        return answers.get(item["item_id"])
    if index >= len(answers):
        return None
    return answers[index]


def _apply(session, item, answer, diagnoser, reason, eig):
    if item["kind"] == "coding":
        code = answer["code"] if isinstance(answer, dict) else str(answer)
        if diagnoser is None:
            raise TypeError("a coding item needs a diagnoser")
        raw = diagnoser(dict(item["problem"]), code)
        parsed = _parse_diagnosis(raw)
        _apply_coding(session, item, parsed, raw)
        passed = parsed["passed"]
        outcome = _coding_outcome(parsed)
    else:
        choice = answer["answer"] if isinstance(answer, dict) else answer
        passed = _apply_trace(session, item, choice)
        outcome = "pass" if passed else "wrong"
    _elo(session, item, passed)
    _serve(session, item)
    session["administered"].append({
        "item_id": item["item_id"],
        "sector": item["sector"],
        "passed": passed,
        "eig_bits": float(eig),
        "reason": reason,
        "outcome": outcome,
    })


def _serve(session, item):
    session["used"].add(item["item_id"])
    session["covered"].add(item["sector"])
    session["last"] = item["sector"]
    if item["kind"] == "coding":
        session["n_coding"] += 1
        session["coding_n"][item["sector"]] = session["coding_n"].get(item["sector"], 0) + 1
    else:
        session["n_trace"] += 1


def _elo(session, item, passed):
    sector = item["sector"]
    expected = pass_probability(item["difficulty"], session["theta"][sector])
    outcome = 1.0 if passed else 0.0
    session["theta"][sector] = session["theta"][sector] + ELO_K * (outcome - expected)


def _parse_diagnosis(raw):
    raw = raw or {}
    status = raw.get("status") or "fail"
    top = list(raw.get("top") or [])
    posterior = dict(raw.get("posterior") or {})
    latent = raw.get("latent")
    latent_class = _latent_class(latent)
    top1, top1_p = _top1(top, posterior)
    passed = status == "correct" or latent_class is not None
    fail_k = None
    if not passed and top1 in MISCONCEPTIONS and top1_p >= DIAGNOSIS_ACTIVE_P:
        fail_k = top1
    return {
        "status": status,
        "top": top,
        "posterior": posterior,
        "latent": latent,
        "twin_set": raw.get("twin_set"),
        "passed": passed,
        "fail_k": fail_k,
        "latent_class": latent_class if passed else None,
    }


def _latent_class(latent):
    if isinstance(latent, str):
        return latent
    if isinstance(latent, dict):
        return latent.get("class", latent.get("cls"))
    return None


def _top1(top, posterior):
    if not top:
        return None, 0.0
    first = top[0]
    if isinstance(first, str):
        return first, float(posterior.get(first, 0.0))
    if isinstance(first, dict):
        cls = first.get("id", first.get("class"))
        if "p" in first:
            return cls, float(first["p"])
        return cls, float(posterior.get(cls, 0.0))
    return None, 0.0


def _coding_outcome(parsed):
    if parsed["passed"]:
        return "pass"
    if parsed["fail_k"]:
        return f"fail_{parsed['fail_k']}"
    return "fail"


def _apply_coding(session, item, parsed, raw):
    targets = _targets(item, parsed["fail_k"], parsed["latent_class"])
    ambiguous = _ambiguous_members(parsed)
    snapshot = _snapshot(raw)
    seen = set()
    for cls in targets + ambiguous:
        if cls in seen or cls not in MISCONCEPTIONS:
            continue
        seen.add(cls)
        if cls in targets:
            _update_coding_class(session, item, cls, parsed)
            session["tested"].add(cls)
        if cls in ambiguous:
            session["ambiguous"][cls] = parsed["twin_set"]
        session["evidence"][cls].append({
            "item_id": item["item_id"],
            "diagnosis": snapshot,
        })


def _targets(item, fail_k, latent_class):
    keys = []
    for cls, raw in item["exposure"].items():
        if cls in MISCONCEPTIONS and float(raw) >= EXAM_MIN_EXPOSURE:
            keys.append(cls)
    for extra in (fail_k, latent_class):
        if extra in MISCONCEPTIONS and extra not in keys:
            keys.append(extra)
    return keys


def _ambiguous_members(parsed):
    if parsed["status"] != "ambiguous" and not parsed["twin_set"]:
        return []
    members = []
    twin = parsed["twin_set"]
    if twin in TWIN_SETS:
        members.extend(TWIN_SETS[twin]["members"])
    for entry in (parsed["top"] or [])[:2]:
        if isinstance(entry, str):
            members.append(entry)
        elif isinstance(entry, dict):
            members.append(entry.get("id") or entry.get("class"))
    return [cls for cls in members if cls in MISCONCEPTIONS]


def _snapshot(raw):
    try:
        return json.loads(json.dumps(raw))
    except TypeError:
        return {
            "status": raw.get("status"),
            "top": raw.get("top"),
            "posterior": raw.get("posterior"),
            "latent": raw.get("latent"),
        }


def _update_coding_class(session, item, cls, parsed):
    p = session["p"][cls]
    state = session["states"][cls]
    exposure = item["exposure"].get(cls)
    passed = parsed["passed"]
    if passed and parsed["latent_class"] == cls:
        new_p = update_p(p, "latent_pass", exposure=exposure)
    elif not passed and parsed["fail_k"] == cls:
        new_p = update_p(p, "signature_failure", exposure=exposure)
    else:
        new_p = update_p(p, "exam_code", correct=passed)
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


def _apply_trace(session, item, choice):
    """Score the option with §6.2, then update each exposed class with ``exam_trace``."""
    table = likelihood(item["options"], item["correct"], item["belief"])
    answer_s = str(choice)
    is_correct = answer_s == str(item["correct"])
    for cls, raw in item["exposure"].items():
        if cls not in MISCONCEPTIONS or float(raw) < EXAM_MIN_EXPOSURE:
            continue
        p = session["p"][cls]
        state = session["states"][cls]
        new_p = update_p(
            p, "exam_trace", correct=is_correct, belief=item["belief"], class_id=cls,
        )
        exposed = float(raw) >= EXAM_MIN_EXPOSURE
        ghost_failed = (not is_correct) and state in ("STABLE", "MASTERED") and exposed
        new_state = next_state(
            state,
            new_p,
            exam_update=True,
            exam_passed=is_correct and exposed,
            exposure=float(raw),
            ghost_failed=ghost_failed,
        )
        session["p"][cls] = new_p
        session["states"][cls] = new_state
        session["tested"].add(cls)
        session["evidence"][cls].append({
            "item_id": item["item_id"],
            "answer": answer_s,
            "p_answer": table[cls].get(answer_s, 0.0),
        })
    return is_correct
