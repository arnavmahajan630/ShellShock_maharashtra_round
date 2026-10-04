"""Next practice item, and how an answer is scored (ml_plan/05 §4, §5).

Selection uses the exam's per-class expected information gain (03 §8.5.4)
and none of the exam's composition rules: no sector quotas, no opening-item
rule, no coverage or ghost bonus. Options stay in the order they have in the
item. The public item has no ``correct``, ``belief``, or ``explain``.

``ask_reason`` is true only when the chosen option is the belief answer of
two or more classes.
"""
from __future__ import annotations

from ml.bayes.likelihood import likelihood
from ml.contracts.classes import CLASS_INFO, LABELS, MISCONCEPTIONS
from ml.contracts.params import POPULATION_PRIOR
from ml.contracts.schemas import ClassUpdate, PublicQuizItem
from ml.exam.select import belief_exposure, class_eigs, tie_key
from ml.learner.knowledge import update_p


def next_item(learner_state, items, *, type=None, concept=None):
    """``(public item, reason)``. ``type`` and ``concept`` are optional filters."""
    state = learner_state or {}
    type_filter = type
    asked = set(state.get("asked") or [])
    p_active = _p_map(state)
    learner_id = str(state.get("learner_id") or "quiz")
    best = None
    best_key = None
    best_parts = None
    for item in items:
        if not _matches(item, type_filter, concept, asked):
            continue
        parts = _parts(item, p_active)
        eig = sum(parts.values())
        key = (eig, -tie_key(learner_id, "quiz", item["item_id"]))
        if best is None or key > best_key:
            best = item
            best_key = key
            best_parts = parts
    if best is None:
        raise ValueError("no quiz item matches the filters")
    return _public(best), _reason(best, best_parts or {})


def score_answer(learner_state, item, answer):
    """Score one ``mcq``, ``predict_output``, or ``next_state`` answer.

    Likelihood grades the option. ``update_p`` moves P(A). Passing ``belief``
    and ``class_id`` lets a ``predict_output`` item use the trap row when the
    class is in the belief map.
    """
    if item.get("type") not in ("mcq", "predict_output", "next_state"):
        raise ValueError(f"cannot score quiz type {item.get('type')!r}")
    options = item.get("options") or []
    correct = item.get("correct")
    belief = dict(item.get("belief") or {})
    table = likelihood(options, correct, belief)
    answer_s = str(answer)
    known = set(table[LABELS[0]])
    is_correct = answer_s == str(correct) and answer_s in known
    p_active = _p_map(learner_state or {})
    updates = []
    for cls in _classes(item):
        before = float(p_active[cls]) if cls in p_active else POPULATION_PRIOR
        after = update_p(
            before,
            item["type"],
            correct=is_correct,
            belief=belief,
            class_id=cls,
        )
        updates.append(ClassUpdate(cls=cls, p_before=before, p_after=after).model_dump(by_alias=True))
    return {
        "correct": is_correct,
        "correct_answer": str(correct),
        "explain": item.get("explain") or "",
        "updates": updates,
        "ask_reason": answer_asks_reason(belief, answer_s),
    }


def answer_asks_reason(belief, chosen):
    """True when ``chosen`` is the belief answer of at least two classes."""
    text = str(chosen)
    return sum(1 for value in (belief or {}).values() if str(value) == text) >= 2


def _matches(item, type_filter, concept, asked):
    if item.get("item_id") in asked:
        return False
    if type_filter is not None and item.get("type") != type_filter:
        return False
    if concept is not None and item.get("concept") != concept:
        return False
    return True


def _parts(item, p_active):
    options = item.get("options") or []
    correct = item.get("correct")
    belief = item.get("belief") or {}
    if not options or correct is None:
        return {}
    try:
        exposure = belief_exposure(options, correct, belief)
    except ValueError:
        return {}
    filled = dict(p_active)
    for cls in exposure:
        if cls not in filled:
            filled[cls] = POPULATION_PRIOR
    return class_eigs(filled, exposure)


def _classes(item):
    ordered = []
    seen = set()
    for cls in list(item.get("classes") or []) + list((item.get("belief") or {}).keys()):
        if cls in MISCONCEPTIONS and cls not in seen:
            seen.add(cls)
            ordered.append(cls)
    return ordered


def _p_map(learner_state):
    if not isinstance(learner_state, dict):
        return {}
    if "misconceptions" in learner_state:
        out = {}
        for cls, entry in learner_state["misconceptions"].items():
            if isinstance(entry, dict):
                out[cls] = float(entry.get("p_active", POPULATION_PRIOR))
            else:
                out[cls] = float(entry)
        return out
    if any(key in learner_state for key in ("p_active", "states", "learner_id", "asked")):
        return {cls: float(value) for cls, value in (learner_state.get("p_active") or {}).items()}
    return {
        cls: float(value)
        for cls, value in learner_state.items()
        if cls in MISCONCEPTIONS
    }


def _public(item):
    data = {
        "item_id": item["item_id"],
        "type": item["type"],
        "concept": item["concept"],
        "code": item.get("code") or "",
        "question": item["question"],
        "options": list(item.get("options") or []),
        "difficulty": int(item.get("difficulty") or 1),
    }
    return PublicQuizItem.model_validate(data).model_dump()


def _reason(item, parts):
    if not parts and item.get("type") == "reasoning":
        return f"{item['item_id']} — reasoning item; the sentence is what separates the classes."
    ranked = sorted(parts, key=lambda cls: (-parts[cls], cls))
    names = [CLASS_INFO[cls]["name"] for cls in ranked[:2]]
    if len(names) == 2:
        about = f"{names[0]} and {names[1]}"
    elif names:
        about = names[0]
    else:
        about = "no class above the exposure floor"
    return f"{item['item_id']} — most informative about {about}."
