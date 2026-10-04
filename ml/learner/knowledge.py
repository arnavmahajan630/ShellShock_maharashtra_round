"""Knowledge updates for P(A), the probability a misconception is active (ml_plan/03 §8.2, ml_plan/05 §4).

One update per observation. There is no ``max(P, posterior)`` write (notes/W0.md decision 19).
Callers start P where they store it (population prior, exam prior); this module does not.

``likelihood_ratio`` is the shared shape. A signature failure uses the problem's exposure
``e_ik`` (or ``FAIL_SIGNATURE_IF_ACTIVE`` when the problem lists none) against
``FAIL_SIGNATURE_IF_NOT``. A matched sentence uses ``TEXT_IF_ACTIVE`` against ``TEXT_IF_NOT``.
Item responses are the same ratio with guess and slip:

    correct:  if_active = g,       if_not = 1 - s
    wrong:    if_active = 1 - g,   if_not = s

``g`` and ``s`` come from ``ITEM_GUESS_SLIP``. ``belief_mcq`` already stores
``(1 - P_BELIEF) * Q_CORRECT``; do not recompute it. A hint rewrites ``g`` for that call
only: ``min(HINT_GUESS_CAP, g + HINT_GUESS_BONUS)``.

If a ratio's denominator is 0, P is returned unchanged.

``update_p(p, kind, ...)`` is the dispatcher. ``kind`` is an item type or one of
``signature_failure``, ``latent_pass``, ``intervention``, ``forgetting``,
``predict_output_trap``. ``predict_output`` whose ``belief`` map contains ``class_id``
resolves to ``predict_output_trap`` and uses the trap row (03 §8.3, 05 §4).

Exam coding, for a later caller (03 §8.5.3, same formulas as §8.2): ``signature_failure``
when this class is the diagnosed failure, ``latent_pass`` when a pass carries ``latent``
for this class, ``exam_code`` when a pass has no latent class (and for a failure that is
not this class's signature). Exam trace items use ``exam_trace``, which is the
``belief_mcq`` row.
"""
from __future__ import annotations

from ml.contracts.classes import MISCONCEPTIONS
from ml.contracts.params import (
    DIAGNOSIS_ACTIVE_P,
    FAIL_SIGNATURE_IF_ACTIVE,
    FAIL_SIGNATURE_IF_NOT,
    FORGET_RATE,
    HINT_GUESS_BONUS,
    HINT_GUESS_CAP,
    ITEM_GUESS_SLIP,
    LATENT_EXPOSURE_FACTOR,
    LEARN_RATE,
    TEXT_IF_ACTIVE,
    TEXT_IF_NOT,
    TEXT_MATCH_P,
)

# kind → ITEM_GUESS_SLIP row. predict_output_trap is the second trap pool (03 §8.3).
ITEM_RESPONSE_KIND = {
    "same_family_code": "same_family_code",
    "transfer_code": "transfer_code",
    "trap": "trap",
    "belief_mcq": "belief_mcq",
    "ghost": "ghost",
    "exam_code": "exam_code",
    "exam_trace": "belief_mcq",
    "mcq": "belief_mcq",
    "predict_output": "belief_mcq",
    "predict_output_trap": "trap",
    "next_state": "belief_mcq",
}


def _merge(event, kwargs):
    if event is None:
        merged = {}
    elif isinstance(event, dict):
        merged = dict(event)
    else:
        raise TypeError("event must be a dict")
    merged.update(kwargs)
    return merged


def likelihood_ratio(p, if_active, if_not):
    """P ← P·if_active / (P·if_active + (1−P)·if_not). Denominator 0 leaves P unchanged."""
    p = float(p)
    numer = p * float(if_active)
    denom = numer + (1.0 - p) * float(if_not)
    if denom == 0.0:
        return p
    return numer / denom


def _exposure(exposure):
    if exposure is None:
        return FAIL_SIGNATURE_IF_ACTIVE
    return float(exposure)


def signature_failure(p, exposure=None):
    """Failed code task diagnosed as this class (03 §8.2). Caller gates on the posterior."""
    return likelihood_ratio(p, _exposure(exposure), FAIL_SIGNATURE_IF_NOT)


def latent_pass(p, exposure=None):
    """Passed code task whose latent class is this one: the same ratio with e_ik scaled."""
    return likelihood_ratio(p, _exposure(exposure) * LATENT_EXPOSURE_FACTOR, FAIL_SIGNATURE_IF_NOT)


def guess_with_hint(g):
    return min(HINT_GUESS_CAP, float(g) + HINT_GUESS_BONUS)


def item_response(p, g, s, *, correct, hint=False):
    g = guess_with_hint(g) if hint else float(g)
    s = float(s)
    if correct:
        return likelihood_ratio(p, g, 1.0 - s)
    return likelihood_ratio(p, 1.0 - g, s)


def intervention(p):
    """Intervention completed: P ← P·(1−ℓ)."""
    return float(p) * (1.0 - LEARN_RATE)


def forget(p, levels=1):
    """Forgetting before a ghost return, once per intervening level."""
    p = float(p)
    n = int(levels)
    if n < 0:
        n = 0
    for _ in range(n):
        p = p + FORGET_RATE * (1.0 - p)
    return p


def resolve_kind(kind, event=None, **kwargs):
    """Item kind after the predict_output trap rule. ``predict_output_trap`` is explicit."""
    ev = _merge(event, kwargs)
    if kind == "predict_output":
        belief = ev.get("belief") or {}
        class_id = ev.get("class_id")
        if class_id is not None and class_id in belief:
            return "predict_output_trap"
    return kind


def _correct(ev):
    if "correct" in ev:
        return bool(ev["correct"])
    if "passed" in ev:
        return bool(ev["passed"])
    raise ValueError("item response needs correct")


def _named_response(p, row, ev):
    g, s = ITEM_GUESS_SLIP[row]
    return item_response(p, g, s, correct=_correct(ev), hint=bool(ev.get("hint", False)))


def _debug_line(p, ev):
    if ev.get("planted") is None:
        return float(p)
    g, s = ITEM_GUESS_SLIP["debug_line"]
    return item_response(p, g, s, correct=_correct(ev), hint=bool(ev.get("hint", False)))


def _fix_bug(p, ev):
    if "planted" not in ev:
        raise ValueError("fix_bug needs planted")
    planted = ev["planted"]
    class_id = ev.get("class_id")
    hint = bool(ev.get("hint", False))
    g, s = ITEM_GUESS_SLIP["fix_bug"]
    tests_pass = ev.get("tests_pass", ev.get("passed", False))
    if tests_pass:
        if class_id is not None and class_id != planted:
            return float(p)
        return item_response(p, g, s, correct=True, hint=hint)
    top1 = ev.get("top1")
    if top1 == planted:
        if class_id is not None and class_id != planted:
            return float(p)
        return item_response(p, g, s, correct=False, hint=hint)
    posterior = ev.get("posterior")
    if (top1 in MISCONCEPTIONS and top1 != planted and posterior is not None
            and posterior >= DIAGNOSIS_ACTIVE_P):
        if class_id is None or class_id == top1:
            return signature_failure(p, ev.get("exposure", ev.get("e_ik")))
        return float(p)
    return float(p)


def _text(p, ev):
    if ev.get("unsure") or ev.get("status") == "unsure":
        return float(p)
    if "matched" in ev:
        matched = bool(ev["matched"])
    else:
        matched = ev.get("status") == "matched"
    if not matched:
        return float(p)
    p_text = ev.get("p_text")
    if p_text is None or float(p_text) < TEXT_MATCH_P:
        return float(p)
    return likelihood_ratio(p, TEXT_IF_ACTIVE, TEXT_IF_NOT)


def update_p(p, kind, event=None, **kwargs):
    """Apply one knowledge update. ``kind`` selects the row; see the module docstring."""
    ev = _merge(event, kwargs)
    resolved = resolve_kind(kind, ev)
    if resolved == "signature_failure":
        return signature_failure(p, ev.get("exposure", ev.get("e_ik")))
    if resolved == "latent_pass":
        return latent_pass(p, ev.get("exposure", ev.get("e_ik")))
    if resolved == "intervention":
        return intervention(p)
    if resolved == "forgetting":
        return forget(p, ev.get("levels", 1))
    if resolved == "debug_line":
        return _debug_line(p, ev)
    if resolved == "fix_bug":
        return _fix_bug(p, ev)
    if resolved == "complete_snippet":
        if not ev.get("passed", ev.get("correct", False)):
            return float(p)
        g, s = ITEM_GUESS_SLIP["same_family_code"]
        return item_response(p, g, s, correct=True, hint=bool(ev.get("hint", False)))
    if resolved in ("reasoning", "why"):
        return _text(p, ev)
    if resolved in ITEM_RESPONSE_KIND:
        return _named_response(p, ITEM_RESPONSE_KIND[resolved], ev)
    raise ValueError(f"unknown knowledge update {kind!r}")
