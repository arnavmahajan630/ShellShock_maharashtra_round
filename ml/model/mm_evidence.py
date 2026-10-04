"""Evidence fusion module for multimodal attempts.

Combines deterministic ground truth with judge cues and behavioral signals
into a posterior distribution across the problem's candidate classes.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from ml.contracts.classes import CLASS_INFO, LABELS, band

LIKELIHOODS_FILE = Path(__file__).resolve().parent / "mm_likelihoods.json"
try:
    LIKELIHOODS = json.loads(LIKELIHOODS_FILE.read_text(encoding="utf-8"))
except Exception:
    LIKELIHOODS = {
        "deterministic_match": 0.90,
        "deterministic_divergence": 0.75,
        "judge_cue_strong": 0.85,
        "judge_cue_medium": 0.70,
        "hedge_penalty": 0.08,
        "correct_path_prob": 0.95,
    }


def compute_posterior(
    candidates: List[str],
    deterministic_misc: Optional[str],
    cues: List[Any],
    signals: Any,
    is_correct: bool,
    needs_probe: bool = False,
    prior: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Computes posterior distribution over candidates and full 19-class label space."""
    if is_correct:
        status = "correct"
        top = [{
            "id": "CORRECT",
            "p": LIKELIHOODS["correct_path_prob"],
            "name": "Correct",
            "subtitle": "Trace matches ground truth",
            "band": "Likely",
        }]
        full_posterior = {lbl: 0.001 for lbl in LABELS}
        full_posterior["CORRECT"] = LIKELIHOODS["correct_path_prob"]
        return {
            "status": status,
            "top": top,
            "posterior": full_posterior,
            "twin_set": None,
            "needs_probe": False,
        }

    # Initialize scores for candidate classes
    scores = {}
    for c in candidates:
        scores[c] = prior.get(c, 1.0 / max(1, len(candidates))) if prior else 1.0

    # 1. Deterministic weight
    if deterministic_misc and deterministic_misc in scores:
        scores[deterministic_misc] *= 3.0
        for c in candidates:
            if c != deterministic_misc:
                scores[c] *= 0.20
    elif deterministic_misc:
        scores[deterministic_misc] = 2.0

    # 2. Judge cues
    for cue in cues:
        c_misc = getattr(cue, "misconception", None) or (cue.get("misconception") if isinstance(cue, dict) else None)
        c_str = getattr(cue, "strength", 0.7) or (cue.get("strength", 0.7) if isinstance(cue, dict) else 0.7)
        if c_misc in scores:
            scores[c_misc] *= (1.0 + c_str * 0.5)

    # 3. Hedge penalty (lowers top certainty slightly)
    hedge_count = getattr(signals, "hedge_count", 0) or (signals.get("hedge_count", 0) if isinstance(signals, dict) else 0)
    penalty = min(0.15, hedge_count * LIKELIHOODS["hedge_penalty"])

    # Normalize candidate scores
    total_score = sum(scores.values()) or 1.0
    cand_probs = {c: val / total_score for c, val in scores.items()}

    # Sort descending
    sorted_cands = sorted(cand_probs.items(), key=lambda x: x[1], reverse=True)
    top_class_id, top_p = sorted_cands[0] if sorted_cands else ("OTHER", 0.5)

    # Apply hedge penalty to top probability
    top_p = max(0.40, top_p - penalty)

    # Check for ambiguity
    is_ambiguous = needs_probe
    if len(sorted_cands) >= 2:
        margin = abs(sorted_cands[0][1] - sorted_cands[1][1])
        if margin < 0.15 or needs_probe:
            is_ambiguous = True

    status = "ambiguous" if is_ambiguous else "confident"

    # Build top card list
    top_cards = []
    for cid, p in sorted_cands[:2 if is_ambiguous else 1]:
        info = CLASS_INFO.get(cid, {})
        top_cards.append({
            "id": cid,
            "p": round(p if not is_ambiguous else 0.48, 2),
            "name": info.get("name", cid),
            "subtitle": info.get("subtitle", ""),
            "band": band(p),
        })

    # Full 19-class posterior embedding
    full_posterior = {lbl: 0.002 for lbl in LABELS}
    for cid, p in cand_probs.items():
        full_posterior[cid] = round(p, 4)

    twin_set = None
    if is_ambiguous:
        if set(candidates) == {"M01", "M08"}:
            twin_set = "T1"
        elif set(candidates) == {"M06", "M07"}:
            twin_set = "T3"

    return {
        "status": status,
        "top": top_cards,
        "posterior": full_posterior,
        "twin_set": twin_set,
        "needs_probe": is_ambiguous,
    }
