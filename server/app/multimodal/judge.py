"""Judge module for evaluating multimodal student responses.

Builds prompts, calls LLM, verifies verbatim quotes, and returns structured JudgeEvidence.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict
from server.app.multimodal.llm import call_llm
from server.app.multimodal.models import Claim, Cue, JudgeEvidence, MMResponse

log = logging.getLogger("relearn.multimodal.judge")

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load_prompt(filename: str) -> str:
    path = PROMPTS_DIR / filename
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _verify_quotes(evidence: JudgeEvidence, learner_text: str) -> JudgeEvidence:
    """Drops cues and claims whose quote is not a verbatim substring of the learner's words."""
    if not learner_text:
        return evidence

    cleaned_cues = []
    for cue in evidence.cues:
        if not cue.quote or cue.quote in learner_text or cue.quote.lower() in learner_text.lower():
            cleaned_cues.append(cue)
        else:
            log.debug("Dropped cue with non-verbatim quote: %r", cue.quote)
    evidence.cues = cleaned_cues

    cleaned_claims = []
    for claim in evidence.claims:
        if not claim.quote or claim.quote in learner_text or claim.quote.lower() in learner_text.lower():
            cleaned_claims.append(claim)
    evidence.claims = cleaned_claims

    return evidence


def judge_flowchart_response(
    problem: Dict[str, Any],
    response: MMResponse,
    grounding: Dict[str, Any],
) -> JudgeEvidence:
    """Judges Level A flowchart attempt."""
    prompt_template = _load_prompt("judge_image.md")
    explanation = response.explanation_text or ""

    # Prepare default mock response
    primary = grounding.get("matched_misconception")
    is_correct = grounding.get("is_correct", False)
    mock_dict = {
        "rubric": [{"id": "r1", "met": is_correct, "quote": None}],
        "cues": [{"misconception": primary, "strength": 0.85, "quote": explanation[:30]}] if primary else [],
        "claims": [{"kind": "branch_taken", "value": grounding.get("divergence_node") or "n4", "quote": ""}],
        "verdict": "correct" if is_correct else "incorrect",
        "confidence": 0.90 if primary else 0.60,
    }
    mock_json = json.dumps(mock_dict)

    prompt = (
        prompt_template
        .replace("{problem_name}", str(problem.get("name", "Flowchart Trace")))
        .replace("{candidates}", ", ".join(problem.get("candidates", [])))
        .replace("{correct_path}", str(problem.get("correct_path", [])))
        .replace("{correct_output}", str(problem.get("correct_output", "")))
        .replace("{student_path}", str(response.path or []))
        .replace("{student_output}", str(response.predicted_output or ""))
        .replace("{explanation}", explanation)
    )

    raw = call_llm(prompt, mock_default=mock_json)
    try:
        data = json.loads(raw)
        # Parse into JudgeEvidence
        cues = [Cue(**c) for c in data.get("cues", []) if c.get("misconception") in problem.get("candidates", [])]
        claims = [Claim(**cl) for cl in data.get("claims", [])]
        evidence = JudgeEvidence(
            rubric=data.get("rubric", []),
            cues=cues,
            claims=claims,
            verdict=data.get("verdict", "incorrect"),
            confidence=float(data.get("confidence", 0.7)),
            raw_response=raw,
        )
    except Exception as exc:
        log.warning("Failed to parse judge JSON (%s); using mock fallback", exc)
        evidence = JudgeEvidence(
            rubric=mock_dict["rubric"],
            cues=[Cue(**c) for c in mock_dict["cues"]],
            claims=[Claim(**cl) for cl in mock_dict["claims"]],
            verdict=mock_dict["verdict"],
            confidence=mock_dict["confidence"],
            raw_response=raw,
        )

    return _verify_quotes(evidence, explanation)


def judge_voice_response(
    problem: Dict[str, Any],
    response: MMResponse,
    grounding: Dict[str, Any],
) -> JudgeEvidence:
    """Judges Level B voice loop attempt."""
    prompt_template = _load_prompt("judge_voice.md")
    transcript = response.transcript or response.explanation_text or ""

    primary = grounding.get("matched_misconception")
    is_correct = grounding.get("is_correct", False)
    mock_dict = {
        "rubric": [{"id": "r1", "met": is_correct, "quote": None}],
        "cues": [{"misconception": primary, "strength": 0.85, "quote": transcript[:30]}] if primary else [],
        "claims": [{"kind": "iteration_count", "value": "3" if primary == "M01" else "4", "quote": ""}],
        "verdict": "correct" if is_correct else "incorrect",
        "confidence": 0.85 if primary else 0.60,
    }
    mock_json = json.dumps(mock_dict)

    prompt = (
        prompt_template
        .replace("{problem_name}", str(problem.get("name", "Voice Trace")))
        .replace("{code}", str(problem.get("code", "")))
        .replace("{candidates}", ", ".join(problem.get("candidates", [])))
        .replace("{transcript}", transcript)
    )

    raw = call_llm(prompt, mock_default=mock_json)
    try:
        data = json.loads(raw)
        cues = [Cue(**c) for c in data.get("cues", []) if c.get("misconception") in problem.get("candidates", [])]
        claims = [Claim(**cl) for cl in data.get("claims", [])]
        evidence = JudgeEvidence(
            rubric=data.get("rubric", []),
            cues=cues,
            claims=claims,
            verdict=data.get("verdict", "incorrect"),
            confidence=float(data.get("confidence", 0.7)),
            raw_response=raw,
        )
    except Exception as exc:
        log.warning("Failed to parse voice judge JSON (%s); using mock fallback", exc)
        evidence = JudgeEvidence(
            rubric=mock_dict["rubric"],
            cues=[Cue(**c) for c in mock_dict["cues"]],
            claims=[Claim(**cl) for cl in mock_dict["claims"]],
            verdict=mock_dict["verdict"],
            confidence=mock_dict["confidence"],
            raw_response=raw,
        )

    return _verify_quotes(evidence, transcript)
