"""Personalized feedback narration for multimodal attempts.

Generates droid feedback in the learner's own terms with safe template fallback.
"""
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from ml.contracts.classes import CLASS_INFO
from server.app.multimodal.llm import call_llm

log = logging.getLogger("relearn.multimodal.narrate")

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def generate_droid_feedback(
    problem: Dict[str, Any],
    misconception_id: Optional[str],
    learner_words: str,
    divergence_info: str,
) -> str:
    """Generates personalized droid feedback or falls back to canonical explanation."""
    canonical = problem.get("canonical_explanation", "Review the control flow and boundary conditions.")
    if not misconception_id or misconception_id == "CORRECT":
        return "Telemetry verified! Your trace matches the program's actual execution path perfectly."

    misc_name = CLASS_INFO.get(misconception_id, {}).get("name", misconception_id)

    prompt_file = PROMPTS_DIR / "narrate.md"
    template = prompt_file.read_text(encoding="utf-8") if prompt_file.exists() else ""

    fallback_text = f"Notice: at {divergence_info}, the path diverged due to {misc_name}. {canonical}"

    if not template:
        return fallback_text

    prompt = (
        template
        .replace("{problem_name}", str(problem.get("name", "Mission")))
        .replace("{misconception_name}", misc_name)
        .replace("{misconception_id}", str(misconception_id))
        .replace("{student_words}", learner_words[:200] if learner_words else "the chosen path")
        .replace("{divergence_info}", str(divergence_info))
        .replace("{canonical_explanation}", canonical)
    )

    try:
        droid_text = call_llm(prompt, mock_default=fallback_text).strip()
        # Clean quotes if any
        if droid_text.startswith('"') and droid_text.endswith('"'):
            droid_text = droid_text[1:-1].strip()
        if len(droid_text) > 20:
            return droid_text
    except Exception as exc:
        log.warning("Narration LLM call failed (%s); using fallback template", exc)

    return fallback_text
