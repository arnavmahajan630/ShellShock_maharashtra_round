"""Live router for multimodal attempts: POST /mm/attempt.

Automatically discovered by server/app/main.py via pkgutil.iter_modules().
Conforms strictly to the existing AttemptResponse schema so all downstream
screens (Diagnosis, Probe, Intervention, Transfer Trap, Verdict) work unchanged.
"""
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict
from fastapi import APIRouter, Body, HTTPException, Request

from ml.model.mm_evidence import compute_posterior
from server.app.multimodal.grounding import ground_flowchart, ground_voice_loop
from server.app.multimodal.judge import judge_flowchart_response, judge_voice_response
from server.app.multimodal.models import MMAttemptRequest, MMResponse, MMSignals
from server.app.multimodal.narrate import generate_droid_feedback

log = logging.getLogger("relearn.routes.mm_attempt")

router = APIRouter()

MM_FIXTURES = Path(__file__).resolve().parent.parent.parent / "fixtures" / "mm_problems.json"
_MM_PROBLEMS: Dict[str, Any] = {}


def _get_mm_problems() -> Dict[str, Any]:
    global _MM_PROBLEMS
    if not _MM_PROBLEMS and MM_FIXTURES.exists():
        try:
            _MM_PROBLEMS = json.loads(MM_FIXTURES.read_text(encoding="utf-8"))
        except Exception as exc:
            log.error("Failed to load mm_problems.json: %s", exc)
    return _MM_PROBLEMS


@router.get("/mm/problem/{problem_id}")
async def get_mm_problem(problem_id: str):
    """Returns full multimodal problem specification."""
    probs = _get_mm_problems()
    prob = probs.get(problem_id)
    if not prob:
        raise HTTPException(status_code=404, detail=f"Multimodal problem '{problem_id}' not found.")
    return prob


@router.post("/mm/attempt")
async def mm_attempt(request: Request):
    """Processes a multimodal trace attempt and returns standard AttemptResponse."""
    body = await request.json()
    problem_id = body.get("problem_id")
    modality = body.get("modality", "flowchart_trace")
    resp_dict = body.get("response", {})
    response = MMResponse(**resp_dict) if isinstance(resp_dict, dict) else MMResponse()
    signals_dict = body.get("signals", {})
    signals = MMSignals(**signals_dict) if isinstance(signals_dict, dict) else MMSignals()

    probs = _get_mm_problems()
    problem = probs.get(problem_id)
    if not problem:
        raise HTTPException(status_code=404, detail=f"Multimodal problem '{problem_id}' not found.")

    prob_type = problem.get("type", modality)
    t0 = time.perf_counter()

    # 1. Deterministic Grounding
    if prob_type in ("flowchart_trace", "image"):
        grounding = ground_flowchart(problem, response)
        judge_ev = judge_flowchart_response(problem, response, grounding)
    else:
        # voice_trace / voice
        judge_ev = judge_voice_response(problem, response, {"matched_misconception": None, "is_correct": False})
        grounding = ground_voice_loop(problem, judge_ev.claims, response)

    is_correct = grounding["is_correct"]
    matched_misc = grounding["matched_misconception"]
    needs_probe = grounding.get("needs_probe", False)

    # 2. Bayesian Evidence Fusion
    posterior_res = compute_posterior(
        candidates=problem.get("candidates", ["M06", "M07"]),
        deterministic_misc=matched_misc,
        cues=judge_ev.cues,
        signals=signals,
        is_correct=is_correct,
        needs_probe=needs_probe,
    )

    top_cards = posterior_res["top"]
    primary_id = top_cards[0]["id"] if top_cards else None

    # 3. Personalized Droid Feedback
    divergence_info = (
        f"node {grounding.get('divergence_node')}"
        if grounding.get("divergence_node")
        else f"{grounding.get('actual_iterations')} iterations"
    )
    learner_text = response.explanation_text or response.transcript or ""
    droid_feedback = generate_droid_feedback(
        problem=problem,
        misconception_id=primary_id,
        learner_words=learner_text,
        divergence_info=divergence_info,
    )

    # Build evidence items for standard Diagnosis screen
    evidence_items = []
    if grounding.get("evidence_text"):
        evidence_items.append({"type": "RUN", "chip": "RUN", "text": grounding["evidence_text"]})
    for cue in judge_ev.cues:
        if cue.quote:
            evidence_items.append({"type": "YOU PREDICTED", "chip": "YOU PREDICTED", "text": f'"{cue.quote}" indicates belief in {cue.misconception}.'})

    # Probe configuration if ambiguous
    next_probe = None
    if posterior_res["status"] == "ambiguous":
        twin_set = posterior_res["twin_set"]
        if twin_set == "T3":
            next_probe = {
                "probe_id": "P_T3_mm",
                "prompt": "In C, what is the result of `if (x = 1)` when x was initially 0?",
                "code": "int x = 0;\nif (x = 1) {\n    // does this run?\n}",
                "options": ["Always runs (assignment evaluates to 1)", "Does not run (0 != 1)"],
                "eig_bits": 0.95,
            }
        elif twin_set == "T1":
            next_probe = {
                "probe_id": "P_T1_a",
                "prompt": "What is the index of the first element in cells[]?",
                "code": "int cells[4] = {1, 2, 3, 4};",
                "options": ["0", "1"],
                "eig_bits": 0.95,
            }

    # 4. Standard Diagnosis Response Model
    diagnosis = {
        "status": posterior_res["status"],
        "posterior": posterior_res["posterior"],
        "top": top_cards,
        "twin_set": posterior_res["twin_set"],
        "two_bug": False,
        "novelty": {
            "knn_dist": 0.05,
            "tau_d": 0.35,
            "p_max": top_cards[0]["p"] if top_cards else 0.5,
            "tau_p": 0.40,
            "abstain": False,
        },
        "evidence": evidence_items,
        "next_probe": next_probe,
        "model_version": "mm-hybrid-v2.1",
    }

    # Assembled AttemptResponse
    attempt_id = f"at_{problem_id}_{uuid.uuid4().hex[:8]}"
    return {
        "attempt_id": attempt_id,
        "gate": {"code": "G0", "message": "Gate check passed."},
        "trace": {
            "status": "ok",
            "returned": grounding.get("actual_returned", response.predicted_output),
            "printed": droid_feedback,
            "steps": [],
            "loop_iters": {},
            "branch": {},
            "events": [],
            "effects_count": {},
            "max_depth": 0,
            "truncated": False,
            "per_test": [],
        },
        "tests": {
            "passed": 1 if is_correct else 0,
            "total": 1,
            "results": [{
                "args": [problem.get("inputs", {})],
                "expected": {"output": problem.get("correct_output", "1")},
                "got": {"output": response.predicted_output or "0"},
                "pass": is_correct,
            }],
        },
        "diagnosis": diagnosis,
        "mm_feedback": droid_feedback,
        "mm_debug": {
            "grounding": grounding,
            "judge_evidence": judge_ev.model_dump(),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        },
    }
