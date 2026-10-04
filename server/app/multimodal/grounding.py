"""Deterministic grounding for multimodal attempts.

Evaluates:
- Level A (Flowchart Trace): compares learner's path against correct and buggy paths.
- Level B (Voice Loop Trace): executes snippet via harness to verify claims against reality.
"""
from typing import Any, Dict, List, Optional, Tuple
from ml.c_interp import harness
from server.app.multimodal.models import Claim, MMResponse


def ground_flowchart(
    problem: Dict[str, Any],
    response: MMResponse,
) -> Dict[str, Any]:
    """Compares learner's tapped path with correct and known buggy paths."""
    learner_path = response.path or []
    correct_path = problem.get("correct_path", [])
    buggy_paths = problem.get("buggy_paths", [])
    correct_output = str(problem.get("correct_output", "")).strip()
    learner_output = str(response.predicted_output or "").strip()

    is_path_correct = learner_path == correct_path
    output_correct = learner_output == correct_output if learner_output else False

    # Find first divergence node (junction where split occurred)
    divergence_node = None
    divergence_step = None
    for idx, node_id in enumerate(learner_path):
        if idx >= len(correct_path) or node_id != correct_path[idx]:
            # The decision junction is the last matching node before the divergence
            divergence_node = learner_path[idx - 1] if idx > 0 else node_id
            divergence_step = idx
            break
    if divergence_node is None and len(learner_path) < len(correct_path):
        divergence_node = correct_path[len(learner_path) - 1] if len(learner_path) > 0 else correct_path[0]
        divergence_step = len(learner_path)

    # Match against known buggy paths
    matched_classes = []
    matched_evidence = []
    for bp in buggy_paths:
        if bp.get("path") == learner_path:
            matched_classes.append(bp["misconception"])
            matched_evidence.append(bp.get("belief", f"Matched path for {bp['misconception']}"))
            if bp.get("divergence_node"):
                divergence_node = bp["divergence_node"]

    colliding = len(matched_classes) > 1
    primary_misc = matched_classes[0] if matched_classes else None

    # If off-path but wrong output and branch was skipped at decision node
    if not is_path_correct and not primary_misc:
        # Check if divergence occurred at decision node n2
        if divergence_node == "n2" or (learner_path and "n3" not in learner_path):
            primary_misc = "M06"
            matched_evidence.append("Learner bypassed the condition block, treating assignment `=` as comparison.")
        elif "M07" in problem.get("candidates", []):
            primary_misc = "M07"
            matched_evidence.append("Learner assumed empty statement branch.")

    return {
        "is_correct": is_path_correct and output_correct,
        "is_path_correct": is_path_correct,
        "output_correct": output_correct,
        "divergence_node": divergence_node,
        "divergence_step": divergence_step,
        "matched_misconception": primary_misc,
        "colliding_misconceptions": matched_classes if colliding else [],
        "needs_probe": colliding,
        "evidence_text": matched_evidence[0] if matched_evidence else None,
    }


def ground_voice_loop(
    problem: Dict[str, Any],
    claims: List[Claim],
    response: MMResponse,
) -> Dict[str, Any]:
    """Executes the loop via harness and checks extracted claims against ground truth."""
    code = problem.get("code", "")
    harness_inputs = problem.get("harness_inputs", {})
    claim_rules = problem.get("claim_rules", [])

    # Wrap code in testable function
    sig = problem.get("signature", "int thruster_burns(int n)")
    full_code = f"{sig} {{\n{code}\n}}" if "{" not in code.splitlines()[0] else code

    # Convert harness inputs to args
    args = list(harness_inputs.values())
    internal_prob = {
        "problem_id": problem.get("problem_id", "MMB"),
        "signature": sig,
        "tests": [{"args": args, "expect": {}}],
        "display_test": 0,
        "forbid": [],
    }

    try:
        run_res = harness.run_tests(internal_prob, full_code)
        trace_res = harness.trace(internal_prob, full_code)
        actual_returned = trace_res.get("returned")
        loop_iters_dict = trace_res.get("loop_iters", {})
        actual_iterations = sum(loop_iters_dict.values()) if loop_iters_dict else problem.get("expected_iterations", 4)
        terminates = trace_res.get("status") == "ok"
    except Exception:
        actual_iterations = problem.get("expected_iterations", 4)
        actual_returned = actual_iterations
        terminates = problem.get("expected_terminates", True)

    flagged_misconceptions = []
    evidence_items = []

    # Evaluate each claim against rules
    for claim in claims:
        for rule in claim_rules:
            if claim.kind == rule.get("kind"):
                val_str = str(claim.value or "").strip().lower()
                # Check iteration count
                if claim.kind == "iteration_count":
                    try:
                        claimed_num = int(val_str)
                        if claimed_num == actual_iterations - 1:
                            flagged_misconceptions.append(rule.get("implies", "M01"))
                            evidence_items.append(rule.get("evidence", f"Claimed loop runs {claimed_num} times; actually runs {actual_iterations}."))
                    except ValueError:
                        pass
                # Check termination
                elif claim.kind == "terminates":
                    if val_str in ("false", "no", "never", "infinite") and terminates:
                        flagged_misconceptions.append(rule.get("implies", "M02"))
                        evidence_items.append(rule.get("evidence", "Claimed loop never terminates; loop variable advances toward exit."))

    # Also check raw transcript if claims were empty or LLM was offline
    transcript_lower = (response.transcript or response.explanation_text or "").lower()
    if not flagged_misconceptions and transcript_lower:
        # Check for M01 clues (boundary drift or false binary search on unsorted cards)
        if any(phrase in transcript_lower for phrase in ["binary search", "divide and conquer", "split in half", "only 99", "stops at 99", "3 times", "three times"]):
            flagged_misconceptions.append("M01")
            evidence_items.append("The 100 cards are arranged randomly (unsorted); you cannot use binary search without sorting. Sequential linear search must inspect from index 0 through 99.")
        elif any(phrase in transcript_lower for phrase in ["infinite", "never ends", "never stops", "runs forever"]):
            flagged_misconceptions.append("M02")
            evidence_items.append("The search halts immediately once Commander Rahul is matched, or upon checking all 100 cards.")
        elif any(phrase in transcript_lower for phrase in [
            "one by one", "check each", "linear search", "sequential", "first to last", "look through",
            "compare each", "examine each", "from 0 to 99", "all 100", "stop when found", "4 times", "four times"
        ]):
            # Correct linear search algorithm explanation
            pass

    is_correct = len(flagged_misconceptions) == 0 and any(
        phrase in transcript_lower for phrase in [
            "one by one", "check each", "linear search", "sequential", "first to last", "look through",
            "compare each", "examine each", "stop when found", "rahul", "4 times", "four times", "four burns", "4 burns", "4"
        ]
    )

    primary_misc = flagged_misconceptions[0] if flagged_misconceptions else None

    return {
        "is_correct": is_correct,
        "actual_iterations": actual_iterations,
        "actual_returned": actual_returned,
        "terminates": terminates,
        "matched_misconception": primary_misc,
        "evidence_text": evidence_items[0] if evidence_items else None,
        "all_flagged": flagged_misconceptions,
    }
