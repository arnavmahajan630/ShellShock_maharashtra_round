"""Unit tests for deterministic multimodal grounding and harness verification."""
import pytest
from server.app.multimodal.grounding import ground_flowchart, ground_voice_loop
from server.app.multimodal.models import Claim, MMResponse


def test_ground_flowchart_correct():
    item = {
        "problem_id": "MMA-01",
        "correct_path": ["n1", "n2", "n3", "n4"],
        "correct_output": "1",
        "buggy_paths": [
            {
                "misconception": "M06",
                "path": ["n1", "n2", "n4"],
                "output": "0",
                "divergence_node": "n2",
                "belief": "Believed `=` compares code to 42."
            }
        ]
    }
    response = MMResponse(path=["n1", "n2", "n3", "n4"], predicted_output="1")
    res = ground_flowchart(item, response)
    assert res["is_correct"] is True
    assert res["is_path_correct"] is True
    assert res["output_correct"] is True
    assert res["divergence_node"] is None
    assert res["matched_misconception"] is None


def test_ground_flowchart_buggy_m06():
    item = {
        "problem_id": "MMA-01",
        "correct_path": ["n1", "n2", "n3", "n4"],
        "correct_output": "1",
        "buggy_paths": [
            {
                "misconception": "M06",
                "path": ["n1", "n2", "n4"],
                "output": "0",
                "divergence_node": "n2",
                "belief": "Believed `=` compares code to 42."
            }
        ]
    }
    response = MMResponse(path=["n1", "n2", "n4"], predicted_output="0")
    res = ground_flowchart(item, response)
    assert res["is_correct"] is False
    assert res["is_path_correct"] is False
    assert res["divergence_node"] == "n2"
    assert res["matched_misconception"] == "M06"
    assert "Believed `=` compares code to 42" in res["evidence_text"]


def test_ground_voice_correct_iteration():
    item = {
        "problem_id": "MMB-01",
        "code": "int burns = 0;\nfor (int i = 0; i <= n; i++) {\n    burns++;\n}\nreturn burns;",
        "harness_inputs": {"n": 3},
        "expected_iterations": 4,
        "expected_terminates": True,
        "claim_rules": [
            {
                "kind": "iteration_count",
                "condition": "claimed_value == actual - 1",
                "implies": "M01",
                "evidence": "Claimed 3 burns instead of 4"
            }
        ]
    }
    claims = [
        Claim(kind="iteration_count", value="4", quote="it loops four times")
    ]
    response = MMResponse(transcript="the loop runs 4 times for i = 0 to 3")
    res = ground_voice_loop(item, claims, response)
    assert res["actual_iterations"] == 4
    assert res["terminates"] is True
    assert res["matched_misconception"] is None
    assert res["is_correct"] is True


def test_ground_voice_boundary_drift_m01():
    item = {
        "problem_id": "MMB-01",
        "code": "int burns = 0;\nfor (int i = 0; i <= n; i++) {\n    burns++;\n}\nreturn burns;",
        "harness_inputs": {"n": 3},
        "expected_iterations": 4,
        "expected_terminates": True,
        "claim_rules": [
            {
                "kind": "iteration_count",
                "condition": "claimed_value == actual - 1",
                "implies": "M01",
                "evidence": "Claimed 3 burns instead of 4"
            }
        ]
    }
    claims = [
        Claim(kind="iteration_count", value="3", quote="burns 3 times because n is 3")
    ]
    response = MMResponse(transcript="it runs 3 times")
    res = ground_voice_loop(item, claims, response)
    assert res["actual_iterations"] == 4
    assert res["matched_misconception"] == "M01"
    assert res["is_correct"] is False
