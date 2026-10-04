"""Integration tests for multimodal attempt endpoint and schemas."""
import pytest
from fastapi.testclient import TestClient
from server.app.main import app

client = TestClient(app)


def test_get_mm_problem():
    res = client.get("/mm/problem/MMA-01")
    assert res.status_code == 200
    data = res.json()
    assert data["problem_id"] == "MMA-01"
    assert data["type"] == "flowchart_trace"
    assert "graph" in data
    assert len(data["graph"]["nodes"]) == 4


def test_mm_attempt_flowchart_correct():
    payload = {
        "learner_id": "test_learner",
        "problem_id": "MMA-01",
        "modality": "image",
        "response": {
            "path": ["n1", "n2", "n3", "n4"],
            "predicted_output": "1",
            "explanation_text": "Power is 10 which is >= 5, so condition evaluates true and shield is set to 1."
        },
        "signals": {
            "response_ms": 2500,
            "hedge_count": 0
        }
    }
    res = client.post("/mm/attempt", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "attempt_id" in data
    assert data["gate"]["code"] == "G0"
    assert data["tests"]["passed"] == 1
    assert data["tests"]["total"] == 1
    assert data["diagnosis"]["status"] == "correct"
    assert data["diagnosis"]["top"][0]["id"] == "CORRECT"


def test_mm_attempt_flowchart_buggy_m06():
    payload = {
        "learner_id": "test_learner",
        "problem_id": "MMA-01",
        "modality": "image",
        "response": {
            "path": ["n1", "n2", "n4"],
            "predicted_output": "0",
            "explanation_text": "I think power condition failed or was bypassed, so shield remains 0."
        },
        "signals": {
            "response_ms": 3200,
            "hedge_count": 1
        }
    }
    res = client.post("/mm/attempt", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "attempt_id" in data
    assert data["gate"]["code"] == "G0"
    assert data["tests"]["passed"] == 0
    assert data["tests"]["total"] == 1
    diag = data["diagnosis"]
    assert diag["status"] in ("confident", "likely")
    top_ids = [c["id"] for c in diag["top"]]
    assert "M06" in top_ids
    assert diag["top"][0]["id"] == "M06"
    assert any(e["chip"] == "YOU PREDICTED" for e in diag["evidence"])


def test_mm_attempt_voice_correct():
    payload = {
        "learner_id": "test_learner",
        "problem_id": "MMB-01",
        "modality": "voice",
        "response": {
            "transcript": "Since the cards are arranged randomly, I will use linear search and inspect cards one by one from the first to last, stopping when Commander Rahul is found.",
            "transcript_edited": False,
            "explanation_text": "linear search one by one"
        },
        "signals": {
            "response_ms": 4000,
            "hedge_count": 0,
            "speech_rate_wps": 2.5
        }
    }
    res = client.post("/mm/attempt", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["gate"]["code"] == "G0"
    assert data["tests"]["passed"] == 1
    assert data["tests"]["total"] == 1
    assert data["diagnosis"]["status"] == "correct"


def test_mm_attempt_voice_boundary_drift_m01():
    payload = {
        "learner_id": "test_learner",
        "problem_id": "MMB-01",
        "modality": "voice",
        "response": {
            "transcript": "Maybe I will use binary search and split in half to find Commander Rahul.",
            "transcript_edited": False,
            "explanation_text": "binary search split in half"
        },
        "signals": {
            "response_ms": 4500,
            "hedge_count": 1,
            "speech_rate_wps": 2.1
        }
    }
    res = client.post("/mm/attempt", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["gate"]["code"] == "G0"
    assert data["tests"]["passed"] == 0
    diag = data["diagnosis"]
    assert diag["status"] in ("confident", "likely")
    top_ids = [c["id"] for c in diag["top"]]
    assert "M01" in top_ids
    assert diag["top"][0]["id"] == "M01"
