"""Checks that the contracts, the sample data and the fake API responses agree with each other.

Run from the repo root:  .venv\\Scripts\\python -m pytest tests/test_contracts.py
"""
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ml import runner
from ml.contracts import schemas as S
from ml.contracts.classes import CLASS_INFO, LABELS, MISCONCEPTIONS, REASON_LABELS, TWIN_SETS, band
from ml.contracts.feature_names import FEATURES, GROUPS, MASK_PRECONDITIONS
from ml.contracts.params import ITEM_GUESS_SLIP
from ml.contracts.subset import EFFECT_FIELDS, EVENT_FIELDS
from server.app.main import FIXTURE_ROUTES, app

FIXTURES = Path(__file__).parent / "fixtures"
SERVER_FIXTURES = Path(__file__).parent.parent / "server" / "fixtures"
client = TestClient(app)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_class_lists():
    assert len(MISCONCEPTIONS) == 17 and len(LABELS) == 19 and len(REASON_LABELS) == 18
    assert LABELS[-2:] == ["CORRECT", "OTHER"]
    assert set(LABELS) == set(CLASS_INFO)
    for info in TWIN_SETS.values():
        assert set(info["members"]) <= set(MISCONCEPTIONS)
    assert [band(p) for p in (0.9, 0.5, 0.2)] == ["Likely", "Possible", "Unsure"]


def test_feature_list():
    assert len(FEATURES) == len(set(FEATURES))
    assert sum(len(g) for g in GROUPS.values()) == len(FEATURES)
    assert len(GROUPS["F"]) == 2 * len(MISCONCEPTIONS)
    for cls, needed in MASK_PRECONDITIONS.items():
        assert cls in MISCONCEPTIONS and set(needed) <= set(FEATURES)


def test_params():
    for guess, slip in ITEM_GUESS_SLIP.values():
        assert 0 < guess < 1 and 0 < slip < 1


@pytest.mark.parametrize("path", sorted((FIXTURES / "problems").glob("*.json")), ids=lambda p: p.stem)
def test_sample_problems(path):
    problem = S.Problem.model_validate(load(path))
    assert len(problem.correct_variants) >= 3
    assert 4 <= len(problem.tests) <= 6
    assert sum(t.sample for t in problem.tests) == 2
    assert (problem.planet is None) != (problem.sector is None)
    assert set(problem.exposure) <= set(MISCONCEPTIONS)


@pytest.mark.parametrize("path", sorted((FIXTURES / "traces").glob("*.json")), ids=lambda p: p.stem)
def test_sample_traces(path):
    trace = S.Trace.model_validate(load(path))
    assert [s.i for s in trace.steps] == list(range(len(trace.steps)))
    for event in trace.events:
        assert event.type in EVENT_FIELDS
        assert set(EVENT_FIELDS[event.type]) <= set(event.model_extra)
    for step in trace.steps:
        for effect in step.effects:
            name, *fields = effect.split(":")
            assert name in EFFECT_FIELDS and len(fields) == len(EFFECT_FIELDS[name])


def test_sample_reasons():
    rows = [S.ReasonRow.model_validate(json.loads(line))
            for line in (FIXTURES / "reasons_sample.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 60
    assert {r.label for r in rows} <= set(REASON_LABELS)


def test_feature_matrix():
    data = np.load(FIXTURES / "features_synth.npz")
    assert list(data["features"]) == FEATURES and list(data["labels"]) == LABELS
    assert data["X"].shape == (len(data["y"]), len(FEATURES))


def test_runner_reports_missing_interpreter():
    if runner.available("interp") and hasattr(__import__("ml.c_interp.harness", fromlist=["x"]), "trace"):
        pytest.skip("interpreter is built")
    with pytest.raises(runner.BackendUnavailable):
        runner.trace(load(FIXTURES / "problems" / "P03_total_energy.json"), "int f() { return 0; }")


def _call(method, path, body=None):
    path = (path.replace("{problem_id}", "P11").replace("{learner_id}", "demo-learner")
            .replace("{exam_id}", "ex_fixture"))
    return client.get(path) if method == "GET" else client.post(path, json=body or {})


@pytest.mark.parametrize("method,path,filename", FIXTURE_ROUTES, ids=lambda v: str(v))
def test_every_endpoint_answers(method, path, filename):
    assert (SERVER_FIXTURES / filename).exists()
    response = _call(method, path)
    assert response.status_code == 200
    assert response.json() is not None


def test_health():
    S.HealthResponse.model_validate(client.get("/health").json())


def test_fixture_cases_follow_the_request():
    """Skipped for any endpoint that a live router has taken over."""
    fixture_paths = set(client.get("/_routes").json()["fixture_routes"])
    if "POST /attempt" in fixture_paths:
        by_problem = {p: client.post("/attempt", json={"problem_id": p}).json() for p in ("P03", "P11", "Q06", "Q17")}
        assert by_problem["P03"]["diagnosis"]["status"] == "ambiguous"
        assert by_problem["P03"]["diagnosis"]["next_probe"]["probe_id"] == "P_T1_a"
        assert by_problem["P11"]["diagnosis"]["top"][0]["id"] == "M07"
        assert by_problem["Q06"]["diagnosis"]["top"][0]["id"] == "D03"
        assert by_problem["Q17"]["diagnosis"]["status"] == "correct"
        assert client.post("/attempt", json={"problem_id": "GATE"}).json()["gate"]["code"] == "G3b"
    if "POST /exam/answer" in fixture_paths:
        order = ["Q06", "Q08", "xt_rec_1"]
        nexts = [client.post("/exam/answer", json={"item_id": i}).json()["next_item"] for i in order]
        assert [n and n["item_id"] for n in nexts] == ["Q08", "xt_rec_1", None]
    if "POST /quiz/answer" in fixture_paths:
        assert client.post("/quiz/answer", json={"answer": "3"}).json()["ask_reason"] is True
        assert client.post("/quiz/answer", json={"answer": "4"}).json()["correct"] is True
    if "GET /problems/{problem_id}" in fixture_paths:
        assert client.get("/problems/Q17").json()["problem_id"] == "Q17"


def test_public_shapes_hide_answers():
    for problem in client.get("/problems").json():
        assert not {"tests", "correct_variants", "exposure"} & set(problem)
        assert "belief" not in (problem["predict_item"] or {})
    for item in client.get("/code-items").json():
        assert not {"planted", "bug_lines", "fix", "op_id"} & set(item)
    assert not {"correct", "belief", "explain"} & set(client.get("/quiz/next").json()["item"])
