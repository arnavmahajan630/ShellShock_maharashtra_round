"""Package S2: diagnosis pipeline and routes.

Acceptance (06 §3): hard-twin code returns `ambiguous`; after two probes the posterior is
>= 0.85; under 300 ms per request. The tests call the real app through TestClient with the
real diagnoser artifact in ml/artifacts/ (build it with `python -m ml.model.train`).
"""
import json
import statistics
import time
from pathlib import Path

import pytest

from ml.contracts import schemas
from server.app import pipeline
from tests.S2.conftest import P03_LE, P03_OK, P11_SEMI

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "server" / "fixtures"
PROBES = {p["probe_id"]: p for p in json.loads((ROOT / "ml" / "data" / "probes.json").read_text(encoding="utf-8"))}

Q06_NO_TEMP = """void bubble_sort(int a[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1 - i; j++) {
            if (a[j] > a[j + 1]) {
                a[j] = a[j + 1];
                a[j + 1] = a[j];
            }
        }
    }
}"""


def believer_answer(cls, probe):
    """What a learner who holds misconception `cls` answers: the belief answer if the probe has one."""
    return probe["belief"].get(cls, probe["correct"])


def attempt(client, problem_id, code, **extra):
    response = client.post("/attempt", json={"learner_id": "t", "problem_id": problem_id, "code": code,
                                             "events": [], **extra})
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------- acceptance 1: hard twin is ambiguous

def test_hard_twin_is_ambiguous(client):
    body = attempt(client, "P03", P03_LE)
    d = body["diagnosis"]
    assert body["gate"]["code"] == "G0"
    assert d["status"] == "ambiguous"
    assert d["twin_set"] == "T1"
    assert {t["id"] for t in d["top"]} == {"M01", "M08"}
    assert d["next_probe"]["probe_id"].startswith("P_T1_")
    assert d["next_probe"]["eig_bits"] >= 0.10
    assert any(e["type"] == "RUN" and "identical for both explanations" in e["text"] for e in d["evidence"])
    assert d["probes_asked"] == []
    assert body["tests"]["passed"] == 0


# ---------------------------------------------------------------- acceptance 2: two probes reach 0.85

@pytest.mark.parametrize("believes", ["M01", "M08"])
def test_two_probes_reach_085_through_probe_answer(client, believes):
    body = attempt(client, "P03", P03_LE)
    attempt_id, diagnosis, asked = body["attempt_id"], body["diagnosis"], 0
    while diagnosis["status"] == "ambiguous":
        probe = diagnosis["next_probe"]
        response = client.post("/probe/answer", json={
            "learner_id": "t", "attempt_id": attempt_id, "probe_id": probe["probe_id"],
            "answer": believer_answer(believes, PROBES[probe["probe_id"]])})
        assert response.status_code == 200, response.text
        diagnosis = response.json()["diagnosis"]
        asked += 1
        assert asked <= 2
    assert asked >= 1
    assert diagnosis["status"] == "confident"
    assert diagnosis["top"][0]["id"] == believes
    # decide() (03 §5.5) calls a lead of 0.25 over the runner-up "confident", so one probe may stop short of
    # 0.85 (see notes/S2.md); two probes always end at or above it (test_both_probes_always_reach_085).
    assert diagnosis["posterior"][believes] >= 0.85 if asked == 2 else diagnosis["posterior"][believes] >= 0.5
    assert len(diagnosis["probes_asked"]) == asked
    assert diagnosis["twin_set"] == "T1"
    assert diagnosis["next_probe"] is None
    assert any(e["type"] == "PROBE" for e in diagnosis["evidence"])


@pytest.mark.parametrize("believes,first,second", [("M01", "P_T1_a", "P_T1_b"), ("M08", "P_T1_a", "P_T1_b")])
def test_both_probes_always_reach_085(client, believes, first, second):
    """The worst order for M01 (its first answer is the correct one) still ends above 0.85 after two."""
    body = attempt(client, "P03", P03_LE)
    attempt_id = body["attempt_id"]
    for probe_id in (first, second):
        response = client.post("/probe/answer", json={
            "learner_id": "t", "attempt_id": attempt_id, "probe_id": probe_id,
            "answer": believer_answer(believes, PROBES[probe_id])})
        assert response.status_code == 200
    d = response.json()["diagnosis"]
    assert d["top"][0]["id"] == believes and d["posterior"][believes] >= 0.85
    assert d["probes_asked"] == [first, second]


@pytest.mark.parametrize("believes", ["M01", "M08"])
def test_lab_diagnose_replays_probe_answers_in_order(client, believes):
    answers = [{"probe_id": pid, "answer": believer_answer(believes, PROBES[pid])} for pid in ("P_T1_a", "P_T1_b")]
    response = client.post("/lab/diagnose", json={"problem_id": "P03", "code": P03_LE, "probe_answers": answers})
    assert response.status_code == 200
    body = response.json()
    assert "attempt_id" not in body
    d = body["diagnosis"]
    assert d["top"][0]["id"] == believes and d["posterior"][believes] >= 0.85
    assert d["probes_asked"] == ["P_T1_a", "P_T1_b"]
    # without answers: the same code is ambiguous again (stateless)
    again = client.post("/lab/diagnose", json={"problem_id": "P03", "code": P03_LE}).json()
    assert again["diagnosis"]["status"] == "ambiguous"


def test_probe_answer_without_attempt_id_replays_from_code(client):
    """The app also sends problem_id and code; a lost attempt id is not fatal."""
    response = client.post("/probe/answer", json={
        "learner_id": "t", "attempt_id": "at_gone", "problem_id": "P03", "code": P03_LE,
        "probe_id": "P_T1_a", "answer": "5"})
    assert response.status_code == 200
    d = response.json()["diagnosis"]
    assert d["top"][0]["id"] == "M08" and d["probes_asked"] == ["P_T1_a"]


def test_probe_answer_errors(client):
    body = attempt(client, "P03", P03_LE)
    bad = client.post("/probe/answer", json={"attempt_id": body["attempt_id"], "probe_id": "P_T1_a", "answer": "99"})
    assert bad.status_code == 422 and "error" in bad.json()
    unknown = client.post("/probe/answer", json={"attempt_id": body["attempt_id"], "probe_id": "P_NOPE", "answer": "1"})
    assert unknown.status_code == 422
    lost = client.post("/probe/answer", json={"attempt_id": "at_gone", "probe_id": "P_T1_a", "answer": "4"})
    assert lost.status_code in (200, 404)           # no problem_id: falls to the fixture; never a stack trace
    # a rejected answer does not poison the attempt
    ok = client.post("/probe/answer", json={"attempt_id": body["attempt_id"], "probe_id": "P_T1_a", "answer": "5"})
    assert ok.status_code == 200 and ok.json()["diagnosis"]["probes_asked"] == ["P_T1_a"]


def test_a_probe_asked_twice_counts_once(client):
    body = attempt(client, "P03", P03_LE)
    for answer in ("5", "4"):
        d = client.post("/probe/answer", json={"attempt_id": body["attempt_id"], "probe_id": "P_T1_a",
                                               "answer": answer}).json()["diagnosis"]
    assert d["probes_asked"] == ["P_T1_a"]
    assert d["top"][0]["id"] == "M08"               # the first answer stands


# ---------------------------------------------------------------- other statuses and shapes

def test_confident_and_correct_and_dsa(client):
    semi = attempt(client, "P11", P11_SEMI)["diagnosis"]
    assert semi["status"] in ("confident", "ambiguous")
    assert semi["top"][0]["id"] == "M07"

    ok = attempt(client, "P03", P03_OK)
    assert ok["tests"]["passed"] == ok["tests"]["total"] == 5
    assert ok["diagnosis"]["status"] == "correct"
    assert ok["diagnosis"]["top"][0]["id"] == "CORRECT"
    assert ok["diagnosis"]["evidence"] == []

    sort = attempt(client, "Q06", Q06_NO_TEMP)
    assert sort["tests"]["passed"] < sort["tests"]["total"]
    assert sort["diagnosis"]["status"] in ("confident", "ambiguous", "two_bug", "novel")
    assert "D03" in [t["id"] for t in sort["diagnosis"]["top"]]


def test_gate_blocks_before_the_model(client):
    body = attempt(client, "P03", "def total_energy(cells, n):\n    return sum(cells)\n")
    assert body["gate"]["code"] == "G3b"
    assert body["gate"]["message"] == "That's Python. This ship only speaks C."
    assert body["trace"] is None and body["tests"] is None and body["diagnosis"] is None
    empty = attempt(client, "P03", "   ")
    assert empty["gate"]["code"] == "G1"
    run = client.post("/run", json={"problem_id": "P03", "code": ""}).json()
    assert run["gate"]["code"] == "G1" and run["trace"] is None and run["tests"] is None


def test_run_has_no_diagnosis_and_honours_sample_only(client):
    full = client.post("/run", json={"problem_id": "P03", "code": P03_LE}).json()
    assert set(full) == {"gate", "trace", "tests", "model_version", "latency_ms"}
    assert full["tests"]["total"] == 5 and full["tests"]["passed"] == 0
    assert full["trace"]["events"] and full["trace"]["events"][0]["type"] == "oob_read"
    sample = client.post("/run", json={"problem_id": "P03", "code": P03_OK, "sample_only": True}).json()
    assert 0 < sample["tests"]["total"] < 5
    assert sample["tests"]["passed"] == sample["tests"]["total"]


def test_response_shapes_match_the_contract_and_fixtures(client):
    body = attempt(client, "P03", P03_LE)
    schemas.AttemptResponse.model_validate(body)
    fixture = json.loads((FIXTURES / "attempt.json").read_text(encoding="utf-8"))["cases"]["P03"]
    assert set(body) == set(fixture)
    assert set(body["diagnosis"]) >= set(fixture["diagnosis"])
    assert set(body["trace"]) == set(fixture["trace"])
    assert set(body["tests"]) == set(fixture["tests"])
    assert set(body["diagnosis"]["next_probe"]) == set(fixture["diagnosis"]["next_probe"])
    assert set(body["diagnosis"]["top"][0]) == set(fixture["diagnosis"]["top"][0])
    assert body["diagnosis"]["novelty"].keys() == fixture["diagnosis"]["novelty"].keys()
    assert all(set(e) >= {"type", "text"} for e in body["diagnosis"]["evidence"])
    assert abs(sum(body["diagnosis"]["posterior"].values()) - 1.0) < 0.01
    assert set(body["diagnosis"]["posterior"]) == set(fixture["diagnosis"]["posterior"])

    lab = client.post("/lab/diagnose", json={"problem_id": "P03", "code": P03_LE}).json()
    schemas.LabDiagnoseResponse.model_validate(lab)
    assert set(lab) == set(json.loads((FIXTURES / "lab_diagnose.json").read_text(encoding="utf-8"))["cases"]["P03"])

    run = client.post("/run", json={"problem_id": "P03", "code": P03_LE}).json()
    schemas.RunResponse.model_validate(run)

    answered = client.post("/probe/answer", json={"attempt_id": body["attempt_id"], "probe_id": "P_T1_a",
                                                  "answer": "5"}).json()
    schemas.ProbeAnswerResponse.model_validate(answered)
    assert set(answered) == set(json.loads((FIXTURES / "probe_answer.json").read_text(encoding="utf-8")))


def test_prediction_is_scored_with_the_bayes_table(client):
    plain = client.post("/lab/diagnose", json={"problem_id": "P03", "code": P03_LE}).json()["diagnosis"]
    # 'How many cells are read?' -> 3 is the belief answer of M01 and M08, 4 is correct
    wrong = client.post("/lab/diagnose", json={"problem_id": "P03", "code": P03_LE, "prediction": "3"}).json()["diagnosis"]
    right = client.post("/lab/diagnose", json={"problem_id": "P03", "code": P03_LE, "prediction": "4"}).json()["diagnosis"]
    assert any(e["type"] == "YOU PREDICTED" for e in wrong["evidence"])
    both = lambda d: d["posterior"]["M01"] + d["posterior"]["M08"]
    assert both(wrong) > both(plain) > both(right)


def test_attempt_reads_a_prediction_event(client):
    event = {"type": "predict", "payload": {"answer": "3"}}
    body = attempt(client, "P03", P03_LE, events=[event])
    assert any(e["type"] == "YOU PREDICTED" for e in body["diagnosis"]["evidence"])


def test_unknown_ids_still_get_fixtures(client):
    gate = client.post("/attempt", json={"problem_id": "GATE", "code": "x"}).json()
    assert gate["gate"]["code"] == "G3b" and gate["diagnosis"] is None
    assert client.post("/run", json={"problem_id": "nope", "code": "x"}).status_code == 200


def test_older_planet_problems_keep_working(client):
    body = client.post("/attempt", json={"learner_id": "t", "problem_id": "P20", "code": "", "events": []}).json()
    assert body["gate"]["code"] == "G1" and body["diagnosis"] is None and body["attempt_id"].startswith("at_P20")


# ---------------------------------------------------------------- code items (05 §4)

FIX_ITEM = {"item_id": "fb_P03_test", "type": "fix_bug", "problem_id": "P03", "planted": "M01", "op_id": "m01_le",
            "starter": P03_LE, "bug_lines": [3]}


def test_fix_bug_rule(client, monkeypatch):
    monkeypatch.setattr(pipeline, "_code_items", lambda: {FIX_ITEM["item_id"]: FIX_ITEM})

    still = attempt(client, "P03", P03_LE, code_item_id=FIX_ITEM["item_id"])
    assert still["code_item"]["response"] == "wrong"
    assert still["code_item"]["apply_diagnosis"] is False and still["code_item"]["bug_lines_unchanged"] is True
    assert (still["code_item"]["g"], still["code_item"]["s"]) == (0.35, 0.15)
    assert still["diagnosis"]["top"][0]["id"] in ("M01", "M08")

    fixed = attempt(client, "P03", P03_OK, code_item_id=FIX_ITEM["item_id"])
    assert fixed["code_item"]["response"] == "correct" and fixed["code_item"]["apply_diagnosis"] is False

    other = attempt(client, "P03", P03_OK.replace("int total = 0;", "int total;"), code_item_id=FIX_ITEM["item_id"])
    assert other["code_item"]["response"] in ("other_bug", "wrong")
    if other["code_item"]["response"] == "other_bug":
        assert other["code_item"]["apply_diagnosis"] is True and other["code_item"]["other"] != "M01"

    plain = attempt(client, "P03", P03_LE)
    assert "code_item" not in plain
    unknown = attempt(client, "P03", P03_LE, code_item_id="nope")
    assert "code_item" not in unknown


def test_complete_snippet_is_a_normal_attempt(client, monkeypatch):
    item = {"item_id": "cs_P03_test", "type": "complete_snippet", "problem_id": "P03", "starter": "", "holes": []}
    monkeypatch.setattr(pipeline, "_code_items", lambda: {item["item_id"]: item})
    body = attempt(client, "P03", P03_OK, code_item_id=item["item_id"])
    assert body["code_item"]["apply_diagnosis"] is True and body["code_item"]["response"] is None


def test_code_item_for_another_problem_is_ignored(client, monkeypatch):
    monkeypatch.setattr(pipeline, "_code_items", lambda: {FIX_ITEM["item_id"]: FIX_ITEM})
    body = attempt(client, "P11", P11_SEMI, code_item_id=FIX_ITEM["item_id"])
    assert "code_item" not in body


# ---------------------------------------------------------------- pipeline pieces

def test_stateless_analysis_does_not_use_a_learner_prior(client):
    a = pipeline.analyse(pipeline.get_problem("P03"), P03_LE)
    assert a.base == pytest.approx({k: v / sum(a.p_code.values()) for k, v in a.p_code.items()})


def test_learner_prior_hook_nudges_but_never_dominates(client, monkeypatch):
    problem = pipeline.get_problem("P03")
    monkeypatch.setattr(pipeline, "learner_prior_fn", lambda learner_id: {"M08": 0.9, "M01": 0.02})
    a = pipeline.analyse(problem, P03_LE, learner_id="x")
    assert a.base["M08"] > a.base["M01"]
    assert a.base["M08"] < 0.7
    monkeypatch.setattr(pipeline, "learner_prior_fn", lambda learner_id: (_ for _ in ()).throw(RuntimeError("db down")))
    assert pipeline.analyse(problem, P03_LE, learner_id="x").base is not None


def test_no_model_falls_back_to_the_older_path(client, monkeypatch):
    monkeypatch.setitem(pipeline._state, "model", None)
    monkeypatch.setitem(pipeline._state, "model_error", "no artifact")
    body = client.post("/attempt", json={"learner_id": "t", "problem_id": "P03", "code": P03_LE, "events": []}).json()
    assert body["model_version"] == "rules-v1" and body["gate"]["code"] == "G0"
    assert body["diagnosis"] is not None
    monkeypatch.setitem(pipeline._state, "model_error", None)


def test_internal_errors_do_not_leak_a_stack_trace(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("secret path C:\\x")
    monkeypatch.setattr(pipeline, "attempt", boom)
    response = client.post("/attempt", json={"learner_id": "t", "problem_id": "P03", "code": P03_LE, "events": []})
    assert response.status_code == 500
    assert "Traceback" not in response.text and "secret" not in response.text


# ---------------------------------------------------------------- acceptance 3: under 300 ms

def _corpus():
    cases = [("P03", P03_LE), ("P03", P03_OK), ("P11", P11_SEMI), ("Q06", Q06_NO_TEMP),
             ("P03", "def f(x):\n  return 1\n"), ("P03", "")]
    rows = []
    for name in ("dataset.jsonl", "realistic_llm.jsonl"):
        path = ROOT / "ml" / "data" / name
        if path.exists():
            rows += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    step = max(1, len(rows) // 80)
    cases += [(r["problem_id"], r["code"]) for r in rows[::step]]
    return cases


def test_latency_under_300_ms(client):
    cases = _corpus()
    assert len(cases) >= 60
    wall, reported, slowest = [], [], []
    for problem_id, code in cases:
        t0 = time.perf_counter()
        response = client.post("/attempt", json={"learner_id": "t", "problem_id": problem_id, "code": code, "events": []})
        wall.append((time.perf_counter() - t0) * 1000)
        assert response.status_code == 200, response.text
        reported.append(response.json()["latency_ms"])
        slowest.append((wall[-1], problem_id))
    # the two follow-up requests of a hard-twin diagnosis
    attempt_id = client.post("/attempt", json={"learner_id": "t", "problem_id": "P03", "code": P03_LE,
                                               "events": []}).json()["attempt_id"]
    for probe_id in ("P_T1_a", "P_T1_b"):
        t0 = time.perf_counter()
        r = client.post("/probe/answer", json={"attempt_id": attempt_id, "probe_id": probe_id, "answer": "4"})
        wall.append((time.perf_counter() - t0) * 1000)
        assert r.status_code == 200
    ordered = sorted(wall)
    p50, p95, worst = statistics.median(ordered), ordered[int(0.95 * (len(ordered) - 1))], ordered[-1]
    print(f"\nS2 latency over {len(wall)} requests: p50 {p50:.0f} ms, p95 {p95:.0f} ms, max {worst:.0f} ms "
          f"(reported latency_ms max {max(reported):.0f}); slowest {sorted(slowest)[-3:]}")
    assert p95 < 300
    assert worst < 300 and max(reported) < 300
