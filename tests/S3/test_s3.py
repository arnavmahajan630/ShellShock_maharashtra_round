"""S3 acceptance: a scripted learner finishes an exam, and a collision quiz answer
asks for a reason that /reason can settle. An unsure reader does not move P(A).
"""
import json
from pathlib import Path

from ml.contracts.schemas import ExamReport, QuizAnswerResponse, ReasonResponse

ROOT = Path(__file__).resolve().parents[2]
COLLISION = "qz_m01m08_mcq_01"
COLLISION_ANSWER = "8"
M08_SENTENCE = "Counting starts at 1, so a[4] is the 4th number in the list."
UNSURE_SENTENCE = "asdf qqq zzz"
HIDDEN = ("fix", "planted", "bug_lines", "explain", "exposes", "op_id", "correct", "belief")


def _learner(client, callsign="S3"):
    response = client.post("/learner", json={"callsign": callsign})
    assert response.status_code == 200, response.text
    return response.json()["learner_id"]


def _p(client, learner_id, cls):
    body = client.get(f"/learner/{learner_id}").json()
    return body["misconceptions"][cls]["p_active"]


def test_routes_take_over_fixtures(client):
    listed = client.get("/_routes").json()
    for name in ("code_items", "exam", "metrics", "quiz", "reason"):
        assert name in listed["live_routers"]
    fixture = " ".join(listed["fixture_routes"])
    for path in ("/exam/start", "/exam/answer", "/exam/finish", "/exam/probe", "/quiz/next",
                 "/quiz/answer", "/code-items", "/reason", "/lab/reason", "/metrics"):
        assert path not in fixture


def test_scripted_learner_finishes_exam(client):
    learner_id = _learner(client, "EXAM")
    started = client.post("/exam/start", json={"learner_id": learner_id, "demo": True})
    assert started.status_code == 200, started.text
    body = started.json()
    assert body["progress"] == {"k": 1, "n": 3}
    assert body["time_limit_s"] == 1500
    exam_id = body["exam_id"]
    item = body["item"]
    assert "correct" not in item and "belief" not in item and "exposure" not in item

    seen = []
    for _ in range(5):
        payload = {"exam_id": exam_id, "item_id": item["item_id"], "ms": 1000, "sample_runs": 0}
        if item["kind"] == "coding":
            payload["code"] = item.get("starter") or ""
        else:
            assert item["options"]
            payload["answer"] = item["options"][0]
        answered = client.post("/exam/answer", json=payload)
        assert answered.status_code == 200, answered.text
        answer_body = answered.json()
        assert answer_body["logged"] is True
        assert "diagnosis" not in answer_body
        seen.append(item["item_id"])
        item = answer_body["next_item"]
        if item is None:
            break
        assert "correct" not in item and "belief" not in item
    assert item is None
    assert len(seen) == 3

    finished = client.post("/exam/finish", json={"exam_id": exam_id})
    assert finished.status_code == 200, finished.text
    report = finished.json()["report"]
    ExamReport.model_validate(report)
    assert report["exam_id"] == exam_id
    assert report["learner_id"] == learner_id
    assert report["items_answered"] == 3
    assert [row["item_id"] for row in report["adaptivity_log"]] == seen
    assert all(row["reason"] for row in report["adaptivity_log"])

    again = client.get(f"/exam/{exam_id}/report")
    assert again.status_code == 200
    assert again.json()["report"]["adaptivity_log"] == report["adaptivity_log"]

    probed = client.post("/exam/probe", json={"exam_id": exam_id, "probe_id": "P_T1_a", "answer": "5"})
    assert probed.status_code == 200, probed.text
    probed_report = probed.json()["report"]
    ExamReport.model_validate(probed_report)
    assert probed_report["exam_id"] == exam_id


def test_collision_asks_reason_and_reason_settles_it(client):
    learner_id = _learner(client, "QUIZ")
    answered = client.post("/quiz/answer", json={
        "learner_id": learner_id, "item_id": COLLISION, "answer": COLLISION_ANSWER, "ms": 400,
    })
    assert answered.status_code == 200, answered.text
    body = answered.json()
    QuizAnswerResponse.model_validate(body)
    assert body["ask_reason"] is True
    assert body["correct"] is False
    classes = {row["class"] for row in body["updates"]}
    assert classes == {"M01", "M08"}
    assert abs(body["updates"][0]["p_after"] - body["updates"][1]["p_after"]) < 1e-9
    tied = _p(client, learner_id, "M01")
    assert abs(_p(client, learner_id, "M08") - tied) < 1e-6

    unsure = client.post("/reason", json={
        "learner_id": learner_id,
        "ref": {"kind": "quiz", "id": COLLISION},
        "text": UNSURE_SENTENCE,
    })
    assert unsure.status_code == 200, unsure.text
    unsure_body = unsure.json()
    ReasonResponse.model_validate(unsure_body)
    assert unsure_body["status"] == "unsure"
    assert unsure_body["updates"] == []
    assert abs(_p(client, learner_id, "M08") - tied) < 1e-6
    assert abs(_p(client, learner_id, "M01") - tied) < 1e-6

    settled = client.post("/reason", json={
        "learner_id": learner_id,
        "ref": {"kind": "quiz", "id": COLLISION},
        "text": M08_SENTENCE,
    })
    assert settled.status_code == 200, settled.text
    settled_body = settled.json()
    ReasonResponse.model_validate(settled_body)
    assert settled_body["status"] == "matched"
    assert settled_body["top"][0]["id"] == "M08"
    assert settled_body["reader"] in ("tfidf", "biencoder", "frozen")
    moved = {row["class"]: row for row in settled_body["updates"]}
    assert "M08" in moved
    assert moved["M08"]["p_after"] > moved["M08"]["p_before"]
    assert _p(client, learner_id, "M08") > _p(client, learner_id, "M01")


def test_correct_collision_option_does_not_ask(client):
    learner_id = _learner(client, "QUIZ2")
    answered = client.post("/quiz/answer", json={
        "learner_id": learner_id, "item_id": COLLISION, "answer": "unpredictable",
    })
    assert answered.status_code == 200, answered.text
    assert answered.json()["correct"] is True
    assert answered.json()["ask_reason"] is False


def test_quiz_next_hides_answers(client):
    learner_id = _learner(client, "NEXT")
    response = client.get("/quiz/next", params={"learner_id": learner_id})
    assert response.status_code == 200, response.text
    item = response.json()["item"]
    assert item["item_id"]
    for key in ("correct", "belief", "explain", "classes", "expected"):
        assert key not in item
    assert response.json()["reason"]


def test_code_items_are_public_and_debug_scores(client):
    listed = client.get("/code-items", params={"problem_id": "P04", "type": "debug_line"})
    assert listed.status_code == 200, listed.text
    rows = listed.json()
    assert isinstance(rows, list) and rows
    ids = {row["item_id"] for row in rows}
    assert "dl_P04_m05_01" in ids
    assert "dl_P04_ok_01" in ids
    for row in rows:
        for key in HIDDEN:
            assert key not in row

    snippets = client.get("/code-items", params={"problem_id": "P03", "type": "complete_snippet"})
    assert snippets.status_code == 200
    for row in snippets.json():
        assert "fix" not in row
        assert row["type"] == "complete_snippet"
        assert row.get("holes")

    learner_id = _learner(client, "CODE")
    hit = client.post("/code-items/debug", json={"learner_id": learner_id, "item_id": "dl_P04_m05_01", "line": 2})
    assert hit.status_code == 200, hit.text
    hit_body = hit.json()
    assert hit_body["correct"] is True
    assert 2 in hit_body["bug_lines"]
    assert hit_body["fix"]["changed_lines"] == [2]
    assert hit_body["updates"][0]["class"] == "M05"
    assert hit_body["updates"][0]["p_after"] < hit_body["updates"][0]["p_before"]

    miss = client.post("/code-items/debug", json={"learner_id": learner_id, "item_id": "dl_P04_m05_01", "line": 9})
    assert miss.status_code == 200, miss.text
    assert miss.json()["correct"] is False
    assert miss.json()["updates"][0]["p_after"] > miss.json()["updates"][0]["p_before"]

    learner_id = _learner(client, "NOBUG")
    clear = client.post("/code-items/debug", json={"learner_id": learner_id, "item_id": "dl_P04_ok_01", "line": None})
    assert clear.status_code == 200, clear.text
    assert clear.json()["correct"] is True
    assert clear.json()["updates"] == []
    assert clear.json()["fix"] is None


def test_lab_reason_is_stateless(client):
    response = client.post("/lab/reason", json={"text": M08_SENTENCE})
    assert response.status_code == 200, response.text
    body = response.json()
    ReasonResponse.model_validate(body)
    assert body["updates"] == []
    assert body["status"] == "matched"


def test_metrics_copies_cards_and_marks_gaps(client):
    response = client.get("/metrics")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body.get("stub") is not True
    e1_file = json.loads((ROOT / "ml" / "eval" / "cards" / "E01.json").read_text(encoding="utf-8"))
    e1 = next(card for card in body["cards"] if card["id"] == "E1")
    assert e1["metrics"]["macro_f1"] == e1_file["metrics"]["macro_f1"]
    assert body["model_version"] == e1_file["model_version"]
    ids = [card["id"] for card in body["cards"]]
    assert ids[:18] == [f"E{number}" for number in range(1, 19)]
    assert body["effective_n"]["unique_prog_op"] == "missing"
    assert body["domains"] == "missing"
    assert body["effective_n"]["rows"] == 6753
    assert body["effective_n"]["unique_hash"] == 1845
    rules = {row["id"]: row["status"] for row in body["eda"]["rules"]}
    assert rules["10.1"] == "PASS"
    assert rules["10.2"] == "FAIL"
    e8 = json.loads((ROOT / "ml" / "eval" / "cards" / "E08.json").read_text(encoding="utf-8"))
    rules_row = next(row for row in body["baselines"] if row["name"] == "rules")
    assert rules_row["E1"] == e8["metrics"]["rules"]["E1"]["macro_f1"]
    assert body["cross_domain"]["shortcut_auc"]["estimate"] == json.loads(
        (ROOT / "ml" / "eval" / "out" / "e-b" / "e15_card.json").read_text(encoding="utf-8")
    )["metrics"]["shortcut_auc"]["estimate"]


def test_unknown_learner_and_exam(client):
    missing = client.post("/exam/start", json={"learner_id": "nobody"})
    assert missing.status_code == 404
    assert "error" in missing.json()
    assert "Traceback" not in missing.text
    report = client.get("/exam/ex_missing/report")
    assert report.status_code == 404
