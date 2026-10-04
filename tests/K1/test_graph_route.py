"""Package K1: GET /learner/{id}/graph and the saving that feeds it, through the real app.

Every test uses a temp database (conftest). The model artifact is needed; without it the
session fixture skips.
"""
import os

from ml.contracts.params import POPULATION_PRIOR
from ml.learner import concept_graph as cg
from server.app.store import Store

P11_ASSIGN = """int door_open(int code) {
    if (code = 42) {
        return 1;
    }
    return 0;
}"""

P11_OK = """int door_open(int code) {
    if (code == 42) {
        return 1;
    }
    return 0;
}"""

P03_LE = """int total_energy(int cells[], int n) {
    int total = 0;
    for (int i = 0; i <= n; i++) {
        total += cells[i];
    }
    return total;
}"""

P20_OK = """int clamp_range(int val, int lo, int hi) {
    if (val < lo) {
        return lo;
    }
    if (val > hi) {
        return hi;
    }
    return val;
}"""

T3_OK = """void bubble_sort(int a[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1 - i; j++) {
            if (a[j] > a[j + 1]) {
                int t = a[j];
                a[j] = a[j + 1];
                a[j + 1] = t;
            }
        }
    }
}"""


def make_learner(full, learner_id="demo-learner"):
    response = full.post("/learner", json={"callsign": "PILOT", "learner_id": learner_id})
    assert response.status_code == 200, response.text
    assert response.json()["learner_id"] == learner_id
    return learner_id


def chart(full, learner_id):
    response = full.get(f"/learner/{learner_id}/graph")
    assert response.status_code == 200, response.text
    return response.json()


def star(view, star_id):
    return next(s for s in view["stars"] if s["id"] == star_id)


def attempt(full, learner_id, problem_id, code, **extra):
    response = full.post("/attempt", json={"learner_id": learner_id, "problem_id": problem_id, "code": code,
                                           "events": [], **extra})
    assert response.status_code == 200, response.text
    return response.json()


def test_unknown_learner_is_404(full):
    response = full.get("/learner/nobody/graph")
    assert response.status_code == 404 and isinstance(response.json()["error"], str)


def test_learner_can_be_created_with_a_chosen_id(full):
    make_learner(full)
    again = full.post("/learner", json={"callsign": "PILOT", "learner_id": "demo-learner"})
    assert again.status_code == 409 and "exists" in again.json()["error"]
    assert full.post("/learner", json={"callsign": "PILOT", "learner_id": "no spaces"}).status_code == 422
    assert full.post("/learner", json={"callsign": "PILOT"}).json()["learner_id"].startswith("lrn_")


def test_blank_chart_shape(full):
    view = chart(full, make_learner(full))
    assert set(view) >= {"learner_id", "canvas", "topics", "stars", "edges", "summary", "suggested_next",
                         "model_version", "latency_ms"}
    assert view["summary"]["cleared"] == 0 and view["summary"]["charted"] == 42
    assert len(view["topics"]) == 10 and view["suggested_next"]["route"].startswith("/")
    assert set(star(view, "count_loop")) == {"id", "label", "kind", "topic", "pos", "state", "cleared", "brightness",
                                             "risk", "at_risk", "risk_from", "hazards", "problems", "playable",
                                             "suggested"}


def test_a_mission_marks_the_chart_and_it_survives_a_new_store(full, temp_db):
    learner_id = make_learner(full)

    wrong = attempt(full, learner_id, "P11", P11_ASSIGN, prediction="0")
    assert wrong["diagnosis"]["top"][0]["id"] == "M06" and wrong["diagnosis"]["status"] == "confident"
    learner = full.get(f"/learner/{learner_id}").json()
    assert learner["misconceptions"]["M06"]["state"] == "ACTIVE"
    assert learner["misconceptions"]["M06"]["p_active"] > 0.5
    assert learner["nodes"]["conditions:P11"]["status"] == "unstable"
    view = chart(full, learner_id)
    assert star(view, "equality_check")["state"] == "attempted"
    halo = star(view, "branch_bands")               # P12 exposes M06 too: the mistake spreads on the chart
    assert halo["at_risk"] and halo["risk_from"][0]["id"] == "M06"
    assert view["summary"]["at_risk"] >= 3

    solved = attempt(full, learner_id, "P11", P11_OK)
    assert solved["tests"]["passed"] == solved["tests"]["total"]
    lit = star(chart(full, learner_id), "equality_check")
    assert lit["cleared"] and lit["state"] == "shaky"           # cleared, but M06 is still ACTIVE
    assert 0 < lit["brightness"] < 0.7

    reopened = Store(temp_db).get_learner(learner_id)           # a new store on the same file
    again = cg.build_view(cg.load_graph(), cg.load_problems(), reopened)
    assert star(again, "equality_check")["state"] == "shaky"
    assert len(reopened["attempts"]) == 2


def test_an_unknown_learner_writes_nothing(full, temp_db):
    body = attempt(full, "t", "P11", P11_ASSIGN)
    assert body["diagnosis"]["status"] == "confident"
    assert not os.path.exists(temp_db)                          # the database was not even created


def test_a_probe_answer_that_settles_the_diagnosis_is_saved_once(full):
    learner_id = make_learner(full)
    first = attempt(full, learner_id, "P03", P03_LE, prediction="3")
    assert first["diagnosis"]["status"] == "ambiguous"
    before = full.get(f"/learner/{learner_id}").json()["misconceptions"]
    assert before["M08"]["p_active"] == POPULATION_PRIOR and before["M01"]["p_active"] == POPULATION_PRIOR

    probe = first["diagnosis"]["next_probe"]
    answer = {"learner_id": learner_id, "attempt_id": first["attempt_id"], "probe_id": probe["probe_id"],
              "answer": "5", "problem_id": "P03", "code": P03_LE}
    settled = full.post("/probe/answer", json=answer).json()["diagnosis"]
    assert settled["status"] == "confident" and settled["top"][0]["id"] == "M08"
    after = full.get(f"/learner/{learner_id}").json()
    p_once = after["misconceptions"]["M08"]["p_active"]
    assert p_once > 0.5 and after["misconceptions"]["M08"]["state"] == "ACTIVE"
    assert len(after["attempts"]) == 1                          # the same attempt, not a second row

    full.post("/probe/answer", json=answer)                     # the same answer again changes nothing
    assert full.get(f"/learner/{learner_id}").json()["misconceptions"]["M08"]["p_active"] == p_once


def test_a_rule_based_mission_lights_its_star_without_moving_any_probability(full):
    learner_id = make_learner(full)
    body = attempt(full, learner_id, "P20", P20_OK)
    assert body["model_version"] == "rules-v1" and body["tests"]["passed"] == body["tests"]["total"]
    learner = full.get(f"/learner/{learner_id}").json()
    assert learner["nodes"]["functions:P20"]["status"] == "done"
    assert {e["p_active"] for e in learner["misconceptions"].values()} == {POPULATION_PRIOR}
    lit = star(chart(full, learner_id), "clamp_range")
    assert lit["state"] == "known" and lit["brightness"] == 1.0


def test_a_solved_trial_lights_its_exam_star(full):
    learner_id = make_learner(full)
    exam_id = full.post("/trials/start", json={"learner_id": learner_id}).json()["exam_id"]
    wrong = full.post("/trials/answer", json={"exam_id": exam_id, "item_id": "T2_binary_search", "code": ""}).json()
    assert not wrong["diagnosis"]["is_correct"]
    right = full.post("/trials/answer", json={"exam_id": exam_id, "item_id": "T3_bubble_sort", "code": T3_OK}).json()
    assert right["diagnosis"]["is_correct"]
    view = chart(full, learner_id)
    assert star(view, "bubble_sort")["cleared"] and not star(view, "binary_search")["cleared"]
    learner = full.get(f"/learner/{learner_id}").json()
    assert {e["p_active"] for e in learner["misconceptions"].values()} == {POPULATION_PRIOR}
