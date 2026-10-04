"""Package S1: SQLite store + learner routes.

Acceptance (06 3): seed a learner -> reassess -> STABLE (`test_acceptance_seed_reassess_stable`,
also runnable as `python -m tests.S1.demo_flow`). Every test uses a temp database.
"""
import pytest

from ml.contracts import schemas
from ml.contracts.classes import MISCONCEPTIONS
from ml.contracts.params import STABLE_P
from server.app import pipeline, store as store_module
from tests.S1 import demo_flow
from tests.S1.conftest import P03_LE


def reassess(client, learner_id, cls, item_id, item_type, **result):
    response = client.post("/reassess", json={"learner_id": learner_id, "class": cls, "item_id": item_id,
                                              "item_type": item_type, "result": result})
    assert response.status_code == 200, response.text
    return response.json()


def seeded(client, learner_id="demo-learner"):
    response = client.post(f"/learner/{learner_id}/seed")
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------- learner

def test_create_and_get_learner(client):
    created = client.post("/learner", json={"callsign": "ACE"}).json()
    assert created["learner_id"].startswith("lrn_") and created["model_version"]
    other = client.post("/learner", json={"callsign": "ACE"}).json()
    assert other["learner_id"] != created["learner_id"]           # not the fixture's constant id

    got = client.get(f"/learner/{created['learner_id']}")
    assert got.status_code == 200
    body = got.json()
    schemas.LearnerResponse.model_validate(body)
    assert body["callsign"] == "ACE" and body["attempts"] == [] and body["nodes"] == {}
    assert list(body["misconceptions"]) == MISCONCEPTIONS
    assert all(e["state"] == "UNSEEN" and e["p_active"] == 0.1 for e in body["misconceptions"].values())


def test_create_needs_callsign(client):
    assert client.post("/learner", json={}).status_code == 422
    assert client.post("/learner", json={"callsign": "  "}).status_code == 422


def test_unknown_learner_is_404_without_a_stack_trace(client):
    response = client.get("/learner/nope")
    assert response.status_code == 404 and set(response.json()) == {"error"}
    assert client.post("/reassess", json={"learner_id": "nope", "class": "M08", "item_id": "trap_M08",
                                          "item_type": "trap", "result": {"correct": True}}).status_code == 404


def test_seed_matches_the_contract(client):
    body = seeded(client)
    schemas.LearnerResponse.model_validate(body)
    m = body["misconceptions"]
    assert (m["M01"]["state"], m["M08"]["state"], m["M06"]["state"]) == ("STABLE", "ACTIVE", "MASTERED")
    assert m["M08"]["p_active"] == 0.9
    assert sum(1 for e in m.values() if e["state"] == "UNSEEN") == 14
    assert len(body["attempts"]) == 12
    assert [a["attempt_id"] for a in body["attempts"]][:3] == ["at_001", "at_002", "at_003"]
    assert body["callsign"] == "NOVA" and body["nodes"]["loops:P03"] == {"status": "unstable", "stars": 0}


def test_seed_resets_and_is_deterministic(client):
    first = seeded(client, "L1")
    reassess(client, "L1", "M08", "trap_M08", "trap", correct=True)
    again = seeded(client, "L1")
    for key in ("misconceptions", "attempts", "nodes", "callsign"):
        assert again[key] == first[key]
    assert seeded(client, "L2")["attempts"][0]["attempt_id"] == "L2:at_001"   # attempt ids are global keys


def test_seed_keeps_the_callsign_of_an_existing_learner(client):
    learner_id = client.post("/learner", json={"callsign": "ACE"}).json()["learner_id"]
    assert seeded(client, learner_id)["callsign"] == "ACE"


# ---------------------------------------------------------------- acceptance

def test_acceptance_seed_reassess_stable(full):
    """06 3 S1: seed learner -> reassess -> STABLE, through the real app (all routers loaded)."""
    final = demo_flow.run(demo_flow.AppClient(full), say=lambda *_: None)
    assert final["state"] == "STABLE" and final["p_active"] < STABLE_P


def test_stable_by_reassess_alone(client):
    """No recorded intervention: a reassess item on an ACTIVE class counts it as taken."""
    seeded(client)
    mid = reassess(client, "demo-learner", "M08", "trap_M08", "trap", correct=True, answer="outside the array")
    assert mid["state"] == "PROBATION" and mid["resolved_level"] is None
    assert [c["met"] for c in mid["conditions"]] == [False, True, False]        # 0.19 is not under 0.15 yet
    # intervention 0.9 -> 0.585, trap correct (g .15, s .10) -> 0.190
    assert mid["p_active"] == pytest.approx(0.19, abs=0.005)
    final = reassess(client, "demo-learner", "M08", mid["next_item"]["item_id"], "transfer_code", passed=True)
    assert final["state"] == "STABLE" and final["resolved_level"] == "STABLE"
    assert final["p_active"] == pytest.approx(0.0765, abs=0.002)
    assert final["next_item"] is None and all(c["met"] for c in final["conditions"])
    schemas.ReassessResponse.model_validate(final)


# ---------------------------------------------------------------- reassess

def test_conditions_have_the_plan_shape_and_details(client):
    seeded(client)
    body = reassess(client, "demo-learner", "M08", "trap_M08", "trap", correct=False, answer="9")
    ids = [c["id"] for c in body["conditions"]]
    assert ids == ["p_active", "trap", "transfer"]
    trap = body["conditions"][1]
    assert not trap["met"] and trap["detail"] == "predicted 9, actual: outside the array"   # from items.json
    assert body["conditions"][2]["detail"] == "not attempted yet"
    assert body["conditions"][0]["label"] == "Misconception probability < 0.15"


def test_failed_trap_sends_the_learner_back_to_treating(client):
    seeded(client)
    body = reassess(client, "demo-learner", "M08", "trap_M08", "trap", correct=False, answer="9")
    # 0.9 -> 0.585 (intervention) -> 0.92 (wrong trap): over 0.5 after an item, so PROBATION -> TREATING
    assert body["p_active"] > 0.5 and body["state"] == "TREATING" and body["next_item"] is None
    entry = store_module.get_store().get_entry("demo-learner", "M08")
    assert entry["flags"]["trap_passed"] is False
    # the next intervention request goes to the next modality
    again = client.post("/learner/demo-learner/intervene", json={"class": "M08", "problem_id": "P03",
                                                                 "code": P03_LE}).json()
    assert again["learner_state"]["state"] == "TREATING"


def test_passing_the_same_family_does_not_count_as_transfer(client):
    seeded(client)
    reassess(client, "demo-learner", "M08", "trap_M08", "trap", correct=True)
    # P03 is the family where M08 was found (array_accumulate): passing it is not a transfer
    body = reassess(client, "demo-learner", "M08", "P03", "transfer_code", passed=True)
    assert body["state"] == "PROBATION"
    transfer = body["conditions"][2]
    assert not transfer["met"] and "does not count" in transfer["detail"]


def test_transfer_alone_cannot_make_it_stable(client):
    """The trap carries the weight (03 8.4): passing a transfer without the trap is not enough."""
    seeded(client)
    problem = next(p for p in pipeline._problems().values() if p["family"] == "array_max")
    body = reassess(client, "demo-learner", "M08", problem["problem_id"], "transfer_code", passed=True)
    assert body["state"] == "PROBATION"
    assert [c["met"] for c in body["conditions"]] == [False, False, True]
    assert body["conditions"][2]["detail"].startswith(problem["problem_id"])
    assert body["next_item"] == {"item_id": "trap_M08", "item_type": "trap"}


def test_next_item_order_is_trap_then_transfer(client):
    seeded(client)
    first = reassess(client, "demo-learner", "M08", "intervention", "intervention_done")
    assert first["state"] == "PROBATION"
    assert first["next_item"] == {"item_id": "trap_M08", "item_type": "trap"}
    second = reassess(client, "demo-learner", "M08", "trap_M08", "trap", correct=True)
    nxt = second["next_item"]
    assert nxt["item_type"] == "transfer_code"
    family = pipeline.get_problem(nxt["item_id"])["family"]
    assert family in ("index_access", "array_max") and family != "array_accumulate"


def test_relapse_and_ghost_return(client):
    seeded(client)
    # M01 STABLE at 0.08. One wrong trap is not enough (0.43), two are (0.86).
    one = reassess(client, "demo-learner", "M01", "trap_M01", "trap", correct=False, answer="4")
    assert one["state"] == "STABLE" and one["p_active"] == pytest.approx(0.425, abs=0.005)
    two = reassess(client, "demo-learner", "M01", "trap_M01", "trap", correct=False, answer="4")
    assert two["state"] == "RELAPSED" and two["resolved_level"] is None
    # a ghost return passed from STABLE (0.08 -> 0.025) reaches MASTERED
    seeded(client, "g")
    up = reassess(client, "g", "M01", "P02", "ghost", correct=True)
    assert up["state"] == "MASTERED" and up["p_active"] < 0.10 and up["resolved_level"] == "MASTERED"


def test_ghost_failure_relapses(client):
    seeded(client)
    body = reassess(client, "demo-learner", "M01", "P02", "ghost", correct=False)
    assert body["state"] == "RELAPSED"


def test_relapsed_class_goes_through_treating_again(client):
    seeded(client)
    reassess(client, "demo-learner", "M01", "P02", "ghost", correct=False)
    body = reassess(client, "demo-learner", "M01", "x", "intervention_done")
    assert body["state"] == "PROBATION"                         # relapse -> (implicit) treating -> probation
    entry = store_module.get_store().get_entry("demo-learner", "M01")
    assert entry["flags"]["trap_passed"] is False               # the checklist starts over


def test_reassess_validation(client):
    seeded(client)
    base = {"learner_id": "demo-learner", "class": "M08", "item_id": "x", "item_type": "trap", "result": {"correct": 1}}
    assert client.post("/reassess", json={k: v for k, v in base.items() if k != "item_id"}).status_code == 422
    assert client.post("/reassess", json={**base, "result": "yes"}).status_code == 422
    assert client.post("/reassess", json={**base, "result": {}}).status_code == 422
    assert client.post("/reassess", json={**base, "item_type": "nonsense"}).status_code == 422
    assert client.post("/reassess", json={**base, "class": "ZZ9"}).status_code == 422
    assert client.post("/reassess", json={**base, "class": "M09"}).status_code == 422


def test_hint_is_weaker_evidence(client):
    seeded(client, "a")
    seeded(client, "b")
    plain = reassess(client, "a", "M08", "P09", "transfer_code", passed=True)
    hinted = reassess(client, "b", "M08", "P09", "transfer_code", passed=True, hint=True)
    assert hinted["p_active"] > plain["p_active"]


def test_predict_output_with_the_class_belief_is_a_trap(client):
    seeded(client)
    body = reassess(client, "demo-learner", "M08", "qi_1", "predict_output", correct=True, belief={"M08": "9"})
    assert body["conditions"][1]["met"] is True and body["state"] == "PROBATION"


# ---------------------------------------------------------------- intervene

def test_intervene_by_learner_path(client):
    seeded(client)
    response = client.post("/learner/demo-learner/intervene",
                           json={"class": "M08", "problem_id": "P03", "code": P03_LE})
    assert response.status_code == 200, response.text
    body = response.json()
    schemas.Intervention.model_validate(body)
    assert body["class"] == "M08" and body["modality"] == "memory_strip"
    assert body["learner_state"]["state"] == "TREATING"
    assert body["learner_state"]["interventions"] == ["memory_strip"]
    assert client.get("/learner/demo-learner").json()["misconceptions"]["M08"]["state"] == "TREATING"

    # second request: the modality already tried is skipped
    again = client.post("/learner/demo-learner/intervene",
                        json={"class": "M08", "problem_id": "P03", "code": P03_LE}).json()
    assert again["modality"] != "memory_strip"
    assert again["learner_state"]["interventions"] == ["memory_strip", again["modality"]]
    assert again["learner_state"]["state"] == "TREATING"

    # reassess after the recorded intervention: TREATING -> PROBATION with the learning step applied
    body = reassess(client, "demo-learner", "M08", "trap_M08", "trap", correct=True)
    assert body["state"] == "PROBATION" and body["p_active"] == pytest.approx(0.19, abs=0.005)


def test_intervene_contract_body_with_learner_id_and_attempt_id(client):
    """03 11: {learner_id, attempt_id, class, modality?}. The attempt must be known to the store."""
    seeded(client)
    store_module.get_store().record_attempt("demo-learner", "at_x1", "P03", 0, 5, "confident", "M08", code=P03_LE)
    response = client.post("/intervene", json={"learner_id": "demo-learner", "attempt_id": "at_x1",
                                               "class": "M08", "modality": "counterexample"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["modality"] == "counterexample" and body["counterexample"]
    assert client.get("/learner/demo-learner").json()["misconceptions"]["M08"]["state"] == "TREATING"


def test_intervene_stateless_without_a_learner(client):
    response = client.post("/intervene", json={"class": "M08", "problem_id": "P03", "code": P03_LE})
    assert response.status_code == 200
    assert response.json()["modality"] == "memory_strip" and "learner_state" not in response.json()


def test_intervene_does_not_move_an_unseen_class(client):
    seeded(client)
    body = client.post("/learner/demo-learner/intervene",
                       json={"class": "M01", "problem_id": "P03", "code": P03_LE}).json()
    assert body["learner_state"]["state"] == "STABLE"            # STABLE stays STABLE (no diagnosis, no relapse)


def test_intervene_errors(client):
    seeded(client)
    assert client.post("/learner/demo-learner/intervene", json={"problem_id": "P03", "code": P03_LE}).status_code == 422
    assert client.post("/learner/demo-learner/intervene", json={"class": "M08"}).status_code == 422
    assert client.post("/learner/demo-learner/intervene",
                       json={"class": "M08", "attempt_id": "unknown"}).status_code == 422
    assert client.post("/learner/nobody/intervene",
                       json={"class": "M08", "problem_id": "P03", "code": P03_LE}).status_code == 404
    response = client.post("/intervene", json={"class": "M08", "problem_id": "NOPE", "code": "int f(){return 0;}"})
    assert response.status_code == 404 and set(response.json()) == {"error"}


# ---------------------------------------------------------------- attempt results (not in the contract)

def test_attempt_result_updates_knowledge(client):
    learner_id = client.post("/learner", json={"callsign": "ACE"}).json()["learner_id"]
    url = f"/learner/{learner_id}/attempt-result"
    base = {"attempt_id": "a1", "problem_id": "P11", "passed": 0, "total": 5, "status": "confident", "top": "M07",
            "top_p": 0.8}
    out = client.post(url, json=base).json()
    (update,) = out["updates"]
    assert update["class"] == "M07" and update["state"] == "ACTIVE" and update["p_after"] > 0.5
    # P11's authored exposure for M07 is used, or the 0.5 default
    after = client.get(f"/learner/{learner_id}").json()
    assert after["misconceptions"]["M07"]["state"] == "ACTIVE" and len(after["attempts"]) == 1
    assert after["attempts"][0]["top"] == "M07"

    # an ambiguous attempt is logged but moves nothing
    out = client.post(url, json={**base, "attempt_id": "a2", "status": "ambiguous", "top": "M01"}).json()
    assert out["updates"] == []
    assert client.get(f"/learner/{learner_id}").json()["misconceptions"]["M01"]["state"] == "UNSEEN"

    # a pass whose latent class is M07 raises it a little further
    p_before = after["misconceptions"]["M07"]["p_active"]
    out = client.post(url, json={**base, "attempt_id": "a3", "passed": 5, "status": "correct", "top": "CORRECT",
                                 "latent": "M07"}).json()
    assert out["updates"][0]["kind"] == "latent_pass" and out["updates"][0]["p_after"] > p_before

    assert client.post(url, json={"attempt_id": "a4"}).status_code == 422


def test_attempt_result_node_progress(client):
    learner_id = client.post("/learner", json={"callsign": "ACE"}).json()["learner_id"]
    client.post(f"/learner/{learner_id}/attempt-result",
                json={"attempt_id": "n1", "problem_id": "P11", "passed": 5, "total": 5, "status": "correct",
                      "top": "CORRECT", "node": "conditions:P11"})
    assert client.get(f"/learner/{learner_id}").json()["nodes"]["conditions:P11"] == {"status": "done", "stars": 3}


# ---------------------------------------------------------------- the store itself

def test_state_survives_a_restart(temp_db):
    first = store_module.Store(str(temp_db))
    first.seed("keep")
    first.reassess("keep", "M08", "trap_M08", "trap", {"correct": True})
    second = store_module.Store(str(temp_db))                    # a new process would open the same file
    entry = second.get_entry("keep", "M08")
    assert entry["state"] == "PROBATION" and entry["flags"]["trap_passed"] is True
    assert [e["type"] for e in second.events("keep")] == ["seed", "reassess"]


def test_database_path_comes_from_the_environment(temp_db):
    assert store_module.db_path() == str(temp_db)
    store_module.get_store().create_learner("x", "lrn_path")
    assert temp_db.exists()


def test_default_database_is_git_ignored():
    import subprocess
    from pathlib import Path
    default = Path(store_module.DEFAULT_DB)
    out = subprocess.run(["git", "check-ignore", str(default)], capture_output=True, text=True,
                         cwd=store_module.ROOT)
    assert out.returncode == 0, "server/data/relearn.db must be ignored by git"


def test_prior_hook_for_the_pipeline(client):
    from server.app.routes import learner
    assert callable(pipeline.learner_prior_fn)
    assert learner._learner_prior("nobody") is None             # no database yet: nothing is created
    fresh = client.post("/learner", json={"callsign": "ACE"}).json()["learner_id"]
    assert pipeline.learner_prior_fn(fresh) is None             # no history: the stateless path
    seeded(client)
    prior = pipeline.learner_prior_fn("demo-learner")
    assert prior["M08"] == 0.9 and prior["M01"] == 0.08 and set(prior) == set(MISCONCEPTIONS)


def test_concurrent_writes_do_not_lose_updates(temp_db):
    import threading
    store = store_module.Store(str(temp_db))
    store.seed("c")
    def work():
        for _ in range(5):
            store.reassess("c", "M08", "P09", "transfer_code", {"passed": False})
    threads = [threading.Thread(target=work) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert store.get_entry("c", "M08")["times_seen"] == 1 + 20      # seeded once + 20 reassess calls, none lost


# ---------------------------------------------------------------- how it sits in the real app

def test_real_app_serves_the_learner_routes_not_fixtures(full):
    assert full.post("/learner", json={"callsign": "A"}).json()["learner_id"] != "demo-learner"
    live = full.get("/_routes").json()
    assert "learner" in live["live_routers"]
    for gone in ("POST /learner", "POST /reassess", "GET /learner/{learner_id}"):
        assert gone not in live["fixture_routes"]


def test_intervene_conflict_status(full):
    """POST /intervene is registered by routes/intervene.py and by routes/learner.py (see notes/S1.md).

    main.py includes routers alphabetically and Starlette uses the first match, so the earlier
    stateless route answers as long as routes/intervene.py exists. The test pins that rule: whoever
    comes first in LIVE_ROUTERS wins, and the learner-aware version is reachable on its own path.
    """
    from server.app.main import LIVE_ROUTERS
    full.post("/learner/demo-learner/seed")
    body = full.post("/intervene", json={"learner_id": "demo-learner", "class": "M08", "problem_id": "P03",
                                         "code": P03_LE}).json()
    learner_wins = "learner_state" in body
    if "intervene" in LIVE_ROUTERS:
        assert learner_wins == (LIVE_ROUTERS.index("learner") < LIVE_ROUTERS.index("intervene"))
        assert not learner_wins                                  # today: the older stateless route answers
        assert body["model_version"] == "rules-v1"
    else:
        assert learner_wins


def test_real_app_intervene_by_learner_path(full):
    full.post("/learner/demo-learner/seed")
    response = full.post("/learner/demo-learner/intervene",
                         json={"class": "M08", "problem_id": "P03", "code": P03_LE})
    assert response.status_code == 200
    assert response.json()["learner_state"]["state"] == "TREATING"
