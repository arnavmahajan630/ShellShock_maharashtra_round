"""Package K1: the star chart file and its builder (no server, no model)."""
import math

import pytest

from ml.contracts.classes import MISCONCEPTIONS
from ml.contracts.params import POPULATION_PRIOR
from ml.learner import concept_graph as cg
from server.app.store import Store


@pytest.fixture(scope="module")
def graph():
    return cg.load_graph()


@pytest.fixture(scope="module")
def problems():
    return cg.load_problems()


def blank(**extra):
    entry = {"state": "UNSEEN", "p_active": POPULATION_PRIOR}
    return {"learner_id": "x", "misconceptions": {cls: dict(entry) for cls in MISCONCEPTIONS},
            "nodes": {}, "attempts": [], **extra}


def star(view, star_id):
    return next(s for s in view["stars"] if s["id"] == star_id)


def test_chart_file_is_consistent(graph, problems):
    assert cg.check(graph, problems) == []
    families = {p["family"] for p in problems.values()}
    assert len(families) == len(problems) == 34
    assert {s["family"] for s in graph["skills"] if s.get("family")} == families
    assert len(graph["skills"]) == 42 and len(graph["uncharted"]) == 16
    assert all("topic" not in s and "family" not in s and "problems" not in s for s in graph["uncharted"])


def test_check_reports_what_is_wrong(graph, problems):
    broken = {**graph, "skills": graph["skills"][1:],
              "prereqs": graph["prereqs"] + [["loops", "variables"], ["loops", "nowhere"]]}
    errors = cg.check(broken, problems)
    assert any("has no skill" in e for e in errors)
    assert any("nowhere" in e for e in errors)
    assert any("form a loop" in e for e in errors)


def test_layout_and_edges(graph, problems):
    view = cg.build_view(graph, problems, blank())
    ids = {s["id"] for s in view["stars"]} | {t["id"] for t in view["topics"]}
    assert all(e["from"] in ids and e["to"] in ids and e["kind"] in ("line", "prereq") for e in view["edges"])
    width, height = view["canvas"]["width"], view["canvas"]["height"]
    for s in view["stars"]:
        assert 0 <= s["pos"][0] <= width and 0 <= s["pos"][1] <= height
    loops = next(t for t in view["topics"] if t["id"] == "loops")
    for s in (s for s in view["stars"] if s["topic"] == "loops"):
        assert math.dist(s["pos"], loops["pos"]) == pytest.approx(loops["radius"], abs=0.02)
    assert loops["skills_total"] == 6 and loops["skills_cleared"] == 0


def test_blank_learner_has_nothing_lit_and_no_halos(graph, problems):
    view = cg.build_view(graph, problems, blank())
    skills = [s for s in view["stars"] if s["kind"] == "skill"]
    assert {s["state"] for s in skills} == {"unexplored"}
    assert {s["state"] for s in view["stars"] if s["kind"] == "uncharted"} == {"uncharted"}
    assert view["summary"] == {"charted": 42, "cleared": 0, "known": 0, "shaky": 0, "at_risk": 0, "uncharted": 16}
    count_loop = star(view, "count_loop")           # P01: M01 0.7, M02 0.7, M07 0.4, every P at 0.10
    assert count_loop["risk"] == pytest.approx(1 - 0.93 * 0.93 * 0.96, abs=1e-4)
    assert count_loop["brightness"] is None and count_loop["risk_from"] == []
    assert {h["id"] for h in count_loop["hazards"]} == {"M01", "M02", "M07"}
    next_ = view["suggested_next"]
    chosen = star(view, next_["star_id"])
    assert chosen["suggested"] and chosen["playable"] == next_["route"] and not chosen["cleared"]
    assert sum(s["suggested"] for s in view["stars"]) == 1
    assert cg.build_view(graph, problems, blank())["suggested_next"] == next_       # deterministic


def test_seeded_learner(graph, problems, tmp_path):
    learner = Store(tmp_path / "seed.db").seed("demo-learner")      # M08 ACTIVE 0.90, M01 STABLE 0.08, M06 MASTERED
    view = cg.build_view(graph, problems, learner)

    known = star(view, "count_loop")                # P01 done; M01 0.08, M02 and M07 at the prior
    assert known["state"] == "known" and known["risk"] is None
    assert known["brightness"] == pytest.approx((1 - 0.7 * 0.08) * (1 - 0.7 * 0.1) * (1 - 0.4 * 0.1), abs=1e-4)

    shaky = star(view, "index_access")              # P08 done, but it exposes M08, which is ACTIVE
    assert shaky["state"] == "shaky" and shaky["cleared"]
    assert shaky["risk_from"] == [{"id": "M08", "name": "Index Origin Fault"}]
    assert not shaky["at_risk"]

    halo = star(view, "array_max")                  # P09 not cleared; M08 0.90 with exposure 0.7
    assert halo["state"] == "unexplored" and halo["at_risk"]
    assert halo["risk_from"][0]["id"] == "M08" and halo["risk"] > 0.6

    tried = star(view, "array_accumulate")          # P03: three failed attempts, node "unstable"
    assert tried["state"] == "attempted" and tried["at_risk"]

    app_only = star(view, "sync_ratio")             # the app's P13: no linked misconception
    assert app_only["state"] == "known" and app_only["brightness"] == 1.0 and app_only["hazards"] == []
    assert star(view, "ratio")["state"] == "unexplored"      # the bank's P13 cannot be cleared from records

    summary = view["summary"]
    assert summary["cleared"] == summary["known"] + summary["shaky"] == 7
    assert summary["shaky"] == 1 and summary["at_risk"] >= 2
    conditions = next(t for t in view["topics"] if t["id"] == "conditions")
    assert conditions["skills_cleared"] == 4


def test_a_weak_link_is_not_a_halo(graph, problems):
    learner = blank()
    learner["misconceptions"]["M06"] = {"state": "ACTIVE", "p_active": 0.72}
    view = cg.build_view(graph, problems, learner)
    assert star(view, "branch_bands")["at_risk"]                # P12 exposes M06 at 0.7
    weak = star(view, "linear_search")                          # Q01 exposes M06 at 0.4 only
    assert not weak["at_risk"] and weak["risk_from"] == []
    assert "M06" in [h["id"] for h in weak["hazards"]]


def test_a_solved_trial_lights_its_exam_star(graph, problems):
    view = cg.build_view(graph, problems, blank(nodes={"trials:T3_bubble_sort": {"status": "done", "stars": 3}}))
    assert star(view, "bubble_sort")["state"] == "known"
    assert star(view, "selection_sort")["state"] == "unexplored"


def test_a_passing_attempt_clears_even_after_a_later_failure(graph, problems):
    attempts = [{"problem_id": "P11", "passed": 5, "total": 5, "status": "correct", "top": "CORRECT"},
                {"problem_id": "P11", "passed": 1, "total": 5, "status": "confident", "top": "M06"}]
    learner = blank(attempts=attempts, nodes={"conditions:P11": {"status": "unstable", "stars": 3}})
    assert star(cg.build_view(graph, problems, learner), "equality_check")["cleared"]


def test_suggestion_follows_what_is_uncertain(graph, problems):
    learner = blank()
    learner["misconceptions"]["M04"] = {"state": "UNSEEN", "p_active": 0.45}     # most uncertain: near one half
    view = cg.build_view(graph, problems, learner)
    chosen = star(view, view["suggested_next"]["star_id"])
    assert "M04" in [h["id"] for h in chosen["hazards"]]
    assert "Fraction Shear" in view["suggested_next"]["reason"]
