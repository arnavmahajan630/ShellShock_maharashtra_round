"""Star chart: the learner's knowledge laid over a map of C (package K1).

    from ml.learner.concept_graph import load_graph, load_problems, build_view
    view = build_view(load_graph(), load_problems(), learner)     # learner = Store.get_learner(...)

The map is `ml/data/concept_graph.json` (hand-written): 10 taught topics, one skill per problem
family of the bank plus the app-only planet missions, the parts of C with no levels
("uncharted"), and the "needs this first" lines. Nothing is trained here. A star's numbers come
from two things that already exist: the learner's P(active) per misconception (the knowledge
table) and the exposure e_ik each problem states for the misconceptions it can show.

For a skill with exposures e_k and probabilities P_k:

    solid       = product over k of (1 - e_k * P_k)
    brightness  = solid            (cleared skills)
    risk        = 1 - solid        (skills not cleared yet)
    shaky       = cleared, and a linked misconception is ACTIVE, TREATING, PROBATION or RELAPSED
    at_risk     = not cleared, and a linked misconception has P >= DIAGNOSIS_ACTIVE_P and
                  e_k * P_k >= AT_RISK_MIN (a diagnosed mistake with a weak link is not a halo)

A skill is cleared when one of its record ids has an attempt with every test passed, or a node
marked "done". Record ids are the bank problem of the family plus `also_cleared_by`; a skill
with `"record_ids": []` cannot be cleared from stored records (the app serves a different
problem under that id, see P13 and P14 in notes/K1.md).

The suggested next star is the playable, uncleared skill with the largest expected information
gain (`ml.exam.select.class_eigs`, the exam's own score). It is the most informative next step,
not a teaching order.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from ml.contracts.classes import CLASS_INFO, MISCONCEPTIONS
from ml.contracts.params import DIAGNOSIS_ACTIVE_P, POPULATION_PRIOR
from ml.exam.select import class_eigs

ROOT = Path(__file__).resolve().parents[2]
GRAPH_PATH = ROOT / "ml" / "data" / "concept_graph.json"
PROBLEM_DIRS = (ROOT / "ml" / "problems" / "main", ROOT / "ml" / "problems" / "dsa")

LIVE_STATES = ("ACTIVE", "TREATING", "PROBATION", "RELAPSED")
FIRST_ANGLE = -90.0             # the first skill of a constellation sits at the top
AT_RISK_MIN = 0.35              # e_k * P_k: a strong link (0.7) at the diagnosis threshold (0.5)

_cache = {"graph_mtime": None, "graph": None, "problems": None}


# ---------------------------------------------------------------- loading

def load_graph(path=GRAPH_PATH):
    path = Path(path)
    if path != GRAPH_PATH:
        return json.loads(path.read_text(encoding="utf-8"))
    mtime = path.stat().st_mtime
    if _cache["graph_mtime"] != mtime:
        _cache["graph"] = json.loads(path.read_text(encoding="utf-8"))
        _cache["graph_mtime"] = mtime
    return _cache["graph"]


def load_problems():
    """The problem bank by id (ml/problems/main and ml/problems/dsa)."""
    if _cache["problems"] is None:
        found = {}
        for folder in PROBLEM_DIRS:
            for path in sorted(folder.glob("*.json")):
                problem = json.loads(path.read_text(encoding="utf-8"))
                found[problem["problem_id"]] = problem
        _cache["problems"] = found
    return _cache["problems"]


def _by_family(problems):
    return {problem["family"]: problem for problem in problems.values() if problem.get("family")}


def check(graph, problems):
    """Everything wrong with the chart file, as a list of sentences. Empty when it is consistent."""
    errors = []
    topics = {topic["id"] for topic in graph["topics"]}
    families = _by_family(problems)
    ids = [skill["id"] for skill in graph["skills"]] + [star["id"] for star in graph["uncharted"]]
    for repeated in sorted({i for i in ids if ids.count(i) > 1} | (set(ids) & topics)):
        errors.append(f"id {repeated} is used twice")
    used = [skill["family"] for skill in graph["skills"] if skill.get("family")]
    for family in sorted(set(families) - set(used)):
        errors.append(f"bank family {family} has no skill")
    for family in sorted({f for f in used if used.count(f) > 1}):
        errors.append(f"family {family} has two skills")
    for skill in graph["skills"]:
        if skill["topic"] not in topics:
            errors.append(f"skill {skill['id']} names unknown topic {skill['topic']}")
        if skill.get("family"):
            if skill["family"] not in families:
                errors.append(f"skill {skill['id']} names unknown family {skill['family']}")
        elif not skill.get("problems"):
            errors.append(f"skill {skill['id']} has neither a family nor problems")
    known = topics | set(ids)
    for a, b in graph["prereqs"]:
        for end in (a, b):
            if end not in known:
                errors.append(f"prereq end {end} does not exist")
    if _has_cycle(graph["prereqs"]):
        errors.append("the prereq lines form a loop")
    width, height = graph["canvas"]["width"], graph["canvas"]["height"]
    for star in _placed(graph):
        x, y = star["pos"]
        if not (0 <= x <= width and 0 <= y <= height):
            errors.append(f"{star['id']} is outside the canvas")
    return errors


def _has_cycle(pairs):
    after = {}
    for a, b in pairs:
        after.setdefault(a, []).append(b)
    done, open_ = set(), set()

    def visit(node):
        if node in done:
            return False
        if node in open_:
            return True
        open_.add(node)
        if any(visit(nxt) for nxt in after.get(node, ())):
            return True
        open_.discard(node)
        done.add(node)
        return False

    return any(visit(node) for node in list(after))


# ---------------------------------------------------------------- layout

def _placed(graph):
    """Every star with a position: skills on a ring around their topic, uncharted stars as written."""
    topics = {topic["id"]: topic for topic in graph["topics"]}
    rings = {}
    for skill in graph["skills"]:
        rings.setdefault(skill["topic"], []).append(skill)
    out = []
    for topic_id, members in rings.items():
        topic = topics.get(topic_id)
        if topic is None:
            continue
        cx, cy = topic["pos"]
        radius = float(topic.get("radius", 5))
        for index, skill in enumerate(members):
            if skill.get("pos"):
                pos = list(skill["pos"])
            else:
                angle = math.radians(FIRST_ANGLE + 360.0 * index / len(members))
                pos = [round(cx + radius * math.cos(angle), 2), round(cy + radius * math.sin(angle), 2)]
            out.append({**skill, "pos": pos})
    out.extend(dict(star) for star in graph["uncharted"])
    return out


def _lines(graph):
    """Constellation lines: consecutive skills of a topic, closed into a ring when there are 3 or more."""
    rings = {}
    for skill in graph["skills"]:
        rings.setdefault(skill["topic"], []).append(skill["id"])
    edges = []
    for members in rings.values():
        pairs = list(zip(members, members[1:]))
        if len(members) >= 3:
            pairs.append((members[-1], members[0]))
        edges.extend({"from": a, "to": b, "kind": "line"} for a, b in pairs)
    return edges


# ---------------------------------------------------------------- the learner's view

def _records(learner):
    """(ids cleared, ids tried) from the learner's attempts and nodes."""
    cleared, tried = set(), set()
    for attempt in learner.get("attempts") or []:
        tried.add(attempt["problem_id"])
        if attempt["total"] > 0 and attempt["passed"] == attempt["total"]:
            cleared.add(attempt["problem_id"])
    for key, node in (learner.get("nodes") or {}).items():
        record_id = key.split(":", 1)[-1]
        tried.add(record_id)
        if (node or {}).get("status") == "done":
            cleared.add(record_id)
    return cleared, tried


def _hazards(exposure, entries):
    rows = []
    for cls, strength in (exposure or {}).items():
        if cls not in MISCONCEPTIONS:
            continue
        entry = entries.get(cls) or {}
        rows.append({
            "id": cls, "name": CLASS_INFO[cls]["name"], "subtitle": CLASS_INFO[cls].get("subtitle"),
            "p_active": round(float(entry.get("p_active", POPULATION_PRIOR)), 4),
            "state": entry.get("state", "UNSEEN"), "exposure": float(strength),
        })
    rows.sort(key=lambda row: (-row["exposure"] * row["p_active"], row["id"]))
    return rows


def _named(rows):
    return [{"id": row["id"], "name": row["name"]} for row in rows]


def build_view(graph, problems, learner):
    """The chart for one learner. `learner` is the dict of `Store.get_learner`."""
    entries = learner.get("misconceptions") or {}
    p_active = {cls: float(entry.get("p_active", POPULATION_PRIOR)) for cls, entry in entries.items()}
    cleared_ids, tried_ids = _records(learner)
    families = _by_family(problems)

    stars, gains = [], {}
    for index, raw in enumerate(_placed(graph)):
        if "topic" not in raw:
            stars.append({
                "id": raw["id"], "label": raw["label"], "kind": "uncharted", "topic": None, "pos": raw["pos"],
                "state": "uncharted", "cleared": False, "brightness": None, "risk": None, "at_risk": False,
                "risk_from": [], "hazards": [], "problems": [], "playable": None, "suggested": False,
            })
            continue
        problem = families.get(raw.get("family"))
        if problem is not None:
            shown = [{"problem_id": problem["problem_id"], "name": problem["name"]}]
            exposure = problem.get("exposure") or {}
            record_ids = [problem["problem_id"]]
        else:
            shown = list(raw.get("problems") or [])
            exposure = {}
            record_ids = [p["problem_id"] for p in shown]
        if "record_ids" in raw:
            record_ids = list(raw["record_ids"])
        record_ids += list(raw.get("also_cleared_by") or [])

        hazards = _hazards(exposure, entries)
        solid = 1.0
        for row in hazards:
            solid *= 1.0 - row["exposure"] * row["p_active"]
        cleared = any(record_id in cleared_ids for record_id in record_ids)
        tried = any(record_id in tried_ids for record_id in record_ids)
        live = [row for row in hazards if row["state"] in LIVE_STATES]
        high = [row for row in hazards if row["p_active"] >= DIAGNOSIS_ACTIVE_P
                and row["exposure"] * row["p_active"] >= AT_RISK_MIN]
        if cleared:
            state = "shaky" if live else "known"
        else:
            state = "attempted" if tried else "unexplored"
        stars.append({
            "id": raw["id"], "label": raw["label"], "kind": "skill", "topic": raw["topic"], "pos": raw["pos"],
            "state": state, "cleared": cleared,
            "brightness": round(solid, 4) if cleared else None,
            "risk": None if cleared else round(1.0 - solid, 4),
            "at_risk": bool(high) and not cleared,
            "risk_from": _named(live if cleared else high),
            "hazards": hazards, "problems": shown, "playable": raw.get("playable"), "suggested": False,
        })
        if not cleared and raw.get("playable"):
            parts = class_eigs(p_active, exposure)
            gains[raw["id"]] = (sum(parts.values()), -index, parts)

    suggested = None
    if gains:
        star_id = max(gains, key=lambda key: gains[key][:2])
        star = next(s for s in stars if s["id"] == star_id)
        star["suggested"] = True
        suggested = {"star_id": star_id, "label": star["label"], "route": star["playable"],
                     "reason": _reason(gains[star_id][2])}

    topics = []
    for topic in graph["topics"]:
        members = [s for s in stars if s["topic"] == topic["id"]]
        topics.append({"id": topic["id"], "label": topic["label"], "pos": topic["pos"],
                       "radius": topic.get("radius", 5), "skills_total": len(members),
                       "skills_cleared": sum(s["cleared"] for s in members)})

    charted = [s for s in stars if s["kind"] == "skill"]
    return {
        "learner_id": learner.get("learner_id"),
        "canvas": dict(graph["canvas"]),
        "topics": topics,
        "stars": stars,
        "edges": _lines(graph) + [{"from": a, "to": b, "kind": "prereq"} for a, b in graph["prereqs"]],
        "summary": {
            "charted": len(charted),
            "cleared": sum(s["cleared"] for s in charted),
            "known": sum(s["state"] == "known" for s in charted),
            "shaky": sum(s["state"] == "shaky" for s in charted),
            "at_risk": sum(s["at_risk"] for s in charted),
            "uncharted": len(stars) - len(charted),
        },
        "suggested_next": suggested,
    }


def _reason(parts):
    ranked = sorted(parts, key=lambda cls: (-parts[cls], cls))
    names = [CLASS_INFO[cls]["name"] for cls in ranked[:2] if parts[cls] > 0]
    if len(names) == 2:
        return f"Most informative about {names[0]} and {names[1]}."
    if names:
        return f"Most informative about {names[0]}."
    return "Not explored yet."
