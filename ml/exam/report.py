"""Debrief for one exam (ml_plan/03 §8.5.6).

Findings use the state ``next_state`` already produced while the exam ran.
Status words are the five names in §8.5.6. A tested class that matches none of
them is left out; that gap is written up in notes/D3.md.

The 0.3 and 0.7 cutoffs are the words in §8.5.6. They are not in ``params.py``.
"""
from __future__ import annotations

from ml.bayes.eig import expected_information_gain, load_probes
from ml.contracts.classes import CLASS_INFO, LABELS, MISCONCEPTIONS, TWIN_SETS
from ml.contracts.params import DIAGNOSIS_ACTIVE_P, EIG_MIN_BITS, POPULATION_PRIOR, PRIOR_FLOOR
from ml.contracts.schemas import (
    AdaptivityEntry,
    ExamReport,
    Finding,
    PublicProbe,
    Recommendation,
    SectorScore,
)
from ml.exam.select import practice_catalog
from ml.learner.state_machine import TRANSFER_FAMILIES

# §8.5.6, not stored in params.py.
P_BAND_LOW = 0.3
P_BAND_HIGH = 0.7
MAX_DEFERRED_PROBES = 3
RECOMMENDATION_COUNT = 2


def build_report(session):
    """An ``ExamReport`` for a finished session. Nothing here is shown mid-exam."""
    findings = _findings(session)
    uncertain = [finding.cls for finding in findings if finding.status == "UNCERTAIN"]
    report = ExamReport(
        exam_id=session["exam_id"],
        learner_id=session["learner_id"],
        items_answered=len(session["administered"]),
        time_used_s=int(session.get("time_used_s") or 0),
        sectors=_sectors(session),
        findings=findings,
        deferred_probes=_probes(session, uncertain),
        recommendations=_recommendations(session, findings),
        adaptivity_log=_log(session),
    )
    return ExamReport.model_validate(report.model_dump())


def _sectors(session):
    rows = []
    for sector in session["sectors"]:
        served = [row for row in session["administered"] if row["sector"] == sector]
        rows.append(SectorScore(
            sector=sector,
            items=[row["item_id"] for row in served],
            passed=sum(1 for row in served if row["passed"]),
            rating_before=float(session["theta0"][sector]),
            rating_after=float(session["theta"][sector]),
        ))
    return rows


def _log(session):
    entries = []
    for index, row in enumerate(session["administered"], start=1):
        entries.append(AdaptivityEntry(
            order=index,
            item_id=row["item_id"],
            sector=row["sector"],
            eig_bits=float(row["eig_bits"]),
            reason=row["reason"],
            outcome=row["outcome"],
        ))
    return entries


def _findings(session):
    """One pass assigns a status; a second pass fills ``twin_set`` when both twins are uncertain."""
    draft = []
    for cls in MISCONCEPTIONS:
        status = _status(session, cls)
        if status is None:
            continue
        draft.append((cls, status))
    uncertain = {cls for cls, status in draft if status == "UNCERTAIN"}
    findings = []
    for cls, status in draft:
        twin = session["ambiguous"].get(cls)
        if status == "UNCERTAIN" and not twin:
            twin = _partner_twin(cls, uncertain)
        findings.append(Finding(
            cls=cls,
            status=status,
            p_active=float(session["p"][cls]),
            name=CLASS_INFO[cls]["name"],
            subtitle=CLASS_INFO[cls]["subtitle"],
            state_after=session["states"][cls],
            twin_set=twin,
            evidence=list(session["evidence"].get(cls) or []),
        ))
    return findings


def _status(session, cls):
    evidence = session["evidence"].get(cls) or []
    ambiguous = cls in session["ambiguous"]
    if not evidence and not ambiguous:
        return "NOT_TESTED"
    state = session["states"][cls]
    p = float(session["p"][cls])
    p0 = float(session["p0"][cls])
    if state == "RELAPSED":
        return "RELAPSED"
    if state == "MASTERED":
        return "HELD"
    was_new = session["state0"][cls] == "UNSEEN" or p0 <= P_BAND_LOW
    if was_new and p >= DIAGNOSIS_ACTIVE_P:
        return "NEW"
    if ambiguous or P_BAND_LOW <= p <= P_BAND_HIGH:
        return "UNCERTAIN"
    return None


def _partner_twin(cls, uncertain):
    for set_id, info in TWIN_SETS.items():
        members = info["members"]
        if cls not in members:
            continue
        if any(other != cls and other in uncertain for other in members):
            return set_id
    return None


def _recommendations(session, findings):
    used = {row["item_id"] for row in session["administered"]}
    catalog = practice_catalog()
    strings = session.get("strings", True)
    rows = []
    for finding in findings:
        if finding.status not in ("NEW", "RELAPSED"):
            continue
        families = set(TRANSFER_FAMILIES.get(finding.cls, ()))
        problems = []
        for entry in catalog:
            if entry["problem_id"] in used or entry["family"] not in families:
                continue
            if not strings and entry.get("sector") == "strings":
                continue
            problems.append(entry["problem_id"])
            if len(problems) == RECOMMENDATION_COUNT:
                break
        if problems:
            rows.append(Recommendation(cls=finding.cls, problems=problems))
    return rows


def _probes(session, uncertain):
    """Up to three probes for UNCERTAIN findings, ranked by §6.4 expected information gain."""
    if not uncertain:
        return []
    posterior = _posterior_from_knowledge(session["p"])
    wanted = set(uncertain)
    ranked = []
    for probe in load_probes():
        belief = probe.get("belief") or {}
        if not wanted.intersection(belief):
            continue
        score = expected_information_gain(
            posterior, probe["options"], probe["correct"], belief,
        )
        if score < EIG_MIN_BITS:
            continue
        ranked.append((score, probe))
    ranked.sort(key=lambda pair: (-pair[0], pair[1]["probe_id"]))
    chosen = []
    for score, probe in ranked[:MAX_DEFERRED_PROBES]:
        belief = probe.get("belief") or {}
        for_class = max(
            (cls for cls in belief if cls in wanted),
            key=lambda cls: float(session["p"].get(cls, 0.0)),
        )
        chosen.append(PublicProbe(
            probe_id=probe["probe_id"],
            prompt=probe["prompt"],
            code=probe["code"],
            options=list(probe["options"]),
            eig_bits=float(score),
            for_class=for_class,
        ))
    return chosen


def _posterior_from_knowledge(p_active):
    """A simplex for probe EIG. Class mass follows P(A); CORRECT and OTHER stay at the floor."""
    raw = {}
    for label in LABELS:
        if label in MISCONCEPTIONS:
            raw[label] = max(float(p_active.get(label, POPULATION_PRIOR)), PRIOR_FLOOR)
        else:
            raw[label] = PRIOR_FLOOR
    total = sum(raw.values())
    return {label: value / total for label, value in raw.items()}
