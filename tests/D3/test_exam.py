"""D3: adaptive exam, fixed blueprint, and a report with one reason per item."""
import json
import math
from pathlib import Path

import pytest

from ml.contracts import schemas as S
from ml.contracts.classes import SECTORS
from ml.contracts.params import EXAM_LENGTH, FAIL_SIGNATURE_IF_NOT
from ml.exam.run import run_exam
from ml.exam.select import class_eig, load_pool, public_exam_item

ROOT = Path(__file__).resolve().parents[2]
DSA = ROOT / "ml" / "problems" / "dsa"


def _pass_diagnoser(problem, code):
    return {"status": "correct", "top": [], "posterior": {}, "latent": None}


def _fail_diagnoser(problem, code):
    return {
        "status": "confident",
        "top": [{"id": "D01", "p": 0.91}],
        "posterior": {"D01": 0.91},
        "latent": None,
    }


def _answers(item):
    if item["kind"] == "trace":
        return item["correct"]
    return "int unused(void) { return 0; }"


def _run(**kwargs):
    defaults = dict(
        learner_id="ada",
        exam_id="ex1",
        answers=_answers,
        diagnoser=_pass_diagnoser,
        mode="demo",
        p_active={"M01": 0.2},
        states={"M01": "STABLE"},
        stable_since={"M01": "Loops"},
        time_used_s=90,
    )
    defaults.update(kwargs)
    return run_exam(**defaults)


def _kinds(report):
    coding = trace = 0
    for entry in report.adaptivity_log:
        if entry.item_id.startswith("xt_"):
            trace += 1
        else:
            coding += 1
    return coding, trace


def _difficulty(problem_id):
    for path in DSA.glob("Q*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["problem_id"] == problem_id:
            return data["difficulty"]
    raise AssertionError(problem_id)


def test_class_eig_matches_the_binary_formula():
    p = 0.5
    e = 0.5
    fail_if_not = FAIL_SIGNATURE_IF_NOT
    p_fail = p * e + (1 - p) * fail_if_not

    def posterior(like_active, like_not):
        return (p * like_active) / (p * like_active + (1 - p) * like_not)

    def entropy(q):
        if q <= 0.0 or q >= 1.0:
            return 0.0
        return -(q * math.log2(q) + (1.0 - q) * math.log2(1.0 - q))

    expected = entropy(p) - (
        p_fail * entropy(posterior(e, fail_if_not))
        + (1 - p_fail) * entropy(posterior(1 - e, 1 - fail_if_not))
    )
    assert class_eig(p, e) == pytest.approx(expected)


def test_demo_exam_validates_with_a_reason_per_item():
    report = _run(diagnoser=_fail_diagnoser)
    payload = report.model_dump(by_alias=True)
    S.ExamReport.model_validate(payload)
    coding, trace = _kinds(report)
    assert (coding, trace) == EXAM_LENGTH["demo"]
    assert report.items_answered == coding + trace
    assert len(report.adaptivity_log) == report.items_answered
    assert _difficulty(report.adaptivity_log[0].item_id) == 1
    sectors = [entry.sector for entry in report.adaptivity_log]
    assert all(left != right for left, right in zip(sectors, sectors[1:]))
    for entry in report.adaptivity_log:
        assert entry.reason.strip()
        assert entry.reason != "fixed blueprint"
    by_sector = {row.sector: row for row in report.sectors}
    assert list(by_sector) == list(SECTORS)
    for entry in report.adaptivity_log:
        assert entry.item_id in by_sector[entry.sector].items


def test_same_seed_picks_the_same_items():
    first = _run(diagnoser=_fail_diagnoser)
    second = _run(diagnoser=_fail_diagnoser)
    assert [entry.item_id for entry in first.adaptivity_log] == [
        entry.item_id for entry in second.adaptivity_log
    ]


def test_adaptivity_off_still_returns_a_report():
    report = _run(adaptive=False, diagnoser=_fail_diagnoser)
    S.ExamReport.model_validate(report.model_dump(by_alias=True))
    coding, trace = _kinds(report)
    assert (coding, trace) == EXAM_LENGTH["demo"]
    assert report.items_answered == 3
    assert [entry.reason for entry in report.adaptivity_log] == ["fixed blueprint"] * 3
    assert [entry.item_id for entry in report.adaptivity_log] == ["Q01", "xt_sort_1", "Q06"]


def test_full_exam_covers_every_sector():
    report = _run(mode="full", diagnoser=_pass_diagnoser)
    S.ExamReport.model_validate(report.model_dump(by_alias=True))
    coding, trace = _kinds(report)
    assert (coding, trace) == EXAM_LENGTH["full"]
    sectors = [entry.sector for entry in report.adaptivity_log]
    assert all(left != right for left, right in zip(sectors, sectors[1:]))
    assert set(sectors) == set(SECTORS)
    coding_per = {}
    for entry in report.adaptivity_log:
        if entry.item_id.startswith("Q"):
            coding_per[entry.sector] = coding_per.get(entry.sector, 0) + 1
    assert all(count <= 2 for count in coding_per.values())
    assert _difficulty(report.adaptivity_log[0].item_id) == 1


def test_stable_pass_can_be_mastered():
    report = _run(
        adaptive=False,
        p_active={"M01": 0.05},
        states={"M01": "STABLE"},
        diagnoser=_pass_diagnoser,
    )
    held = next(finding for finding in report.findings if finding.cls == "M01")
    assert held.status == "HELD"
    assert held.state_after == "MASTERED"


def test_stable_fail_can_relapse():
    report = _run(
        adaptive=False,
        p_active={"M01": 0.05},
        states={"M01": "STABLE"},
        diagnoser=_fail_diagnoser,
    )
    relapsed = next(finding for finding in report.findings if finding.cls == "M01")
    assert relapsed.status == "RELAPSED"
    assert relapsed.state_after == "RELAPSED"
    new = next(finding for finding in report.findings if finding.cls == "D01")
    assert new.status == "NEW"
    assert new.state_after == "ACTIVE"
    assert new.evidence and new.evidence[0]["item_id"] == "Q01"


def test_public_exam_item_hides_the_answer_key():
    pool = {item["item_id"]: item for item in load_pool()}
    coding = public_exam_item(pool["Q01"])
    trace = public_exam_item(pool["xt_sort_1"])
    S.ExamItem.model_validate(coding)
    S.ExamItem.model_validate(trace)
    for public in (coding, trace):
        assert "exposure" not in public
        assert "correct" not in public
        assert "belief" not in public
    assert trace["options"] == pool["xt_sort_1"]["options"]
    assert coding["sample_tests"]
