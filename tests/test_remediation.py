"""Tests for the Misconception Clearance & Remediation Guide Engine."""
from fastapi.testclient import TestClient

from server.app.main import app
from server.app.remediation import (
    EXPERT_REMEDIATIONS,
    enrich_report_with_remediations,
    get_remediation_for_finding,
)

client = TestClient(app)


def test_expert_remediations_exist_for_all_canonical_classes():
    """Ensure all core DSA and C misconception classes have concrete remediations."""
    core_classes = ["M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08", "M09", "M10",
                    "D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08"]
    for cls in core_classes:
        assert cls in EXPERT_REMEDIATIONS, f"Missing remediation for {cls}"
        rem = EXPERT_REMEDIATIONS[cls]
        assert "root_cause" in rem and len(rem["root_cause"]) > 10
        assert "rule_to_remember" in rem and len(rem["rule_to_remember"]) > 10
        assert "code_fix" in rem
        assert "wrong" in rem["code_fix"] and "right" in rem["code_fix"]
        assert "self_check" in rem and len(rem["self_check"]) > 5


def test_get_remediation_for_finding():
    finding = {
        "class": "D03",
        "name": "Cargo Overwrite",
        "subtitle": "Swapping without a temp loses a value",
        "trial_title": "Bubble Sort",
        "evidence": [{"type": "CODE", "text": "a[j] = a[j+1];"}],
    }
    rem = get_remediation_for_finding(finding)
    assert rem["class"] == "D03"
    assert rem["name"] == "Cargo Overwrite"
    assert "buffer" in rem["rule_to_remember"].lower() or "temp" in rem["rule_to_remember"].lower()
    assert "temp" in rem["code_fix"]["right"]


def test_enrich_report_with_findings():
    report = {
        "exam_id": "test_exam",
        "items_total": 5,
        "items_passed": 4,
        "score_pct": 80,
        "sectors": [
            {"sector": "sorting", "name": "Sorting Algorithms", "items": ["T3"], "passed": 0, "rating_before": 1400, "rating_after": 1370}
        ],
        "findings": [
            {
                "class": "D03",
                "name": "Cargo Overwrite",
                "subtitle": "Swapping without a temp loses a value",
                "item_id": "T3_bubble_sort",
                "trial_title": "Bubble Sort",
                "evidence": [{"type": "CODE", "text": "a[j] = a[j+1];"}],
            }
        ],
        "recommendations": [],
    }
    enriched = enrich_report_with_remediations(report)
    assert "remediations" in enriched
    assert len(enriched["remediations"]) == 1
    assert enriched["remediations"][0]["class"] == "D03"


def test_enrich_report_with_zero_findings_provides_mastery():
    report = {
        "exam_id": "perfect_exam",
        "items_total": 5,
        "items_passed": 5,
        "score_pct": 100,
        "sectors": [],
        "findings": [],
        "recommendations": [],
    }
    enriched = enrich_report_with_remediations(report)
    assert "remediations" in enriched
    assert len(enriched["remediations"]) == 1
    assert enriched["remediations"][0]["class"] == "MASTERY"


def test_trials_finish_endpoint_includes_remediations():
    res = client.post("/trials/finish", json={"exam_id": "ex_unit_test"})
    assert res.status_code == 200
    data = res.json()
    assert "report" in data
    assert "remediations" in data["report"]
    assert len(data["report"]["remediations"]) > 0


def test_trials_remediation_route():
    res = client.post("/trials/remediation", json={"finding": {"class": "M01", "name": "Off by One"}})
    assert res.status_code == 200
    data = res.json()
    assert "remediation" in data
    assert data["remediation"]["class"] == "M01"
    assert "<=" in data["remediation"]["code_fix"]["wrong"]
    assert "<" in data["remediation"]["code_fix"]["right"]
