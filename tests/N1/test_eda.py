"""N1: red-rule checks from 03 §10, plus the written report."""
import json
from pathlib import Path

import numpy as np

from ml.contracts.classes import LABELS
from ml.contracts.feature_names import GROUP_A
from ml.eda.checks import (
    check_counts,
    check_duplicates,
    check_events,
    check_exposure,
    check_length,
    check_masking,
    check_predicates,
    check_stumps,
    load_jsonl,
    operator_drop_status,
    pairwise_pearson,
)
from ml.eda.report import render

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ml" / "data" / "dataset.jsonl"
REPORT = ROOT / "docs" / "eda.md"


def _row(label, n, **extra):
    rows = []
    for index in range(n):
        item = {
            "label": label,
            "source": "A",
            "problem_id": "P01",
            "is_two_bug": False,
            "soft_label": None,
            "ast_hash": f"{label}-{index}",
            "aug": [],
            "code": "int f(){return 0;}",
        }
        item.update(extra)
        rows.append(item)
    return rows


def test_class_floor():
    rows = []
    for label in LABELS:
        rows.extend(_row(label, 80))
    passed = check_counts(rows)
    assert passed["status"] == "PASS"
    assert passed["min_count"] == 80
    short = [row for row in rows if row["label"] != "M06"]
    short.extend(_row("M06", 79))
    failed = check_counts(short)
    assert failed["status"] == "FAIL"
    assert failed["min_class"] == "M06"
    assert failed["min_count"] == 79


def test_hash_collision_allows_only_m01_m08():
    rows = [
        {"ast_hash": "h1", "label": "M01", "problem_id": "P03"},
        {"ast_hash": "h1", "label": "M08", "problem_id": "P03"},
        {"ast_hash": "h2", "label": "M04", "problem_id": "P13"},
        {"ast_hash": "h2", "label": "M10", "problem_id": "P13"},
    ]
    result = check_duplicates(rows)
    assert result["status"] == "FAIL"
    assert result["n_allowed_pairs"] == 1
    assert result["n_bad"] == 1
    assert result["bad"][0]["labels"] == ["M04", "M10"]


def test_drop_bounds():
    assert operator_drop_status(0, 0, 0)["verdict"] == "no_attempts"
    assert operator_drop_status(10, 8, 1)["verdict"] == "pass"  # upper = 0.20
    proven = operator_drop_status(10, 0, 5)  # lower = 0.50, upper = 1
    assert proven["verdict"] == "fail"
    assert operator_drop_status(10, 2, 1)["verdict"] == "inconclusive"  # lower 0.10, upper 0.80
    assert operator_drop_status(10, 8, 0)["verdict"] == "pass"
    edge = operator_drop_status(10, 0, 4)  # lower == 0.40, not greater
    assert edge["verdict"] == "inconclusive"
    assert operator_drop_status(5, 4, 3)["verdict"] == "inconsistent"


def test_length_separation():
    rows = _row("CORRECT", 30, code="x" * 20)
    rows.extend(_row("M01", 30, code="y" * 400))
    for label in LABELS:
        if label in ("CORRECT", "M01"):
            continue
        rows.extend(_row(label, 30, code="x" * 20))
    result = check_length(rows)
    flags = {item["label"]: item["separates"] for item in result["classes"]}
    assert flags["M01"] is True
    assert flags["M02"] is False
    assert result["status"] == "FAIL"


def test_stump_flags_non_defining_only():
    column = np.array([[0], [0], [0], [1], [1], [1]], dtype=float)
    y = np.array(["CORRECT", "CORRECT", "CORRECT", "OTHER", "OTHER", "OTHER"])
    tripped = check_stumps(["t_is_void"], column, y)
    assert tripped["status"] == "FAIL"
    assert tripped["rows"][0]["accuracy"] == 1
    spared = check_stumps(["a_assign_in_cond"], column, y)
    assert spared["status"] == "PASS"
    assert spared["rows"][0]["defining"] is True


def test_predicate_bleed():
    names = ["a_init_inside_loop", "a_assign_in_cond"]
    matrix = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.8, 1.0],
        [0.8, 1.0],
    ])
    y = np.array(["M03", "M03", "M06", "M06"])
    result = check_predicates(names, matrix, y)
    assert result["status"] == "FAIL"
    assert any(item["row"] == "M06" and item["col"] == "M03" and item["mean"] > 0.3 for item in result["offenders"])


def test_masking_zeroes_true_class():
    names = list(GROUP_A)
    matrix = np.zeros((2, len(names)), dtype=float)
    write = names.index("a_has_array_write")
    matrix[1, write] = 1.0
    y = ["D03", "D03"]
    ids = [
        {"id": "a", "label": "D03", "problem_id": "Q06"},
        {"id": "b", "label": "D03", "problem_id": "Q06"},
    ]
    result = check_masking(names, matrix, y, ids)
    assert result["status"] == "FAIL"
    assert result["n_masked"] == 1
    assert result["examples"][0]["id"] == "a"


def test_events_concentration():
    rows = []
    for _ in range(6):
        rows.append({"label": "M08", "source": "A", "trace_summary": {"events": ["oob_read"]}})
    for _ in range(6):
        rows.append({"label": "M05", "source": "A", "trace_summary": {"events": ["uninit_read"]}})
    assert check_events(rows)["status"] == "PASS"
    mixed = [{"label": "M03", "source": "A", "trace_summary": {"events": ["oob_read", "uninit_read"]}}] * 5
    mixed.append({"label": "M08", "source": "AMB", "trace_summary": {"events": ["oob_read"]}})
    mixed.append({"label": "M05", "source": "A", "trace_summary": {"events": ["uninit_read"]}})
    assert check_events(mixed)["status"] == "FAIL"


def test_exposure_gap_is_strict():
    rows = []
    for _ in range(10):
        rows.append({
            "problem_id": "Q01", "label": "M01", "is_two_bug": False, "aug": [],
            "verified": {"tests_failed": 2},
        })
    on_the_line = check_exposure(rows, [], {"Q01": {"M01": 0.7}})
    assert on_the_line["status"] == "PASS"
    assert abs(on_the_line["cells"][0]["gap"] - 0.3) < 1e-9
    over = check_exposure(rows, [], {"Q01": {"M01": 0.4}})
    assert over["status"] == "FAIL"
    assert over["cells"][0]["flag"] is True


def test_pairwise_pearson_matches_complete_rows():
    matrix = np.array([
        [1.0, np.nan],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
    ])
    corr = pairwise_pearson(matrix)
    assert abs(corr[0, 1] - 1.0) < 1e-6


def test_render_lists_statuses():
    text = render("scope text", [{
        "id": "10.1",
        "title": "Counts",
        "rule": "any class < 80 rows",
        "status": "PASS",
        "headline": "minimum is M06 = 116",
        "body": "body",
        "plots": [],
    }])
    assert "| 10.1 |" in text
    assert "PASS" in text
    assert "scope text" in text


def test_real_dataset_checks_are_consistent():
    rows = load_jsonl(DATA)
    assert len(rows) >= 4000
    counts = check_counts(rows)
    assert counts["n_rows"] == len(rows)
    assert counts["min_count"] == min(counts["per_class"].values())
    assert counts["status"] == ("FAIL" if counts["min_count"] < 80 else "PASS")
    duplicates = check_duplicates(rows)
    assert duplicates["n_rows"] == len(rows)
    assert duplicates["status"] == ("FAIL" if duplicates["n_bad"] else "PASS")
    assert duplicates["ratio"] == duplicates["n_unique_hash"] / len(rows)
    events = check_events(rows)
    for item in events["focus"]:
        if item["n"]:
            assert item["n_expected"] / item["n"] == item["share"]
    assert events["status"] in ("PASS", "FAIL")


def test_one_real_program_parses():
    from ml.features.ast_feats import ast_features

    with DATA.open(encoding="utf-8") as handle:
        row = json.loads(handle.readline())
    feats, meta = ast_features(row["code"])
    assert meta["parse_ok"] is True
    assert set(GROUP_A) <= set(feats)


def test_adversarial_auc_separates_and_ignores_a_copy():
    from ml.eda.checks import adversarial_auc, check_shift

    rng = np.random.default_rng(0)
    names = ["a_shift", "a_noise"]
    train = np.column_stack([np.zeros(40), rng.normal(size=40)])
    other = np.column_stack([np.ones(40), rng.normal(size=40)])
    groups = [f"g{i}" for i in range(40)]
    separated = adversarial_auc(train, groups, other, groups, names, num_boost_round=20)
    assert separated["auc"] > 0.85
    assert separated["status"] == "FAIL"
    assert separated["top"][0]["name"] == "a_shift"
    assert separated["lo"] is not None and separated["lo"] <= separated["auc"] <= separated["hi"]

    same = adversarial_auc(train, groups, train.copy(), [f"e{i}" for i in range(40)], names, num_boost_round=20)
    assert same["auc"] <= 0.85
    assert same["status"] == "PASS"

    # A shared problem id is one group. The splitter must accept that.
    shared = adversarial_auc(
        train,
        ["P01"] * 20 + ["P02"] * 20,
        other[:20],
        ["P01"] * 10 + ["P03"] * 10,
        names,
        num_boost_round=10,
    )
    assert shared["auc"] is not None

    both = check_shift(
        {**separated, "counts": {"n": 40}},
        {**same, "counts": {"n": 40}},
    )
    assert both["status"] == "FAIL"
    assert "LLM-written stand-in for R-team, not hand-written" in both["headline"]
    only_itsp = check_shift(
        {**same, "counts": {"n": 40}},
        {**separated, "counts": {"n": 40}},
    )
    assert only_itsp["status"] == "PASS"


def test_external_inputs_drop_labels_and_fill_group_c(tmp_path):
    from ml.eda.eda import feature_matrix, load_inputs

    path = tmp_path / "slice.jsonl"
    path.write_text(
        json.dumps({
            "code": "float mean(float a[], int n) { return a[0]; }",
            "label": "OTHER",
            "soft_label": {"OTHER": 1},
            "labels_all": ["OTHER"],
            "rater2_label": "OTHER",
            "problem_id": "ITSP-1",
        }) + "\n",
        encoding="utf-8",
    )
    rows = load_inputs(path)
    assert "label" not in rows[0]
    assert "rater2_label" not in rows[0]
    names, matrix, parse_fail = feature_matrix(rows, {}, signature_fallback=True)
    assert parse_fail == 0
    assert matrix[0, names.index("t_returns_float")] == 1.0
    _, bare, _ = feature_matrix(rows, {}, signature_fallback=False)
    assert np.isnan(bare[0, names.index("t_returns_float")])


def test_written_report_lists_every_rule():
    text = REPORT.read_text(encoding="utf-8")
    for rule_id in [f"10.{index}" for index in range(1, 14)]:
        assert f"| {rule_id} |" in text
    for name in ("class_problem.png", "predicate.png", "pca_class.png", "events.png"):
        assert (ROOT / "ml" / "eda" / "figures" / name).is_file()
