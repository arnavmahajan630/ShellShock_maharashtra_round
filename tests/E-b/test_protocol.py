"""Protocol checks for E-b. No retraining and no feature build."""
from __future__ import annotations

import numpy as np

from ml.bayes.likelihood import likelihood
from ml.contracts.classes import LABELS
from ml.eval._eb_data import Bundle
from ml.eval._eb_metrics import CARD_KEYS, cluster_mean_ci, depth1_auc, new_card
from ml.eval._eb_probe import answer_distribution
from ml.eval._eb_splits import choose_operator, plan_jobs
from ml.eval.e05_twins import _pair_hit
from ml.eval.e06_loco import _strict_columns


def _bundle():
    rows = [
        ("M01", "a_op", "H1", "train", 0),
        ("M01", "b_op", "H2", "train", 1),
        ("M08", "a_op", "H1", "train", 2),
        ("M02", "z_op", "H2", "train", 3),
        ("M01", "b_op+x_op", "H3", "train", 4),
        ("CORRECT", "c_op", "C1", "train", 5),
        ("CORRECT", "d_op", "C2", "train", 6),
        ("M01", "b_op", "H9", "holdout_problem", 7),
    ]
    n = len(rows)
    y = np.array([LABELS.index(row[0]) for row in rows], dtype=np.int64)
    return Bundle(
        kind=np.array(["bank"] * n),
        split=np.array([row[3] for row in rows]),
        y=y,
        label=np.array([row[0] for row in rows]),
        op_id=np.array([row[1] for row in rows]),
        ast_hash=np.array([row[2] for row in rows]),
        domain=np.array(["main"] * n),
        problem_id=np.array([f"P{i}" for i in range(n)]),
        id=np.array([f"r{i}" for i in range(n)]),
    )


def test_choose_operator_is_lexicographic_last_with_enough_rows():
    assert choose_operator({"a_op": 10, "m_op": 3, "c_op": 12}, min_rows=8) == "c_op"
    assert choose_operator({"only": 20}) is None


def test_holdout_drops_operator_and_shared_hash():
    jobs, skipped = plan_jobs(_bundle(), min_rows=1)
    job = next(item for item in jobs if item["class"] == "M01")
    assert job["op_id"] == "b_op"
    assert job["test_index"].tolist() == [1]
    train = set(job["train_index"].tolist())
    assert 1 not in train
    assert 3 not in train
    assert 4 not in train
    assert 0 in train
    assert 7 not in train
    assert job["dropped_for_hash"] == 1
    assert any(item["class"] == "M06" for item in skipped)


def test_engine_keeps_published_belief_probability():
    options = ["4", "5", "depends on values"]
    table = likelihood(options, "4", {"M08": "5", "M01": "4"})
    drawn = answer_distribution(options, "4", {"M08": "5"}, "M08", 0.9)
    assert abs(table["M08"]["5"] - 0.6) < 1e-9
    assert abs(drawn["5"] - 0.9) < 1e-9
    assert abs(sum(drawn.values()) - 1) < 1e-9


def test_pair_hit_tie_follows_label_order():
    posterior = {name: 0.0 for name in LABELS}
    posterior["M02"] = 0.2
    posterior["M07"] = 0.2
    assert _pair_hit(posterior, "M02", "M07") == 1.0
    assert _pair_hit(posterior, "M07", "M02") == 0.0
    posterior["M07"] = 0.3
    assert _pair_hit(posterior, "M02", "M07") == 0.0


def test_cluster_interval_contains_the_mean():
    interval = cluster_mean_ci([1, 0, 1, 1], ["P1", "P1", "P2", "P3"], n_boot=200, seed=1)
    assert interval["lo"] <= interval["estimate"] <= interval["hi"]
    assert interval["n"] == 4
    assert interval["n_clusters"] == 3


def test_depth1_stump_separates_a_pure_feature():
    X = np.array([[0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    auc, split = depth1_auc(X, y, groups, n_boot=40, seed=0)
    assert auc["estimate"] == 1.0
    assert split["feature_index"] == 0


def test_card_has_the_metrics_schema():
    card = new_card("E3", "title", "slice", 3, {"accuracy": 1}, "caveat")
    for key in CARD_KEYS:
        assert key in card


def test_strict_loco_blanks_mask_preconditions_it_would_otherwise_delete():
    removed, blanked = _strict_columns("D08")
    assert "a_str_literal_compare" in blanked
    assert "a_str_literal_compare" not in removed
    assert "b_str_literal_compare" in removed
    _, blanked_m01 = _strict_columns("M01")
    assert blanked_m01 == []
