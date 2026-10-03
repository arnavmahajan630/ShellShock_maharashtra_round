"""M1: data loading, folds, training, the artifact and reloading it.

Run from the repo root:  .venv\\Scripts\\python -m pytest tests/M1 -q
"""
import hashlib
import json

import numpy as np
import pytest

from ml.contracts.classes import LABELS
from ml.contracts.feature_names import FEATURES, GROUP_A, GROUP_B, GROUP_C, GROUP_R
from ml.model import data as D
from ml.model import train as T
from ml.model.calibrate import softmax
from ml.model.mask import mask_proba
from ml.model.predict import Diagnoser
from tests.F2.helpers import P03, P11, load
from tests.fixtures import build as B


# ---------------------------------------------------------------- data

def test_load_npz(synth):
    assert synth.X.shape == (1900, len(FEATURES)) and synth.X.dtype == np.float32
    assert synth.Y.shape == (1900, len(LABELS)) and np.allclose(synth.Y.sum(axis=1), 1)
    assert (synth.Y.argmax(axis=1) == synth.y).all() and len(np.unique(synth.groups)) == 28
    assert synth.domain is None and synth.codes is None and len(synth.data_hash()) == 16
    assert synth.subset([0, 5]).X.shape == (2, len(FEATURES))


def test_class_weights_and_soft_expansion():
    """Two plain M01 rows, one plain M08 row and one T1 row (0.5 / 0.5)."""
    Y = np.zeros((4, len(LABELS)))
    m01, m08 = LABELS.index("M01"), LABELS.index("M08")
    Y[0, m01] = Y[1, m01] = Y[2, m08] = 1.0
    Y[3, m01] = Y[3, m08] = 0.5
    weights = D.class_weights(Y)
    # mass: M01 2.5, M08 1.5, total 4, two classes present -> 4 / (2 * mass)
    assert weights[m01] == pytest.approx(0.8) and weights[m08] == pytest.approx(4 / 3)
    assert weights.sum() == pytest.approx(0.8 + 4 / 3)             # absent classes get 0
    X = np.arange(8, dtype=np.float32).reshape(4, 2)
    Xe, ye, we, source = D.expand_soft(X, Y)
    assert source.tolist() == [0, 1, 2, 3, 3] and ye.tolist() == [m01, m01, m08, m01, m08]
    assert we.tolist() == pytest.approx([0.8, 0.8, 4 / 3, 0.4, 2 / 3])  # soft_p x class weight
    assert (Xe[3] == Xe[4]).all()                                   # the soft row is duplicated
    for cls in (m01, m08):                                          # balanced: equal total weight per class
        assert we[ye == cls].sum() == pytest.approx(2.0)


def row(i, **over):
    base = {"id": f"A-{i:04d}", "source": "A", "problem_id": "P03", "family": "array_accumulate", "split": "train",
            "code": B.P03_LE, "label": "M01", "ast_hash": f"h{i}"}
    return dict(base, **over)


def test_training_rows_filter():
    rows = [row(0), row(1, split="holdout_problem"), row(2, verified={"tests_failed": 0, "tests_total": 5,
                                                                      "passes_by_luck": True}),
            row(3, source="U"), row(4, source="R-blind"), row(5, source="AMB"), row(6, source="E"), row(7, source="X")]
    assert [r["id"] for r in D.training_rows(rows)] == ["A-0000", "A-0005", "A-0006"]


def test_soft_label_vector():
    plain = D.soft_label_vector(row(0, label="M07"))
    assert plain[LABELS.index("M07")] == 1 and plain.sum() == 1
    twin = D.soft_label_vector(row(0, soft_label={"M01": 0.5, "M08": 0.5}))
    assert twin[LABELS.index("M01")] == 0.5 and twin[LABELS.index("M08")] == 0.5
    two_bug = D.soft_label_vector(row(0, label="M03", is_two_bug=True, labels_all=["M03", "M07"]))
    assert two_bug[LABELS.index("M03")] == 0.5 and two_bug[LABELS.index("M07")] == 0.5


def test_dropout_rows_are_seeded():
    a, b = D.dropout_rows(4000, seed=42), D.dropout_rows(4000, seed=42)
    assert (a == b).all() and 0.13 < a.mean() < 0.17
    assert (D.dropout_rows(4000, seed=7) != a).any() and not D.dropout_rows(100, rate=0.0).any()


def fake_backend():
    """Stands in for ml.runner: returns the fixture trace for the two P03 programs it knows."""
    traces = {B.P03_LE: load("traces/p03_le_oob_read.json"),
              P03["correct_variants"][0]: B.run_all(P03, lambda rec, c, n: B.p03_for(rec, c, n, rel_le=False))[0]}
    calls = {"trace": 0, "run": 0}

    def trace_fn(problem, code):
        calls["trace"] += 1
        return traces[code]

    def run_tests_fn(problem, code):
        calls["run"] += 1
        results = []
        for test in problem["tests"]:
            cells, n = test["args"]
            got = sum(cells[:n]) if code == P03["correct_variants"][0] else -1
            results.append({"args": test["args"], "expected": test["expect"], "got": {"returned": got},
                            "pass": got == test["expect"]["returned"]})
        return {"status": "ok", "backend": "interp",
                "tests": {"passed": sum(r["pass"] for r in results), "total": len(results), "results": results}}

    return trace_fn, run_tests_fn, calls


def group_cols(names):
    return [FEATURES.index(n) for n in names]


def test_build_matrix_with_a_fake_backend():
    trace_fn, run_tests_fn, calls = fake_backend()
    rows = [row(i) for i in range(40)]
    X = D.build_matrix(rows, {"P03": P03}, trace_fn=trace_fn, run_tests_fn=run_tests_fn, seed=42)
    drop = D.dropout_rows(40, seed=42)
    assert X.shape == (40, len(FEATURES)) and 0 < drop.sum() < 40
    assert np.isnan(X[drop][:, group_cols(GROUP_B + GROUP_R)]).all()            # dropout rows: no trace features
    assert not np.isnan(X[:, group_cols(GROUP_A + GROUP_C)]).any()
    kept = X[~drop]
    assert (kept[:, FEATURES.index("b_oob_read_idx_eq_n")] == 1).all()
    assert (kept[:, FEATURES.index("b_iter_delta_const_pm1")] == 1).all()       # the reference trace was used
    assert not np.isnan(kept[:, group_cols(GROUP_R)]).any()                     # the reference runner was used
    assert (X[:, FEATURES.index("a_main_cond_op_le")] == 1).all()
    assert calls["trace"] == (~drop).sum() + 1                                  # one reference trace per problem


def test_load_dataset_filters_builds_and_caches(tmp_path):
    trace_fn, run_tests_fn, calls = fake_backend()
    rows = [row(0), row(1, source="AMB", soft_label={"M01": 0.5, "M08": 0.5}, labels_all=["M01", "M08"]),
            row(2, split="holdout_problem"), row(3, source="U")]
    path = tmp_path / "dataset.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    data = D.load_dataset(path, problems={"P03": P03}, trace_fn=trace_fn, run_tests_fn=run_tests_fn, dropout=0.0)
    assert len(data) == 2 and data.ids == ["A-0000", "A-0001"] and data.groups.tolist() == ["P03", "P03"]
    assert data.domain.tolist() == ["main", "main"] and data.codes == [B.P03_LE, B.P03_LE]
    assert data.y.tolist() == [LABELS.index("M01")] * 2 and data.Y[1, LABELS.index("M08")] == 0.5
    assert len(list(tmp_path.glob("features_*.npz"))) == 1
    before = dict(calls)
    again = D.load_dataset(path, problems={"P03": P03}, trace_fn=trace_fn, run_tests_fn=run_tests_fn, dropout=0.0)
    assert calls == before and np.array_equal(again.X, data.X, equal_nan=True)   # second load reads the cache
    assert D.domain_of({"sector": "sorting"}) == "dsa" and D.domain_of(P11) == "main"


def test_load_dataset_without_a_backend_says_so(tmp_path):
    from ml import runner
    if runner.available("interp"):
        pytest.skip("interpreter is built")
    path = tmp_path / "dataset.jsonl"
    path.write_text(json.dumps(row(0)) + "\n", encoding="utf-8")
    with pytest.raises(runner.BackendUnavailable):
        D.load_dataset(path, problems={"P03": P03}, dropout=0.0)


# ---------------------------------------------------------------- folds

def test_folds_are_grouped_by_problem(synth):
    folds = T.make_folds(synth.groups)
    assert len(folds) == 5
    seen = np.zeros(len(synth), dtype=int)
    for train_idx, valid_idx in folds:
        assert not set(synth.groups[train_idx]) & set(synth.groups[valid_idx])
        seen[valid_idx] += 1
    assert (seen == 1).all()                            # every row is out-of-fold exactly once
    again = T.make_folds(synth.groups)
    assert all((a[1] == b[1]).all() for a, b in zip(folds, again))


def test_folds_are_stratified_by_domain():
    """13 main and 15 DSA problems (03 §5.3): every fold holds both domains."""
    groups = np.repeat(np.arange(28), 10)
    domain = np.where(groups < 13, "main", "dsa")
    for train_idx, valid_idx in T.make_folds(groups, domain):
        assert set(domain[valid_idx]) == {"main", "dsa"}
        assert 2 <= len(set(groups[valid_idx][domain[valid_idx] == "main"])) <= 3
        assert not set(groups[train_idx]) & set(groups[valid_idx])


def test_inner_split_is_grouped(synth):
    train_idx, _ = T.make_folds(synth.groups)[0]
    fit_idx, stop_idx = T.inner_split(train_idx, synth.groups)
    assert not set(synth.groups[fit_idx]) & set(synth.groups[stop_idx])
    assert sorted(np.concatenate([fit_idx, stop_idx]).tolist()) == sorted(train_idx.tolist())
    assert 0.15 < len(stop_idx) / len(train_idx) < 0.35


def test_grid_is_the_twelve_configs_of_the_plan():
    assert len(T.GRID) == 12
    assert {g["num_leaves"] for g in T.GRID} == {7, 15, 31}
    assert {g["min_data_in_leaf"] for g in T.GRID} == {10, 30} and {g["feature_fraction"] for g in T.GRID} == {0.6, 0.9}
    p = T.BASE_PARAMS
    assert p["num_threads"] == 4 and p["deterministic"] is True and p["objective"] == "multiclass"
    assert (p["learning_rate"], p["max_depth"], p["bagging_fraction"], p["bagging_freq"], p["lambda_l2"], p["seed"]) == \
        (0.05, 5, 0.8, 1, 1.0, 42)
    assert T.MAX_ROUNDS == 400 and T.EARLY_STOPPING == 30 and T.N_SPLITS == 5


def test_macro_f1_accepts_either_label_of_a_soft_row():
    Y = np.zeros((4, len(LABELS)))
    m01, m08, m07 = LABELS.index("M01"), LABELS.index("M08"), LABELS.index("M07")
    Y[0, m01] = Y[1, m07] = 1.0
    Y[2, m01] = Y[2, m08] = Y[3, m01] = Y[3, m08] = 0.5
    y = np.array([m01, m07, m01, m01])
    assert T.macro_f1(Y, y, np.array([m01, m07, m08, m01])) == 1.0          # M08 on a T1 row is right
    assert T.macro_f1(Y, y, np.array([m01, m07, m07, m01])) < 1.0


# ---------------------------------------------------------------- training and the artifact

def test_artifact_files_and_meta(trained, synth):
    summary, out = trained
    folder = out / summary["model_version"]
    assert sorted(p.name for p in folder.iterdir()) == ["meta.json", "model.txt"]
    text = (folder / "model.txt").read_text(encoding="utf-8")
    assert summary["model_version"] == "diagnoser_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["classes"] == LABELS and meta["features"] == FEATURES
    for key in ("temperature", "tau_p", "tau_d", "novelty", "data_hash", "git_commit", "params", "num_boost_round",
                "cv", "calibration", "model_version"):
        assert key in meta, key
    assert 0.5 <= meta["temperature"] <= 5.0 and 0.05 <= meta["tau_p"] <= 0.95 and meta["tau_d"] > 0
    assert meta["data_hash"] == synth.data_hash() and meta["n_rows"] == 1900 and meta["n_problems"] == 28
    assert meta["params"]["num_threads"] == 4 and meta["params"]["deterministic"] is True
    assert meta["params"]["num_class"] == 19
    reference = meta["novelty"]["reference"]
    assert reference["dtype"] == "float32" and reference["shape"][0] <= 4000
    assert len(meta["cv"]["grid"]) == 2 and meta["cv"]["n_splits"] == 5
    assert meta["chosen"] in ({"num_leaves": 7}, {"num_leaves": 15, "min_data_in_leaf": 30})


def test_grid_choice_is_best_macro_f1_then_log_loss(trained):
    grid = trained[0]["cv"]["grid"]
    best = min(grid, key=lambda r: (-round(r["macro_f1_mean"], 6), r["log_loss_mean"]))
    assert trained[0]["chosen"] == best["params"]


def test_out_of_fold_predictions_cover_every_row(trained, synth):
    summary = trained[0]
    oof = summary["oof_logits"]
    assert oof.shape == (1900, 19) and np.isfinite(oof).all() and (np.abs(oof).sum(axis=1) > 0).all()
    assert not np.isnan(summary["oof_distances"]).any()
    # the reported masked OOF score is exactly mask.py applied to the calibrated OOF probabilities
    probs = mask_proba(softmax(oof, summary["temperature"]), synth.X)
    pred = probs.argmax(axis=1)
    assert summary["cv"]["oof_masked"]["accuracy"] == pytest.approx((pred == synth.y).mean())
    assert summary["cv"]["oof_masked"]["macro_f1"] == pytest.approx(T.macro_f1(synth.Y, synth.y, pred))


def test_the_model_learned_the_telltale_columns(trained):
    """Not a quality claim: one column per class is informative, so anything far above chance
    (1/19 = 0.05) shows that training, folds and scoring are wired together correctly."""
    cv = trained[0]["cv"]
    assert cv["oof_unmasked"]["macro_f1"] > 0.35
    assert cv["oof_masked"]["macro_f1"] > 0.20          # masking columns are random in this matrix
    assert len(cv["folds"]) == 5 and all(f["n"] > 0 for f in cv["folds"])


def test_reloaded_artifact_gives_the_same_predictions(trained, synth):
    summary, _ = trained
    model = Diagnoser.load(summary["path"])
    assert model.model_version == summary["model_version"] and model.labels == LABELS
    reloaded = model.logits(synth.X)
    assert np.allclose(reloaded, summary["final_logits"], atol=1e-9)
    assert (reloaded.argmax(axis=1) == summary["final_logits"].argmax(axis=1)).all()
    twice = Diagnoser.load(summary["path"])
    assert np.array_equal(twice.proba(synth.X[:50]), model.proba(synth.X[:50]))
    assert np.array_equal(twice.knn_distance(synth.X[:50]), model.knn_distance(synth.X[:50]))


def test_latest_finds_the_artifact(trained, tmp_path):
    summary, out = trained
    assert Diagnoser.latest(out).model_version == summary["model_version"]
    with pytest.raises(FileNotFoundError):
        Diagnoser.latest(tmp_path)


def test_training_is_deterministic(synth, tmp_path):
    small = synth.subset(np.arange(0, 1900, 2))
    a = T.train(small, tmp_path / "a", grid=[{}], max_rounds=25, log=None)
    b = T.train(small, tmp_path / "b", grid=[{}], max_rounds=25, log=None)
    assert a["model_version"] == b["model_version"]
    assert a["temperature"] == b["temperature"] and a["tau_d"] == b["tau_d"] and a["tau_p"] == b["tau_p"]
    assert np.array_equal(a["oof_logits"], b["oof_logits"])


def test_soft_rows_train_as_weighted_duplicates(synth, tmp_path):
    """A matrix with soft labels goes through the same code (T1-style rows: 0.5 / 0.5)."""
    small = synth.subset(np.arange(0, 1900, 3))
    m01, m08 = LABELS.index("M01"), LABELS.index("M08")
    twins = np.where(np.isin(small.y, [m01, m08]))[0][::2]
    small.Y[twins] = 0.0
    small.Y[twins, m01] = small.Y[twins, m08] = 0.5
    summary = T.train(small, tmp_path, grid=[{}], max_rounds=25, log=None)
    assert np.isfinite(summary["oof_logits"]).all() and summary["cv"]["oof_unmasked"]["macro_f1"] > 0.2


def test_command_line(tmp_path, capsys):
    assert T.main(["--data", str(tmp_path / "missing.jsonl"), "--out", str(tmp_path)]) == 2
    assert "does not exist" in capsys.readouterr().err
    matrix = str(B.HERE / "features_synth.npz")
    assert T.main(["--data", matrix, "--out", str(tmp_path), "--no-grid", "--max-rounds", "15"]) == 0
    out = capsys.readouterr().out
    assert "made-up matrix" in out and "grouped-CV macro-F1" in out
    assert len(list(tmp_path.glob("diagnoser_*/model.txt"))) == 1
