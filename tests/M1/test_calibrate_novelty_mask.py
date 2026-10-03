"""M1: temperature scaling, novelty score and thresholds, structural masking."""
import json

import numpy as np
import pytest

from ml.contracts.classes import LABELS
from ml.contracts.feature_names import CLASS_DEFINING_FEATURES, FEATURES, GROUP_C, MASK_PRECONDITIONS
from ml.contracts.params import NOVEL_P_OTHER
from ml.features.extract import extract
from ml.model import calibrate as C
from ml.model import novelty as N
from ml.model.data import one_hot
from ml.model.mask import allowed_classes, apply_mask, mask_proba, masked_out
from tests.F1.snippets import SNIPPETS

K = len(LABELS)


def col(name):
    return FEATURES.index(name)


def lab(name):
    return LABELS.index(name)


# ---------------------------------------------------------------- calibration

def test_softmax_and_nll():
    p = C.softmax(np.array([[0.0, np.log(3.0)]]))
    assert p == pytest.approx(np.array([[0.25, 0.75]]))
    assert C.softmax(np.array([[1000.0, 0.0]]))[0, 0] == pytest.approx(1.0)        # no overflow
    assert C.softmax(np.array([[0.0, np.log(3.0)]]), T=2.0)[0, 1] == pytest.approx(np.sqrt(3) / (1 + np.sqrt(3)))
    Y = np.array([[0.5, 0.5]])
    assert C.nll(p, Y) == pytest.approx(-(0.5 * np.log(0.25) + 0.5 * np.log(0.75)))


def test_ece_and_reliability_by_hand():
    """Four predictions at 0.95 (three right) and four at 0.55 (two right)."""
    probs = np.array([[0.95, 0.05]] * 4 + [[0.55, 0.45]] * 4)
    correct = np.array([True, True, True, False, True, True, False, False])
    rows = C.reliability(probs, correct)
    assert len(rows) == 10 and rows[9]["count"] == 4 and rows[5]["count"] == 4
    assert rows[9]["accuracy"] == 0.75 and rows[9]["confidence"] == pytest.approx(0.95)
    assert rows[0]["count"] == 0 and rows[0]["accuracy"] is None
    assert C.ece(probs, correct) == pytest.approx(0.5 * abs(0.75 - 0.95) + 0.5 * abs(0.5 - 0.55))
    json.dumps(rows)


def sample_labels(probs, seed):
    rng = np.random.default_rng(seed)
    return np.array([rng.choice(probs.shape[1], p=row) for row in probs])


def test_temperature_recovers_a_known_scale():
    """Labels drawn from softmax(z); the model reports 2.5 z (overconfident). T must come back near 2.5."""
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1.5, size=(6000, 5))
    Y = one_hot(sample_labels(C.softmax(z), 1), 5)
    report = C.calibrate(2.5 * z, Y)
    assert report["temperature"] == pytest.approx(2.5, abs=0.15)
    assert report["ece_after"] < report["ece_before"] and report["ece_after"] < 0.03
    assert report["nll_after"] < report["nll_before"] and report["brier_after"] < report["brier_before"]
    assert report["n"] == 6000 and len(report["reliability_after"]) == 10


def test_temperature_stays_inside_its_bounds():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, size=(500, 4))
    Y = one_hot(z.argmax(axis=1), 4)                    # always right: T wants to go to 0
    assert C.fit_temperature(z, Y) == pytest.approx(0.5, abs=1e-3)
    Y = one_hot(rng.integers(0, 4, 500), 4)             # pure noise: T wants to go to infinity
    assert C.fit_temperature(10 * z, Y) == pytest.approx(5.0, abs=1e-3)


def test_soft_rows_count_as_correct_for_either_label():
    Y = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
    probs = np.array([[0.2, 0.7, 0.1], [0.6, 0.3, 0.1]])
    assert C.is_correct(probs, Y).tolist() == [True, False]


def test_calibration_on_the_trained_model(trained):
    report = trained[0]["calibration"]
    assert report["nll_after"] <= report["nll_before"] + 1e-9         # T minimises NLL on the OOF logits
    assert 0 <= report["ece_before"] <= 1 and 0 <= report["ece_after"] <= 1
    assert sum(r["count"] for r in report["reliability_after"]) == 1900


# ---------------------------------------------------------------- novelty

def toy_matrix(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, len(FEATURES)), dtype=np.float32)
    X[:, col("a_n_loops")] = rng.integers(0, 3, n)
    X[:, col("b_pass_frac")] = rng.random(n)
    X[:, col("r_eq_zero")] = rng.random(n)
    X[:, col("t_is_void")] = rng.integers(0, 2, n)      # group C: never part of the space
    X[::10, col("b_pass_frac")] = np.nan
    return X


def test_space_is_a_b_r_without_constant_columns():
    X = toy_matrix()
    space = N.Novelty.fit(X)
    assert [FEATURES[i] for i in space.columns] == ["a_n_loops", "b_pass_frac", "r_eq_zero"]
    assert not set(FEATURES[i] for i in space.columns) & set(GROUP_C)
    assert space.reference.dtype == np.float32 and space.reference.shape == (300, 3)
    filled = X[:, col("b_pass_frac")].astype(np.float64)
    assert space.median[1] == pytest.approx(np.nanmedian(filled))
    Z = space.transform(X)
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-5) and np.allclose(Z.std(axis=0), 1, atol=1e-5)
    blank = X[:1].copy()
    blank[0, col("b_pass_frac")] = np.nan               # NaN -> the column median
    assert space.transform(blank)[0, 1] == pytest.approx((np.nanmedian(filled) - space.mean[1]) / space.std[1], abs=1e-5)


def test_distance_is_the_mean_of_the_five_nearest():
    X = np.zeros((7, len(FEATURES)), dtype=np.float32)
    X[:, col("a_n_loops")] = [0, 1, 2, 3, 4, 5, 6]
    space = N.Novelty.fit(X)
    std = np.std([0, 1, 2, 3, 4, 5, 6])
    query = np.zeros((1, len(FEATURES)), dtype=np.float32)
    query[0, col("a_n_loops")] = 10
    # nearest five to 10 are 6, 5, 4, 3, 2 -> distances 4, 5, 6, 7, 8 in raw units
    assert space.distance(query)[0] == pytest.approx(np.mean([4, 5, 6, 7, 8]) / std, rel=1e-5)
    # a reference row scored against the reference: without self it is 1, 2, 3, 4, 5 away from row 0
    assert space.distance(X[:1])[0] == pytest.approx(np.mean([0, 1, 2, 3, 4]) / std, rel=1e-5)
    assert space.distance(X[:1], exclude_self=True)[0] == pytest.approx(np.mean([1, 2, 3, 4, 5]) / std, rel=1e-5)


def test_reference_is_capped_and_round_trips_through_meta():
    X = toy_matrix(n=600)
    space = N.Novelty.fit(X, max_rows=200, seed=1)
    assert space.reference.shape == (200, 3)
    again = N.Novelty.from_meta(json.loads(json.dumps(space.to_meta())))
    probe = toy_matrix(n=40, seed=9)
    assert np.array_equal(again.distance(probe), space.distance(probe))
    assert np.array_equal(N.Novelty.fit(X, max_rows=200, seed=1).reference, space.reference)   # seeded sample


def test_exclude_drops_a_class_signature_from_the_space():
    """E6 "LOCO strict": the defining features of the held-out class leave the kNN space."""
    X = toy_matrix()
    X[:, col("b_step_cap")] = np.arange(len(X)) % 2
    with_it = N.Novelty.fit(X)
    without = N.Novelty.fit(X, exclude=CLASS_DEFINING_FEATURES["M02"])
    assert col("b_step_cap") in with_it.columns and col("b_step_cap") not in without.columns
    assert len(without.columns) == len(with_it.columns) - 1


def test_far_rows_are_farther_than_the_threshold(trained, model, synth):
    summary = trained[0]
    assert summary["tau_d"] == pytest.approx(np.percentile(summary["oof_distances"], 99))
    far = synth.X[:5].copy()
    far[:, [col(n) for n in FEATURES if n.startswith("a_")]] = 40.0     # nothing like any training row
    assert (model.knn_distance(far) > model.tau_d).all()
    assert (model.knn_distance(synth.X[:200]) <= model.tau_d).mean() > 0.9


def test_oof_distances_use_only_the_other_folds():
    X = toy_matrix(n=120)
    folds = [(np.arange(60, 120), np.arange(0, 60)), (np.arange(0, 60), np.arange(60, 120))]
    out = N.oof_distances(X, folds)
    assert np.allclose(out[:60], N.Novelty.fit(X[60:]).distance(X[:60]))
    assert not np.isnan(out).any()


def test_tau_p_by_hand():
    """Ten predictions at p_max 0.9 (all right), ten at 0.6 (six right), ten at 0.7 (two right)."""
    def block(p, right, total=10):
        probs = np.tile([p, 1 - p], (total, 1))
        return probs, np.array([True] * right + [False] * (total - right))

    parts = [block(0.9, 10), block(0.6, 6), block(0.7, 2)]
    probs = np.vstack([p for p, _ in parts])
    correct = np.concatenate([c for _, c in parts])
    # all 30: 18/30 = 0.60; p_max >= 0.61: blocks 0.9 and 0.7 -> 12/20; p_max >= 0.71: 10/10
    assert N.choose_tau_p(probs, correct) == pytest.approx(0.71)
    assert N.choose_tau_p(probs, correct, target=0.6) == pytest.approx(0.05)
    assert N.choose_tau_p(probs, np.zeros(30, dtype=bool)) == pytest.approx(0.95)   # never reached


def vector(**given):
    rest = (1.0 - sum(given.values())) / (K - len(given))
    return np.array([given.get(name, rest) for name in LABELS])


def test_assess_rules():
    kw = dict(tau_d=3.0, tau_p=0.55)
    confident = N.assess(1.0, vector(M07=0.88, M06=0.05), **kw)
    assert confident == {"knn_dist": 1.0, "tau_d": 3.0, "p_max": 0.88, "tau_p": 0.55, "abstain": False}
    assert N.assess(3.5, vector(M07=0.88, M06=0.05), **kw)["abstain"] is True           # far from training data
    assert N.assess(1.0, vector(OTHER=NOVEL_P_OTHER, M07=0.3), **kw)["abstain"] is True  # p_other >= 0.5
    assert N.assess(1.0, vector(OTHER=0.49, M07=0.5), tau_d=3.0, tau_p=0.4)["abstain"] is False
    assert N.assess(1.0, vector(M07=0.40, M04=0.35), **kw)["abstain"] is True           # unsure, not a twin pair
    assert N.assess(1.0, vector(M01=0.46, M08=0.44), **kw)["abstain"] is False          # unsure, but T1 twins
    assert N.assess(1.0, vector(D05=0.40, D06=0.38), **kw)["abstain"] is False          # T7 twins


# ---------------------------------------------------------------- masking

def feature_row(**on):
    x = np.zeros(len(FEATURES))
    for name, value in on.items():
        x[col(name)] = value
    return x


def test_preconditions_table():
    assert masked_out(feature_row()) == ["D02", "D03", "D04", "D05", "D06", "D07", "D08"]
    assert masked_out(feature_row(a_has_recursion=1)) == ["D02", "D03", "D04", "D08"]
    assert masked_out(feature_row(a_mid_like_var=1)) == ["D03", "D04", "D05", "D06", "D07", "D08"]
    assert masked_out(feature_row(a_has_array_write=1)) == ["D02", "D05", "D06", "D07", "D08"]
    for needed in MASK_PRECONDITIONS["D08"]:            # any one of the three opens D08
        assert "D08" not in masked_out(feature_row(**{needed: 1}))
    everything = feature_row(a_has_recursion=1, a_mid_like_var=1, a_has_array_write=1, a_char_array_param=1)
    assert masked_out(everything) == []
    main_classes = allowed_classes(feature_row())[0][:9]
    assert main_classes.all() and allowed_classes(feature_row())[0][lab("CORRECT")]


def test_masking_zeroes_and_renormalises():
    probs = vector(D05=0.4, M02=0.3, D02=0.1)
    out = mask_proba(probs, feature_row())[0]
    assert out[lab("D05")] == 0 and out[lab("D02")] == 0 and out.sum() == pytest.approx(1.0)
    kept = 1.0 - 0.4 - 0.1 - 5 * (0.2 / 16)             # D05, D02 and the five other masked classes
    assert out[lab("M02")] == pytest.approx(0.3 / kept)
    same = mask_proba(probs, feature_row(a_has_recursion=1, a_mid_like_var=1, a_has_array_write=1,
                                         a_char_array_param=1))[0]
    assert same == pytest.approx(probs)


def test_unknown_preconditions_do_not_mask_and_empty_rows_survive():
    unknown = feature_row()
    unknown[[col(n) for n in FEATURES if n.startswith("a_")]] = np.nan      # code did not parse
    assert masked_out(unknown) == []
    only_masked = np.zeros(K)
    only_masked[lab("D05")] = 1.0
    out = apply_mask(only_masked, allowed_classes(feature_row()))[0]
    assert out[lab("D05")] == 1.0                       # all mass was on a masked class: left as it is


def test_masking_follows_the_code_not_the_problem():
    """Real feature rows: a plain loop cannot be a recursion or string class; a factorial can."""
    blank = {"signature": "", "tests": []}
    loop_row, _ = extract(blank, SNIPPETS["sum_for_lt"])
    assert masked_out(loop_row) == ["D02", "D03", "D04", "D05", "D06", "D07", "D08"]
    rec_row, _ = extract(blank, SNIPPETS["fact_no_base"])
    assert masked_out(rec_row) == ["D02", "D03", "D04", "D08"]
    sort_row, _ = extract(blank, SNIPPETS["bubble_no_temp"])
    assert masked_out(sort_row) == ["D02", "D05", "D06", "D07", "D08"]
    search_row, _ = extract(blank, SNIPPETS["bsearch_low_mid"])
    assert "D02" not in masked_out(search_row)
    text_row, _ = extract(blank, SNIPPETS["vowels_string_literal"])
    assert "D08" not in masked_out(text_row)


def test_model_probabilities_are_calibrated_then_masked(model, synth):
    X = synth.X[:100]
    masked, plain = model.proba(X), model.proba(X, masked=False)
    assert plain == pytest.approx(C.softmax(model.logits(X), model.temperature))
    assert masked == pytest.approx(mask_proba(plain, X))
    assert np.allclose(masked.sum(axis=1), 1.0) and (masked[~allowed_classes(X)] == 0).all()
    with pytest.raises(ValueError):
        model.proba(np.zeros((1, 5)))
