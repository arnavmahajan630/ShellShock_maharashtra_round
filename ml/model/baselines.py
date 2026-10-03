"""Baselines for E8 (plans/03 §5.7), package M1.

1. majority class
2. rules: one predicate per class over the extracted features, tried in the fixed priority
   order of 03 §5.7; first match wins; nothing matches -> CORRECT if every test passed, else OTHER
3. TF-IDF (character 3–5-grams on normalised code) + logistic regression
4. TF-IDF + LightGBM

The zero-shot DeepSeek row is run by package E-a through ml/text/llm_client.py.
All baselines are scored on the same grouped folds as the diagnoser (`run_baselines`).
The TF-IDF baselines need the source text, so they are skipped for a bare feature matrix.
"""
from __future__ import annotations

import re

import lightgbm as lgb
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ml.contracts.classes import LABELS
from ml.contracts.feature_names import FEATURES
from ml.features.ast_feats import strip_source
from ml.model.train import macro_f1

RULE_ORDER = ["D08", "D05", "D06", "D07", "D03", "D02", "D04", "D01",
              "M07", "M06", "M05", "M02", "M10", "M04", "M03", "M08", "M01"]


# ---------------------------------------------------------------- 1. majority

def majority_predict(y_train, n):
    return np.full(n, np.bincount(y_train).argmax(), dtype=np.int64)


# ---------------------------------------------------------------- 2. rules

class _Row:
    """Feature access by name; a missing (NaN) value reads as 0 so no rule fires on it."""

    def __init__(self, x, index):
        self.x, self.index = x, index

    def __getitem__(self, name):
        value = self.x[self.index[name]]
        return 0.0 if np.isnan(value) else float(value)

    def missing(self, name):
        return bool(np.isnan(self.x[self.index[name]]))


RULES = {
    "D08": lambda f: f["a_str_literal_compare"] or f["a_array_name_compare"] or f["b_str_literal_compare"]
    or f["b_array_compare"],
    "D05": lambda f: f["a_has_recursion"] and (
        not f["a_base_case_present"]
        or (not f["a_base_case_static_reachable"] and not f["a_rec_arg_form_same"] and not f["a_rec_arg_form_plus1"])),
    "D06": lambda f: f["a_has_recursion"] and (f["a_rec_arg_form_same"] or f["a_rec_arg_form_plus1"]
                                               or f["b_rec_arg_constant"] or f["b_rec_arg_growing"]),
    "D07": lambda f: f["a_rec_call_discarded"] or f["b_discarded_call_value"],
    "D03": lambda f: f["a_swap_no_temp"] or f["b_multiset_changed"] or f["r_eq_shift_without_wrap"] > 0,
    "D02": lambda f: f["a_mid_assign_no_offset"],
    "D04": lambda f: f["a_outer_bound_const1"] or f["r_eq_one_pass"] > 0
    or (f["a_single_loop_with_swap"] and f["a_index_pair_plus1"]),
    "D01": lambda f: f["a_return_in_loop_else"] or f["a_assign_in_loop_else"],
    "M07": lambda f: f["a_empty_body_if"] or f["a_empty_body_for"] or f["a_empty_body_while"] or f["b_empty_body_exec"],
    "M06": lambda f: f["a_assign_in_cond"] or f["b_assign_in_cond_rt"],
    "M05": lambda f: f["a_decl_noinit_read_first"] or f["b_uninit_read"],
    "M02": lambda f: f["b_step_cap"] or (f["a_n_loops"] and (
        not f["a_update_present"] or not f["a_update_var_is_cond_var"] or f["a_update_in_branch"])),
    "M10": lambda f: f["a_printf_in_nonvoid"] and (f["a_nonvoid_missing_return"] or f["a_return_const_only"]
                                                   or f["b_printed_eq_ref_return"] > 0),
    "M04": lambda f: f["a_int_div_into_float"] or f["a_cast_wraps_division"] or f["r_eq_floor_ref"] > 0,
    "M03": lambda f: f["a_init_inside_loop"] or f["r_eq_ref_last_only"] >= 0.5,
    "M08": lambda f: f["b_oob_read_idx_eq_n"] or f["a_index_n"] or f["a_index_const1"] or f["a_array_loop_le_n"]
    or f["a_index_mirror_n_minus_i"] or (f["a_index_i_plus_1"] and not f["a_index_pair_plus1"])
    or (f["a_array_loop_start1"] and not f["a_index_const0"]),
    "M01": lambda f: f["b_iter_delta_const_pm1"] or f["a_bound_form_n_plus_1"]
    or (f["a_main_cond_op_le"] and f["a_bound_form_n"] and f["a_init_form_0"] and f["a_update_dir"] > 0),
}


def rules_predict_one(x, features=FEATURES, labels=LABELS, index=None):
    """Label index for one feature row."""
    row = _Row(np.asarray(x, dtype=np.float64), index or {name: i for i, name in enumerate(features)})
    for cls in RULE_ORDER:
        if RULES[cls](row):
            return labels.index(cls)
    passed_all = row.missing("b_pass_frac") or row["b_pass_frac"] >= 1.0
    return labels.index("CORRECT" if passed_all else "OTHER")


def rules_predict(X, features=FEATURES, labels=LABELS):
    index = {name: i for i, name in enumerate(features)}
    return np.array([rules_predict_one(x, features, labels, index) for x in np.atleast_2d(X)], dtype=np.int64)


# ---------------------------------------------------------------- 3 and 4. TF-IDF on the source text

def normalise_code(code):
    """Comments and preprocessor lines removed, whitespace collapsed (identifiers are kept)."""
    return re.sub(r"\s+", " ", strip_source(code)).strip()


def tfidf(train_codes, test_codes, max_features=20000):
    vectoriser = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), sublinear_tf=True, max_features=max_features)
    train = vectoriser.fit_transform([normalise_code(c) for c in train_codes])
    return train, vectoriser.transform([normalise_code(c) for c in test_codes])


def tfidf_lr_predict(train_codes, y_train, test_codes, seed=42):
    train, test = tfidf(train_codes, test_codes)
    model = LogisticRegression(max_iter=1000, C=10.0, class_weight="balanced", random_state=seed)
    return model.fit(train, y_train).predict(test).astype(np.int64)


def tfidf_lgbm_predict(train_codes, y_train, test_codes, seed=42, rounds=150):
    train, test = tfidf(train_codes, test_codes)
    classes = np.unique(y_train)
    remap = {int(c): k for k, c in enumerate(classes)}
    params = dict(objective="multiclass", num_class=len(classes), learning_rate=0.1, num_leaves=15,
                  min_data_in_leaf=5, feature_fraction=0.5, lambda_l2=1.0, verbose=-1, seed=seed,
                  deterministic=True, force_row_wise=True, num_threads=4)
    if len(classes) < 2:
        return np.full(test.shape[0], classes[0], dtype=np.int64)
    booster = lgb.train(params, lgb.Dataset(train.astype(np.float32), label=[remap[int(v)] for v in y_train]),
                        num_boost_round=rounds)
    return classes[np.asarray(booster.predict(test.astype(np.float32))).argmax(axis=1)].astype(np.int64)


# ---------------------------------------------------------------- same folds as the diagnoser

def run_baselines(data, folds):
    """Out-of-fold predictions and scores for every baseline that can run on `data`."""
    predictions = {"majority": np.zeros(len(data), dtype=np.int64),
                   "rules": rules_predict(data.X, data.features, data.labels)}
    if data.codes is not None:
        predictions["tfidf_lr"] = np.zeros(len(data), dtype=np.int64)
        predictions["tfidf_lgbm"] = np.zeros(len(data), dtype=np.int64)
    for train_idx, valid_idx in folds:
        predictions["majority"][valid_idx] = majority_predict(data.y[train_idx], len(valid_idx))
        if data.codes is not None:
            train_codes = [data.codes[i] for i in train_idx]
            valid_codes = [data.codes[i] for i in valid_idx]
            predictions["tfidf_lr"][valid_idx] = tfidf_lr_predict(train_codes, data.y[train_idx], valid_codes)
            predictions["tfidf_lgbm"][valid_idx] = tfidf_lgbm_predict(train_codes, data.y[train_idx], valid_codes)
    scores = {}
    for name, pred in predictions.items():
        per_fold = [macro_f1(data.Y[va], data.y[va], pred[va]) for _, va in folds]
        hit = data.Y[np.arange(len(pred)), pred] > 0
        scores[name] = {"macro_f1_mean": float(np.mean(per_fold)), "macro_f1_std": float(np.std(per_fold)),
                        "accuracy": float(hit.mean()), "n": int(len(pred))}
    skipped = [] if data.codes is not None else ["tfidf_lr", "tfidf_lgbm"]
    return {"scores": scores, "predictions": predictions, "skipped": skipped}
