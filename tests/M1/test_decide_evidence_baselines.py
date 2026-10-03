"""M1: decision logic, evidence sentences, baselines, and the whole chain on the made-up matrix."""
import math

import numpy as np
import pytest

from ml.contracts import schemas as S
from ml.contracts.classes import LABELS, MISCONCEPTIONS
from ml.contracts.feature_names import FEATURES
from ml.contracts.params import CONFIDENT_MARGIN, CONFIDENT_P, EIG_MIN_BITS, MAX_PROBES, TWO_BUG_P2
from ml.features.extract import extract
from ml.model import baselines as BL
from ml.model import evidence as E
from ml.model.data import TrainData, one_hot
from ml.model.decide import decide, normalise
from ml.model.train import make_folds
from tests.F1.snippets import SNIPPETS
from tests.F2.helpers import P03, load
from tests.fixtures import build as B

PROBE = {"probe_id": "P_T1_a", "prompt": "Index of the last valid cell?", "code": "int a[5];",
         "options": ["4", "5", "depends on values"]}
NOT_NOVEL = {"knn_dist": 1.8, "tau_d": 3.2, "p_max": 0.46, "tau_p": 0.55, "abstain": False}


def post(**given):
    """A full posterior: the named values, the rest spread evenly."""
    rest = (1.0 - sum(given.values())) / (len(LABELS) - len(given))
    return {name: given.get(name, rest) for name in LABELS}


def check(diagnosis):
    S.Diagnosis.model_validate(diagnosis)
    return diagnosis


# ---------------------------------------------------------------- decide: one test per branch of 03 §5.5

def test_thresholds_come_from_params():
    assert (CONFIDENT_P, CONFIDENT_MARGIN, EIG_MIN_BITS, MAX_PROBES, TWO_BUG_P2) == (0.75, 0.25, 0.10, 2, 0.25)


def test_gate():
    d = check(decide({}, gate_code="G3b", model_version="diagnoser_x"))
    assert d["status"] == "gate" and d["posterior"] == {} and d["top"] == [] and d["model_version"] == "diagnoser_x"


def test_tests_pass_and_model_agrees():
    d = check(decide(post(CORRECT=0.97), tests_passed=True, novelty=NOT_NOVEL))
    assert d["status"] == "correct" and d["latent"] is None
    assert [t["id"] for t in d["top"]] == ["CORRECT"] and d["top"][0]["band"] == "Likely"


def test_tests_pass_but_a_misconception_is_named():
    """Passes by luck: shown as correct, with the latent class for the knowledge model."""
    d = check(decide(post(M03=0.62, CORRECT=0.30), tests_passed=True))
    assert d["status"] == "correct" and d["latent"] == {"class": "M03", "p": 0.62}
    weak = check(decide(post(M03=0.45, CORRECT=0.40), tests_passed=True))
    assert weak["status"] == "correct" and weak["latent"] is None          # below 0.5: not reported
    # passing tests win over novelty and probing
    d = decide(post(M01=0.46, M08=0.44), tests_passed=True, novelty=dict(NOT_NOVEL, abstain=True),
               best_probe=lambda p: (PROBE, 0.9))
    assert d["status"] == "correct" and d["next_probe"] is None


def test_novel():
    d = check(decide(post(M07=0.40, M04=0.35), novelty=dict(NOT_NOVEL, abstain=True),
                     best_probe=lambda p: (PROBE, 0.9), fix_check=lambda a, b: True))
    assert d["status"] == "novel" and d["next_probe"] is None and d["two_bug"] is False


def test_two_bug_needs_p2_and_the_fix_check():
    calls = []

    def fix_check(a, b):
        calls.append((a, b))
        return True

    d = check(decide(post(M03=0.50, M07=0.30), novelty=NOT_NOVEL, fix_check=fix_check))
    assert d["status"] == "two_bug" and d["two_bug"] is True and calls == [("M03", "M07")]
    assert decide(post(M03=0.50, M07=0.30), fix_check=lambda a, b: False)["status"] != "two_bug"
    assert decide(post(M03=0.70, M07=0.24), fix_check=fix_check)["status"] != "two_bug"        # p2 < 0.25
    assert decide(post(M03=0.50, M07=0.30))["status"] != "two_bug"                             # no fixer supplied
    calls.clear()
    assert decide(post(M03=0.50, CORRECT=0.30), fix_check=fix_check)["status"] != "two_bug"     # no fixer for CORRECT
    assert calls == []


def test_ambiguous_asks_the_best_probe():
    seen = {}

    def best_probe(posterior):
        seen.update(posterior)
        return PROBE, 0.71

    d = check(decide(post(M01=0.46, M08=0.44), novelty=NOT_NOVEL, best_probe=best_probe, model_version="diagnoser_x"))
    assert d["status"] == "ambiguous" and d["twin_set"] == "T1"
    assert d["next_probe"] == dict(PROBE, eig_bits=0.71)
    assert [t["id"] for t in d["top"]] == ["M01", "M08"] and [t["band"] for t in d["top"]] == ["Possible", "Unsure"]
    assert seen["M01"] == pytest.approx(0.46) and sum(seen.values()) == pytest.approx(1.0)
    assert d["posterior"]["M01"] == 0.46 and list(d["posterior"]) == LABELS


def test_no_useful_probe_means_confident_with_band_possible():
    for best_probe in (None, lambda p: None, lambda p: (PROBE, 0.09)):
        d = check(decide(post(M06=0.42, M07=0.40), novelty=NOT_NOVEL, best_probe=best_probe))
        assert d["status"] == "confident" and d["next_probe"] is None
        assert d["top"][0]["band"] == "Possible"        # 0.42 alone would be "Unsure"
        assert d["twin_set"] == "T3"
    exactly = decide(post(M06=0.42, M07=0.40), best_probe=lambda p: (PROBE, 0.10))
    assert exactly["status"] == "ambiguous"             # >= 0.10 bits


def test_probe_budget():
    asked = ["P_T1_a", "P_T1_b"]
    d = check(decide(post(M01=0.46, M08=0.44), best_probe=lambda p: (PROBE, 0.9), probes_asked=asked))
    assert d["status"] == "confident" and d["probes_asked"] == asked and d["next_probe"] is None
    one = decide(post(M01=0.46, M08=0.44), best_probe=lambda p: (PROBE, 0.9), probes_asked=asked[:1])
    assert one["status"] == "ambiguous"


def test_confident():
    d = check(decide(post(M07=0.88, M06=0.05), novelty=NOT_NOVEL, best_probe=lambda p: (PROBE, 0.9)))
    assert d["status"] == "confident" and d["top"][0] == {
        "id": "M07", "p": 0.88, "name": "Ghost Semicolon", "subtitle": "A stray `;` gives the if/loop an empty body",
        "band": "Likely"}
    assert d["twin_set"] is None and d["next_probe"] is None            # clear call: no twin talk
    # p1 < 0.75 but the margin is wide: still confident, the probe is not consulted
    wide = decide(post(M07=0.70, M06=0.10), best_probe=lambda p: (_ for _ in ()).throw(AssertionError))
    assert wide["status"] == "confident" and wide["top"][0]["band"] == "Possible"
    after_probe = check(decide(post(M08=0.90, M01=0.07), probes_asked=["P_T1_a"]))
    assert after_probe["status"] == "confident" and after_probe["twin_set"] == "T1"


def test_posterior_is_normalised_and_ordered():
    assert sum(normalise({"M01": 2.0, "M08": 2.0}).values()) == pytest.approx(1.0)
    d = decide({"M01": 3.0, "M08": 1.0})
    assert d["posterior"]["M01"] == 0.75 and d["posterior"]["D08"] == 0.0
    tie = decide(post(M08=0.3, M01=0.3))
    assert tie["top"][0]["id"] == "M01"                 # ties go to the contract's label order
    with pytest.raises(ValueError):
        decide({"M01": 0.0})


# ---------------------------------------------------------------- evidence

P03_REF_TRACE, _ = B.run_all(P03, lambda rec, c, n: B.p03_for(rec, c, n, rel_le=False))


def p03_twin():
    return extract(P03, B.P03_LE, load("traces/p03_le_oob_read.json"), P03_REF_TRACE,
                   lambda args: sum(args[0][:args[1]]))


def contributions_for(cls, **weights):
    contrib = np.zeros((len(LABELS), len(FEATURES) + 1))
    contrib[LABELS.index(cls), -1] = 5.0                # the bias column is never evidence
    for name, value in weights.items():
        contrib[LABELS.index(cls), FEATURES.index(name)] = value
    return contrib


def test_top_contributors_become_sentences_in_order():
    row, meta = p03_twin()
    contrib = contributions_for("M08", b_oob_read_idx_eq_n=3.0, a_main_cond_op_le=2.0,
                                t_has_array_param=1.5,          # no sentence for task meta: skipped
                                a_empty_body_for=1.2,           # value is 0 and there is no "zero" sentence: skipped
                                a_array_loop_le_n=1.0, a_index_i=0.5, a_init_form_0=-4.0)
    items = E.build_evidence(contrib, "M08", meta)
    for item in items:
        S.EvidenceItem.model_validate(item)
    assert items == [
        {"type": "RUN", "text": "Reads `cells[3]`, one cell past the end (line 4).", "line": 4,
         "feature": "b_oob_read_idx_eq_n", "weight": round(3.0 / 9.2, 3)},
        {"type": "CODE", "text": "Loop condition uses `<=` (line 3).", "line": 3,
         "feature": "a_main_cond_op_le", "weight": round(2.0 / 9.2, 3)},
        {"type": "CODE", "text": "The loop over `cells` runs while `i <= n`, so its last pass uses `cells[n]` (line 3).",
         "line": 3, "feature": "a_array_loop_le_n", "weight": round(1.0 / 9.2, 3)},
        {"type": "RUN", "text": "None of the 5 tests passed.", "feature": "b_pass_frac"},
    ]
    other_class = E.build_evidence(contrib, "M01", meta)                # nothing pushes toward M01 here
    assert [i["feature"] for i in other_class] == ["b_pass_frac"]


def test_hard_twin_fixer_and_bayes_items():
    row, meta = p03_twin()
    contrib = contributions_for("M01", a_main_cond_op_le=2.0, b_iter_delta_const_pm1=1.0)
    bayes = [{"type": "YOU PREDICTED", "text": "You predicted 3 cells would be read; 4 were."}]
    items = E.build_evidence(contrib, "M01", meta, status="ambiguous", twin_set="T1", extra_items=bayes,
                             fix_desc="`<=` to `<` on line 3", fix_line=3)
    for item in items:
        S.EvidenceItem.model_validate(item)
    assert [i.get("feature") for i in items] == ["a_main_cond_op_le", "b_iter_delta_const_pm1", "b_pass_frac",
                                                 "f_fix", None, None]
    assert items[1]["text"] == "Loop ran 4 times; the mission needed 3."
    assert items[3] == {"type": "CODE", "text": "Changing only `<=` to `<` on line 3 makes every test pass.",
                        "feature": "f_fix", "line": 3}
    assert items[4] == bayes[0]
    assert items[5] == {"type": "RUN", "text": "This code is identical for both explanations. Asking one question."}
    structural = E.build_evidence(contrib, "M01", meta, status="ambiguous", twin_set="T3")
    assert E.HARD_TWIN_TEXT not in [i["text"] for i in structural]      # only HARD twin sets get the sentence
    confident = E.build_evidence(contrib, "M01", meta, status="confident", twin_set="T1")
    assert E.HARD_TWIN_TEXT not in [i["text"] for i in confident]


def test_run_summary_is_not_repeated_and_needs_a_trace():
    row, meta = p03_twin()
    contrib = contributions_for("M08", b_pass_frac=2.0, a_main_cond_op_le=1.0)
    items = E.build_evidence(contrib, "M08", meta)
    assert [i["feature"] for i in items] == ["b_pass_frac", "a_main_cond_op_le"]       # no second summary
    no_trace_row, no_trace_meta = extract(P03, B.P03_LE)
    items = E.build_evidence(contributions_for("M08", a_main_cond_op_le=1.0, b_oob_read_idx_eq_n=3.0), "M08",
                             no_trace_meta)
    assert [i["feature"] for i in items] == ["a_main_cond_op_le"]       # nothing ran: no RUN items at all


def test_at_most_three_model_items():
    row, meta = p03_twin()
    contrib = contributions_for("M08", a_main_cond_op_le=5, a_bound_form_n=4, a_init_form_0=3, a_index_i=2,
                                a_array_loop_le_n=1)
    items = E.build_evidence(contrib, "M08", meta)
    assert [i["feature"] for i in items] == ["a_main_cond_op_le", "a_bound_form_n", "a_init_form_0", "b_pass_frac"]
    weights = [i["weight"] for i in items[:3]]
    assert weights == sorted(weights, reverse=True) and all(0 < w <= 1 for w in weights)
    assert len(E.build_evidence(contrib, "M08", meta, k=5)) == 6


def test_contributions_of_the_real_model_add_up_to_its_logit(model, synth):
    x = synth.X[7]
    contrib = model.contributions(x)
    assert contrib.shape == (len(LABELS), len(FEATURES) + 1)
    assert contrib.sum(axis=1) == pytest.approx(model.logits(x)[0], abs=1e-6)       # SHAP additivity
    ranked = E.top_contributors(E.class_contributions(contrib, "M01"))
    assert all(c > 0 for _, c, _ in ranked) and sum(s for _, _, s in ranked) == pytest.approx(1.0)
    assert [c for _, c, _ in ranked] == sorted((c for _, c, _ in ranked), reverse=True)


# ---------------------------------------------------------------- baselines

def test_majority():
    assert BL.majority_predict(np.array([3, 3, 5, 3, 1]), 4).tolist() == [3, 3, 3, 3]


BLANK = {"signature": "", "tests": []}
RULE_CASES = [
    ("vowels_string_literal", "D08"), ("array_name_compare", "D08"),
    ("fact_no_base", "D05"), ("fact_unreachable_base", "D05"),
    ("fact_same_arg", "D06"), ("fact_grow_arg", "D06"),
    ("fact_discarded", "D07"),
    ("bubble_no_temp", "D03"),
    ("bsearch_low_mid", "D02"),
    ("bubble_outer_once", "D04"), ("bubble_single_pass", "D04"),
    ("search_else_return", "D01"), ("search_flag_reset", "D01"),
    ("for_semicolon", "M07"), ("if_semicolon", "M07"), ("while_semicolon", "M07"),
    ("if_assign", "M06"),
    ("uninit_accumulator", "M05"),
    ("sum_while_no_update", "M02"), ("while_update_in_branch", "M02"), ("while_wrong_var", "M02"),
    ("printf_no_return", "M10"), ("printf_return_zero", "M10"),
    ("int_div_returned_as_float", "M04"), ("cast_after_division", "M04"),
    ("reset_decl_in_loop", "M03"), ("reset_assign_in_loop", "M03"),
    ("last_index_n", "M08"), ("sum_one_based_m08", "M08"), ("index_i_plus_1", "M08"), ("max_first_const1", "M08"),
    ("reverse_mirror_m08_full_bound", "M08"),
    ("sum_for_le_twin", "M08"),                     # the hard twin: M08 comes before M01 in the fixed order
    ("count_n_plus_1", "M01"),
    # correct programs, no trace: nothing fires
    ("sum_for_lt", "CORRECT"), ("sum_one_based_correct", "CORRECT"), ("sum_countdown_correct", "CORRECT"),
    ("debug_printf_correct", "CORRECT"), ("bubble_correct", "CORRECT"), ("selection_sort_correct", "CORRECT"),
    ("bsearch_correct_overflow_safe", "CORRECT"), ("fact_correct", "CORRECT"), ("palindrome_correct", "CORRECT"),
    ("search_flag_break_correct", "CORRECT"), ("two_pointer_correct", "CORRECT"), ("cast_numerator_correct", "CORRECT"),
    ("max_first_const0_correct", "CORRECT"), ("is_sorted_else_return", "D01"),
]


@pytest.mark.parametrize("snippet,expected", RULE_CASES)
def test_rules_on_real_feature_rows(snippet, expected):
    row, _ = extract(BLANK, SNIPPETS[snippet])
    assert LABELS[BL.rules_predict_one(row)] == expected


def test_rules_priority_and_fallbacks():
    assert BL.RULE_ORDER == ["D08", "D05", "D06", "D07", "D03", "D02", "D04", "D01",
                             "M07", "M06", "M05", "M02", "M10", "M04", "M03", "M08", "M01"]
    assert set(BL.RULE_ORDER) == set(MISCONCEPTIONS) and set(BL.RULES) == set(MISCONCEPTIONS)
    x = np.zeros(len(FEATURES))
    x[FEATURES.index("b_pass_frac")] = 1.0
    assert LABELS[BL.rules_predict_one(x)] == "CORRECT"
    x[FEATURES.index("b_pass_frac")] = 0.4
    assert LABELS[BL.rules_predict_one(x)] == "OTHER"                   # fails tests, no predicate
    x[FEATURES.index("a_empty_body_if")] = 1
    x[FEATURES.index("a_index_n")] = 1
    assert LABELS[BL.rules_predict_one(x)] == "M07"                     # M07 is tried before M08
    x[FEATURES.index("a_str_literal_compare")] = 1
    assert LABELS[BL.rules_predict_one(x)] == "D08"                     # D08 is tried first
    row, _ = extract(P03, B.P03_LE, load("traces/p03_le_oob_read.json"), P03_REF_TRACE,
                     lambda args: sum(args[0][:args[1]]))
    assert LABELS[BL.rules_predict_one(row)] == "M08"
    assert BL.rules_predict(np.vstack([row, x])).tolist() == [LABELS.index("M08"), LABELS.index("D08")]


def test_normalise_code():
    code = "int f(int n) {\n    // comment\n    return n;   /* x */\n}\n"
    assert BL.normalise_code(code) == "int f(int n) { return n; }"


def code_corpus():
    """A small labelled corpus from the F1 snippets, three 'problems' as groups."""
    picks = [("sum_for_lt", "CORRECT"), ("sum_countdown_correct", "CORRECT"), ("decl_then_assign_correct", "CORRECT"),
             ("for_semicolon", "M07"), ("if_semicolon", "M07"), ("while_semicolon", "M07"),
             ("if_assign", "M06"), ("while_assign", "M06"),
             ("uninit_accumulator", "M05"), ("uninit_max", "M05")]
    codes, y, groups = [], [], []
    for repeat in range(3):
        for name, label in picks:
            codes.append(SNIPPETS[name].replace("total", ["total", "sum", "acc"][repeat]))
            y.append(LABELS.index(label))
            groups.append(repeat)
    y = np.array(y)
    X = np.vstack([extract(BLANK, c)[0] for c in codes]).astype(np.float32)
    return TrainData(X=X, Y=one_hot(y, len(LABELS)), y=y, groups=np.array(groups), codes=codes)


def test_tfidf_baselines_run_on_source_text():
    data = code_corpus()
    train_idx, test_idx = np.where(data.groups < 2)[0], np.where(data.groups == 2)[0]
    train_codes, test_codes = [data.codes[i] for i in train_idx], [data.codes[i] for i in test_idx]
    lr = BL.tfidf_lr_predict(train_codes, data.y[train_idx], test_codes)
    gbm = BL.tfidf_lgbm_predict(train_codes, data.y[train_idx], test_codes)
    assert lr.shape == gbm.shape == (10,) and set(lr) <= set(data.y) and set(gbm) <= set(data.y)
    assert (lr == data.y[test_idx]).mean() >= 0.8       # the test codes are renamed copies of the training ones


def test_run_baselines_with_and_without_source_text(synth):
    data = code_corpus()
    folds = make_folds(data.groups, n_splits=3)
    report = BL.run_baselines(data, folds)
    assert set(report["scores"]) == {"majority", "rules", "tfidf_lr", "tfidf_lgbm"} and report["skipped"] == []
    assert report["scores"]["rules"]["accuracy"] == 1.0                 # these snippets are what the rules look for
    assert report["scores"]["majority"]["accuracy"] == pytest.approx(0.3)
    for score in report["scores"].values():
        assert 0 <= score["macro_f1_mean"] <= 1 and score["n"] == 30
    matrix_only = BL.run_baselines(synth, make_folds(synth.groups))
    assert set(matrix_only["scores"]) == {"majority", "rules"} and matrix_only["skipped"] == ["tfidf_lr", "tfidf_lgbm"]
    assert matrix_only["scores"]["majority"]["accuracy"] < 0.1          # 19 balanced classes


# ---------------------------------------------------------------- the whole chain

def test_chain_on_the_made_up_matrix(trained, model, synth):
    """train -> calibrate -> novelty thresholds -> mask -> decide -> evidence, on rows of the matrix.

    The posterior is the model's own masked, calibrated probability (the Bayes layer, package D1,
    sits between the two in the server). Evidence needs extractor metadata, which a made-up row
    does not have, so the metadata here says "no sentence values": every status must still come
    out as a valid diagnosis.
    """
    summary = trained[0]
    statuses = set()
    for i in range(0, 400, 7):
        x = synth.X[i]
        probs = model.proba(x)[0]
        assert probs.sum() == pytest.approx(1.0)
        novelty = model.novelty(x, probs)
        assert novelty["tau_d"] == round(summary["tau_d"], 4) and novelty["tau_p"] == round(summary["tau_p"], 4)
        diagnosis = decide(model.posterior_dict(probs), novelty=novelty, model_version=model.model_version,
                           best_probe=lambda p: (PROBE, 0.5), fix_check=lambda a, b: False)
        meta = {"features": {name: float(v) for name, v in zip(FEATURES, x)}, "lines": {}, "values": {},
                "has_trace": not math.isnan(float(x[FEATURES.index("b_pass_frac")]))}
        top1 = diagnosis["top"][0]["id"]
        diagnosis["evidence"] = E.build_evidence(model.contributions(x), top1, meta, status=diagnosis["status"],
                                                 twin_set=diagnosis["twin_set"])
        S.Diagnosis.model_validate(diagnosis)
        assert diagnosis["model_version"] == summary["model_version"]
        statuses.add(diagnosis["status"])
    assert statuses <= {"confident", "ambiguous", "novel"} and "novel" in statuses


def test_chain_with_real_extractor_metadata(model):
    """The same chain for a real program (the P03 hard twin fixture): sentences are filled.
    The model was trained on random columns, so which class it names is meaningless here."""
    row, meta = p03_twin()
    probs = model.proba(row)[0]
    novelty = model.novelty(row, probs)
    diagnosis = decide(model.posterior_dict(probs), novelty=dict(novelty, abstain=False),
                       model_version=model.model_version)
    top1 = diagnosis["top"][0]["id"]
    diagnosis["evidence"] = E.build_evidence(model.contributions(row), top1, meta, status=diagnosis["status"],
                                             twin_set=diagnosis["twin_set"])
    S.Diagnosis.model_validate(diagnosis)
    assert 1 <= len(diagnosis["evidence"]) <= 4
    assert all("{" not in item["text"] for item in diagnosis["evidence"])
    assert diagnosis["evidence"][-1] == {"type": "RUN", "text": "None of the 5 tests passed.", "feature": "b_pass_frac"} \
        or any(item["feature"] == "b_pass_frac" for item in diagnosis["evidence"])
    masked = {"D02", "D03", "D04", "D05", "D06", "D07", "D08"}         # a plain loop: none of these structures
    assert all(diagnosis["posterior"][name] == 0 for name in masked)
