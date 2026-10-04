# E-b — Retrain experiments E3, E5, E6, E15

Ran on 2026-10-04 from the repo root with `.venv\Scripts\python -m ml.eval._eb_run`. Feature extraction 82s (6,811 rows, 0 failures). The four experiments then finished in one process, exit code 0, about 14 minutes. LightGBM was capped at 4 threads. Every retrain used the default config in `ml/model/train.py` (`grid=[{}]`), not the 12-config search. 06 §6 budgets about 7 minutes for all evaluation fits; a grid inside each retrain is about 12× that, and every comparison below uses the same config.

Models and cards are under `ml/eval/out/e-b/`. Nothing was written to `ml/artifacts/` or `ml/data/`.

## Built

- `ml/eval/e03_operator_holdout.py`, `e05_twins.py`, `e06_loco.py`, `e15_cross_domain.py`
- Helpers `ml/eval/_eb_data.py`, `_eb_metrics.py`, `_eb_probe.py`, `_eb_splits.py`, `_eb_train.py`, `_eb_run.py`
- `tests/E-b/test_protocol.py` (8 tests, no retraining)
- Cards and plots in `ml/eval/out/e-b/`:
  - `e03_card.json`, `e03_operator_holdout.png`
  - `e05_card.json`, `e05_twins.png`
  - `e06_card.json`, `e06_loco.png`, `e06_risk_coverage.png`
  - `e15_card.json`, `e15_cross_domain.png`
- Feature cache: `ml/eval/out/e-b/cache/features.npz` (full row and the no-trace row used for the 15% dropout)
- One diagnoser folder per retrain (`diagnoser_<sha8>/` with `model.txt` and `meta.json`)

`ml/eval/__init__.py` already existed and was not edited.

## Results

Intervals are 95% cluster bootstraps by `problem_id` (1,000 resamples), except E6's mean AUROC, which resamples the 17 class scores. A slice whose rows all sit on one problem, or a class where every row is right or every row is wrong, has `lo == hi == estimate`. That is the bootstrap, not a missing interval.

### E3 — unseen operator

Unweighted mean of per-class accuracies **0.668** (15 classes). Micro accuracy **0.649** [0.462, 0.858], n = 1,280. Accuracy counts a soft-label hit as correct; on these held-out operators it matches primary-label accuracy.

The held-out operator is the lexicographically last base operator (`op_id` with no `+`) that has at least 8 TRAIN rows. Two-bug rows that contain it are removed from training and are not in the test. Rows that share an `ast_hash` with the test are removed from training.

| class | held-out operator | n | accuracy |
|---|---|---:|---:|
| M01 | m01_while_le | 14 | 1.000 |
| M02 | m02_wrong_var | 49 | 1.000 |
| M03 | m03_decl_in_loop | 150 | 0.993 [0.980, 1.000] |
| M04 | m04_drop_cast | 51 | 1.000 |
| M05 | m05_drop_init_counter | 141 | 1.000 |
| M07 | m07_while_semi | 93 | 0.108 [0.000, 0.309] |
| M08 | m08_onebased | 76 | 1.000 |
| M10 | m10_printf_return0 | 124 | 0.992 [0.976, 1.000] |
| D02 | d02_low_mid | 102 | 1.000 |
| D03 | d03_rotate_no_save | 51 | 0.000 |
| D04 | d04_outer_once | 102 | 0.000 |
| D05 | d05_unreachable_base | 101 | 0.000 |
| D06 | d06_same_arg | 102 | 0.000 |
| CORRECT | nm_yoda | 24 | 1.000 |
| OTHER | oth_wrong_var | 100 | 0.920 [0.724, 1.000] |

M07 and D03–D06 do not survive a change of surface form. The other ten classes do.

### E5 — twins

Full TRAIN model `diagnoser_e2f2edbd`. Slice is HOLDOUT-P (938) plus R-llm (52 class-labelled rows). Six R-llm rows labelled GATE are excluded. No R-llm `ast_hash` appears in TRAIN. R-llm is the LLM-written stand-in from `ml/data/realistic_llm.jsonl`, not hand-written R-team and not R-blind.

Structural pair accuracy (unweighted mean of the seven twin sets): **0.891** [0.783, 0.997], n = 454 set-evaluations. Micro **0.879** [0.534, 0.996]. Main problems **1.000** (n = 229). DSA problems **0.807** [0.648, 0.970]. M-class rows **1.000**. D-class rows **0.616**. Per set: T2 1.00 (n=4), T3 1.00 (n=4), T4 1.00 (n=159), T6 0.75 (n=4), T7 0.989 (n=91), T8 0.988 (n=86), T9 0.509 (n=106). T9 (D01 vs M03) is the set that misses. T2, T3 and T6 have only four rows each.

Hard twins are the 44 soft-labelled T1 rows. Ambiguity-flag rate **1.000** (every row, 5 problems). Code-only accuracy **0.500** [0.448, 0.800]. After the engine's probes (20 draws per row; engine likelihood stays at p_b = 0.6):

| true p_b | post-probe accuracy |
|---:|---|
| 0.4 | 0.792 [0.590, 0.810] |
| 0.5 | 0.814 [0.759, 0.850] |
| 0.6 | 0.853 [0.659, 0.875] |
| 0.7 | 0.891 [0.850, 0.907] |
| 0.8 | 0.908 [0.890, 0.930] |
| 0.9 | 0.910 [0.800, 0.936] |

Hard-labelled M01/M08 rows (not the soft pair) are flagged ambiguous only **0.122** [0.000, 0.338] of the time (n = 98). That is the contrast, not the headline.

### E6 — leave-one-class-out

Strict mean AUROC **0.602** [0.541, 0.666]. Naive mean AUROC **0.733** [0.657, 0.808]. The gap is the one 03 §4.8 describes: naive still sees `CLASS_DEFINING_FEATURES[k]`. M07 is the extreme (strict 0.492, naive 0.993). D02 and D08 do not move (0.468 and 0.523 either way).

Strict, M-classes **0.622** [0.526, 0.708]; D-classes **0.579** [0.484, 0.675]. Strict on main-problem rows **0.711** [0.610, 0.812] (9 classes have both a positive and a negative there). Strict on DSA-problem rows **0.499** [0.412, 0.587] (16 classes). Mean abstain precision at the retrained model's own `tau_d` / `tau_p`: strict **0.354** [0.238, 0.479], naive **0.332** [0.217, 0.455]. Mean abstain rate on the held-out class: strict **0.324** [0.190, 0.475]. Card n = 4,785 is the class-k rows counted once each. The 0.70 AUROC target in 03 §0.3 is met by the naive mean and missed by the strict mean.

### E15 — cross-domain

Train on the 13 main problems only (`diagnoser_0a225f85`, 2,365 rows). M-class rows on the 15 DSA training problems (n = 1,198):

- macro-recall **0.797** [0.714, 0.865]
- macro-F1 **0.817** [0.729, 0.880]
- primary accuracy **0.733** [0.629, 0.814]

The full model, scored out of fold on those same rows (`diagnoser_e2f2edbd`): macro-recall **0.632** [0.549, 0.873], macro-F1 **0.645** [0.568, 0.876]. That control is harder in two ways that are both in the card: 15% of its training rows have no trace, and it has to separate D-classes as well. Grouped out-of-fold macro-F1 of the full model is **0.728** [0.693, 0.915] on main (n = 2,365) and **0.697** [0.596, 0.876] on DSA (n = 3,450).

HOLDOUT-P DSA M-class rows (n = 59, all on one problem, so the interval has no width): main-only macro-recall 0.720, full-model macro-recall 0.940.

Depth-1 stump on the five `t_*` features, grouped out-of-fold, D-class vs M-class: AUC **0.457** [0.311, 0.667], n = 4,206 (1,527 D, 2,679 M). The in-sample stump splits on `t_returns_float` at 0.5. This is under the 0.8 shortcut red flag in 03's risk table.

## Skipped

- E3 for M06, D01, D07, D08. Each has one base operator with kept TRAIN rows (`m06_if_assign`, `d01_else_return`, `d07_discard`, `d08_str_literal`). Compound `+` ids are two-bug rows, not a second operator. They are listed on the card and not scored as zero.
- E6 part (c), U1/U2. No unseen-class jsonl is in `ml/data`. Abstain rate on that slice was not filled in. The risk–coverage curve is the novelty score on class-k rows versus HOLDOUT-P, not on U.
- R-blind. It is not in the repo, and 03 §9.1 says it is never inspected. E5 uses R-llm and names it as such.
- The 12-config grid, inside every retrain. Same default config for all of them.
- Fixer k in E6, and the fixer in E5. The AUROC does not use it. `two_bug` cannot fire because `fix_check` is not passed.
- No new packages. Nothing was installed.

## Decisions

1. **Operator rule.** Lexicographically last base operator with at least 8 TRAIN rows. `splits.json` is in `ml/data/`, which this package does not write, so the rule lives in `ml/eval/_eb_splits.py` and is recorded on the E3 card.
2. **Dropout.** 15% of each training matrix is the no-trace extraction (`ml.model.data.FEATURE_DROPOUT`, seed 42), matching `build_matrix`. Test rows use the full trace.
3. **E5 pair accuracy** is the unweighted mean of per-set accuracies. Within a pair, the true class must have the higher probability after `prior()`. Micro accuracy is on the card beside it. T5 is omitted because M09 is not a model class.
4. **Probes.** The simulated learner draws with the true p_b, q = 0.5 and s = 0.1, before the probability floor. `update()` still uses `P_BELIEF` 0.6. A row is probed only when `decide()` returns `ambiguous`, including the novelty check. 20 repetitions, seed 42.
5. **E6 strict columns.** Defining features are removed. D08's `a_str_literal_compare` and `a_array_name_compare` are also mask preconditions; deleting them makes `allowed_classes` raise, so those two columns are set to NaN instead. The kNN fit then drops them (zero variance). The All-NaN median warning on that fit, and on the main-only model (DSA-only columns are empty), comes from that path. The novelty code replaces an all-NaN median with 0 and drops the column.
6. **Positives for E6** are every bank row whose primary label is k. A soft-label twin whose primary label is the other class is in neither the positives nor the negatives. Negatives are HOLDOUT-P rows with no soft mass on k.

## Commands

```
.venv\Scripts\python -m pytest tests/E-b -q
.venv\Scripts\python -m ml.eval._eb_run
.venv\Scripts\python -m ml.eval.e03_operator_holdout
.venv\Scripts\python -m ml.eval.e05_twins
.venv\Scripts\python -m ml.eval.e06_loco
.venv\Scripts\python -m ml.eval.e15_cross_domain
```

A finished retrain is reused when `summary.json` or `result.json` carries stamp `eb-1`. Bump `STAMP` in `ml/eval/_eb_data.py` to force a refit.
