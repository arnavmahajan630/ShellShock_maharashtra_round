# E-a — Diagnoser experiments E1, E2, E4, E7, E8, E9, E11, E12, E13

All nine cards are written to `ml/eval/cards/` (`E01.json`, `E02.json`, `E04.json`, `E07.json`, `E08.json`, `E09.json`, `E11.json`, `E12.json`, `E13.json`, plus PNGs and `E08_deepseek_answers.json`). Each card follows the 03 §9.4 shape (`id`, `title`, `slice`, `n`, `metrics`, `per_class`, `plots`, `caveat`) with extra keys.

## Read this first

- **R-blind (40, FE) does not exist.** Nothing was run on it. The 03 §0.3 headline (macro-F1 ≥ 0.60 on R-blind) is **not measured**. Every card that uses R says so in `data_status`.
- **R-team is replaced by R-llm: 58 LLM-written programs** (`ml/data/realistic_llm.jsonl`, source `R-llm`; 52 scored by the model + 6 junk rows for the gate). They are not hand-written, not blind, not real students; the second rater is an LLM too. Every R number below is on that stand-in.
- **The shipped temperature makes calibration worse (E7).** ECE goes 0.036 → 0.303 after temperature scaling; the ≤ 0.08 target is missed. This is M1's `ml/model/` (not edited). A temperature of 1.2 would pass (see E7).
- **DeepSeek zero-shot scores higher than the diagnoser on R-llm (0.917 vs 0.783). Do not read that as "DeepSeek is better".** DeepSeek wrote the programs, was told the label while writing them, and is also the second rater.
- **The model used is trained here, not loaded from `ml/artifacts/`** (none existed). `diagnoser_40e8e6ff` is in `ml/eval/_ea_cache/` and was trained with M1's `train()` and the 12-config grid on the 5,815 TRAIN rows of `dataset.jsonl` (28 problems). Nothing was written to `ml/artifacts/`. I1 should decide whether to retrain there; the numbers will move a little if the dataset changes.

## Results

| Exp | Slice (n) | Headline | Target / note |
|---|---|---|---|
| **E1** grouped CV | TRAIN, 28 problems, 5 folds (5,815 rows; 1,558 unique AST hashes) | macro-F1 **0.827 ± 0.094** (fold mean); pooled OOF 0.748 (cluster-bootstrap 95% CI 0.679–0.891); acc 0.831; D-classes on DSA problems 0.810 (CI 0.656–0.990, n = 1,527) | targets 0.80 / 0.75 met; no red flag (≥ 0.98). Weak problems: Q14 0.22 acc, Q12 0.56, Q07 0.60, Q11 0.63 |
| **E2** unseen problems | HOLDOUT-P: P04 P09 P13 Q04 Q09 Q16 (938) | macro-F1 **0.799** (cluster CI 0.699–0.989, 6 clusters); acc 0.880; main 1.000 (n = 536), DSA 0.642 (n = 402); D-classes on DSA **0.50** (CI 0.00–0.67, n = 174) | overall target 0.70 met; D-on-DSA target 0.65 **missed**. Main-domain 1.000 = the same-generator ceiling (no ast_hash overlap with TRAIN, checked) |
| **E4** realistic | R-llm, LLM-written stand-in (52 scored) | macro-F1 **0.783** (row-bootstrap CI 0.630–0.904); acc 0.827; main 0.684 (n = 19), DSA 0.779 (n = 33) | R-blind missing, so the 0.60 headline is unmeasured. Fixer finds a verified fix for the true class on only 18/35 = **51%** of misconception programs (same for top-3). Gate catches 5/6 junk rows, 0/52 normal programs wrongly gated (RL-055 slips through). Label κ vs LLM rater 0.906 (n = 58, agreement 0.914) |
| **E7** calibration | TRAIN out-of-fold | T = 2.447. ECE **0.036 → 0.303** (after-CI 0.219–0.384), NLL 2.632 → 1.889, Brier 0.270 → 0.347 | target ECE ≤ 0.08 **missed**. With T chosen to minimise ECE instead (informational, T = 1.2): OOF 0.014, HOLDOUT-P 0.072, R-llm 0.089 |
| **E8** baselines | macro-F1 on E1 / E2 / E4 | majority 0.026 / 0.035 / 0.017; rules 0.826 / 0.664 / 0.697; TF-IDF+LR 0.439 / 0.419 / 0.463; TF-IDF+LGBM 0.484 / 0.451 / 0.353; **diagnoser 0.827 / 0.799 / 0.783**; DeepSeek zero-shot – / – / **0.917** (CI 0.85–1.00) | DeepSeek ran once: 52 calls, 38,972 prompt + 1,864 completion tokens, model `deepseek-flash`, no failures. The rules baseline ties the diagnoser on E1 and is lower on E2 and E4 |
| **E9** ablation (E1 / E2 / E4) | | A 0.758 / 0.740 / 0.702; A+B 0.842 / 0.799 / 0.782; A+B+R 0.830 / 0.797 / 0.782; **A+B+R+C 0.827 / 0.799 / 0.783**; AST-only (trained without any trace) 0.738 / 0.727 / 0.689; shipped model fed AST-only input – / 0.686 / 0.694; **+ F (offline) 0.849 / 0.994 / 0.749** | B adds ~8 points over A; R and C add nothing measurable. F: E2 → 0.994 (CI 0.97–1.00) is the generator memorised (03 §4.4), and F lowers E4 (0.749). The shipped model is correctly the row without F |
| **E11** robustness | HOLDOUT-P (938) + R-llm (52) | top-1 unchanged **100%** under the five rewrites (3,919 HOLDOUT-P rewrites, 248 R-llm) and under two extra stress rewrites (1,318 / 60). Near-miss CORRECT false alarm: 0/170 on HOLDOUT-P; 7.5% (of 730) out-of-fold on TRAIN | target ≥ 0.90 met, but **mostly by construction**: feature vectors are bit-identical in 100% of HOLDOUT-P and 95.6% of R-llm cases (generator code is already canonical) |
| **E12** failure audit | wrong with conf ≥ 0.6 on E2 + E4 | 21 clusters (94 rows; copies of one mutant are shown once): 15 on HOLDOUT-P (88 rows), 6 on R-llm. 15 of the 21 are the Q04, Q16 and Q09 held-out forms (below); every cluster has a hand-written "why" | confidence ≥ 0.6 on the calibrated probability gives only 32 HOLDOUT-P rows and 0 R-llm rows (the temperature flattens it), so rows with T = 1 confidence ≥ 0.6 are included and marked |
| **E13** real students | ITSP slice, 396 programs (Strong) | **partial**, AST-only, labels unverified: 396/396 get AST features; agreement with the 19 rule-labelled rows: macro-F1 0.345, acc 0.368 (n = 19); with all 396 candidate labels: macro-F1 0.173 (cluster CI 0.045–0.321) | no trace features (programs use `main`/`scanf`/`malloc`). 38% of programs are predicted M10 (whole-program printing looks like M10) |

The main failure families in E12: (1) Q04 D01 written as a flag reset in the `else` branch (operator `d01_else_flag_reset`, present only on that held-out problem; every training D01 row is `else return`), predicted OTHER (6 clusters; a 7th is the same mutant plus a debug `printf`, predicted M10); (2) Q16 D06 `sum_digits(n % 10)`, where the argument is an "other" form that none of D06's own features cover, predicted OTHER (6 clusters); (3) Q09 M08 with a constant-index slip, predicted OTHER at low confidence (2 clusters); (4) a `printf` next to a `return` (debug print) pushing M10 with ~5 SHAP units (HOLDOUT-P A-002701 and R-llm RL-030). The remaining R-llm misses are listed with reasons in the card.

## What was built

`ml/eval/` (all mine, prefixed or numbered):

| File | Role |
|---|---|
| `_ea_common.py` | feature matrices (cached), trains the diagnoser once (`python -m ml.eval._ea_common`), scoring helpers (soft-label-aware macro-F1, bootstrap CIs, per-class tables, confusion plot), `write_card` |
| `_ea_perturb.py` | semantics-preserving rewrites: rename, reformat, comments, dead variable, for↔while, plus two stress rewrites (flip comparison, `i++` → `i += 1`) |
| `_ea_manual_why.json` | the 21 hand-written "why" sentences for E12 |
| `e01_grouped_cv.py` `e02_unseen_problems.py` `e04_realistic.py` `e07_calibration.py` `e08_baselines.py` `e09_ablation.py` `e11_robustness.py` `e12_failure_audit.py` `e13_real_students.py` | one experiment each, `python -m ml.eval.eNN_...` |
| `ml/eval/cards/` | the cards and PNGs |
| `ml/eval/_ea_cache/` | feature matrices, the trained model, OOF logits, baseline predictions, group-F values (4 MB; gitignore it or delete it, it regenerates) |
| `tests/E-a/test_ea_experiments.py` | 33 tests: helpers, every rewrite keeps the test results of P03, cards exist with the right shape, R cards say "LLM stand-in" and "R-blind missing", no key in any card or note |

`ml/eval/__init__.py` already existed (empty); not touched.

Order to rerun everything: `python -m ml.eval._ea_common` (≈ 3 min incl. training), then each `eNN` (E8 ≈ 6 min for the TF-IDF fits, E9 ≈ 10 min incl. group F; E8's DeepSeek answers are cached and cost nothing on rerun).

## Decisions where the plan is silent

1. **Folds, metric and CI:** E1 uses M1's own folds. "macro-F1" for E1 is the mean of per-fold macro-F1 (what the plan asks); the pooled OOF value and its cluster bootstrap are reported next to it, because the two differ (0.827 vs 0.748). M1's rule is kept: macro-F1 over classes present in the scored rows, and naming either label of a soft-labelled row counts as right. E2 and E1 use a cluster bootstrap by problem; E4 and E13 use a row bootstrap (1,000 resamples).
2. **Probabilities scored** are the shipped ones: temperature, then structural masking.
3. **R GATE rows** (6 junk programs) are not model inputs; they go through `server/app/gate.check` (read-only import) in E4.
4. **E4 fixer coverage** uses `ml/learner/fixer.probe` for the true class and for the model's top-3 misconception classes.
5. **E8 DeepSeek:** one call per program, temperature 0, hidden reasoning off, model from `.env` (`deepseek-flash`), through `ml/text/llm_client.py`; the prompt (task text + signature + code + the 19 labels with subtitle and belief) was written once and not tuned. Answers are in `ml/data/llm_raw/ea_e08_zero_shot/` (the client's cache) and in `E08_deepseek_answers.json`. The model sees no test results; the diagnoser does. The key was never printed or written (a test checks every card and this note).
6. **E9** retrains each row with M1's chosen hyper-parameters and no new grid; ablated columns are set to NaN (all mask preconditions are group A, so masking is identical in every row). AST-only is retrained on rows extracted with no trace at all. Group F was computed for all 17 classes for every TRAIN row (not just the top-3 serving would use): 5,815 rows in 220 s.
7. **E11** keeps a rewrite only if the problem's tests give the same pass/fail vector before and after (all did; the drop counts are in the card). "Near-miss CORRECT" = generator rows with `op_id` starting `nm_`.
8. **E12** clusters rows by (set, problem, labels, normalised AST hash) because the generator's style augmentations produce many copies of one mutant; `cluster_size` and `cluster_ids` are in the card.
9. **E13** needs a problem per program. ITSP has none, so the entry function is the function that contains the first differing line between the buggy and corrected files, and the corrected file is the reference. The ITSP code is not copied into the card.

## Skipped, faked or only partly done

- **R-blind: not run (does not exist).** Not faked, not substituted silently.
- **E13 is partial:** no full-feature run is possible (interpreter cannot run these programs); labels are the rule candidates of X1, not hand-checked; n for real classes is 1–7 per label. Treat as a feasibility check.
- **E12 "why" is a model-error explanation written after reading the code and the SHAP-style contributors, not a relabelling.** For the six R-llm rows it also says where the intended belief and the run disagree.
- **E12 confidence:** because of the E7 temperature, the plan's "confidence ≥ 0.6" selects too few rows; I added the T = 1 selection and marked it.
- **No per-class numbers for R-llm are meaningful** (1–2 items per class).
- `E5`, `E6`, `E3`, `E15` (E-b), `E10`, `E14` (V1), `E16–E18` (E-c) are not mine and were not touched.

## For I1 and other packages (nothing was edited outside my paths)

1. **M1 / calibration:** `ml/model/calibrate.py` fits T by NLL, which here is dominated by a few confident mistakes on whole unseen problems (worst 5% of rows = 52% of NLL), so T = 2.45 and the model is under-confident (mean confidence 0.40–0.67 for classes that are 80–100% right). `tau_p` came out at 0.29 for the same reason. T = 1.2 gives ECE 0.014 on OOF, 0.072 on HOLDOUT-P and 0.089 on R-llm (n = 52). Decide in I1 (pick T by ECE, or keep NLL and report it as a known miss). Everything downstream that thresholds on p (`decide`, novelty `tau_p`, Bayes) inherits this.
2. **Held-out operators:** the two biggest holdout failures are surface forms the generator makes only on a held-out problem. E3 (E-b) measures this directly; a second D01 operator on a training problem would likely fix the Q04 family.
3. **`a_printf_in_nonvoid` over-fires on debug prints:** the cause of M10 predictions on a function that does return (HOLDOUT-P A-002701, RL-030, and 38% of the ITSP slice). A rule "printf and no return of the printed value" would separate them.
4. **Fixer coverage on realistic code is 51%** (R1's 80% target was on held-out generator mutants).
5. **`.gitignore`** (root, not mine): add `ml/eval/_ea_cache/`.
6. `ml/data/llm_raw/ea_e08_zero_shot/` is new (52 files) inside T1's folder; it is the client's cache by design.

## Commands

```
.venv\Scripts\python -m ml.eval._ea_common
.venv\Scripts\python -m ml.eval.e01_grouped_cv      # e02_unseen_problems e04_realistic e07_calibration
.venv\Scripts\python -m ml.eval.e08_baselines [--no-llm]
.venv\Scripts\python -m ml.eval.e09_ablation [--f-budget 1200]
.venv\Scripts\python -m ml.eval.e11_robustness      # e12_failure_audit e13_real_students
.venv\Scripts\python -m pytest tests/E-a -q         # 33 passed
```
