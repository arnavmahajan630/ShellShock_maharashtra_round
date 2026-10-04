# E-c — Sentence reader and item-bank experiments

Built on 2026-10-04. **E16, E17 and E18 each write their card.** E16 was written first (an earlier session); E17 and E18 were added once the quiz bank (B4), the Bayes layer (D1) and the new `ml/text/serve.py` (T2) were in. The code-item half of E18 was filled in once C4 had built `build_code_items.py` and `code_items.json` (see "E18" below). The quiz-bank numbers were left as they were.

## Built

| File | What it is |
|---|---|
| `ml/eval/e16_text.py` | E16: `python -m ml.eval.e16_text`; `build_card(...)` and `run()` return the card as a dict |
| `ml/eval/e16_card.json`, `e16_card.md` | the E16 card (JSON, and the same as tables) |
| `ml/eval/e17_why.py` | E17: `python -m ml.eval.e17_why` |
| `ml/eval/e17_card.json`, `e17_card.md` | the E17 card |
| `ml/eval/e18_bank.py` | E18: `python -m ml.eval.e18_bank` (`--backend interp` to skip gcc) |
| `ml/eval/e18_card.json`, `e18_card.md` | the E18 card |
| `ml/eval/_ec_common.py` | helpers for E17/E18: 4-thread cap for onnxruntime, bootstraps, card writer. (E16 does not use it.) |
| `tests/E-c/test_e16_text.py`, `test_e17_why.py`, `test_e18_bank.py` | 31 tests in all |

`DeepSeek` was not called by any of the three.

**Test folder:** `tests/E-c/`, with no `__init__.py`. "E-c" is not a valid Python package name, so pytest imports the test file by its own name; that works as long as no other folder without an `__init__.py` has a `test_e16_text.py`.

**Card layout (03 leaves it open):** one JSON file per experiment next to its script, `ml/eval/e16_card.json`. The top level has the fields of 03 §9.4 (`id`, `title`, `slice`, `n`, `metrics`, `per_class`, `plots`, `caveat`); these describe the default reader (`tfidf`) on the held-out contexts. The E16 tables are extra keys: `readers`, `rows`, `per_voice`, `language`, `mohler`, `leave_one_class_out`, `with_code_masking`, `per_class_by_reader`, `not_run`, `columns`. For I1: append the JSON to `cards` in `metrics.json`, or call `ml.eval.e16_text.run()`. A run takes 5–9 minutes on the laptop CPU, almost all of it the two 436 MB models reading the 2,442 Mohler answers and the 3,307 sentences.

## What the sets are

- **test** = the test split of `ml/data/reasons.jsonl`: 539 sentences from contexts never seen in training. LLM-written, same generator as the training sentences.
- **persona** = `ml/data/reasons_persona.jsonl`, 270 sentences: the **LLM-written stand-in for classmates' sentences**. A different model and prompt. It is not text written by students.
- **Mohler** = 2,442 short answers by real CS students to 87 questions of an introductory computer-science course (C++ and data structures) (`ml/external/mohler.py`). None shows one of our 17 mistakes, so the right answer for every one of them is `unsure`.

**Caveat printed on the card:** the training sentences are LLM-written, and so are both test sets. Only Mohler is real student text, and it has no labels for our classes. No number here measures accuracy on real students' sentences.

Nothing was tuned on the test split or the persona set. Thresholds come from the val split (word-count 0.584, frozen 0.585) and from the artifact's `meta.json` (fine-tuned 0.871).

## Results

Accuracy, macro-F1 and top-3 ignore the threshold. "Answers" is the share at or above the threshold. Intervals are 95%, from 1,000 resamples of whole contexts.

| Reader | Set | n | Accuracy | 95% interval | Macro-F1 | Top-3 | Answers | Right when it answers |
|---|---|---|---|---|---|---|---|---|
| tfidf | test | 539 | 0.720 | 0.63 to 0.82 | 0.714 | 0.876 | 54% | 89% |
| tfidf | persona | 270 | 0.730 | 0.67 to 0.79 | 0.762 | 0.900 | 61% | 83% |
| tfidf | persona, picked by itself | 163 | 0.681 | 0.58 to 0.76 | 0.695 | 0.896 | 58% | 77% |
| tfidf | persona, told which option | 107 | 0.804 | 0.71 to 0.89 | 0.829 | 0.907 | 64% | 91% |
| biencoder | test | 539 | 0.718 | 0.62 to 0.82 | 0.702 | 0.891 | 54% | 94% |
| biencoder | persona | 270 | 0.707 | 0.67 to 0.75 | 0.762 | 0.919 | 64% | 81% |
| biencoder | persona, picked by itself | 163 | 0.638 | 0.57 to 0.70 | 0.688 | 0.902 | 64% | 73% |
| biencoder | persona, told which option | 107 | 0.813 | 0.73 to 0.89 | 0.803 | 0.944 | 65% | 93% |
| frozen | test | 539 | 0.603 | 0.47 to 0.74 | 0.596 | 0.852 | 47% | 84% |
| frozen | persona | 270 | 0.667 | 0.60 to 0.73 | 0.697 | 0.893 | 53% | 83% |
| frozen | persona, picked by itself | 163 | 0.687 | 0.61 to 0.76 | 0.701 | 0.902 | 53% | 81% |
| frozen | persona, told which option | 107 | 0.636 | 0.51 to 0.76 | 0.655 | 0.879 | 53% | 84% |

What this says:
- Word-count and fine-tuned are level on accuracy; frozen is behind on the held-out contexts. The intervals are wide (the test split has one context per mistake type).
- **No reader keeps the 90% promise outside validation.** The thresholds were set for 90% on val. On the held-out contexts: word-count 89%, fine-tuned 94%, frozen 84%. On the stand-in set: 83%, 81%, 83%; and on its "picked by itself" rows 77%, 73%, 81%. Expect lower again on real students.
- The word-count and fine-tuned numbers are the same as the table in `notes/T1.md`. The frozen reader is 0.603 here against 0.605 on Colab (one sentence; ONNX file from the model's repo here, PyTorch there).

### Hinglish against English

| Reader | Set | Language | n | Accuracy | Top-3 | Answers | Right when it answers |
|---|---|---|---|---|---|---|---|
| tfidf | test | English | 454 | 0.725 | 0.881 | 56% | 88% |
| tfidf | test | Hinglish | 85 | 0.694 | 0.847 | 41% | 97% |
| tfidf | persona | English | 225 | 0.711 | 0.893 | 64% | 81% |
| tfidf | persona | Hinglish | 45 | 0.822 | 0.933 | 47% | 95% |
| biencoder | test | English | 454 | 0.718 | 0.892 | 56% | 94% |
| biencoder | test | Hinglish | 85 | 0.718 | 0.882 | 47% | 88% |
| biencoder | persona | English | 225 | 0.698 | 0.916 | 66% | 80% |
| biencoder | persona | Hinglish | 45 | 0.756 | 0.933 | 56% | 88% |
| frozen | test | English | 454 | 0.617 | 0.861 | 50% | 84% |
| frozen | test | Hinglish | 85 | 0.529 | 0.800 | 32% | 89% |
| frozen | persona | English | 225 | 0.662 | 0.893 | 57% | 83% |
| frozen | persona | Hinglish | 45 | 0.689 | 0.889 | 31% | 79% |

Hinglish is the `hinglish` voice in the test split and the `friend` character in the stand-in set. Every reader answers Hinglish less often. The unchanged English model (frozen) loses the most accuracy on it. The full per-voice table (6 voices per set) is in `e16_card.md`; groups there have 44–95 sentences, so single rows move by several points on a handful of sentences.

### Mohler: should say unsure

| Reader | n | Unsure | Wrongly `matched` | `correct_reasoning` | Most often matched to |
|---|---|---|---|---|---|
| tfidf | 2,442 | 89% | 10% | 1% | D04 2.9%, M08 2.7%, D06 2.1% |
| biencoder | 2,442 | 87% | 10% | 3% | D04 5.5%, D06 1.8%, M08 1.7% |
| frozen | 2,442 | 83% | 12% | 5% | D04 6.5%, D06 1.6%, M08 1.4% |

One real student answer in ten that is about something else is read as one of our mistakes. The answers are about sorting, recursion and arrays among other things, so they share words with D04, D06 and M08. Each whole answer (median 15 words) was read as one sentence.

### Leave-one-class-out

Fine-tuned column: **run on Colab on 2026-10-04, copied from notes/T1.md** (17 retrains; each time the reader is trained without any sentence of the class, then asked to find them from the description). Unchanged-model column: computed here. `BAAI/bge-base-en-v1.5` with no training at all picks the nearest description (texts from the artifact's `descriptions.json`), on the same sentences: every sentence of the class, all splits.

| Class | Fine-tuned, class held out: top-1 | Unchanged model: top-1 | Unchanged model: top-3 | n |
|---|---|---|---|---|
| M01 | 0.000 | 0.594 | 0.928 | 180 |
| M02 | 0.067 | 0.184 | 0.520 | 179 |
| M03 | 0.106 | 0.550 | 0.933 | 180 |
| M04 | 0.217 | 0.811 | 0.961 | 180 |
| M05 | 0.155 | 0.540 | 0.862 | 174 |
| M06 | 0.011 | 0.256 | 0.594 | 180 |
| M07 | 0.000 | 0.144 | 0.344 | 180 |
| M08 | 0.044 | 0.700 | 0.878 | 180 |
| M10 | 0.272 | 0.550 | 0.794 | 180 |
| D01 | 0.144 | 0.322 | 0.433 | 180 |
| D02 | 0.311 | 0.517 | 0.672 | 180 |
| D03 | 0.250 | 0.633 | 0.817 | 180 |
| D04 | 0.267 | 0.456 | 0.717 | 180 |
| D05 | 0.011 | 0.190 | 0.430 | 179 |
| D06 | 0.011 | 0.383 | 0.656 | 180 |
| D07 | 0.139 | 0.589 | 0.883 | 180 |
| D08 | 0.028 | 0.583 | 0.844 | 180 |
| **mean** | **0.120** | **0.471** | **0.722** | |

Fine-tuned mean top-3 is 0.283 (the notes give top-3 per class for three classes only: D02 0.600, M01 0.017, M07 0.067).

**For a mistake with a description and no training sentences, the unchanged model is the better reader in every one of the 17 classes** (mean 0.47 against 0.12). Fine-tuning on the other classes pulls sentences toward the descriptions it trained on. If a new mistake has to be added without sentences, the unchanged model's nearest description is the option to use, and it still gets under half right.

On the test split alone the unchanged model's nearest description scores 0.490 top-1 and 0.731 top-3 (n 539), the same as the Colab run.

### Extra: with the code passed for masking

With each row's code passed to `read()`, 120 of 539 test rows and 64 of 270 stand-in rows have code that is used for masking (whole and self-contained; `notes/T2.md` decision 4). Accuracy does not change for any reader; "answers" rises by at most one point. Before the self-contained rule was added, one test context lost its class and the word-count reader fell from 0.720 to 0.664: that is how the rule was found. The rule was not chosen by score: it is "do not mask code that calls a helper it does not show".

### Not run

- **DeepSeek zero-shot row** (optional in 05 §9): not run. No LLM API call was made.

## Decisions

1. Macro-F1 is over the labels present in the slice, as in `check_readers.scores` (imported, not copied), so the numbers agree with T1.
2. Intervals resample contexts, not sentences: the 30 sentences of one context are about the same question.
3. Stand-in voices are reported per character with the `_told` rows merged in (the 12 raw voices have 14–30 sentences each).
4. On Mohler, `correct_reasoning` is counted apart from `matched`: it names no mistake.
5. The copied leave-one-class-out numbers are constants in `e16_text.py`; a test compares them with the line in `notes/T1.md`.
6. The Mohler loader looks in this checkout's `ml/data/external/mohler` first, then where `ml/external/common.py` keeps downloads. Without the files the section says "not run".
7. A reader that does not load is listed as not loaded and has no rows. Without the unchanged model the leave-one-class-out table has only the copied column.

## Not verified

- Every accuracy number is on LLM-written sentences.
- The leave-one-class-out retrains were not rerun here; their numbers are copied.
- The Mohler answers are C++ and data-structures answers, not answers to our questions. They show how a reader behaves on off-topic student text, not on an off-topic answer to one of our items.
- Per-class precision and recall are in the card (`per_class_by_reader`) but were not examined.

## E17 — does asking "why?" add anything? (`ml/eval/e17_why.py`)

**Collision items only** (13 items; each has one wrong option that two mistakes share: M01/M08 x5, M02/M07 x2, D02/M02 x2, D05/D06 x2, D01/M03 x2). For each, and for each mistake c in the pair, one trial per sentence written for c. **Option alone** is `ml.bayes.update` on the shared answer (the two mistakes tie, so naming one is a coin flip: 0.50 always). **Option + sentence** adds `ml.bayes.apply_text` with the reader's probabilities (05 §4), skipped when the reader is unsure. Score = the top label is c (a tie counts 1/k). Quoted below inside the pair.

| Reader | Sentences | Trials | Option alone | Option + sentence | Gain | Gain 95% (resample contexts) | Reader unsure |
|---|---|---|---|---|---|---|---|
| tfidf | test | 780 | 0.50 | 0.72 | +0.22 | 0.15 to 0.29 | 53% |
| tfidf | persona | 310 | 0.50 | 0.76 | +0.26 | 0.19 to 0.33 | 45% |
| biencoder | test | 780 | 0.50 | 0.69 | +0.19 | 0.11 to 0.27 | 61% |
| biencoder | persona | 310 | 0.50 | 0.82 | +0.32 | 0.26 to 0.38 | 35% |
| frozen | test | 780 | 0.50 | 0.67 | +0.17 | 0.10 to 0.24 | 59% |
| frozen | persona | 310 | 0.50 | 0.74 | +0.24 | 0.18 to 0.30 | 49% |

What this says:
- A sentence helps with every reader and both pools; the interval for the gain stays above zero. The `test` pool has one context per mistake (9 contexts here) and `persona` two (18), so the context-resampled interval is the one to quote; the item- and trial-resampled intervals in the card are narrower and say less.
- **The reader is unsure on about half of these sentences** (35–61%), and an unsure reader leaves the posterior untouched, so those trials stay at 0.50. When the reader does answer, the pair is right 92–100% of the time ("answered only" in the card). If the sentence were always used (even below the threshold), the pair would be right 0.79–0.96 of the time (tfidf 0.91 test, 0.96 persona): the rule "skip when unsure" costs about 0.19 on these pairs. That variant is **not** the 05 §4 rule; it is in the card only to show the cost. For S3/D1: on a collision item the sentence only has to choose between two known mistakes, so a lower threshold, or using it unthresholded, is worth trying. Not changed here.
- Weakest pairs (tfidf, test): M02/M07 0.64 and D05/D06 0.67; strongest D02/M02 0.88. M01 inside M01/M08 gets only 0.58, the one pair the dataset itself cannot always separate (the M01/M08 label clash allowed in 06 §3, C3).

**Not measured, and it matters:** no sentence was ever written *for these items* (the T1 sentences come from separate hand-written contexts; DeepSeek was not called). Each trial uses a sentence written for mistake c about some other question, which the reader reads alone. A real "why?" sentence would be about the item, and might be easier or harder. All sentences are LLM-written; nothing here is a real student. Also not simulated: whether a learner with mistake c picks the shared option (taken from the 03 §6.2 table as written, p_b = 0.6).

**Decisions:** the model prior is uniform over the 19 labels, the learner prior is the population value (`ml.bayes.prior`); `CORRECT_REASON` is counted as the posterior's `CORRECT`; masking uses `serve.allowed_labels` on the item's code with the item's own classes never removed (the `keep=` argument of `read()`); the three readers' own thresholds are used as shipped. `serve.read()` itself is not called per trial (the sentences repeat across items), but the same functions are, so the answers match.

## E18 — is the item bank sound? (`ml/eval/e18_bank.py`)

**Quiz half, full:** `ml.items.verify_quiz` re-run over all 115 items on both backends (structure, the 05 §3.1 targets, each answer and each believed program re-worked by running the code). **Passed, 0 problems.** 100% verified by the interpreter; 104 of 115 (90.4%) also agreed by gcc, the other 11 are interpreter-only because C leaves them undefined; 0 marked manual; 13 collision items (target 12); every one of the 17 mistakes meets the targets (M01, M02, M03, M07, M08, D01, D02, D05, D06 have more than the minimum because of the collision items); the `verified` flags in the file match the fresh run; 128 believed programs run. The run takes about 35–65 s. Descriptive counts in the card: the correct option's position (tested against uniform within each item's own option count, p = see card), the correct option being the unique longest (31 of 98), difficulty (61 / 41 / 13, the drafter's estimate), concept counts, source (96 DeepSeek-drafted and checked by running, 19 hand-written).

**Verified means the code gives the stated answer and the believed program gives the stated belief answer.** It does not mean a student with that mistake picks that option; no student has answered these items. Nobody has skimmed the DeepSeek-drafted items (06 §5).

**Code half, run:** `.venv\Scripts\python -m ml.items.build_code_items --check` → `CHECK PASSED: 143 items` (gcc cross-check on). 67 `complete_snippet` over 34 problems (33 with 2, P13 with 1; target 1–2 per problem), 34 `fix_bug` (2 per class, all 17), 42 `debug_line` (2 planted per class, plus 8 with `planted` null = 8/42 = 19.0%, and 05 §3.2 asks for about 20%). All three count targets met. The quiz half was not re-run, so its numbers are the ones already on the card (115 items, 27.7 s, interpreter 100%, gcc 104/115).

## Skipped or weaker than the plan

- E16 was rerun at the end of this session (about 6 min, onnxruntime capped at 4 threads) and wrote a card identical to the committed one.
- E16: the DeepSeek zero-shot row (no API call allowed); the human set (never collected; the persona set stands in and is not student text); the fine-tuned reader's leave-one-class-out is copied from T1, not rerun.
- E17: sentences written for the collision items themselves (none exist).

## For other packages

- **C4:** E18 recorded `python -m ml.items.build_code_items --check` (`CHECK PASSED: 143 items`) and the 05 §3.2 counts. The quiz half was not re-run.
- **D1 / S3:** see the E17 bullet about the threshold on collision items.
- **I1:** the three cards are `ml/eval/e16_card.json`, `e17_card.json`, `e18_card.json` (same layout as 03 §9.4: `id`, `title`, `slice`, `n`, `metrics`, `per_class`, `plots`, `caveat`, plus extra keys). Each script has a `run()`; E17 and E18 write their files in `main()`.
- During this session `ml/eval/e16_text.py` was overwritten by mistake with a first draft of an E16 rewrite and put back byte-for-byte from `HEAD`; its content is the committed version.

## Commands

```
.venv\Scripts\python -m pytest tests/E-c -q
.venv\Scripts\python -m ml.eval.e16_text      # 5-9 min
.venv\Scripts\python -m ml.eval.e17_why       # about 1.5 min
.venv\Scripts\python -m ml.eval.e18_bank      # about 1 min
```