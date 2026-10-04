# E-c — Sentence reader and item-bank experiments

Built on 2026-10-04. **Only E16 is built.** E17 and E18 are not, and there are no placeholder scripts for them; see the end of this file.

## Built

| File | What it is |
|---|---|
| `ml/eval/e16_text.py` | the experiment: `python -m ml.eval.e16_text`; `build_card(...)` and `run()` return the card as a dict |
| `ml/eval/e16_card.json` | the card (40 KB) |
| `ml/eval/e16_card.md` | the same as tables (10 KB) |
| `tests/E-c/test_e16_text.py` | 13 tests |

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

## E17 and E18: not built

**E17 — does asking "why?" add anything?** (collision items only: option alone against option + sentence). It waits for:
1. `ml/data/quiz_items.json` with its collision items (at least 12, 05 §3.1) and `ml/items/verify_quiz.py` (package B4). `ml/items/` is empty.
2. The Bayes layer (package D1, `ml/bayes/`): the option update of 03 §6.2 and the sentence update of 05 §4. `ml/bayes/` is empty.
3. "Why" sentences written for those collision items, each labelled with the class its writer holds. None exist: the T1 contexts are hand-written questions, not the quiz items. They come from T1's second pass over B4's items (06 §3, T1 row), which needs DeepSeek, or from classmates.
4. The sentence reader (T2). Done.

**E18 — is the item bank sound?** (`verify_quiz.py` and `build_code_items.py --check` over every item). It waits for:
1. `ml/data/quiz_items.json` and `ml/items/verify_quiz.py` (package B4).
2. `ml/data/code_items.json` and `ml/items/build_code_items.py` with `--check` (package C4). C4 needs the generator's verified dataset (C3; `ml/generate/` is empty), which needs the problems (B1, B2; `ml/problems/` is empty) and the operators (C1, C2).
3. Both backends through `ml/runner.py`: the interpreter (`ml/c_interp/`, present) and gcc (`ml/oracle/`, present).

## Commands

```
.venv\Scripts\python -m pytest tests/E-c -q
.venv\Scripts\python -m ml.eval.e16_text
```
