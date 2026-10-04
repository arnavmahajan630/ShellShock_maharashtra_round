# E16: how well does the sentence reader work?

Generated 2026-10-04 by `python -m ml.eval.e16_text`.

**Caveat.** The training sentences are LLM-written, and so are both test sets. Only Mohler is real student text, and it has no labels for our classes, so it can only show how often a reader stays unsure. No number on this card measures accuracy on real students' sentences.

## Readers

| Reader | Loaded | Threshold | Version |
|---|---|---|---|
| tfidf | yes | 0.584 | 89d0e16d |
| biencoder | yes | 0.871 | 9d7b8962 |
| frozen | yes | 0.585 | dfdaf841 |

Serving order: tfidf, biencoder, frozen, then none. Default here: `tfidf`. Thresholds were chosen on the val split; nothing was tuned on the sets below.

## 1 and 2. Held-out contexts and the second test set

`test` = Held-out contexts (test split of reasons.jsonl; LLM-written, same generator as training). `persona` = **LLM-written stand-in for classmates' sentences** (`reasons_persona.jsonl`, a different model and prompt from the training sentences). It is not text written by students.

| Reader | Set | n | Accuracy | 95% interval | Macro-F1 | Top-3 | Answers | Right when it answers |
|---|---|---|---|---|---|---|---|---|
| tfidf | test | 539 | 0.720 | 0.63 to 0.82 | 0.714 | 0.876 | 54% | 89% |
| tfidf | persona | 270 | 0.730 | 0.67 to 0.79 | 0.762 | 0.900 | 61% | 83% |
| tfidf | persona_picked_by_itself | 163 | 0.681 | 0.58 to 0.76 | 0.695 | 0.896 | 58% | 77% |
| tfidf | persona_told | 107 | 0.804 | 0.71 to 0.89 | 0.829 | 0.907 | 64% | 91% |
| biencoder | test | 539 | 0.718 | 0.62 to 0.82 | 0.702 | 0.891 | 54% | 94% |
| biencoder | persona | 270 | 0.707 | 0.67 to 0.75 | 0.762 | 0.919 | 64% | 81% |
| biencoder | persona_picked_by_itself | 163 | 0.638 | 0.57 to 0.70 | 0.688 | 0.902 | 64% | 73% |
| biencoder | persona_told | 107 | 0.813 | 0.73 to 0.89 | 0.803 | 0.944 | 65% | 93% |
| frozen | test | 539 | 0.603 | 0.47 to 0.74 | 0.596 | 0.852 | 47% | 84% |
| frozen | persona | 270 | 0.667 | 0.60 to 0.73 | 0.697 | 0.893 | 53% | 83% |
| frozen | persona_picked_by_itself | 163 | 0.687 | 0.61 to 0.76 | 0.701 | 0.902 | 53% | 81% |
| frozen | persona_told | 107 | 0.636 | 0.51 to 0.76 | 0.655 | 0.879 | 53% | 84% |

Intervals resample whole contexts. "Answers" is the share at or above the threshold; the rest are `unsure`.

## 3. By voice

| Reader | Set | Voice | Language | n | Accuracy | Top-3 | Answers | Right when it answers |
|---|---|---|---|---|---|---|---|---|
| tfidf | test | confident | English | 85 | 0.776 | 0.965 | 60% | 92% |
| tfidf | test | hinglish | Hinglish | 85 | 0.694 | 0.847 | 41% | 97% |
| tfidf | test | over_explaining | English | 95 | 0.705 | 0.874 | 58% | 87% |
| tfidf | test | typos | English | 95 | 0.726 | 0.884 | 48% | 89% |
| tfidf | test | unsure | English | 95 | 0.705 | 0.842 | 53% | 92% |
| tfidf | test | very_short | English | 84 | 0.714 | 0.845 | 63% | 81% |
| tfidf | persona | persona_crammer | English | 45 | 0.756 | 0.911 | 69% | 84% |
| tfidf | persona | persona_friend | Hinglish | 45 | 0.822 | 0.933 | 47% | 95% |
| tfidf | persona | persona_hostel_2am | English | 46 | 0.652 | 0.891 | 63% | 76% |
| tfidf | persona | persona_nervous | English | 45 | 0.756 | 0.867 | 56% | 84% |
| tfidf | persona | persona_topper | English | 44 | 0.727 | 0.955 | 66% | 76% |
| tfidf | persona | persona_tracer | English | 45 | 0.667 | 0.844 | 64% | 86% |
| biencoder | test | confident | English | 85 | 0.812 | 0.941 | 55% | 98% |
| biencoder | test | hinglish | Hinglish | 85 | 0.718 | 0.882 | 47% | 88% |
| biencoder | test | over_explaining | English | 95 | 0.726 | 0.853 | 57% | 94% |
| biencoder | test | typos | English | 95 | 0.695 | 0.895 | 52% | 98% |
| biencoder | test | unsure | English | 95 | 0.653 | 0.842 | 61% | 91% |
| biencoder | test | very_short | English | 84 | 0.714 | 0.940 | 54% | 91% |
| biencoder | persona | persona_crammer | English | 45 | 0.778 | 0.933 | 80% | 81% |
| biencoder | persona | persona_friend | Hinglish | 45 | 0.756 | 0.933 | 56% | 88% |
| biencoder | persona | persona_hostel_2am | English | 46 | 0.652 | 0.891 | 65% | 73% |
| biencoder | persona | persona_nervous | English | 45 | 0.667 | 0.889 | 67% | 77% |
| biencoder | persona | persona_topper | English | 44 | 0.682 | 0.955 | 59% | 85% |
| biencoder | persona | persona_tracer | English | 45 | 0.711 | 0.911 | 60% | 85% |
| frozen | test | confident | English | 85 | 0.682 | 0.894 | 54% | 89% |
| frozen | test | hinglish | Hinglish | 85 | 0.529 | 0.800 | 32% | 89% |
| frozen | test | over_explaining | English | 95 | 0.600 | 0.853 | 46% | 86% |
| frozen | test | typos | English | 95 | 0.632 | 0.853 | 45% | 86% |
| frozen | test | unsure | English | 95 | 0.611 | 0.832 | 52% | 84% |
| frozen | test | very_short | English | 84 | 0.560 | 0.881 | 54% | 73% |
| frozen | persona | persona_crammer | English | 45 | 0.756 | 0.911 | 71% | 91% |
| frozen | persona | persona_friend | Hinglish | 45 | 0.689 | 0.889 | 31% | 79% |
| frozen | persona | persona_hostel_2am | English | 46 | 0.543 | 0.891 | 59% | 70% |
| frozen | persona | persona_nervous | English | 45 | 0.622 | 0.844 | 49% | 82% |
| frozen | persona | persona_topper | English | 44 | 0.750 | 0.932 | 57% | 88% |
| frozen | persona | persona_tracer | English | 45 | 0.644 | 0.889 | 51% | 83% |

Persona voices merge the `_told` rows with the rest. Hinglish against English:

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

## 4. Mohler: should say unsure

Mohler short answers (real CS students; none of our classes applies). n = 2442. Every 'matched' is wrong: these answers are about other questions and show none of our 17 mistakes. 'correct_reasoning' is listed apart; it updates nothing about a mistake.

| Reader | n | Unsure | Wrongly `matched` | `correct_reasoning` | Most often matched to |
|---|---|---|---|---|---|
| tfidf | 2442 | 89% | 10% | 1% | D04 2.9%, M08 2.7%, D06 2.1%, M05 0.9%, D08 0.9% |
| biencoder | 2442 | 87% | 10% | 3% | D04 5.5%, D06 1.8%, M08 1.7%, M05 0.9%, D03 0.1% |
| frozen | 2442 | 83% | 12% | 5% | D04 6.5%, D06 1.6%, M08 1.4%, D08 0.7%, M05 0.6% |

## 5. Leave-one-class-out

Fine-tuned reader: **run on Colab on 2026-10-04, copied from notes/T1.md**. Trained without any sentence of the class, then asked to find them from its description.

Unchanged model: BAAI/bge-base-en-v1.5 with no training at all: nearest description (texts from ml/artifacts/reason_9d7b8962/descriptions.json), computed here on the same sentences (every sentence of the class).

| Class | Fine-tuned, class held out: top-1 | top-3 | Unchanged model: top-1 (all sentences) | top-3 | n | Unchanged model: top-1 (test split) | n |
|---|---|---|---|---|---|---|---|
| M01 | 0.000 | 0.017 | 0.594 | 0.928 | 180 | 0.567 | 30 |
| M02 | 0.067 | - | 0.184 | 0.520 | 179 | 0.233 | 30 |
| M03 | 0.106 | - | 0.550 | 0.933 | 180 | 0.633 | 30 |
| M04 | 0.217 | - | 0.811 | 0.961 | 180 | 0.800 | 30 |
| M05 | 0.155 | - | 0.540 | 0.862 | 174 | 0.828 | 29 |
| M06 | 0.011 | - | 0.256 | 0.594 | 180 | 0.467 | 30 |
| M07 | 0.000 | 0.067 | 0.144 | 0.344 | 180 | 0.167 | 30 |
| M08 | 0.044 | - | 0.700 | 0.878 | 180 | 0.433 | 30 |
| M10 | 0.272 | - | 0.550 | 0.794 | 180 | 0.633 | 30 |
| D01 | 0.144 | - | 0.322 | 0.433 | 180 | 0.233 | 30 |
| D02 | 0.311 | 0.600 | 0.517 | 0.672 | 180 | 0.533 | 30 |
| D03 | 0.250 | - | 0.633 | 0.817 | 180 | 0.733 | 30 |
| D04 | 0.267 | - | 0.456 | 0.717 | 180 | 0.367 | 30 |
| D05 | 0.011 | - | 0.190 | 0.430 | 179 | 0.300 | 30 |
| D06 | 0.011 | - | 0.383 | 0.656 | 180 | 0.333 | 30 |
| D07 | 0.139 | - | 0.589 | 0.883 | 180 | 0.767 | 30 |
| D08 | 0.028 | - | 0.583 | 0.844 | 180 | 0.767 | 30 |
| **mean of 17** | 0.120 | 0.283 | 0.471 | 0.722 |  | 0.517 |  |

The fine-tuned top-3 per class is in the notes for three classes only.

## Extra: with the code passed for masking

The same sentences with each row's code passed to read(). Only whole, self-contained code is used for masking (it parses as C functions and calls nothing it does not define), so bare quiz fragments change nothing.

| Reader | Set | Rows with usable code | n | Accuracy | Top-3 | Answers | Right when it answers |
|---|---|---|---|---|---|---|---|
| tfidf | test | 120 | 539 | 0.720 | 0.876 | 55% | 90% |
| tfidf | persona | 64 | 270 | 0.730 | 0.900 | 61% | 83% |
| biencoder | test | 120 | 539 | 0.718 | 0.891 | 55% | 94% |
| biencoder | persona | 64 | 270 | 0.707 | 0.919 | 65% | 81% |
| frozen | test | 120 | 539 | 0.603 | 0.853 | 48% | 84% |
| frozen | persona | 64 | 270 | 0.667 | 0.893 | 53% | 83% |

## 6. Not run

- **DeepSeek zero-shot**: Optional row of 05 §9. Not run: no LLM API call was made for this card.
