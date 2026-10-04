# E17: does asking "why?" add anything?

Generated 2026-10-04 by `python -m ml.eval.e17_why`.

**Caveat.** Every reason sentence used here is LLM-written. The 270 'persona' sentences are role-played students (a different model and prompt from the training sentences); no real student sentence was collected. Results on real phrasing are not measured. The sentences were not written for these items (see not_measured).

Collision items: 13 items, 13 shared wrong answers. Each trial is one sentence written for one of the mistakes that share the answer. **The sentences are about other questions**, not these items.

## Headline (naming the right mistake among those that share the answer)

| Reader | Sentences | Trials | Option alone | Option + sentence | Gain | Gain 95% (resample contexts) | Reader unsure |
|---|---|---|---|---|---|---|---|
| tfidf | test | 780 | 0.50 | 0.72 | +0.22 | 0.15 to 0.29 | 53% |
| tfidf | persona | 310 | 0.50 | 0.76 | +0.26 | 0.19 to 0.33 | 45% |
| biencoder | test | 780 | 0.50 | 0.69 | +0.19 | 0.11 to 0.27 | 61% |
| biencoder | persona | 310 | 0.50 | 0.82 | +0.32 | 0.26 to 0.38 | 35% |
| frozen | test | 780 | 0.50 | 0.67 | +0.17 | 0.10 to 0.24 | 59% |
| frozen | persona | 310 | 0.50 | 0.74 | +0.24 | 0.18 to 0.30 | 49% |

Option alone is a tie between the mistakes in the group, so it scores 1/size (0.50 for every pair here). "Reader unsure" trials leave the posterior unchanged.

## Same, over all 19 labels and the sentence-only ablation

| Reader | Sentences | Option alone (19 labels) | Option + sentence (19 labels) | Sentence alone (group) | Variant: sentence always used (group) | 95% (contexts) | Option + sentence, answered only | Answered |
|---|---|---|---|---|---|---|---|---|
| tfidf | test | 0.50 | 0.72 | 0.91 | 0.91 | 0.86 to 0.96 | 0.96 | 370 |
| tfidf | persona | 0.50 | 0.76 | 0.96 | 0.96 | 0.93 to 0.99 | 0.98 | 169 |
| biencoder | test | 0.50 | 0.69 | 0.85 | 0.85 | 0.72 to 0.96 | 0.98 | 304 |
| biencoder | persona | 0.50 | 0.82 | 0.92 | 0.92 | 0.85 to 0.98 | 1.00 | 201 |
| frozen | test | 0.50 | 0.67 | 0.79 | 0.79 | 0.63 to 0.93 | 0.92 | 317 |
| frozen | persona | 0.50 | 0.74 | 0.88 | 0.88 | 0.76 to 0.98 | 0.97 | 158 |

How to read this. The 05 §4 rule skips the sentence when the reader is unsure, and on these items the reader is unsure for roughly half of the sentences; those trials stay at a coin flip, which is what pulls "option + sentence" below "sentence alone". "Sentence alone" and the variant never skip, so they do not pay for the threshold; the variant is **not** the rule in 05 §4, it is shown to say what the threshold costs on these pairs. When the reader does answer, the pair is right almost every time ("answered only"). "Sentence alone" is scored inside the group only, so it is not comparable with the 19-label columns.

## Intervals for the headline rows

| Reader | Sentences | Contexts | Option + sentence | 95% (contexts) | 95% (items) | 95% (trials) | Gain | Gain 95% (contexts) | Gain 95% (items) | Gain 95% (trials) |
|---|---|---|---|---|---|---|---|---|---|---|
| tfidf | test | 9 | 0.72 | 0.65 to 0.79 | 0.68 to 0.76 | 0.70 to 0.74 | +0.22 | 0.15 to 0.29 | 0.18 to 0.26 | 0.20 to 0.24 |
| tfidf | persona | 18 | 0.76 | 0.69 to 0.83 | 0.73 to 0.79 | 0.73 to 0.79 | +0.26 | 0.19 to 0.33 | 0.23 to 0.29 | 0.23 to 0.29 |
| biencoder | test | 9 | 0.69 | 0.61 to 0.77 | 0.65 to 0.73 | 0.67 to 0.71 | +0.19 | 0.11 to 0.27 | 0.15 to 0.23 | 0.17 to 0.21 |
| biencoder | persona | 18 | 0.82 | 0.76 to 0.88 | 0.80 to 0.85 | 0.80 to 0.85 | +0.32 | 0.26 to 0.38 | 0.30 to 0.35 | 0.30 to 0.35 |
| frozen | test | 9 | 0.67 | 0.60 to 0.74 | 0.64 to 0.70 | 0.65 to 0.69 | +0.17 | 0.10 to 0.24 | 0.14 to 0.20 | 0.15 to 0.19 |
| frozen | persona | 18 | 0.74 | 0.68 to 0.80 | 0.71 to 0.77 | 0.71 to 0.77 | +0.24 | 0.18 to 0.30 | 0.21 to 0.27 | 0.21 to 0.27 |

## By pair of mistakes (reader tfidf, test sentences)

| Pair | Items | Trials | Option alone | Option + sentence | Gain | Per mistake (alone→with sentence) |
|---|---|---|---|---|---|---|
| D01/M03 | 2 | 120 | 0.50 | 0.72 | +0.22 | D01: 0.50→0.77, M03: 0.50→0.67 |
| D02/M02 | 2 | 120 | 0.50 | 0.88 | +0.38 | D02: 0.50→0.95, M02: 0.50→0.82 |
| D05/D06 | 2 | 120 | 0.50 | 0.67 | +0.17 | D05: 0.50→0.72, D06: 0.50→0.62 |
| M01/M08 | 5 | 300 | 0.50 | 0.70 | +0.20 | M01: 0.50→0.58, M08: 0.50→0.82 |
| M02/M07 | 2 | 120 | 0.50 | 0.64 | +0.14 | M02: 0.50→0.65, M07: 0.50→0.63 |

## Not measured

- Sentences written for these items: none exist. Trials use sentences about other questions, written for the same mistake. A sentence about the item itself could be easier or harder to read.
- Real students: the learner never exists here. The 'learner' is a sentence drawn from a pool of LLM-written text.
- The probability that a learner with mistake c picks the shared answer (p_b): the option update is taken from the table as written (03 §6.2), not simulated.
- Whether the app asks the follow-up on every collision item: ask_reason is true on all of these by construction (collision_answers).

## Protocol

- **units**: a trial = (collision item, shared wrong answer, mistake c in the group, one sentence written for c)
- **option_alone**: ml.bayes.prior (uniform model prior, population learner prior) then ml.bayes.update with the shared wrong answer
- **option_plus_sentence**: then ml.bayes.apply_text with the reader's probabilities (CORRECT_REASON counted as CORRECT), skipped when the reader says unsure
- **masking**: serve.allowed_labels(item code) then serve.mask_probs, with the item's own classes never removed (the keep= argument of read())
- **score**: top label equals c; a tie of k labels counts 1/k. 'group' = among the mistakes that share the answer, 'all19' = over all 19 labels
- **intervals**: 95% percentile, 1000 resamples. _ci_contexts resamples whole sentence contexts (the one to quote: the sentences of one context are about one question, and the test pool has one context per mistake); _ci_items resamples the 13 items, which differ little within a pair of mistakes, so it is narrow; _ci_trials resamples single trials and is narrowest
