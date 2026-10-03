# 05 — All eight question types + the sentence reader (adds to v3)

> Adds to `03_ML_IMPLEMENTATION_PLAN_v3.md`; nothing in 03 is renamed or removed. Where this file is silent, 03 applies. New ids, fields, paths and endpoints here are contracts in the same sense as 03.
>
> Assumptions (change here if wrong): topics stay as in v3 (no pointers); "debugging" = tap the wrong line; "fix bug" = repair code we hand over; "reasoning" = the learner types one sentence. DeepSeek is used **offline only**, to draft content and write training sentences. The running app never calls it.

## 1. Which part judges which type

| Type | `type` id | Learner gives | Judged by | Trained? |
|---|---|---|---|---|
| MCQ | `mcq` | one option | Bayes layer (03 §6.2 likelihood table) | no |
| Predict the output | `predict_output` | one option | same | no |
| Predict next state | `next_state` | one option | same; the right answer is read from the trace | no |
| Debugging | `debug_line` | a line number or "no bug" | knowledge model (§4) | no |
| Complete the snippet | `complete_snippet` | code (holes filled) | diagnoser, through `/attempt` | LightGBM (03 §5) |
| Fix bug | `fix_bug` | code (repaired) | diagnoser, through `/attempt`, with the rule in §4 | LightGBM |
| Build the algo | (missions, exam coding items) | code | diagnoser | LightGBM |
| Reasoning | `reasoning` | one typed sentence | **sentence reader** (§6) → Bayes layer | **new model** |

## 2. New files

| Path | Contents | Made by |
|---|---|---|
| `ml/data/quiz_items.json` | `mcq`, `predict_output`, `next_state`, `reasoning` items | authored (DeepSeek drafts, a script verifies, a human skims) |
| `ml/data/code_items.json` | `complete_snippet`, `fix_bug`, `debug_line` items | generated from the problem bank + operators |
| `ml/data/reasons.jsonl` | sentences labelled by class | DeepSeek (training) + classmates (test only) |
| `ml/data/llm_raw/` | every raw DeepSeek response, one file per call | written by the client; committed, so nothing is paid for twice |
| `ml/data/external/` | public datasets | downloaded; **gitignored**, never committed |
| `ml/items/` | `verify_quiz.py`, `draft_quiz_llm.py`, `build_code_items.py` | |
| `ml/quiz/` | `select.py` (next practice item) | |
| `ml/text/` | the sentence reader (§6) and data generation (§7) | |
| `ml/oracle/` | gcc harness: compile a learner function with a generated `main`, return `returned` / `printed` | |
| `ml/runner.py` | `run_tests(problem, code, backend)` and `trace(problem, code, test_index)`; backends `interp` and `gcc` | |

`ml/runner.py` exists so that work which only needs pass/fail can start on the gcc backend before the interpreter is finished.

## 3. Item shapes

### 3.1 `quiz_items.json`
```json
{
  "item_id": "qz_m01_po_01",
  "type": "predict_output",
  "concept": "loops",
  "classes": ["M01"],
  "code": "int s = 0;\nfor (int i = 0; i <= 3; i++) s += i;\nprintf(\"%d\", s);",
  "question": "What does this print?",
  "options": ["3", "6", "10"],
  "correct": "6",
  "belief": {"M01": "3"},
  "state_query": null,
  "explain": "i takes 0, 1, 2 and 3, so the body runs four times.",
  "difficulty": 1,
  "verified": {"interp": true, "gcc": true, "manual": false}
}
```
- `next_state` adds `"state_query": {"var": "i", "after_line": 2, "hit": 3}` (value of `var` after line 2 has run for the 3rd time). `correct` must equal the value in the trace's `steps[]`.
- `reasoning` has no `options`; it has `"expected": "CORRECT_REASON"` and `classes` (the mistakes a wrong reason would show). The answer goes to `/reason`.
- Rules checked by `verify_quiz.py`: 3–4 distinct options; `correct` is one of them; every `belief` value is an option and differs from `correct`; a class appears in `belief` at most once; items with runnable code agree with the interpreter **and** gcc. Concept-only items (no runnable code) are marked `manual: true` and listed for a human to skim.
- Target: per class at least 2 `mcq`, 2 `predict_output`, 1 `next_state`, 1 `reasoning` (17 classes → about 100 items). At least 12 **collision** items, where one wrong option is the belief answer of two classes.
- None of these reuse the probes, traps or exam trace items of 03 (those stay separate so practice does not leak into reassessment).

### 3.2 `code_items.json`
```json
{"item_id": "cs_P03_01", "type": "complete_snippet", "problem_id": "P03",
 "starter": "int total_energy(int cells[], int n) {\n    int total = ____;\n    for (int i = ____; i ____ n; i++) total += cells[i];\n    return total;\n}",
 "holes": [{"id": "h1", "line": 2, "kind": "acc_init", "choices": ["0", "1", "cells[0]"]},
           {"id": "h2", "line": 3, "kind": "loop_init", "choices": ["0", "1"]},
           {"id": "h3", "line": 3, "kind": "loop_rel", "choices": ["<", "<=", "!="]}],
 "exposes": ["M01", "M05", "M08"]}

{"item_id": "fb_P03_m05_01", "type": "fix_bug", "problem_id": "P03", "planted": "M05", "op_id": "m05_drop_init_acc",
 "starter": "<verified buggy mutant>", "bug_lines": [2]}

{"item_id": "dl_P03_m03_01", "type": "debug_line", "problem_id": "P03", "planted": "M03", "op_id": "m03_assign_in_loop",
 "code": "<verified buggy mutant>", "bug_lines": [4], "allow_no_bug": true,
 "explain": "total is set back to 0 on every pass.", "fix": {"code": "...", "changed_lines": [4]}}
```
- Holes sit exactly where a mutation operator applies (loop start, loop comparison, update, index, accumulator start, base case, recursive argument, swap lines, `return`). The app replaces each `____` and submits whole code. `choices` are optional chips for phone typing.
- `fix_bug` and `debug_line` code comes from verified dataset rows (03 §3.5.4), unaugmented, one bug each. About 20% of `debug_line` items have `"planted": null` and correct code, so "no bug" is a real answer.
- Targets: 1–2 `complete_snippet` per problem; 2 `fix_bug` and 2 `debug_line` per class.

## 4. Scoring rules (extends 03 §8.2; all numbers hand-set, like the rest of that table)

| Item | Update |
|---|---|
| `mcq`, `predict_output`, `next_state` | Bayes layer with 03 §6.2 (`p_b` 0.6, `q` 0.5, `s` 0.1), then the knowledge-model row "probe / MCQ with belief distractor" |
| `debug_line`, planted k | correct line: g = 0.30, s = 0.20. `planted: null` items give no update |
| `fix_bug`, planted k, all tests pass | item response "correct" with g = 0.35, s = 0.15 |
| `fix_bug`, planted bug still there (top-1 = k and `bug_lines` unchanged) | item response "wrong" with the same g and s. **Do not** apply the Diagnosis rule: the learner did not write this bug |
| `fix_bug`, a different bug j appears (top-1 = j ≠ k, p ≥ 0.5) | normal Diagnosis rule for j: the learner wrote it |
| `complete_snippet` | a normal attempt: full diagnosis; on pass, the "same-family code task" row |
| `reasoning` or a "why?" follow-up, reader says k | Bayes layer: `p(k) ← p(k) · (0.02 + P_text(k))^0.5`, renormalise. Skipped when the reader is unsure. Knowledge model: if matched with `P_text(k) ≥ 0.6`, likelihoods 0.5 (active) vs 0.05 (not active) |

A "why?" follow-up is asked after a quiz answer only when the chosen option is the belief answer of two or more classes (`ask_reason: true`). That is the case the option alone cannot settle.

## 5. API additions (extends 03 §11; same envelope rules)

| Method | Path | Request | Response |
|---|---|---|---|
| GET | `/quiz/next` | `?learner_id=&type=&concept=` (filters optional) | `{item, reason}`; item has public fields only (no `correct`, `belief`, `explain`) |
| POST | `/quiz/answer` | `{learner_id, item_id, answer, ms}` | `{correct, correct_answer, explain, updates: [{class, p_before, p_after}], ask_reason}` |
| GET | `/code-items` | `?problem_id=&type=` | `[code item, public fields]` (no `planted`, `bug_lines`, `fix`) |
| POST | `/code-items/debug` | `{learner_id, item_id, line}` (`line: null` = "no bug") | `{correct, bug_lines, explain, fix, updates}` |
| POST | `/attempt` | gains optional `code_item_id` | unchanged; applies the §4 `fix_bug` rule when the id is a `fix_bug` item |
| POST | `/reason` | `{learner_id, ref: {kind: "quiz" \| "attempt", id}, text}` | `{status: "matched" \| "correct_reasoning" \| "unsure", top: [{id, name, p}], reader: "biencoder" \| "frozen" \| "tfidf" \| "none", updates}` |
| POST | `/lab/reason` | `{text, code?}` | same, stateless (for judges) |

`/quiz/next` picks by the same expected-information-gain code as the exam (03 §8.5.4) without the exam's composition rules; `reason` is the same kind of one-line "why this question".

## 6. Sentence reader (`ml/text/`)

**Job:** given a code snippet and one sentence from the learner, say which mistake the sentence shows, or say "unsure".

- **Labels:** the 17 classes + `CORRECT_REASON`. "Unsure" is not a label; it is the answer when confidence is below a threshold.
- **Input:** `code + "\n[WHY] " + sentence`. The chosen option is **not** part of the input: the Bayes layer already counts it, and feeding it here would count it twice.
- **Three readers behind one interface** (`serve.py: read(code, text) -> {probs, status, reader}`), tried in this order at start-up: fine-tuned → frozen → word-count → none.

| Reader | What it is | Training | Load |
|---|---|---|---|
| `tfidf` | word and character counts + logistic regression | seconds, laptop CPU | no extra installs |
| `frozen` | sentences turned into vectors by an unchanged `BAAI/bge-base-en-v1.5`, then logistic regression on the vectors | no GPU training; about 1–2 min to embed 3k sentences on CPU (estimate) | model file about 420 MB |
| `biencoder` | the same model fine-tuned so a sentence lands next to the **description** of its mistake (description = name + wrong belief from 03 §3.1) | about 3–10 min on a free T4 or on the RTX 4050 (estimate); 4 epochs, lr 2e-5, batch 32 (T4) or 16 (4050), max length 256, mixed precision, cross-entropy over cosine similarity to all descriptions, temperature 0.05 | 3–4 GB GPU memory (estimate) |

- **Why descriptions:** a new mistake can be added by writing its description, with no retraining. That is what the leave-one-out test (E16) measures.
- **Threshold:** chosen on validation so that accepted answers are at least 90% right; below it the status is `unsure` and nothing is updated.
- **Where to train:** Kaggle or Colab, with `ml/text/kaggle_train.py` (one self-contained script: installs its own packages, reads `reasons.jsonl`, trains, exports). Reason: GPU PyTorch is about 5–6 GB on a drive that is 95% full. The leave-one-out run is 17 retrains, about 1–2 hours on a T4 (estimate), so that one should be in the cloud regardless.
- **Running it on the laptop:** export to ONNX in the cloud; the laptop then needs only `onnxruntime` and `tokenizers` (tens of MB) plus the model file. Fallback if the export misbehaves: CPU-only PyTorch (about 1 GB, estimate).
- **Artifacts:** `ml/artifacts/reason_<sha8>/{model.onnx, tokenizer.json, descriptions.json, meta.json}`; gitignored (over GitHub's 100 MB file limit), shared as a download. `tfidf` and `frozen` heads are small and are committed.
- Model id is a config value. `google/embeddinggemma-300m` is a newer option that handles more languages, but it needs a licence click-through on Hugging Face.

## 7. DeepSeek (offline only)

- **Key:** `DEEPSEEK_API_KEY` in a root `.env` (gitignored; see `.env.example`). Never printed, never written to `llm_raw/`.
- **Model name:** `DEEPSEEK_MODEL` in `.env`. Third-party pages say `deepseek-chat` was retired and `deepseek-flash` is current; **check DeepSeek's own model list before the first run**.
- **Client:** `ml/text/llm_client.py`: OpenAI-compatible calls, JSON output, temperature per job, retries, and every raw response saved under `ml/data/llm_raw/<job>/<hash>.json` so a rerun reads from disk instead of paying again.

| Job | Script | Output | Size | Cost (estimate) |
|---|---|---|---|---|
| Reason sentences | `ml/text/gen_reasons.py` | `reasons.jsonl` | about 150 per label × 18 ≈ 2,700 | well under $1 |
| Quiz item drafts | `ml/items/draft_quiz_llm.py` | candidates for `quiz_items.json` | about 150 drafts for about 100 kept | well under $1 |
| Zero-shot baseline (optional) | `ml/eval/e16_text.py` | one row in E16 | test sentences only | cents |

**Reason sentences, rules for the generator:**
- One call = one (class, context, voice). Context = a trap, probe or quiz item where that class has a belief answer, plus the answer the learner gave. Voices: at least 6 (confident, unsure, very short, typos, Hinglish, over-explaining).
- The prompt gives the wrong belief and asks for what such a student would *say*; it forbids naming the rule or the class.
- Filters: drop sentences that contain a class name or the description's own wording; drop near-duplicates; drop anything under 3 words.
- Row: `{id, label, context_id, code, chosen, text, voice, source: "deepseek" | "human", split}`.
- Splits are **by context**, never by row, so the test contexts are unseen. Each class needs at least 6 contexts.
- `source: "human"` rows (50 or more, typed by classmates through the app's "why?" box) are test only.

## 8. Public datasets

No public dataset has C beginners' work labelled by mistake type, so none is used for training. All of these go under `ml/data/external/` and are for evaluation only.

| Dataset | Use here | Notes |
|---|---|---|
| ITSP (`github.com/jyi/ITSP`) | real-student test slice, 03 §3.4 protocol X | no licence in the repo: do not redistribute |
| Codeflaws (`github.com/codeflaws/codeflaws`) | "bugs we have no class for" slice: 3,902 buggy/fixed C pairs with tests | contest code with `scanf`, so the interpreter rejects most of it: AST-only features. Measures how often the model abstains or says OTHER |
| IntroClass (`github.com/ProgramRepair/IntroClass`) | same use as Codeflaws, smaller | BSD-3 |
| Mohler / Texas short answers | "should say unsure" set for the sentence reader: real CS student sentences that match none of our classes | graded 0–5 for correctness, not tagged by mistake |
| McMining (`github.com/taisazero/mcminer`) | source of ideas for extra mistake descriptions | Python, bugs injected by an LLM; not used as data |

## 9. New experiments (extend 03 §9.2; cards go into `metrics.json` the same way)

| ID | Question | Protocol | Metrics |
|---|---|---|---|
| **E16** | How well does the sentence reader work? | three readers (+ optional DeepSeek zero-shot) on: held-out contexts; leave-one-class-out (the class has a description but no training sentences); the human set; the Mohler "should say unsure" set | macro-F1, top-3, accuracy among accepted answers, how often it says unsure, per-voice breakdown (English vs Hinglish) |
| **E17** | Does asking "why?" add anything? | collision items only: option alone vs option + sentence | accuracy of naming the right class; this is the ablation for the reasoning type |
| **E18** | Is the item bank sound? | `verify_quiz.py` + `build_code_items.py --check` over every item | % verified by interpreter and gcc; count marked manual; items per class and type |

Caveat to print on E16 and E17: training sentences are written by an LLM; only the human set reflects real phrasing.

## 10. What changes for the app (Expo)

- New screens or cards: quiz card (options), line-tap on the code view (`debug_line`), hole chips (`complete_snippet`), a one-line "why?" box (`reasoning` and follow-ups).
- 02 is written for a Vite web app. On Expo, CodeMirror needs a WebView or is replaced by a plain monospace text box; Tailwind and Framer Motion need their React Native equivalents. The API is the same either way.
- The phone must reach the laptop: server on `0.0.0.0`, same Wi-Fi or Tailscale.
