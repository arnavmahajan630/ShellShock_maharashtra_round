# 06 — Work packages for parallel agents (ML + server)

> Splits everything in `03_ML_IMPLEMENTATION_PLAN_v3.md` and `05_QUESTION_TYPES_AND_SENTENCE_READER.md` into packages that different agents can build at the same time. This file replaces the build order in 04 §6. The app (Expo) is not covered here; its contract is 03 §11 + 05 §5.
>
> No schedule is given on purpose. Each package lists what it needs before it can start and how to tell it is done.

## 1. How the split works

1. **One package, one owner, no shared files.** Every file belongs to exactly one package (the "Owns" column). An agent edits only what it owns.
2. **Contracts are frozen first.** Package W0 writes the shared shapes (`ml/contracts/`) and fake responses for every endpoint. After W0, nobody edits them.
3. **Stand-ins remove the waiting.** Most packages need the C interpreter, which is the largest single piece. W0 provides three stand-ins so they can start at once:
   - `ml/runner.py`, which can run tests through **gcc** (package A2) instead of the interpreter;
   - hand-written sample traces in `tests/fixtures/traces/`;
   - three sample problems (P03, P11, Q17) in `tests/fixtures/problems/`.
4. **Each package has two finish lines:** "Starts after" (what must exist to begin) and "Final check needs" (what must exist to run its real acceptance test).
5. **One git worktree and branch per package** (`pkg/<id>`). Because ownership never overlaps, branches merge without conflicts. All worktrees use the one root `.venv`.

## 2. Rules every agent gets

- Read: 03, 05, this file, and `ml/contracts/`. Build only your package.
- Edit only the paths in your "Owns" cell, plus `tests/<your id>/` and `notes/<your id>.md`.
- Never edit `ml_plan/`, `design_plan/`, `ml/contracts/`, `ml/runner.py`, `server/app/main.py`, `requirements.txt`. If a contract looks wrong or a package is missing something, write it in `notes/<your id>.md` and work around it inside your own files.
- Need a new pip package? Write it in your notes; do not install it.
- Class ids, feature names, JSON fields and paths from 03 and 05 are fixed. Do not rename.
- Finish = your acceptance command passes, and `notes/<your id>.md` says what was built, what was skipped, and what is still faked.
- Never print, log or commit `DEEPSEEK_API_KEY`. Only T1 and B4 may call DeepSeek, and only through `ml/text/llm_client.py`. E-a may call it once for the E8 zero-shot row on R, through the same client.
- Strings (Q13–Q15, D08) are built last in every package and are the first thing dropped.
- Decisions from `04_FEASIBILITY_AND_ML_BUILD_PLAN.md` that override 03 where the two disagree:
  - The generator changes the syntax tree and reprints with `pycparser.c_generator`. Text edits by coordinate exist only in the fixer (03 §3.5).
  - Comment stripping happens in `preprocess.py` before parse: comments become spaces, newlines stay (03 §2.5).
  - R-team is 30 items, not 60. The realistic set is 70 (03 §3.4).
  - The fine-tuned sentence reader is Strong. Core is `tfidf` and `frozen`, and the reader embeds the sentence alone (05 §6).
  - Group F is not a model input. `FEATURES` in `ml/contracts/feature_names.py` is A + B + R + C (03 §4.4).

## 3. Packages

### Wave 0 — one agent, alone

| ID | Package | Owns | Done when |
|---|---|---|---|
| **W0** | Scaffold and contracts | root files (`requirements.txt`, `Makefile`, `.gitignore`, `.env.example`), `ml/contracts/`, `ml/runner.py`, `server/app/main.py`, `server/fixtures/`, `server/READY.md`, `tests/fixtures/`, `tests/test_contracts.py`, `legacy/` | every endpoint in 03 §11 and 05 §5 returns its fixture; `pytest tests/test_contracts.py` passes; first commit made |

W0 builds:
- Root `.venv` and `requirements.txt` (04 §4, plus `python-dotenv`, `openai`, `onnxruntime`, `tokenizers`). Old pointer build moved to `legacy/pointers_mcq/`.
- Empty package tree for every folder in 03 §12 and 05 §2, with `__init__.py`.
- `ml/contracts/classes.py`: the 19 class ids with ship name, subtitle (02 §2.3) and wrong belief (03 §3.1).
- `ml/contracts/schemas.py`: typed shapes for trace, problem, dataset row, diagnosis, intervention, exam report, knowledge state, quiz item, code item, reason row.
- `ml/contracts/feature_names.py`: the ordered feature list of 03 §4 (the agreement between the feature packages and the model package).
- `ml/contracts/subset.py`: supported and rejected C constructs (03 §2.1), interpreter limits, event and effect formats, gate messages.
- `ml/contracts/params.py`: every hand-set number of 03 §5.5, §6, §8 and 05 §4, so all packages use the same ones.
- `ml/runner.py`: `run_tests(problem, code, backend="auto")`, `trace(problem, code, test_index)`; `auto` uses the interpreter when present, else gcc.
- `server/app/main.py`: loads every router found in `server/app/routes/`; any endpoint without a live router answers from `server/fixtures/`. Nobody else needs to touch it.
- `tests/fixtures/`: 3 sample problems, 3 hand-written traces (one with `oob_read`, one `step_cap_hit`, one recursion), a 60-row sample `reasons.jsonl`, a small made-up feature matrix.

### Wave 1 — all can start as soon as W0 is merged

| ID | Package | Owns | Final check needs | Done when |
|---|---|---|---|---|
| **A1** | C interpreter (03 §2), incl. comment stripping | `ml/c_interp/` | A3 | `pytest tests/A3` green |
| **A2** | gcc stand-in: compile a learner function with a generated `main`, time-outs, results in the `run_tests` shape | `ml/oracle/` | — | `python -m ml.oracle.selfcheck` |
| **A3** | Interpreter tests written from the spec, one per row of 03 §2.2 and §2.5; expected values cross-checked with gcc where C defines the behaviour | `tests/A3/` | A2 | tests run; each names its spec row |
| **B1** | 16 main problems (03 §3.3.1) + checker | `ml/problems/main/`, `ml/problems/check.py` | A1 | `python -m ml.problems.check main` on gcc, then on the interpreter |
| **B2** | 18 DSA problems (03 §3.3.2), strings last | `ml/problems/dsa/` | A1, B1's checker | `python -m ml.problems.check dsa` |
| **B3** | 14 probes, 17 traps, ~24 exam trace items (03 §6.3, §8.3, §8.5.1) + verifier | `ml/data/probes.json`, `ml/data/items.json`, `ml/data/exam_items.json`, `ml/bayes/verify_probes.py` | A1 | verifier passes on both backends |
| **B4** | ~100 quiz items (05 §3.1) + verifier + DeepSeek drafting script | `ml/data/quiz_items.json`, `ml/items/verify_quiz.py`, `ml/items/draft_quiz_llm.py` | A1 (for `next_state`), key | `python -m ml.items.verify_quiz`; targets of 05 §3.1 met |
| **C1** | Main-class operators and their predicates (03 §3.5.1, M + AMB + OTHER + near-miss), as syntax-tree changes | `ml/generate/ops_main.py`, `ml/generate/predicates_main.py` | A1, B1 | every operator yields a parsing mutant that fails a test on the sample problems |
| **C2** | DSA operators and predicates (D01–D08 + DSA surfaces of M-classes) | `ml/generate/ops_dsa.py`, `ml/generate/predicates_dsa.py` | A1, B2 | same |
| **C3** | Generator pipeline: registry, augment, verify, ambiguity, caps, card (03 §3.5.2–3.9) | rest of `ml/generate/` | A1, B1, B2, C1, C2 | `make data` → ≥ 4,000 rows, every class ≥ 80, only {M01, M08} label clashes |
| **F1** | AST features + masking preconditions (03 §4.1) | `ml/features/ast_feats.py` | — | features match hand-computed values on 20 snippets |
| **F2** | Trace and output-relation features (03 §4.2, §4.3) | `ml/features/trace_feats.py`, `ml/features/relation_feats.py` | A1 | values correct on fixture traces, then on live ones |
| **M1** | Model: train, calibrate, novelty, mask, decide, evidence, baselines (03 §5) | `ml/model/` | C3, F3 | runs end to end on the made-up matrix; then `make train` prints grouped-CV scores |
| **D1** | Bayes layer, likelihoods, probe choice (03 §6) + the sentence-reader update (05 §4) | `ml/bayes/` except `verify_probes.py` | B3 | unit tests: T1 posterior reaches 0.85 after probes |
| **D2** | Knowledge model + state machine (03 §8.1–8.4) + new item types (05 §4) | `ml/learner/knowledge.py`, `ml/learner/state_machine.py` | — | unit tests for every transition and every row of 05 §4 |
| **D3** | Exam selector, observation model, report (03 §8.5) + quiz selector (05 §5) | `ml/exam/`, `ml/quiz/` | B2, B3, B4 | scripted exam with a fake diagnoser produces a report with a reason per item |
| **R1** | Verified fixer, counterexample search, fix-probe features (03 §7.2, §7.3, §4.4) | `ml/learner/fixer.py`, `ml/learner/counterexample.py`, `ml/features/fix_feats.py` | A1, C3 | fixer repairs ≥ 80% of held-out mutants |
| **R2** | Timeline, call-stack, window and memory builders; intervention policy (03 §7.1, §7.4) | `ml/learner/timeline.py`, `ml/learner/interventions.py` | A1, R1 | packages match the 03 §11.2 shape on fixture traces |
| **T1** | DeepSeek client + reason sentences (05 §7) | `ml/text/llm_client.py`, `ml/text/gen_reasons.py`, `ml/text/prompts/`, `ml/data/reasons.jsonl`, `ml/data/llm_raw/` | key; B4 for the second pass | ≥ 150 kept sentences per label, ≥ 6 contexts per class, filters applied, rerun costs nothing |
| **T2** | Sentence reader: three readers, cloud training script, ONNX export, serving (05 §6) | rest of `ml/text/` | T1; one Kaggle or Colab run by a human | `read()` works with `tfidf` and `frozen` on the 60-row sample (sentence only, code used for masking). `biencoder` is Strong: the same call uses it when `ml/artifacts/` has the file, and skips it when not |
| **X1** | Public data: fetch ITSP, Codeflaws, IntroClass, Mohler; label by diff (03 §3.4 protocol X) | `ml/external/`, `ml/data/itsp_slice.jsonl` | a human hand-check of labels | slices load; counts per class printed |
| **G1** | Gate (03 §3.7.1) | `server/app/gate.py` | — | the gate fixtures of 03 §13 K6 behave as specified |
| **S1** | SQLite store + learner routes (`/learner`, seed, `/reassess`, `/intervene`) | `server/app/store.py`, `server/app/routes/learner.py` | D2, R1, R2 | curl script: seed learner → reassess → STABLE |

### Wave 2 — start when the named packages are merged

| ID | Package | Owns | Starts after | Done when |
|---|---|---|---|---|
| **F3** | Feature assembly + ~100 evidence sentences (03 §5.6) | `ml/features/extract.py`, `ml/features/evidence_templates.json` | F1, F2 | one call returns the full ordered row for a program |
| **C4** | Build `complete_snippet`, `fix_bug`, `debug_line` items (05 §3.2) | `ml/items/build_code_items.py`, `ml/data/code_items.json` | C3 | `--check` passes; targets of 05 §3.2 met |
| **N1** | EDA (03 §10) | `ml/eda/` | C3 | `docs/eda.md` written; red rules listed |
| **S2** | Diagnosis pipeline + routes (`/run`, `/attempt` incl. `code_item_id`, `/probe/answer`, `/lab/diagnose`) | `server/app/pipeline.py`, `server/app/routes/attempt.py` | G1, D1, F3, M1 | hard-twin code returns `ambiguous`; after two probes ≥ 0.85; under 300 ms |
| **S3** | Exam, quiz, code-item, reason and metrics routes | `server/app/routes/exam.py`, `quiz.py`, `code_items.py`, `reason.py`, `metrics.py` | D3, T2, C4, S1 | scripted learner finishes an exam; quiz answer on a collision item returns `ask_reason: true` and `/reason` settles it |
| **V1** | Simulated learners: resolution and exam (03 §9.3, §9.3b; E10, E14) | `ml/eval/sim_learners.py`, `sim_exam.py`, `e10_*.py`, `e14_*.py` | D2, D3 | both tables produced with intervals |
| **E-a** | Diagnoser experiments E1, E2, E4, E7, E8, E9, E11, E12, E13 | those `ml/eval/eNN_*.py` | M1 | each writes its card |
| **E-b** | Retrain experiments E3, E5, E6, E15 | those `ml/eval/eNN_*.py` | M1, D1 | each writes its card |
| **E-c** | Sentence reader and item-bank experiments E16, E17, E18 (05 §9) | `ml/eval/e16_*.py`, `e17_*.py`, `e18_*.py` | T2, B4, C4 | each writes its card |

### Wave 3 — one agent, alone

| ID | Package | Owns | Done when |
|---|---|---|---|
| **I1** | Integration: run `make data eda train eval fixtures` end to end, fix breaks between packages, read every `notes/*.md`, assemble `metrics.json`, write the cards | `ml/eval/run_all.py`, `scripts/`, `docs/`, `server/READY.md` | acceptance criteria of 01 §5 pass; app works with the backend stopped (fixtures dumped from the live API) |

I1 is the only agent allowed to edit other packages' files, and only for small fixes.

## 4. Suggested lanes

Wave 1 has 23 packages that can all run at once. With fewer agents, give each agent one lane and let it work down the lane:

| Lane | Packages in order |
|---|---|
| 1 Interpreter | A1 |
| 2 Checking | A2 → A3 → G1 → X1 |
| 3 Problems | B1 → B2 |
| 4 Items | B3 → B4 → C4 |
| 5 Generator | C1 → C2 → C3 → N1 |
| 6 Features and model | F1 → F2 → F3 → M1 → E-a → E-b |
| 7 Learner logic | D1 → D2 → D3 → V1 |
| 8 Repair | R1 → R2 → S1 |
| 9 Sentences | T1 → T2 → E-c |
| 10 Server | (after lanes 6–9) S2 → S3 |

More agents: split a lane at any arrow whose right side does not list the left side under "Starts after" or "Final check needs" (for example C1 and C2, B1 and B2, D1 and D2, F1 and F2).

Merge order for I1, and the order to prefer when agents finish out of sequence. The first list is what makes the Loops safe point (04 §6) real; nothing after it blocks that demo.

1. Must merge first, in this order: **A2, A1, B1, C1, C3, F1, F2, F3, M1, D1, D2, R1, R2, G1, S1, S2.**
2. Then everything else in wave 1, in any order, then wave 2, then I1.

W0 is already merged. A package later in the list may start before an earlier one (06 §3), but I1 does not treat the loop as closed until this sequence is in.

## 5. Things only a human can do

| Task | Needed by |
|---|---|
| Put the DeepSeek key and model name in `.env` | T1, B4 |
| Run `ml/text/kaggle_train.py` on Kaggle or Colab, download the result into `ml/artifacts/` | T2, E-c |
| ~~Hand-write R-team (30 programs)~~ Not done by hand: replaced by 58 LLM-written programs in `ml/data/realistic_llm.jsonl` (source `R-llm`, see `notes/T1.md`). E-a reports them as an LLM-written stand-in, never as hand-written. Arnav's R-blind (40) is still open | E-a |
| Skim the quiz items marked `manual` and a sample of the DeepSeek-drafted ones | B4 |
| Hand-check the auto-labels on the ITSP slice | X1 |
| ~~Get 50 or more classmates to type a "why" sentence~~ Not collected: replaced by 270 role-played sentences in `ml/data/reasons_persona.jsonl` (see `notes/T1.md`). E-c reports them as LLM-written, never as real student text | E-c |

## 6. Training runs: time and load

Measured on the ML laptop where marked; the rest are estimates.

| Run | Where | Time | Load |
|---|---|---|---|
| Build the dataset (`make data`) | laptop CPU | 10–15 min in one process, 2–3 min across 8 | under 2 GB RAM |
| Extract features | laptop CPU | 8–17 min in one process, a few minutes across 8 | under 2 GB RAM |
| Train the code model once | laptop CPU, `num_threads=4` | 3.5 s (measured) | under 1 GB RAM |
| All code-model runs in the evaluation (about 120) | laptop CPU | about 7 min | same |
| Word-count sentence reader | laptop CPU | seconds | negligible |
| Frozen sentence reader (embed 3k sentences, fit) | laptop CPU | 1–2 min | 420 MB model file; about 1 GB RAM |
| Fine-tuned sentence reader, one run | Kaggle / Colab T4, or the RTX 4050 | 3–10 min | 3–4 GB GPU memory |
| Fine-tuned reader, leave-one-class-out (17 runs) | Kaggle / Colab | 1–2 hours | same; free tier is enough |
| DeepSeek generation (sentences + quiz drafts) | DeepSeek's servers | 10–30 min of API calls | under $2 |
| Simulations (E10, E14) | laptop CPU | a few minutes | negligible |

Disk on the laptop: about 1 GB for everything above, plus about 0.5 GB for the sentence reader's model file and ONNX runtime. GPU PyTorch (5–6 GB) is needed only if the fine-tuning is done locally.

## 7. Prompt to start an agent

```
You are building package <ID> of the Re:Learn ML backend.
Read ml_plan/03_ML_IMPLEMENTATION_PLAN_v3.md, ml_plan/05_QUESTION_TYPES_AND_SENTENCE_READER.md,
ml_plan/06_AGENT_WORK_PACKAGES.md (your row in §3 and all of §2), ml/contracts/, and notes/W0.md
(decisions made where the plans are silent).
Build only the paths in your "Owns" cell. Follow every rule in 06 §2.
Use ml/runner.py with the gcc backend and tests/fixtures/ for anything that is not merged yet.
Finish when your "Done when" check passes; then write notes/<ID>.md.
```
