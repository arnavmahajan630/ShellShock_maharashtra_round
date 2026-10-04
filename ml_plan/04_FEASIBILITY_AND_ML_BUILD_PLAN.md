# 04 — Feasibility check and ML build plan (for the v3 plans)

Checked on 2026-10-04 against `01_SCOPE_v3.md`, `02_DESIGN_SCOPE_v3.md` and `03_ML_IMPLEMENTATION_PLAN_v3.md`, on the ML laptop (i5-13420H, 8 cores, 16 GB RAM, RTX 4050 6 GB, Windows 11). Numbers marked **measured** were run on this laptop today; the rest are estimates.

## 1. Verdict

- **Hardware is not a limit.** Everything in v3 runs on the laptop's CPU. Nothing needs the GPU and nothing needs the cloud.
- **Disk: about 1 GB for the ML side, about 1.5 GB if the frontend also runs on this laptop.** About 25 GB is free, so it fits.
- **Time is the limit.** The ML column of the 12-hour plan adds up to roughly 15–17 hours of work by my estimate (§5). The build order in §6 is arranged so the required parts work by hour 6.5 and later parts can be cut without breaking the demo.

## 2. Disk space

| Item | Size | Basis |
|---|---|---|
| Python environment: pycparser, lightgbm, scikit-learn, scipy, numpy, pandas, matplotlib, pytest, fastapi, uvicorn | 420 MB | measured |
| Dataset (~9k rows), feature cache, splits | under 100 MB | estimate |
| Saved model | 13 MB each; keep only the final one | measured at 400 rounds, 19 classes |
| Plots, `metrics.json`, fixtures, SQLite | under 50 MB | estimate |
| ITSP student-code clone (Strong tier only) | under 100 MB | estimate |
| **ML total** | **about 0.7–1 GB** | |
| Frontend `node_modules` (Vite, React, Tailwind, Framer Motion, CodeMirror) + Kenney art + fonts | 0.4–0.6 GB | estimate; only if run on this laptop |

The C: drive is 95% full (25.3 GB free of 476 GB at the last check; it was 17.8 GB earlier in the day). That is enough, but if space is needed:

| Folder | Size | Safe to clear? |
|---|---|---|
| `AppData\Local\pip\cache` | 8.3 GB | Yes: `pip cache purge` (do it **after** the install in §4, which reuses the cache) |
| `.ollama` | 8.0 GB | Only if you don't need those local models; v3 uses none |
| `AppData\Local\Docker` | 101.5 GB | Only if you don't need those images; v3 does not use Docker |

Not needed by v3, so do not install: PyTorch (several GB with CUDA), any code language model, UMAP (optional in the EDA; PCA is enough), SHAP (LightGBM's `pred_contrib` already gives the contributions).

## 3. What runs where

| Job | Where | Time on this laptop | Basis |
|---|---|---|---|
| One LightGBM training run (10k rows × 110 features, 19 classes, 400 rounds) | CPU | 3.5 s with 4 threads; 5.2 s with 12 | measured on a matrix shaped like ours (mostly 0/1 flags) |
| All training runs in the eval suite: 12-config grid × 5 folds, plus E3, E6 (17 retrains), E9, E15, about 120 runs | CPU | about 7 minutes | 120 × measured run time |
| Building the dataset (every candidate program is run through the interpreter) | CPU | 10–15 min in one process, 2–3 min across 8 | estimate from v3's 30 ms per test suite |
| Feature extraction, including "would this fix repair it" features | CPU | the slowest job: 8–17 min in one process, a few minutes across 8 | estimate from v3's 250 ms cap, ~2–4k distinct programs |
| Serving one diagnosis | CPU | model 3 ms, with explanation weights 23 ms | measured; the interpreter and fixers use the rest of the 300 ms budget |
| Memory | | under 2 GB of 16 GB | estimate |

- **GPU:** unused. The LightGBM build from pip has no GPU support (tested: "GPU Tree Learner was not enabled in this build"), and at 10k rows a GPU would not be faster.
- **Cloud:** none. No Colab, no API keys. v3 has no LLM, so the DeepSeek key is unused (only the optional LLM baseline row in E8 would use it).
- **Internet is needed only for:** `pip install`, `npm install`, the Kenney art download, Google Fonts (bundle the font files locally so the offline demo works), and the ITSP clone.
- Set `num_threads=4` in the LightGBM params: 12 threads was slower than 4 on this CPU.
- v3 §5.2 says the 12-config grid takes "well under a minute". At the measured rate it is 3–4 minutes. Harmless, but the eval slot should allow for it.

## 4. Setup on this machine

```powershell
cd C:\Users\harsh\hacks\bitnbuild
python -m venv .venv
.venv\Scripts\python -m pip install pycparser lightgbm scikit-learn scipy numpy pandas matplotlib pytest fastapi uvicorn
```

The wheels are already in the pip cache from today's size check, so this takes about a minute. Put the same list in a root `requirements.txt`, and delete `server\.venv` (it only has fastapi and uvicorn).

`make` is not installed, so `make demo` will not run as written. `mingw32-make` is installed (MSYS2) and reads the same Makefile: use `mingw32-make demo`, or run `choco install make` once.

## 5. Problems found in the v3 plans

| # | Finding | What to do |
|---|---|---|
| 1 | **pycparser rejects comments** (tested on 3.0: "Comments are not supported"). v3 §2.5 does not list comment stripping, yet the `comments` augmentation and the realistic-set brief both add comments. | In `preprocess.py`, replace comments with spaces (keep newlines) so line numbers stay correct. Do this first. |
| 2 | **v3 rejects pointers** (gate G4). Everything already built is about pointers: 24 MCQs, misconceptions M1–M6, `engine.py`. | Move `server/main.py`, `engine.py`, `simulate.py`, `verify_items.py`, `server/data/` to `legacy/pointers_mcq/`. Mark the root `PLAN.md` as superseded. The old `M1…M6` ids are unrelated to v3's `M01…M10`. |
| 3 | **02 is written for a Vite web app, but the team is keeping Expo** (decided 2026-10-04). CodeMirror is a browser editor and does not run in React Native without a WebView; Tailwind and Framer Motion are browser libraries too. | Arnav swaps those three for React Native equivalents. The API and the ML side do not change. |
| 4 | `00_PROJECT_CONTEXT.md` is referenced by all three files but is not in `plans/` or on this laptop. | Get it from Arnav. The ML side is not blocked: the contracts are all in 03. |
| 5 | The "~9k rows" are about 2k distinct programs. Renaming, brace style, comments and spacing do not change any AST or trace feature, so the model sees those rows as copies. v3 admits this ("effective N"). | No action needed for training speed. Always quote the effective N beside the row count. |
| 6 | Operators that edit source text by pycparser coordinates are fiddly: a `BinaryOp`'s coordinate is where its left side starts, not where the operator is. | For the dataset, change the syntax tree and print it with `pycparser.c_generator`. Keep text edits only in the fixer, where the learner sees a diff of their own code. No contract changes. |
| 7 | The ML hours do not fit (table below). | Follow the order in §6 and cut from the end. |

Reusable from the pointer build: the FastAPI + SQLite pattern, the probability update in `engine.py` (same maths as v3 §6.1), the simulated-learner pattern in `simulate.py` (for E10 and E14), and `gcc` (15.2, installed) as a second opinion on the interpreter: compile each correct variant and compare outputs.

### ML time: v3 budget against my estimate

One person driving a coding agent. These are judgments, not measurements.

| Card | v3 | Estimate | Why |
|---|---|---|---|
| K0 contracts + fixtures | 0:30 | 0:30 | |
| K1 interpreter | 1:30 | 2:00 | strings and recursion tracing on top of the core |
| K2 34 problems | 1:15 | 1:30 | 3 variants and 4–6 tests each, all verified |
| K3 generator | 0:50 | 2:00–2:30 | about 55 operators, each with a predicate; the largest underestimate |
| K4 EDA | 0:10 | 0:15 | |
| K5 features + model | 0:45 | 1:30–2:00 | about 110 features and about 100 evidence sentences |
| K6 gate | 0:15 | 0:30 | |
| K7 fixer, counterexample, builders | 1:00 | 1:30 | 17 fixers |
| K8 Bayes + probes | 0:45 | 0:45 | |
| K9 knowledge model + exam | 1:00 | 1:30–2:00 | two subsystems in one slot |
| K10 hand-write 60 programs | 0:45 | 1:30 | 45 seconds per program is not realistic |
| K11 eval E1–E15 | 1:15 | 2:00 | |
| **Total** | **10:00** | **15–17 h** | |

## 6. Build order (ML side)

Rule: build the Loops planet end to end with a real model first, then widen. The API, class ids and JSON shapes in 03 §11 do not change, so the frontend is unaffected by the order. Mark each endpoint in `server/READY.md` when it goes live (02 §11, step 7).

| Time | Build | Done when |
|---|---|---|
| **0:00–0:30** Setup | §4 setup; move the pointer build to `legacy/`; FastAPI serving fixture JSON for every endpoint in 03 §11; first commit and push | Arnav gets valid JSON from every endpoint |
| **0:30–4:30** Loops slice | Interpreter core: ints, floats, arrays, loops, ifs, functions, comment stripping, all main-game events (no strings, no call-stack trace yet) · the 16 main problems, checked against gcc · M-class, AMB, OTHER and near-miss operators + verifier + ambiguity pass · feature groups A, B, R · LightGBM, temperature, masking, novelty, decision, evidence · gate · `/run`, `/attempt`, `/lab/diagnose` | `for(i=0;i<=n;i++) sum+=a[i];` on P03 comes back `ambiguous` between M01 and M08; `lo < x < hi` comes back `novel`; grouped-CV score printed |
| **4:30–6:30** Close the loop | Bayes layer + 8 main probes + probe choice by information gain · fixers for M-classes + counterexample + timeline · `/probe/answer`, `/intervene` · knowledge model, state machine, 9 trap items · `/reassess`, `/learner`, seed learner | A curl script goes play → fail → probe → 0.85 → intervention → trap + transfer → STABLE |
| | **Safe point.** All seven requirements of the problem statement can now be shown on the main game. Everything below adds to it. | |
| **6:30–9:00** DSA | Interpreter: recursion `call`/`ret`, depth cap, array write and compare effects · 15 DSA problems (all except strings Q13–Q15) · D01–D07 operators, features, fixers · retrain one shared model · exam with the fixed blueprint first (03 §8.5.7), then the adaptive selector · debrief report · 6 DSA probes · about 12 exam trace items | A scripted learner finishes an exam and the report has one NEW and one HELD finding; the four Bug Lab DSA presets diagnose correctly |
| | Strings (Q13–Q15, D08) only if DSA finishes before 9:00. v3's own cut order drops D08 first. | |
| **9:00–10:45** Evaluation | Hand-write R-team (30, not 60) and collect Arnav's R-blind (40) · run in this order and stop when time runs out: E1, E2, E4, E5, E10, E14, E15, E12, then E6, E7, E8, E9, E3, E11 · plots, `metrics.json`, `/metrics` | `/metrics` returns a card, with slice, n and caveat, for every experiment that ran |
| **10:45–11:15** | Dump fixtures from the live API; dataset card and model card | Frontend works with the backend stopped |
| **11:15–12:00** | Two rehearsals, backup recording | |

### Cut lines

- **Behind at 4:30:** serve the rules baseline behind `/attempt` (03 §13 fallback) and keep going; train the model when the features are ready.
- **Behind at 6:30:** drop group F features and the two-bug check. Keep the fixer for the intervention screen.
- **Behind at 9:00:** ship the exam with the fixed blueprint (same API) and skip E14.
- **Never cut** (from 01 §8): interpreter, gate, hard twin + probe, verified fixer, knowledge model + trap, exam start → finish → debrief, Lab Report with honest caveats, fixtures.

### Decisions (made 2026-10-04)

1. Expo stays (§5, row 3).
2. R-team is 30 items, not 60, making the realistic set 70 instead of 100.
3. Strings (D08, Q13–Q15, SignalTiles) are the first content cut. If cut, Arnav does not need to build SignalTiles.
4. All eight question types are in scope: see `05_QUESTION_TYPES_AND_SENTENCE_READER.md`.

The build order in this section is replaced by the parallel work packages in `06_AGENT_WORK_PACKAGES.md`.
