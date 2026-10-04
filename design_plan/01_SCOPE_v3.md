# 01 — PROJECT SCOPE v3 (12-hour build, 3 planets + Deep Space Trials)

> **Replaces `01_SCOPE.md` and `01_SCOPE_v2.md`.** Source of truth for coding agents together with `02_DESIGN_SCOPE_v3.md` and `03_ML_IMPLEMENTATION_PLAN_v3.md`. If `00_PROJECT_CONTEXT.md` disagrees, the v3 files win (patches in §10).
>
> Build window: **12 hours, 2 people, AI-assisted coding.** **FE** = Arnav (frontend + game), **ML** = ML teammate (interpreter, data, model, API). "Both" = sync tasks.
>
> **What's new in v3 (judges' feedback):** a **DSA "Test Yourself" level** — *Deep Space Trials* — with an **adaptive exam** over Searching, Sorting, Array techniques, Strings and Recursion, **8 new DSA misconception classes**, a Debrief report and a DSA Practice Arena. It is **open from the start** (recommended after the 3 planets, because the exam then rechecks your earlier repairs). The main game grows from 1 to **3 playable planets**.

---

## 0. Gap analysis (v1 → v2 fixes, still in force) + v3 additions

| # | Gap | Fix |
|---|---|---|
| G1 | Simulated prediction features leak the label | LightGBM sees code + trace only; predictions/probes go through a Bayesian evidence layer with documented likelihoods (ML §6) |
| G2 | v1 "twins" were mostly separable by one flag | HARD twin (code-identical, soft labels, needs a probe) + STRUCTURAL twins; v3 adds 4 DSA twin sets |
| G3 | Core input was tap-to-edit only | Free typing is Core (missions, transfer, exam, Bug Lab) |
| G4 | "Posterior < 0.15" check collapses into "tests pass" | 2-state knowledge model; trap items (no execution) carry the weight |
| G5 | "Resolved" before the ghost recheck | States `ACTIVE → TREATING → PROBATION → STABLE → MASTERED`, `RELAPSED` |
| G6 | Missing verified fix, evidence text, try-your-own-bug page | Verified minimal fixer, SHAP-to-sentence evidence, Bug Lab |
| G7 | History as a simulated feature | Capped prior in the Bayes layer |
| G8 | No beginner cohort for a human study | Blind-authored realistic set (now 100 items incl. DSA) + ITSP real-student slice |
| G9 | One held-out class | LOCO over all 17 classes + 2 never-trained classes |
| G10 | No outlier/gate handling | Gate before the model + kNN novelty |
| G11 | Weeks of scope | 12-hour plan with checkpoints and cut lines (§6, §8) |
| G12 | LLM personas + LLM baseline | Dropped; no LLM in the product |
| **G13 (v3)** | **No DSA** (judges' feedback) | Deep Space Trials: 18 basic DSA problems, 8 DSA misconception classes, adaptive exam, debrief, practice arena |
| **G14 (v3)** | "Resolved" was only ever checked inside the same planet | **The exam is the final ghost return:** main-game misconceptions are re-tested on DSA surfaces (e.g. off-by-one inside bubble sort) → MASTERED or RELAPSED |
| **G15 (v3)** | Adaptivity was only in *teaching*, not in *assessment* | Exam items are chosen by **expected information gain** over the learner's misconception states + topic coverage + target difficulty; every pick is explained in the debrief ("why this question") |

### Why no LLM
Not needed. Everything (diagnosis, probes, adaptive selection, interventions, resolution) is deterministic, offline and explainable from AST + execution evidence. "No LLM in the loop" is a pitch point.

---

## 1. Scope philosophy

1. Judges score the ML loop: **diagnose → (probe) → intervene → prove resolution → track → evaluate**, now **plus assess (DSA exam) → debrief → practise**.
2. **Three planets playable** (Conditions → Loops → Arrays). Loops stays the deepest and is the main demo planet. Variables and Functions are teasers; their exercises appear as transfer tasks.
3. **Deep Space Trials** is a separate mode at the edge of the galaxy, **unlocked from the start**. A soft banner recommends clearing the 3 planets first.
4. Nothing becomes a model feature unless execution or the AST can verify it.
5. Honesty is a feature: every Lab Report number states its slice and caveat.
6. The demo survives backend or Wi-Fi failure (fixtures, everything local).

---

## 2. Locked tech decisions

| Decision | Choice |
|---|---|
| Frontend | Vite + React + TypeScript + Tailwind + Framer Motion (keep Next.js only if already scaffolded) |
| Arena + DSA visualisers | React + CSS sprites / `<canvas>`; no Phaser. 3 DSA visualisers: **ArrayBars**, **SignalTiles**, **WarpStack** |
| Editor | CodeMirror 6 (`@uiw/react-codemirror`, C mode) |
| Art | Kenney.nl CC0 packs + 1 droid sprite + Google Fonts (Press Start 2P, VT323, JetBrains Mono) |
| Backend | FastAPI + SQLite, single process, local for the demo |
| C execution | Own pycparser tree-walking interpreter; v3 adds **char arrays/strings, call-stack tracing, in-place array checks** |
| ML | LightGBM (19 outputs) + Bayes evidence layer + knowledge model + **adaptive exam selector** |
| LLM | None |
| Audio | ≤ 6 SFX |
| Run | `make demo` → backend `:8000`, frontend `:5173`, seeded demo learner (3 planets cleared) |

---

## 3. Tiers

### CORE

| # | Feature | Owner | Cx |
|---|---|---|---|
| C1 | Contracts frozen (trace, events, diagnosis, exam JSON, endpoints) + fixtures | Both | S |
| C2 | Interpreter + events + world builtins; **v3:** strings, recursion trace, array-arg checks | ML | L |
| C3 | Problem bank: **16 main** (Conditions 4, Loops 6, Arrays 4, Variables 1, Functions 1) + **18 DSA** (Searching 5, Sorting 3, Array techniques 4, Strings 3, Recursion 3) | ML | L |
| C4 | Mutation generator (M + D operators) + verifier + soft labels → dataset (~9k rows) | ML | M |
| C5 | Features (incl. DSA features) + LightGBM + grouped CV + calibration + gate + novelty + class masking | ML | M |
| C6 | Bayes layer + **14 probes** + EIG | ML | M |
| C7 | Knowledge model + `/reassess` + state machine | ML | M |
| C8 | Verified fixer (M + D) + counterexample + timeline / call-stack / window builders | ML | M |
| C9 | **Adaptive exam engine** (`/exam/*`), debrief report, "why this item" log | ML | M |
| C10 | Eval suite E1–E15 → `metrics.json` | ML | M |
| C11 | Galaxy Map (3 live planets + open Trials wormhole) + 3 Planet Paths | FE | M |
| C12 | Mission screen (Predict → Code → Run → Arena) | FE | L |
| C13 | Diagnosis modal + Probe | FE | M |
| C14 | Intervention (timeline, memory strip, value meter, **call stack**, **window strip**) | FE | M |
| C15 | Transfer + Trap + Verdict; Mental Model (Core + Deep Space constellations) | FE | M |
| C16 | **Trials Lobby + Exam Runner + DSA visualisers** | FE | L |
| C17 | **Debrief report** (with deferred probes) | FE | M |
| C18 | Lab Report (incl. DSA + adaptive-exam tabs) + Bug Lab (DSA presets) | FE | M |
| C19 | HUD, demo learner, judge mode (`?judge=1`: all planets open + footer links), fixtures fallback | FE | S |

### STRONG (only after the Core demo path works end to end)

| # | Feature |
|---|---|
| S1 | Ghost Return nodes on planet paths (engine is Core; the exam also acts as a ghost return) |
| S2 | Quickfire MCQ nodes |
| S3 | Orbit hint ladder (missions only; never in the exam) |
| S4 | M09 PASS_BY_VALUE + T5 twin |
| S5 | ITSP real-student slice (60-min time-box) |
| S6 | Two-bug UI |
| S7 | Modality switching on Not Yet |
| S8 | Settings + reduced motion |
| S9 | **DSA Practice Arena** as a full screen (Core fallback: Debrief "Train this" opens Bug Lab with the problem preloaded) |
| S10 | Exam history (retake, compare reports) |

### STRETCH (roadmap slide)
Teacher heatmap · Variables & Functions planets · pointers / linked lists · stacks & queues · 2-D arrays · boss fight · Python frontend · real classroom study · fitted knowledge-tracing parameters · IRT-calibrated exam difficulties from real data.

---

## 4. Content scope

| Item | v2 | **v3** |
|---|---|---|
| Playable planets | Loops | **Conditions, Loops, Arrays** |
| Trained classes | 9 | **17** = M01–M08, M10 + **D01–D08** (+ CORRECT, OTHER) |
| Unseen classes | U1, U2 | U1, U2 (+ LOCO over all 17) |
| Problems | 14 | **34** (16 main + 18 DSA); 6 held out |
| Probes | 8 | **14** (+ trap items: 17) |
| Exam | — | 10 items (5 coding + 5 trace), ~25 min; demo mode 3 items |
| Realistic test set | 60 | **100** (R-blind 40 by FE, R-team 60 by ML; ≥ 40 DSA) |
| Interventions | 3 modalities | 5 (+ `call_stack`, `window_strip`) |

### 4.1 DSA misconception classes (summary; full spec in ML §3.1)
| ID | Name | Typical wrong C |
|---|---|---|
| D01 | SEARCH_EARLY_EXIT | `if (a[i]==x) return i; else return -1;` inside the loop; `else found = 0;` |
| D02 | BSEARCH_NO_SHRINK | `low = mid;` / `high = mid;` |
| D03 | SWAP_OVERWRITE | `a[i] = a[j]; a[j] = a[i];` (no temp) |
| D04 | SINGLE_PASS_SORT | only one pass of bubble/selection |
| D05 | MISSING_BASE_CASE | no / unreachable base case |
| D06 | RECURSION_NO_SHRINK | `f(n)` or `f(n+1)` in the recursive call |
| D07 | RECURSIVE_RESULT_DISCARDED | `f(n-1); return n;` |
| D08 | STRING_EQ_COMPARE | `s[i] == "a"`, `s == t` |

### 4.2 DSA problems (18; full spec in ML §3.3)
Searching: linear search · count occurrences · binary search · contains (flag) · count guesses (halving) — Sorting: bubble sort · selection sort · is-sorted — Array techniques: reverse in place · second largest · rotate left by one · pair-sum (two pointers) — Strings: length without strlen · count vowels · palindrome — Recursion: factorial · sum of digits · recursive array sum.

---

## 5. Acceptance criteria (done at T+11:15)

1. **Loops end-to-end** on the real backend: play → fail → diagnosis → intervention → transfer + trap → STABLE.
2. **Hard-twin demo** (`for(i=0;i<=n;i++) sum+=a[i];`) → ambiguous → probe → ≥ 0.85.
3. **Conditions planet:** `if (lo < x < hi)` → "Unknown anomaly" (never-trained class).
4. **Arrays planet** playable end-to-end (fixtures acceptable for one node).
5. **Trials:** open from the start; adaptive exam runs (10 items, or 3 in demo mode); each next item comes with a logged reason; Debrief shows sector scores, DSA findings, deferred probes, and at least one **main-game misconception promoted to MASTERED or flagged RELAPSED** on a DSA surface.
6. DSA diagnosis works for at least: swap without temp, `low = mid`, missing base case, early `return -1` (Bug Lab presets).
7. **Gate:** empty / Python / hard-coded / `strlen` in the no-strlen task → correct messages.
8. **Lab Report** shows E1–E15, incl. DSA per-class results, cross-domain transfer (E15) and adaptive-vs-fixed exam (E14).
9. `make demo` works; backend killed → fixtures with an "OFFLINE REPLAY" badge.

---

## 6. Twelve-hour build plan

> T+hh:mm from start. Checkpoints are hard; take the fallback immediately if one fails.

| Time | ML teammate | FE (Arnav) | Checkpoint / fallback |
|---|---|---|---|
| 0:00–0:30 | **Both:** freeze contracts (ML §11 incl. `/exam/*`), fixtures for: T1 ambiguous attempt, M07 confident, CORRECT, D03 confident, one exam flow (3 items + report), metrics stub; repo + `make demo` stub | same | Contracts committed |
| 0:30–2:00 | Interpreter core (§2) + strings + recursion trace + array-arg checks + unit tests | App shell, tokens, kit components, Galaxy Map (3 live planets + open Trials wormhole), Planet Paths on fixtures | **CP1 @2:00:** all main correct variants run. *Fallback:* defer strings to CP3 |
| 2:00–3:15 | Problem bank: 16 main + 18 DSA (variants + tests), `problems.check` | Mission screen: CodeMirror, Predict, Run, Arena (cannon, drone bay, door worlds) | **CP2 @3:15:** all 34 problems' variants pass. *Fallback:* drop Q12, Q18, P17 |
| 3:15–4:15 | Operators (M + D), verifier, ambiguity, caps → `dataset.jsonl`; EDA v1 | Diagnosis modal (all states) + Probe card | **CP3 @4:15:** ≥ 4,000 rows, every class ≥ 80 |
| 4:15–5:15 | Features (A/B/R/C + DSA) + LightGBM + calibration + gate + masking; `/attempt`, `/lab/diagnose`, `/run` live | Intervention screen: timeline, memory strip, value meter | **CP4 @5:15:** grouped-CV printed; `/attempt` < 300 ms. *Fallback:* rules baseline behind the API |
| 5:15–6:15 | Fixer (M + D) + group F + counterexample + timeline/call-stack/window builders; `/intervene` | Transfer + Trap + Verdict; Mental Model (2 constellations) | **CP5 @6:15:** fixer coverage ≥ 80% on held-out mutants |
| 6:15–7:00 | Bayes layer + 14 probes + EIG + `verify_probes` | **FE writes R-blind (40 items, 25 main + 15 DSA) without having seen `operators.py`** (40 min) | **CP6 @7:00:** T1 + T7 curl scripts pass |
| 7:00–8:00 | Knowledge model + `/reassess` + `/learner` + seed; **exam engine** `/exam/start|answer|finish|probe` | Trials Lobby + Exam Runner + ArrayBars / SignalTiles / WarpStack | **CP7 @8:00:** scripted learner completes an exam via curl |
| 8:00–8:45 | ML writes R-team (60 items) | Debrief report + deferred probes; intervention `call_stack` + `window_strip` widgets | — |
| 8:45–10:00 | Eval E1–E15, sim learners (resolution + exam), plots, `metrics.json`; realistic sets run once | Lab Report (all tabs) + Bug Lab (main + DSA presets); Practice Arena if ahead (else Debrief → Bug Lab) | **CP8 @10:00:** `/metrics` complete |
| 10:00–10:45 | ITSP slice (Strong) **or** buffer for failed checkpoints | Arrays + Conditions path polish; judge mode; reduced motion | — |
| 10:45–11:15 | **Both:** integration, fixtures dumped from the live API, demo seed | | Feature freeze @11:15 |
| 11:15–12:00 | **Both:** 2 rehearsals, backup screen recording, README + model/dataset cards | | Ship |

---

## 7. Demo script (≈ 5 min)

1. **Hook (15 s):** galaxy: Conditions and Arrays stars steady, Loops star flickering, the Trials wormhole glowing at the edge.
2. **Loops mission (60 s):** predict → type `i <= n` → 11 pulls, last one into the void → diagnosis M01 0.46 vs M08 0.44: "Code alone can't separate these" → probe "last valid index of `int a[5]`?" → answer 5 → M08 0.9.
3. **Intervention → trap → transfer → STABLE (45 s).**
4. **Conditions in Bug Lab (15 s):** `lo < x < hi` → Unknown anomaly (never trained on this).
5. **Deep Space Trials (75 s):** open the wormhole → adaptive exam (demo mode, 3 items). Item 1 is bubble sort: the player swaps without a temp → bars show a duplicated crate. Item 2 is chosen "because your record suggests boundary bugs; this checks them inside a sort" (shown in the debrief). Item 3: a recursion trace question.
6. **Debrief (40 s):** sector scores; *Swap Overwrite* found; Boundary Drift **held** → MASTERED (fixed in Loops, still fixed inside a sort); deferred probe answered → finding sharpened; "Train this" → practice.
7. **Lab Report (40 s):** realistic-set headline, LOCO AUROC, DSA per-class, **cross-domain transfer** ("trained only on main-game problems, still spots off-by-ones inside sorts at X"), adaptive exam finds more misconceptions per item than a fixed exam, false-resolve table.
8. **Close:** "Every bug tells a story about how you think — even in your first sorting algorithm."

---

## 8. Risks and cut lines

**Cut order if behind:** Strong tier → Practice Arena screen (use Bug Lab) → WarpStack animation (static frame list) → SignalTiles (use ArrayBars with chars) → D08 strings class (string problems keep M-classes) → exam adaptivity (fall back to a fixed blueprint: one item per sector, same API) → Arrays planet path polish (keep 3 nodes) → arena VFX → audio.

**Never cut:** interpreter, gate, hard-twin + probe, verified fixer, knowledge model + trap, exam start→finish→debrief (even if fixed-order), Lab Report with honest caveats, fixtures.

| Risk | Likelihood | Mitigation |
|---|---|---|
| Interpreter scope grows (strings, recursion) | High | Strings limited to char arrays + literals + `strlen`; recursion depth cap 100; unit test per event |
| 34 problems × 3 variants is a lot of authoring | High | Agent drafts, `problems.check` verifies; CP2 fallback drops 3 problems |
| Synthetic scores "too perfect" | High | Headline = R-blind; red-flag thresholds; operator holdout; LOCO; E15 cross-domain |
| Adaptive exam looks like magic | Med | "Why this item" log in the debrief; E14 simulation vs fixed exam |
| Two modes = two products | Med | Exam reuses CodeCard, D04 diagnosis panel, D05 probe, D06 intervention |
| Demo time | Med | Demo mode = 3 exam items; seeded learner |

---

## 9. Unlock rules
- Planet order: **Conditions → Loops → Arrays.** A planet unlocks when the previous one's path is complete (all mission nodes passed; ghost node not required).
- **Deep Space Trials is unlocked from the start.** The lobby shows a soft banner if fewer than 3 planets are complete: "Recommended after the 3 planets: the Trials recheck your earlier repairs." It never blocks.
- If the exam is taken first, it still works: main-game classes start at the population prior, so findings are "new" rather than "held/relapsed".
- **Judge mode:** URL `?judge=1` or Settings → "Demo: open all planets". Seeded demo learner (`POST /learner/{id}/seed`) has all 3 planets complete, M01 STABLE, M08 ACTIVE, M06 MASTERED.
- Retakes: allowed; each exam starts from the learner's current knowledge state.

---

## 10. Patches to `00_PROJECT_CONTEXT.md`
1. §1: add "Plus a DSA *Test Yourself* mode (Deep Space Trials), open from the start, per judges' feedback: basic searching, sorting, array techniques, strings and recursion in C."
2. §2 Core loop: append `DEEP SPACE TRIALS (open any time; best after the 3 planets) → adaptive exam → DEBRIEF → PRACTICE`.
3. §2 planet ladder: "Playable: Conditions → Loops → Arrays. Variables and Functions are teasers. Deep Space Trials is open from the start."
4. §2 fiction: "Stars flicker while ACTIVE/TREATING/PROBATION, steady when STABLE, fully lit when MASTERED (ghost return or exam), red flicker when RELAPSED. The Trials are a wormhole at the galaxy's edge: the ship's AI tests whether your repairs hold in deep space."
5. §3 table: Team "2 people + AI agents; 12-hour build"; Tech "Vite + React + TS (no Phaser); FastAPI + SQLite; pycparser interpreter with strings + recursion"; Language "C (no pointers, no scanf; char arrays and recursion allowed)"; Input "free typing core".
6. §4 differentiators: add "Adaptive DSA exam chosen by expected information gain, with an explained 'why this question' log" and "The exam is the final ghost return for main-game misconceptions". Replace #3 with "code-identical twin handling (soft labels + probe)", #5 with "blind-authored realistic set + ITSP real-student slice", and #8 with "no LLM in the loop; feature ablations prove the game/execution signals add lift".
7. §5 architecture: add "Exam selector (EIG over knowledge states + coverage + target difficulty)" under FastAPI.
8. §6 mapping: add DSA rows: swap without temp → *cargo crate duplicates*; `low = mid` → *scanner window freezes, reactor overheats*; single-pass sort → *only the heaviest crate reaches the end*; missing base case / no shrink → *warp gates open forever*; discarded recursive result → *message lost between gates*; early `return -1` → *scanner gives up after the first beacon*; `==` on strings → *decoder never matches the signal*.
9. §7 file index → the three v3 files.
