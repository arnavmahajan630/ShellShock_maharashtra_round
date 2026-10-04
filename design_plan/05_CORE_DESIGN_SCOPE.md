# 05 — CORE DESIGN SCOPE (what we design and build, v3)

> Scope only: screens, functions, states, flows and content. No visual-style decisions. Derived from `01_SCOPE_v3.md` and `02_DESIGN_SCOPE_v3.md`. Anything not listed here is Strong, Stretch or cut (see §8).

---

## 1. What "Core" means

Core is the smallest product that delivers the full demo path end to end:

1. **Main game:** three playable planets (Conditions → Loops → Arrays). Each mission runs *predict → code → run → diagnose → (probe) → intervene → trap + transfer → verdict*.
2. **Deep Space Trials:** an adaptive DSA exam (open from the start) with a debrief report.
3. **Judge pages:** Lab Report and Bug Lab.
4. **Resilience:** fixtures fallback, demo learner, judge mode.

**Core = 14 screens** (D01–D11, D16, D17, D18), plus three in-flow views that live inside those screens: concept briefing (inside D02), predict phase (inside D03) and trap check (inside D07).

---

## 2. Core screen inventory

| ID | Screen | Purpose | Main user actions |
|---|---|---|---|
| D01 | Title + Galaxy Map | Entry and hub | Enter callsign or resume; choose a planet; open the Trials; open Mental Model; footer links to Lab Report / Bug Lab |
| D02 | Planet Path (one component, 3 configs) | Per-planet progression | See node states; start a node; open the briefing |
| D03 | Mission (Predict → Code → Run) | The core play loop | Predict; type code; run; watch the world; see intended vs actual |
| D04 | Diagnosis | Explain what the system found | Read finding, confidence and evidence; continue to probe or intervention |
| D05 | Probe | Split ambiguous diagnoses | Answer one question (max 2 per diagnosis) |
| D06 | Intervention | Fix the belief, not just the line | Work the primary panel; read the counterexample; see the verified fix |
| D07 | Transfer + Trap | Test whether it stuck | Answer a no-execution trap; complete a transfer mission |
| D08 | Verdict | Say whether the misconception is stable | Read the result; return to the path or retry |
| D09 | Mental Model | Show the learner state | Switch constellations; open a star's detail card |
| D10 | Lab Report | Show the ML evidence honestly | Switch tabs; read metrics with slice, n and caveat |
| D11 | Bug Lab | Let judges try their own code | Pick problem or preset; diagnose; run split mode on twins |
| D16 | Trials Lobby | Entry to the DSA exam | Read rules; start the exam; (judge) demo mode |
| D17 | Exam Runner | Take the adaptive exam | Code or answer; run samples; submit; skip |
| D18 | Debrief Report | Payoff of the exam | Read findings; answer deferred probes; read "why these questions"; choose next step |

---

## 3. Per-screen scope

### D01 — Title + Galaxy Map
**Functions**
- Callsign entry for a new player; "Resume as {callsign}" for a returning one.
- Galaxy with five bodies: three playable planets (Conditions, Loops, Arrays) and two teasers (Variables, Functions) marked "signal lost".
- Planet unlock order: Conditions → Loops → Arrays (a planet unlocks when the previous path's mission nodes are complete; the ghost node is not required).
- A planet shows an unstable indicator when any of its misconceptions is ACTIVE, TREATING or PROBATION.
- Deep Space Trials entry, always open. If fewer than 3 planets are complete, a hover note recommends finishing them first (never blocks).
- Buttons: Mental Model. Footer: Lab Report, Bug Lab.

**States to design:** first visit · mid-game · seeded demo learner · hover note on Trials · hover note on a teaser planet · connecting/offline.
**Data:** `GET /learner/{id}`, `POST /learner`.

### D02 — Planet Path
**Functions**
- One component with three configurations. Six nodes each:

| Planet | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Conditions | Briefing | Predict warm-up | `door_open` | `shield_mode` | `in_range` | Ghost: `max_of_three` |
| Loops | Briefing | Predict warm-up | `fire_shots` | `total_energy` | `charge_steps` | Ghost: `power_up` |
| Arrays | Briefing | Predict warm-up | `last_beacon` | `avg_fuel` | `sum_first_k` | Ghost: `max_shield` |

- Node states: locked, available, current, done (1–3 stars), unstable.
- The ghost node looks and behaves like a normal mission node in Core; its recheck logic runs in the engine.
- **Concept briefing** (node 1): a short guided explanation with one worked example. It never lists misconceptions.

**Data:** `GET /learner/{id}`, `GET /problems`.

### D03 — Mission (Predict → Code → Run)
**Functions (three phases in one screen)**
1. **Predict:** read-only code, a numeric input or answer choices, lock-in.
2. **Code:** free typing in a code editor, starter code with a TODO, objective banner, Run button.
3. **Run:** `POST /attempt`; the world plays the effects; intended vs actual counters; trace scrubber.

**World types (Core):** cannon (`fire`), drone bay (`read_cell` / `read_void`), reactor (step-cap overheat), door (`door_open` / `door_closed`), fuel gauge (`fuel_set`), radar (value display).
**States:** untouched · edited · running · success · partial · failure → D04 · gate message · interpreter error.
**Gate messages** (shown in-world, no diagnosis): empty, unchanged, parse error, Python, C++, unsupported construct, forbidden builtin, wrong signature, no recursion used, hard-coded answer, code too large.
**Events sent:** `prediction_made`, `code_edit` (throttled), `attempt_submitted`.

### D04 — Diagnosis
**Sections:** observed vs expected · diagnosis (name + plain subtitle + confidence band) · confidence for the top 3 · evidence list · action.

| State | When | Action |
|---|---|---|
| scanning | request in flight | none |
| confident | one clear finding | go to D06 |
| ambiguous | two candidates, probe available | go to D05 |
| novel | doesn't match any known pattern | go to D06 (generic inspection of the trace) |
| gate | code was rejected before analysis | back to code |
| read-only | embedded in D18 | none |

Evidence text comes only from the backend; evidence lines can highlight the matching code line.

### D05 — Probe
- One question with three answers, max 2 per diagnosis. After an answer, the confidence updates and the screen returns to D04.
- The system says "Logged." and never reveals the answer.
- Missions use `POST /probe/answer`; the Debrief uses `POST /exam/probe` (deferred).

### D06 — Intervention
**Always shown:** a counterexample ("try a different input: intended vs yours"), the player's code minimally fixed with a "Verified" note (or "Reference solution"), and two lines of class copy.
**Primary panel depends on `modality`:**

| Modality | Used for |
|---|---|
| trace timeline (step list with flags, "set i" question) | loop/condition/return classes and early-exit search |
| memory strip (array cells with a void cell) | index and swap/sort classes |
| window strip (low/mid/high per iteration) | binary-search window |
| value meter (exact vs truncated, static meter) | int division, uninitialised variable |
| call stack (recursion replay) | all recursion classes |

Event sent: `intervention_completed`. Switching modality after a failed attempt is Strong.

### D07 — Transfer + Trap
- **Trap first:** a prediction with no code execution.
- **Then a transfer mission** from a different problem family (reuses D03).
- One call to `POST /reassess` per item. Progress is visible (trap, transfer 1, transfer 2).

### D08 — Verdict
- **STABLE:** the misconception shown as steady; note that full lock comes after a later recheck (ghost return or the Trials).
- **NOT YET:** the unmet conditions as a ✓/✗ list plus an "active" meter; a button to try again (back to D06).
- **MASTERED** and **RELAPSED** appear as toasts (from a ghost return or the exam).

### D09 — Mental Model
- Two constellations: **Core** (9 stars) and **Deep Space** (8 stars; "uncharted" until the learner has attempted any DSA item).
- Star states: UNSEEN, ACTIVE, TREATING, PROBATION, STABLE, MASTERED, RELAPSED (each as shape + label).
- Star detail card: state, active probability, times seen, last 3 evidence items, interventions tried, where it was last tested, recheck queued.
- Empty state for a brand-new learner.

### D10 — Lab Report
Seven tabs; every card shows its **slice**, **n** and **caveat**:
1. Overview (KPIs) · 2. Diagnosis (per-class table, confusion matrix, unseen problems, baselines) · 3. Twins (structural pair accuracy; hard-twin flag rate and simulated post-probe accuracy) · 4. Unseen & calibration (LOCO AUROC, U1/U2 examples, reliability, ECE, risk–coverage) · 5. DSA (D-class metrics, cross-domain transfer, held-out DSA problems) · 6. Resolution & Exam (policy table; adaptive vs fixed vs random exam) · 7. Ablation & data (feature ablation, dataset card summary, EDA highlights, failure audit).
Source: `GET /metrics`. The LLM baseline row reads "not run".

### D11 — Bug Lab
- Problem selector grouped by Conditions / Loops / Arrays / Variables / Functions / DSA sectors, a code editor, optional prediction input, Diagnose.
- Result shows the diagnosis (D04 layout), a mini trace and the matching DSA visualiser.
- Ambiguous result → inline probe (stateless, answers replayed in order).
- **Presets:** hard twin A/B · M02 vs M07 · M03 vs M05 · chained comparison (novel) · Python code (gate) · hard-coded answer (gate) · swap without temp · binary search `low = mid` · missing base case · recursion `f(n)` · early `return -1` · `s[i] == "a"` · off-by-one inside bubble sort.
- **Split mode** shows two twins side by side.
- Endpoint: `POST /lab/diagnose` (no learner prior, nothing stored).

### D16 — Trials Lobby
- Five sector cards with the learner's rating (or "uncharted"): Searching, Sorting, Array Techniques, Strings, Recursion.
- Rules: 10 trials · about 25 min · no hints · questions adapt to you · full debrief at the end.
- Soft banner if fewer than 3 planets are complete (never blocks).
- Buttons: Start Trials; (judge) Demo mode: 3 trials.
- `POST /exam/start`.

### D17 — Exam Runner
- Header: item k of N, sector, total timer, Skip (counts as not passed, no diagnosis).
- **Coding item:** prompt, signature, free-typing editor with starter, the sector's visualiser, **Run samples** (the two visible tests only, `POST /run` with `sample_only`), Submit.
- **Trace item:** read-only code, question, choices or numeric answer.
- **During the exam:** no hints, no diagnosis, no companion lines. After Submit: "Logged · Routing next trial…" then the next item.
- **Time-out:** auto-submit current code and go to the debrief.
- Visualisers: **ArrayBars** (searching, sorting, array techniques), **SignalTiles** (strings), **WarpStack** (recursion).
- Endpoint: `POST /exam/answer`.

### D18 — Debrief Report
1. **Summary:** five sector scores with rating change, total time, items passed.
2. **Sharpen your report:** up to 3 deferred probes for ambiguous findings.
3. **Findings** (cards with a status): NEW · RELAPSED · HELD → MASTERED · UNCERTAIN (links to a deferred probe) · NOT TESTED (collapsed). Each shows name + subtitle, confidence band, the evidence items (expand to a read-only D04) and a "Train this" button.
4. **Why these questions:** one line per item with a small information-gain bar.
5. **Next steps:** Train weak spots (opens Bug Lab with the problem preloaded in Core) · Retake · Back to galaxy.
- Endpoints: `POST /exam/finish`, `POST /exam/probe`.

---

## 4. Core flows

**Main game**
```
D01 → D02 (→ briefing) → D03
  success → D02
  gate / interpreter error → stay in D03
  system failure → D04
     confident → D06 | ambiguous → D05 → D04 | novel → D06
  D06 → D07 (trap, then transfer) → D08
     STABLE → D02 | NOT YET → D06
```

**Trials**
```
D01 → D16 → D17 (×N, 10 items or 3 in demo mode) → D18
D18 → deferred probe (D05 component) → D18
D18 → Train this → D11 (preloaded) | Retake → D16 | Back → D01
```

**Judge / explorer**
```
?judge=1 → all planets open, "Seed demo learner", demo mode in D16
Footer → D10 ⇄ D11
Backend down → fixtures with an "OFFLINE REPLAY" badge on every screen
```

---

## 5. Core content

| Item | Core amount |
|---|---|
| Playable planets | 3: Conditions, Loops, Arrays |
| Teaser planets | 2: Variables, Functions (signal lost) |
| Path nodes per planet | 6: briefing, predict warm-up, 3 missions, 1 ghost-looking mission |
| Main problems | 16 (Conditions 4, Loops 6, Arrays 4, Variables 1, Functions 1) |
| DSA problems | 18 (Searching 5, Sorting 3, Array techniques 4, Strings 3, Recursion 3) |
| Misconception classes shown | Core constellation 9 (M01–M08, M10) + Deep Space 8 (D01–D08) |
| Never-trained classes (novel demo) | 2 |
| Probes | 14 (8 main + 6 DSA), plus trap items |
| Intervention modalities | 5 (trace timeline, memory strip, window strip, value meter, call stack) |
| Exam | 10 items (5 coding + 5 trace), about 25 min; demo mode 3 items |
| World types | 6 (cannon, drone bay, reactor, door, fuel gauge, radar) |
| DSA visualisers | 3 (ArrayBars, SignalTiles, WarpStack) |
| Lab Report tabs | 7 |
| Bug Lab presets | 13 |

### Misconception display (each shown with a ship name and a plain subtitle)
- **Core:** Boundary Drift (M01), Stalled Thruster (M02), Memory Wipe (M03), Fraction Shear (M04), Static Signal (M05), Sensor Overwrite (M06), Ghost Semicolon (M07), Index Origin Fault (M08), Silent Messenger (M10).
- **Deep Space:** Premature Abort (D01), Frozen Window (D02), Cargo Overwrite (D03), Half-Sorted Hold (D04), Endless Warp (D05), Static Warp (D06), Lost Echo (D07), Signal Mismatch (D08).
- **Special:** Unknown Anomaly (novel), Unclassified Fault (OTHER).

---

## 6. Core rules

- **Learner states:** `UNSEEN → ACTIVE → TREATING → PROBATION → STABLE → MASTERED`, plus `RELAPSED`. A mission can reach STABLE; only a ghost return or the exam can promote to MASTERED.
- **Probes:** max 2 per diagnosis; never reveal the answer.
- **Exam:** no hints, no diagnosis, no companion during the exam; everything appears in the debrief. Each next item is chosen adaptively and its reason is logged for the debrief.
- **Trials access:** open from the start; the soft banner never blocks. An exam taken first still works; main-game classes start from the population prior, so findings read as NEW.
- **Copy:** never "wrong"; use "System failure" (world), "Analysis" (diagnostic), "Logged" (exam and probe).
- **Evidence text** is generated by the backend only; the frontend adds no field names of its own.
- **Gates** stop analysis before the model for the cases listed under D03.
- **Fixtures:** every endpoint has a fixture fallback, including one full exam (3 items + report).
- **Judge mode** (`?judge=1`): all planets open, footer links, demo mode, seed button.

---

## 7. Core acceptance criteria (design + demo path)

1. Loops end to end on the real backend: play → fail → diagnosis → intervention → transfer + trap → STABLE.
2. Hard-twin demo (`for(i=0;i<=n;i++) sum+=a[i];`): ambiguous → probe → confident (≥ 0.85).
3. Conditions planet: `if (lo < x < hi)` → Unknown Anomaly (never trained on it).
4. Arrays planet playable end to end (fixtures acceptable for one node).
5. Trials: open from the start; the adaptive exam runs (10 items, or 3 in demo mode); each next item has a logged reason; the debrief shows sector scores, DSA findings, deferred probes and at least one main-game misconception promoted to MASTERED or flagged RELAPSED on a DSA surface.
6. DSA diagnosis works for at least: swap without temp, `low = mid`, missing base case, early `return -1` (Bug Lab presets).
7. Gate messages correct for empty, Python, hard-coded and `strlen` in the no-strlen task.
8. Lab Report shows all experiments including DSA per-class results, cross-domain transfer and adaptive-vs-fixed exam.
9. `make demo` works; with the backend killed, fixtures run with the "OFFLINE REPLAY" badge.

---

## 8. Explicitly NOT in Core

| Item | Tier |
|---|---|
| Ghost-return special UI (the engine is Core) | Strong |
| Quickfire timed MCQ nodes (D12) | Strong |
| Orbit hint ladder (D14), missions only | Strong |
| Settings screen and reduced-motion controls (D15) | Strong |
| DSA Practice Arena as a full screen (D19); Core fallback = "Train this" opens Bug Lab | Strong |
| Two-bug diagnosis state | Strong |
| Modality switching after "Not yet" | Strong |
| M09 pass-by-value and its twin | Strong |
| Past exam reports / retake comparison | Strong |
| Real-student data slice | Strong |
| Teacher heatmap, Variables and Functions as playable planets, pointers and linked lists, stacks and queues, 2-D arrays, boss fight, Python frontend | Stretch |
| Boss fight, game-over screen, ending cutscene, results screen | Cut (replaced by the Trials and a toast) |

**Design cut order if time runs short:** Practice Arena → WarpStack animation (use a static frame list) → SignalTiles (use ArrayBars with characters) → Arrays planet polish (keep 3 nodes) → arena effects.
**Never cut:** the gate, the hard-twin + probe flow, the verified fix, the trap, the exam start-to-debrief path, the Lab Report with honest caveats, and fixtures.

---

## 9. Design workload summary

| Group | Screens | Distinct states to design (approx.) |
|---|---|---|
| Entry and hub | D01, D02 (+ briefing) | 12 |
| Mission loop | D03, D04, D05 | 22 |
| Teaching loop | D06, D07, D08 | 14 |
| Learner model | D09 | 4 |
| Trials | D16, D17, D18 | 20 |
| Judge pages | D10, D11 | 10 |
| **Total** | **14 screens** | **about 80 states** |
