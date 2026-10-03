# 02 — DESIGN SCOPE v3 (3 planets + Deep Space Trials, 12-hour build)

> **Replaces `02_SCREEN_LIST.md` and `02_DESIGN_SCOPE_v2.md`.** Visual tokens from `00_PROJECT_CONTEXT.md` §3 still apply. All API/JSON shapes are defined in `03_ML_IMPLEMENTATION_PLAN_v3.md` §11; agents must not invent fields.
>
> **Core screens:** D01–D11 (main game + judge pages) and **D16–D18 (Deep Space Trials: Lobby, Exam Runner, Debrief)**. **Strong:** D12–D15, D19 (Practice Arena). Merged/cut v1 screens: §9.

---

## 1. Design gaps fixed

| Gap | Fix |
|---|---|
| Predict / Code Lab / Arena were 3 navigations per attempt | One **Mission screen** (D03) with phases Predict → Code → Run |
| No place for judges to try their own code | **D11 Bug Lab** (main + DSA presets, side-by-side twins) |
| Diagnosis didn't explain *why* the model is unsure | Ambiguous state: "Code alone can't separate these. One question will." |
| Gate cases unhandled | Gate state in D04 with in-world messages |
| "Resolved" before the ghost recheck | Verdict says **STABLE**; **MASTERED** only after a ghost return or the exam |
| Star states colour-only | Colour + shape/icon + label |
| Flicker/scanlines with no motion setting | `prefers-reduced-motion` respected |
| Raw confidence % | Bars + calibrated band words + abstain line |
| No offline resilience | Fixtures mode + "OFFLINE REPLAY" badge |
| **(v3) No DSA** | **Deep Space Trials**: Lobby → adaptive Exam Runner → Debrief → Practice; 3 DSA visualisers |
| **(v3) Exam adaptivity invisible** | Debrief shows a **"Why these questions"** log, one line per item |
| **(v3) Exam vs learning conflict** | No hints or diagnoses *during* the exam; everything in the Debrief, incl. deferred probes |

---

## 2. Global design system

### 2.1 Tokens
```
--bg:#0B0E17  --panel:#141A2A  --panel-2:#1C2438  --line:#2A3550
--cyan:#3FE0FF (action)  --amber:#FFC247 (reward)  --mint:#5CF2A2 (success/STABLE)
--magenta:#FF4FA3 (ACTIVE)  --red:#FF5A5A (RELAPSED)  --violet:#9B7BFF (Deep Space Trials accent)
--text:#E8ECF6  --muted:#8A94AD
Fonts: Press Start 2P (titles), VT323 (diagnostic layer), JetBrains Mono (code)
Pixel scale: integer 2×/3×, image-rendering: pixelated. Radius 0 (game), 2px (diagnostic).
```
Layers: **Game** (bevel buttons, bouncy motion 200–300 ms), **Diagnostic** (VT323, scanlines 6%, linear 150 ms), **Trials** (violet accent, calmer, exam-like: no confetti during the exam, timer always visible).

### 2.2 Global components

| Component | States | Notes |
|---|---|---|
| `TopHUD` | normal, low-energy, exam (timer + item k/N replaces hearts) | XP, streak, credits, settings, OFFLINE REPLAY badge |
| `BevelButton` | default, hover, pressed, disabled, danger, loading | |
| `AnswerPill` | idle, selected, correct, wrong, timed-out, disabled | keys 1–4 |
| `PlanetNode` | locked, available, current, done (1–3 stars), unstable | icons: briefing / predict / code / quickfire / ghost |
| `StarGlyph` | UNSEEN ○, ACTIVE ✶ magenta flicker, TREATING ✶ + wrench, PROBATION ✶ amber pulse, STABLE ★ mint, MASTERED ★ amber + ring, RELAPSED ✶ red + crack | always shape + label |
| `CodeCard` | read-only, editable, running (line follows trace), error | CodeMirror 6 |
| `TraceScrubber` | idle, playing, paused, at-event | event ticks: `oob_read`, `uninit_read`, `step_cap`, `depth_cap` |
| `OrbitBubble` | idle, thinking, alert, celebrating, concerned | silent during the exam |
| `ConfidenceBars` | — | top-3 + band words + abstain line |
| `EvidenceItem` | chips `CODE` `RUN` `YOU PREDICTED` `PROBE` `HISTORY` `EXAM` | line refs highlight CodeCard |
| **`ArrayBars`** (v3) | idle, reading (cell glow), writing (bar morph), compare (two bars flash), swap (bars cross), void-read (bar outside the frame glitches), markers | markers from `problem.markers` (e.g. `i`, `j`, `low`, `mid`, `high`) drawn as pixel arrows under the bars |
| **`SignalTiles`** (v3) | idle, reading, mismatch, terminator | one tile per char; `'\0'` is a dim **END** tile; reads past END glitch |
| **`WarpStack`** (v3) | push (gate opens, frame slides in with args), pop (return value travels down), overflow (stack hits ceiling → red "WARP OVERFLOW") | from `call`/`ret` effects; caps at 12 visible frames + "…+N" |
| `Toast` | streak, level-up, star stabilised, mastered, relapse | |

### 2.3 Misconception display names (ship name + plain subtitle, always both)

**Core constellation (main game)**
| ID | Ship name | Plain subtitle |
|---|---|---|
| M01 | Boundary Drift | Loop runs one step too many or too few |
| M02 | Stalled Thruster | Loop variable never moves toward the exit |
| M03 | Memory Wipe | Running total is reset inside the loop |
| M04 | Fraction Shear | int ÷ int drops the decimals |
| M05 | Static Signal | Variable used before it was given a value |
| M06 | Sensor Overwrite | `=` stores a value; `==` compares |
| M07 | Ghost Semicolon | A stray `;` gives the if/loop an empty body |
| M08 | Index Origin Fault | Arrays start at 0; last cell is n−1 |
| M10 | Silent Messenger | `printf` shows a value; `return` hands it back |
| M09 *(Strong)* | Copy Module | A function gets a copy of the variable |

**Deep Space constellation (DSA)**
| ID | Ship name | Plain subtitle |
|---|---|---|
| D01 | Premature Abort | The search gives up after the first miss |
| D02 | Frozen Window | Binary search window never shrinks (`low = mid`) |
| D03 | Cargo Overwrite | Swapping without a temp loses a value |
| D04 | Half-Sorted Hold | One pass doesn't sort the whole array |
| D05 | Endless Warp | Recursion has no reachable base case |
| D06 | Static Warp | The recursive call doesn't move toward the base case |
| D07 | Lost Echo | The recursive call's result is thrown away |
| D08 | Signal Mismatch | `==` doesn't compare text in C |
| novel | Unknown Anomaly | Doesn't match any known pattern |
| OTHER | Unclassified Fault | A bug we don't have a lesson for yet |

### 2.4 Copy rules
Never "wrong": "System failure" (world), "Analysis" (diagnostic), "Logged" (exam). Evidence text comes from the backend only. Bands: ≥ 0.75 Likely, 0.45–0.75 Possible, < 0.45 Unsure.

### 2.5 Accessibility & motion
Reduced motion: no flicker (2 s pulse), no shake, static scanlines, ArrayBars swap = instant swap + highlight. Contrast ≥ 4.5:1. Keyboard: Run `Ctrl/Cmd+Enter`, Submit (exam) `Ctrl/Cmd+Shift+Enter`, pills `1–4`.

### 2.6 Responsive
Desktop 1440×900: arena/visualiser 55% left, CodeCard 45% right. Mobile 390×844: stacked; visualiser 40vh; scrubber as bottom sheet; token chips above the keyboard (`<` `<=` `;` `==` `[` `]` `++` `'` `"` `\0`).

### 2.7 Fixtures mode
`VITE_FIXTURES=1` or automatic after 2 failed calls → `/fixtures/*.json`, badge "OFFLINE REPLAY". Includes one complete exam flow (3 items + report).

### 2.8 Judge mode
`?judge=1`: all planets open, footer links to Lab Report and Bug Lab, "Demo mode (3 items)" toggle in the Trials Lobby, "Seed demo learner" button.

---

## 3. Navigation

```
D01 Galaxy ─┬─▶ D02 Planet Path (Conditions → Loops → Arrays, sequential)
            │        └─▶ D03 Mission ─fail─▶ D04 Diagnosis ─▶ (D05 Probe) ─▶ D06 Intervention ─▶ D07 Transfer+Trap ─▶ D08 Verdict
            ├─▶ D16 Trials Lobby (OPEN FROM START) ─▶ D17 Exam Runner ×N ─▶ D18 Debrief ─▶ D19 Practice (or D11 Bug Lab preloaded)
            ├─▶ D09 Mental Model (Core + Deep Space constellations)
            └─ footer / ?judge=1: D10 Lab Report ⇄ D11 Bug Lab
```

---

## 4. Main-game screens (Core)

### D01 — Title + Galaxy Map
- Starfield, logo, callsign input / "Resume as {callsign}".
- **Planets:** Conditions (live), Loops (live after Conditions), Arrays (live after Loops); Variables and Functions "signal lost" (teaser tooltip).
- **Deep Space Trials wormhole** (violet swirl) at the edge: **always open.** If < 3 planets complete, hover note: "Recommended after the 3 planets: the Trials recheck your earlier repairs."
- Buttons: Mental Model; footer: Lab Report · Bug Lab.
- Data: `GET /learner/{id}`. Acceptance: a planet flickers if any of its misconceptions is ACTIVE/TREATING/PROBATION.

### D02 — Planet Path (one component, 3 configs)
| Planet | Node 1 | Node 2 | Node 3 | Node 4 | Node 5 | Node 6 |
|---|---|---|---|---|---|---|
| Conditions | Briefing | Predict warm-up | P11 `door_open` | P12 `shield_mode` | P16 `in_range` | Ghost: P17 `max_of_three` |
| Loops | Briefing | Predict warm-up | P01 `fire_shots` | **P03 `total_energy` (hard-twin demo)** | P05 `charge_steps` | Ghost: P06 `power_up` |
| Arrays | Briefing | Predict warm-up | P08 `last_beacon` | P07 `avg_fuel` | P10 `sum_first_k` | Ghost: P09 `max_shield` |
Ghost nodes look like normal missions (UI Strong, engine Core).

### D03 — Mission (Predict → Code → Run)
Predict (read-only code + numeric/pills + lock-in) → Code (free typing, objective banner, starter + TODO, Run) → Run (`POST /attempt` → arena plays `world_effects`; Intended vs Actual counters; scrubber).
Worlds: cannon (`fire`), drone bay (`read_cell`/`read_void`), reactor (`step_cap` overheat), door (`door_open/closed`), gauge (`fuel_set`), radar (value display).
States: untouched · edited · running · success · partial · failure → D04 · gate message · interpreter error.

### D04 — Diagnosis Telemetry
Sections: Observed vs Expected · Diagnosis (ship name + subtitle + band) · ConfidenceBars · Evidence list · Action.
| State | Trigger | UI | Action |
|---|---|---|---|
| scanning | in flight | radar sweep (600 ms–2 s) | — |
| confident | `status="confident"` | single name | Run repair protocol → D06 |
| ambiguous | `status="ambiguous"` + `next_probe` | two names + "Code alone can't separate these. One question will." | Answer probe → D05 |
| two-bug | `status="two_bug"` | both shown, top-1 first | → D06 |
| novel | `status="novel"` | UNKNOWN ANOMALY, no class bars | Inspect trace → D06 generic |
| gate | `gate.code != "G0"` | `gate.message` | Back to code |
| read-only (v3) | embedded in D18 Debrief | same layout, no action button | — |

### D05 — Probe Encounter
Bridge/door scene, Orbit "Let me check one thing.", code snippet, 3 pills → `POST /probe/answer` (missions) or `POST /exam/probe` (debrief) → animate bars old → new. Max 2 per diagnosis. Says "Logged.", never reveals the answer.

### D06 — Intervention
Three panels; `intervention.modality` decides which is primary:
| modality | Primary panel | Used for |
|---|---|---|
| `trace_timeline` | step list with flags (`extra`, `missing`, `reset`, `overwrite`, `empty_body`, `early_return`) + "set i" question | M01 M02 M03 M06 M07 M10 D01 |
| `memory_strip` | ArrayBars with void cell at n; reads/writes highlighted | M08 D03 D04 |
| `window_strip` | ArrayBars + `low/mid/high` markers per iteration; frozen window highlighted | D02 |
| `value_meter` | exact vs truncated / static meter | M04 M05 |
| `call_stack` | WarpStack replay; missing base / non-shrinking args / discarded return highlighted | D05 D06 D07 |
| `counterexample`, `minimal_fix` | make those panels primary | 2nd modality |
Always: counterexample panel ("Try n = 3: Intended / Yours") and "Your code, minimally fixed" diff with "Verified: passes all tests" (or "Reference solution"). 2-line class copy from the backend. Event `intervention_completed`.

### D07 — Transfer + Trap
Trap prediction (no execution) first, then a transfer mission from a different family (reuses D03). `POST /reassess` per item.

### D08 — Verdict
STABLE: flicker → steady mint, "Stable. Full lock after a later recheck (ghost return or the Trials)." NOT YET: itemised `conditions[]` ✓/✗ + `p_active` meter + "Try a different explanation".

### D09 — Mental Model
Two constellations: **Core** (9 stars) and **Deep Space** (8 stars; "uncharted" dots until the learner has attempted any DSA item). Click a star → card: state, `p_active`, times seen, last 3 evidence items, interventions tried, where it was last tested (planet node or exam item), ghost/exam recheck queued.

---

## 5. Deep Space Trials (Core, v3)

### D16 — Trials Lobby
- **Purpose:** entry to the DSA "Test Yourself" level. **Open from the start.**
- **Layout:** violet wormhole header "DEEP SPACE TRIALS"; 5 sector cards with icon + 1-line description + the learner's sector rating (bar, "uncharted" if none):
  | Sector | World framing | Visualiser |
  |---|---|---|
  | Searching | Beacon Scan: find the right beacon | ArrayBars (+ `low/mid/high`) |
  | Sorting | Cargo Hold: order crates by weight | ArrayBars |
  | Array Techniques | Convoy Ops: reverse, rotate, pair up ships | ArrayBars (+ `i/j`) |
  | Strings | Signal Decoder: read the transmission | SignalTiles |
  | Recursion | Warp Gates: each gate opens the next | WarpStack |
- **Rules card:** "10 trials · about 25 min · no hints · questions adapt to you · full debrief at the end".
- Soft banner (only if < 3 planets complete): "Recommended after the 3 planets: the Trials also recheck your earlier repairs." Never blocks.
- Buttons: **Start Trials** · (judge) **Demo mode: 3 trials** · past reports (Strong S10).
- API: `POST /exam/start {learner_id, length}` → first item.

### D17 — Exam Runner
- **Header:** item k/N, sector chip, total timer (25:00, amber at 5:00, red at 1:00), "Skip" (counts as not passed, no diagnosis).
- **Coding item:** prompt + signature + CodeCard (free typing, starter) + **visualiser** for the sector + **Run samples** (runs only the 2 visible sample tests via `POST /run {sample_only:true}`; shows pass/fail and animates the visualiser) + **Submit**.
- **Trace item:** read-only code + question + pills/numeric (same component as Predict).
- **During the exam:** no diagnosis, no hints, no Orbit lines; after Submit show "Logged · Routing next trial…" (600 ms) → next item from the response.
- **Time-out:** auto-submit current code; go to D18.
- **Events:** `exam_answer {exam_id, item_id, code|answer, ms, sample_runs}` via `POST /exam/answer`.
- **Acceptance:** with fixtures, a 3-item exam runs end-to-end; visualisers animate bubble-sort swaps, binary-search markers and a recursion stack from trace effects.

### D18 — Debrief Report
- **Purpose:** the payoff: what the Trials revealed, in the diagnostic layer.
- **Sections (top → bottom):**
  1. **Summary:** 5 sector bars (score + rating change), total time, items passed.
  2. **Sharpen your report (deferred probes):** up to 3 probes for ambiguous findings (D05 component, `POST /exam/probe`); findings update live.
  3. **Findings:** cards per class with a status chip:
     - **NEW** (DSA or main class first seen in the exam) — magenta
     - **RELAPSED** (was STABLE/MASTERED, failed again on a DSA surface) — red
     - **HELD → MASTERED** (was STABLE, passed a DSA item that exposes it) — amber ring
     - **UNCERTAIN** (evidence split, e.g. two recursion explanations) — violet, with a link to the deferred probe above
     - **NOT TESTED** (collapsed list)
     Each card: ship name + subtitle, posterior band, the item(s) as evidence (expand → read-only D04 panel with code + evidence), "Train this" button.
  4. **Why these questions:** ordered list, one line per item from `adaptivity_log[].reason` (e.g. "Q05 bubble_sort: checks Boundary Drift inside a sort (your Loops record) + covers Sorting"). Shows information gain as a small bar.
  5. **Next steps:** "Train weak spots" (→ D19 / Bug Lab preloaded) · "Retake" · "Back to galaxy".
- **API:** `POST /exam/finish` → `report`; `POST /exam/probe` → updated `report`.
- **Acceptance:** demo fixture shows ≥ 1 NEW DSA finding, 1 HELD → MASTERED, 1 deferred probe, and a 3-line "why" list.

### D19 — DSA Practice Arena *(Strong; Core fallback = Bug Lab preloaded)*
Sector tabs → problem list (with "recommended" badges from the debrief) → reuses D03 Code/Run + D04 + D05 + D06 + D07/D08 with the DSA visualiser as the arena. `POST /attempt` (same as missions).

---

## 6. Judge pages (Core)

### D10 — Lab Report
Tabs (all from `GET /metrics`; every card shows `slice`, `n`, `caveat`):
1. **Overview:** KPI tiles: grouped-CV macro-F1, R-blind macro-F1, LOCO AUROC, false-resolve (ours vs naive), **adaptive-exam detection gain**.
2. **Diagnosis:** per-class table (M + D), confusion matrix, unseen problems, baselines (LLM row "not run" unless present).
3. **Twins:** STRUCTURAL pair accuracy (T2–T4, T6–T9), HARD twin flag rate + simulated post-probe accuracy vs `p_b`.
4. **Unseen & calibration:** LOCO AUROC per class, U1/U2 examples, reliability diagram, ECE, risk–coverage.
5. **DSA (v3):** D-class P/R/F1; **cross-domain transfer** (trained on main problems only → M-classes found inside DSA problems); DSA held-out problems.
6. **Resolution & Exam (v3):** policy table (naive / 4-check / ours); **adaptive vs fixed vs random exam** (misconceptions correctly identified per item, coverage, Brier of final `p_active`).
7. **Ablation & data:** feature-group ablation, dataset card summary, effective N, EDA highlights, failure audit.

### D11 — Bug Lab
Problem dropdown grouped (Conditions / Loops / Arrays / Variables / Functions / **DSA sectors**), CodeCard, optional prediction input, Diagnose → inline D04 + mini trace + the right visualiser; ambiguous → inline probe (stateless `probe_answers[]`).
**Presets:** Hard twin A/B · M02 vs M07 · M03 vs M05 · Chained comparison (novel) · Python code (gate) · Hard-coded answer (gate) · **Swap without temp** · **Binary search `low = mid`** · **Missing base case** · **Recursion `f(n)`** · **Early `return -1`** · **`s[i] == "a"`** · **Off-by-one inside bubble sort** (cross-domain). Split mode for twins side by side.

---

## 7. Strong add-ons
| ID | Screen | Notes |
|---|---|---|
| D12 | Quickfire node | timed MCQ; distractor → class |
| D13 | Ghost Return node | disguised mission; pass → MASTERED |
| D14 | Orbit hint ladder | missions only, never in Trials |
| D15 | Settings | audio, reduced motion, font size, reset, judge mode |
| D19 | Practice Arena | §5 |

---

## 8. Event contract (FE → BE)
Envelope `{type, ts, learner_id, problem_id?, attempt_id?, exam_id?, payload}`.
| type | payload | endpoint |
|---|---|---|
| `prediction_made` | `{item_id, value, confidence?, ms}` | `/attempt` or `/reassess` |
| `attempt_submitted` | `{code}` | `/attempt` |
| `code_edit` | `{ms_since_start, chars}` (throttled) | `/attempt` |
| `hint_used` | `{level}` | `/attempt` |
| `probe_answered` | `{probe_id, answer}` | `/probe/answer` or `/exam/probe` |
| `mcq_answered` | `{item_id, choice, ms}` | `/attempt` (Strong) |
| `intervention_completed` | `{class, modality, steps, ms}` | `/intervene` |
| `transfer_result` | `{item_id, item_type, passed, prediction?}` | `/reassess` |
| `sample_run` (v3) | `{item_id, passed, total}` | `/run` (`sample_only`) |
| `exam_answer` (v3) | `{item_id, code? , answer?, ms, sample_runs, skipped}` | `/exam/answer` |

---

## 9. Assets (MVP)
Kenney CC0 space/pixel packs (ship, lasers, explosions, crates for Cargo Hold, small ships for Convoy, beacons), 1 droid sprite, star glyph SVGs (7 states), wormhole (CSS conic-gradient + pixel noise), WarpStack gate frame (CSS), 6 SFX (laser, explosion, error, success, scan, warp), 3 Google Fonts.

## 10. v1 → v3 screen map
S01+S02 → D01 · S03 → D02 (×3 planets) · S04 → node 1 of D02 · S05 → D12 · S06+S07+S08 → D03 · S09 → D04 · S10 → D06 · S11 → D05 (Core) · S12 → D07 · S13 → D08 · S14 Boss → cut (the Trials replace it as the end-game) · S15 → D09 · S16 → toast · S17 → D10 · S18 → cut · S19 → D13 · S20 → cut · new: D11 Bug Lab, **D16 Trials Lobby, D17 Exam Runner, D18 Debrief, D19 Practice**.

## 11. FE build order (agent-ready)
1. Tokens, fonts, kit route `/kit`: BevelButton, TopHUD, StarGlyph, CodeCard, ConfidenceBars, EvidenceItem, **ArrayBars, SignalTiles, WarpStack** (each driven by a fixture trace).
2. `src/api.ts` with fixtures fallback for every endpoint, incl. `/exam/*`.
3. D01 → D02 → D03 → D04 → D05.
4. D06 (5 modalities) → D07 → D08 → D09.
5. **D16 → D17 → D18.**
6. D10 → D11 → (D19 if ahead).
7. Swap fixtures for the real API endpoint by endpoint as ML marks them ready in `server/READY.md`.
