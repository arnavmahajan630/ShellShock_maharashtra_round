# Re:Learn Multimodal Missions — Test Guide & Input Reference

This document provides the exact inputs, correct answers, and intentional buggy answers (with their triggered misconceptions and UI behaviors) for both multimodal levels.

---

## Level A: Shield Relay · Flowchart Trace

* **Planet**: Aegis Grid (`/planet/conditions`)
* **Node on Map**: `shield_trace` (3rd node on orbit, replacing `door_open`)
* **Direct URL**: [http://localhost:5173/planet/conditions/mission/MMA-01](http://localhost:5173/planet/conditions/mission/MMA-01)
* **UI Theme**: Blue & Yellow Pixel UI (`deep-space` #0c1937, `void-blue` #153268, `starlight` #fcb143, `warp-cyan` #2de7fc) with `"Press Start 2P"` and `"VT323"` typography.
* **Initial Telemetry Inputs**: `power = 10`, `shield = 0`
* **Underlying Logic**:
  ```c
  int power = 10;
  int shield = 0;
  if (power >= 5) {
      shield = 1;
  }
  return shield;
  ```
* **Graph Structure**:
  * `n1`: Start (`power = 10, shield = 0`)
  * `n2`: Check Power (`if (power >= 5)`)
  * `n3`: Raise Shield (`shield = 1`)
  * `n4`: Return (`shield`)
* **Edges**:
  * `n1 → n2`: Unconditional transition
  * `n2 → n3`: `true (power >= 5)`
  * `n2 → n4`: `false (power < 5)`
  * `n3 → n4`: Unconditional transition

---

### Case A1: Correct Trace (Passes)

| Field | Input / Action |
|---|---|
| **Path Selection** | Tap nodes on SVG or click step buttons: `n1` → `n2` → `n3` → `n4` |
| **Predicted Output** | Click `1 (shield active / 1)` button (or type `1`) |
| **Reasoning (Optional)** | `Power is initialized to 10, which satisfies power >= 5. The condition evaluates to true, taking the branch to n3 where shield becomes 1, returning 1.` |

#### Expected System Response:
* **Gate & Tests**: `Passed: 1 / 1`
* **Diagnosis Status**: `correct` (`CORRECT: 0.95`)
* **UI**: Green checkmark on `SignalGateWorld`. Clicking **Continue** returns to orbit with `shield_trace` marked as passed.

---

### Case A2: Buggy Trace — Misconception `M06` (Bypassed Condition)

Learner assumes the condition evaluates to false or was bypassed, taking the false branch directly to `n4`.

| Field | Input / Action |
|---|---|
| **Path Selection** | Tap nodes on SVG or click step buttons: `n1` → `n2` → `n4` *(skipping n3)* |
| **Predicted Output** | Click `0 (shield off / 0)` button (or type `0`) |
| **Reasoning (Optional)** | `Assumed power condition failed or was bypassed, so shield remains 0.` |

#### Expected System Response:
* **Gate & Tests**: `Passed: 0 / 1`
* **Path Replay Visualizer**:
  * Alerts `⚠ PATH DIVERGENCE DETECTED AT NODE: n2`
  * Explains: *"With power = 10, the condition power >= 5 is true (10 >= 5), so the active shield branch (n3: shield = 1) is followed instead of skipping directly to n4."*
  * Step-by-step replay compares `n1 → n2 → n4` against `n1 → n2 → n3 → n4`.
* **Diagnosis Screen**:
  * Flags divergence at `n2` and provides remediation for condition branch tracing.
* **Next Steps**: Continues to Diagnostic Telemetry → Intervention → Transfer Trap.

---

### Case A3: Buggy Trace — Misconception `M07` (Ghost Semicolon / Unchanged Return)

Learner traces the true branch to `n3`, but predicts `0` (assuming `shield` did not update).

| Field | Input / Action |
|---|---|
| **Path Selection** | Tap nodes: `n1` → `n2` → `n3` → `n4` |
| **Predicted Output** | Click `0 (shield off / 0)` |
| **Reasoning (Optional)** | `Thought the block did not update the shield value.` |

#### Expected System Response:
* **Gate & Tests**: `Passed: 0 / 1` (path was correct, but predicted output was incorrect `0`)
* **Diagnosis Screen**: Flags `M07` evidence and guides learner to intervention.

---
---

## Level B: Astronaut ID Search · Voice Algorithm Trace

* **Planet**: Miner's Belt / Mars (`/planet/loops`)
* **Node on Map**: `astronaut_search` (3rd node on belt, replacing `fire_shots`)
* **Direct URL**: [http://localhost:5173/planet/loops/mission/MMB-01](http://localhost:5173/planet/loops/mission/MMB-01)
* **UI Theme**: Blue & Yellow Pixel UI (`deep-space` #0c1937, `void-blue` #153268, `starlight` #fcb143, `warp-cyan` #2de7fc) with `"Press Start 2P"` and `"VT323"` typography.
* **Question / Prompt**:
  > *"A space station has 100 identification cards belonging to different astronauts, arranged randomly. You need to find the card belonging to Commander Rahul. Explain how you would find it."*
* **Underlying Logic**:
  ```c
  // Search 100 astronaut cards arranged randomly
  int find_commander_rahul(AstronautCard cards[100]) {
      for (int i = 0; i < 100; i++) {
          if (is_commander_rahul(cards[i])) {
              return i; // Found Rahul's card
          }
      }
      return -1; // Not found
  }
  ```

---

### Case B1: Correct Algorithm Explanation (Passes)

| Field | Input / Action |
|---|---|
| **Speech / Text** | Hold <kbd>SPACE</kbd> or click **Hold Talk** and say:<br>*"Because the 100 cards are arranged randomly and unsorted, you must use linear search. Inspect the cards one by one from the first card (index 0) to the last card (index 99), checking each card to see if it belongs to Commander Rahul, and stopping as soon as his card is found."*<br>*(Or type into manual text mode: `Since the cards are arranged randomly, I will use linear search and inspect cards one by one from the first to last, stopping when Commander Rahul is found.`)* |
| **Hedges** | 0 hedge words |

#### Expected System Response:
* **Gate & Tests**: `Passed: 1 / 1`
* **Card Scan Iteration Gauge**:
  * Visual dial animates across search iterations
  * Droid Report: *"The 100 astronaut cards are arranged randomly, requiring a sequential linear search from index 0 to 99 until Commander Rahul is located."*
* **Diagnosis Status**: `correct`
* **UI**: Green checkmark on `SignalGateWorld`. Clicking **Continue** returns to orbit with `astronaut_search` marked as passed.

---

### Case B2: Buggy Explanation — Misconception `M01` (Boundary Drift / False Binary Search on Unsorted Data)

Learner attempts binary search on unsorted/random cards, or claims checking stops 1 card short.

| Field | Input / Action |
|---|---|
| **Speech / Text** | Say or type:<br>*"I will use binary search and split the cards in half repeatedly to find Commander Rahul."*<br>*(Or: `"Check cards up to 99 but skip the last one."`)* |
| **Hedge Detection** | Detects cognitive uncertainty markers |

#### Expected System Response:
* **Gate & Tests**: `Passed: 0 / 1`
* **Diagnosis Screen**:
  * Top Misconception: **`M01` — Boundary Drift** (`p ~ 0.75`, Band: `Likely`)
  * Evidence: `[RUN] The 100 cards are arranged randomly (unsorted); you cannot use binary search without sorting. Sequential linear search must inspect from index 0 through 99.`
* **Next Steps**: Continues to Intervention (`M01`) → Ambiguity Probe → Transfer Trap (`MMB-01T`).

---

### Case B3: Buggy Explanation — Misconception `M02` (Infinite Horizon / Stalled Loop)

Learner believes the search loops indefinitely and never stops.

| Field | Input / Action |
|---|---|
| **Speech / Text** | Say or type:<br>*"The search runs forever and never terminates because the cards are random."* |

#### Expected System Response:
* **Gate & Tests**: `Passed: 0 / 1`
* **Diagnosis Screen**:
  * Top Misconception: **`M02` — Infinite Horizon** (`p ~ 0.86`, Band: `Likely`)
  * Subtitle: *"Loop condition never becomes false"*
  * Evidence: `[RUN] The search halts immediately once Commander Rahul is matched, or upon checking all 100 cards.`

---

## Quick Verification Checklist

| Step | Action | Expected Visual |
|---|---|---|
| **1. Planet Map (Conditions)** | Go to [http://localhost:5173/planet/conditions](http://localhost:5173/planet/conditions) | Node `airlock_trace` at position 3 |
| **2. Test A1 (Correct)** | Click `airlock_trace`, tap `n1-n2-n3-n4`, click `1` | Tests pass 1/1, return to map |
| **3. Test A2 (Bug M06)** | Click `airlock_trace`, tap `n1-n2-n4`, click `0` | Divergence at `n2` shown, diagnoses `M06` |
| **4. Planet Map (Loops / Mars)** | Go to [http://localhost:5173/planet/loops](http://localhost:5173/planet/loops) | Node `astronaut_search` at position 3 |
| **5. Test B1 (Correct)** | Click `astronaut_search`, speak/type linear search algorithm | Gauge evaluates, tests pass 1/1 |
| **6. Test B2 (Bug M01)** | Click `astronaut_search`, speak/type `"use binary search and split in half"` | Diagnoses `M01`, shows unsorted random card rule |
