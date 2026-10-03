# 03 — ML IMPLEMENTATION PLAN v3 (complete, agent-ready, incl. DSA "Deep Space Trials")

> **Replaces `03_ML_SPEC.md` and `03_ML_IMPLEMENTATION_PLAN_v2.md`.** Owner: ML teammate. Build window: **12 hours** with AI coding agents.
> Companion files: `01_SCOPE_v3.md` (tiers, hour plan), `02_DESIGN_SCOPE_v3.md` (screens that consume this API).
> **v3 adds:** 8 DSA misconception classes (D01–D08), 18 basic DSA problems (Searching, Sorting, Array techniques, Strings, Recursion), interpreter support for strings + recursion tracing, 4 DSA twin sets + 6 probes, DSA traps, and an **adaptive exam engine** (§8.5) whose results also act as the final ghost return for main-game misconceptions. Main game: 3 playable planets (Conditions, Loops, Arrays).
> **Rule for agents:** class IDs, feature names, JSON fields and file paths in this document are contracts. Don't rename them.
>
> **Accuracy stance:** high accuracy is *not* the goal and is *not* expected. The goal is a **correctly evaluated, honestly reported** diagnosis loop. A synthetic score ≥ 0.98 is treated as a **red flag** (leakage or template memorisation), not a win.

---

## 0. One-page summary

### 0.1 What changed from v1 and why

| v1 | v2 / v3 | Reason |
|---|---|---|
| LightGBM on AST + trace + **simulated prediction** + interaction + **history** features | LightGBM on **AST + trace + output-relation + fix-probe** features only | Simulated predictions are generated from the label, so they leak it. History is simulated too and self-reinforcing |
| Twins T1–T4, mostly separable by one flag | **HARD twin** (code-identical, soft-labelled) + 3 STRUCTURAL twins | Makes the model's ambiguity real and the probe necessary |
| Probe likelihoods "estimated from belief-agent simulation" | **Hand-set, documented likelihoods** (3 parameters) + **sensitivity sweep** in evaluation | Can't fit them without real learners; say so and test robustness |
| Beta(α, β) + decay + 4 hard conditions | **2-state knowledge model** (BKT-style) with **item-specific guess/slip**; trap items carry the weight | Explains "lucky guesser" and "pattern copier" directly; one Bayesian engine end to end |
| 10 classes, 24 problems, 15 probes, 250-person human set, LLM personas | **17 classes (9 main + 8 DSA), 34 problems (16 main + 18 DSA), 14 probes, 100-item blind-authored realistic set, ITSP real-student slice, no LLM** | 12-hour build incl. the DSA level |
| 1 held-out class | **LOCO** (leave-one-class-out) over all 17 + 2 never-trained classes | One class = one noisy number |
| Template-family split | Problem split + **operator-variant holdout** + **cross-domain transfer** (main → DSA) | Families ≈ problems; operator holdout tests unseen *surface forms*; cross-domain tests whether intro misconceptions are recognised inside algorithms |
| — (v3) | **Adaptive DSA exam**: items chosen by expected information gain over the learner's misconception states + coverage + target difficulty; doubles as the final ghost return | Judges asked for DSA; adaptive assessment is an ML contribution, not just more content |
| — | **Gate** for empty / Python / hard-coded / unsupported input; **kNN novelty** | Outliers must not get confident labels |
| — | **Verified minimal fixer** (inverse operators) reused as **features** and for **two-bug detection** | Table stakes vs Da GOATS + strong problem-agnostic signal |

### 0.2 Architecture

```
                         ┌──────────────── offline ────────────────┐
problems/*.json ─► generate/ (operators → verify → soft labels → dedupe) ─► data/dataset.jsonl
                                                    │                    ─► eda/eda.py → docs/eda.md
                                                    ▼
                               features/extract.py ─► model/train.py (LightGBM, grouped CV)
                                                    ─► calibrate.py (temperature) ─► novelty.py (τ, kNN)
                                                    ─► artifacts/diagnoser_<hash>/ ─► eval/run_all.py → metrics.json
                         └──────────────────────────────────────────┘
online (FastAPI):
code ─► GATE ─► c_interp (trace, tests) ─► features ─► LightGBM ─► temperature ─► p_code(k)
                                                                                    │
           learner prior (capped) ───────────────────────────────────────────────► Bayes layer ◄── predictions / MCQ / probe answers
                                                                                    │                     (hand-set likelihoods)
                                                                     status: confident | ambiguous(+EIG probe) | novel | two_bug
                                                                                    │
                                     intervention engine (timeline, counterexample, verified fix)
                                                                                    │
                                     knowledge model (2-state, guess/slip) ─► state machine ─► STABLE / MASTERED / RELAPSED
                                                                                    ▲                    │
DSA exam:  /exam/start ─► exam selector (EIG over P(A_k) + sector coverage + Elo difficulty) ─► item ─► /exam/answer
           (diagnosis computed silently per item, fed into the knowledge model) ─► /exam/finish ─► debrief report
           (findings NEW / HELD→MASTERED / RELAPSED, deferred probes, "why this item" log)
```

### 0.3 MVP targets (goals to measure against, not promises)

| Metric | Slice | Target | Red flag |
|---|---|---|---|
| macro-F1 | grouped 5-fold CV by problem | ≥ 0.80 | ≥ 0.98 |
| macro-F1 | 3 unseen problems | ≥ 0.70 | 1.00 |
| macro-F1 | realistic set R-blind (headline) | ≥ 0.60 | — |
| pair accuracy | STRUCTURAL twins | ≥ 0.85 | — |
| ambiguity-flag rate | HARD twin items | ≥ 0.80 | < 0.3 (model is guessing confidently) |
| AUROC | LOCO novelty (mean over classes) | ≥ 0.70 | — |
| ECE | OOF after temperature | ≤ 0.08 | — |
| false-resolve rate | simulated pattern-copiers | ours ≤ ½ × naive | — |
| invariance | semantics-preserving perturbations | ≥ 0.90 unchanged | — |
| macro-F1 (D-classes only) | grouped CV, DSA problems | ≥ 0.75 | ≥ 0.98 |
| macro-F1 (D-classes only) | 3 held-out DSA problems | ≥ 0.65 | 1.00 |
| M-class recall inside DSA problems | **cross-domain** (trained on main problems only) | ≥ 0.50 | — |
| profile-recovery F1 per item | adaptive exam vs fixed exam (simulation) | adaptive > fixed by ≥ 10% relative | — |

---

## 1. Problem-statement requirement → ML component (model draft per scope item)

| # | PS requirement | What we build | Model / method | Inputs → outputs | Evaluated by | MVP cut |
|---|---|---|---|---|---|---|
| 1 | **Misconception Dataset** | Verified synthetic corpus (mutation operators over correct variants, main + DSA) + hard negatives + soft-labelled ambiguity groups + blind-authored realistic set + ITSP real-student slice | Rule-based generator + execution verifier | problems → `dataset.jsonl` (~9k rows), `realistic_*.csv` (100), `itsp_slice.jsonl` | Dataset card, EDA, effective-N | ITSP slice is Strong |
| 2 | **Misconception Model** | Diagnoser | **LightGBM multiclass** (19 outputs) on 5 feature groups, temperature-calibrated, structural class masking | code + trace → `p_code(k)` | E1–E4, E7, E8, E9, E15 | — |
| 3 | **Misconception Differentiation** | Twin handling | Structural twins: features. HARD twins: **soft labels** → honest ~50/50 → **Bayes layer + EIG probe** | `p_code` + probe answers → posterior | E5 (code-only vs + probe, p_b sweep) | 14 probes |
| 4 | **Adaptive Intervention** | Intervention engine | Rule policy: class → modality; **counterexample search** + **verified minimal fixer** + trace-timeline builder | class + learner code + trace → intervention package | Fixer coverage % on realistic set; demo | 2nd modality Strong |
| 5 | **Resolution Assessment** | Reassessment | **2-state knowledge model** per misconception, item-specific guess/slip; trap items (belief ≠ correct); ghost return | item results → `p_active`, state, unmet conditions | E10 simulated learners vs naive and 4-check policies | Ghost UI Strong (engine Core) |
| 6 | **Learner Model** | Per-learner store | Knowledge states + attempt log + capped prior into the Bayes layer | events → `/learner/{id}` | Demo + E10 | — |
| 7 | **Model Evaluation (incl. unseen)** | Eval suite | Grouped CV, unseen problems, operator holdout, realistic set, LOCO novelty, U1/U2 unseen classes, calibration, baselines, ablation, robustness, failure audit, cross-domain transfer | → `metrics.json` + PNGs | Lab Report (D10) | LLM baseline optional |
| 8 | **DSA "Test Yourself" (judges' requirement)** | Deep Space Trials | Same diagnoser on DSA problems + **adaptive item selector** (EIG over knowledge states, sector coverage, Elo-style difficulty) + debrief generator | exam answers → silent diagnoses → knowledge updates → report | E14 adaptive vs fixed vs random exam (simulation); E4/E15 on DSA items | Fixed blueprint fallback (same API) |

**Why LightGBM and not a fine-tuned code model (CodeBERT etc.):** ~2k effective unique programs from 34 problems. A text model memorises templates (Da GOATS' own TF-IDF result: 0.49–0.57 macro-F1 on unseen problems). Hand-built **problem-agnostic** features + trees generalise across problems, run in < 5 ms on CPU, and explain themselves through SHAP contributions. A code LM is a roadmap item once real student data exists.

---

## 2. Execution engine (`ml/c_interp/`) — shared by game and ML

### 2.1 Frozen C subset
**Supported:** `int`, `float`, `double` (treated as float), `char` (as int, char literals), 1D arrays (`int a[5]`, `int a[] = {..}`, array params), `+ - * / %`, `+= -= *= /= %=`, `++/--` (pre/post), relational, `&& || !` (**short-circuit**, as in C: the right operand is not evaluated when the left one decides the result), ternary, casts `(int)` `(float)` `(double)`, `if/else`, `for` (C99 decl in init), `while`, `do-while`, `break`, `continue`, `return`, user functions (scalars by value, **arrays by reference**, as in real C), **recursion (depth cap 100)**, **`char` arrays and string literals** (`char s[] = "level";`, `'\0'` terminator, `char s[]` params, `printf("%s")`), builtin **`strlen`** (unless the problem lists it in `forbid`), `printf` (`%d %i %f %.Nf %c %s %% %lf %ld`), `#define NAME literal` (textual substitution), `#include` lines (stripped).
**World builtins** (the game API): `fire()`, `launch()`, `open_door()`, `close_door()`, `scan(int x)`. Each call appends a world effect.
**Rejected (gate G4):** pointers (`*p`, `&x`, `char *s`), `struct`, `malloc`, `<string.h>` functions other than `strlen` (`strcmp`, `strcpy`…), `scanf` *(Strong: allowed only for the ITSP slice via an input queue)*, `goto`, multi-dim arrays, `switch` *(add only if time)*.

### 2.2 Deterministic semantics and events

| Situation | Behaviour | Event (with `line`) |
|---|---|---|
| Read of uninitialised local | returns `GARBAGE = -858993460` | `uninit_read {var}` |
| Array read out of bounds | returns `GARBAGE` | `oob_read {arr, idx, size}` |
| Array write out of bounds | ignored | `oob_write {arr, idx, size}` |
| int / int | C truncation toward zero | `intdiv {remainder_nonzero, into_float}`; `into_float` is true when the result is assigned to / returned as float |
| Division by zero | halt, status `runtime_error` | `div_zero` |
| Condition expression is an assignment | evaluates normally | `assign_in_cond {var, value}` |
| Body is `EmptyStatement` (stray `;`) | executes nothing | `empty_body {kind: if/for/while}` |
| Each loop iteration | counter `loop_iters["L<line>"] += 1` | — |
| Each `if` | `branch["B<line>"] = {true: n, false: m}` | — |
| > 5,000 executed statements | halt, status `timeout` | `step_cap_hit` |
| Non-void function falls off the end | returns `GARBAGE` | `missing_return` |
| int overflow | wrap to 32-bit | `overflow` |
| Array param read (in tasks with array input) | — | world effect `read_cell {i}` or `read_void {i}` if OOB |
| Array param **write** (v3) | performed (shared with the caller) | world effect `write_cell {i, v}` |
| Relational compare of two array cells (v3) | normal | world effect `compare {i, j}` (drives ArrayBars flashes) |
| String literal `"…"` initialising `char s[]` (v3) | list of char codes + `0` terminator | — |
| `==`/`!=` with a string literal or two array names (v3) | evaluates to **false** / true (address comparison, as real C would) | `str_literal_compare {line}` / `array_compare {line}` |
| Function call / return (v3) | push / pop frame | world effects `call {fn, depth, args}`, `ret {fn, depth, value}`; counter `max_depth` |
| Recursion deeper than 100 (v3) | halt, status `timeout` | `depth_cap_hit {fn, last_args}` |
| Recursive call result unused (v3) | normal | `discarded_call_value {fn, line}` (an `ExprStmt` whose expression is a call to a non-void user function) |
| Call to `strlen` in a problem whose `forbid` lists it (v3) | not executed | gate G4b (see §3.7.1) |

### 2.3 Harness and tests
A problem defines the entry function. The harness parses the learner file, calls the entry function with each test's `args`, and records `returned`, `printed` and world-effect counts. A test passes if every key in `expect` matches (floats ±1e-3).
**v3:** Python `str` args become `char[]` with a terminator; `expect` may contain `"array0": [...]` (final state of argument 0 after the call, for in-place sorts/reverses), `"max_depth_le": 30` (efficiency guard for recursion). Each problem marks 2 tests as `"sample": true`; `/run` with `sample_only` runs only those (exam Run button).

### 2.4 Trace JSON (contract; consumed by FE arena and by features)
```json
{
  "status": "ok|timeout|runtime_error|parse_error|unsupported",
  "returned": 26, "printed": "",
  "steps": [{"i": 0, "line": 4, "vars": {"i": 0, "total": 0}, "events": [], "effects": ["read_cell:0"]}],
  "loop_iters": {"L4": 4}, "branch": {"B6": {"true": 3, "false": 1}},
  "events": [{"type": "oob_read", "line": 5, "arr": "cells", "idx": 3, "size": 3}],
  "effects_count": {"fire": 0, "read_cell": 3, "read_void": 1, "write_cell": 0, "compare": 0, "call": 1},
  "max_depth": 1,
  "truncated": false,
  "per_test": [{"status": "ok", "returned": 26, "printed": "", "loop_iters": {"L4": 4}, "branch": {}, "effects_count": {"read_cell": 3, "read_void": 1}, "max_depth": 1}]
}
```
DSA traces use the same shape: ArrayBars replays `read_cell / write_cell / compare` effects with marker variables (`problem.markers`, e.g. `["low","mid","high"]`) taken from `steps[].vars`; WarpStack replays `call / ret`; SignalTiles replays `read_cell` on `char[]` params.
Steps are recorded for **one display test** (the problem's `display_test`); counters and events are aggregated for all tests in `tests[]`. Cap 2,000 recorded steps (`truncated: true`).
`per_test` is one object per test, in test order: `{status, returned, printed, loop_iters, branch, effects_count, max_depth}`. The same-named fields on the trace stay the **sums** over tests (`max_depth` stays the **maximum**) so the frontend keeps reading them. Features that say "on every test" (`b_iter_delta_const_pm1`, `b_branch_always`, `b_branch_never`, `b_return_first_iter`) read `per_test`, never the sums.

### 2.5 Implementation notes
- `pycparser.CParser().parse(src)`. Preprocess: normalise smart quotes, **replace `//` and `/* */` comments with spaces (keep the newlines, so line numbers stay)**, strip `#include`, apply `#define`, reject other `#` lines. pycparser rejects comments, and the `comments` augmentation plus the realistic-set brief both add them. pycparser doesn't need prototypes for `printf` or builtins (implicit declarations parse fine).
- Tree-walking evaluator: one method per node type (`FileAST, FuncDef, Decl, TypeDecl, ArrayDecl, InitList, Compound, Assignment, BinaryOp, UnaryOp, Constant, ID, ArrayRef, FuncCall, If, For, While, DoWhile, Break, Continue, Return, EmptyStatement, Cast, TernaryOp, ExprList`). Use Python exceptions for `Break/Continue/Return`. Expect ~500–700 lines.
- Values carry a type tag (`int|float`). int ops wrap at 32 bits. Arrays are Python lists of tagged values, shared by reference when passed.
- Every node has `coord.line`; keep it for events, evidence and fixer edits.
- **Unit tests** (`c_interp/tests/`): one per row of §2.2 + all correct variants of all problems pass their tests. Optional: differential check against `gcc` for defined-behaviour programs if gcc is installed.
- **Performance budget:** a problem's whole test suite in < 30 ms (sorting tests use n ≤ 8; recursion n ≤ 12). Fix-probe features are **not** model inputs (§4.4). After the model, fixers run for the **top-3 classes only**, ≤ 5 candidates each, and stop at the first candidate that passes. Worst case before the cache: 3 × 5 × 30 ms = 450 ms. The p95 target for `/attempt` is 300 ms, which holds on a cache hit and on the common case where the first candidate of each class passes. Cache by AST hash.
- **gcc differential:** compare the interpreter with gcc only for programs whose trace has none of `uninit_read`, `oob_read`, `overflow`, `str_literal_compare`. Those are undefined behaviour in real C, so a mismatch there is not an interpreter bug.
- **v3 extra unit tests:** string literal init + terminator; `s[i] == "a"` → false + event; in-place sort visible to the caller; `depth_cap_hit` on missing base case; `discarded_call_value`; `strlen` forbid.

---

## 3. Data

### 3.1 Taxonomy v3 (9 main + 8 DSA classes)

| ID | Name | Typical wrong C | Wrong belief (used for probe/trap answers) | Defining evidence |
|---|---|---|---|---|
| M01 | LOOP_BOUND_OFF_BY_ONE | counting loop `i<=n`, `i<n-1`, start `1` in a counting loop, countdown `i>=0` | "`i<=n` (from 0) runs n times"; "the bound is the last value excluded/included" mixed up | iteration delta ±1 vs reference, no OOB |
| M02 | LOOP_NO_PROGRESS | missing `i++`, `i--` with `<`, update on wrong var, update inside an `if` | "the loop variable advances by itself / the loop ends by itself" | `step_cap_hit`, no update or wrong direction |
| M03 | ACCUMULATOR_RESET | `int t=0;` or `t=0;` inside the loop body | "the total keeps its value across passes even if re-set" | output = last element's contribution |
| M04 | INT_DIVISION | `float avg = sum / n;`, `(float)(a/b)`, `x / 2` for decimals | "int / int gives the exact decimal" | `intdiv` with `remainder_nonzero` and `into_float`; output = floor(ref) |
| M05 | UNINITIALIZED_VAR | `int total;` then `total += …` | "local variables start at 0" | `uninit_read`, garbage output |
| M06 | ASSIGN_IN_CONDITION | `if (x = 5)`, `while (k = 0)` | "`=` compares" | `assign_in_cond`, branch always/never taken |
| M07 | STRAY_SEMICOLON | `if (c);`, `for (…);`, `while (…);` | "a `;` after the header is harmless" | `empty_body`; body runs once / always, or step cap on `while` |
| M08 | ARRAY_INDEX_BASE | `a[n]`, `i=1; i<=n` with `a[i]`, `a[i+1]`, `a[1]` as the first | "arrays are indexed 1..n" | `oob_read` at idx = n, first element skipped |
| M10 | PRINT_NOT_RETURN | `printf("%d", r);` instead of `return r;` | "printing the value hands it back" | `printed` equals the reference return value; returned garbage/0 |
| **D01** | SEARCH_EARLY_EXIT | `if (a[i]==x) return i; else return -1;` inside the loop; `if (a[i]==x) found=1; else found=0;` | "the else branch means 'not this one, keep looking'; it doesn't end or overwrite anything" | function returns during iteration 1 (`b_return_first_iter`); result depends only on the first / last element (`r_eq_first_only`, `r_eq_last_only`) |
| **D02** | BSEARCH_NO_SHRINK | `low = mid;` / `high = mid;` | "mid is excluded from the window automatically" | `step_cap_hit` with `low`/`high` frozen across iterations (`b_window_frozen`) |
| **D03** | SWAP_OVERWRITE | `a[i] = a[j]; a[j] = a[i];`; rotate with `a[i] = a[i+1]` and the first value never saved | "two assignments happen at the same time" | output multiset ≠ input multiset (a value duplicated, one lost) |
| **D04** | SINGLE_PASS_SORT | missing outer loop; outer loop `i < 1`; selection sort that only places the minimum | "one pass over the array sorts it" | output = reference's *single pass* on the input (`r_eq_one_pass`); max at the end, rest unsorted |
| **D05** | MISSING_BASE_CASE | no base case; unreachable base (`if (n == 0)` with `f(n-2)` on odd n) | "recursion stops by itself when the work is done" | `depth_cap_hit`; no base-case `return` ever executed |
| **D06** | RECURSION_NO_SHRINK | `return n * f(n);`, `f(n+1)`, `sum_digits(n % 10)` | "the recursive call moves toward the base case on its own" | `depth_cap_hit` with **constant or growing** argument across frames |
| **D07** | RECURSIVE_RESULT_DISCARDED | `f(n-1); return n;` / `sum(a, n-1); return a[n-1];` | "the recursive call's result is combined automatically" | `discarded_call_value`; output = top frame's own contribution only |
| **D08** | STRING_EQ_COMPARE | `s[i] == "a"`, `if (s == t)` | "`==` compares text" | `str_literal_compare` / `array_compare`; branch never taken |
| CORRECT | — | passes all tests (incl. near-miss styles) | — | — |
| OTHER | non-targeted bug | wrong operator, swapped operands, wrong variable, wrong constant | — | fails tests, no class predicate |
| *M09 (Strong)* | PASS_BY_VALUE | `void boost(int s){ s += 10; }` expecting caller change | "parameters alias the caller's variable" | caller scalar unchanged, param written |
| **U1 (unseen)** | CHAINED_COMPARISON | `if (0 < x < 10)` | "`a < x < b` checks a range" | test only; **no feature is designed for it** |
| **U2 (unseen)** | OR_CHAIN | `if (code == 42 \|\| 7)` | "`\|\|` applies the `==` to both sides" | test only |

**Rule:** no feature may target U1/U2 (that would make the unseen test fake).
**Literature grounding:** classes M01–M10 all appear in published novice-error catalogues (Sorva 2012; Qian & Lehman 2017; Ettles et al. 2018). DSA classes: recursion mental-model errors such as missing/unreachable base cases and "looping" models of recursion (Götschi, Sanders & Galpin 2003), and algorithm/data-structure misconceptions incl. search and sort errors (Danielsiek, Paul & Vahrenhold 2012). Cite these in the dataset card.

**Where main-game classes show up inside DSA problems** (this is what the exam rechecks): M01/M08 in pair loops (`j < n` with `a[j+1]`), mirrors (`a[n-i]`), reverse bounds (`i < n` instead of `n/2` reverses twice); M02 in two-pointer loops (one pointer never moves); M03/M05 in counters and recursive accumulators; M06 in search conditions; M07 after `for` in sorts; M10 in `is_sorted`/`is_palindrome` that print instead of return.

### 3.2 Twin sets v3

| Set | Type | Members | Shared symptom | What separates them |
|---|---|---|---|---|
| **T1** | **HARD (code-identical)** | M01 ⟷ M08 | `for(i=0;i<=n;i++) s+=a[i];` reads `a[n]`; `for(i=1;i<n;i++)` skips `a[0]` | **Nothing in the code.** Soft label 0.5/0.5 → model reports ambiguity → probe P_T1_a/b |
| T2 | STRUCTURAL | M02 ⟷ M07 | `while` never ends (reactor overheats) | missing/backwards update vs `;` after `while(...)` |
| T3 | STRUCTURAL | M06 ⟷ M07 (on `if`) | branch always runs | `=` in condition (variable overwritten) vs empty body |
| T4 | STRUCTURAL | M03 ⟷ M05 | wrong total | output = last element vs `uninit_read`/garbage |
| *T5 (Strong)* | STRUCTURAL | M09 ⟷ M10 | right value printed inside the function, caller sees nothing | param write vs `printf` replacing `return` |
| **T6** (v3) | STRUCTURAL | D02 ⟷ M02 | search loop never ends (reactor overheats) | `low = mid`/`high = mid` vs no update at all |
| **T7** (v3) | STRUCTURAL | D05 ⟷ D06 | warp overflow (`depth_cap_hit`) | no reachable base case vs argument doesn't shrink |
| **T8** (v3) | STRUCTURAL | D07 ⟷ M10 | right work done, wrong value comes back | discarded recursive value vs `printf` instead of `return` |
| **T9** (v3) | STRUCTURAL | D01 ⟷ M03 | result depends only on one element | `else return/overwrite` inside the loop vs accumulator reset |

Non-ambiguous M01 forms (counting loops, no array) and non-ambiguous M08 forms (`a[n]` literal, `a[i+1]`, `a[1]` as first) are hard-labelled. **Only array-traversal loops with a shifted bound/start are soft-labelled.** v3: the same rule covers DSA surfaces: the bubble-sort pair loop `for (j = 0; j < n; j++) if (a[j] > a[j+1])` and the linear search `for (i = 0; i <= n; i++)` are soft-labelled M01/M08 too (operators `amb_pair_bound`, `amb_le_array`). This is the honest version of the "twin" claim: *where code differs, the model separates; where code is identical, the model asks.*

### 3.3 Problem bank (16 main + 18 DSA + 1 Strong)

#### 3.3.1 Main game (planets)

| ID | Function | World | Family | Planet | Targets | Split |
|---|---|---|---|---|---|---|
| P01 | `void fire_shots(int n)` → `fire()` ×n | cannon | count_loop | loops | M01 M02 M07 | train |
| P02 | `void countdown(int n)` prints n..1 then `launch()` | launchpad | countdown_loop | loops | M01 M02 M07 | train |
| P03 | `int total_energy(int cells[], int n)` | drone bay | array_accumulate | loops | M01 M03 M05 M07 M08 T1 | train (demo) |
| P04 | `int count_overheated(int t[], int n, int limit)` | reactor | array_count_if | loops | M01 M03 M05 M06 M08 | **held out** |
| P05 | `int charge_steps(int level, int target)` (+7 per step) | charger | while_progress | loops | M02 M05 M06 M07 | train |
| P06 | `int power_up(int base, int k)` (base^k) | amplifier | accumulate_product | loops | M01 M02 M03 M05 | train |
| P07 | `float avg_fuel(int tanks[], int n)` | fuel gauge | average | arrays | M03 M04 M05 M08 | train |
| P08 | `int last_beacon(int ids[], int n)` | beacon | index_access | arrays | M08 M10 | train |
| P09 | `int max_shield(int s[], int n)` | shield grid | array_max | arrays | M01 M05 M06 M08 | **held out** |
| P10 | `int sum_first_k(int a[], int n, int k)` | cargo | prefix_loop | arrays | M01 M03 M08 | train |
| P11 | `int door_open(int code)` → 1 iff code == 42 | door | equality_check | conditions | M06 M07 M10 | train |
| P12 | `int shield_mode(int energy)` (<30→0, <70→1, else 2) | shield | branch_bands | conditions | M06 M07 | train |
| P13 | `float fuel_percent(int fuel, int cap)` | gauge | ratio | variables | M04 M10 | **held out** |
| P14 | `int distance(int a, int b)` (absolute difference) | radar | return_value | functions | M10 M06 | train |
| P16 | `int in_range(int x, int lo, int hi)` → 1 iff lo < x < hi | shield gate | range_check | conditions | M06 M07 OTHER (U1 shows up here naturally; never trained) | train |
| P17 | `int max_of_three(int a, int b, int c)` | radar | nested_branch | conditions | M05 M06 M07 M10 | train |
| *P15* | `void boost(int shield)` + caller check | ship module | pass_by_value | functions | M09 | Strong |

Each problem file `ml/problems/Pxx_<name>.json`:
```json
{
  "problem_id": "P03", "name": "total_energy", "planet": "loops", "world": "drone_bay",
  "family": "array_accumulate", "split": "train",
  "prompt": "Pull energy from every cell and return the total.",
  "signature": "int total_energy(int cells[], int n)",
  "starter": "int total_energy(int cells[], int n) {\n    // TODO\n}",
  "correct_variants": ["...for...", "...while...", "...alt style (index from n-1 down)..."],
  "tests": [
    {"args": [[2,4,6], 3], "expect": {"returned": 12}},
    {"args": [[5], 1], "expect": {"returned": 5}},
    {"args": [[1,2,3,4], 4], "expect": {"returned": 10}},
    {"args": [[0,0,9], 3], "expect": {"returned": 9}, "adversarial": "M03 passes by luck"},
    {"args": [[7,1], 2], "expect": {"returned": 8}}
  ],
  "display_test": 0,
  "predict_item": {"item_id": "PR_P03", "code": "...", "question": "How many cells are read?", "options": ["3","4","2"], "correct": "4", "belief": {"M01": "3", "M08": "3"}},
  "allowed_ops": ["m01_*", "amb_*", "m03_*", "m05_*", "m07_for_semi", "m08_*", "oth_*"]
}
```
Rules: ≥ 3 correct variants (for / while / different style); 4–6 tests incl. **≥ 1 adversarial test where some misconception passes by luck** (so "tests pass" never equals "learned"); every correct variant must pass all tests (CI check).

**v3 problem fields** (all problems): `"sector"` (DSA only: `searching|sorting|array_tech|strings|recursion`), `"difficulty"` (1–3), `"markers"` (vars shown under ArrayBars, e.g. `["low","mid","high"]`), `"forbid"` (e.g. `["strlen"]`), `"exposure"` (class → probability that a learner with that misconception fails this problem *with that class's signature*; authored 0.7 for primary targets, 0.4 for secondary, verified in §10.12), tests with `"sample": true` (2 per problem).

**Held-out problems** P04, P09, P13 (main) and Q04, Q09, Q16 (DSA) are chosen so that every class still appears in ≥ 2 training problems and each held-out problem has a *different* family from all training problems. (Held-out problems still appear in the *game*, e.g. P09 as a ghost node, Q-problems in the exam; "held out" only means "never used to train or select the model".)

#### 3.3.2 DSA problems — Deep Space Trials (18)

| ID | Function | Sector | Family | Diff | Targets (exposure ≥ 0.4) | Split |
|---|---|---|---|---|---|---|
| Q01 | `int linear_search(int a[], int n, int x)` → index or −1 | searching | linear_search | 1 | D01 AMB(M01/M08) M06 M07 | train |
| Q02 | `int count_occurrences(int a[], int n, int x)` | searching | count_match | 1 | M03 M05 M06 M08 | train |
| Q03 | `int binary_search(int a[], int n, int x)` (sorted) → index or −1 | searching | binary_search | 3 | D02 M01 M02 | train |
| Q04 | `int contains(int a[], int n, int x)` → 1/0 via a `found` flag | searching | flag_search | 1 | D01 M05 M06 | **held out** |
| Q05 | `int count_guesses(int secret, int lo, int hi)` (halving guesses) | searching | guess_halving | 2 | D02 M02 M04 | train |
| Q06 | `void bubble_sort(int a[], int n)` (in place) | sorting | bubble_sort | 2 | D03 D04 AMB(pair) M07 | train |
| Q07 | `void selection_sort(int a[], int n)` | sorting | selection_sort | 3 | D03 D04 M01 M05 | train |
| Q08 | `int is_sorted(int a[], int n)` → 1/0 | sorting | pairwise_check | 1 | D01 AMB(pair) M06 M10 | train |
| Q09 | `int second_largest(int a[], int n)` | array_tech | running_max2 | 2 | M05 M08 OTHER | **held out** |
| Q10 | `void reverse(int a[], int n)` (in place) | array_tech | two_pointer_swap | 2 | D03 M01 M08 | train |
| Q11 | `void rotate_left(int a[], int n)` (by one) | array_tech | shift | 2 | D03 M08 AMB(pair) | train |
| Q12 | `int pair_sum_exists(int a[], int n, int target)` (sorted, two pointers) | array_tech | two_pointer_scan | 3 | M02 M01 D01 | train |
| Q13 | `int str_length(char s[])` (no `strlen`) | strings | string_scan | 1 | M01 M02 M05 | train |
| Q14 | `int count_vowels(char s[])` | strings | string_count | 1 | D08 M03 M05 M06 | train |
| Q15 | `int is_palindrome(char s[])` | strings | string_two_pointer | 2 | M08 D01 D08 M10 | train |
| Q16 | `int sum_digits(int n)` (recursive) | recursion | rec_digits | 2 | D05 D06 D07 | **held out** |
| Q17 | `int factorial(int n)` (recursive) | recursion | rec_product | 1 | D05 D06 D07 | train |
| Q18 | `int array_sum_rec(int a[], int n)` (recursive) | recursion | rec_array | 2 | D05 D06 D07 M08 | train |

Coverage check (CI): every D-class has ≥ 2 training problems (D01: Q01 Q08 Q15 · D02: Q03 Q05 · D03: Q06 Q07 Q10 Q11 · D04: Q06 Q07 · D05–D07: Q17 Q18 · D08: Q14 Q15). Recursion problems must state "use recursion" and the gate rejects a loop-only solution (G5b) so recursion classes stay meaningful. `Q13` forbids `strlen`.

### 3.4 Data sources and availability

| Code | Source | Size (target) | Labels | Used for | Train? |
|---|---|---|---|---|---|
| **A** | Mutation generator over correct variants, main + DSA (§3.5) | ~6,000 rows after dedupe (~2,800 main, ~3,200 DSA) | operator → class (verified) | train / CV / unseen-problem test | ✔ |
| **E** | Hard negatives (near-miss CORRECT), two-bug compositions, weird-but-valid OTHER | ~2,200 | constructed | train / eval slices | ✔ (two-bug as soft) |
| **AMB** | Ambiguity groups (T1 incl. DSA pair loops) | ~400 | soft 0.5/0.5 | train + twin eval | ✔ (weighted) |
| **R-blind** | 40 items hand-written by **FE** (has not seen operators): 25 main + 15 DSA | 40 | intended label + 2nd-rater | **headline test, run once** | ✘ |
| **R-team** | 30 items hand-written by ML (04 decision; was 60): about 18 main + 12 DSA | 30 | same | test + adversarial validation | ✘ |
| **U** | U1/U2 unseen-class items (hand-written + operator) | 30 | U1/U2 | novelty test | ✘ |
| **X** *(Strong)* | ITSP real-student slice, diff-auto-labelled + hand-checked | 40–80 | auto + manual | external real-student test | ✘ |

#### Ready-made public data (ITSP and IntroClass pages checked October 2026; others from the literature)

| Dataset | What it is | Lang | Misconception labels? | Fit | Usable in 12 h? |
|---|---|---|---|---|---|
| **ITSP** (IIT Kanpur, `github.com/jyi/ITSP`) | 661 student programs from CS-101 (2015–16) collected through Prutor; **buggy + later-corrected version by the same student**, tests per problem | C | No, but **the buggy→correct diff points at the misconception** | **High**: real Indian CS1 students, C, paired fixes; its later labs are a natural place to look for DSA-style bugs (arrays, strings, recursion), so filter by topic when sampling | **Yes, time-boxed 60 min** as an eval slice. No explicit licence in the repo → evaluation only, cite the FSE'17 paper, don't redistribute |
| **IntroClass** (`github.com/ProgramRepair/IntroClass`) | 6 small assignments (checksum, digits, grade, median, smallest, syllables), many buggy student versions, black-/white-box tests, BSD-3 | C | No (defects, not beliefs) | Medium: full programs with `scanf`, mostly conditional logic | Not in MVP; good post-hackathon source for OTHER/novelty realism |
| **C-Pack of IPAs** (APR 2024) | Benchmark of C90 introductory programming assignments | C | No | Medium | Roadmap |
| **Project CodeNet** (IBM, `github.com/IBM/Project_CodeNet`) | ~14M submissions in 55 languages from AIZU/AtCoder with verdicts (incl. Wrong Answer); AIZU has intro-programming problems | C incl. | No (verdict only) | High as an *unlabelled* pool for weak labelling | No (multi-GB download); roadmap |
| **Prutor / DeepFix** | Student C programs with **compile** errors | C | Error class | Low: syntax, not misconceptions | No |
| **Eedi — Mining Misconceptions in Mathematics** (Kaggle 2024) | MCQ distractors mapped to misconceptions | Maths | Yes | Method analogue for distractor→class mapping in Quickfire/probes | Cite only |
| Literature catalogues: Sorva (2012) misconception catalogue; Qian & Lehman (2017, ACM TOCE) review; Ettles, Luxton-Reilly & Denny (2018, ACE) common logic errors; Götschi, Sanders & Galpin (2003) mental models of recursion; Danielsiek, Paul & Vahrenhold (2012) algorithms & data-structures misconceptions | Taxonomy evidence | — | — | Grounds the 17 classes | Cite in the dataset card |

**Conclusion:** no public C dataset ships *misconception* labels. The ready data that helps within 12 hours is ITSP (real students, paired fixes) as an external test slice. Everything trainable is generated and verified by us.

#### R (realistic set) protocol — the honest replacement for the human study
1. **Blind authoring:** FE writes R-blind (40: 25 main + 15 DSA) at T+6:15 *without having seen* `operators.py`. ML writes R-team (**30**, 04 decision) separately at T+8:00. The combined realistic set is 70.
2. **Brief to authors:** "Write like a first-semester student who has *this* belief: odd names (`ans`, `temp1`, `x2`), debug `printf`s, comments (Hinglish fine), `while` instead of `for`, extra variables, partial solutions, different structure from the reference. Use only the frozen subset. Pick any bank problem."
3. **Composition (70):** 4 per class is no longer possible. Aim for at least 2 items per class across the combined set, plus CORRECT, OTHER, at least 4 HARD-twin items (incl. 2 DSA pair loops) and at least 4 gate/outlier items (incl. a loop-only "recursive" answer and `strlen` in Q13). U1/U2 live in set U, not here.
4. **Second rater:** the other teammate labels each item from code alone (≤ 10 min). Report % agreement + Cohen's κ; disagreements → final label by discussion, keep both raw labels.
5. **Verify by execution:** misconception items should fail ≥ 1 test. Items that pass all tests are kept in a **"passes-by-luck"** slice (evidence for the resolution argument).
6. **Freeze:** commit with hash before running any model on it. R-blind is evaluated **once**, at the end.

#### X (ITSP) protocol (Strong, 60-minute time-box)
1. `git clone https://github.com/jyi/ITSP` (small).
2. For each `*_buggy.c` / `*_correct.c` pair: `difflib.unified_diff`; keep pairs with **one hunk ≤ 3 lines**.
3. Map the hunk with regex rules to a candidate class (`<=`↔`<` in a loop header → M01/M08-AMB; `=`→`==` in a condition → M06; removed `;` after `)` → M07; added `= 0` on a declaration → M05; `(float)` or `.0` added → M04; accumulator init moved out of the loop → M03; added `i++` → M02; `printf`→`return` → M10; temp variable introduced around a swap → D03; `mid`→`mid±1` → D02; base case added in a recursive function → D05; `return` added before a recursive call → D07; `"x"`→`'x'` → D08; `else return` removed from a loop → D01; else → OTHER).
4. Hand-check 40–80 items (~30 s each); drop unclear ones.
5. Run AST features; trace features = NaN unless `scanf` support is added (input queue fed from the ITSP test inputs). The model is trained with feature-dropout (§4.6), so it degrades gracefully.
6. Report as "real-student slice, n = …, AST(+trace) features, auto-labelled and hand-verified".

### 3.5 Generation pipeline (`ml/generate/`)

```
for problem in train+holdout problems:
  for cv in problem.correct_variants:                  # 3 per problem
    emit CORRECT(cv) + near-miss CORRECT transforms    # §3.5.3
    for op in OPERATORS if op matches problem.allowed_ops:
      for site in op.sites(cv):                        # every place the op applies
        mutant = op.apply(cv, site)
        for aug in sample(AUGMENTATIONS, k=4):         # style variance
          row = verify(augment(mutant, aug))           # §3.5.4
          if row: yield row
  compositions(two-bug, 8%) ; OTHER ops ; AMB ops
→ ambiguity pass (§3.5.5) → dedupe + caps (§3.5.6) → dataset.jsonl
```
Operators change the **syntax tree and reprint it with `pycparser.c_generator`** (a `BinaryOp`'s coordinate is where its left operand starts, not where the operator is, so text edits by coordinate are reserved for the fixer, where the learner sees a diff of their own code). Each operator has a stable `op_id`. **`op_variant` equals `op_id`**: E3 holds out one operator per class that has at least two. The row field `variant` is the correct-variant id (`cv1`, `cv2`, `cv3`) and is unrelated. Each operator has an inverse in the fixer.

#### 3.5.1 Operator catalogue

| Class | op_id | Edit | Applies when |
|---|---|---|---|
| M01 | `m01_le` | `i < n` → `i <= n` | counting loop, body doesn't index an array with `i` |
| M01 | `m01_nminus1` | `i < n` → `i < n - 1` | counting or array loop (it under-reads, no OOB → hard M01) |
| M01 | `m01_start1_count` | `i = 0` → `i = 1` | counting loop without array indexing |
| M01 | `m01_countdown_ge0` | `i > 0` → `i >= 0` | countdown loop |
| M01 | `m01_while_le` | `while (i < n)` → `while (i <= n)` | counting while loop |
| M08 | `m08_index_n` | `a[n - 1]` → `a[n]` | direct last-element access |
| M08 | `m08_onebased` | `i = 0; i < n` → `i = 1; i <= n` (keeps `a[i]`) | array loop |
| M08 | `m08_iplus1` | `a[i]` → `a[i + 1]` | array loop |
| M08 | `m08_first1` | `a[0]` → `a[1]` | first-element access / max init |
| AMB(M01,M08) | `amb_le_array` | `i < n` → `i <= n` | loop body indexes `a[i]` |
| AMB(M01,M08) | `amb_start1_array` | `i = 0` → `i = 1` (bound unchanged) | loop body indexes `a[i]` |
| M02 | `m02_drop_update` | delete `i++` (for-update or last body stmt) | any loop |
| M02 | `m02_reverse` | `i++` → `i--` | loop with `<`/`<=` bound |
| M02 | `m02_update_in_branch` | move `i++` inside an existing `if` in the body | while loop with an `if` |
| M02 | `m02_wrong_var` | `i++` → `k++` / `n++` on another in-scope var | while loop |
| M03 | `m03_decl_in_loop` | move `int t = 0;` into the loop body (first stmt) | accumulator |
| M03 | `m03_assign_in_loop` | add `t = 0;` as the first body stmt | accumulator / counter |
| M04 | `m04_drop_cast` | `(float)sum / n` → `sum / n` | float result |
| M04 | `m04_int_literal` | `/ 2.0` → `/ 2`, `* 100.0` → `* 100` | float result |
| M04 | `m04_cast_late` | `(float)a / b` → `(float)(a / b)` | float result |
| M05 | `m05_drop_init_acc` | `int t = 0;` → `int t;` | accumulator |
| M05 | `m05_drop_init_counter` | `int c = 0;` → `int c;` | counter |
| M05 | `m05_drop_init_max` | `int m = a[0];` → `int m;` | max/min |
| M06 | `m06_if_assign` | `==` → `=` in `if` | if with `==` against a constant or var |
| M06 | `m06_while_assign` | `!=`/`==` → `=` in `while` | while with equality |
| M07 | `m07_if_semi` | `if (c) {` → `if (c); {` | any if |
| M07 | `m07_for_semi` | `for (...) {` → `for (...); {` | any for |
| M07 | `m07_while_semi` | `while (...) {` → `while (...); {` | any while |
| M10 | `m10_printf_noreturn` | `return e;` → `printf("%d", e);` | non-void entry function |
| M10 | `m10_printf_return0` | `return e;` → `printf("%d", e); return 0;` | non-void entry function |
| OTHER | `oth_swap_operands` | `a - b` → `b - a` | non-loop-control arithmetic |
| OTHER | `oth_wrong_op` | `+=` → `-=`, `*` → `+` | accumulator update |
| OTHER | `oth_wrong_const` | change a non-bound literal (`42` → `24`, `7` → `6`) | constants |
| OTHER | `oth_wrong_var` | use `n` where `a[i]` was used, etc. | loop body |
| OTHER | `oth_flip_branch_rel` | `>` → `<` in a non-loop `if` | conditions |
| OTHER | `oth_drop_stmt` | delete a non-target statement | any |
| **DSA surfaces of main classes (v3)** | | | |
| AMB(M01,M08) | `amb_pair_bound` | `j < n - 1` → `j < n` (body uses `a[j+1]`) | bubble sort, is_sorted, rotate |
| M08 | `m08_mirror` | `a[n - 1 - i]` / `s[len - 1 - i]` → `a[n - i]` / `s[len - i]` | reverse, palindrome |
| M01 | `m01_half_bound` | `i < n / 2` → `i < n` (reverses twice) or `i <= n / 2` | reverse, palindrome |
| M02 | `m02_pointer_stuck` | delete `i++` or `j--` in one branch of a two-pointer loop | pair_sum_exists |
| M10 | `m10_print_bool` | `return 1;`/`return 0;` → `printf("yes")`/`printf("no")` | is_sorted, is_palindrome |
| **DSA classes (v3)** | | | |
| D01 | `d01_else_return` | add `else return -1;` (or `else return 0;`) to the match `if` inside the loop | linear search, is_sorted, palindrome |
| D01 | `d01_else_flag_reset` | add `else found = 0;` to the match `if` | flag search, count-style checks |
| D02 | `d02_low_mid` | `low = mid + 1` → `low = mid` | binary search, guess halving |
| D02 | `d02_high_mid` | `high = mid - 1` → `high = mid` | same |
| D03 | `d03_drop_temp` | `t = a[i]; a[i] = a[j]; a[j] = t;` → `a[i] = a[j]; a[j] = a[i];` | sorts, reverse |
| D03 | `d03_rotate_no_save` | remove `int first = a[0];` and use `a[n-1] = a[0];` after shifting | rotate |
| D04 | `d04_drop_outer` | remove the outer loop (keep one inner pass with `i = 0`) | bubble, selection |
| D04 | `d04_outer_once` | outer bound `i < n - 1` → `i < 1` | bubble, selection |
| D05 | `d05_drop_base` | delete the base-case `if (...) return ...;` | all recursion |
| D05 | `d05_unreachable_base` | `n <= 0` → `n == 0` **and** step `n - 1` → `n - 2` (only kept if some test has odd n) | factorial, sum_digits |
| D06 | `d06_same_arg` | `f(n - 1)` → `f(n)`; `f(n / 10)` → `f(n % 10)` | all recursion |
| D06 | `d06_grow_arg` | `f(n - 1)` → `f(n + 1)` | factorial, array_sum_rec |
| D07 | `d07_discard` | `return n * f(n - 1);` → `f(n - 1); return n;` (operator-aware: `+`, `*`) | all recursion |
| D08 | `d08_str_literal` | `s[i] == 'a'` → `s[i] == "a"` | count_vowels, palindrome |
| D08 | `d08_array_eq` | char-by-char comparison loop → `if (s == t)` | palindrome helper variant |
| U1 *(test only)* | `u1_chained` | `lo < x && x < hi` → `lo < x < hi` | range checks (P16, P12 + 2 extra snippets) |
| U2 *(test only)* | `u2_or_chain` | `c == 42 \|\| c == 7` → `c == 42 \|\| 7` | equality alternatives |

#### 3.5.2 Style augmentations (applied after mutation; each row stores `aug: [...]`)
`rename` (identifier pools: `sum/s/tot/total/ans/res`, `i/j/k/idx/x`, `cnt/count/c`) · `brace_style` (K&R / Allman / no braces for single statements) · `incr_form` (`i++`, `++i`, `i += 1`, `i = i + 1`) · `decl_hoist` (`int i;` at the top vs C99 `for (int i…)`) · `comments` (English + Hinglish templates, e.g. `// yaha total add karo`) · `debug_printf` (insert `printf("i=%d\n", i);`, tagged `debug=true`) · `spacing` · `redundant_parens` · `unused_var`. **Never** augment in a way that changes semantics; the verifier re-runs every augmented row.

#### 3.5.3 Hard negatives (CORRECT near-misses), ≥ 30% of CORRECT
`i <= n - 1` · `i = 1; i <= n` in **counting** loops (correct count) · `for (i = 1; i <= n; i++) s += a[i - 1];` (correct 1-based traversal: **key M08 negative**) · `i = n - 1; i >= 0; i--` · early-exit loops with `break` · `debug_printf` + correct `return` (**key M10 negative**) · `(float)` cast on the denominator or `* 1.0` · `int t; t = 0;` (declared without initialiser, assigned before use: **key M05 negative**) · `if (5 == x)` · single-statement `if` without braces.
**DSA near-misses (v3):** bubble sort with an early-exit `swapped` flag · bubble with inner bound `j < n - 1 - i` *and* `j < n - 1` · binary search with `mid = low + (high - low) / 2` and `while (low <= high)` · swaps with the temp in a different order (`t = a[j]; a[j] = a[i]; a[i] = t;`) · linear search with a `found` flag + `break` (**key D01 negative**) · `else continue;` inside a search loop · recursion with base `n <= 1` vs `n == 0` · accumulator-parameter recursion (`f(n - 1, acc * n)`) · string loops using `s[i] != '\0'` vs `s[i]` truthiness · palindrome using `len - 1 - i` with `i < len / 2` · `printf` debug inside a recursive function that still returns correctly (**key D07/M10 negative**).

#### 3.5.4 Verifier (hard rules)
A row is kept only if **all** hold:
1. Parses and stays within the subset (else drop + count in `drop_reasons`).
2. `CORRECT`: passes all tests.
3. Misconception class: fails ≥ 1 test **and** its structural predicate holds (e.g. M07: an `EmptyStatement` body exists at the edited site; M05: a variable is read before assignment on some path). Rows that pass all tests are moved to the **passes-by-luck** pool (used in evaluation, not training).
4. `OTHER`: fails ≥ 1 test **and no class predicate fires** (else drop — it's an accidental class instance).
5. Two-bug: both predicates hold and the code fails a test that each single fix alone doesn't repair.
6. **DSA predicates (v3):** D01 return/assignment inside an `else` of the match `if` within the loop; D02 assignment `low = mid` / `high = mid` and `step_cap_hit` or a wrong result on ≥ 1 test; D03 consecutive cross-assignments without a temp and multiset change on ≥ 1 test; D04 output equals the reference's single pass on ≥ 1 failing test; D05 no reachable base case (static check + `depth_cap_hit`); D06 recursive argument constant or growing; D07 `discarded_call_value`; D08 `str_literal_compare` or `array_compare` event.
Log drop rate per operator. An operator with > 40% drops is buggy: fix it, don't ship it.

#### 3.5.5 Ambiguity pass (soft labels)
1. Compute a **normalised AST hash**: rename identifiers to canonical order (`v0, v1…`), drop comments, drop `debug=true` printfs, normalise `i++`/`++i`/`i+=1`, canonical brace form.
2. Group rows by hash. If a group carries **≥ 2 distinct labels**:
   - If the labels are exactly `{M01, M08}` → expected T1 ambiguity: assign `soft_label = {M01: 0.5, M08: 0.5}` to every row in the group.
   - Any other label clash → **generator bug**: print it, fix the operator. Never silently soft-label it.
3. Training uses soft labels by **row duplication with weights**: one copy per label, `weight = soft_p × class_weight`. This is exactly cross-entropy with soft targets in LightGBM.

#### 3.5.6 Dedupe and caps
- Exact-hash duplicates within the same label → keep 1.
- Cap **≤ 15 rows per (problem, op_id)** and **≤ 6 augmentations per (problem, variant, op_id)** so no template dominates.
- Class target after caps: each M-class 300–550 (main + DSA surfaces), each D-class 250–400, CORRECT ~1,300 (incl. ≥ 400 near-miss), OTHER ~600, AMB ~400, two-bug ~450. Total ≈ 9,000 rows.
- Report **effective N**: `n_rows`, `n_unique_hash`, `n_unique(problem, variant, op_id)` (≈ 34 × 3 × ~20 ≈ 2,000). Put effective N on the Lab Report. Da GOATS didn't, and judges notice.
- Report counts **per domain** (main vs DSA) so the DSA classes can't hide behind main-game volume.

### 3.6 Row schema (`data/dataset.jsonl`)
```json
{
  "id": "A-004311", "source": "A|E|AMB|R-blind|R-team|U|X",
  "problem_id": "P03", "family": "array_accumulate", "split": "train|holdout_problem",
  "variant": "cv1", "op_id": "amb_le_array", "op_variant": "amb_le_array", "aug": ["rename","debug_printf"],
  "code": "...",
  "label": "M01", "soft_label": {"M01": 0.5, "M08": 0.5}, "labels_all": ["M01","M08"],
  "is_two_bug": false, "ambiguous_group": "h_9f2c…", "ast_hash": "h_9f2c…",
  "verified": {"tests_failed": 2, "tests_total": 5, "predicate": ["M01_iter_delta","M08_oob_n"], "passes_by_luck": false},
  "trace_summary": {"status": "ok", "events": ["oob_read"], "loop_iters_delta": 1},
  "author": null, "rater2_label": null
}
```

### 3.7 Outliers and edge cases

#### 3.7.1 Serving-time gate (`server/app/gate.py`), run **before** the model

| Code | Condition | In-world message (FE shows as-is) | Diagnosis? |
|---|---|---|---|
| G0 | ok | — | yes |
| G1 | empty / whitespace / comments only | "Navigation core is empty." | no |
| G2 | identical to starter after normalisation | "No changes detected in the core." | no |
| G3a | parse error | "Core rejected line {n}: {short reason}." | no |
| G3b | looks like Python (`def `, `print(`, `elif`, `:`+indent) | "That's Python. This ship only speaks C." | no |
| G3c | looks like C++ (`cout`, `std::`, `<iostream>`) | "That's C++. This ship only speaks C." | no |
| G4 | unsupported construct (§2.1) | "This ship doesn't understand `{construct}` yet." | no |
| G4b | uses a forbidden builtin (e.g. `strlen` in Q13) | "Trial rules: count it yourself, no `strlen`." | no |
| G5 | entry signature missing / wrong arity | "Mission needs `{signature}`." | no |
| G5b | recursion problem solved without recursion (no self-call) | "This trial needs a warp gate: solve it with recursion." | no |
| G6 | **test gaming**: entry function never reads its parameters **and** returns/prints a constant | "Your core gives the same answer whatever the input. The mission changes its inputs." | no (logged) |
| G7 | > 120 lines or > 4 KB | "Core too large for this mission." | no |
| G8 | smart quotes / non-ASCII punctuation | auto-normalise silently and continue | yes |
| — | timeout / runtime error | not a gate: becomes trace evidence (M02/M07, div-by-zero message) | yes |

#### 3.7.2 Outliers deliberately put into training data
- Near-miss CORRECT (§3.5.3) so `<=`, `i=1`, `printf`, `int t;` alone never decide a class.
- Two-bug compositions (8%) with soft labels so mixed evidence isn't forced into one class.
- Weird-but-valid programs (3% of CORRECT/OTHER): extra helper functions, duplicated computation, unused arrays.
- **Feature dropout** rows (15%): trace-based groups set to NaN so the model survives missing traces (ITSP, unsupported runs).
- **DSA outliers (v3):** sorts that are correct but O(n³) or use extra arrays; recursion with extra helper functions; palindrome via a reversed copy; binary search on a 1-element array; all kept as CORRECT so "unusual" ≠ "wrong".

#### 3.7.3 Outliers in evaluation
R contains 4 gate/outlier items; U has 30 unseen-class items; E11 perturbs every realistic item; EDA lists the top-20 IsolationForest outliers of the training matrix for manual inspection (they're usually operator bugs).

### 3.8 Synthetic-data overfitting controls (checklist, all Core)

| Risk | Control | Where measured |
|---|---|---|
| Template memorisation | Problem-grouped CV; 3 held-out problems with unique families; effective-N reporting | E1, E2 |
| Operator memorisation (one surface form per class) | ≥ 2 operators per class; **operator-variant holdout** | E3 |
| Style shortcuts (length, comments, names) | No text/length/identifier features; augmentation balances style; shortcut stump scan | EDA §10.4 |
| Label leakage via simulated signals | No simulated features in LightGBM; Bayes layer uses fixed likelihoods | design §6 |
| Fix-probe features = inverse of generator | Fixers are **broader** than operators (try several sites and edits); ablation with/without group F on R | E9 |
| Hyper-parameter overfitting to test | Tiny grid, selected on grouped CV only; R-blind run once | §5 |
| Optimistic synthetic numbers | Headline = R-blind; red-flag thresholds (§0.3); failure audit | E4, E12 |
| Distribution shift synthetic → real | Adversarial validation vs R-team only; ITSP slice | EDA §10.7, E13 |
| Domain shortcut (DSA problems → D-classes) | Coarse task meta only; **structural class masking** instead of problem ids; M-classes generated on DSA surfaces too; cross-domain experiment | E15, EDA §10.12 |

### 3.9 Dataset card (`docs/dataset_card.md`, generated by `generate/card.py`)
Sources and counts (class × source × problem), operator catalogue with drop rates, verification rules, ambiguity groups, effective N, augmentation list, near-miss share, known biases (author-written templates, single cohort style, no real learners in training), ITSP licence note, R protocol + κ, intended use (hackathon demo and research prototype; not for grading students).

---

## 4. Features (`ml/features/`)

**Design rule: every feature is problem-agnostic.** Behaviour is measured *relative to the reference solution on the same inputs*, so a feature means the same thing on P03 and on an unseen P09. No raw text, identifiers, line counts, `problem_id`, `op_id` or template ids.

**Main loop.** `a_main_cond_op_*`, `a_bound_form_*`, `a_init_form_*` and `a_update_*` describe one loop: the loop with the most executed iterations on the display test, ties going to the outermost. Nested loops also emit `a_outer_cond_op_*` and `a_outer_bound_*` for the outermost loop and `a_inner_cond_op_*` for the loop directly inside it; all of those are 0 when the function has one loop. The same rule picks the reference's main loop.

**Reference.** Every learner-vs-reference feature uses `correct_variants[0]`, not a blend of the three variants.

### 4.1 Group A — AST structure (`ast_feats.py`), ~30 features
| Feature | Meaning |
|---|---|
| `a_n_loops`, `a_max_nesting` | structure size |
| `a_main_cond_op_{lt,le,gt,ge,ne,eq,other}` | relational operator of the main loop condition |
| `a_bound_form_{n, n_minus_1, n_plus_1, const, other}` | bound expression shape (param-relative) |
| `a_init_form_{0, 1, n, n_minus_1, other}` | loop-variable initial value |
| `a_update_present`, `a_update_dir` (+1/−1/0), `a_update_var_is_cond_var`, `a_update_in_branch` | progress structure |
| `a_outer_cond_op_{lt,le,gt,ge,ne,eq,other}`, `a_inner_cond_op_{…}`, `a_outer_bound_{n, n_minus_1, n_plus_1, const, other}` | nested loops only; 0 when there is one loop |
| `a_empty_body_{if,for,while}` | `EmptyStatement` directly as a body |
| `a_assign_in_cond` | `Assignment` node as an `if`/`while`/`for` condition |
| `a_decl_noinit_read_first` | declared without initialiser and the first use is a read / compound assign |
| `a_init_inside_loop` | var assigned a constant inside a loop body *and* compound-updated in the same body |
| `a_int_div_into_float`, `a_cast_wraps_division`, `a_float_literal_in_div` | division typing |
| `a_index_{i, i_plus_1, i_minus_1, n, n_minus_1, const0, const1}` | array index expression shapes |
| `a_array_loop_start1`, `a_array_loop_le_n` | array traversal shape |
| `a_printf_in_nonvoid`, `a_nonvoid_missing_return`, `a_return_const_only` | output vs return |
| `a_reads_all_params` | used by the gate and as a weak feature |
| `a_param_written` *(M09, Strong)* | param assigned inside the function |
| **DSA (v3)** | |
| `a_n_nested_loops`, `a_outer_bound_const1`, `a_single_loop_with_swap` | sort structure (D04) |
| `a_return_in_loop_else`, `a_assign_in_loop_else` | return / flag overwrite inside the `else` of the match `if` (D01) |
| `a_mid_assign_no_offset` | `low = mid` or `high = mid` without ±1 (D02) |
| `a_mid_like_var` | a var assigned `(x + y) / 2` or `x + (y - x) / 2` (masking precondition for D02) |
| `a_swap_no_temp` | two consecutive statements `X = Y; Y = X;` over array cells / vars (D03) |
| `a_swap_with_temp` | the correct 3-statement pattern (negative evidence) |
| `a_index_pair_plus1`, `a_index_mirror_n_minus_i`, `a_index_mirror_n_minus_1_minus_i`, `a_half_bound` | pair/mirror index shapes (M01/M08 on DSA surfaces) |
| `a_has_recursion`, `a_n_self_calls` | self-call present (masking precondition for D05–D07) |
| `a_base_case_present`, `a_base_case_static_reachable` | base `if` that returns without recursion; reachability heuristic: base compares the param to a constant and the recursive arg moves monotonically toward it by 1 (or `/10` toward 0) |
| `a_rec_arg_form_{minus1, minus2, div10, same, plus1, other}` | shape of the recursive argument (D05/D06) |
| `a_rec_call_discarded` | self-call used as a statement, value unused (D07) |
| `a_str_literal_compare`, `a_array_name_compare` | `==` with `"…"` or between two array names (D08) |
| `a_char_array_param`, `a_uses_terminator` | strings structure |

### 4.1b Structural class masking (v3, `model/mask.py`)
Some classes are impossible unless a structure exists. After calibration, set `p(k) = 0` and renormalise when the precondition is absent: D05–D07 need `a_has_recursion`; D08 needs a string literal or `a_char_array_param`; D02 needs `a_mid_like_var`; D03/D04 need an array write. Preconditions come from the **code**, never from the problem id, so masking also works in Bug Lab on arbitrary code. Masking is applied identically in training-time OOF evaluation.

### 4.2 Group B — execution/trace (`trace_feats.py`), ~20 features
`b_pass_frac` · `b_status_{timeout, runtime_error}` · `b_step_cap` · `b_uninit_read` · `b_oob_read` · `b_oob_read_idx_eq_n` · `b_oob_read_idx_neg` · `b_oob_write` · `b_intdiv_trunc_nonzero` · `b_intdiv_into_float` · `b_iter_delta_mean` (main-loop iterations − reference's, averaged over tests) · `b_iter_delta_const_pm1` (delta is exactly +1 or −1 on every test) · `b_body_once_vs_many` (body ran once where the reference ran ≥ 2) · `b_empty_body_exec` · `b_assign_in_cond_rt` · `b_branch_always` / `b_branch_never` (an `if` is constant across tests while the reference's varies) · `b_missing_return` · `b_returned_garbage` · `b_printed_nonempty` · `b_printed_eq_ref_return` (**M10 signal**).
**DSA (v3):** `b_depth_cap` · `b_max_depth_ratio` (learner max depth ÷ reference's) · `b_rec_arg_constant` (same args in consecutive frames) · `b_rec_arg_growing` · `b_base_return_executed` · `b_discarded_call_value` · `b_return_first_iter` (function returned during the first loop iteration on a test where the reference iterated ≥ 2) · `b_window_frozen` (in the main loop, the pair of loop-bound vars is unchanged across 2 consecutive iterations) · `b_multiset_changed` (output array isn't a permutation of the input) · `b_sorted_frac` (fraction of adjacent pairs in order in `array0`) · `b_n_outer_passes_ratio` · `b_str_literal_compare` · `b_array_compare`.

### 4.3 Group R — output relations (`relation_feats.py`), ~10 features
For each failing test, compute **counterfactual reference outputs** by running the *reference* on modified inputs, then check which one the learner's output equals. The feature is the fraction of failing tests where the relation holds:
`r_eq_ref_drop_first` (ref on `a[1:]`) · `r_eq_ref_drop_last` (ref on `a[:-1]`) · `r_eq_ref_last_only` (ref on `[a[-1]]`, the **M03 signal**) · `r_eq_ref_first_only` · `r_eq_ref_n_minus_1` / `r_eq_ref_n_plus_1` (scalar `n` ±1) · `r_eq_floor_ref` (**M04**) · `r_is_garbage` · `r_eq_zero` · `r_off_by_value_1`.
**DSA (v3):** `r_eq_one_pass` (learner's `array0` equals the reference's **first pass only**, computed by running a provided `one_pass` helper of the reference; **D04 signal**) · `r_eq_first_check_only` / `r_eq_last_check_only` (result equals the reference run on only the first / last element: **D01** early return / flag reset) · `r_eq_top_frame_only` (output equals the top frame's own contribution, e.g. `n` for factorial, `a[n-1]` for array sum: **D07 signal**) · `r_eq_reversed_twice` (array unchanged after a reverse: `m01_half_bound`) · `r_eq_shift_without_wrap` (rotate lost the first value).
These generalise across problems because they're defined by input transformations, not by problem-specific values.

### 4.4 Group F — fix-probe, computed after the model, not an input
For each of the **top-3 classes** after masking and temperature (§5.5): run its fixer (§7.3, up to 5 candidate edits); `f_fix_k` = 1 if any candidate passes all tests, `f_fixgain_k` = best Δ pass fraction. These 34 numbers (`ml/contracts/feature_names.py`: `OFFLINE_FEATURES`) are **not columns of the training matrix**. On synthetic data the class-k fixer is the inverse of the operator that built the row, so `f_fix_k` is a near-perfect label and a model trained on it memorizes the generator (E1/E2 then hit the ≥ 0.98 red flag and `/attempt` spends ~17 × 5 suites). For T1 code the M01 and M08 fixes are the same edit, so both fire: the ambiguity is preserved. Results feed `two_bug_check`, the `f_fix` evidence sentence and the intervention's minimal fix. E9 still reports an offline row that *adds* F to the model, so the Lab Report shows what was given up. Budget: stop as soon as one candidate per class passes; cache by AST hash.

### 4.5 Group C — coarse task meta, 5 features
`t_has_array_param`, `t_returns_float`, `t_is_void`, `t_has_char_array_param`, `t_mutates_array_arg`. (Coarse only. Anything finer — sector, problem id, "is DSA" — lets the model learn "problem → class". E15 checks this.)

### 4.6 Missing values
LightGBM handles NaN natively. 15% of training rows have groups B and R set to NaN (feature dropout), so AST-only inputs (ITSP, unsupported runs) still get sensible posteriors. Group F is not a model input (§4.4). Ablation E9 reports AST-only performance explicitly, and one extra row with F added back.

### 4.7 Explicitly **not** features
Simulated predictions, belief-agreement vectors, history, hint usage, time taken, text n-grams, identifiers, `n_lines`. (Predictions/probes/history live in the Bayes layer; TF-IDF exists only as a baseline.)

### 4.8 Novelty score (`model/novelty.py`)
- `knn_dist`: mean Euclidean distance to the 5 nearest training rows in standardised A+B+R space (NaN → column median; drop zero-variance columns). Threshold `τ_d` = 99th percentile of out-of-fold distances.
- `p_max`: calibrated max probability; `p_other`.
- **novel** if `knn_dist > τ_d` **or** `p_other ≥ 0.5` **or** (`p_max < τ_p` **and** the top-2 aren't a known twin pair). `τ_p` is chosen on OOF predictions so that accepted (non-abstained) predictions have ≥ 90% precision.
- **LOCO.** E6 "LOCO strict" (§9.2) drops `CLASS_DEFINING_FEATURES[k]` (`ml/contracts/feature_names.py`) from this space before scoring class k's rows, and does not run fixer k. "LOCO naive" is the same score without that drop, reported beside it. The naive number is optimistic: k's own signature features are near-constant 0 in a model that never saw k.

---

## 5. Diagnoser model (`ml/model/`)

### 5.1 Labels
19 outputs: `M01 M02 M03 M04 M05 M06 M07 M08 M10 D01 D02 D03 D04 D05 D06 D07 D08 CORRECT OTHER` (+ `M09` if Strong S4 lands → 20). **One model for both domains** (main + DSA): M-classes occur inside DSA problems, and a shared model is what makes the exam's "did your Loops repair hold inside a sort?" question answerable. **Inputs are groups A, B, R and C.** Group F is computed afterwards for the top-3 classes (§4.4, §5.5) and is not in the matrix. Soft-labelled rows are duplicated with weights (§3.5.5). Two-bug rows: 0.5/0.5 duplication.

### 5.2 LightGBM configuration
```python
params = dict(objective="multiclass", num_class=19, learning_rate=0.05,
              num_leaves=15, max_depth=5, min_data_in_leaf=20,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, verbose=-1, seed=42, deterministic=True)
num_boost_round ≤ 400, early_stopping_rounds = 30 on an inner grouped validation fold
sample_weight = soft_weight × balanced_class_weight
```
**Grid (12 configs):** `num_leaves ∈ {7, 15, 31}` × `min_data_in_leaf ∈ {10, 30}` × `feature_fraction ∈ {0.6, 0.9}`. Select by **mean grouped-CV macro-F1** on training problems; tie-break by log-loss. Takes well under a minute on ~9k rows.

### 5.3 Training procedure (`train.py`)
1. Load `dataset.jsonl`, drop `split == holdout_problem`, the passes-by-luck pool and U rows.
2. `StratifiedGroupKFold(n_splits=5)` over the 28 training problems (13 main + 15 DSA), groups = `problem_id`, stratified by domain (main vs DSA) so each fold holds both. `GroupKFold` cannot stratify; do not use it here. → OOF probabilities for every row (used for calibration, τ, novelty, E1).
3. Pick config (§5.2), refit on all training problems → `model.txt`.
4. Temperature scaling on OOF logits (§5.4); novelty thresholds on OOF (§4.8).
5. Save `artifacts/diagnoser_<sha8>/{model.txt, meta.json}`; `meta.json` = classes, feature list + order, T, τ_p, τ_d, kNN reference matrix (≤ 4k rows, float32), training data hash, git commit.
6. Evaluate holdout problems, operator holdout, R, U, X via `eval/run_all.py` (§9).

### 5.4 Calibration
Fit a scalar temperature T by minimising NLL of `softmax(logits / T)` on OOF logits (`scipy.optimize.minimize_scalar`, bounds [0.5, 5]). Report ECE (10 equal-width bins) before/after and a reliability diagram. If per-class reliability is bad for one class, report it; don't add per-class isotonic in the MVP (too little data per class).

### 5.5 Decision logic (`model/decide.py`)
Structural masking (§4.1b) is applied to `p_code` first; then the Bayes layer (§6) produces `posterior`. Fixers for the **top-3** classes run here (§4.4), not inside the model:
```
if gate != G0                                   → status = "gate"
elif tests all pass:
     if top1 == CORRECT or p(top1) < 0.5        → status = "correct"
     else                                       → status = "correct", latent = {class: top1, p: p(top1)}
elif novel (§4.8)                               → status = "novel"
elif two_bug_check(top1, top2)                  → status = "two_bug"
elif p1 < 0.75 and p1 - p2 < 0.25:
     if eig_best_probe(posterior) ≥ 0.10 bits   → status = "ambiguous", next_probe = argmax EIG
     else                                       → status = "confident" (band "Possible")
else                                            → status = "confident"
```
`latent` is the passes-by-luck case: the tests passed, but the model still names a misconception at ≥ 0.5. The attempt is shown as correct. The knowledge model applies the §8.2 likelihood ratio with `e_ik × LATENT_EXPOSURE_FACTOR` (`0.5` in `params.py`), so a lucky pass is weak evidence that k is active, not evidence that it is gone.
`two_bug_check`: p2 ≥ 0.25 **and** fix(top1) alone fails tests **and** fix(top2) alone fails **and** fix(top1)∘fix(top2) passes. Both fixes come from the top-3 fixer run above.

### 5.6 Evidence generation (`model/evidence.py`)
1. `booster.predict(X, pred_contrib=True)` → SHAP-style contributions per class; take the slice for `top1`.
2. Top-3 positive contributors → sentence templates filled from extractor metadata (`feature_lines`, values). The template file is `evidence_templates.json`: one entry per feature in `FEATURES`, plus the post-model fixer sentence. A class's defining features are exactly `CLASS_DEFINING_FEATURES[k]`, and EDA 10.5 reads that same map.
   - `a_main_cond_op_le` → "Loop condition uses `<=` (line {line})."
   - `b_iter_delta_const_pm1` → "Loop ran {actual} times; the mission needed {expected}."
   - `b_oob_read_idx_eq_n` → "Reads `{arr}[{n}]`, one cell past the end (line {line})."
   - `r_eq_ref_last_only` → "Your total equals only the last cell's value."
   - `b_printed_eq_ref_return` → "The right value was printed, but the function returned {returned}."
   - `f_fix_k` → "Changing only {fix_desc} makes every test pass." (from the post-model fixer, §4.4; this is not a model feature)
   - `b_multiset_changed` → "After your swap, crate {v} appears twice and {w} is gone."
   - `a_mid_assign_no_offset` → "`{var} = mid` keeps mid inside the window (line {line}); the window stops shrinking."
   - `b_depth_cap` + `a_base_case_present=0` → "Warp gates opened {depth} deep and never closed: no base case."
   - `b_rec_arg_constant` → "Every gate was called with the same value ({arg}); it never gets closer to the base."
   - `r_eq_top_frame_only` → "Only the last gate's value came back; the result of `{fn}({arg})` was thrown away (line {line})."
   - `b_return_first_iter` → "The search stopped after checking only the first beacon."
   - `a_str_literal_compare` → "`{expr}` compares a character with a string literal; in C this is never equal."
   (One template per feature in `evidence_templates.json`, ~100 entries. Agents generate these from the feature table above.)
3. Always add ≤ 1 `RUN` item from the trace, plus `YOU PREDICTED` / `PROBE` / `HISTORY` items from the Bayes layer when present.
4. For HARD-twin status, add: `{"type": "RUN", "text": "This code is identical for both explanations. Asking one question."}`.

### 5.7 Baselines (`model/baselines.py`)
1. Majority class. 2. **Rules**: class predicates in fixed priority (gate → D08 → D05 → D06 → D07 → D03 → D02 → D04 → D01 → M07 → M06 → M05 → M02 → M10 → M04 → M03 → M08 → M01 → OTHER/CORRECT). 3. TF-IDF (char 3–5-grams on normalised code) + logistic regression. 4. TF-IDF + LightGBM. 5. **Zero-shot DeepSeek** on R (R-blind and R-team), through `ml/text/llm_client.py`: same label list, JSON output, temperature 0. This is an E-a row. If the key is missing the Lab Report row says "not run" and the rest of E8 still ships. Cost is cents; do not skip it only because the hour is late.

---

## 6. Bayes evidence layer + active probing (`ml/bayes/`)

### 6.1 Posterior
```
classes K = {M01…M10, D01…D08, CORRECT, OTHER}  (masked classes removed, §4.1b)
p0(k)   = calibrated LightGBM probability (code + trace)
π_L(k)  = learner's current P(active_k) from the knowledge model (§8), floored at 0.02
π_L(CORRECT) = π_L(OTHER) = 1.0          (history does not move these two)
prior   : p(k) ∝ p0(k) · π_L(k)^γ          γ = 0.3  (history can nudge, never dominate)
response r_j (prediction, MCQ, probe):  p(k) ← p(k) · P(r_j | k)  then renormalise
```
Responses are treated as conditionally independent given the class (stated assumption). LightGBM is trained class-balanced, so multiplying by a prior is a legitimate prior correction.

### 6.2 Likelihood tables (hand-set, documented — **no fitting on simulated data**)
For an item j with options `O_j`, correct answer `c_j`, and authored belief answers `b_j(k)`:

| Learner class | P(answer = a) |
|---|---|
| misconception k, item diagnostic for k (`b_j(k)` defined and ≠ `c_j`) | `p_b` if a = `b_j(k)`; `(1−p_b)·q` if a = `c_j`; remainder spread evenly over other options |
| misconception k, item **not** diagnostic for k | `1−s` if a = `c_j`; `s` spread evenly over the rest |
| CORRECT | `1−s` if a = `c_j`; `s` spread evenly |
| OTHER | 0.4 if a = `c_j`; 0.6 spread evenly |

Defaults **p_b = 0.6, q = 0.5, s = 0.1**; every probability floored at 0.02 and renormalised. Free numeric predictions map to options `{correct, each belief value, "other"}`. These three numbers are design assumptions. E5 sweeps the *simulated* learner's true `p_b` from 0.4 to 0.9 while the engine keeps 0.6, to show robustness to the assumption being wrong.

### 6.3 Probe bank (`data/probes.json`), 14 probes (8 main + 6 DSA)
Each probe's correct answer is **verified by running its code in the interpreter** (`bayes/verify_probes.py`).

| probe_id | Set | Code / prompt | Options | Correct | Belief answers |
|---|---|---|---|---|---|
| P_T1_a | T1 | `int a[5];` — index of the last valid cell? | 4 / 5 / depends on values | 4 | M08: 5 · M01: 4 |
| P_T1_b | T1 | `for (i = 0; i <= 3; i++) fire();` — shots? | 3 / 4 / 5 | 4 | M01: 3 · M08: 4 |
| P_T2_a | T2 | `int i = 0; while (i < 3) { fire(); }` — shots? | 3 / never stops / 0 | never stops | M02: 3 · M07: never stops |
| P_T2_b | T2 | `for (i = 0; i < 3; i++); fire();` — shots? | 1 / 3 / 0 | 1 | M07: 3 · M02: 1 |
| P_T3_a | T3 | `int x = 3; if (x = 5) fire();` — value of x after? | 3 / 5 / error | 5 | M06: 3 · M07: 5 |
| P_T3_b | T3 | `int x = 2; if (x > 10); fire();` — does it fire? | yes / no / error | yes | M07: no · M06: yes |
| P_T4_a | T4 | `int t; t = t + 5;` — value of t? | 5 / 0 / unpredictable | unpredictable | M05: 5 · M03: unpredictable |
| P_T4_b | T4 | `int s = 0; for (i = 0; i < 3; i++) { s = 0; s = s + 2; }` — s after? | 2 / 6 / 0 | 2 | M03: 6 · M05: 2 |
| P_T6_a | T6 | Window `low = 0, high = 8`, `mid = 4`; target > `a[4]`. After `low = mid;` the window is? | 4..8 / 5..8 / 0..8 | 4..8 | D02: 5..8 · M02: 4..8 |
| *(P_T2_a reused)* | T6 | `int i = 0; while (i < 3) { fire(); }` — shots? | 3 / never stops / 0 | never stops | M02: 3 · D02: never stops |
| P_T7_a | T7 | `int h(int n) { return h(n - 1); }` — `h(2)`? | 0 / never stops / 2 | never stops | D05: 0 · D06: never stops |
| P_T7_b | T7 | `int g(int n) { if (n == 0) return 1; return g(n); }` — `g(2)`? | 1 / never stops / 2 | never stops | D06: 1 · D05: never stops |
| P_T8_a | T8 | `int f(int n) { if (n == 0) return 0; f(n - 1); return n; }` — `f(3)`? | 3 / 6 / 0 | 3 | D07: 6 · M10: 3 |
| P_T8_b | T8 | `int dbl(int x) { printf("%d", x + x); return 1; }` `int y = dbl(5);` — y? | 1 / 10 / error | 1 | M10: 10 · D07: 1 |
| P_T9_a | T9 | `for (i = 0; i < 3; i++) { if (a[i] == 5) return i; else return -1; }` with `a = {2, 5, 9}` — result? | 1 / −1 / 0 | −1 | D01: 1 · M03: −1 |
| *(P_T4_b reused)* | T9 | as above | 2 / 6 / 0 | 2 | M03: 6 · D01: 2 |
Reused probes are the same JSON object with several entries in its `belief` map; a probe is never shown twice to the same learner within one diagnosis.

### 6.4 Probe selection by expected information gain (`bayes/eig.py`)
```
candidates = probes whose belief map touches any of the top-3 classes, not yet asked in this attempt
P(a)       = Σ_k p(k) · P(a | k)
EIG(j)     = H(p) − Σ_a P(a) · H(p(· | a))          (bits)
ask argmax EIG if EIG ≥ 0.10; stop when max p ≥ 0.85, or 2 probes asked, or no candidate ≥ 0.10
fallback   : if the bank has no candidate, use the rule table {twin set → probe_a}
```
~40 lines of numpy. The same likelihood code powers MCQ (Strong D12) and the mission's predict item.

### 6.5 Predict items in missions
Each mission's `predict_item` (§3.3) is scored with the same likelihood table *before* the code runs. Example (P03): code with `i <= n` over 3 cells, "How many cells are read?" Correct = 4. If the learner says 3, that supports M01/M08 (both believe it reads 3), which matters for the later knowledge update, not for T1 separation. This is why T1 needs the dedicated probes.

### 6.6 Exam trace items (v3)
The exam's no-code items (§8.5) are scored with the same likelihood table. They come from a separate **exam item bank** (`data/exam_items.json`, ~24 items: the 17 trap items in §8.3 are *not* reused, so the exam doesn't leak into reassessment), each with `belief` answers for 1–3 classes and an interpreter-verified correct answer.

---

## 7. Intervention engine (`ml/learner/interventions.py`, `fixer.py`, `counterexample.py`)

### 7.1 Policy
```
modality order per class:
  M01, M02, M07(for/while) : trace_timeline  → counterexample → minimal_fix
  M08                      : memory_strip    → counterexample → minimal_fix
  M03, M05                 : trace_timeline (value of total per pass) → minimal_fix
  M04                      : value_meter (exact vs truncated) → counterexample
  M06, M07(if)             : trace_timeline (x before/after the condition) → minimal_fix
  M10                      : trace_timeline (printed vs returned) → minimal_fix
  D01                      : trace_timeline (flag `early_return` at iteration 0) → counterexample (target at index 1) → minimal_fix
  D02                      : window_strip (low/mid/high per iteration, frozen window flagged) → minimal_fix
  D03                      : memory_strip (write-by-write replay, duplicated value flagged) → counterexample (2-element array) → minimal_fix
  D04                      : memory_strip (pass-by-pass; shows only one pass happened) → counterexample (reverse-sorted input)
  D05, D06                 : call_stack (frames with args; missing base / non-shrinking arg flagged) → counterexample (n = 1, 2) → minimal_fix
  D07                      : call_stack (return values travelling down; the dropped one flagged) → minimal_fix
  D08                      : value_meter (char code vs string literal) → minimal_fix
  novel / OTHER            : trace_timeline (generic) + minimal_fix if any fixer passes, else reference
next modality = first one not yet tried for (learner, class); never repeat a failed one
log (learner, class, modality, outcome) for a future bandit (roadmap)
```
Every package includes **all three panels** (timeline/strip/meter + counterexample + fix); `modality` only decides which panel is primary and which question is asked.

### 7.2 Counterexample search
Enumerate small inputs (`n ∈ {1, 2, 3}`, arrays from `[[3], [3, 5], [2, 4, 6], [1, 2, 3, 4]]`, scalars from the problem's test ranges; **DSA pools:** sorts `[[2, 1], [3, 1, 2], [4, 3, 2, 1]]`, searches target at index 0 / 1 / last / absent, strings `"a"`, `"ab"`, `"aba"`, `"abca"`, recursion `n ∈ {0, 1, 2, 3}` and `n = 10` for digits). Pick the **smallest** input where the learner's output ≠ the reference's. Prefer inputs where the class's **belief answer equals the intended output** (cognitive conflict: "you expected 12; your code gives 12 + static"). Return `{input, intended, yours, effect_diff}`.

### 7.3 Verified minimal fixer
Per class, candidate source edits at AST-located sites (inverse operators, deliberately **broader** than the generator):
| Class | Candidate edits (try in order, ≤ 5) |
|---|---|
| M01 / M08 | `<=`→`<` on the main bound; `<`→`<=`; `n-1`→`n`; start `1`→`0`; `a[i+1]`→`a[i]`; `a[n]`→`a[n-1]`; `a[1]`→`a[0]` (first-element contexts) |
| M02 | append `i++;` to the loop body; `i--`→`i++`; move an update out of an `if` |
| M03 | hoist the in-loop init/declaration to before the loop |
| M04 | wrap the numerator in `(float)`; `2`→`2.0` in the division; move the cast inside |
| M05 | add `= 0` (accumulator/counter), `= 1` (product), `= a[0]` (max/min) to the declaration |
| M06 | `=`→`==` in conditions |
| M07 | delete the `;` after `if(...)/for(...)/while(...)` |
| M10 | `printf(fmt, e);` → `return e;` (remove a trailing `return 0;` if present) |
| D01 | delete the `else return …;` / `else flag = 0;` branch; move `return -1;` after the loop |
| D02 | `low = mid` → `low = mid + 1`; `high = mid` → `high = mid - 1` |
| D03 | rewrite `X = Y; Y = X;` as `int t = X; X = Y; Y = t;`; in rotate, save `a[0]` before the shift |
| D04 | wrap the single pass in `for (int p = 0; p < n - 1; p++)`; fix outer bound `i < 1` → `i < n - 1` |
| D05 | insert a base case before the recursive call: `if (n <= 0) return B;` with `B ∈ {0, 1}`, or `if (n == 0) return 0;` for array recursion (try each, keep the one that passes) |
| D06 | recursive arg `n` → `n - 1`; `n + 1` → `n - 1`; `n % 10` → `n / 10` |
| D07 | `f(x); return y;` → try `return y * f(x);`, `return y + f(x);`, `return f(x);` |
| D08 | `"c"` → `'c'` in comparisons; array-name `==` → call a char-by-char loop helper *(reference fallback if none)* |
Accept the **first candidate (fewest changed characters) that passes all tests**. Output `{kind: "minimal", code, changed_lines, rule}`. If none passes, return a reference correct variant (`kind: "reference"`), never identical to the learner's code. Measure **fixer coverage** on R (target ≥ 85% minimal).

### 7.4 Timeline builder
Align learner and reference traces on the display test by loop iteration; produce `[{step, line, vars, effect, flag}]` with `flag ∈ {extra, missing, void_read, static, reset, overwrite, empty_body}`. `question = {prompt: "At which value of i should the attack stop?", answer: <ref final i>}` (template per class).
**v3 builders:** `call_stack` → `[{depth, fn, args, returned, flag}]` from `call`/`ret` effects with flags `no_base`, `same_arg`, `growing_arg`, `dropped_value`, capped at 12 frames + `"…+N"`; `window_strip` → per iteration `{low, mid, high, cmp}` with flag `frozen` when (low, high) repeats; `memory_strip` (DSA) → write-by-write frames `{i, old, new}` with flags `duplicate`, `lost`, `one_pass_end`.

---

## 8. Learner model, knowledge tracing and resolution (`ml/learner/`)

### 8.1 Per (learner, misconception) state
`P(A_k)` = probability the misconception is currently **active**. Start at 0.10 (population prior). Stored in SQLite with a JSON evidence log.

### 8.2 Updates (2-state, BKT-style; parameters hand-set from item design)
One rule for a code attempt. There is no `max(P, p_k)` write.
- **Failed code task whose diagnosis top-1 is k** (posterior ≥ 0.5): likelihood ratio with `P(fail_k | A) = e_ik` and `P(fail_k | ¬A) = 0.03`. `e_ik` is the problem's authored exposure for k; when the problem lists none, use `FAIL_SIGNATURE_IF_ACTIVE` (0.5). This is the same update the exam uses (§8.5.3).
  - `P(A) ← P(A)·e_ik / (P(A)·e_ik + (1−P(A))·0.03)`
- **Passed code task with `latent` = k** (§5.5): the same ratio with `e_ik × 0.5`.
- **Passed code task, no latent class:** the item-response row below ("correct").
- **Intervention completed:** learning transition `P(A) ← P(A)·(1 − ℓ)`, ℓ = 0.35.
- **Item response** with guess `g = P(correct | A)` and slip `s = P(wrong | ¬A)`, for traps, probes, MCQs, trace items and code tasks that are not a k-signature failure:
  - correct: `P(A) ← P(A)·g / (P(A)·g + (1−P(A))·(1−s))`
  - wrong: `P(A) ← P(A)·(1−g) / (P(A)·(1−g) + (1−P(A))·s)`
- **Hint used:** `g ← min(0.9, g + 0.2)` for that item (passing with help is weaker evidence).
- **Forgetting before a ghost return:** `P(A) ← P(A) + φ·(1 − P(A))`, φ = 0.05 per intervening level.
- **State:** after the update, §8.4 fires from the new `P(A)`. A diagnosis moves the state to ACTIVE when the updated `P(A) ≥ 0.5` (and to RELAPSED if it was STABLE or MASTERED).

| Item type | g = P(correct \| active) | s = P(wrong \| resolved) | Why |
|---|---|---|---|
| same-family code task | 0.45 | 0.10 | the fix pattern can be copied |
| transfer code task (different family) | 0.30 | 0.15 | needs the idea; tests may still miss the boundary |
| **trap prediction** (belief ≠ correct) | **0.15** | 0.10 | a believer gives the specific wrong answer |
| probe / MCQ with belief distractor | derived from §6.2: (1−p_b)·q ≈ 0.20 | 0.10 | |
| ghost return (later, unscaffolded) | 0.25 | 0.15 | |
| **exam coding item** (v3) | pass with no latent class: 0.30. A `fail_k` or a `latent` class uses the §8.2 ratio instead of this row | 0.15 | no hints, new surface |
| **exam trace item** with a k-belief option (v3) | from §6.2 | 0.10 | |

### 8.3 Trap items (`data/items.json`), one per class (17), verified by the interpreter
| Class | Trap | Correct | Belief answer |
|---|---|---|---|
| M01 | `for (i = 2; i <= 6; i++) fire();` shots? | 5 | 4 |
| M02 | `int i = 0; while (i < 4) { fire(); }` shots? | never stops | 4 |
| M03 | `for (i = 0; i < 3; i++) { int s = 0; s += 5; }` s at the end of the last pass? | 5 | 15 |
| M04 | `int a = 7, b = 2; float r = a / b;` r? | 3.0 | 3.5 |
| M05 | `int c; c++;` c? | unpredictable | 1 |
| M06 | `int x = 0; if (x = 0) fire(); else open_door();` what happens? | door opens | fires |
| M07 | `int shield = 10; if (shield < 0); fire();` does it fire? | yes | no |
| M08 | `int a[4] = {3,5,7,9};` what is `a[4]`? | outside the array | 9 |
| M10 | `int f(int x){ printf("%d", x*2); return 0; }` `y = f(4);` y? | 0 | 8 |
| D01 | `for (i = 0; i < 4; i++) { if (a[i] == 9) return i; else return -1; }` with `a = {1, 9, 3, 4}` → result? | −1 | 1 |
| D02 | `low = 2, high = 3, mid = 2`, target > `a[2]`; after `low = mid;` the next mid is? | 2 (stuck) | 3 |
| D03 | `a = {4, 7}; a[0] = a[1]; a[1] = a[0];` → a? | {7, 7} | {7, 4} |
| D04 | `{4, 3, 2, 1}` after **one** bubble pass? | {3, 2, 1, 4} | {1, 2, 3, 4} |
| D05 | `int f(int n) { return n + f(n - 1); }` → `f(2)`? | never stops | 3 |
| D06 | `int f(int n) { if (n <= 0) return 0; return 1 + f(n); }` → `f(2)`? | never stops | 2 |
| D07 | `int fact(int n) { if (n == 1) return 1; fact(n - 1); return n; }` → `fact(4)`? | 4 | 24 |
| D08 | `char c = 'a'; if (c == "a") fire(); else open_door();` what happens? | door opens | fires |

One trap per class is burned once the learner leaves PROBATION. A RELAPSED learner needs another. `predict_output` quiz items (05 §3.1) whose `belief` map contains k are a second trap pool for that class: scored with the trap row of the table above, and never the probe that was already asked in the original diagnosis.

**Transfer families** (different from where the misconception was found): M01 {count_loop, countdown_loop, prefix_loop} · M02 {while_progress, count_loop} · M03/M05 {array_accumulate, accumulate_product, array_count_if} · M04 {average, ratio} · M06/M07 {equality_check, branch_bands, while_progress} · M08 {index_access, array_accumulate, array_max} · M10 {return_value, index_access, equality_check, pairwise_check} · **D01** {linear_search, pairwise_check, string_two_pointer, flag_search} · **D02** {binary_search, guess_halving} · **D03** {bubble_sort, selection_sort, two_pointer_swap, shift} · **D04** {bubble_sort, selection_sort} · **D05/D06/D07** {rec_product, rec_digits, rec_array} · **D08** {string_count, string_two_pointer}. Main-game classes may also transfer onto DSA families (e.g. M01 → pairwise_check) once the learner has opened the Trials.

### 8.4 State machine (`state_machine.py`)
```
UNSEEN ──updated P(A)≥0.5──▶ ACTIVE ──intervention start──▶ TREATING ──intervention done──▶ PROBATION
PROBATION ──[P(A)<0.15 ∧ trap passed ∧ ≥1 transfer in a different family passed]──▶ STABLE
PROBATION ──[P(A)>0.5 after an item]──▶ TREATING (next modality)
STABLE ──[ghost return passed ∧ P(A)<0.10]──▶ MASTERED
STABLE ──[exam item with e_ik ≥ 0.4 passed ∧ P(A)<0.10]──▶ MASTERED      (v3: the exam is a ghost return)
UNSEEN ──[exam update leaves P(A)≥0.5]──▶ ACTIVE                          (v3: "NEW" finding in the debrief)
STABLE / MASTERED ──[updated P(A)≥0.5, or a ghost item failed]──▶ RELAPSED ──intervention──▶ TREATING
```
Transitions read the `P(A)` produced by §8.2. They do not copy the model's posterior.
`/reassess` returns `conditions[]`, e.g.
`[{"id":"p_active","label":"Misconception probability < 0.15","met":false,"detail":"0.31"}, {"id":"trap","label":"Trap item passed","met":false,"detail":"predicted 9, actual: outside the array"}, {"id":"transfer","label":"Different-family transfer passed","met":true,"detail":"P09 max_shield"}]`.

**Why this fixes Da GOATS' admitted weakness:** their "model no longer detects it" check collapses into "tests pass". Here a passed transfer is *weak* evidence (g = 0.30), the **trap** (no execution, g = 0.15) carries the weight, and repeated evidence accumulates instead of four independent yes/no gates.

### 8.5 Adaptive exam engine — Deep Space Trials (`ml/exam/`, v3)

**Goal:** in ~25 minutes, (a) estimate which of the 17 misconceptions are active, especially the D-classes; (b) **recheck main-game STABLE misconceptions on DSA surfaces** (the final ghost return); (c) give a fair per-sector picture. It's a diagnostic, not a grade.

#### 8.5.1 Item pool
- **Coding items:** Q01–Q18. Fields: `sector`, `difficulty ∈ {1,2,3}`, `exposure {class: e_ik}` (§3.3.1), 2 sample tests.
- **Trace items:** `data/exam_items.json`, ~24 (≈ 5 per sector), no execution, 3 options, `belief` map for 1–3 classes, correct answer verified by the interpreter. Examples: Searching "binary search on `{1,3,5,7,9}` for 7: how many mids are checked?"; Sorting "after the 2nd outer pass of bubble sort on `{5,1,4,2}`?"; Array tech "`reverse` with `i < n` on `{1,2,3}` gives?" (M01 belief: `{3,2,1}`); Strings "`strlen("abc")`?" (M01 belief: 4); Recursion "how many calls does `fact(3)` make?". Their exposure = `P(belief answer | A_k)` from §6.2.

#### 8.5.2 Learner state at exam start
- `P(A_k)` for all 17 classes from the knowledge model; D-classes start at the population prior 0.15 if unseen.
- Sector rating `θ_s` (Elo-style), start 1300 (+100 if all 3 planets are complete). Item rating `r_i` = 1200 / 1400 / 1600 for difficulty 1 / 2 / 3. `P_pass(i) = 1 / (1 + 10^((r_i − θ_s) / 400))`.

#### 8.5.3 Observation model
- Coding item i: outcome `o ∈ {pass, fail_k, fail_other}` where `fail_k` = silent diagnosis top-1 is k with posterior ≥ 0.5. The knowledge update is the §8.2 ratio: `P(fail_k | A_k) = e_ik`, `P(fail_k | ¬A_k) = 0.03`. A pass with a `latent` class uses `e_ik × 0.5`. There is no second, exam-only formula.
- Trace item: answer likelihood from §6.2.
- Classes are updated independently (stated approximation).

#### 8.5.4 Selection rule (`exam/select.py`)
```
Each item is a set of independent per-class binary tests (classes are updated independently, §8.5.3).
For every k with e_ik ≥ 0.2:
  P(fail_k | A_k) = e_ik,  P(fail_k | ¬A_k) = 0.03
  P(fail_k) = P(A_k)·e_ik + (1 − P(A_k))·0.03
  EIG_k = H(P(A_k)) − [ P(fail_k)·H(P(A_k) | fail_k) + (1 − P(fail_k))·H(P(A_k) | not fail_k) ]
EIG(i) = Σ_k EIG_k                                          (binary entropies, bits)
cost(i)  = 1.0 for coding (≈ 4 min), 0.3 for trace (≈ 1 min)
S(i)     = EIG(i) / cost(i)
         + 0.5 · [sector(i) not yet covered]
         − 0.4 · | P_pass(i) − 0.6 |                       (aim for ~60% success)
         + 0.3 · Σ_{k ∈ STABLE, untested in this exam} e_ik  (ghost-return bonus)
pick argmax S(i) subject to:
  10 items = 5 coding + 5 trace   (demo mode: 2 coding + 1 trace)
  every sector gets ≥ 1 item; ≤ 2 coding items per sector; no two consecutive items from the same sector
  item 1 = a difficulty-1 coding item from the sector with the highest prior EIG
  ties broken by a seed derived from (learner_id, exam_id) → reproducible
reason string = top-2 classes by EIG contribution + the bonus that fired, e.g.
  "Q06 bubble_sort — most informative about Boundary Drift (stable since Loops → recheck) and Cargo Overwrite; covers Sorting."
```

#### 8.5.5 Updates after each answer (silent: nothing shown to the learner)
1. Coding: run gate → interpreter → diagnoser → Bayes layer with **probes suppressed** (deferred to the debrief); store the diagnosis.
2. Knowledge update (§8.2) with item types `exam_code` / `exam_trace`.
3. Elo: `θ_s ← θ_s + 40 · (outcome − P_pass)` (pass = 1, fail/skip = 0).
4. State transitions: §8.4 (MASTERED via exam, RELAPSED, NEW → ACTIVE).
5. Time-out: unanswered items are *unobserved* (no evidence), and the report says so.

#### 8.5.6 Debrief report (`exam/report.py`)
- `sectors[]`: items, passed, rating before/after.
- `findings[]` with status: **NEW** (P(A) ≥ 0.5, was UNSEEN or ≤ 0.3), **RELAPSED**, **HELD** (→ MASTERED), **UNCERTAIN** (0.3–0.7 or an ambiguous twin diagnosis), **NOT_TESTED** (no item with e ≥ 0.2). Each finding carries its evidence items (diagnosis objects, read-only).
- `deferred_probes[]`: up to 3, for UNCERTAIN findings, chosen by EIG (§6.4); answering via `/exam/probe` updates the findings.
- `recommendations[]`: per NEW/RELAPSED finding, 2 practice problems from its transfer families that weren't in the exam.
- `adaptivity_log[]`: `{order, item_id, sector, eig_bits, reason, outcome}`.

#### 8.5.7 Fallback (same API)
Fixed blueprint if the selector fails a checkpoint: `[Q01, xt_sort_1, Q06, xt_rec_1, Q17, xt_str_1, Q14, xt_search_1, Q10, xt_bsearch_1]`; `reason = "fixed blueprint"`. If strings were cut (D08, Q13–Q15), use `[Q01, xt_sort_1, Q06, xt_rec_1, Q17, xt_search_1, Q10, xt_bsearch_1, Q03, xt_arr_1]` and skip the strings sector. E14 compares both anyway.

#### 8.5.8 Honest framing
Exposure values, Elo constants and weights are **hand-set design choices**. Elo here is a convenience, not a calibrated IRT model. Classes are updated independently. The adaptive policy is evaluated only in simulation (E14), under parameter ranges different from the engine's.

---

## 9. Evaluation (`ml/eval/`, one script per experiment, `run_all.py` → `docs/metrics.json` + PNGs)

### 9.1 Splits (fixed seeds, committed as `data/splits.json`)
| Slice | Contents | Used for | Touch policy |
|---|---|---|---|
| TRAIN | A+E+AMB rows of the 28 training problems (13 main + 15 DSA) | fit, 5-fold grouped CV, calibration, τ | free |
| HOLDOUT-P | A+E+AMB rows of P04, P09, P13, Q04, Q09, Q16 | unseen-problem test (reported per domain) | evaluate after model freeze |
| TRAIN-MAIN (v3) | TRAIN restricted to the 13 main problems | cross-domain experiment E15 (separate retrain) | free |
| OPHOLD | for each class with ≥ 2 operators, one `op_id` removed from TRAIN (`op_variant` = `op_id`, §3.5) | unseen surface-form test (separate retrains) | free |
| LUCK | verified "passes-by-luck" mutants | resolution argument, CORRECT false-negatives | free |
| R-team | 30 realistic items (04 decision) | test + adversarial validation (inputs only) | evaluate once at the end; inputs may be inspected |
| **R-blind** | 40 realistic items by FE (25 main + 15 DSA) | **headline** | **never inspected; evaluated once** |
| U | 30 U1/U2 items | unseen-class test | evaluate once |
| X | ITSP slice (Strong) | real-student test | evaluate once |
Leakage asserts: no `ast_hash` appears in two slices; R/U/X hashes never in TRAIN; operator-holdout retrains never see the held-out `op_variant`.

### 9.2 Experiments

| ID | Question | Protocol | Metrics | Plot / table |
|---|---|---|---|---|
| **E1** | How well does it generalise to new problems (in-distribution style)? | 5-fold StratifiedGroupKFold by problem, stratified by domain, on TRAIN (§5.3) | macro-F1 mean ± std, accuracy, per-class P/R/F1, per-problem acc | confusion matrix (OOF) |
| **E2** | Unseen problems | train on TRAIN, test HOLDOUT-P | macro-F1, per-problem acc, cluster-bootstrap CI | table |
| **E3** | Unseen surface forms | per class with ≥ 2 operators: drop one `op_id` (`op_variant` = `op_id`), retrain, test on it | accuracy on the held-out operator (per class) | bar chart |
| **E4** | **Realistic code (headline)** | final model on R-blind, then R-team | macro-F1 + 95% bootstrap CI, per-class, confusion, fixer coverage, κ of the labels | table + confusion |
| **E5** | **Twins** | STRUCTURAL: pair accuracy on twin rows (HOLDOUT-P + R). HARD: ambiguity-flag rate (status = ambiguous); then **simulated probing**: answers drawn from the true class with true `p_b ∈ {0.4 … 0.9}` while the engine assumes 0.6 | pair acc; flag rate; post-probe acc vs `p_b` | line chart (acc vs p_b), pre/post bars |
| **E6** | **Unseen misconceptions** | (a) **LOCO strict:** for each class k, retrain without k **and without** `CLASS_DEFINING_FEATURES[k]`; do not run fixer k; novelty score on all of k's rows vs seen-class HOLDOUT-P rows → AUROC. (b) **LOCO naive:** the same retrain but the defining features stay, reported next to strict so the gap is visible. (c) U1/U2: abstain rate, and which class it is confused with when not abstained | per-class AUROC, mean, both ways; abstain precision; risk–coverage | bar chart, risk–coverage curve |
| **E7** | Calibration | OOF before/after T | ECE, NLL, Brier | reliability diagram |
| **E8** | Baselines | same splits for majority, rules, TF-IDF+LR, TF-IDF+LGBM, and DeepSeek zero-shot on R via `ml/text/llm_client.py` (§5.7) | macro-F1 on E1, E2, E4 | table |
| **E9** | Feature ablation | A → A+B → A+B+R → A+B+R+C, and **AST-only**. One extra offline row adds group F (§4.4); the shipped model is the row without F | macro-F1 on E1 and R | grouped bars |
| **E10** | **Resolution** | simulated learners × policies (§9.3) | false-resolve, false-not-yet, items-to-decision | table + bar |
| **E11** | Robustness | semantics-preserving perturbations of R + HOLDOUT-P (rename, reformat, comments, dead var, for↔while); near-miss CORRECT false-alarm rate | % unchanged predictions; false-alarm rate | table |
| **E12** | Failure audit | all wrong predictions with confidence ≥ 0.6 on E2/E4 | code, true, pred, conf, top-3 SHAP, written "why" | table (Lab Report) |
| *E13* | Real students (Strong) | X slice | macro-F1 + CI, AST-only vs full | table |
| **E14** (v3) | **Does the adaptive exam find more?** | simulated learners with hidden misconception profiles (§9.3b) × exam policies: adaptive (§8.5), fixed blueprint, random (same 5 + 5 composition) | profile-recovery F1 (active classes found vs truth), Brier of final `P(A_k)`, ghost-return coverage of STABLE classes, items-to-first-finding | line chart (F1 vs items asked), table |
| **E15** (v3) | **Cross-domain transfer** | (a) train on TRAIN-MAIN only, test M-class rows *inside DSA problems*; (b) full model, metrics split main vs DSA; (c) domain-shortcut check: can a stump on `t_*` features predict D vs M class? | M-class recall/F1 on DSA surfaces; per-domain macro-F1; shortcut AUC | grouped bars |

**Per-domain reporting (v3):** E1, E2, E4, E5, E6 are reported overall **and** separately for main vs DSA problems and for M- vs D-classes.

**Statistics:** bootstrap with 1,000 resamples for CIs; cluster bootstrap by `problem_id` for E1/E2; report n next to every number; never report a metric without its slice name.

### 9.3 Resolution simulation (`eval/sim_learners.py`)
Learner types (ground truth hidden from the engine):
| Type | Truth after the intervention | Behaviour |
|---|---|---|
| truly_fixed | inactive | correct with prob 1 − s_sim on every item |
| pattern_copier | **active** | same-family code passes 0.9; transfer code passes 0.5; trap → belief answer with p_b_sim |
| lucky_guesser | **active** | code passes 0.35; MCQ/trap uniform random |
| forgetful | inactive, then active before the ghost return with prob 0.6 | as truly_fixed, then as pattern_copier |
| slow | active after the 1st intervention, inactive after the 2nd | |
Policies on the same item budget (≤ 4 items + 1 ghost):
- **naive:** resolved after 1 correct follow-up.
- **4-check (Da GOATS-style):** new task ∧ tests pass ∧ model p_k < 0.25 (≈ always true when tests pass) ∧ concept MCQ correct.
- **ours:** §8.
Metrics: **false-resolve** = P(declared STABLE | truly active), **false-not-yet** = P(not STABLE within budget | truly inactive), mean items. Simulation parameters (`g_sim, s_sim, p_b_sim`) are drawn from ranges **±50% around** the engine's values, so the engine is never tested only under its own assumptions. 2,000 learners per type × 20 seeds → mean ± 95% interval.
*Stated caveat:* simulated learners embody our assumptions about learner behaviour. The table shows how the policies behave **if** learners act like this, not that real learners do.

### 9.3b Exam simulation (`eval/sim_exam.py`, E14)
- 2,000 simulated learners. Hidden truth: each of the 17 classes active independently with probability 0.25 (D-classes) / 0.15 (M-classes); 30% of learners carry 1–3 STABLE main classes (truly inactive) from "playing the game".
- Responses: coding item i fails with k's signature with probability `e_ik_sim` if k active (`e_ik_sim` drawn ±50% around the authored `e_ik`), passes with probability `1 − max_k(...)` × skill noise; trace items answered with `p_b_sim ∈ [0.4, 0.9]`.
- Policies: adaptive / fixed blueprint / random, each 10 items (5 + 5). Same diagnoser noise model for all (confusion matrix from E1 OOF used to corrupt `fail_k` into another class).
- Metrics: profile-recovery F1 after each item (curve), final Brier, % of STABLE classes rechecked, mean items to the first true finding. 20 seeds → mean ± 95% interval.
- Caveat on the Lab Report: "simulated learners follow our assumptions; this shows the selection logic works as designed, not that it is optimal for real learners."

### 9.4 `metrics.json` schema (consumed by D10)
```json
{
  "model_version": "diagnoser_3f9a12cd", "generated_at": "...", "effective_n": {"rows": 4012, "unique_hash": 1730, "unique_prog_op": 812},
  "cards": [
    {"id": "E1", "title": "Grouped CV by problem", "slice": "TRAIN (11 problems, 5 folds)", "n": 3100,
     "metrics": {"macro_f1": 0.84, "macro_f1_std": 0.05, "acc": 0.88}, "per_class": {...},
     "plots": ["e1_confusion.png"], "caveat": "Synthetic, same generator as training. Optimistic."}
  ],
  "baselines": [{"name": "rules", "E1": 0.71, "E2": 0.66, "E4": 0.52}],
  "domains": {"main": {"E1": 0.0, "E2": 0.0, "E4": 0.0}, "dsa": {"E1": 0.0, "E2": 0.0, "E4": 0.0}},
  "cross_domain": {"m_recall_on_dsa_trained_main_only": 0.0, "shortcut_auc": 0.0},
  "exam_sim": {"curves": {"adaptive": [], "fixed": [], "random": []}, "final_f1": {}, "stable_recheck_rate": {}},
  "resolution": {"rows": [{"policy": "naive", "type": "pattern_copier", "false_resolve": 0.87}]},
  "failure_audit": [{"id": "R-b-07", "code": "...", "true": "M03", "pred": "M05", "conf": 0.71, "why": "..."}]
}
```
(Numbers above are placeholders for shape only.)

---

## 10. EDA plan (`ml/eda/eda.py` → `docs/eda.md` + `docs/eda/*.png`)

Run after dataset v1 (CP2) and again after every generator change. Each check has an **action rule**.

| # | Check | Output | Action rule |
|---|---|---|---|
| 10.1 | Counts: class × problem, class × source, soft-label groups, two-bug share | heatmap | any class < 80 rows → add sites/variants before training |
| 10.2 | Verification: drop rate per operator, passes-by-luck rate per class | table | operator drop > 40% → fix the operator |
| 10.3 | Duplicates: `n_unique_hash / n_rows`; largest hash groups; **cross-label hash collisions** | table | collision other than {M01, M08} → generator bug |
| 10.4 | **Shortcut scan:** depth-1 stump per feature (macro-acc); code length per class (KS test vs CORRECT) | ranked table, violin plot | a non-defining feature (e.g. style artefact) with stump acc > 0.6, or length separating a class → add augmentation |
| 10.5 | Predicate matrix: mean of each class's `CLASS_DEFINING_FEATURES` per class | heatmap (should be diagonal-dominant) | off-diagonal bleed > 0.3 → check operator scope |
| 10.6 | Feature health: constant/near-constant, NaN rate, pairs with \|corr\| > 0.95 | list | drop constants; keep one of each correlated pair |
| 10.7 | **Distribution shift:** adversarial validation TRAIN vs **R-team** (inputs only; grouped CV LightGBM) | AUC + top drifting features | AUC > 0.85 → add augmentations that cover the drifting features (e.g. while-forms, debug prints); **never** look at R-blind |
| 10.8 | 2-D projection (PCA; UMAP if installed) coloured by class and by source | scatter | classes overlapping heavily → expected for T1; anything else → look at features |
| 10.9 | **Outliers:** IsolationForest (contamination 0.02) on TRAIN features | top-20 table with code | inspect; generator bugs → fix; genuine weird code → keep |
| 10.10 | Trace-event co-occurrence per class | heatmap | sanity: `oob_read` should concentrate in M08/AMB; `uninit_read` in M05 |
| 10.11 | Realistic set: class counts, κ, passes-by-luck items | table | κ < 0.6 → refine class definitions in the dataset card |
| 10.12 (v3) | **Exposure sanity:** for each DSA problem, empirical rate at which class-k mutants fail with k's signature vs the authored `e_ik` | table | \|empirical − authored\| > 0.3 → re-author `e_ik` from the empirical value (allowed: it's synthetic data, not a test set) |
| 10.13 (v3) | **Domain balance & masking:** class × domain counts; % of rows where masking zeroes the true class (must be 0) | heatmap + count | any masked true label → fix the precondition feature |

---

## 11. Serving API contract (`server/app/`, FastAPI)

All responses include `model_version` and `latency_ms`. CORS: `localhost:*`. Errors never return stack traces.

| Method | Path | Request | Response |
|---|---|---|---|
| GET | `/health` | — | `{ok, model_version, n_features, db: "ok"}` |
| GET | `/problems` | — | `[{problem_id, name, planet, sector, difficulty, world, signature, starter, prompt, predict_item, markers, sample_tests}]` (public fields only: no hidden tests, no correct variants, no exposure) |
| GET | `/problems/{id}` | — | one problem (public fields) |
| POST | `/learner` | `{callsign}` | `{learner_id}` |
| POST | `/learner/{id}/seed` | — | seeded demo learner (3 planets complete; M01 STABLE, M08 ACTIVE, M06 MASTERED, rest UNSEEN; ~12 attempts) |
| POST | `/run` | `{problem_id, code, sample_only?: bool}` | `{gate, trace, tests}` (no diagnosis; Predict reveal and the exam's "Run samples") |
| POST | `/attempt` | `{learner_id, problem_id, code, events[]}` | `{attempt_id, gate, trace, tests, diagnosis}` |
| POST | `/probe/answer` | `{learner_id, attempt_id, probe_id, answer}` | `{diagnosis}` (updated; may contain another `next_probe`) |
| POST | `/intervene` | `{learner_id, attempt_id, class, modality?}` | intervention package (§11.2) |
| POST | `/reassess` | `{learner_id, class, item_id, item_type, result}` | `{state, p_active, conditions[], resolved_level, next_item}` |
| GET | `/learner/{id}` | — | `{learner_id, misconceptions{}, nodes{}, attempts[]}` |
| POST | `/lab/diagnose` | `{problem_id, code, prediction?, probe_answers?: [{probe_id, answer}]}` | `{gate, trace, tests, diagnosis}`; stateless (no learner prior, nothing stored); probe answers are applied in order, so Bug Lab can replay the hard-twin flow |
| GET | `/metrics` | — | `metrics.json` (§9.4) |
| POST | `/exam/start` (v3) | `{learner_id, length?: 10, demo?: bool}` | `{exam_id, item, progress: {k, n}, time_limit_s: 1500}` |
| POST | `/exam/answer` (v3) | `{exam_id, item_id, code?, answer?, ms, sample_runs, skipped?}` | `{logged: true, next_item \| null, progress}`; **no diagnosis returned** |
| POST | `/exam/finish` (v3) | `{exam_id}` | `{report}` (§11.4) |
| POST | `/exam/probe` (v3) | `{exam_id, probe_id, answer}` | `{report}` (findings updated) |
| GET | `/exam/{id}/report` (v3) | — | `{report}` |

`gate` object everywhere: `{"code": "G0|G1|…|G7|G4b|G5b", "message": "…"}`. `tests`: `{"passed": 3, "total": 5, "results": [{"args": […], "expected": …, "got": …, "pass": true}]}`.
`item` object (exam): `{"item_id": "Q06", "kind": "coding|trace", "sector": "sorting", "difficulty": 2, "prompt": "…", "signature": "…", "starter": "…", "markers": ["i","j"], "sample_tests": […]}` or for trace `{"item_id": "xt_sort_1", "kind": "trace", "code": "…", "question": "…", "options": […]}`.

### 11.1 `diagnosis` object
```json
{
  "status": "confident|ambiguous|novel|two_bug|correct|gate",
  "posterior": {"M01": 0.46, "M08": 0.44, "M07": 0.03, "CORRECT": 0.01, "OTHER": 0.02, "...": 0.0},
  "top": [{"id": "M01", "p": 0.46, "name": "Boundary Drift", "subtitle": "Loop runs one step too many or too few", "band": "Possible"},
          {"id": "M08", "p": 0.44, "name": "Index Origin Fault", "subtitle": "Arrays start at 0; last cell is n−1", "band": "Unsure"}],
  "twin_set": "T1", "two_bug": false,
  "novelty": {"knn_dist": 1.8, "tau_d": 3.2, "p_max": 0.46, "tau_p": 0.55, "abstain": false},
  "evidence": [
    {"type": "CODE", "text": "Loop condition uses `<=` (line 4).", "line": 4, "feature": "a_main_cond_op_le", "weight": 0.9},
    {"type": "RUN", "text": "Reads `cells[3]`, one cell past the end.", "line": 5, "feature": "b_oob_read_idx_eq_n", "weight": 0.7},
    {"type": "RUN", "text": "This code is identical for both explanations. Asking one question."}
  ],
  "next_probe": {"probe_id": "P_T1_a", "prompt": "Index of the last valid cell?", "code": "int a[5];", "options": ["4", "5", "depends on values"], "eig_bits": 0.71},
  "probes_asked": [], "latent": null, "model_version": "diagnoser_3f9a12cd"
}
```

### 11.2 Intervention package
```json
{
  "class": "M08", "modality": "memory_strip", "next_modalities": ["counterexample", "minimal_fix"],
  "copy": ["Arrays start at cell 0.", "An array of n cells ends at cell n−1; cell n is the void."],
  "question": {"prompt": "Which is the last cell the loop should read?", "answer": 2},
  "timeline": [{"step": 3, "line": 5, "vars": {"i": 3}, "effect": "read_void:3", "flag": "void_read"}],
  "memory_strip": {"array": "cells", "values": [2, 4, 6], "reads": [0, 1, 2, 3]},
  "counterexample": {"input": {"cells": [3], "n": 1}, "intended": 3, "yours": -858993457, "effect_diff": "1 extra void read"},
  "fix": {"kind": "minimal", "code": "...", "changed_lines": [4], "rule": "<= to <", "verified": true}
}
```

### 11.3 Non-functional
p95 `/attempt` < 300 ms on a laptop (interpreter + LightGBM + fixers for the top-3 classes only, cached by AST hash; see §2.5). `/exam/answer` < 400 ms (diagnosis + selection). Seeds fixed. SQLite tables: `learners`, `attempts`, `knowledge(learner_id, class, p_active, state, log_json)`, `events`. **Fixtures:** `server/fixtures/*.json` regenerated from the live API at T+10:45 by `scripts/dump_fixtures.py`, then copied to the FE. SQLite v3 adds `exams(exam_id, learner_id, state_json, log_json, report_json)`.

### 11.4 Exam report object
```json
{
  "exam_id": "ex_7c1", "learner_id": "…", "items_answered": 10, "time_used_s": 1320,
  "sectors": [{"sector": "sorting", "items": ["Q06", "xt_sort_1"], "passed": 1, "rating_before": 1400, "rating_after": 1372}],
  "findings": [
    {"class": "D03", "status": "NEW", "p_active": 0.82, "name": "Cargo Overwrite", "subtitle": "Swapping without a temp loses a value",
     "evidence": [{"item_id": "Q06", "diagnosis": {"status": "confident", "top": [{"id": "D03", "p": 0.88}], "evidence": ["…"]}}]},
    {"class": "M01", "status": "HELD", "p_active": 0.06, "state_after": "MASTERED", "evidence": [{"item_id": "Q08"}]},
    {"class": "D05", "status": "UNCERTAIN", "p_active": 0.48, "twin_set": "T7"}
  ],
  "deferred_probes": [{"probe_id": "P_T7_a", "prompt": "…", "code": "…", "options": ["0", "never stops", "2"], "for_class": "D05"}],
  "recommendations": [{"class": "D03", "problems": ["Q10", "Q11"]}],
  "adaptivity_log": [{"order": 2, "item_id": "Q08", "sector": "sorting", "eig_bits": 0.62, "reason": "Rechecks Boundary Drift (stable since Loops) on a pair loop; covers Sorting.", "outcome": "pass"}]
}
```

---

## 12. Repo layout and commands

```
ml/
  c_interp/      preprocess.py interp.py values.py harness.py tests/
  problems/      main/P01_fire_shots.json … P17_max_of_three.json   dsa/Q01_linear_search.json … Q18_array_sum_rec.json
  generate/      operators.py predicates.py augment.py verify.py ambiguity.py build_dataset.py card.py
  data/          dataset.jsonl splits.json realistic_blind.csv realistic_team.csv unseen.jsonl probes.json items.json exam_items.json itsp_slice.jsonl
  features/      ast_feats.py trace_feats.py relation_feats.py fix_feats.py extract.py evidence_templates.json
  model/         train.py calibrate.py novelty.py mask.py decide.py evidence.py baselines.py
  exam/          select.py observe.py report.py blueprint.json
  bayes/         likelihood.py eig.py verify_probes.py
  learner/       knowledge.py state_machine.py interventions.py fixer.py counterexample.py timeline.py
  eda/           eda.py
  eval/          e01_cv.py … e15_cross_domain.py sim_learners.py sim_exam.py run_all.py
  artifacts/     diagnoser_<sha8>/
server/app/      main.py gate.py store.py routes/ ; server/fixtures/
docs/            dataset_card.md model_card.md metrics.json eda.md *.png
Makefile:  make data | make eda | make train | make eval | make serve | make fixtures | make demo
```

---

## 13. Twelve-hour ML timeline with agent task cards

> **Superseded by `06_AGENT_WORK_PACKAGES.md`.** The cards below are the original hour-by-hour list. Agents follow 06, not this table. Where this table disagrees with 06 or with the amendments in §2–§11, §2–§11 and 06 win.

Each card = what to ask your coding agent + the acceptance test to run before moving on. Paste §2–§11 of this file as context with every card.

| Time | Card | Ask the agent to… | Acceptance test |
|---|---|---|---|
| 0:00–0:30 | **K0 Contracts** (both) | `server/fixtures/`: 3 `/attempt` (T1 ambiguous, M07 confident, CORRECT) + 1 DSA `/attempt` (D03), `/intervene`, `/reassess`, a 3-item exam flow (`/exam/start`, 2× `/exam/answer`, `/exam/finish` with report), stub `metrics.json`; FastAPI serving fixtures for every endpoint in §11 | FE gets valid JSON from every endpoint |
| 0:30–2:00 | **K1 Interpreter** | §2 incl. v3 strings, recursion trace, array-arg checks, all events + harness + unit tests | `pytest ml/c_interp` green |
| 2:00–3:15 | **K2 Problems** | 16 main + 18 DSA JSON per §3.3 (3 variants, 4–6 tests incl. adversarial + 2 samples, exposure, markers, forbid) | `python -m ml.problems.check`: all variants pass; coverage check passes |
| 3:15–4:05 | **K3 Generator** | Operators §3.5.1 (M + DSA surfaces + D), augmentations, verifier incl. DSA predicates, ambiguity, caps, card | ≥ 4,000 rows; every class ≥ 80; only {M01, M08} collisions |
| 4:05–4:15 | **K4 EDA v1** | 10.1–10.6, 10.9, 10.12, 10.13 | `docs/eda.md`; act on red rules |
| 4:15–5:00 | **K5 Features + model** | §4 A/B/R/C incl. DSA features + masking; §5 training, calibration, novelty, decide, evidence | E1 printed (overall + per domain); `/attempt` < 300 ms |
| 5:00–5:15 | **K6 Gate + Bug Lab + run** | §3.7.1 incl. G4b/G5b; `/lab/diagnose`; `/run` with `sample_only` | 8 gate fixtures behave as specified |
| 5:15–6:15 | **K7 Fixer + counterexample + builders + `/intervene`** | §7 incl. D fixers, `call_stack`/`window_strip` builders. Fixers run for the top-3 classes after the model (§4.4); do not retrain and do not add group F to the matrix | fixer coverage ≥ 80% on HOLDOUT-P mutants (main and DSA) |
| 6:15–7:00 | **K8 Bayes + probes** | §6 incl. 14 probes, exam trace items, verify_probes | T1 and T7 curl scripts reach ≥ 0.85 after probes |
| 7:00–8:00 | **K9 Knowledge + reassess + learner + exam** | §8.1–8.4 + SQLite + `/reassess`, `/learner`, seed; **§8.5 exam engine** + `/exam/*` + report | scripted learner: mission loop → STABLE; then exam (10 items) → report with ≥ 1 HELD and ≥ 1 NEW |
| 8:00–8:45 | **K10 Realistic set** (human) | ML writes R-team (30) by hand; collect FE's R-blind (40); second-rater labels; freeze hash | files committed, κ computed |
| 8:45–10:00 | **K11 Eval** | `run_all.py`: E1–E15 (E3/E6/E15 retrains in loops), sim learners, sim exam, plots, `metrics.json` | `/metrics` returns all cards |
| 10:00–10:45 | **K13 ITSP slice** (Strong) or buffer | §3.4 X protocol (60-min cap) | E13 card present |
| 10:45–11:15 | **K12 Fixtures + cards** (both) | dump fixtures from the live API; model card + dataset card | FE works with the backend killed |

**Checkpoint fallbacks:** K1 late → defer strings (drop Q13–Q15 and D08 until CP3). K2 late → drop Q12, Q18, P17. K5 late → serve the **rules baseline** behind the same API; keep training in the background. K8 late → rule-table probes (twin set → probe_a). K9 exam late → fixed blueprint (§8.5.7) behind the same `/exam/*` API. K11 late → E1, E2, E4, E5, E10, E14, E15, E12 first.

---

## 14. Risk register (ML)

| Risk | Signal | Mitigation |
|---|---|---|
| Perfect synthetic scores | E1/E2 ≥ 0.98 | Treat as a red flag: run the shortcut scan, check hash leakage, lead with R-blind |
| Model ≈ rules | E8 rules within 0.02 of ours | Say so; show where it differs (R, calibration, abstain, ambiguity) — the Bayes layer and resolution are the ML contribution beyond rules |
| HARD twin not flagged | flag rate < 0.5 | Check soft-label duplication weights; check that no feature differs between AMB groups |
| Fix-probe features dominate | F ablation on R drops > 0.2 when removed but E3 is weak | Keep F, report it; broaden fixers |
| Probe assumptions wrong | — | E5 sweep shows the dependency; keep the honest framing |
| Latency | `/attempt` > 300 ms | Cache by AST hash; cap fixer candidates to 3; skip group F for timeouts |
| Interpreter semantic bug | gcc mismatch / weird traces | Unit tests per event; differential gcc check if available |
| D-classes learned from problem type, not code | E15 shortcut AUC > 0.8; D-class errors only on held-out DSA problems | Coarse meta only; masking from code; generate M-classes on DSA surfaces |
| Recursion classes collapse (D05 vs D06 both = depth cap) | T7 pair accuracy < 0.7 | Strengthen `a_rec_arg_form_*`, `b_rec_arg_constant`; rely on P_T7 probes |
| Exam too long for the demo | — | `demo: true` → 3 items |
| Adaptive exam no better than fixed | E14 shows no gain | Report it honestly; the debrief and ghost-return recheck still work with the blueprint |

---

## 15. Limitations (state them on the Lab Report)
1. Training data is synthetic, from our own templates; the realistic set is hand-written by the team (R-blind reduces, but doesn't remove, author bias). No real learner used the system.
2. Probe, trap and knowledge-model parameters are **design assumptions**, not fitted; simulations test robustness to them, not their truth.
3. Code-identical twins can't be separated from code at all; the system relies on the learner's answers, which can be noisy.
4. Fixed C subset (no pointers, `scanf`, structs; strings only as char arrays) and a fixed bank of 34 problems; "DSA" means basic searching, sorting, array techniques, strings and recursion, not data structures such as linked lists, stacks or trees.
5. Confidence is calibrated on synthetic out-of-fold data; on real code it may be over-confident (see the failure audit).
6. Resolution is probabilistic evidence of a changed belief, not proof of learning.
7. Exam item exposures, Elo constants and selection weights are hand-set; adaptivity is validated only in simulation.

---

## 16. References
- ITSP dataset (IIT Kanpur CS-101 student C programs, buggy + corrected pairs): https://github.com/jyi/ITSP
- IntroClass benchmark (C, student submissions, BSD-3): https://github.com/ProgramRepair/IntroClass
- C-Pack of IPAs: a C90 program benchmark of introductory programming assignments (APR 2024): https://doi.org/10.1145/3643788.3648010
- Project CodeNet (IBM): https://github.com/IBM/Project_CodeNet
- Prutor: a system for tutoring CS1 and collecting student programs: https://arxiv.org/abs/1608.03828
- Ettles, Luxton-Reilly, Denny (2018). Common logic errors made by novice programmers. ACE '18: https://dl.acm.org/doi/10.1145/3160489.3160493
- Qian & Lehman (2017). Students' misconceptions and other difficulties in introductory programming: a literature review. ACM TOCE.
- Sorva (2012). Visual Program Simulation in Introductory Programming Education (doctoral thesis, Aalto University), appendix catalogue of novice misconceptions.
- Eedi — Mining Misconceptions in Mathematics (Kaggle, 2024): distractor-to-misconception mapping.
- Götschi, Sanders & Galpin (2003). Mental models of recursion. SIGCSE '03.
- Danielsiek, Paul & Vahrenhold (2012). Detecting and understanding students' misconceptions related to algorithms and data structures. SIGCSE '12.
- pycparser: https://github.com/eliben/pycparser · LightGBM `pred_contrib` (SHAP contributions): LightGBM `Booster.predict` docs.
- Competitor reference: https://github.com/itsjayrane/DaGOATS_maharashtra_round (README "Limitations" section).
