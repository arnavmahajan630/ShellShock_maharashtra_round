# E18: is the item bank sound?

Generated 2026-10-04 by `python -m ml.eval.e18_bank`.

**Caveat.** 'Verified' means: re-running the code gives the stated answer, and re-running the believed program gives the stated belief answer. It does not mean a student holding that mistake would choose that option; no student has answered these items. Nobody has skimmed the DeepSeek-drafted items (06 §5). Counts are for the files on disk at the time of the run.

**Scope.** The plan asks for verify_quiz.py + build_code_items.py --check over every item. The quiz half ran in full. The code-item half ran (CHECK PASSED: 143 items).

## Quiz bank (`quiz_items.json`)

| Measure | Value |
|---|---|
| Items | 115 (mcq 42, next_state 17, predict_output 39, reasoning 17) |
| `verify_quiz` over every item, both backends | PASSED |
| Verified by the interpreter | 100.0% (all 115 run, 0 with a problem) |
| Also agreed by gcc | 104 of 115 = 90.4% |
| Interpreter only (C leaves it undefined, so gcc is not asked) | 11 |
| Believed programs run (behind the belief answers) | 128 |
| `verified` flags in the file match the fresh run | yes |
| Marked manual | 0 |
| Collision items (target 12) | 13 |
| Run time | 27.7 s |

### Items per mistake and type (05 §3.1 target: 2 mcq, 2 predict_output, 1 next_state, 1 reasoning)

| Mistake | mcq | predict_output | next_state | reasoning |
|---|---|---|---|---|
| M01 | 4 | 5 | 1 | 1 |
| M02 | 5 | 3 | 1 | 1 |
| M03 | 3 | 3 | 1 | 1 |
| M04 | 2 | 2 | 1 | 1 |
| M05 | 2 | 2 | 1 | 1 |
| M06 | 2 | 2 | 1 | 1 |
| M07 | 4 | 2 | 1 | 1 |
| M08 | 4 | 5 | 1 | 1 |
| M10 | 2 | 2 | 1 | 1 |
| D01 | 3 | 3 | 1 | 1 |
| D02 | 3 | 3 | 1 | 1 |
| D03 | 2 | 2 | 1 | 1 |
| D04 | 2 | 2 | 1 | 1 |
| D05 | 4 | 2 | 1 | 1 |
| D06 | 4 | 2 | 1 | 1 |
| D07 | 2 | 2 | 1 | 1 |
| D08 | 2 | 2 | 1 | 1 |
| target | 2 | 2 | 1 | 1 |

All targets met.
A collision item counts toward each mistake in its `classes`.

### Descriptive counts (no pass/fail)

| Count | Value |
|---|---|
| Position of the correct option (0 = first) | {"0": 28, "1": 19, "2": 32, "3": 19} over 98 choice items; chi-square p = 0.2004 |
| Correct option is the unique longest | 31 of 98 |
| Options per item | {"3": 39, "4": 59} |
| Difficulty (drafter's estimate, unpiloted) | {"1": 61, "2": 41, "3": 13} |
| Concept | {"array_tech": 6, "arrays": 10, "conditions": 12, "functions": 6, "loops": 20, "recursion": 20, "searching": 16, "sorting": 7, "strings": 6, "variables": 12} |
| Source | {"DeepSeek draft, checked by running": 96, "hand-written": 19} |

## Code bank (`code_items.json`)

| Measure | Value |
|---|---|
| `build_code_items --check` | PASSED |
| Check line | `CHECK PASSED: 143 items` |
| Items | 143 (complete_snippet 67, debug_line 42, fix_bug 34) |
| debug_line items with no bug (05 §3.2: about 20%) | 8/42 = 19.0% |
| Counts meet the 05 §3.2 targets | yes |
| gcc | on |

### Counts versus targets (05 §3.2)

| Kind | Count | Target | Versus target |
|---|---|---|---|
| complete_snippet | 67 over 34 problems (33 problems with 2; P13 with 1) | 1-2 per problem | met |
| fix_bug | 34 | 2 per class | met |
| debug_line | 34 planted, plus 8 with planted null | 2 per class | met |

### Planted items per mistake (target: 2 fix_bug and 2 debug_line)

| Mistake | fix_bug | debug_line |
|---|---|---|
| M01 | 2 | 2 |
| M02 | 2 | 2 |
| M03 | 2 | 2 |
| M04 | 2 | 2 |
| M05 | 2 | 2 |
| M06 | 2 | 2 |
| M07 | 2 | 2 |
| M08 | 2 | 2 |
| M10 | 2 | 2 |
| D01 | 2 | 2 |
| D02 | 2 | 2 |
| D03 | 2 | 2 |
| D04 | 2 | 2 |
| D05 | 2 | 2 |
| D06 | 2 | 2 |
| D07 | 2 | 2 |
| D08 | 2 | 2 |
| target | 2 | 2 |

All three 05 §3.2 targets met.
A problem with no `complete_snippet` is absent from the file, so only `--check` can see it. This run exited 0, so none were missing.

Checker output:

```
items complete_snippet=67 fix_bug=34 debug_line=42
fix_bug per class M01=2 M02=2 M03=2 M04=2 M05=2 M06=2 M07=2 M08=2 M10=2 D01=2 D02=2 D03=2 D04=2 D05=2 D06=2 D07=2 D08=2
debug_line per class M01=2 M02=2 M03=2 M04=2 M05=2 M06=2 M07=2 M08=2 M10=2 D01=2 D02=2 D03=2 D04=2 D05=2 D06=2 D07=2 D08=2
debug_line no-bug 8/42
complete_snippet per problem P01=2 P02=2 P03=2 P04=2 P05=2 P06=2 P07=2 P08=2 P09=2 P10=2 P11=2 P12=2 P13=1 P14=2 P16=2 P17=2 Q01=2 Q02=2 Q03=2 Q04=2 Q05=2 Q06=2 Q07=2 Q08=2 Q09=2 Q10=2 Q11=2 Q12=2 Q13=2 Q14=2 Q15=2 Q16=2 Q17=2 Q18=2
gcc cross-check on
CHECK PASSED: 143 items
```
