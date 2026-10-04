# E14 Exam — simulated learners, not real students

Simulated learners follow our assumptions; this shows the selection logic works as designed, not that it is optimal for real learners.

Population: simulated learners (not real students).
n = 2000 simulated learners per seed, 20 seeds (seed 1414). Cells are the mean and the 95% t interval across seeds.
Items to first true finding: learners with no hidden active class are left out; never found counts as 10.
Diagnoser noise: E1 out-of-fold confusion on TRAIN (calibrated, masked argmax | dataset label).

| Policy | Profile-recovery F1 | Brier of P(A) | STABLE classes rechecked | Items to first true finding |
|---|---|---|---|---|
| adaptive (simulated) | 0.495 [0.493, 0.497] | 0.137 [0.136, 0.138] | 0.938 [0.935, 0.941] | 3.717 [3.676, 3.758] |
| fixed (simulated) | 0.464 [0.460, 0.467] | 0.131 [0.130, 0.131] | 0.782 [0.777, 0.786] | 4.295 [4.258, 4.331] |
| random (simulated) | 0.426 [0.423, 0.428] | 0.132 [0.131, 0.132] | 0.749 [0.742, 0.755] | 5.611 [5.581, 5.641] |

Profile-recovery F1 after each item (simulated):

| Item | adaptive | fixed | random |
|---|---|---|---|
| 1 | 0.158 [0.155, 0.161] | 0.160 [0.157, 0.163] | 0.088 [0.086, 0.089] |
| 2 | 0.263 [0.260, 0.265] | 0.219 [0.217, 0.222] | 0.143 [0.141, 0.144] |
| 3 | 0.324 [0.320, 0.327] | 0.205 [0.203, 0.208] | 0.189 [0.187, 0.192] |
| 4 | 0.390 [0.386, 0.393] | 0.261 [0.259, 0.264] | 0.231 [0.228, 0.233] |
| 5 | 0.417 [0.413, 0.420] | 0.343 [0.340, 0.346] | 0.267 [0.264, 0.270] |
| 6 | 0.405 [0.402, 0.409] | 0.361 [0.358, 0.364] | 0.301 [0.298, 0.303] |
| 7 | 0.421 [0.419, 0.424] | 0.397 [0.394, 0.401] | 0.333 [0.330, 0.336] |
| 8 | 0.443 [0.441, 0.445] | 0.440 [0.437, 0.444] | 0.364 [0.362, 0.366] |
| 9 | 0.468 [0.465, 0.471] | 0.464 [0.461, 0.468] | 0.394 [0.392, 0.396] |
| 10 | 0.495 [0.493, 0.497] | 0.464 [0.460, 0.467] | 0.426 [0.423, 0.428] |

Final F1, simulated: adaptive 0.495, fixed 0.464, relative gain 0.067. Claim (adaptive > fixed by >= 10% relative) met: False.
Move versus the stand-in run (simulated, same seed and n): adaptive +0.013, fixed +0.010, random +0.008. Relative gain 0.062 -> 0.067 (delta +0.005). 10% bar on the stand-in run: False; on this E1 run: False.
