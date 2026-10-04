# E10 Resolution — simulated learners, not real students

Simulated learners embody our assumptions about learner behaviour. The table shows how the policies behave if learners act like this, not that real learners do.

Population: simulated learners (not real students).
n = 2000 simulated learners per type per seed, 20 seeds (seed 1010). Cells are the mean and the 95% t interval across seeds.
A dash means the conditioning set is empty for that type (nobody is active, or nobody is inactive).

| Policy | Simulated learner type | False-resolve | False-not-yet | Items to decision |
|---|---|---|---|---|
| naive | truly_fixed (simulated) | — | 0.000 [0.000, 0.000] | 1.117 [1.112, 1.122] |
| 4-check | truly_fixed (simulated) | — | 0.019 [0.017, 0.020] | 2.241 [2.236, 2.246] |
| ours | truly_fixed (simulated) | — | 0.182 [0.176, 0.188] | 3.312 [3.306, 3.318] |
| naive | pattern_copier (simulated) | 0.968 [0.967, 0.970] | — | 1.189 [1.184, 1.194] |
| 4-check | pattern_copier (simulated) | 0.390 [0.386, 0.395] | — | 3.536 [3.530, 3.542] |
| ours | pattern_copier (simulated) | 0.106 [0.103, 0.109] | — | 3.961 [3.957, 3.964] |
| naive | lucky_guesser (simulated) | 0.812 [0.808, 0.816] | — | 2.354 [2.341, 2.366] |
| 4-check | lucky_guesser (simulated) | 0.371 [0.366, 0.376] | — | 2.769 [2.760, 2.779] |
| ours | lucky_guesser (simulated) | 0.094 [0.090, 0.097] | — | 3.959 [3.954, 3.964] |
| naive | forgetful (simulated) | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.120 [1.116, 1.125] |
| 4-check | forgetful (simulated) | 0.981 [0.979, 0.983] | 0.019 [0.016, 0.022] | 2.245 [2.241, 2.249] |
| ours | forgetful (simulated) | 0.477 [0.468, 0.486] | 0.181 [0.177, 0.185] | 3.312 [3.307, 3.317] |
| naive | slow (simulated) | 0.719 [0.714, 0.723] | — | 2.293 [2.278, 2.308] |
| 4-check | slow (simulated) | 0.267 [0.263, 0.271] | — | 2.969 [2.958, 2.980] |
| ours | slow (simulated) | 0.272 [0.254, 0.290] | 0.274 [0.270, 0.279] | 4.223 [4.217, 4.229] |

Pattern-copier false-resolve, simulated: ours 0.106, naive 0.968, ratio 0.109. Claim (ours ≤ ½ × naive) met: True.
