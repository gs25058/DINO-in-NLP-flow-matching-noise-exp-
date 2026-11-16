# Context-reading teacher (local→global)

Stage: stopped_by_user
Teacher: clean B inside a 5..7-sentence same-paragraph window, B tokens pooled.
Student: three noisy solo-B views (weights 1, 1, β); identical in all arms.
A: teacher solo B. B: real window. C: other-document context, exact per-side token lengths.

Design: full. B and C only; independent lockstep TPE over 20 axes (40 attempted trials each, seed 42, median pruning on current dev from step 2000). Objective and selection: late dev = mean of steps 3500/3750/3999. Primary test checkpoint: last.

| trial | B late dev | B best | C late dev | C best |
|---|---|---|---|---|
| 0 | 69.68 | 69.94 | 66.98 | 72.51 |
| 1 | 60.55 | 70.14 | 60.60 | 71.70 |
| 2 | 66.23 | 68.74 | 61.10 | 70.87 |
| 3 | 59.49 | 68.89 | 53.72 | 67.96 |
| 4 | 55.64 | 69.47 | 61.44 | 70.67 |
| 5 | 58.99 | 69.95 | 53.95 | 68.49 |
| 6 | 67.60 | 68.39 | 63.90 | 69.30 |
| 7 | 51.69 | 67.64 | 51.20 | 68.51 |
| 8 | 67.57 | 70.99 | 66.00 | 71.31 |
| 9 | 65.81 | 68.36 | 67.69 | 69.34 |
| 10 | 69.66 | 71.82 | pruned |  |
| 11 | 66.07 | 68.42 | 67.38 | 69.73 |
| 12 | 65.49 | 71.44 | 67.92 | 70.01 |
| 13 | 70.78 | 71.08 | 57.59 | 71.58 |
| 14 | 57.72 | 67.36 | 56.11 | 70.24 |

B search best: trial 13, late dev 70.78 (teacher/student temp 0.161/0.129); 0 of 15 pruned.

C search best: trial 12, late dev 67.92 (teacher/student temp 0.151/0.119); 1 of 15 pruned.


Missing results are unexecuted, not estimated. No statistical significance is claimed.
Error: train_search_B_t15_s42 failed (143); see /src/gs25058/noise_experiment/noise_experiment/flowdino/results/experiments/context_teacher_full_20261116b/commands/train_search_B_t15_s42.log
