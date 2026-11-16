# Context-reading teacher (local→global)

Stage: failed
Teacher: clean B inside a 5..7-sentence same-paragraph window, B tokens pooled.
Student: three noisy solo-B views (weights 1, 1, β); identical in all arms.
A: teacher solo B. B: real window. C: other-document context, exact per-side token lengths.

Design: full. B and C only; independent lockstep TPE over 20 axes (40 attempted trials each, seed 42, median pruning on current dev from step 2000). Objective and selection: late dev = mean of steps 3500/3750/3999. Primary test checkpoint: last.

| trial | B late dev | B best | C late dev | C best |
|---|---|---|---|---|
| 0 | 69.68 | 69.94 | 66.98 | 72.51 |
| 1 | 60.55 | 70.14 | 60.60 | 71.70 |
| 2 | 66.23 | 68.74 | 61.10 | 70.87 |

B search best: trial 0, late dev 69.68 (teacher/student temp 0.135/0.151); 0 of 3 pruned.

C search best: trial 0, late dev 66.98 (teacher/student temp 0.135/0.151); 0 of 3 pruned.


Missing results are unexecuted, not estimated. No statistical significance is claimed.
Error: student B tokens/noise differ across arms
