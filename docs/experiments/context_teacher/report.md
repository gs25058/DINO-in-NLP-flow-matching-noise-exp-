# Context-reading teacher (local→global)

Stage: complete
Teacher: clean B inside a 5..7-sentence same-paragraph window, B tokens pooled.
Student: three noisy solo-B views (weights 1, 1, β); identical in all arms.
A: teacher solo B. B: real window. C: other-document context, exact per-side token lengths.

Design: tuned. Independent equal-budget Optuna per arm (12 trials, seed 42, lockstep), top-3 confirmed on seeds 101..103, selection by fresh-seed dev mean; C also run at B's parameters.

| trial | A dev best | B dev best | C dev best |
|---|---|---|---|
| 0 | 71.56 | 69.94 | 72.51 |
| 1 | 71.39 | 71.77 | 72.82 |
| 2 | 72.46 | 71.39 | 72.21 |
| 3 | 71.15 | 69.58 | 70.07 |
| 4 | 71.37 | 68.09 | 70.38 |
| 5 | 73.16 | 72.05 | 72.23 |
| 6 | 71.66 | 71.37 | 72.12 |
| 7 | 72.87 | 70.98 | 73.16 |
| 8 | 67.40 | 70.57 | 70.33 |
| 9 | 72.25 | 71.82 | 72.84 |
| 10 | 72.13 | 70.39 | 70.83 |
| 11 | 72.29 | 72.41 | 71.51 |

A search best: trial 5, 73.16 (backbone_lr=4.093e-05, head_lr=0.001098, beta=0.3976).

B search best: trial 11, 72.41 (backbone_lr=5.555e-05, head_lr=0.0009281, beta=0.5911).

C search best: trial 7, 73.16 (backbone_lr=8.099e-05, head_lr=0.001061, beta=1.093).

| arm | seed | dev best | dev last | best step | wall s | teacher nonpad tokens |
|---|---|---|---|---|---|---|
| A | 101 | 71.65 | 71.11 | 1750 | 1054 | 3636355 |
| A | 102 | 71.77 | 71.77 | 3999 | 1009 | 3636498 |
| A | 103 | 72.22 | 72.16 | 2500 | 1061 | 3637062 |
| B | 101 | 71.44 | 70.64 | 500 | 1214 | 21142994 |
| B | 102 | 71.38 | 69.24 | 250 | 2135 | 21146745 |
| B | 103 | 71.48 | 71.02 | 1000 | 1232 | 21150795 |
| C | 101 | 73.49 | 66.57 | 250 | 1211 | 21142994 |
| C | 102 | 72.72 | 65.34 | 500 | 1223 | 21146745 |
| C | 103 | 72.96 | 66.02 | 250 | 1248 | 21150795 |
| C_matched_B | 101 | 72.11 | 70.46 | 250 | 1266 | 21142994 |
| C_matched_B | 102 | 73.22 | 67.83 | 250 | 1270 | 21146745 |
| C_matched_B | 103 | 71.34 | 68.76 | 500 | 1269 | 21150795 |

A dev best: 71.88 ± 0.30 (sample SD).

B dev best: 71.44 ± 0.05 (sample SD).

C dev best: 73.06 ± 0.40 (sample SD).

C_matched_B dev best: 72.22 ± 0.94 (sample SD).

| arm | checkpoint | STS7 per seed 101/102/103 | mean |
|---|---|---|---|
| A | best | 62.72 / 64.27 / 64.32 | 63.77 |
| A | last | 64.09 / 64.27 / 64.79 | 64.38 |
| B | best | 61.50 / 60.85 / 62.33 | 61.56 |
| B | last | 62.63 / 61.06 / 62.82 | 62.17 |
| C | best | 64.07 / 61.37 / 62.60 | 62.68 |
| C | last | 57.44 / 55.54 / 57.88 | 56.95 |
| C_matched_B | best | 61.90 / 63.45 / 61.69 | 62.35 |
| C_matched_B | last | 60.37 / 59.21 / 58.93 | 59.51 |

B vs C (STS7, best): 기각, paired Δ=-1.12 pp, per seed -2.56, -0.52, -0.27.

B vs C_matched_B (STS7, best): 기각, paired Δ=-0.79 pp, per seed -0.40, -2.60, +0.64.

B vs A (STS7, best): 기각, paired Δ=-2.21 pp, per seed -1.22, -3.42, -1.99.

Missing results are unexecuted, not estimated. No statistical significance is claimed.
