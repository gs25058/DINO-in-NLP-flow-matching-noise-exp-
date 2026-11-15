# Surrounding-context asymmetric distillation

Stage: failed
Teacher: clean B only. Student: two solo B views plus a weighted third view.
All arms: content-only B pooling; covariance on solo view1 once/step.
Native wiki1m scores are historical references only; comparisons use the same new Wikipedia centers.

| combination | A dev best | B dev best | C dev best | B−C (pp) |
|---|---|---|---|---|
| 0 | 70.85560008293064 | 70.18222611053353 | 70.1935246068126 | -0.011298496279067649 |
| 1 | 72.30525514536139 | 71.78428498196348 | 71.66181630727009 | 0.12246867469339406 |
| 2 | 71.1109894799958 | 70.29873352371314 | 71.40399290339195 | -1.105259379678813 |
| 3 | 71.24076035306555 | 70.1594405666087 | 69.83193820601072 | 0.32750236059797544 |
| 4 | 71.2116334108019 | 70.90471865714457 | 70.3507172472583 | 0.5540014098862684 |
| 5 | 70.71195246232631 | 69.39505097611686 | 69.35592216800369 | 0.039128808113176206 |
| 6 | 71.75361372029568 | 70.82421582577344 | 70.45250175732244 | 0.3717140684509985 |
| 7 | 71.40726908830605 | 71.07792193426121 | 71.70098479544167 | -0.6230628611804576 |
| 8 | 71.78605741721925 | 70.41401868433057 | 71.3056582452241 | -0.8916395608935233 |
| 9 | 69.65864097790372 | 71.20370522836475 | 70.2751756687535 | 0.9285295596112491 |
| 10 | 71.64075064467991 | 71.67453456671609 | 71.09641159592796 | 0.5781229707881295 |
| 11 | 71.30568689315866 | 70.67461631148674 | 70.66146815508482 | 0.013148156401925348 |

Context diagnostic JSON files are separate; attention visualization is not used as evidence.
Missing results are unexecuted, not estimated. All checkpoint selection uses solo-sentence dev STS.
Error: train_confirm_A_t8_s103 failed (1); see /src/gs25058/noise_experiment/noise_experiment/flowdino/results/experiments/context_distill_20261110/commands/train_confirm_A_t8_s103.log
