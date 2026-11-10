# Surrounding-context asymmetric distillation

Stage: paired_search
Teacher: clean B only. Student: two solo B views plus a weighted third view.
All arms: content-only B pooling; covariance on solo view1 once/step.
Native wiki1m scores are historical references only; comparisons use the same new Wikipedia centers.

| combination | A dev best | B dev best | C dev best | B−C (pp) |
|---|---|---|---|---|

Context diagnostic JSON files are separate; attention visualization is not used as evidence.
Missing results are unexecuted, not estimated. All checkpoint selection uses solo-sentence dev STS.
