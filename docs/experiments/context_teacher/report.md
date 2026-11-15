# Context-reading teacher (local→global)

Stage: stopped_by_user
Teacher: clean B inside a 5..7-sentence same-paragraph window, B tokens pooled.
Student: three noisy solo-B views (weights 1, 1, β); identical in all arms.
A: teacher solo B. B: real window. C: other-document context, exact per-side token lengths.
Fixed hyperparameters: context_distill combination 1. Selection: none (fixed config).


Missing results are unexecuted, not estimated. No statistical significance is claimed.
Error: train_run_B_s101 failed (143); see /src/gs25058/noise_experiment/noise_experiment/flowdino/results/experiments/context_teacher_20261115/commands/train_run_B_s101.log
