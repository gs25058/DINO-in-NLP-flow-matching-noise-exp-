# Weak student view experiment

Stage: stopped_by_user
Clean EMA; CE8192 + iso768; one weak + two original strong student views.
Control = full-clean-EMA trial1 hyperparameters; all runs start from pretrained BERT.
Primary selection: 768d STS-B dev. Test only after selection freeze.

| trial | state | dev best | dev last |
|---|---|---|---|
| 0 | COMPLETE | 0.6993708680210493 | 0.6931461487130139 |
| 1 | COMPLETE | 0.6568516551448786 | 0.6411393537939708 |
| 2 | COMPLETE | 0.6424419255560443 | 0.6341842776364431 |
| 3 | COMPLETE | 0.690098834105452 | 0.6876283358737499 |
| 4 | RUNNING |  |  |
| 5 | RUNNING |  |  |

Error: train_search_bottleneck_t5_s42 failed (143); see /src/gs25058/noise_experiment/noise_experiment/flowdino/results/experiments/weak_student_view_20261110/commands/train_search_bottleneck_t5_s42.log
