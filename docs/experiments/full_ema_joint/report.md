# FlowDINO: 완전한 clean EMA + mean/cov 분리 공동 튜닝

상태: **search**.
campaign: `/src/gs25058/noise_experiment/noise_experiment/flowdino/results/experiments/full_ema_joint_20261110`. 실행 소스 `207dee68160397e4b34f4e85bd714dd63577b4f5`, `exp/flowdino-full-clean-ema`.
DINO8192+iso768 유지, mean/cov 독립 가중치 및 기존19축을 탐색. teacher는 자체 EMA word lookup 사용.
완전한 EMA 조건 20 trial ×4000 step. pruning 없음. 재검증 seed101/102/103.
768차원 STS-B dev로 선택하고, 동결 후 768차원 STS7으로 판정한다. 256차원은 진단이다.

전체 명령·exit code: commands/와 status/. resolved config: configs/. 소스 diff와 SHA: source_manifest.json, source_diff.patch.
실행기: `scripts/run_bottleneck.py --full-clean-ema --campaign results/experiments/full_ema_joint_20261110 --gpus auto,auto --trials-per-arm 20`

탐색 공간: `{'loss.cov_iso_mean_lambda': [0.05, 8.0, True], 'loss.cov_iso_cov_lambda': [0.05, 8.0, True], 'train.lr': [2e-05, 0.0002, True], 'train.head_lr': [0.0001, 0.002, True], 'train.grad_clip': [1.0, 6.0, False], 'train.weight_decay': [0.01, 0.3, True], 'train.warmup_frac': [0.01, 0.2, False], 'train.momentum_end': [0.998, 0.9999, False], 'train.momentum_ramp_frac': [0.2, 1.0, False], 'loss.teacher_temp': [0.08, 0.18, False], 'loss.warmup_teacher_temp': [0.03, 0.08, False], 'loss.student_temp': [0.1, 0.25, False], 'loss.teacher_temp_warmup_frac': [0.02, 0.3, False], 'loss.center_momentum': [0.8, 0.99, False], 'loss.cov_ema_momentum': [0.75, 0.99, False], 'augment.t_lo': [0.0, 0.6, False], '_t_span': [0.05, 0.6, False], '_t_start_frac': [0.1, 1.0, False], 'augment.warmup_frac': [0.0, 0.15, False]}`.
trial0은 과거 joint trial13, trial1은 R18 설정. 모든 새 탐색은 teacher_input=ema.
과거67.097±0.260은 seed42/43/44의 joint13 참조다. 현재 baseline 표는 이전 R18 seed101/102/103 참조다.
교사 입력과 다른 하이퍼파라미터도 달라질 수 있어 순수한 EMA 수정의 인과효과라고 주장하지 않는다.


## 마지막 관측 진단 (선택에 사용하지 않음)

| run | step | 256 rank | 256 mean norm | 256 uniformity | view cosine | weighted iso mean/cov |
|---|---|---|---|---|---|---|
| bneckiso_bneck_iso_20261104_confirm_baseline_t0_s101 | 3999 | 58.24271443792668 | 0.08900696039199829 | -2.2194650173187256 | 0.7242178618907928 | 0.035260845184868286 / 0.04038382958562702 |
| bneckiso_bneck_iso_20261104_confirm_baseline_t0_s102 | 3999 | 54.808250485742654 | 0.10002864897251129 | -2.0675079822540283 | 0.7907576113939285 | 0.038014851839450546 / 0.04128703818428425 |
| bneckiso_bneck_iso_20261104_confirm_baseline_t0_s103 | 3999 | 55.38466244482019 | 0.08411945402622223 | -2.3781545162200928 | 0.8014882206916809 | 0.04020567702345093 / 0.04321064273103442 |
| full_ema_full_ema_joint_20261110_smoke_baseline_s42 | 59 | 97.86142877536967 | 0.3058982491493225 | -2.601166009902954 |  | 0.5768529530725361 / 0.049236543641375766 |
| full_ema_full_ema_joint_20261110_smoke_bottleneck_s42 | 59 | 97.84669879731388 | 0.30583685636520386 | -2.600468873977661 |  | 0.5781593854448412 / 0.04906302093080241 |

성분별 raw loss·backbone/MLP gradient는 components.jsonl과 gradient_diagnostics.jsonl에 보존한다.

## 동결 후 STS7 test

| group | seed | checkpoint | 768 STS7 ×100 | 256 STS7 ×100 (진단) |
|---|---|---|---|---|
| baseline | 101 | best | 65.4942 | 59.4499 |
| baseline | 101 | last | 65.8205 | 57.8691 |
| baseline | 102 | best | 65.4591 | 55.8487 |
| baseline | 102 | last | 65.4591 | 55.8487 |
| baseline | 103 | best | 66.4953 | 59.1656 |
| baseline | 103 | last | 66.4953 | 59.1656 |

baseline_best: n=3, mean 65.8162, sample SD 0.5883.

baseline_last: n=3, mean 65.9250, sample SD 0.5259.

개별 7-task 점수는 results/analysis/*.json에 모두 보존한다.

## 해석과 미실행 항목

미완료 단계와 점수는 추정해서 채우지 않는다. 완료 상태 전에는 최종 결론을 내리지 않는다.
새 완전EMA FlowDINO와 역사적 기준선의 비교다. 공동 튜닝 및 teacher 입력이 함께 달라져 특정 변경의 인과 효과를 분리하지 못한다.
256차원 bottleneck 지표만 개선되면 768차원 임베딩 개선으로 판정하지 않는다.
과거 R18 보고 점수 대신 현재 campaign의 대조군을 비교 기준으로 삼는다.
