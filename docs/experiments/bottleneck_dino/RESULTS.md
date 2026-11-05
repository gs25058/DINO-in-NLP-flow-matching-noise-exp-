# 실험2: 직접256d DINO+iso 최종 결과

완료: 2026-11-05T18:55:00.074768+09:00. 실행 소스 `8471057ed928aaac151f71ea0fe7b108ac83cb96`, 브랜치 `exp/bottleneck-dino256-optuna`.

주 평가인 **768d STS7은65.816→64.049로1.767점 하락**했고 세 시드 모두 하락했다. 사전 등록 개선 가설은 **기각**이다.
256d bottleneck의 STS7은58.155→60.355로2.200점 개선됐다. 병목 표현 개선이 주 평가 임베딩 개선으로 이어지지 않았다.

## 실행한 조건과 튜닝

8192 expansion을 생성하지 않고, MLP→L2의256d를 직접 DINO logits로 사용했다. teacher256d EMA centering·temperature sharpening·student temperature를 유지하고, 동일256d에 기존 EMA mean/cov iso를 적용했다. cosine·768d iso·별도 predictor는 없다.
bert-base-uncased, mean+L2 추론, batch32, noisy student3, clean EMA teacher, max128, 기존 flow noise·마스킹·데이터 순서·optimizer/EMA·4000-step 스케줄·FP32/TF32를 유지했다.
기준선은 사용자 지시대로 **재탐색하지 않았다**. 기존 선택 config와 seed101/102/103 best/last의6개 STS7 결과를 해시 검증 후 참조했다.
새 구조만 Optuna20 trial×4000step. backbone/head LR 각각 기준×[0.5,2] log, student T [0.0755,0.302] log, teacher/student T ratio [0.1,0.99] log, iso mean/cov 공통λ [0.001,10] log를 탐색했다. teacher warmup initial T는 min(기존 initial T, final T)이다.
TPE seed20261104, startup5, multivariate, pruning 없음. 탐색 seed42, 모든 run은 독립 사전학습 초기화다. 완주 top3(trial15/6/12)를 새 seed101/102/103에서 각각 재학습9회 후 dev 평균으로 trial15를 선택했다.
총 새 full training29회 +60-step smoke2회 +새 test6회 =37개 실행 모두 exit0. 기존 기준선의4000-step 학습·재탐색·STS7 재평가는 없었다. 기준선60-step smoke는 코드 보존 검증에만 썼다.

## 3시드 평균 ± sample SD, Spearman×100

| 조건 | 768 dev best | 768 dev last | 768 STS7 best | 768 STS7 last | 256 dev (선택된 best) | 256 STS7 best | 256 STS7 last |
|---|---|---|---|---|---|---|---|
| 이미 튜닝된 FlowDINO (고정 참조) | 73.623 ± 0.713 | 73.456 ± 1.002 | 65.816 ± 0.588 | 65.925 ± 0.526 | 64.331 ± 1.781 | 58.155 ± 2.002 | 57.628 ± 1.672 |
| 직접256d DINO CE+iso | 70.962 ± 0.568 | 70.939 ± 0.608 | 64.049 ± 0.409 | 64.062 ± 0.387 | 68.108 ± 0.170 | 60.355 ± 0.204 | 60.366 ± 0.189 |

best는 **768d dev로 고른 동일 checkpoint**다. 256d test를 보고 checkpoint를 다시 고르지 않았다. 256d dev 마지막 평균은 기준선62.934, 새 구조68.052다.

## 시드별 점수

| 조건 | seed | 768 dev best | 768 dev last | best loop step | 768 STS7 best | 768 STS7 last | 256 STS7 best | 256 STS7 last |
|---|---|---|---|---|---|---|---|---|
| 이미 튜닝된 FlowDINO (고정 참조) | 101 | 72.799 | 72.299 | 2500 | 65.494 | 65.821 | 59.450 | 57.869 |
| 이미 튜닝된 FlowDINO (고정 참조) | 102 | 74.043 | 74.043 | 3999 | 65.459 | 65.459 | 55.849 | 55.849 |
| 이미 튜닝된 FlowDINO (고정 참조) | 103 | 74.026 | 74.026 | 3999 | 66.495 | 66.495 | 59.166 | 59.166 |
| 직접256d DINO CE+iso | 101 | 70.316 | 70.247 | 3750 | 63.581 | 63.619 | 60.174 | 60.208 |
| 직접256d DINO CE+iso | 102 | 71.384 | 71.384 | 3999 | 64.230 | 64.230 | 60.576 | 60.576 |
| 직접256d DINO CE+iso | 103 | 71.186 | 71.186 | 3999 | 64.336 | 64.336 | 60.314 | 60.314 |

새 구조−기준선의 paired768d STS7 차이는 seed101 **−1.914**,102 **−1.229**,103 **−2.159**점이다. 평균−1.767, paired sample SD0.482점이다.
사전 등록상 평균Δ≤0이고 최소2/3 비양수면 기각이다.3시드로 통계적 유의성이나 구조의 보편적 열등함을 주장하지 않는다.

## 태스크별 best checkpoint의3시드 평균

| task | 기준선768 | 직접DINO768 | 기준선256 | 직접DINO256 |
|---|---|---|---|---|
| STS12 | 58.399 | 55.863 | 58.065 | 52.876 |
| STS13 | 72.132 | 69.798 | 58.663 | 65.398 |
| STS14 | 58.203 | 56.515 | 48.312 | 52.194 |
| STS15 | 70.133 | 69.952 | 63.179 | 66.392 |
| STS16 | 70.687 | 68.439 | 63.268 | 63.843 |
| STSBenchmark | 67.026 | 63.704 | 55.690 | 59.657 |
| SICK-R | 64.133 | 64.073 | 59.908 | 62.123 |

## 재검증과 최종 설정

| trial | 새3시드 768d dev best 평균 | sample SD |
|---|---|---|
| 15 | 70.962 | 0.568 |
| 6 | 68.880 | 1.146 |
| 12 | 69.766 | 0.645 |

최종 trial15: backbone LR5.147714796903899e−5, head LR0.00031962576827391267, student T0.10791228321287379, final teacher T0.10177588383777081, iso mean/cov λ5.711075084680992이다.
새 구조 seed42 탐색 최고는70.238(trial15). 기존 FlowDINO 같은 seed42 dev best73.690보다3.452점 낮다. 단일 seed 점수로 선택하지 않고 새3시드 결과로 확정했다.
기존 hyperparameter를 그대로 넣은 새 구조 trial0의67.502보다 튜닝 후70.238로2.736점 올랐다. 튜닝의 효과는 있었지만 이번 범위와 예산에서는 기준선에 미치지 못했다.
재사용 config: `configs/bottleneck_dino/selected.yaml`. 기존 기준선: `configs/coviso_relax/base.yaml`. 두 config는 실제 resolved config와 비교 검증했다.
실제 새 선택 run config: `/src/gs25058/noise_experiment/noise_experiment/flowdino/results/experiments/bneck_dino256_20261105/configs/confirm_bottleneck_t15_s101.yaml`. 기준선의 실제 config: `/src/gs25058/noise_experiment/noise_experiment/flowdino/results/experiments/bneck_iso_20261104/configs/confirm_baseline_t0_s101.yaml`.

## 분포·gradient·구현 검증

| 조건 | last768 rank | last768 alignment | last768 uniformity | last256 rank | last256 alignment | last256 uniformity | last256 mean norm |
|---|---|---|---|---|---|---|---|
| 이미 튜닝된 FlowDINO (고정 참조) | 326.8026 | 0.2710 | -3.3041 | 56.1452 | 0.1701 | -2.2217 | 0.0911 |
| 직접256d DINO CE+iso | 245.7110 | 0.2506 | -3.0324 | 103.6242 | 0.2178 | -2.9368 | 0.0831 |

256d rank는56.15→103.62로 증가했지만768d rank는326.80→245.71로 감소했다. 진단상 병목의 분포가 더 퍼졌으나 주 표현의 STS 개선은 확인되지 않았다. Uniformity 목표값이나 다른 서버의 붕괴 임계치를 강제하지 않았다.
최종 새 run의 마지막 step에서 backbone gradient norm은 CE0.437~0.532, weighted iso mean0.237~0.351, weighted iso cov0.073~0.083이었다. MLP에서도 각 항의 gradient를 측정했다. 손실 절대값만으로 지배성을 판단하지 않았다.
선택 checkpoint 검사: expansion parameter 없음, center shape256, covariance256×256, cosine0, 추가 auxiliary iso0. 기존 iso 모듈 자체를256d로 이동했으므로 추가 iso 모듈은 없다.
기준선 GPU smoke의 공통 loss 성분·gradient norm·평가 최대 차이는0.0이었다. 공유 backbone/MLP 초기값, 첫 noisy view, 데이터 순서 hash와 고정된 학습·평가 규칙을 검증했다.
DINO CE 값은 logit class 수가8192→256으로 바뀌어 직접 크기를 비교할 수 없다. 작은 CE 값만으로 새 모델이 더 좋다고 해석하지 않는다.

## 앞선 실험1과의 관계

실험1은 기존 DINO8192+iso768에 cos256+iso256을 추가한 조건이며, 선택된3시드768d STS7은64.366±0.621,256d는60.092±1.015였다. 이번 실험2는 각각64.049±0.409,60.355±0.204다.
두 실험 모두 주 평가에서 고정 기준선65.816±0.588을 넘지 못했다. 실험1 대비 실험2의 작은 평균 차이를 유의한 우열로 주장하지 않는다. 두 조건의 온도·가중치·기타 선택값은 독립 튜닝됐다.
기존 baseline test는 이미 알려져 있던 참조다. 새 test는 selection.json 동결 뒤 계산했고 test를 보며 설정·범위·후보를 변경하지 않았다. 탐색 범위와 이력도 기준선과 달라 expansion 하나의 인과 효과를 분리하거나 expansion이 필수라고 단정할 수 없다.

## 원본 기록

실제 시작 명령:

```bash
scripts/run_bottleneck.py --direct-bottleneck-dino --campaign results/experiments/bneck_dino256_20261105 --reference-baseline results/experiments/bneck_iso_20261104 --gpus auto,auto --trials-per-arm 20
```

전체 trial·재검증은 campaign report.md/SQLite와 `confirmations.json`, 선택은 `selection.json`, 실제 명령·exit는 `commands/`·`status/`, 개별 test는 `results/analysis/*.json`에 있다.
정리 원자료는 campaign `final_summary.json`, checkpoint 검사는 `checkpoint_structure_audit.json`, 기준선 GPU 비교는 `baseline_gpu_equivalence_audit.json`, 초기값·노이즈·데이터 비교는 `comparability_audit.json`, 참조 출처는 `fixed_baseline.json`에 보존했다.
실행하지 않은 결과를 추정하지 않았다. 계획한 새29회 학습과6개 test에 미완료 항목은 없다. 결과 확인 후 추가 탐색·GPU 학습을 시작하지 않았다.
