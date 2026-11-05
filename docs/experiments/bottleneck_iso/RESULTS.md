# Bottleneck cosine + isotropy 최종 결과

완료: 2026-11-04T23:40:37.980861+09:00. 실행 소스 `ee13213dc05e9ac3f82ae970329ccf0e2f655a83` / `exp/bottleneck-iso-paired-optuna`.

주 평가인 **768d STS7 best의3시드 평균**은 기준선65.816, cosine+iso256 64.366으로 **−1.450점**이다. 세 시드 모두 하락했다.
iso만 제거한 조건과의 차이는 +0.162점이지만3시드 중2개에서 하락해 iso의 STS 개선 가설은 **판정불가**다.

## 실행과 선택

20+20 Optuna trial×4000step, 조건별 top3×새 seed101/102/103 재검증18회, 최종 설정의 두 성분 제거 비교6회: 총64회 full training 완료.
보정1회·양쪽60-step smoke·best/last STS7 24평가까지 총91개 실행 기록의 exit code가 모두0이다. source snapshot hash 검증과 초기값·데이터·batch 순서 audit도 통과했다.
최초 조건은 실제 R18 기준선 `configs/coviso_relax/base.yaml`. 기존768d iso 가중치5.874093568592943와 noise 등은 유지했다. 4000-step 스케줄을 압축하거나 weight resume하지 않았다.
공통 탐색은 backbone/head LR 기준×[0.5,2] log, 최종 teacher T [0.10,0.16]. 보조 가중치 범위·상태 보존 보정은 `OPTUNA_SPEC.md`와 `validation.json`에 있다.
탐색 seed42 dev 최고 trial은 baseline11(74.782), bottleneck1(73.424)이었지만, 새3시드 dev 평균으로 최종 선택한 것은 baseline0 / bottleneck15다. test 이전에 selection.json을 동결했다.

## 최종3시드 평균 ± sample SD (Spearman×100)

| 조건 | dev best | dev last | 768d STS7 best | 768d STS7 last | 256d STS7 best (진단) |
|---|---|---|---|---|---|
| 기존 FlowDINO | 73.623 ± 0.713 | 73.456 ± 1.002 | 65.816 ± 0.588 | 65.925 ± 0.526 | 58.155 ± 2.002 |
| cosine + iso256 | 72.687 ± 0.446 | 72.364 ± 0.255 | 64.366 ± 0.621 | 64.322 ± 0.987 | 60.092 ± 1.015 |
| cosine만 유지 (iso256=0) | 73.220 ± 1.117 | 73.140 ± 1.211 | 64.204 ± 0.978 | 64.820 ± 0.070 | 59.485 ± 0.164 |
| iso256만 유지 (cosine=0) | 71.889 ± 0.344 | 67.779 ± 0.788 | 62.386 ± 0.279 | 61.571 ± 0.280 | 59.266 ± 1.171 |

`iso_off`와 `aux_off`는 실험군 최종 설정에서 해당 가중치만0으로 변경했다. 개별 ablation에 맞춰 재튜닝하거나 test로 후보를 다시 선택하지 않았다.

## 시드별 결과

| 조건 | seed | dev best | dev last | best loop step | 768d STS7 best | 768d STS7 last |
|---|---|---|---|---|---|---|
| 기존 FlowDINO | 101 | 72.799 | 72.299 | 2500 | 65.494 | 65.821 |
| 기존 FlowDINO | 102 | 74.043 | 74.043 | 3999 | 65.459 | 65.459 |
| 기존 FlowDINO | 103 | 74.026 | 74.026 | 3999 | 66.495 | 66.495 |
| cosine + iso256 | 101 | 72.418 | 72.142 | 2500 | 65.080 | 65.462 |
| cosine + iso256 | 102 | 72.440 | 72.307 | 3500 | 63.948 | 63.759 |
| cosine + iso256 | 103 | 73.202 | 72.642 | 2750 | 64.070 | 63.746 |
| cosine만 유지 (iso256=0) | 101 | 72.164 | 71.948 | 1250 | 63.075 | 64.901 |
| cosine만 유지 (iso256=0) | 102 | 73.105 | 73.105 | 3999 | 64.785 | 64.785 |
| cosine만 유지 (iso256=0) | 103 | 74.390 | 74.369 | 3500 | 64.751 | 64.774 |
| iso256만 유지 (cosine=0) | 101 | 72.018 | 68.433 | 1750 | 62.629 | 61.277 |
| iso256만 유지 (cosine=0) | 102 | 72.150 | 66.905 | 1250 | 62.082 | 61.601 |
| iso256만 유지 (cosine=0) | 103 | 71.499 | 67.999 | 2500 | 62.447 | 61.835 |

## Best checkpoint 태스크별3시드 평균

| task | 기준선 | cosine+iso256 | iso_off | cosine_off |
|---|---|---|---|---|
| STS12 | 58.399 | 56.569 | 56.011 | 52.843 |
| STS13 | 72.132 | 70.952 | 71.374 | 69.431 |
| STS14 | 58.203 | 56.453 | 56.763 | 55.498 |
| STS15 | 70.133 | 67.815 | 68.382 | 65.782 |
| STS16 | 70.687 | 68.878 | 68.186 | 66.394 |
| STSBenchmark | 67.026 | 65.460 | 65.558 | 62.956 |
| SICK-R | 64.133 | 64.435 | 63.153 | 63.799 |

## 사전 등록 가설 판정

- H1: 동일 튜닝 예산 개선: **기각**. paired Δ×100=[-0.414, -1.511, -2.425], 평균 -1.450점.
- H2: 같은 설정에서 iso256 추가 효과: **판정불가**. paired Δ×100=[2.005, -0.837, -0.681], 평균 +0.162점.
- H3: 같은 설정에서 cosine 추가 효과: **성립**. paired Δ×100=[2.451, 1.866, 1.623], 평균 +1.980점.

성립은 평균Δ≥0.003이고 최소2/3 양수, 기각은 평균≤0이고 최소2/3 비양수, 그 외는 판정불가라는 기존 규칙을 적용했다. 3시드로 통계적 유의성은 주장하지 않는다.
cosine의 긍정 결과는 iso256과 실험군 hyperparameter를 고정한 조건부 효과다. 기존 FlowDINO 전반보다 낫거나 모든 설정에서 효과가 있다고 일반화하지 않는다.
iso는 dev best에서 평균−0.533점, STS7 last에서−0.498점이다. 사전 등록된 best 결과의 +0.162점만으로 일관된 개선을 주장할 수 없다.

## 분포와 gradient 진단

| 조건 | 768 rank | 768 alignment | 768 uniformity | 256 rank | 256 alignment | 256 uniformity | 256 mean norm |
|---|---|---|---|---|---|---|---|
| 기존 FlowDINO | 326.8026 | 0.2710 | -3.3041 | 56.1452 | 0.1701 | -2.2217 | 0.0911 |
| cosine + iso256 | 329.5169 | 0.2664 | -3.4712 | 57.8399 | 0.1086 | -1.6772 | 0.5748 |
| cosine만 유지 (iso256=0) | 317.9353 | 0.2643 | -3.4287 | 54.5779 | 0.0479 | -0.7778 | 0.8348 |
| iso256만 유지 (cosine=0) | 275.0292 | 0.3319 | -3.3089 | 48.4508 | 0.2580 | -2.5486 | 0.1740 |

위 표는 **last checkpoint**의3시드 평균이다. iso_off 대비 iso 추가로256d mean norm은0.835→0.575, uniformity는−0.778→−1.677로 바뀌었다. 분포의 방향 집중은 줄었지만 centered rank는54.58→57.84의 작은 변화이며 STS 개선은 입증되지 않았다.
기준선 대비256d STS7 best는58.155→60.092로 개선됐으나768d는65.816→64.366으로 하락했다. bottleneck의 개선이 주 표현으로 전달되었다는 가설을 지지하지 않는다.
최종 실험군의 마지막 step에서 weighted cosine backbone gradient norm은0.109~0.133, CE는0.031~0.039로 cosine이 CE의약3.38~3.61배다. 초기 보정 비율은 학습 내내 유지되지 않았다. 성분별 gradient는 `components.jsonl` / `gradient_diagnostics.jsonl`에 보존했다.
centered rank·uniformity는 진단이며 특정 목표값이나 외부 붕괴 임계치를 강제하지 않았다.

## 정확한 실행 설정과 원본 기록

최종 실험군: backbone LR9.140175995078903e−5, head LR0.00030548108587646737, 최종 teacher T0.14551971443452516, cosine λ0.12204655086730903, iso256 λ0.03395064496716301.

- 기존 FlowDINO seed101 resolved config: `/src/gs25058/noise_experiment/noise_experiment/flowdino/results/experiments/bneck_iso_20261104/configs/confirm_baseline_t0_s101.yaml`. seed102/103은 같은 폴더에 있다.
- cosine + iso256 seed101 resolved config: `/src/gs25058/noise_experiment/noise_experiment/flowdino/results/experiments/bneck_iso_20261104/configs/confirm_bottleneck_t15_s101.yaml`. seed102/103은 같은 폴더에 있다.
- cosine만 유지 (iso256=0) seed101 resolved config: `/src/gs25058/noise_experiment/noise_experiment/flowdino/results/experiments/bneck_iso_20261104/configs/iso_off_s101.yaml`. seed102/103은 같은 폴더에 있다.
- iso256만 유지 (cosine=0) seed101 resolved config: `/src/gs25058/noise_experiment/noise_experiment/flowdino/results/experiments/bneck_iso_20261104/configs/aux_off_s101.yaml`. seed102/103은 같은 폴더에 있다.

실제 시작 명령:

```bash
scripts/run_bottleneck.py --with-bneck-iso --campaign results/experiments/bneck_iso_20261104 --gpus auto,auto --trials-per-arm 20
```

전체40 trial·상위3 설정 재검증은 `docs/experiments/bottleneck_iso/report.md`와 campaign report, 실제 guarded 명령·옵션/exit는 `commands/`·`status/`, 데이터/소스/초기값은 `source_manifest.json`·각 run의 provenance, 비교 가능성은 `comparability_audit.json`에 있다.
모든 개별7-task 점수와256d 진단은 `results/analysis/*.json`, 위 표의 원자료는 campaign `final_summary.json`에 보존했다.
과거 R18의2시드 dev≈74.65/STS7≈66.09와 현재 대조군73.623/65.816은 다르다. GPU·시드 집합 차이를 포함하므로 과거 점수를 맞추는 튜닝은 하지 않았고 현재 대조군과 비교했다.
추가 미실행 학습·평가는 없다. iso mean/cov 분리 효과나 모든 hyperparameter에서의 효과는 이번 설계가 검증하지 않는다. 코드 변경·추가 탐색·test 후 재선택을 하지 않았다.
