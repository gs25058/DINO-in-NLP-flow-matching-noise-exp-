# KoLeo × lrsplit Optuna 탐색 (study: `tune_r5d_bert_lrsplit_c_koleo_lrsplit`)

**데이터 출처**
- 전체 30 trial 기록: `results/analysis/tune_r5d_bert_lrsplit_c_koleo_lrsplit_trials.csv`
- best trial 요약: `results/analysis/tune_koleo_lrsplit_best.json`
- search space 정의: `scripts/tune.py` `SEARCH_SPACES["koleo_lrsplit"]`
- base config: `configs/r5d_bert_lrsplit_c.yaml` (BERT, R5 스케줄+head_lr 분리+LN/bias wd 제외+grad_clip 이미 적용된 위에 4개 값만 Optuna가 덮어씀)
- 예산: trial당 750 step(base 실제 길이 1500의 절반), seed=42 고정, n_trials=30

**주의(예산 불일치)**: 이 문서의 모든 값은 **750-step 예산** 기준이다. 기존 수동 grid 최고(`r6_bert_koleo_lrsplit_c`, koleo_λ=0.2 + lr=2e-4)의 0.666은 **1500-step 전체 학습** 기준이라 직접 비교가 아니다. best trial을 `configs/r6_bert_koleo_lrsplit_optuna_best_t27.yaml`로 승격하며 `max_steps`를 1500으로 되돌렸으니, grid 최고를 실제로 넘는지는 그 run을 돌려서 확인해야 한다 — 아직 미검증.

---

## 결과 요약

| 순위 | trial | sts_b_dev(750step) | koleo_λ | lr | head_lr | grad_clip |
|---|---|---|---|---|---|---|
| 1 | 27 | **0.6822** | 0.423 | 1.39e-4 | 7.66e-4 | 5.21 |
| 2 | 26 | 0.6778 | 0.413 | 3.07e-4 | 7.59e-4 | 5.33 |
| 3 | 25 | 0.6761 | 0.166 | 2.27e-4 | 7.70e-4 | 5.35 |
| 4 | 1 | 0.6727 | 0.300 | 4.14e-4 | 8.94e-4 | 2.32 |
| 5 | 23 | 0.6709 | 0.283 | 3.83e-4 | 7.64e-4 | 5.14 |
| … | | | | | | |
| 30 | 12 | **0.0395**(붕괴) | 0.484 | 5.00e-4 | 9.82e-4 | 5.99 |

30 trial 중앙값 0.6474, trial 12를 제외한 평균 0.6478 — trial 12 하나만 명백한 이상치(collapse)이고 나머지는 0.61~0.68 사이에 완만하게 분포.

---

## 발견 1. 단순 상관계수는 trial 12(붕괴) 하나 때문에 부호가 뒤집혀 있었다

전체 30 trial로 계산하면 4개 파라미터 모두 `sts_b_dev`와 **음의 상관**(-0.21~-0.29)으로 나와서 "값을 키우면 나빠진다"는 그릇된 결론을 낼 뻔했다. trial 12(0.0395로 유일하게 붕괴)를 제외하고 다시 계산하면 부호가 완전히 뒤집힌다:

| 파라미터 | 상관계수(30개 전체) | 상관계수(trial 12 제외) |
|---|---|---|
| koleo_lambda | -0.250 | **+0.592** |
| lr | -0.290 | +0.005 (사실상 무관) |
| head_lr | -0.248 | **+0.426** |
| grad_clip | -0.212 | **+0.479** |

**결론: koleo_lambda·head_lr·grad_clip 세 값 모두 (기존에 고정해뒀던 범위보다) 높을수록 좋다. lr 자체의 절대값은 이 구간 안에서는 거의 무관하다.** 이전 수동 grid는 이 세 파라미터 중 `koleo_lambda`만(그것도 0.05~0.2로 좁게) 탐색했고 `head_lr`/`grad_clip`은 아예 손대지 않았었다 — 이번 탐색으로 처음 이 두 축의 유효 범위를 확인한 것.

---

## 발견 2. 붕괴는 특정 축이 아니라 "네 축이 동시에 극단"일 때만 일어난다

trial 12 vs 정상 range를 비교하면, 개별 축을 극단으로 밀어붙인 다른 trial들은 전혀 붕괴하지 않았다:

| trial | koleo_λ | lr | head_lr | grad_clip | sts_b_dev |
|---|---|---|---|---|---|
| 11 (koleo 최댓값) | **0.490** | 0.031e-2 | 0.041e-2 | 5.97 | 0.6567 (정상) |
| 21 (lr 최댓값) | 0.110 | **0.539e-2** | 0.053e-2 | 1.69 | 0.6153 (정상) |
| 27 (최고 기록) | 0.423 | 0.139e-2 | **0.077e-2** | 5.21 | 0.6822 (최고) |
| **12 (붕괴)** | 0.484 | **0.500e-2** | **0.098e-2** | **5.99** | **0.0395** |

trial 12만 koleo_λ(2위), lr(3위), head_lr(**1위**), grad_clip(**1위**)이 동시에 상위권이다 — 다른 어떤 trial도 4개 중 3개 이상을 동시에 극단으로 밀어붙이지 않았다. **개별 축은 각자 안전 마진이 넓지만, lr을 탐색 범위 상단(Table 1 원값 5.4e-4에 근접) 가까이 두면서 동시에 head_lr·grad_clip·koleo_λ까지 전부 상단으로 밀면 그 조합에서만 무너진다.** 최고 기록(trial 27)이 lr을 오히려 낮게(1.39e-4, lrsplit_b의 1e-4에 더 가까움) 가져간 것도 이와 일치한다 — 다른 세 축을 밀어붙일 여유를 lr을 눌러서 만든 셈.

**실무 규칙**: head_lr·grad_clip·koleo_λ를 공격적으로 올리려면 backbone lr은 낮게(1e-4~1.4e-4대) 유지할 것. lr까지 같이 올리려면(2e-4 이상) 나머지 세 축 중 최소 하나는 낮춰야 한다(trial 1: lr=4.14e-4면서 grad_clip만 낮음(2.32)이 대표 사례, 0.6727로 여전히 상위권).

---

## 발견 3. head_lr·grad_clip은 이번에 처음 탐색됐고, 둘 다 기존 고정값보다 위가 유리했다

- **head_lr**: 이전엔 항상 Table 1 값(5.4e-4)에 고정. 상위 7개 trial 전부 7.6e-4~8.9e-4에 몰려 있다 — 기존 고정값의 **1.4~1.65배**. 9.8e-4(trial 12)까지 가면 위 발견 2의 조합과 맞물려 위험.
- **grad_clip**: 이전엔 항상 3.0 고정. 상위권 대부분(27/26/25/23/29/28)이 5.1~5.5로 수렴 — 즉 **더 느슨한 clipping**이 유리했다. 예외는 trial 1(2.32)로, lr을 높게 가져간 대신 clip을 세게 건 조합.

---

## 다음에 확인할 것
- `configs/r6_bert_koleo_lrsplit_optuna_best_t27.yaml`(1500-step, 시드 42/43)을 돌려서 grid 최고(0.666, 1500-step)를 실제로 넘는지 검증 — 750-step 결과(0.6822)가 그대로 유지될지, 후반부에 발견 2의 조합 위험(head_lr/grad_clip이 높은 채 1500 step까지 감)이 뒤늦게 나타날지가 핵심 질문.
- 이번 탐색은 lr 상한을 Table 1 원값(5.4e-4)으로 뒀다 - 발견 2가 맞다면 lr을 더 낮은 상한(예: 2e-4)으로 좁히고 head_lr/grad_clip/koleo_λ 세 축만 더 넓게 재탐색하는 쪽이 다음 Optuna 라운드로 유망해 보인다.
