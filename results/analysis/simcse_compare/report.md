# SimCSE 비교 — 공식 체크포인트 재측정 + 원본 재현 학습

**데이터 출처**
- 공식 체크포인트 재측정: `scripts/compare_simcse.py` → `results/analysis/simcse_compare/metrics.json`
- 원본 재현 학습: `scripts/train_simcse_baseline.py`, run `simcse_repro_1epoch_s42` (1 epoch=15,401 step 중 **7,700에서 사용자 지시로 정지**)
- 예산 맞춤 비교 run: `simcse_bert_bs32_st1500_s{42,43}`, `simcse_bert_bs64_st750_s{42,43}` — **원본 설정이 아니므로 아래 §5 참고**
- 곡선 원본: `results/analysis/simcse_compare/curves_eval.csv`, `curves_train.csv`
- 우리 비교 대상: `r8_bert_coviso_sched_optuna_t50_s{42,43}` (cov_iso + 튜닝된 스케줄)

---

## 1. 평가 파이프라인 검증 — 논문을 재현한다

공식 `unsup-simcse-bert-base-uncased`를 **우리 지표 코드로** 재서 논문 Table 5와 대조:

| 태스크 | 논문 | 우리 측정(cls) |
|---|---|---|
| STS12 | 68.40 | 68.40 |
| STS13 | 82.41 | 82.41 |
| STS14 | 74.38 | 74.38 |
| STS15 | 80.91 | 80.91 |
| STS16 | 78.56 | 78.56 |
| STS-B | 76.85 | 76.84 |
| SICK-R | 72.23 | 72.55 |
| **avg7** | **76.25** | **76.29** |

소수점 둘째 자리까지 일치한다(SICK-R만 데이터셋 버전 차이로 보이는 0.3). **`sts_suite_spearman`/`sts_b_dev_metrics`가 논문 프로토콜과 동일하게 동작함이 확인됐다** — 이후 우리 수치도 같은 잣대로 신뢰할 수 있다.

부수 확인: `analysis_v2.md`가 "재계산 아님"이라고 명시하며 쓰던 SimCSE 참고치(alignment≈0.18, uniformity≈−2.6)도 직접 측정값(0.200 / −2.72)과 부합했다.

## 2. 원본 재현 학습 — 절반만으로 논문에 2.5점 차

공식 `run_unsup_example.sh`를 그대로 따랐다: batch 64, lr 3e-5, **max_seq_length 32**, temp 0.05, dropout 0.1, warmup 없음 + **linear** decay, **fp16**, CLS+MLP(`mlp_only_train`), 셔플 1-epoch 순회, 125 step마다 평가 후 **best 채택**(`load_best_model_at_end`).

| | STS-B dev | avg7(test) |
|---|---|---|
| 공식 체크포인트(1 epoch 완주) | 0.8245 | 76.29 |
| **우리 재현 (절반, best@step 3625)** | **0.8049** | **73.78** |

절반 학습 + 문장 985,723개(원본 1M)로 2.5점 차. **구현이 원본에 충실하다고 판단하기에 충분하다.**

## 3. 학습 곡선 — 논문에 없는 것

논문에는 step별 STS 곡선이 없다(Figure 2의 alignment–uniformity 평면 궤적이 유일하며 렌더링 이미지다). 공식 저장소에도 로그가 없다(`figure/`에는 `demo.gif`, `model.png`뿐).

```
step:      0     125     500    1000    1500    2000    3000   3625★    5000    7625
STS:   0.317   0.555   0.639   0.674   0.730   0.756   0.790   0.805    0.767   0.798
align: 0.174   0.277   0.284   0.278   0.260   0.238   0.231   0.224    0.209   0.221
unif: -1.019  -2.300  -2.446  -2.487  -2.487  -2.391  -2.413  -2.406   -2.353  -2.432
rank:    142     241     266     271     266     248     238     235      238     225
```

**두 가지가 드러난다.**

1. **uniformity는 step 500이면 사실상 끝난다** (−1.02 → −2.45, 이후 −2.35~−2.49에서 평평). 학습 후반의 성능 향상은 균일성에서 오는 게 아니다.
2. **alignment는 U자다.** 0.174 → step 250~500에서 0.284로 최악 → 이후 0.21까지 회복. 즉 **후반 향상은 거의 전적으로 alignment 회복에서 온다.**

"SimCSE는 alignment를 지키면서 uniformity만 개선한다"는 통념은 **최종 상태에만 맞고 과정에는 틀리다** — 초반에는 우리 방법만큼 alignment가 무너진다.

## 4. 우리 방법과의 비교

| 시점 | SimCSE(원본 설정) | 우리 r8 |
|---|---|---|
| step 1500 (동일 예산) | **0.7303** (cls) | 0.7120 (mean, 2시드) |
| 최고 (step 3625) | **0.8049** (cls) / **0.8158** (mean) | — |
| avg7 | **73.78** | 63.75 |
| 최종 alignment | 0.207~0.224 | 0.283 |
| 최종 uniformity | −2.38~−2.41 | −3.44 |

**동일 step에서도 SimCSE가 앞서고, 학습을 늘리면 격차가 크게 벌어진다.**

우리 r8은 SimCSE보다 **훨씬 균일한데(−3.44 vs −2.4) 정렬은 나쁘다(0.283 vs 0.21).** §3의 곡선과 겹쳐 보면 해석이 분명해진다 — 균일성은 이미 필요 이상으로 확보했고, SimCSE가 후반에 얻는 alignment 회복 구간을 우리는 겪지 못한 채 1500 step에서 끝난다.

## 5. 방법론 교훈 — 이 비교에서 결론을 세 번 갈아엎었다

기록으로 남긴다. 같은 실수를 반복하지 않기 위함이다.

1. **1차 오류(pooling 불일치)**: SimCSE는 CLS(공식), 우리는 mean으로 재놓고 나란히 놓아 "우리가 0.108 앞선다"고 결론냈다. 같은 pooling으로 맞추자 사라진 격차였다.
2. **2차 오류(구현 불충실)**: 자체 구현 SimCSE가 `max_seq_length` 128(공식 32), cosine 스케줄(공식 linear), fp16 미사용, 복원추출 샘플링(공식 epoch 순회)이었다. 원본 설정으로 고치자 **같은 step에서 0.604 → 0.7303**으로 뛰었다. "동률"이라는 2차 결론도 틀렸다.
3. **지표 혼동**: 논문의 76.85는 **7-task test 평균**, 우리가 보던 0.84는 **STS-B dev 단일**이다. 서로 다른 값을 비교하면 안 된다.

**규칙**: 외부 기준선과 비교할 때는 (a) pooling·지표·split을 명시적으로 맞추고, (b) 자체 구현이면 원본 저장소의 실행 스크립트와 항목별로 대조하고, (c) 재현 검증(공개 체크포인트 수치 맞추기)을 먼저 통과시킨 뒤 비교에 들어간다.

## 6. 다음에 할 것

- **우리 config를 같은 길이(7,700 step)로 확장**: 우리도 alignment U자 회복 구간에 들어가는가, 아니면 cov_iso의 균일성 압력이 회복을 막는가. 지금 가장 정보량이 큰 실험.
- 우리 방법의 uniformity 목표를 낮추는 실험(cov_iso λ 축소): −3.44는 과한 것으로 보인다.
- 미검증으로 남은 cov_iso config 2개(t30, t39).

---

## 부록: 로그·데이터 위치

| 항목 | 경로 |
|---|---|
| 재현 학습 로그 | `results/logs/simcse_repro_1epoch_s42/train.log` (836줄, EVAL 62개, step 0~7700) |
| 재현 stdout/stderr | `results/logs/simcse_repro_1epoch_s42/launch.log` |
| 예산 맞춤 run | `results/logs/simcse_bert_bs{32_st1500,64_st750}_s{42,43}/train.log` |
| TensorBoard | `results/tensorboard/simcse_*/` (`train/*`, `eval/*` 태그 — 우리 run과 동일) |
| 체크포인트 | `checkpoints/simcse_repro_1epoch_s42/best.pt` (step 3625), 나머지 run은 `last.pt` |
| **평가 곡선 CSV** | `results/analysis/simcse_compare/curves_eval.csv` (98행: run, step, sts_b_dev, alignment, uniformity, eff_rank, max_sv_ratio) |
| **학습 곡선 CSV** | `results/analysis/simcse_compare/curves_train.csv` (1,527행: loss, infonce_acc, pos_cos, neg_cos, lr, grad_norm 등 + 우리 run의 DINO 지표) |
| 공식 체크포인트 재측정 | `results/analysis/simcse_compare/metrics.json` |

두 CSV에는 우리 `r8_bert_coviso_sched_optuna_t50_s{42,43}`도 같은 스키마로 포함되어 있어 바로 겹쳐 분석할 수 있다.
