# R7 cov_iso (분포 통계량 정규화) vs KoLeo (쌍별 반발) — BERT (P-I1~I5 판정)

**데이터 출처**
- 백본: `bert-base-uncased` (2026-09-03 합의: SimCSE 등과 공정 비교를 위해 모든 실험 BERT로 진행 — 이 문서는 그 합의 이후 버전. 이전에 돌렸던 ModernBERT 본 실험/3000-step 연장판은 폐기했다).
- 대조군(재사용, 재실행 안 함): `results/logs_archive/r5d_bert_combined[_seed43]`(정규화 없음), `results/logs_archive/r6_bert_koleo_a_s{42,43}`(쌍별 반발, koleo_λ=0.05).
- 본 실험: `configs/r7_coviso_bert_lam{0.5,1,2}.yaml` (`r5d_bert.yaml` 기반, koleo 명시적 off), 시드 42/43, 1500 step.
- λ 후보 근거: `results/analysis/coviso/` 캘리브레이션(ModernBERT/r5d_combined 기반 200-step dry-run) + BERT swap 확인 run에서 λ=2.0까지 안정성 확인 — BERT 전용 재캘리브레이션은 하지 않음.
- 확인 run: `configs/r7_bert_coviso_swap_t27.yaml` (BERT 최고점 라인 `r6_bert_koleo_lrsplit_optuna_best_t27`에서 KoLeo→cov_iso λ=2.0 교체, 시드 42).
- 그림: `results/analysis/coviso/overlay.png`
- 구현/캘리브레이션 스케일 조정 이력(`*D` 제거)은 `src/loss.py::CovIsoPenalty` 및 이전 세션 기록 참고.

---

## 결과 표

| run | 피크 STS | 최종 STS(1499) | 최종 alignment | 최종 uniformity | 최종 eff_rank |
|---|---|---|---|---|---|
| r5d_bert(정규화 없음) s42 | 0.6191 | **0.6119** | 0.2121 | -1.8988 | 220.93 |
| r5d_bert(정규화 없음) s43 | 0.6146 | **0.6106** | *(로깅 결측)* | *(로깅 결측)* | 221.62 |
| r6_bert_koleo_a(쌍별) s42 | 0.6311 | 0.6291 | 0.4008 | -3.1567 | 331.81 |
| r6_bert_koleo_a(쌍별) s43 | 0.6259 | 0.6180 | 0.4017 | -3.1216 | 331.67 |
| r7 coviso λ=0.5 s42 | 0.6150 | 0.6102 | 0.3427 | -2.9231 | 229.59 |
| r7 coviso λ=0.5 s43 | 0.6104 | 0.6066 | 0.3451 | -2.9099 | 229.73 |
| r7 coviso λ=1.0 s42 | 0.6145 | 0.6116 | 0.3466 | -2.9472 | 235.06 |
| r7 coviso λ=1.0 s43 | 0.6106 | 0.6074 | 0.3455 | -2.9310 | 235.22 |
| r7 coviso λ=2.0 s42 | 0.6191 | **0.6119** | 0.3486 | -2.9649 | 243.35 |
| r7 coviso λ=2.0 s43 | 0.6134 | 0.6083 | 0.3518 | -2.9629 | 243.38 |
| (참고) r7_bert_coviso_swap_t27(튜닝된 lr/head_lr/grad_clip 라인, λ=2.0) s42 | - | 0.6307 | 0.3621 | -3.0666 | 258.56 |
| (참고, 동일 라인 KoLeo판) r6_bert_koleo_lrsplit_optuna_best_t27 s42/s43 | - | 0.6816 / 0.6652 | - | - | 433 |

---

## 게이트 판정 — **발동**

사전 조건: "전 λ에서 raw STS가 r5d 대비 하락하거나 학습 불안정이면 멈추고 보고."

**전 λ, 전 시드에서 최종 STS가 r5d_bert 대비 하락하거나 정확히 동률이다** (λ=2.0 s42만 0.6119로 정확히 동률, 나머지 5개 조합은 전부 하락). 학습 자체는 불안정하지 않았다(loss 발산·H_pt 급락 없음, `L_iso`는 매 λ에서 정상적으로 감소) — 붕괴가 아니라 **"안정적으로 학습되지만 STS에는 사실상 아무 영향이 없다"**는 판정이다. 게이트에 따라 P-I2/P-I3(후처리 rescore 비교)를 위한 추가 계산은 진행하지 않고 여기서 멈춘다.

---

## 예측별 판정

**P-I1 (raw STS가 r5d 초과): 기각.** 위 게이트 판정과 동일 근거. λ를 0.5→2.0으로 올려도 STS가 거의 안 움직인다(0.607~0.612 범위에 전부 수렴) — ModernBERT 캘리브레이션에서 보였던 "λ가 클수록 유리"한 경향이 BERT에서는 재현되지 않았다.

**P-I2 (rescore 기준 raw-후처리 격차 축소): 판정불가.** 게이트로 인해 rescore 계산을 진행하지 않음.

**P-I3 (최종 raw가 [후처리 보정값, KoLeo] 사이): 판정불가.** 같은 이유. 원 예측의 참고 구간(0.70~0.74 등)은 ModernBERT 스케일이라 BERT(raw ~0.61)에 그대로 대입할 수도 없다.

**P-I4 (uniformity/eff_rank 개선이 KoLeo보다 작거나 같음): 성립.** eff_rank 개선폭: cov_iso(λ=2, 최선) +22(221→243) < KoLeo +111(221→332). uniformity 개선폭: cov_iso -1.06(-1.90→-2.96) < KoLeo -1.24(-1.90→-3.16→여기선 -3.16 근사치, s42 기준). 두 지표 모두 cov_iso가 KoLeo보다 확실히 작게 개선된다.

**P-I5 (후반 alignment 악화 기울기가 KoLeo보다 완만): 성립, 그것도 뚜렷하게.** step 750→1499 구간 alignment 변화: KoLeo(r6_bert_koleo_a_s42) 0.394→0.401 (**계속 악화**, +0.007), cov_iso(λ=2, s42) 0.360→0.349 (**오히려 개선**, -0.011). 쌍별 반발이 없는 cov_iso는 후반부에 alignment가 계속 갉아먹히는 KoLeo의 병리를 보이지 않는다 — METHOD.md §9의 "쌍별 반발이 alignment 침식의 원인"이라는 가설과 정확히 일치하는 방향.

---

## 종합 해석

**"쌍별 반발 vs 분포 통계량" 구분 자체는 실제로 의미 있는 기전 차이를 만든다** — P-I4/P-I5가 이를 보여준다: cov_iso는 KoLeo처럼 후반부 alignment를 갉아먹지 않으면서도 uniformity/eff_rank를 어느 정도(KoLeo의 1/5~1/3 수준) 개선한다. 이는 METHOD.md §9의 정체성 구분이 공허한 철학적 구분이 아니라 실제 학습 동역학에 드러나는 차이임을 뒷받침한다.

**그러나 BERT에서는 그 개선폭이 STS로 이어지기엔 너무 작다.** λ를 4배(0.5→2.0) 늘려도 STS는 사실상 그대로다 — ModernBERT 캘리브레이션에서 관찰된 "λ가 클수록 유리" 패턴이 BERT의 학습 동역학(이 프로젝트에서 이미 여러 번 확인된, BERT가 ModernBERT보다 훨씬 불안정하고 별도 안정화 스택이 필요했던 배경)에서는 재현되지 않는다. 튜닝된 lr/head_lr/grad_clip 라인(swap 확인 run)에서도 cov_iso(0.6307)는 같은 자리의 KoLeo(0.68/0.67)에 크게 못 미친다.

**권고**: 이 λ 범위·설계의 cov_iso는 BERT에서 KoLeo를 대체할 성능을 보여주지 못했다. 다음으로 시도해볼 만한 방향(범위 밖, 결정 필요): (1) BERT 전용 캘리브레이션으로 훨씬 큰 λ(4, 8 등) 탐색, (2) `cov_iso_start_step`을 더 일찍/늦게 당기기, (3) EMA momentum을 낮춰 배치 통계에 더 민감하게 반응하도록 조정. 어느 쪽이든 추가 GPU 실험이 필요해 게이트에 따라 사용자 확인 없이는 진행하지 않는다.
