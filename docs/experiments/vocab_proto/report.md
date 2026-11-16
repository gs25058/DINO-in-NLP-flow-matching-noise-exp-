# 어휘 prototype DINO (mean pooling 고정)

Stage: complete

C0 = t13 재사용(재학습 없음). C1 사전학습 MLM head(학습), C2 같은 head 동결, C3 단어 임베딩 k-means 8192 + 무작위 transform. 사전 등록: OPTUNA_SPEC.md.

## Phase 0 정합성 (학습 0 step, STS-B dev)

| 항목 | probe | 측정 | 통과 |
|---|---|---|---|
| head_raw | 0.5938 | 0.5938 | 예 |
| embedding_raw | 0.5931 | 0.5931 | 예 |
| H_temp1 | 6.081 | 6.0806 | 예 |
| KL_temp1 | 7.582 | 7.5171 | 예 |

C1: K=30522, temp1 H/lnK 0.589, 문장 간 KL 7.517, head raw dev 0.5938.

C3: K=8192, temp1 H/lnK 0.987, 문장 간 KL 0.227, head raw dev 0.5126.
C2 학습 step 0 (anchor, warmup teacher temp 0.233, center=0): H_pt/lnK 0.034, batch_KL 1.798, teacher pair KL 8.049, teacher copy rate 0.625.
C1 학습 step 0 (anchor, warmup teacher temp 0.233, center=0): H_pt/lnK 0.034, batch_KL 1.798, teacher pair KL 8.049, teacher copy rate 0.625.
C3 학습 step 0 (anchor, warmup teacher temp 0.058, center=0): H_pt/lnK 0.047, batch_KL 8.697, teacher pair KL 24.083, teacher copy rate 0.013.

## Phase 1 탐색 (seed 42, 조건별 독립 TPE, 최대 40 attempted)

| 조건 | attempted | complete | pruned | failed | best trial | dev best | τs | τt/τs | 마지막 H/lnK |
|---|---|---|---|---|---|---|---|---|---|
| C1 | 40 | 20 | 20 | 0 | 16 | 78.45 | 1.290 | 0.535 | 0.034 |
| C2 | 40 | 18 | 22 | 0 | 9 | 78.31 | 1.355 | 0.335 | 0.016 |
| C3 | 40 | 20 | 20 | 0 | 11 | 73.05 | 0.129 | 0.558 | 0.034 |

선택(3시드 dev best 평균, test 이전 동결): C1 trial 29 (τs 1.086, τt/τs 0.184), C2 trial 9 (τs 1.355, τt/τs 0.335), C3 trial 11 (τs 0.129, τt/τs 0.558).

## 표 1. STS-B dev와 STS7 test (×100, 3시드 42/43/44)

| 조건 | dev best 평균 | dev best 시드별 | STS7 raw best 평균 | raw best 시드별 | raw last 평균 | center_pc2 best 평균 | center_pc2 last 평균 | center best 평균 |
|---|---|---|---|---|---|---|---|---|
| C0 t13 (random 8192) | 73.90 | 73.72 / 73.52 / 74.47 | 67.10 | 67.40 / 66.95 / 66.95 | 67.04 | 67.14 | 67.08 | 67.06 |
| C1 word head, trained | 77.26 | 77.90 / 76.67 / 77.19 | 69.00 | 69.07 / 69.02 / 68.90 | 66.74 | 68.69 | 66.64 | 68.74 |
| C2 word head, frozen | 77.82 | 78.31 / 77.78 / 77.39 | 69.15 | 67.70 / 70.48 / 69.26 | 63.24 | 68.84 | 63.16 | 68.94 |
| C3 k-means 8192, random transform | 72.09 | 73.05 / 70.95 / 72.27 | 64.17 | 64.23 / 63.54 / 64.75 | 62.14 | 63.55 | 62.79 | 63.67 |

C0 dev는 coviso 보고서 값이다. C0 STS7은 같은 평가 코드(TF32)로 다시 계산한 값이며, 기록값 67.10(best raw)과의 차이는 아래 재현 항목에 적는다. 후처리 통계는 학습 코퍼스 고정 10,000문장(seed 0)에서 구했고 평가 데이터는 쓰지 않았다.

## 표 2. target 정보량과 기하 (final 3시드 평균; step 0 / 2000 / 3999)

| 조건 | H_pt/lnK | batch_KL (student) | teacher pair KL | teacher copy rate | eff_rank |
|---|---|---|---|---|---|
| C0 t13 (random 8192) | 0.979 / 0.996 / 0.997 | 0.030 / 0.050 / 0.046 | — / — / — | — / — / — | 219.7 / 328.5 / 335.8 |
| C1 word head, trained | 0.017 / 0.012 / 0.006 | 2.135 / 18.019 / 19.540 | 21.137 / 104.334 / 110.625 | 0.599 / 0.180 / 0.183 | 219.7 / 210.5 / 213.1 |
| C2 word head, frozen | 0.058 / 0.025 / 0.011 | 1.927 / 14.011 / 16.334 | 5.900 / 52.478 / 57.356 | 0.599 / 0.175 / 0.204 | 219.7 / 179.1 / 185.7 |
| C3 k-means 8192, random transform | 0.019 / 0.109 / 0.055 | 10.009 / 9.795 / 15.753 | 68.260 / 21.271 / 37.273 | 0.009 / 0.002 / 0.006 | 219.7 / 194.1 / 185.9 |

batch_KL은 기존 지표(student 분포, 무작위 256쌍 대칭 KL)이고 teacher pair KL은 배치 안 모든 문장쌍의 teacher(centered, 해당 step teacher 온도) 대칭 KL이다. C0는 copy rate를 기록하지 않았다.

## 사전 등록 예측 판정

- **P1** C1 > C0 (STS7 raw, dev-best ckpt): **성립**; Δ시드별 +1.67, +2.08, +1.96, 평균 +1.90pp.
- **P2** C1 target 정보량 (max H/lnK ≤ 0.95 그리고 median batch_KL ≥ 0.47): **성립**; s42: max H/lnK 0.028, median batch_KL 17.549; s43: max H/lnK 0.032, median batch_KL 17.502; s44: max H/lnK 0.031, median batch_KL 17.630.
- **P3** C3 − C1: **MLM head의 학습된 지식이 핵심(C3가 0.003 넘게 낮음)**; Δ시드별 -4.84, -5.49, -4.15, 평균 -4.83pp. 시드 일관성은 판정 규칙에 없으므로 시드별 값을 함께 본다.
- **P4** C1 − C2 (적응 효과): **기각** — 고정된 의미 좌표계로 충분하다(적응 이득 없음); Δ시드별 +1.37, -1.45, -0.36.
- **P5** 어휘 복사 퇴화 (step≥3000 student top-10 copy rate, 높은 t ≥ 0.9×낮은 t 그리고 낮은 t ≥ 0.3): **판정불가**; s42: 표본 충분한 t 구간 1개; s43: 표본 충분한 t 구간 1개; s44: 표본 충분한 t 구간 1개.

미실행 단계는 추정하지 않는다. 3시드로 통계적 유의성을 주장하지 않는다.
