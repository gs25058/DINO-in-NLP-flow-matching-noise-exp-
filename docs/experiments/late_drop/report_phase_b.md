# 후반 하락: 원인 요인 스케줄 튜닝 (Phase B)

Stage: complete

C1/C2 각각 vocab_proto 공간 + 스케줄 4축(sharpening 완화 u·시작·길이, teacher 동결 시점), late dev 목적함수, seed 42, 조건당 40 attempted. 확인과 test는 seed 101/102/103, C0 t13도 같은 seed로 재학습.

재현 검증(anchor가 이전 run과 모든 기록 일치): C2_t1 = results/experiments/late_drop_20261117/iv_C2_t9_teacher_freeze, C1_t1 = results/experiments/late_drop_20261117/iv_C1_t16_teacher_freeze, C2_t0 = results/experiments/late_drop_20261117/iv_C2_t9_sharpen_off, C1_t0 = results/experiments/late_drop_20261117/iv_C1_t16_sharpen_off, C2_t2 = results/experiments/vocab_proto_20261116/search_C2_t3, C1_t2 = results/experiments/vocab_proto_20261116/search_C1_t29.

| 조건 | attempted | complete | pruned | failed | best trial | late dev | u | start | len | freeze |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 40 | 21 | 19 | 0 | 30 | 78.28 | 0.78 | 0.36 | 0.01 | 0.92 |
| C2 | 40 | 23 | 17 | 0 | 35 | 78.64 | 0.08 | 0.48 | 0.05 | 0.19 |

선택(test 전 동결): C1 trial 30 (u 0.78, start 0.36, len 0.01, freeze 0.92), C2 trial 35 (u 0.08, start 0.48, len 0.05, freeze 0.19).

## STS7 test (×100, seed 101/102/103)

| 그룹 | late dev 평균 | raw last 평균 | raw last 시드별 | raw best 평균 | best − last | center_pc2 last |
|---|---|---|---|---|---|---|
| C0 | 73.99 | 67.10 | 66.94 / 66.57 / 67.80 | 67.14 | 0.04 | 67.07 |
| C1 | 78.33 | 69.11 | 68.77 / 69.26 / 69.30 | 69.36 | 0.25 | 69.01 |
| C2 | 77.78 | 69.31 | 69.28 / 69.37 / 69.30 | 69.50 | 0.18 | 69.04 |
| C1_off | 77.11 | 67.44 | 66.92 / 67.68 / 67.71 | 69.58 | 2.14 | 67.39 |
| C2_off | 75.68 | 65.67 | 65.24 / 65.93 / 65.83 | 69.38 | 3.71 | 65.54 |

## 사전 등록 판정

- **R1 C1 vs C0 (STS7 raw last):** **성립**; Δ +1.83, +2.70, +1.50.
- **R1 C2 vs C0 (STS7 raw last):** **성립**; Δ +2.34, +2.81, +1.50.
- **R2 C1 후반 하락 해소 (best − last ≤ 0.5pp, 시드별 ≤ 1.0pp):** **성립**; +0.00, +0.38, +0.38.
- **R2 C2 후반 하락 해소 (best − last ≤ 0.5pp, 시드별 ≤ 1.0pp):** **성립**; +0.11, +0.21, +0.22.
- **R3 C1 스케줄 자체 효과 (같은 설정, 스케줄 끔 대비, STS7 raw last):** **성립**; Δ +1.84, +1.58, +1.59.
- **R3 C2 스케줄 자체 효과 (같은 설정, 스케줄 끔 대비, STS7 raw last):** **성립**; Δ +4.03, +3.44, +3.47.

미실행 단계는 추정하지 않는다. 3시드로 통계적 유의성을 주장하지 않는다.
