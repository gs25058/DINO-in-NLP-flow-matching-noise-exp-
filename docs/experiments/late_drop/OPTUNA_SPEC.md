# 후반 하락 원인 규명과 원인 요인 스케줄 튜닝 — 사전 등록

2026-11-17 실행 전 등록. 브랜치 `exp/vocab-prototype-dino`. 대상은 vocab_proto의 단어 head 조건이다.
vocab_proto의 test 결과는 이미 보았음을 공개한다. 이 단계는 STS-B dev만 사용하며 test는 Phase B 동결 뒤에만 본다.

## 동기 (기존 로그에서 확인한 사실)

- C1/C2의 dev best는 step 250–1250이고 last checkpoint STS7은 C0보다 낮다.
- 후반에 alignment가 나빠지고 uniformity는 계속 커진다. teacher 문장 간 KL도 계속 증가한다.
- backbone gradient에서 cov 항은 DINO의 1/1000 이하이고 mean 항은 1/30–1/600이다. grad clip은 거의 항상 걸린다.
- 같은 조건 안에서도 후반 dev를 유지하는 설정(C2 trial 3·28)과 떨어지는 설정(C2 trial 9, C1 trial 16·33)이 공존한다.
  상관만으로는 sharpening, centering, EMA teacher, 노이즈를 분리할 수 없다.

## Phase A: 단일 요인 개입 (seed 고정, 18 run)

- 대상: C2 trial 9(seed 42, 43), C1 trial 16(seed 42). vocab_proto에서 하락이 가장 큰 설정이며, 원래 run을 그대로 재사용한다.
- step 1000부터 한 요인만 바꾼다(`src/intervention.py`, RNG를 소비하지 않음).
  - `sharpen_off`: teacher 최종 온도 = student 온도
  - `center_freeze`: center momentum = 1
  - `iso_off`: mean/cov λ = 0
  - `teacher_freeze`: teacher EMA momentum = 1
  - `noise`: 노이즈 범위를 C0(t13)의 [0.466, 0.902]로 (curriculum 종료 상태)
  - `lr_zero`: 모든 lr = 0 (조기 종료 기준선, 판정 대상 아님)
- 정합성: step 1000 이전 `components.jsonl`과 `evaluations.jsonl`이 원래 run과 정확히 같아야 하며, 다르면 campaign을 멈춘다.
- 지표: late dev = step 3500/3750/3999 dev 평균. 회복률 = (개입 late − 원래 late) / (원래 best − 원래 late).
- 판정(요인별): 3 run 모두 회복률 ≥ 0.5이면 **원인**, 평균 ≥ 0.25이면 **부분 원인**, 평균 ≤ −0.25이면 **악화**, 그 외 **원인 아님**.

## Phase B: 원인 요인의 스케줄 튜닝 (Phase A 결과를 보고 확정, test 전 커밋)

- Phase A에서 원인 또는 부분 원인으로 판정된 요인만 스케줄 축으로 만든다(예: teacher 온도 후반 완화의 시작 step·끝값·형태,
  center momentum 후반값, teacher momentum 후반 동결 시점). 원인 요인이 없으면 Phase B는 하지 않고 그대로 보고한다.
- 기존 탐색 공간 위에 스케줄 축을 더해, 조건별 독립 TPE로 탐색한다. 목적함수는 late dev이며 dev best는 쓰지 않는다.
- 확인: top 3 × 새 seed 101/102/103, 3시드 late dev 평균으로 동결한 뒤 STS7 test를 본다. 주 지표는 **last checkpoint**다.
- 비교 대상: C0 t13 last(67.04)와 vocab_proto 선택 설정의 last.
- 정확한 축·범위·예산은 Phase A 판정 직후 이 문서에 추가하고 커밋한 다음 실행한다.

## Phase A 결과 (2026-11-17, campaign `results/experiments/late_drop_20261117`)

18 run 모두 step 1000 이전 기록이 원래 run과 정확히 같았다. 회복률은 C2 t9 s42 / C2 t9 s43 / C1 t16 s42 순서다.

| 개입 | 회복률 | 판정 |
|---|---|---|
| sharpen_off | +0.99 / +0.90 / +0.68 | 원인 |
| teacher_freeze | +0.91 / +0.84 / +0.65 | 원인 |
| noise | +0.05 / +0.21 / +0.14 | 원인 아님 |
| center_freeze | −0.04 / −0.05 / +0.03 | 원인 아님 |
| iso_off | −0.01 / −0.14 / −0.06 | 원인 아님 |
| (lr_zero, 기준선) | +0.66 / +0.49 / +0.71 | — |

C2에서는 sharpen_off·teacher_freeze가 lr_zero보다 late dev가 높다. 멈추는 것보다 학습을 계속하며 원인만 없애는 쪽이 낫다.

## Phase B 확정 설계 (Phase A 판정 직후, test 전 등록)

- 조건: C1, C2. C3는 vocab_proto에서 C0보다 낮아 제외한다.
- 공간: vocab_proto 공간(C1 19축, C2 18축)에 스케줄 4축을 더한다(`src/late_schedule.py`).
  - `_late_relax` u ∈ [0,1]: 후반 teacher/student 비율 = 1 − (1 − r)(1 − u), r은 기본 비율. u=1이면 sharpening을 완전히 없앤다.
  - `_late_start_frac` ∈ [0.05, 0.75], `_late_len_frac` ∈ [0, 0.5]: 완화 시작과 cosine 완화 길이(0이면 즉시 전환).
  - `train.teacher_freeze_frac` ∈ [0.1, 1.0]: 이 시점부터 teacher EMA momentum 1(1.0이면 동결하지 않음).
  - u=0과 freeze 1.0이면 config 키를 추가하지 않으므로 vocab_proto 설정과 비트 단위로 같다.
- 탐색: 조건별 독립 TPE(seed 20261118, startup 8, multivariate), 조건당 40 attempted, seed 42.
  목적함수는 late dev(step 3500/3750/3999 평균)이고, MedianPruner(startup 8, warmup 2000, 간격 250, 최소 4)가 현재 dev를 비교한다.
- anchor 5개(조건별): C2는 t9+sharpen_off@1000, t9+teacher_freeze@1000, t3(스케줄 없음), t3+sharpen_off@1000, t28+sharpen_off@1000.
  C1은 t16+sharpen_off@1000, t16+teacher_freeze@1000, t29(스케줄 없음), t29+sharpen_off@1000, t14+sharpen_off@1000.
- 재현 검증: 처음 3개 anchor는 Phase A 개입 run 또는 vocab_proto run과 모든 기록이 같아야 하며, 다르면 campaign을 멈춘다.
- 확인: 조건별 late dev top 3 × seed 101/102/103(18 run). C0 t13도 같은 seed로 재학습한다(3 run, seed 짝 비교용).
  3시드 late dev 평균으로 동결하고 `selection.json`을 test 전에 기록한다.
- 스케줄 효과 대조: 선택된 설정에서 스케줄 4축만 중립으로 둔 run을 같은 seed로 돌린다(6 run).
- test: C0, C1, C2, C1_off, C2_off × seed 3개 × best/last(30 eval). **주 지표는 STS7 raw last**다.
- 판정(seed 짝, 규칙은 평균 Δ ≥ 0.003이고 2/3 양수면 성립, 평균 ≤ 0이고 2/3 비양수면 기각, 그 외 판정불가):
  - **R1** C1 또는 C2 vs C0, STS7 raw last.
  - **R2** 후반 하락 해소: best − last 평균 ≤ 0.5pp이고 시드별 ≤ 1.0pp이면 성립, 평균 > 1.0pp이면 기각, 그 외 판정불가.
  - **R3** 스케줄 자체 효과: 같은 설정에서 스케줄 켬 vs 끔, STS7 raw last.
- 예산: 탐색 최대 80 + 확인 21 + 대조 6 = 최대 107 full run, 그 뒤 eval 30회.
- 동시 실행: 사용자가 2026-11-17 긴급 상황을 이유로 계정 2개 제한을 이번 작업에 한해 해제했다. 유휴 GPU 4개(5–8)에 4 slot으로
  실행하며(조건당 동시 2 trial, TPE constant liar), gs25009/gs25049 GPU 제외·GPU당 1 프로세스·유휴 GPU만 사용 규칙은 그대로다.
  gpu_guard의 기본 상한 2는 바꾸지 않았고, 실행할 때만 `FLOWDINO_GPU_MAX_PROCESSES`로 올린다.

## 동결 전 test 열람 공개 (2026-11-17, 사용자 요청)

확인 단계 도중(21개 중 4개 완료) 사용자 요청으로 탐색 seed 42의 최고 run 2개(C2 trial 35, C1 trial 30)와
C0 t13 seed 42의 best/last checkpoint에서 STS7을 계산했다(`pre_freeze_test_seen.json`, `scripts/full_metrics_by_space.py`).
선택 규칙(seed 101–103 late dev 평균)은 이미 고정된 기계적 규칙이라 바꾸지 않는다. 이 시점부터 `selection.json`의
`test_scores_seen: false`는 "확인 run(seed 101–103)의 test를 보지 않았다"는 뜻이며, 보고서에도 이 열람을 공개한다.
