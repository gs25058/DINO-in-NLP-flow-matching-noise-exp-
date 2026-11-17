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
