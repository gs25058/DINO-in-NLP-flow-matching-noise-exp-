# 공동 Optuna 최적화 — 변경된 사전 등록

2026-10-28, 사용자 지시로 고정 성분 매트릭스를 중단하고 **성능 최적화**로 전환했다.
성분별 인과 판정·조건부 감쇠 대신, 아래의 공동 탐색과 추가 시드 검증으로 후보를 선택한다.
이미 완료된 control 42/43은 보존하고, 중단된 both1 run은 완료 trial로 사용하지 않는다.

## 목적과 고정 조건

STS-B dev **best checkpoint**의 Spearman을 최대화한다. last도 기록한다.
bert-base-uncased, mean+L2, batch 32, 학생 뷰 3개, clean EMA teacher, epoch 순회,
길이 128, 4000-step 최대 예산, 평가 250 step 및 마지막, FP32+TF32+batched views를 고정한다.
iBOT·토큰 회귀·새 증강을 추가하지 않는다. 각 trial은 동일 seed 42의 사전학습 초기값에서 독립 시작한다.

## 공동 탐색 공간과 예산

정확한 19개 축과 범위는 `src/joint_search.py::SPACE`가 실행 명세다.
mean/cov 가중치, backbone/head LR, grad clip, weight decay, LR warmup,
EMA 종착 모멘텀·ramp, teacher/student 온도와 teacher 온도 warmup,
centering/covariance EMA, 기존 Gaussian noise의 범위·초기 범위·warmup을 함께 탐색한다.
상속 이후의 resolved config를 run마다 저장한다. 노이즈 t_lo<t_start≤t_max<1을 파생식으로 보장한다.
이 범위 안의 경험적 최적값이며 전역 최적해라고 주장하지 않는다.

| 축 | 범위 | 표집 |
|---|---|---|
| mean / cov 가중치 (각각) | 0.05–8 | log |
| backbone LR | 2e-5–2e-4 | log |
| head LR | 1e-4–2e-3 | log |
| gradient clip | 1–6 | linear |
| weight decay | 0.01–0.3 | log |
| LR warmup fraction | 0.01–0.2 | linear |
| teacher EMA end | 0.998–0.9999 | linear |
| teacher EMA ramp fraction | 0.2–1 | linear |
| teacher temperature | 0.08–0.18 | linear |
| initial teacher temperature | 0.03–0.08 | linear |
| student temperature | 0.1–0.25 | linear |
| teacher temperature warmup fraction | 0.02–0.3 | linear |
| center momentum | 0.8–0.99 | linear |
| covariance EMA momentum | 0.75–0.99 | linear |
| noise t_lo | 0–0.6 | linear |
| noise span | 0.05–0.6 | linear |
| initial position within noise span | 0.1–1 | linear |
| noise warmup fraction | 0–0.15 | linear |

t_max=min(0.99,t_lo+span), t_start=t_lo+(t_max−t_lo)×initial_position이다.
teacher EMA 시작값 0.997, 정규화 시작 step 56, head 크기와 구조는 고정한다.

기본 총 **40 attempted trial 이내**(완료 기준선 1개 포함, pruned/failed도 예산에 포함).
고정 seed TPE(multivariate, constant_liar), startup 8개. 원래 baseline과 두 λ=1인 anchor를 포함한다.
진행 중인 고정 소스 campaign은 검증한 e7ee9d0 control seed42를 trial0으로 재사용한다.
리팩토링 후 새 campaign은 동일 방법·예산으로 대조군 42/43을 독립 학습하고 seed42를 trial0으로 기록한다.
소스·config·데이터가 완전히 동일한 완료 대조군에 한해 명시적 import를 허용한다.
이는 실행 소스 검증을 위한 경로이며 목적함수·탐색 공간·예산·선택 규칙은 변경하지 않는다.
원본 resolved config·checkpoint·명령·데이터/초기값 증거 경로를 함께 보존한다.

MedianPruner: 완료 startup 8개, warmup 2000 step, 500-step 간격, 최소 비교값 4개.
중간 보고값은 그때까지의 dev best다. 초반이 느린 설정을 즉시 탈락시키지 않는다.
pruned run은 완료 4000-step 결과와 구분하며, 최종 후보는 반드시 4000 step 완주한 trial에서만 선택한다.
학습 완료값 없는 오류는 FAIL로 남긴다. GPU는 계정 전체 최대 2개, 매 작업 공통 admission 검사.
유휴 GPU가 없으면 자동 admission 대기하며, 대기 시간은 학습 시간이나 완료 실험으로 계산하지 않는다.

## 추가 시드와 test 평가

완료 trial의 dev best 상위 3개 설정을 고정하고 seed43·44로 검증한다.
대조군도 3시드로 비교한다(control seed43은 같은 소스로 완주한 대조군을 사용).
각 후보의 3시드 dev best 평균으로 최종 후보를 선택하고 동률은 trial 번호로 결정한다.
3시드 mean/SD, paired Δ, last와 표현 기하, 손실/gradient를 모두 보고한다.

후보와 대조군을 동결한 `selection.json` 생성 이후에만 best/last STS7 test를 평가한다.
test 점수를 보며 다른 trial을 고르거나 탐색 범위를 수정하지 않는다.
`configs/coviso_relax/selected_joint.yaml`에 재사용 가능한 정확한 resolved 설정/선택 출처를 남긴다.

여러 축이 같이 바뀌므로 어느 개별 항의 인과 효과로 해석하지 않는다.
3시드 평균 dev 향상 ≥0.003이며 최소 2개 paired 양수면 개선 성립,
평균 ≤0이고 최소 2개 비양수면 기각, 그 외는 판정불가로 보고한다.
last와 STS7은 별도로 공개한다. 붕괴 감시의 과거 기준값을 복사하지 않는다.
