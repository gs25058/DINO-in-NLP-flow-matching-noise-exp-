# 약한 학생 뷰 실험: clean EMA FlowDINO

2026-11-10, exp/weak-student-view. 사용자 요청으로 실행 전 등록한다.

## 가설과 기준선

학생3뷰 중1뷰를 약한 노이즈로 바꾸면 의미 관계 보존이 개선되어 최종768d STS가 오른다는 가설이다.
기준선은 full_ema_joint_20261110의 dev-best trial1 실제 설정이다(구조·LR·온도·노이즈는 R18 계보).
과거75.0869 dev /66.9478 STS7은 참조다. 새 대조군을 새 소스에서 독립 실행하고 그것으로 비교한다.
기준선 config는 configs/weak_student_view/base.yaml이며 실제 선택 config와의 일치를 검증한다.
bert-base-uncased, mean+L2 768, MLP+L2 256→8192, DINO CE8192+iso768,
완전한 clean EMA teacher(자체 EMA word lookup), batch32, 학생3뷰, 길이128,4000step을 유지한다.
cosine·iso256·직접256DINO·새 contrastive 항은 없다. 모든 run은 pretrained 초기값에서 새로 시작한다.

## 뷰와 탐색

첫 학생뷰만 t~Uniform(0,weak_t_max), 별도 warmup 없이 전 구간 같은 범위를 사용한다.
나머지2뷰는 실제 대조군의 원래 flow-noise와 warmup을 유지한다.
특수/패딩 토큰 마스킹, t[B]→epsilon[B,L,D]의 뷰별 난수 추출 순서를 보존한다.
같은 seed/입력에서 강한2뷰와 RNG 상태가 bitwise 같아야 한다. teacher는 항상 clean이다.
손실은 여전히3뷰를 동일 비중으로 평균하며 covariance EMA는 각뷰마다 기존 순서로1회씩 갱신한다.

20 trial ×4000step, seed42, 최대2 GPU에서 병렬 TPE.
탐색 변수3개: weak_t_max=[.05,.30] log, mean lambda=[.05,8] log, cov lambda=[.05,8] log.
다른 모든 설정은 고정한다. LR·head LR·온도·strong noise를 함께 튜닝하지 않는다.
첫 trial은 weak_t_max=.15, 두 가중치는 각각5.874093568592943.
TPE seed20261110, startup5, multivariate+constant_liar, pruning 없음.
두60-step smoke를 통과한 뒤, 첫trial과 새 기준선 seed42를 동시에4000step 실행한다.
이 두 run은 약한뷰 여부만 다른 일차 비교다. 첫trial은20개 예산에 포함한다.
실패/부분학습은 성능 결과로 사용하지 않고 controller가 종료해 기록을 보존한다.

## 확인과 평가

목적함수는768d STS-B dev Spearman best. step0/250/…/3999, 최초 동률을 선택한다.
완주 top3를 새 seed101/102/103에서 각각4000step 재검증한다(9run).
그 dev-best 평균으로 선택을 동결한다. 그 후 두 대조군을 같은3시드로 새로 학습한다:
1. 원래 기준 설정+원래3 strong뷰.
2. 최종 weak 설정에서 weak_view만 제거한3 strong뷰(동일 가중치 대조군).
두번째 비교는 학습된 체크포인트에 변형을 적용하는 것이 아니라 동일 pretrained 초기값에서 새로 시작한다.
총 full run36회(탐색20+후보확인9+원기준선42포함4+동일가중치대조3), smoke2회.
세최종조건 각각3seed×best/last의 STS7을 동결후 평가한다(18eval).
이미 기준선의 과거 test를 봤음을 공개하며 새로운 test로 설정을 변경하지 않는다.

주판정은 weak 조건과 동일 가중치 strong 대조군의 dev-selected best 768d STS7 paired 차이다.
평균차≥.003이고2/3 양수면 성립, 평균≤0이고2/3 비양수면 기각, 나머지는 판정불가.
기준선 대비 차이는 튜닝을 포함한 방법 비교다. alignment/uniformity/rank와256d STS는 진단이다.
어떤 uniformity 수치도 목표로 강제하지 않는다. 단일seed pilot만으로 개선을 확정하지 않는다.

## 추적과 실행

resolved config, commit/diff, source SHA, data SHA, 초기값, ordered batch hash,
first-view SHA, loss component·기여·기존 sparse gradient 진단을 기록한다.
first-view는 같은seed+augmentation 설정끼리 비교한다. 초기값·데이터 순서는 설정이 달라도 같아야 한다.
GPU_POLICY.md를 준수하며 계정전체최대2GPU process, 유휴GPU만 admission한다.
활성 스냅샷과 과거 산출물을 수정하지 않으며 main merge/push는 하지 않는다.

실행: `source scripts/env.sh` 후 offline 환경에서
`uv run --no-sync python scripts/run_weak_view.py --campaign results/experiments/weak_student_view_20261110`.
장시간 controller는 setsid nohup으로 실행한다. GPU가 없으면 admission에서 대기한다.
