# 후속 실험 보고서

현재 새 실험: [공분산 정규화 완화·성분 분리](coviso_relax/README.md).

아래 문서는 원격 `origin/exp/r19-simcse-gram`의 **211ea78d24cba6b8921b81024ac331686781a609**에서
선별해 원문 그대로 보관했다. 원본 위치는 각각 `results/analysis/<실험>/report.md`다.
코드, 체크포인트, DB, 이미지와 실험 전용 설정은 가져오지 않았다. 보고서 안의 원래 경로는 당시 원격 트리 기준이다.

| 문서 | 이해에 도움이 되는 내용 |
|---|---|
| [R13](r13/report.md) | teacher sharpening과 뷰 구조 변경이 기대만큼 성능을 높이지 못한 이유 |
| [R14~R17](r14_r17/report.md) | 긴 학습 예산, 토큰 예측, 폐기한 실험의 설계와 당시 진행 상황 |
| [R18](r18/report.md) | 4000-step 대조군, iBOT 부정 결과, STS-B와 7-task 순위 차이 |
| [R19](r19/report.md) | SimCSE 구현 차이 교정 근거와 gram 실험 부정 결과 |

서로 다른 시점의 기록이다. 특히 R14~R17의 중간 상태는 후속 R18 기록과 함께 읽는다.
9월 보고서 수치를 10월 재실험 수치로 대체하거나 역으로 덮어쓰지 않는다.

이 로컬 checkout의 10월 고밀도 궤적 재실험 기록은
[`results/analysis/dense_trajectory_20261021/README.md`](../../results/analysis/dense_trajectory_20261021/README.md)에 있다.
당시 소스는 `results/experiments/dense_trajectory_20261021/{code,early_code}`로 고정되어 있다.
이 경로들은 기존 로컬 산출물이며 Git에서 제외된다. 새 SimCSE 구현의 버전 표식은
`simcse_original_aligned_v1`이다. 버전 표식이 없는 이전 결과를 새 구현 결과로 해석하지 않는다.
