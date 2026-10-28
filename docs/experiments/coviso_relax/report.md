# 공분산 정규화 완화·성분 분리 — 실행 보고서

상태: **고정 매트릭스 중단 — 공동 Optuna로 전환 준비**. campaign: `/src/gs25058/noise_experiment/noise_experiment/flowdino/results/experiments/coviso_relax_20261028`.
완료 control 두 시드는 보존한다. both1 두 시드는 사용자 지시로 중단했으며, 완료 결과로 사용하지 않는다.
현재 탐색 설계는 OPTUNA_SPEC.md를 따른다. 아래는 중단 시점의 대조군 기록이다.
사전 등록은 SPEC.md이며, test 점수는 설정/후보 선택에 사용하지 않는다.

## 기준선과 실행 소스

실제 R18 원격 기준선 `eed56a1a9f669129a8dbd428443f563ccec35074`; λ₀=5.874093568592943.
작업 시작 commit `e7ee9d0042452c43327580bb90d1a823babf9604`, branch `exp/coviso-component-relaxation`.
실행한 전체 소스·변경 내역·파일 해시는 campaign의 source/, source_manifest.json, source_diff.patch에 보존한다.
R18 과거 STS-B dev 0.7465 / STS7 66.09는 참고값이며, 현 A6000 대조군과 직접 동등하다고 가정하지 않는다.

## 시드별 dev 및 진단 (실측)

| arm | seed | best dev | best loop step | last dev | alignment last | uniformity last | rank last |
|---|---|---|---|---|---|---|---|
| control | 42 | 0.736902 | 2750 | 0.732824 | 0.270792 | -3.295936 | 322.75 |
| control | 43 | 0.738153 | 3000 | 0.736416 | 0.265249 | -3.204976 | 309.48 |

## 평균과 시드 변동

표준편차는 표본 SD이며 신뢰구간이 아니다.

| arm | n | best mean ± SD | last mean ± SD |
|---|---|---|---|
| control | 2 | 0.737527 ± 0.000884 | 0.734620 ± 0.002539 |

## 손실 성분과 backbone gradient

가중 기여는 loss 절대값이며 영향력은 별도 gradient 진단과 함께 본다.

| arm | seed | L_DINO last | L_mean last | L_cov last | weighted mean | weighted cov | g_mean/g_DINO | g_cov/g_DINO |
|---|---|---|---|---|---|---|---|---|
| control | 42 | 8.970794 | 0.005889 | 0.006789 | 0.034590 | 0.039877 | 1.1915 | 0.2968 |
| control | 43 | 8.972128 | 0.005900 | 0.007393 | 0.034658 | 0.043426 | 1.4397 | 0.3087 |

시간에 따른 변화는 각 run의 components.jsonl / gradient_diagnostics.jsonl / evaluations.jsonl에 보존한다.

가설 판정: **판정불가(고정 매트릭스 미완료)**.

## STS7 (설정 동결 후 test)

| arm | seed | checkpoint | STS7 (%) |
|---|---|---|---|
| 미실행 | — | — | — |

## 실제 명령과 완료 상태

- `train_both1_s42` exit=실행 중/미완료
  `/src/gs25058/noise_experiment/noise_experiment/flowdino/.venv/bin/python3 /src/gs25058/noise_experiment/noise_experiment/flowdino/results/experiments/coviso_relax_20261028/source/scripts/gpu_guard.py --gpu 6 --wait --timeout 7200 -- /src/gs25058/noise_experiment/noise_experiment/flowdino/.venv/bin/python3 -m src.train --config /src/gs25058/noise_experiment/noise_experiment/flowdino/results/experiments/coviso_relax_20261028/configs/both1_s42.yaml --device cuda`
- `train_both1_s43` exit=실행 중/미완료
  `/src/gs25058/noise_experiment/noise_experiment/flowdino/.venv/bin/python3 /src/gs25058/noise_experiment/noise_experiment/flowdino/results/experiments/coviso_relax_20261028/source/scripts/gpu_guard.py --gpu 9 --wait --timeout 7200 -- /src/gs25058/noise_experiment/noise_experiment/flowdino/.venv/bin/python3 -m src.train --config /src/gs25058/noise_experiment/noise_experiment/flowdino/results/experiments/coviso_relax_20261028/configs/both1_s43.yaml --device cuda`
- `train_control_s42` exit=0
  `/src/gs25058/noise_experiment/noise_experiment/flowdino/.venv/bin/python3 /src/gs25058/noise_experiment/noise_experiment/flowdino/results/experiments/coviso_relax_20261028/source/scripts/gpu_guard.py --gpu 6 --wait --timeout 7200 -- /src/gs25058/noise_experiment/noise_experiment/flowdino/.venv/bin/python3 -m src.train --config /src/gs25058/noise_experiment/noise_experiment/flowdino/results/experiments/coviso_relax_20261028/configs/control_s42.yaml --device cuda`
- `train_control_s43` exit=0
  `/src/gs25058/noise_experiment/noise_experiment/flowdino/.venv/bin/python3 /src/gs25058/noise_experiment/noise_experiment/flowdino/results/experiments/coviso_relax_20261028/source/scripts/gpu_guard.py --gpu 9 --wait --timeout 7200 -- /src/gs25058/noise_experiment/noise_experiment/flowdino/.venv/bin/python3 -m src.train --config /src/gs25058/noise_experiment/noise_experiment/flowdino/results/experiments/coviso_relax_20261028/configs/control_s43.yaml --device cuda`

## 구현 검증

변경 전 94개, 변경 후 106개 CPU 테스트 통과.
기존 수치·기울기·EMA 순서와 횟수, 가중치 0, teacher/EMA detach, 감쇠 경계, 단일 학습 루프와 best/last 저장을 검사했다.
별도 GPU 스모크: `results/experiments/coviso_relax_smoke_20261028`. control/cov1 60 step 정상 종료, 동일 초기 student 해시 확인.
스모크는 4000-step 스케줄을 유지했으며 본 실험 표에는 포함하지 않았다.

남은 교락: 현재 서버/라이브러리와 과거 R18 환경 차이, 2~3시드의 표본 한계.
공통 초기값·학습 스케줄·데이터·평가를 고정한 범위에서만 가중치 효과를 해석한다.
uniformity와 rank는 진단 지표이며 SimCSE 수치를 목표로 강제하지 않았다.
