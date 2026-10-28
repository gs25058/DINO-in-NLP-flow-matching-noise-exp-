# 공분산 정규화 완화·성분 분리 — 실행 보고서

상태: **smoke_complete**. campaign: `/src/gs25058/noise_experiment/noise_experiment/flowdino/results/experiments/coviso_relax_smoke_20261028`.
사전 등록은 SPEC.md이며, test 점수는 설정/후보 선택에 사용하지 않는다.

## 기준선과 실행 소스

실제 R18 원격 기준선 `eed56a1a9f669129a8dbd428443f563ccec35074`; λ₀=5.874093568592943.
작업 시작 commit `504a48921f195f9b9aae0f8833d511f61f41430c`, branch `exp/coviso-component-relaxation`.
실행한 전체 소스·변경 내역·파일 해시는 campaign의 source/, source_manifest.json, source_diff.patch에 보존한다.
R18 과거 STS-B dev 0.7465 / STS7 66.09는 참고값이며, 현 A6000 대조군과 직접 동등하다고 가정하지 않는다.

## 시드별 dev 및 진단 (실측)

| arm | seed | best dev | best loop step | last dev | alignment last | uniformity last | rank last |
|---|---|---|---|---|---|---|---|

## 평균과 시드 변동

표준편차는 표본 SD이며 신뢰구간이 아니다.

| arm | n | best mean ± SD | last mean ± SD |
|---|---|---|---|

## 손실 성분과 backbone gradient

가중 기여는 loss 절대값이며 영향력은 별도 gradient 진단과 함께 본다.

| arm | seed | L_DINO last | L_mean last | L_cov last | weighted mean | weighted cov | g_mean/g_DINO | g_cov/g_DINO |
|---|---|---|---|---|---|---|---|---|

시간에 따른 변화는 각 run의 components.jsonl / gradient_diagnostics.jsonl / evaluations.jsonl에 보존한다.

가설 판정: **판정불가(고정 매트릭스 미완료)**.

## STS7 (설정 동결 후 test)

| arm | seed | checkpoint | STS7 (%) |
|---|---|---|---|
| 미실행 | — | — | — |

## 실제 명령과 완료 상태

- `train_control_s42` exit=0
  `/src/gs25058/noise_experiment/noise_experiment/flowdino/.venv/bin/python3 /src/gs25058/noise_experiment/noise_experiment/flowdino/results/experiments/coviso_relax_smoke_20261028/source/scripts/gpu_guard.py --gpu 6 --wait --timeout 7200 -- /src/gs25058/noise_experiment/noise_experiment/flowdino/.venv/bin/python3 -m src.train --config /src/gs25058/noise_experiment/noise_experiment/flowdino/results/experiments/coviso_relax_smoke_20261028/configs/control_s42.yaml --device cuda --stop-after-steps 60`
- `train_cov1_s42` exit=0
  `/src/gs25058/noise_experiment/noise_experiment/flowdino/.venv/bin/python3 /src/gs25058/noise_experiment/noise_experiment/flowdino/results/experiments/coviso_relax_smoke_20261028/source/scripts/gpu_guard.py --gpu 9 --wait --timeout 7200 -- /src/gs25058/noise_experiment/noise_experiment/flowdino/.venv/bin/python3 -m src.train --config /src/gs25058/noise_experiment/noise_experiment/flowdino/results/experiments/coviso_relax_smoke_20261028/configs/cov1_s42.yaml --device cuda --stop-after-steps 60`

남은 교락: 현재 서버/라이브러리와 과거 R18 환경 차이, 2~3시드의 표본 한계.
공통 초기값·학습 스케줄·데이터·평가를 고정한 범위에서만 가중치 효과를 해석한다.
uniformity와 rank는 진단 지표이며 SimCSE 수치를 목표로 강제하지 않았다.
