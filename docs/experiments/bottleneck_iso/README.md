# Bottleneck isotropy 실험

[사전 등록](OPTUNA_SPEC.md)의 baseline 대 cosine+256d iso 비교다.
학습은 `src/train.py` / `src/loss.py` 공통 경로를 쓰고 별도 루프를 만들지 않는다.

```bash
source scripts/env.sh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
setsid nohup uv run --no-sync python scripts/run_bottleneck.py \
  --with-bneck-iso --campaign results/experiments/bneck_iso_20261104 \
  --gpus auto,auto --trials-per-arm 20 \
  > results/logs/bneck_iso_20261104.supervisor.log 2>&1 < /dev/null &
disown
```

한 GPU에서는 `--gpus auto`. `--prepare-only`는 소스 고정·실행 준비만 한다.
재시작 대신 새 campaign을 사용하며 기존 SQLite의 sampler 상태를 조용히 초기화하지 않는다.
각 run은 독립 초기화되고 학습 weight resume은 없다.

- 기본 config: `configs/bottleneck_iso/base.yaml` (추가항 모두0, 공통 진단 켬).
- 추가항 양수의 정확한 실행 config: campaign의 `configs/{label}_s{seed}.yaml`.
- 보정/범위: `calibration.json`, `search_registration.json`.
- 선택/순수 ablation: `selection.json`, `final_runs.json` (`iso_off`/`aux_off`).
- 실제 완료 여부: `state.json`, `status/`, run별 `completion.json`.
- 보고서: [report.md](report.md), campaign의 `report.md`.

```bash
CUDA_VISIBLE_DEVICES='' UV_CACHE_DIR=/tmp/flowdino-uv-cache \
  uv run --no-sync python -m pytest -q
uv run --no-sync python scripts/report_bottleneck.py \
  --campaign results/experiments/bneck_iso_20261104
```

중지할 때는 현재 campaign의 정확한 supervisor PID에만 SIGTERM을 보낸다.
guard의 자식 종료·예약 해제를 확인한 다음 별도 호출로 다음 작업을 실행한다.
예전 cosine campaign은 사용자 요청으로 중단됐으며 완료 결과와 부분 로그를 그대로 보존한다.

## 2026-11-04 실행 확인

실행 소스는 `ee13213`, campaign은 `results/experiments/bneck_iso_20261104`다.
수정 전71개/수정 후84개 테스트가 통과했다. GPU 보정에서9개 상태 보존 검사를 모두 통과했고,
두 조건60-step smoke도 각각 exit0으로 완료했다. iso=0 GPU smoke의 기존 공통
출력·손실·gradient·평가 최대 차이는0이며 초기값과 batch 순서 hash도 같았다.
근거는 [validation.json](validation.json)과 campaign의 `zero_iso_gpu_equivalence_audit.json`에 있다.

실측 보정 log 범위는 cosine `[0.0017612116928148694, 0.17612116928148694]`,
iso256 `[0.00828181379315871, 0.828181379315871]`이다. 각각 trial의 가중치가 이 범위에서
선택된다는 뜻이며, 학습 전체의 gradient 비율을 고정하는 방식은 아니다.
현재20+20 trial 탐색을 시작했으며 4000-step 결과·최종 개선 판정은 아직 없다.
GPU 규약에 따라 동일 계정의 다른 작업이 있으면 다음 작업의 admission이 대기할 수 있다.
