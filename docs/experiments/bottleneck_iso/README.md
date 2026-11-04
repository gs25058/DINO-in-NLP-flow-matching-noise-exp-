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
