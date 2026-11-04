# Bottleneck cosine paired experiment

방법·예산·판정: [OPTUNA_SPEC.md](OPTUNA_SPEC.md). 결과: [report.md](report.md).
초기 R18 설정: `configs/bottleneck_cosine/base.yaml`.

```bash
source scripts/env.sh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p results/logs
setsid nohup uv run --no-sync python scripts/run_bottleneck.py \
  --campaign results/experiments/bneck_cosine_20261104 --gpus auto,auto \
  > results/logs/bneck_cosine_20261104.supervisor.log 2>&1 < /dev/null &
disown
```

같은 이름의 campaign은 다시 만들지 않는다. 예산은 각20trial×4000step이며 축소하지 않았다.
GPU 한 장이면 `--gpus auto`를 쓴다. 공통 admission 검사로 계정 전체 GPU process≤2를 보장한다.

실행 순서: 초기 gradient calibration → 보정 수치 사전 등록 → 양쪽60-step smoke →
20+20 full trials → 각 top3×새3시드 → dev로 선택 동결 → experimental λ=0×3시드 → STS7.
각 run은 새 초기값에서 시작한다. optimizer/EMA 갱신 없는 보정 코드는 `src/bottleneck.py`에 있다.
보조 손실은 `src/loss.py`, cached bottleneck은 `src/model.py`, 학습은 공통 `Trainer`를 사용한다.

`--prepare-only`는 실행 소스/config/명세를 먼저 고정한다. `--execute`는 그 frozen source에서만
새 study를 실행한다. 진행한 study의 sampler를 조용히 재시작하지 않고 실패 산출물을 덮어쓰지 않는다.

```bash
uv run --no-sync python scripts/report_bottleneck.py --campaign results/experiments/bneck_cosine_20261104
```

보고서는 실제 완료 데이터만 읽는다. 각 명령은 commands/*.json, exit code는 status/,
source SHA와 git diff는 source_manifest.json/source_diff.patch, 데이터·초기값은 각 run provenance,
배치 순서는 batch_order.json과 comparability_audit.json에서 확인한다.
선택 trial은 selection.json, 각 조건 top3는 shortlist.json, 재검증은 confirmations.json,
최종군/aux_off는 final_runs.json, 개별768d·256d test는 results/analysis/*.json이다.

기존 미커밋 보고서와 기존 선택 config는 이번 실험의 입력이나 수정 대상으로 사용하지 않는다.
