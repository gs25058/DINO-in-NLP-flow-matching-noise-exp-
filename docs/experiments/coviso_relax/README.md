# 공분산 정규화 완화·성분 분리 실행

- [현재 사전 등록](OPTUNA_SPEC.md): 사용자 지시로 전환한 19축 공동 Optuna, 최대 40 trial, 상위 3개 추가 시드 검증.
- [처음 명세](SPEC.md): 중단된 고정 매트릭스 설계 기록. 이미 완료된 대조군은 보존·검증 후 재사용한다.
- [실험 보고서](report.md): 실행기가 현재 campaign의 실제 완료 결과로 갱신한다.
- [원격 기준선 출처](baseline_provenance.json): R18 control과 정확한 상속 계보/λ₀.

```bash
source scripts/env.sh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
mkdir -p results/logs
# GPU 번호는 GPU_POLICY에 따라 현재 점유·소유자를 확인한 것으로 지정한다.
setsid nohup "$PY" scripts/run_coviso_optuna.py \
  --campaign results/experiments/<new-campaign-name> --gpus "$GPUS" --trials 40 \
  --import-baseline results/experiments/coviso_relax_20261028 \
  > results/logs/<new-campaign-name>.supervisor.log 2>&1 < /dev/null &
disown
```

`--gpus`에 한 장 또는 두 장을 지정한다. 공통 GPU admission 검사로 각 실행 직전에 이 계정의
전체 사용량·소유자·유휴 상태를 확인한다. 제3자 GPU 작업을 종료하지 않는다.
`--gpus auto,auto`는 최대 두 개의 실행 슬롯을 두고, 규약에 맞는 유휴 GPU가 생길 때 자동 배정한다.
선택된 실제 GPU 번호와 UUID는 각 commands/*.log에 남긴다.

공통 학습 루프의 control/cov1 각 60-step GPU 스모크는 이미 완료했다(`validation.json`).
4000-step 스케줄을 압축하지 않았으며 스모크를 Optuna의 완료 trial에 섞지 않는다.
준비만 하려면 `--prepare-only`, 이미 만든 불변 campaign을 실행하려면
`--campaign <path> --execute`를 사용한다. 이미 trial을 시작한 study의 sampler를 조용히 재초기화하지 않는다.
실패/불완전 run의 결과를 덮어쓰거나 weight resume하지 않는다.

각 campaign은 source snapshot과 SHA-256, git commit·diff, SPEC, 실행 명령/exit code,
resolved YAML, 데이터 해시, 초기 student 해시, best/last checkpoint, JSONL 진단을 갖는다.
선택은 dev만 사용하며 `selection.json`을 기록한 뒤 STS7을 평가한다. STS7의 개별 7개 과제 점수는
`results/analysis/sts7_*.json`에 있고 보고서는 평균과 시드 변동을 집계한다.

후속 직접 자기증류의 공통 기준선: `configs/coviso_relax/control.yaml`.
실측으로 선택된 공동 최적화 config: `configs/coviso_relax/selected_joint.yaml` 및 `<campaign>/selected_joint.yaml`.
이 파일은 3시드 dev 후보 선택이 끝나야 생성되며, 탐색 중 임의의 trial을 최종 후보로 표기하지 않는다.
최종 판정이 성립하지 않으면 control을 기본으로 유지한다.
