# 공분산 정규화 완화·성분 분리 실행

- [사전 등록 명세](SPEC.md): 고정 4×2, 조건부 단일 항 cosine 감쇠, seed 44, test 평가 순서와 판정 기준.
- [실험 보고서](report.md): 실행기가 현재 campaign의 실제 완료 결과로 갱신한다.
- [원격 기준선 출처](baseline_provenance.json): R18 control과 정확한 상속 계보/λ₀.

```bash
source scripts/env.sh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
mkdir -p results/logs
# GPU 번호는 GPU_POLICY에 따라 현재 점유·소유자를 확인한 것으로 지정한다.
setsid nohup "$PY" scripts/run_coviso_relax.py \
  --campaign results/experiments/<new-campaign-name> --gpus "$GPUS" \
  > results/logs/<new-campaign-name>.supervisor.log 2>&1 < /dev/null &
disown
```

`--gpus`에 한 장 또는 두 장을 지정한다. 공통 GPU admission 검사로 각 실행 직전에 이 계정의
전체 사용량·소유자·유휴 상태를 확인한다. 제3자 GPU 작업을 종료하지 않는다.

먼저 별도 campaign에 `--smoke`를 붙여 control/cov1 각 60 step을 확인할 수 있다.
이때 4000-step LR/teacher/정규화 스케줄은 압축하지 않는다. 스모크를 고정 매트릭스 결과에 섞지 않는다.
준비만 하려면 `--prepare-only`, 이미 만든 불변 campaign을 실행하려면
`--campaign <path> --execute`를 사용한다. 실패/불완전 run의 결과를 덮어쓰거나 weight resume하지 않는다.

각 campaign은 source snapshot과 SHA-256, git commit·diff, SPEC, 실행 명령/exit code,
resolved YAML, 데이터 해시, 초기 student 해시, best/last checkpoint, JSONL 진단을 갖는다.
선택은 dev만 사용하며 `selection.json`을 기록한 뒤 STS7을 평가한다. STS7의 개별 7개 과제 점수는
`results/analysis/sts7_*.json`에 있고 보고서는 평균과 시드 변동을 집계한다.

후속 직접 자기증류의 공통 기준선: `configs/coviso_relax/control.yaml`.
실측으로 선택된 완화 config: `<campaign>/selected_relaxation.yaml`.
최종 판정이 성립하지 않으면 control을 기본으로 유지한다.
