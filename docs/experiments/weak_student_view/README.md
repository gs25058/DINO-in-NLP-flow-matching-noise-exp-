# 약한 학생 뷰 1 + 기존 학생 뷰 2

[명세](OPTUNA_SPEC.md) · [결과](report.md)

기준선: `configs/weak_student_view/base.yaml`
약한뷰 pilot: `configs/weak_student_view/pilot.yaml`
실행기: `scripts/run_weak_view.py`

`source scripts/env.sh`와 HF offline 변수를 설정한 뒤:

```bash
setsid nohup uv run --no-sync python scripts/run_weak_view.py \
  --campaign results/experiments/weak_student_view_20261110 \
  > results/logs/weak_student_view_20261110.supervisor.log 2>&1 < /dev/null &
disown
```

공통 `src.train`을 사용한다. Optuna20trial과 후속3시드검증·대조군·STS7이 자동 실행된다.
최대2개 trial을 병렬 실행하며 GPU admission에서 계정전체2프로세스 상한을 지킨다.
`state.json`, `paired_pilot.json`, `selection.json`, `final_runs.json`, `report.md`로 결과를 확인한다.
`bottleneck`은 기존 공통 실행기의 내부 arm 이름으로, 이 실험의 DINO는8192d, iso는768d다.
`aux_off`는 cosine 제거가 아니라 선택된 weak 설정에서 weak뷰만 제거한 대조군이다.
