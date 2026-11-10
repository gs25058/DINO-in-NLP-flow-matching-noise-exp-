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

## 중단된 탐색 재개

`resume_weak_view.py`는 원래 동결 source로 학습·평가를 실행한다.
완료 trial은 재사용하며 중단 trial은 같은 설정·seed로 새 run 이름에서4000step 다시 학습한다.
체크포인트에 optimizer/scheduler/RNG 상태가 없으므로 부분 학습을 이어붙이지 않는다.
원래 부분 로그·checkpoint·config는 덮어쓰지 않고, 재개 전 DB/state/report를 별도 복사한다.
총 탐색 trial은 여전히20개다. 이후 기존 명세의 top3×3시드 확인과 대조군·STS7이 자동 진행된다.
원 sampler RNG가 저장되지 않았으므로 재개 이후 새 제안은 seed20261111로 명시적으로 등록한다.
범위·예산·선택 규칙·기존 결과는 유지하며 제안 순서가 원래 직렬 실행과 동일하다고 주장하지 않는다.
추가 controller는 campaign `continuations/retry1/controller.py`로 동결하고 SHA를 등록한다.
