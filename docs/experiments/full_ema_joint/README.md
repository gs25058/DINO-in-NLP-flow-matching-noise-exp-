# 완전한 clean EMA FlowDINO

기존 DINO CE8192 + iso768을 유지하며 teacher word lookup도 EMA를 사용한다.
mean/cov 가중치를 독립적으로 포함한 기존19축을 Optuna20 trial로 튜닝한다.
[사전 등록](OPTUNA_SPEC.md), config는 `configs/full_ema_joint/base.yaml`이다.

```bash
source scripts/env.sh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
setsid nohup uv run --no-sync python scripts/run_bottleneck.py --full-clean-ema \
  --campaign results/experiments/full_ema_joint_20261110 --gpus auto,auto \
  --trials-per-arm 20 > results/logs/full_ema_joint_20261110.supervisor.log 2>&1 < /dev/null &
disown
```

trial0은 과거 joint13, trial1은 R18 초매개변수이며 모두 완전EMA로 새로 학습한다.
60-step smoke를 통과한 뒤 탐색하고, top3×새3시드 확인 후 dev로 선택을 고정한다.
최종 best/last의768d STS7을 주 평가하고256d는 진단으로 보고한다.
과거 점수는 참조이며 새 완전EMA 설정의 재현값으로 대신하지 않는다.
실행 상태는 campaign `state.json`, 명령·exit는 `commands/`·`status/`, 결과는 [report.md](report.md)다.

사용자 요청으로 탐색도 최대2개 trial 병렬 실행으로 전환했다.
[전환 명세](PARALLEL_CONTINUATION.md)에 기존 trial 보존 및 sampler 변경을 기록한다.
실제 전환 정보는 campaign의 `continuations/parallel_20261110/registration.json`,
실행 로그는 `results/logs/full_ema_joint_20261110.parallel.retry1.supervisor.log`에서 확인한다.
