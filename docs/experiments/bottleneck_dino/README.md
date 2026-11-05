# 직접256d DINO+iso 실험

[사전 등록](OPTUNA_SPEC.md). 실험2는 cosine이 아니라 **centering/sharpening을 포함한 DINO CE + iso**다.

- 구조: BERT→mean→MLP→L2의256d를 직접 DINO logits로 사용.8192 확장 없음.
- 기준선: 기존 최적 FlowDINO config·3시드 결과를 고정. 재탐색 없음.
- 새 구조: LR2개·두 온도·iso 가중치 Optuna20×4000step 후 top3×3시드 확인.
- 주 평가768d STS.256d STS와 분포 지표는 진단.

```bash
source scripts/env.sh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
setsid nohup uv run --no-sync python scripts/run_bottleneck.py \
  --direct-bottleneck-dino --campaign results/experiments/bneck_dino256_20261105 \
  --reference-baseline results/experiments/bneck_iso_20261104 --gpus auto,auto \
  --trials-per-arm 20 > results/logs/bneck_dino256_20261105.supervisor.log 2>&1 < /dev/null &
disown
```

공통 config는 `configs/bottleneck_dino/base.yaml`, 직접 조건은 `direct.yaml`이다.
고정 참조 검증은 campaign `fixed_baseline.json`, 실제 trial은 `configs/search_bottleneck_t*_s42.yaml`에 있다.
진행은 `state.json`·[report.md](report.md), 선택은 `selection.json`, 검증은 [validation.json](validation.json)에 기록한다.
새 모델 test는 선택 고정 후6평가이며, 기존 기준선 test를 참조한 점수와 구분한다.

2026-11-05: 구현 후88개 테스트 통과. seed42/101/102/103의 기존 초기 full hash를
정확히 재현했고, 실제 campaign에서 source·data·평가 cache·고정 설정을 검증했다.
실행 소스는 `8471057`이다. 새29회 full training과6개 test까지 모두 완료했다.
[최종 결과](RESULTS.md): 768d STS7은 기준선65.816±0.588, 직접DINO64.049±0.409로
개선 가설은 기각됐다.256d 진단 STS7은58.155→60.355로 개선됐다.
재사용 설정은 `configs/bottleneck_dino/selected.yaml`이며 실제 resolved config와 일치한다.

[손실별 역할·동역학 상세 분석](LOSS_DYNAMICS.md)에는 step별 gradient, teacher entropy,
실패 trial의 과도한 sharpening, 표현 공간별 기하 변화와 teacher 입력의 구현 주의점을 정리했다.
재현 실행기는 `scripts/analyze_loss_dynamics.py`이며, 학습 없이 완료 로그와 CPU checkpoint를 읽는다.
