# 어휘 prototype DINO — 실행 안내

[사전 등록](OPTUNA_SPEC.md), [기준선 출처](baseline_provenance.json), [사전 검증](validation.json), [보고서](report.md).

```bash
source scripts/env.sh
# C3 k-means 캐시(한 번만, CPU). 이미 있으면 덮어쓰지 않는다.
uv run --no-sync python scripts/prepare_vocab_kmeans.py --threads 8
# campaign: 불변 source snapshot을 만든 뒤 Phase 0 → smoke → 탐색 → 확인 → test를 순서대로 진행한다.
setsid nohup uv run --no-sync python scripts/run_vocab_proto.py \
  --campaign /abs/path/results/experiments/vocab_proto_YYYYMMDD > campaign.log 2>&1 & disown
```

- 조건 차이는 `model.head.{head_type, head_freeze, kmeans_path}`와 탐색값뿐이다. 학습 루프는 공통이다.
- 모든 학습과 평가는 `scripts/gpu_guard.py`(`--gpu auto`, 최대 2개)를 거친다.
- 진단: 각 run의 `components.jsonl`(H_pt_norm, teacher_pair_KL, copy rate), `evaluations.jsonl`(head 이탈량,
  입력 임베딩 이동량), `vocab_topwords.jsonl`(고정 5문장 top-10 단어).
- 중지: controller PID에만 SIGTERM을 보내면 자신이 띄운 guard 프로세스 그룹만 종료한다.
