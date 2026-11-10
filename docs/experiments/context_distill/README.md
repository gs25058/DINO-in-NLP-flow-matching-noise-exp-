# 주변 문맥 비대칭 자기증류

[사전 등록](OPTUNA_SPEC.md) · [결과](report.md)

- 데이터: `src/context_data.py`, `scripts/prepare_context_data.py`
- 입력/노이즈/attention 차단: `src/context.py`
- 공통 모델·손실·학습·평가: `src/model.py`, `src/loss.py`, `src/train.py`, `src/evaluate.py`
- 실행: `scripts/run_context_distill.py`; heldout 진단: `scripts/diagnose_context.py`
- 설정: `configs/context_distill/{base,A,B,C,D}.yaml`

원자료는 [Wikipedia 데이터셋](https://huggingface.co/datasets/wikimedia/wikipedia)의
고정 revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`, 영어20231101 첫parquet shard다.
문서ID와 원문 text가 필요하다. wiki1m 행 인접성을 대신 쓰지 않는다.

```bash
# 기존 환경 유지: uv sync를 실행하지 않는다.
source scripts/env.sh
# 네트워크 접근이 있는 준비 단계에서 (약420MB):
uv run --no-sync python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='wikimedia/wikipedia', repo_type='dataset', revision='b04c8d1ceb2f5cd4588862100d08de323dccfbaa', filename='20231101.en/train-00000-of-00041.parquet', local_dir='data/context_raw'))"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
# CPU는 원자료 처리와 고정 word-embedding 통계만 계산한다. CPU 장시간 학습은 하지 않는다.
OMP_NUM_THREADS=2 uv run --no-sync python scripts/prepare_context_data.py \
  --raw-shard data/context_raw/20231101.en/train-00000-of-00041.parquet \
  --out-dir data/context_wikipedia_v1 --max-documents 20000 --max-train 150000
setsid nohup uv run --no-sync python scripts/run_context_distill.py \
  --campaign results/experiments/context_distill_20261110 \
  > results/logs/context_distill_20261110.supervisor.log 2>&1 < /dev/null &
disown
```

이미 준비된 캐시는 덮어쓰지 않는다. 다른 위치에 준비하면 base config의 data 경로5개를 함께 바꾼다.
원자료·cache·stats SHA와 문서 분리 목록은 data/…/manifest.json에 있다.
실행기는 데이터 무결성 확인 후 불변 source를 만들고, 3조건 smoke를 통과해야 full training을 시작한다.
최대2개GPU를 admission으로 사용한다. GPU가 없으면 대기하며, 실패/부분학습을 완료 점수로 쓰지 않는다.

`pairs.json`: 동일 조합 A/B/C run과 paired dev 차이를 재구성할 수 있는 원자료.
`shortlist.json`·`confirmations.json`·`selection.json`: 새3seed 검증과 test 이전 선택 동결.
`far_context_decision.json`: test 이전 D실행 여부. `final_runs.json`: 모든 최종 run의 정확한 config경로.
`results/analysis/*.json`: task별 STS7 및 heldout 진단. `report.md`: 실제 상태·집계·판정·비용.
