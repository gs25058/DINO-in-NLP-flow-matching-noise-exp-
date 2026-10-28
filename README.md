# FlowDINO: 공동 Optuna 실험

영어 wiki1m 문장을 BERT로 인코딩하고, 깨끗한 EMA teacher의 prototype 분포를
가우시안 보간 노이즈를 받은 학생 뷰 3개가 예측한다. 문장 표현은 마지막 hidden state의
masked mean + L2이며, DINO cross entropy와 평균·공분산 정규화를 함께 최적화한다.

이 브랜치의 실행 명세는 [METHOD.md](METHOD.md)와
[공동 탐색 사전 등록](docs/experiments/coviso_relax/OPTUNA_SPEC.md)이다.

## 코드 읽는 순서

| 파일 | 역할 |
|---|---|
| `configs/coviso_relax/base.yaml` | 실제 R18 대조군의 정확한 활성 설정 |
| `src/train.py` | config·epoch 배치 준비, `Trainer.training_step`, optimizer/EMA, 평가·저장 |
| `src/augment.py` | 토큰 표준화 → 독립 t·ε → 노이즈 보간, 특수/패딩 토큰 보존 |
| `src/model.py` | BERT → masked mean → 문장 임베딩과 DINO head, teacher EMA |
| `src/loss.py` | DINO target/center, 순차 covariance EMA, 두 성분 가중치와 gradient 진단 |
| `src/evaluate.py` | STS-B dev·표현 기하·effective rank·고정 7-task test |
| `src/joint_search.py` | 19개 탐색 축과 유효한 config 생성 |
| `src/provenance.py` | 불변 소스 스냅샷, config·데이터·초기값 해시, 실행 기록 |
| `scripts/run_coviso_optuna.py` | 유한 탐색 → 추가 시드 → 후보 동결 → STS7 |
| `scripts/analyze_optuna.py` | 시드별 표·진단·판정 보고서, 선택 config 내보내기 |

`training_step`은 토큰화 → 원본 임베딩 → 노이즈 뷰 → teacher/student → 손실까지만 계산한다.
`optimize`는 gradient 진단·AdamW·scheduler·teacher EMA를 처리한다.
`evaluate`는 측정과 best 선택을 담당하고, `run`은 이 순서를 연결한다.

## 환경과 데이터

Python 3.11, uv, PyTorch 2.5.1/cu121을 사용한다. GPU 사용은 [공용 규약](GPU_POLICY.md)을 따른다.

```bash
source scripts/env.sh
uv sync
```

현재 실행 중인 campaign은 고정 소스와 기존 가상환경을 사용한다. 이 작업에서는 가상환경을
동기화하지 않았고, 기존 데이터·checkpoint·실행 소스도 변경하지 않았다.
이미 실행 중인 campaign과 같은 환경을 사용할 때는 `uv run --no-sync`를 쓴다.

학습은 config에 지정된 세 캐시를 읽는다: `sentences.jsonl`, `rank_eval_sentences.jsonl`,
`embed_stats_bert_base_uncased.pt`. 현재 캐시를 재생성하지 않는다.
새 환경에서 준비할 때만 **새 출력 위치**에 실행한다.

```bash
flowdino_gpu_python auto scripts/prepare_data.py \
  --config configs/coviso_relax/control.yaml --out-dir data --device cuda
```

wiki1m의 10자 미만 문장을 제거하고, seed 0으로 고른 10,000개 문장의 토큰 임베딩에서
μ·σ를 계산한다. 이와 겹치지 않는 2,048개 문장을 rank 진단에 쓴다. 토큰 길이 제한은 128이다.

## 학습과 스모크

```bash
source scripts/env.sh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
flowdino_gpu_python auto -m src.train --config configs/coviso_relax/control.yaml --device cuda
```

스모크에는 `--stop-after-steps 60`을 덧붙인다. **4000-step 스케줄을 압축하지 않는다.**
독립 실행마다 `run_name`을 바꾸고, 같은 이름의 산출물을 겹쳐 쓰지 않는다.

학습 출력은 `results/logs/<run>/`의 resolved config·provenance·components/gradient/evaluation
JSONL과 completion, `results/tensorboard/<run>/`, `checkpoints/<run>/{best,last}.pt`이다.

## 공동 탐색

```bash
mkdir -p results/logs
setsid nohup uv run --no-sync python scripts/run_coviso_optuna.py \
  --campaign results/experiments/<새-campaign> --gpus auto,auto --trials 40 \
  > results/logs/<새-campaign>.supervisor.log 2>&1 < /dev/null &
disown
```

계정 전체 GPU 프로세스 최대 2개, 빈 GPU가 없으면 대기한다. 한 장이면 `--gpus auto`로 순차 실행한다.
새 campaign의 대조군도 같은 소스에서 독립 시작한다. `--import-baseline <campaign>`은
**학습·평가 소스와 config·데이터가 동일한** 완료 대조군에만 허용한다. 리팩토링 전 campaign을
새 코드의 완료 대조군으로 자동 인정하지 않는다. 기존 study의 sampler를 재초기화하거나
학습 checkpoint에서 이어 학습하지 않는다.

최대 40 trial은 기준선·pruned·failed를 포함한다. 각 trial은 seed 42, 최대 4000 step이다.
완주 상위 3개 설정과 대조군을 seed 43·44로 검증하고, 3시드 dev best 평균으로 후보를 고른다.
`selection.json`을 저장한 뒤 대조군/후보의 best·last를 7-task test에서 평가한다.

```bash
uv run --no-sync python scripts/analyze_optuna.py --campaign results/experiments/<campaign>
# 동결된 19개 값으로 현재 명세의 재사용 config 생성; 기존 파일은 덮어쓰지 않는다.
uv run --no-sync python scripts/analyze_optuna.py \
  --campaign results/experiments/<campaign> --out-config configs/coviso_relax/<선택-config>.yaml
```

campaign의 `report.md`, `trials.json`, `diagnostic_summary.json`과 `results/analysis/sts7_*.json`을 읽는다.
현재 진행하던 campaign은 `results/experiments/coviso_joint_optuna_20261028`이고,
리팩토링 전 `e1d2ea3`의 불변 소스로 계속 실행한다. 추가 campaign을 중복으로 시작할 필요는 없다.
기존 미커밋 보고서도 이번 정리 커밋에서 제외한다.

선택 후 별도 평가가 필요하면 해당 campaign의 고정 source에서 실행한다. 새 코드로 만든 checkpoint는:

```bash
flowdino_gpu_python auto scripts/eval_sts7.py --runs <run> --which best --device cuda --tf32 \
  --out results/analysis/<새-결과>.json
```

## 검증

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  uv run --no-sync python -m pytest -q
```

삭제한 실험만을 위한 테스트는 제거했다. 유지한 수식·gradient·EMA·sampling·평가와 실제
전체 실행 경로를 검증하며, 변경 전 tiny-BERT의 두 설정×두 시드 기준값과 정확히 비교한다.
[정리 근거와 검증 범위](docs/refactor-20261028.md)에 제한 사항을 기록한다.
