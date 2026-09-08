# FlowDINO-Text

DINO 자기증류(EMA teacher + centering/sharpening) 구조는 유지하고, 증강만 flow matching식
가우시안 노이즈 보간으로 교체했을 때의 문장 임베딩 품질을 검증하는 소형 실험 저장소.

실험 조건(run)은 **오직 `configs/*.yaml` 값만** 다르다. 학습 루프에 조건별 분기는 없다.

## 요구 사항

- Python 3.11 (`pyproject.toml`의 `requires-python`)
- [uv](https://docs.astral.sh/uv/) — 모든 실행은 `uv run ...`으로 통일한다 (venv activate / `pip install` 금지)
- CUDA GPU 1장 (CPU로도 돌지만 1500 step은 비현실적으로 느리다)
- `pyproject.toml`은 PyTorch cu121 휠 인덱스를 가리킨다. 서버 드라이버가 다르면
  `[[tool.uv.index]]`의 URL을 cu124 등으로 바꿀 것.

## 설치

```bash
git clone <this repo> && cd flowdino
source scripts/env.sh     # UV_CACHE_DIR / HF_HOME을 스크래치로 (홈 쿼터 보호)
uv sync                   # uv.lock 기준 고정 버전 설치
```

`scripts/env.sh`는 매 셸마다 `source` 해야 한다. 계산 노드가 오프라인이면 스크립트 안의
`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` 주석을 해제한다.

## 1. 데이터 준비 (네트워크 되는 노드에서 1회)

```bash
source scripts/env.sh
uv run python scripts/prepare_data.py --device cuda
```

SimCSE의 `wiki1m_for_simcse.txt`를 받아 `data/` 아래에 캐시를 만든다:

| 파일 | 내용 |
|---|---|
| `data/sentences.jsonl` | 필터링된 전체 학습 문장 (10자 미만 제거) |
| `data/rank_eval_sentences.jsonl` | effective rank 평가용 고정 2048 문장 |
| `data/embed_stats.pt` | 토큰 임베딩 통계 `{"mean": mu[D], "std": sigma[D]}` |

`data/`는 `.gitignore` 대상이다 — 저장소에 담지 않고 이 스크립트로 재생성한다.

> **알려진 제약:** 현재 `prepare_data.py`는 백본이 `answerdotai/ModernBERT-base`로 하드코딩돼
> 있어 `data/embed_stats.pt`만 만든다. 반면 저장소의 모든 run config는 `bert-base-uncased`용
> `data/embed_stats_bert_base_uncased.pt`를 참조하므로, BERT 학습을 새로 시작하려면
> `prepare_data.py`의 `BACKBONE`/출력 파일명을 BERT로 바꿔 한 번 더 돌려야 한다.

## 2. 학습

```bash
source scripts/env.sh
uv run python -m src.train --config configs/r7_bert_coviso_optuna_t35.yaml --device cuda
```

주요 인자: `--max-steps`(config 값 override, 스모크용), `--seed`, `--run-name-suffix`.

10분 넘는 학습은 붙잡지 말고 백그라운드로 던진다:

```bash
mkdir -p results/logs   # 리다이렉트 대상 디렉토리는 셸이 만들어주지 않는다
nohup uv run python -m src.train --config configs/<run>.yaml \
  > results/logs/<run>.log 2>&1 &
```

산출물: `results/logs/<run_name>/train.log`, `results/tensorboard/<run_name>/`,
`checkpoints/<run_name>/last.pt`. 전부 `.gitignore` 대상이다.

`configs/base.yaml`(ModernBERT) / `configs/base_bert.yaml`(BERT)은 공통 설정이고
`run_name`이 없어 직접 실행할 수 없다. run config가 `extends:`로 상속해서 쓴다.

## 3. 평가 / 분석

```bash
uv run python -m src.evaluate --which last          # 7-task STS + 길이 probing + rank -> results/summary.md
uv run python scripts/analyze_checkpoints.py --device cuda   # alignment/uniformity 등 -> results/analysis/
uv run python scripts/plot_logs.py                 # 학습 곡선 -> results/plots/
```

`src/evaluate.py`는 모델 형상을 `configs/base.yaml`에서 읽으므로 ModernBERT 체크포인트
기준이다. BERT run은 체크포인트에 저장된 `model_cfg`를 그대로 쓰는
`scripts/analyze_checkpoints.py` / `scripts/rescore_checkpoints.py` 쪽을 쓴다.

## 4. 하이퍼파라미터 탐색

```bash
uv run python scripts/tune.py --base-config configs/r5d_bert_lrsplit_c.yaml \
  --space koleo_lrsplit --n-trials 20 --max-steps 750 --n-jobs 2 --gpus 0,1
```

Optuna study는 `results/analysis/<study_name>.db`(SQLite)에 저장되어 재개 가능하다.

## 테스트

```bash
uv run python -m pytest -q
```

외부 파일이나 네트워크 없이 도는 단위 테스트다. 학습 루프를 건드리기 전에 먼저 통과시킨다.

## run / config 작명 규칙

`<실험단계>_[<백본>]_<메커니즘><값>[_<메커니즘2><값2>...]_s<시드>`

값을 이름에 직접 인코딩한다 (`r6_bert_koleo0.2_lr2e-4`). 실험단계 R번호는 실험 계보를
추적하는 고리이므로 유지한다. ModernBERT는 백본 토큰을 생략하고, BERT는 `bert`를 붙인다.
메커니즘이 3개를 넘어 이름이 길어지면 값 나열 대신 출처를 남긴다
(`r6_bert_koleo_lrsplit_optuna_best_t27`) — 정확한 값은 config yaml과
`results/analysis/<study_name>_trials.csv`에 있다.

## 저장소 구조

```
src/          학습 루프(train) · 증강(augment) · 손실(loss) · 스케줄(schedules) · 평가(evaluate)
configs/      run별 yaml (extends로 base 상속)
scripts/      데이터 준비 · Optuna 탐색 · 체크포인트 분석 · 플롯
tests/        단위 테스트
data/ results/ checkpoints/ wandb/    생성물 (gitignore)
```
