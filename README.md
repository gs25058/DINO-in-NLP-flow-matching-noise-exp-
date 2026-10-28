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

공용 서버의 모든 GPU 실행은 [GPU_POLICY.md](GPU_POLICY.md)를 따른다. 아래 `$GPU`는
점유와 소유자를 확인한 물리 GPU 번호다. 학습·평가·튜닝을 합쳐 계정당 최대 2개다.

```bash
source scripts/env.sh
flowdino_gpu_python "$GPU" scripts/prepare_data.py --device cuda
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
flowdino_gpu_python "$GPU" -m src.train --config configs/r7_bert_coviso_optuna_t35.yaml --device cuda
```

주요 인자: `--max-steps`(config 값 override, 스모크용), `--seed`, `--run-name-suffix`.

10분 넘는 학습은 붙잡지 말고 백그라운드로 던진다:

```bash
mkdir -p results/logs   # 리다이렉트 대상 디렉토리는 셸이 만들어주지 않는다
nohup uv run python scripts/gpu_guard.py --gpu "$GPU" --wait -- \
  "$UV_PROJECT_ENVIRONMENT/bin/python" -m src.train --config configs/<run>.yaml \
  > results/logs/<run>.log 2>&1 &
```

산출물: `results/logs/<run_name>/train.log`, `results/tensorboard/<run_name>/`,
`checkpoints/<run_name>/last.pt`. 전부 `.gitignore` 대상이다.

`configs/base.yaml`(ModernBERT) / `configs/base_bert.yaml`(BERT)은 공통 설정이고
`run_name`이 없어 직접 실행할 수 없다. run config가 `extends:`로 상속해서 쓴다.

## 3. 평가 / 분석

`source scripts/env.sh`는 미리 설정한 `HF_HOME`, `UV_CACHE_DIR`, `UV_PROJECT_ENVIRONMENT`,
`PY`를 보존한다. 기본 환경은 이 checkout의 `.venv`이며 실행기는 그 Python을 재사용한다.
worktree에서는 `UV_PROJECT_ENVIRONMENT`에 공용 `.venv` 경로를 지정하면 환경 중복 생성을 피할 수 있다.

`train.data_order: epoch`를 명시하면 섞은 데이터셋을 중복 없이 순회하고 마지막 불완전 배치는 버린다.
기본 `sample`은 기존 매 step `random.sample`의 난수 순서를 유지한다.

```bash
flowdino_gpu_python "$GPU" scripts/eval_sts7.py --config configs/base_bert.yaml --runs <run>_s42 <run>_s43
flowdino_gpu_python "$GPU" scripts/analyze_checkpoints.py --device cuda
uv run python scripts/plot_logs.py                 # 학습 곡선 -> results/plots/
```

`src/evaluate.py`는 모델 형상을 `configs/base.yaml`에서 읽으므로 ModernBERT 체크포인트
기준이다. BERT run은 체크포인트에 저장된 `model_cfg`를 그대로 쓰는
`scripts/analyze_checkpoints.py` / `scripts/rescore_checkpoints.py` 쪽을 쓴다.

`eval_sts7.py`는 저장된 모델 설정을 우선 사용하며 FlowDINO, 구형/수정된 SimCSE 체크포인트,
`--runs pretrained`를 지원한다. `--pooling auto`가 기본이며 FlowDINO/학습 전 모델은 mean(`last`),
SimCSE는 CLS로 평가한다. 결과 JSON에는 run별 pooling과 baseline 버전을 기록한다.

현재 `train_simcse_baseline.py`는 원격 `1d177db`의 기준선 교정을 반영한다:
불필요한 BERT pooler 제거, MLP 정규분포 초기화, weight decay 0, grad clip 1, 부분 배치 유지.
새 체크포인트는 `baseline_version=simcse_original_aligned_v1`이며 기존 run 이름을 덮어쓰지 않는다.
이전 발표문/10월 dense 재실험은 당시 소스 스냅샷으로 해석한다. gram 실험 기제는 포함하지 않는다.

## 4. 하이퍼파라미터 탐색

```bash
uv run python scripts/tune.py --base-config configs/r5d_bert_lrsplit_c.yaml \
  --space koleo_lrsplit --n-trials 20 --max-steps 750 --n-jobs 2 --gpus 0,1
```

Optuna study는 `results/analysis/<study_name>.db`(SQLite)에 저장되어 재개 가능하다.
`tune.py`는 GPU 공통 검사를 trial마다 실행하며 `--n-jobs`는 최대 2다.
`--save-checkpoints` 또는 감독기의 `SAVE_CKPT=1`로 사후 평가용 체크포인트를 남긴다.
감독기의 `TIMEOUT=<초>`는 trial 실행 시간 제한에 전달된다. 대기 시간은 실행 시간에 포함하지 않는다.

```bash
flowdino_gpu_python "$GPU" scripts/eval_study_sts7.py \
  --study <study_name> --config configs/<base>.yaml --top 10 --device cuda
```

사후 STS7은 증분 캐시와 두 지표의 순위 표를 만든다. pooling/모델 설정/평가 코드/체크포인트가
달라지면 캐시를 재사용하지 않는다. 누락되거나 붕괴한 trial은 평가 표에서 제외하며 체크포인트를 삭제하지 않는다.

붕괴 감시는 기본 off이며, 켜려면 같은 base·예산·서버의 완료 로그에서 기준값을 먼저 잰다:

```bash
uv run python scripts/measure_guard_refs.py --run <base_run> --max-steps <budget>
```

출력된 `collapse_guard` 설정을 새 실험 config에 명시한다. 감시 시작 전에는 검사하지 않고,
entropy/batch-KL은 연속 위반을 요구한다. rank 붕괴도 검사한다. 붕괴로 중단된 trial은 마지막 유효
STS-B 값(없으면 0)으로 기록하고, 일반 오류나 시간 초과는 미완료로 처리한다.

후속 실험의 결론과 출처는 [실험 보고서 목록](docs/experiments/README.md),
이번 반영 내역은 [통합 기록](docs/upstream-integration-20261028.md)에 정리했다.

## 실측 궤적 분석과 절제 설정

`scripts/plot_dense_trajectory.py`는 CSV/JSONL의 실제 측정값을 읽어 모델별 공통 step의
시드 평균·범위, 관측된 최고점과 alignment–uniformity 궤적을 만든다. 입력 경로와 새 출력
폴더를 명시해야 하며 입력 데이터나 기존 결과를 덮어쓰지 않는다.
발표문 수치 대입, 보간, 별도 replay의 자동 병합은 하지 않는다.

```bash
uv run python scripts/plot_dense_trajectory.py \
  --data results/analysis/<experiment>/all_seed_evaluations.csv \
  --out results/analysis/<new-analysis-directory>
```

입력 필드: `model`, `seed`, `optimizer_steps`, `sts_b_dev`, `alignment`, `uniformity`.
시드별 실제 관측 최고점과 입력 SHA-256도 기록한다. 실행이 끝났는지는 입력 실험 기록에서 확인한다.
10월 고밀도 재실험의 완료 여부·설정·초기 replay 검증은 기존 실험 폴더에 보존돼 있다.

`configs/r12_bert_noise_optuna_t35_coviso0.yaml`은 R12 기준선에서 `cov_iso_lambda`만 0으로
바꾼 통제 절제 설정이다. 공분산 정규화 제거 효과를 재검증할 때 재사용할 수 있다.
다른 하이퍼파라미터를 재튜닝한 비교나 4000-step R18 비교와는 구분한다.
일회성 발표 그림·고정 실행기 정리 내역은 [로컬 변경 정리](docs/local-cleanup-20261028.md)에 있다.

## 테스트

새 공분산 성분 완화 실험은 [실행 안내](docs/experiments/coviso_relax/README.md)와
[사전 등록 명세](docs/experiments/coviso_relax/SPEC.md)를 따른다.
기존 config의 `cov_iso_lambda`는 보존하며, mean/cov 개별 가중치는 명시한 경우에만 우선한다.

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
