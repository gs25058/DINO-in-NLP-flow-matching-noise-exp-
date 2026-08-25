# FlowDINO-Text: 소형 검증 프로젝트

## 목적 (3줄)
- DINO 자기증류 구조(EMA teacher + centering/sharpening)는 유지하고, 증강만 flow matching식 가우시안 노이즈 보간으로 교체했을 때 문장 임베딩 품질을 검증한다.
- 기존 dropout 증강 실험은 이미 완료되었으므로 **재실험하지 않는다**. 비교 기준은 기존 결과 수치(STS-B Spearman 70.6, 250 step 이후 하락, effective rank 감소)를 외부 baseline으로 사용한다.
- 방법의 상세 스펙과 검증할 예측은 `METHOD.md`가 단일 진실 소스(single source of truth)다. 구현이 METHOD.md와 충돌하면 METHOD.md를 따르고, METHOD.md를 바꿔야 한다면 먼저 사용자에게 확인한다.

## 서버 제약 (중요)
- 공용 서버, **sudo 없음**. 시스템 패키지 설치 금지. 모든 것은 유저 공간에서 해결한다.
- 모든 파이썬 실행은 `uv run python ...`으로 통일한다. venv activate, `pip install` 직접 호출 금지.
- 캐시는 홈 쿼터 보호를 위해 스크래치로: 실행 전 `source scripts/env.sh` (UV_CACHE_DIR, HF_HOME 설정).
- 장시간 학습(>10분)은 직접 붙잡지 말고 `nohup uv run python -m src.train --config ... > results/logs/RUN.log 2>&1 &`로 던지고 로그를 tail로 확인한다. Slurm이 있으면 `sbatch scripts/slurm_train.sh`.
- 계산 노드가 오프라인일 수 있음: 모델/데이터 다운로드는 로그인 노드에서 `uv run python scripts/prepare_data.py`로 선행하고, 학습 시 `HF_HUB_OFFLINE=1`.

## 구현 원칙
1. **조건 간 코드 경로 차이 금지**: 실험 조건(run)들은 오직 config yaml 값만 다르다. if-분기로 조건별 다른 학습 루프를 만들지 않는다.
2. **증강은 `src/augment.py`의 인터페이스 뒤에만 존재**한다. 학습 루프는 증강의 내용을 모른다.
3. 노이즈는 HF 모델의 `inputs_embeds` 인자로 주입한다. backbone forward 코드를 수정하지 않는다.
4. loss 함수는 `(loss, aux_dict)`를 반환하고 aux_dict에 H(p_t), KL(p_t||p_s), (사용 시) L_vel을 항상 포함한다. METHOD.md의 예측 P1–P4 검증이 이 로깅에 의존한다.
5. 학습 루프를 짜기 **전에** `tests/`의 단위 테스트를 먼저 작성·통과시킨다 (아래 체크리스트).

## 작업 순서 (반드시 이 순서)
1. `uv sync` 후 `scripts/prepare_data.py`로 FineWeb 문장 추출 + 임베딩 통계(μ, σ) 사전 계산 → `data/` 캐시
2. `tests/test_augment.py` 작성·통과:
   - t=0에서 FlowNoiseAug가 항등(원본과 allclose)인지
   - t=1에서 출력의 차원별 분산이 표준화 공간에서 ≈1인지
   - 패딩/특수 토큰 위치에 노이즈가 들어가지 않는지
   - curriculum 스케줄이 step에 따라 단조 증가하고 상한을 넘지 않는지
3. `tests/test_loss.py`: teacher==student 로짓일 때 KL≈0, H가 온도에 따라 예상대로 변하는지
4. run R1(anchor, t 고정, 최소 구성)을 500 step 스모크 테스트 → loss 개형이 기존 dropout 실험(초반 증가 후 감소)과 유사한지 확인
5. 매트릭스 R1→R2→R3→R4 순서로 실행 (METHOD.md §6). 각 run은 시드 2개 이상.
6. `src/evaluate.py`로 STS-B(mteb) + 길이 probing + rank 지표를 결과 테이블로 집계 → `results/summary.md`

## 판정 기준 (미리 고정)
- 성공: 어떤 노이즈 설정이 STS 피크 ≥ 70 (기존 dropout 피크와 동등) **이면서** 피크 후 500 step 동안 하락 폭이 기존보다 작음
- 부분 성공: STS는 낮아도 P1–P3 예측(METHOD.md)이 로그에서 확인됨 → 분석 결과로 가치 있음
- 실패 시 디버깅 순서: (1) t 스케일/표준화 버그 → (2) 조건부 붕괴 지표(batch KL) 확인 → (3) 그 후에만 "가설 기각" 논의

## 하지 말 것
- dropout 증강 재구현/재실험
- 기존 하이퍼파라미터(Table 1: logit dim 8192, teacher momentum 0.997, τ_t=0.082, τ_s=0.151, center momentum 0.92, AdamW lr 5.4e-4 등)를 근거 없이 변경 — 노이즈 관련 신규 하이퍼파라미터만 탐색 대상
- 결과 요약에서 시드 1개 수치로 결론 내리기
