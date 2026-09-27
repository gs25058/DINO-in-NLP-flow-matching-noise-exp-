# R19 — SimCSE InfoNCE의 유사도 행렬을 gram 변환(exp → Sinkhorn-Knopp → SVD top-k 제거)으로 교체

작성 시점: 2026-09-20. 예측은 1-epoch 매트릭스 **실행 전**, 300-step 스윕 결과를 보고 등록했다.

## 1. 기제 (정확한 정의)

SimCSE 비지도 학습의 손실은 두 dropout 뷰 z1, z2에 대한 InfoNCE다.
원본은 cos 유사도 행렬 G(=gram, [B,B], 대각이 positive 쌍)를 온도로 나눠 바로 CE에 넣는다.
R19는 G 자리에 다음 변환을 거친 행렬을 넣는다 — **학습 목적함수만 바뀌고 평가 경로는 그대로다.**

```
G   = cos(z1_i, z2_j)                      # [B,B], 대각이 positive
P   = Sinkhorn-Knopp(exp(G))               # 이중확률화 (로그 영역 3 iter)
P_k = sum_{i<k} sigma_i u_i v_i^T          # SVD 상위 k개 특이성분
S   = (P - P_k) * scale                    # 이 값을 "cos 유사도"로 간주
L   = CrossEntropy(S / 0.05, diag)         # 원본과 같은 InfoNCE, 같은 온도
```

- k = 5 (top-5 특이값 제거).
- 상위 성분 제거는 좌특이벡터 U_k(no_grad) 투영을 빼는 방식이다. 값은 SVD 절단과 동일하고
  특이값 축퇴에서의 backward 불안정만 피한다(koleo가 최근접 이웃 index를 no_grad로 고르는 것과 같은 규약).
- 구현: `src/loss.py::gram_sinkhorn_topk`, `scripts/train_simcse_baseline.py::info_nce`.
  `--gram-topk 0`(기본)이면 원본 SimCSE와 완전히 같은 경로다.

## 2. 원본 저장소 대조 수정 (princeton-nlp/SimCSE @ 49fa580)

같은 세션에서 원본을 클론해 대조하고, 우리 기준선이 원본과 갈라지던 지점을 고쳤다.

| 항목 | 우리(수정 전) | 원본 | 조치 |
|---|---|---|---|
| weight decay | torch AdamW 기본 0.01 | HF TrainingArguments 기본 0.0 | 0.0으로 (`--weight-decay`) |
| grad clip | `max_norm=1e10`(사실상 없음) | HF 기본 max_grad_norm 1.0 | 1.0으로 (`--max-grad-norm`) |
| BERT 자체 pooler | 포함(옵티마이저에도 포함) | `BertModel(config, add_pooling_layer=False)` | 제거 |
| MLP pooler 초기화 | PyTorch 기본 kaiming uniform | `init_weights` → N(0, 0.02) | N(0, 0.02), bias 0 |
| 1 epoch step 수 | `len//batch` (마지막 부분 배치 버림) | HF DataLoader `drop_last=False` | ceil + 부분 배치 유지 |

확인 결과 **차이가 아니었던** 것들(수정 불필요):
- best 체크포인트 선정 주기: 원본은 `load_best_model_at_end`일 때 매 eval(125 step)마다 저장 판정
  (transformers 4.2.1 `DefaultFlowCallback`에서 확인) — 우리와 동일.
- cos 유사도의 fp16: `cosine_similarity`는 autocast fp32 목록이라 원본도 fp32로 계산된다.
- 2뷰 forward를 한 번에 묶는지 여부: dropout 마스크가 표본별 독립이라 분포상 동일.
- token_type_ids: 단문이라 양쪽 모두 0.

의도적으로 유지한 차이: **학습 데이터**. 원본은 wiki1m 100만 줄 전부, 우리는 10자 미만을 거른
985,723문장(`data/sentences.jsonl`)이다. 우리 DINO run과 데이터를 맞추는 것이 이 기준선의 목적이다.

## 3. 스케일 스윕 (300 step, 1시드, LR 스케줄 압축) — 예측의 근거

`scale`은 변환 행렬에 곱하는 상수로, 온도와 같은 축이다(원본 cos는 [-1,1]인데 이중확률행렬은 ~1/B).

| arm | STS-B dev | alignment | uniformity | eff_rank | loss(끝) |
|---|---|---|---|---|---|
| ctrl (원본 SimCSE) | 0.6077 | 0.3095 | -2.494 | 273.7 | 0.0001 |
| gram5 scale=1 | 0.6946 | 0.1604 | -2.409 | 213.0 | 3.78 |
| gram5 scale=8 | **0.7069** | 0.1617 | -2.439 | 221.1 | 1.36 |
| gram5 scale=64(=B) | 0.5737 | 0.2862 | -2.356 | 257.7 | 0.0000 |
| gram5 scale=512 | 0.4468 | 0.1627 | -1.157 | 145.1 | 0.0000 |

읽기: scale이 커지면 **Sinkhorn이 이미 푼 매칭을 그대로 읽는 꼴**이 되어 loss가 0으로 포화하고
성능이 무너진다. 신호가 남는 구간(scale 1~8)에서만 대조군을 이겼다. 기본값은 scale=1(문자 그대로).

## 4. 사전 등록 예측 (1 epoch = 15,402 step, 시드 42/43)

- **P-19-1**: gram arm(scale 1)의 STS-B dev 피크가 대조군보다 높다.
  근거는 §3이지만 300 step은 대조군이 아직 궤적 초반이라(1 epoch에서 대조군은 0.80대로 간다)
  **뒤집힐 가능성이 크다**고 본다. 기각되면 "짧은 예산에서의 이득"으로 기록한다.
- **P-19-2**: gram arm의 alignment가 대조군보다 **낮다(좋다)**. 300 step에서 0.16 vs 0.31로
  격차가 크고, 이 축은 우리 프로젝트의 남은 격차(C-1)라 방향이 유지되면 그 자체로 정보다.
- **P-19-3**: gram arm의 eff_rank는 대조군보다 낮다(213~221 vs 274). 즉 이득이 있다면
  uniformity/rank를 키워서가 아니라 **정렬 쪽**에서 온다.

판정은 2시드 평균으로 하고, 7-task(test, cls pooling)는 best 체크포인트로 따로 잰다(A-5).

## 5. 결과

(실행 중 — 채울 것)
