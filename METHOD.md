# FlowDINO-Text 방법 스펙 v0.1

이 문서가 구현의 단일 진실 소스다. 기호는 중간보고서 식 (6)–(12)를 따른다.

## 1. 개요

DINO-NLP 구조(EMA teacher, DINO head, centering/sharpening, CE loss)는 기존 실험과 동일하게 유지한다.
바뀌는 것은 **뷰 생성**뿐이다: dropout 대신, 토큰 임베딩 공간에서 flow matching식 선형 보간으로
노이즈 레벨 t가 다른 뷰를 만들어 teacher(깨끗한 쪽)와 student(노이즈 쪽)에 준다.

**추가 결정 (노이즈 격리)**: 노이즈 run에서는 backbone의 모든 dropout(attention/hidden/embedding)을
teacher·student 공통으로 0으로 설정한다(config: `hidden_dropout: 0.0` 등). 기존 dropout 실험의
Table 1은 backbone dropout=0.103이었지만, 이 값이 남아 있으면 두 뷰의 차이가 노이즈 단독 효과인지
dropout과 섞인 효과인지 격리되지 않으므로 노이즈 실험에서는 0으로 고정한다.

**출처 메모**: 아래 §4.1의 DINO head 구조/centering 공식과 §5의 effective rank 공식은 기존
dropout 실험의 실제 완료 결과가 기록된 중간보고서(`referecne_files/`)의 식 (8)–(12), Table 1,
Fig 9–10을 근거로 한다. `referecne_files/Unsupervised-Context-Embedding` git 저장소에 커밋된
코드는 이 보고서 결과를 낸 코드가 아니라(전체 브랜치/커밋에 DINOHead/logit_dim/teacher_temp 등
문구가 전혀 없음을 확인함) 이후 문맥 임베딩 방향으로 발전한 별개 실험(Matryoshka cosine/KL loss,
Muon+AdamW, projection_dim=1024)이므로 이식 대상에서 제외한다. 저장소에서 그대로 가져오는 것은
mean pooling 구현(§4.1)과 EMA 파라미터 업데이트 메커니즘(매 step 갱신이라는 형태)뿐이다.

핵심 논리 (이론 분석 요약):
- dropout 증강에서는 두 뷰가 사실상 동일 → 학습 신호가 온도 차이 항에서만 나옴 → 자기 샤프닝 폭주
  (loss 하락 ≈ H(p_t) 하락) → 과샤프닝으로 effective rank 감소, STS 하락.
- 노이즈 증강은 뷰 간 실질적 정보 차이를 t로 만들 수 있음 → KL(p_t||p_s)에 t에 비례하는 소거 불가능한
  바닥이 생김 → 샤프닝 폭주가 억제되고, 클러스터 경계가 저밀도 분리 원리를 따름.

## 데이터

기존 dropout 실험은 FineWeb sample-10BT를 pysbd로 문장 분리해서 썼지만, 이번 노이즈 실험은
**SimCSE 공식 학습 데이터로 변경**한다: HF `princeton-nlp/datasets-for-simcse`의
`wiki1m_for_simcse.txt` (영어 위키 100만 문장, 이미 문장 단위라 pysbd 불필요).
필터링은 10자 미만 문장 제거 + 토큰화 후 128토큰 초과 절사만 유지(기존과 동일 기준, pysbd 단계만
제거). §2.1의 μ,σ 통계와 §5의 effective rank 평가용 2048문장도 이 wiki1m 데이터에서 뽑는다
(전자는 학습 문장 10k 샘플, 후자는 이와 겹치지 않는 고정 2048문장).

## 2. 노이즈 뷰 생성 (FlowNoiseAug)

### 2.1 표준화 공간
백본은 `answerdotai/ModernBERT-base`(기존 dropout 실험과 동일, HF 체크포인트). word embedding
접근은 모듈 경로를 하드코딩하지 않고 `model.get_input_embeddings()`(HF 공식 API)로 얻는다.
노이즈 주입 후에는 `model(inputs_embeds=x', attention_mask=...)`로 forward한다 — backbone forward
코드 수정 없음.

사전에 학습 문장 N개(기본 10k, §"데이터" 절의 wiki1m 소스에서 샘플)에서 backbone 토큰 임베딩
(위치 임베딩 합산 전, word embedding lookup 출력)의 차원별 평균 μ ∈ R^D, 표준편차 σ ∈ R^D를
계산해 캐시한다.

    x̂ = (x − μ) / σ          # 표준화
    x_t = (1 − t)·x̂ + t·ε,  ε ~ N(0, I_D)   # flow matching 보간 (토큰별 독립 ε)
    x'  = x_t · σ + μ         # 역표준화 후 inputs_embeds로 주입

이렇게 하면 t의 의미가 보편화된다: t=0은 원본, t=1은 "임베딩 분포와 스케일이 맞는 순수 노이즈".
노이즈 스케일 하이퍼파라미터가 t 하나로 통합된다.

### 2.2 적용 범위
- t는 **문장(뷰)당 하나** 샘플링. 토큰별 t는 사용하지 않는다(뷰 = 하나의 노이즈 레벨이라는 해석 유지).
- ε은 토큰별 독립.
- [CLS], [SEP], 패딩 토큰에는 노이즈를 주지 않는다. `attention_mask`는 패딩만 구분하므로,
  `tokenizer(..., return_special_tokens_mask=True)`의 `special_tokens_mask`로 CLS/SEP를 함께
  식별해 노이즈에서 제외한다.

### 2.3 시간 조건화 (선택, 기본 OFF)
플래그 `time_embed: true`일 때, sinusoidal 시간 임베딩 τ(t) ∈ R^D를 모든 (비특수) 토큰 임베딩에 더한다.
추론(평가) 시에는 t=0이므로 τ(0)을 더한다. 기본은 OFF — 임베딩 모델로서 단순성을 유지하고,
R4 이후 ablation으로만 확인한다. **이번 R1–R4 매트릭스(§6) 범위 밖**이다: 플래그와 구현은 갖추되
매트릭스 실행에는 사용하지 않는다.

## 3. teacher / student 시간 배정

### 모드 A (anchor) — 기본
- teacher: t_T = 0 (항상 깨끗한 뷰)
- student: K개의 뷰, t_k ~ U(t_lo, t_hi(step))

### 모드 C (consistency)
- student: t_S ~ U(t_lo, t_hi(step))
- teacher: t_T = max(0, t_S − Δt) — 같은 ε을 공유한 인접 시간쌍 (consistency model의 인접 timestep 대응)
- Δt 기본값 0.1. 같은 ε 공유가 중요: 노이즈 실현이 다르면 인접성이 깨진다.
- K개 student 뷰 중 **뷰 0**과만 Δt로 인접 페어링한다(teacher_embeds는 설계상 단일 뷰). 나머지
  K−1개 student 뷰는 consistency 페어링 없이 multi-crop적 추가 정보 스케일로만 기여한다 (승인됨).

모드 A를 먼저 검증하고(R1–R2), 모드 C는 R3에서. A가 전혀 안 되는데 C만 되는 경우는 기대하지 않으므로
A 결과가 R3 진행 여부의 게이트다.

### curriculum (t_hi 스케줄)
consistency model의 간격 스케줄에서 착안:

    t_hi(step) = t_start + (t_max − t_start) · min(1, step / warmup_steps)

기본값: t_start = 0.05, t_max = 0.5, warmup_steps = 전체의 40%. t_lo = 0.02 고정.
이 스케줄이 "초기에 두 추정의 차가 너무 큼" 문제의 해법이다.

### multi-crop 대응
student 뷰 수 K = 3 (기존 N_local=3과 동일 예산). 각 뷰가 독립적으로 t를 샘플링하므로
여러 노이즈 레벨 = 여러 정보 스케일이 되어 multi-crop의 역할을 대신한다.

## 4. loss

### 4.1 주 loss: DINO CE (기존과 동일)

**Final Embedding (평가/rank 지표용)**: backbone 마지막 hidden state를 masked mean pooling한 뒤
L2 정규화. 기존 저장소 코드 그대로 이식:
`sum(h * mask.unsqueeze(-1).float(), dim=1) / clamp(mask.float().sum(dim=1), min=1e-9)`.

**DINO-NLP head (로짓, loss 전용 — Final Embedding과 별개)**: 기존 실험 보고서 식 (8)–(9), Fig 4
구조(BERT → mean pooling → MLP bottleneck → weight-norm Expanding → logits)를 따르되, 정확한
hidden/bottleneck 차원은 기존 문서 어디에도 남아있지 않아(git 저장소에도 없음) 이번 프로젝트
규모에 맞게 가볍게 새로 정한다: `Linear(backbone_hidden→backbone_hidden) → GELU →
Linear(backbone_hidden→256)` (bottleneck_dim=256) → L2 정규화 → `weight_norm(Linear(256→8192,
bias=False))` (logit_dim=8192, Table 1). 이 hidden/bottleneck 차원 2개는 근거 문서가 없는
신규 구조 결정이며, 노이즈 관련 하이퍼파라미터 탐색 대상이 아니라 고정값으로 취급한다.

**Centering & sharpening (식 6, 7, 10, 11)**: centering은 teacher 로짓에만 적용, EMA로 갱신되는
중심벡터 c (모멘텀 0.920)를 로짓에서 뺀 뒤 온도로 나눠 softmax. student는 centering 없이 온도만
다르게 적용:

    c ← m·c + (1−m)·mean_batch(ℓ_t),  m = center_momentum = 0.920
    p_t = softmax((ℓ_t − c) / τ_t),  τ_t = 0.082
    p_s = softmax(ℓ_s / τ_s),        τ_s = 0.151

**centering 대안 - uniform_push (centering-uniform-push 브랜치, `loss.centering: uniform_push`)**:
위 EMA 업데이트에 더해, marginal usage entropy H(p̄_t)를 높이는(=prototype 사용을 균등화하는)
방향의 gradient를 c에 한 스텝 추가로 더한다:

    c ← c − η · ∇_c [ Σ_k p̄_t,k log p̄_t,k ]   (η = uniform_push_lr, autograd로 계산)

η=0이면 순수 EMA와 완전히 동일(회귀 테스트로 확인). Sinkhorn-Knopp centering(DINOv2 방식)도
검토했으나, batch_size=32에 logit_dim=8192라 배치당 prototype 균형화가 사실상 무의미해 보류함
(SwAV류 큐 확장이 필요 - 별도 브랜치 검토 대상). uniform_push는 배치 크기에 안전하게 동작하는
경량 대안.

**EMA teacher**: teacher는 학생 backbone+head 파라미터의 EMA(θ_t ← λθ_t + (1−λ)θ_s, λ=teacher
momentum=0.997), 매 optimizer step마다(grad accumulation 완료 후) 갱신. stop-gradient 적용.
갱신 메커니즘 자체(매 step 파라미터 EMA)는 기존 저장소의 `EMACallback` 패턴을 이식하되, 저장소의
std/RMS 정규화 확장은 보고서 식과 다르므로 포함하지 않는다 — 위 centering 식만 사용한다.

K개 student 뷰에 대한 CE 평균. 로깅을 위해 H(p_t)와 KL(p_t||p_s)를 분리 반환한다:
L = H(p_t) + KL(p_t||p_s).

### 4.2 보조 loss: velocity 예측 (선택, R4)
student 토큰 출력 h_i에서 2층 MLP head로 flow matching velocity를 예측:

    v̂_i = MLP(h_i),   target v_i = ε_i − x̂_i     (표준화 공간)
    L_vel = mean_{i: 비특수, 비패딩} ||v̂_i − v_i||² / D
    L_total = L_DINO + λ·L_vel,   λ = 0.05

의도: "입력 정보를 버리면 풀 수 없는 과제"를 걸어 rank 붕괴 압력을 낮추는 정보 보존 정규화.
velocity head는 평가 시 버린다. λ는 {0.02, 0.05, 0.1}만 탐색.

## 5. 진단 지표 (모든 run에서 로깅)

| 지표 | 목적 |
|---|---|
| H(p_t) | 자기 샤프닝 속도. dropout 실험 대비 감소가 느려야 함 (예측 P1) |
| H(p̄_t) | marginal usage entropy: p̄_t = mean_batch(p_t)의 엔트로피. H(p_t)(샘플별 평균)와 별개로, 8192개 prototype을 얼마나 고르게 쓰는지 보는 붕괴 진단. 상한 log(8192)≈9.01, 0에 가까우면 소수 prototype만 사용(=붕괴) (R5) |
| KL(p_t‖p_s) | t에 비례하는 양의 바닥에 머물러야 함 (예측 P2) |
| batch-KL | **조건부 붕괴 경보**. 배치 내 모든 student 뷰 출력(N=B·K개) 중 무작위 256쌍을 매 로깅 시점마다 샘플링해 대칭화 KL의 평균: `mean[(KL(p_i‖p_j) + KL(p_j‖p_i)) / 2]`. →0이면 student가 입력을 무시하고 marginal만 출력 중. t가 너무 크다는 신호. centering으로는 안 잡히므로 별도 감시 필수. 정밀도보다 추세가 중요하므로 전수(O(N²)) 계산은 하지 않는다 |
| Effective Rank, Max SV Ratio | 고정 평가 문장 2048개(§"데이터"의 wiki1m 소스에서 샘플, 학습 문장과 분리)의 pooled 임베딩 X를 열 평균 센터링 후 SVD. `p_i = σ_i² / Σ_j σ_j²`, `Effective Rank = exp(−Σ_i p_i log p_i)`, `Max SV Ratio = p_1`. 기존 저장소·git 히스토리 어디에도 이 계산 코드가 없음을 확인함(신규 구현) — 보고서 Fig 9–10의 지표명("SV Ratio", "Effective Rank")과 방향성만 참고. 후반 rank 감소가 완화되는지 (예측 P3) |
| STS-B Spearman (mteb, 250 step마다) | 학습 중 빠른 추적용 주 성능 지표. STS-B dev split |
| Alignment, Uniformity (매 평가 시, 상시) | Wang & Isola(2020)/SimCSE 논문 Fig.2·각주3과 동일 정의: ppos=STS-B dev에서 score≥4.0(정규화 스케일 0.8) 쌍, pdata=STS-B dev 전체 문장. `alignment = E‖f(x)−f(y)‖²`, `uniformity = log E[exp(−2‖f(x)−f(y)‖²)]` (둘 다 낮을수록 좋음). `scripts/plot_logs.py`가 run마다 SimCSE Fig.2 스타일 궤적 그래프(`align_uniform_<run>.png`)와 전체 run 비교(`align_uniform_comparison.png`)를 자동 생성. 참고: SimCSE-BERT-base(unsup) 논문 보고치는 uniformity≈−2.6 — 이 프로젝트 최고 run(r5d)도 −0.77 수준으로 한 자릿수 이상 못 미침 (analysis_v2.md Q1) |
| 길이 probing R² (평가 시) | STS-B 문장 임베딩 → 토큰 길이 ridge 회귀. dropout 실험보다 낮아야 함 (예측 P4, ESimCSE 비판 해소) |
| 최종 7-task 평균 Spearman (평가 시) | SimCSE 논문과 동일 태스크셋: STS12, STS13, STS14, STS15, STS16, STSBenchmark, SICK-R (mteb 태스크명 확인됨, 그대로 사용). 각 run의 best/last checkpoint에서 1회 |

## 6. 실험 매트릭스

| run | 모드 | t 설정 | K | L_vel | 목적 |
|---|---|---|---|---|---|
| R1a/b/c | A | 고정 t ∈ {0.1, 0.25, 0.5}, curriculum 없음 | 1 | ✗ | t 민감도, 유효 구간 파악 |
| R2 | A | curriculum (§3) | 3 | ✗ | 본 제안 기본형 |
| R3 | C | curriculum, Δt=0.1 | 3 | ✗ | consistency 모드 검증 |
| R4 | A | R2와 동일 | 3 | ✓ λ=0.05 | 정보 보존 정규화 효과 |

- 공통: 기존 Table 1 하이퍼파라미터 고정, 1500 step (R2/R4는 3000 step 연장판 1개 추가 — 후반 하락 관찰용), 시드 ≥ 2.
- 실행 순서 게이트: R1에서 어떤 t도 STS 60을 못 넘으면 구현 버그부터 의심 (CLAUDE.md 디버깅 순서).

## 7. 예측 (사전 등록)

- P1: R2의 H(p_t) 감소 속도 < 기존 dropout 실험의 H(p_t) 감소 속도 (기존 로그와 비교)
- P2: R1에서 수렴 후 KL(p_t‖p_s)의 평균값이 t에 대해 단조 증가
- P3: R2/R4의 피크 후 effective rank 감소 기울기 < 기존 실험, R4 < R2
- P4: R2의 길이 probing R² < 기존 dropout 실험의 R²
- P5 (위험 예측): t 고정 0.5(R1c)에서 batch-KL이 낮게 유지되면 조건부 붕괴 → STS 낮음

P1–P4 중 3개 이상 성립하면 성능과 무관하게 이론 분석이 데이터로 뒷받침된 것.

## 8. 알려진 위험과 대응

- t 너무 작음 → dropout과 동일 병리로 회귀 (H 폭주). 대응: R1a가 이를 보여주는 대조군 역할.
- t 너무 큼 → 조건부 붕괴. 대응: batch-KL 감시 + curriculum 상한.
- 표준화 통계가 부정확 → t의 의미 왜곡. 대응: 10k 문장으로 추정, 테스트에서 t=1 분산≈1 검증.
- head/임베딩 공간 불일치 문제는 이 실험이 해결하지 않는다. rank 지표가 여전히 나쁘면
  다음 단계에서 KoLeo/Gram anchoring을 별도 축으로 도입한다 (이번 범위 아님).

## 9. 정규화 항의 정체성 기준 (pairwise vs 분포 통계량)

이 프로젝트에서 임베딩 기하(anisotropy/rank 붕괴)를 완화하는 정규화 항을 추가로 도입할 때 지키는
기준선(centering-uniform-push 브랜치, R6/R7 계열에서 명문화):

- **금지**: 표본이 표본을 직접 미는 항(쌍별/pairwise 반발). 배치 내 개별 샘플 쌍 사이의 거리나
  유사도를 직접 밀어내거나 당기는 항 — 예: KoLeo(§4.1의 loss와 별개로 도입된 최근접 이웃 반발,
  `src/loss.py::koleo_loss`)는 이 범주라 **금지 대상**. 표본 개수(O(B²) 또는 O(B·K)) 쌍에 대해
  개별적으로 gradient가 걸리는 형태는 전부 여기 속한다.
- **허용**: 분포 통계량(배치 평균, 배치 공분산 등)에 대한 제약. 개별 표본 쌍이 아니라 배치 전체의
  요약 통계량 하나에 페널티가 걸리고, 그 gradient가 각 표본으로 균등하게(직접적인 쌍별 상호작용
  없이) 흘러들어가는 형태 — 기존 EMA centering(§4.1)과 uniform_push(§4.1)가 이미 이 범주이고,
  R7의 `CovIsoPenalty`(EMA 공분산을 등방으로 미는 페널티, `src/loss.py::CovIsoPenalty`)도 동일
  범주로 허용.

**근거**: 쌍별 반발 항은 배치 구성에 따라 개별 표본이 서로를 직접 밀어내므로 표본 수준 재배열
효과가 커 SimCSE류 InfoNCE와 메커니즘이 유사해진다 — 이 프로젝트가 검증하려는 "노이즈 증강만으로
얻는 이득"과 "표본 간 직접 상호작용으로 얻는 이득"을 섞이게 만든다. 분포 통계량 제약은 (uniform_push
가 이미 그렇듯) 배치 전체의 형태만 규제하고 표본 간 직접 상호작용이 없어, 노이즈 증강 효과와
혼동 없이 별도 축으로 다룰 수 있다.
