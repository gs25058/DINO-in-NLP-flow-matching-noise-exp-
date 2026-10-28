# 현재 실험의 학습 방법

## 데이터와 예측 대상

`princeton-nlp/datasets-for-simcse/wiki1m_for_simcse.txt`의 영어 문장 중 10자 미만을 제거한다.
학습 캐시는 985,723문장이며 실제 run의 provenance에 파일 해시·행 수를 기록한다.
seed별 Python RNG로 순서를 섞어 epoch를 순회하고, batch 32에 못 미치는 마지막 배치는 버린다.
토큰화는 최대 128토큰이다. 입력 정답은 따로 없고, 깨끗한 EMA teacher의 prototype 확률분포를 예측한다.

## 뷰 생성

BERT word embedding lookup 출력 x의 차원별 고정 μ,σ로 표준화한다.

    x̂ = (x−μ)/σ
    x' = [(1−t)x̂ + tε]σ + μ,  ε ~ N(0,I)
    t_hi(s) = t_start + (t_max−t_start) min(1,s/warmup_steps)

학생 뷰 3개 각각에서 문장당 t 하나를 U(t_lo,t_hi)에서 뽑은 다음 토큰별 ε를 뽑는다.
CLS·SEP·패딩은 원본을 유지한다. teacher는 항상 원본 x를 받는다.
BERT dropout은 0이다. 추론에도 masked mean+L2만 사용한다.

## 모델과 손실

BERT-base-uncased 마지막 hidden state를 attention mask로 mean pooling한다.
이 pooled vector를 L2 정규화한 것이 문장 임베딩 z다. head는 다음과 같다.

    pooled → Linear(768,768) → GELU → Linear(768,256)
           → L2 normalize → weight_norm(Linear(256,8192,bias=False)) → logits

DINO는 teacher 로짓을 이전 center로 보정하고 온도로 sharpen한다.

    p_t = stopgrad(softmax((logits_t−center)/τ_t))
    L_DINO = mean_views mean_batch CE(p_t, softmax(logits_s/τ_s))
    center ← m_c center + (1−m_c) mean_batch(logits_t)

부가 진단 H(p_t), KL(p_t||p_s), H(mean_batch(p_t))도 기록한다.

학생의 z에서 μ와 M2를 구하고, 갱신 전 EMA와 현재의 미분 가능한 통계를 섞는다.

    μ_blend = m μ_ema.detach() + (1−m) mean(z)
    M2_blend = m M2_ema.detach() + (1−m) zᵀz/B
    C = M2_blend − μ_blend μ_blendᵀ
    L_mean = ||μ_blend||²
    L_cov = ||C / max(tr(C),1e−8) − I/D||²_F
    L_total = L_DINO + λ_mean L_mean + λ_cov L_cov

각 뷰 손실을 평균한다. 두 가중치가 같으면 기존의 λ·mean(L_mean+L_cov) 계산 순서까지 유지한다.
초기 EMA는 μ=0, M2=I/D이며, 손실을 구한 **뒤** no_grad로 현재 통계를 반영한다.
한 optimizer step에서 세 뷰의 EMA를 순서대로 세 번 갱신한다. step 56 전에도 통계는 갱신하지만
정규화 기여는 0이다. 과거 EMA나 teacher로 gradient가 흐르지 않는다.
두 가중치는 config의 `cov_iso_mean_lambda`, `cov_iso_cov_lambda`로 명시한다.
실제 R18 대조군에서는 둘 다 정확히 5.874093568592943이다.

## 최적화와 EMA 순서

backbone/head 각각 AdamW 학습률을 쓰고, bias·normalization 파라미터의 weight decay는 0이다.
LR은 warmup+cosine, teacher 온도와 EMA 모멘텀도 지정된 기간에 cosine으로 증가한다.
그룹 순서, gradient clipping, 각 스케줄의 정수 경계는 기존 실제 R18 구현과 동일하다.

    loss와 성분별 gradient 진단 → zero_grad → backward → clip
    → AdamW.step → LR scheduler.step → teacher momentum 설정 → teacher EMA 한 번

FP32 파라미터, TF32, 학생 뷰의 batched forward를 사용한다. AMP는 쓰지 않는다.
매 10 step과 마지막 step의 batch-KL은 Torch RNG를 소비한다. 이 위치·빈도를 보존한다.

## 측정과 선택

250 step마다 및 마지막 step에서 동일한 STS-B dev validation을 측정한다.
mean+L2, head bottleneck+L2, head logits+L2를 같은 backbone pass에서 평가하며,
**모델 선택은 mean+L2 dev Spearman만** 사용한다. 동률이면 먼저 관찰한 checkpoint를 유지한다.

alignment는 STS-B score≥0.8인 쌍의 제곱 거리 평균, uniformity는 dev 전체 문장에서
seed 0으로 뽑은 500쌍에 대한 log mean exp(−2·distance²)이다.
rank는 고정 2,048문장 임베딩을 열 평균 중심화한 뒤 SVD를 구하고,
p_i=σ_i²/Σσ_j², effective rank=exp(−Σp_i log p_i), max SV ratio=p_1로 계산한다.
특정 uniformity나 rank 목표값을 강제하지 않는다.

가중치 적용 후 성분 기여와 소수 진단 step의 backbone gradient norm을 기록한다.
loss 절대값만으로 지배 항을 판단하지 않는다.
추가 시드와 선택이 끝난 뒤에만 best/last의 STS12–16, STSBenchmark, SICK-R test와
동일 가중 7-task 평균을 보고한다. 자세한 탐색 범위·예산·판정 기준은
[OPTUNA_SPEC.md](docs/experiments/coviso_relax/OPTUNA_SPEC.md)에 있다.
