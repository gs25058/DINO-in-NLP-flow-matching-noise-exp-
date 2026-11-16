# 어휘 prototype DINO (mean pooling 고정) — 사전 등록

2026-11-16 학습 시작 전 등록. 브랜치 `exp/vocab-prototype-dino`(출발 be730f7).
이 파일은 학습 run 이전에 커밋하며, test 결과를 본 뒤 설정·범위·판정 규칙을 바꾸지 않는다.

## 목적

DINO head의 무작위 prototype 8192개를 사전학습 MLM head(30,522 단어)로 바꾸면 768d STS가 오르는가.
오른다면 원인이 "의미 있는 prototype 방향"인지 "MLM head에 학습된 지식"인지 가른다.
pooling은 모든 조건에서 masked mean + L2다. [MASK] 프롬프트 pooling, bag-of-words anchor,
토큰 단위 soft MLM은 범위 밖이며 구현하지 않는다.

## 조건

- **C0** 기준선: coviso joint trial 13(t13), `configs/coviso_relax/selected_joint.yaml`. 재학습하지 않는다.
  seed 42/43/44 checkpoint와 dev 값(73.716/73.518/74.472)을 재사용한다. 기록된 STS7 raw best는 67.10이다.
- **C1** 단어 head(학습): pooled 768d → MLM transform(dense+GELU+LN) → decoder(30522)+bias.
  transform·decoder·bias 모두 사전학습 값의 복사본이다. decoder는 입력 단어 임베딩과 tie를 끊은 별도 저장소다.
- **C2** 단어 head(동결): C1과 같고 head 전체를 동결한다(optimizer group 없음, teacher head도 불변). backbone만 학습한다.
- **C3** k-means 대조군: 입력 단어 임베딩 행렬(= tie된 decoder와 같은 공간) 30,522행을 Euclidean k-means
  (k=8192, k-means++ 초기화, seed 0)로 군집화한 뒤 중심을 L2 정규화해 decoder를 초기화한다(bias 0).
  transform은 C1과 같은 구조지만 BERT식 무작위 초기화(normal 0.02, bias 0, LN 1/0)이며 사전학습 weight는 쓰지 않는다.
  캐시 `data/vocab_kmeans_bert_base_uncased_k8192_seed0.pt`: 39회에서 수렴, 빈 군집 0개, 크기 1 군집 4,708개,
  최대 크기 995개다. sha256은 `.json` 메타에 기록한다.
- 768d embedding은 head 이전 pooled 벡터이므로 head 교체와 무관하게 정의가 같다. cov-iso는 기존처럼 768d embedding에 적용한다.
  t13에는 KoLeo가 없으므로 어떤 조건에도 KoLeo를 넣지 않는다.
- DINO center 차원은 K(30522 또는 8192)다. 진단용 bottleneck은 L2 정규화한 transform 출력(768d)이다.

## 구현 규칙과 사전 검증 (학습 전 완료)

- 새 키 `model.head.head_type: {random, vocab, vocab_kmeans}`, `model.head.head_freeze`, `model.head.kmeans_path`.
  키가 없으면 기존 `DINOHead` 경로를 그대로 탄다.
- **bit-identical 검증:** t13 config, seed 42, 60 step에서 legacy(be730f7)와 이 브랜치를 실행했다.
  components 122개 값, 평가 32개 값, 초기 hash, 첫 노이즈 뷰, 데이터 순서가 정확히 같다(최대 차이 0).
  provenance에는 backbone 전용 hash 키(`initial_backbone_sha256`) 하나만 추가된다.
- **head 생성과 RNG:** vocab head는 분리된(forked) CPU RNG 아래에서 만든다. C1–C3는 같은 seed·노이즈 설정에서 같은
  노이즈 스트림을 쓴다(감사로 확인). C0과 비트 단위로 같은 노이즈는 아니다. C0는 재학습하지 않으므로 비교는 seed 짝으로 한다.
- **입력 임베딩:** 모든 조건(C0 포함)에서 backbone의 입력 단어 임베딩은 backbone optimizer로 학습된다.
  따라서 "입력 임베딩 변화량 0"은 "head 갱신이 입력 임베딩에 닿지 않는다"로 검증한다. decoder 저장소 비공유 assert,
  head만의 gradient step 뒤 입력 임베딩이 불변인지 확인하는 테스트, 학습 중 `decoder_shares_input_storage`(항상 0)
  기록으로 확인한다. 입력 임베딩의 실제 이동량(`input_embedding_drift`)도 진단으로 기록하고, C0 t13 checkpoint의 값과 비교한다.
  노이즈 표준화 통계(μ,σ)는 사전 계산된 고정 파일이며 head와 무관하다.
- **param group:** C1 head lr = backbone lr × `_head_lr_mult`(backbone lr 계열), C3 head lr = 기존 무작위 head lr 범위.
  C2는 head group이 없다. LN/bias는 기존 규칙대로 weight decay 0이다.
- **grad clip:** 요청 문서의 3.0 고정은 "C0과 head만 다르다"는 원칙과 충돌한다(t13 = 5.337). 확장 탐색 지시에 따라
  grad clip은 C0의 탐색 축과 같은 범위 [1,6]에서 탐색한다. anchor 1은 t13 값, anchor 2는 3.0을 쓴다.
- 학습 루프·loss·스케줄·데이터 순서·정밀도(FP32+TF32, batched views)·평가 빈도는 기존과 같다.
  진단은 RNG를 소비하지 않는다(새 batch_KL 호출 없음).

## Phase 0 (학습 0 step, CPU) — 통과 후에만 Phase 1

probe 기준: STS-B dev에서 mean pooling → 단어 분포 raw 0.5938, 768d mean 0.5931, temp 1·dev 평균 중심화 H 6.081 nat,
문장 간 KL 7.582. 허용치: Spearman ±0.001, H ±3%, KL ±5%(probe는 무작위 2000쌍, 이번에는 모든 쌍 평균).
사전 측정 결과(campaign에서 다시 실행): 0.59375 / 0.59312 / 6.081 / 7.517, 전부 통과. C3는 H/lnK 0.987, KL 0.227.
학습 step 0의 H_pt/lnK·batch_KL은 smoke에서 기록한다(이때 center=0이고 warmup teacher 온도이므로 probe와 정의가 다르다).

## Phase 1 탐색 (조건별 독립 Optuna, seed 42)

사용자 요청(2026-11-16)으로 축과 예산을 bottleneck_dino(5축·20 trial)보다 넓힌다.

- **공통 15축:** C0을 낳은 coviso 19축 공간(`src/joint_search.py::SPACE`)에서 절대 온도 3축과 head lr을 뺀 것.
  cov-iso mean/cov λ, backbone lr, grad clip, weight decay, LR warmup, teacher momentum 끝값·ramp,
  teacher temp warmup 기간, center momentum, covariance EMA momentum, 노이즈 t_lo·t span·t_start 비율·노이즈 warmup.
  범위는 C0 탐색과 같다.
- **온도 3축:** student temp τs: C1/C2 [0.3, 2] log, C3 [0.05, 2] log(기존 범위와 넓은 범위 모두 포함).
  teacher 최종 온도 = `_teacher_ratio` × τs, ratio [0.1, 1] log. teacher warmup 시작 온도 = `_warmup_ratio` × teacher 최종,
  [0.2, 1]. 구성상 teacher는 student보다 날카롭거나 같다.
- **head 축:** C1 `_head_lr_mult` [0.25, 4] log, C2 없음, C3 head lr [1e-4, 2e-3] log. 총 C1 19축, C2 18축, C3 19축.
- 조건별 독립 TPE(seed 20261117, startup 8, multivariate), **조건당 40 attempted trial**(pruned·failed 포함; C0의 탐색 예산과 같다).
  MedianPruner(startup 8 trials, warmup 1500 step, 간격 250, 최소 4 trial)는 지금까지의 dev best를 비교하며 마지막 평가에서는
  자르지 않는다. 각 조건은 직전 trial이 끝나야 다음 trial을 ask하고, GPU 2슬롯은 시도 수가 가장 적은 조건부터 채운다.
- anchor 2개를 먼저 넣는다. 공통 축은 t13 값이다. C1/C2는 (τs 1.0, ratio 0.5)와 (τs 0.5, ratio 0.5, grad clip 3.0),
  C3는 t13 온도 그대로와 (τs 1.0, ratio 0.5, grad clip 3.0)이다.
- 목적함수: 768d STS-B dev best(250 step마다, 4000 step), C0의 선택 규칙과 같다.
- **보수적 해 점검:** 선택된 설정의 τt/τs와 마지막 H_pt/lnK를 보고한다. ratio > 0.9 이거나 H_pt/lnK가 0.98 이상이면 "신호가 사라지는 해"로 표시한다.

## Phase 2 확인과 test

- 조건별 완주 trial의 seed-42 dev best top 3를 seed 43/44로 재학습한다(18 run). seed 42는 탐색 run을 재사용하며, C0 t13의 확인 방식과 같다.
- seed 42/43/44 dev best 평균으로 조건별 설정을 동결하고 `selection.json`을 test 전에 기록한다.
- 선택된 3 seed의 best/last checkpoint에서 STS7 test를 평가한다. C0 t13 checkpoint도 같은 코드로 다시 평가한다.
- **주 지표:** STS7 raw, dev-best checkpoint(C0 기록과 같은 규약). last와 후처리 값도 함께 보고한다.
- **후처리:** 학습 코퍼스(wiki1m 캐시)에서 seed 0으로 뽑은 고정 10,000문장으로 평균과 상위 PC를 구한다.
  center / center_pc1 / center_pc2 각각에서 중심화와 PC 제거 후 L2 정규화를 하며, center_pc2를 대표 후처리 값으로 표기한다.
  test 데이터의 통계는 쓰지 않는다. C0에도 같은 후처리를 적용한다.

## 사전 등록 예측

판정은 각각 성립 / 기각 / 판정불가다. 판정불가를 억지로 결론으로 바꾸지 않는다. seed 짝 차이 Δ는 STS7 raw best 기준이다.

- **P1 (주 가설):** C1 − C0 평균 Δ ≥ 0.003이고 3 seed 중 2개 이상 양수이면 성립, 평균 ≤ 0이고 2개 이상 비양수이면 기각, 그 외 판정불가.
- **P2 (target 정보량):** C1 final run 각 seed에서 기록된 모든 step의 H_pt/lnK 최댓값 ≤ 0.95(t13 ≈ 0.997)이고
  batch_KL 중앙값 ≥ 0.47(t13 0.047의 10배)이면 그 seed는 통과, 둘 다 실패하면 실패다.
  2 seed 이상 통과면 성립, 2 seed 이상 실패면 기각, 그 외 판정불가. P1 기각·P2 성립이면 "정보량이 큰 target이 STS로 이어지지 않는다"로 기록한다.
- **P3 (귀속):** C3 − C1 평균 |Δ| ≤ 0.003이면 "prototype 방향의 의미가 핵심", 평균 Δ < −0.003이면 "MLM head의 학습된 지식이 핵심",
  평균 Δ > +0.003이면 판정불가. 시드별 값을 함께 보고한다.
- **P4 (동결):** C1 − C2에 P1과 같은 규칙을 적용한다. 성립 = head 적응이 필요하다, 기각 = 고정된 의미 좌표계로 충분하다.
- **P5 (어휘 복사 퇴화):** student top-10 단어 중 입력 문장에 실제로 나온 토큰의 비율(copy rate)을 각 student view의 t 구간
  ([0,.5), [.5,.7), [.7,.85), [.85,1])별로 step ≥ 3000에서 집계한다. 표본 200개 이상인 가장 낮은 t 구간과 가장 높은 t 구간을 비교한다.
  seed별로 높은 t ≥ 0.9 × 낮은 t 그리고 낮은 t ≥ 0.3이면 퇴화, 높은 t < 0.7 × 낮은 t이면 비퇴화(기각)다.
  2 seed 이상 같은 결과일 때 그 판정을 내리고, 그 외는 판정불가다.
  teacher copy rate(t와 무관, centered top-10)는 별도로 기록한다.
  주의: teacher는 clean이므로 "t 구간별 copy rate"는 student 예측에서만 정의된다.

## 로깅 추가 (vocab head 조건에서만)

H_pt/ln K, teacher 문장 간 대칭 KL(배치 내 모든 쌍, 결정적), teacher copy rate, student copy rate(t 구간별)와 표본 수,
τt/τs, 평가마다 head 파라미터별 상대 이탈량 ‖W−W0‖/‖W0‖, 입력 임베딩 상대 이동량, decoder 저장소 공유 여부.
평가마다 고정 5문장(“A woman is riding on a horse.”, “A woman is carrying a boy.” 포함)에 대해 teacher의 centered top-10
단어를 `vocab_topwords.jsonl`에 기록한다(C3는 군집별 중심에 가장 가까운 단어 3개).

## 예산

smoke 3(60 step) + 탐색 최대 120 + 확인 18 = 최대 138 full run, 그 뒤 STS7 24 eval(C0 6 포함). 동시 GPU 최대 2.
run당 약 20–25분이므로 pruning 전 기준 최대 약 30시간이다.
