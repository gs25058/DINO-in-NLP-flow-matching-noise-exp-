# 분석 v2 — 왜 STS 고점이 0.70 근처에 갇히는가

**데이터 출처**
- Part A: `scripts/analyze_checkpoints.py` → `results/analysis/checkpoint_metrics.csv` (35행: run 32개 + backbone별 pretrained 기준선 2개 + 계산 상세는 npz/png)
- Part C: `configs/r2_diag.yaml`(붕괴 대표) / `configs/r5d_diag.yaml`(안정 대표), 둘 다 seed 42, 1500 step, 진단 로깅 전부 on. 원 로그: `results/logs/r2_diag/train.log`, `results/logs/r5d_diag/train.log`. 겹쳐그림: `results/analysis/partc_overlay.png`, `results/analysis/partc_grad_norms.png`
- 참고 상수: SimCSE-BERT-base(unsup) 논문 보고치 — alignment ≈ 0.18, uniformity ≈ **−2.6** (원 논문 부록 수치, 재계산 아님 — 자릿수 비교용 참고치로만 사용)

**주의(데이터 결함 하나)**: Part A 최초 실행에서 `r2_anchor_curriculum` 체크포인트가 이전 세션의 3-step 회귀 테스트로 덮어써져 있던 것이 드러남(sts=0.484로 비정상). Part C에서 `r2_diag`로 실제 1500-step 재실행을 했으므로 이 분석에서 "r2"는 전부 `r2_diag`(정상) 수치를 쓴다. Part A CSV의 `r2_anchor_curriculum` 행 자체는 손상된 채로 남아있으니 별도로 갱신하지 않는 한 신뢰하지 말 것.

---

## Q1. uniformity와 alignment 중 무엇이 부족한가?

**결론: uniformity가 압도적으로 부족하다. alignment는 이미 SimCSE와 비슷한 수준이다.**

| | alignment (↓좋음) | uniformity (↓좋음, 더 음수가 좋음) |
|---|---|---|
| pretrained ModernBERT-base | 0.055 | **−0.172** |
| pretrained bert-base-uncased | 0.204 | −0.674 |
| r2 (붕괴 대표, 1500 step) | 0.15*(참고: r1c 계열 0.12~0.16 범위) | −0.46 ~ −0.57 |
| r5d_combined (최고 성능) | 0.146 | −0.574 |
| r5d_bert_combined (BERT 최고) | 0.219 | −0.772 |
| SimCSE-BERT-base(unsup), 참고치 | ≈0.18 | ≈**−2.6** |

(alignment는 STS-B dev 500쌍 중 score≥0.8[원 척도 4.0] 서브셋, uniformity는 같은 500쌍 전체. `checkpoint_metrics.csv` 전체 run은 각자 0.11~0.28 / −0.18~−0.85 범위에 분포)

- **alignment**: 우리 최고 run(r5d_combined=0.146, r5d_bert_combined=0.219)이 SimCSE 참고치(≈0.18)와 **같은 자릿수**다. 오히려 ModernBERT 쪽은 SimCSE보다 낮다(더 좋다). alignment는 병목이 아니다.
- **uniformity**: pretrained 상태(ModernBERT −0.17)에서 학습을 거쳐 최고 −0.77(BERT)/−0.57(ModernBERT)까지 개선되지만, SimCSE의 −2.6에는 **한 자릿수 이상 못 미친다**. `results/analysis/alignment_uniformity.png`에서 전체 run이 SimCSE가 있을 위치(왼쪽 바깥)에서 한참 떨어진 좁은 띠에 뭉쳐 있는 것으로도 확인됨.
- **해석**: SimCSE의 InfoNCE는 배치 내 negative pair를 명시적으로 밀어내는 항이 있어 uniformity를 직접 최적화한다. 본 방법(DINO CE + centering/sharpening)은 collapse(점으로 뭉침, uniformity→0)만 막을 뿐 구형(sphere) 전체에 고르게 펴는 힘이 없다 — **STS 고점이 0.70 근처에 갇히는 근본 원인은 negative-pair 기반 uniformity 압력의 부재로 보인다.**

---

## Q2. step 250 피크의 정체 — pretrained 공간의 정제인가?

**결론: 부분적으로만 맞다. "정제"보다는 "완만한 연속적 표류(drift) 중 STS-B와 우연히 가장 잘 맞는 지점"에 가깝고, 진짜 피크는 250이 아니라 150~500 사이에 넓게 걸친 잡음 섞인 구간이다.**

Part A의 CKA는 최종 체크포인트 기준이라 피크 시점(step 250) CKA를 직접 계산하지 못했다 — 대신 Part C의 `dense_early_eval`(step≤300은 25-step 간격) + `diag_drift`(인접 평가 시점 임베딩 코사인)로 대리 판단했다.

- **피크 위치 재확인(dense eval)**: r2_diag의 STS-B dev는 150(0.680)/275(0.679)/**500(0.681, 관측된 최댓값)**에서 서로 0.01 이내로 몰려 있다 — 250이라는 특정 지점이 아니라 **150~500 step의 넓은 평탄부**가 실체다. 기존 250-step 간격 로깅은 이 평탄부를 "step 250 피크"로 단순화해 보여준 것.
- **drift_cosine이 이 구간에서 낮아지지 않는다**: step25~300 구간 내내 0.93~0.98로 높게 유지된다(=인접 평가 지점 사이 임베딩이 큰 폭으로 재배열되지 않음). 즉 "피크"는 급격한 재구성이 끝나고 멈춘 스냅샷이 아니라, **작은 폭의 변화가 계속 누적되는 과정 중 한 지점**이다.
- **eff_rank는 STS보다 먼저 꺾인다**: r2_diag의 eff_rank는 step 200~225(≈177)에서 이미 정점을 찍고 step 300(148)부터 뚜렷이 하락하는데, STS-B dev는 step 500(0.681)까지도 최댓값 근처를 유지한다 — **표현 붕괴(rank 감소)가 STS 하락보다 먼저, 더 조용히 시작된다.** STS 하락은 뒤따라오는 지연된 증상이다.
- r5d_diag는 같은 구간(step300)에서 drift_cosine=0.982(r2의 0.930보다 훨씬 안정)이고 eff_rank도 179로 계속 상승 중 — **분기(divergence)는 이미 step 300 이전부터 시작되어 있었다.**

**정리**: "250 피크 = pretrained 표현의 국소 최적 정제"라는 가설은 다소 부정확하다. 실제로는 연속적인 완만한 표류가 처음부터 진행 중이고, STS-B라는 특정 척도가 우연히 이 초기~중기 구간(150-500)과 가장 잘 정렬되어 있을 뿐이며, 그 정렬이 깨지는 것보다 앞서 rank 붕괴가 이미 시작된다.

---

## Q3. 학습 신호는 어디서 나오고, 후반에 얼마나 약해지는가?

**결론: 신호는 약해지지 않는다 — 오히려 세 파라미터 그룹(backbone/bottleneck/prototype) 전부에서 계속 커진다. r2가 무너지는 것은 "신호 고갈"이 아니라 "억제되지 않는 자기강화(self-reinforcing sharpening)"다. r5d는 반대로 신호가 정상적으로 감쇠하며 수렴한다.**

`results/analysis/partc_grad_norms.png` 참조.

| 시점 | r2 backbone / bottleneck / prototype | r5d backbone / bottleneck / prototype |
|---|---|---|
| step 0 | 0.080 / 0.036 / 0.015 | 0.168 / 0.101 / 0.016 |
| step 500 | 0.070 / 0.077 / 0.004 | 0.012 / 0.019 / 0.001 |
| step 1000 | 0.145 / 0.115 / 0.031 | 0.015 / 0.019 / 0.001 |
| step 1499 | **0.224** / 0.123 / **0.093** | 0.011 / 0.020 / 0.001 |

- r2: 세 그룹 모두 학습 후반으로 갈수록 grad norm이 **커진다**(특히 prototype은 step500→1499 사이 약 23배 증가, backbone도 3배 이상). 이는 "학습 신호가 약해져서" 붕괴하는 게 아니라, DINO CE가 스스로를 계속 더 강하게 밀어붙이는 자기강화 루프에 들어갔다는 뜻이다.
- r5d: 세 그룹 모두 초반에 감소한 뒤 낮은 값에서 **평탄하게 유지**(prototype은 사실상 바닥 0.001에 고정) — 정상적으로 수렴하는 최적화의 모습.
- **P2 3중 재검증(batch-KL 스냅샷 → 전 구간 평균 → 이번 뷰별 t-bin)**: r2_diag의 t-bin별 KL은 step 500(전 구간 0.024~0.025), step 1000(0.095 안팎), step 1499(0.309 안팎) 어느 시점에서도 **5개 구간이 서로 사실상 동일**하다 — t에 따른 KL 차이가 전혀 보이지 않는다. r5d_diag도(고정 구간이라 bin 3·4만 존재) 두 구간이 항상 거의 같다. 세 번째 방법으로도 **P2(KL이 t에 단조 증가)는 기각**된다 — KL의 크기는 t가 아니라 "얼마나 학습이 진행됐는지(=얼마나 sharpening됐는지)"에 좌우되는 것으로 보인다.

---

## Q4. head/프로토타입 쪽에 숨은 병목이 있는가?

**결론: 그렇다. r2는 학습이 진행되며 사용하는 프로토타입 수가 뚜렷이 줄어들고, r5d는 끝까지 거의 전부를 사용한다. 이것이 이번 분석에서 가장 깨끗하게 갈리는 지표다.**

| 시점 | r2 active prototypes (/8192) | r5d active prototypes (/8192) |
|---|---|---|
| step 0 | 6597 | 6597 |
| step 100 | 7327 | 7331 |
| step 500 | 7177 | 7293 |
| step 1000 | 6571 | 7284 |
| step 1499 | **4882 (60%)** | **7268 (89%)** |

(`active_prototypes` = p̄_t를 내림차순 정렬해 누적합 0.9에 도달하는 최소 개수, `src/diagnostics.py::active_prototype_count`)

- r2는 step 500~700 부근부터 급격히 줄기 시작해 최종적으로 8192개 중 4882개(60%)만 사용 — 정확히 STS/eff_rank가 무너지기 시작하는 시점과 겹친다.
- r5d는 처음부터 끝까지 7270~7330(89%) 부근에서 거의 움직이지 않는다.
- Part A의 **prototype 벡터 자체의 pairwise 코사인**(`proto_cos_mean`)은 조금 다른 그림을 보인다: 1500-step 정상 run들(r1/r3/r5 계열)은 대체로 0.05~0.15로 서로 비슷하고, r5d(0.0006~0.003)가 오히려 **가장 낮다**(=프로토타입 벡터들이 가장 서로 독립적/직교에 가까움). 반면 **3000/5000-step 연장판**(r2_ext3000=0.264, r4_ext3000=0.233, r5c_ext3000=0.238, r5d_ext5000=0.107)은 뚜렷이 상승한다.
  - 즉 "사용 편중(active count 감소)"은 표준 1500-step 예산 안에서도 빠르게(r2에서 이미) 나타나는 **빠른 신호**이고, "가중치 자체의 군집화(proto_cos 상승)"는 3000+ step에서야 뚜렷해지는 **느린 신호**다 — 같은 방향의 병목이 서로 다른 시간 스케일로 나타난다.
  - r5d조차 5000 step까지 가면 proto_cos이 0.0006→0.107로 뛰는데, 이는 이전 대화에서 확인한 "r5d_combined도 3500~5000 step에서는 결국 평평해지며 소폭의 재붕괴 조짐을 보인다"는 관찰과 일치한다 — head 병목은 R5d로 "해소"된 게 아니라 "훨씬 늦춰진" 것이다.

---

## 보너스: teacher vs student 최종 평가

**결론: teacher가 일관되게 student와 같거나 낫다. 특히 후반 하락 구간에서 격차가 커진다 — 최종 평가를 teacher 임베딩으로 바꾸는 것을 제안한다.**

| run | student STS(최종) | teacher STS(최종) | 차이 |
|---|---|---|---|
| r2_diag | 0.578 | **0.611** | +0.034 |
| r5d_diag | 0.715 | **0.719** | +0.003 |

- r2_diag는 step 500 부근까지 teacher와 student가 비슷하다가(둘 다 0.66~0.70대), student가 무너지기 시작하는 step 750 이후 teacher가 계속 더 높게 유지된다(step1000: teacher 0.669 vs student 0.619, step1250: 0.639 vs 0.595) — **teacher는 EMA로 평활화되어 있어 student의 급격한 후반 붕괴를 일부 흡수**한다. 다만 teacher도 결국(step500 피크 0.704 → step1499 0.611) 같은 방향으로 하락하므로 "teacher면 문제가 없다"는 아니고 "덜/늦게 겪는다" 정도다.
- r5d_diag는 애초에 안정적이라 teacher/student 차이가 거의 없다(+0.003).
- **제안**: 이후 최종 평가(및 최고 checkpoint 선정)에 student 대신 teacher 임베딩을 기본으로 쓰는 걸 검토할 가치가 있다. 특히 후반 하락이 있는 조건을 비교할 때 student 기준 수치는 붕괴를 과장해서 보여줄 수 있다. 다만 두 run 모두에서 teacher도 결국 같은 방향으로 움직이므로, "붕괴 자체를 막는" 해법이 아니라 "측정 시점의 잡음을 줄이는" 정도로 이해해야 한다.

---

## 판정불가로 남긴 것

- **Q2의 CKA-vs-pretrained 피크 시점 값**: Part B에 주기적 CKA 로깅을 추가하지 않아 dense_early_eval 구간에서 CKA 자체는 계산하지 못했다(대신 drift_cosine으로 대체 판단). 정확한 "CKA가 피크에서 높고 이후 하락하는가"를 보려면 diag_drift와 별개로 학습 중 CKA-vs-pretrained를 매 dense eval마다 계산하는 기능이 추가로 필요하다.
- **prototype 코사인과 active count 중 어느 쪽이 "원인"이고 어느 쪽이 "결과"인지**: 두 지표가 다른 시간 스케일에서 같은 방향을 가리키는 것은 확인했지만, 인과관계(사용 편중이 먼저 벡터 군집화를 유발하는지, 혹은 그 반대인지)는 이번 데이터로는 판별 불가 — 두 run만으로는 시간 지연 관계를 인과로 해석하기엔 근거가 약함.
