# 손실의 역할과 학습 동역학

2026-11-05. 완료한 실험의 로그·checkpoint를 읽는 사후 분석이다. 학습·튜닝·test 재선택은 하지 않았다.
세 시드101/102/103의 선택된 설정을 비교하며, 나쁜 trial 예시는 탐색 seed42로 별도 표시한다.
숫자는 특별한 설명이 없으면 해당 step의3시드 평균이다. 관측과 수학적 성질, 원인에 대한 가설을 구분한다.

## 핵심 판단

1. 직접256d 조건은 최종 선택 run에서 붕괴하지 않았다. KL은 줄고256d rank·STS는 후반까지 증가했다.
2. 제약을 거는 공간이 달라졌다. 새 조건은 DINO와iso 모두 MLP 뒤에서만 적용되며, 기존768d 문장 공간에 직접 적용하던 iso가 없어졌다. 병목 개선이768d 개선을 보장하지 않는 가장 유력한 구조적 해석이다.
3. 일부 나쁜 trial은 uniform collapse와 다른 동역학이다. teacher가 거의 one-hot으로 과도하게 확신하고, student KL·gradient norm·clipping이 커지면서 STS가 악화됐다.
4. loss 값의 크기로 영향력을 판단할 수 없다. 직접256d의 마지막 mean loss 기여는 CE의0.8%지만 MLP gradient norm은 CE의1.93배다.
5. 실제 teacher 입력 word embedding은 student lookup이다. Transformer/head는 EMA지만 모든 입력까지 완전한 EMA라는 설명은 정확하지 않다. 공통 구현이며 이번 비교의 원인으로 확정할 수 없다.

## 데이터와 재현

원본은 `results/experiments/bneck_dino256_20261105`와 `bneck_iso_20261104`의
`final_runs.json`, run별 components/gradient/evaluations JSONL, 마지막 checkpoint다.
그룹 이름은 baseline=기존FlowDINO, direct256=실험2, cos_iso=실험1,
cos_only=실험1에서iso256만0, iso_only=실험1에서cosine만0을 뜻한다.
cos_only/iso_only에도 기존 DINO8192와iso768은 남아 있다. DINO-off 실험이 아니다.

- 손실·teacher entropy·gradient: 학습 minibatch32, student3뷰. gradient 상세는0/56/250/1000/2000/3000/3999에만 있다.
- STS·alignment/uniformity: 고정 STS-B dev. uniformity는500쌍, alignment는의미 유사도 높은 쌍의 거리다.
- rank·centroid·background cosine: 별도 고정2048문장, clean student 표현이다.
- train의 teacher/student view cosine과 clean2048의 서로 다른 문장 cosine은 모집단이 다르다. 둘을 같은 배치의 positive-negative margin으로 빼지 않는다.
- gradient norm은 clipping 전이다. parameter 수·head/backbone LR·AdamW 전처리가 달라 실제 update 크기와 동일하지 않다.

```bash
CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/flowdino-mpl \
  uv run --no-sync python scripts/analyze_loss_dynamics.py \
  --out results/analysis/<새-분석-폴더> --checkpoint-drift
```

실행 산출물: `results/analysis/loss_dynamics_20261105/`의 summary.json,
components.csv, gradients.csv, evaluations.csv, search_trials.csv, gradient_interference.csv.
그래프는 PNG/SVG/PDF다. shaded band는3시드 sample SD이며, 시간축 간격의 불확실성 추정은 아니다.

![두 공간의 동역학](../../../results/analysis/loss_dynamics_20261105/overview.png)

## 1. DINO CE: clean teacher의 상대적인 특징 반응을 noisy student가 예측

    p_t = stopgrad(softmax((q_t − c) / T_t))
    p_s = softmax(q_s / T_s)
    CE = H(p_t) + KL(p_t || p_s)
    dCE/dq_s = (p_s − p_t) / T_s  # batch/view 평균 계수 생략

기존 q는 W b인8192d prototype logits, 직접 조건 q는 L2 normalized b256 그 자체다.
DINO는 같은 문장의 증강 간 확률분포를 맞춘다. cosine처럼 벡터 방향을 직접 동일하게 만드는 손실은 아니다.
human STS 쌍을 학습 정답으로 보지 않으므로, invariance나pseudo-label 구분이 곧 의미 유사도 ranking 개선이라는 보장은 없다.

centering은 teacher의 좌표별 공통 bias를 제거하고, sharpening은 target을 뾰족하게 한다.
두 연산의 균형을 사용하는 설계는 [DINO 원논문](https://ar5iv.labs.arxiv.org/html/2104.14294)에 따른다.
원논문의 비전용 온도 임계값을 본 NLP/256d 조건에 그대로 이식하지 않는다.

### raw CE가 오르는데도 학습은 진행될 수 있다

| 직접256d step | H(teacher) | KL(t||s) | CE |
|---|---:|---:|---:|
| 0 | 4.7705 | 0.2543 | 5.0248 |
| 250 | 5.3436 | 0.1162 | 5.4598 |
| 3999 | 5.3476 | 0.0367 | 5.3843 |

teacher 온도를 올리는 동안 target entropy가 증가해서 CE가 올라간다. student의 matching gap은 오히려 줄었다.
반대로 실패 trial2(seed42)는 마지막 CE가약1.946으로 더 작지만, teacher entropy0.0277과KL1.918의 합이다.
같은 trial의768d dev last는18.79다. 낮은 CE를 좋은 학습이라고 해석하면 정반대 결론이 나온다.
log(K)도8192d=9.0109,256d=5.5452로 다르다. 두 구조의 raw CE를 직접 비교하면 안 된다.

### 성공한 직접 조건은 uniform target으로 사라지지 않았다

마지막 직접조건의 H(teacher)/log256은약0.9644, H(mean teacher)는5.5359다.
배치 내 분포 차이를 나타내는 H(mean p_t)−mean H(p_t)는0.1883nats다.
기준선은 같은 차이가0.0484nats다. 직접 조건의 target은 기준선보다 uniform에서 더 떨어져 있으며,
모든 문장이 동일한 uniform distribution이라는 설명은 관측과 맞지 않는다.
이 배치 차이는 semantic information 자체가 아니며, batch32에서 전 데이터의 prototype 사용 빈도를 나타내지도 않는다.

## 2. iso mean: 공통 방향을 줄여 centered variance의 총량을 확보

    L_mean = ||mu||²
    E||z||² = ||mu||² + tr(C)

unit-normalized z에서는 E||z||²=1이므로, global mean을 줄이면 centered variance의 총량이 증가한다.
모든 문장이 공통 방향으로 몰리는 것을 억제하는 역할이다. teacher centering과 동일한 연산은 아니다.
teacher centering은 target logits에만 적용되고, mean loss는 noisy student 표현에 직접 gradient를 준다.
softmax는 모든 좌표에 동일한 상수를 더해도 변하지 않지만 geometric mean loss는 그렇지 않다.

단, 여기 mu는 전체 데이터의 정확한 평균이 아니라 detached 과거 EMA와 현재 batch를 혼합한 통계다.
유한 batch·서로 상관된3뷰·EMA의 sampling noise 때문에 mean loss가0이 아니라고 곧바로 심한 전역 bias라 할 수 없다.

## 3. iso covariance: variance를 일부 방향에만 쓰지 않게 한다

    L_cov = ||C/tr(C) − I/D||²_F
          = sum_i p_i² − 1/D,  p_i = eigenvalue_i(C)/tr(C)

trace clamp가 비활성이고PSD인 경우의 식이다. 총 variance의 크기 대신 spectrum의 집중을 줄인다.
역 participation rank는1/(L_cov+1/D)다. 실제 로그 effective rank는 Shannon entropy 기반이고,
행렬도 training EMA가 아닌 clean2048 covariance이므로 두 숫자가 일치할 필요는 없다.

이것은 단순 off-diagonal covariance loss와도 다르고, 각 축의 variance 하한을 별도 hinge로 강제하는
[VICReg](https://arxiv.org/abs/2105.04906)와도 다르다. 우리 구현은 mean+unit normalization이
총 variance를 간접 유지하고 covariance 항이 그 분배를 조절하는 형태다.

normalized second moment의 trace는초기 I/D와 각 batch에서 약1이다. 따라서 혼합 covariance trace는
대략1−L_mean이다. 선택된 직접 run의 마지막은약0.9924, 실패 trial2도약0.8917이다.
1e−8 clamp 경계와 거리가 멀다. 이번 나쁜 trial의 큰 gradient를 trace 분모 폭주라고 설명할 근거는 없다.

## 4. 실제 영향력: 숫자는 covariance가 커도 gradient는 mean이 더 크다

직접256d 마지막 step3999의3시드 평균, 적용한 가중치 포함:

| 항 | scalar contribution | backbone gradient norm | MLP gradient norm |
|---|---:|---:|---:|
| DINO CE | 5.3843 | 0.4756 | 0.0569 |
| iso mean | 0.0436 | 0.2880 | 0.1096 |
| iso covariance | 0.0856 | 0.0773 | 0.0192 |

mean은 CE의약0.8% 크기지만 backbone gradient는CE의0.61배, MLP에서는1.93배다.
covariance의 scalar는mean의약1.96배인데 backbone gradient는mean의0.27배다.
동일 lambda5.711로 묶였다고 두 항이 같은 학습 영향을 갖는 것은 아니다.
이것은 해당 parameter group에서의 gradient 비교이며 AdamW update 기여도의 분해는 아니다.

![직접 조건의 성분과 gradient](../../../results/analysis/loss_dynamics_20261105/direct_gradients.png)

## 5. 시간축: 초기 재배치, 중기 맞춤, 후반 다른 공간에서의 수렴

### step0–55: CE만 활성, 통계는 이미 워밍업

covariance EMA는 처음부터뷰마다 갱신하되 loss contribution은0이다.
따라서 iso activation56은 통계의 cold start가 아니라 gradient를 켜는 경계다.

### step56–315: mean 제거와 스케줄 변화가 겹침

| step56 gradient norm | 기준선 backbone | 직접조건 backbone | 직접조건 MLP |
|---|---:|---:|---:|
| DINO | 0.2101 | 0.9167 | 0.3422 |
| mean | 2.7875 | 1.2863 | 1.0652 |
| covariance | 0.1322 | 0.1405 | 0.0771 |

기준선에서는 mean768의 gradient가CE의약13.3배다. 이 초기 단계에서 backbone 기하를 강하게 재배치한다.
직접 조건에서는 MLP mean의 gradient가CE의약3.1배다. 같은 regularizer라도 직접 작동하는 모듈이 다르다.
두 조건의256d clean centroid norm은초기0.746에서250step에 baseline0.169/direct0.163으로 감소했다.

noise warmup은101, LR warmup은206, teacher temperature warmup은315step에 끝난다.
이 구간의 변화에 원인을 하나만 붙일 수 없다. 데이터가 지나며 augmentation 난도와학습률·target softness가 함께 달라진다.

### step1000–3999: 직접 조건은 학습되지만768d 분산은 크게 늘지 않음

| 직접조건 | step1000 | step3999 |
|---|---:|---:|
| KL(t||s) | 0.0553 | 0.0367 |
| 768d dev | 66.68 | 70.94 |
| 256d dev | 64.24 | 68.05 |
| 768d rank | 245.30 | 245.71 |
| 256d rank | 85.21 | 103.62 |

선택 run의 best dev 평균70.96과 last70.94는 거의 같다. 후반에 모델 전체가 무너져서 기준선을 놓쳤다는 설명은 맞지 않는다.
rank가 거의 일정해도768d STS는 개선됐다. rank와STS 사이도 일대일 관계가 아니다.
cosine LR은 마지막에 거의0이므로 gradient가 남아 있어도 parameter update는 작아진다.

### EMA의 세 가지 시간 척도

- covariance: momentum0.81951을뷰마다 적용. 반감기약3.48뷰=1.16optimizer step. 한 step의 이전 통계 잔여율은 m³=0.5504다.
- teacher center: momentum0.92, step당 한 번. 반감기약8.31step.
- teacher parameters: momentum0.997→0.998501, 고정 계수로 환산한 국소 반감기약231→462step.

이들은 서로 다른 통계·목표를 추적한다. covariance는 상당히 빠른 noisy batch 통계이며,
teacher parameter EMA는 느린 target을 만든다. 후술하는 student word lookup 때문에 teacher 입력 전체가 이 반감기를 따르지는 않는다.
모든 과거 통계·teacher에는 gradient가 흐르지 않고, 이 시간 방향으로 backpropagation을 unroll하지 않는다.

## 6. 손실 간 충돌은 일부 있으나 실패의 단일 원인으로 보기 어려움

직접 조건은 파라미터가 backbone+MLP이고 loss가 세 개뿐이므로 기록된 norm으로다음 값을 복원했다.

    beta = (||g_total||² − sum_i ||g_i||²) / (2 sum_{i<j} ||g_i|| ||g_j||)

이는 pairwise cosine의 norm 가중 평균이다. 양수면 순 강화, 음수면 순 상쇄다.
각 성분 norm은 backbone/MLP norm의 제곱합으로 계산하고, 먼저 seed별 복원 후 평균했다.
CE-only step0에서 전체 norm 일치 오차는약1e−7 수준이었다.

| step | beta mean ± seed SD |
|---|---:|
| 56 | +0.186 ± 0.078 |
| 250 | +0.130 ± 0.156 |
| 1000 | +0.083 ± 0.092 |
| 2000 | +0.060 ± 0.057 |
| 3000 | −0.169 ± 0.117 |
| 3999 | −0.065 ± 0.141 |

측정한 초기·중기 지점에서는대체로협조적이고3000에서는순상쇄가 있다. 마지막은seed별부호도다르다.
어느 쌍이 충돌하는지, MLP 안에서만 어떤 방향인지, 실제 AdamW update 간섭이 무엇인지는 복원하지 못한다.
baseline은 expansion gradient norm이 별도로 기록되지 않아 같은 복원을 적용하지 않았다.
따라서 'mean과DINO가 항상 싸워서 성능이 낮다'는 단정은 지지되지 않는다.

## 7. 실패 trial: 과도한 sharpening과unit-logit의 표현 한계

seed42의 마지막 관측:

| trial | teacher/student T | iso lambda | H(teacher) | KL | logged clipping | 768d last |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.403 | 1.410 | 0.1116 | 1.200 | 90.0% | 36.18 |
| 2 | 0.173 | 0.0050 | 0.0277 | 1.918 | 95.0% | 18.79 |
| 3 | 0.154 | 0.0304 | 0.0055 | 2.390 | 93.3% | 19.46 |
| 15 | 0.943 | 5.711 | 5.3441 | 0.0527 | 0% | 70.24 |

clipping은401개 logged step에서의 비율이다. 모든4000step의 정확한 빈도는 아니다.
나쁜 trial은 target이거의one-hot인데 noisy student가 맞추지 못하고, 큰 gradient가 지속된다.
teacher의 batch entropy는약2.8도 남아 있어 모든 문장이 하나의 동일한 출력으로 붕괴했다고 말하면 부정확하다.
batch32에서 H(mean p)가작다고 전 데이터에서 소수좌표만 사용한다고도 단정할 수 없다.

![좋은 설정과나쁜설정](../../../results/analysis/loss_dynamics_20261105/sharp_trials.png)

### unit-normalized256d를 logits로 쓰면 student 확률에 상한이 생긴다

||b||=1인K차원 vector에 대해한 좌표의 최대 softmax 확률은

    p_max = 1 / (1 + (K−1) exp(−sqrt(K/(K−1))/T_s))

다. 다른 K−1좌표를 모두동일한음수로 두고 주좌표와의 margin을 최대로 만들면 얻는다.
K=256에서 trial2의 T_s=0.2071은 최대0.331, trial3의 T_s=0.2517은 최대0.174다.
그런데 실제 teacher entropy는거의0이다. 이 두 경우student는teacher처럼one-hot이될수없고,
낮은teacher temperature와높은student temperature가 표현가능성 측면에서도 불일치한다.
모든 student 좌표에 같은 확률 상한이 있으므로 CE≥−log(p_max), 따라서
KL≥max(0,−log(p_max)−H_teacher)다. 마지막 로그로 계산한 최소 KL 하한은
trial2 약1.078nats, trial3 약1.745nats다. 관측 KL은 각각1.918,2.390이었다.
이는 matching gap이0이 될 수 없다는 수학적 제약이다. 제약된 stationary point에서 gradient까지
반드시 커야 한다는 뜻은 아니며, 실제 큰 gradient는 별도로 로그에서 관측한 사실이다.
trial15의 T_s=0.1079에서는 이 상한이0.977이며 실제 target은 더 부드럽다.
따라서 이 상한만으로 선택된 최종 run의 기준선 대비 하락을 설명할 수는 없다.
trial1은 상한만으로 설명되지 않는다. augmentation·LR·EMA·iso의 공동효과도 남아 있다.
모든 변수는 함께 튜닝됐으므로 sharpening 또는lambda 하나의 인과효과로 주장하지 않는다.

기존 expansion의 W b는logit vector 전체가unit norm으로 제한되지 않는다.
이 차이는단지'클래스수가8192→256'이라는설명보다구체적인차이다. 그렇다고확장이필수라는증거는아니다.

## 8. 왜 병목은 좋아지고 768d는 나빠졌나

기존 경로에서 CE8192는 MLP를 거쳐 흐르고, iso768은 평가 공간에 직접 걸린다.
직접 조건에서는 CE256과 iso256 모두 MLP 뒤에 걸리고, 768d 출력에는 직접 기하 제약이 없다.

    baseline: h768 -> z768 -> iso768
                  -> MLP -> b256 -> W8192 -> CE
    direct:   h768 -> z768 -> STS768 evaluation only
                  -> MLP -> b256 -> CE256 + iso256

MLP는 768d의 일부 변형을 흡수해서 256d 분포를 좋게 만들 수 있다. 국소적으로 loss의 h에 대한 gradient는
J_b(h)^T를 통과하므로 b를 거의 바꾸지 않는 방향은 약하게 제약된다. L2 뒤 256d Jacobian의 rank는 최대255다.
단, 기존 CE8192도 같은 bottleneck을 거치므로 '8192라서 gradient rank가8192'는 틀리다.
샘플별 Jacobian이 다르므로 전체768d 데이터 rank가255 이하라는 주장도 틀리다.

| 마지막clean진단 | 기준선 | 직접조건 |
|---|---:|---:|
| 768d rank | 326.80 | 245.71 |
| 768d mean norm | 0.1152 | 0.1754 |
| 768d uniformity | −3.304 | −3.032 |
| 768d positive-pair alignment | 0.2710 | 0.2506 |
| 256d rank | 56.15 | 103.62 |
| 256d uniformity | −2.222 | −2.937 |

768d의 의미상 유사한 문장 쌍의 거리도 더 작지만 STS7은 낮다. invariance·spread·rank 중 어느 하나만 좋다고 semantic ranking이 좋아지는 것은 아니다.
규제 위치가 바뀌어 표현 공간별 기하가 달라졌다는 해석이 데이터와 잘 맞지만, 온도·가중치 등도 변했으므로 인과 분리는 아직 없다.

## 9. 실험1의 cosine: 유용한 invariance와 공통 방향 지름길이 동시에 존재

unit vector에서1−cos는0.5||b_s−b_t||²다. 같은 문장의 정확한 방향을 맞추지만 teacher의 공통 성분까지 맞춘다.
마지막3시드 평균:

| 조건 | train view cosine | clean background cosine | clean256 mean norm |
|---|---:|---:|---:|
| 기준선 | 0.772 | 0.0079 | 0.091 |
| cos있음,iso256없음 | 0.957 | 0.6968 | 0.835 |
| cos+iso256 | 0.919 | 0.3305 | 0.575 |

unit vector의 서로 다른 문장 평균 cosine은(n||mu||²−1)/(n−1)이므로,
공통 방향이 강하면 문장 내용과 무관하게 cosine이 높아진다. 다만 train/clean 평가셋이 다르므로
위 두 cosine을 동일 데이터의 margin으로 해석하지 않는다.
iso 추가는 이 설정에서 공통 방향을 줄였지만 최종 STS 개선은 일관되지 않았다.

cos+iso256의 마지막 weighted loss는 cos0.00986, mean256 0.01283, cov256 0.00123이다.
backbone gradient norm은 각각0.1244,0.01448,0.00156으로 cos가 mean보다 약8.6배 크다.
작은 cos loss가 작은 영향을 뜻하지 않는다. cos gradient는 DINO CE의 약3.5배이며 초기 보정 비율은 학습 중 고정되지 않았다.
반면 cos만 제거하면 같은 선택 설정의 STS7이64.37→62.39로 떨어졌다.
cosine이 무조건 해롭다는 결론도 틀리다. 이 설정에서 invariance에 기여하지만 기준선65.82를 넘기지는 못했다.

## 10. 별도로 기록할 구현 세부: teacher word embedding은 student의 현재값

`Trainer.forward_views`는 student.get_input_embeddings()(ids)를 계산한 뒤,
같은 clean tensor를 teacher의 inputs_embeds로 전달한다. 따라서 teacher 자체의 EMA word lookup 행렬은 이 경로에서 사용되지 않는다.
teacher의 Transformer·position/token-type embedding·head는 EMA이며, no_grad/stop_gradient는 정상적으로 적용된다.

선택된 last checkpoint에서 student/teacher word lookup 행렬의 상대 L2 차이는
기준선 약0.178%, 직접 조건 약0.149%다. 입력 단계에서 실제로 차이가 있으며0은 아니다.
이는 전체 행렬 norm 비율이지 사용 빈도가 높은 token의 영향 또는 출력 perturbation 크기는 아니다.
모든 비교 조건의 공통 구현이므로 현재 성능 차이의 원인으로 확정하지 않는다.
완전 EMA teacher를 의도했다면 별도 수정 후 재검증할 항목이며, 이번 분석 중 알고리즘은 수정하지 않았다.

과거 baseline checkpoint에는 DINO center도 저장되지 않았다. 새 직접 조건에는256d center를 저장한다.
따라서 과거 center와 noisy batch를 그대로 복원한 전체 gradient 각도 분석에는 한계가 있다.

## 부록: iso가 완전 붕괴 상태에서 탈출을 보장하는가

그렇지 않다. 모든 문장이 b=(1,...,1)/sqrt(256), teacher center도 b인 정확한 대칭 고정점을 만들었다.
현재 코드를 double precision CPU에서 계산하면 CE=log256, mean=1, cov=1/256인데,
L2 정규화 이전 gradient norm은 각각0, 약1.2e−18,0이다.
unit sphere의 mean gradient가 정규화의 radial null direction에 놓일 수 있기 때문이다.
이는 가능한 stationary point의 존재만 보인 것이지 안정한 attractor나 실제 실패 trial의 원인을 증명한 것은 아니다.
관측한 나쁜 trial은 오히려 teacher entropy가 매우 낮았으므로 이 uniform 예시와 구분해야 한다.

## 남은 가설과 다음에 분리할 요인

새 학습은 시작하지 않았다. 원인을 분리하려면 다음 두 축이 특히 명확하다.

1. 8192 확장을 제거한 채 iso768을 추가로 유지하는 조건: 평가 공간의 직접 기하 규제가 사라진 효과를 검증한다.
2. iso256은 normalized b에 유지하되 DINO logit scale을 분리하는 조건: unit-logit 상한과 강한 sharpening의 불일치를 검증한다.

어느 조건이 더 좋을지는 현재 자료로 예측하지 않는다. 비교 시 새 구조의 온도·가중치도 튜닝해야 하며,
일부 loss를 고정값으로 끄고 나온 점수를 그 구조의 최적 성능으로 해석하면 안 된다.
