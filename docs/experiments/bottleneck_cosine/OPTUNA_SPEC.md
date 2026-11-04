# Bottleneck cosine — 사전 등록

2026-11-04. 구현 시작 checkout은 `exp/coviso-component-relaxation`의
`f1192dba2bb61e51a1f21cff33f5ce6f18f08df3`이다.
작업 브랜치는 `exp/bottleneck-cosine-paired-optuna`이며 main 병합·원격 push를 하지 않는다.
기존 보고서·선택 config·실험 산출물은 보존한다.

## 질문과 실제 기준선

기존 DINO CE와 768차원 공분산 정규화를 유지하면서, 같은 문장의 clean EMA teacher와
noise student의 **정규화된 256차원 bottleneck**을 직접 맞추면 768차원 문장 STS가 개선되는가?

실제 R18 4000-step 대조군의 refactored config인 `configs/coviso_relax/base.yaml`을 상속한다.
기준선 계보와 정확한 값은 `baseline_provenance.json` 및 campaign resolved config에 남긴다.
이전 공동 탐색 trial13은 dev 개선 판정을 통과하지 못했으므로 이번 기준선으로 채택하지 않는다.
과거 보고 점수를 현재 대조군 실측으로 대신하지 않는다.

고정 조건: bert-base-uncased, masked mean+L2 768d, DINO head 256→8192,
batch32, student3 views, clean teacher, 기존 Gaussian flow interpolation,
epoch 순서·길이128, 4000 optimizer steps, 평가 loop step0/250/…/3750/3999,
FP32+TF32·batched views·AMP 없음. covariance는 768d normalized embedding에 그대로 적용하며
두 가중치 모두 정확히 **5.874093568592943**, EMA momentum0.8195105695006707, activation step56이다.
노이즈 t_lo0.3919166650119331, t_start0.4638073616095715, t_max0.8811226221985871,
warmup fraction0.02545633588184152도 양쪽에서 고정한다.

## 유일한 목적함수 차이

head MLP→L2 직후, 8192 expansion 직전에서 이미 계산한 벡터를 재사용한다.

    L_bneck = mean_views mean_batch [1 − dot(b_s, stopgrad(b_t))]
    L_total = 기존 DINO+mean/cov 손실 + λ_bneck L_bneck

보조항은 step0부터 적용한다. predictor, 768d/8192d 추가 손실, 새 증강을 넣지 않는다.
기존 CE target/center·view별 covariance EMA·optimizer·teacher EMA 순서는 유지한다.
새 키가 없거나 λ=0이면 기존 total loss에 0을 더하는 연산도 하지 않는다.
명시적인 λ=0은 비교용 view cosine만 no_grad로 기록한다.

## 시작 전 gradient 보정

학습 seed42와 동일한 사전학습 초기값, 첫 epoch minibatch32, loop step0에서 한 번 측정한다.
teacher는 student의 deepcopy이며 양쪽 시작 초기값은 같다. 평가 데이터는 사용하지 않는다.

- DINO CE와 보조 cosine 각각이 backbone에 주는 L2 gradient norm g_D, g_B를 autograd.grad로 읽는다.
- λ_eq=g_D/g_B. 범위는 **[0.01 λ_eq, λ_eq]**, log 분포, 첫 보조항 trial의 λ는 **0.1 λ_eq**다.
- 초기 weighted gradient가 CE의 1%~100%인 구간을 탐색한다. 학습 내내 이 비율이라고 가정하지 않는다.
- center/covariance·teacher EMA를 갱신하지 않고 optimizer/scheduler step이나 .grad 축적도 하지 않는다.
- model/teacher/criterion·optimizer/scheduler·.grad와 Python/Torch/CUDA RNG의 상태 보존을 검사한다.
- 0/비정상 gradient이면 보정을 중단하고 임의 범위를 만들지 않는다.
- 실제 수치와 config/초기값/데이터 해시는 calibration.json, provenance에 기록한다.
  search_registration.json을 **스모크/탐색 전에** 저장한다. 결과를 본 뒤 범위를 바꾸지 않는다.

## 같은 예산의 Optuna 두 조건

| 공통 변수 | 범위 | 분포 |
|---|---|---|
| backbone LR | [3.4382477416615245e-5, 1.3752990966646098e-4] = 기준×[0.5,2] | log |
| head LR | [2.9124167925665e-4, 1.1649667170266e-3] = 기준×[0.5,2] | log |
| 최종 teacher 온도 | [0.10,0.16] | linear |

baseline은 위 3개만, bottleneck은 같은 3개와 보정한 λ를 탐색한다. 다른 모든 값은 기준선 고정이다.
각각 **20 trial ×4000 step**, 총 40 full-budget trial이다. 각 조건 trial0의 공통 값은 정확한 기존
기준 설정이며 baseline λ=0, experimental λ=0.1λ_eq다. 기존 완료 trial을 새 예산에 가져오지 않는다.
TPE seed20261104, startup5, multivariate, **pruning 없음**으로 동일 적용한다.
각 pair에서 baseline/experimental 하나씩 실행·완료한 뒤 다음 pair를 뽑는다.
GPU 한 장이면 순차 실행한다. runtime error는 중단·보고하며 단축 결과를 완료 trial로 쓰지 않는다.

탐색 학습 seed42는 양 조건에서 동일하다. 모든 run은 독립 사전학습 초기값에서 시작하고 weight resume하지 않는다.
source snapshot·commit/diff, resolved config, 실제 명령/exit code, 데이터 SHA, 초기 model SHA,
실제 batch index 순서 SHA를 기록하고 같은 seed의 비교 가능성을 검사한다.

예산을 바꾸려면 시작 전에 양쪽에 같은 trial 수와 이유를 기록해야 한다. 기본 예산은 줄이지 않는다.

## 스모크, 재검증, ablation, test

먼저 각 조건 60-step smoke로 step56을 넘기되 전체4000-step 스케줄을 압축하지 않는다.
양쪽 모두 통과해야 본 탐색을 시작한다. 스모크 점수는 후보 선택에서 제외한다.

완주한 각 조건의 768d STS-B dev best 상위3개를 고정하고 **새 seed101/102/103**에서 모두 4000 step 재학습한다.
총18회다. 각 조건에서 새3시드의 768d dev best 평균이 가장 높은 설정을 선택한다.
동률은 trial번호가 작은 설정을 고른다. checkpoint는 주기 평가의 dev 최고점, 동률은 최초 관측이다.
256d dev/STS7이나 cosine은 선택에 사용하지 않는다.

selection.json을 고정한 다음, 실험군 최종 설정에서 **λ_bneck만 0**으로 바꿔 같은3시드로
추가3회 학습한다. 다른 설정이 달라지지 않았는지 검사한다. 이 비교 결과로 최종 설정을 다시 고르지 않는다.
따라서 기본 full training 총61회(40+18+3)이며 calibration/smoke는 별도다.

그 후 최종 baseline/experimental/aux_off의 3시드×best/last에서 7-task test를 평가한다.
동일 forward의 768d 문장 임베딩이 주 평가, 256d bottleneck은 별도 진단 결과다.
태스크: STS12/13/14/15/16/STSBenchmark/SICK-R. test를 보며 후보·가중치·범위를 바꾸지 않는다.

## 가설과 판정

두 질문을 구분한다.

1. 같은 튜닝 예산: 선택된 experimental 대 선택된 baseline.
2. 같은 hyperparameter: 선택된 experimental 대 그 설정의 aux_off.

**최종 주 지표는 dev로 선택한 best checkpoint의 768d STS7 평균**이다.
각 비교에서 seed별 paired 차이를 구하고, 평균 Spearman Δ≥0.003(×100 점수로 +0.3점)이면서
3시드 중 최소2개 양수면 성립, 평균≤0이고 최소2개 비양수면 기각, 나머지는 판정불가다.
3시드 표본 SD와 개별 태스크·last·dev도 모두 공개하며 통계적 유의성을 주장하지 않는다.
dev는 checkpoint/설정 선택에 사용하고 dev 차이도 별도 보고한다.
256d 지표만 좋아지면 최종 임베딩 개선으로 결론 내리지 않는다.

GPU/데이터가 없으면 상태와 미실행 단계를 기록한다. CPU 장시간 학습이나 점수 추정으로 대신하지 않는다.
