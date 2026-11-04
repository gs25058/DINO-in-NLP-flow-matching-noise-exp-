# Bottleneck cosine + isotropy — 사전 등록

2026-11-04. 시작 소스 `d590993`, 별도 작업 브랜치 `exp/bottleneck-iso-paired-optuna`.
기존 cosine 탐색은 사용자의 중단 승인으로 종료했다. 기존 5쌍의 완료 trial과 부분 run을
`results/experiments/bneck_cosine_20261104`에 보존하며 새 탐색 예산으로 가져오지 않는다.
main 병합·원격 push를 하지 않는다. 이 명세는 새 GPU 보정·학습 전에 작성했다.

## 질문과 기준선

기존 768d iso와 DINO CE를 유지하면서, 정규화된 256d bottleneck에 직접 cosine과
iso를 함께 걸면 768d 문장 임베딩의 STS가 개선되는가? 동일 설정에서 iso만 제거했을 때
보다 좋아지는가? 이전 t3에서 관측한 cosine 상승·256d STS 하락은 동기이며,
iso로 해결된다는 증거 또는 붕괴 확정 판정으로 간주하지 않는다.

실제 R18 대조군 계보 `origin/exp/r18-ibot-token-ce` / `eed56a1`의
`configs/r18_bert_ctrl.yaml`에서 정리한 `configs/coviso_relax/base.yaml`을 사용한다.
`baseline_provenance.json`에 원본 계보·파일 해시를 보존한다. 재구성 기준선이 아니다.
과거 R18 dev 약0.7465/STS7 약66.09는 참고 기록이며 현재 서버 재현값으로 쓰지 않는다.

새 기본 config는 `configs/bottleneck_iso/base.yaml`이다. bert-base-uncased,
mean+L2 768d, MLP→L2 256d→8192 prototype head, batch32, noisy student3,
clean EMA teacher, 기존 flow interpolation·특수토큰 마스킹, max128, FP32+TF32,
4000 optimizer step, 데이터 순서·optimizer·teacher momentum·모든 스케줄은 기존과 같다.
iBOT·토큰 회귀·predictor·새 증강은 추가하지 않는다.

768d iso는 기존 위치·두 가중치 **5.874093568592943**·EMA momentum
**0.8195105695006707**·activation step **56**을 그대로 유지한다.
noise 값과 초기/최종 teacher 온도 스케줄 등 다른 고정 값도 변경하지 않는다.
기존 cosine 단독 실험의 테스트·실행 경로는 공통 루프에서 유지한다.

## 손실과 상태

캐시된 MLP→L2 벡터 b만 사용한다. 추가 forward나 predictor는 없다.

    L_cos = mean_views mean_batch(1 − dot(b_s, stopgrad(b_t)))
    mu = m * mu_ema.detach() + (1−m) * mean_batch(b_s)
    M2 = m * M2_ema.detach() + (1−m) * b_s.T @ b_s / B
    C = M2 − outer(mu,mu)
    L_iso256 = mean_views(||mu||² + ||C/max(tr(C),1e−8) − I/256||²_F)
    L_total = DINO + existing_iso768 + lambda_cos*L_cos + lambda_iso256*L_iso256

256d 통계는 768d와 독립인 `CovIsoPenalty` 인스턴스다. momentum과 activation56은
768d를 따른다. 활성 iso run에서 step56 전에도 세 학생 뷰 순서로 통계를 세 번 갱신한다.
각 손실은 갱신 전 EMA와 현재의 미분 가능한 통계를 사용한 뒤 no_grad로 갱신한다.
cosine은 step0부터, iso256은 step56부터 손실에 포함한다. mean/cov의 가중치는 하나로 묶고
차원별 추가 계수는 없다. teacher·과거 EMA에는 gradient가 흐르지 않는다.
768d covariance·DINO center·optimizer·teacher EMA의 기존 순서와 횟수는 바꾸지 않는다.

새 키 `loss.bneck_iso_lambda`가 없거나0이면 새 통계 모듈을 생성하지 않고 total에0을 더하지도 않는다.
cosine의 기존 zero/missing 동작도 그대로다. checkpoint에 활성 256d iso 상태를 별도 키로 저장하며
모델 weight 형식은 바꾸지 않는다. 모든 run은 사전학습 초기값에서 독립 시작한다.

## 시작 전 보정과 탐색

seed42의 동일 초기 checkpoint와 첫 minibatch32를 사용한다. CE·cosine·iso256 각각의
backbone gradient norm을 측정한다. iso는 **일회용 초기 통계**에서 세 뷰를 순서대로 계산하며,
측정 시만 activation을 무시한다. live objective의 통계는 전혀 갱신하지 않는다.
model/teacher/criterion/optimizer/scheduler/.grad와 Python/Torch/CUDA RNG의 보존을 검증한다.
head MLP의 gradient도 함께 기록한다. 0·비정상 gradient이면 중단한다.

각 추가항 j에 대해 lambda_eq_j = g_CE/g_j, log 범위 **[0.01 lambda_eq_j, lambda_eq_j]**,
시작 가중치 **0.1 lambda_eq_j**로 사전 고정한다. 두 항은 독립 탐색하므로 두 weighted norm의 합은
초기에 최대 CE의2배일 수 있다. 이는 gradient 방향·합성 norm의 예측이 아니다.
학습 도중 gradient 비율이 고정된다고 가정하지 않으며 실제 희소 진단값으로 확인한다.
보정 수치는 `calibration.json`, 탐색 범위는 학습 전 `search_registration.json`에 저장한다.

| 조건 | 탐색 변수 | 예산 |
|---|---|---|
| baseline: DINO+iso768 | backbone LR, head LR, 최종 teacher T | 20×4000 |
| bottleneck: 위 손실+cosine+iso256 | 같은 공통3개 + lambda_cos + lambda_iso256 | 20×4000 |

공통 공간은 두 조건 모두 backbone LR=기준×[0.5,2] log,
head LR=기준×[0.5,2] log, 최종 teacher T=[0.10,0.16] linear다.
기준 LR=6.876495483323049e−5, head LR=0.0005824833585133, 최종T=0.1347501197840043.
trial0은 공통값을 이 기준에 두고 추가항은 각각 보정 anchor를 사용한다. baseline의 추가항은 둘 다0.
768d iso·noise는 양쪽에서 고정한다. 기존 run/짧은 run을 4000-step 결과로 대체하지 않는다.

학습 seed42, TPE seed20261104, startup5, multivariate, pruning 없음.
각 조건 하나씩 완료한 뒤 다음 pair를 제안한다. GPU admission은 GPU_POLICY.md를 따르고
계정 전체 최대2개 작업, 한 장만 가능하면 순차 실행한다. 예산 축소는 시작 전 양쪽 동일하게
이유를 기록해야 하며 기본 예산은 축소하지 않는다.

## 검증·선택·ablation·최종 평가

수정 전 전체 테스트71개를 실행했다. zero/missing iso의 기존 손실·gradient·EMA·RNG 보존,
256/768 EMA 분리와 세 번 갱신, teacher/과거 통계 detach, student backbone/MLP gradient,
activation55/56 경계, 상태 보존 보정과 전체 실행기를 검증한다.
양 조건60-step GPU smoke를 먼저 완료한다. 4000-step 스케줄은 압축하지 않는다.

탐색 완료 각 조건 top3를 **새 시드101/102/103**에서 각각4000 step 재학습한다(18회).
평가는 loop step0/250/…/3750/3999이며, **768d STS-B dev Spearman best**로 checkpoint를 고른다.
동률은 최초 관측, 설정 선택은 새3시드 dev best 평균, 동률은 작은 trial번호다.
256d STS·cosine·rank나 test는 후보 선택에 사용하지 않는다.

`selection.json` 동결 후 최종 실험군에서 다음 두 조건을 같은3시드로 추가 실행한다(6회).

- `iso_off`: **iso256만0**, cosine과 나머지 hyperparameter는 그대로. iso 추가 효과 비교다.
- `aux_off`: **cosine만0**, iso256과 나머지는 그대로. cosine 추가 효과 비교다.

이 결과로 후보를 다시 선택하지 않는다. 총 full run **64회 = 40+18+6**, 보정과60-step smoke는 별도.
최종 baseline/bottleneck/iso_off/aux_off 각3시드×best/last의 STS7을 선택 동결 후 평가한다(24평가).
주 평가는 기존 추론용 **768d** 임베딩이다. 256d STS는 같은 forward의 진단으로 구분한다.
test를 보며 설정·범위·판정 기준을 바꾸지 않는다.

## 기록·가설·판정

모든 run의 resolved config, 실제 명령·옵션·exit, source commit/diff/hash,
초기 model hash, 데이터 identity/hash/행수, seed와 batch index 순서 hash를 남긴다.
CE, iso768 mean/cov, cosine, iso256 mean/cov 및 각 weighted contribution을 기록한다.
step0/56/250/1000/2000/3000/3999에서 backbone gradient,
CE/cosine/iso256의 MLP gradient를 측정한다. loss 절대값만으로 지배성을 판단하지 않는다.

768/256 alignment·uniformity와 고정 rank2048 문장의 centered effective rank·max SV ratio,
uncentered mean norm·평균 off-diagonal cosine을 함께 기록한다.
zero centered energy의 진단 rank는0으로 정의한다. uniformity −2.4 등 목표나 외부 붕괴 임계치를 강제하지 않는다.

H1: 동일 튜닝 예산의 bottleneck이 baseline보다 주 지표를 개선한다.
H2: 같은 설정에서 bottleneck이 iso_off보다 개선한다(iso의 조건부 효과).
H3: 같은 설정에서 bottleneck이 aux_off보다 개선한다(cosine의 조건부 효과).
주 지표는 **dev로 선택한 best checkpoint의 768d STS7 7-task 평균**이다.
각 비교의 seed별 paired delta 평균≥0.003이고 최소2/3 양수면 성립,
평균≤0이고 최소2/3 비양수면 기각, 그 외/미완료는 판정불가다.
각 seed의 dev best/last·7-task 개별 점수·평균·sample SD를 모두 보고한다.
3시드로 통계적 유의성을 주장하지 않는다. 병목만 개선되면 최종 임베딩 개선으로 결론 내리지 않는다.
H1은 두 추가항과 재튜닝의 합성 비교다. H2/H3는 선택된 한 설정의 조건부 효과이며
모든 설정의 인과 효과 또는 iso mean/cov 개별 효과를 입증하지 않는다.

GPU/데이터 부족 시 가능한 구현·테스트·실행 준비만 완료하고 미실행 사유를 기록한다.
장시간 CPU 대체나 미실행 점수 추정은 하지 않는다.
