# 실험2: 256d bottleneck DINO CE + isotropy

2026-11-05, `fffb47b`에서 `exp/bottleneck-dino256-optuna`로 분기했다.
사용자의 두 차례 정정을 반영한다. 실험2는 cosine+iso가 아니라 **DINO CE+iso**다.
실험1(DINO8192+iso768+cos256+iso256) 결과·미커밋 보고서와 기존 기준선을 보존한다.
main 병합·원격 push를 하지 않는다. 이 문서는 새 학습 전에 작성했다.

## 모델·손실

    BERT -> masked mean pooled768 -> MLP(768,768,256) -> L2 -> b256
    z768 = L2(pooled768)  # 주 평가용이며 이 조건의 직접 손실은 없음
    p_t = stopgrad(softmax((b_teacher - previous_center256) / teacher_T))
    L_DINO256 = mean_views mean_batch CE(p_t, softmax(b_student / student_T))
    L = L_DINO256 + lambda_iso * mean_views (L_mean256 + L_cov256)

8192차원 확장 파라미터 자체를 생성하지 않는다. normalized b256을 logits로 직접 재사용한다.
teacher는 clean 원문을 보는 frozen EMA copy이며 predictor·cosine·768d iso는 없다.
teacher center는256차원이고 기존 momentum0.92를 사용한다. target 생성 후 clean teacher
벡터의 배치 평균으로 한 번 갱신한다. temperature sharpening, student temperature 모두 유지한다.

iso는 기존 `CovIsoPenalty`의 통계·수식·trace clamp1e−8·detach를256차원에 그대로 적용한다.
평균/공분산의 가중치는 하나로 묶고, momentum0.8195105695006707와 activation56을 유지한다.
세 학생 뷰마다 손실을 계산한 뒤 통계를 순서대로 세 번 갱신한다. step56 전에도 통계는 갱신한다.
추가768d covariance 모듈은 생성하지 않는다. teacher와 과거 통계로 gradient가 흐르지 않는다.

기존 no-key config는 기존 모델·손실 경로와 수치를 유지한다. 체크포인트에는256d center와
covariance 상태를 저장하며, `model.head.logit_dim: null`로 확장 없는 모델을 정확하게 복원한다.

bert-base-uncased, mean+L2, max128, batch32, noisy student3, clean teacher,
기존 Gaussian flow noise와 마스킹·data epoch 순서·AdamW·EMA·스케줄·4000step·FP32/TF32를 유지한다.
추론 주 평가는768d mean+L2, 256d는 별도 진단이다. 두 공간의 점수를 혼용하지 않는다.

## 이미 튜닝된 FlowDINO 기준선: 재탐색 금지

사용자는 기준선이 이미 튜닝됐으므로 재탐색하지 말라고 지정했다.
`results/experiments/bneck_iso_20261104/selection.json`의 baseline trial0과
그 final_runs의 seed101/102/103 best/last 평가를 고정 참조한다. 실제 R18 계보의
config는 `configs/coviso_relax/base.yaml`, 정확한 mean/cov 가중치는5.874093568592943이다.

기준선 source snapshot, 데이터 SHA,4000-step 완료, 각 seed의 초기 full model hash를 확인한다.
CPU에서 **초기화만** 재구성하여 full hash가 과거 run과 같은지 검증하고, backbone+MLP 공유
파라미터 hash를 새 run의 초기값과 대조한다. 기존 로그·checkpoint·원본 점수는 수정하지 않는다.
기존6개 best/last STS7 결과를 참조로 복사하고 기준선4000-step 재학습·STS7 재평가는 하지 않는다.
기준선 test는 이미 알려져 있음을 명시한다. 새 모델의 test를 설정 선택에 사용하지 않는다.

양 조건60-step 스모크 중 기준선은 기존 source와의 동작 보존 확인용이며 재튜닝이 아니다.
GPU의 view RNG는 동일 seed에서 시작한다. head 초기화는 CPU에서 수행한다. 실제 첫 noisy view
hash와 seed별 데이터 순서 hash를 기록·비교한다. 별도 RNG reset이나 노이즈 알고리즘 변경은 하지 않는다.

## 새 구조만 Optuna20 trial ×4000step

| 탐색 축 | 범위 | 분포 |
|---|---|---|
| backbone LR | R18×[0.5,2] | log |
| head MLP LR | R18×[0.5,2] | log |
| student T | [0.0755,0.302] | log |
| teacher/student T 비율 r | [0.1,0.99] | log |
| iso mean/cov 공통 lambda | [0.001,10] | log |

final teacher T = r×student T. warmup initial T = min(기존0.0494051972253961, final T).
기존4000-step teacher warmup fraction은 그대로다. teacher가 student보다 낮은 온도를 쓰도록 한다.
출력 logit 공간이 바뀌므로 온도와 iso 가중치를 고정한 한 run으로 구조를 평가하지 않는다.
여기에는 cosine 보조항이 없고, iso 범위는 폭넓게 사전 고정했으므로 과거 cosine gradient 보정을 재사용하지 않는다.

trial0은 기존 R18 LR/head LR/student T/final teacher T/정확한 lambda를 시작점으로 넣되,
모델과 iso 위치는 새256d 조건이다. 이 단일 trial 점수로 최종 결론을 내리지 않는다.
TPE seed20261104, startup5, multivariate, pruning 없음. 학습 seed42, 독립 사전학습 시작,
모든 완료 trial은4000 optimizer step을 채워야 한다. 오류·중단·스모크를 full 결과로 세지 않는다.

baseline은 이번 budget에 포함하지 않고 기존 최적값을 고정한다. 이전 baseline 탐색과 새 구조의
탐색 축/범위가 다르므로 엄밀한 동일 탐색공간 비교나 한 변경의 인과 효과라고 주장하지 않는다.
이 비교는 **기존 최적 기준선 대 새 구조의 튜닝 후 성능**이다.

## 재검증·선택·평가

새20개 trial의768d STS-B dev best top3를 고정하고, 탐색에 쓰지 않은 seed101/102/103에서
모두4000step 재학습한다(9회). 각 config의 새3시드 dev best 평균으로 최종 설정을 고른다.
동률은 작은 trial번호다. checkpoint는 loop step0/250/…/3750/3999의 dev 최고, 동률은 최초다.

총 **29회 새 full training + 2회60-step smoke**, 최종 새 모델의3시드×best/last STS7 **6평가**다.
미튜닝 성분 제거 run을 구조별 최적 성능 비교로 사용하지 않는다. 기준선은 이미 완료된3시드를 쓴다.
`selection.json` 동결 후에만 새6개 test를 계산한다. test를 보며 범위·온도·iso·후보를 바꾸지 않는다.

STS12/13/14/15/16/STSBenchmark/SICK-R의 동일 가중 평균, 주 지표는 **dev로 고른 best의768d STS7**.
256d STS·alignment/uniformity·두 공간의 effective rank·mean norm은 진단이다.
새 run의 DINO CE·H_pt/KL/H_p_bar·iso mean/cov·weighted contribution·backbone/MLP gradient를 기록한다.
head가 없으므로8192d head STS를 계산하거나256d 점수로 이름만 바꿔 대체하지 않는다.

가설: 튜닝된 직접256d DINO+iso가 고정된 FlowDINO 기준선을 개선한다.
동일3시드 paired STS7 delta 평균≥0.003이고 최소2/3 양수면 성립,
평균≤0이고 최소2/3 비양수면 기각, 그 외/미완료면 판정불가다.
개별 시드·task·best/last·표본SD를 보고하며3시드로 통계적 유의성을 주장하지 않는다.
Uniformity 목표값·다른 서버의 붕괴 임계치는 강제하지 않는다. 과거점수를 맞추는 튜닝은 하지 않는다.

## 실행과 증거

```bash
uv run --no-sync python scripts/run_bottleneck.py --direct-bottleneck-dino \
  --campaign results/experiments/bneck_dino256_20261105 --gpus auto,auto \
  --reference-baseline results/experiments/bneck_iso_20261104 --trials-per-arm 20
```

`GPU_POLICY.md`의 계정 최대2개·차단 소유자·공통 admission을 적용한다. 한 장이면 순차 실행한다.
소스 commit/diff/hash 고정, resolved config, 실제 명령·exit, 데이터/초기값/첫 view/batch 순서
hash와 seed를 남긴다. 기존19축 결과와 모든 미커밋 변경은 보존한다.
GPU/데이터가 없으면 구현·테스트·스모크 준비를 마치고 대기/미실행 이유를 기록한다.
장시간 CPU 학습이나 미실행 점수 추정으로 대체하지 않는다.
