# 공분산 등방성 정규화 완화와 구성 항 분리 — 사전 등록

**프로토콜 전환:** 2026-10-28 사용자 요청으로 고정 매트릭스를 중단했다.
현재 실행 명세는 [공동 Optuna 최적화](OPTUNA_SPEC.md)다. 아래는 원래 사전 등록 기록으로 보존한다.

등록일: 2026-10-28 (Asia/Seoul). 새 실험을 위한 별도 명세이며 기존 METHOD.md는 변경하지 않는다.
작업 시작: `536eea1`, clean `exp/coviso-ablation-sts7`에서 `exp/coviso-component-relaxation` 생성.
main 병합·원격 push 없음. 실행 전에 이 명세와 소스 스냅샷의 해시를 고정한다.

## 1. 실제 기준선

`origin/exp/r18-ibot-token-ce`의 `eed56a1a9f669129a8dbd428443f563ccec35074`에서
`configs/r18_bert_ctrl.yaml`과 상속 계보를 직접 해석했다. 정확한 resolved 원본은
`configs/coviso_relax/r18_reference.yaml`, 파일별 출처는 `baseline_provenance.json`이다.
재구성 기준선이 아니다. 과거 0.7465 / 66.09는 RTX 6000 Ada의 참고값이며 현재 재현값이 아니다.

λ₀ = **5.874093568592943**, covariance EMA momentum = **0.8195105695006707**, 활성 시작 = step **56**.
bert-base-uncased, mean + L2, batch 32, 학생 뷰 3개, clean EMA teacher, 기존 anchor flow noise.
4000 step, epoch 순회, 길이 128, eval 250 step마다 및 마지막 step 3999.
R18 실행기의 `--batch-views --tf32`를 모든 arm에 동일 적용한다. AMP는 사용하지 않는 FP32 학습이다.
LR·head LR·온도·모멘텀·스케줄·노이즈의 모든 정확한 값은 resolved 기준선을 따른다.
iBOT·token latent·mask/cutoff/correlated noise·velocity·KoLeo는 사용하지 않는다.
원본의 iBOT 전용 inactive 설정은 실험 base에서 명시적으로 off/제거한다.

현재 코어에 없는 R18 가속 옵션만 기본 off인 선택 기능으로 이식한다. 실험 기제는 이식하지 않는다.
각 seed는 동일 사전학습 파일과 seed별 동일 초기 head에서 독립적으로 시작한다. 학습 체크포인트 resume 금지.

## 2. 손실과 호환성 계약

L_mean = ||μ_blend||², L_cov = ||C/tr(C) − I/D||²_F.
L = L_DINO + λ_mean L_mean + λ_cov L_cov.

실제 기존 순서는 각 학생 뷰에서 현재 배치 평균·2차 모멘트를 계산하고 **갱신 전** detached EMA와
블렌드 → trace clamp_min(1e-8)로 공분산 정규화 → 두 loss 산출 → no_grad EMA 갱신이다.
뷰는 0,1,2 순서로 처리하여 step당 **3회** 갱신한다. step 56 전에도 EMA를 갱신하지만 loss에는 더하지 않는다.
성분 분리는 같은 forward의 미분 가능한 scalar를 재사용하며 EMA 갱신을 추가하지 않는다.
교사·과거 EMA에는 gradient가 흐르지 않는다. 각 뷰 손실을 평균한 뒤 가중한다.

`loss.cov_iso_mean_lambda`, `loss.cov_iso_cov_lambda`는 각각 명시된 값(0 포함)이 우선이다.
키가 없거나 null이면 상속 완료된 `loss.cov_iso_lambda`로 fallback한다. legacy 값도 없으면 0이다.
새 키/스케줄이 없으면 기존 `(평균한 L_mean+L_cov) * cov_iso_lambda` 연산 순서를 유지한다.
두 resolved 가중치가 같을 때도 같은 결합 순서로 계산한다. 두 초기 가중치가 0이면 기존처럼 모듈을 호출하지 않는다.
스케줄로 가중치가 줄어도 초기 활성 모듈의 EMA 갱신 횟수는 바뀌지 않는다.

성분별 schedule 키는 `cov_iso_mean_schedule`, `cov_iso_cov_schedule`이며 null/미지정이면 상수다.
cosine start_frac=0.5, end_frac=1.0, min_ratio=0.1: indices 0..1999는 유지,
step 2000에서도 시작 가중치, step 3999에서 정확히 0.1배. 경계 밖은 clamp한다.

## 3. 고정 가중치 4 × 2 매트릭스

| arm | λ_mean | λ_cov | seed |
|---|---|---|---|
| control | λ₀ | λ₀ | 42,43 |
| both1 | 1 | 1 | 42,43 |
| cov1 | λ₀ | 1 | 42,43 |
| mean1 | 1 | λ₀ | 42,43 |

실험별 차이는 config로만 표현하며 학습 루프를 복제하지 않는다. 두 GPU 슬롯이 허용되면
서로 다른 GPU에 독립 run을 하나씩 실행하며, 한 장만 가능하면 순차 실행한다. 계정 전체 2개 제한 준수.

## 4. 가설, 판정과 조건부 감쇠

- H1: both1이 control보다 STS-B dev best 평균을 ≥0.003 높이고 paired 차이가 두 seed 모두 양수이며,
  last 평균 감소가 0.002를 넘지 않는다 → 초기 지지.
- H2-cov/H2-mean: cov1/mean1에도 각각 같은 기준을 적용한다. 이 단일 항 개입이 해당 항의 완화 근거다.
- 손실 크기·uniformity·effective rank만 좋아졌다는 이유로 가설을 지지하지 않는다.
- 위 조건을 통과하지 않고 두 seed가 모두 비양수이면 초기 기각, 혼합된 결과/작은 차이/미완료는 판정불가.

fixed matrix가 끝난 후 dev best 평균이 가장 높은 완화 arm을 고정 후보로 택한다(동률은 arm 이름순).
이 후보가 위 지지 기준을 만족할 때만, 후보에서 완화된 항 중 H2 단일 항 근거가 있는 항을 감쇠한다.
각 근거 확보 항을 **하나씩** 감쇠하는 arm을 만들어 seed 42,43에서 대응 상수 후보와 비교한다.
두 항 동시 감쇠는 하지 않는다. 둘 다 근거가 없으면 감쇠하지 않고 이유를 보고한다.
모든 감쇠는 앞 50% 유지, 뒤 50% cosine, 마지막 0.1배로 고정하며 다른 탐색은 하지 않는다.
감쇠가 대응 상수 후보보다 낫다는 판정에도 동일한 ≥0.003 / 두 paired 양수 / last 감소 ≤0.002 기준을 쓴다.

fixed/허용된 decay 중 dev best 평균이 가장 높은 완화 arm을 최종 후보로 동결한다.
control과 최종 후보에 seed 44를 추가한다(후보가 약해도 재검증하여 부정 결과를 확인).
최종 지지: 3시드 best paired 평균 ≥0.003, 3개 중 최소 2개 양수, last 평균 감소 ≤0.002.
3시드 best paired 평균 ≤0이며 최소 2개 비양수이면 기각, 그 외는 판정불가.
이는 작은 표본의 운영 판정이며 통계적 유의성 주장이나 모집단 CI가 아니다.

## 5. 평가와 진단

dev best는 동일 eval cadence에서 **학생 final embedding의 STS-B dev 최고값**, 동률은 이른 step이다.
R18 원본은 last만 저장했다. best 저장을 모든 arm에 동일하게 추가하고 last도 함께 보존한다.
기존 post-update step 표기를 유지하며 optimizer_steps=loop_step+1을 함께 기록한다.
dev 선택과 seed 44 검증을 끝내고 `selection.json`을 동결한 뒤에만 best/last의 STS7 test를 한 번 평가한다.
test 결과로 arm·스케줄·선택 규칙을 바꾸지 않는다. 평가도 train과 같은 TF32, mean+L2, 길이 128이다.

로그: L_DINO, L_mean, L_cov, λ_mean, λ_cov, 적용 후 성분 기여, alignment, uniformity,
effective rank, dev score. 진단 steps={0,56,250,1000,2000,3000,3999}에서
autograd.grad(retain_graph=True)로 DINO와 각 가중 성분의 backbone gradient norm을 각각 측정한다.
진단은 .grad를 바꾸거나 EMA를 다시 갱신하지 않는다. 기록만 추가하고 학습 경로는 바꾸지 않는다.
uniformity -2.4를 목표로 하지 않는다. collapse guard는 모든 arm off이며 NaN/Inf는 명시적 실패다.

각 run: resolved YAML, 소스 commit·diff·파일 해시, frozen source, 시드, 데이터 경로·SHA-256·문장 수,
환경/라이브러리/GPU 정보, 실행 명령·정밀도·옵션, 초기 student 가중치 해시를 보존한다.
미완료 run은 건너뛰어 새 run처럼 이어붙이지 않고 별도 실패 상태로 남긴다.

## 6. 두 번째 실험에 넘길 산출물

기준선 경로: `configs/coviso_relax/control.yaml`.
상수 완화 경로: `configs/coviso_relax/{both1,cov1,mean1}.yaml`.
선택된 완화 설정은 campaign의 `selected_relaxation.yaml`로 복사하고 원래 arm/선택 근거를 기록한다.
선택된 run은 다음 실험의 config 기준선으로만 재사용하며, 사전학습부터 새로 시작한다.
결과가 불충분하면 완화를 권하지 않고 control을 기본으로 명시한다.
