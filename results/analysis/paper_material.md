# 논문 재료 추출 — FlowDINO-Text (2026-09-16 기준)

repo 전체(main + exp/* 브랜치 + 로컬 보고서)에서 논문으로 쓸 수 있는 것만 골라 정리한다.
수치는 전부 실측 로그·Optuna DB에서 재추출했고 출처를 병기한다. 규모: 1500-step 완주 run 215개,
Optuna trial 229개, 단위 테스트 67개, src 1,395줄.

---

## 0. 한 문장 요약

**DINO 자기증류를 문장 임베딩에 쓸 때 dropout 뷰를 flow-matching식 가우시안 노이즈 뷰로 바꾸면
"피크 후 하락"이 사라지고, 이후 성능은 손실항·teacher 온도·뷰 구조가 아니라 노이즈 분포(t 범위)와
LR warmup 길이가 결정한다** — 그리고 이를 4개의 사전 등록된 음성 결과로 뒷받침한다.

---

## 1. 논문의 뼈대

### 1-1. 핵심 주장 (positive)
1. 노이즈 뷰가 dropout 뷰의 자기샤프닝 폭주를 억제한다 (§2-1)
2. 성능을 올린 것은 **증강 분포**와 **스케줄 길이**뿐이다 (§2-2, §2-3)
3. 분포 통계량 정규화(cov_iso)는 쌍별 반발(KoLeo)과 학습 동역학이 실제로 다르다 (§2-4)

### 1-2. 핵심 주장 (negative, 사전 등록)
4. teacher 샤프닝은 판별 엔진이 아니다 — 신호를 24배 되살려도 alignment 불변 (§3-1)
5. pooled 공간 정렬 항은 포화가 원인이 아니라 형태가 문제다 (§3-2)
6. "iid 노이즈가 pooling에서 소멸한다"는 통념이 틀렸다 — 84% 통과 (§4-2)

### 1-3. 방법론 기여
7. 샤프닝의 설계 좌표는 τ가 아니라 H(p_t)다 — 개루프 3.4~4.4배 초과, 폐루프 정확 도달 (§4-1)
8. 학습 없는 뷰 난이도 측정이 학습 중 난이도를 예측한다 (§4-2)
9. DINO head는 STS 정보를 일관되게 잃으며 원인은 uniformity다 (§4-3)

---

## 2. Positive 결과 — 표와 수치

### 2-1. 피크 후 하락의 소거 (프로젝트 성공 기준)

| | 피크 STS-B dev | 피크 시점 | 이후 하락 |
|---|---|---|---|
| dropout DINO (외부 기준선, 기존 실험) | 70.6 | step 250 | 하락 (250 이후) |
| **노이즈 DINO r8** (2시드) | 71.83 / 70.90 | step 1250 | **−0.1 / −0.2** |

CLAUDE.md의 사전 등록 성공 기준("피크 ≥ 70 이면서 피크 후 500 step 하락폭이 기존보다 작음")을
두 시드 모두 충족. 출처: `results/analysis/schedule_tuning/report.md` §1.

이론적 근거(METHOD.md §1): dropout 뷰는 두 뷰가 사실상 동일해 학습 신호가 온도 차이에서만 나오고
loss 하락 ≈ H(p_t) 하락이 되어 과샤프닝→rank 감소로 간다. 노이즈 뷰는 KL(p_t‖p_s)에 t에 비례하는
소거 불가능한 바닥을 만든다.

### 2-2. 노이즈 분포가 지배적 레버다 (+0.021)

Optuna 40 trial, 5차원(t_lo, 폭, curriculum 시작비, warmup_frac, 뷰 수), base r8.
출처: `results/analysis/tune_r8_bert_coviso_sched_optuna_t50_flow_noise_t_trials.csv`

| 축 | STS 상관 (n=40) |
|---|---|
| **t 범위 폭** | **+0.388** |
| t_lo | −0.362 |
| t_max | −0.003 |
| warmup_frac (curriculum) | −0.226 |
| 뷰 수 | +0.104 |

상위 10 trial은 전부 t_max ∈ [0.77, 0.88], 폭 ∈ [0.37, 0.60]. 기존 챔피언 U(0.35, 0.50)은
상한이 낮고 폭이 좁았다. **t를 좁게 고정하지 말고 넓게 흩뿌려야 한다.** curriculum은 여전히 불필요.

2시드 검증 (상위 5개 전부 승격):

| config | t 범위 | STS (2시드) | r8 대비 |
|---|---|---|---|
| **t35 (새 챔피언)** | U(0.392, 0.881) | **0.7333** (0.7309/0.7357) | **+0.021** |
| t16 | U(0.231, 0.827) | 0.7323 | +0.020 |
| t33 | U(0.242, 0.803) | 0.7317 | +0.020 |
| t38 | U(0.407, 0.775) | 0.7302 | +0.018 |
| t12 | U(0.224, 0.799) | 0.7293 | +0.017 |
| r8 (구 챔피언) | U(0.35, 0.50) | 0.7120 (0.7173/0.7066) | — |

5/5가 시드 편차(≤0.005) 밖에서 이겼다. 우연한 한 점이 아니라 영역 전체가 잘못 잡혀 있었다.

### 2-3. warmup이 4~6배 길었고 범인은 LR 스케줄이다 (+0.034)

Optuna 60 trial, 8차원(값·기간·방식), base r7 cov_iso. **60개 중 55개가 base를 넘겼다.**
출처: `results/analysis/schedule_tuning/report.md`

| 스케줄 | 이전 | 튜닝 후 |
|---|---|---|
| LR warmup | frac 0.30 (step 450) | **0.052 (step 77)** |
| teacher_temp warmup | frac 0.30 (step 450) | 0.079 (step 118) |
| momentum ramp | 전 구간 | frac 0.41 (step 611), 종착 0.9995→0.9985 |

두 warmup이 같은 frac 0.30으로 묶여 있어 step 450의 전환이 어느 쪽인지 교락돼 있었다.
독립 탐색으로 분리: **LR warmup 기간 상관 −0.82 vs teacher_temp warmup −0.37.**
영향력 순위: 기간 > 값 > 방식(linear/cosine 차이 0.005~0.008로 무의미).

방법론 교훈(논문에 쓸 만함): **상위 1개만 검증하면 틀린다.** 750-step 1위(t18)가 1500-step에서
2·3위에 역전됐다. 이 프로젝트에서 750-step 순위와 1500-step 순위의 Spearman은 +0.19에 불과했다.

### 2-4. 분포 통계량 정규화 vs 쌍별 반발 — 기전 차이가 실재한다

METHOD.md §9의 정체성 기준: 표본이 표본을 직접 미는 항(KoLeo, O(B²) 쌍별 gradient)은 금지,
배치 요약 통계량 하나에 페널티가 걸리는 항(centering, uniform_push, **cov_iso**)만 허용.
근거: 쌍별 반발은 InfoNCE와 기전이 같아져 "노이즈 증강만의 이득"과 섞인다.

| | KoLeo (쌍별) | cov_iso (분포 통계량) |
|---|---|---|
| eff_rank 개선 (r5d 221 →) | +111 (→332) | +22 (→243) |
| 후반 alignment | 갉아먹음 (koleo decay로도 미해결) | 유지 |
| Optuna 최적 (750-step) | 0.6822 | **0.703** |
| 최종 계보 위치 | 탈락 | **채택 (r7→r8→r12)** |

cov_iso는 KoLeo의 1/5~1/3만 펴면서 alignment를 해치지 않는다. 수동 grid(λ≤2)에서는 게이트가
걸렸으나 Optuna(λ≤8, cov_ema_momentum 낮춤)에서 뒤집힘 — **"게이트는 λ 범위 아티팩트"**라는
교훈도 방법론 항목으로 쓸 수 있다. 출처: `results/analysis/coviso/report.md`, 메모리
`project_r7_coviso_gated`.

수식(loss.py CovIsoPenalty): 배치 평균 μ와 2차 모멘트 M₂를 EMA로 추적하되 detach된 EMA를 현재
배치 통계와 같은 momentum으로 재블렌드 → gradient는 (1−m) 몫으로만 흐른다.
L = ‖μ_blend‖² + ‖C_norm − I/D‖²_F. 최초 스펙의 ×D 스케일링은 300배 과대 gradient로 즉시
rank-1 붕괴를 일으켜 제거(캘리브레이션 dry-run 기록 있음).

### 2-5. SimCSE와의 비교 — 같은 예산에서는 앞서고, 전체 예산에서는 뒤진다

평가 파이프라인 검증: 공식 SimCSE-BERT-base 체크포인트를 우리 코드로 재측정해 논문 avg7
76.25 → **76.29**(태스크별 소수점 2자리 일치). 출처: `results/analysis/simcse_compare/report.md`.

| 조건 | SimCSE | 우리 (챔피언) |
|---|---|---|
| **동일 예산** (batch 32, 1500 step, mean pooling) | 0.6042 | **0.7333** (+0.129) |
| SimCSE 원본 설정의 step 1500 (batch 64) | 0.7303 (cls) | 0.7333 |
| SimCSE 원본 완주 (1 epoch) | **0.7980** (dev) / 76.29 (avg7) | — |
| 7-task avg (mean pooling) | 76.29 | 64.23 |
| 7-task avg (first_last pooling) | — | 65.26 |

표현 구조가 정반대다:

| | alignment ↓ | uniformity ↓ | eff_rank ↑ |
|---|---|---|---|
| SimCSE | **0.221** | −2.43 | 225 |
| 우리 | 0.297 | **−3.40** | **301** |

**우리는 잘 펴고, SimCSE는 잘 붙인다.** 남은 STS 격차는 전적으로 alignment에 있다.
이것이 §3의 음성 결과를 해석하는 열쇠다.

---

## 3. Negative 결과 — 사전 등록·2시드·기제 확인까지 된 것만

이 프로젝트의 가장 강한 부분이다. 넷 다 "구현이 약해서"가 아니라 **기제가 확실히 걸린 상태에서**
효과가 없음을 보였다.

### 3-1. 후반 teacher 샤프닝 ≠ 판별 엔진 (R13-A)

가설: DINO의 prototype softmax ≡ InfoNCE 표본 softmax이므로 teacher 온도 인하 = 판별 해상도
상승 = alignment 회복. 원본 DINO는 τ 0.04→0.07의 날카로운 설계점, 우리는 붕괴 때문에 0.135
plateau. cov_iso가 응축을 막는 지금 후반 재샤프닝이 판별 효과만 취할 수 있는가?

| arm (r8 base, 2시드) | ΔH(p_t) 달성 | KL(p_t‖p_s) plateau | alignment | STS |
|---|---|---|---|---|
| r8 기준 | 0 | 0.0025 | 0.2838 | 0.7120 |
| 개루프 τ→0.07 | −0.226 | 0.0175 | 0.2823 | 0.7115 |
| **개루프 τ→0.05** | **−0.664** | **0.0603 (24×)** | 0.2816 | 0.7087 |
| 폐루프 δ0.066 | −0.066 | 0.0062 | 0.2830 | 0.7123 |
| 폐루프 δ0.152 | −0.149 | 0.0138 | 0.2810 | 0.7118 |

- 기제는 의심 없이 걸렸다: H(p_t)가 균등 상한(ln 8192 = 9.011)에서 0.024 떨어진 plateau를 완전히
  벗어나 −0.664 nats, 학습 신호 KL은 **24배** 회복.
- **alignment는 0.001 단위로도 안 움직였다.** 개입 구간 궤적이 r8과 겹친다.
- 붕괴 없음(eff_rank 301). cov_iso의 안전 여유는 가정보다 훨씬 크다.
- 챔피언 base로 옮겨 10 trial 튜닝해도 **중앙값 0.7328 = 챔피언 0.7333**, corr(delta, STS) = −0.696.
  개입이 셀수록 나쁘다.

**결론: "후반에 teacher가 줄 정보가 없어서 정체한다"는 가설 기각.** 정보를 되살려도 표현은 안 바뀐다.
출처: `.claude/worktrees/r13-sharpen-views/results/analysis/r13/report.md` §6-1.

### 3-2. BYOL식 pooled 정렬 항 — 크기·난이도가 아니라 형태의 문제 (R10)

| arm (r8 base, 2시드) | STS | alignment | pos_cos_raw 종착 |
|---|---|---|---|
| r8 기준 | 0.7120 | 0.2838 | — |
| λ=0.5 | 0.6785 | 0.3368 | 0.978 |
| λ=0.1 | 0.6996 | 0.2994 | 0.975 |
| λ=0.1 + t 제어기 (밴드 [0.85,0.90] 유지) | 0.6901 | 0.3029 | 0.89 (유지됨) |

λ 0.5와 0.1의 pos_cos_raw 궤적이 거의 겹치고 둘 다 0.975 포화 — **λ는 도착점이 아니라 속도만
바꾼다.** 포화가 원인인지 보려고 t 제어기로 pos_cos_raw를 [0.85, 0.90]에 붙들었더니(제어기는 정확히
작동, t 0.425→0.63 평형) alignment가 0.299→0.303. **포화는 원인이 아니다.**
출처: `results/analysis/r10_align_tctrl/report.md`.

### 3-3. 주기적 head 재초기화 (R9)

sync 리셋(student+teacher+center)은 두 head logit이 같아져 KL이 공짜로 충족된다 — 실측 리셋 직후
KL 0.0043→0.0075, step 0의 0.127에 한참 못 미침. student만 리셋하면 KL은 튀지만 3000-step
2시드에서 0.6970 vs 대조군 0.7042. 출처: `results/analysis/r9_head_reinit/report.md`.

### 3-4. 상관 노이즈는 구조로 작동하지 않는다 (R13-B)

pooled 난이도를 맞춘 쌍 비교 (§4-2의 B0 측정으로 설계):

| pooled | iid 대조군 | 구조 arm | 판정 |
|---|---|---|---|
| ~0.91 | iid U(0.45,0.55) **0.7077** | rho 0.5 0.7017 | **iid 승** |
| ~0.88 | iid U(0.50,0.60) 0.6920 | rho 1.0 **0.7043** | 구조 승 |
| ~0.88 | iid U(0.50,0.60) 0.6920 | cutoff 0.1 **0.7109** | **구조 승 (+0.019)** |

cutoff만 유의미하고(alignment −0.026, uniformity·rank도 동시 개선 — 매트릭스에서 유일),
상관 노이즈는 약한 설정에서 같은 난이도의 iid보다 나쁘다. 단 어느 arm도 r8을 넘지 못했고,
챔피언 base 튜닝 4 trial도 전부 미달(최대 0.7248). 뷰를 어렵게 해도 KL 바닥은 0.0033~0.0041로
거의 안 오른다(샤프닝의 0.0603과 대비).

### 3-5. 넷을 합친 해석

두 개의 서로 다른 "후반 정체 원인" 가설이 각각 반증됐다: (a) 신호 부족 — 24배 되살려도 무반응,
(b) positive가 쉬움 — 난이도 0.943→0.893으로 내려도 STS 하락. 남는 해석은 **이 목적함수가
alignment를 만드는 기제를 갖고 있지 않다**는 것이고, SimCSE와의 격차가 정확히 alignment라는 §2-5와
맞물린다. 이번에 STS를 올린 유일한 것이 t 분포(같은 목적함수 안의 증강 분포 이동)였다는 사실이
이를 뒷받침한다.

---

## 4. 방법론 기여

### 4-1. 샤프닝의 설계 좌표는 τ가 아니라 H(p_t)다 — 실증

정적 체크포인트에서 잰 H(τ) 곡선으로 개루프 종점을 정했더니 학습 중 **logit 스케일이 자라** 같은 τ가
훨씬 낮은 H를 만들었다:

| arm | 목표 ΔH | 실제 ΔH | 오차 |
|---|---|---|---|
| 개루프 τ→0.07 | −0.066 | −0.226 | **3.4×** |
| 개루프 τ→0.05 | −0.152 | −0.664 | **4.4×** |
| 폐루프 δ=0.066 | −0.066 | −0.066 | ✓ |
| 폐루프 δ=0.152 | −0.152 | −0.149 | ✓ |

폐루프(EntropyCtrl, loss.py)는 anchor에서 cosine으로 목표를 내리고 τ ← clip(τ·exp(−gain·err))에
스텝당 1% 상대 변화 제한. 목표 구간 종료 후 H가 아래로 새자 τ를 0.0926→0.1049로 되올려 교정.
gain은 측정 H(τ) 곡선 위 시뮬레이션으로 정했다(스펙 0.02는 +0.023 nats 정상오차, 0.3 채택).

### 4-2. "iid 노이즈는 mean pooling에서 소멸"은 틀렸고, 구조는 더 효율적이다

`scripts/measure_view_hardness.py` — 학습 없이 512문장에 대해 student 뷰 vs 깨끗한 뷰의 pooled
코사인과 토큰 코사인을 잰다.

| 설정 | pooled | 토큰 | 토큰 손상량 |
|---|---|---|---|
| iid t=0.425 | 0.9450 | 0.8410 | |
| iid t=0.63 | 0.7712 | 0.6347 | |
| **변화** | **−0.174** | −0.206 | pooled가 토큰의 **84%** |

1/L 소멸 논증은 pooled 출력에 직접 노이즈를 더할 때만 성립한다. 입력 임베딩에 넣어 12층
비선형 인코더를 통과시키면 각 토큰의 섭동이 attention을 바꾸며 전파된다.

| pooled 수준 | 설정 | 토큰 손상량 |
|---|---|---|
| 0.91 | iid U(0.45,0.55) | 0.229 |
| 0.91 | rho=0.5 | **0.159** |
| 0.88 | iid U(0.50,0.60) | 0.280 |
| 0.88 | rho=1.0 / cutoff 0.1 | **0.159 / 0.163** |

같은 pooled 난이도에 iid는 토큰을 44~72% 더 망가뜨린다. 그리고 **이 사전 측정이 학습 중
pos_cos_raw를 예측했다**(cutoff 0.8894 예측 → 0.8964 실측, rho0.5 0.9138 → 0.9175). 학습 없이
뷰 난이도를 설계할 수 있다.

### 4-3. DINO head는 STS 정보를 일관되게 잃으며, 잃는 것은 uniformity다

r8 챔피언에서 세 readout 공간을 같은 backbone forward로 동시 평가 (`src/evaluate.py` EVAL_SPACES):

| 공간 | STS | alignment | uniformity |
|---|---|---|---|
| Final embedding (mean-pool) | **0.7173** | 0.2834 | **−3.470** |
| head bottleneck (256-d) | 0.6946 | 0.2374 | −3.148 |
| head logits (8192-d) | 0.6840 | **0.2271** | −3.033 |

학습 내내 순서가 한 번도 안 뒤집혔다. head 공간은 alignment가 **더 좋은데**(DINO loss가 직접
최적화하는 공간이니 당연) uniformity가 나빠 STS가 낮다. R10의 align 항은 이 격차를 0.033→0.050으로
벌렸다. "평가는 Final embedding으로"라는 선택의 사후 근거이자, DINO head가 문장 표현으로서
무엇을 잃는지에 대한 첫 정량.

### 4-4. 실험 설계 관행 (부록/방법론 절)

- **사전 등록**: R9~R13 전부 P-*/P-S*/P-V* 예측을 실행 전에 박고 판정. 성공/실패 시그니처를 미리 고정.
- **난이도 짝맞춤 대조군**: 구조 vs 난이도를 가르기 위해 pooled 난이도를 맞춘 iid 대조군 배치.
- **붕괴 감시**: H(p̄_t)/eff_rank/batch_KL을 base 자신의 plateau 기준 상대 임계로 감시, 연속 3회 위반 시
  중단, 중단 trial은 pruned가 아니라 마지막 유효 점수로 채점(TPE가 붕괴 영역을 학습하게).
  **교훈: batch_KL은 증강 설정 간 비교 불가** — t 범위가 넓어지면 절반으로 떨어지면서 STS는 오른다.
- **다중 시드 규칙**: 단일 시드 결론 금지. 7-task 평균은 2시드로 ±1점 불확실성(SICK-R이 3.4~3.8점).
- **1500-step 튜닝 원칙**: frac 기반 스케줄 파라미터는 절대 step으로 전이되지 않는다.

---

## 5. 한계 — 논문에 정직하게 써야 할 것

1. **SimCSE 원본 예산에는 못 미친다**: dev 0.7333 vs 0.7980(−0.065), avg7 64.2 vs 76.3(−12).
   우리 방법을 SimCSE 예산(1 epoch)으로 늘려본 적이 없다 — 가장 명백한 공백.
2. **격차는 alignment다**: 0.297 vs 0.221. uniformity·rank는 우리가 낫다. 이 목적함수에서
   alignment를 얻는 방법을 못 찾았다(§3).
3. **BERT-base 단일 backbone**, 시드 2개, 1500 step.
4. **후처리 없는 raw 수치로 보고**해야 한다. first_last pooling(+1.03 avg7)과 center_pc2(평가 셋
   통계 사용)는 내부 비교용이지 SimCSE 공식 수치(후처리 없음)와 나란히 놓을 수 없다.
5. 원 사전 등록 P1~P5(METHOD.md §7)는 부분 판정만 남아 있다(P2는 3중 재검증됨,
   `results/analysis_v2.md`). 논문에 넣으려면 P1/P3/P4를 현 챔피언에서 재판정해야 한다.
6. 개루프/폐루프 "경로 의존성" 비교는 개루프가 목표를 빗나가 설계대로 성립하지 못했다.

---

## 6. 논문 형태 제안

**제목 후보**: "Where the Gains Come From: Augmentation Distribution, Not Loss Engineering, in
DINO-style Sentence Embedding" — 또는 음성 결과 중심으로 "What Doesn't Help DINO-for-Text:
Four Pre-registered Negative Results".

**본문 표**: (1) 계보표 §2-2 하단 확장(R6→R7→R8→R12), (2) R13 매트릭스 §3-1+§3-4,
(3) SimCSE 구조 대비 §2-5, (4) head 공간 §4-3.

**본문 그림**: (1) 노이즈 튜닝 40 trial 산점도(폭 vs STS, t_max 색), (2) H(p_t)·τ·alignment 궤적
r8 대비 겹침(개루프/폐루프), (3) pooling 통과율(§4-2 표를 막대로), (4) SimCSE vs 우리의
alignment–uniformity 평면 위치.

**핵심 메시지 순서**: 하락 소거(§2-1) → 이득은 분포·스케줄(§2-2, 2-3) → 손실항·온도·뷰는 무효(§3)
→ 그래서 남은 것은 alignment이고 목적함수 자체의 문제(§3-5, §2-5) → 방법론(§4).

---

## 부록: 수치 출처 색인

| 수치 | 파일 |
|---|---|
| 계보 STS/align/unif/rank | `results/logs/<run>_s{42,43}/train.log` 최종 EVAL |
| 노이즈 튜닝 40 trial | `results/analysis/tune_r8_bert_coviso_sched_optuna_t50_flow_noise_t_trials.csv` |
| 스케줄 튜닝 60 trial | `results/analysis/tune_r7_bert_coviso_optuna_t35_r5_schedules_full_trials.csv` |
| R13 매트릭스·튜닝 | `.claude/worktrees/r13-sharpen-views/results/analysis/r13/report.md`, `*.db` |
| B0 난이도 측정 | `.claude/worktrees/r13-sharpen-views/results/analysis/r13/view_hardness.json` |
| SimCSE 비교 | `results/analysis/simcse_compare/{report.md,metrics.json,curves_*.csv}` |
| 7-task | `results/analysis/champion_sts7{,_first_last}.json` |
| R9/R10 | `results/analysis/r{9_head_reinit,10_align_tctrl}/report.md` |
| cov_iso vs KoLeo | `results/analysis/coviso/report.md`, `koleo_schedule/report.md` |
