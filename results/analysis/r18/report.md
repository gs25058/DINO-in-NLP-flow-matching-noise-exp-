# R18: iBOT식 토큰 수준 프로토타입 CE (R15 재설계)

**브랜치**: `exp/r18-ibot-token-ce` (migrate/new-server 위, R15 마스킹 뷰 코드 재사용) · **서버**: RunPod RTX 6000 Ada 1장
**기준선**: 챔피언 `r12_bert_noise_optuna_t35` — STS-B dev 0.7333 (1500 step, 2시드, 이전 서버·가속 없음)
**작성 시작**: 2026-09-17 (실행 전 사전 등록)

---

## 0. 배경

### R15(data2vec식 회귀) 실패 진단
마스크 토큰 위치에서 teacher hidden state(top-6층 평균)를 회귀했고 실패했다.
- STS 0.63 정체, eff_rank 초기값 220 → 225 정체, uniformity −3.0에서 멈춤.
- alignment 0.27은 개선이 아니라 pretrained BERT(0.195) 근처에 머문 결과.
- 원인 (1) λ 과대: L_tok 0.2 vs KL 0.02로 토큰 손실이 student 기울기의 약 10배.
- 원인 (2, 근본): teacher 자신의 hidden을 맞추는 목표가 pretrained 모델에서 "BERT 그대로 있어라"는 앵커가 되어
  STS 이득의 원천인 확산을 막는다(data2vec은 from-scratch라 이 자기참조가 무해했다).

### R14(4000 step)의 교훈
이 방법에서 STS 이득은 확산에서 오고, 확산은 alignment를 갉아먹으며, 확산을 멈추면 이득이 사라진다.
확산하면서 alignment를 공급하는 기제가 필요하다.

### R18의 논리
목표를 raw hidden state가 아니라 **학습 중인 DINO head가 만드는 토큰별 프로토타입 분포**로 바꾼다(iBOT, DINOv2).
- head가 움직이면 목표도 움직이므로 raw BERT 기하에 앵커되지 않는다.
- centering/sharpening이 토큰 수준에서도 "다른 서랍으로 가라"는 판별 압력을 만든다.
- 문장당 마스크 토큰 수만큼 판별 사건이 늘어난다. 문장 수준 CE가 alignment에 무력했던 이유가 판별 해상도
  부족이라면, 이것이 정체성(쌍별 반발 금지) 안에서 해상도를 올리는 방법이다.
- 표본 간 상호작용은 프로토타입(파라미터)을 통해서만 일어나므로 정체성 준수.

---

## 1. 구현 (커밋 `72f06fe`, 테스트 `3cb66b6`)

| 요소 | 내용 |
|---|---|
| 토큰 head | `IBOTTokenHeads`(src/model.py): 마스크 위치 마지막 층 hidden [N_mask, D] → DINO head → [N_mask, 8192]. teacher는 깨끗한 입력의 같은 위치(no_grad). `loss.ibot_head`: shared(문장 head 공유, 기본) / separate(별도 head + 같은 momentum의 EMA 사본, student state_dict 밖에 두어 기존 strict 로더 호환) |
| 토큰 CE | `IBOTTokenLoss`(src/loss.py): 문장 center와 독립인 `token_center`(같은 center_momentum). p_t = softmax((ℓ_t − c_tok)/τ_t), p_s = softmax(ℓ_s/τ_s), 마스크 위치 평균 CE. 분포는 갱신 전 center로 계산 후 center 갱신(문장 쪽과 같은 순서). `loss.ibot_use_token_center` 기본 true |
| λ 자동 조정 | `IBOTLambdaCtrl`: `loss.ibot_grad_ratio`가 있으면 log_every(10) 스텝마다 backbone 기울기 노름 ‖g_ce‖(문장 DINO CE), ‖g_ibot‖(λ 곱하기 전)를 따로 재서 목표 λ* = ratio·‖g_ce‖/‖g_ibot‖, λ ← 0.9λ + 0.1λ*, [0.01, 10] clip. 첫 관측은 λ*로 바로 이동 |
| 마스킹 뷰 | R15 구현 재사용: student 뷰 0 하나, 비특수 토큰의 mask_ratio를 [MASK]로 치환 후 노이즈 |
| 로깅 | L_ibot, H_pt_tok, H_p_bar_tok, KL_tok, ibot_lambda (ratio 모드: ibot_g_ce, ibot_g_tok, ibot_grad_ratio_obs) |
| 기본값 | ibot_lambda 0.0, ibot_grad_ratio null → 기존과 bit-identical |

검증:
- 단위 테스트 130개 통과(신규 17: λ=0 동일, 토큰 head shape, teacher no_grad, token_center 독립, grad_ratio clip,
  separate head 파라미터 비공유 + EMA 추종, 마스크 0개 처리, H+KL 분해).
- 챔피언 config 300 step: 변경 전후 로그·EVAL 34줄 완전 동일.
- 스모크 60 step(shared λ0.3 / separate ratio0.5, 가속 on/off): 정상 종료, H_p_bar_tok 9.01(붕괴 없음), separate 체크포인트 strict 로드 확인.
- 관측: ratio 0.5 목표에 대해 실측 비율이 초반 0.74~0.79. λ가 10 step마다만 EMA로 따라가(시간상수 약 100 step)
  문장 CE 기울기가 빠르게 줄어드는 초반에는 목표보다 높게 머문다.

---

## 2. 매트릭스와 실행 방식

base `configs/r18_base.yaml` = 챔피언 config, max_steps 4000, data_order epoch, eval 250 step마다, mask 0.15.
전 run `--batch-views --tf32`, 시드 {42, 43}.

| run | ibot | head | mask | 역할 |
|---|---|---|---|---|
| `r18a_lam0.3` | λ 0.3 | shared | 0.15 | 게이트 arm |
| `r18a_lam1.0` | λ 1.0 | shared | 0.15 | |
| `r18a_ratio0.5` | grad ratio 0.5 | shared | 0.15 | |
| `r18b_sep_ratio0.5` | grad ratio 0.5 | separate | 0.15 | F-18-1 대조 |
| `r18a_ratio0.5_mask0.3` | grad ratio 0.5 | shared | 0.30 | |
| `r18_bert_ctrl` (추가) | 없음 | – | – | 동일 조건 대조군 + 붕괴 감시 기준값 plateau |
| `r12_bert_noise_optuna_t35_ada[_accel]` (추가) | 없음 | – | – | 챔피언 1500 step 새 서버 기준선, 가속 없이/가속 |

추가한 두 arm의 이유:
- **동일 조건 대조군**: 챔피언(0.7333)과 R14 4000-step 궤적은 이전 서버·가속 없음이다. 가속은 이전 서버에서 챔피언을
  −0.0075 옮겼고 이 서버에서도 TF32를 켜면 step 0 평가부터 값이 달라진다. R14와의 직접 비교에는 서버·가속 차이가
  섞이므로, 같은 서버·같은 플래그의 iBOT-off 4000-step run을 함께 돌린다. 붕괴 감시 기준값도 이 run의 plateau에서 잰다.
- **1500-step 기준선 2종**: 새 서버 기준선(가속 없음)과 가속 효과 크기를 한 번에 얻는다.

실행 순서(`scripts/run_r18.sh`, GPU 1장 순차 — 이 서버에서 run을 겹치면 총 처리량이 떨어졌다):
1. `gate`: r18a_lam0.3 s42 (step 1000 게이트) → r18_bert_ctrl s42, s43
2. 붕괴 감시 기준값 측정 → `r18_base.yaml`에 기록
3. `matrix`: 나머지 9 run
4. `base1500`: 4 run
5. `sts7`: 최고 arm 2시드 + 대조군 2시드

코드는 detached worktree 스냅샷에서 실행한다(실행 중 개발 변경이 다음 run에 섞이지 않게).
게이트 arm s42는 기준값 확정 전이라 붕괴 감시 없이 돈다(감시는 중단만 하고 수치는 바꾸지 않는다).

---

## 3. 사전 등록

### 게이트
첫 arm(r18a_lam0.3 s42) step 1000에서 eff_rank가 R15처럼 초기값 근처(< 240)에 정체하면 즉시 중단·보고.
앵커 해소 실패라는 뜻이며, 매트릭스 진행 전에 원인(head 공유가 앵커를 재도입? centering 누락?)을 본다.

### "살아난 R15"의 정의 = 다음 둘의 동시 성립
- **P-18a**: eff_rank가 R15처럼 정체하지 않고 성장(step 2000에 ≥ 300) — 앵커 해소의 증거.
- **P-18b**: 동일 step에서 alignment가 R14보다 낮음(개선) — 특히 R14가 침식하는 step 1500~4000 구간에서 격차가 벌어져야 함.
- 둘 다 성립 시 **P-18c**: STS-B dev 최고값 > 0.7333 및 4000-step 마감값이 R14(0.7087)보다 높음. 7-task 평균 기록.

판정 비교 대상: 위 문장 그대로 R14/챔피언 수치로 판정하고, 같은 판정을 **동일 조건 대조군 `r18_bert_ctrl`**
(P-18b의 R14 → ctrl, P-18c의 0.7333 → ctrl 최고값, 0.7087 → ctrl 마감값)로도 한다. 두 판정이 갈리면 둘 다 보고하고
서버·가속 차이로 설명되는지 본다.

### 실패 분기
- **F-18-1**: rank 정체 + alignment 유지 → 프로토타입 목표도 앵커(head가 pretrained 표현을 그대로 양자화) → separate head 결과와 대조.
- **F-18-2**: rank 성장 + alignment 무개선 → 토큰 수준 판별도 pooled alignment에 무력 → "판별 해상도" 가설 기각.
  이 경우 정체성 제약 안의 토큰 수준 경로는 소진된 것으로 명시.
- **F-18-3**: H_p_bar_tok 급락 또는 토큰 분포 붕괴 → token_center/온도 재조정 후 1회 재시도, 그래도 실패면 F-18-1/2로 분류.

### 부수 관측
토큰 수준 프로토타입이 의미 클러스터를 이루는가(emerging property의 첫 텍스트 증거 후보) — 마스크 토큰 상위 할당
프로토타입별로 실제 토큰을 20개씩 뽑아 정성 표 1개.

### 플롯
R14/R15/R18(+ctrl) 겹친 STS·alignment·uniformity·eff_rank·H_pt·KL.

---

## 4. 비교 기준 수치 (사용자 보고, 이전 서버·가속 없음)

| 기준 | 값 |
|---|---|
| 챔피언 1500 step | STS-B dev 0.7333 (0.7309 / 0.7357) |
| R14 4000 step | alignment 0.2765 @750 → 0.3327 @3999, eff_rank 303 → 394, 마감 STS-B dev 0.7087 |
| R15 | STS 0.63 정체, eff_rank 220 → 225, uniformity −3.0, alignment 0.27 |

알려진 한계: R14/R15 step별 학습 로그는 이 서버로 이관되지 않았다(`results/logs/` 제외). P-18b의 R14 구간 비교와
겹친 플롯에는 이전 서버 로그가 필요하다. 없으면 R14 비교는 위 두 지점(750, 3999)만으로 하고 구간 비교는 ctrl로 한다.

---

## 5. 결과

(실행 후 기록)
