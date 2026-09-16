# 스케줄 튜닝 (r5_schedules_full) — warmup이 4~6배 길었다

**데이터 출처**
- study: `tune_r7_bert_coviso_optuna_t35_r5_schedules_full` (60 trial, **1500-step**, seed 42 고정, 2 concurrent)
  - 전체 기록: `results/analysis/tune_r7_bert_coviso_optuna_t35_r5_schedules_full_trials.csv`
  - 감독 로그: `results/logs/schedule_study_supervisor.log` (2026-09-06 23:00 ~ 09-07 03:48, 60/60 완주, 재시작 0회)
- base config: `configs/r7_bert_coviso_optuna_t35.yaml` (cov_iso Optuna 2위 설정, 1500-step 2시드 평균 0.6779로 당시 최고)
- 승격 config: `configs/r8_bert_coviso_sched_optuna_t50.yaml` (best trial #50)
- 검증 run: `results/logs/r8_bert_coviso_sched_optuna_t50_s{42,43}/train.log`
- 탐색 공간 정의: `scripts/tune.py` `SEARCH_SPACES["r5_schedules_full"]` (8차원)

---

## 1. 결과

| config | s42 | s43 | 평균 | 피크(step) | 최종 eff_rank | 최종 alignment |
|---|---|---|---|---|---|---|
| **r8_bert_coviso_sched_optuna_t50** | **0.7173** | **0.7066** | **0.7120** | 0.7183(1250) / 0.7090(1250) | 312 / 313 | 0.283 / 0.284 |
| r7_bert_coviso_optuna_t35 (base) | 0.6755 | 0.6803 | 0.6779 | 0.6755(1499) / 0.6813(1250) | 308 / 306 | 0.313 / 0.304 |
| r6_bert_koleo_lrsplit_optuna_best_t27 | 0.6816 | 0.6652 | 0.6734 | 0.6995(750) / 0.6671(1000) | 433 / 434 | 0.505 / 0.501 |

**base 대비 +0.034, 직전 KoLeo 챔피언 대비 +0.039.** 60 trial 중 **55개가 base를 넘겼다** — 운 좋은 한 점이 아니라 스케줄 영역 전체가 잘못 잡혀 있었다는 뜻이다. 시드 42가 튜닝 시점 값(0.7173)을 정확히 재현해 승격 config가 trial과 동일함도 확인됐다.

**CLAUDE.md 사전 등록 성공 기준 충족**: 피크 ≥ 0.70을 두 시드 모두 만족(0.7183 / 0.7090, 후처리 없는 raw)하고, 피크(step 1250) 이후 하락이 −0.001 / −0.002로 사실상 없다. 외부 기준선인 기존 dropout 실험은 70.6에서 250 step 이후 하락이었다.

---

## 2. 무엇이 잘못돼 있었나 — warmup 기간

| 축 | 기존 (DINO 원본 기본값) | 최적 | 배수 |
|---|---|---|---|
| **LR warmup** | frac 0.30 = step 450 | frac 0.052 = **step 77** | **1/5.8** |
| teacher_temp warmup | frac 0.30 = step 450 | frac 0.079 = step 118 | 1/3.8 |
| teacher_temp 값 | 0.082 → 0.11 (선형) | 0.049 → 0.135 (cosine) | — |
| momentum ramp | 전 구간(step 1500) | frac 0.41 = step 611 종료 | 1/2.5 |
| momentum 종착값 | 0.9995 | 0.9985 | — |

R5 실험에서 "warmup을 넣으면 좋아진다"까지는 확인했지만 **값은 DINO 원본 기본값을 그대로 써왔고, 그 뒤로 한 번도 탐색되지 않았다.** 학습의 실질적 이득은 step ~250에서 대부분 끝나는데 warmup이 450까지 이어지는 타이밍 불일치가 실재했다.

## 3. 교락 해제로 범인이 특정됐다

`train.warmup_frac`(LR)과 `loss.teacher_temp_warmup_frac`이 **둘 다 0.3으로 하드코딩**돼 있어 두 warmup이 항상 step 450에 함께 끝났다. 그래서 그 지점의 전환이 LR 때문인지 온도 때문인지 구분이 원리적으로 불가능했다. 이번에 독립 축으로 분리한 결과:

| 파라미터 | value와의 상관 | 상위10 평균 | 하위10 평균 |
|---|---|---|---|
| **`train.warmup_frac` (LR 기간)** | **−0.816** | 0.066 | 0.291 |
| `loss.warmup_teacher_temp` (시작 온도) | −0.457 | 0.048 | 0.071 |
| `train.momentum_ramp_frac` | −0.391 | 0.445 | 0.789 |
| `loss.teacher_temp_warmup_frac` | −0.369 | 0.117 | 0.269 |
| `train.momentum_end` | −0.255 | 0.9984 | 0.9991 |
| `loss.teacher_temp` (끝 온도) | −0.161 | 0.142 | 0.151 |

**step 450의 전환은 주로 LR 스케줄이었다.** teacher_temp warmup의 영향은 그 절반 이하다.

**영향력 순위: 기간 > 값 > 방식.** 방식(linear vs cosine)은 가장 약했다 — teacher_temp는 cosine 0.6997 vs linear 0.6946, momentum은 cosine 0.7007 vs linear 0.6928로 차이가 0.005~0.008에 불과하고, 상위 8개 trial에도 두 방식이 섞여 있다. 스케줄 방식을 탐색 대상으로 만드느라 코드까지 추가했지만, 실제로 중요한 건 **언제 끝나는가**였다.

## 4. 정정 — "후반 하락"은 cov_iso가 이미 고쳤던 문제다

작업 중 "모든 run이 500~750에서 피크 찍고 미끄러진다"고 서술했으나, 피크 시점을 확인하니 **부정확하다**:

- **하락은 KoLeo 계열의 특성이었다** — `r6_bert_koleo_..._t27` s42는 0.6995(750) → 0.6816(1499)로 뚜렷이 하락.
- **cov_iso base(t35)는 이미 하락하지 않았다** — s42는 step 1499가 곧 피크(끝까지 상승 중), s43은 1250 피크 후 −0.001.

즉 **두 개선은 서로 독립적이다**: cov_iso가 궤적의 *모양*(후반 하락)을 고쳤고, 스케줄 튜닝은 그 위에서 *수준*을 +0.034 끌어올렸다. 스케줄 튜닝이 하락을 고친 것이 아니다.

## 5. 기하학적 대비 — cov_iso는 rank를 덜 쓰고 더 잘한다

| | eff_rank | alignment(↓) | uniformity(↓) | STS |
|---|---|---|---|---|
| KoLeo 계열 | **433** | 0.505 | −3.51 | 0.673 |
| cov_iso + 튜닝 스케줄 | 312 | **0.283** | −3.44 | **0.712** |

KoLeo는 eff_rank를 훨씬 크게 키우지만 alignment를 그만큼 희생하고, 최종 점수는 오히려 낮다. 쌍별 반발이 alignment를 침식한다는 P-I5 가설과 일관되며, **등방성을 rank로 밀어붙이는 것이 목표가 아님**을 보여준다.

---

## 6. 방법론 교훈 (다음 튜닝에 그대로 적용할 것)

1. **스케줄 튜닝은 배포 길이(1500 step)에서 해야 한다.** 모든 파라미터가 `frac × max_steps`라 750-step에서 튜닝한 frac은 절대 step이 절반이 되어 전이되지 않는다. 독립적으로도 cov_iso study에서 750-step 1위(t18)가 1500-step에서는 3위로 밀린 전례가 있다 — **탐색 예산이 성능의 좋은 프록시라는 가정은 이 프로젝트에서 이미 두 번 깨졌다.**
2. **상위 1개만 검증하면 안 된다.** cov_iso 상위 5개를 검증한 결과 750-step 1위(t18, 0.6739)보다 2위(t35, 0.6779)·3위(t34, 0.6771)가 1500-step에서 더 좋았다. t18만 봤다면 "cov_iso는 1500에서 안 통한다"는 잘못된 결론이 났을 것이다.
3. **중도 사망 trial을 점수로 기록하지 말 것.** 이전 cov_iso study는 40개 중 11개가 step 30~80에서 조용히 죽었는데, `train.log`에 step 0 EVAL이 남아 있어 사전학습 초기값(0.5931)이 정상 점수로 보고됐고 TPE가 그 영역을 "탐색했는데 나쁨"으로 학습했다. `parse_final_sts(min_step=max_steps-1)`로 수정한 뒤 이번 study는 **60/60 COMPLETE, 사망 0건**.
4. **trial 체크포인트는 저장하지 말 것.** 아무도 읽지 않는데 30-trial 한 번에 ~25GB가 쌓였다. `save_checkpoint: false`를 trial config에 심어 해결.
5. **승격 config는 값과 *타입*까지 검증할 것.** pandas의 numpy 스칼라를 그대로 포맷하면 YAML에 `np.float64(0.0517...)`가 문자열로 박힌다. 이번에 실제로 발생했고 검증 단계에서 잡았다.
6. **무인 장시간 sweep에는 감독 패턴이 유효하다.** `setsid`로 PPID 1 분리 + Optuna SQLite 재개 + 남은 trial 수만 재요청. 이번엔 재시작이 한 번도 필요 없었지만, 세션 종료와 무관하게 완주했다.

## 7. 남은 질문

- **이 스케줄이 정규화 방식과 독립인가?** 같은 스케줄을 KoLeo 라인(`r6_bert_koleo_lrsplit_optuna_best_t27`)에 얹어보면, 스케줄 개선분(+0.034)이 cov_iso 고유의 것인지 일반적인 것인지 갈린다. 가장 저렴하고 정보량이 큰 다음 실험.
- **더 짧은 warmup은?** 최적값이 탐색 하한(LR frac 0.05, temp frac 0.05) 근처에 몰려 있다 — 하한을 더 낮춰 재탐색할 여지가 있다.
- cov_iso 미검증 config 2개(t30, t39)는 config만 남아 있다.
- 이번 스케줄로 얻은 궤적이 1500 step 이후에도 계속 오르는지(ext3000).
