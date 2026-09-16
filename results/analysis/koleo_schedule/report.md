# KoLeo λ 감쇠 스케줄 (P-KS1~3 판정)

**데이터 출처**
- 대조군(상수 λ): `results/logs/r6_bert_koleo_lrsplit_optuna_best_t27_s{42,43}` (기존 run, 재실행 안 함)
- a: `configs/r6_bert_koleo_lrsplit_optuna_best_t27_koleomin0.25.yaml` (hold_frac=0.2→step300, decay_frac=0.4→step900, min_ratio=0.25 → λ 0.4227→0.1057)
- b: `configs/r6_bert_koleo_lrsplit_optuna_best_t27_koleomin0.yaml` (동일 hold/decay, min_ratio=0.0 → λ 0.4227→0)
- 구현: `src/schedules.py` `koleo_lambda_schedule`/`resolve_koleo_lambda`, `src/train.py` 통합. 유닛 테스트 `tests/test_schedules.py`(신규 5개) 포함 전체 41개 통과, 기존 config(스케줄 키 미지정)는 `resolve_koleo_lambda`가 `lam_max`를 그대로 반환함을 별도 검증 — bit-identical 확인됨.
- 그림: `results/analysis/koleo_schedule/overlay.png`

---

## 결과 표

| run | 피크 STS | 피크 step | 최종 STS(1499) | 피크→최종 낙폭 | 최종 alignment | 최종 uniformity | 최종 eff_rank |
|---|---|---|---|---|---|---|---|
| baseline s42 | 0.6995 | 750 | 0.6816 | -0.0179 | 0.5053 | -3.5278 | 433.06 |
| baseline s43 | 0.6671 | 1000 | 0.6652 | -0.0019 | 0.5010 | -3.4801 | 433.63 |
| a(koleomin0.25) s42 | 0.6680 | 250 | 0.6601 | -0.0079 | 0.4484 | -3.3451 | 406.72 |
| a(koleomin0.25) s43 | 0.6784 | 500 | 0.6488 | -0.0296 | 0.4653 | -3.3809 | 406.04 |
| b(koleomin0) s42 | 0.6680 | 250 | 0.6493 | -0.0187 | 0.4184 | -3.2335 | 374.53 |
| b(koleomin0) s43 | 0.6913 | 500 | 0.6546 | -0.0367 | 0.4294 | -3.3135 | 375.52 |

---

## P-KS1: alignment가 step 750 이후 평탄화 — **성립**

baseline은 alignment가 끝까지 계속 오른다(step250 0.43→최종 0.50/0.50, +0.07 안팎). a/b는 hold 구간(≤300) 이후 증가폭이 뚜렷이 줄어든다 — a: 0.434→0.448/0.437→0.465(+0.01~0.03), b: 0.437→0.418/0.437→0.429(사실상 정체 또는 소폭 하락). **λ를 낮추면 alignment 악화가 예측대로 느려진다는 점은 확인됐다.**

## P-KS2: STS 피크가 상수 λ와 동등 이상 + 피크→최종 하락 없음 — **기각**

- **피크 자체가 시드마다 갈린다**: s42는 a/b 둘 다 baseline보다 낮고(0.6995→0.6680), s43은 a/b 둘 다 baseline보다 높다(0.6671→0.6784/0.6913). 시드 2개로는 "피크가 유지·상승한다"고 말할 근거가 부족하다.
- **더 중요한 건 최종값이다**: **a/b 4개 run 전부 최종 STS(1499)가 대응하는 baseline 시드보다 낮다**(0.6601<0.6816, 0.6488<0.6652, 0.6493<0.6816, 0.6546<0.6652) — 예외 없이 일관됨.
- **피크→최종 낙폭도 줄지 않았다**: 기대와 반대로 4개 중 3개(a-s43, b-s42, b-s43)가 baseline의 낙폭(−0.0179/−0.0019)보다 **더 크게** 떨어졌다. 유일하게 낙폭이 baseline보다 작은 a-s42(−0.0079)도 애초에 피크 자체가 낮아서 절대값 이득은 없다.
- **결론: "λ를 늦게 줄이면 정점을 지키면서 하락만 막을 수 있다"는 예측은 틀렸다.** 전체 곡선이 아래로 밀리는 대가가 낙폭 감소보다 크다.

## P-KS3: uniformity 손실 미미(-3.2보다 안 나빠짐) — **부분 성립 + 용량 대 alignment 트레이드오프 확인**

리터럴 기준(-3.2)은 a/b 6개 run 전부 지킨다(가장 나쁜 값도 b-s42의 -3.2335). 다만 baseline 대비로는 전부 나쁘고(-3.35~-3.38 vs -3.48~-3.53), **a보다 b가 확실히 더 나쁘다**(uniformity: a 평균 -3.363 vs b 평균 -3.274 / eff_rank: a 평균 406.4 vs b 평균 375.0). 반대로 **alignment는 b가 a보다 더 좋다**(평균 0.4239 vs 0.4569) — λ를 완전히(0) 끄면 alignment는 더 개선되지만 uniformity/eff_rank가 그만큼 되돌아간다(재붕괴 방향).

사전 등록된 판정 규칙("b에서 alignment는 좋아지되 uniformity/eff_rank가 되돌아가면 후반에도 소량의 KoLeo가 필요 → a 채택")이 **정확히 이 패턴과 일치한다.** a와 b 중 고른다면 a(λ/4로 유지)가 맞는 선택.

---

## 게이트 판정

사전 조건("a/b 모두에서 STS 피크가 상수 λ 대비 하락하면 멈추고 보고")은 피크 기준 리터럴로는 절반만 해당(s42만 하락, s43은 상승)이라 자동으로 걸리지는 않는다. 하지만 **최종값 기준으로는 4/4가 전부 baseline보다 낮다**는, 오히려 더 일관된 반대 증거가 나왔다. 두 신호를 종합하면:

**"L_koleo 포화 이후에도 반발력이 alignment를 계속 갉아먹는다"는 진단 자체는 (P-KS1이 보여주듯) 맞다. 하지만 "그 반발력을 늦게 줄이면 STS가 개선된다"는 후속 예측은 틀렸다** — alignment 개선이 STS 이득으로 이어지지 않고, 오히려 uniformity/eff_rank 손실(특히 b)이나 전반적인 스케일 하락(a도 포함)으로 상쇄된다.

**권고**: 이 스케줄 설계(a, koleomin0.25)를 최종 채택하지 않는다 — baseline(상수 λ, `r6_bert_koleo_lrsplit_optuna_best_t27`)이 시드 평균으로 여전히 더 높은 최종 STS를 낸다. 사용자가 미리 지정한 게이트의 취지대로, alignment 악화의 다른 원인(노이즈 t 분포, teacher momentum 동결과의 상호작용)을 재조사하는 쪽으로 방향을 돌리는 것을 제안한다.
