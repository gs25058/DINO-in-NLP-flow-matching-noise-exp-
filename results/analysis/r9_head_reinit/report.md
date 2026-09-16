# R9: 주기적 DINO head 재초기화

**코드 위치**: 브랜치 `exp/r9-head-reinit` (main에 병합하지 않음)
**작성**: 2026-09-09

## 동기

백본이 head의 특정 prototype 배치에 과적합되는지 확인하려 했다. 주기적으로 head를 난수로
되돌리면 백본이 매번 새 배치를 다시 맞춰야 하므로, 그 재학습 gradient가 표현을 개선할
것이라는 가설이었다.

## 결과

3000 step, `head_reinit_every_steps=1000`.

| 조건 | s42 | s43 | 시드 평균 | 대조군 대비 |
|---|---|---|---|---|
| 대조군 (리셋 없음) | 0.7135 | 0.6949 | **0.7042** | — |
| student만 리셋 | 0.7129 | 0.6811 | **0.6970** | −0.007 |
| sync 리셋 (student+teacher+center) | 0.7073 @step1630 | — | 판정 불가 | — |

**sync 조건은 결론을 낼 수 없다.** 시드 1개이고 그마저 step 1630에서 수동 중단됐다
(09-08 16:26). CLAUDE.md의 "시드 1개 수치로 결론 내리기 금지"에 해당한다.

**완결된 비교는 student-only 하나뿐이고, 그 결과는 소폭 악화다.**

## 왜 sync가 작동하지 않는가 (진단은 유효)

점수는 판정 불가지만 로그의 진단 지표는 명확하다. student와 teacher head를 **같은 난수로**
리셋하면 두 logit이 거의 같아져 `KL(p_t||p_s)`가 공짜로 충족된다. 실측:

| step | KL_pt_ps | 비고 |
|---|---|---|
| 0 | 0.1272 | 최초 난수 head |
| 990 | 0.0043 | 리셋 직전 |
| 1010 | 0.0075 | 리셋 직후 |
| 1050 | 0.0062 | 회복 |

step 0의 0.1272에 한참 못 미친다. "다시 배우라"는 압력이 loss에 거의 실리지 않는다는 뜻이다.
`student_only` scope는 정확히 이걸 피하려고 학습된 teacher를 남기는 변형이고, 그래서 완주까지
돌린 것도 이쪽이다. 다만 그렇게 해도 대조군을 넘지 못했다.

## 남은 의문

- student-only의 s43(0.6811)이 s42(0.7129)보다 0.032 낮다. 시드 분산이 효과 크기(0.007)보다
  훨씬 커서, 2시드로는 "악화"조차 확정하기 어렵다. 결론을 강하게 쓰려면 시드가 더 필요하다.
- sync 조건을 끝까지 돌려본 적이 없다. 위 진단대로라면 무의미할 가능성이 높지만, 확인된 것은
  아니다.

## 재현

```bash
git checkout exp/r9-head-reinit
uv run python -m src.train --config configs/r9_bert_t50_steps3000.yaml --seed 42                      # 대조군
uv run python -m src.train --config configs/r9_bert_t50_steps3000_headreinit1000_studentonly.yaml --seed 42
```
