# 문맥을 읽는 teacher: local→global 문장 자기증류

2026-11-15 실행 전 등록. 브랜치 exp/context-asymmetric-distillation.
직전 context_distill(student가 문맥을 봄)은 B−C≈0, A가 B/C보다 약 0.55pp 높았다.
그 설계는 정보가 많은 쪽이 student여서 문맥을 쓸 압력이 없었다. 이번에는 DINO의
global(teacher)→local(student) 방향을 따른다. 탐색이 아니라 고정 설정의 paired 비교다.

## 가설과 판정

주변 문장을 함께 읽은 teacher의 B 표현을 문맥 없는 student B가 맞추도록 하면,
문맥에서 얻는 이해가 단일 문장 임베딩에 내면화되어768d STS가 오르는가?
- A: teacher도 solo clean B (문맥 없음, 기준선)
- B: teacher가 실제 5..7문장 창 안의 clean B를 읽고 B 토큰만 pooling
- C: teacher가 다른 문서의 문맥(좌/우 토큰 수 정확히 일치) 안의 clean B를 읽음
B−C가 핵심(실제 문맥의 내용 효과), B−A는 보조(형식·정규화 효과 포함).
주평가: dev-best checkpoint의768d STS7 test, seed101/102/103 paired.
평균차≥.003이고2/3 양수면 성립, 평균≤0이고2/3 비양수면 기각, 그 외 판정불가. 유의성 주장 없음.

## 데이터 (context-windows-v2, 새 캐시 data/context_wikipedia_v2)

v1과 같은 원자료·문장 분리·문서 split(sha256 doc ID 90/5/5), 첫40000문서·train 최대150000
(20000문서로는 train 86389개라 4000step×32에 못 미쳐 실행 전 40000으로 늘렸다).
같은 문단 안에서 B 앞뒤로 각각 2..3개의 온전한 문장(창 전체5..7문장). 문맥 문장은 자르지 않는다:
가까운2문장이 쪽당160 WordPiece에 들어가지 않으면 B 제외, 3번째는 들어갈 때만 추가.
B는3..126 WordPiece. 창 안 문장 중복·10자 미만 문장 제외. far(D) arm 없음.
무관 문맥: 같은 split 다른 문서, 쪽별 정확한 토큰 수 일치(donor의 B 쪽 끝 기준으로 자름; 문장
경계가 잘릴 수 있음을 공개), 고정seed 256회 rejection 실패 시 모든 arm에서 B 제외.
teacher 입력: [CLS] 왼쪽 [SEP] B [SEP] 오른쪽 [SEP], 최대450, token_type 0.

## 학습 (조건 차이는 context.teacher_mode 하나)

hyperparameter는 context_distill 조합1(A의 탐색·3seed 확인 최선)로 고정: backbone LR 5.2414e-5,
head LR 3.2383e-4, β=.50856. A에 맞춘 설정이므로 B에 보수적이다.
student: 세 개의 solo noisy B 뷰(가중치1,1,β), B noise·stream·covariance(첫 뷰,1회)·center·EMA는
context_distill과 동일하며 세 arm에서 비트 단위로 같다. teacher는 노이즈 없는 자체 EMA로 stop-gradient.
4000step, batch32, dev 평가 step0/250/…/3999, 768d 단일문장 dev Spearman best(동률은 최초).

## 실행·산출물

smoke 60step×3arm(seed42) 후 감사: 같은 seed에서 student B 토큰·노이즈·초기값·데이터 순서 동일,
B/C teacher 토큰 길이 동일. 그 뒤 A/B/C×seed101..103 =9 full run(seed 순서로 짝지어 진행),
최대2GPU, gpu_guard admission. 선택 단계 없음(고정 설정)을 test 전에 selection.json에 기록하고,
best/last checkpoint STS7 18 eval. test 결과로 설정·범위를 바꾸지 않는다. 기존 test 점수를 본 적 있음을 공개.
총예산: smoke3 + full9 + eval18. 미실행은 추정하지 않는다.

## 개정 (2026-11-15, 사용자 요청): arm별 독립 Optuna (design=tuned)

고정 설정은 A에 맞춘 값이라 B/C에 불공정하다는 지적으로, 고정 설계 캠페인
context_teacher_20261115는 첫 full run 도중(완료 run 없음, test 미관찰) 중단했다. 산출물은 보존한다.
고정 설계 코드는 `--design fixed`로 남기고, 새 캠페인은 `--design tuned`로 실행한다.

- arm마다 독립 TPE study(같은 공간·예산·sampler seed20261115, startup4, multivariate, NopPruner).
  탐색3축: backbone LR=[3.4382e−5,1.3753e−4] log, head LR=[2.9124e−4,1.1650e−3] log, β=[.25,2] log
  (context_distill과 같은 범위; student 문맥이 없어 weak_t_max는 쓰지 않는다).
- 첫 trial은 세 arm 모두 조합1(anchor). 같은 seed라 startup 무작위 점도 세 arm이 같고 TPE부터 갈라진다.
- lockstep: 모든 arm의 trial n이 tell된 뒤에만 trial n+1을 ask한다. arm당12trial×4000step, seed42.
- arm별 search dev top3를 seed101/102/103으로 확인(27run), 새3seed dev-best 평균으로 arm별 설정 동결.
- 인과 대조: B 선택 설정 그대로의 C(C_matched_B)3run.
- 판정(STS7 test, dev-best checkpoint, seed paired): 주비교는 각자 튜닝된 B vs C(사용자가 요청한 공정 비교),
  보조로 B vs C_matched_B(같은 hyperparameter), B vs A를 같은 규칙(평균≥.003·2/3 양수 성립)으로 보고.
- 예산: smoke3 + search36 + confirm27 + matched3 = 69 run, 그 뒤 best/last STS7 24 eval. 최대2GPU.

## 결과 요약: design=tuned (context_teacher_tuned_20261115, 완료)

STS7 test(dev-best ckpt) 평균: A 63.77, B 61.56, C 62.68, C_matched_B 62.35. B vs C·C_matched_B·A 모두 기각.
진단: B/C의 teacher 분포가 균등분포로 붕괴(C 후반 H_pt 9.0107≈ln8192, batch_KL 1e-4).
66 run에서 log(후반 batch_KL)과 dev(last−best) 상관 0.83. dev-best는 붕괴 직전 step250 근처를 선택했다.
고정했던 teacher temp .135/student .151(sharpening 비 약1.1)이 붕괴를 허용했을 가능성이 크다.

## 개정 2 (2026-11-16, 사용자 요청): B·C 전체 축 튜닝 (design=full)

A는 재실행하지 않는다(tuned 결과를 참고 기준으로만 보고; A는3축만 튜닝되어 B/C에 유리한 비교임을 공개).
- 공간: coviso_relax의 19축 공동 공간(`src/joint_search.py::SPACE`) + β[.25,2] log, 단 teacher temp
  하한을 .03으로 넓힌다([.03,.18] linear). 총20축. `scripts/run_context_teacher.py::FULL_SPACE`가 실행 명세.
- arm별 독립 TPE(seed20261115, startup8, multivariate), lockstep, arm당40 attempted trial(pruned 포함).
  anchor 2개: 측정된 base 점, 같은 점에 teacher temp=.05(강한 sharpening).
- 목적함수(사전 등록 변경): late dev = step3500/3750/3999 dev Spearman 평균. dev-best는 붕괴 직전 초반
  봉우리를 고르므로 쓰지 않는다. MedianPruner(startup8, warmup2000, 간격500, 최소4)는 현재 dev 값을 비교한다.
- 확인: arm별 완주 trial top3(late dev) × seed101..103, 새 3seed late dev 평균으로 동결. C_matched_B 3run.
- 판정: STS7 test, **last checkpoint**가 주판정(best도 보고). B vs C(각자 튜닝) 주비교, B vs C_matched_B 보조,
  규칙은 동일(평균≥.003·2/3 양수 성립 / 평균≤0·2/3 비양수 기각 / 그 외 판정불가).
- 예산: smoke2 + search≤80 + confirm18 + matched3, 그 뒤 STS7 18 eval. 최대2GPU.

### 재시작 기록 (2026-11-16)

context_teacher_full_20261116은 trial 0..2 완료 뒤 감사 버그로 멈췄다: 같은 seed의 student noise를
trial 간에도 같다고 요구했으나 full 설계는 노이즈 설정을 튜닝한다. 같은 trial의 B/C noise는 일치했다.
감사를 (seed, 노이즈 설정 hash) 단위로 고친 뒤 같은 설계·예산으로 context_teacher_full_20261116b를 새로 시작한다.
이전 3 trial은 보존하되 새 study에 넣지 않는다(test 미관찰).
