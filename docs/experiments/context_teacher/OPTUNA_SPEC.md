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
