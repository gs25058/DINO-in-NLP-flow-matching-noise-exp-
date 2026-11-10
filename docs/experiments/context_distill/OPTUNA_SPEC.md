# 주변 문맥을 이용한 비대칭 노이즈 자기증류

2026-11-10 실행 전 등록. 브랜치 exp/context-asymmetric-distillation, 출발 commit eca56e7.
이전 weak-view 실험은 사용자 요청으로 controller3494287을 종료했고 GPU 자식 종료도 별도로 확인했다.
기존 미커밋 보고서·config·스크립트·모든 학습 산출물은 보존한다. main 병합·push는 하지 않는다.

## 가설과 판정

실제 이웃 A,C가 손상된 중심 B 표현을 돕고, 이 효과가 문맥 없는768d STS에도 남는가?
A,C를 positive target으로 사용하지 않는다. teacher target은 항상 clean B의 EMA DINO 분포다.
필수 비교는 A(문맥 없음), B(실제 이웃), C(다른 문서의 무관한 문맥)다.
B−C가 핵심이며, B−A만 양수이면 입력 형식/일반적 정규화 설명이 남는다.

최종 선택 B 설정과 같은 매개변수의 C를 새3시드에서 비교한다(C_matched_B).
B와 자체 최적 C의 선택이 같으면 해당 확인 run을 재사용하고, 다르면 C를3회 추가 실행한다.
주평가는 dev-selected best checkpoint의768d STS7 test, paired 평균차≥.003이고2/3 양수이면 성립,
평균≤0이고2/3 비양수이면 기각, 그외/미실행은 판정불가다. 통계적 유의성을 주장하지 않는다.
각 arm을 같은 예산으로 최적화한 A/B/C 결과와 각 공통 조합별 B−C도 별도로 보고한다.

## 소스와 데이터

공통 hyperparameter 계보는 full-clean-EMA trial1/R18(정확한 mean/cov 가중치 각각5.874093568592943).
과거 wiki1m 75.09dev/66.95STS7은 역사적 참고일 뿐 새 데이터의 대조군 결과가 아니다.
새 캐시는 아래 문서 코퍼스에서만 만든다. 기존 wiki1m의 인접 행은 쓰지 않는다.

- 데이터셋: wikimedia/wikipedia, subset20231101.en, revision b04c8d1ceb2f5cd4588862100d08de323dccfbaa.
- 원자료: 20231101.en/train-00000-of-00041.parquet(420296449bytes), parquet id/url/title/text 필드.
- 라이선스: CC-BY-SA-3.0 및 GFDL. 원문 URL과 문서 ID를 각 triple에 유지한다.
- 이번 예산의 코퍼스: 첫 shard의 첫20000문서, train 최대150000triple. Wikipedia 전체 무작위 표본이라고 주장하지 않는다.
- 전처리: src/context_data.py의 context-triples-v1. 원문 한 줄을 보수적 문단 경계로 삼는다.
- 문장 분리: 문장부호 뒤 공백+대문자/숫자 등을 경계로 하는 고정 규칙. 약어·이니셜·소수점이 있는 모호한 문단은 제외한다.
  제목/목록/표/마크업 줄·미완결 꼬리·3문장 미만 문단도 제외한다. 규칙 기반 분리이므로 완전한 언어학적 정답을 주장하지 않는다.
- 문서 ID/문서 내용 중복과 대소문자·공백 정규화된 B 중복을 제거한다. A=B 또는 B=C인 triple은 제외한다.
- 같은 문단의 연속3문장만 생성하고 각 문서ID·문단ID·문장index·doc내 분리index·A/B/C·원문char[start,end)·source row를 저장한다.
- sha256(document ID)로 train90%, stats5%, heldout5%를 분리한다. 세집합 문서ID 교집합은 반드시 공집합이다.
- 통계는 stats 문서의 첫10000개 B, rank용2048B와 문맥 진단은 heldout 문서에서 얻는다. 학습 문서와 겹치지 않는다.
- 무관한 문맥은 같은 split의 다른 문서에서 뽑고 B의 중복 텍스트가 A′/B′/C′에 없는지 검사한다.
  좌/우의 실제 WordPiece 수를 각각 맞춰 자른다. source doc/row/positions/text를 유지한다.
  64회 고정seed rejection에서 길이 조건을 만족하지 못하면 그 B를 모든 arm에서 제외한다.
- 같은 문서 비인접 문맥은 분리된 문장index 거리≥8, 실제 문맥과 같은 길이, 원 A/B/C 중복 제외다.
  이 후보가 없는 B는 캐시 준비 때부터 모든 arm에서 제외한다. optional D에서도 B 목록을 바꾸지 않는다.
- 원자료 SHA256, tokenizer명, 전처리 버전, cache SHA, stats SHA와 실제 수량은 data/context_wikipedia_v1/manifest.json에 남긴다.
  캐시 token ID는 고정 BERT tokenizer에서 얻고 모든 조건이 같은 B와 순서를 사용한다.

## 토큰과 pooling

B는3..126내용 WordPiece인 문장만 사용한다. 긴 B를 문맥 때문에 더 자르지 않고 모든 arm에서 제외한다.
solo=[CLS] B [SEP], 최대128. context=[CLS] A_tail [SEP] B [SEP] C_head [SEP], 최대226.
A는 끝48, C는 앞48내용 토큰이다. BERT512위치 범위 안이다. token_type_ids는 전부0이다.
sentence_span_mask(-1/0/1/2), center_pooling_mask, special_tokens_mask, attention_mask를 분리한다.
context에서는 B 내용 토큰 hidden state만 평균하고 L2 정규화한다. A/C·CLS/SEP·padding은 pooling에 넣지 않는다.
teacher·solo student·context student·STS 추론 모두 내용 토큰 pooling으로 통일한다.
따라서 과거 native baseline의 CLS/SEP 포함 mean pooling과 다르다. A/B/C 전부 같은 새 규약을 적용한다.
head는 기존 MLP768→256+L2→8192 weight-normalized linear를 유지한다. predictor 등 새 head는 없다.

## 노이즈·손실·업데이트

차원별 고정 stats μ,σ로 기존 xhat=(x−μ)/σ, x′=((1−t)xhat+tε)σ+μ를 사용한다.
B는 원 기준선의 t_lo=.3919166650119331,t_start=.4638073616095715,t_max=.8811226221985871,
warmup fraction=.02545633588184152를 유지한다. A/C는 U(0,weak_t_max), 별도 curriculum은 없다.
t는 문장당1개, epsilon은 토큰별이다. 추가한 CLS/SEP·padding에는 노이즈를 넣지 않는다.
(seed,step,view,sentence-role)별 독립 generator, B epsilon budget126과 A/C budget48로
배치 문맥 길이와 관계없이 B noise를 동일하게 만든다. B/C는 A/C t/epsilon도 같은 stream을 쓴다.

teacher는 자체 EMA word embedding·encoder·head로 clean B만 보고 stop-gradient한다.
L_distill=(CE_solo1+CE_solo2+beta*CE_third)/(2+beta). A의 third도 동일 beta를 적용한다.
공분산 정규화는 세 조건 모두 첫 solo B 뷰의768d z 하나에만 적용한다.
이 규약은 과거3뷰 순차 covariance EMA3회와 다르다. 이번에는 optimizer step당 covariance1회,
DINO center1회(target은 이전 center), teacher EMA1회(optimizer 뒤)로 고정한다.
수식·가중치·covariance activation56·기존AdamW/TF32/스케줄/mean-cov 통계 detach는 유지한다.
iBOT·token regression·bottleneck-only·cosine auxiliary·외부 의미교사·LLM 데이터는 없다.

## 예산·튜닝·선택

필수3조건 각각12trial×4000step, seed42, batch32. 같은12개의 공통 조합을 A/B/C에 짝지어 적용한다.
TPE seed20261111, startup4, multivariate, NopPruner. A/B/C dev-best 평균을 조합 제안 목적함수로 사용한다.
한 조합의 세 full run이 끝난 후 tell한다. 동시에 최대2GPU, 한 슬롯이면 admission 순차 실행이다.
매 조합의 B−C를 저장하며 arm별 최선은 해당 arm의 dev로 따로 선택한다.

탐색4축: backbone LR=[3.4382477416615245e−5,1.3752990966646098e−4] log,
head LR=[.00029124167925665,.0011649667170266] log,
beta=[.25,2] log, weak_t_max=[.02,.20] log.
A의 active변수는3개이며 weak_t_max는 값만 공유하고 계산에는 사용하지 않는다. B/C는4개로 완전히 같다.
첫 조합은 원 LR·head LR, beta1, weak 상한.1. B noise와 covariance 가중치/EMA/온도 등은 고정한다.
평가빈도는 step0/250/…/3999, 768d 단일문장 dev Spearman best, 동률은 최초 checkpoint다.
각 arm top3를 새seed101/102/103으로 확인한다(27full run). 새3seed dev-best 평균으로 최종 설정을 동결한다.
선택된 B와 같은 설정의 C를 확보하고, 이 paired dev차가평균≥.003이고2/3양수면 D(같은문서거리≥8)를3seed 실행한다.
D의 설정은 B와 동일하며 별도 튜닝은 없다. 이 결정은 새로운test 이전에 기록한다.
총예산: 검색36+확인27=63full run, C_matched_B 최대3+D 최대3으로69full run 이하, smoke3회.
그 뒤 최종3seed best/last의768d STS7을 평가한다(그룹 수에 따라24..30eval).
기존test 점수를 이미 봤음을 공개하며, 새test 결과로 선택/탐색범위/추가튜닝을 변경하지 않는다.

## 진단·비용·산출물

문서heldout에서 최대문서당2개씩64triples를 고정한다. B noise t=.05,.4,.8 각각에서
실제 문맥/길이일치 무관 문맥/B only/실제문맥의 attention 차단을 비교한다.
교사p_t(B) 대비 CE·KL·768d cosine을 보고한다. no_grad·no optimizer/EMA/center/covariance updates다.
attention 차단은 모든 층에서 B 내용토큰과 모든 non-B 토큰 사이를 양방향 차단한다.
CLS/SEP도 non-B이므로 다른토큰을 통한 전달 경로가 없다. 2-layer BERT eager/SDPA에서 불변성을 검증한다.
같은 A/C 아래 B를 다른 heldout 문서의 길이일치 토큰으로 교체하고 teacher/student 출력 변화를 보고한다.
동일길이 donor가 없는 사례는 swap 진단에서 제외하고 실제 sample수를 보고한다. 이 진단은 STS 개선 증거가 아니다.

각 run에 resolved config, source commit/diff/hash, 데이터SHA, seed, 실제명령/exit, 모델 초기값,
B순서hash, first B-token/noise/teacher hash, B/C input layout, 분항 CE·covariance·gradient 진단을 기록한다.
nonpad student token 수(교사제외), 학습+정기평가 wall time, peak allocated GPU memory를 기록한다.
B/C는 토큰길이까지 같고 A/B/C는B샘플수/step이 같다. A보다 비싼데 좋아진것만으로 효율 개선을 주장하지 않는다.
보고서에는 paired차, 각최선, best/last·seed변동, 모든7task·mean, diagnostics·비용과 미실행 이유를 남긴다.
