# FlowDINO: 완전한 clean EMA teacher + mean/cov 분리 공동 탐색

2026-11-10 사전 등록. 시작 commit819e673, 브랜치exp/flowdino-full-clean-ema.
기존 미커밋 변경·모든 과거 산출물은 보존한다. main 병합·원격 push는 하지 않는다.

## 방법과 변경

기존 bert-base-uncased FlowDINO를 사용한다. masked mean+L2 768d 문장 임베딩,
MLP→L2 256d→8192 선형 확장, DINO CE8192와 iso768을 유지한다.
cosine·iso256·직접DINO256·iBOT·토큰회귀·predictor는 없다.

    L = L_DINO + lambda_mean * ||mu||²
               + lambda_cov * ||C/max(tr(C),1e-8) - I/768||²_F

두 가중치는 독립 탐색한다. 기존 detached EMA moment 혼합·뷰당 한 번씩3회 갱신,
activation56·분모 clamp·center의 target 생성 후1회 갱신·optimizer 후teacher EMA1회를 유지한다.

새 train.teacher_input=ema는 teacher 자체의 EMA word lookup으로 token ID를 벡터로 바꾼다.
teacher는 그 clean 벡터를 자신의 EMA position/type embeddings·Transformer·head로 처리한다.
student만 기존 flow interpolation noise를 받는다. Teacher의 word lookup·encoder·head 모두
requires_grad=False/no_grad이고 모든 파라미터는 optimizer 후 EMA로 한 번 갱신한다.
centering·teacher sharpening·student 온도는 유지한다.
키가 없으면 기존 student word lookup 공유 경로를 보존하여 과거 run의 재현을 깨지 않는다.

batch32, student3 views, max128, epoch data order,4000 optimizer step,
FP32+TF32·batched views·기존 AdamW와 cosine 스케줄을 사용한다.

## 예산과 탐색 공간

새 조건만 **20 trial ×4000step** 실행한다. TPE seed20261104, startup5,
multivariate, pruning 없음. 실패·부분 run은 완료 결과로 사용하지 않는다.
GPU admission은 GPU_POLICY.md에 따른다. 계정 최대2개, 한 슬롯이면 순차 실행한다.

앞서 STS7 최고였던20261028 joint trial13을 trial0의 시작점으로 enqueue한다.
trial1은 R18 원 설정이며 **이 trial도 완전한 EMA**다. 나머지18개를 TPE가 제안한다.
과거 완료 점수를 새 trial로 가져오거나 checkpoint에서 이어 학습하지 않는다.
모든 새 trial은 동일 seed42의 사전학습 초기값에서 독립 시작한다.

기존19축의 정확한 범위는 src/joint_search.py의 SPACE와 search_registration.json에 고정한다.

| 축 | 범위 | 분포 |
|---|---|---|
| mean / cov lambda 각각 | [0.05,8] | log |
| backbone LR | [2e-5,2e-4] | log |
| head LR | [1e-4,2e-3] | log |
| gradient clip | [1,6] | linear |
| weight decay | [0.01,0.3] | log |
| LR warmup fraction | [0.01,0.2] | linear |
| teacher EMA end | [0.998,0.9999] | linear |
| EMA ramp fraction | [0.2,1] | linear |
| final teacher T | [0.08,0.18] | linear |
| initial teacher T | [0.03,0.08] | linear |
| student T | [0.1,0.25] | linear |
| teacher T warmup fraction | [0.02,0.3] | linear |
| center momentum | [0.8,0.99] | linear |
| covariance momentum | [0.75,0.99] | linear |
| noise t_lo | [0,0.6] | linear |
| noise span | [0.05,0.6] | linear |
| noise start fraction within span | [0.1,1] | linear |
| noise warmup fraction | [0,0.15] | linear |

t_max=min(0.99,t_lo+span), t_start=t_lo+(t_max-t_lo)*start_fraction.
teacher momentum start는0.997로 고정한다. 19축을 함께 바꾸므로 차이를 특정 항의 효과로 단정하지 않는다.

## 검증과 선택

기존 테스트와60-step GPU smoke를 먼저 실행하며4000-step 스케줄은 압축하지 않는다.
기존 입력 모드와 새 EMA 모드의 smoke는 동일 trial13 설정을 사용한다. step0 초기값과 first view는
같아야 하지만, 학습 중 teacher word embedding이 달라지므로60step 결과의 동등성을 요구하지 않는다.

목적함수는768d STS-B dev Spearman best다. 평가 step0/250/…/3750/3999, 동률은 최초다.
완주 top3를 탐색에 쓰지 않은 seed101/102/103에서 각각4000step 재학습한다(9회).
새3시드 dev best 평균으로 최종 설정을 선택하고 selection.json을 고정한다.
그 뒤 최종 모델의3시드×best/last에서 STS7을 평가한다(6회).256d는 별도 진단이며 선택에 쓰지 않는다.
총 새 full training29회, smoke2회다. test로 범위·설정·체크포인트 선택을 바꾸지 않는다.

과거 joint13의 seed42/43/44 dev-best STS7은67.097±0.260이다.
과거 R18 seed101/102/103은65.816±0.588로 기존 평가를 참조하고 재탐색·STS7 재평가는 하지 않는다.
R18과 같은 seed의 paired 비교도 역사적 참조와의 성능 비교다. teacher 입력과 다른 초매개변수가
함께 바뀌므로 완전EMA나 특정 정규화 항의 인과효과를 분리한 결과로 해석하지 않는다.

역사적 R18과 paired STS7 차이에 기존 판정 규칙을 적용한다. 평균Δ≥0.003이고2/3 양수면 성립,
평균≤0이고2/3 비양수면 기각, 그 외/미완료는 판정불가다. 통계적 유의성을 주장하지 않는다.
과거 baseline test가 이미 알려졌음을 명시하며, 새 test는 선택 동결 뒤에만 측정한다.
과거67.10을 맞추기 위한 비공개 재튜닝은 하지 않는다.

run별 resolved config, source commit/diff/hash, 데이터hash, seed, 실제 명령/exit,
초기값·batch 순서hash, 실제 teacher/student first-view hash, 분항 loss·weighted contribution,
고정 step의 backbone gradient를 기록한다. 역사적 데이터·로그·checkpoint를 수정하지 않는다.
first-view 동등성은 동일 seed·노이즈 설정끼리 검증한다. 서로 다른 노이즈 설정에서도
사전학습 초기값과 데이터 순서는 seed별로 같아야 한다.
GPU가 없으면 admission에서 대기하고 장시간 CPU 학습이나 점수 추정으로 대체하지 않는다.
