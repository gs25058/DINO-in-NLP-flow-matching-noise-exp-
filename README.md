# FlowDINO

현재 실험: [주변 문맥 비대칭 자기증류](docs/experiments/context_distill/README.md).

현재 실험: [약한 학생 뷰 1 + 기존 뷰 2](docs/experiments/weak_student_view/README.md).

FlowDINO는 라벨이 없는 영어 문장으로 문장 임베딩을 학습하는 연구입니다. 깨끗한 문장을 입력받는 teacher의 출력 분포를, 가우시안 노이즈가 섞인 문장을 입력받는 student가 예측합니다. 여기에 임베딩의 평균과 공분산을 정규화해 표현 공간을 개선합니다.

## 방법

- **인코더:** `bert-base-uncased`의 마지막 hidden state를 attention mask로 평균한 뒤 L2 정규화하여 768차원 문장 임베딩을 만듭니다.
- **입력 뷰:** word embedding을 차원별 평균과 표준편차로 정규화한 뒤 가우시안 노이즈와 보간하여 student 뷰 3개를 생성합니다. 특수 토큰과 패딩은 유지하며 teacher는 깨끗한 입력을 사용합니다.
- **Teacher:** student 파라미터의 지수이동평균(EMA)으로 갱신합니다. 현재 실험에서는 teacher의 word embedding을 포함한 전체 모델에 EMA를 적용합니다.
- **학습 목표:** 256차원 bottleneck과 8192차원 출력 head를 거친 teacher와 student의 분포에 DINO cross-entropy를 적용합니다. 768차원 문장 임베딩에는 평균과 공분산 정규화를 더합니다.

전체 손실은 `L = L_DINO + λ_mean L_mean + λ_cov L_cov`이며, 현재 실험에서는 두 정규화 가중치를 독립적으로 탐색합니다.

## 실험 조건

학습 데이터는 SimCSE의 영어 Wikipedia 문장으로, 10자 미만 문장을 제거한 뒤 985,723문장을 사용합니다. 배치 크기는 32, 최대 길이는 128토큰, 학습 길이는 trial당 4,000 optimizer step입니다. AdamW와 warmup 후 cosine 학습률 스케줄을 사용하고 backbone과 head의 학습률을 별도로 설정합니다.

현재 실험은 완전한 clean EMA teacher 조건에서 정규화 가중치, 학습률, temperature, EMA 및 노이즈 설정을 포함한 19개 축을 공동 탐색합니다. Seed 42에서 20개 trial을 학습하고, 상위 3개 설정을 seed 101, 102, 103에서 재학습합니다. 자세한 설계와 탐색 범위는 [실험 명세](docs/experiments/full_ema_joint/OPTUNA_SPEC.md)를 참고하세요.

## 평가

학습 중 250 step마다와 마지막 step에 **STS-B dev Spearman 상관계수**를 측정합니다. 768차원 문장 임베딩의 dev 성능으로 checkpoint와 최종 설정을 선택한 뒤, STS12–16, STS Benchmark, SICK-R의 7개 test 과제에서 평가합니다. 256차원 bottleneck 성능과 임베딩 공간의 기하 지표는 별도 진단으로 기록합니다.

과거 실험에는 256차원 bottleneck의 cosine 및 isotropy 보조 손실과 직접 256차원 DINO가 포함됩니다. 현재 실험은 teacher 입력과 여러 하이퍼파라미터를 함께 바꾸므로, 과거 결과와의 차이를 특정 변경 하나의 효과로 단정할 수 없습니다.
