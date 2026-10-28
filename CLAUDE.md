# 현재 브랜치 작업 지침

이 브랜치는 BERT flow-noise DINO의 19축 공동 Optuna 실험만 지원한다.
방법은 [METHOD.md](METHOD.md), 예산·선택 규칙은
[OPTUNA_SPEC.md](docs/experiments/coviso_relax/OPTUNA_SPEC.md)를 따른다.

- 조건 차이는 config 값으로 표현한다. 별도 학습 루프나 과거 실험 호환 코드를 추가하지 않는다.
- 노이즈 생성과 마스킹은 `src/augment.py`에서 처리하고 `inputs_embeds`로 주입한다.
- DINO target은 갱신 전 center, covariance loss는 갱신 전 EMA를 사용한다.
  covariance는 뷰마다 순차 갱신하고, teacher 파라미터는 optimizer step 뒤 각각 한 번 갱신한다.
- RNG 순서도 실험 동작이다. batch-KL의 평가 빈도와 난수 샘플링을 임의로 변경하지 않는다.
- 현재 비교의 loss·gradient·스케줄·데이터 순서·정밀도·선택 규칙을 구조 정리와 함께 변경하지 않는다.
- 변경 전후의 작은 수치 검증을 먼저 확보하고 유지한 기능의 테스트를 통과시킨다.
- [GPU_POLICY.md](GPU_POLICY.md)를 따른다. 계정 전체 학습·평가 프로세스 최대 2개,
  gs25009/gs25049 점유 GPU 제외, 실행 직전 공통 admission 검사.
- Python은 `uv run`으로 시작하고 자식은 `sys.executable`을 사용한다. sudo와 시스템 설치는 하지 않는다.
- `source scripts/env.sh`로 기존 환경과 scratch 캐시를 재사용한다. 활성 실험의 가상환경을
  동기화하거나 데이터를 재생성하지 않는다. 기존 환경에서는 `uv run --no-sync`를 사용한다.
- 장시간 작업은 `setsid nohup`과 불변 소스 스냅샷으로 실행한다. 중지할 때 자신의 정확한 PID만 사용한다.
- 활성 snapshot·로그·데이터·checkpoint는 수정하지 않는다. 정리 작업의 임시 수치 기준은 `/tmp`에 둔다.
- Git 커밋 메시지는 영어 Conventional Commits 형식이다. 현재 브랜치에만 커밋하고 승인 없는 push/merge를 하지 않는다.
