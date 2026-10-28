# 원격 권장 변경 로컬 반영 — 2026-10-28

작업 브랜치: `exp/coviso-ablation-sts7`, 기준 HEAD `a6d285e`.
원격 검토 기준: `origin/exp/r19-simcse-gram` = `211ea78d24cba6b8921b81024ac331686781a609`.
브랜치 전체를 병합하지 않고, 권장한 코어·도구·문서를 현재 작업 파일에 선별 반영했다.
기존 staged/unstaged 작업은 유지하며 커밋이나 원격 push는 하지 않았다.

| 원격 근거 | 반영 내용 |
|---|---|
| `faeca6b`, `d40441b` | `env.sh`에서 기존 캐시/환경/PY 보존, 공용 Python 재사용, TensorBoard 실행 환경 일치, `HF_HUB_CACHE` 사용 |
| `574b12b` | 임베딩 통계를 CPU로 로드한 뒤 증강 모듈에서 입력 장치에 맞게 사용 |
| `797bf1b` | `BatchIndexSampler`와 회귀 테스트. 기본 sample은 이전 난수 순서 유지, epoch는 config로 선택 |
| `1d177db` | SimCSE BERT pooler 제거, MLP 초기화, weight decay 0, grad clip 1, 부분 배치 유지. gram 변환은 제외 |
| `1d177db` + 로컬 평가기 | FlowDINO/SimCSE 체크포인트와 pretrained 지원. auto pooling은 FlowDINO mean, SimCSE CLS. 모델 설정 및 run별 평가 메타데이터 기록 |
| `0efe50d` 이후 운영 문서 | 루트 `GPU_POLICY.md`, `CLAUDE.md` 보완. `METHOD.md` 내용은 보존하며 문서의 Git ignore 해제 |
| 후속 보고서 | `docs/experiments/`에 R13, R14~R17, R18, R19 원문과 출처 보관 |
| `b64e9c0` | opt-in 체크포인트 보존, 범용 `eval_study_sts7.py`로 사후 순위 비교와 증분 캐시 지원 |
| `eccd552`, `3fdec7a` 및 calibration 도구 | 자식 Python 재사용, timeout 전달, base 로그에서 guard 기준 측정, 기본 off인 학습 측 detector와 붕괴 trial 점수 처리 |

공용 서버에 맞춰 `scripts/gpu_guard.py`를 추가했다. 각 실행 직전에 GPU/PID 소유자, 계정 전체
동시 사용 수, 유휴 상태와 메모리를 확인한다. 파일 잠금과 대기 예약으로 새 실행기끼리의 시작 경쟁도 막는다.
study/dense/early/coviso 실행기의 모든 GPU 작업에 연결했으며, coviso 실행기의 학습 실패 전파도 수정했다.
단독 서버용 R18/R19 실행기는 가져오지 않았다. 직접 실행하는 다른 GPU 도구는 GPU_POLICY의 공통 실행기로 감싼다.

붕괴 감시 기준은 전역 기본값을 넣지 않았다. 활성화 시 같은 base의 명시적인 기준값을 요구하고,
측정 도구는 각 seed의 완료/plateau 기록을 확인하며 원본 로그 경로와 SHA-256을 남긴다.
기준선, 예산 또는 서버가 달라지면 다시 측정한다. 기존 config에서 감시를 자동으로 켜지 않았다.
체크포인트 보존도 선택 사항이며 자동 삭제·pruning 도구는 추가하지 않았다.

새 SimCSE의 표식은 `simcse_original_aligned_v1`이다. 이전 발표문과 10월 dense 결과는 이전 구현의
측정값이다. 새 구현 수치로 바꿔 부르지 않으며, 수정된 학습을 새로 실행하지 않았다.

검증:

- 변경 전 CPU 단위 테스트 67개, 반영 후 **91개 통과**. 네트워크와 GPU를 사용하지 않았다.
- 기존 sample 난수 순서 일치, epoch 중복 방지, SimCSE 부분 배치/MLP, 구형 pooler 키 수용과 누락 가중치 거부, pretrained/teacher 호환성 확인.
- GPU 소유자·계정 제한·대기 예약·메모리 검사와 CPU 자식의 timeout 종료/예약 해제 확인.
- guard 시작 시점·연속 위반·기준값 측정·붕괴 점수 처리, 자식 인터프리터/체크포인트 플래그, 사후 평가 캐시 무효화 확인.
- 변경한 셸 스크립트 6개의 `bash -n` 및 tracked diff의 `git diff --check` 통과.
- 사용자 지정 캐시/환경과 공백이 있는 Python 경로 보존을 셸에서 확인.
- 10월 frozen 소스 8개 manifest 해시 일치. 가져온 보고서 4개는 원격 원문과 바이트 단위 일치. `METHOD.md` 내용 불변.

기존 원자료·발표용 그림·frozen source는 보존했다. 미커밋 플롯/절제 실험 작업도 남아 있다.

후속 정리: 위 내용은 통합 직후 기록이다. 이후 사용자 요청으로 일회성 플롯·실행기는
활성 코드에서 제외하고 로컬에 보관했다. 실측 분석기는 범용화했고 절제 설정은 유지했다.
현재 파일 구성과 판단 근거는 [로컬 변경 정리](local-cleanup-20261028.md)를 따른다.
