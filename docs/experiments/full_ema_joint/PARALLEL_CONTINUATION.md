# 2-GPU 병렬 탐색 전환

2026-11-10 사용자 요청: GPU를 가급적 2개 사용한다.
기존 탐색은 한 arm을 한 번씩 실행해 GPU 슬롯이 둘이어도 탐색은 직렬이었다.

`scripts/parallel_full_ema.py`는 기존 campaign의 Optuna DB를 계속 사용한다.
완료 trial은 재실행하지 않으며, 진행 중인 trial은 원래 GPU에서 끝까지 실행해 인계한다.
다음 trial부터 최대 두 개를 동시에 실행한다. 기존 trial을 포함해 총20개 예산을 유지한다.
확인 실험과 최종 평가는 기존 실행기의 두 GPU 슬롯을 사용한다.
GPU admission은 계속 gpu_guard를 거치며 계정 전체 프로세스 상한2개를 유지한다.

원래 controller의 스케줄링 thread만 SIGSTOP으로 일시 정지한다. 기존 GPU guard와
학습 자식은 계속 실행한다. 종료·4000-step 완료를 확인한 뒤 옛 controller만 종료하고
해당 trial 결과를 같은 study에 기록한다. 실제 학습을 중단하거나 다시 시작하지 않는다.
진행 중이던 trial의 status에는 인계 사실과 완료 검증을 명시한다.

활성 source snapshot·원본 사전 등록·config·기존 로그·checkpoint는 수정하지 않는다.
새 controller는 별도 경로에 복사하고 SHA256을 기록하며 학습/평가 코드는 원래
동결 source에서 import한다. 학습 수식·19축 탐색 범위·4000-step·seed42·선택 규칙은 같다.

기존 sampler의 RNG 상태가 저장되지 않았으므로 동일 제안 순서를 재현한다고 주장하지 않는다.
전환 이후 TPE seed는20261110, startup5·multivariate·constant_liar=True,
pruning 없음이다. constant liar는 진행 중 trial을 고려하여 동시 제안의 중복을 줄인다.
전환 시점의 완료/진행 trial, PID, 예산, sampler, controller SHA256은 새 trial을
할당하기 전에 campaign `continuations/parallel_20261110/registration.json`에 기록한다.
원래 seed20261104 직렬 탐색과 제안 경로가 달라짐을 공개한다.

검증은 CPU에서 진행 trial 인계, 기존 완료값 보존, 남은 예산만 사용, 실제 두 worker 동시 실행,
한 worker 실패 시 다른 worker 종료 전달을 확인한다. 실서버 전환 후에는 실제 GPU PID와
새 trial resolved config의 teacher_input=ema, CE8192+iso768 및 초기화·데이터 순서 감사를 확인한다.
