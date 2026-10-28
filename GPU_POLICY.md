# 공용 서버 GPU 사용 규약

이 checkout은 공용 서버 규약을 따른다. 원격의 단독 서버용 실행기 기본값으로 대체하지 않는다.
근거: 원격 `eed56a1`의 `CLAUDE.md` A-2~A-5와 이 서버에서 합의한 운영 규칙.

- **이 계정의 GPU 프로세스는 총 2개 이하.** 학습, Optuna, 평가, 스모크 테스트를 모두 센다.
- **gs25009 또는 gs25049의 프로세스가 있는 GPU에는 작업을 올리지 않는다.**
- 매 실행 직전에 아래 점유 현황과 PID 소유자를 확인한다. 학습에서 평가로 전환할 때도 다시 확인한다.
- 유휴 GPU를 우선하고 `CUDA_VISIBLE_DEVICES`로 선택한 물리 GPU를 명시한다. 공통 실행기는 번호를 UUID로 해석해 고정한다. 장시간 작업은 여유 메모리 35 GB 이상을 요구한다.
- 10분을 넘는 작업은 `nohup` 또는 별도 세션에서 실행한다. 종료는 자신의 작업 PID로만 한다.
- 종료와 재실행은 별도 호출로 나눈다: 종료 → 종료 확인 → 재실행. 광범위한 `pkill -f`를 쓰지 않는다.

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv
ps -o user= -p <확인할_PID>
```

공통 실행기는 `scripts/gpu_guard.py`다. 선택한 GPU에 compute process가 없고, util < 30%,
여유 메모리 ≥ 35000 MiB일 때 실행한다. 소유자를 확인할 수 없거나 조회가 실패하면 실행하지 않는다.
이 계정의 현재 GPU PID와 실행 대기 예약을 합쳐 최대 2개까지 허용하며, 같은 GPU의 중복 예약도 막는다.
조회와 예약은 계정 공통 파일 잠금으로 직렬화한다. 다른 사용자의 실행기나 이 검사를 거치지 않는
외부 작업까지 잠글 수는 없으므로 직접 실행할 때도 위 규약을 지킨다. 실행기는 한 작업당 GPU 프로세스 하나를 전제로 한다.

```bash
source scripts/env.sh
# GPU에는 위에서 확인한 물리 GPU 번호를 설정한다.
uv run python scripts/gpu_guard.py --gpu "$GPU" --wait -- \
  "$UV_PROJECT_ENVIRONMENT/bin/python" -m src.train --config configs/<run>.yaml
```

`--wait`는 조건을 만족할 때까지 대기한다. 생략하면 실행을 거부하고 종료한다.
`--timeout <초>`는 자식 작업의 실행 시간을 제한하고, 시간 초과 또는 종료 신호를 받으면
자신이 실행한 프로세스 그룹만 종료한 뒤 예약을 해제한다.

`--gpu auto`는 동일한 계정 잠금 안에서 모든 GPU를 검사하여 규약을 만족하는 완전 유휴 GPU를
선택한 뒤 UUID로 고정한다. 유휴 GPU가 없으면 `--wait`로 대기한다. 점유 GPU에 함께 올리지 않는다.

`run_study.sh`/`tune.py`는 매 trial을 이 실행기로 전달한다. 직접 실행하는 다른 GPU 도구도
위 방식이나 `env.sh`의 `flowdino_gpu_python` 함수로 감싸서 사용한다.
날짜·run이 고정된 과거 실행기는 로컬 실험 보관소로 옮겼다([정리 기록](docs/local-cleanup-20261028.md)).
