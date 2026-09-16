# FlowDINO-Text: 소형 검증 프로젝트

## 목적 (3줄)
- DINO 자기증류 구조(EMA teacher + centering/sharpening)는 유지하고, 증강만 flow matching식 가우시안 노이즈 보간으로 교체했을 때 문장 임베딩 품질을 검증한다.
- 기존 dropout 증강 실험은 이미 완료되었으므로 **재실험하지 않는다**. 비교 기준은 기존 결과 수치(STS-B Spearman 70.6, 250 step 이후 하락, effective rank 감소)를 외부 baseline으로 사용한다.
- 방법의 상세 스펙과 검증할 예측은 `METHOD.md`가 단일 진실 소스(single source of truth)다. 구현이 METHOD.md와 충돌하면 METHOD.md를 따르고, METHOD.md를 바꿔야 한다면 먼저 사용자에게 확인한다.

## 서버 제약 (중요)
> 이 절은 이전 공용 서버(A6000 x10, sudo 없음) 기준이다. 새 서버에서 재검토할 것.
- 공용 서버, **sudo 없음**. 시스템 패키지 설치 금지. 모든 것은 유저 공간에서 해결한다.
- 모든 파이썬 실행은 `uv run python ...`으로 통일한다. venv activate, `pip install` 직접 호출 금지.
- 캐시는 홈 쿼터 보호를 위해 스크래치로: 실행 전 `source scripts/env.sh` (UV_CACHE_DIR, HF_HOME 설정).
- 장시간 학습(>10분)은 직접 붙잡지 말고 `nohup uv run python -m src.train --config ... > results/logs/RUN.log 2>&1 &`로 던지고 로그를 tail로 확인한다. Slurm이 있으면 `sbatch scripts/slurm_train.sh`.
- 계산 노드가 오프라인일 수 있음: 모델/데이터 다운로드는 로그인 노드에서 `uv run python scripts/prepare_data.py`로 선행하고, 학습 시 `HF_HUB_OFFLINE=1`.

## 구현 원칙
1. **조건 간 코드 경로 차이 금지**: 실험 조건(run)들은 오직 config yaml 값만 다르다. if-분기로 조건별 다른 학습 루프를 만들지 않는다.
2. **증강은 `src/augment.py`의 인터페이스 뒤에만 존재**한다. 학습 루프는 증강의 내용을 모른다.
3. 노이즈는 HF 모델의 `inputs_embeds` 인자로 주입한다. backbone forward 코드를 수정하지 않는다.
4. loss 함수는 `(loss, aux_dict)`를 반환하고 aux_dict에 H(p_t), KL(p_t||p_s), (사용 시) L_vel을 항상 포함한다. METHOD.md의 예측 P1–P4 검증이 이 로깅에 의존한다.
5. 학습 루프를 짜기 **전에** `tests/`의 단위 테스트를 먼저 작성·통과시킨다 (아래 체크리스트).

## 작업 순서 (반드시 이 순서)
1. `uv sync` 후 `scripts/prepare_data.py`로 FineWeb 문장 추출 + 임베딩 통계(μ, σ) 사전 계산 → `data/` 캐시
2. `tests/test_augment.py` 작성·통과:
   - t=0에서 FlowNoiseAug가 항등(원본과 allclose)인지
   - t=1에서 출력의 차원별 분산이 표준화 공간에서 ≈1인지
   - 패딩/특수 토큰 위치에 노이즈가 들어가지 않는지
   - curriculum 스케줄이 step에 따라 단조 증가하고 상한을 넘지 않는지
3. `tests/test_loss.py`: teacher==student 로짓일 때 KL≈0, H가 온도에 따라 예상대로 변하는지
4. run R1(anchor, t 고정, 최소 구성)을 500 step 스모크 테스트 → loss 개형이 기존 dropout 실험(초반 증가 후 감소)과 유사한지 확인
5. 매트릭스 R1→R2→R3→R4 순서로 실행 (METHOD.md §6). 각 run은 시드 2개 이상.
6. `src/evaluate.py`로 STS-B(mteb) + 길이 probing + rank 지표를 결과 테이블로 집계 → `results/summary.md`

## 판정 기준 (미리 고정)
- 성공: 어떤 노이즈 설정이 STS 피크 ≥ 70 (기존 dropout 피크와 동등) **이면서** 피크 후 500 step 동안 하락 폭이 기존보다 작음
- 부분 성공: STS는 낮아도 P1–P3 예측(METHOD.md)이 로그에서 확인됨 → 분석 결과로 가치 있음
- 실패 시 디버깅 순서: (1) t 스케일/표준화 버그 → (2) 조건부 붕괴 지표(batch KL) 확인 → (3) 그 후에만 "가설 기각" 논의

## 커밋 메시지
- 항상 영어, Conventional Commits 스타일(`type: concise description`, 예: `feat:`, `fix:`, `chore:`, `docs:`, `refactor:`)로 작성한다.
- 한국어 문구, "(6단계)" 같은 단계 번호 접미사, 캐주얼한 설명은 금지.

## run/config 작명 규칙
run_name(=config 파일명)은 `<실험단계>_[<백본>]_<메커니즘><값>[_<메커니즘2><값2>...]_s<시드>` 형식으로,
**값을 이름에 직접 인코딩**한다 — a/b/c 같은 tier 문자는 config를 열어보기 전엔 무슨 값인지 알 수
없어 가독성이 떨어지므로 지양한다.
- 실험단계: METHOD.md 매트릭스의 R번호(r1~r6)를 그대로 유지 — 실험 계보를 추적하는 유일한 고리라
  버리지 않는다.
- 백본: ModernBERT는 생략(기본값), BERT는 `bert`.
- 메커니즘 토큰은 실제 스칼라 값을 짧게(유효숫자 2~3자리) 붙인다. 예: `koleo0.2`(koleo_lambda),
  `lr2e-4`(backbone lr), `upush4.1`(uniform_push_lr), `ttemp0.14`(teacher_temp 튜닝값).
  on/off만 있는 boolean은 값 대신 짧은 단어로: `embedpush`(임베딩공간 push), `tuned`(Optuna 최적값 다수 적용).
- 시드는 항상 맨 뒤 `_s<시드>` (중간에 끼워넣지 않음).
- 메커니즘이 3개를 넘어가 이름이 지나치게 길어지면, 값 나열 대신 출처를 이름에 남긴다
  (예: `r6_bert_koleo_lrsplit_optuna_best_t<trial번호>`) — 정확한 값은 config yaml과
  `results/analysis/<study_name>_trials.csv`에 있으므로 이름은 "어디서 왔는지"만 알려주면 된다.
- 예시: `r6_bert_koleo_lrsplit_c` (X) → `r6_bert_koleo0.2_lr2e-4` (O)
- 기존 파일은 소급 리네임하지 않는다(활성 run/checkpoint 경로가 깨질 수 있음) — 새로 만드는
  config부터 적용한다.

## 하지 말 것
- dropout 증강 재구현/재실험
- 기존 하이퍼파라미터(Table 1: logit dim 8192, teacher momentum 0.997, τ_t=0.082, τ_s=0.151, center momentum 0.92, AdamW lr 5.4e-4 등)를 근거 없이 변경 — 노이즈 관련 신규 하이퍼파라미터만 탐색 대상
- 결과 요약에서 시드 1개 수치로 결론 내리기


---
---

# 부록 A. 작업 규칙 (전역 지침 + 피드백 메모리 통합, 2026-09-16 서버 이전 시 이관)

## A-1. 커밋 메시지 (전역 규칙과 동일, 위 "커밋 메시지" 절 참고)
- 영어, Conventional Commits, 한국어·단계 번호 접미사·캐주얼 문구 금지. 대화가 한국어여도 동일.
- 계기(2026-08-26): "docs: 매트릭스 최종 집계 results/summary.md (6단계)" 같은 커밋에 대한 사용자 지적.

## A-2. 백그라운드 프로세스 재시작
- kill과 relaunch를 **한 Bash 호출에 묶지 않는다.** 세 번에 나눈다: (1) kill → (2) 프로세스/포트가 사라졌는지 확인 → (3) `setsid nohup ... & disown`으로 실행.
- 이유: 한 호출에 묶으면 새로 띄운 프로세스가 자신도 SIGTERM을 받아 죽었다(exit 144, 2026-09-02 세 번 재현, setsid/nohup/disown으로도 안 막힘). 하네스가 호출 단위로 프로세스 그룹을 정리하는 것으로 추정.
- `pkill -f <패턴>`/`pgrep -f <패턴>`은 **자기 자신의 bash 명령줄에도 매칭된다**(패턴 문자열이 명령줄에 들어 있으므로). 대기 루프가 영원히 안 끝나거나 자기 셸을 죽인다. 프로세스 확인은 `ps ... | grep -v grep` 또는 `nvidia-smi --query-compute-apps=pid` 기준으로 하고, 종료는 PID를 명시한다.

## A-3. GPU 동시 사용 제한 — **이전 공용 서버 전용** (새 서버를 단독 사용하면 재검토)
- 이 계정의 GPU 프로세스는 학습·Optuna trial·평가·스모크 테스트를 **모두 합쳐 최대 2개**. 이미 2개면 새 작업은 줄 세운다.
- 이유: 5개를 동시에 띄운 날, 학습 중인 GPU에 잠깐 띄운 스모크 테스트 때문에 학습 run(`r5d_bert_lrsplit_b_s42`)이 트레이스백 없이 죽었다(OOM 추정).
- 매트릭스·다중 시드는 2개씩 띄우고 슬롯이 비면 다음을 넣는다.

## A-4. GPU 선택 규칙 — **이전 공용 서버 전용**
- 띄우기 전에 두 쿼리로 점유를 확인: `nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv`, `nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv`.
- **`/src/gs25009/`, `/src/gs25049/` 사용자의 프로세스가 있는 GPU에는 절대 올리지 않는다**(팀원, 예외 없음).
- 유휴(util 0%, 메모리 거의 0) GPU 우선. 없으면 가장 붐비는 GPU를 피한다. 항상 `CUDA_VISIBLE_DEVICES`로 고정(기본 device 0에 의존 금지). run/배치마다 다시 확인.
- 이유(2026-09-02): 팀원 gs25009의 Optuna 작업이 100%로 쓰던 GPU 0에 학습을 올려 양쪽이 3~4배 느려졌다.
- 오래 도는 study는 util 순간값이 버스트성 사용자 때문에 0%로 찍혀 오판하기 쉽다. 긴 작업은 메모리 여유(>=35GB)도 함께 요구했다.

## A-5. 실험 운영 원칙 (이번 프로젝트에서 사용자와 합의되거나 사고로 배운 것)
- **단일 시드로 결론 내리지 않는다.** 기본 2시드. 7-task 평균은 2시드로도 ±1점 불확실(SICK-R 시드 편차 3~4점).
- **사전 등록**: 새 실험은 예측(성공/실패 시그니처)을 실행 전에 보고서에 박고, 성립/기각/판정불가로 판정한다.
- **Optuna 결과는 반드시 1500-step(또는 배포 길이) 2시드로 재검증**한다. 상위 1개만 검증하지 않는다(750-step 1위가 1500-step 3위로 밀린 전례). 짧은 예산 순위와 배포 예산 순위의 Spearman은 +0.19였다.
- **frac 기반 스케줄 파라미터는 다른 max_steps로 전이되지 않는다.** 튜닝은 배포 길이에서 한다.
- **붕괴 감시(collapse_guard) 기준값은 config 계보를 넘어 재사용 금지.** 쓰려는 base 자신의 plateau에서 `scripts/measure_guard_refs.py`로 잰다. batch_KL은 증강·예산에 따라 수 배 달라진다(1500-step 챔피언 0.030, r8 0.059, 7700-step base 0.200). R13에서 r8 기준값을 재사용해 챔피언 자신이 붕괴로 판정, 23 trial이 전부 날아갔다.
- **붕괴로 중단된 trial은 pruned가 아니라 마지막 유효 점수로 채점**한다(값 없는 pruned는 TPE가 그 영역을 계속 재탐색).
- **코드 스냅샷에서 장시간 실행**: 실행 중인 study/매트릭스가 읽는 파일을 수정하지 않도록, 실행은 detached worktree 스냅샷에서 하고 개발은 다른 worktree에서 한다.
- **worktree에서 `uv run`을 쓰면 5.6GB venv를 새로 만든다.** 공용 venv의 python을 직접 쓰거나(`PYTHONPATH=<worktree>`), `scripts/run_study.sh`에 `PY=` 를 준다. `tune.py`는 trial을 `sys.executable`로 띄운다.
- **실험 코드는 main을 비대하게 만들지 않는다**: 실험 기제는 `exp/*` 브랜치에 두고 결과는 보고서로 남긴다. main에는 코어와 검증된 도구만. 결과 보고서(`results/`)는 원칙적으로 로컬 전용이었다.
- **tokenization은 최적화 대상이 아니다**(599ms step 중 10.7ms, 1.8%).

---

# 부록 B. 프로젝트 지식 (프로젝트 메모리 통합)

## B-1. 백본은 BERT만 (사용자·팀 합의, 2026-09-03)
- 모든 실험은 `bert-base-uncased`(`configs/base_bert.yaml` 계열). ModernBERT가 raw 성능은 높지만 SimCSE 등 비교 기준이 BERT라 불공정 비교가 된다.
- 과제 지시가 ModernBERT를 명시하면 기본값으로 쓰지 말고 사용자에게 되묻는다. (초기 R7 cov_iso 본실험은 ModernBERT로 돌았음 - 재실행 여부는 미정으로 남았었다.)

## B-2. 8/24~9/2 세션 요약 (R1~R6)
- 옛 dropout 저장소 코드는 보고된 70.6과 불일치 → METHOD.md를 참고 PDF 4개로 재작성(데이터는 FineWeb 대신 wiki1m_for_simcse.txt).
- Table 1의 lr=5.4e-4는 post-LN BERT를 완전 붕괴시킨다(eff_rank≈1). 단순 lr 스윕으로는 해결 안 됨.
- **R5(teacher_temp warmup + momentum ramp)가 BERT 붕괴를 해결**(eff_rank~220, STS 0.61). ModernBERT 0.578→0.68~0.70.
- **P2(KL이 t에 단조) 기각**(3000-step 연장까지 재검증). 새 증거 없이 재론하지 않는다.
- uniformity 계산 버그 수정(2026-09-02): STS 쌍이 아니라 무작위 쌍으로 계산해야 한다(-0.4 → -1.3~-2.6).
- 평가 후처리(first_last pooling + center_pc2)는 raw를 크게 올리지만(0.60→0.70~0.75) **보고 수치는 후처리 없는 raw**로 한다. center_pc2는 평가 셋 자체에서 통계를 뽑는다.

## B-3. KoLeo x lrsplit Optuna (2026-09-02, 30 trial, 750-step)
- 붕괴는 **한 축이 아니라 lr/head_lr/grad_clip/koleo가 동시에 높을 때**만 일어났다(trial 12). 그 한 trial을 빼면 상관 부호가 뒤집힌다(koleo +0.59, head_lr +0.43, grad_clip +0.48).
- 교훈: Optuna 상관은 **최악 outlier를 빼고 다시 확인**한다. backbone lr을 낮게(~1e-4) 두면 나머지를 올릴 여유가 생긴다.

## B-4. KoLeo 감쇠 스케줄 기각 (2026-09-02)
- 후반 koleo_lambda 감쇠는 alignment 표류를 멈췄지만(P-KS1 성립) **최종 STS가 4 run 모두 상수-λ보다 낮았다.**
- 교훈: **alignment 개선만으로 개입이 STS에 도움 된다고 보지 않는다.** 최종/피크 STS를 직접 확인.

## B-5. R7 cov_iso — 게이트는 λ 범위 아티팩트였다
- `CovIsoPenalty`(EMA 공분산 등방 페널티, 분포 통계량 = METHOD §9상 허용). 최초 스펙의 `*D` 스케일은 gradient를 300배 키워 rank-1 붕괴 → 제거.
- 수동 grid λ∈{0.5,1,2}는 무효과(게이트 발동)였으나, **Optuna λ∈[0.1,8]에서 KoLeo를 이겼다**(0.703 vs 0.6822 @750). 최적 λ≈5~8, cov_ema_momentum≈0.82~0.87.
- cov_iso는 KoLeo와 달리 **후반 alignment를 갉아먹지 않는다** → METHOD §9의 쌍별 vs 분포 통계량 구분은 실제 기전 차이다.
- 당시 `parse_final_sts`가 중도 사망 run을 step-0 값으로 채점하던 버그 → 이후 `min_step`으로 수정.

## B-6. R8 스케줄 튜닝 — BERT 계열 최대 이득 (2026-09-07)
- 60 trial(1500-step) 중 55개가 base를 이겼다. 챔피언 `r8_bert_coviso_sched_optuna_t50`: 0.7173/0.7066, **평균 0.7120** — CLAUDE.md 성공 기준 최초 충족(피크>=0.70, 피크 후 하락 -0.001/-0.002).
- **warmup이 4~6배 길었다**: LR warmup 0.30→0.052, teacher_temp warmup 0.30→0.079, momentum ramp 전 구간→0.41(종착 0.9995→0.9985).
- 두 warmup을 분리 탐색해 교락 해제: **LR warmup 기간 상관 -0.82 vs temp -0.37 → 범인은 LR 스케줄.** 영향력: 기간 > 값 > 방식(linear/cosine 차 0.005~0.008).
- 주의: 성공 기준(70.6)은 소규모 in-house dropout run 기준이지 SimCSE가 아니다. 공식 SimCSE를 우리 코드로 재면 avg7 76.29(논문 76.25 재현) vs r8 63.75.
- 무인 study 감독기 패턴(`scripts/run_study.sh`, setsid, SQLite 재개, 최대 20회 재시작)은 재사용 가치가 있다.

## B-7. 학습 가속 옵션
- `--batch-views`(K개 student 뷰를 forward 1회로), `--tf32`(Ampere 이상, 게이트 있음), `--attn-impl`. config로도 켤 수 있으나 **기본 off**. 가속 플래그는 config가 아니라 실행기 인자로 준다(수치가 머신에 묶이지 않게).
- 실측(A6000, 200 step): batch_views 단독 0.92배(느려짐), tf32 단독 1.13배, **둘 다 1.39배**. SDPA는 transformers 5.15가 이미 기본 선택 → 이득 0.
- **2026-09-16 재현 확인 실패**: 챔피언 1500-step을 가속으로 돌리면 0.7309 → **0.7234 (-0.0075)**, 허용 ±0.005 밖. 비가속은 전 지표 bit-identical. → R14 이후 전부 가속 없이 실행. 가속을 쓰려면 모든 arm에 켜고 기준선도 가속으로 다시 잰다.

---

# 부록 C. 현재 상태 (2026-09-16, 서버 이전 시점)

## C-1. 챔피언
- **`r12_bert_noise_optuna_t35`** (main): STS-B dev **0.7333** (0.7309/0.7357). 노이즈 t 분포 튜닝(`flow_noise_t`, 40 trial)으로 r8 대비 +0.021.
  - 핵심 발견: t 범위를 **넓고 높게**. 상위 10 trial 전부 t_max 0.77~0.88, 폭 0.37~0.60(r8은 U(0.35,0.50)). corr(폭, STS)=+0.39. curriculum은 여전히 불필요.
  - config: t~U(0.392, 0.881), 뷰 3, anchor / teacher temp 0.0494→0.1348 / momentum→0.9985 / lr 6.88e-5, head_lr 5.82e-4 / cov_iso λ 5.87.
- 7-task(test): **64.23** (mean pooling, 보고 기준) / 65.26 (first_last). 하네스 SimCSE 재현 73.78, 공식 76.29.
- 표현 구조: 우리는 잘 펴고(uniformity -3.40, rank 301) SimCSE는 잘 붙인다(alignment 0.221 vs 우리 0.297). **남은 격차는 alignment.**
- 동일 예산(배치 32, 1500 step)에서는 SimCSE(0.6042)를 크게 앞선다.

## C-2. 기각된 가설 (사전 등록, 2시드, 기제 작동 확인 후 기각)
- **R9 head 재초기화**(`exp/r9-head-reinit`): 대조군 0.7042 > student-only 리셋 0.6970. sync 리셋은 KL이 공짜로 충족돼 재학습 압력 없음.
- **R10 BYOL식 pooled align 항**(`exp/r10-align-tctrl`, origin에 있음): λ는 포화 속도만 바꾼다. t 제어기로 포화를 막아도 alignment 0.299→0.303.
- **R13 후반 teacher 샤프닝**(`exp/r13-sharpen-views`): KL(p_t‖p_s)을 24배 되살려도 alignment 무반응. 챔피언 base 10 trial 튜닝 중앙값 0.7328 = 챔피언, corr(delta, STS)=-0.70. **설계 좌표는 τ가 아니라 H(p_t)**(개루프는 logit 스케일 증가로 목표 3~4배 초과, 폐루프는 정확 도달).
- **R13 뷰 구조**: "iid 노이즈는 mean pooling에서 소멸"은 틀렸다(pooled 변화가 토큰 변화의 84%). 같은 pooled 난이도에서 구조적 열화가 토큰 손상을 44~72% 덜 쓰지만, cutoff만 유의(iid 대비 +0.019), 상관 노이즈는 무효. 챔피언 base에서는 cutoff도 미달.
- DINO head 통과 공간은 STS 정보를 일관되게 잃는다(잃는 것은 uniformity).

## C-3. 진행 중이던 사이클: R14~R17 (`exp/r14-r16-alignment` → 이 브랜치)
원칙: uniformity/rank를 키우는 항 추가 금지, alignment를 만드는 축만. 정체성 제약 유지, BERT, 예산 <= 1 epoch.
보고서(로컬): `results/analysis/r14_r17/report.md`.
- **R14 예산 스케일링**: `r14_bert_champ_7700`(스케줄 frac 유지, `data_order: epoch`)이 **두 시드 모두 step 3000 정점 후 끝까지 하락** — 0.7357→0.7161, 0.7286→0.7039 (평균 0.7100 < 1500-step 챔피언). alignment 0.25→0.31~0.34로 되돌아가고 uniformity -3.70으로 과확산. **1500-step 튜닝값은 5배 예산으로 옮겨가지 않는다.**
  - mom999, 15600 arm은 미실행.
- **긴 예산 재튜닝 study** (`tune_r14_bert_champ_7700_tunebase_r14_long_budget`, 공간 `r14_long_budget`: cov_iso_lambda / momentum_end / lr / head_lr / teacher_temp, 14 trial x 7700 step): **서버 이전으로 시작 직후 중단**(완료 0). 새 서버에서 처음부터 다시 돌릴 것. base `configs/r14_bert_champ_7700_tunebase.yaml`의 붕괴 감시 기준값(H_p_bar_t 9.0041, eff_rank 316.89, batch_KL 조건 off)도 새 서버 기준선에서 다시 잰다.
- **R15 토큰 latent 예측**(data2vec식, `token_latent_lambda` 등): 구현·테스트·스모크 완료. config 4개(`r15_bert_tok_lam{0.5,1.0,2.0}`, `..._lam1.0_mask0.30`)와 실행기(`scripts/run_r15.sh`) 준비. `configs/r15_base.yaml`의 `GATE: PENDING`이 남아 있어 실행기가 시작을 거부한다 — 재튜닝 결과로 base를 정하고 해제.
- **R16 역번역 positive: 사용자 지시로 전부 삭제**(코드·의존성·산출물).
- **R17**: R15+R16 결합으로 정의돼 있어 R16 삭제로 스펙대로는 성립하지 않음. R15 결과 후 재논의.

## C-4. 새 서버에서 먼저 할 것
1. `uv sync` 후 `scripts/prepare_data.py`로 `data/` 재생성(wiki1m 985,723문장 + 임베딩 통계). 데이터는 git에 없다.
2. **기준선 재측정이 최우선.** GPU·드라이버·커널이 바뀌면 수치가 달라진다(TF32 하나로 -0.0075 이동한 실측 있음). 챔피언 1500-step과 r14_7700을 새 서버에서 다시 돌리고, 이후 모든 비교는 이 서버 수치가 아니라 **새 서버 기준선**과 한다. 체크포인트도 이 과정에서 재생성된다.
3. 이 서버 전용 절대경로(`/src/gs25058/...`)가 스크립트에 남아 있다: `scripts/compare_simcse.py`(HF_HUB), `scripts/run_r13_*.sh`·`run_r14.sh`·`run_r15.sh`(PY, UV_CACHE_DIR, HF_HOME), `scripts/plot_r14_vs_simcse.py`(MAIN). 환경변수화할 것.
4. 부록 A-3/A-4의 GPU 규칙이 새 서버에 맞는지 사용자와 재확인.
5. 이관 방식(2026-09-16 합의):
   - **git(`migrate/new-server`)**: `results/analysis/`의 보고서(.md)·그림(.png)·Optuna DB(.db)·trial CSV·JSON, `results/analysis_v2.md`, `results/summary.md`. `results/`는 여전히 gitignore 대상이라 `git add -f`로만 추적된다. `simcse_compare/curves_eval.csv`도 들어왔으니 `plot_r14_vs_simcse.py`의 MAIN 경로는 repo 내부로 바꾸면 된다.
   - **git에 없음**: 학습 로그(`results/logs/`), TensorBoard, `results/plots/`, 임베딩 캐시(.npz), 비핵심 체크포인트(`checkpoints_archive/`, 이전 서버에만 존재).
   - **체크포인트는 사용자가 수동 이관**: 이전 서버 `checkpoints/`에 핵심 7개만 남겨 두었다(약 5.5G) — `r12_bert_noise_optuna_t35_s{42,43}`, `r8_bert_coviso_sched_optuna_t50_s{42,43}`, `r14_bert_champ_7700_s{42,43}`(각 `last.pt`), `simcse_repro_1epoch_s42/best.pt`. 새 서버에서 `checkpoints/` 아래 같은 이름으로 두면 `eval_sts7.py` 등이 그대로 동작한다.
   - `exp/r9-head-reinit`도 원격에 push했다. 그 밖의 로컬 브랜치(backbone-bert-base, backup-original-history-20260826, wip/pre-restructure-snapshot, worktree-tb-recent-launcher)는 이관하지 않았다.
