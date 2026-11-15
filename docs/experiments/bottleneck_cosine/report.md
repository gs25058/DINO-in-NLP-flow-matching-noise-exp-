# Bottleneck cosine — 공정한 공동 튜닝 비교

상태: **failed**.
campaign: `/src/gs25058/noise_experiment/noise_experiment/flowdino/results/experiments/bneck_cosine_20261104`. 실행 소스 `c11fe95172804442292ca14a871ad2ae2beadc08`, `exp/bottleneck-cosine-paired-optuna`.
R18 실제 4000-step 대조군을 사용한다. covariance와 noise는 양쪽에서 고정한다.
각 조건 20 trial ×4000 step; pruning 없음. 재검증 seed101/102/103.
주 지표는 768차원 embedding STS-B dev. 256차원 STS와 view cosine은 진단이다.

전체 명령·exit code: commands/와 status/. resolved config: configs/. 소스 diff와 SHA: source_manifest.json, source_diff.patch.
실행기: `scripts/run_bottleneck.py --campaign results/experiments/bneck_cosine_20261104 --gpus auto,auto`

## 탐색 공간과 보정

공통 범위: `{'train.lr': [3.4382477416615244e-05, 0.00013752990966646098, True], 'train.head_lr': [0.00029124167925665, 0.0011649667170266, True], 'loss.teacher_temp': [0.1, 0.16, False]}`.
초기 g_DINO=0.21253528, g_bneck=1.206756.
λ 동등값=0.17612117, log 범위=[0.0017612116928148694, 0.17612116928148694], 시작 λ=0.017612117.
상태 보존 검증: `{'student_unchanged': True, 'teacher_unchanged': True, 'criterion_buffers_unchanged': True, 'optimizer_unchanged': True, 'scheduler_unchanged': True, 'parameter_grads_unchanged': True, 'python_rng_restored': True, 'torch_rng_restored': True, 'cuda_rng_restored': True}`.
초기 한 minibatch의 gradient 규모이며 학습 전체에서 같은 비율이라고 가정하지 않는다.

## baseline 탐색

| trial | state | 768 dev best | last | params |
|---|---|---|---|---|
| 0 | COMPLETE | 0.7369020822480099 | 0.7328244811510823 | `{'train.lr': 6.876495483323049e-05, 'train.head_lr': 0.0005824833585133, 'loss.teacher_temp': 0.1347501197840043}` |
| 1 | COMPLETE | 0.7177115426343476 | 0.7126874707194608 | `{'train.lr': 3.771838776080576e-05, 'train.head_lr': 0.00030048864042319317, 'loss.teacher_temp': 0.11619034347406178}` |
| 2 | COMPLETE | 0.7350307208698788 | 0.7343511260251819 | `{'train.lr': 7.99142496008563e-05, 'train.head_lr': 0.0008674732469185291, 'loss.teacher_temp': 0.12127854940900888}` |
| 3 | COMPLETE | 0.7232612258559689 | 0.7105975184831167 | `{'train.lr': 0.00011912203784280311, 'train.head_lr': 0.0007988631104858635, 'loss.teacher_temp': 0.11431739652873835}` |
| 4 | COMPLETE | 0.7335822261377665 | 0.7291052336867883 | `{'train.lr': 4.385906788469344e-05, 'train.head_lr': 0.0007112024523581739, 'loss.teacher_temp': 0.1135202001190929}` |
| 5 | FAIL |  |  | `{'train.lr': 7.92434524912073e-05, 'train.head_lr': 0.0003668221785738668, 'loss.teacher_temp': 0.14653212483092365}` |
## bottleneck 탐색

| trial | state | 768 dev best | last | params |
|---|---|---|---|---|
| 0 | COMPLETE | 0.7298778913729116 | 0.7262700419099777 | `{'train.lr': 6.876495483323049e-05, 'train.head_lr': 0.0005824833585133, 'loss.teacher_temp': 0.1347501197840043, 'loss.bneck_cosine_lambda': 0.017612116928148694}` |
| 1 | COMPLETE | 0.7189645908364987 | 0.7175503704221223 | `{'train.lr': 3.771838776080576e-05, 'train.head_lr': 0.00030048864042319317, 'loss.teacher_temp': 0.11619034347406178, 'loss.bneck_cosine_lambda': 0.029012879442276773}` |
| 2 | COMPLETE | 0.7242313054190075 | 0.6985658308104641 | `{'train.lr': 0.00010240937834797631, 'train.head_lr': 0.0004761782789428013, 'loss.teacher_temp': 0.1537808479950183, 'loss.bneck_cosine_lambda': 0.05029711072933828}` |
| 3 | COMPLETE | 0.7349087180355126 | 0.7259538487842718 | `{'train.lr': 4.78633060880435e-05, 'train.head_lr': 0.0003715144905525836, 'loss.teacher_temp': 0.13864130398428137, 'loss.bneck_cosine_lambda': 0.004971470851393267}` |
| 4 | COMPLETE | 0.7216983991743289 | 0.7214276210320917 | `{'train.lr': 0.00011461651062393558, 'train.head_lr': 0.000377699032300739, 'loss.teacher_temp': 0.12224866198743439, 'loss.bneck_cosine_lambda': 0.1516733052015782}` |
| 5 | FAIL |  |  | `{'train.lr': 4.169030201732292e-05, 'train.head_lr': 0.0004322805598391723, 'loss.teacher_temp': 0.13673611503862598, 'loss.bneck_cosine_lambda': 0.0037910062756688714}` |

## 해석과 미실행 항목

미완료 단계와 점수는 추정해서 채우지 않는다. 완료 상태 전에는 최종 결론을 내리지 않는다.
다른 학습률/온도를 고른 양 조건의 차이는 동일 튜닝 예산 비교이며, 보조항 자체의 효과는 aux_off와 비교한다.
256차원 bottleneck 지표만 개선되면 768차원 임베딩 개선으로 판정하지 않는다.
과거 R18 보고 점수 대신 현재 campaign의 대조군을 비교 기준으로 삼는다.

실행 오류: `train_search_bottleneck_t5_s42 failed (143); see /src/gs25058/noise_experiment/noise_experiment/flowdino/results/experiments/bneck_cosine_20261104/commands/train_search_bottleneck_t5_s42.log`
