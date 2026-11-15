# FlowDINO: 완전한 clean EMA + mean/cov 분리 공동 튜닝

상태: **failed**.
campaign: `/src/gs25058/noise_experiment/noise_experiment/flowdino/results/experiments/full_ema_joint_20261110`. 실행 소스 `207dee68160397e4b34f4e85bd714dd63577b4f5`, `exp/flowdino-full-clean-ema`.
DINO8192+iso768 유지, mean/cov 독립 가중치 및 기존19축을 탐색. teacher는 자체 EMA word lookup 사용.
완전한 EMA 조건 20 trial ×4000 step. pruning 없음. 재검증 seed101/102/103.
768차원 STS-B dev로 선택하고, 동결 후 768차원 STS7으로 판정한다. 256차원은 진단이다.

전체 명령·exit code: commands/와 status/. resolved config: configs/. 소스 diff와 SHA: source_manifest.json, source_diff.patch.
실행기: `scripts/run_bottleneck.py --full-clean-ema --campaign results/experiments/full_ema_joint_20261110 --gpus auto,auto --trials-per-arm 20`

탐색 공간: `{'loss.cov_iso_mean_lambda': [0.05, 8.0, True], 'loss.cov_iso_cov_lambda': [0.05, 8.0, True], 'train.lr': [2e-05, 0.0002, True], 'train.head_lr': [0.0001, 0.002, True], 'train.grad_clip': [1.0, 6.0, False], 'train.weight_decay': [0.01, 0.3, True], 'train.warmup_frac': [0.01, 0.2, False], 'train.momentum_end': [0.998, 0.9999, False], 'train.momentum_ramp_frac': [0.2, 1.0, False], 'loss.teacher_temp': [0.08, 0.18, False], 'loss.warmup_teacher_temp': [0.03, 0.08, False], 'loss.student_temp': [0.1, 0.25, False], 'loss.teacher_temp_warmup_frac': [0.02, 0.3, False], 'loss.center_momentum': [0.8, 0.99, False], 'loss.cov_ema_momentum': [0.75, 0.99, False], 'augment.t_lo': [0.0, 0.6, False], '_t_span': [0.05, 0.6, False], '_t_start_frac': [0.1, 1.0, False], 'augment.warmup_frac': [0.0, 0.15, False]}`.
trial0은 과거 joint trial13, trial1은 R18 설정. 모든 새 탐색은 teacher_input=ema.
과거67.097±0.260은 seed42/43/44의 joint13 참조다. 현재 baseline 표는 이전 R18 seed101/102/103 참조다.
교사 입력과 다른 하이퍼파라미터도 달라질 수 있어 순수한 EMA 수정의 인과효과라고 주장하지 않는다.

## bottleneck 탐색

| trial | state | 768 dev best | last | params |
|---|---|---|---|---|
| 0 | COMPLETE | 0.7418775896860009 | 0.7418775896860009 | `{'loss.cov_iso_mean_lambda': 1.4948540903362064, 'loss.cov_iso_cov_lambda': 2.247944015207464, 'train.lr': 6.545727532374416e-05, 'train.head_lr': 0.00039969567127027346, 'train.grad_clip': 5.337049917615548, 'train.weight_decay': 0.13821244390893206, 'train.warmup_frac': 0.08771058784538847, 'train.momentum_end': 0.9981343830152652, 'train.momentum_ramp_frac': 0.34038897874030694, 'loss.teacher_temp': 0.12506484413951288, 'loss.warmup_teacher_temp': 0.05837008069937486, 'loss.student_temp': 0.12655703783527755, 'loss.teacher_temp_warmup_frac': 0.16719777593815738, 'loss.center_momentum': 0.9187320587014133, 'loss.cov_ema_momentum': 0.7879439828646531, 'augment.t_lo': 0.4664484163749474, '_t_span': 0.4355063277399266, '_t_start_frac': 0.10809492348141264, 'augment.warmup_frac': 0.0019071258426626332}` |
| 1 | COMPLETE | 0.7508694272284513 | 0.7507333762299004 | `{'loss.cov_iso_mean_lambda': 5.874093568592943, 'loss.cov_iso_cov_lambda': 5.874093568592943, 'train.lr': 6.876495483323049e-05, 'train.head_lr': 0.0005824833585133, 'train.grad_clip': 5.590297628724676, 'train.weight_decay': 0.226, 'train.warmup_frac': 0.0517088539939031, 'train.momentum_end': 0.9985014352751024, 'train.momentum_ramp_frac': 0.4077115063691495, 'loss.teacher_temp': 0.1347501197840043, 'loss.warmup_teacher_temp': 0.0494051972253961, 'loss.student_temp': 0.151, 'loss.teacher_temp_warmup_frac': 0.0789677221858058, 'loss.center_momentum': 0.92, 'loss.cov_ema_momentum': 0.8195105695006707, 'augment.t_lo': 0.3919166650119331, '_t_span': 0.48920595718665394, '_t_start_frac': 0.1469538453927879, 'augment.warmup_frac': 0.02545633588184152}` |
| 2 | COMPLETE | 0.6326783144358481 | 0.6135589693880111 | `{'loss.cov_iso_mean_lambda': 0.07017770604424832, 'loss.cov_iso_cov_lambda': 0.056061650408123234, 'train.lr': 3.7227944137975816e-05, 'train.head_lr': 0.0006187755805635586, 'train.grad_clip': 4.9365060815109185, 'train.weight_decay': 0.03340796023310159, 'train.warmup_frac': 0.18030601865089121, 'train.momentum_end': 0.9993829449041193, 'train.momentum_ramp_frac': 0.39089862038317796, 'loss.teacher_temp': 0.09756008250150149, 'loss.warmup_teacher_temp': 0.06220108665356782, 'loss.student_temp': 0.13380050029773224, 'loss.teacher_temp_warmup_frac': 0.2631897263896739, 'loss.center_momentum': 0.8356269242015258, 'loss.cov_ema_momentum': 0.8389946479497375, 'augment.t_lo': 0.580529277217129, '_t_span': 0.4066112761378284, '_t_start_frac': 0.1556060668028934, 'augment.warmup_frac': 0.08707391992715142}` |
| 3 | COMPLETE | 0.6016672366622591 | 0.5671947766435624 | `{'loss.cov_iso_mean_lambda': 0.05685019613864201, 'loss.cov_iso_cov_lambda': 0.07210922899054537, 'train.lr': 2.8738950632815198e-05, 'train.head_lr': 0.00012645552379914212, 'train.grad_clip': 5.494394307367731, 'train.weight_decay': 0.06248213732140922, 'train.warmup_frac': 0.03661627501411219, 'train.momentum_end': 0.9982895260667638, 'train.momentum_ramp_frac': 0.5378339263923317, 'loss.teacher_temp': 0.17848211163602723, 'loss.warmup_teacher_temp': 0.0421819995345288, 'loss.student_temp': 0.11150749661254404, 'loss.teacher_temp_warmup_frac': 0.03259840829980853, 'loss.center_momentum': 0.8545795862098643, 'loss.cov_ema_momentum': 0.9635499500051959, 'augment.t_lo': 0.12487410600080345, '_t_span': 0.29866210973195045, '_t_start_frac': 0.30865299057523515, 'augment.warmup_frac': 0.14765847717938105}` |
| 4 | COMPLETE | 0.671034626545087 | 0.6559504975794033 | `{'loss.cov_iso_mean_lambda': 2.8409167559332205, 'loss.cov_iso_cov_lambda': 0.1199702689470352, 'train.lr': 2.9296344482476285e-05, 'train.head_lr': 0.0005294531098328371, 'train.grad_clip': 4.147220487157387, 'train.weight_decay': 0.10502689492846337, 'train.warmup_frac': 0.11560340044783456, 'train.momentum_end': 0.9983724562633798, 'train.momentum_ramp_frac': 0.27446077908737226, 'loss.teacher_temp': 0.11075199061874669, 'loss.warmup_teacher_temp': 0.050174617158267265, 'loss.student_temp': 0.1432784797515098, 'loss.teacher_temp_warmup_frac': 0.2733248271701453, 'loss.center_momentum': 0.9509264309580696, 'loss.cov_ema_momentum': 0.8475224734170657, 'augment.t_lo': 0.31452339461628614, '_t_span': 0.49004030485855055, '_t_start_frac': 0.9015431058080938, 'augment.warmup_frac': 0.06397946719608526}` |
| 5 | COMPLETE | 0.7379415414145603 | 0.7300151490284955 | `{'loss.cov_iso_mean_lambda': 2.057155824889888, 'loss.cov_iso_cov_lambda': 6.811776339691536, 'train.lr': 0.00010010097167433929, 'train.head_lr': 0.0012719189628506618, 'train.grad_clip': 5.965751903573681, 'train.weight_decay': 0.16046619172014906, 'train.warmup_frac': 0.10897180349241739, 'train.momentum_end': 0.9986637603252139, 'train.momentum_ramp_frac': 0.5926669731250752, 'loss.teacher_temp': 0.17345412835159746, 'loss.warmup_teacher_temp': 0.04918410411873384, 'loss.student_temp': 0.18834400993905734, 'loss.teacher_temp_warmup_frac': 0.11818738274084684, 'loss.center_momentum': 0.8873876046935931, 'loss.cov_ema_momentum': 0.8512472474691837, 'augment.t_lo': 0.4361495179418676, '_t_span': 0.41023052033187835, '_t_start_frac': 0.5409983506045668, 'augment.warmup_frac': 0.045138042838815576}` |
| 6 | COMPLETE | 0.6592102591705532 | 0.6571050114282772 | `{'loss.cov_iso_mean_lambda': 7.377303651099855, 'loss.cov_iso_cov_lambda': 0.8123264998254532, 'train.lr': 4.209773466743342e-05, 'train.head_lr': 0.0013952467053881575, 'train.grad_clip': 3.4527342128449554, 'train.weight_decay': 0.16632301280300388, 'train.warmup_frac': 0.15155707613188973, 'train.momentum_end': 0.9996069974295528, 'train.momentum_ramp_frac': 0.3683640155399158, 'loss.teacher_temp': 0.14272839054932202, 'loss.warmup_teacher_temp': 0.06184942280074804, 'loss.student_temp': 0.11539582167779605, 'loss.teacher_temp_warmup_frac': 0.06160178667205788, 'loss.center_momentum': 0.9362171550881745, 'loss.cov_ema_momentum': 0.8523540032787028, 'augment.t_lo': 0.46286043971140656, '_t_span': 0.5613859415328457, '_t_start_frac': 0.11741613379574725, 'augment.warmup_frac': 0.00622004215875014}` |
| 7 | COMPLETE | 0.7229259809640912 | 0.7227579528206939 | `{'loss.cov_iso_mean_lambda': 1.8294368198679152, 'loss.cov_iso_cov_lambda': 1.5838983182197353, 'train.lr': 5.510813134312931e-05, 'train.head_lr': 0.000383328114290917, 'train.grad_clip': 5.85042830858527, 'train.weight_decay': 0.025715014965830157, 'train.warmup_frac': 0.08573511439631061, 'train.momentum_end': 0.9980021088292867, 'train.momentum_ramp_frac': 0.225425021782804, 'loss.teacher_temp': 0.1673276505156831, 'loss.warmup_teacher_temp': 0.04358593633065442, 'loss.student_temp': 0.20528822832750832, 'loss.teacher_temp_warmup_frac': 0.023454472149103595, 'loss.center_momentum': 0.8811235229620862, 'loss.cov_ema_momentum': 0.7789405830758627, 'augment.t_lo': 0.34093945037674894, '_t_span': 0.4499251627912057, '_t_start_frac': 0.1840296519514704, 'augment.warmup_frac': 0.067316263647764}` |
| 8 | RUNNING |  |  | `{'loss.cov_iso_mean_lambda': 5.829874670999802, 'loss.cov_iso_cov_lambda': 1.058396564059913, 'train.lr': 2.5263914386132785e-05, 'train.head_lr': 0.0013774488374124977, 'train.grad_clip': 4.578992448002537, 'train.weight_decay': 0.11958875533282551, 'train.warmup_frac': 0.037614409037069384, 'train.momentum_end': 0.9982358758583776, 'train.momentum_ramp_frac': 0.6353262936087805, 'loss.teacher_temp': 0.17060746653759962, 'loss.warmup_teacher_temp': 0.03873083680725507, 'loss.student_temp': 0.15775580508271478, 'loss.teacher_temp_warmup_frac': 0.06623617914504949, 'loss.center_momentum': 0.9486823366354393, 'loss.cov_ema_momentum': 0.8318544256050342, 'augment.t_lo': 0.5624192235399205, '_t_span': 0.2764823256812321, '_t_start_frac': 0.3273419564297515, 'augment.warmup_frac': 0.027507168085700757}` |
| 9 | RUNNING |  |  | `{'loss.cov_iso_mean_lambda': 0.554690152344319, 'loss.cov_iso_cov_lambda': 5.205824078278007, 'train.lr': 5.649715806367647e-05, 'train.head_lr': 0.00024174729539425225, 'train.grad_clip': 5.255270914911297, 'train.weight_decay': 0.10773797089829296, 'train.warmup_frac': 0.020255294836462927, 'train.momentum_end': 0.9991865323212685, 'train.momentum_ramp_frac': 0.4514884381425672, 'loss.teacher_temp': 0.147570868171874, 'loss.warmup_teacher_temp': 0.04692979312855461, 'loss.student_temp': 0.13350149071711057, 'loss.teacher_temp_warmup_frac': 0.07273991504051609, 'loss.center_momentum': 0.9326195003793379, 'loss.cov_ema_momentum': 0.8617801761793666, 'augment.t_lo': 0.5498839211716721, '_t_span': 0.41035669037940603, '_t_start_frac': 0.5239065584016767, 'augment.warmup_frac': 0.06940237546696243}` |

## 마지막 관측 진단 (선택에 사용하지 않음)

| run | step | 256 rank | 256 mean norm | 256 uniformity | view cosine | weighted iso mean/cov |
|---|---|---|---|---|---|---|
| bneckiso_bneck_iso_20261104_confirm_baseline_t0_s101 | 3999 | 58.24271443792668 | 0.08900696039199829 | -2.2194650173187256 | 0.7242178618907928 | 0.035260845184868286 / 0.04038382958562702 |
| bneckiso_bneck_iso_20261104_confirm_baseline_t0_s102 | 3999 | 54.808250485742654 | 0.10002864897251129 | -2.0675079822540283 | 0.7907576113939285 | 0.038014851839450546 / 0.04128703818428425 |
| bneckiso_bneck_iso_20261104_confirm_baseline_t0_s103 | 3999 | 55.38466244482019 | 0.08411945402622223 | -2.3781545162200928 | 0.8014882206916809 | 0.04020567702345093 / 0.04321064273103442 |
| full_ema_full_ema_joint_20261110_search_bottleneck_t0_s42 | 3999 | 48.52988765327737 | 0.219265878200531 | -2.312138795852661 |  | 0.010082643819900577 / 0.017212109241421726 |
| full_ema_full_ema_joint_20261110_search_bottleneck_t1_s42 | 3999 | 53.71320070169955 | 0.10664977133274078 | -2.2814314365386963 |  | 0.03547400733775974 / 0.03956955229514528 |
| full_ema_full_ema_joint_20261110_search_bottleneck_t2_s42 | 3999 | 57.16483295922368 | 0.11953799426555634 | -2.0767509937286377 |  | 0.000707033925997812 / 0.0008796682142899671 |
| full_ema_full_ema_joint_20261110_search_bottleneck_t3_s42 | 3999 | 2.9779839623176065 | 0.3397549092769623 | -1.0738633871078491 |  | 0.00032926994840115826 / 0.001515818922783759 |
| full_ema_full_ema_joint_20261110_search_bottleneck_t4_s42 | 3999 | 30.872043246578976 | 0.11067487299442291 | -2.0325794219970703 |  | 0.02633256597717311 / 0.0022060083155229196 |
| full_ema_full_ema_joint_20261110_search_bottleneck_t5_s42 | 3999 | 43.44386552128828 | 0.13114367425441742 | -2.7145814895629883 |  | 0.00984493877917094 / 0.033075332243472574 |
| full_ema_full_ema_joint_20261110_search_bottleneck_t6_s42 | 3999 | 24.03280301956844 | 0.17132841050624847 | -2.271582841873169 |  | 0.024077550239594632 / 0.006499986180292292 |
| full_ema_full_ema_joint_20261110_search_bottleneck_t7_s42 | 3999 | 73.0016348111844 | 0.04540933296084404 | -2.789815664291382 |  | 0.015039317010212182 / 0.01642827124509618 |
| full_ema_full_ema_joint_20261110_search_bottleneck_t8_s42 | 3500 | 12.777479084927734 | 0.2172398865222931 | -1.7480438947677612 |  | 0.025281637440432426 / 0.007441155226478065 |
| full_ema_full_ema_joint_20261110_search_bottleneck_t9_s42 | 2500 | 15.933695390084077 | 0.16120709478855133 | -1.6819820404052734 |  | 0.0030916148699951596 / 0.020820659826158463 |
| full_ema_full_ema_joint_20261110_smoke_baseline_s42 | 59 | 97.86142877536967 | 0.3058982491493225 | -2.601166009902954 |  | 0.5768529530725361 / 0.049236543641375766 |
| full_ema_full_ema_joint_20261110_smoke_bottleneck_s42 | 59 | 97.84669879731388 | 0.30583685636520386 | -2.600468873977661 |  | 0.5781593854448412 / 0.04906302093080241 |

성분별 raw loss·backbone/MLP gradient는 components.jsonl과 gradient_diagnostics.jsonl에 보존한다.

## 동결 후 STS7 test

| group | seed | checkpoint | 768 STS7 ×100 | 256 STS7 ×100 (진단) |
|---|---|---|---|---|
| baseline | 101 | best | 65.4942 | 59.4499 |
| baseline | 101 | last | 65.8205 | 57.8691 |
| baseline | 102 | best | 65.4591 | 55.8487 |
| baseline | 102 | last | 65.4591 | 55.8487 |
| baseline | 103 | best | 66.4953 | 59.1656 |
| baseline | 103 | last | 66.4953 | 59.1656 |

baseline_best: n=3, mean 65.8162, sample SD 0.5883.

baseline_last: n=3, mean 65.9250, sample SD 0.5259.

개별 7-task 점수는 results/analysis/*.json에 모두 보존한다.

## 해석과 미실행 항목

미완료 단계와 점수는 추정해서 채우지 않는다. 완료 상태 전에는 최종 결론을 내리지 않는다.
새 완전EMA FlowDINO와 역사적 기준선의 비교다. 공동 튜닝 및 teacher 입력이 함께 달라져 특정 변경의 인과 효과를 분리하지 못한다.
256차원 bottleneck 지표만 개선되면 768차원 임베딩 개선으로 판정하지 않는다.
과거 R18 보고 점수 대신 현재 campaign의 대조군을 비교 기준으로 삼는다.

실행 오류: `143`
