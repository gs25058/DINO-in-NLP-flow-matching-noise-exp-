# 공동 Optuna 실험

실행 명령과 코드 흐름은 루트 [README](../../../README.md)에 있다.
[사전 등록](OPTUNA_SPEC.md), [실행 보고서](report.md),
[기준선 출처](baseline_provenance.json), [구조 정리 검증](../../refactor-20261028.md)을 참고한다.

기준선은 `configs/coviso_relax/control.yaml`이다. 새 campaign은 현재 소스에서 대조군을 학습한다.
소스·설정·데이터가 같은 완료 대조군만 `--import-baseline`으로 재사용할 수 있다.

진행하던 `results/experiments/coviso_joint_optuna_20261028`은 `e1d2ea3`의 고정 소스로
계속 실행한다. 구조 정리 때문에 중단하거나 다시 샘플링하지 않는다.
현재 코드의 분석기는 이 campaign의 실제 기록도 읽을 수 있고,
`--out-config <새 경로>`로 동결된 선택 값 19개를 현재의 활성 config 형식으로 내보낼 수 있다.
