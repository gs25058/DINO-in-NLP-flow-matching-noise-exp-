# Bottleneck isotropy 실행 보고

구현·검증 중. GPU 보정과 학습은 아직 실행하지 않았다. H1/H2/H3 모두 **판정불가**.
사전 등록은 `OPTUNA_SPEC.md`, 기준선은 `configs/bottleneck_iso/base.yaml`에 있다.
수정 전 기존 테스트는 71개 통과했다. 이전 cosine 탐색은 사용자 승인으로 종료했으며
`results/experiments/bneck_cosine_20261104/stopped_by_user.json`에 종료 확인을 기록했다.
미실행 점수는 채우지 않는다.
