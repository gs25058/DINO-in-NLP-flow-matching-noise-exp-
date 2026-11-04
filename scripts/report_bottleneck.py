"""Measured results for the paired bottleneck experiment; never selects on test."""

import argparse
import statistics
import sys
from pathlib import Path

import optuna

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.provenance import read_json, read_records


def judge_deltas(deltas):
    mean = statistics.mean(deltas)
    verdict = (
        "성립"
        if mean >= 0.003 and sum(d > 0 for d in deltas) >= 2
        else "기각"
        if mean <= 0 and sum(d <= 0 for d in deltas) >= 2
        else "판정불가"
    )
    return {
        "verdict": verdict,
        "paired_best_deltas": deltas,
        "paired_mean": mean,
        "paired_sample_sd": statistics.stdev(deltas),
    }


def paired_judgment(left, right):
    return judge_deltas(
        [
            right[s]["metrics"]["best_sts"] - left[s]["metrics"]["best_sts"]
            for s in ("101", "102", "103")
        ]
    )


def render(campaign):
    manifest = read_json(campaign / "source_manifest.json")
    state = read_json(campaign / "state.json")
    lines = [
        "# Bottleneck cosine — 공정한 공동 튜닝 비교",
        "",
        f"상태: **{state['stage']}**.",
        f"campaign: `{campaign}`. 실행 소스 `{manifest['git']['commit']}`, `{manifest['git']['branch']}`.",
        "R18 실제 4000-step 대조군을 사용한다. covariance와 noise는 양쪽에서 고정한다.",
        f"각 조건 {manifest['trials_per_arm']} trial ×4000 step; pruning 없음. 재검증 seed101/102/103.",
        "주 지표는 768차원 embedding STS-B dev. 256차원 STS와 view cosine은 진단이다.",
        "",
        "전체 명령·exit code: commands/와 status/. resolved config: configs/. 소스 diff와 SHA: source_manifest.json, source_diff.patch.",
        f"실행기: `{' '.join(manifest.get('launch_argv', []))}`",
        "",
    ]
    calibration = campaign / "calibration.json"
    if calibration.exists():
        cal = read_json(calibration)
        lines += [
            "## 탐색 공간과 보정",
            "",
            f"공통 범위: `{manifest['common_space']}`.",
            f"초기 g_DINO={cal['grad_dino_backbone']:.8g}, g_bneck={cal['grad_bneck_unweighted_backbone']:.8g}.",
            f"λ 동등값={cal['lambda_equivalent']:.8g}, log 범위={cal['lambda_range']}, 시작 λ={cal['lambda_anchor']:.8g}.",
            f"상태 보존 검증: `{cal['state_proof']}`.",
            "초기 한 minibatch의 gradient 규모이며 학습 전체에서 같은 비율이라고 가정하지 않는다.",
            "",
        ]
    for arm in ("baseline", "bottleneck"):
        database = campaign / f"{arm}.db"
        if not database.exists():
            continue
        study = optuna.load_study(
            study_name=f"{campaign.name}_{arm}", storage=f"sqlite:///{database}"
        )
        lines += [
            f"## {arm} 탐색",
            "",
            "| trial | state | 768 dev best | last | params |",
            "|---|---|---|---|---|",
        ]
        for trial in study.trials:
            m = trial.user_attrs.get("run", {}).get("metrics", {})
            lines.append(
                f"| {trial.number} | {trial.state.name} | {m.get('best_sts', '')} | {m.get('last_sts', '')} | `{trial.params}` |"
            )
    confirmations = campaign / "confirmations.json"
    if confirmations.exists():
        confirmed = read_json(confirmations)
        for arm, candidates in confirmed.items():
            lines += [
                f"## {arm} 상위 설정 재검증",
                "",
                "| trial | seed | 768 best | 768 last | 256 dev last (진단) | view cosine last |",
                "|---|---|---|---|---|---|",
            ]
            for number, seeds in candidates.items():
                for seed, run in seeds.items():
                    m = run["metrics"]
                    logdir = campaign / "results/logs" / run["run_name"]
                    ev = read_records(logdir / "evaluations.jsonl")[-1]
                    loss = read_records(logdir / "components.jsonl")[-1]
                    lines.append(
                        f"| {number} | {seed} | {m['best_sts']:.6f} | {m['last_sts']:.6f} | {ev.get('sts_b_dev_bottleneck', '')} | {loss.get('bneck_cosine', '')} |"
                    )
                vals = [r["metrics"]["best_sts"] for r in seeds.values()]
                if vals:
                    lines.append(
                        f"\ntrial {number}: best mean {statistics.mean(vals):.6f}, sample SD {statistics.stdev(vals) if len(vals) > 1 else 0:.6f}.\n"
                    )
    final = campaign / "final_runs.json"
    if final.exists():
        groups = read_json(final)
        lines += [
            "## 최종 설정과 λ=0 비교",
            "",
            "| group | seed | 768 dev best | last |",
            "|---|---|---|",
        ]
        for group, seeds in groups.items():
            for seed, run in seeds.items():
                m = run["metrics"]
                lines.append(f"| {group} | {seed} | {m['best_sts']:.6f} | {m['last_sts']:.6f} |")
            vals = [r["metrics"]["best_sts"] for r in seeds.values()]
            lines.append(
                f"\n{group}: mean {statistics.mean(vals):.6f}, sample SD {statistics.stdev(vals):.6f}.\n"
            )
        if "aux_off" in groups:
            lines += [
                "",
                f"dev 차이(선택 지표), 동일 예산: `{paired_judgment(groups['baseline'], groups['bottleneck'])}`.",
                f"dev 차이, 동일 hyperparameter 보조항: `{paired_judgment(groups['aux_off'], groups['bottleneck'])}`.",
            ]
    evaluations = (
        sorted((campaign / "results/analysis").glob("*.json"))
        if (campaign / "results/analysis").exists()
        else []
    )
    if evaluations:
        aggregates = {}
        lines += [
            "",
            "## 동결 후 STS7 test",
            "",
            "| group | seed | checkpoint | 768 STS7 ×100 | 256 STS7 ×100 (진단) |",
            "|---|---|---|---|---|",
        ]
        for path in evaluations:
            doc = read_json(path)
            if "experiment" not in doc:
                continue
            info = doc["experiment"]
            score = next(iter(doc["runs"].values()))
            diagnostic = next(iter(doc["bottleneck_diagnostic"].values()))
            lines.append(
                f"| {info['group']} | {info['seed']} | {info['which']} | {score['avg_7task'] * 100:.4f} | {diagnostic['avg_7task'] * 100:.4f} |"
            )
            aggregates.setdefault(f"{info['group']}_{info['which']}", []).append(
                score["avg_7task"] * 100
            )
        for key, values in aggregates.items():
            lines.append(
                f"\n{key}: n={len(values)}, mean {statistics.mean(values):.4f}, sample SD {statistics.stdev(values) if len(values) > 1 else 0:.4f}."
            )
        if all(
            len(aggregates.get(f"{group}_best", [])) == 3
            for group in ("baseline", "bottleneck", "aux_off")
        ):
            for group in ("baseline", "aux_off"):
                deltas = [
                    (right - left) / 100
                    for left, right in zip(
                        aggregates[f"{group}_best"], aggregates["bottleneck_best"]
                    )
                ]
                lines += [
                    "",
                    f"**주 평가: bottleneck 대 {group}, 768d STS7 best: {judge_deltas(deltas)}**",
                ]
        lines += ["", "개별 7-task 점수는 results/analysis/*.json에 모두 보존한다."]
    lines += [
        "",
        "## 해석과 미실행 항목",
        "",
        "미완료 단계와 점수는 추정해서 채우지 않는다. 완료 상태 전에는 최종 결론을 내리지 않는다.",
        "다른 학습률/온도를 고른 양 조건의 차이는 동일 튜닝 예산 비교이며, 보조항 자체의 효과는 aux_off와 비교한다.",
        "256차원 bottleneck 지표만 개선되면 768차원 임베딩 개선으로 판정하지 않는다.",
        "과거 R18 보고 점수 대신 현재 campaign의 대조군을 비교 기준으로 삼는다.",
    ]
    if state.get("error"):
        lines += ["", f"실행 오류: `{state['error']}`"]
    text = "\n".join(lines) + "\n"
    (campaign / "report.md").write_text(text)
    Path(manifest["report"]).write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    render(parser.parse_args().campaign.resolve())


if __name__ == "__main__":
    main()
