"""Dev-only selection summaries, measured diagnostics, and frozen STS7 reports."""

import argparse
import json
import statistics
import sys
from pathlib import Path

import optuna
import yaml
from optuna.trial import TrialState

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.joint_search import config_from_parameters
from src.provenance import read_json, read_records, write_json
from src.train import load_config


def judgment(control, candidate):
    deltas = [
        candidate[s]["metrics"]["best_sts"] - control[s]["metrics"]["best_sts"]
        for s in (42, 43, 44)
    ]
    mean = statistics.mean(deltas)
    status = (
        "성립"
        if mean >= 0.003 and sum(d > 0 for d in deltas) >= 2
        else ("기각" if mean <= 0 and sum(d <= 0 for d in deltas) >= 2 else "판정불가")
    )
    return {
        "verdict": status,
        "paired_best_deltas": deltas,
        "paired_best_mean": mean,
        "paired_last_mean": statistics.mean(
            candidate[s]["metrics"]["last_sts"] - control[s]["metrics"]["last_sts"]
            for s in (42, 43, 44)
        ),
    }


def render(campaign, manifest, study, stage, confirmations=None, error=None):
    trials = study.get_trials(deepcopy=False)
    counts = {state.name: sum(t.state == state for t in trials) for state in TrialState}
    lines = [
        "# 공동 Optuna 최적화 — 실행 보고서",
        "",
        f"상태: **{stage}**. `{campaign}`.",
        "",
        "공동 성능 최적화 사전 등록: OPTUNA_SPEC.md.",
        "여러 축을 함께 바꾸므로 특정 정규화 항의 인과 효과를 이 결과로 판정하지 않는다.",
        "",
        f"실행 소스: `{manifest['git']['commit']}`, branch `{manifest['git']['branch']}`.",
        f"기준선 원격 소스: `{manifest['baseline']['upstream_commit']}`.",
        f"실제 R18 대조군 λ₀={manifest['baseline']['lambda_0']} (재구성 기준선 아님).",
        "실제 명령·exit code: campaign의 commands/ 및 status/. 실행 소스와 diff: source_manifest.json, source_diff.patch, source/.",
        (
            "control 42/43은 핵심 소스·resolved config·데이터 해시·4000-step 완료를 확인하고 재사용했다."
            if (campaign / "imported_control.json").exists()
            else "control 42/43은 같은 실행 소스와 예산으로 독립 학습했다."
        ),
        f"trial 상태: `{json.dumps(counts)}`. 최대 예산 `{manifest['trials']}` (pruned/failed/기준선 포함).",
        "",
        "## seed42 탐색 결과 (완주 trial만 후보 자격)",
        "",
        "| trial | state | dev best | dev last | best step |",
        "|---|---|---|---|---|",
    ]
    for t in trials:
        m = t.user_attrs.get("run", {}).get("metrics", {})
        lines.append(
            f"| {t.number} | {t.state.name} | {m.get('best_sts', '')} | {m.get('last_sts', '')} | {m.get('best_loop_step', '')} |"
        )
    lines += [
        "",
        "trial별 정확한 19개 값은 trials.json, resolved config는 configs/ 및 각 run의 resolved.yaml에 기록한다.",
        "진단은 각 로그의 components.jsonl·gradient_diagnostics.jsonl·evaluations.jsonl에 있다.",
    ]
    if confirmations:
        lines += [
            "",
            "## 추가 시드 검증",
            "",
            "| 설정 | seed | best dev | last dev | alignment last | uniformity last | effective rank last |",
            "|---|---|---|---|---|---|---|",
        ]
        for label, seeds in sorted(confirmations.items()):
            for seed, run in sorted(seeds.items()):
                m = run["metrics"]
                rows = read_records(
                    campaign / "results/logs" / run["run_name"] / "evaluations.jsonl"
                )
                last = rows[-1]
                lines.append(
                    f"| {label} | {seed} | {m['best_sts']:.6f} | {m['last_sts']:.6f} | {last['alignment']:.5f} | {last['uniformity']:.5f} | {last['eff_rank']:.2f} |"
                )
        lines += [
            "",
            "| 설정 | n | best mean ± sample SD | last mean ± sample SD |",
            "|---|---|---|---|",
        ]
        for label, seeds in sorted(confirmations.items()):
            vals = [
                [run["metrics"][key] for run in seeds.values()] for key in ("best_sts", "last_sts")
            ]
            summaries = [
                f"{statistics.mean(v):.6f} ± {statistics.stdev(v) if len(v) > 1 else 0:.6f}"
                for v in vals
            ]
            lines.append(f"| {label} | {len(seeds)} | {summaries[0]} | {summaries[1]} |")
    selection_path = campaign / "selection.json"
    if selection_path.exists():
        selection = read_json(selection_path)
        lines += [
            "",
            f"최종 후보: trial `{selection['candidate']}`. 판정: **{selection['judgment']['verdict']}**.",
            f"paired best 평균 Δ={selection['judgment']['paired_best_mean']:.6f}.",
            "선택은 3시드 dev만 사용했고 selection.json을 동결한 뒤 test 평가를 시작했다.",
            "재사용 기준선: configs/coviso_relax/control.yaml. 선택 조합: configs/coviso_relax/selected_joint.yaml.",
        ]
    evaluations = (
        sorted((campaign / "results/analysis").glob("sts7_*.json"))
        if (campaign / "results/analysis").exists()
        else []
    )
    if evaluations:
        scores = {}
        lines += [
            "",
            "## STS7 test (선택 후 측정, ×100)",
            "",
            "| 설정/seed/checkpoint | STS12 | STS13 | STS14 | STS15 | STS16 | STS-B | SICK-R | 평균 |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for path in evaluations:
            data = read_json(path)
            row = next(iter(data["runs"].values()))
            label = path.stem.removeprefix("sts7_")
            lines.append(
                "| "
                + label
                + " | "
                + " | ".join(
                    f"{row[k] * 100:.3f}"
                    for k in (
                        "STS12",
                        "STS13",
                        "STS14",
                        "STS15",
                        "STS16",
                        "STSBenchmark",
                        "SICK-R",
                        "avg_7task",
                    )
                )
                + " |"
            )
            group = label.rsplit("_s", 1)[0] + "_" + label.rsplit("_", 1)[1]
            scores.setdefault(group, []).append(row["avg_7task"] * 100)
        for group, vals in sorted(scores.items()):
            lines.append(
                f"\n{group}: n={len(vals)}, 평균 {statistics.mean(vals):.3f}, sample SD {statistics.stdev(vals) if len(vals) > 1 else 0:.3f}."
            )
    if confirmations:
        lines += [
            "",
            "## 마지막 step 손실과 backbone gradient norm",
            "",
            "gradient는 각 가중치를 적용한 항을 독립 미분한 값이며 벡터 방향이 달라 norm의 합이 총 norm은 아니다.",
            "",
            "| 설정/seed | DINO | mean raw | cov raw | mean 기여 | cov 기여 | g DINO | g mean | g cov |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        diagnostics = {}
        for label, seeds in sorted(confirmations.items()):
            for seed, run in sorted(seeds.items()):
                logdir = campaign / "results/logs" / run["run_name"]
                records = read_records(logdir / "components.jsonl") + read_records(
                    logdir / "gradient_diagnostics.jsonl"
                )
                rows = sorted(
                    (r for r in records if "grad_dino_backbone" in r),
                    key=lambda r: r["loop_step"],
                )
                if not rows:
                    continue
                last = rows[-1]
                diagnostics[f"{label}_s{seed}"] = {
                    "first": rows[0],
                    "last": last,
                    "sampled_gradient_steps": rows,
                }
                lines.append(
                    f"| {label}/s{seed} | "
                    + " | ".join(
                        f"{last[k]:.6g}"
                        for k in (
                            "L_DINO",
                            "L_mean",
                            "L_cov",
                            "weighted_mean",
                            "weighted_cov",
                            "grad_dino_backbone",
                            "grad_mean_backbone",
                            "grad_cov_backbone",
                        )
                    )
                    + " |"
                )
        write_json(campaign / "diagnostic_summary.json", diagnostics)
    lines += [
        "",
        "## 해석과 남은 제약",
        "",
        "R18 과거 dev 약0.7465, 7-task 약66.09는 참고 기록이다. 현재 환경의 대조군을 비교 기준으로 사용한다.",
        "노이즈 강도·최적화·EMA가 함께 바뀌므로 어느 단일 성분 완화의 인과 효과인지는 미확인이다.",
        "alignment/uniformity/rank는 진단 지표다. 손실의 절대 크기만으로 지배 항을 판단하지 않는다.",
        "미완주 trial 및 미평가 test 결과를 추정치로 채우지 않는다.",
    ]
    if error:
        lines += ["", f"실행 오류: `{error}`"]
    text = "\n".join(lines) + "\n"
    (campaign / "report.md").write_text(text)
    Path(manifest["report"]).write_text(text)
    write_json(
        campaign / "trials.json",
        [
            {
                "number": t.number,
                "state": t.state.name,
                "value": t.value,
                "params": t.params,
                "user_attrs": t.user_attrs,
                "intermediate_values": t.intermediate_values,
            }
            for t in trials
        ],
    )
    write_json(
        campaign / "state.json",
        {"stage": stage, "trial_counts": counts, "error": error},
    )


def export_selected_config(campaign, base, destination):
    """Build a clean config from the frozen 19-value dev selection."""
    if destination.exists():
        raise RuntimeError(f"preserving existing selected config: {destination}")
    selection = read_json(campaign / "selection.json")
    cfg = config_from_parameters(base, selection["params"])
    cfg.update(
        run_name="r18_bert_joint_optuna_selected_s42",
        seed=42,
        experiment={
            "selection_campaign": str(campaign),
            "selected_trial": selection["candidate"],
            "selected_from_dev_only": True,
            "recommended": selection["judgment"]["verdict"] == "성립",
        },
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(cfg, sort_keys=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument(
        "--out-config",
        type=Path,
        help="export the frozen selection without overwriting a config",
    )
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    manifest = read_json(campaign / "source_manifest.json")
    study = optuna.load_study(
        storage=f"sqlite:///{campaign / 'study.db'}", study_name=campaign.name
    )
    state = read_json(campaign / "state.json")
    confirmations = None
    if (campaign / "confirmations.json").exists():
        confirmations = read_json(campaign / "confirmations.json")
    elif (campaign / "control_runs.json").exists():
        confirmations = {"control": read_json(campaign / "control_runs.json")}
    elif (campaign / "imported_control.json").exists():
        confirmations = {"control": read_json(campaign / "imported_control.json")["runs"]}
    render(campaign, manifest, study, state["stage"], confirmations, state.get("error"))
    if args.out_config:
        export_selected_config(
            campaign,
            load_config(ROOT / "configs/coviso_relax/base.yaml"),
            args.out_config,
        )


if __name__ == "__main__":
    main()
