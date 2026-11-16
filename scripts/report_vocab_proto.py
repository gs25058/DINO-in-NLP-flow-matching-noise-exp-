"""Render the vocab-prototype campaign report and the pre-registered P1–P5 verdicts.

Missing stages are shown as unexecuted, never estimated.
"""

import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.provenance import read_json, read_records  # noqa: E402

CONDITIONS = ("C1", "C2", "C3")
SEEDS = (42, 43, 44)
C0_CAMPAIGN = "results/experiments/coviso_joint_optuna_20261028"
C0_RUN = "r18_bert_joint_optuna_coviso_joint_optuna_20261028_t13_s{seed}"
C0_K = 8192
STEPS = (0, 2000, 3999)
NAMES = {"C0": "C0 t13 (random 8192)", "C1": "C1 word head, trained", "C2": "C2 word head, frozen",
         "C3": "C3 k-means 8192, random transform"}
# P2 thresholds (pre-registered): t13 H_pt/lnK ≈ 0.997, t13 batch_KL 0.047.
P2_MAX_HNORM = 0.95
P2_MIN_KL = 0.47
P5_MIN_SAMPLES = 200


def verdict(diffs):
    if len(diffs) != 3:
        return "판정불가(미실행)"
    mean = statistics.mean(diffs)
    if mean >= 0.003 and sum(d > 0 for d in diffs) >= 2:
        return "성립"
    if mean <= 0 and sum(d <= 0 for d in diffs) >= 2:
        return "기각"
    return "판정불가"


def fmt(x, scale=100, digits=2):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * scale:.{digits}f}"


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def logs(campaign, c, seed, final):
    if c == "C0":
        base = Path(read_json(campaign / "source_manifest.json")["data_root"]) / C0_CAMPAIGN
        return base / "results/logs" / C0_RUN.format(seed=seed)
    run = final.get(c, {}).get(str(seed))
    return campaign / "results/logs" / run["run_name"] if run else None


def at_step(rows, step):
    match = [r for r in rows if r["loop_step"] == step]
    return match[0] if match else None


def scores(campaign):
    out = {}
    for file in sorted((campaign / "results/analysis").glob("*.json")):
        doc = read_json(file)
        ex = doc.get("experiment")
        if not ex:
            continue
        run = next(iter(doc["runs"]))
        row = {"raw": doc["runs"][run]["avg_7task"], "tasks": doc["runs"][run]}
        for variant, values in doc.get("postprocessed", {}).get(run, {}).items():
            row[variant] = values["avg_7task"]
        out[(ex["condition"], ex["seed"], ex["which"])] = row
    return out


def c0_dev():
    return {42: 0.737162, 43: 0.735175, 44: 0.744721}  # coviso report, t13 best dev (seed 42 = search)


def table1(campaign, final, test):
    lines = ["## 표 1. STS-B dev와 STS7 test (×100, 3시드 42/43/44)", "",
             "| 조건 | dev best 평균 | dev best 시드별 | STS7 raw best 평균 | raw best 시드별 | raw last 평균 "
             "| center_pc2 best 평균 | center_pc2 last 평균 | center best 평균 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c in ("C0", *CONDITIONS):
        if c == "C0":
            dev = [c0_dev()[s] for s in SEEDS]
        else:
            dev = [final[c][str(s)]["metrics"]["best_sts"] if c in final else None for s in SEEDS]

        def get(seed, which, key):
            row = test.get((c, seed, which))
            return row.get(key) if row else None

        raw = [get(s, "best", "raw") for s in SEEDS]
        lines.append(
            f"| {NAMES[c]} | {fmt(mean(dev))} | {' / '.join(fmt(x) for x in dev)} | {fmt(mean(raw))} | "
            f"{' / '.join(fmt(x) for x in raw)} | {fmt(mean([get(s, 'last', 'raw') for s in SEEDS]))} | "
            f"{fmt(mean([get(s, 'best', 'center_pc2') for s in SEEDS]))} | "
            f"{fmt(mean([get(s, 'last', 'center_pc2') for s in SEEDS]))} | "
            f"{fmt(mean([get(s, 'best', 'center') for s in SEEDS]))} |")
    lines += ["", "C0 dev는 coviso 보고서 값이다. C0 STS7은 같은 평가 코드(TF32)로 다시 계산한 값이며, "
              "기록값 67.10(best raw)과의 차이는 아래 재현 항목에 적는다. 후처리 통계는 학습 코퍼스 고정 "
              "10,000문장(seed 0)에서 구했고 평가 데이터는 쓰지 않았다.", ""]
    return lines


def table2(campaign, final):
    lines = ["## 표 2. target 정보량과 기하 (final 3시드 평균; step 0 / 2000 / 3999)", "",
             "| 조건 | H_pt/lnK | batch_KL (student) | teacher pair KL | teacher copy rate | eff_rank |",
             "|---|---|---|---|---|---|"]
    for c in ("C0", *CONDITIONS):
        cells = {k: [] for k in ("h", "kl", "tkl", "copy", "rank")}
        for step in STEPS:
            vals = {k: [] for k in cells}
            for s in SEEDS:
                path = logs(campaign, c, s, final)
                if path is None or not (path / "components.jsonl").exists():
                    continue
                comp = at_step(read_records(path / "components.jsonl"), step)
                ev = at_step(read_records(path / "evaluations.jsonl"), step)
                if comp:
                    vals["h"].append(comp.get("H_pt_norm", comp["H_pt"] / math.log(C0_K)))
                    vals["kl"].append(comp["batch_KL"])
                    vals["tkl"].append(comp.get("teacher_pair_KL"))
                    vals["copy"].append(comp.get("copy_rate_teacher"))
                if ev:
                    vals["rank"].append(ev["eff_rank"])
            for k in cells:
                m = mean(vals[k])
                cells[k].append("—" if m is None else f"{m:.3f}" if k != "rank" else f"{m:.1f}")
        lines.append(f"| {NAMES[c]} | " + " | ".join(" / ".join(v) for v in cells.values()) + " |")
    lines += ["", "batch_KL은 기존 지표(student 분포, 무작위 256쌍 대칭 KL)이고 teacher pair KL은 배치 안 모든 "
              "문장쌍의 teacher(centered, 해당 step teacher 온도) 대칭 KL이다. C0는 copy rate를 기록하지 않았다.", ""]
    return lines


def student_copy(path, from_step=3000):
    totals = {}
    for row in read_records(path / "components.jsonl"):
        if row["loop_step"] < from_step:
            continue
        for key, value in row.items():
            if key.startswith("copy_rate_student_t"):
                bin_ = key[len("copy_rate_student_"):]
                count = row[f"copy_count_student_{bin_}"]
                total = totals.setdefault(bin_, [0.0, 0])
                total[0] += value * count
                total[1] += count
    return {b: (v / n, n) for b, (v, n) in sorted(totals.items()) if n}


def predictions(campaign, final, test):
    lines = ["## 사전 등록 예측 판정", ""]
    raw = {c: {s: (test.get((c, s, "best")) or {}).get("raw") for s in SEEDS} for c in ("C0", *CONDITIONS)}

    def diffs(a, b):
        if any(raw[a][s] is None or raw[b][s] is None for s in SEEDS):
            return []
        return [raw[a][s] - raw[b][s] for s in SEEDS]

    d1 = diffs("C1", "C0")
    p1 = verdict(d1)
    lines.append(f"- **P1** C1 > C0 (STS7 raw, dev-best ckpt): **{p1}**; Δ시드별 "
                 f"{', '.join(f'{d*100:+.2f}' for d in d1) or '미실행'}"
                 f"{f', 평균 {statistics.mean(d1)*100:+.2f}pp' if d1 else ''}.")
    seeds_ok, seeds_bad, detail = 0, 0, []
    for s in SEEDS:
        path = logs(campaign, "C1", s, final)
        if path is None:
            continue
        rows = read_records(path / "components.jsonl")
        hmax = max(r["H_pt_norm"] for r in rows)
        kl = statistics.median(r["batch_KL"] for r in rows)
        a, b = hmax <= P2_MAX_HNORM, kl >= P2_MIN_KL
        seeds_ok += a and b
        seeds_bad += (not a) and (not b)
        detail.append(f"s{s}: max H/lnK {hmax:.3f}, median batch_KL {kl:.3f}")
    p2 = "판정불가(미실행)" if not detail else "성립" if seeds_ok >= 2 else "기각" if seeds_bad >= 2 else "판정불가"
    lines.append(f"- **P2** C1 target 정보량 (max H/lnK ≤ {P2_MAX_HNORM} 그리고 median batch_KL ≥ {P2_MIN_KL}): "
                 f"**{p2}**; {'; '.join(detail) or '미실행'}.")
    if p1 == "기각" and p2 == "성립":
        lines.append("  - 기록: 정보량이 큰 target이 STS로 이어지지 않는다.")
    d3 = diffs("C3", "C1")
    if len(d3) == 3:
        m = statistics.mean(d3)
        p3 = ("prototype 방향의 의미가 핵심(|Δ|≤0.003)" if abs(m) <= 0.003 else
              "MLM head의 학습된 지식이 핵심(C3가 0.003 넘게 낮음)" if m < -0.003 else "판정불가(C3가 C1보다 0.003 넘게 높음)")
        lines.append(f"- **P3** C3 − C1: **{p3}**; Δ시드별 {', '.join(f'{d*100:+.2f}' for d in d3)}, 평균 {m*100:+.2f}pp. "
                     "시드 일관성은 판정 규칙에 없으므로 시드별 값을 함께 본다.")
    else:
        lines.append("- **P3** 판정불가(미실행).")
    d4 = diffs("C1", "C2")
    p4 = verdict(d4)
    meaning = {"성립": "head 적응이 필요하다", "기각": "고정된 의미 좌표계로 충분하다(적응 이득 없음)"}.get(p4, "")
    lines.append(f"- **P4** C1 − C2 (적응 효과): **{p4}**{(' — ' + meaning) if meaning else ''}; Δ시드별 "
                 f"{', '.join(f'{d*100:+.2f}' for d in d4) or '미실행'}.")
    per_seed, detail = [], []
    for s in SEEDS:
        path = logs(campaign, "C1", s, final)
        if path is None:
            continue
        bins = {b: v for b, v in student_copy(path).items() if v[1] >= P5_MIN_SAMPLES}
        if len(bins) < 2:
            per_seed.append("판정불가")
            detail.append(f"s{s}: 표본 충분한 t 구간 {len(bins)}개")
            continue
        (lo_bin, (lo, _)), (hi_bin, (hi, _)) = min(bins.items()), max(bins.items())
        v = "성립" if hi >= 0.9 * lo and lo >= 0.3 else "기각" if hi < 0.7 * lo else "판정불가"
        per_seed.append(v)
        detail.append(f"s{s}: {lo_bin} {lo:.3f} → {hi_bin} {hi:.3f}")
    p5 = next((v for v in ("성립", "기각") if per_seed.count(v) >= 2), "판정불가" if per_seed else "판정불가(미실행)")
    lines.append(f"- **P5** 어휘 복사 퇴화 (step≥3000 student top-10 copy rate, 높은 t ≥ 0.9×낮은 t 그리고 낮은 t ≥ 0.3): "
                 f"**{p5}**; {'; '.join(detail) or '미실행'}.")
    lines.append("")
    return lines


def search_summary(campaign):
    if not (campaign / "search.json").exists():
        return []
    found = read_json(campaign / "search.json")
    lines = ["## Phase 1 탐색 (seed 42, 조건별 독립 TPE, 최대 40 attempted)", "",
             "| 조건 | attempted | complete | pruned | failed | best trial | dev best | τs | τt/τs | 마지막 H/lnK |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for c in CONDITIONS:
        rows = found.get(c, {})
        complete = {n: r for n, r in rows.items() if r["state"] == "COMPLETE"}
        states = [r["state"] for r in rows.values()]
        if complete:
            n, r = max(complete.items(), key=lambda kv: (kv[1]["metrics"]["best_sts"], -int(kv[0])))
            last = read_records(campaign / "results/logs" / r["run_name"] / "components.jsonl")[-1]
            cells = [n, fmt(r["metrics"]["best_sts"]), f"{r['params']['loss.student_temp']:.3f}",
                     f"{r['params']['_teacher_ratio']:.3f}", f"{last['H_pt_norm']:.3f}"]
        else:
            cells = ["—"] * 5
        lines.append(f"| {c} | {len(rows)} | {states.count('COMPLETE')} | {states.count('PRUNED')} | "
                     f"{states.count('FAIL')} | " + " | ".join(cells) + " |")
    lines.append("")
    if (campaign / "selection.json").exists():
        sel = read_json(campaign / "selection.json")
        lines.append("선택(3시드 dev best 평균, test 이전 동결): " + ", ".join(
            f"{c} trial {n} (τs {sel['selected_params'][c]['loss.student_temp']:.3f}, "
            f"τt/τs {sel['selected_params'][c]['_teacher_ratio']:.3f}"
            f"{', 보수적 해 주의' if sel['selected_params'][c]['_teacher_ratio'] > 0.9 else ''})"
            for c, n in sel["selected_trials"].items()) + ".")
        lines.append("")
    return lines


def phase0_lines(campaign):
    if not (campaign / "phase0.json").exists():
        return []
    p0 = read_json(campaign / "phase0.json")
    lines = ["## Phase 0 정합성 (학습 0 step, STS-B dev)", "",
             "| 항목 | probe | 측정 | 통과 |", "|---|---|---|---|"]
    for key, row in p0["checks"].items():
        if "probe" in row:
            lines.append(f"| {key} | {row['probe']} | {row['measured']:.4f} | {'예' if row['ok'] else '아니오'} |")
    for c, row in p0["conditions"].items():
        lines.append(f"\n{c}: K={row['K']}, temp1 H/lnK {row['H_over_lnK_temp1.0']:.3f}, 문장 간 KL "
                     f"{row['KL_pairs_temp1.0']:.3f}, head raw dev {row['head_raw']:.4f}.")
    if (campaign / "smoke.json").exists():
        for c, r in read_json(campaign / "smoke.json").items():
            first = read_records(campaign / "results/logs" / r["run_name"] / "components.jsonl")[0]
            lines.append(f"{c} 학습 step 0 (anchor, warmup teacher temp {first['teacher_temp']:.3f}, center=0): "
                         f"H_pt/lnK {first['H_pt_norm']:.3f}, batch_KL {first['batch_KL']:.3f}, "
                         f"teacher pair KL {first['teacher_pair_KL']:.3f}, teacher copy rate {first['copy_rate_teacher']:.3f}.")
    lines.append("")
    return lines


def render(campaign):
    state = read_json(campaign / "state.json")
    final = read_json(campaign / "final_runs.json") if (campaign / "final_runs.json").exists() else {}
    test = scores(campaign)
    lines = ["# 어휘 prototype DINO (mean pooling 고정)", "", f"Stage: {state['stage']}", "",
             "C0 = t13 재사용(재학습 없음). C1 사전학습 MLM head(학습), C2 같은 head 동결, "
             "C3 단어 임베딩 k-means 8192 + 무작위 transform. 사전 등록: OPTUNA_SPEC.md.", ""]
    lines += phase0_lines(campaign) + search_summary(campaign)
    if final or test:
        lines += table1(campaign, final, test) + table2(campaign, final) + predictions(campaign, final, test)
    lines += ["미실행 단계는 추정하지 않는다. 3시드로 통계적 유의성을 주장하지 않는다."]
    if state.get("error"):
        lines.append("Error: " + state["error"])
    text = "\n".join(lines) + "\n"
    (campaign / "report.md").write_text(text)
    target = Path(read_json(campaign / "source_manifest.json")["report"])
    target.write_text(text)
