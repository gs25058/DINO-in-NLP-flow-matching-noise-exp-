"""checkpoints/ 전체를 {pooling: last/first_last} x {postprocess: none/center/center_pc1/
center_pc2} 조합으로 STS-B dev Spearman 소급 재채점 -> results/analysis/rescore.md.
student와 (teacher_state_dict가 있는 체크포인트는) teacher 둘 다 채점한다.

체크포인트 포맷 3종을 모두 처리:
  - 최신: {"state_dict", "model_cfg", "teacher_state_dict"}
  - 중간: {"state_dict", "model_cfg"} (teacher_state_dict 없음)
  - 최구(레거시): state_dict 자체가 최상위(model_cfg 없음) - backbone은 파라미터 이름으로 추론

가장 좋은 (pooling, postprocess) 조합 1개를 student 평균 Spearman 기준으로 고른 뒤,
--best-combo-runs로 지정한 대표 run들만 7-task(SimCSE 스위트) 평균까지 추가 채점한다
(전체 61 checkpoints x 7-task는 비용이 커서 기본은 STS-B dev만).

실행:
    uv run python scripts/rescore_checkpoints.py
    uv run python scripts/rescore_checkpoints.py --best-combo-runs r5d_combined,r2_anchor_curriculum,...
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import AutoTokenizer

from src.evaluate import _load_stsb_dev, embed_sentences, postprocess_embeddings, sts_suite_spearman
from src.model import DinoTextModel

ROOT = Path(__file__).resolve().parent.parent
CKPT_ROOT = ROOT / "checkpoints"
OUT_PATH = ROOT / "results" / "analysis" / "rescore.md"

POOLINGS = ["last", "first_last"]
POSTPROCESSES = ["none", "center", "center_pc1", "center_pc2"]
# Table 1 고정값(CLAUDE.md) - model_cfg 없는 레거시 체크포인트의 fallback.
_LEGACY_BOTTLENECK_DIM = 256
_LEGACY_LOGIT_DIM = 8192


def _infer_backbone_from_state_dict(state_dict: dict) -> str:
    keys = state_dict.keys()
    if any("attn.Wqkv" in k or "final_norm" in k for k in keys):
        return "answerdotai/ModernBERT-base"
    if any("encoder.layer." in k for k in keys):
        return "bert-base-uncased"
    raise ValueError("backbone을 파라미터 이름으로 추론할 수 없음 (알려진 두 backbone 키 패턴에 안 맞음)")


def load_checkpoint_entities(ckpt_path: Path):
    """반환: [(entity_name, state_dict, backbone, bottleneck_dim, logit_dim), ...]
    ("student"는 항상, "teacher"는 teacher_state_dict가 있을 때만)."""
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj:
        student_sd = obj["state_dict"]
        model_cfg = obj.get("model_cfg")
        teacher_sd = obj.get("teacher_state_dict")
    else:
        student_sd = obj
        model_cfg = None
        teacher_sd = None

    if model_cfg is not None:
        backbone = model_cfg["backbone"]
        bdim = model_cfg["head"]["bottleneck_dim"]
        ldim = model_cfg["head"]["logit_dim"]
    else:
        backbone = _infer_backbone_from_state_dict(student_sd)
        bdim, ldim = _LEGACY_BOTTLENECK_DIM, _LEGACY_LOGIT_DIM

    entities = [("student", student_sd, backbone, bdim, ldim)]
    if teacher_sd is not None:
        entities.append(("teacher", teacher_sd, backbone, bdim, ldim))
    return entities


def score_entity_stsb(model, tokenizer, device, batch_size=256) -> list[dict]:
    """(pooling, postprocess) 전 조합에 대해 STS-B dev Spearman 계산. embedding은
    pooling별로 한 번만 계산하고 postprocess는 그 위에서 재사용(전 조합 재순전파 방지)."""
    s1, s2, scores = _load_stsb_dev()
    rows = []
    for pooling in POOLINGS:
        e1 = embed_sentences(model, tokenizer, s1, device, batch_size=batch_size, pooling=pooling)
        e2 = embed_sentences(model, tokenizer, s2, device, batch_size=batch_size, pooling=pooling)
        pool = torch.cat([e1, e2], dim=0)
        for pp in POSTPROCESSES:
            pe1 = postprocess_embeddings(e1, fit_embeds=pool, method=pp)
            pe2 = postprocess_embeddings(e2, fit_embeds=pool, method=pp)
            cos = F.cosine_similarity(pe1, pe2, dim=-1).numpy()
            spearman = float(spearmanr(cos, scores)[0])
            rows.append({"pooling": pooling, "postprocess": pp, "sts_b_dev": spearman})
    return rows


def write_grid_md(all_rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# 전 체크포인트 소급 재채점 (STS-B dev Spearman)", ""]

    # --- 1) run x entity x pooling 별 4-postprocess 표 ---
    lines.append("## 조합별 상세 (run x entity x pooling)")
    cols = ["run", "backbone", "entity", "pooling"] + POSTPROCESSES
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    keyed = {}
    for r in all_rows:
        k = (r["run"], r["backbone"], r["entity"], r["pooling"])
        keyed.setdefault(k, {})[r["postprocess"]] = r["sts_b_dev"]
    for (run, backbone, entity, pooling), pp_vals in sorted(keyed.items()):
        cells = [run, backbone, entity, pooling] + [f"{pp_vals.get(pp, float('nan')):.4f}" for pp in POSTPROCESSES]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # --- 2) backbone x postprocess 평균 개선(student만, "none" 대비 delta) ---
    lines.append("## backbone별 후처리 평균 효과 (student만, vs postprocess=none, Δspearman)")
    lines.append("| backbone | " + " | ".join(POSTPROCESSES[1:]) + " |")
    lines.append("|" + "---|" * (1 + len(POSTPROCESSES) - 1))
    by_backbone = {}
    for (run, backbone, entity, pooling), pp_vals in keyed.items():
        if entity != "student" or "none" not in pp_vals:
            continue
        by_backbone.setdefault(backbone, {pp: [] for pp in POSTPROCESSES[1:]})
        for pp in POSTPROCESSES[1:]:
            if pp in pp_vals:
                by_backbone[backbone][pp].append(pp_vals[pp] - pp_vals["none"])
    for backbone, pp_deltas in sorted(by_backbone.items()):
        cells = [backbone]
        for pp in POSTPROCESSES[1:]:
            vals = pp_deltas[pp]
            cells.append(f"{sum(vals)/len(vals):+.4f}" if vals else "n/a")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # --- 3) student vs teacher (같은 run/pooling/postprocess에서 존재하는 쌍만) ---
    lines.append("## student vs teacher (teacher_state_dict 있는 run만, pooling=last/postprocess=none 기준)")
    lines.append("| run | student | teacher | teacher - student |")
    lines.append("|---|---|---|---|")
    runs_with_teacher = sorted({r["run"] for r in all_rows if r["entity"] == "teacher"})
    for run in runs_with_teacher:
        backbones = [b for (rn, b, e, p) in keyed if rn == run]
        if not backbones:
            continue
        backbone = backbones[0]
        s_val = keyed.get((run, backbone, "student", "last"), {}).get("none")
        t_val = keyed.get((run, backbone, "teacher", "last"), {}).get("none")
        if s_val is None or t_val is None:
            continue
        lines.append(f"| {run} | {s_val:.4f} | {t_val:.4f} | {t_val - s_val:+.4f} |")
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pick_best_combo(all_rows: list[dict]) -> tuple[str, str]:
    """(pooling, postprocess) 조합 중 student 평균 spearman이 가장 높은 조합 선택."""
    sums, counts = {}, {}
    for r in all_rows:
        if r["entity"] != "student":
            continue
        k = (r["pooling"], r["postprocess"])
        sums[k] = sums.get(k, 0.0) + r["sts_b_dev"]
        counts[k] = counts.get(k, 0) + 1
    means = {k: sums[k] / counts[k] for k in sums}
    best = max(means, key=means.get)
    return best, means[best]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", default="last", help="checkpoints/<run>/<which>.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument(
        "--best-combo-runs", default="",
        help="best (pooling,postprocess) 조합으로 7-task까지 추가 채점할 대표 run 이름들(콤마 구분). "
             "빈 문자열이면 7-task 추가 채점을 건너뜀 (STS-B dev 그리드만).",
    )
    parser.add_argument(
        "--skip-grid", action="store_true",
        help="61개 체크포인트 전체 그리드를 다시 돌리지 않고 --best-pooling/--best-postprocess로 "
             "지정한 조합으로 --best-combo-runs의 7-task만 채점(이미 grid를 돌려서 best combo를 "
             "아는 경우 재실행 낭비 방지용).",
    )
    parser.add_argument("--best-pooling", choices=POOLINGS, default="first_last")
    parser.add_argument("--best-postprocess", choices=POSTPROCESSES, default="center_pc2")
    args = parser.parse_args()

    tokenizer_cache: dict[str, AutoTokenizer] = {}

    if not args.skip_grid:
        all_rows = []
        run_dirs = sorted(p for p in CKPT_ROOT.iterdir() if p.is_dir())
        for i, run_dir in enumerate(run_dirs):
            ckpt_path = run_dir / f"{args.which}.pt"
            if not ckpt_path.exists():
                continue
            try:
                entities = load_checkpoint_entities(ckpt_path)
            except Exception as e:
                print(f"[rescore] {run_dir.name}: 로드 실패, 스킵 ({e})")
                continue

            for entity_name, state_dict, backbone, bdim, ldim in entities:
                if backbone not in tokenizer_cache:
                    tokenizer_cache[backbone] = AutoTokenizer.from_pretrained(backbone)
                tokenizer = tokenizer_cache[backbone]

                model = DinoTextModel(backbone, bdim, ldim).to(args.device)
                model.load_state_dict(state_dict)
                model.eval()

                rows = score_entity_stsb(model, tokenizer, args.device)
                for r in rows:
                    r.update(run=run_dir.name, backbone=backbone, entity=entity_name)
                all_rows.extend(rows)

                best_here = max(r["sts_b_dev"] for r in rows)
                print(f"[rescore] ({i+1}/{len(run_dirs)}) {run_dir.name} [{entity_name}, {backbone}]: "
                      f"best sts_b_dev over grid={best_here:.4f}")

                del model
                torch.cuda.empty_cache()

        write_grid_md(all_rows, Path(args.out))
        print(f"[rescore] wrote {args.out} ({len(all_rows)} rows)")

        best_pooling, best_pp = pick_best_combo(all_rows)[0]
        print(f"[rescore] best (pooling, postprocess) by mean student STS-B dev: "
              f"({best_pooling}, {best_pp})")
    else:
        best_pooling, best_pp = args.best_pooling, args.best_postprocess
        print(f"[rescore] --skip-grid: 그리드 재계산 없이 지정된 조합 사용 -> "
              f"pooling={best_pooling}, postprocess={best_pp}")

    if args.best_combo_runs:
        rep_runs = [r.strip() for r in args.best_combo_runs.split(",") if r.strip()]
        extra_lines = [
            "", f"## 7-task 평균 (best combo: pooling={best_pooling}, postprocess={best_pp})", "",
            "| run | entity | avg_7task (raw, postprocess=none) | avg_7task (best combo) | 개선 |",
            "|---|---|---|---|---|",
        ]
        for run_name in rep_runs:
            ckpt_path = CKPT_ROOT / run_name / f"{args.which}.pt"
            if not ckpt_path.exists():
                print(f"[rescore] 7-task: {run_name} 체크포인트 없음, 스킵")
                continue
            entities = load_checkpoint_entities(ckpt_path)
            for entity_name, state_dict, backbone, bdim, ldim in entities:
                tokenizer = tokenizer_cache.setdefault(backbone, AutoTokenizer.from_pretrained(backbone))
                model = DinoTextModel(backbone, bdim, ldim).to(args.device)
                model.load_state_dict(state_dict)
                model.eval()

                raw_scores = sts_suite_spearman(model, tokenizer, args.device)

                # best combo(포스트프로세스 포함) 7-task: task별로 e1/e2 뽑아 postprocess 적용.
                from src.evaluate import STS_SUITE
                from datasets import load_dataset
                task_scores = {}
                for task_name, hf_path in STS_SUITE.items():
                    ds = load_dataset(hf_path, split="test")
                    s1, s2, scores = list(ds["sentence1"]), list(ds["sentence2"]), list(ds["score"])
                    e1 = embed_sentences(model, tokenizer, s1, args.device, pooling=best_pooling)
                    e2 = embed_sentences(model, tokenizer, s2, args.device, pooling=best_pooling)
                    pool = torch.cat([e1, e2], dim=0)
                    pe1 = postprocess_embeddings(e1, fit_embeds=pool, method=best_pp)
                    pe2 = postprocess_embeddings(e2, fit_embeds=pool, method=best_pp)
                    cos = F.cosine_similarity(pe1, pe2, dim=-1).numpy()
                    task_scores[task_name] = float(spearmanr(cos, scores)[0])
                best_avg = sum(task_scores.values()) / len(task_scores)

                extra_lines.append(
                    f"| {run_name} | {entity_name} | {raw_scores['avg_7task']:.4f} | {best_avg:.4f} "
                    f"| {best_avg - raw_scores['avg_7task']:+.4f} |"
                )
                print(f"[rescore] 7-task {run_name}[{entity_name}]: raw={raw_scores['avg_7task']:.4f} "
                      f"best_combo={best_avg:.4f}")
                del model
                torch.cuda.empty_cache()

        with open(args.out, "a", encoding="utf-8") as f:
            f.write("\n".join(extra_lines) + "\n")
        print(f"[rescore] appended 7-task section to {args.out}")


if __name__ == "__main__":
    main()
