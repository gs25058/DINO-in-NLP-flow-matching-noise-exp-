"""study의 trial별 7-task 평균을 재서 기록하고, 남겨둘 이유가 없는 체크포인트를 지운다.

튜닝 목표는 STS-B dev 하나지만, 그 순위가 7-task 순위와 어긋나는 trial이 있으면 그 자체가 기록할
가치가 있는 결과다(이 프로젝트의 보고 지표는 7-task 평균이다). trial 체크포인트가 있어야 하므로
study를 `SAVE_CKPT=1`(run_study.sh) 또는 `--save-checkpoints`(tune.py)로 돌렸어야 한다.

trial당 약 860MB라 50 trial이면 43GB다. 대여 서버라 용량이 한정적이므로 --prune으로 정리한다:
**평가해서 점수를 JSON에 남긴 뒤** 아래 중 어디에도 해당하지 않는 체크포인트만 지운다
(점수 기록은 남으므로 지워도 결과는 보존된다).
  - 7-task 평균 >= --min-7task (기본 0.65)
  - STS-B dev 상위 --keep-top (기본 5)
  - 7-task 상위 --keep-top
  - STS-B dev가 study 최고값에서 --keep-margin (기본 0.005) 이내
아직 평가하지 않은 trial, 완료되지 않은 trial(진행 중), study 소속이 아닌 체크포인트는 절대 건드리지 않는다.
--prune-tb를 함께 주면 같은 기준으로 results/tensorboard/<study>_t<n>도 지운다(trial당 약 3.3MB).
용량보다 TensorBoard run 목록을 줄이는 효과가 크다. 지워도 점수·파라미터는 study DB와 JSON에 남는다.

결과는 증분 캐시에 쌓이므로 study가 도는 중에 여러 번 돌려도 이미 잰 trial은 건너뛴다.

실행:
    uv run python scripts/r18_trial_sts7.py --study tune_r18_bert_tunebase_r18_ibot_wide --prune
    uv run python scripts/r18_trial_sts7.py --study ... --prune --dry-run   # 지울 목록만 보기
    uv run python scripts/r18_trial_sts7.py --study ... --prune --prune-tb   # TensorBoard run도 함께
산출물: results/analysis/r18/<study>_sts7.json (캐시) / _sts7.md (STS-B 순위와 나란히 놓은 표)
"""
import argparse
import json
import sys
from pathlib import Path

import optuna
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.evaluate import sts_suite_spearman  # noqa: E402
from src.model import DinoTextModel  # noqa: E402
from src.train import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--config", default="configs/r18_bert_tunebase.yaml", help="모델 차원/backbone을 읽을 config")
    ap.add_argument("--top", type=int, default=0, help="STS-B 상위 N개만 평가 (0 = 체크포인트가 있는 전부)")
    ap.add_argument("--pooling", default="last", choices=["last", "first_last", "cls"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--db", default=None,
                    help="study SQLite 경로 (기본 <repo>/results/analysis/<study>.db). "
                         "실행 스냅샷 worktree에서 돌린 study는 그쪽 results/analysis에 쌓인다")
    ap.add_argument("--prune", action="store_true",
                    help="평가·기록이 끝난 trial 중 남길 이유가 없는 체크포인트를 지운다")
    ap.add_argument("--dry-run", action="store_true", help="--prune과 함께: 지울 목록만 출력하고 실제로 지우지 않는다")
    ap.add_argument("--prune-tb", action="store_true",
                    help="같은 기준으로 results/tensorboard/<study>_t<n>도 지운다 (run 목록 정리용, 약 3.3MB/trial)")
    ap.add_argument("--min-7task", type=float, default=0.65, help="이 값 이상이면 보존 (기본 0.65)")
    ap.add_argument("--keep-top", type=int, default=5, help="STS-B / 7-task 각각 상위 N개는 보존 (기본 5)")
    ap.add_argument("--keep-margin", type=float, default=0.005,
                    help="STS-B가 study 최고값에서 이 차이 이내면 보존 (기본 0.005)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / "analysis" / "r18"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"{args.study}_sts7.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    db = Path(args.db) if args.db else ROOT / "results" / "analysis" / f"{args.study}.db"
    if not db.exists():
        raise SystemExit(f"study DB가 없다: {db} (--db로 경로를 지정한다)")
    study = optuna.load_study(study_name=args.study, storage=f"sqlite:///{db}")
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    done.sort(key=lambda t: -t.value)
    if args.top:
        done = done[:args.top]

    cfg = load_config(ROOT / args.config)
    tok = AutoTokenizer.from_pretrained(cfg["model"]["backbone"])
    todo = [t for t in done if str(t.number) not in cache
            and (ROOT / "checkpoints" / f"{args.study}_t{t.number}" / "last.pt").exists()]
    print(f"[trial_sts7] 완료 trial {len(done)}개 / 캐시 {len(cache)}개 / 이번에 평가 {len(todo)}개")

    for t in todo:
        run = f"{args.study}_t{t.number}"
        model = DinoTextModel(cfg["model"]["backbone"], cfg["model"]["head"]["bottleneck_dim"],
                              cfg["model"]["head"]["logit_dim"]).to(args.device).eval()
        sd = torch.load(ROOT / "checkpoints" / run / "last.pt", map_location=args.device, weights_only=True)
        missing, _ = model.load_state_dict(sd["state_dict"], strict=False)
        if missing:
            raise RuntimeError(f"{run}: 모델 파라미터 누락 {missing[:5]}")
        scores = sts_suite_spearman(model, tok, args.device, pooling=args.pooling)
        cache[str(t.number)] = {"sts_b_dev": t.value, "params": t.params,
                                **{k: float(v) for k, v in scores.items()}}
        cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  t{t.number}: STS-B {t.value:.4f} / 7-task {scores['avg_7task'] * 100:.2f}")
        del model
        torch.cuda.empty_cache()

    rows = [(int(n), v) for n, v in cache.items()]
    if not rows:
        print("[trial_sts7] 평가된 trial이 없다 (체크포인트 없음 - SAVE_CKPT=1로 돌렸는지 확인)")
        return
    by_sts = {n: i + 1 for i, (n, _) in enumerate(sorted(rows, key=lambda r: -r[1]["sts_b_dev"]))}
    by_7 = {n: i + 1 for i, (n, _) in enumerate(sorted(rows, key=lambda r: -r[1]["avg_7task"]))}

    lines = [f"# {args.study} - trial별 STS-B dev vs 7-task ({args.pooling} pooling)", "",
             "목적함수는 STS-B dev 하나다. 7-task는 사후 기록용이며, 두 순위가 어긋나는 trial(|순위차| >= 5)에 *를 붙였다.",
             "비교 기준: 대조군 r18_bert_ctrl seed42 STS-B 0.7440 / 7-task 65.98.", "",
             "| trial | STS-B dev | 7-task | STS-B 순위 | 7-task 순위 | 주요 파라미터 |", "|---|---|---|---|---|---|"]
    for n, v in sorted(rows, key=lambda r: -r[1]["avg_7task"]):
        p = v["params"]
        gap = "*" if abs(by_sts[n] - by_7[n]) >= 5 else ""
        keys = ["loss.ibot_lambda", "augment.mask_ratio", "loss.cov_iso_lambda", "loss.teacher_temp",
                "train.lr", "train.head_lr", "train.momentum_end"]
        brief = " ".join(f"{k.split('.')[-1]}={p[k]:.4g}" for k in keys if k in p)
        for k in ("loss.ibot_head", "loss.ibot_use_token_center", "augment.token_latent_views"):
            if k in p:
                brief += f" {k.split('.')[-1]}={p[k]}"
        lines.append(f"| t{n}{gap} | {v['sts_b_dev']:.4f} | {v['avg_7task'] * 100:.2f} | "
                     f"{by_sts[n]} | {by_7[n]} | {brief} |")
    md = out_dir / f"{args.study}_sts7.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    best_sts = max(rows, key=lambda r: r[1]["sts_b_dev"])
    best_7 = max(rows, key=lambda r: r[1]["avg_7task"])
    print(f"[trial_sts7] STS-B 최고: t{best_sts[0]} {best_sts[1]['sts_b_dev']:.4f} "
          f"(7-task {best_sts[1]['avg_7task'] * 100:.2f})")
    print(f"[trial_sts7] 7-task 최고: t{best_7[0]} {best_7[1]['avg_7task'] * 100:.2f} "
          f"(STS-B {best_7[1]['sts_b_dev']:.4f})")
    print(f"[trial_sts7] 저장: {md}")

    if args.prune:
        prune(args, cache, rows, by_sts, by_7, best_sts[1]["sts_b_dev"])


def keep_reasons(v: dict, n: int, by_sts: dict, by_7: dict, best_sts: float, args) -> list[str]:
    """이 trial의 체크포인트를 남길 이유들. 비어 있으면 지워도 되는 trial이다."""
    reasons = []
    if v["avg_7task"] >= args.min_7task:
        reasons.append(f"7-task {v['avg_7task'] * 100:.2f} >= {args.min_7task * 100:.1f}")
    if by_sts[n] <= args.keep_top:
        reasons.append(f"STS-B 상위 {by_sts[n]}위")
    if by_7[n] <= args.keep_top:
        reasons.append(f"7-task 상위 {by_7[n]}위")
    if v["sts_b_dev"] >= best_sts - args.keep_margin:
        reasons.append(f"STS-B 최고값 -{args.keep_margin} 이내")
    return reasons


def prune(args, cache: dict, rows: list, by_sts: dict, by_7: dict, best_sts: float) -> None:
    """평가가 끝난 trial의 체크포인트만 대상으로, 남길 이유가 없는 것을 지운다.

    캐시(JSON)에 점수가 남아 있는 trial만 후보다 - 아직 평가하지 않았거나 진행 중인 trial은
    애초에 rows에 없으므로 건드리지 않는다. 경로도 이 study의 trial 이름으로만 만든다.
    """
    freed = 0
    kept, removed = [], []
    targets = ["checkpoints"] + (["tensorboard"] if getattr(args, "prune_tb", False) else [])
    for n, v in sorted(rows, key=lambda r: -r[1]["sts_b_dev"]):
        dirs = [ROOT / ("results/tensorboard" if t == "tensorboard" else t) / f"{args.study}_t{n}"
                for t in targets]
        dirs = [d for d in dirs if d.exists()]
        if not dirs:
            continue
        size = sum(f.stat().st_size for d in dirs for f in d.rglob("*") if f.is_file())
        reasons = keep_reasons(v, n, by_sts, by_7, best_sts, args)
        if reasons:
            kept.append((n, size, ", ".join(reasons)))
            continue
        removed.append((n, size, f"STS-B {v['sts_b_dev']:.4f} ({by_sts[n]}위) / 7-task {v['avg_7task'] * 100:.2f} ({by_7[n]}위)"))
        freed += size
        if not args.dry_run:
            for d in dirs:
                for f in sorted(d.rglob("*"), key=lambda p: -len(p.parts)):
                    f.unlink() if f.is_file() else f.rmdir()
                d.rmdir()

    head = "[prune] 지울 대상(dry-run)" if args.dry_run else "[prune] 삭제함"
    for n, size, why in removed:
        print(f"  {head}: t{n} ({size / 2**30:.2f} GiB) - {why}")
    for n, size, why in kept:
        print(f"  [prune] 보존: t{n} ({size / 2**30:.2f} GiB) - {why}")
    verb = "확보 예정" if args.dry_run else "확보"
    print(f"[prune] {len(removed)}개 {'삭제 대상' if args.dry_run else '삭제'}, {freed / 2**30:.2f} GiB {verb} "
          f"/ 보존 {len(kept)}개")


if __name__ == "__main__":
    main()
