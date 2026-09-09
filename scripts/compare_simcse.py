"""SimCSE 공식 체크포인트를 우리 지표 코드로 직접 재측정해 우리 run과 비교한다.

지금까지 results/analysis_v2.md는 SimCSE의 alignment/uniformity를 *논문 인용값*
(≈0.18 / ≈−2.6, "재계산 아님"이라고 명시)으로 써왔다. 이 스크립트는 같은 코드·같은
STS-B dev 쌍·같은 난수 시드로 직접 재서 그 자리를 대체 가능한 수치로 만든다.

주의 두 가지:
1) pooling - SimCSE 비지도판의 공식 평가는 CLS(cls_before_pooler)다. 우리 파이프라인은
   mean("last")이 기본이라 둘 다 재서 함께 보고한다. MLP pooler는 backbone만 싣기 때문에
   자연히 제외되어 cls_before_pooler와 일치한다.
2) SimCSE 체크포인트 로드 - HF repo에 safetensors가 없고 .bin만 있는데, transformers 5.x는
   torch<2.6에서 .bin 로드를 거부한다(CVE-2025-32434). 캐시에 자동 변환된 safetensors가
   있으므로 그것을 직접 읽는다 - torch.load 경로를 아예 타지 않아 해당 게이트와 무관하다.

실행:
    uv run python scripts/compare_simcse.py --device cuda
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer, BertModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluate import (  # noqa: E402
    effective_rank_metrics, embed_sentences, postprocess_embeddings, sts_b_dev_metrics,
)
from src.model import DinoTextModel  # noqa: E402
from src.train import load_config, load_sentences  # noqa: E402

HF_HUB = Path("/src/gs25058/scratch/.hf_home/hub")
# 우리 모델은 mean pooling으로 학습했으므로 mean 계열만 잰다. "cls"는 SimCSE 비지도판의
# 공식 평가 방식이라 SimCSE에만 추가로 적용한다 - 우리 모델의 CLS 토큰은 학습 신호를 받은
# 적이 없어 그 수치는 해석 가능한 정보가 아니다.
OUR_POOLINGS = ["last", "first_last"]
SIMCSE_POOLINGS = ["cls", "last", "first_last"]
POSTPROCESS = ["none", "center", "center_pc1", "center_pc2"]


def load_simcse_backbone(name: str) -> BertModel:
    """캐시된 safetensors에서 SimCSE backbone을 싣는다(스냅샷이 config용/가중치용으로 갈려 있다)."""
    root = HF_HUB / f"models--princeton-nlp--{name}"
    cfg_dir = Path(glob.glob(f"{root}/snapshots/*/config.json")[0]).parent
    st_path = glob.glob(f"{root}/snapshots/*/model.safetensors")
    if not st_path:
        raise FileNotFoundError(
            f"{name}: 캐시에 model.safetensors가 없다. 네트워크가 있는 노드에서 한 번 "
            f"from_pretrained를 시도하면 transformers가 자동 변환해 캐시에 남긴다."
        )
    model = BertModel(AutoConfig.from_pretrained(cfg_dir))
    sd = load_file(st_path[0])
    sd.pop("embeddings.position_ids", None)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{name} 가중치 불일치: missing={missing[:5]} unexpected={unexpected[:5]}")
    return model.eval()


def wrap_as_dino(backbone: BertModel, device: str) -> DinoTextModel:
    """우리 지표 함수는 DinoTextModel 인터페이스를 요구한다. head는 쓰이지 않으므로
    랜덤 초기화 그대로 둔다(analyze_checkpoints.py의 pretrained 기준선과 동일한 방식)."""
    m = DinoTextModel.__new__(DinoTextModel)
    torch.nn.Module.__init__(m)
    m.backbone = backbone
    m.head = torch.nn.Linear(backbone.config.hidden_size, 8).to(device)
    return m.to(device).eval()


def load_our_run(run_name: str, device: str) -> DinoTextModel | None:
    ckpt_path = ROOT / "checkpoints" / run_name / "last.pt"
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = ckpt["model_cfg"]
    m = DinoTextModel(cfg["backbone"], cfg["head"]["bottleneck_dim"], cfg["head"]["logit_dim"]).to(device)
    m.load_state_dict(ckpt["state_dict"])
    return m.eval()


def measure(model, tokenizer, device, rank_sentences, pooling: str) -> dict:
    """학습 중 평가와 '같은 함수'로 잰다 - 지표 정의가 갈리지 않게 하기 위함."""
    spearman, alignment, uniformity = sts_b_dev_metrics(model, tokenizer, device, pooling=pooling)
    eff_rank, max_sv_ratio = effective_rank_metrics(
        model, tokenizer, rank_sentences, device, pooling=pooling
    )
    return {
        "sts_b_dev": spearman, "alignment": alignment, "uniformity": uniformity,
        "eff_rank": eff_rank, "max_sv_ratio": max_sv_ratio,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--our-runs", default="r8_bert_coviso_sched_optuna_t50_s42,"
                                          "r7_bert_coviso_optuna_t35_s42,"
                                          "r6_bert_koleo_lrsplit_optuna_best_t27_s42")
    ap.add_argument("--out", default="results/analysis/simcse_compare")
    args = ap.parse_args()

    base_cfg = load_config(ROOT / "configs" / "base_bert.yaml")
    tokenizer = AutoTokenizer.from_pretrained(base_cfg["model"]["backbone"])
    rank_sentences = load_sentences(ROOT / base_cfg["data"]["rank_eval_path"])

    targets: list[tuple[str, DinoTextModel, list[str]]] = []
    for name in ["unsup-simcse-bert-base-uncased", "sup-simcse-bert-base-uncased"]:
        targets.append((f"simcse_{name.split('-simcse')[0]}",
                        wrap_as_dino(load_simcse_backbone(name), args.device), SIMCSE_POOLINGS))
    # 학습 전 BERT (하한 기준선) - 우리와 같은 mean 계열로만 본다
    targets.append(("pretrained_bert",
                    DinoTextModel(base_cfg["model"]["backbone"], 256, 8192).to(args.device).eval(), OUR_POOLINGS))
    for run in [r for r in args.our_runs.split(",") if r]:
        m = load_our_run(run, args.device)
        if m is None:
            print(f"  [skip] checkpoint 없음: {run}")
            continue
        targets.append((run, m, OUR_POOLINGS))

    rows = []
    for label, model, poolings in targets:
        for pooling in poolings:
            r = measure(model, tokenizer, args.device, rank_sentences, pooling)
            r.update(model=label, pooling=pooling)
            rows.append(r)
            print(f"  {label:<48} {pooling:<10} sts={r['sts_b_dev']:.4f} "
                  f"eff_rank={r['eff_rank']:.1f} align={r['alignment']:.4f} unif={r['uniformity']:.4f}")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n  wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
