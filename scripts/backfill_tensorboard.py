"""이미 완료된 results/logs/*/train.log(TensorBoard 로깅 추가 이전 run들 포함)를
results/tensorboard/<run_name>/ 이벤트 파일로 소급 변환한다. scripts/plot_logs.py의
파서를 재사용해 train.py의 실시간 TB 로깅과 동일한 태그 이름(train/*, eval/*)을 쓴다.

실행: uv run python scripts/backfill_tensorboard.py [--overwrite]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.utils.tensorboard import SummaryWriter

from scripts.plot_logs import discover_logs, parse_log

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "tensorboard"


def backfill_one(name: str, log_path: Path, overwrite: bool) -> bool:
    out_dir = OUT_DIR / name
    if out_dir.exists() and not overwrite:
        return False
    d = parse_log(log_path)
    if not d["train"] and not d["eval"]:
        return False

    writer = SummaryWriter(log_dir=str(out_dir))
    for step, kv in d["train"]:
        for k, v in kv.items():
            writer.add_scalar(f"train/{k}", v, step)
    for step, kv in d["eval"]:
        for k, v in kv.items():
            writer.add_scalar(f"eval/{k}", v, step)
    writer.close()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="이미 변환된 run도 다시 씀")
    args = parser.parse_args()

    logs = discover_logs()
    written = 0
    for name, path in logs.items():
        if backfill_one(name, path, args.overwrite):
            written += 1
            print(f"[backfill_tensorboard] {name}")
    print(f"[backfill_tensorboard] {written}/{len(logs)} run 변환 -> {OUT_DIR}")


if __name__ == "__main__":
    main()
