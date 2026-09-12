#!/usr/bin/env python3
"""Stage 2 locally (smoke tests; the real runs use scripts/modal_stage2.py).

Usage:
    uv run python scripts/train_stage2.py --base fusion --train-split dev --epochs 1 --max-train-rows 64 --out results/smoke_stage2
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
from pathlib import Path

from meld_emotion.config import FEATURE_CACHE_DIR, PREPROCESSED_DIR, REPO_ROOT
from meld_emotion.training.config import stage2_config
from meld_emotion.training.train_stage2 import train_stage2


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="fusion", help="Stage 1 preset to start from (must use faces)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--face-trainable-layers", type=int, default=None)
    parser.add_argument("--lr-face", type=float, default=None)
    parser.add_argument("--loader-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--init-from", type=Path, default=None, help="Stage 1 best.pt (default: results/<base>/seed0/best.pt)")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--dev-split", default="dev")
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    overrides = {k: v for k, v in {"seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
                                   "face_trainable_layers": args.face_trainable_layers, "lr_face": args.lr_face,
                                   "loader_workers": args.loader_workers, "device": args.device}.items() if v is not None}
    config = stage2_config(args.base, **overrides)
    init_from = args.init_from or REPO_ROOT / config.init_from
    out_dir = args.out or REPO_ROOT / "results" / config.name / f"seed{config.seed}"
    train_stage2(config, FEATURE_CACHE_DIR, PREPROCESSED_DIR, out_dir, init_from=init_from,
                 train_split=args.train_split, dev_split=args.dev_split,
                 test_split="test" if args.eval_test else None, max_train_rows=args.max_train_rows)


if __name__ == "__main__":
    main()
