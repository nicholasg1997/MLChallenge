#!/usr/bin/env python3
"""Train one Stage 1 configuration on the cached MELD features and evaluate it.

Usage:
    uv run python scripts/train_meld.py --ablation fusion --seed 0
    uv run python scripts/train_meld.py --ablation text_only_k4 --seed 0 --eval-test
    uv run python scripts/train_meld.py --ablation fusion --train-split dev --epochs 1   # smoke test

Writes results/<ablation>/seed<N>/{best.pt,log.jsonl,results.json}.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # before torch is imported (design doc §6)

import argparse
from pathlib import Path

from meld_emotion.config import FEATURE_CACHE_DIR, REPO_ROOT
from meld_emotion.training.config import ABLATIONS, config_for
from meld_emotion.training.train import train


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ablation", choices=sorted(ABLATIONS), default="fusion")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--class-weight-alpha", type=float, default=None)
    parser.add_argument("--sentiment-lambda", type=float, default=None)
    parser.add_argument("--mask-vision-over-seconds", type=float, default=None)
    parser.add_argument("--device", default=None, help="mps, cpu, or auto")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--dev-split", default="dev")
    parser.add_argument("--eval-test", action="store_true", help="evaluate the best checkpoint on test (once!)")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    overrides = {k: v for k, v in {
        "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size, "n_layers": args.n_layers,
        "class_weight_alpha": args.class_weight_alpha, "sentiment_lambda": args.sentiment_lambda,
        "mask_vision_over_seconds": args.mask_vision_over_seconds, "device": args.device,
    }.items() if v is not None}
    config = config_for(args.ablation, **overrides)
    out_dir = args.out or REPO_ROOT / "results" / config.name / f"seed{config.seed}"
    train(config, FEATURE_CACHE_DIR, out_dir, train_split=args.train_split, dev_split=args.dev_split,
          test_split="test" if args.eval_test else None, max_train_rows=args.max_train_rows)


if __name__ == "__main__":
    main()
