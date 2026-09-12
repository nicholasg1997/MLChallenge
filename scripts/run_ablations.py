#!/usr/bin/env python3
"""Run several Stage 1 ablations × seeds back to back (each ~30 min on MPS).
Skips any (ablation, seed) whose results.json already exists, so it can be
re-run after an interruption.

Usage:
    uv run python scripts/run_ablations.py --seeds 0            # all presets, one seed
    uv run python scripts/run_ablations.py --ablations fusion text_only_k4 --seeds 0 1 2 --eval-test
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import time

from meld_emotion.config import FEATURE_CACHE_DIR, REPO_ROOT
from meld_emotion.training.config import ABLATIONS, config_for
from meld_emotion.training.train import train


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ablations", nargs="+", default=sorted(ABLATIONS), choices=sorted(ABLATIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    started = time.perf_counter()
    for name in args.ablations:
        for seed in args.seeds:
            out_dir = REPO_ROOT / "results" / name / f"seed{seed}"
            if (out_dir / "results.json").exists():
                print(f"[skip] {name} seed{seed}: results.json exists")
                continue
            overrides = {"seed": seed, **({"epochs": args.epochs} if args.epochs else {})}
            train(config_for(name, **overrides), FEATURE_CACHE_DIR, out_dir,
                  test_split="test" if args.eval_test else None)
    print(f"All done in {(time.perf_counter() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
