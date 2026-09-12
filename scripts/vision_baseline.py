#!/usr/bin/env python3
"""Zero-training vision-only baseline on dev and/or test: average the frozen
face encoder's own probabilities over each clip's faces. Writes
results/vision_baseline/results.json.

Usage:
    uv run python scripts/vision_baseline.py --splits dev test
"""
import argparse
import json
from pathlib import Path

from meld_emotion.config import FEATURE_CACHE_DIR, REPO_ROOT
from meld_emotion.training.baselines import face_meld_label_order, vision_baseline_metrics
from meld_emotion.training.dataset import load_ok_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=["dev"], choices=["train", "dev", "test"])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "vision_baseline")
    args = parser.parse_args()

    face_labels = face_meld_label_order()
    results = {"name": "vision_only_zero_training", "face_label_order": list(face_labels)}
    for split in args.splits:
        rows = load_ok_rows(FEATURE_CACHE_DIR / split / "manifest.jsonl")
        results[split] = vision_baseline_metrics(rows, FEATURE_CACHE_DIR, face_labels)
        m = results[split]
        print(f"{split}: n={m['n']} weighted_f1={m['weighted_f1']:.4f} macro_f1={m['macro_f1']:.4f} "
              f"accuracy={m['accuracy']:.4f} no_face_clips={m['no_face_clips']}")
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"-> {args.out / 'results.json'}")


if __name__ == "__main__":
    main()
