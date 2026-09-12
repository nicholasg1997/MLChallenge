#!/usr/bin/env python3
"""Print the Stage 1 ablation table (markdown) from results/, including the
zero-training vision baseline if scripts/vision_baseline.py has been run.

Usage:
    uv run python scripts/ablation_table.py
"""
import argparse
import json
from pathlib import Path

from meld_emotion.config import REPO_ROOT
from meld_emotion.training.results import ablation_table, collect_results, summarise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--out", type=Path, default=None, help="also write the table to this .md file")
    args = parser.parse_args()

    baseline_path = args.results / "vision_baseline" / "results.json"
    baseline = json.load(open(baseline_path)) if baseline_path.exists() else None
    table = ablation_table(summarise(collect_results(args.results)), baseline)
    print(table)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table + "\n")
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
