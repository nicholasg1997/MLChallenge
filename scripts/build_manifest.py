#!/usr/bin/env python3
"""Write data/meld/features/<split>/manifest.jsonl joining every labeled
utterance to its cached features, and print pipeline statistics.

Usage:
    uv run python scripts/build_manifest.py --split dev
"""
import argparse
import json

from meld_emotion.config import FEATURE_CACHE_DIR, LABELS_DIR, PREPROCESSED_DIR, SPLITS, split_video_dir
from meld_emotion.data.labels import load_split
from meld_emotion.data.manifest import build_manifest, manifest_stats, write_manifest
from meld_emotion.data.video_index import build_video_index


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=SPLITS, required=True)
    args = parser.parse_args()

    utterances = load_split(args.split, labels_dir=LABELS_DIR)
    index = build_video_index(split_video_dir(args.split))
    rows = build_manifest(args.split, utterances, index,
                          preprocessed_dir=PREPROCESSED_DIR, cache_dir=FEATURE_CACHE_DIR)
    out_path = FEATURE_CACHE_DIR / args.split / "manifest.jsonl"
    write_manifest(rows, out_path)
    print(f"Wrote {len(rows)} rows -> {out_path}")
    print(json.dumps(manifest_stats(rows), indent=2))


if __name__ == "__main__":
    main()
