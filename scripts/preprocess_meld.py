#!/usr/bin/env python3
"""Preprocess every clip in one MELD split: decode at ~3fps, detect + track
faces, crop and letterbox, write crops + metadata.json per clip under
data/meld/preprocessed/<split>/. Resumable: clips with an existing
metadata.json are skipped unless --overwrite.

Usage:
    uv run python scripts/preprocess_meld.py --split dev --workers 8
"""
import argparse
import os
import time

from meld_emotion.config import LABELS_DIR, PREPROCESSED_DIR, SPLITS, split_video_dir
from meld_emotion.data.labels import load_split
from meld_emotion.data.preprocess import preprocess_split
from meld_emotion.data.video_index import build_video_index


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--limit", type=int, default=None, help="process only the first N rows")
    parser.add_argument("--overwrite", action="store_true", help="redo clips that already have metadata")
    args = parser.parse_args()

    utterances = load_split(args.split, labels_dir=LABELS_DIR)
    if args.limit:
        utterances = utterances[:args.limit]
    index = build_video_index(split_video_dir(args.split))
    print(f"{args.split}: {len(utterances)} utterances, {len(index)} video files, "
          f"{args.workers} workers -> {PREPROCESSED_DIR / args.split}")

    started = time.perf_counter()
    counts = preprocess_split(args.split, utterances, index, out_dir=PREPROCESSED_DIR,
                              workers=args.workers, overwrite=args.overwrite)
    elapsed = time.perf_counter() - started
    print(f"Done in {elapsed / 60:.1f} min: {dict(counts)} (total={len(utterances)})")


if __name__ == "__main__":
    main()
