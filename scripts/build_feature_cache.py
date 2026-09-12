#!/usr/bin/env python3
"""Build the frozen vision feature cache for one MELD split from already
preprocessed clips (run scripts/preprocess_meld.py first). Resumable: clips
with an existing .npz are skipped unless --overwrite.

Usage:
    uv run python scripts/build_feature_cache.py --split dev
"""
import argparse
import sys
import time

from meld_emotion.config import FEATURE_CACHE_DIR, PREPROCESSED_DIR, SPLITS
from meld_emotion.data.cache import build_split_cache
from meld_emotion.vision.encoders import FaceEmotionEncoder, SceneEncoder, default_device


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--device", default=default_device(), help="'mps' or 'cpu'")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    split_dir = PREPROCESSED_DIR / args.split
    if not split_dir.exists() or not any(p.is_dir() for p in split_dir.iterdir()):
        sys.exit(f"No preprocessed clips found under {split_dir}\n"
                  f"Run this first: uv run python scripts/preprocess_meld.py --split {args.split}")

    print(f"Loading encoders on {args.device}...")
    face_encoder = FaceEmotionEncoder(device=args.device)
    scene_encoder = SceneEncoder(device=args.device)

    started = time.perf_counter()
    counts = build_split_cache(args.split, face_encoder, scene_encoder, overwrite=args.overwrite)
    elapsed = time.perf_counter() - started
    print(f"Done in {elapsed / 60:.1f} min: {dict(counts)} -> {FEATURE_CACHE_DIR / args.split}")


if __name__ == "__main__":
    main()
