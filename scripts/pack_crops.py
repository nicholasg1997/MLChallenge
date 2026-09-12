#!/usr/bin/env python3
"""Pack one split's face crops + metadata.json into a single tar for Modal
(270K small files are far slower to upload than three tarballs).

Usage:
    uv run python scripts/pack_crops.py --split train      # -> data/meld/crops_train.tar
"""
import argparse
import tarfile
import time

from meld_emotion.config import DATA_DIR, PREPROCESSED_DIR, SPLITS


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=SPLITS, required=True)
    args = parser.parse_args()
    out = DATA_DIR / f"crops_{args.split}.tar"
    started = time.perf_counter()
    n = 0
    with tarfile.open(out, "w") as tar:
        for clip_dir in sorted((PREPROCESSED_DIR / args.split).iterdir()):
            if not (clip_dir / "metadata.json").exists():
                continue
            for p in [clip_dir / "metadata.json"] + sorted(clip_dir.glob("frame*_face*.jpg")):
                tar.add(p, arcname=f"{args.split}/{clip_dir.name}/{p.name}")
                n += 1
    print(f"{n} files -> {out} ({out.stat().st_size / 2**30:.2f} GB) in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
