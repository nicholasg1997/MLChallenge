"""Runs the frozen vision encoders over preprocessed crops, one batch per
clip, and persists the results as one .npz per clip. Resumable."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from meld_emotion.config import FEATURE_CACHE_DIR, PREPROCESSED_DIR


def cache_path_for(cache_dir: Path, split: str, clip_name: str) -> Path:
    return Path(cache_dir) / split / f"{clip_name}.npz"


def build_clip_cache(clip_dir: Path, face_encoder, scene_encoder) -> dict | None:
    with open(clip_dir / "metadata.json") as f:
        meta = json.load(f)
    if meta["status"] != "ok":
        return None

    face_images, face_track_ids, face_frame_idx = [], [], []
    scene_images, scene_frame_idx, shot_cut = [], [], []
    for frame in meta["frames"]:
        for face in frame["faces"]:
            face_images.append(Image.open(clip_dir / face["path"]).convert("RGB"))
            face_track_ids.append(face["track_id"])
            face_frame_idx.append(frame["sample_index"])
        scene_images.append(Image.open(clip_dir / frame["scene_path"]).convert("RGB"))
        scene_frame_idx.append(frame["sample_index"])
        shot_cut.append(bool(frame["shot_cut"]))

    faces = face_encoder.encode_batch(face_images)   # (0, D) arrays when there are no faces
    return {
        "dialogue_id": meta["dialogue_id"],
        "utterance_id": meta["utterance_id"],
        "face_features": faces["features"],
        "face_probs": faces["probs"],
        "face_track_ids": np.array(face_track_ids, dtype=np.int64),
        "face_frame_idx": np.array(face_frame_idx, dtype=np.int64),
        "scene_features": scene_encoder.encode_batch(scene_images),
        "scene_frame_idx": np.array(scene_frame_idx, dtype=np.int64),
        "shot_cut": np.array(shot_cut, dtype=bool),
    }


def save_clip_cache(cache: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **cache)


def build_split_cache(split: str, face_encoder, scene_encoder,
                      preprocessed_dir: Path = PREPROCESSED_DIR, cache_dir: Path = FEATURE_CACHE_DIR,
                      overwrite: bool = False, progress_every: int = 500) -> Counter:
    counts: Counter = Counter()
    clip_dirs = sorted(p for p in (Path(preprocessed_dir) / split).iterdir()
                       if (p / "metadata.json").exists())
    for i, clip_dir in enumerate(clip_dirs, 1):
        out_path = cache_path_for(cache_dir, split, clip_dir.name)
        if (out_path.exists() and not overwrite
                and out_path.stat().st_mtime >= (clip_dir / "metadata.json").stat().st_mtime):
            counts["skipped_existing"] += 1
        else:
            cache = build_clip_cache(clip_dir, face_encoder, scene_encoder)
            if cache is None:
                counts["skipped_not_ok"] += 1
            else:
                save_clip_cache(cache, out_path)
                counts["cached"] += 1
        if progress_every and i % progress_every == 0:
            print(f"  {i}/{len(clip_dirs)} {dict(counts)}", flush=True)
    return counts
