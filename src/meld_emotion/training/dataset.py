"""Dataset over the data plan's outputs: manifest.jsonl rows + one .npz per
clip. Yields token ids for the text (with dialogue context) and the
variable-size face/scene token sets the fusion model consumes."""
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from meld_emotion.data.labels import EMOTIONS, SENTIMENTS
from meld_emotion.data.manifest import read_manifest
from meld_emotion.training.config import TrainConfig
from meld_emotion.training.text import format_context

EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}
SENTIMENT_TO_IDX = {s: i for i, s in enumerate(SENTIMENTS)}


def remap_track_ids(track_ids: np.ndarray, max_slots: int) -> np.ndarray:
    """Raw per-clip track ids are unique but sparse (they reach 55 on dev).
    Remap by first appearance to 0..max_slots-1; later distinct ids share
    the overflow slot max_slots."""
    order: dict[int, int] = {}
    out = np.empty(len(track_ids), dtype=np.int64)
    for i, t in enumerate(track_ids.tolist()):
        if t not in order:
            order[t] = len(order)
        out[i] = min(order[t], max_slots)
    return out


def load_ok_rows(manifest_path: Path) -> list[dict]:
    return [r for r in read_manifest(manifest_path) if r["status"] == "ok"]


def infer_feature_dims(rows: list[dict], cache_dir: Path) -> tuple[int, int]:
    z = load_clip_features(Path(cache_dir) / rows[0]["feature_path"])
    return int(z["face_features"].shape[1]), int(z["scene_features"].shape[1])


FEATURE_KEYS = ("face_features", "face_frame_idx", "face_track_ids", "scene_features", "scene_frame_idx")


def load_clip_features(path: Path) -> dict[str, np.ndarray]:
    """The arrays training needs from one clip's .npz, fully materialised."""
    with np.load(path) as z:
        return {k: z[k] for k in FEATURE_KEYS}


class MeldFeatureDataset(Dataset):
    def __init__(self, rows: list[dict], cache_dir: Path, config: TrainConfig,
                 encode_fn: Callable[[str, str], dict[str, list[int]]],
                 preload: bool = True, load_workers: int = 16):
        self.rows = rows
        self.cache_dir = Path(cache_dir)
        self.config = config
        self.encoded = [encode_fn(format_context(r["context_prev"], config.context_k), r["text"])
                        for r in rows]
        self.lengths = [len(e["input_ids"]) for e in self.encoded]
        # Preload every clip's arrays once (a split is ~1 GB in RAM). One np.load
        # per __getitem__ is ~1 ms on a local SSD but a network round-trip on a
        # Modal Volume: with num_workers=0 that left the GPU idle for nearly the
        # whole epoch (observed 0% utilisation). Threads overlap the reads, so
        # the one-time load is seconds rather than minutes.
        self.features: list[dict[str, np.ndarray]] | None = None
        if preload:
            paths = [self.cache_dir / r["feature_path"] for r in rows]
            with ThreadPoolExecutor(max_workers=load_workers) as pool:
                self.features = list(pool.map(load_clip_features, paths))

    def __len__(self) -> int:
        return len(self.rows)

    def _clip_features(self, i: int) -> dict[str, np.ndarray]:
        if self.features is not None:
            return self.features[i]
        return load_clip_features(self.cache_dir / self.rows[i]["feature_path"])

    def __getitem__(self, i: int) -> dict:
        row, cfg = self.rows[i], self.config
        z = self._clip_features(i)
        face_feat, face_frame, face_track = z["face_features"], z["face_frame_idx"], z["face_track_ids"]
        scene_feat, scene_frame = z["scene_features"], z["scene_frame_idx"]

        vision_masked = (cfg.mask_vision_over_seconds is not None
                         and row["duration_s"] > cfg.mask_vision_over_seconds)
        if not cfg.use_faces or vision_masked:
            face_feat, face_frame, face_track = face_feat[:0], face_frame[:0], face_track[:0]
        if not cfg.use_scene or vision_masked:
            scene_feat, scene_frame = scene_feat[:0], scene_frame[:0]

        enc = self.encoded[i]
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "face_feat": torch.from_numpy(np.ascontiguousarray(face_feat, dtype=np.float32)),
            "face_frame": torch.from_numpy(np.minimum(face_frame, cfg.max_frames - 1).astype(np.int64)),
            "face_track": torch.from_numpy(remap_track_ids(face_track, cfg.max_track_slots)),
            "scene_feat": torch.from_numpy(np.ascontiguousarray(scene_feat, dtype=np.float32)),
            "scene_frame": torch.from_numpy(np.minimum(scene_frame, cfg.max_frames - 1).astype(np.int64)),
            "emotion": torch.tensor(EMOTION_TO_IDX[row["emotion"]], dtype=torch.long),
            "sentiment": torch.tensor(SENTIMENT_TO_IDX[row["sentiment"]], dtype=torch.long),
            "clip": f"dia{row['dialogue_id']}_utt{row['utterance_id']}",
        }


class BucketBatchSampler(Sampler[list[int]]):
    """Batches of similar token length. RoBERTa's cost is linear in padded
    tokens; on dev with k=4, random batches pad to ~114 tokens and
    length-sorted buckets to ~59. Shuffles inside mega-chunks of
    `chunk_batches` batches and shuffles batch order, fresh each epoch."""

    def __init__(self, lengths: list[int], batch_size: int, shuffle: bool = True,
                 seed: int = 0, chunk_batches: int = 50):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.chunk = batch_size * chunk_batches
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self) -> list[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.lengths)) if self.shuffle else np.arange(len(self.lengths))
        batches = []
        for start in range(0, len(order), self.chunk):
            chunk = order[start:start + self.chunk]
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            batches += [chunk[i:i + self.batch_size].tolist() for i in range(0, len(chunk), self.batch_size)]
        if self.shuffle:
            batches = [batches[i] for i in rng.permutation(len(batches))]
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)


def _pad_stack(seqs: list[torch.Tensor], pad_value) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack variable-length (N, ...) tensors into (B, Nmax, ...) plus a bool
    validity mask (B, Nmax). Works for N=0 rows and for 1-D or 2-D items."""
    n_max = max((s.shape[0] for s in seqs), default=0)
    trailing = seqs[0].shape[1:]
    out = torch.full((len(seqs), n_max, *trailing), pad_value, dtype=seqs[0].dtype)
    mask = torch.zeros(len(seqs), n_max, dtype=torch.bool)
    for b, s in enumerate(seqs):
        out[b, :s.shape[0]] = s
        mask[b, :s.shape[0]] = True
    return out, mask


def collate(batch: list[dict], pad_id: int) -> dict:
    input_ids, _ = _pad_stack([b["input_ids"] for b in batch], pad_id)
    attention_mask, _ = _pad_stack([b["attention_mask"] for b in batch], 0)
    face_feat, face_mask = _pad_stack([b["face_feat"] for b in batch], 0.0)
    face_frame, _ = _pad_stack([b["face_frame"] for b in batch], 0)
    face_track, _ = _pad_stack([b["face_track"] for b in batch], 0)
    scene_feat, scene_mask = _pad_stack([b["scene_feat"] for b in batch], 0.0)
    scene_frame, _ = _pad_stack([b["scene_frame"] for b in batch], 0)
    return {
        "input_ids": input_ids, "attention_mask": attention_mask,
        "face_feat": face_feat, "face_mask": face_mask, "face_frame": face_frame, "face_track": face_track,
        "scene_feat": scene_feat, "scene_mask": scene_mask, "scene_frame": scene_frame,
        "emotion": torch.stack([b["emotion"] for b in batch]),
        "sentiment": torch.stack([b["sentiment"] for b in batch]),
        "clips": [b["clip"] for b in batch],
    }
