"""Stage 2 data: the Stage 1 item (text, scene features, frame/track ids)
plus each clip's face crops as uint8 tensors, read from the preprocessing
pass's JPEGs, capped per clip and augmented in training."""
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from meld_emotion.data.preprocess import clip_dir_for
from meld_emotion.training.config import TrainConfig
from meld_emotion.training.dataset import MeldFeatureDataset, collate

CROP_SIZE = 224


def train_transform():
    """Mild augmentation so the ViT specialises to expressions, not to the
    six actors' faces (the live demo must generalise past them)."""
    return transforms.Compose([
        transforms.RandomResizedCrop(CROP_SIZE, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.PILToTensor(),                       # uint8 CHW; normalised on the device
    ])


def eval_transform():
    return transforms.Compose([transforms.Resize((CROP_SIZE, CROP_SIZE)), transforms.PILToTensor()])


def subsample_faces(n: int, cap: int) -> np.ndarray:
    if n <= cap:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, cap).round().astype(np.int64))


def face_crop_paths(clip_dir: Path) -> list[Path]:
    """Face JPEG paths in metadata order — the same order as the cached
    face_frame_idx / face_track_ids rows (the cache was built by iterating
    frames, then faces)."""
    with open(clip_dir / "metadata.json") as f:
        meta = json.load(f)
    return [clip_dir / face["path"] for frame in meta["frames"] for face in frame["faces"]]


class MeldCropDataset(Dataset):
    def __init__(self, rows: list[dict], preprocessed_dir: Path, cache_dir: Path, config: TrainConfig,
                 encode_fn: Callable[[str, str], dict[str, list[int]]], train: bool):
        self.base = MeldFeatureDataset(rows, cache_dir, config, encode_fn)   # text, scene, frame/track ids
        self.config = config
        self.transform = train_transform() if (train and config.face_augment) else eval_transform()
        self.crop_paths = [face_crop_paths(clip_dir_for(preprocessed_dir, r["split"], r["dialogue_id"], r["utterance_id"]))
                           for r in rows]
        for i, (paths, feats) in enumerate(zip(self.crop_paths, self.base.features)):
            if len(paths) != len(feats["face_frame_idx"]):
                raise ValueError(f"{rows[i]['feature_path']}: {len(paths)} crops on disk vs "
                                 f"{len(feats['face_frame_idx'])} cached face rows")
        self.lengths = self.base.lengths

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> dict:
        from PIL import Image
        item = self.base[i]
        del item["face_feat"]
        keep = subsample_faces(len(self.crop_paths[i]), self.config.max_faces_per_clip)
        if not self.config.use_faces:
            keep = keep[:0]
        item["face_frame"] = item["face_frame"][keep]
        item["face_track"] = item["face_track"][keep]
        pixels = [self.transform(Image.open(self.crop_paths[i][j]).convert("RGB")) for j in keep]
        item["face_pixels"] = torch.stack(pixels) if pixels else torch.zeros((0, 3, CROP_SIZE, CROP_SIZE), dtype=torch.uint8)
        return item


def collate_crops(batch: list[dict], pad_id: int) -> dict:
    """Stage 1 collate (with a 1-wide face_feat placeholder that Stage2Model
    overwrites) plus every crop in the batch flattened to one tensor, with
    the (sample, slot) each crop belongs to."""
    placeholder = [{**b, "face_feat": torch.zeros((b["face_pixels"].shape[0], 1))} for b in batch]
    out = collate(placeholder, pad_id)
    out["face_pixels"] = torch.cat([b["face_pixels"] for b in batch]) if batch else torch.zeros((0, 3, CROP_SIZE, CROP_SIZE), dtype=torch.uint8)
    out["face_batch_idx"] = torch.cat([torch.full((b["face_pixels"].shape[0],), bi, dtype=torch.long) for bi, b in enumerate(batch)])
    out["face_slot"] = torch.cat([torch.arange(b["face_pixels"].shape[0], dtype=torch.long) for b in batch])
    return out
