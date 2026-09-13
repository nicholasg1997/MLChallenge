"""Shared fixtures: a synthetic feature cache + manifests in the exact layout
the data plan writes, small enough to train on in a unit test."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from meld_emotion.data.labels import EMOTIONS, SENTIMENTS

FACE_DIM, SCENE_DIM = 8, 4


def whitespace_encode(context: str, current: str) -> dict[str, list[int]]:
    """Stand-in for encode_text: word -> small id, context before current."""
    words = (context + " " + current).split() if context else current.split()
    ids = [0] + [(abs(hash(w)) % 60) + 2 for w in words] + [1]   # 0=<s>, 1=</s>, pad=1 unused
    return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class StubTextEncoder(nn.Module):
    """Same contract as build_text_encoder(): .hidden_size and .last_hidden_state."""
    def __init__(self, hidden_size=32, vocab=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(vocab, hidden_size)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.emb(input_ids))


def write_synthetic_split(root, split, n_clips, seed=0, n_frames=3, faces_per_frame=(0, 1, 2),
                          long_clip_every=0):
    """Writes <root>/<split>/manifest.jsonl and <root>/<split>/dia<D>_utt<U>.npz.
    Faces per frame cycle by utterance id (utt0 -> 0, utt1 -> 1, utt2 -> 2, ...)
    so every dialogue has a zero-face clip."""
    rng = np.random.default_rng(seed)
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_clips):
        dia, utt = i // 4, i % 4
        n_faces_per = faces_per_frame[utt % len(faces_per_frame)]
        face_frame = np.repeat(np.arange(n_frames), n_faces_per)
        face_track = np.tile(np.arange(n_faces_per) * 7, n_frames)   # sparse raw ids, e.g. 0, 7
        n_faces = len(face_frame)
        npz_name = f"dia{dia}_utt{utt}.npz"
        np.savez(split_dir / npz_name,
                 face_features=rng.standard_normal((n_faces, FACE_DIM)).astype(np.float32),
                 face_probs=(lambda p: p / p.sum(1, keepdims=True))(rng.random((n_faces, 7)).astype(np.float32)),
                 face_track_ids=face_track.astype(np.int64),
                 face_frame_idx=face_frame.astype(np.int64),
                 scene_features=rng.standard_normal((n_frames, SCENE_DIM)).astype(np.float32),
                 scene_frame_idx=np.arange(n_frames, dtype=np.int64),
                 shot_cut=np.zeros(n_frames, dtype=bool),
                 dialogue_id=dia, utterance_id=utt)
        duration = 20.0 if (long_clip_every and i % long_clip_every == 0) else 2.5
        rows.append({"split": split, "dialogue_id": dia, "utterance_id": utt, "speaker": "Joey",
                     "text": f"utterance number {i} here", "context_prev": [f"earlier line {j}" for j in range(utt)],
                     "emotion": EMOTIONS[i % 7], "sentiment": SENTIMENTS[i % 3],
                     "status": "ok", "feature_path": f"{split}/{npz_name}", "duration_s": duration,
                     "n_frames": n_frames, "n_faces": n_faces, "n_shot_cuts": 0})
    # one non-ok row that must be filtered out
    rows.append({**rows[-1], "utterance_id": 99, "status": "decode_failed", "feature_path": None})
    with open(split_dir / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return split_dir / "manifest.jsonl"


@pytest.fixture
def synthetic_features(tmp_path):
    """Returns (root, train_manifest, dev_manifest) with 28 train / 14 dev clips."""
    root = tmp_path / "features"
    train = write_synthetic_split(root, "train", 28, seed=0, long_clip_every=7)
    dev = write_synthetic_split(root, "dev", 14, seed=1)
    return root, train, dev


import torch as _torch
from dataclasses import asdict as _asdict


def write_synthetic_checkpoint(path, d_model=32, n_heads=4, ff_dim=64, n_layers=1,
                               face_dim=8, scene_dim=4, use_text=True, stage2=False):
    """A checkpoint in exactly train.py's save format (train_stage2.py's when
    stage2=True: adds face_encoder_state / stage / base_checkpoint), small
    enough to build and load in a unit test."""
    from meld_emotion.training.config import TrainConfig
    from meld_emotion.training.model import FusionModel
    config = TrainConfig(d_model=d_model, n_heads=n_heads, ff_dim=ff_dim, n_layers=n_layers,
                         use_text=use_text, use_scene=not stage2, use_track_id=not stage2,
                         face_trainable_layers=4 if stage2 else 0)
    text_encoder = StubTextEncoder(d_model) if use_text else None
    model = FusionModel(config, text_encoder, face_dim=face_dim, scene_dim=scene_dim)
    ckpt = {"model_state": model.state_dict(), "config": _asdict(config), "epoch": 1,
            "dev_weighted_f1": 0.5, "face_dim": face_dim, "scene_dim": scene_dim}
    if stage2:
        ckpt.update({"face_encoder_state": {"dummy.weight": _torch.zeros(1)}, "stage": 2,
                     "base_checkpoint": "results/fusion_faces_only/seed0/best.pt"})
    _torch.save(ckpt, path)
    return config


@pytest.fixture
def synthetic_checkpoint(tmp_path):
    path = tmp_path / "best.pt"
    config = write_synthetic_checkpoint(path)
    return path, config


@pytest.fixture
def synthetic_stage2_checkpoint(tmp_path):
    path = tmp_path / "stage2.pt"
    config = write_synthetic_checkpoint(path, stage2=True)
    return path, config
