import json

import numpy as np
import pytest
import torch
from PIL import Image

from conftest import whitespace_encode


def write_synthetic_crops(features_root, preprocessed_root, split):
    """For every ok row in <features_root>/<split>/manifest.jsonl, write the
    preprocessed clip dir the data plan would have written: metadata.json
    with one face entry per cached face row (same order) and a 224x224 JPEG
    per face whose red channel encodes its index in steps of 20 (JPEG
    compression perturbs values by a few units, so tests check ±4)."""
    from meld_emotion.data.manifest import read_manifest
    from meld_emotion.data.preprocess import clip_dir_for
    for row in read_manifest(features_root / split / "manifest.jsonl"):
        if row["status"] != "ok":
            continue
        z = np.load(features_root / row["feature_path"])
        clip_dir = clip_dir_for(preprocessed_root, split, row["dialogue_id"], row["utterance_id"])
        clip_dir.mkdir(parents=True, exist_ok=True)
        frames = {}
        for i, (frame_idx, track_id) in enumerate(zip(z["face_frame_idx"].tolist(), z["face_track_ids"].tolist())):
            name = f"frame{frame_idx}_face{track_id}.jpg"
            Image.new("RGB", (224, 224), ((i * 20) % 256, 40, 200)).save(clip_dir / name, quality=95)
            frames.setdefault(frame_idx, []).append({"track_id": track_id, "box": [0, 0, 1, 1], "score": 0.9, "path": name})
        meta = {"status": "ok", "frames": [{"sample_index": f, "faces": frames.get(f, []), "scene_path": f"frame{f}_scene.jpg",
                                             "shot_cut": False} for f in sorted(set(z["scene_frame_idx"].tolist()) | set(frames))]}
        with open(clip_dir / "metadata.json", "w") as fh:
            json.dump(meta, fh)


@pytest.fixture
def synthetic_crops(synthetic_features, tmp_path):
    root, train, dev = synthetic_features
    pre = tmp_path / "preprocessed"
    write_synthetic_crops(root, pre, "train")
    write_synthetic_crops(root, pre, "dev")
    return root, pre


def test_stage2_config_derives_from_a_face_preset():
    from meld_emotion.training.config import stage2_config
    c = stage2_config("fusion", seed=3)
    assert (c.name, c.face_trainable_layers, c.batch_size, c.seed) == ("stage2_fusion", 4, 16, 3)
    assert c.use_faces and c.use_text and c.init_from == "results/fusion/seed0/best.pt"
    with pytest.raises(KeyError):
        stage2_config("text_only_k4")


def test_transforms_produce_uint8_chw_224():
    from meld_emotion.training.crops import eval_transform, train_transform
    img = Image.new("RGB", (224, 224), (10, 20, 30))
    for t in (train_transform(), eval_transform()):
        x = t(img)
        assert x.dtype == torch.uint8 and x.shape == (3, 224, 224)
    assert torch.equal(eval_transform()(img)[:, 0, 0], torch.tensor([10, 20, 30], dtype=torch.uint8))


def test_subsample_faces_is_sorted_covers_the_range_and_is_identity_under_cap():
    from meld_emotion.training.crops import subsample_faces
    assert subsample_faces(5, cap=64).tolist() == [0, 1, 2, 3, 4]
    idx = subsample_faces(431, cap=64)
    assert len(idx) == 64 and idx[0] == 0 and idx[-1] == 430 and np.all(np.diff(idx) > 0)


def test_crop_dataset_items_carry_pixels_aligned_with_frame_and_track(synthetic_crops):
    from meld_emotion.training.config import stage2_config
    from meld_emotion.training.crops import MeldCropDataset
    from meld_emotion.training.dataset import load_ok_rows
    root, pre = synthetic_crops
    rows = load_ok_rows(root / "train" / "manifest.jsonl")
    ds = MeldCropDataset(rows, pre, root, stage2_config("fusion"), whitespace_encode, train=False)
    item = ds[6]                                         # dia1_utt2: 2 faces x 3 frames
    assert item["face_pixels"].shape == (6, 3, 224, 224) and item["face_pixels"].dtype == torch.uint8
    assert "face_feat" not in item
    assert item["face_frame"].tolist() == [0, 0, 1, 1, 2, 2] and item["face_track"].tolist() == [0, 1, 0, 1, 0, 1]
    assert abs(item["face_pixels"][3, 0, 0, 0].item() - 60) <= 4   # 4th crop: red = 3 * 20, within JPEG error
    assert item["scene_feat"].shape[0] == 3 and ds.lengths[6] == len(item["input_ids"])
    empty = ds[4]                                        # dia1_utt0: 0 faces
    assert empty["face_pixels"].shape == (0, 3, 224, 224)


def test_crop_dataset_caps_faces_per_clip_consistently(synthetic_crops):
    from meld_emotion.training.config import stage2_config
    from meld_emotion.training.crops import MeldCropDataset
    from meld_emotion.training.dataset import load_ok_rows
    root, pre = synthetic_crops
    rows = load_ok_rows(root / "train" / "manifest.jsonl")
    ds = MeldCropDataset(rows, pre, root, stage2_config("fusion", max_faces_per_clip=4), whitespace_encode, train=False)
    item = ds[6]
    assert item["face_pixels"].shape[0] == 4 == len(item["face_frame"]) == len(item["face_track"])
    assert item["face_frame"].tolist() == sorted(item["face_frame"].tolist())


def test_collate_crops_flattens_faces_with_batch_and_slot_indices(synthetic_crops):
    from meld_emotion.training.config import stage2_config
    from meld_emotion.training.crops import MeldCropDataset, collate_crops
    from meld_emotion.training.dataset import load_ok_rows
    root, pre = synthetic_crops
    rows = load_ok_rows(root / "train" / "manifest.jsonl")
    ds = MeldCropDataset(rows, pre, root, stage2_config("fusion"), whitespace_encode, train=False)
    batch = collate_crops([ds[4], ds[5], ds[6]], pad_id=1)   # 0, 3, 6 faces
    assert batch["face_pixels"].shape == (9, 3, 224, 224)
    assert batch["face_batch_idx"].tolist() == [1, 1, 1, 2, 2, 2, 2, 2, 2]
    assert batch["face_slot"].tolist() == [0, 1, 2, 0, 1, 2, 3, 4, 5]
    assert batch["face_mask"].sum(1).tolist() == [0, 3, 6] and batch["face_feat"].shape == (3, 6, 1)
    assert batch["scene_feat"].shape[0] == 3 and batch["emotion"].shape == (3,)
