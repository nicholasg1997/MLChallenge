import numpy as np
import pytest
import torch

from conftest import FACE_DIM, SCENE_DIM, whitespace_encode


def test_remap_track_ids_by_first_appearance_with_overflow_slot():
    from meld_emotion.training.dataset import remap_track_ids
    ids = np.array([7, 0, 7, 55, 0, 3, 55])
    assert remap_track_ids(ids, max_slots=16).tolist() == [0, 1, 0, 2, 1, 3, 2]
    assert remap_track_ids(ids, max_slots=2).tolist() == [0, 1, 0, 2, 1, 2, 2]   # 3rd+ distinct -> overflow=2
    assert remap_track_ids(np.array([], dtype=np.int64), 16).shape == (0,)


def test_load_ok_rows_drops_non_ok_rows(synthetic_features):
    from meld_emotion.training.dataset import load_ok_rows
    _, train, _ = synthetic_features
    rows = load_ok_rows(train)
    assert len(rows) == 28 and all(r["status"] == "ok" for r in rows)


def test_infer_feature_dims_reads_widths_even_from_a_zero_face_clip(synthetic_features):
    from meld_emotion.training.dataset import infer_feature_dims, load_ok_rows
    root, train, _ = synthetic_features
    rows = load_ok_rows(train)
    assert rows[0]["n_faces"] == 0                       # first clip cycles to 0 faces
    assert infer_feature_dims(rows, root) == (FACE_DIM, SCENE_DIM)


def test_dataset_item_shapes_labels_and_context(synthetic_features):
    from meld_emotion.training.config import TrainConfig
    from meld_emotion.training.dataset import EMOTION_TO_IDX, MeldFeatureDataset, load_ok_rows
    root, train, _ = synthetic_features
    rows = load_ok_rows(train)
    ds = MeldFeatureDataset(rows, root, TrainConfig(context_k=2), whitespace_encode)
    item = ds[5]                                          # dia1_utt1: 1 face/frame, 1 previous line
    assert item["face_feat"].shape == (3, FACE_DIM) and item["face_feat"].dtype == torch.float32
    assert item["face_frame"].tolist() == [0, 1, 2]
    assert item["face_track"].tolist() == [0, 0, 0]       # raw id 0 in every frame -> slot 0
    assert item["scene_feat"].shape == (3, SCENE_DIM)
    assert item["emotion"].item() == EMOTION_TO_IDX[rows[5]["emotion"]]
    assert item["input_ids"].dtype == torch.long and item["input_ids"][0] == 0
    assert item["clip"] == "dia1_utt1"
    n_context_words = len(" ".join(rows[5]["context_prev"][-2:]).split())
    assert len(item["input_ids"]) == 2 + n_context_words + len(rows[5]["text"].split())


def test_dataset_clamps_frame_positions_and_remaps_tracks(synthetic_features, tmp_path):
    from meld_emotion.training.config import TrainConfig
    from meld_emotion.training.dataset import MeldFeatureDataset, load_ok_rows
    root, train, _ = synthetic_features
    rows = load_ok_rows(train)
    row = rows[6]                                         # 2 faces/frame -> raw ids 0 and 7
    z = dict(np.load(root / row["feature_path"]))
    z["face_frame_idx"] = z["face_frame_idx"] + 40        # push past the 32-slot table
    z["scene_frame_idx"] = z["scene_frame_idx"] + 40
    np.savez(root / row["feature_path"], **z)
    ds = MeldFeatureDataset([row], root, TrainConfig(max_frames=32, max_track_slots=16), whitespace_encode)
    item = ds[0]
    assert item["face_frame"].max().item() == 31 and item["scene_frame"].max().item() == 31
    assert sorted(set(item["face_track"].tolist())) == [0, 1]


def test_dataset_switches_off_disabled_modalities_and_masks_long_clips(synthetic_features):
    from meld_emotion.training.config import TrainConfig
    from meld_emotion.training.dataset import MeldFeatureDataset, load_ok_rows
    root, train, _ = synthetic_features
    rows = load_ok_rows(train)
    no_faces = MeldFeatureDataset(rows, root, TrainConfig(use_faces=False), whitespace_encode)[5]
    assert no_faces["face_feat"].shape == (0, FACE_DIM) and no_faces["scene_feat"].shape[0] == 3
    no_scene = MeldFeatureDataset(rows, root, TrainConfig(use_scene=False), whitespace_encode)[5]
    assert no_scene["scene_feat"].shape == (0, SCENE_DIM) and no_scene["face_feat"].shape[0] == 3
    masked = MeldFeatureDataset(rows, root, TrainConfig(mask_vision_over_seconds=15.0), whitespace_encode)
    assert rows[7]["duration_s"] == 20.0
    assert masked[7]["face_feat"].shape[0] == 0 and masked[7]["scene_feat"].shape[0] == 0
    assert masked[8]["scene_feat"].shape[0] == 3


def test_dataset_exposes_token_lengths(synthetic_features):
    from meld_emotion.training.config import TrainConfig
    from meld_emotion.training.dataset import MeldFeatureDataset, load_ok_rows
    root, train, _ = synthetic_features
    ds = MeldFeatureDataset(load_ok_rows(train), root, TrainConfig(), whitespace_encode)
    assert ds.lengths == [len(ds[i]["input_ids"]) for i in range(len(ds))]


def test_bucket_sampler_covers_every_index_once_and_groups_similar_lengths():
    from meld_emotion.training.dataset import BucketBatchSampler
    rng = np.random.default_rng(0)
    lengths = rng.integers(5, 120, size=203).tolist()
    sampler = BucketBatchSampler(lengths, batch_size=16, shuffle=True, seed=0, chunk_batches=4)
    batches = list(sampler)
    assert len(sampler) == len(batches) == 13
    assert sorted(i for b in batches for i in b) == list(range(203))
    spread_bucketed = np.mean([max(lengths[i] for i in b) - min(lengths[i] for i in b) for b in batches])
    spread_random = np.mean([np.ptp([lengths[i] for i in rng.permutation(203)[:16]]) for _ in range(13)])
    assert spread_bucketed < spread_random / 2


def test_bucket_sampler_is_deterministic_per_epoch_and_sorted_when_not_shuffling():
    from meld_emotion.training.dataset import BucketBatchSampler
    lengths = [30, 5, 20, 5, 30, 20, 7, 8]
    a = BucketBatchSampler(lengths, batch_size=2, shuffle=True, seed=1)
    b = BucketBatchSampler(lengths, batch_size=2, shuffle=True, seed=1)
    assert list(a) == list(b)
    a.set_epoch(1)
    assert list(a) != list(b)
    ordered = list(BucketBatchSampler(lengths, batch_size=2, shuffle=False))
    assert [lengths[i] for batch in ordered for i in batch] == sorted(lengths)


def test_preloaded_and_lazy_datasets_yield_identical_items(synthetic_features):
    from meld_emotion.training.config import TrainConfig
    from meld_emotion.training.dataset import MeldFeatureDataset, load_ok_rows
    root, train, _ = synthetic_features
    rows = load_ok_rows(train)
    eager = MeldFeatureDataset(rows, root, TrainConfig(), whitespace_encode, preload=True)
    lazy = MeldFeatureDataset(rows, root, TrainConfig(), whitespace_encode, preload=False)
    assert eager.features is not None and len(eager.features) == len(rows) and lazy.features is None
    for i in (0, 5, 6, len(rows) - 1):
        a, b = eager[i], lazy[i]
        assert a.keys() == b.keys()
        for k in a:
            assert (a[k] == b[k]) if k == "clip" else torch.equal(a[k], b[k]), k


def test_collate_pads_and_masks_variable_length_sets(synthetic_features):
    from meld_emotion.training.config import TrainConfig
    from meld_emotion.training.dataset import MeldFeatureDataset, collate, load_ok_rows
    root, train, _ = synthetic_features
    rows = load_ok_rows(train)
    ds = MeldFeatureDataset(rows, root, TrainConfig(), whitespace_encode)
    batch = collate([ds[4], ds[5], ds[6]], pad_id=1)      # 0, 1, 2 faces per frame
    assert batch["face_feat"].shape == (3, 6, FACE_DIM)
    assert batch["face_mask"].sum(1).tolist() == [0, 3, 6]
    assert batch["scene_feat"].shape == (3, 3, SCENE_DIM) and batch["scene_mask"].all()
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    lengths = [len(ds[i]["input_ids"]) for i in (4, 5, 6)]
    assert batch["attention_mask"].sum(1).tolist() == lengths
    assert (batch["input_ids"][0, lengths[0]:] == 1).all()
    assert batch["emotion"].shape == (3,) and batch["sentiment"].shape == (3,)
    assert batch["clips"] == ["dia1_utt0", "dia1_utt1", "dia1_utt2"]
