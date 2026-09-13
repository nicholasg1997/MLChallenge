import numpy as np
import pytest
import torch

from conftest import StubTextEncoder
from meld_emotion.data.labels import EMOTIONS, SENTIMENTS


def _model(config):
    from meld_emotion.training.model import FusionModel
    return FusionModel(config, StubTextEncoder(32) if config.use_text else None, face_dim=8, scene_dim=4).eval()


def _cfg(**kw):
    from meld_emotion.training.config import TrainConfig
    return TrainConfig(**{**dict(d_model=32, n_heads=4, ff_dim=64, n_layers=1), **kw})


ENC = {"input_ids": [0, 5, 6, 1], "attention_mask": [1, 1, 1, 1]}


def test_build_item_clamps_frames_remaps_tracks_and_keeps_dims_from_the_checkpoint():
    from meld_emotion.inference.predict import build_item
    item = build_item(_cfg(max_frames=32, max_track_slots=16), ENC,
                      face_feat=np.ones((3, 8), np.float32), face_frame_idx=[0, 40, 44],
                      face_track_raw=[7, 55, 7], scene_feat=np.ones((2, 4), np.float32),
                      face_dim=8, scene_dim=4, clip="dia1_utt1")
    assert item["face_frame"].tolist() == [0, 31, 31] and item["face_track"].tolist() == [0, 1, 0]
    assert item["scene_frame"].tolist() == [0, 1] and item["clip"] == "dia1_utt1"
    assert item["input_ids"].tolist() == ENC["input_ids"] and item["emotion"].item() == 0


def test_build_item_uses_checkpoint_dims_for_empty_modalities():
    from meld_emotion.inference.predict import build_item
    item = build_item(_cfg(), ENC, face_feat=np.zeros((0, 8), np.float32), face_frame_idx=[], face_track_raw=[],
                      scene_feat=np.zeros((0, 4), np.float32), face_dim=768, scene_dim=512, clip="x")
    assert item["face_feat"].shape == (0, 768) and item["scene_feat"].shape == (0, 512)


def test_build_item_drops_modalities_the_config_switched_off():
    from meld_emotion.inference.predict import build_item
    item = build_item(_cfg(use_scene=False), ENC, face_feat=np.ones((2, 8), np.float32), face_frame_idx=[0, 1],
                      face_track_raw=[0, 0], scene_feat=np.ones((2, 4), np.float32), face_dim=8, scene_dim=4, clip="x")
    assert item["scene_feat"].shape == (0, 4) and item["face_feat"].shape == (2, 8)


def test_build_item_applies_the_stage2_face_cap_only_for_stage2_configs():
    from meld_emotion.inference.predict import build_item
    feats = np.arange(100, dtype=np.float32)[:, None].repeat(8, 1)
    kw = dict(face_frame_idx=list(range(100)), face_track_raw=[0] * 100, scene_feat=np.zeros((0, 4), np.float32),
              face_dim=8, scene_dim=4, clip="x")
    stage2 = build_item(_cfg(face_trainable_layers=4, max_faces_per_clip=64), ENC, feats, **kw)
    stage1 = build_item(_cfg(face_trainable_layers=0, max_faces_per_clip=64), ENC, feats, **kw)
    assert stage2["face_feat"].shape[0] == 64 == len(stage2["face_frame"]) == len(stage2["face_track"])
    assert stage2["face_feat"][0, 0] == 0 and stage2["face_feat"][-1, 0] == 99      # uniform, endpoints kept
    assert stage1["face_feat"].shape[0] == 100


def test_predict_returns_normalised_distributions_and_text_masking_changes_them():
    from meld_emotion.inference.predict import build_item, dummy_text_encoding, predict
    cfg = _cfg()
    model = _model(cfg)
    item = build_item(cfg, ENC, np.random.default_rng(0).standard_normal((3, 8)).astype(np.float32), [0, 1, 2],
                      [0, 0, 0], np.random.default_rng(1).standard_normal((2, 4)).astype(np.float32),
                      face_dim=8, scene_dim=4, clip="x")
    emo, sent = predict(model, item, pad_id=1, device="cpu")
    assert list(emo) == list(EMOTIONS) and list(sent) == list(SENTIMENTS)
    assert sum(emo.values()) == pytest.approx(1.0, abs=1e-5) and sum(sent.values()) == pytest.approx(1.0, abs=1e-5)
    masked, _ = predict(model, item, pad_id=1, device="cpu", force_drop_text=True)
    assert masked != emo

    class _Tok:
        cls_token_id = 0
    assert dummy_text_encoding(_Tok()) == {"input_ids": [0], "attention_mask": [1]}
    dummy_item = build_item(cfg, dummy_text_encoding(_Tok()), item["face_feat"].numpy(), [0, 1, 2], [0, 0, 0],
                            item["scene_feat"].numpy(), face_dim=8, scene_dim=4, clip="x")
    provisional, _ = predict(model, dummy_item, pad_id=1, device="cpu", force_drop_text=True)
    assert provisional == pytest.approx(masked, abs=1e-6)   # text masked -> the text content is irrelevant


def test_predict_is_finite_with_no_visual_tokens_at_all():
    from meld_emotion.inference.predict import build_item, predict
    cfg = _cfg()
    item = build_item(cfg, ENC, np.zeros((0, 8), np.float32), [], [], np.zeros((0, 4), np.float32),
                      face_dim=8, scene_dim=4, clip="x")
    emo, _ = predict(_model(cfg), item, pad_id=1, device="cpu")
    assert all(np.isfinite(v) for v in emo.values())
