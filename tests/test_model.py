import pytest
import torch

from conftest import FACE_DIM, SCENE_DIM, StubTextEncoder, whitespace_encode


def _batch(synthetic_features, config, idxs=(4, 5, 6)):
    from meld_emotion.training.dataset import MeldFeatureDataset, collate, load_ok_rows
    root, train, _ = synthetic_features
    ds = MeldFeatureDataset(load_ok_rows(train), root, config, whitespace_encode)
    return collate([ds[i] for i in idxs], pad_id=1)


def _model(config, text_encoder=StubTextEncoder(hidden_size=32)):
    from meld_emotion.training.model import FusionModel
    return FusionModel(config, text_encoder, face_dim=FACE_DIM, scene_dim=SCENE_DIM)


def _cfg(**kw):
    from meld_emotion.training.config import TrainConfig
    return TrainConfig(d_model=32, n_heads=4, ff_dim=64, n_layers=2, **kw)


def test_forward_produces_logits_of_the_right_shape(synthetic_features):
    cfg = _cfg()
    out = _model(cfg).eval()(_batch(synthetic_features, cfg))
    assert out["emotion_logits"].shape == (3, 7) and out["sentiment_logits"].shape == (3, 3)
    assert torch.isfinite(out["emotion_logits"]).all()


def test_forward_works_for_text_only_vision_only_and_no_scene(synthetic_features):
    for cfg in (_cfg(use_faces=False, use_scene=False), _cfg(use_text=False), _cfg(use_scene=False)):
        model = _model(cfg, text_encoder=None if not cfg.use_text else StubTextEncoder(32)).eval()
        out = model(_batch(synthetic_features, cfg))
        assert out["emotion_logits"].shape == (3, 7) and torch.isfinite(out["emotion_logits"]).all()


def test_a_sample_with_zero_visual_tokens_still_gets_finite_logits(synthetic_features):
    cfg = _cfg()
    batch = _batch(synthetic_features, cfg, idxs=(4,))           # clip with 0 faces
    batch["scene_mask"][:] = False                              # and now no scene tokens either
    out = _model(cfg).eval()(batch)
    assert torch.isfinite(out["emotion_logits"]).all()


def test_padding_tokens_do_not_change_the_prediction(synthetic_features):
    cfg = _cfg()
    model = _model(cfg).eval()
    single = _batch(synthetic_features, cfg, idxs=(5,))
    padded = _batch(synthetic_features, cfg, idxs=(5, 6))       # sample 5 now padded to sample 6's lengths
    a = model(single)["emotion_logits"][0]
    b = model(padded)["emotion_logits"][0]
    assert torch.allclose(a, b, atol=1e-5)


def test_force_drop_flags_change_the_output_and_track_id_flag_is_respected(synthetic_features):
    cfg = _cfg()
    model = _model(cfg).eval()
    batch = _batch(synthetic_features, cfg, idxs=(6,))
    full = model(batch)["emotion_logits"]
    assert not torch.allclose(full, model(batch, force_drop_vision=True)["emotion_logits"])
    assert not torch.allclose(full, model(batch, force_drop_text=True)["emotion_logits"])
    no_track = _model(_cfg(use_track_id=False)).eval()
    batch2 = _batch(synthetic_features, cfg, idxs=(6,))
    batch2["face_track"] = batch2["face_track"] + 5              # different slots
    assert torch.allclose(no_track(batch)["emotion_logits"], no_track(batch2)["emotion_logits"])


def test_modality_dropout_never_drops_both_and_never_drops_an_absent_modality(synthetic_features):
    cfg = _cfg(modality_dropout=0.5)
    model = _model(cfg).train()
    batch = _batch(synthetic_features, cfg, idxs=(4, 5, 6))      # sample 0 has 0 faces but has scene
    batch["scene_mask"][0] = False                              # now sample 0 has no vision at all
    torch.manual_seed(0)
    seen_text_drop = seen_vision_drop = False
    for _ in range(200):
        drop_text, drop_vision = model.modality_dropout_masks(batch)
        assert not (drop_text & drop_vision).any()
        assert not drop_text[0] and not drop_vision[0]           # nothing to fall back on / nothing to drop
        seen_text_drop |= bool(drop_text[1:].any())
        seen_vision_drop |= bool(drop_vision[1:].any())
    assert seen_text_drop and seen_vision_drop
    model.eval()
    drop_text, drop_vision = model.modality_dropout_masks(batch)
    assert not drop_text.any() and not drop_vision.any()         # eval mode: no dropout


def test_parameter_groups_split_text_encoder_from_the_rest(synthetic_features):
    cfg = _cfg()
    model = _model(cfg)
    text_ids = {id(p) for p in model.text_parameters()}
    fusion_ids = {id(p) for p in model.fusion_parameters()}
    assert text_ids and fusion_ids and not (text_ids & fusion_ids)
    assert text_ids | fusion_ids == {id(p) for p in model.parameters()}
