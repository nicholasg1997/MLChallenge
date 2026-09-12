import json

import torch

from conftest import StubTextEncoder, whitespace_encode
from test_crops import write_synthetic_crops
from test_stage2_model import StubFaceEncoder


def _cfg(**kw):
    from meld_emotion.training.config import stage2_config
    defaults = dict(d_model=32, n_heads=4, ff_dim=64, n_layers=1, batch_size=8, epochs=2, patience=5,
                    device="cpu", face_augment=True, loader_workers=0)
    return stage2_config("fusion", **{**defaults, **kw})


def _stage1(synthetic_features, tmp_path):
    from meld_emotion.training.config import TrainConfig
    from meld_emotion.training.train import train
    root, _, _ = synthetic_features
    s1 = TrainConfig(d_model=32, n_heads=4, ff_dim=64, n_layers=1, batch_size=8, epochs=1, device="cpu")
    train(s1, root, tmp_path / "s1", encode_fn=whitespace_encode, text_encoder=StubTextEncoder(32), pad_id=1, log=lambda *_: None)
    return tmp_path / "s1" / "best.pt"


def test_stage2_optimizer_has_three_learning_rates(synthetic_features):
    from meld_emotion.training.model import FusionModel
    from meld_emotion.training.stage2_model import Stage2Model
    from meld_emotion.training.train_stage2 import build_stage2_optimizer
    cfg = _cfg(lr_face=1e-5, lr_text=2e-5, lr_fusion=1e-3)
    model = Stage2Model(FusionModel(cfg, StubTextEncoder(32), face_dim=8, scene_dim=4), StubFaceEncoder(8))
    assert sorted(g["lr"] for g in build_stage2_optimizer(model, cfg).param_groups) == [1e-5, 2e-5, 1e-3]


def test_train_stage2_end_to_end_initialises_from_stage1_and_writes_artifacts(synthetic_features, tmp_path):
    from meld_emotion.training.train_stage2 import train_stage2
    root, _, _ = synthetic_features
    pre = tmp_path / "pre"
    write_synthetic_crops(root, pre, "train")
    write_synthetic_crops(root, pre, "dev")
    s1 = _stage1(synthetic_features, tmp_path)
    out = tmp_path / "s2"
    seen = []
    results = train_stage2(_cfg(seed=1), root, pre, out, test_split="dev", init_from=s1,
                           encode_fn=whitespace_encode, text_encoder=StubTextEncoder(32), face_encoder=StubFaceEncoder(8),
                           pad_id=1, log=lambda *_: None, on_epoch_end=seen.append)
    assert results["stage"] == 2 and results["base_checkpoint"] == str(s1) and results["face_trainable_params"] > 0
    assert [r["epoch"] for r in seen] == [1, 2]
    for key in ("dev", "dev_masked_text_only", "dev_masked_vision_only", "test"):
        assert 0.0 <= results[key]["emotion"]["weighted_f1"] <= 1.0
    ckpt = torch.load(out / "best.pt", weights_only=False)
    assert "face_encoder_state" in ckpt and "proj.weight" in ckpt["face_encoder_state"]
    assert json.load(open(out / "results.json"))["config"]["face_trainable_layers"] == 4
    assert (out / "test_predictions.jsonl").exists()


def test_train_stage2_refuses_to_run_without_a_base_checkpoint(synthetic_features, tmp_path):
    import pytest
    from meld_emotion.training.train_stage2 import train_stage2
    root, _, _ = synthetic_features
    with pytest.raises(FileNotFoundError):
        train_stage2(_cfg(init_from=None), root, tmp_path / "pre", tmp_path / "s2",
                     encode_fn=whitespace_encode, text_encoder=StubTextEncoder(32), face_encoder=StubFaceEncoder(8), pad_id=1)
