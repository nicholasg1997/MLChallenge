import torch
import torch.nn as nn

from conftest import StubTextEncoder, whitespace_encode
from test_crops import write_synthetic_crops


class StubFaceEncoder(nn.Module):
    """Same contract as TrainableFaceEncoder, tiny: mean colour -> feature."""
    def __init__(self, dim=8):
        super().__init__()
        self.feature_dim = dim
        self.proj = nn.Linear(3, dim)

    def trainable_parameters(self):
        return self.parameters()

    def export_state(self):
        return {k: v.detach().cpu() for k, v in self.state_dict().items()}

    def forward(self, pixels_uint8):
        if pixels_uint8.shape[0] == 0:
            return torch.zeros((0, self.feature_dim))
        return self.proj(pixels_uint8.float().mean(dim=(2, 3)) / 255.0)


def _cfg(**kw):
    from meld_emotion.training.config import stage2_config
    return stage2_config("fusion", d_model=32, n_heads=4, ff_dim=64, n_layers=1, **kw)


def _batch(synthetic_features, tmp_path, cfg, idxs=(4, 5, 6)):
    from meld_emotion.training.crops import MeldCropDataset, collate_crops
    from meld_emotion.training.dataset import load_ok_rows
    root, train, _ = synthetic_features
    pre = tmp_path / "pre"
    write_synthetic_crops(root, pre, "train")
    ds = MeldCropDataset(load_ok_rows(train), pre, root, cfg, whitespace_encode, train=False)
    return collate_crops([ds[i] for i in idxs], pad_id=1)


def test_stage2_model_runs_the_face_encoder_and_the_fusion_model(synthetic_features, tmp_path):
    from meld_emotion.training.model import FusionModel
    from meld_emotion.training.stage2_model import Stage2Model
    cfg = _cfg()
    fusion = FusionModel(cfg, StubTextEncoder(32), face_dim=8, scene_dim=4)
    model = Stage2Model(fusion, StubFaceEncoder(8)).eval()
    out = model(_batch(synthetic_features, tmp_path, cfg))
    assert out["emotion_logits"].shape == (3, 7) and torch.isfinite(out["emotion_logits"]).all()
    assert not torch.allclose(out["emotion_logits"], model(_batch(synthetic_features, tmp_path, cfg), force_drop_vision=True)["emotion_logits"])


def test_stage2_parameter_groups_partition_all_parameters(synthetic_features, tmp_path):
    from meld_emotion.training.model import FusionModel
    from meld_emotion.training.stage2_model import Stage2Model
    cfg = _cfg()
    model = Stage2Model(FusionModel(cfg, StubTextEncoder(32), face_dim=8, scene_dim=4), StubFaceEncoder(8))
    groups = [{id(p) for p in g} for g in (model.text_parameters(), model.fusion_parameters(), model.face_parameters())]
    assert all(groups) and not (groups[0] & groups[1]) and not (groups[1] & groups[2]) and not (groups[0] & groups[2])
    assert groups[0] | groups[1] | groups[2] == {id(p) for p in model.parameters()}


def test_init_fusion_from_stage1_loads_the_stage1_weights(synthetic_features, tmp_path):
    from meld_emotion.training.config import TrainConfig
    from meld_emotion.training.model import FusionModel
    from meld_emotion.training.stage2_model import init_fusion_from_stage1
    from meld_emotion.training.train import train
    root, _, _ = synthetic_features
    s1 = TrainConfig(d_model=32, n_heads=4, ff_dim=64, n_layers=1, batch_size=8, epochs=1, device="cpu")
    train(s1, root, tmp_path / "s1", encode_fn=whitespace_encode, text_encoder=StubTextEncoder(32), pad_id=1, log=lambda *_: None)
    fresh = FusionModel(_cfg(), StubTextEncoder(32), face_dim=8, scene_dim=4)
    before = fresh.emotion_head.weight.detach().clone()
    ckpt = init_fusion_from_stage1(fresh, tmp_path / "s1" / "best.pt")
    assert ckpt["epoch"] == 1 and not torch.equal(before, fresh.emotion_head.weight)
    trained = torch.load(tmp_path / "s1" / "best.pt", weights_only=False)["model_state"]["emotion_head.weight"]
    assert torch.equal(fresh.emotion_head.weight, trained)
