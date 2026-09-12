import pytest
import torch


def test_normalize_pixels_maps_uint8_to_minus_one_one():
    from meld_emotion.training.face_encoder import normalize_pixels
    x = torch.tensor([[[[0, 128, 255]]]], dtype=torch.uint8).expand(1, 3, 1, 3)
    y = normalize_pixels(x)
    assert y.dtype == torch.float32
    assert torch.allclose(y[0, 0, 0], torch.tensor([-1.0, 128 / 127.5 - 1, 1.0]), atol=1e-6)


def test_scatter_faces_places_each_crop_in_its_sample_and_slot():
    from meld_emotion.training.face_encoder import scatter_faces
    feats = torch.arange(1, 5, dtype=torch.float32)[:, None].expand(4, 2)        # 4 crops, D=2
    out = scatter_faces(feats, torch.tensor([0, 0, 2, 2]), torch.tensor([0, 1, 0, 1]), B=3, fmax=3)
    assert out.shape == (3, 3, 2)
    assert out[0, 0, 0] == 1 and out[0, 1, 0] == 2 and out[2, 1, 0] == 4
    assert torch.all(out[1] == 0) and torch.all(out[0, 2] == 0)
    assert scatter_faces(torch.zeros(0, 2), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long), B=2, fmax=0).shape == (2, 0, 2)


@pytest.mark.network
def test_trainable_face_encoder_matches_the_pretrained_forward_and_trains_only_the_top():
    from meld_emotion.training.face_encoder import TrainableFaceEncoder, normalize_pixels
    from meld_emotion.vision.encoders import FaceEmotionEncoder
    enc = TrainableFaceEncoder(trainable_layers=4).eval()
    ref = FaceEmotionEncoder(device="cpu")
    pixels = torch.randint(0, 256, (2, 3, 224, 224), dtype=torch.uint8)
    with torch.no_grad():
        ours = enc(pixels)
        theirs = ref.model.vit(pixel_values=normalize_pixels(pixels)).last_hidden_state[:, 0]
    assert ours.shape == (2, 768) and torch.allclose(ours, theirs, atol=1e-4)
    enc.chunk_size = 1                                   # chunked path must give the same features
    with torch.no_grad():
        assert torch.allclose(enc(pixels), theirs, atol=1e-4)
    assert sum(p.numel() for p in enc.trainable_parameters()) == 28_353_024
    enc.train()
    enc(pixels).sum().backward()
    assert enc.vit.layers[11].attention.q_proj.weight.grad is not None
    assert enc.vit.layers[7].attention.q_proj.weight.grad is None
    assert enc(torch.zeros((0, 3, 224, 224), dtype=torch.uint8)).shape == (0, 768)
    state = enc.export_state()
    ref.model.load_state_dict(state)                     # the demo's encoder can load Stage 2 weights
