import numpy as np
import torch

from conftest import StubTextEncoder


class _StubFaceDetector:
    pass


class _StubFaceEncoder:
    feature_dim = 8
    meld_labels = ("neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear")

    def encode_batch(self, images):
        return {"features": np.zeros((len(images), 8), dtype=np.float32),
                "probs": np.zeros((len(images), 7), dtype=np.float32)}


class _StubSceneEncoder:
    feature_dim = 4

    def encode(self, image):
        return np.zeros(4, dtype=np.float32)


class _StubTokenizer:
    pad_token_id = 1
    cls_token_id = 0


def _stub_factories():
    return dict(text_encoder_factory=lambda: StubTextEncoder(32),
                tokenizer_factory=lambda: _StubTokenizer(),
                face_encoder_factory=lambda: _StubFaceEncoder(),
                scene_encoder_factory=lambda: _StubSceneEncoder(),
                face_detector_factory=lambda: _StubFaceDetector())


def test_loads_a_matching_model_and_reproduces_the_checkpoints_own_state(synthetic_checkpoint):
    from meld_emotion.inference.loader import load_inference_bundle
    path, config = synthetic_checkpoint
    bundle = load_inference_bundle(path, device="cpu", **_stub_factories())
    assert bundle.device == "cpu" and bundle.stage == 1
    assert (bundle.face_dim, bundle.scene_dim, bundle.checkpoint_path) == (8, 4, path)
    assert bundle.config.d_model == config.d_model and bundle.config.n_layers == config.n_layers
    assert bundle.tokenizer.pad_token_id == 1
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for key, value in bundle.model.state_dict().items():
        assert torch.equal(value, ckpt["model_state"][key])


def test_vision_only_checkpoint_skips_the_text_encoder(tmp_path):
    from conftest import write_synthetic_checkpoint
    from meld_emotion.inference.loader import load_inference_bundle
    path = tmp_path / "vision_only.pt"
    write_synthetic_checkpoint(path, use_text=False)
    factories = _stub_factories()
    factories["text_encoder_factory"] = lambda: (_ for _ in ()).throw(AssertionError("must not be called"))
    factories["tokenizer_factory"] = lambda: (_ for _ in ()).throw(AssertionError("must not be called"))
    bundle = load_inference_bundle(path, device="cpu", **factories)
    assert bundle.tokenizer is None and bundle.model.text_encoder is None


def test_resolve_device_auto_is_used_when_device_not_given(synthetic_checkpoint, monkeypatch):
    from meld_emotion.inference import loader as loader_module
    from meld_emotion.inference.loader import load_inference_bundle
    path, _ = synthetic_checkpoint
    monkeypatch.setattr(loader_module, "resolve_device", lambda name: "cpu" if name == "auto" else name)
    assert load_inference_bundle(path, **_stub_factories()).device == "cpu"


def test_stage2_checkpoint_loads_the_fine_tuned_face_weights_by_default(synthetic_checkpoint,
                                                                         synthetic_stage2_checkpoint, monkeypatch):
    from meld_emotion.inference import loader as loader_module
    from meld_emotion.inference.loader import load_inference_bundle
    seen = []

    class _RecordingFaceEncoder(_StubFaceEncoder):
        def __init__(self, device=None, weights_path=None):
            seen.append(weights_path)

    monkeypatch.setattr(loader_module, "FaceEmotionEncoder", _RecordingFaceEncoder)
    factories = _stub_factories()
    del factories["face_encoder_factory"]                 # use the default (real) factory path
    s2_path, s2_config = synthetic_stage2_checkpoint
    bundle = load_inference_bundle(s2_path, device="cpu", **factories)
    assert bundle.stage == 2 and seen[-1] == s2_path
    assert not bundle.config.use_scene and not bundle.config.use_track_id
    s1_path, _ = synthetic_checkpoint
    load_inference_bundle(s1_path, device="cpu", **factories)
    assert seen[-1] is None                                # Stage 1 checkpoint: pretrained ViT, no override
