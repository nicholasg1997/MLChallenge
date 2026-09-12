import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.network


def _images(n, seed=0):
    rng = np.random.default_rng(seed)
    return [Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(n)]


@pytest.fixture(scope="module")
def face_encoder():
    from meld_emotion.vision.encoders import FaceEmotionEncoder
    return FaceEmotionEncoder(device="cpu")


@pytest.fixture(scope="module")
def scene_encoder():
    from meld_emotion.vision.encoders import SceneEncoder
    return SceneEncoder(device="cpu")


def test_face_labels_come_from_the_checkpoint_and_map_onto_meld(face_encoder):
    assert set(face_encoder.labels) == {"sad", "disgust", "angry", "neutral", "fear", "surprise", "happy"}
    assert set(face_encoder.meld_labels) == {"sadness", "disgust", "anger", "neutral", "fear", "surprise", "joy"}
    assert face_encoder.feature_dim == 768


def test_face_encode_batch_shapes_and_probabilities(face_encoder):
    out = face_encoder.encode_batch(_images(3))
    assert out["features"].shape == (3, 768)
    assert out["probs"].shape == (3, 7)
    assert out["features"].dtype == np.float32
    assert np.allclose(out["probs"].sum(axis=1), 1.0, atol=1e-4)


def test_face_encode_batch_of_nothing_returns_empty_arrays_with_the_right_width(face_encoder):
    out = face_encoder.encode_batch([])
    assert out["features"].shape == (0, 768)
    assert out["probs"].shape == (0, 7)


def test_face_probs_match_the_checkpoints_own_forward_pass(face_encoder):
    # Guards the CLS-pooling path: our features must be exactly what the
    # checkpoint's classifier consumes (post-LayerNorm CLS), so the probs we
    # return equal the model's own prediction.
    image = _images(1)[0]
    ours = face_encoder.encode(image)["probs"]
    inputs = face_encoder.processor(images=image, return_tensors="pt")
    with torch.no_grad():
        theirs = torch.softmax(face_encoder.model(**inputs).logits, dim=-1)[0].numpy()
    assert np.allclose(ours, theirs, atol=1e-5)


def test_scene_encoder_shapes(scene_encoder):
    assert scene_encoder.feature_dim == 512
    assert scene_encoder.encode_batch(_images(2)).shape == (2, 512)
    assert scene_encoder.encode(_images(1)[0]).shape == (512,)
    assert scene_encoder.encode_batch([]).shape == (0, 512)
