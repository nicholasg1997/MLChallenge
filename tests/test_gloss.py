import numpy as np
import pytest
import torch


def test_scene_gloss_picks_the_closest_bank_phrase():
    from meld_emotion.inference.gloss import SCENE_PROMPT_BANK, scene_gloss
    bank = torch.nn.functional.normalize(torch.eye(len(SCENE_PROMPT_BANK), 8), dim=-1)
    query = bank[3].numpy() * 5.0  # far from unit norm on purpose; scene_gloss must normalise it
    assert scene_gloss(query, bank, margin=0.05)[0] == SCENE_PROMPT_BANK[3]


def test_scene_gloss_reports_only_the_top1_when_the_next_is_within_margin():
    from meld_emotion.inference.gloss import scene_gloss
    bank = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.999, 0.045], [0.0, 1.0]]), dim=-1)
    assert len(scene_gloss(np.array([1.0, 0.0], dtype=np.float32), bank, margin=0.5)) == 1


def test_face_gloss_reads_the_vision_only_top2_and_counts_faces():
    from meld_emotion.inference.gloss import face_gloss
    probs = {"neutral": 0.23, "joy": 0.05, "surprise": 0.52, "anger": 0.1, "sadness": 0.05, "disgust": 0.03, "fear": 0.02}
    assert face_gloss(2, probs) == "2 faces visible, reading surprise (52%) or neutral (23%)"
    assert face_gloss(1, probs).startswith("1 face visible, reading surprise")
    assert face_gloss(0, probs) == "no faces visible"
    assert face_gloss(0, None) == "no faces visible"


def test_embed_prompt_bank_is_l2_normalised_and_unwraps_pooler_output():
    from meld_emotion.inference.gloss import SCENE_PROMPT_BANK, embed_prompt_bank

    class _StubOut:
        def __init__(self, n):
            self.pooler_output = torch.randn(n, 8) * 10  # deliberately not unit norm

    class _StubModel:
        def get_text_features(self, **kwargs):
            return _StubOut(len(SCENE_PROMPT_BANK))

    class _StubTokenizer:
        def __call__(self, texts, return_tensors, padding):
            return {"input_ids": torch.zeros(len(texts), 1, dtype=torch.long)}

    embeddings = embed_prompt_bank(_StubModel(), _StubTokenizer(), "cpu")
    norms = embeddings.norm(dim=-1)
    assert embeddings.shape == (len(SCENE_PROMPT_BANK), 8)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
