import json

import pytest


def test_compare_reports_argmax_match_and_max_abs_diff():
    from scripts.consistency_check import compare
    a = {"neutral": 0.60, "joy": 0.20, "surprise": 0.10, "anger": 0.05, "sadness": 0.03, "disgust": 0.01, "fear": 0.01}
    b = {"neutral": 0.61, "joy": 0.19, "surprise": 0.10, "anger": 0.05, "sadness": 0.03, "disgust": 0.01, "fear": 0.01}
    result = compare(a, b)
    assert result["argmax_match"] is True and result["max_abs_diff"] == pytest.approx(0.01, abs=1e-6)


def test_compare_detects_a_real_argmax_mismatch():
    from scripts.consistency_check import compare
    a = {"neutral": 0.51, "joy": 0.49, "surprise": 0.0, "anger": 0.0, "sadness": 0.0, "disgust": 0.0, "fear": 0.0}
    b = {"neutral": 0.40, "joy": 0.60, "surprise": 0.0, "anger": 0.0, "sadness": 0.0, "disgust": 0.0, "fear": 0.0}
    assert compare(a, b)["argmax_match"] is False


def test_batch_predict_one_reads_the_stored_probs_in_emotion_order(tmp_path):
    from meld_emotion.data.labels import EMOTIONS
    from scripts.consistency_check import batch_predict_one
    path = tmp_path / "test_predictions.jsonl"
    probs = [0.05, 0.6, 0.1, 0.05, 0.1, 0.05, 0.05]
    path.write_text(json.dumps({"clip": "dia1_utt1", "pred": "joy", "probs": probs}) + "\n"
                    + json.dumps({"clip": "dia1_utt2", "pred": "anger", "probs": probs[::-1]}) + "\n")
    result = batch_predict_one(path, "dia1_utt1")
    assert list(result["emotion_probs"]) == list(EMOTIONS)
    assert result["emotion_probs"]["joy"] == pytest.approx(0.6)
    with pytest.raises(KeyError):
        batch_predict_one(path, "dia9_utt9")


class _WhitespaceTokenizer:
    """HF-call contract encode_text relies on, over conftest's whitespace ids."""
    pad_token_id = 1

    def __call__(self, first, second=None, truncation=None, max_length=None):
        from conftest import whitespace_encode
        return whitespace_encode(first if second is not None else "", second if second is not None else first)


def test_local_batch_predictions_match_the_inference_path_item_by_item(synthetic_features, synthetic_checkpoint):
    """The batch path (padded batches of 4 through evaluate()) and the replay
    path's predict() (one item, no padding) must agree on identical inputs --
    the same guard the real check applies, minus the video."""
    import torch
    from meld_emotion.data.labels import EMOTIONS
    from meld_emotion.inference.predict import predict
    from meld_emotion.training.dataset import MeldFeatureDataset, load_ok_rows
    from meld_emotion.training.model import FusionModel
    from meld_emotion.training.text import encode_text
    from conftest import StubTextEncoder
    from scripts.consistency_check import local_batch_predictions
    import functools
    root, _, dev_manifest = synthetic_features
    rows = load_ok_rows(dev_manifest)[:6]
    ckpt_path, config = synthetic_checkpoint

    torch.manual_seed(0)
    batch = local_batch_predictions(ckpt_path, rows, device="cpu", features_dir=root,
                                    text_encoder_factory=lambda: StubTextEncoder(32),
                                    tokenizer_factory=_WhitespaceTokenizer)
    assert set(batch) == {f"dia{r['dialogue_id']}_utt{r['utterance_id']}" for r in rows}
    for probs in batch.values():
        assert list(probs["emotion_probs"]) == list(EMOTIONS)
        assert sum(probs["emotion_probs"].values()) == pytest.approx(1.0, abs=1e-5)

    torch.manual_seed(0)
    model = FusionModel(config, StubTextEncoder(32), face_dim=8, scene_dim=4)
    model.load_state_dict(torch.load(ckpt_path, weights_only=False)["model_state"])
    model.eval()
    encode_fn = functools.partial(encode_text, _WhitespaceTokenizer(), max_length=config.max_text_tokens)
    dataset = MeldFeatureDataset(rows, root, config, encode_fn)
    for i, row in enumerate(rows):
        single, _ = predict(model, dataset[i], pad_id=1, device="cpu")
        clip = f"dia{row['dialogue_id']}_utt{row['utterance_id']}"
        for e in EMOTIONS:
            assert single[e] == pytest.approx(batch[clip]["emotion_probs"][e], abs=1e-5)
