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
