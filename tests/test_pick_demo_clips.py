import json


def _predictions():
    return [
        {"clip": "dia1_utt0", "true_emotion": "joy", "pred_emotion": "joy", "n_faces": 3, "n_frames": 3, "duration_s": 2.0},
        {"clip": "dia1_utt1", "true_emotion": "joy", "pred_emotion": "neutral", "n_faces": 12, "n_frames": 3, "duration_s": 2.0},
        {"clip": "dia1_utt2", "true_emotion": "anger", "pred_emotion": "anger", "n_faces": 0, "n_frames": 3, "duration_s": 2.0},
        {"clip": "dia1_utt3", "true_emotion": "anger", "pred_emotion": "sadness", "n_faces": 4, "n_frames": 3, "duration_s": 20.0},
        {"clip": "dia1_utt4", "true_emotion": "neutral", "pred_emotion": "neutral", "n_faces": 2, "n_frames": 3, "duration_s": 2.0},
    ]


def test_stratified_candidates_samples_per_emotion():
    from scripts.pick_demo_clips import stratified_candidates
    picked = stratified_candidates(_predictions(), per_emotion=1, seed=0)
    assert {p["true_emotion"] for p in picked} == {"joy", "anger", "neutral"} and len(picked) == 3


def test_hard_case_candidates_finds_many_faces_zero_faces_and_long_clips():
    from scripts.pick_demo_clips import hard_case_candidates
    result = hard_case_candidates(_predictions())
    assert result["many_faces"][0]["clip"] == "dia1_utt1"
    assert [c["clip"] for c in result["zero_faces"]] == ["dia1_utt2"]
    assert [c["clip"] for c in result["long"]] == ["dia1_utt3"]


def test_wrong_prediction_candidates_finds_mismatches_only():
    from scripts.pick_demo_clips import wrong_prediction_candidates
    assert {c["clip"] for c in wrong_prediction_candidates(_predictions())} == {"dia1_utt1", "dia1_utt3"}


def test_load_test_predictions_joins_the_batch_output_with_the_manifest(synthetic_features, tmp_path):
    from meld_emotion.training.dataset import load_ok_rows
    from scripts.pick_demo_clips import load_test_predictions
    root, _, dev_manifest = synthetic_features
    rows = load_ok_rows(dev_manifest)
    preds_path = tmp_path / "test_predictions.jsonl"
    with open(preds_path, "w") as f:
        for r in rows:
            f.write(json.dumps({"clip": f"dia{r['dialogue_id']}_utt{r['utterance_id']}", "pred": "joy",
                                "probs": [0.1, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1]}) + "\n")
    predictions = load_test_predictions(preds_path, dev_manifest)
    assert len(predictions) == len(rows) == 14
    first = predictions[0]
    assert first["pred_emotion"] == "joy" and first["true_emotion"] == rows[0]["emotion"]
    assert first["n_faces"] == rows[0]["n_faces"] and first["duration_s"] == rows[0]["duration_s"]
