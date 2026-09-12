import numpy as np

from meld_emotion.data.labels import EMOTIONS


def test_vision_baseline_averages_face_probs_and_falls_back_when_no_faces(synthetic_features):
    from meld_emotion.training.baselines import vision_baseline_predictions
    from meld_emotion.training.dataset import EMOTION_TO_IDX, load_ok_rows
    root, train, _ = synthetic_features
    rows = load_ok_rows(train)[:6]
    # Overwrite clip 1's probs so the mean over its faces is unambiguous.
    row = rows[1]
    z = dict(np.load(root / row["feature_path"]))
    probs = np.zeros_like(z["face_probs"]); probs[:, 3] = 1.0        # face-head column 3
    z["face_probs"] = probs
    np.savez(root / row["feature_path"], **z)
    face_labels = ("sadness", "disgust", "anger", "neutral", "fear", "surprise", "joy")   # column 3 = neutral
    face_labels = tuple(face_labels[i] if i != 3 else "surprise" for i in range(7))      # remap col 3 -> surprise
    y_true, y_pred = vision_baseline_predictions(rows, root, face_labels, fallback="neutral")
    assert y_true.tolist() == [EMOTION_TO_IDX[r["emotion"]] for r in rows]
    assert y_pred[1] == EMOTION_TO_IDX["surprise"]
    assert rows[0]["n_faces"] == 0 and y_pred[0] == EMOTION_TO_IDX["neutral"]


def test_vision_baseline_metrics_reports_no_face_count(synthetic_features):
    from meld_emotion.training.baselines import vision_baseline_metrics
    from meld_emotion.training.dataset import load_ok_rows
    root, train, _ = synthetic_features
    rows = load_ok_rows(train)
    face_labels = ("sadness", "disgust", "anger", "neutral", "fear", "surprise", "joy")
    m = vision_baseline_metrics(rows, root, face_labels)
    assert m["n"] == 28 and m["no_face_clips"] == sum(1 for r in rows if r["n_faces"] == 0)
    assert set(m["per_class_f1"]) == set(EMOTIONS)
