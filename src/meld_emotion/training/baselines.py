"""Zero-training vision-only baseline (design doc §7): the frozen face
encoder's own 7-way softmax, averaged over every detected face in the clip.
No parameters are trained; it anchors the ablation table."""
from pathlib import Path

import numpy as np

from meld_emotion.data.labels import EMOTIONS
from meld_emotion.training.dataset import EMOTION_TO_IDX
from meld_emotion.training.metrics import compute_metrics
from meld_emotion.vision.encoders import FACE_MODEL_ID, FACE_TO_MELD_LABEL


def face_meld_label_order(model_id: str = FACE_MODEL_ID) -> tuple[str, ...]:
    """The face head's logit order in MELD spelling, from the checkpoint's
    config.json (cached by the data plan's encoder tests / cache build)."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id)
    return tuple(FACE_TO_MELD_LABEL[cfg.id2label[i]] for i in range(cfg.num_labels))


def vision_baseline_predictions(rows: list[dict], cache_dir: Path, face_labels: tuple[str, ...],
                                fallback: str = "neutral") -> tuple[np.ndarray, np.ndarray]:
    col_to_idx = np.array([EMOTION_TO_IDX[label] for label in face_labels])
    y_true, y_pred = [], []
    for row in rows:
        probs = np.load(Path(cache_dir) / row["feature_path"])["face_probs"]
        y_true.append(EMOTION_TO_IDX[row["emotion"]])
        y_pred.append(int(col_to_idx[probs.mean(0).argmax()]) if len(probs) else EMOTION_TO_IDX[fallback])
    return np.array(y_true), np.array(y_pred)


def vision_baseline_metrics(rows: list[dict], cache_dir: Path, face_labels: tuple[str, ...],
                            fallback: str = "neutral") -> dict:
    y_true, y_pred = vision_baseline_predictions(rows, cache_dir, face_labels, fallback)
    metrics = compute_metrics(y_true, y_pred, EMOTIONS)
    metrics["no_face_clips"] = int(sum(1 for r in rows if r["n_faces"] == 0))
    return metrics
