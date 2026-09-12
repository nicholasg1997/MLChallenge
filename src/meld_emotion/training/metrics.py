"""Loss and metrics (design doc §4.3): class-weighted emotion CE + λ·sentiment
CE; weighted-F1 primary, macro-F1 secondary, per-class F1, confusion matrix."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score


def class_weights(labels: np.ndarray, n_classes: int, alpha: float) -> torch.Tensor:
    """w_c ∝ (1/freq_c)^alpha, rescaled to mean 1 so the loss scale doesn't
    depend on alpha. alpha=0 -> uniform; alpha=1 -> full inverse frequency."""
    counts = np.bincount(np.asarray(labels), minlength=n_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)                  # absent class: treat as one example
    freq = counts / counts.sum()
    w = (1.0 / freq) ** alpha
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def joint_loss(out: dict, batch: dict, emotion_weight: torch.Tensor | None,
               sentiment_lambda: float, label_smoothing: float) -> tuple[torch.Tensor, dict[str, float]]:
    emotion = F.cross_entropy(out["emotion_logits"], batch["emotion"],
                              weight=emotion_weight, label_smoothing=label_smoothing)
    sentiment = F.cross_entropy(out["sentiment_logits"], batch["sentiment"])
    total = emotion + sentiment_lambda * sentiment
    return total, {"emotion": float(emotion.item()), "sentiment": float(sentiment.item())}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: tuple[str, ...]) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    idx = list(range(len(labels)))
    per_class = f1_score(y_true, y_pred, labels=idx, average=None, zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "weighted_f1": float(f1_score(y_true, y_pred, labels=idx, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=idx, average="macro", zero_division=0)),
        "per_class_f1": {label: float(f) for label, f in zip(labels, per_class)},
        "confusion": confusion_matrix(y_true, y_pred, labels=idx).tolist(),
    }
