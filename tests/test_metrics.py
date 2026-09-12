import numpy as np
import pytest
import torch


def test_class_weights_follow_inverse_frequency_to_alpha_and_average_to_one():
    from meld_emotion.training.metrics import class_weights
    labels = np.array([0] * 8 + [1] * 2)                       # 80% / 20%
    w = class_weights(labels, n_classes=2, alpha=1.0)
    assert w[1] / w[0] == pytest.approx(4.0)
    assert w.mean().item() == pytest.approx(1.0)
    w_half = class_weights(labels, n_classes=2, alpha=0.5)
    assert w_half[1] / w_half[0] == pytest.approx(2.0)
    assert torch.equal(class_weights(labels, n_classes=2, alpha=0.0), torch.ones(2))


def test_class_weights_handle_an_absent_class():
    from meld_emotion.training.metrics import class_weights
    w = class_weights(np.array([0, 0, 1]), n_classes=3, alpha=1.0)
    assert torch.isfinite(w).all() and w.shape == (3,)


def test_joint_loss_combines_emotion_and_lambda_weighted_sentiment():
    from meld_emotion.training.metrics import joint_loss
    out = {"emotion_logits": torch.zeros(4, 7), "sentiment_logits": torch.zeros(4, 3)}
    batch = {"emotion": torch.tensor([0, 1, 2, 3]), "sentiment": torch.tensor([0, 1, 2, 0])}
    total, parts = joint_loss(out, batch, emotion_weight=None, sentiment_lambda=0.5, label_smoothing=0.0)
    assert parts["emotion"] == pytest.approx(np.log(7), abs=1e-5)
    assert parts["sentiment"] == pytest.approx(np.log(3), abs=1e-5)
    assert total.item() == pytest.approx(np.log(7) + 0.5 * np.log(3), abs=1e-5)


def test_compute_metrics_on_a_known_confusion():
    from meld_emotion.training.metrics import compute_metrics
    labels = ("a", "b", "c")
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 0, 2, 1])
    m = compute_metrics(y_true, y_pred, labels)
    assert m["n"] == 6 and m["accuracy"] == pytest.approx(4 / 6)
    assert m["confusion"] == [[2, 0, 0], [1, 1, 0], [0, 1, 1]]
    assert m["per_class_f1"]["a"] == pytest.approx(2 * 2 / (2 * 2 + 1))     # tp=2 fp=1 fn=0
    assert 0 < m["macro_f1"] < 1 and 0 < m["weighted_f1"] < 1
    assert set(m["per_class_f1"]) == set(labels)


def test_compute_metrics_keeps_every_label_even_if_never_predicted():
    from meld_emotion.training.metrics import compute_metrics
    m = compute_metrics(np.array([0, 0]), np.array([0, 0]), ("a", "b"))
    assert m["per_class_f1"]["b"] == 0.0 and m["confusion"] == [[2, 0], [0, 0]]
