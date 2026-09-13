import numpy as np
import pytest
import torch

from conftest import StubTextEncoder
from meld_emotion.inference.events import EventEmitter


class _StubFaceDetector:
    pass


class _StubFaceEncoder:
    feature_dim = 8

    def encode_batch(self, images):
        n = len(images)
        feats = np.stack([np.full(8, 0.1 * (i + 1), np.float32) for i in range(n)]) if n else np.zeros((0, 8), np.float32)
        return {"features": feats, "probs": np.zeros((n, 7), dtype=np.float32)}


class _StubSceneEncoder:
    feature_dim = 4

    def encode(self, image):
        return np.full(4, 0.5, dtype=np.float32)


class _StubTokenizer:
    pad_token_id = 1
    cls_token_id = 0

    def __call__(self, *args, **kwargs):
        raise AssertionError("turn.py must go through encode_text, not call the tokenizer directly")


def _bundle(use_faces=True, use_scene=True, stage2=False):
    from meld_emotion.inference.loader import InferenceBundle
    from meld_emotion.training.config import TrainConfig
    from meld_emotion.training.model import FusionModel
    config = TrainConfig(d_model=32, n_heads=4, ff_dim=64, n_layers=1, use_faces=use_faces, use_scene=use_scene,
                         context_k=2, face_trainable_layers=4 if stage2 else 0)
    model = FusionModel(config, StubTextEncoder(32), face_dim=8, scene_dim=4).eval()
    return InferenceBundle(model=model, config=config, tokenizer=_StubTokenizer(), face_detector=_StubFaceDetector(),
                           face_encoder=_StubFaceEncoder(), scene_encoder=_StubSceneEncoder(), device="cpu",
                           face_dim=8, scene_dim=4, checkpoint_path=None, stage=2 if stage2 else 1)


def _fake_frame(color=(50, 50, 50)):
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:] = color
    return frame


class _Collector(EventEmitter):
    """A real EventEmitter whose `emit` records events in a list instead of
    writing JSON lines to a stream -- so it still exposes the typed
    `.provisional()`/`.final()`/`.token()`/`.done()` helpers TurnProcessor
    calls, exactly like the real out=file-handle EventEmitter would."""
    def __init__(self):
        super().__init__(out=None)
        self.events = []

    def emit(self, event):
        self.events.append(event)


def test_push_frame_emits_a_provisional_event_from_the_masked_fusion_model(monkeypatch):
    from meld_emotion.inference import turn as turn_module
    from meld_emotion.inference.predict import build_item, dummy_text_encoding, predict
    monkeypatch.setattr(turn_module, "detect_faces", lambda detector, frame: [(0, 0, 10, 10, 0.9)])
    bundle, collector = _bundle(), _Collector()
    tp = turn_module.TurnProcessor(bundle, emitter=collector)
    tp.start_turn("dia1_utt1")
    event = tp.push_frame(_fake_frame())
    assert event["phase"] == "provisional" and event["turn_id"] == "dia1_utt1" and event["frame"] == 0
    assert event["faces_seen"] == 1 and collector.events[-1] == event
    # the provisional state IS the fusion model with text masked over the tokens so far
    item = build_item(bundle.config, dummy_text_encoding(bundle.tokenizer), np.full((1, 8), 0.1, np.float32), [0], [0],
                      np.full((1, 4), 0.5, np.float32), face_dim=8, scene_dim=4, clip="dia1_utt1")
    expected, _ = predict(bundle.model, item, pad_id=1, device="cpu", force_drop_text=True)
    assert event["provisional_expression"] == pytest.approx(expected, abs=1e-6)
    assert tp.last_provisional == event["provisional_expression"]


def test_shot_cut_between_sampled_frames_resets_the_tracker_before_detection(monkeypatch):
    from meld_emotion.inference import turn as turn_module
    order = []
    monkeypatch.setattr(turn_module, "detect_faces", lambda detector, frame: order.append("detect") or [(0, 0, 10, 10, 0.9)])

    class _Tracker:
        def reset(self): order.append("reset")
        def update(self, boxes): order.append("update"); return [0] * len(boxes)
    monkeypatch.setattr(turn_module, "FaceTracker", lambda: _Tracker())
    monkeypatch.setattr(turn_module, "shot_change_score", lambda a, b: 0.9)  # every transition is "a cut"

    tp = turn_module.TurnProcessor(_bundle(), emitter=_Collector())
    tp.start_turn("dia1_utt1")
    tp.push_frame(_fake_frame((10, 10, 10)))
    tp.push_frame(_fake_frame((200, 200, 200)))
    assert order == ["reset", "detect", "update", "reset", "detect", "update"]   # start_turn reset; cut reset before detect


def test_end_turn_builds_the_training_shaped_batch_and_emits_final_without_response(monkeypatch):
    from meld_emotion.inference import turn as turn_module
    monkeypatch.setattr(turn_module, "detect_faces", lambda detector, frame: [(0, 0, 10, 10, 0.9)])
    monkeypatch.setattr(turn_module, "encode_text",
                        lambda tok, ctx, cur, max_len: {"input_ids": [0, 5, 6, 1], "attention_mask": [1, 1, 1, 1]})
    collector = _Collector()
    tp = turn_module.TurnProcessor(_bundle(), emitter=collector)
    tp.start_turn("dia1_utt1")
    tp.push_frame(_fake_frame())
    tp.push_frame(_fake_frame())
    final = tp.end_turn("You did WHAT?", ["earlier line one", "earlier line two"])
    assert final["phase"] == "final" and "response" not in final and "latency_ms" not in final
    assert final["text"] == "You did WHAT?" and final["faces_seen"] == 1
    assert set(final["emotion_probs"]) == {"neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"}
    assert final["emotion"] in final["emotion_probs"] and final["sentiment"] in final["sentiment_probs"]
    assert collector.events[-1] == final


def test_end_turn_carries_the_caller_supplied_visual_cues_and_defaults_to_empty(monkeypatch):
    from meld_emotion.inference import turn as turn_module
    monkeypatch.setattr(turn_module, "detect_faces", lambda detector, frame: [])
    monkeypatch.setattr(turn_module, "encode_text",
                        lambda tok, ctx, cur, max_len: {"input_ids": [0, 1], "attention_mask": [1, 1]})
    tp = turn_module.TurnProcessor(_bundle(), emitter=_Collector())
    tp.start_turn("dia1_utt1")
    assert tp.end_turn("hello", [], visual_cues=["two people", "no faces visible"])["visual_cues"] == ["two people", "no faces visible"]
    tp.start_turn("dia1_utt2")
    assert tp.end_turn("hello again", [])["visual_cues"] == []


def test_end_turn_with_zero_frames_still_produces_finite_predictions(monkeypatch):
    from meld_emotion.inference import turn as turn_module
    monkeypatch.setattr(turn_module, "encode_text",
                        lambda tok, ctx, cur, max_len: {"input_ids": [0, 1], "attention_mask": [1, 1]})
    tp = turn_module.TurnProcessor(_bundle(), emitter=_Collector())
    tp.start_turn("dia1_utt1")
    final = tp.end_turn("hello", [])
    assert final["faces_seen"] == 0 and tp.last_provisional is None
    assert all(np.isfinite(v) for v in final["emotion_probs"].values())
    assert np.allclose(tp.mean_scene_embedding, 0.0) and tp.mean_scene_embedding.shape == (4,)


def test_state_accumulates_across_the_turn_and_resets_on_start_turn(monkeypatch):
    from meld_emotion.inference import turn as turn_module
    monkeypatch.setattr(turn_module, "detect_faces", lambda detector, frame: [(0, 0, 10, 10, 0.9), (20, 20, 10, 10, 0.9)])
    tp = turn_module.TurnProcessor(_bundle(), emitter=_Collector())
    tp.start_turn("dia1_utt1")
    tp.push_frame(_fake_frame())
    tp.push_frame(_fake_frame())
    assert tp.max_faces_seen == 2 and np.allclose(tp.mean_scene_embedding, 0.5)
    assert len(tp.last_boxes) == 2 and len(tp.last_track_ids) == 2
    tp.start_turn("dia1_utt2")
    assert tp.max_faces_seen == 0 and tp.last_provisional is None and tp.last_boxes == []
