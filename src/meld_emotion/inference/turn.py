"""One inference core (design doc §2, §4.2, §4.3). push_frame runs the same
per-frame vision logic preprocess_clip uses -- shot-cut check against the
previous SAMPLED frame, then face detect/track/encode, then scene encode
for the gloss -- and publishes a provisional state: the fusion model's own
vision-only prediction over the face tokens so far (text masked). end_turn
adds the real text and runs the model for the final event, which never
carries a response: the LM is a separate, slower step so it never gates
the state (design doc §2)."""
from typing import Optional

import numpy as np
from PIL import Image

from meld_emotion.data.preprocess import crop_face, letterbox
from meld_emotion.inference.loader import InferenceBundle
from meld_emotion.inference.predict import build_item, dummy_text_encoding, predict
from meld_emotion.training.text import encode_text, format_context
from meld_emotion.vision.face_detector import detect_faces
from meld_emotion.vision.tracker import FaceTracker, SHOT_CUT_THRESHOLD, shot_change_score


class TurnProcessor:
    def __init__(self, bundle: InferenceBundle, emitter):
        self.bundle = bundle
        self.emitter = emitter
        self.turn_id: Optional[str] = None
        self._tracker = FaceTracker()
        self._pad_id = bundle.tokenizer.pad_token_id if bundle.tokenizer is not None else 0
        self._reset_state()

    def start_turn(self, turn_id: str) -> None:
        self.turn_id = turn_id
        self._tracker.reset()
        self._reset_state()

    def _reset_state(self) -> None:
        self._prev_frame = None
        self._face_features: list[np.ndarray] = []
        self._face_frame_idx: list[int] = []
        self._face_track_raw: list[int] = []
        self._scene_features: list[np.ndarray] = []
        self.last_provisional: Optional[dict] = None
        self.max_faces_seen = 0
        self.last_boxes: list[tuple] = []
        self.last_track_ids: list[int] = []
        self.last_crops: list[Image.Image] = []     # the face crops of the last sampled frame (PIL RGB)

    @property
    def mean_scene_embedding(self) -> np.ndarray:
        if not self._scene_features:
            return np.zeros(self.bundle.scene_dim, dtype=np.float32)
        return np.mean(self._scene_features, axis=0).astype(np.float32)

    def _item(self, enc: dict) -> dict:
        face_feat = np.concatenate(self._face_features) if self._face_features \
            else np.zeros((0, self.bundle.face_dim), np.float32)
        scene_feat = np.stack(self._scene_features) if self._scene_features \
            else np.zeros((0, self.bundle.scene_dim), np.float32)
        return build_item(self.bundle.config, enc, face_feat, self._face_frame_idx, self._face_track_raw, scene_feat,
                          face_dim=self.bundle.face_dim, scene_dim=self.bundle.scene_dim, clip=self.turn_id)

    def push_frame(self, frame_bgr: np.ndarray) -> dict:
        sample_i = len(self._scene_features)
        # Shot-cut check against the previous SAMPLED frame, before detection (as preprocess_clip does).
        if self._prev_frame is not None and shot_change_score(self._prev_frame, frame_bgr) >= SHOT_CUT_THRESHOLD:
            self._tracker.reset()
        self._prev_frame = frame_bgr

        boxes = detect_faces(self.bundle.face_detector, frame_bgr)
        track_ids = self._tracker.update(boxes)
        self.last_boxes, self.last_track_ids = boxes, track_ids
        crops, kept_tracks = [], []
        for (x, y, w, h, score), track_id in zip(boxes, track_ids):
            crop = crop_face(frame_bgr, (x, y, w, h))
            if crop is not None:
                crops.append(Image.fromarray(crop[:, :, ::-1]))   # OpenCV BGR -> PIL RGB
                kept_tracks.append(track_id)
        self.last_crops = crops
        enc = self.bundle.face_encoder.encode_batch(crops)
        if len(kept_tracks):
            self._face_features.append(np.asarray(enc["features"], np.float32))
            self._face_frame_idx += [sample_i] * len(kept_tracks)
            self._face_track_raw += kept_tracks
        self.max_faces_seen = max(self.max_faces_seen, len(kept_tracks))

        self._scene_features.append(np.asarray(self.bundle.scene_encoder.encode(
            Image.fromarray(letterbox(frame_bgr)[:, :, ::-1])), np.float32))

        provisional, _ = predict(self.bundle.model, self._item(dummy_text_encoding(self.bundle.tokenizer)),
                                 self._pad_id, self.bundle.device, force_drop_text=True)
        self.last_provisional = provisional
        return self.emitter.provisional(self.turn_id, sample_i, len(kept_tracks), provisional)

    def end_turn(self, text: str, context_prev: list[str], visual_cues: list[str] | None = None) -> dict:
        cfg = self.bundle.config
        enc = encode_text(self.bundle.tokenizer, format_context(context_prev, cfg.context_k), text, cfg.max_text_tokens)
        emotion_probs, sentiment_probs = predict(self.bundle.model, self._item(enc), self._pad_id, self.bundle.device)
        return self.emitter.final(self.turn_id, text, max(emotion_probs, key=emotion_probs.get), emotion_probs,
                                  max(sentiment_probs, key=sentiment_probs.get), sentiment_probs,
                                  self.max_faces_seen, list(visual_cues or []))
