"""Turn a turn's accumulated tokens into the exact batch the training
pipeline built, and run the fusion model on it (design doc §4.3).

Used twice per turn: with `force_drop_text=True` after every sampled frame
(the provisional, vision-only state -- the fusion model's own reading of
the faces so far, which is what modality dropout trained it to do) and
with the real text at end-of-turn (the final state).
"""
import numpy as np
import torch

from meld_emotion.data.labels import EMOTIONS, SENTIMENTS
from meld_emotion.training.config import TrainConfig
from meld_emotion.training.crops import subsample_faces
from meld_emotion.training.dataset import collate, remap_track_ids


def dummy_text_encoding(tokenizer) -> dict:
    cls_id = getattr(tokenizer, "cls_token_id", None)
    return {"input_ids": [0 if cls_id is None else int(cls_id)], "attention_mask": [1]}


def build_item(config: TrainConfig, enc: dict, face_feat: np.ndarray, face_frame_idx: list[int],
               face_track_raw: list[int], scene_feat: np.ndarray, *, face_dim: int, scene_dim: int,
               clip: str) -> dict:
    face_feat = np.asarray(face_feat, dtype=np.float32).reshape(-1, face_dim)
    face_frame = np.asarray(face_frame_idx, dtype=np.int64)
    face_track = np.asarray(face_track_raw, dtype=np.int64)
    scene_feat = np.asarray(scene_feat, dtype=np.float32).reshape(-1, scene_dim)

    if config.face_trainable_layers > 0:
        # The Stage 2 batch path capped crops per clip (crops.py); replay must match it.
        keep = subsample_faces(len(face_feat), config.max_faces_per_clip)
        face_feat, face_frame, face_track = face_feat[keep], face_frame[keep], face_track[keep]
    if not config.use_faces:
        face_feat, face_frame, face_track = face_feat[:0], face_frame[:0], face_track[:0]
    if not config.use_scene:
        scene_feat = scene_feat[:0]

    return {
        "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
        "face_feat": torch.from_numpy(np.ascontiguousarray(face_feat)),
        "face_frame": torch.from_numpy(np.minimum(face_frame, config.max_frames - 1)),
        "face_track": torch.from_numpy(remap_track_ids(face_track, config.max_track_slots)),
        "scene_feat": torch.from_numpy(np.ascontiguousarray(scene_feat)),
        "scene_frame": torch.from_numpy(np.minimum(np.arange(len(scene_feat), dtype=np.int64), config.max_frames - 1)),
        "emotion": torch.tensor(0, dtype=torch.long),
        "sentiment": torch.tensor(0, dtype=torch.long),
        "clip": clip,
    }


@torch.no_grad()
def predict(model, item: dict, pad_id: int, device: str, force_drop_text: bool = False) -> tuple[dict, dict]:
    batch = collate([item], pad_id=pad_id)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    out = model(batch, force_drop_text=force_drop_text)
    emotion = torch.softmax(out["emotion_logits"].float(), dim=-1)[0].cpu().numpy()
    sentiment = torch.softmax(out["sentiment_logits"].float(), dim=-1)[0].cpu().numpy()
    return ({e: float(p) for e, p in zip(EMOTIONS, emotion)},
            {s: float(p) for s, p in zip(SENTIMENTS, sentiment)})
