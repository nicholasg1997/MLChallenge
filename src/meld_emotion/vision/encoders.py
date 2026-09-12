"""Pretrained, frozen vision encoders (design doc §4.1).

- FaceEmotionEncoder: dima806/facial_emotions_image_detection (ViT-Base,
  ~86M). Per face crop -> the 768-d post-LayerNorm CLS feature the
  checkpoint's classifier consumes, plus its own 7-way expression
  probabilities (used for the provisional state, the zero-training
  vision-only baseline, and the visual gloss -- never as classifier input).
- SceneEncoder: CLIP ViT-B/32 image tower (~88M). Per letterboxed frame ->
  the 512-d projected image embedding. The full CLIPModel (incl. the ~63M
  text tower) is loaded because the visual-gloss prompt bank needs the text
  tower later; both towers are counted in the parameter budget.

Both run under torch.no_grad and are never trained in this pipeline.
"""
from pathlib import Path

import numpy as np
import torch
from transformers import (AutoImageProcessor, AutoModelForImageClassification,
                          CLIPImageProcessor, CLIPModel)

FACE_MODEL_ID = "dima806/facial_emotions_image_detection"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_BATCH_SIZE = 64

# The checkpoint's label names -> MELD's. Same seven categories, different spelling.
FACE_TO_MELD_LABEL = {
    "sad": "sadness", "disgust": "disgust", "angry": "anger", "neutral": "neutral",
    "fear": "fear", "surprise": "surprise", "happy": "joy",
}


def default_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


class FaceEmotionEncoder:
    def __init__(self, device: str | None = None, batch_size: int = DEFAULT_BATCH_SIZE,
                 weights_path: "str | Path | None" = None):
        self.device = device or default_device()
        self.batch_size = batch_size
        self.processor = AutoImageProcessor.from_pretrained(FACE_MODEL_ID)
        self.model = AutoModelForImageClassification.from_pretrained(FACE_MODEL_ID).to(self.device).eval()
        if not (hasattr(self.model, "vit") and hasattr(self.model, "classifier")):
            raise TypeError(f"{FACE_MODEL_ID} loaded as {type(self.model).__name__}; "
                            f"expected ViTForImageClassification with .vit and .classifier")
        if weights_path is not None:
            # Stage 2 fine-tuned weights (design doc §6). The classifier head in
            # this state is the *original* head and is stale for these features.
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(ckpt["face_encoder_state"])
            self.model.to(self.device).eval()
        cfg = self.model.config
        self.labels = tuple(cfg.id2label[i] for i in range(cfg.num_labels))
        self.meld_labels = tuple(FACE_TO_MELD_LABEL[label] for label in self.labels)
        self.feature_dim = cfg.hidden_size

    @torch.no_grad()
    def encode_batch(self, images: list) -> dict:
        features, probs = [], []
        for chunk in _chunks(list(images), self.batch_size):
            inputs = self.processor(images=chunk, return_tensors="pt").to(self.device)
            hidden = self.model.vit(pixel_values=inputs["pixel_values"]).last_hidden_state
            cls = hidden[:, 0, :]  # post-LayerNorm CLS: exactly what the classifier sees
            logits = self.model.classifier(cls)
            features.append(cls.float().cpu().numpy())
            probs.append(torch.softmax(logits, dim=-1).float().cpu().numpy())
        if not features:
            return {"features": np.zeros((0, self.feature_dim), np.float32),
                    "probs": np.zeros((0, len(self.labels)), np.float32)}
        return {"features": np.concatenate(features).astype(np.float32),
                "probs": np.concatenate(probs).astype(np.float32)}

    def encode(self, image) -> dict:
        out = self.encode_batch([image])
        return {"features": out["features"][0], "probs": out["probs"][0]}


class SceneEncoder:
    def __init__(self, device: str | None = None, batch_size: int = DEFAULT_BATCH_SIZE):
        self.device = device or default_device()
        self.batch_size = batch_size
        self.processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)
        self.model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(self.device).eval()
        self.feature_dim = self.model.config.projection_dim

    @torch.no_grad()
    def encode_batch(self, images: list) -> np.ndarray:
        features = []
        for chunk in _chunks(list(images), self.batch_size):
            inputs = self.processor(images=chunk, return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            if not isinstance(image_features, torch.Tensor):
                # transformers >=5 returns a BaseModelOutputWithPooling; .pooler_output
                # here is the same visual_projection(pooled_output) a bare-tensor return used to be.
                image_features = image_features.pooler_output
            features.append(image_features.float().cpu().numpy())
        if not features:
            return np.zeros((0, self.feature_dim), np.float32)
        return np.concatenate(features).astype(np.float32)

    def encode(self, image) -> np.ndarray:
        return self.encode_batch([image])[0]
