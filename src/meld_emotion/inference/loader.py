"""Load a checkpoint into an inference-ready bundle: the fused model plus
the tokenizer and vision encoders it needs at serving time (design doc
§4.1). A Stage 2 checkpoint (train_stage2.py) also carries the fine-tuned
face-ViT weights, which are loaded into FaceEmotionEncoder so the demo's
face features match the ones the classifier was trained on. The model is
built from the checkpoint's own config/face_dim/scene_dim before its
weights are loaded."""
from dataclasses import dataclass
from pathlib import Path

import torch

from meld_emotion.training.config import TrainConfig
from meld_emotion.training.model import FusionModel
from meld_emotion.training.text import build_text_encoder, build_tokenizer
from meld_emotion.training.train import resolve_device
from meld_emotion.vision.encoders import FaceEmotionEncoder, SceneEncoder
from meld_emotion.vision.face_detector import build_face_detector


@dataclass
class InferenceBundle:
    model: FusionModel
    config: TrainConfig
    tokenizer: object | None
    face_detector: object
    face_encoder: object
    scene_encoder: object
    device: str
    face_dim: int
    scene_dim: int
    checkpoint_path: Path
    stage: int


def load_inference_bundle(checkpoint_path: Path, device: str = "auto", *,
                          text_encoder_factory=None, tokenizer_factory=None,
                          face_encoder_factory=None, scene_encoder_factory=None,
                          face_detector_factory=None) -> InferenceBundle:
    checkpoint_path = Path(checkpoint_path)
    device = resolve_device(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = TrainConfig(**ckpt["config"])
    stage = int(ckpt.get("stage", 1))

    text_encoder = tokenizer = None
    if config.use_text:
        build_text = text_encoder_factory or (
            lambda: build_text_encoder(config.text_model, config.text_trainable_layers))
        build_tok = tokenizer_factory or (lambda: build_tokenizer(config.text_model))
        text_encoder, tokenizer = build_text(), build_tok()

    model = FusionModel(config, text_encoder, face_dim=ckpt["face_dim"], scene_dim=ckpt["scene_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    weights_path = checkpoint_path if "face_encoder_state" in ckpt else None
    build_face_enc = face_encoder_factory or (lambda: FaceEmotionEncoder(device=device, weights_path=weights_path))
    build_scene_enc = scene_encoder_factory or (lambda: SceneEncoder(device=device))
    build_detector = face_detector_factory or build_face_detector

    return InferenceBundle(model=model, config=config, tokenizer=tokenizer,
                           face_detector=build_detector(), face_encoder=build_face_enc(),
                           scene_encoder=build_scene_enc(), device=device,
                           face_dim=int(ckpt["face_dim"]), scene_dim=int(ckpt["scene_dim"]),
                           checkpoint_path=checkpoint_path, stage=stage)
