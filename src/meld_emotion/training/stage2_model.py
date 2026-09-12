"""Stage 2 = the face encoder in front of the unchanged Stage 1 FusionModel:
crops -> CLS features -> scattered into the (B, Fmax, 768) face_feat slot."""
from pathlib import Path

import torch
import torch.nn as nn

from meld_emotion.training.face_encoder import scatter_faces
from meld_emotion.training.model import FusionModel
from meld_emotion.training.train import load_checkpoint


class Stage2Model(nn.Module):
    def __init__(self, fusion: FusionModel, face_encoder: nn.Module):
        super().__init__()
        self.fusion = fusion
        self.face_encoder = face_encoder
        self.config = fusion.config

    # parameter groups for the three learning rates
    def text_parameters(self):
        return self.fusion.text_parameters()

    def fusion_parameters(self):
        return self.fusion.fusion_parameters()

    def face_parameters(self):
        return self.face_encoder.parameters()

    def modality_dropout_masks(self, batch: dict):
        return self.fusion.modality_dropout_masks(batch)

    def forward(self, batch: dict, force_drop_text: bool = False, force_drop_vision: bool = False) -> dict:
        B, fmax = batch["face_mask"].shape
        feats = self.face_encoder(batch["face_pixels"])
        batch = {**batch, "face_feat": scatter_faces(feats, batch["face_batch_idx"], batch["face_slot"], B, fmax)}
        return self.fusion(batch, force_drop_text=force_drop_text, force_drop_vision=force_drop_vision)


def init_fusion_from_stage1(fusion: FusionModel, path: Path) -> dict:
    """Load a Stage 1 checkpoint (text encoder incl. its fine-tuned layers,
    projectors, fusion transformer, heads) into a fresh FusionModel."""
    return load_checkpoint(Path(path), fusion)
