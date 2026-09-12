"""The face ViT inside the training loop (design doc §6, Stage 2): frozen
bottom layers run under no_grad, only the top `trainable_layers` (+ the
final LayerNorm) keep activations and gradients. Output = the post-LayerNorm
CLS feature, exactly what Stage 1 cached."""
import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification

from meld_emotion.vision.encoders import FACE_MODEL_ID

FACE_MEAN, FACE_STD = 0.5, 0.5   # the checkpoint's ViTImageProcessor: rescale 1/255, then (x - 0.5) / 0.5


def normalize_pixels(x_uint8: torch.Tensor) -> torch.Tensor:
    return (x_uint8.float() / 255.0 - FACE_MEAN) / FACE_STD


def _run_layer(layer: nn.Module, h: torch.Tensor) -> torch.Tensor:
    out = layer(h)
    return out[0] if isinstance(out, tuple) else out


class TrainableFaceEncoder(nn.Module):
    def __init__(self, model_id: str = FACE_MODEL_ID, trainable_layers: int = 4):
        super().__init__()
        model = AutoModelForImageClassification.from_pretrained(model_id, attn_implementation="eager")
        self.model = model                      # kept whole so export_state() matches FaceEmotionEncoder.model
        self.vit = model.vit
        self.feature_dim = model.config.hidden_size
        n_layers = len(self.vit.layers)
        self.n_frozen = n_layers - trainable_layers
        for p in model.parameters():
            p.requires_grad = False
        for layer in self.vit.layers[self.n_frozen:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.vit.layernorm.parameters():
            p.requires_grad = True

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def export_state(self) -> dict:
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

    def forward(self, pixels_uint8: torch.Tensor) -> torch.Tensor:
        if pixels_uint8.shape[0] == 0:
            return torch.zeros((0, self.feature_dim), device=pixels_uint8.device)
        x = normalize_pixels(pixels_uint8)
        with torch.no_grad():
            h = self.vit.embeddings(x)
            for layer in self.vit.layers[:self.n_frozen]:
                h = _run_layer(layer, h)
        for layer in self.vit.layers[self.n_frozen:]:
            h = _run_layer(layer, h)
        return self.vit.layernorm(h)[:, 0]


def scatter_faces(feats: torch.Tensor, batch_idx: torch.Tensor, slot: torch.Tensor, B: int, fmax: int) -> torch.Tensor:
    """(Ntot, D) crop features -> (B, fmax, D) with zeros in unused slots."""
    out = torch.zeros((B, fmax, feats.shape[1]), dtype=feats.dtype, device=feats.device)
    if feats.shape[0]:
        out[batch_idx, slot] = feats
    return out
