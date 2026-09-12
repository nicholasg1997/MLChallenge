"""Single-stream fusion model (design doc §4.3).

Token sequence per sample:  [FUSE] + text tokens + face tokens + scene tokens
Each token = projected feature + modality-type embedding (+ frame-position
embedding for visual tokens, + track-ID embedding for face tokens). One
pre-LN transformer encoder attends over the whole set -- every text token
sees every visual token and vice versa in every layer. Both heads read the
[FUSE] output. Padding and dropped modalities are excluded via the key
padding mask; [FUSE] is never masked, so no sample is ever all-masked.
"""
import torch
import torch.nn as nn

from meld_emotion.training.config import TrainConfig

TYPE_FUSE, TYPE_TEXT, TYPE_FACE, TYPE_SCENE = 0, 1, 2, 3


class FusionModel(nn.Module):
    def __init__(self, config: TrainConfig, text_encoder: nn.Module | None,
                 face_dim: int, scene_dim: int, n_emotions: int = 7, n_sentiments: int = 3):
        super().__init__()
        self.config = config
        d = config.d_model
        if config.use_text:
            if text_encoder is None:
                raise ValueError("use_text=True requires a text encoder")
            self.text_encoder = text_encoder
            self.text_proj = nn.Linear(text_encoder.hidden_size, d) if text_encoder.hidden_size != d else nn.Identity()
        else:
            self.text_encoder = None
            self.text_proj = None
        self.face_proj = nn.Linear(face_dim, d) if config.use_faces else None
        self.scene_proj = nn.Linear(scene_dim, d) if config.use_scene else None

        self.fuse_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.fuse_token, std=0.02)
        self.type_emb = nn.Embedding(4, d)
        self.frame_emb = nn.Embedding(config.max_frames, d)
        self.track_emb = nn.Embedding(config.max_track_slots + 1, d)
        self.input_norm = nn.LayerNorm(d)
        self.input_dropout = nn.Dropout(config.dropout)
        layer = nn.TransformerEncoderLayer(d, config.n_heads, config.ff_dim, config.dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, config.n_layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(d)
        self.emotion_head = nn.Linear(d, n_emotions)
        self.sentiment_head = nn.Linear(d, n_sentiments)

    # --- parameter groups for the two learning rates ---
    def text_parameters(self):
        return self.text_encoder.parameters() if self.text_encoder is not None else iter(())

    def fusion_parameters(self):
        text_ids = {id(p) for p in self.text_parameters()}
        return (p for p in self.parameters() if id(p) not in text_ids)

    # --- modality dropout (training only) ---
    def modality_dropout_masks(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sample: drop all text (p) or all vision (p), never both, and
        never a modality the sample doesn't have or the other side lacks."""
        cfg = self.config
        B = batch["input_ids"].shape[0]
        device = batch["input_ids"].device
        has_text = batch["attention_mask"].bool().any(1) if cfg.use_text else torch.zeros(B, dtype=torch.bool, device=device)
        has_vision = torch.zeros(B, dtype=torch.bool, device=device)
        if cfg.use_faces:
            has_vision |= batch["face_mask"].any(1)
        if cfg.use_scene:
            has_vision |= batch["scene_mask"].any(1)
        if not self.training or cfg.modality_dropout <= 0:
            return torch.zeros(B, dtype=torch.bool, device=device), torch.zeros(B, dtype=torch.bool, device=device)
        r = torch.rand(B, device=device)
        p = cfg.modality_dropout
        both = has_text & has_vision
        drop_text = (r < p) & both
        drop_vision = (r >= p) & (r < 2 * p) & both
        return drop_text, drop_vision

    def forward(self, batch: dict, force_drop_text: bool = False, force_drop_vision: bool = False) -> dict:
        cfg = self.config
        B = batch["input_ids"].shape[0]
        device = batch["input_ids"].device
        drop_text, drop_vision = self.modality_dropout_masks(batch)
        if force_drop_text:
            drop_text = torch.ones(B, dtype=torch.bool, device=device)
        if force_drop_vision:
            drop_vision = torch.ones(B, dtype=torch.bool, device=device)

        tokens = [self.fuse_token.expand(B, 1, -1) + self.type_emb.weight[TYPE_FUSE]]
        valid = [torch.ones(B, 1, dtype=torch.bool, device=device)]

        if cfg.use_text:
            h = self.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
            tokens.append(self.text_proj(h) + self.type_emb.weight[TYPE_TEXT])
            valid.append(batch["attention_mask"].bool() & ~drop_text[:, None])
        if cfg.use_faces:
            f = self.face_proj(batch["face_feat"]) + self.type_emb.weight[TYPE_FACE] + self.frame_emb(batch["face_frame"])
            if cfg.use_track_id:
                f = f + self.track_emb(batch["face_track"])
            tokens.append(f)
            valid.append(batch["face_mask"] & ~drop_vision[:, None])
        if cfg.use_scene:
            s = self.scene_proj(batch["scene_feat"]) + self.type_emb.weight[TYPE_SCENE] + self.frame_emb(batch["scene_frame"])
            tokens.append(s)
            valid.append(batch["scene_mask"] & ~drop_vision[:, None])

        x = self.input_dropout(self.input_norm(torch.cat(tokens, dim=1)))
        key_padding_mask = ~torch.cat(valid, dim=1)          # True = ignore
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        fused = self.final_norm(h[:, 0])
        return {"emotion_logits": self.emotion_head(fused), "sentiment_logits": self.sentiment_head(fused)}
