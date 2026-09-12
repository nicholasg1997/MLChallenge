"""Training configuration. Every ablation in design doc §7 is a preset of one
TrainConfig; the model, dataset and loop read only this object."""
from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class TrainConfig:
    name: str = "fusion"

    # --- modality / ablation switches ---
    use_text: bool = True
    use_faces: bool = True
    use_scene: bool = True
    use_track_id: bool = True
    context_k: int = 4                       # previous utterances fed to the text encoder
    mask_vision_over_seconds: float | None = None   # treat clips longer than this as vision-missing

    # --- text encoder ---
    text_model: str = "roberta-base"
    text_trainable_layers: int = 6           # top N of the 12 encoder layers (design doc §6)
    max_text_tokens: int = 256

    # --- fusion transformer (design doc §4.3) ---
    d_model: int = 768
    n_layers: int = 2
    n_heads: int = 8
    ff_dim: int = 2048
    dropout: float = 0.1
    max_frames: int = 32                     # frame-position table; sample_index is clamped
    max_track_slots: int = 16                # track-ID table; ids beyond go to one overflow slot
    modality_dropout: float = 0.15

    # --- loss ---
    class_weight_alpha: float = 0.5          # w_c ∝ (1/freq_c)^alpha, tuned on dev
    sentiment_lambda: float = 0.3
    label_smoothing: float = 0.0

    # --- optimisation ---
    lr_text: float = 2e-5
    lr_fusion: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    epochs: int = 12
    warmup_fraction: float = 0.06
    patience: int = 3                        # epochs without dev weighted-F1 improvement
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "auto"                     # "auto" -> mps if available else cpu

    def __post_init__(self):
        if not (self.use_text or self.use_faces or self.use_scene):
            raise ValueError("at least one modality must be enabled")
        if self.context_k < 0:
            raise ValueError("context_k must be >= 0")

    @property
    def use_vision(self) -> bool:
        return self.use_faces or self.use_scene

    def to_dict(self) -> dict:
        return asdict(self)


ABLATIONS: dict[str, TrainConfig] = {
    "text_only_k0": TrainConfig(name="text_only_k0", use_faces=False, use_scene=False, context_k=0),
    "text_only_k4": TrainConfig(name="text_only_k4", use_faces=False, use_scene=False),
    "vision_only": TrainConfig(name="vision_only", use_text=False),
    "fusion": TrainConfig(name="fusion"),
    "fusion_no_scene": TrainConfig(name="fusion_no_scene", use_scene=False),
    "fusion_no_context": TrainConfig(name="fusion_no_context", context_k=0),
    "fusion_no_trackid": TrainConfig(name="fusion_no_trackid", use_track_id=False),
    # Stage 1 found the scene token and track-ID embedding each slightly negative at one seed;
    # this preset tests whether the two removals stack.
    "fusion_faces_only": TrainConfig(name="fusion_faces_only", use_scene=False, use_track_id=False),
}


def config_for(name: str, **overrides) -> TrainConfig:
    if name not in ABLATIONS:
        raise KeyError(f"unknown ablation {name!r}; choose from {sorted(ABLATIONS)}")
    return replace(ABLATIONS[name], **overrides)
