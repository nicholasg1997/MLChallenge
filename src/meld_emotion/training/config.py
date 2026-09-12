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

    # --- Stage 2: face encoder in the loop (design doc §6). 0 = Stage 1 (cached features) ---
    face_trainable_layers: int = 0          # top N of the 12 ViT layers (+ final LayerNorm)
    lr_face: float = 1e-5
    max_faces_per_clip: int = 64            # p99 is ~95; uniform subsample keeps frame coverage
    face_augment: bool = True               # random resized crop / flip / colour jitter on train crops
    init_from: str | None = None            # Stage 1 checkpoint to initialise everything but the ViT
    loader_workers: int = 0                 # DataLoader workers for JPEG decoding (Modal: 6)
    amp_bf16: bool = True                   # bf16 autocast on CUDA only

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


def stage2_config(base: str, **overrides) -> TrainConfig:
    """Stage 2 preset derived from a Stage 1 preset: same modalities, the face
    ViT's top 4 layers trainable, initialised from that preset's seed-0 run."""
    if base not in ABLATIONS or base.startswith("text_only"):
        raise KeyError(f"Stage 2 needs a Stage 1 preset that uses faces; got {base!r}")
    defaults = dict(name=f"stage2_{base}", face_trainable_layers=4, batch_size=16, epochs=8,
                    lr_text=1e-5, init_from=f"results/{base}/seed0/best.pt")
    return replace(ABLATIONS[base], **{**defaults, **overrides})
