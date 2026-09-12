# Stage 1 Training & Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the text + vision fusion classifier (emotion + sentiment heads)
on the cached MELD features locally on MPS, evaluate it against dev/test with
the metrics the design doc commits to, and produce the full Stage 1 ablation
table — the submittable core of the project.

**Architecture:** A `meld_emotion.training` package. A `Dataset` reads the
per-split `manifest.jsonl` + per-clip `.npz` produced by the data plan and
yields text token ids (with dialogue context) plus variable-length face and
scene token sets. A single-stream `FusionModel` projects everything into one
d=768 token sequence — `[FUSE] + text tokens + face tokens + scene tokens`
with modality/frame/track embeddings — runs a small pre-LN transformer
encoder over it, and reads two linear heads off `[FUSE]`. RoBERTa-base is
fine-tuned (top half) inside the loop; the vision encoders are frozen and
already cached. A single `train()` function is configured by a
`TrainConfig`; every ablation is a preset of that config. The same
`train()` runs on MPS (smoke tests, fallback) and on a Modal A10G (the real
runs, ~1 min/epoch, all ablations in parallel) — only `device` differs.
Everything is unit-tested offline with stub text encoders and synthetic
feature files; only the RoBERTa-specific tests need the network.

**Tech Stack:** Python 3.11, uv, pytest, PyTorch (MPS), Hugging Face
`transformers` (RoBERTa-base), NumPy, scikit-learn (F1 / confusion matrix).

## Global Constraints

- Inference runs on-device (design doc §2). Training location is
  unconstrained: Stage 1 runs on a Modal A10G (Task 9) with the 16GB M1 /
  MPS as the smoke-test and fallback path; the code is device-agnostic
  (`device="auto"` → cuda, else mps, else cpu). `PYTORCH_ENABLE_MPS_FALLBACK=1`
  is set by the CLIs before torch is imported; fp32 throughout; DataLoader
  `num_workers=0`.
  Known MPS gap (verified): Hugging Face's fused SDPA attention raises
  "does not support dropout" in train mode once the lower RoBERTa layers are
  frozen, so the text encoder is loaded with `attn_implementation="eager"`.
- Parameter budget: ≤6B on the inference path (design doc §4.1). This plan
  adds RoBERTa-base (~125M), the fusion transformer (~11M at 2 layers,
  ~22M at 4), projectors (~1M) and heads (<0.1M). Running total with the
  data plan's vision encoders: ~375M before the response LM. Every
  `results.json` records total and trainable parameter counts.
- Metrics (design doc §4.3, §7): **weighted-F1 primary**, macro-F1
  secondary, per-class F1 and a confusion matrix; sentiment accuracy/F1
  alongside. Dev selects checkpoints and hyperparameters; **test is
  evaluated once per reported configuration**.
- Label order is `meld_emotion.data.labels.EMOTIONS` =
  `("neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear")` and
  `SENTIMENTS` = `("neutral", "positive", "negative")`, everywhere.
- Dialogue context (design doc §4.3): previous *k* utterances, oldest
  truncated first, formatted `<s> u[t-k] </s> … </s> u[t-1] </s></s> u[t] </s>`,
  256 tokens max, *k*=4 default. Speaker names are never used.
- Modality dropout (design doc §4.3): p=0.15 drop all visual tokens, p=0.15
  drop all text tokens, never both, never a modality the sample lacks.
- Frame-position table has 32 slots (`sample_index` clamped); track-ID
  table has 16 slots + 1 overflow, ids remapped per clip by first
  appearance. (Measured on dev: `sample_index` reaches 44, raw track ids
  reach 55.)
- Training must finish in well under 24 hours per run; a Stage 1 run is
  expected to take ~30 minutes. Seeds are fixed and recorded.
- Dependencies via `uv add` only; `uv run pytest` must pass offline in
  seconds; tests needing Hugging Face downloads carry `@pytest.mark.network`.
- Inputs this plan consumes, exactly as the data plan writes them:
  `data/meld/features/<split>/manifest.jsonl` rows with keys `split,
  dialogue_id, utterance_id, speaker, text, context_prev, emotion, sentiment,
  status, feature_path, duration_s, n_frames, n_faces, n_shot_cuts`; and
  `data/meld/features/<feature_path>` `.npz` files with arrays
  `face_features (N,768) f32, face_probs (N,7) f32, face_track_ids (N,)
  i64, face_frame_idx (N,) i64, scene_features (M,512) f32,
  scene_frame_idx (M,) i64, shot_cut (M,) bool, dialogue_id, utterance_id`.

---

## File Structure

```
pyproject.toml                       # MODIFIED (Task 5): scikit-learn
src/meld_emotion/training/
  __init__.py
  config.py                          # TrainConfig + ABLATIONS presets (Task 1)
  text.py                            # context formatting, tokenizer, RoBERTa partial freeze (Task 2)
  dataset.py                         # manifest+npz Dataset, track remap, collate (Task 3)
  model.py                           # FusionModel: projectors, embeddings, encoder, heads, modality dropout (Task 4)
  metrics.py                         # class weights, joint loss, F1/confusion metrics (Task 5)
  baselines.py                       # zero-training vision-only baseline (Task 6)
  train.py                           # optimiser/scheduler, evaluate(), train() with early stopping + checkpoints (Task 7)
  results.py                         # aggregate results.json files into the ablation table (Task 8)
scripts/
  train_meld.py                      # one run: --ablation <preset> --seed N (Task 7)
  modal_train.py                     # the same runs on a Modal A10G, ablations × seeds in parallel (Task 9)
  vision_baseline.py                 # zero-training baseline numbers (Task 6)
  run_ablations.py                   # loop presets × seeds (Task 8)
  ablation_table.py                  # markdown table from results/ (Task 8)
tests/
  test_train_config.py, test_text.py, test_dataset.py, test_model.py,
  test_metrics.py, test_baselines.py, test_train_loop.py, test_results.py
  conftest.py                        # shared synthetic-feature fixture + stubs (Task 3)
```

---

### Task 1: Training config and ablation presets

**Files:**
- Create: `src/meld_emotion/training/__init__.py`
- Create: `src/meld_emotion/training/config.py`
- Test: `tests/test_train_config.py`

**Interfaces:**
- Produces: frozen dataclass `TrainConfig` (fields below);
  `ABLATIONS: dict[str, TrainConfig]` with keys `text_only_k0,
  text_only_k4, vision_only, fusion, fusion_no_scene, fusion_no_context,
  fusion_no_trackid`; `config_for(name: str, **overrides) -> TrainConfig`
  (raises `KeyError` on an unknown name); `TrainConfig.to_dict() -> dict`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_train_config.py`:

```python
import pytest


def test_default_config_is_the_full_fusion_model():
    from meld_emotion.training.config import TrainConfig
    c = TrainConfig()
    assert (c.use_text, c.use_faces, c.use_scene, c.use_track_id) == (True, True, True, True)
    assert c.context_k == 4 and c.max_text_tokens == 256
    assert c.d_model == 768 and c.max_frames == 32 and c.max_track_slots == 16
    assert c.modality_dropout == pytest.approx(0.15)


def test_every_ablation_preset_is_named_after_its_key():
    from meld_emotion.training.config import ABLATIONS
    assert set(ABLATIONS) == {"text_only_k0", "text_only_k4", "vision_only", "fusion",
                              "fusion_no_scene", "fusion_no_context", "fusion_no_trackid"}
    for key, cfg in ABLATIONS.items():
        assert cfg.name == key


def test_presets_flip_exactly_the_intended_switches():
    from meld_emotion.training.config import ABLATIONS
    t0 = ABLATIONS["text_only_k0"]
    assert (t0.use_text, t0.use_faces, t0.use_scene, t0.context_k) == (True, False, False, 0)
    assert ABLATIONS["text_only_k4"].context_k == 4 and not ABLATIONS["text_only_k4"].use_faces
    assert not ABLATIONS["vision_only"].use_text
    assert not ABLATIONS["fusion_no_scene"].use_scene
    assert ABLATIONS["fusion_no_context"].context_k == 0
    assert not ABLATIONS["fusion_no_trackid"].use_track_id


def test_config_for_applies_overrides_without_mutating_the_preset():
    from meld_emotion.training.config import ABLATIONS, config_for
    c = config_for("fusion", seed=3, epochs=1)
    assert (c.seed, c.epochs, c.name) == (3, 1, "fusion")
    assert ABLATIONS["fusion"].seed == 0


def test_config_for_rejects_unknown_preset():
    from meld_emotion.training.config import config_for
    with pytest.raises(KeyError):
        config_for("bogus")


def test_a_config_with_no_modality_is_rejected():
    from meld_emotion.training.config import TrainConfig
    with pytest.raises(ValueError):
        TrainConfig(use_text=False, use_faces=False, use_scene=False)


def test_to_dict_round_trips_through_constructor():
    from meld_emotion.training.config import TrainConfig
    c = TrainConfig(n_layers=4, seed=7)
    assert TrainConfig(**c.to_dict()) == c
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_train_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meld_emotion.training'`

- [ ] **Step 3: Implement**

Create `src/meld_emotion/training/__init__.py` (empty file).

Create `src/meld_emotion/training/config.py`:

```python
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
}


def config_for(name: str, **overrides) -> TrainConfig:
    if name not in ABLATIONS:
        raise KeyError(f"unknown ablation {name!r}; choose from {sorted(ABLATIONS)}")
    return replace(ABLATIONS[name], **overrides)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_train_config.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/meld_emotion/training/__init__.py src/meld_emotion/training/config.py tests/test_train_config.py
git commit -m "Add Stage 1 TrainConfig with ablation presets"
```

---

### Task 2: Text formatting, tokenizer, and RoBERTa with partial freeze

**Files:**
- Create: `src/meld_emotion/training/text.py`
- Test: `tests/test_text.py`

**Interfaces:**
- Consumes: `TrainConfig.text_model`, `.text_trainable_layers`,
  `.max_text_tokens`, `.context_k` (Task 1).
- Produces: `format_context(context_prev: list[str], k: int) -> str` (the
  last *k* previous utterances joined by `" </s> "`, `""` when *k*=0 or no
  context); `build_tokenizer(model_name: str) -> PreTrainedTokenizerBase`
  (with `truncation_side="left"`); `encode_text(tokenizer, context: str,
  current: str, max_length: int) -> dict[str, list[int]]` (keys
  `input_ids`, `attention_mask`); `build_text_encoder(model_name: str,
  trainable_layers: int) -> nn.Module` returning a Hugging Face encoder whose
  `forward(input_ids=, attention_mask=)` has `.last_hidden_state (B, T,
  hidden_size)` and which exposes `.hidden_size: int`, with only the top
  `trainable_layers` encoder layers trainable; `count_parameters(module) ->
  tuple[int, int]` (total, trainable).

The text-encoder contract used by the model and the tests: any `nn.Module`
with `.hidden_size` whose call returns an object with `.last_hidden_state`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_text.py`:

```python
import pytest
import torch


def test_format_context_takes_the_last_k_previous_utterances():
    from meld_emotion.training.text import format_context
    assert format_context(["a", "b", "c", "d"], k=2) == "c </s> d"
    assert format_context(["a", "b"], k=4) == "a </s> b"


def test_format_context_is_empty_for_k_zero_or_no_history():
    from meld_emotion.training.text import format_context
    assert format_context(["a", "b"], k=0) == ""
    assert format_context([], k=4) == ""


def test_count_parameters_separates_frozen_from_trainable():
    from meld_emotion.training.text import count_parameters
    m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    for p in m[0].parameters():
        p.requires_grad = False
    total, trainable = count_parameters(m)
    assert total == 4 * 3 + 3 + 3 * 2 + 2
    assert trainable == 3 * 2 + 2


@pytest.mark.network
def test_encode_text_puts_context_first_current_last_and_truncates_oldest_context():
    from meld_emotion.training.text import build_tokenizer, encode_text, format_context
    tok = build_tokenizer("roberta-base")
    ctx = format_context(["line zero", "line one", "line two", "line three"], k=4)
    enc = encode_text(tok, ctx, "current line", max_length=16)
    assert len(enc["input_ids"]) == 16 and len(enc["attention_mask"]) == 16
    decoded = tok.decode(enc["input_ids"])
    assert decoded.endswith("</s></s>current line</s>")
    assert "line zero" not in decoded and "line three" in decoded   # oldest context dropped first


@pytest.mark.network
def test_encode_text_without_context_is_a_plain_single_sequence():
    from meld_emotion.training.text import build_tokenizer, encode_text
    tok = build_tokenizer("roberta-base")
    enc = encode_text(tok, "", "current line", max_length=16)
    assert tok.decode(enc["input_ids"]) == "<s>current line</s>"


@pytest.mark.network
def test_build_text_encoder_freezes_all_but_the_top_layers():
    from meld_emotion.training.text import build_text_encoder, count_parameters
    enc = build_text_encoder("roberta-base", trainable_layers=6)
    assert enc.hidden_size == 768
    total, trainable = count_parameters(enc)
    assert 120_000_000 < total < 130_000_000
    assert 40_000_000 < trainable < 45_000_000          # 6 of 12 layers ≈ 42.5M
    assert not any(p.requires_grad for p in enc.embeddings.parameters())
    assert all(p.requires_grad for p in enc.encoder.layer[-1].parameters())
    assert not any(p.requires_grad for p in enc.encoder.layer[0].parameters())
    out = enc(input_ids=torch.tensor([[0, 100, 2]]), attention_mask=torch.tensor([[1, 1, 1]]))
    assert out.last_hidden_state.shape == (1, 3, 768)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_text.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meld_emotion.training.text'`

- [ ] **Step 3: Implement**

Create `src/meld_emotion/training/text.py`:

```python
"""Dialogue-context text formatting and the RoBERTa text encoder.

Format (design doc §4.3): <s> u[t-k] </s> ... </s> u[t-1] </s></s> u[t] </s>
The context is the tokenizer's *first* sequence, the current utterance the
second, so `truncation="only_first"` with `truncation_side="left"` drops the
oldest context first and never touches the current utterance.
"""
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

CONTEXT_SEPARATOR = " </s> "   # RoBERTa's sep token; tokenizes to the real </s> id


def format_context(context_prev: list[str], k: int) -> str:
    if k <= 0 or not context_prev:
        return ""
    return CONTEXT_SEPARATOR.join(context_prev[-k:])


def build_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.truncation_side = "left"
    return tokenizer


def encode_text(tokenizer, context: str, current: str, max_length: int) -> dict[str, list[int]]:
    if context:
        enc = tokenizer(context, current, truncation="only_first", max_length=max_length)
    else:
        enc = tokenizer(current, truncation=True, max_length=max_length)
    return {"input_ids": list(enc["input_ids"]), "attention_mask": list(enc["attention_mask"])}


def build_text_encoder(model_name: str, trainable_layers: int) -> nn.Module:
    """Pretrained encoder with everything frozen except the top `trainable_layers`
    transformer layers (design doc §6: partial fine-tune on ~10K examples)."""
    # attn_implementation="eager": with the bottom layers frozen, MPS's fused
    # scaled_dot_product_attention picks a no-grad kernel that raises
    # "does not support dropout" in train mode (verified on torch 2.14).
    model = AutoModel.from_pretrained(model_name, add_pooling_layer=False, attn_implementation="eager")
    for p in model.parameters():
        p.requires_grad = False
    n_layers = model.config.num_hidden_layers
    for layer in model.encoder.layer[n_layers - trainable_layers:]:
        for p in layer.parameters():
            p.requires_grad = True
    model.hidden_size = model.config.hidden_size
    return model


def count_parameters(module: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_text.py -v` → Expected: 3 passed, 3 deselected
Run: `uv run pytest tests/test_text.py -m network -v` → Expected: 3 passed (first run downloads roberta-base, ~500MB)

- [ ] **Step 5: Commit**

```bash
git add src/meld_emotion/training/text.py tests/test_text.py
git commit -m "Add dialogue-context text encoding and RoBERTa with top-half fine-tuning"
```

---

### Task 3: Dataset over manifest + feature cache, and collate

**Files:**
- Create: `src/meld_emotion/training/dataset.py`
- Create: `tests/conftest.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `TrainConfig` (Task 1); `format_context` (Task 2);
  `EMOTIONS`, `SENTIMENTS` from `meld_emotion.data.labels`;
  `read_manifest` from `meld_emotion.data.manifest`.
- Produces: `EMOTION_TO_IDX`, `SENTIMENT_TO_IDX: dict[str, int]`;
  `remap_track_ids(track_ids: np.ndarray, max_slots: int) -> np.ndarray`;
  `load_ok_rows(manifest_path: Path) -> list[dict]`; `infer_feature_dims(
  rows, cache_dir) -> tuple[int, int]` (face_dim, scene_dim);
  `MeldFeatureDataset(rows, cache_dir: Path, config: TrainConfig,
  encode_fn: Callable[[str, str], dict[str, list[int]]])` whose items are
  dicts with `input_ids (T,) long`, `attention_mask (T,) long`, `face_feat
  (Nf, Df) float`, `face_frame (Nf,) long`, `face_track (Nf,) long`,
  `scene_feat (Ns, Ds) float`, `scene_frame (Ns,) long`, `emotion () long`,
  `sentiment () long`, `clip: str`; `collate(batch: list[dict], pad_id: int)
  -> dict` with `input_ids (B,T)`, `attention_mask (B,T)`, `face_feat
  (B,Fmax,Df)`, `face_mask (B,Fmax) bool`, `face_frame`, `face_track
  (B,Fmax)`, `scene_feat (B,Smax,Ds)`, `scene_mask (B,Smax) bool`,
  `scene_frame (B,Smax)`, `emotion (B,)`, `sentiment (B,)`, `clips:
  list[str]`; `MeldFeatureDataset.lengths: list[int]` (token count per
  item); `BucketBatchSampler(lengths: list[int], batch_size: int, shuffle:
  bool = True, seed: int = 0, chunk_batches: int = 50)` — a
  `torch.utils.data.Sampler` yielding lists of indices grouped by similar
  token length, with `set_epoch(epoch: int)` for a fresh shuffle each epoch.

Why the sampler: RoBERTa's step time is linear in padded tokens. Measured on
dev with *k*=4, random batches of 32 pad to a mean max of 114 tokens while
length-sorted buckets pad to 59 — a 1.9× cut in text-encoder compute, which
is nearly all of the step. `encode_fn` is how the tokenizer is injected: production passes
`functools.partial(encode_text, tokenizer, max_length=...)` (Task 2); tests
pass a whitespace tokenizer. The tokenizer is applied eagerly in `__init__`
so `__getitem__` only loads one `.npz`.

- [ ] **Step 1: Write the shared synthetic-feature fixture**

Create `tests/conftest.py`:

```python
"""Shared fixtures: a synthetic feature cache + manifests in the exact layout
the data plan writes, small enough to train on in a unit test."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from meld_emotion.data.labels import EMOTIONS, SENTIMENTS

FACE_DIM, SCENE_DIM = 8, 4


def whitespace_encode(context: str, current: str) -> dict[str, list[int]]:
    """Stand-in for encode_text: word -> small id, context before current."""
    words = (context + " " + current).split() if context else current.split()
    ids = [0] + [(abs(hash(w)) % 60) + 2 for w in words] + [1]   # 0=<s>, 1=</s>, pad=1 unused
    return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class StubTextEncoder(nn.Module):
    """Same contract as build_text_encoder(): .hidden_size and .last_hidden_state."""
    def __init__(self, hidden_size=32, vocab=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(vocab, hidden_size)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.emb(input_ids))


def write_synthetic_split(root, split, n_clips, seed=0, n_frames=3, faces_per_frame=(0, 1, 2),
                          long_clip_every=0):
    """Writes <root>/<split>/manifest.jsonl and <root>/<split>/dia<D>_utt<U>.npz.
    Faces per frame cycle by utterance id (utt0 -> 0, utt1 -> 1, utt2 -> 2, ...)
    so every dialogue has a zero-face clip."""
    rng = np.random.default_rng(seed)
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_clips):
        dia, utt = i // 4, i % 4
        n_faces_per = faces_per_frame[utt % len(faces_per_frame)]
        face_frame = np.repeat(np.arange(n_frames), n_faces_per)
        face_track = np.tile(np.arange(n_faces_per) * 7, n_frames)   # sparse raw ids, e.g. 0, 7
        n_faces = len(face_frame)
        npz_name = f"dia{dia}_utt{utt}.npz"
        np.savez(split_dir / npz_name,
                 face_features=rng.standard_normal((n_faces, FACE_DIM)).astype(np.float32),
                 face_probs=(lambda p: p / p.sum(1, keepdims=True))(rng.random((n_faces, 7)).astype(np.float32)),
                 face_track_ids=face_track.astype(np.int64),
                 face_frame_idx=face_frame.astype(np.int64),
                 scene_features=rng.standard_normal((n_frames, SCENE_DIM)).astype(np.float32),
                 scene_frame_idx=np.arange(n_frames, dtype=np.int64),
                 shot_cut=np.zeros(n_frames, dtype=bool),
                 dialogue_id=dia, utterance_id=utt)
        duration = 20.0 if (long_clip_every and i % long_clip_every == 0) else 2.5
        rows.append({"split": split, "dialogue_id": dia, "utterance_id": utt, "speaker": "Joey",
                     "text": f"utterance number {i} here", "context_prev": [f"earlier line {j}" for j in range(utt)],
                     "emotion": EMOTIONS[i % 7], "sentiment": SENTIMENTS[i % 3],
                     "status": "ok", "feature_path": f"{split}/{npz_name}", "duration_s": duration,
                     "n_frames": n_frames, "n_faces": n_faces, "n_shot_cuts": 0})
    # one non-ok row that must be filtered out
    rows.append({**rows[-1], "utterance_id": 99, "status": "decode_failed", "feature_path": None})
    with open(split_dir / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return split_dir / "manifest.jsonl"


@pytest.fixture
def synthetic_features(tmp_path):
    """Returns (root, train_manifest, dev_manifest) with 28 train / 14 dev clips."""
    root = tmp_path / "features"
    train = write_synthetic_split(root, "train", 28, seed=0, long_clip_every=7)
    dev = write_synthetic_split(root, "dev", 14, seed=1)
    return root, train, dev
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_dataset.py`:

```python
import numpy as np
import pytest
import torch

from conftest import FACE_DIM, SCENE_DIM, whitespace_encode


def test_remap_track_ids_by_first_appearance_with_overflow_slot():
    from meld_emotion.training.dataset import remap_track_ids
    ids = np.array([7, 0, 7, 55, 0, 3, 55])
    assert remap_track_ids(ids, max_slots=16).tolist() == [0, 1, 0, 2, 1, 3, 2]
    assert remap_track_ids(ids, max_slots=2).tolist() == [0, 1, 0, 2, 1, 2, 2]   # 3rd+ distinct -> overflow=2
    assert remap_track_ids(np.array([], dtype=np.int64), 16).shape == (0,)


def test_load_ok_rows_drops_non_ok_rows(synthetic_features):
    from meld_emotion.training.dataset import load_ok_rows
    _, train, _ = synthetic_features
    rows = load_ok_rows(train)
    assert len(rows) == 28 and all(r["status"] == "ok" for r in rows)


def test_infer_feature_dims_reads_widths_even_from_a_zero_face_clip(synthetic_features):
    from meld_emotion.training.dataset import infer_feature_dims, load_ok_rows
    root, train, _ = synthetic_features
    rows = load_ok_rows(train)
    assert rows[0]["n_faces"] == 0                       # first clip cycles to 0 faces
    assert infer_feature_dims(rows, root) == (FACE_DIM, SCENE_DIM)


def test_dataset_item_shapes_labels_and_context(synthetic_features):
    from meld_emotion.training.config import TrainConfig
    from meld_emotion.training.dataset import EMOTION_TO_IDX, MeldFeatureDataset, load_ok_rows
    root, train, _ = synthetic_features
    rows = load_ok_rows(train)
    ds = MeldFeatureDataset(rows, root, TrainConfig(context_k=2), whitespace_encode)
    item = ds[5]                                          # dia1_utt1: 1 face/frame, 1 previous line
    assert item["face_feat"].shape == (3, FACE_DIM) and item["face_feat"].dtype == torch.float32
    assert item["face_frame"].tolist() == [0, 1, 2]
    assert item["face_track"].tolist() == [0, 0, 0]       # raw id 0 in every frame -> slot 0
    assert item["scene_feat"].shape == (3, SCENE_DIM)
    assert item["emotion"].item() == EMOTION_TO_IDX[rows[5]["emotion"]]
    assert item["input_ids"].dtype == torch.long and item["input_ids"][0] == 0
    assert item["clip"] == "dia1_utt1"
    n_context_words = len(" ".join(rows[5]["context_prev"][-2:]).split())
    assert len(item["input_ids"]) == 2 + n_context_words + len(rows[5]["text"].split())


def test_dataset_clamps_frame_positions_and_remaps_tracks(synthetic_features, tmp_path):
    from meld_emotion.training.config import TrainConfig
    from meld_emotion.training.dataset import MeldFeatureDataset, load_ok_rows
    root, train, _ = synthetic_features
    rows = load_ok_rows(train)
    row = rows[6]                                         # 2 faces/frame -> raw ids 0 and 7
    z = dict(np.load(root / row["feature_path"]))
    z["face_frame_idx"] = z["face_frame_idx"] + 40        # push past the 32-slot table
    z["scene_frame_idx"] = z["scene_frame_idx"] + 40
    np.savez(root / row["feature_path"], **z)
    ds = MeldFeatureDataset([row], root, TrainConfig(max_frames=32, max_track_slots=16), whitespace_encode)
    item = ds[0]
    assert item["face_frame"].max().item() == 31 and item["scene_frame"].max().item() == 31
    assert sorted(set(item["face_track"].tolist())) == [0, 1]


def test_dataset_switches_off_disabled_modalities_and_masks_long_clips(synthetic_features):
    from meld_emotion.training.config import TrainConfig
    from meld_emotion.training.dataset import MeldFeatureDataset, load_ok_rows
    root, train, _ = synthetic_features
    rows = load_ok_rows(train)
    no_faces = MeldFeatureDataset(rows, root, TrainConfig(use_faces=False), whitespace_encode)[5]
    assert no_faces["face_feat"].shape == (0, FACE_DIM) and no_faces["scene_feat"].shape[0] == 3
    no_scene = MeldFeatureDataset(rows, root, TrainConfig(use_scene=False), whitespace_encode)[5]
    assert no_scene["scene_feat"].shape == (0, SCENE_DIM) and no_scene["face_feat"].shape[0] == 3
    masked = MeldFeatureDataset(rows, root, TrainConfig(mask_vision_over_seconds=15.0), whitespace_encode)
    assert rows[7]["duration_s"] == 20.0
    assert masked[7]["face_feat"].shape[0] == 0 and masked[7]["scene_feat"].shape[0] == 0
    assert masked[8]["scene_feat"].shape[0] == 3


def test_dataset_exposes_token_lengths(synthetic_features):
    from meld_emotion.training.config import TrainConfig
    from meld_emotion.training.dataset import MeldFeatureDataset, load_ok_rows
    root, train, _ = synthetic_features
    ds = MeldFeatureDataset(load_ok_rows(train), root, TrainConfig(), whitespace_encode)
    assert ds.lengths == [len(ds[i]["input_ids"]) for i in range(len(ds))]


def test_bucket_sampler_covers_every_index_once_and_groups_similar_lengths():
    from meld_emotion.training.dataset import BucketBatchSampler
    rng = np.random.default_rng(0)
    lengths = rng.integers(5, 120, size=203).tolist()
    sampler = BucketBatchSampler(lengths, batch_size=16, shuffle=True, seed=0, chunk_batches=4)
    batches = list(sampler)
    assert len(sampler) == len(batches) == 13
    assert sorted(i for b in batches for i in b) == list(range(203))
    spread_bucketed = np.mean([max(lengths[i] for i in b) - min(lengths[i] for i in b) for b in batches])
    spread_random = np.mean([np.ptp([lengths[i] for i in rng.permutation(203)[:16]]) for _ in range(13)])
    assert spread_bucketed < spread_random / 2


def test_bucket_sampler_is_deterministic_per_epoch_and_sorted_when_not_shuffling():
    from meld_emotion.training.dataset import BucketBatchSampler
    lengths = [30, 5, 20, 5, 30, 20, 7, 8]
    a = BucketBatchSampler(lengths, batch_size=2, shuffle=True, seed=1)
    b = BucketBatchSampler(lengths, batch_size=2, shuffle=True, seed=1)
    assert list(a) == list(b)
    a.set_epoch(1)
    assert list(a) != list(b)
    ordered = list(BucketBatchSampler(lengths, batch_size=2, shuffle=False))
    assert [lengths[i] for batch in ordered for i in batch] == sorted(lengths)


def test_collate_pads_and_masks_variable_length_sets(synthetic_features):
    from meld_emotion.training.config import TrainConfig
    from meld_emotion.training.dataset import MeldFeatureDataset, collate, load_ok_rows
    root, train, _ = synthetic_features
    rows = load_ok_rows(train)
    ds = MeldFeatureDataset(rows, root, TrainConfig(), whitespace_encode)
    batch = collate([ds[4], ds[5], ds[6]], pad_id=1)      # 0, 1, 2 faces per frame
    assert batch["face_feat"].shape == (3, 6, FACE_DIM)
    assert batch["face_mask"].sum(1).tolist() == [0, 3, 6]
    assert batch["scene_feat"].shape == (3, 3, SCENE_DIM) and batch["scene_mask"].all()
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    lengths = [len(ds[i]["input_ids"]) for i in (4, 5, 6)]
    assert batch["attention_mask"].sum(1).tolist() == lengths
    assert (batch["input_ids"][0, lengths[0]:] == 1).all()
    assert batch["emotion"].shape == (3,) and batch["sentiment"].shape == (3,)
    assert batch["clips"] == ["dia1_utt0", "dia1_utt1", "dia1_utt2"]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meld_emotion.training.dataset'`

- [ ] **Step 4: Implement**

Create `src/meld_emotion/training/dataset.py`:

```python
"""Dataset over the data plan's outputs: manifest.jsonl rows + one .npz per
clip. Yields token ids for the text (with dialogue context) and the
variable-size face/scene token sets the fusion model consumes."""
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from meld_emotion.data.labels import EMOTIONS, SENTIMENTS
from meld_emotion.data.manifest import read_manifest
from meld_emotion.training.config import TrainConfig
from meld_emotion.training.text import format_context

EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}
SENTIMENT_TO_IDX = {s: i for i, s in enumerate(SENTIMENTS)}


def remap_track_ids(track_ids: np.ndarray, max_slots: int) -> np.ndarray:
    """Raw per-clip track ids are unique but sparse (they reach 55 on dev).
    Remap by first appearance to 0..max_slots-1; later distinct ids share
    the overflow slot max_slots."""
    order: dict[int, int] = {}
    out = np.empty(len(track_ids), dtype=np.int64)
    for i, t in enumerate(track_ids.tolist()):
        if t not in order:
            order[t] = len(order)
        out[i] = min(order[t], max_slots)
    return out


def load_ok_rows(manifest_path: Path) -> list[dict]:
    return [r for r in read_manifest(manifest_path) if r["status"] == "ok"]


def infer_feature_dims(rows: list[dict], cache_dir: Path) -> tuple[int, int]:
    z = np.load(Path(cache_dir) / rows[0]["feature_path"])
    return int(z["face_features"].shape[1]), int(z["scene_features"].shape[1])


class MeldFeatureDataset(Dataset):
    def __init__(self, rows: list[dict], cache_dir: Path, config: TrainConfig,
                 encode_fn: Callable[[str, str], dict[str, list[int]]]):
        self.rows = rows
        self.cache_dir = Path(cache_dir)
        self.config = config
        self.encoded = [encode_fn(format_context(r["context_prev"], config.context_k), r["text"])
                        for r in rows]
        self.lengths = [len(e["input_ids"]) for e in self.encoded]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row, cfg = self.rows[i], self.config
        z = np.load(self.cache_dir / row["feature_path"])
        face_feat, face_frame, face_track = z["face_features"], z["face_frame_idx"], z["face_track_ids"]
        scene_feat, scene_frame = z["scene_features"], z["scene_frame_idx"]

        vision_masked = (cfg.mask_vision_over_seconds is not None
                         and row["duration_s"] > cfg.mask_vision_over_seconds)
        if not cfg.use_faces or vision_masked:
            face_feat, face_frame, face_track = face_feat[:0], face_frame[:0], face_track[:0]
        if not cfg.use_scene or vision_masked:
            scene_feat, scene_frame = scene_feat[:0], scene_frame[:0]

        enc = self.encoded[i]
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "face_feat": torch.from_numpy(np.ascontiguousarray(face_feat, dtype=np.float32)),
            "face_frame": torch.from_numpy(np.minimum(face_frame, cfg.max_frames - 1).astype(np.int64)),
            "face_track": torch.from_numpy(remap_track_ids(face_track, cfg.max_track_slots)),
            "scene_feat": torch.from_numpy(np.ascontiguousarray(scene_feat, dtype=np.float32)),
            "scene_frame": torch.from_numpy(np.minimum(scene_frame, cfg.max_frames - 1).astype(np.int64)),
            "emotion": torch.tensor(EMOTION_TO_IDX[row["emotion"]], dtype=torch.long),
            "sentiment": torch.tensor(SENTIMENT_TO_IDX[row["sentiment"]], dtype=torch.long),
            "clip": f"dia{row['dialogue_id']}_utt{row['utterance_id']}",
        }


class BucketBatchSampler(Sampler[list[int]]):
    """Batches of similar token length. RoBERTa's cost is linear in padded
    tokens; on dev with k=4, random batches pad to ~114 tokens and
    length-sorted buckets to ~59. Shuffles inside mega-chunks of
    `chunk_batches` batches and shuffles batch order, fresh each epoch."""

    def __init__(self, lengths: list[int], batch_size: int, shuffle: bool = True,
                 seed: int = 0, chunk_batches: int = 50):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.chunk = batch_size * chunk_batches
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self) -> list[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.lengths)) if self.shuffle else np.arange(len(self.lengths))
        batches = []
        for start in range(0, len(order), self.chunk):
            chunk = order[start:start + self.chunk]
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            batches += [chunk[i:i + self.batch_size].tolist() for i in range(0, len(chunk), self.batch_size)]
        if self.shuffle:
            batches = [batches[i] for i in rng.permutation(len(batches))]
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)


def _pad_stack(seqs: list[torch.Tensor], pad_value) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack variable-length (N, ...) tensors into (B, Nmax, ...) plus a bool
    validity mask (B, Nmax). Works for N=0 rows and for 1-D or 2-D items."""
    n_max = max((s.shape[0] for s in seqs), default=0)
    trailing = seqs[0].shape[1:]
    out = torch.full((len(seqs), n_max, *trailing), pad_value, dtype=seqs[0].dtype)
    mask = torch.zeros(len(seqs), n_max, dtype=torch.bool)
    for b, s in enumerate(seqs):
        out[b, :s.shape[0]] = s
        mask[b, :s.shape[0]] = True
    return out, mask


def collate(batch: list[dict], pad_id: int) -> dict:
    input_ids, _ = _pad_stack([b["input_ids"] for b in batch], pad_id)
    attention_mask, _ = _pad_stack([b["attention_mask"] for b in batch], 0)
    face_feat, face_mask = _pad_stack([b["face_feat"] for b in batch], 0.0)
    face_frame, _ = _pad_stack([b["face_frame"] for b in batch], 0)
    face_track, _ = _pad_stack([b["face_track"] for b in batch], 0)
    scene_feat, scene_mask = _pad_stack([b["scene_feat"] for b in batch], 0.0)
    scene_frame, _ = _pad_stack([b["scene_frame"] for b in batch], 0)
    return {
        "input_ids": input_ids, "attention_mask": attention_mask,
        "face_feat": face_feat, "face_mask": face_mask, "face_frame": face_frame, "face_track": face_track,
        "scene_feat": scene_feat, "scene_mask": scene_mask, "scene_frame": scene_frame,
        "emotion": torch.stack([b["emotion"] for b in batch]),
        "sentiment": torch.stack([b["sentiment"] for b in batch]),
        "clips": [b["clip"] for b in batch],
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: 10 passed

- [ ] **Step 6: Commit**

```bash
git add src/meld_emotion/training/dataset.py tests/conftest.py tests/test_dataset.py
git commit -m "Add feature-cache Dataset with context encoding, track remap, frame clamp, padded collate, and length-bucketed batching"
```

---

### Task 4: Single-stream fusion model with modality dropout

**Files:**
- Create: `src/meld_emotion/training/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `TrainConfig` (Task 1); the batch dict from `collate` (Task 3);
  a text encoder per the Task 2 contract.
- Produces: `FusionModel(config: TrainConfig, text_encoder: nn.Module |
  None, face_dim: int, scene_dim: int, n_emotions: int = 7, n_sentiments:
  int = 3)` with `forward(batch: dict, force_drop_text: bool = False,
  force_drop_vision: bool = False) -> dict` (keys `emotion_logits (B,7)`,
  `sentiment_logits (B,3)`); `modality_dropout_masks(self, batch) ->
  tuple[Tensor, Tensor]` (`drop_text (B,) bool`, `drop_vision (B,) bool`);
  `FusionModel.text_parameters()` / `.fusion_parameters()` iterators for the
  optimiser's two learning rates (Task 7).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_model.py`:

```python
import pytest
import torch

from conftest import FACE_DIM, SCENE_DIM, StubTextEncoder, whitespace_encode


def _batch(synthetic_features, config, idxs=(4, 5, 6)):
    from meld_emotion.training.dataset import MeldFeatureDataset, collate, load_ok_rows
    root, train, _ = synthetic_features
    ds = MeldFeatureDataset(load_ok_rows(train), root, config, whitespace_encode)
    return collate([ds[i] for i in idxs], pad_id=1)


def _model(config, text_encoder=StubTextEncoder(hidden_size=32)):
    from meld_emotion.training.model import FusionModel
    return FusionModel(config, text_encoder, face_dim=FACE_DIM, scene_dim=SCENE_DIM)


def _cfg(**kw):
    from meld_emotion.training.config import TrainConfig
    return TrainConfig(d_model=32, n_heads=4, ff_dim=64, n_layers=2, **kw)


def test_forward_produces_logits_of_the_right_shape(synthetic_features):
    cfg = _cfg()
    out = _model(cfg).eval()(_batch(synthetic_features, cfg))
    assert out["emotion_logits"].shape == (3, 7) and out["sentiment_logits"].shape == (3, 3)
    assert torch.isfinite(out["emotion_logits"]).all()


def test_forward_works_for_text_only_vision_only_and_no_scene(synthetic_features):
    for cfg in (_cfg(use_faces=False, use_scene=False), _cfg(use_text=False), _cfg(use_scene=False)):
        model = _model(cfg, text_encoder=None if not cfg.use_text else StubTextEncoder(32)).eval()
        out = model(_batch(synthetic_features, cfg))
        assert out["emotion_logits"].shape == (3, 7) and torch.isfinite(out["emotion_logits"]).all()


def test_a_sample_with_zero_visual_tokens_still_gets_finite_logits(synthetic_features):
    cfg = _cfg()
    batch = _batch(synthetic_features, cfg, idxs=(4,))           # clip with 0 faces
    batch["scene_mask"][:] = False                              # and now no scene tokens either
    out = _model(cfg).eval()(batch)
    assert torch.isfinite(out["emotion_logits"]).all()


def test_padding_tokens_do_not_change_the_prediction(synthetic_features):
    cfg = _cfg()
    model = _model(cfg).eval()
    single = _batch(synthetic_features, cfg, idxs=(5,))
    padded = _batch(synthetic_features, cfg, idxs=(5, 6))       # sample 5 now padded to sample 6's lengths
    a = model(single)["emotion_logits"][0]
    b = model(padded)["emotion_logits"][0]
    assert torch.allclose(a, b, atol=1e-5)


def test_force_drop_flags_change_the_output_and_track_id_flag_is_respected(synthetic_features):
    cfg = _cfg()
    model = _model(cfg).eval()
    batch = _batch(synthetic_features, cfg, idxs=(6,))
    full = model(batch)["emotion_logits"]
    assert not torch.allclose(full, model(batch, force_drop_vision=True)["emotion_logits"])
    assert not torch.allclose(full, model(batch, force_drop_text=True)["emotion_logits"])
    no_track = _model(_cfg(use_track_id=False)).eval()
    batch2 = _batch(synthetic_features, cfg, idxs=(6,))
    batch2["face_track"] = batch2["face_track"] + 5              # different slots
    assert torch.allclose(no_track(batch)["emotion_logits"], no_track(batch2)["emotion_logits"])


def test_modality_dropout_never_drops_both_and_never_drops_an_absent_modality(synthetic_features):
    cfg = _cfg(modality_dropout=0.5)
    model = _model(cfg).train()
    batch = _batch(synthetic_features, cfg, idxs=(4, 5, 6))      # sample 0 has 0 faces but has scene
    batch["scene_mask"][0] = False                              # now sample 0 has no vision at all
    torch.manual_seed(0)
    seen_text_drop = seen_vision_drop = False
    for _ in range(200):
        drop_text, drop_vision = model.modality_dropout_masks(batch)
        assert not (drop_text & drop_vision).any()
        assert not drop_text[0] and not drop_vision[0]           # nothing to fall back on / nothing to drop
        seen_text_drop |= bool(drop_text[1:].any())
        seen_vision_drop |= bool(drop_vision[1:].any())
    assert seen_text_drop and seen_vision_drop
    model.eval()
    drop_text, drop_vision = model.modality_dropout_masks(batch)
    assert not drop_text.any() and not drop_vision.any()         # eval mode: no dropout


def test_parameter_groups_split_text_encoder_from_the_rest(synthetic_features):
    cfg = _cfg()
    model = _model(cfg)
    text_ids = {id(p) for p in model.text_parameters()}
    fusion_ids = {id(p) for p in model.fusion_parameters()}
    assert text_ids and fusion_ids and not (text_ids & fusion_ids)
    assert text_ids | fusion_ids == {id(p) for p in model.parameters()}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meld_emotion.training.model'`

- [ ] **Step 3: Implement**

Create `src/meld_emotion/training/model.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_model.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/meld_emotion/training/model.py tests/test_model.py
git commit -m "Add single-stream FusionModel with modality/frame/track embeddings and modality dropout"
```

---

### Task 5: Class weights, joint loss, and metrics

**Files:**
- Create: `src/meld_emotion/training/metrics.py`
- Modify: `pyproject.toml` (via `uv add scikit-learn`)
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: model output dict and batch dict (Tasks 3, 4).
- Produces: `class_weights(labels: np.ndarray, n_classes: int, alpha: float)
  -> torch.Tensor` (shape `(n_classes,)`, mean 1.0, all-ones when
  alpha=0); `joint_loss(out: dict, batch: dict, emotion_weight:
  torch.Tensor | None, sentiment_lambda: float, label_smoothing: float) ->
  tuple[torch.Tensor, dict[str, float]]` (total, `{"emotion": …,
  "sentiment": …}`); `compute_metrics(y_true: np.ndarray, y_pred:
  np.ndarray, labels: tuple[str, ...]) -> dict` with keys `accuracy`,
  `weighted_f1`, `macro_f1`, `per_class_f1: dict[str, float]`, `confusion:
  list[list[int]]` (rows = true, cols = predicted, in `labels` order),
  `n: int`.

- [ ] **Step 1: Add scikit-learn**

Run: `uv add "scikit-learn>=1.4"`
Expected: `pyproject.toml` and `uv.lock` updated.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_metrics.py`:

```python
import numpy as np
import pytest
import torch


def test_class_weights_follow_inverse_frequency_to_alpha_and_average_to_one():
    from meld_emotion.training.metrics import class_weights
    labels = np.array([0] * 8 + [1] * 2)                       # 80% / 20%
    w = class_weights(labels, n_classes=2, alpha=1.0)
    assert w[1] / w[0] == pytest.approx(4.0)
    assert w.mean().item() == pytest.approx(1.0)
    w_half = class_weights(labels, n_classes=2, alpha=0.5)
    assert w_half[1] / w_half[0] == pytest.approx(2.0)
    assert torch.equal(class_weights(labels, n_classes=2, alpha=0.0), torch.ones(2))


def test_class_weights_handle_an_absent_class():
    from meld_emotion.training.metrics import class_weights
    w = class_weights(np.array([0, 0, 1]), n_classes=3, alpha=1.0)
    assert torch.isfinite(w).all() and w.shape == (3,)


def test_joint_loss_combines_emotion_and_lambda_weighted_sentiment():
    from meld_emotion.training.metrics import joint_loss
    out = {"emotion_logits": torch.zeros(4, 7), "sentiment_logits": torch.zeros(4, 3)}
    batch = {"emotion": torch.tensor([0, 1, 2, 3]), "sentiment": torch.tensor([0, 1, 2, 0])}
    total, parts = joint_loss(out, batch, emotion_weight=None, sentiment_lambda=0.5, label_smoothing=0.0)
    assert parts["emotion"] == pytest.approx(np.log(7), abs=1e-5)
    assert parts["sentiment"] == pytest.approx(np.log(3), abs=1e-5)
    assert total.item() == pytest.approx(np.log(7) + 0.5 * np.log(3), abs=1e-5)


def test_compute_metrics_on_a_known_confusion():
    from meld_emotion.training.metrics import compute_metrics
    labels = ("a", "b", "c")
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 0, 2, 1])
    m = compute_metrics(y_true, y_pred, labels)
    assert m["n"] == 6 and m["accuracy"] == pytest.approx(4 / 6)
    assert m["confusion"] == [[2, 0, 0], [1, 1, 0], [0, 1, 1]]
    assert m["per_class_f1"]["a"] == pytest.approx(2 * 2 / (2 * 2 + 1))     # tp=2 fp=1 fn=0
    assert 0 < m["macro_f1"] < 1 and 0 < m["weighted_f1"] < 1
    assert set(m["per_class_f1"]) == set(labels)


def test_compute_metrics_keeps_every_label_even_if_never_predicted():
    from meld_emotion.training.metrics import compute_metrics
    m = compute_metrics(np.array([0, 0]), np.array([0, 0]), ("a", "b"))
    assert m["per_class_f1"]["b"] == 0.0 and m["confusion"] == [[2, 0], [0, 0]]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meld_emotion.training.metrics'`

- [ ] **Step 4: Implement**

Create `src/meld_emotion/training/metrics.py`:

```python
"""Loss and metrics (design doc §4.3): class-weighted emotion CE + λ·sentiment
CE; weighted-F1 primary, macro-F1 secondary, per-class F1, confusion matrix."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score


def class_weights(labels: np.ndarray, n_classes: int, alpha: float) -> torch.Tensor:
    """w_c ∝ (1/freq_c)^alpha, rescaled to mean 1 so the loss scale doesn't
    depend on alpha. alpha=0 -> uniform; alpha=1 -> full inverse frequency."""
    counts = np.bincount(np.asarray(labels), minlength=n_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)                  # absent class: treat as one example
    freq = counts / counts.sum()
    w = (1.0 / freq) ** alpha
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def joint_loss(out: dict, batch: dict, emotion_weight: torch.Tensor | None,
               sentiment_lambda: float, label_smoothing: float) -> tuple[torch.Tensor, dict[str, float]]:
    emotion = F.cross_entropy(out["emotion_logits"], batch["emotion"],
                              weight=emotion_weight, label_smoothing=label_smoothing)
    sentiment = F.cross_entropy(out["sentiment_logits"], batch["sentiment"])
    total = emotion + sentiment_lambda * sentiment
    return total, {"emotion": float(emotion.item()), "sentiment": float(sentiment.item())}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: tuple[str, ...]) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    idx = list(range(len(labels)))
    per_class = f1_score(y_true, y_pred, labels=idx, average=None, zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "weighted_f1": float(f1_score(y_true, y_pred, labels=idx, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=idx, average="macro", zero_division=0)),
        "per_class_f1": {label: float(f) for label, f in zip(labels, per_class)},
        "confusion": confusion_matrix(y_true, y_pred, labels=idx).tolist(),
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/meld_emotion/training/metrics.py tests/test_metrics.py
git commit -m "Add class-weighted joint loss and F1/confusion metrics"
```

---

### Task 6: Zero-training vision-only baseline

**Files:**
- Create: `src/meld_emotion/training/baselines.py`
- Create: `scripts/vision_baseline.py`
- Test: `tests/test_baselines.py`

**Interfaces:**
- Consumes: `load_ok_rows`, `EMOTION_TO_IDX` (Task 3); `compute_metrics`
  (Task 5); `FACE_MODEL_ID`, `FACE_TO_MELD_LABEL` from
  `meld_emotion.vision.encoders`.
- Produces: `face_meld_label_order(model_id: str = FACE_MODEL_ID) ->
  tuple[str, ...]` (the cached face head's logit order, in MELD spelling —
  reads only the checkpoint's `config.json`); `vision_baseline_predictions(
  rows: list[dict], cache_dir: Path, face_labels: tuple[str, ...], fallback:
  str = "neutral") -> tuple[np.ndarray, np.ndarray]` (`y_true`, `y_pred`
  as `EMOTIONS` indices; clips with no faces predict `fallback`);
  `vision_baseline_metrics(rows, cache_dir, face_labels) -> dict`
  (`compute_metrics` output plus `no_face_clips: int`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_baselines.py`:

```python
import numpy as np

from meld_emotion.data.labels import EMOTIONS


def test_vision_baseline_averages_face_probs_and_falls_back_when_no_faces(synthetic_features):
    from meld_emotion.training.baselines import vision_baseline_predictions
    from meld_emotion.training.dataset import EMOTION_TO_IDX, load_ok_rows
    root, train, _ = synthetic_features
    rows = load_ok_rows(train)[:6]
    # Overwrite clip 1's probs so the mean over its faces is unambiguous.
    row = rows[1]
    z = dict(np.load(root / row["feature_path"]))
    probs = np.zeros_like(z["face_probs"]); probs[:, 3] = 1.0        # face-head column 3
    z["face_probs"] = probs
    np.savez(root / row["feature_path"], **z)
    face_labels = ("sadness", "disgust", "anger", "neutral", "fear", "surprise", "joy")   # column 3 = neutral
    face_labels = tuple(face_labels[i] if i != 3 else "surprise" for i in range(7))      # remap col 3 -> surprise
    y_true, y_pred = vision_baseline_predictions(rows, root, face_labels, fallback="neutral")
    assert y_true.tolist() == [EMOTION_TO_IDX[r["emotion"]] for r in rows]
    assert y_pred[1] == EMOTION_TO_IDX["surprise"]
    assert rows[0]["n_faces"] == 0 and y_pred[0] == EMOTION_TO_IDX["neutral"]


def test_vision_baseline_metrics_reports_no_face_count(synthetic_features):
    from meld_emotion.training.baselines import vision_baseline_metrics
    from meld_emotion.training.dataset import load_ok_rows
    root, train, _ = synthetic_features
    rows = load_ok_rows(train)
    face_labels = ("sadness", "disgust", "anger", "neutral", "fear", "surprise", "joy")
    m = vision_baseline_metrics(rows, root, face_labels)
    assert m["n"] == 28 and m["no_face_clips"] == sum(1 for r in rows if r["n_faces"] == 0)
    assert set(m["per_class_f1"]) == set(EMOTIONS)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_baselines.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meld_emotion.training.baselines'`

- [ ] **Step 3: Implement**

Create `src/meld_emotion/training/baselines.py`:

```python
"""Zero-training vision-only baseline (design doc §7): the frozen face
encoder's own 7-way softmax, averaged over every detected face in the clip.
No parameters are trained; it anchors the ablation table."""
from pathlib import Path

import numpy as np

from meld_emotion.data.labels import EMOTIONS
from meld_emotion.training.dataset import EMOTION_TO_IDX
from meld_emotion.training.metrics import compute_metrics
from meld_emotion.vision.encoders import FACE_MODEL_ID, FACE_TO_MELD_LABEL


def face_meld_label_order(model_id: str = FACE_MODEL_ID) -> tuple[str, ...]:
    """The face head's logit order in MELD spelling, from the checkpoint's
    config.json (cached by the data plan's encoder tests / cache build)."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id)
    return tuple(FACE_TO_MELD_LABEL[cfg.id2label[i]] for i in range(cfg.num_labels))


def vision_baseline_predictions(rows: list[dict], cache_dir: Path, face_labels: tuple[str, ...],
                                fallback: str = "neutral") -> tuple[np.ndarray, np.ndarray]:
    col_to_idx = np.array([EMOTION_TO_IDX[label] for label in face_labels])
    y_true, y_pred = [], []
    for row in rows:
        probs = np.load(Path(cache_dir) / row["feature_path"])["face_probs"]
        y_true.append(EMOTION_TO_IDX[row["emotion"]])
        y_pred.append(int(col_to_idx[probs.mean(0).argmax()]) if len(probs) else EMOTION_TO_IDX[fallback])
    return np.array(y_true), np.array(y_pred)


def vision_baseline_metrics(rows: list[dict], cache_dir: Path, face_labels: tuple[str, ...],
                            fallback: str = "neutral") -> dict:
    y_true, y_pred = vision_baseline_predictions(rows, cache_dir, face_labels, fallback)
    metrics = compute_metrics(y_true, y_pred, EMOTIONS)
    metrics["no_face_clips"] = int(sum(1 for r in rows if r["n_faces"] == 0))
    return metrics
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_baselines.py -v`
Expected: 2 passed

- [ ] **Step 5: Add the CLI**

Create `scripts/vision_baseline.py`:

```python
#!/usr/bin/env python3
"""Zero-training vision-only baseline on dev and/or test: average the frozen
face encoder's own probabilities over each clip's faces. Writes
results/vision_baseline/results.json.

Usage:
    uv run python scripts/vision_baseline.py --splits dev test
"""
import argparse
import json
from pathlib import Path

from meld_emotion.config import FEATURE_CACHE_DIR, REPO_ROOT
from meld_emotion.training.baselines import face_meld_label_order, vision_baseline_metrics
from meld_emotion.training.dataset import load_ok_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=["dev"], choices=["train", "dev", "test"])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "vision_baseline")
    args = parser.parse_args()

    face_labels = face_meld_label_order()
    results = {"name": "vision_only_zero_training", "face_label_order": list(face_labels)}
    for split in args.splits:
        rows = load_ok_rows(FEATURE_CACHE_DIR / split / "manifest.jsonl")
        results[split] = vision_baseline_metrics(rows, FEATURE_CACHE_DIR, face_labels)
        m = results[split]
        print(f"{split}: n={m['n']} weighted_f1={m['weighted_f1']:.4f} macro_f1={m['macro_f1']:.4f} "
              f"accuracy={m['accuracy']:.4f} no_face_clips={m['no_face_clips']}")
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"-> {args.out / 'results.json'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run it on the real dev split**

Run: `uv run python scripts/vision_baseline.py --splits dev`
Expected: one line `dev: n=1108 weighted_f1=0.3… macro_f1=0.2… accuracy=0.3… no_face_clips=4` (values in the published ~30–42% vision-only range; the exact numbers go in the write-up) and `results/vision_baseline/results.json` written.

- [ ] **Step 7: Commit**

```bash
git add src/meld_emotion/training/baselines.py tests/test_baselines.py scripts/vision_baseline.py
git commit -m "Add zero-training vision-only baseline from cached face probabilities"
```

---

### Task 7: Training loop, evaluation, checkpoints, and the run CLI

**Files:**
- Create: `src/meld_emotion/training/train.py`
- Create: `scripts/train_meld.py`
- Modify: `.gitignore` (add `/results/`)
- Test: `tests/test_train_loop.py`

**Interfaces:**
- Consumes: everything from Tasks 1–5: `TrainConfig`, `build_tokenizer`,
  `encode_text`, `build_text_encoder`, `count_parameters`, `load_ok_rows`,
  `infer_feature_dims`, `MeldFeatureDataset`, `collate`, `FusionModel`,
  `class_weights`, `joint_loss`, `compute_metrics`.
- Produces: `resolve_device(name: str) -> str`; `seed_everything(seed:
  int)`; `build_optimizer(model: FusionModel, config) -> torch.optim.AdamW`
  (two param groups: text at `lr_text`, fusion at `lr_fusion`);
  `build_scheduler(optimizer, total_steps: int, warmup_fraction: float) ->
  LambdaLR` (linear warmup then cosine to 0); `evaluate(model, loader,
  device, force_drop_text=False, force_drop_vision=False) -> dict` (keys
  `emotion: compute_metrics dict`, `sentiment: compute_metrics dict`,
  `clips: list[str]`, `emotion_pred: list[int]`, `emotion_probs:
  list[list[float]]`); `train(config: TrainConfig, features_dir: Path,
  out_dir: Path, *, train_split="train", dev_split="dev", test_split=None,
  encode_fn=None, text_encoder=None, pad_id=None, max_train_rows=None,
  log=print) -> dict` which writes `<out_dir>/best.pt`, `<out_dir>/log.jsonl`,
  `<out_dir>/results.json` and returns the results dict (keys `config,
  best_epoch, epochs_run, dev, dev_masked_text_only, dev_masked_vision_only,
  test (if test_split), wall_clock_s, peak_rss_mb (+ peak_gpu_mb on cuda),
  params_total, params_trainable, train_rows, dev_rows`);
  `peak_memory_mb(device: str) -> dict`; `load_checkpoint(path, model)`.

`encode_fn` / `text_encoder` / `pad_id` are injection points: `None` builds
RoBERTa from `config.text_model`; the tests inject the stubs from
`conftest.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_train_loop.py`:

```python
import json

import torch

from conftest import StubTextEncoder, whitespace_encode


def _cfg(**kw):
    from meld_emotion.training.config import TrainConfig
    defaults = dict(d_model=32, n_heads=4, ff_dim=64, n_layers=1, batch_size=8, epochs=2,
                    patience=5, device="cpu", modality_dropout=0.15)
    return TrainConfig(**{**defaults, **kw})


def test_scheduler_warms_up_then_decays_to_zero():
    from meld_emotion.training.train import build_scheduler
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=1.0)
    sched = build_scheduler(opt, total_steps=100, warmup_fraction=0.1)
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"]); opt.step(); sched.step()
    assert lrs[0] < lrs[5] < lrs[9]                 # warming up
    assert abs(lrs[10] - 1.0) < 1e-6                 # peak after warmup
    assert lrs[50] < lrs[10] and lrs[-1] < 0.01      # cosine decay to ~0


def test_optimizer_uses_two_learning_rates(synthetic_features):
    from meld_emotion.training.model import FusionModel
    from meld_emotion.training.train import build_optimizer
    cfg = _cfg(lr_text=1e-5, lr_fusion=1e-3)
    model = FusionModel(cfg, StubTextEncoder(32), face_dim=8, scene_dim=4)
    opt = build_optimizer(model, cfg)
    lrs = sorted(g["lr"] for g in opt.param_groups)
    assert lrs == [1e-5, 1e-3]


def test_train_end_to_end_on_synthetic_features_writes_all_artifacts(synthetic_features, tmp_path):
    from meld_emotion.training.train import load_checkpoint, train
    from meld_emotion.training.model import FusionModel
    root, _, _ = synthetic_features
    out = tmp_path / "run"
    results = train(_cfg(seed=1), root, out, test_split="dev",
                    encode_fn=whitespace_encode, text_encoder=StubTextEncoder(32), pad_id=1, log=lambda *_: None)
    assert (out / "best.pt").exists() and (out / "results.json").exists()
    assert len((out / "log.jsonl").read_text().strip().splitlines()) == 2
    for key in ("dev", "dev_masked_text_only", "dev_masked_vision_only", "test"):
        assert 0.0 <= results[key]["emotion"]["weighted_f1"] <= 1.0
        assert len(results[key]["emotion"]["confusion"]) == 7
    assert results["train_rows"] == 28 and results["dev_rows"] == 14
    assert results["params_trainable"] > 0 and results["wall_clock_s"] > 0 and results["peak_rss_mb"] > 10
    assert results["best_epoch"] in (1, 2) and results["config"]["seed"] == 1
    assert json.load(open(out / "results.json"))["best_epoch"] == results["best_epoch"]
    # the checkpoint reloads into a fresh model and reproduces the dev prediction
    model = FusionModel(_cfg(seed=1), StubTextEncoder(32), face_dim=8, scene_dim=4)
    load_checkpoint(out / "best.pt", model)


def test_train_text_only_and_vision_only_presets_run(synthetic_features, tmp_path):
    from meld_emotion.training.train import train
    root, _, _ = synthetic_features
    r1 = train(_cfg(use_faces=False, use_scene=False, context_k=0, epochs=1), root, tmp_path / "t",
               encode_fn=whitespace_encode, text_encoder=StubTextEncoder(32), pad_id=1, log=lambda *_: None)
    assert "dev_masked_text_only" not in r1                       # single modality: no masked variants
    r2 = train(_cfg(use_text=False, epochs=1), root, tmp_path / "v",
               encode_fn=whitespace_encode, text_encoder=None, pad_id=1, log=lambda *_: None)
    assert r2["dev"]["emotion"]["n"] == 14


def test_max_train_rows_limits_the_training_set(synthetic_features, tmp_path):
    from meld_emotion.training.train import train
    root, _, _ = synthetic_features
    r = train(_cfg(epochs=1), root, tmp_path / "s", max_train_rows=10,
              encode_fn=whitespace_encode, text_encoder=StubTextEncoder(32), pad_id=1, log=lambda *_: None)
    assert r["train_rows"] == 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_train_loop.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meld_emotion.training.train'`

- [ ] **Step 3: Implement**

Create `src/meld_emotion/training/train.py`:

```python
"""Stage 1 training loop (design doc §6): AdamW with two learning rates,
linear warmup + cosine decay, gradient clipping, dev-selected early stopping
on weighted-F1, best checkpoint, JSON-lines log, and a results.json that
records everything the write-up needs (metrics, masked-modality variants,
wall-clock, memory, parameter counts)."""
import functools
import json
import math
import random
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from meld_emotion.data.labels import EMOTIONS, SENTIMENTS
from meld_emotion.training.config import TrainConfig
from meld_emotion.training.dataset import (BucketBatchSampler, MeldFeatureDataset, collate,
                                           infer_feature_dims, load_ok_rows)
from meld_emotion.training.metrics import class_weights, compute_metrics, joint_loss
from meld_emotion.training.model import FusionModel
from meld_emotion.training.text import build_text_encoder, build_tokenizer, count_parameters, encode_text


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def peak_memory_mb(device: str) -> dict:
    """ru_maxrss is bytes on macOS but kilobytes on Linux; report both host and GPU peaks."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out = {"peak_rss_mb": rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024}
    if device == "cuda":
        out["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return out


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_optimizer(model: FusionModel, config: TrainConfig) -> torch.optim.AdamW:
    groups = []
    text_params = [p for p in model.text_parameters() if p.requires_grad]
    if text_params:
        groups.append({"params": text_params, "lr": config.lr_text})
    groups.append({"params": [p for p in model.fusion_parameters() if p.requires_grad], "lr": config.lr_fusion})
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def build_scheduler(optimizer, total_steps: int, warmup_fraction: float):
    warmup = max(1, int(total_steps * warmup_fraction))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _to_device(batch: dict, device: str) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model: FusionModel, loader: DataLoader, device: str,
             force_drop_text: bool = False, force_drop_vision: bool = False) -> dict:
    model.eval()
    clips, e_true, e_pred, e_probs, s_true, s_pred = [], [], [], [], [], []
    for batch in loader:
        batch = _to_device(batch, device)
        out = model(batch, force_drop_text=force_drop_text, force_drop_vision=force_drop_vision)
        probs = torch.softmax(out["emotion_logits"], dim=-1)
        clips += batch["clips"]
        e_true += batch["emotion"].tolist(); e_pred += probs.argmax(-1).tolist(); e_probs += probs.cpu().tolist()
        s_true += batch["sentiment"].tolist(); s_pred += out["sentiment_logits"].argmax(-1).tolist()
    return {"emotion": compute_metrics(np.array(e_true), np.array(e_pred), EMOTIONS),
            "sentiment": compute_metrics(np.array(s_true), np.array(s_pred), SENTIMENTS),
            "clips": clips, "emotion_pred": e_pred, "emotion_probs": e_probs}


def _metrics_only(ev: dict) -> dict:
    return {"emotion": ev["emotion"], "sentiment": ev["sentiment"]}


def load_checkpoint(path: Path, model: FusionModel) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return ckpt


def train(config: TrainConfig, features_dir: Path, out_dir: Path, *,
          train_split: str = "train", dev_split: str = "dev", test_split: str | None = None,
          encode_fn=None, text_encoder=None, pad_id: int | None = None,
          max_train_rows: int | None = None, log=print) -> dict:
    features_dir, out_dir = Path(features_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()   # a warm Modal container can carry over a previous job's peak otherwise
    seed_everything(config.seed)
    started = time.perf_counter()

    # --- text side: real RoBERTa unless stubs are injected ---
    if config.use_text and (encode_fn is None or text_encoder is None or pad_id is None):
        tokenizer = build_tokenizer(config.text_model)
        encode_fn = encode_fn or functools.partial(encode_text, tokenizer, max_length=config.max_text_tokens)
        text_encoder = text_encoder or build_text_encoder(config.text_model, config.text_trainable_layers)
        pad_id = tokenizer.pad_token_id if pad_id is None else pad_id
    if not config.use_text:
        encode_fn = encode_fn or (lambda context, current: {"input_ids": [0], "attention_mask": [1]})
        pad_id = 0 if pad_id is None else pad_id
        text_encoder = None

    # --- data ---
    train_rows = load_ok_rows(features_dir / train_split / "manifest.jsonl")
    if max_train_rows:
        train_rows = train_rows[:max_train_rows]
    dev_rows = load_ok_rows(features_dir / dev_split / "manifest.jsonl")
    face_dim, scene_dim = infer_feature_dims(train_rows, features_dir)
    collate_fn = functools.partial(collate, pad_id=pad_id)

    def make_loader(rows, batch_size, shuffle):
        ds = MeldFeatureDataset(rows, features_dir, config, encode_fn)
        sampler = BucketBatchSampler(ds.lengths, batch_size, shuffle=shuffle, seed=config.seed)
        return DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn, num_workers=0), sampler

    train_loader, train_sampler = make_loader(train_rows, config.batch_size, shuffle=True)
    dev_loader, _ = make_loader(dev_rows, config.batch_size * 2, shuffle=False)

    # --- model / optimisation ---
    model = FusionModel(config, text_encoder, face_dim=face_dim, scene_dim=scene_dim).to(device)
    params_total, params_trainable = count_parameters(model)
    optimizer = build_optimizer(model, config)
    total_steps = config.epochs * len(train_loader)
    scheduler = build_scheduler(optimizer, total_steps, config.warmup_fraction)
    emotion_weight = class_weights(np.array([b for b in (EMOTIONS.index(r["emotion"]) for r in train_rows)]),
                                   len(EMOTIONS), config.class_weight_alpha).to(device)
    log(f"[{config.name}] device={device} train={len(train_rows)} dev={len(dev_rows)} "
        f"params={params_total:,} trainable={params_trainable:,} steps/epoch={len(train_loader)}")

    # --- loop ---
    best_f1, best_epoch, epochs_without_gain = -1.0, 0, 0
    log_path = out_dir / "log.jsonl"
    log_path.write_text("")
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch)
        epoch_started = time.perf_counter()
        losses = []
        for batch in train_loader:
            batch = _to_device(batch, device)
            out = model(batch)
            loss, parts = joint_loss(out, batch, emotion_weight, config.sentiment_lambda, config.label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        dev_eval = evaluate(model, dev_loader, device)
        dev_f1 = dev_eval["emotion"]["weighted_f1"]
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "dev_weighted_f1": dev_f1,
                  "dev_macro_f1": dev_eval["emotion"]["macro_f1"], "dev_sentiment_f1": dev_eval["sentiment"]["weighted_f1"],
                  "epoch_s": time.perf_counter() - epoch_started, "lr": scheduler.get_last_lr()[-1]}
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        log(f"  epoch {epoch}: loss={record['train_loss']:.4f} dev_wF1={dev_f1:.4f} "
            f"dev_mF1={record['dev_macro_f1']:.4f} ({record['epoch_s']:.0f}s)")
        if dev_f1 > best_f1:
            best_f1, best_epoch, epochs_without_gain = dev_f1, epoch, 0
            torch.save({"model_state": model.state_dict(), "config": asdict(config), "epoch": epoch,
                        "dev_weighted_f1": dev_f1, "face_dim": face_dim, "scene_dim": scene_dim}, out_dir / "best.pt")
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= config.patience:
                log(f"  early stop: no dev improvement for {config.patience} epochs")
                break

    # --- final evaluation from the best checkpoint ---
    load_checkpoint(out_dir / "best.pt", model)
    model.to(device)
    results = {"config": asdict(config), "device": device, "best_epoch": best_epoch, "epochs_run": epoch,
               "train_rows": len(train_rows), "dev_rows": len(dev_rows),
               "params_total": params_total, "params_trainable": params_trainable,
               "dev": _metrics_only(evaluate(model, dev_loader, device))}
    if config.use_text and config.use_vision:
        results["dev_masked_text_only"] = _metrics_only(evaluate(model, dev_loader, device, force_drop_vision=True))
        results["dev_masked_vision_only"] = _metrics_only(evaluate(model, dev_loader, device, force_drop_text=True))
    if test_split:
        test_rows = load_ok_rows(features_dir / test_split / "manifest.jsonl")
        test_loader, _ = make_loader(test_rows, config.batch_size * 2, shuffle=False)
        test_eval = evaluate(model, test_loader, device)
        results["test"] = _metrics_only(test_eval)
        results["test_rows"] = len(test_rows)
        with open(out_dir / "test_predictions.jsonl", "w") as f:
            for clip, pred, probs in zip(test_eval["clips"], test_eval["emotion_pred"], test_eval["emotion_probs"]):
                f.write(json.dumps({"clip": clip, "pred": EMOTIONS[pred], "probs": probs}) + "\n")
    results["wall_clock_s"] = time.perf_counter() - started
    results.update(peak_memory_mb(device))
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"[{config.name}] best epoch {best_epoch}: dev wF1={results['dev']['emotion']['weighted_f1']:.4f} "
        f"mF1={results['dev']['emotion']['macro_f1']:.4f} in {results['wall_clock_s'] / 60:.1f} min -> {out_dir}")
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_train_loop.py -v`
Expected: 5 passed (the end-to-end test trains two 1-layer epochs on 28 synthetic rows on CPU — a few seconds)

- [ ] **Step 5: Add the run CLI and ignore `results/`**

Create `scripts/train_meld.py`:

```python
#!/usr/bin/env python3
"""Train one Stage 1 configuration on the cached MELD features and evaluate it.

Usage:
    uv run python scripts/train_meld.py --ablation fusion --seed 0
    uv run python scripts/train_meld.py --ablation text_only_k4 --seed 0 --eval-test
    uv run python scripts/train_meld.py --ablation fusion --train-split dev --epochs 1   # smoke test

Writes results/<ablation>/seed<N>/{best.pt,log.jsonl,results.json}.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # before torch is imported (design doc §6)

import argparse
from pathlib import Path

from meld_emotion.config import FEATURE_CACHE_DIR, REPO_ROOT
from meld_emotion.training.config import ABLATIONS, config_for
from meld_emotion.training.train import train


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ablation", choices=sorted(ABLATIONS), default="fusion")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--class-weight-alpha", type=float, default=None)
    parser.add_argument("--sentiment-lambda", type=float, default=None)
    parser.add_argument("--mask-vision-over-seconds", type=float, default=None)
    parser.add_argument("--device", default=None, help="mps, cpu, or auto")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--dev-split", default="dev")
    parser.add_argument("--eval-test", action="store_true", help="evaluate the best checkpoint on test (once!)")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    overrides = {k: v for k, v in {
        "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size, "n_layers": args.n_layers,
        "class_weight_alpha": args.class_weight_alpha, "sentiment_lambda": args.sentiment_lambda,
        "mask_vision_over_seconds": args.mask_vision_over_seconds, "device": args.device,
    }.items() if v is not None}
    config = config_for(args.ablation, **overrides)
    out_dir = args.out or REPO_ROOT / "results" / config.name / f"seed{config.seed}"
    train(config, FEATURE_CACHE_DIR, out_dir, train_split=args.train_split, dev_split=args.dev_split,
          test_split="test" if args.eval_test else None, max_train_rows=args.max_train_rows)


if __name__ == "__main__":
    main()
```

Append to `.gitignore`:

```
# Training outputs (checkpoints, logs, results.json) -- reproducible via scripts/train_meld.py
/results/
```

- [ ] **Step 6: Smoke-test the CLI on real dev features with RoBERTa (train on dev, 1 epoch)**

Run: `uv run python scripts/train_meld.py --ablation fusion --train-split dev --epochs 1 --out results/smoke`
Expected: prints `[fusion] device=mps train=1108 dev=1108 params=136,119,818 trainable=54,592,010 steps/epoch=35`, one `epoch 1:` line with finite loss (training and evaluating on the same split here, so the F1 is meaningless — this checks that RoBERTa + fusion + MPS run end to end), and `results/smoke/results.json`. Note the printed epoch time: it is the cost of 35 bucketed steps over 1108 rows; multiply by ~9 for a train-split epoch. On an idle M1 expect roughly 1–2 s/step (RoBERTa dominates; cost is linear in padded tokens).

Run: `uv run python scripts/train_meld.py --ablation text_only_k4 --train-split dev --epochs 1 --out results/smoke_text`
Expected: same shape of output with `trainable` ≈ RoBERTa's top 6 layers + heads.

- [ ] **Step 7: Commit**

```bash
git add src/meld_emotion/training/train.py tests/test_train_loop.py scripts/train_meld.py .gitignore
git commit -m "Add Stage 1 training loop with early stopping, checkpoints, masked-modality eval, and run CLI"
```

---

### Task 8: Ablation runner and results table

**Files:**
- Create: `src/meld_emotion/training/results.py`
- Create: `scripts/run_ablations.py`
- Create: `scripts/ablation_table.py`
- Test: `tests/test_results.py`

**Interfaces:**
- Consumes: the `results.json` layout written by `train()` (Task 7) and by
  `scripts/vision_baseline.py` (Task 6); `config_for`, `train`.
- Produces: `collect_results(results_dir: Path) -> list[dict]` (every
  `results/<name>/seed*/results.json`, each dict gaining `name` and `seed`);
  `summarise(results: list[dict]) -> list[dict]` (one row per name: `name,
  n_seeds, dev_weighted_f1_mean/std, dev_macro_f1_mean/std,
  test_weighted_f1_mean/std, test_macro_f1_mean/std` (None when no test),
  `dev_sentiment_f1_mean`, `best_epoch_mean`, `wall_clock_min_mean`);
  `ablation_table(summary: list[dict], baseline: dict | None) -> str`
  (markdown).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_results.py`:

```python
import json


def _write(root, name, seed, dev_w, dev_m, test_w=None):
    d = root / name / f"seed{seed}"
    d.mkdir(parents=True)
    r = {"config": {"name": name, "seed": seed}, "best_epoch": 3, "wall_clock_s": 120.0,
         "dev": {"emotion": {"weighted_f1": dev_w, "macro_f1": dev_m}, "sentiment": {"weighted_f1": 0.7}}}
    if test_w is not None:
        r["test"] = {"emotion": {"weighted_f1": test_w, "macro_f1": test_w - 0.1}, "sentiment": {"weighted_f1": 0.7}}
    (d / "results.json").write_text(json.dumps(r))


def test_collect_and_summarise_average_over_seeds(tmp_path):
    from meld_emotion.training.results import collect_results, summarise
    _write(tmp_path, "fusion", 0, 0.60, 0.40, test_w=0.62)
    _write(tmp_path, "fusion", 1, 0.62, 0.42, test_w=0.64)
    _write(tmp_path, "text_only_k4", 0, 0.58, 0.38)
    results = collect_results(tmp_path)
    assert {(r["name"], r["seed"]) for r in results} == {("fusion", 0), ("fusion", 1), ("text_only_k4", 0)}
    summary = {row["name"]: row for row in summarise(results)}
    assert summary["fusion"]["n_seeds"] == 2
    assert abs(summary["fusion"]["dev_weighted_f1_mean"] - 0.61) < 1e-9
    assert abs(summary["fusion"]["test_weighted_f1_mean"] - 0.63) < 1e-9
    assert summary["text_only_k4"]["test_weighted_f1_mean"] is None
    assert summary["fusion"]["wall_clock_min_mean"] == 2.0


def test_ablation_table_renders_markdown_with_baseline_row(tmp_path):
    from meld_emotion.training.results import ablation_table, collect_results, summarise
    _write(tmp_path, "fusion", 0, 0.60, 0.40)
    baseline = {"dev": {"weighted_f1": 0.35, "macro_f1": 0.20}}
    table = ablation_table(summarise(collect_results(tmp_path)), baseline)
    assert table.startswith("| Model |")
    assert "| fusion |" in table and "vision_only_zero_training" in table
    assert "0.600" in table and "0.350" in table
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_results.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meld_emotion.training.results'`

- [ ] **Step 3: Implement**

Create `src/meld_emotion/training/results.py`:

```python
"""Aggregate results/<name>/seed*/results.json into the ablation table
(design doc §7): mean ± std over seeds, dev and (when evaluated) test."""
import json
import statistics
from pathlib import Path

ORDER = ["vision_only_zero_training", "text_only_k0", "text_only_k4", "vision_only", "fusion",
         "fusion_no_scene", "fusion_no_context", "fusion_no_trackid"]


def collect_results(results_dir: Path) -> list[dict]:
    out = []
    for path in sorted(Path(results_dir).glob("*/seed*/results.json")):
        with open(path) as f:
            r = json.load(f)
        r["name"] = r["config"]["name"]
        r["seed"] = r["config"]["seed"]
        out.append(r)
    return out


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.mean(values), (statistics.pstdev(values) if len(values) > 1 else 0.0)


def summarise(results: list[dict]) -> list[dict]:
    by_name: dict[str, list[dict]] = {}
    for r in results:
        by_name.setdefault(r["name"], []).append(r)
    rows = []
    for name, runs in by_name.items():
        row = {"name": name, "n_seeds": len(runs)}
        for split in ("dev", "test"):
            for metric in ("weighted_f1", "macro_f1"):
                vals = [r[split]["emotion"][metric] for r in runs if split in r]
                row[f"{split}_{metric}_mean"], row[f"{split}_{metric}_std"] = _mean_std(vals)
        row["dev_sentiment_f1_mean"], _ = _mean_std([r["dev"]["sentiment"]["weighted_f1"] for r in runs])
        row["best_epoch_mean"], _ = _mean_std([r["best_epoch"] for r in runs])
        row["wall_clock_min_mean"], _ = _mean_std([r["wall_clock_s"] / 60 for r in runs])
        rows.append(row)
    rows.sort(key=lambda r: ORDER.index(r["name"]) if r["name"] in ORDER else len(ORDER))
    return rows


def _fmt(mean, std=None) -> str:
    if mean is None:
        return "—"
    return f"{mean:.3f}" if std is None else f"{mean:.3f} ± {std:.3f}"


def ablation_table(summary: list[dict], baseline: dict | None = None) -> str:
    lines = ["| Model | seeds | dev wF1 | dev mF1 | test wF1 | test mF1 | dev sent. F1 | best ep. | min/run |",
             "|---|---|---|---|---|---|---|---|---|"]
    if baseline:
        b = baseline.get("dev", {})
        t = baseline.get("test", {})
        lines.append(f"| vision_only_zero_training | — | {_fmt(b.get('weighted_f1'))} | {_fmt(b.get('macro_f1'))} "
                     f"| {_fmt(t.get('weighted_f1'))} | {_fmt(t.get('macro_f1'))} | — | — | 0 |")
    for r in summary:
        lines.append(f"| {r['name']} | {r['n_seeds']} | {_fmt(r['dev_weighted_f1_mean'], r['dev_weighted_f1_std'])} "
                     f"| {_fmt(r['dev_macro_f1_mean'], r['dev_macro_f1_std'])} "
                     f"| {_fmt(r['test_weighted_f1_mean'], r['test_weighted_f1_std'])} "
                     f"| {_fmt(r['test_macro_f1_mean'], r['test_macro_f1_std'])} "
                     f"| {_fmt(r['dev_sentiment_f1_mean'])} | {r['best_epoch_mean']:.1f} | {r['wall_clock_min_mean']:.1f} |")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_results.py -v`
Expected: 2 passed

- [ ] **Step 5: Add the two CLIs**

Create `scripts/run_ablations.py`:

```python
#!/usr/bin/env python3
"""Run several Stage 1 ablations × seeds back to back (each ~30 min on MPS).
Skips any (ablation, seed) whose results.json already exists, so it can be
re-run after an interruption.

Usage:
    uv run python scripts/run_ablations.py --seeds 0            # all presets, one seed
    uv run python scripts/run_ablations.py --ablations fusion text_only_k4 --seeds 0 1 2 --eval-test
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import time

from meld_emotion.config import FEATURE_CACHE_DIR, REPO_ROOT
from meld_emotion.training.config import ABLATIONS, config_for
from meld_emotion.training.train import train


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ablations", nargs="+", default=sorted(ABLATIONS), choices=sorted(ABLATIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    started = time.perf_counter()
    for name in args.ablations:
        for seed in args.seeds:
            out_dir = REPO_ROOT / "results" / name / f"seed{seed}"
            if (out_dir / "results.json").exists():
                print(f"[skip] {name} seed{seed}: results.json exists")
                continue
            overrides = {"seed": seed, **({"epochs": args.epochs} if args.epochs else {})}
            train(config_for(name, **overrides), FEATURE_CACHE_DIR, out_dir,
                  test_split="test" if args.eval_test else None)
    print(f"All done in {(time.perf_counter() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
```

Create `scripts/ablation_table.py`:

```python
#!/usr/bin/env python3
"""Print the Stage 1 ablation table (markdown) from results/, including the
zero-training vision baseline if scripts/vision_baseline.py has been run.

Usage:
    uv run python scripts/ablation_table.py
"""
import argparse
import json
from pathlib import Path

from meld_emotion.config import REPO_ROOT
from meld_emotion.training.results import ablation_table, collect_results, summarise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--out", type=Path, default=None, help="also write the table to this .md file")
    args = parser.parse_args()

    baseline_path = args.results / "vision_baseline" / "results.json"
    baseline = json.load(open(baseline_path)) if baseline_path.exists() else None
    table = ablation_table(summarise(collect_results(args.results)), baseline)
    print(table)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table + "\n")
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Verify the table renders from the Task 7 smoke runs**

Run: `uv run python scripts/ablation_table.py --results results`
Expected: a markdown table with the `vision_only_zero_training` row (from Task 6) and no ablation rows yet (the smoke runs were written to `results/smoke*`, which the `*/seed*/` glob ignores).

- [ ] **Step 7: Run the full offline suite**

Run: `uv run pytest -v`
Expected: every test from the data plan and this plan passes; `test_text.py`'s 3 network tests deselected.

- [ ] **Step 8: Commit**

```bash
git add src/meld_emotion/training/results.py tests/test_results.py scripts/run_ablations.py scripts/ablation_table.py
git commit -m "Add ablation runner and results aggregation into the Stage 1 table"
```

---

### Task 9: Run Stage 1 on Modal

Same `train()`, same `TrainConfig`, same `results.json` — only the GPU
changes. One A10G does a RoBERTa-base epoch in about a minute, and Modal runs
each (ablation, seed) in its own container, so the whole one-seed table takes
~15–20 minutes of wall-clock and a couple of dollars. The cached features
(1.0 GB) are uploaded once to a Modal Volume; results come back to the local
`results/` directory so `scripts/ablation_table.py` (Task 8) works unchanged.

**Files:**
- Create: `scripts/modal_train.py`

**Interfaces:**
- Consumes: `config_for`, `train` (Tasks 1, 7); the `data/meld/features/`
  layout; the local editable `meld_emotion` package (shipped into the image
  by `add_local_python_source`).
- Produces: Modal app `meld-stage1` with function `run(ablation, seed,
  eval_test, epochs)`; Volumes `meld-features` (input), `meld-results`
  (output), `meld-hf-cache` (RoBERTa weights, downloaded once).

- [ ] **Step 1: Write the Modal app**

Create `scripts/modal_train.py`:

```python
#!/usr/bin/env python3
"""Stage 1 trainings on Modal (A10G), ablations × seeds in parallel.

One-time setup -- upload the cached features (~1 GB, a few minutes):
    uv run --with modal modal volume create meld-features
    uv run --with modal modal volume put meld-features data/meld/features /features

Run (each ablation×seed gets its own GPU; ~10-15 min each):
    uv run --with modal modal run scripts/modal_train.py --ablations fusion,text_only_k4 --seeds 0
    uv run --with modal modal run scripts/modal_train.py --ablations fusion --seeds 0,1,2 --eval-test

Fetch results into the local results/ tree, then build the table as usual:
    uv run --with modal modal volume get meld-results / results/ --force
    uv run python scripts/ablation_table.py
"""
import modal

app = modal.App("meld-stage1")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.2", "transformers>=4.40,<6", "scikit-learn>=1.4", "numpy>=1.26",
                 "pillow>=10.0", "opencv-python-headless>=4.9")   # cv2: imported transitively via data.preprocess
    .add_local_python_source("meld_emotion")
)
features = modal.Volume.from_name("meld-features", create_if_missing=True)
results = modal.Volume.from_name("meld-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("meld-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=3 * 3600,
              volumes={"/vol": features, "/out": results, "/root/.cache/huggingface": hf_cache})
def run(ablation: str, seed: int, eval_test: bool = False, epochs: int | None = None) -> dict:
    from meld_emotion.training.config import config_for
    from meld_emotion.training.train import train

    overrides = {"seed": seed, "device": "cuda", **({"epochs": epochs} if epochs else {})}
    out_dir = f"/out/{ablation}/seed{seed}"
    r = train(config_for(ablation, **overrides), "/vol/features", out_dir,
              test_split="test" if eval_test else None)
    results.commit()
    return {"ablation": ablation, "seed": seed, "best_epoch": r["best_epoch"],
            "dev_weighted_f1": r["dev"]["emotion"]["weighted_f1"],
            "dev_macro_f1": r["dev"]["emotion"]["macro_f1"],
            "test_weighted_f1": r.get("test", {}).get("emotion", {}).get("weighted_f1"),
            "minutes": round(r["wall_clock_s"] / 60, 1), "peak_gpu_mb": round(r.get("peak_gpu_mb", 0))}


@app.local_entrypoint()
def main(ablations: str = "fusion", seeds: str = "0", eval_test: bool = False, epochs: int = 0):
    jobs = [(a, int(s), eval_test, epochs or None) for a in ablations.split(",") for s in seeds.split(",")]
    print(f"launching {len(jobs)} run(s) on A10G: {jobs}")
    for out in run.starmap(jobs):
        print(out)
```

- [ ] **Step 2: Upload the features once**

Run: `uv run --with modal modal volume create meld-features`
Expected: `Created volume 'meld-features' ...` (or a message that it exists).
Run: `uv run --with modal modal volume put meld-features data/meld/features /features`
Expected: an upload progress bar over ~13,700 files (~1 GB), then completion.
Run: `uv run --with modal modal volume ls meld-features /features`
Expected: `dev  test  train`.

- [ ] **Step 3: Smoke-run one short job (costs cents)**

Run: `uv run --with modal modal run scripts/modal_train.py --ablations fusion --seeds 0 --epochs 1`
Expected: the image builds (first time ~3 min), RoBERTa downloads into the
cache volume (once), then `[fusion] device=cuda train=9988 dev=1108 ...`,
one `epoch 1:` line taking ~60–90 s, and a printed dict with
`dev_weighted_f1` — a real number this time (trained on train, evaluated
on dev); one epoch typically lands somewhere around 0.55–0.60.

- [ ] **Step 4: Fetch, confirm the local table sees it, then clear the smoke result**

Run: `uv run --with modal modal volume get meld-results / results/ --force`
Expected: `results/fusion/seed0/{best.pt,log.jsonl,results.json}` appear locally.
Run: `uv run python scripts/ablation_table.py`
Expected: a row `| fusion | 1 | 0.5xx ± 0.000 | ...` next to the zero-training baseline row.
The 1-epoch smoke result must not be mistaken for a real run later:
`uv run --with modal modal volume rm meld-results /fusion/seed0 --recursive && rm -rf results/fusion/seed0`

- [ ] **Step 5: Commit**

```bash
git add scripts/modal_train.py
git commit -m "Add Modal A10G runner for Stage 1 ablations (same train(), parallel over ablations x seeds)"
```

---

## Running Stage 1 for real

Prerequisite: `data/meld/features/{train,dev,test}/manifest.jsonl` exist with
`status.ok` = 9988 / 1108 / 2610 (the overnight data job).

```bash
# 1. Zero-training anchor (seconds, local)
uv run python scripts/vision_baseline.py --splits dev test

# 2. The submitted model and the text baselines it must beat -- in parallel on Modal
uv run --with modal modal run scripts/modal_train.py --ablations fusion,text_only_k4,text_only_k0 --seeds 0
uv run --with modal modal volume get meld-results / results/ --force && uv run python scripts/ablation_table.py

# 3. If fusion beats text_only_k4 on dev, the rest of the table (also parallel)
uv run --with modal modal run scripts/modal_train.py --ablations vision_only,fusion_no_scene,fusion_no_context,fusion_no_trackid --seeds 0

# 4. Seeds for the rows that go in the write-up, then test -- once per reported config
uv run --with modal modal run scripts/modal_train.py --ablations fusion,text_only_k4 --seeds 1,2
uv run --with modal modal run scripts/modal_train.py --ablations fusion,text_only_k4,text_only_k0,vision_only --seeds 0 --eval-test
uv run --with modal modal volume get meld-results / results/ --force
uv run python scripts/ablation_table.py --out docs/results/stage1_ablations.md
```

Every command above has a local equivalent (`scripts/run_ablations.py`
with the same `--ablations`/`--seeds`/`--eval-test` flags) that runs on MPS
at ~45–60 min per run — the fallback if Modal is unavailable. `--eval-test`
re-trains the run and overwrites that ablation/seed's results directory,
which is intended: the test number belongs to the run that produced it.

**Expected timings (measure and record):** the Task 7 smoke test prints
the epoch time for 1108 rows; a train-split epoch is ~9× that. RoBERTa is
the cost and it is linear in padded tokens — measured 0.85 s/step at 64
tokens vs 3.9 s/step at 256 on a heavily loaded M1 — which is why the
`BucketBatchSampler` matters (batch max ~59 tokens instead of ~114 with
*k*=4). With RoBERTa's top 6 layers trainable at batch 32 (~313
steps/epoch), plan on roughly 3–6 min per epoch on an idle M1 and 8–12
epochs before early stopping, i.e. ~45–60 min per run and ~6 hours for the
whole table at one seed. If that is too slow for the schedule, the levers
in order are: `text_trainable_layers=4` (less backward), `context_k=2`
(fewer tokens), and moving Stage 1 runs to Modal (an A10G does an epoch in
about a minute) — the code is device-agnostic. Each run's `results.json`
records `wall_clock_s` and `peak_rss_mb` for design doc §9.

**Hyperparameters tuned on dev, in this order, only for the `fusion` preset:**
`class_weight_alpha ∈ {0, 0.5, 1}` (weighted-F1 primary, macro-F1
tie-break), `sentiment_lambda ∈ {0.3, 0.5}`, `n_layers ∈ {2, 4}`. Use
`scripts/train_meld.py --ablation fusion --class-weight-alpha 1.0 --out
results/tune_alpha1/seed0` and compare `results.json`. Then update the
`TrainConfig` defaults so the presets share the winning values, and re-run.

**Reading the table (design doc §7 calibration):** published text-only
weighted-F1 on MELD is ~58–66%, vision-only ~42%, and fusion adds ~1–2
points. The claims the write-up may make are exactly what the table
supports: `text_only_k4 − text_only_k0` is the value of context;
`fusion − text_only_k4` is the fusion gain; the `dev_masked_*` entries in
`results.json` show the same checkpoint with one modality removed. If
`fusion` does not beat `text_only_k4` on dev, that is the finding — do not
tune on test to change it.

**Stage 2 gate (design doc §6):** only once `fusion` is at or above the
published text-only range on dev does the Modal plan (unfreeze the face
ViT's top 4 layers, initialise from `results/fusion/seed0/best.pt`) start.

**Mis-cut clips:** the `mask_vision_over_seconds` switch (`--mask-vision-over-seconds 15`)
treats the 37 over-long clips as vision-missing. It is a cheap dev ablation
after the main table; report it only if it moves the number.

## Self-Review

**Spec coverage.** §4.3 text input with context and left truncation →
Task 2; token set, modality/frame/track embeddings, single-stream pre-LN
encoder, `[FUSE]` readout → Task 4; modality dropout (never both, never an
absent modality) → Task 4; emotion + sentiment heads and joint loss with
`w ∝ (1/freq)^α` → Task 5; weighted-F1/macro-F1/per-class/confusion →
Task 5; §6 Stage 1 (vision frozen from cache, RoBERTa top half in the loop,
AdamW, warmup + cosine, grad clip, seeds, dev-based early stopping, test
once) → Task 7; §7 ablation table incl. the zero-training vision baseline
and masked-modality variants → Tasks 6, 7, 8; §9 wall-clock, memory and
parameter counts → Task 7 `results.json`; §5/§11 mis-cut clips →
`mask_vision_over_seconds` (Tasks 1, 3). Stage 2 (Modal), the response LM,
gloss, demos and latency measurement are separate plans by design.

**Placeholder scan.** No TBD/TODO markers; every step has complete code and
an exact command with expected output.

**Type consistency.** `collate(batch, pad_id)` (Task 3) produces exactly
the keys `FusionModel.forward` (Task 4) and `joint_loss`/`evaluate`
(Tasks 5, 7) read: `input_ids, attention_mask, face_feat, face_mask,
face_frame, face_track, scene_feat, scene_mask, scene_frame, emotion,
sentiment, clips`. The text-encoder contract (`.hidden_size`, call returns
`.last_hidden_state`) is met by `build_text_encoder` (Task 2) and
`StubTextEncoder` (conftest), and consumed by `FusionModel.__init__`/
`forward` (Task 4). `FusionModel.text_parameters()`/`fusion_parameters()`
(Task 4) feed `build_optimizer` (Task 7). `compute_metrics` (Task 5)
returns the `weighted_f1`/`macro_f1`/`confusion` keys that `train()`
(Task 7) and `summarise` (Task 8) read; `train()` writes `results.json`
with `config.name`/`config.seed`/`best_epoch`/`wall_clock_s`/`dev`/`test`
exactly as `collect_results`/`summarise` expect, and `vision_baseline.py`
(Task 6) writes the `dev`/`test` keys that `ablation_table`'s baseline row
reads. `EMOTIONS`/`SENTIMENTS` from `meld_emotion.data.labels` are the
single label order used by `EMOTION_TO_IDX` (Task 3), `evaluate` (Task 7)
and the baseline (Task 6). The synthetic `.npz`/manifest fixture in
`conftest.py` uses the exact array names and manifest keys the data plan
writes.
