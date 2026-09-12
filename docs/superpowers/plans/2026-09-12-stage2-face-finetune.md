# Stage 2: Face-Encoder Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Specialise the face encoder to MELD faces — fine-tune the top 4
layers of the dima806 ViT *inside* the fusion model, initialised from the
best Stage 1 checkpoint, on a Modal A10G — and add the resulting `stage2_*`
rows (and a test-once number) to the ablation table, producing a checkpoint
the replay/live demo can load.

**Architecture:** Stage 1 trained the fusion transformer over *cached*
face features. Stage 2 replaces the cached `face_feat` with a live forward
through the face ViT: a `MeldCropDataset` yields each clip's face crops as
uint8 tensors (JPEGs from the preprocessing pass, capped per clip,
augmented in training), a `TrainableFaceEncoder` runs the frozen bottom 8
layers under `no_grad` and the top 4 with gradients, and a `Stage2Model`
scatters the resulting CLS features into the `(B, Fmax, 768)` slot the
unchanged `FusionModel` already consumes. Everything else — text encoder,
scene features, loss, metrics, early stopping, results.json — is reused
from Stage 1. Runs on Modal with the crops shipped as one tarball per split
and bf16 autocast on CUDA.

**Tech Stack:** PyTorch, Hugging Face `transformers` 5.x (`ViTModel` has
`embeddings`, `layers`, `layernorm` — no `.encoder`), torchvision
transforms, Modal.

## Global Constraints

- Design doc §6 (Stage 2): unfreeze the **top 4 layers** of the face ViT,
  initialise fusion/projectors/heads/text from the Stage 1 checkpoint,
  lower learning rate; kept only if it beats Stage 1 on dev. Inference
  path and parameter count are unchanged (the same 86M ViT, now adapted).
- Base preset = whichever Stage 1 row won on dev (`fusion`,
  `fusion_no_scene`, or `fusion_faces_only` — read
  `docs/results/stage1_ablations.md`). The plan is parameterised on it:
  `--base <preset>` everywhere; the Stage 2 preset is `stage2_<base>`.
- Generalisation is the point (the live demo must work on faces that are
  not the six *Friends* actors): most of the ViT stays frozen, the face
  learning rate is 1e-5, and training crops are augmented (random resized
  crop, horizontal flip, colour jitter). Evaluation crops are not.
- Crops per clip are heavy-tailed (train mean 19.7, p99 94, max 431):
  cap at **64 per clip** (uniform subsample preserving frame order; affects
  3.5 % of train clips) and use **batch size 16**. Frozen layers run under
  `no_grad`; only the top 4 layers keep activations.
- The face processor's exact preprocessing, reproduced with torchvision:
  resize to 224×224 (already 224² on disk), scale to [0, 1], normalise
  mean 0.5 / std 0.5 per channel → values in [-1, 1]. Crops on disk are
  JPEGs written by OpenCV in BGR order; `PIL.Image.open` returns RGB —
  correct, no swap needed.
- After Stage 2 the ViT's **original 7-way head is stale** (it was trained
  on the old CLS). Nothing in training uses it. The demo's provisional
  state must come from the fusion model with text masked (`force_drop_text`),
  not from `FaceEmotionEncoder.encode_batch()["probs"]` — see the briefing
  note update in Task 7.
- Same rules as Stage 1: weighted-F1 primary, dev selects, test once with
  `--eval-test`; `uv add` only; `uv run pytest` offline in seconds;
  `network` marker for anything that downloads; every code block runnable.
- Inputs this plan consumes: `data/meld/preprocessed/<split>/dia<D>_utt<U>/`
  with `metadata.json` (`frames[].faces[].path`, in the same order as the
  cache's `face_frame_idx`/`face_track_ids` rows) and the face JPEGs;
  `data/meld/features/<split>/manifest.jsonl` + `.npz` (scene features,
  frame/track indices); `results/<base>/seed0/best.pt` (Stage 1) — on Modal
  it is already at `/out/<base>/seed0/best.pt` on the `meld-results` Volume.

---

## File Structure

```
src/meld_emotion/training/
  config.py            # MODIFIED: face_trainable_layers, lr_face, max_faces_per_clip, face_augment, init_from, loader_workers; stage2_config()
  crops.py             # MeldCropDataset (crops + reuse of MeldFeatureDataset), transforms, collate_crops (Task 1)
  face_encoder.py      # TrainableFaceEncoder (no-grad bottom / grad top), scatter_faces (Task 2)
  stage2_model.py      # Stage2Model = face encoder -> face_feat -> FusionModel; init_from_stage1 (Task 3)
  train_stage2.py      # train_stage2(): 3 LR groups, bf16 autocast on cuda, checkpoint with face weights (Task 4)
  results.py           # MODIFIED: stage2_* rows in ORDER (Task 4)
src/meld_emotion/vision/encoders.py   # MODIFIED: FaceEmotionEncoder(weights_path=...) for the demo (Task 6)
scripts/
  pack_crops.py        # one tar per split of face JPEGs + metadata.json (Task 5)
  train_stage2.py      # local run/smoke (Task 4)
  modal_stage2.py      # Modal runner: extract crops tar, train, commit (Task 5)
tests/
  test_crops.py, test_face_encoder.py, test_stage2_model.py, test_train_stage2.py, test_encoders.py (MODIFIED)
```

---

### Task 1: Crop dataset, transforms, and collate

**Files:**
- Modify: `src/meld_emotion/training/config.py`
- Create: `src/meld_emotion/training/crops.py`
- Test: `tests/test_crops.py`

**Interfaces:**
- Consumes: `MeldFeatureDataset`, `collate`, `load_ok_rows` (Stage 1 dataset);
  `clip_dir_for` (`meld_emotion.data.preprocess`); `TrainConfig`.
- Produces: new `TrainConfig` fields `face_trainable_layers: int = 0`,
  `lr_face: float = 1e-5`, `max_faces_per_clip: int = 64`, `face_augment:
  bool = True`, `init_from: str | None = None`, `loader_workers: int = 0`,
  `amp_bf16: bool = True`; `stage2_config(base: str, **overrides) ->
  TrainConfig` (name `stage2_<base>`, `face_trainable_layers=4`,
  `batch_size=16`, `epochs=8`, `lr_text=1e-5`, `init_from=
  "results/<base>/seed0/best.pt"`); `train_transform()` / `eval_transform()`
  (PIL → uint8 CHW tensor, 224²); `subsample_faces(n: int, cap: int) ->
  np.ndarray` (sorted indices); `MeldCropDataset(rows, preprocessed_dir,
  cache_dir, config, encode_fn, train: bool)` whose items are the Stage 1
  item dict **plus** `face_pixels (N, 3, 224, 224) uint8` with `face_feat`
  removed and `face_frame`/`face_track` subsampled consistently, and
  attribute `.lengths`; `collate_crops(batch, pad_id) -> dict` = Stage 1
  `collate` output (with `face_feat` as an all-zero `(B, Fmax, 1)`
  placeholder) **plus** `face_pixels (Ntot, 3, 224, 224) uint8`,
  `face_batch_idx (Ntot,) long`, `face_slot (Ntot,) long`.

- [ ] **Step 1: Extend `TrainConfig` and add `stage2_config`**

In `src/meld_emotion/training/config.py`, add these fields to `TrainConfig`
after the `# --- optimisation ---` block (before `def __post_init__`):

```python
    # --- Stage 2: face encoder in the loop (design doc §6). 0 = Stage 1 (cached features) ---
    face_trainable_layers: int = 0          # top N of the 12 ViT layers (+ final LayerNorm)
    lr_face: float = 1e-5
    max_faces_per_clip: int = 64            # p99 is ~95; uniform subsample keeps frame coverage
    face_augment: bool = True               # random resized crop / flip / colour jitter on train crops
    init_from: str | None = None            # Stage 1 checkpoint to initialise everything but the ViT
    loader_workers: int = 0                 # DataLoader workers for JPEG decoding (Modal: 6)
    amp_bf16: bool = True                   # bf16 autocast on CUDA only
```

and this function at the end of the file:

```python
def stage2_config(base: str, **overrides) -> TrainConfig:
    """Stage 2 preset derived from a Stage 1 preset: same modalities, the face
    ViT's top 4 layers trainable, initialised from that preset's seed-0 run."""
    if base not in ABLATIONS or base.startswith("text_only"):
        raise KeyError(f"Stage 2 needs a Stage 1 preset that uses faces; got {base!r}")
    defaults = dict(name=f"stage2_{base}", face_trainable_layers=4, batch_size=16, epochs=8,
                    lr_text=1e-5, init_from=f"results/{base}/seed0/best.pt")
    return replace(ABLATIONS[base], **{**defaults, **overrides})
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_crops.py`:

```python
import json

import numpy as np
import pytest
import torch
from PIL import Image

from conftest import whitespace_encode


def write_synthetic_crops(features_root, preprocessed_root, split):
    """For every ok row in <features_root>/<split>/manifest.jsonl, write the
    preprocessed clip dir the data plan would have written: metadata.json
    with one face entry per cached face row (same order) and a 224x224 JPEG
    per face whose red channel encodes its index in steps of 20 (JPEG
    compression perturbs values by a few units, so tests check ±4)."""
    from meld_emotion.data.manifest import read_manifest
    from meld_emotion.data.preprocess import clip_dir_for
    for row in read_manifest(features_root / split / "manifest.jsonl"):
        if row["status"] != "ok":
            continue
        z = np.load(features_root / row["feature_path"])
        clip_dir = clip_dir_for(preprocessed_root, split, row["dialogue_id"], row["utterance_id"])
        clip_dir.mkdir(parents=True, exist_ok=True)
        frames = {}
        for i, (frame_idx, track_id) in enumerate(zip(z["face_frame_idx"].tolist(), z["face_track_ids"].tolist())):
            name = f"frame{frame_idx}_face{track_id}.jpg"
            Image.new("RGB", (224, 224), ((i * 20) % 256, 40, 200)).save(clip_dir / name, quality=95)
            frames.setdefault(frame_idx, []).append({"track_id": track_id, "box": [0, 0, 1, 1], "score": 0.9, "path": name})
        meta = {"status": "ok", "frames": [{"sample_index": f, "faces": frames.get(f, []), "scene_path": f"frame{f}_scene.jpg",
                                             "shot_cut": False} for f in sorted(set(z["scene_frame_idx"].tolist()) | set(frames))]}
        with open(clip_dir / "metadata.json", "w") as fh:
            json.dump(meta, fh)


@pytest.fixture
def synthetic_crops(synthetic_features, tmp_path):
    root, train, dev = synthetic_features
    pre = tmp_path / "preprocessed"
    write_synthetic_crops(root, pre, "train")
    write_synthetic_crops(root, pre, "dev")
    return root, pre


def test_stage2_config_derives_from_a_face_preset():
    from meld_emotion.training.config import stage2_config
    c = stage2_config("fusion", seed=3)
    assert (c.name, c.face_trainable_layers, c.batch_size, c.seed) == ("stage2_fusion", 4, 16, 3)
    assert c.use_faces and c.use_text and c.init_from == "results/fusion/seed0/best.pt"
    with pytest.raises(KeyError):
        stage2_config("text_only_k4")


def test_transforms_produce_uint8_chw_224():
    from meld_emotion.training.crops import eval_transform, train_transform
    img = Image.new("RGB", (224, 224), (10, 20, 30))
    for t in (train_transform(), eval_transform()):
        x = t(img)
        assert x.dtype == torch.uint8 and x.shape == (3, 224, 224)
    assert torch.equal(eval_transform()(img)[:, 0, 0], torch.tensor([10, 20, 30], dtype=torch.uint8))


def test_subsample_faces_is_sorted_covers_the_range_and_is_identity_under_cap():
    from meld_emotion.training.crops import subsample_faces
    assert subsample_faces(5, cap=64).tolist() == [0, 1, 2, 3, 4]
    idx = subsample_faces(431, cap=64)
    assert len(idx) == 64 and idx[0] == 0 and idx[-1] == 430 and np.all(np.diff(idx) > 0)


def test_crop_dataset_items_carry_pixels_aligned_with_frame_and_track(synthetic_crops):
    from meld_emotion.training.config import stage2_config
    from meld_emotion.training.crops import MeldCropDataset
    from meld_emotion.training.dataset import load_ok_rows
    root, pre = synthetic_crops
    rows = load_ok_rows(root / "train" / "manifest.jsonl")
    ds = MeldCropDataset(rows, pre, root, stage2_config("fusion"), whitespace_encode, train=False)
    item = ds[6]                                         # dia1_utt2: 2 faces x 3 frames
    assert item["face_pixels"].shape == (6, 3, 224, 224) and item["face_pixels"].dtype == torch.uint8
    assert "face_feat" not in item
    assert item["face_frame"].tolist() == [0, 0, 1, 1, 2, 2] and item["face_track"].tolist() == [0, 1, 0, 1, 0, 1]
    assert abs(item["face_pixels"][3, 0, 0, 0].item() - 60) <= 4   # 4th crop: red = 3 * 20, within JPEG error
    assert item["scene_feat"].shape[0] == 3 and ds.lengths[6] == len(item["input_ids"])
    empty = ds[4]                                        # dia1_utt0: 0 faces
    assert empty["face_pixels"].shape == (0, 3, 224, 224)


def test_crop_dataset_caps_faces_per_clip_consistently(synthetic_crops):
    from meld_emotion.training.config import stage2_config
    from meld_emotion.training.crops import MeldCropDataset
    from meld_emotion.training.dataset import load_ok_rows
    root, pre = synthetic_crops
    rows = load_ok_rows(root / "train" / "manifest.jsonl")
    ds = MeldCropDataset(rows, pre, root, stage2_config("fusion", max_faces_per_clip=4), whitespace_encode, train=False)
    item = ds[6]
    assert item["face_pixels"].shape[0] == 4 == len(item["face_frame"]) == len(item["face_track"])
    assert item["face_frame"].tolist() == sorted(item["face_frame"].tolist())


def test_collate_crops_flattens_faces_with_batch_and_slot_indices(synthetic_crops):
    from meld_emotion.training.config import stage2_config
    from meld_emotion.training.crops import MeldCropDataset, collate_crops
    from meld_emotion.training.dataset import load_ok_rows
    root, pre = synthetic_crops
    rows = load_ok_rows(root / "train" / "manifest.jsonl")
    ds = MeldCropDataset(rows, pre, root, stage2_config("fusion"), whitespace_encode, train=False)
    batch = collate_crops([ds[4], ds[5], ds[6]], pad_id=1)   # 0, 3, 6 faces
    assert batch["face_pixels"].shape == (9, 3, 224, 224)
    assert batch["face_batch_idx"].tolist() == [1, 1, 1, 2, 2, 2, 2, 2, 2]
    assert batch["face_slot"].tolist() == [0, 1, 2, 0, 1, 2, 3, 4, 5]
    assert batch["face_mask"].sum(1).tolist() == [0, 3, 6] and batch["face_feat"].shape == (3, 6, 1)
    assert batch["scene_feat"].shape[0] == 3 and batch["emotion"].shape == (3,)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_crops.py -v`
Expected: `test_stage2_config_derives_from_a_face_preset` FAILS with `ImportError: cannot import name 'stage2_config'`; the rest FAIL with `ModuleNotFoundError: No module named 'meld_emotion.training.crops'`

- [ ] **Step 4: Implement**

Create `src/meld_emotion/training/crops.py`:

```python
"""Stage 2 data: the Stage 1 item (text, scene features, frame/track ids)
plus each clip's face crops as uint8 tensors, read from the preprocessing
pass's JPEGs, capped per clip and augmented in training."""
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from meld_emotion.data.preprocess import clip_dir_for
from meld_emotion.training.config import TrainConfig
from meld_emotion.training.dataset import MeldFeatureDataset, collate

CROP_SIZE = 224


def train_transform():
    """Mild augmentation so the ViT specialises to expressions, not to the
    six actors' faces (the live demo must generalise past them)."""
    return transforms.Compose([
        transforms.RandomResizedCrop(CROP_SIZE, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.PILToTensor(),                       # uint8 CHW; normalised on the device
    ])


def eval_transform():
    return transforms.Compose([transforms.Resize((CROP_SIZE, CROP_SIZE)), transforms.PILToTensor()])


def subsample_faces(n: int, cap: int) -> np.ndarray:
    if n <= cap:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, cap).round().astype(np.int64))


def face_crop_paths(clip_dir: Path) -> list[Path]:
    """Face JPEG paths in metadata order — the same order as the cached
    face_frame_idx / face_track_ids rows (the cache was built by iterating
    frames, then faces)."""
    with open(clip_dir / "metadata.json") as f:
        meta = json.load(f)
    return [clip_dir / face["path"] for frame in meta["frames"] for face in frame["faces"]]


class MeldCropDataset(Dataset):
    def __init__(self, rows: list[dict], preprocessed_dir: Path, cache_dir: Path, config: TrainConfig,
                 encode_fn: Callable[[str, str], dict[str, list[int]]], train: bool):
        self.base = MeldFeatureDataset(rows, cache_dir, config, encode_fn)   # text, scene, frame/track ids
        self.config = config
        self.transform = train_transform() if (train and config.face_augment) else eval_transform()
        self.crop_paths = [face_crop_paths(clip_dir_for(preprocessed_dir, r["split"], r["dialogue_id"], r["utterance_id"]))
                           for r in rows]
        for i, (paths, feats) in enumerate(zip(self.crop_paths, self.base.features)):
            if len(paths) != len(feats["face_frame_idx"]):
                raise ValueError(f"{rows[i]['feature_path']}: {len(paths)} crops on disk vs "
                                 f"{len(feats['face_frame_idx'])} cached face rows")
        self.lengths = self.base.lengths

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> dict:
        from PIL import Image
        item = self.base[i]
        del item["face_feat"]
        keep = subsample_faces(len(self.crop_paths[i]), self.config.max_faces_per_clip)
        if not self.config.use_faces:
            keep = keep[:0]
        item["face_frame"] = item["face_frame"][keep]
        item["face_track"] = item["face_track"][keep]
        pixels = [self.transform(Image.open(self.crop_paths[i][j]).convert("RGB")) for j in keep]
        item["face_pixels"] = torch.stack(pixels) if pixels else torch.zeros((0, 3, CROP_SIZE, CROP_SIZE), dtype=torch.uint8)
        return item


def collate_crops(batch: list[dict], pad_id: int) -> dict:
    """Stage 1 collate (with a 1-wide face_feat placeholder that Stage2Model
    overwrites) plus every crop in the batch flattened to one tensor, with
    the (sample, slot) each crop belongs to."""
    placeholder = [{**b, "face_feat": torch.zeros((b["face_pixels"].shape[0], 1))} for b in batch]
    out = collate(placeholder, pad_id)
    out["face_pixels"] = torch.cat([b["face_pixels"] for b in batch]) if batch else torch.zeros((0, 3, CROP_SIZE, CROP_SIZE), dtype=torch.uint8)
    out["face_batch_idx"] = torch.cat([torch.full((b["face_pixels"].shape[0],), bi, dtype=torch.long) for bi, b in enumerate(batch)])
    out["face_slot"] = torch.cat([torch.arange(b["face_pixels"].shape[0], dtype=torch.long) for b in batch])
    return out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_crops.py tests/test_train_config.py -v`
Expected: 6 + 8 passed (the existing config tests still pass with the new fields)

- [ ] **Step 6: Commit**

```bash
git add src/meld_emotion/training/config.py src/meld_emotion/training/crops.py tests/test_crops.py
git commit -m "Add Stage 2 crop dataset with per-clip cap, augmentation, and flattened-face collate"
```

---

### Task 2: Trainable face encoder (no-grad bottom, grad top)

**Files:**
- Create: `src/meld_emotion/training/face_encoder.py`
- Test: `tests/test_face_encoder.py`

**Interfaces:**
- Consumes: `FACE_MODEL_ID` from `meld_emotion.vision.encoders`.
- Produces: `normalize_pixels(x_uint8: Tensor) -> Tensor` (float in [-1, 1],
  the face processor's normalisation); class `TrainableFaceEncoder(model_id:
  str = FACE_MODEL_ID, trainable_layers: int = 4)` with `.feature_dim`
  (768), `.forward(pixels_uint8 (N,3,224,224)) -> (N, 768)` (empty input →
  `(0, 768)`), `.trainable_parameters()`, `.export_state() -> dict`
  (the whole ViT + classifier state dict, loadable by
  `FaceEmotionEncoder.model`); `scatter_faces(feats (Ntot, D), batch_idx,
  slot, B: int, fmax: int) -> (B, fmax, D)`.

Verified against transformers 5.17: `ViTModel` children are `embeddings`,
`layers` (ModuleList of 12), `layernorm`; a manual loop over them equals
`vit(pixel_values=...).last_hidden_state`; layer outputs are tensors (handle
a tuple for older versions). Top 4 layers + LayerNorm = 28,353,024 params.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_face_encoder.py`:

```python
import pytest
import torch


def test_normalize_pixels_maps_uint8_to_minus_one_one():
    from meld_emotion.training.face_encoder import normalize_pixels
    x = torch.tensor([[[[0, 128, 255]]]], dtype=torch.uint8).expand(1, 3, 1, 3)
    y = normalize_pixels(x)
    assert y.dtype == torch.float32
    assert torch.allclose(y[0, 0, 0], torch.tensor([-1.0, 128 / 127.5 - 1, 1.0]), atol=1e-6)


def test_scatter_faces_places_each_crop_in_its_sample_and_slot():
    from meld_emotion.training.face_encoder import scatter_faces
    feats = torch.arange(1, 5, dtype=torch.float32)[:, None].expand(4, 2)        # 4 crops, D=2
    out = scatter_faces(feats, torch.tensor([0, 0, 2, 2]), torch.tensor([0, 1, 0, 1]), B=3, fmax=3)
    assert out.shape == (3, 3, 2)
    assert out[0, 0, 0] == 1 and out[0, 1, 0] == 2 and out[2, 1, 0] == 4
    assert torch.all(out[1] == 0) and torch.all(out[0, 2] == 0)
    assert scatter_faces(torch.zeros(0, 2), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long), B=2, fmax=0).shape == (2, 0, 2)


@pytest.mark.network
def test_trainable_face_encoder_matches_the_pretrained_forward_and_trains_only_the_top():
    from meld_emotion.training.face_encoder import TrainableFaceEncoder, normalize_pixels
    from meld_emotion.vision.encoders import FaceEmotionEncoder
    enc = TrainableFaceEncoder(trainable_layers=4).eval()
    ref = FaceEmotionEncoder(device="cpu")
    pixels = torch.randint(0, 256, (2, 3, 224, 224), dtype=torch.uint8)
    with torch.no_grad():
        ours = enc(pixels)
        theirs = ref.model.vit(pixel_values=normalize_pixels(pixels)).last_hidden_state[:, 0]
    assert ours.shape == (2, 768) and torch.allclose(ours, theirs, atol=1e-4)
    assert sum(p.numel() for p in enc.trainable_parameters()) == 28_353_024
    enc.train()
    enc(pixels).sum().backward()
    assert enc.vit.layers[11].attention.q_proj.weight.grad is not None
    assert enc.vit.layers[7].attention.q_proj.weight.grad is None
    assert enc(torch.zeros((0, 3, 224, 224), dtype=torch.uint8)).shape == (0, 768)
    state = enc.export_state()
    ref.model.load_state_dict(state)                     # the demo's encoder can load Stage 2 weights
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_face_encoder.py -v`
Expected: 2 FAIL with `ModuleNotFoundError: No module named 'meld_emotion.training.face_encoder'`, 1 deselected

- [ ] **Step 3: Implement**

Create `src/meld_emotion/training/face_encoder.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_face_encoder.py -v` → Expected: 2 passed, 1 deselected
Run: `uv run pytest tests/test_face_encoder.py -m network -v` → Expected: 1 passed (weights are cached)

- [ ] **Step 5: Commit**

```bash
git add src/meld_emotion/training/face_encoder.py tests/test_face_encoder.py
git commit -m "Add TrainableFaceEncoder: frozen-bottom/trainable-top ViT with scatter into fusion slots"
```

---

### Task 3: Stage2Model and Stage 1 initialisation

**Files:**
- Create: `src/meld_emotion/training/stage2_model.py`
- Test: `tests/test_stage2_model.py`

**Interfaces:**
- Consumes: `FusionModel` (Stage 1), `scatter_faces`, `TrainableFaceEncoder`
  contract (`.feature_dim`, `forward(uint8) -> (N, D)`,
  `.trainable_parameters()`, `.export_state()`), `load_checkpoint` (Stage 1).
- Produces: `Stage2Model(fusion: FusionModel, face_encoder: nn.Module)` with
  `forward(batch, force_drop_text=False, force_drop_vision=False) -> dict`
  (same output as `FusionModel`), `.text_parameters()`,
  `.fusion_parameters()`, `.face_parameters()`, `.modality_dropout_masks`
  (delegated); `init_fusion_from_stage1(fusion: FusionModel, path: Path)
  -> dict` (loads the Stage 1 `model_state` into the fusion model — text
  encoder, projectors, transformer, heads — and returns the checkpoint dict).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stage2_model.py`:

```python
import torch
import torch.nn as nn

from conftest import StubTextEncoder, whitespace_encode
from test_crops import write_synthetic_crops


class StubFaceEncoder(nn.Module):
    """Same contract as TrainableFaceEncoder, tiny: mean colour -> feature."""
    def __init__(self, dim=8):
        super().__init__()
        self.feature_dim = dim
        self.proj = nn.Linear(3, dim)

    def trainable_parameters(self):
        return self.parameters()

    def export_state(self):
        return {k: v.detach().cpu() for k, v in self.state_dict().items()}

    def forward(self, pixels_uint8):
        if pixels_uint8.shape[0] == 0:
            return torch.zeros((0, self.feature_dim))
        return self.proj(pixels_uint8.float().mean(dim=(2, 3)) / 255.0)


def _cfg(**kw):
    from meld_emotion.training.config import stage2_config
    return stage2_config("fusion", d_model=32, n_heads=4, ff_dim=64, n_layers=1, **kw)


def _batch(synthetic_features, tmp_path, cfg, idxs=(4, 5, 6)):
    from meld_emotion.training.crops import MeldCropDataset, collate_crops
    from meld_emotion.training.dataset import load_ok_rows
    root, train, _ = synthetic_features
    pre = tmp_path / "pre"
    write_synthetic_crops(root, pre, "train")
    ds = MeldCropDataset(load_ok_rows(train), pre, root, cfg, whitespace_encode, train=False)
    return collate_crops([ds[i] for i in idxs], pad_id=1)


def test_stage2_model_runs_the_face_encoder_and_the_fusion_model(synthetic_features, tmp_path):
    from meld_emotion.training.model import FusionModel
    from meld_emotion.training.stage2_model import Stage2Model
    cfg = _cfg()
    fusion = FusionModel(cfg, StubTextEncoder(32), face_dim=8, scene_dim=4)
    model = Stage2Model(fusion, StubFaceEncoder(8)).eval()
    out = model(_batch(synthetic_features, tmp_path, cfg))
    assert out["emotion_logits"].shape == (3, 7) and torch.isfinite(out["emotion_logits"]).all()
    assert not torch.allclose(out["emotion_logits"], model(_batch(synthetic_features, tmp_path, cfg), force_drop_vision=True)["emotion_logits"])


def test_stage2_parameter_groups_partition_all_parameters(synthetic_features, tmp_path):
    from meld_emotion.training.model import FusionModel
    from meld_emotion.training.stage2_model import Stage2Model
    cfg = _cfg()
    model = Stage2Model(FusionModel(cfg, StubTextEncoder(32), face_dim=8, scene_dim=4), StubFaceEncoder(8))
    groups = [{id(p) for p in g} for g in (model.text_parameters(), model.fusion_parameters(), model.face_parameters())]
    assert all(groups) and not (groups[0] & groups[1]) and not (groups[1] & groups[2]) and not (groups[0] & groups[2])
    assert groups[0] | groups[1] | groups[2] == {id(p) for p in model.parameters()}


def test_init_fusion_from_stage1_loads_the_stage1_weights(synthetic_features, tmp_path):
    from meld_emotion.training.config import TrainConfig
    from meld_emotion.training.model import FusionModel
    from meld_emotion.training.stage2_model import init_fusion_from_stage1
    from meld_emotion.training.train import train
    root, _, _ = synthetic_features
    s1 = TrainConfig(d_model=32, n_heads=4, ff_dim=64, n_layers=1, batch_size=8, epochs=1, device="cpu")
    train(s1, root, tmp_path / "s1", encode_fn=whitespace_encode, text_encoder=StubTextEncoder(32), pad_id=1, log=lambda *_: None)
    fresh = FusionModel(_cfg(), StubTextEncoder(32), face_dim=8, scene_dim=4)
    before = fresh.emotion_head.weight.detach().clone()
    ckpt = init_fusion_from_stage1(fresh, tmp_path / "s1" / "best.pt")
    assert ckpt["epoch"] == 1 and not torch.equal(before, fresh.emotion_head.weight)
    trained = torch.load(tmp_path / "s1" / "best.pt", weights_only=False)["model_state"]["emotion_head.weight"]
    assert torch.equal(fresh.emotion_head.weight, trained)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_stage2_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meld_emotion.training.stage2_model'`

- [ ] **Step 3: Implement**

Create `src/meld_emotion/training/stage2_model.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_stage2_model.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/meld_emotion/training/stage2_model.py tests/test_stage2_model.py
git commit -m "Add Stage2Model (face encoder -> fusion) and Stage 1 initialisation"
```

---

### Task 4: Stage 2 training loop, checkpoint, local CLI

**Files:**
- Create: `src/meld_emotion/training/train_stage2.py`
- Create: `scripts/train_stage2.py`
- Modify: `src/meld_emotion/training/results.py`
- Test: `tests/test_train_stage2.py`

**Interfaces:**
- Consumes: from Stage 1 `train.py`: `resolve_device`, `seed_everything`,
  `build_scheduler`, `evaluate`, `_metrics_only`, `peak_memory_mb`,
  `_to_device`; `class_weights`, `joint_loss`; `build_tokenizer`,
  `encode_text`, `build_text_encoder`, `count_parameters`; `load_ok_rows`,
  `infer_feature_dims`, `BucketBatchSampler`; `MeldCropDataset`,
  `collate_crops`; `FusionModel`, `Stage2Model`, `init_fusion_from_stage1`,
  `TrainableFaceEncoder`.
- Produces: `build_stage2_optimizer(model: Stage2Model, config) ->
  AdamW` (groups: face at `lr_face`, text at `lr_text`, fusion at
  `lr_fusion`); `train_stage2(config, features_dir, preprocessed_dir,
  out_dir, *, train_split="train", dev_split="dev", test_split=None,
  encode_fn=None, text_encoder=None, face_encoder=None, pad_id=None,
  init_from=None, max_train_rows=None, log=print, on_epoch_end=None) ->
  dict` — same `results.json` layout as Stage 1 plus `"stage": 2`,
  `"base_checkpoint"`, `"face_trainable_params"`; `best.pt` additionally
  holds `"face_encoder_state"` (the whole ViT + classifier state dict) so the
  demo can load it into `FaceEmotionEncoder`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_train_stage2.py`:

```python
import json

import torch

from conftest import StubTextEncoder, whitespace_encode
from test_crops import write_synthetic_crops
from test_stage2_model import StubFaceEncoder


def _cfg(**kw):
    from meld_emotion.training.config import stage2_config
    defaults = dict(d_model=32, n_heads=4, ff_dim=64, n_layers=1, batch_size=8, epochs=2, patience=5,
                    device="cpu", face_augment=True, loader_workers=0)
    return stage2_config("fusion", **{**defaults, **kw})


def _stage1(synthetic_features, tmp_path):
    from meld_emotion.training.config import TrainConfig
    from meld_emotion.training.train import train
    root, _, _ = synthetic_features
    s1 = TrainConfig(d_model=32, n_heads=4, ff_dim=64, n_layers=1, batch_size=8, epochs=1, device="cpu")
    train(s1, root, tmp_path / "s1", encode_fn=whitespace_encode, text_encoder=StubTextEncoder(32), pad_id=1, log=lambda *_: None)
    return tmp_path / "s1" / "best.pt"


def test_stage2_optimizer_has_three_learning_rates(synthetic_features):
    from meld_emotion.training.model import FusionModel
    from meld_emotion.training.stage2_model import Stage2Model
    from meld_emotion.training.train_stage2 import build_stage2_optimizer
    cfg = _cfg(lr_face=1e-5, lr_text=2e-5, lr_fusion=1e-3)
    model = Stage2Model(FusionModel(cfg, StubTextEncoder(32), face_dim=8, scene_dim=4), StubFaceEncoder(8))
    assert sorted(g["lr"] for g in build_stage2_optimizer(model, cfg).param_groups) == [1e-5, 2e-5, 1e-3]


def test_train_stage2_end_to_end_initialises_from_stage1_and_writes_artifacts(synthetic_features, tmp_path):
    from meld_emotion.training.train_stage2 import train_stage2
    root, _, _ = synthetic_features
    pre = tmp_path / "pre"
    write_synthetic_crops(root, pre, "train")
    write_synthetic_crops(root, pre, "dev")
    s1 = _stage1(synthetic_features, tmp_path)
    out = tmp_path / "s2"
    seen = []
    results = train_stage2(_cfg(seed=1), root, pre, out, test_split="dev", init_from=s1,
                           encode_fn=whitespace_encode, text_encoder=StubTextEncoder(32), face_encoder=StubFaceEncoder(8),
                           pad_id=1, log=lambda *_: None, on_epoch_end=seen.append)
    assert results["stage"] == 2 and results["base_checkpoint"] == str(s1) and results["face_trainable_params"] > 0
    assert [r["epoch"] for r in seen] == [1, 2]
    for key in ("dev", "dev_masked_text_only", "dev_masked_vision_only", "test"):
        assert 0.0 <= results[key]["emotion"]["weighted_f1"] <= 1.0
    ckpt = torch.load(out / "best.pt", weights_only=False)
    assert "face_encoder_state" in ckpt and "proj.weight" in ckpt["face_encoder_state"]
    assert json.load(open(out / "results.json"))["config"]["face_trainable_layers"] == 4
    assert (out / "test_predictions.jsonl").exists()


def test_train_stage2_refuses_to_run_without_a_base_checkpoint(synthetic_features, tmp_path):
    import pytest
    from meld_emotion.training.train_stage2 import train_stage2
    root, _, _ = synthetic_features
    with pytest.raises(FileNotFoundError):
        train_stage2(_cfg(init_from=None), root, tmp_path / "pre", tmp_path / "s2",
                     encode_fn=whitespace_encode, text_encoder=StubTextEncoder(32), face_encoder=StubFaceEncoder(8), pad_id=1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_train_stage2.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meld_emotion.training.train_stage2'`

- [ ] **Step 3: Implement**

Create `src/meld_emotion/training/train_stage2.py`:

```python
"""Stage 2 training loop (design doc §6): the Stage 1 loop with the face
ViT in front of the fusion model, three learning rates (face < text <
fusion), bf16 autocast on CUDA, and a checkpoint that also carries the
fine-tuned face-encoder weights for the demo."""
import contextlib
import functools
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from meld_emotion.data.labels import EMOTIONS
from meld_emotion.training.config import TrainConfig
from meld_emotion.training.crops import MeldCropDataset, collate_crops
from meld_emotion.training.dataset import BucketBatchSampler, infer_feature_dims, load_ok_rows
from meld_emotion.training.metrics import class_weights, joint_loss
from meld_emotion.training.model import FusionModel
from meld_emotion.training.stage2_model import Stage2Model, init_fusion_from_stage1
from meld_emotion.training.text import build_text_encoder, build_tokenizer, count_parameters, encode_text
from meld_emotion.training.train import (_metrics_only, _to_device, build_scheduler, evaluate, peak_memory_mb,
                                         resolve_device, seed_everything)


def build_stage2_optimizer(model: Stage2Model, config: TrainConfig) -> torch.optim.AdamW:
    groups = [{"params": [p for p in model.face_parameters() if p.requires_grad], "lr": config.lr_face}]
    text_params = [p for p in model.text_parameters() if p.requires_grad]
    if text_params:
        groups.append({"params": text_params, "lr": config.lr_text})
    groups.append({"params": [p for p in model.fusion_parameters() if p.requires_grad], "lr": config.lr_fusion})
    return torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=config.weight_decay)


def _autocast(device: str, enabled: bool):
    if device == "cuda" and enabled:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def train_stage2(config: TrainConfig, features_dir: Path, preprocessed_dir: Path, out_dir: Path, *,
                 train_split: str = "train", dev_split: str = "dev", test_split: str | None = None,
                 encode_fn=None, text_encoder=None, face_encoder=None, pad_id: int | None = None,
                 init_from: Path | None = None, max_train_rows: int | None = None,
                 log=print, on_epoch_end=None) -> dict:
    features_dir, preprocessed_dir, out_dir = Path(features_dir), Path(preprocessed_dir), Path(out_dir)
    init_from = Path(init_from) if init_from is not None else (Path(config.init_from) if config.init_from else None)
    if init_from is None or not init_from.exists():
        raise FileNotFoundError(f"Stage 2 needs a Stage 1 checkpoint to initialise from; got {init_from}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    seed_everything(config.seed)
    started = time.perf_counter()

    # --- text side (same as Stage 1) ---
    if config.use_text and (encode_fn is None or text_encoder is None or pad_id is None):
        tokenizer = build_tokenizer(config.text_model)
        encode_fn = encode_fn or functools.partial(encode_text, tokenizer, max_length=config.max_text_tokens)
        text_encoder = text_encoder or build_text_encoder(config.text_model, config.text_trainable_layers)
        pad_id = tokenizer.pad_token_id if pad_id is None else pad_id
    if not config.use_text:
        encode_fn = encode_fn or (lambda context, current: {"input_ids": [0], "attention_mask": [1]})
        pad_id = 0 if pad_id is None else pad_id
        text_encoder = None

    # --- face encoder in the loop ---
    if face_encoder is None:
        from meld_emotion.training.face_encoder import TrainableFaceEncoder
        face_encoder = TrainableFaceEncoder(trainable_layers=config.face_trainable_layers)

    # --- data ---
    train_rows = load_ok_rows(features_dir / train_split / "manifest.jsonl")
    if max_train_rows:
        train_rows = train_rows[:max_train_rows]
    dev_rows = load_ok_rows(features_dir / dev_split / "manifest.jsonl")
    _, scene_dim = infer_feature_dims(train_rows, features_dir)
    collate_fn = functools.partial(collate_crops, pad_id=pad_id)

    def make_loader(rows, batch_size, shuffle, train):
        ds = MeldCropDataset(rows, preprocessed_dir, features_dir, config, encode_fn, train=train)
        sampler = BucketBatchSampler(ds.lengths, batch_size, shuffle=shuffle, seed=config.seed)
        return DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn, num_workers=config.loader_workers,
                          persistent_workers=config.loader_workers > 0), sampler

    train_loader, train_sampler = make_loader(train_rows, config.batch_size, shuffle=True, train=True)
    dev_loader, _ = make_loader(dev_rows, config.batch_size, shuffle=False, train=False)

    # --- model: Stage 1 weights for everything but the ViT ---
    fusion = FusionModel(config, text_encoder, face_dim=face_encoder.feature_dim, scene_dim=scene_dim)
    base_ckpt = init_fusion_from_stage1(fusion, init_from)
    model = Stage2Model(fusion, face_encoder).to(device)
    params_total, params_trainable = count_parameters(model)
    face_trainable = sum(p.numel() for p in model.face_parameters() if p.requires_grad)
    optimizer = build_stage2_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config.epochs * len(train_loader), config.warmup_fraction)
    emotion_weight = class_weights(np.array([EMOTIONS.index(r["emotion"]) for r in train_rows]),
                                   len(EMOTIONS), config.class_weight_alpha).to(device)
    log(f"[{config.name}] device={device} train={len(train_rows)} dev={len(dev_rows)} params={params_total:,} "
        f"trainable={params_trainable:,} (face {face_trainable:,}) steps/epoch={len(train_loader)} "
        f"init={init_from} (stage1 dev wF1 {base_ckpt.get('dev_weighted_f1', float('nan')):.4f})")

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
            with _autocast(device, config.amp_bf16):
                out = model(batch)
            out = {k: v.float() for k, v in out.items()}
            loss, _ = joint_loss(out, batch, emotion_weight, config.sentiment_lambda, config.label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        with _autocast(device, config.amp_bf16):
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
            torch.save({"model_state": model.fusion.state_dict(), "face_encoder_state": model.face_encoder.export_state(),
                        "config": asdict(config), "epoch": epoch, "dev_weighted_f1": dev_f1,
                        "face_dim": face_encoder.feature_dim, "scene_dim": scene_dim, "stage": 2,
                        "base_checkpoint": str(init_from)}, out_dir / "best.pt")
        else:
            epochs_without_gain += 1
        if on_epoch_end is not None:
            on_epoch_end(record)
        if epochs_without_gain >= config.patience:
            log(f"  early stop: no dev improvement for {config.patience} epochs")
            break

    # --- final evaluation from the best checkpoint ---
    ckpt = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
    model.fusion.load_state_dict(ckpt["model_state"])
    model.face_encoder.load_state_dict(ckpt["face_encoder_state"]) if not hasattr(model.face_encoder, "model") \
        else model.face_encoder.model.load_state_dict(ckpt["face_encoder_state"])
    model.to(device)
    with _autocast(device, config.amp_bf16):
        results = {"config": asdict(config), "stage": 2, "base_checkpoint": str(init_from), "device": device,
                   "best_epoch": best_epoch, "epochs_run": epoch, "train_rows": len(train_rows), "dev_rows": len(dev_rows),
                   "params_total": params_total, "params_trainable": params_trainable, "face_trainable_params": face_trainable,
                   "dev": _metrics_only(evaluate(model, dev_loader, device))}
        if config.use_text and config.use_vision:
            results["dev_masked_text_only"] = _metrics_only(evaluate(model, dev_loader, device, force_drop_vision=True))
            results["dev_masked_vision_only"] = _metrics_only(evaluate(model, dev_loader, device, force_drop_text=True))
        if test_split:
            test_rows = load_ok_rows(features_dir / test_split / "manifest.jsonl")
            test_loader, _ = make_loader(test_rows, config.batch_size, shuffle=False, train=False)
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

In `src/meld_emotion/training/results.py`, extend `ORDER` so Stage 2 rows
sort after their base:

```python
ORDER = ["vision_only_zero_training", "text_only_k0", "text_only_k4", "vision_only", "fusion",
         "fusion_no_scene", "fusion_no_context", "fusion_no_trackid", "fusion_faces_only",
         "stage2_fusion", "stage2_fusion_no_scene", "stage2_fusion_faces_only", "stage2_fusion_no_trackid"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_train_stage2.py tests/test_results.py -v`
Expected: 3 + 2 passed

- [ ] **Step 5: Add the local CLI**

Create `scripts/train_stage2.py`:

```python
#!/usr/bin/env python3
"""Stage 2 locally (smoke tests; the real runs use scripts/modal_stage2.py).

Usage:
    uv run python scripts/train_stage2.py --base fusion --train-split dev --epochs 1 --max-train-rows 64 --out results/smoke_stage2
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
from pathlib import Path

from meld_emotion.config import FEATURE_CACHE_DIR, PREPROCESSED_DIR, REPO_ROOT
from meld_emotion.training.config import stage2_config
from meld_emotion.training.train_stage2 import train_stage2


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="fusion", help="Stage 1 preset to start from (must use faces)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--face-trainable-layers", type=int, default=None)
    parser.add_argument("--lr-face", type=float, default=None)
    parser.add_argument("--loader-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--init-from", type=Path, default=None, help="Stage 1 best.pt (default: results/<base>/seed0/best.pt)")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--dev-split", default="dev")
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    overrides = {k: v for k, v in {"seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
                                   "face_trainable_layers": args.face_trainable_layers, "lr_face": args.lr_face,
                                   "loader_workers": args.loader_workers, "device": args.device}.items() if v is not None}
    config = stage2_config(args.base, **overrides)
    init_from = args.init_from or REPO_ROOT / config.init_from
    out_dir = args.out or REPO_ROOT / "results" / config.name / f"seed{config.seed}"
    train_stage2(config, FEATURE_CACHE_DIR, PREPROCESSED_DIR, out_dir, init_from=init_from,
                 train_split=args.train_split, dev_split=args.dev_split,
                 test_split="test" if args.eval_test else None, max_train_rows=args.max_train_rows)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Smoke-test locally on MPS with the real ViT (64 dev rows, 1 epoch)**

Run: `uv run python scripts/train_stage2.py --base fusion --train-split dev --epochs 1 --max-train-rows 64 --out results/smoke_stage2`
Expected: `[stage2_fusion] device=mps train=64 dev=1108 params=222,xxx,xxx trainable=8x,xxx,xxx (face 28,353,024) steps/epoch=4 init=.../results/fusion/seed0/best.pt (stage1 dev wF1 0.6210)`, then one `epoch 1:` line (the dev pass over 1108 clips' crops on MPS takes a few minutes — that is the cost of the ViT in the loop and why the real runs are on Modal), and `results/smoke_stage2/{best.pt,results.json}`. `best.pt` is ~860 MB (RoBERTa + ViT); delete `results/smoke_stage2` afterwards.

- [ ] **Step 7: Commit**

```bash
git add src/meld_emotion/training/train_stage2.py src/meld_emotion/training/results.py scripts/train_stage2.py tests/test_train_stage2.py
git commit -m "Add Stage 2 training loop (face ViT in the loop, three LRs, bf16, face weights in checkpoint)"
```

---

### Task 5: Ship the crops and run Stage 2 on Modal

**Files:**
- Create: `scripts/pack_crops.py`
- Create: `scripts/modal_stage2.py`

**Interfaces:**
- Consumes: `train_stage2`, `stage2_config`; the `meld-features` /
  `meld-results` / `meld-hf-cache` Volumes from Stage 1 (the Stage 1
  checkpoint is already at `/out/<base>/seed0/best.pt`).
- Produces: `data/meld/crops_<split>.tar` (face JPEGs + `metadata.json`
  only, ~2.0 GB train / 0.2 dev / 0.5 test); Volume `meld-crops`; Modal app
  `meld-stage2` with `run(base, seed, eval_test, epochs)`.

- [ ] **Step 1: Write the packer**

Create `scripts/pack_crops.py`:

```python
#!/usr/bin/env python3
"""Pack one split's face crops + metadata.json into a single tar for Modal
(270K small files are far slower to upload than three tarballs).

Usage:
    uv run python scripts/pack_crops.py --split train      # -> data/meld/crops_train.tar
"""
import argparse
import tarfile
import time

from meld_emotion.config import DATA_DIR, PREPROCESSED_DIR, SPLITS


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=SPLITS, required=True)
    args = parser.parse_args()
    out = DATA_DIR / f"crops_{args.split}.tar"
    started = time.perf_counter()
    n = 0
    with tarfile.open(out, "w") as tar:
        for clip_dir in sorted((PREPROCESSED_DIR / args.split).iterdir()):
            if not (clip_dir / "metadata.json").exists():
                continue
            for p in [clip_dir / "metadata.json"] + sorted(clip_dir.glob("frame*_face*.jpg")):
                tar.add(p, arcname=f"{args.split}/{clip_dir.name}/{p.name}")
                n += 1
    print(f"{n} files -> {out} ({out.stat().st_size / 2**30:.2f} GB) in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Pack and upload (once; ~2.7 GB total)**

Run, for each split:
```bash
uv run python scripts/pack_crops.py --split train && uv run python scripts/pack_crops.py --split dev && uv run python scripts/pack_crops.py --split test
uv run --with modal modal volume create meld-crops
uv run --with modal modal volume put meld-crops data/meld/crops_train.tar /crops_train.tar
uv run --with modal modal volume put meld-crops data/meld/crops_dev.tar /crops_dev.tar
uv run --with modal modal volume put meld-crops data/meld/crops_test.tar /crops_test.tar
rm data/meld/crops_*.tar
```
Expected: `uv run --with modal modal volume ls meld-crops` lists the three tars.

- [ ] **Step 3: Write the Modal app**

Create `scripts/modal_stage2.py`:

```python
#!/usr/bin/env python3
"""Stage 2 on Modal (A10G): extract the split tarballs to local disk in the
container, then run train_stage2() with the ViT in the loop.

    uv run --with modal modal run --detach scripts/modal_stage2.py --bases fusion --seeds 0
    uv run --with modal modal run --detach scripts/modal_stage2.py --bases fusion,fusion_no_scene --seeds 0,1,2 --eval-test
Results: same layout as Stage 1 under /out/stage2_<base>/seed<N>/ on meld-results.
"""
import modal

app = modal.App("meld-stage2")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.2", "torchvision>=0.17", "transformers>=4.40,<6", "scikit-learn>=1.4", "numpy>=1.26",
                 "pillow>=10.0", "opencv-python-headless>=4.9")
    .add_local_python_source("meld_emotion")
)
features = modal.Volume.from_name("meld-features", create_if_missing=True)
crops = modal.Volume.from_name("meld-crops", create_if_missing=True)
results = modal.Volume.from_name("meld-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("meld-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="A10G", cpu=8, memory=24576, timeout=4 * 3600,
              volumes={"/vol": features, "/crops": crops, "/out": results, "/root/.cache/huggingface": hf_cache})
def run(base: str, seed: int, eval_test: bool = False, epochs: int | None = None) -> dict:
    import subprocess, time
    from meld_emotion.training.config import stage2_config
    from meld_emotion.training.train_stage2 import train_stage2

    started = time.perf_counter()
    for split in ("train", "dev") + (("test",) if eval_test else ()):
        subprocess.run(["tar", "-xf", f"/crops/crops_{split}.tar", "-C", "/tmp"], check=True)
    print(f"crops extracted in {time.perf_counter() - started:.0f}s", flush=True)

    overrides = {"seed": seed, "device": "cuda", "loader_workers": 6, **({"epochs": epochs} if epochs else {})}
    config = stage2_config(base, **overrides)
    out_dir = f"/out/{config.name}/seed{seed}"
    r = train_stage2(config, "/vol/features", "/tmp", out_dir, init_from=f"/out/{base}/seed0/best.pt",
                     test_split="test" if eval_test else None, on_epoch_end=lambda record: results.commit())
    results.commit()
    return {"run": config.name, "seed": seed, "best_epoch": r["best_epoch"],
            "dev_weighted_f1": r["dev"]["emotion"]["weighted_f1"], "dev_macro_f1": r["dev"]["emotion"]["macro_f1"],
            "test_weighted_f1": r.get("test", {}).get("emotion", {}).get("weighted_f1"),
            "minutes": round(r["wall_clock_s"] / 60, 1), "peak_gpu_mb": round(r.get("peak_gpu_mb", 0))}


@app.local_entrypoint()
def main(bases: str = "fusion", seeds: str = "0", eval_test: bool = False, epochs: int = 0):
    jobs = [(b, int(s), eval_test, epochs or None) for b in bases.split(",") for s in seeds.split(",")]
    print(f"launching {len(jobs)} Stage 2 run(s) on A10G: {jobs}")
    for out in run.starmap(jobs):
        print(out)
```

- [ ] **Step 4: Smoke-run one short job (a few cents)**

Run: `uv run --with modal modal run scripts/modal_stage2.py --bases fusion --seeds 0 --epochs 1`
Expected: `crops extracted in ~30s`, then `[stage2_fusion] device=cuda train=9988 dev=1108 ... (face 28,353,024) steps/epoch=625 init=/out/fusion/seed0/best.pt (stage1 dev wF1 0.6210)`, one epoch in roughly 4–6 minutes (625 steps; the ViT forward over ~200K augmented crops dominates), a dev wF1 near the Stage 1 number (the fusion side starts from the Stage 1 weights; epoch 1 mostly measures that nothing is broken), and a returned dict. Then remove the smoke result so it isn't confused with a real run:
`uv run --with modal modal volume rm meld-results /stage2_fusion/seed0 --recursive`

- [ ] **Step 5: Commit**

```bash
git add scripts/pack_crops.py scripts/modal_stage2.py
git commit -m "Add crop packer and Modal A10G runner for Stage 2"
```

---

### Task 6: Let the inference-time face encoder load Stage 2 weights

**Files:**
- Modify: `src/meld_emotion/vision/encoders.py`
- Modify: `tests/test_encoders.py`

**Interfaces:**
- Produces: `FaceEmotionEncoder(device=None, batch_size=64, weights_path:
  str | Path | None = None)` — when given, loads the `face_encoder_state`
  from a Stage 2 `best.pt` into the same ViT, so the demo's encoder is the
  fine-tuned one. `.encode_batch()["features"]` then matches Stage 2's
  training-time features; `["probs"]` is the stale original head and must
  not be used for the provisional state (Task 7).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_encoders.py`:

```python
def test_face_encoder_loads_stage2_weights_from_a_checkpoint(face_encoder, tmp_path):
    import torch
    from meld_emotion.vision.encoders import FaceEmotionEncoder
    state = {k: v.clone() for k, v in face_encoder.model.state_dict().items()}
    key = "vit.layers.11.attention.q_proj.weight"
    state[key] = state[key] + 0.01
    torch.save({"face_encoder_state": state, "stage": 2}, tmp_path / "best.pt")
    tuned = FaceEmotionEncoder(device="cpu", weights_path=tmp_path / "best.pt")
    assert torch.allclose(tuned.model.state_dict()[key], state[key])
    image = _images(1)[0]
    assert not torch.allclose(torch.from_numpy(tuned.encode(image)["features"]), torch.from_numpy(face_encoder.encode(image)["features"]))
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_encoders.py -m network -k stage2 -v`
Expected: FAIL with `TypeError: FaceEmotionEncoder.__init__() got an unexpected keyword argument 'weights_path'`

- [ ] **Step 3: Implement**

In `src/meld_emotion/vision/encoders.py`, change the `FaceEmotionEncoder`
constructor signature and add the load after the `TypeError` check:

```python
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
```

and add `from pathlib import Path` to the imports.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_encoders.py -m network -v` → Expected: 6 passed
Run: `uv run pytest -q` → Expected: everything offline still passes.

- [ ] **Step 5: Commit**

```bash
git add src/meld_emotion/vision/encoders.py tests/test_encoders.py
git commit -m "FaceEmotionEncoder can load Stage 2 fine-tuned weights for the demo"
```

---

### Task 7: Run Stage 2 for real, update the table and the demo briefing

**Files:**
- Modify: `docs/results/stage1_ablations.md` (regenerated)
- Modify: `docs/superpowers/notes/2026-09-12-lm-demo-plan-briefing.md`

- [ ] **Step 1: Run Stage 2 on the winning base(s), three seeds**

Run: `uv run --with modal modal run --detach scripts/modal_stage2.py --bases <winning base> --seeds 0,1,2`
(and a second base if the Stage 1 seeds left two contenders). Expected:
each run ~30–45 min on an A10G (8 epochs × ~4–5 min, early stopping
usually earlier); three containers in parallel.

- [ ] **Step 2: Fetch JSON only (checkpoints are ~860 MB each) and rebuild the table**

```bash
for s in 0 1 2; do mkdir -p results/stage2_<base>/seed$s; for f in results.json log.jsonl; do uv run --with modal modal volume get meld-results /stage2_<base>/seed$s/$f results/stage2_<base>/seed$s/$f --force; done; done
uv run python scripts/ablation_table.py --out docs/results/stage1_ablations.md
```
Expected: a `stage2_<base>` row under its base. **Keep Stage 2 only if its
dev weighted-F1 mean beats the base's** (design doc §6). Look at
`dev_masked_vision_only` too — that is where face specialisation should
show first.

- [ ] **Step 3: Test, once, for the final configuration**

Run: `uv run --with modal modal run --detach scripts/modal_stage2.py --bases <base> --seeds 0 --eval-test`
(or the Stage 1 runner with `--eval-test` if Stage 2 did not win), fetch
`results.json` + `test_predictions.jsonl`, regenerate the table. Then fetch
the winning `best.pt` for the demo:
`uv run --with modal modal volume get meld-results /stage2_<base>/seed0/best.pt results/stage2_<base>/seed0/best.pt`

- [ ] **Step 4: Update the demo briefing**

In `docs/superpowers/notes/2026-09-12-lm-demo-plan-briefing.md`, replace the
bullet that begins `**The frozen face head's raw probabilities are prior-skewed`
with:

```
- **Do not use the face head's probabilities for anything.** They were
  prior-skewed even in Stage 1 (wF1 0.12 on dev) and after Stage 2 the head
  is stale (it was trained on the old CLS). The §2 provisional state is the
  fusion model with text masked -- `model(batch, force_drop_text=True)` over
  the visual tokens accumulated so far -- and the gloss's per-face labels
  (§4.4 b) become that same vision-only prediction's top-2 for the clip.
  Load the encoder with `FaceEmotionEncoder(weights_path=<stage2 best.pt>)`
  when the Stage 2 checkpoint won; the fusion weights come from the same
  file's `model_state`.
```

- [ ] **Step 5: Commit**

```bash
git add docs/results/stage1_ablations.md docs/superpowers/notes/2026-09-12-lm-demo-plan-briefing.md
git commit -m "Record Stage 2 results; demo briefing: provisional state from the masked fusion model"
```

---

## Self-Review

**Spec coverage.** §6 Stage 2 (unfreeze the face ViT's top 4 layers,
initialise from Stage 1, lower LR, keep only if it beats Stage 1 on dev) →
Tasks 2, 3, 4, 7. Same metrics/protocol as Stage 1 → Task 4 reuses
`evaluate`/`joint_loss`/`class_weights`. Unchanged inference path and
parameter count → Task 6 loads the adapted weights into the same encoder.
Generalisation to unseen faces (the live demo) → frozen bottom layers, low
face LR, augmentation (Task 1). Modal execution → Task 5. §7 table → Task 7.

**Placeholder scan.** No TBD/TODO; every step has runnable code and an
exact command with expected output.

**Type consistency.** `collate_crops` (Task 1) emits `face_pixels`,
`face_batch_idx`, `face_slot`, `face_mask (B, Fmax)` and a `(B, Fmax, 1)`
`face_feat` placeholder; `Stage2Model.forward` (Task 3) reads exactly those
and overwrites `face_feat` via `scatter_faces` (Task 2) before calling the
unchanged `FusionModel`. `TrainableFaceEncoder` and the test
`StubFaceEncoder` share `.feature_dim`, `forward(uint8) -> (N, D)`,
`trainable_parameters()`, `export_state()`. `train_stage2` (Task 4) saves
`model_state` (fusion only) + `face_encoder_state` (whole ViT), which
`FaceEmotionEncoder(weights_path=...)` (Task 6) and `init_fusion_from_stage1`
(Task 3, via Stage 1's `load_checkpoint`) consume. `stage2_config` (Task 1)
names runs `stage2_<base>`, matching `ORDER` in `results.py` (Task 4) and
the Modal output path (Task 5).
