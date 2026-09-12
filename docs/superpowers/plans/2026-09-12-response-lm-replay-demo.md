# Response LM + Replay Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the trained Stage 1 checkpoint into a real-time inference
core that emits the design doc's four-phase event stream, add a prompted,
streamed response LM with a visual gloss, and ship the replay demo on ~10
curated MELD test clips — with a batch/replay consistency check, measured
per-event latency, the final parameter-budget table, and a hand-scorable
response rubric. This is design doc build-priority item 3.

**Explicitly out of scope:** the live webcam/microphone/VAD/Whisper path
(design doc build-priority item 4 — a separate, later plan), Stage 2
training, any retraining, and the tri-modal/RL extension.

**Architecture:** One inference core, `TurnProcessor`, does everything a
turn needs: `push_frame(frame_bgr)` runs face detect/track/encode + scene
encode on one sampled frame and returns a `provisional` event; `end_turn
(text, context_prev)` builds the exact model input the training pipeline
used and returns a `final` event. A separate `Responder` wraps `mlx-lm`
and streams `token` events, then a `done` event closes the turn with the
full response and measured latencies. `scripts/replay_demo.py` is the only
thing that knows about `cv2.imshow` — it drives `TurnProcessor` and
`Responder` frame-by-frame against a curated test clip and renders the
window. Everything is unit-tested offline against a synthetic checkpoint
and stub vision/LM components; only checkpoint-loading and LM-benchmark
tests need real weights.

**Tech Stack:** Python 3.11, uv, pytest, PyTorch (MPS/CPU), OpenCV,
`transformers` (RoBERTa, CLIP), `mlx-lm` (4-bit response LM, MLX runtime,
separate from torch).

## Global Constraints

- **Scope boundary:** live webcam/ASR is a separate plan. Do not add VAD,
  microphone capture, or Whisper here — `TurnProcessor`/`Responder` must
  not assume replay-only timing internally (frames arrive one at a time
  either way), but nothing in this plan drives them from a camera.
- **Event stream** (design doc §4.5), exactly these four phases, one JSON
  object per line: `provisional` (`turn_id, phase, frame, faces_seen,
  provisional_expression`), `final` (`turn_id, phase, text, emotion,
  emotion_probs, sentiment, sentiment_probs, faces_seen, visual_cues`) —
  **no `response` or `latency_ms` field on `final`**, `token` (`turn_id,
  phase, text`) carrying one incremental text segment, `done` (`turn_id,
  phase, response, latency_ms`) carrying the full accumulated response and
  `{"state": ms, "first_token": ms, "done": ms}`. (The design doc's one
  combined JSON example under §4.5 shows `response`/`latency_ms` fields
  inline for illustration; its own prose two paragraphs later is explicit
  that `final` is emitted "with response absent" and `done` "carries the
  full response and the measured latencies" — the prose governs, and this
  plan follows it exactly so `final` is never gated on the LM.)
- **Latency targets, replay path** (design doc §2): final state ≤100ms
  after end-of-turn; first response token ≤1.0s after the state; response
  complete ≤2.5s after the state. Per-sampled-frame vision cost must stay
  under the 333ms (~3fps) frame interval. All measured with
  `time.perf_counter()`, p50/p95 reported, never estimated.
- **Response LM selection** (design doc §4.1): the largest of a short list
  of ~3-4B instruction-tuned, permissively-licensed candidates that meets
  the latency targets above, measured on this machine via `mlx-lm` at
  4-bit, any "thinking" mode off. Real, verified candidate HF repos (`mlx-
  community` org, checked to exist on 2026-09-12): `Qwen3-4B-4bit`,
  `Qwen2.5-3B-Instruct-4bit`, `Phi-3.5-mini-instruct-4bit`, with
  `Qwen2.5-1.5B-Instruct-4bit` as the fallback if none of the larger three
  meet target. `mlx_lm.load(repo) -> (model, tokenizer)`;
  `mlx_lm.stream_generate(model, tokenizer, prompt, max_tokens=N)` yields
  `GenerationResponse` objects whose `.text` is the **incremental delta**
  for that step (verified from the installed `mlx-lm`'s source — it is
  `detokenizer.last_segment`, not the cumulative string) and whose
  `.finish_reason` is `None` until the final yield. Qwen3's chat template
  supports `enable_thinking` as a real Jinja variable (verified against its
  published `tokenizer_config.json`); wrap the call in `try/except
  TypeError` since not every candidate's template accepts it.
- **Prompt** (design doc §4.4): fixed persona system prompt (small,
  friendly character robot; react in ≤2 sentences; don't summarize), the
  last 4 context lines, the current line, the predicted emotion with its
  top-2 probabilities, the sentiment, and the visual gloss phrases. Keep it
  under ~200 tokens. Speaker names are never used (matches training,
  design doc §4.3) — never put them in the prompt.
- **Visual gloss feeds the LM only** (design doc §4.4) — never the
  classifier. (a) CLIP zero-shot scene cues: a fixed ~24-phrase bank
  embedded once with CLIP's text tower (reuse the already-loaded
  `SceneEncoder.model`'s text tower — do not load a second `CLIPModel`),
  scored against the turn's mean scene embedding via cosine × CLIP's own
  logit scale (100) then softmax over the bank, top-2 reported only if
  their probability gap clears a small margin. Neither CLIP tower's output
  is L2-normalised by `transformers` — normalise both sides before cosine
  (verified: `get_text_features`/`get_image_features` return
  `BaseModelOutputWithPooling` under the installed `transformers` 5.x;
  unwrap `.pooler_output`, exactly the pattern the merged `SceneEncoder`
  already uses for images). (b) Per-face expression labels, majority-voted
  per track across the turn, from the face head's own softmax argmax
  mapped through `FaceEmotionEncoder.meld_labels` (already on the loaded
  encoder instance — do not recompute the label mapping separately).
- **Face-head prior correction** (a real, measured fact, not a guess): the
  frozen face head's raw softmax is prior-skewed on MELD — it rarely
  predicts "neutral" on real conversational faces. The `provisional` event
  must never show raw per-frame softmax; divide by the empirical mean
  probability vector over the dev cache (`prior.py`, computed once, stored
  as JSON) and renormalise before display. This is a calibration, not a
  learned component, and does not touch the classifier's own input.
- **Model input construction must byte-for-byte match training** (this is
  the single easiest thing to get subtly wrong): frame positions clamped
  via `np.minimum(idx, config.max_frames - 1)`; track ids remapped via
  `meld_emotion.training.dataset.remap_track_ids(ids, config.max_track_slots)`;
  text via `format_context(context_prev, config.context_k)` then
  `encode_text(tokenizer, context, current, config.max_text_tokens)`;
  batching via `collate([item], pad_id=tokenizer.pad_token_id)`; label
  order `meld_emotion.data.labels.EMOTIONS = ("neutral", "joy", "surprise",
  "anger", "sadness", "disgust", "fear")`, `SENTIMENTS = ("neutral",
  "positive", "negative")`.
- **Shot-cut detection compares consecutive SAMPLED frames, not raw decoded
  frames** — exactly as `preprocess_clip` does it (`shot_change_score`
  between the previous *sampled* frame and the current one; the tracker is
  reset when the score is ≥ `SHOT_CUT_THRESHOLD = 0.3`, computed *before*
  face detection runs on the current frame so a reset tracker sees the new
  shot's faces as new tracks). Getting this order wrong (comparing raw
  frames, or detecting faces before checking for a cut) silently breaks
  track-ID continuity in a way no shape-based test catches.
- **Consistency check** (design doc §8.1): the replay path's `final`
  prediction for a curated clip must equal the batch-eval path's
  prediction (same checkpoint, cached features) — assert **argmax
  equality** and **`max|Δprob| < 0.02`**, not bitwise equality (the cache
  encoded JPEG-saved crops; replay encodes in-memory crops, so tiny
  numeric drift is expected). Use the same `sample_frame_indices` and the
  same face-detector threshold (0.75, baked into `build_face_detector`) so
  the frame sets line up.
- **MPS/CPU only, fp32.** No CUDA on this machine. The training-time
  `attn_implementation="eager"` SDPA/dropout gotcha (`text.py`) does not
  apply in eval mode, but the text encoder must still be built via
  `build_text_encoder` (not `AutoModel.from_pretrained` directly) so its
  state-dict keys match the checkpoint.
- **Parameter budget** (design doc §4.1): ceiling 6B; everything before the
  response LM is already known to be ~375M (Stage 1 plan's Global
  Constraints) — this plan's own components (fusion transformer,
  projectors, heads) are already inside that figure; the response LM is
  the only new mass. `results/param_budget.json` must record the actual
  final total with the chosen LM.
- **Response rubric deliverable:** `results/response_rubric.jsonl`, 50
  sampled `{turn_id, prompt, response}` rows with empty rubric fields
  (`tag_consistent, uses_visual_cue, concise, in_character, no_invented_facts`
  all `null`) for hand scoring — this file is itself a plan deliverable,
  not just a script side-effect.
- **Curated clips:** ~10 from *test*, stratified by emotion, including at
  least one the model gets wrong (design doc §8.1) and, where the test
  split contains real examples, the hard shot compositions from design doc
  §5 (a many-face ensemble frame, a clip with zero detected faces, a very
  dark/low-contrast frame) — a heuristic *candidate* list plus a manual
  override, not a fully automated final pick (design doc §5's specific hard
  clips were identified by eye during exploration, not by a reusable rule).
- **Dependencies via `uv add`/`uv run --with` only**; `uv run pytest` must
  pass offline in seconds; anything needing HF downloads or the `mlx_lm`
  package carries `@pytest.mark.network`.
- **Inputs this plan consumes, exactly as the training plan writes them:**
  `results/fusion/seed0/best.pt` — a `torch.save` dict with keys
  `model_state, config (a TrainConfig field dict), epoch, dev_weighted_f1,
  face_dim (768), scene_dim (512)`; `data/meld/features/{dev,test}/manifest.jsonl`
  rows exactly as the data plan writes them (keys `dialogue_id,
  utterance_id, speaker, text, context_prev, emotion, sentiment, status,
  feature_path, duration_s, n_frames, n_faces`); the same directories'
  `.npz` files (`face_features, face_probs, face_track_ids, face_frame_idx,
  scene_features, scene_frame_idx`); real MELD test videos under
  `meld_emotion.config.split_video_dir("test")`, indexed with
  `meld_emotion.data.video_index.build_video_index` (never a global glob —
  IDs collide across splits).

---

## File Structure

```
src/meld_emotion/inference/
  __init__.py
  events.py       # Event dataclasses-as-dicts + JSON-lines emitter (Task 1)
  loader.py       # checkpoint -> InferenceBundle (model, tokenizer, vision encoders) (Task 2)
  prior.py        # face-head prior from the dev cache; provisional-state correction (Task 3)
  gloss.py        # CLIP scene-cue bank + per-track face-label majority vote (Task 4)
  turn.py         # TurnProcessor: push_frame -> provisional; end_turn -> final (Task 5)
  responder.py    # mlx-lm wrapper: prompt building + streaming (Task 6)
scripts/
  select_lm.py         # benchmark candidates, pick per §4.1's rule (Task 6)
  pick_demo_clips.py    # curated clip candidates + final list (Task 7)
  replay_demo.py         # the demo window (Task 8)
  consistency_check.py   # batch vs replay agreement on curated clips (Task 9)
  measure_latency.py     # p50/p95 per event + vision-per-frame cost (Task 10)
  param_budget.py         # final parameter count table (Task 10)
  response_rubric.py      # writes results/response_rubric.jsonl (Task 10)
tests/
  test_events.py, test_loader.py, test_prior.py, test_gloss.py, test_turn.py,
  test_responder.py, test_pick_demo_clips.py, test_consistency_check.py,
  test_measure_latency.py, test_param_budget.py, test_response_rubric.py
  conftest.py           # extends the training-plan conftest: adds a synthetic
                        # checkpoint fixture and a fake video/frame generator (Task 2)
```

---

### Task 1: Event stream

**Files:**
- Create: `src/meld_emotion/inference/__init__.py` (empty)
- Create: `src/meld_emotion/inference/events.py`
- Test: `tests/test_events.py`

**Interfaces:**
- Produces: `EventEmitter(out: TextIO)` with `.emit(event: dict)`,
  `.provisional(turn_id, frame, faces_seen, provisional_expression)`,
  `.final(turn_id, text, emotion, emotion_probs, sentiment, sentiment_probs,
  faces_seen, visual_cues)`, `.token(turn_id, text)`, `.done(turn_id,
  response, latency_ms)`; `LatencyStamps` (`turn_start: float`,
  `state_emitted/first_token_emitted/done_emitted: float | None`) with
  `.as_ms() -> dict` (each key `None` until its stamp is set, else the
  rounded millisecond delta from `turn_start`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_events.py`:

```python
import io
import json

import pytest


def _lines(buf):
    return [json.loads(l) for l in buf.getvalue().strip().splitlines()]


def test_provisional_event_has_exactly_the_spec_fields():
    from meld_emotion.inference.events import EventEmitter
    buf = io.StringIO()
    EventEmitter(buf).provisional("dia1_utt1", 3, 2, {"joy": 0.6, "neutral": 0.4})
    events = _lines(buf)
    assert len(events) == 1
    assert events[0] == {"turn_id": "dia1_utt1", "phase": "provisional", "frame": 3,
                         "faces_seen": 2, "provisional_expression": {"joy": 0.6, "neutral": 0.4}}


def test_final_event_never_carries_response_or_latency():
    from meld_emotion.inference.events import EventEmitter
    buf = io.StringIO()
    EventEmitter(buf).final("dia1_utt1", "hi there", "joy", {"joy": 0.9}, "positive",
                            {"positive": 0.9}, 1, ["one person"])
    event = _lines(buf)[0]
    assert event["phase"] == "final"
    assert "response" not in event and "latency_ms" not in event


def test_token_and_done_events_carry_the_response_and_latencies():
    from meld_emotion.inference.events import EventEmitter
    buf = io.StringIO()
    emitter = EventEmitter(buf)
    emitter.token("dia1_utt1", "Wow")
    emitter.token("dia1_utt1", " really?")
    emitter.done("dia1_utt1", "Wow really?", {"state": 80, "first_token": 600, "done": 1800})
    events = _lines(buf)
    assert [e["phase"] for e in events] == ["token", "token", "done"]
    assert events[0]["text"] == "Wow" and events[1]["text"] == " really?"
    assert events[2] == {"turn_id": "dia1_utt1", "phase": "done", "response": "Wow really?",
                         "latency_ms": {"state": 80, "first_token": 600, "done": 1800}}


def test_latency_stamps_report_none_until_set_then_millisecond_deltas():
    from meld_emotion.inference.events import LatencyStamps
    stamps = LatencyStamps(turn_start=10.0)
    assert stamps.as_ms() == {"state": None, "first_token": None, "done": None}
    stamps.state_emitted = 10.084
    stamps.first_token_emitted = 10.61
    stamps.done_emitted = 11.82
    ms = stamps.as_ms()
    assert ms == {"state": 84, "first_token": 610, "done": 1820}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_events.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meld_emotion.inference'`

- [ ] **Step 3: Implement**

Create `src/meld_emotion/inference/__init__.py` (empty file).

Create `src/meld_emotion/inference/events.py`:

```python
"""JSON-lines event stream shared by both demo frontends (design doc §4.5).

Four phases per turn: `provisional` (during the turn, one per sampled
frame), `final` (state is ready -- response is deliberately absent so the
LM never gates it), `token` (one incremental response chunk), `done`
(closes the turn with the full response and the three measured latencies).
"""
import json
from dataclasses import dataclass
from typing import TextIO


class EventEmitter:
    def __init__(self, out: TextIO):
        self.out = out

    def emit(self, event: dict) -> None:
        self.out.write(json.dumps(event) + "\n")
        self.out.flush()

    def provisional(self, turn_id: str, frame: int, faces_seen: int,
                    provisional_expression: dict) -> None:
        self.emit({"turn_id": turn_id, "phase": "provisional", "frame": frame,
                   "faces_seen": faces_seen, "provisional_expression": provisional_expression})

    def final(self, turn_id: str, text: str, emotion: str, emotion_probs: dict,
              sentiment: str, sentiment_probs: dict, faces_seen: int,
              visual_cues: list[str]) -> None:
        self.emit({"turn_id": turn_id, "phase": "final", "text": text, "emotion": emotion,
                   "emotion_probs": emotion_probs, "sentiment": sentiment,
                   "sentiment_probs": sentiment_probs, "faces_seen": faces_seen,
                   "visual_cues": visual_cues})

    def token(self, turn_id: str, text: str) -> None:
        self.emit({"turn_id": turn_id, "phase": "token", "text": text})

    def done(self, turn_id: str, response: str, latency_ms: dict) -> None:
        self.emit({"turn_id": turn_id, "phase": "done", "response": response,
                   "latency_ms": latency_ms})


@dataclass
class LatencyStamps:
    turn_start: float
    state_emitted: float | None = None
    first_token_emitted: float | None = None
    done_emitted: float | None = None

    def as_ms(self) -> dict:
        def delta(stamp):
            return None if stamp is None else round((stamp - self.turn_start) * 1000)
        return {"state": delta(self.state_emitted), "first_token": delta(self.first_token_emitted),
                "done": delta(self.done_emitted)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_events.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/meld_emotion/inference/__init__.py src/meld_emotion/inference/events.py tests/test_events.py
git commit -m "Add the four-phase JSON-lines event stream and latency stamps"
```

---

### Task 2: Checkpoint loader

**Files:**
- Create: `src/meld_emotion/inference/loader.py`
- Create: `tests/conftest.py` additions (a synthetic-checkpoint fixture)
- Test: `tests/test_loader.py`

**Interfaces:**
- Consumes: `meld_emotion.training.config.TrainConfig`;
  `meld_emotion.training.model.FusionModel`; `meld_emotion.training.text
  .build_text_encoder`, `.build_tokenizer`; `meld_emotion.training.train
  .resolve_device`; `meld_emotion.vision.encoders.FaceEmotionEncoder`,
  `.SceneEncoder`; `meld_emotion.vision.face_detector.build_face_detector`.
- Produces: `InferenceBundle` (frozen-ish dataclass: `model: FusionModel`,
  `config: TrainConfig`, `tokenizer`, `face_detector`, `face_encoder:
  FaceEmotionEncoder`, `scene_encoder: SceneEncoder`, `device: str`);
  `load_inference_bundle(checkpoint_path: Path, device: str = "auto", *,
  text_encoder_factory=None, tokenizer_factory=None,
  face_encoder_factory=None, scene_encoder_factory=None,
  face_detector_factory=None) -> InferenceBundle`. The five `*_factory`
  callables are injection points (each `None` builds the real component);
  tests inject stubs so no network/weights are needed offline.

- [ ] **Step 1: Add the synthetic-checkpoint fixture**

Add to `tests/conftest.py` (this file already exists from the Stage 1
training plan; append these, do not replace the existing content):

```python
import torch as _torch
from dataclasses import asdict as _asdict


def write_synthetic_checkpoint(path, d_model=32, n_heads=4, ff_dim=64, n_layers=1,
                               face_dim=8, scene_dim=4, use_text=True):
    """Writes a checkpoint in exactly train.py's save format, small enough
    to build and load in a unit test."""
    from meld_emotion.training.config import TrainConfig
    from meld_emotion.training.model import FusionModel
    config = TrainConfig(d_model=d_model, n_heads=n_heads, ff_dim=ff_dim, n_layers=n_layers,
                         use_text=use_text)
    text_encoder = StubTextEncoder(d_model) if use_text else None
    model = FusionModel(config, text_encoder, face_dim=face_dim, scene_dim=scene_dim)
    _torch.save({"model_state": model.state_dict(), "config": _asdict(config), "epoch": 1,
                "dev_weighted_f1": 0.5, "face_dim": face_dim, "scene_dim": scene_dim}, path)
    return config


@pytest.fixture
def synthetic_checkpoint(tmp_path):
    path = tmp_path / "best.pt"
    config = write_synthetic_checkpoint(path)
    return path, config
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_loader.py`:

```python
import torch

from conftest import StubTextEncoder


def _stub_factories():
    class _StubFaceDetector: pass
    class _StubFaceEncoder:
        meld_labels = ("neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear")
        def encode_batch(self, images):
            import numpy as np
            return {"features": np.zeros((len(images), 8), dtype=np.float32),
                   "probs": np.zeros((len(images), 7), dtype=np.float32)}
    class _StubSceneEncoder:
        def encode(self, image):
            import numpy as np
            return np.zeros(4, dtype=np.float32)
    class _StubTokenizer:
        pad_token_id = 1
    return dict(text_encoder_factory=lambda: StubTextEncoder(32),
               tokenizer_factory=lambda: _StubTokenizer(),
               face_encoder_factory=lambda: _StubFaceEncoder(),
               scene_encoder_factory=lambda: _StubSceneEncoder(),
               face_detector_factory=lambda: _StubFaceDetector())


def test_loads_a_matching_model_and_reproduces_the_checkpoints_own_state(synthetic_checkpoint):
    from meld_emotion.inference.loader import load_inference_bundle
    path, config = synthetic_checkpoint
    bundle = load_inference_bundle(path, device="cpu", **_stub_factories())
    assert bundle.device == "cpu"
    assert bundle.config.d_model == config.d_model and bundle.config.n_layers == config.n_layers
    assert bundle.tokenizer.pad_token_id == 1
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for key, value in bundle.model.state_dict().items():
        assert torch.equal(value, ckpt["model_state"][key])


def test_vision_only_checkpoint_skips_the_text_encoder(tmp_path):
    from meld_emotion.inference.loader import load_inference_bundle
    from conftest import write_synthetic_checkpoint
    path = tmp_path / "vision_only.pt"
    write_synthetic_checkpoint(path, use_text=False)
    factories = _stub_factories()
    factories["text_encoder_factory"] = lambda: (_ for _ in ()).throw(AssertionError("must not be called"))
    factories["tokenizer_factory"] = lambda: (_ for _ in ()).throw(AssertionError("must not be called"))
    bundle = load_inference_bundle(path, device="cpu", **factories)
    assert bundle.tokenizer is None
    assert bundle.model.text_encoder is None


def test_resolve_device_auto_is_used_when_device_not_given(synthetic_checkpoint, monkeypatch):
    from meld_emotion.inference import loader as loader_module
    from meld_emotion.inference.loader import load_inference_bundle
    path, _ = synthetic_checkpoint
    monkeypatch.setattr(loader_module, "resolve_device", lambda name: "cpu" if name == "auto" else name)
    bundle = load_inference_bundle(path, **_stub_factories())
    assert bundle.device == "cpu"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meld_emotion.inference.loader'`

- [ ] **Step 4: Implement**

Create `src/meld_emotion/inference/loader.py`:

```python
"""Load a Stage 1 checkpoint into an inference-ready bundle: the fused
model plus the text tokenizer and the frozen vision encoders it needs at
serving time (design doc §4.1). The model must be built from the
checkpoint's own recorded config/face_dim/scene_dim before its weights can
be loaded, so this reads the checkpoint file directly rather than going
through train.load_checkpoint (which needs an already-built model)."""
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
    face_encoder: FaceEmotionEncoder
    scene_encoder: SceneEncoder
    device: str


def load_inference_bundle(checkpoint_path: Path, device: str = "auto", *,
                          text_encoder_factory=None, tokenizer_factory=None,
                          face_encoder_factory=None, scene_encoder_factory=None,
                          face_detector_factory=None) -> InferenceBundle:
    device = resolve_device(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = TrainConfig(**ckpt["config"])

    text_encoder = tokenizer = None
    if config.use_text:
        build_text = text_encoder_factory or (
            lambda: build_text_encoder(config.text_model, config.text_trainable_layers))
        build_tok = tokenizer_factory or (lambda: build_tokenizer(config.text_model))
        text_encoder, tokenizer = build_text(), build_tok()

    model = FusionModel(config, text_encoder, face_dim=ckpt["face_dim"], scene_dim=ckpt["scene_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    build_face_enc = face_encoder_factory or (lambda: FaceEmotionEncoder(device=device))
    build_scene_enc = scene_encoder_factory or (lambda: SceneEncoder(device=device))
    build_detector = face_detector_factory or build_face_detector

    return InferenceBundle(model=model, config=config, tokenizer=tokenizer,
                           face_detector=build_detector(), face_encoder=build_face_enc(),
                           scene_encoder=build_scene_enc(), device=device)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_loader.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add src/meld_emotion/inference/loader.py tests/conftest.py tests/test_loader.py
git commit -m "Add checkpoint loader producing an inference-ready model + tokenizer + vision bundle"
```

---

### Task 3: Face-head prior calibration

**Files:**
- Create: `src/meld_emotion/inference/prior.py`
- Test: `tests/test_prior.py`

**Interfaces:**
- Consumes: `meld_emotion.data.manifest.read_manifest`;
  `meld_emotion.data.labels.EMOTIONS`; the dev feature cache's
  `.npz` `face_probs` arrays (already merged data pipeline).
- Produces: `compute_face_prior(dev_manifest_path: Path, cache_dir: Path)
  -> np.ndarray` (shape `(7,)`, native face-head column order, sums to
  1.0); `save_prior(prior, path) -> None`, `load_prior(path) ->
  np.ndarray`; `correct_provisional(raw_probs: np.ndarray, prior:
  np.ndarray, face_to_meld_order: tuple[str, ...]) -> dict[str, float]`
  (keys in `EMOTIONS` canonical order, values sum to 1.0).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_prior.py`:

```python
import json

import numpy as np
import pytest

from meld_emotion.data.labels import EMOTIONS


def test_compute_face_prior_is_the_mean_probability_vector_over_every_face(synthetic_features):
    from meld_emotion.inference.prior import compute_face_prior
    root, train, dev = synthetic_features
    prior = compute_face_prior(dev, root)
    assert prior.shape == (7,)
    assert prior.sum() == pytest.approx(1.0, abs=1e-5)
    assert np.all(prior >= 0)


def test_save_and_load_prior_round_trips(tmp_path):
    from meld_emotion.inference.prior import load_prior, save_prior
    prior = np.array([0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    path = tmp_path / "prior.json"
    save_prior(prior, path)
    loaded = load_prior(path)
    assert np.allclose(loaded, prior)
    assert json.loads(path.read_text())["prior"] == pytest.approx(prior.tolist())


def test_correct_provisional_divides_by_prior_and_relabels_to_meld_order():
    from meld_emotion.inference.prior import correct_provisional
    # native order: [sad, disgust, angry, neutral, fear, surprise, happy]
    face_order = ("sadness", "disgust", "anger", "neutral", "fear", "surprise", "joy")
    raw = np.array([0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1], dtype=np.float32)
    prior = np.array([0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3], dtype=np.float32)  # neutral under-predicted normally
    out = correct_provisional(raw, prior, face_order)
    assert list(out.keys()) == list(EMOTIONS)
    assert out["neutral"] == pytest.approx((0.4 / 0.1) / sum(r / p for r, p in zip(raw, prior)), abs=1e-5)
    assert sum(out.values()) == pytest.approx(1.0, abs=1e-5)


def test_correct_provisional_never_divides_by_zero():
    from meld_emotion.inference.prior import correct_provisional
    face_order = ("sadness", "disgust", "anger", "neutral", "fear", "surprise", "joy")
    raw = np.array([1 / 7] * 7, dtype=np.float32)
    prior = np.array([0.0, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1], dtype=np.float32)  # one zero entry
    out = correct_provisional(raw, prior, face_order)
    assert all(np.isfinite(v) for v in out.values())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_prior.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meld_emotion.inference.prior'`

- [ ] **Step 3: Implement**

Create `src/meld_emotion/inference/prior.py`:

```python
"""Face-head prior calibration.

The frozen face head's raw softmax is prior-skewed on MELD: it rarely
predicts "neutral" on real conversational faces (a generic FER model
trained on posed/peak-expression photos, applied zero-shot). Displaying
these raw probabilities as the live "provisional expression" state would
be actively misleading, so we divide by the empirical mean probability
vector over the dev cache and renormalise. This is a calibration, purely
for display -- it never touches the classifier's own input."""
import json
from pathlib import Path

import numpy as np

from meld_emotion.data.labels import EMOTIONS
from meld_emotion.data.manifest import read_manifest


def compute_face_prior(dev_manifest_path: Path, cache_dir: Path) -> np.ndarray:
    total = np.zeros(7, dtype=np.float64)
    count = 0
    for row in read_manifest(dev_manifest_path):
        if row["status"] != "ok":
            continue
        probs = np.load(Path(cache_dir) / row["feature_path"])["face_probs"]
        total += probs.sum(axis=0)
        count += len(probs)
    prior = total / count
    return (prior / prior.sum()).astype(np.float32)


def save_prior(prior: np.ndarray, path: Path) -> None:
    Path(path).write_text(json.dumps({"prior": prior.tolist()}))


def load_prior(path: Path) -> np.ndarray:
    return np.array(json.loads(Path(path).read_text())["prior"], dtype=np.float32)


def correct_provisional(raw_probs: np.ndarray, prior: np.ndarray,
                        face_to_meld_order: tuple[str, ...]) -> dict[str, float]:
    corrected = raw_probs / np.clip(prior, 1e-6, None)
    corrected = corrected / corrected.sum()
    by_meld_label = {face_to_meld_order[i]: float(corrected[i]) for i in range(len(corrected))}
    return {e: by_meld_label[e] for e in EMOTIONS}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_prior.py -v`
Expected: 4 passed

- [ ] **Step 5: Compute and commit the real prior from the dev cache**

Run:
```bash
uv run python -c "
from pathlib import Path
from meld_emotion.inference.prior import compute_face_prior, save_prior
prior = compute_face_prior(Path('data/meld/features/dev/manifest.jsonl'), Path('data/meld/features'))
save_prior(prior, Path('src/meld_emotion/inference/face_prior.json'))
print(prior)
"
```
Expected: a 7-element vector printed (native face-head order: sad,
disgust, angry, neutral, fear, surprise, happy), heavily skewed away from
index 3 (neutral) per the calibration note above; a small `face_prior.json`
is written and should be committed (unlike `data/`/`models/`/`results/`,
this is a small derived constant the inference code depends on, not a
regenerable bulk artifact).

- [ ] **Step 6: Commit**

```bash
git add src/meld_emotion/inference/prior.py src/meld_emotion/inference/face_prior.json tests/test_prior.py
git commit -m "Add face-head prior calibration for the provisional-state display"
```

---

### Task 4: Visual gloss

**Files:**
- Create: `src/meld_emotion/inference/gloss.py`
- Test: `tests/test_gloss.py`

**Interfaces:**
- Consumes: `meld_emotion.vision.encoders.CLIP_MODEL_ID`; a loaded
  `SceneEncoder.model` (the shared `CLIPModel` instance from `loader.py` —
  this module never loads its own `CLIPModel`, only a lightweight text
  tokenizer for it).
- Produces: `SCENE_PROMPT_BANK: tuple[str, ...]` (~24 fixed phrases);
  `build_clip_tokenizer() -> CLIPTokenizer`; `embed_prompt_bank(clip_model,
  clip_tokenizer, device: str) -> torch.Tensor` (shape `(len(bank), 512)`,
  L2-normalised); `scene_gloss(mean_scene_embedding: np.ndarray,
  bank_embeddings: torch.Tensor, margin: float = 0.05) -> list[str]` (top-2
  phrases if their softmax-over-bank probability gap clears `margin`, else
  just the top-1); `face_gloss(track_expressions: dict[int, list[str]]) ->
  str` (majority vote per track, e.g. `"2 faces visible: surprised,
  neutral"`, or `"no faces visible"`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gloss.py`:

```python
import numpy as np
import pytest
import torch


def test_scene_gloss_picks_the_closest_bank_phrase():
    from meld_emotion.inference.gloss import SCENE_PROMPT_BANK, scene_gloss
    bank = torch.eye(len(SCENE_PROMPT_BANK), 8)  # orthonormal stand-in embeddings
    bank = torch.nn.functional.normalize(bank, dim=-1)
    query = bank[3].numpy() * 5.0  # far from unit norm on purpose; scene_gloss must normalise it
    result = scene_gloss(query, bank, margin=0.05)
    assert result[0] == SCENE_PROMPT_BANK[3]


def test_scene_gloss_reports_only_the_top1_when_the_next_is_within_margin():
    from meld_emotion.inference.gloss import scene_gloss
    # two near-identical directions, one clearly-different one
    bank = torch.tensor([[1.0, 0.0], [0.999, 0.045], [0.0, 1.0]])
    bank = torch.nn.functional.normalize(bank, dim=-1)
    query = np.array([1.0, 0.0], dtype=np.float32)
    result = scene_gloss(query, bank, margin=0.5)   # huge margin -> top1 and top2 (near-tied) fail it
    assert len(result) == 1


def test_face_gloss_majority_votes_per_track_and_counts_faces():
    from meld_emotion.inference.gloss import face_gloss
    assert face_gloss({0: ["surprise", "surprise", "neutral"], 1: ["neutral", "neutral"]}) == \
        "2 faces visible: surprise, neutral"
    assert face_gloss({}) == "no faces visible"
    assert face_gloss({0: []}) == "no faces visible"


def test_embed_prompt_bank_is_l2_normalised(monkeypatch):
    from meld_emotion.inference.gloss import SCENE_PROMPT_BANK, embed_prompt_bank

    class _StubOut:
        def __init__(self, n):
            self.pooler_output = torch.randn(n, 8) * 10  # deliberately not unit norm

    class _StubModel:
        def get_text_features(self, **kwargs):
            return _StubOut(len(SCENE_PROMPT_BANK))

    class _StubTokenizer:
        def __call__(self, texts, return_tensors, padding):
            return {"input_ids": torch.zeros(len(texts), 1, dtype=torch.long)}.__class__(
                {"input_ids": torch.zeros(len(texts), 1, dtype=torch.long)})
        def to(self, device):
            return self

    embeddings = embed_prompt_bank(_StubModel(), _StubTokenizer(), "cpu")
    norms = embeddings.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_gloss.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meld_emotion.inference.gloss'`

- [ ] **Step 3: Implement**

Create `src/meld_emotion/inference/gloss.py`:

```python
"""Visual gloss (design doc §4.4): human-readable cues from tensors already
computed, with no extra forward passes beyond one CLIP text-tower encode
(the bank, cached once). Feeds the response LM only -- never the
classifier, so the classifier's accuracy is never bottlenecked by a
hand-written vocabulary."""
import torch
from transformers import CLIPTokenizer

from meld_emotion.vision.encoders import CLIP_MODEL_ID

SCENE_PROMPT_BANK = (
    "one person", "two people talking", "a group of people", "a dimly lit room",
    "a brightly lit room", "people sitting on a couch", "someone standing",
    "someone leaning in close", "an animated gesture", "someone with arms crossed",
    "a person laughing", "a person crying", "people in a kitchen", "people in a coffee shop",
    "an empty background", "someone pointing", "a crowded room", "two people arguing",
    "a quiet, calm scene", "someone holding an object", "people hugging",
    "a person looking away from the camera", "a close-up of a face", "a wide shot of a room",
)


def build_clip_tokenizer() -> CLIPTokenizer:
    return CLIPTokenizer.from_pretrained(CLIP_MODEL_ID)


def embed_prompt_bank(clip_model, clip_tokenizer, device: str) -> torch.Tensor:
    inputs = clip_tokenizer(list(SCENE_PROMPT_BANK), return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = clip_model.get_text_features(**inputs)
        feats = out if isinstance(out, torch.Tensor) else out.pooler_output
    return torch.nn.functional.normalize(feats, dim=-1)


def scene_gloss(mean_scene_embedding, bank_embeddings: torch.Tensor, margin: float = 0.05) -> list[str]:
    query = torch.nn.functional.normalize(
        torch.as_tensor(mean_scene_embedding, dtype=torch.float32).unsqueeze(0), dim=-1)
    sims = (query @ bank_embeddings.T).squeeze(0) * 100.0  # CLIP's logit scale
    probs = torch.softmax(sims, dim=-1)
    top2 = torch.topk(probs, min(2, len(probs)))
    if len(top2.values) < 2 or (top2.values[0] - top2.values[1]).item() < margin:
        return [SCENE_PROMPT_BANK[top2.indices[0]]]
    return [SCENE_PROMPT_BANK[i] for i in top2.indices.tolist()]


def face_gloss(track_expressions: dict[int, list[str]]) -> str:
    labels = []
    for track_id in sorted(track_expressions):
        exprs = track_expressions[track_id]
        if not exprs:
            continue
        counts: dict[str, int] = {}
        for e in exprs:
            counts[e] = counts.get(e, 0) + 1
        labels.append(max(counts, key=counts.get))
    if not labels:
        return "no faces visible"
    return f"{len(labels)} face{'s' if len(labels) != 1 else ''} visible: " + ", ".join(labels)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_gloss.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/meld_emotion/inference/gloss.py tests/test_gloss.py
git commit -m "Add CLIP scene-cue and per-track face-label visual gloss"
```

---

### Task 5: TurnProcessor — incremental vision pipeline and fused prediction

**Files:**
- Create: `src/meld_emotion/inference/turn.py`
- Test: `tests/test_turn.py`

**Interfaces:**
- Consumes: `InferenceBundle` (Task 2); `correct_provisional` (Task 3);
  `meld_emotion.data.preprocess.{letterbox, crop_face}`;
  `meld_emotion.training.dataset.{collate, remap_track_ids}`;
  `meld_emotion.training.text.{format_context, encode_text}`;
  `meld_emotion.vision.face_detector.detect_faces`;
  `meld_emotion.vision.tracker.{FaceTracker, SHOT_CUT_THRESHOLD,
  shot_change_score}`.
- Produces: `TurnProcessor(bundle: InferenceBundle, prior: np.ndarray,
  emitter: EventEmitter)` with `.start_turn(turn_id: str)`,
  `.push_frame(frame_bgr: np.ndarray) -> dict` (the emitted `provisional`
  event), `.end_turn(text: str, context_prev: list[str], visual_cues:
  list[str] | None = None) -> dict` (the emitted `final` event —
  `visual_cues` defaults to `[]`; the caller computes real cues from the
  two properties below and passes them in before calling `end_turn`, since
  the event is built and emitted exactly once), `.track_expressions ->
  dict[int, list[str]]` (per-track MELD-label history for
  `gloss.face_gloss`), `.mean_scene_embedding -> np.ndarray` (for
  `gloss.scene_gloss`), `.last_boxes -> list[tuple]` /
  `.last_track_ids -> list[int]` (the most recent `push_frame`'s raw
  detector boxes and track ids — demo-only, not part of the event schema,
  for `replay_demo.py` to draw between sampled frames).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_turn.py`:

```python
import numpy as np
import pytest
import torch

from conftest import StubTextEncoder


class _StubFaceDetector:
    pass


class _StubFaceEncoder:
    meld_labels = ("neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear")

    def encode_batch(self, images):
        n = len(images)
        probs = np.zeros((n, 7), dtype=np.float32)
        probs[:, 1] = 1.0  # every face confidently "joy" in its own native order
        return {"features": np.ones((n, 8), dtype=np.float32), "probs": probs}


class _StubSceneEncoder:
    def encode(self, image):
        return np.full(4, 0.5, dtype=np.float32)


class _StubTokenizer:
    pad_token_id = 1

    def __call__(self, *args, **kwargs):
        raise AssertionError("turn.py must go through encode_text, not call the tokenizer directly")


def _bundle(use_faces=True, use_scene=True):
    from meld_emotion.inference.loader import InferenceBundle
    from meld_emotion.training.config import TrainConfig
    from meld_emotion.training.model import FusionModel
    config = TrainConfig(d_model=32, n_heads=4, ff_dim=64, n_layers=1,
                         use_faces=use_faces, use_scene=use_scene, context_k=2)
    text_encoder = StubTextEncoder(32)
    model = FusionModel(config, text_encoder, face_dim=8, scene_dim=4).eval()
    return InferenceBundle(model=model, config=config, tokenizer=_StubTokenizer(),
                           face_detector=_StubFaceDetector(), face_encoder=_StubFaceEncoder(),
                           scene_encoder=_StubSceneEncoder(), device="cpu")


def _fake_frame(color=(50, 50, 50)):
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:] = color
    return frame


def test_push_frame_emits_a_provisional_event_and_never_shows_raw_softmax(monkeypatch):
    from meld_emotion.inference import turn as turn_module
    monkeypatch.setattr(turn_module, "detect_faces", lambda detector, frame: [(0, 0, 10, 10, 0.9)])
    monkeypatch.setattr(turn_module, "encode_text",
                        lambda tok, ctx, cur, max_len: {"input_ids": [0, 1], "attention_mask": [1, 1]})
    prior = np.array([0.5, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)  # neutral over-weighted in prior
    events = []
    tp = turn_module.TurnProcessor(_bundle(), prior, emitter=type("E", (), {"emit": staticmethod(events.append)})())
    tp.start_turn("dia1_utt1")
    event = tp.push_frame(_fake_frame())
    assert event["phase"] == "provisional" and event["turn_id"] == "dia1_utt1" and event["frame"] == 0
    assert event["faces_seen"] == 1
    # raw softmax always says "joy"=1.0 -- after prior correction "neutral" should NOT dominate
    # (prior is high on neutral -> corrected/prior shrinks neutral's share relative to raw joy=1.0)
    assert event["provisional_expression"]["joy"] > event["provisional_expression"]["neutral"]
    assert events[-1] == event


def test_shot_cut_between_sampled_frames_resets_the_tracker(monkeypatch):
    from meld_emotion.inference import turn as turn_module
    calls = []
    monkeypatch.setattr(turn_module, "detect_faces", lambda detector, frame: [(0, 0, 10, 10, 0.9)])

    class _Tracker:
        def __init__(self): self.reset_calls = 0
        def reset(self): self.reset_calls += 1
        def update(self, boxes): calls.append("update"); return [0] * len(boxes)
    tracker = _Tracker()
    monkeypatch.setattr(turn_module, "FaceTracker", lambda: tracker)
    monkeypatch.setattr(turn_module, "shot_change_score", lambda a, b: 0.9)  # always "a cut"

    tp = turn_module.TurnProcessor(_bundle(), np.ones(7) / 7, emitter=type("E", (), {"emit": lambda self, e: None})())
    tp.start_turn("dia1_utt1")
    tp.push_frame(_fake_frame((10, 10, 10)))
    tp.push_frame(_fake_frame((200, 200, 200)))  # visually different frame -> forced "cut" above
    assert tracker.reset_calls >= 2  # once in start_turn, once for the detected cut on frame 2


def test_end_turn_builds_the_training_shaped_batch_and_emits_final_without_response(monkeypatch):
    from meld_emotion.inference import turn as turn_module
    monkeypatch.setattr(turn_module, "detect_faces", lambda detector, frame: [(0, 0, 10, 10, 0.9)])
    monkeypatch.setattr(turn_module, "encode_text",
                        lambda tok, ctx, cur, max_len: {"input_ids": [0, 5, 6, 1], "attention_mask": [1, 1, 1, 1]})
    events = []
    tp = turn_module.TurnProcessor(_bundle(), np.ones(7) / 7,
                                   emitter=type("E", (), {"emit": staticmethod(events.append)})())
    tp.start_turn("dia1_utt1")
    tp.push_frame(_fake_frame())
    tp.push_frame(_fake_frame())
    final = tp.end_turn("You did WHAT?", ["earlier line one", "earlier line two"])
    assert final["phase"] == "final" and "response" not in final and "latency_ms" not in final
    assert final["text"] == "You did WHAT?"
    assert set(final["emotion_probs"]) == {"neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"}
    assert final["emotion"] in final["emotion_probs"]
    assert final["faces_seen"] == 1
    assert events[-1] == final


def test_end_turn_carries_the_caller_supplied_visual_cues_and_defaults_to_empty(monkeypatch):
    from meld_emotion.inference import turn as turn_module
    monkeypatch.setattr(turn_module, "detect_faces", lambda detector, frame: [])
    monkeypatch.setattr(turn_module, "encode_text",
                        lambda tok, ctx, cur, max_len: {"input_ids": [0, 1], "attention_mask": [1, 1]})
    tp = turn_module.TurnProcessor(_bundle(), np.ones(7) / 7, emitter=type("E", (), {"emit": lambda self, e: None})())
    tp.start_turn("dia1_utt1")
    with_cues = tp.end_turn("hello", [], visual_cues=["two people", "1 face visible: joy"])
    assert with_cues["visual_cues"] == ["two people", "1 face visible: joy"]
    tp.start_turn("dia1_utt2")
    without_cues = tp.end_turn("hello again", [])
    assert without_cues["visual_cues"] == []


def test_end_turn_with_zero_frames_still_produces_finite_predictions(monkeypatch):
    from meld_emotion.inference import turn as turn_module
    monkeypatch.setattr(turn_module, "encode_text",
                        lambda tok, ctx, cur, max_len: {"input_ids": [0, 1], "attention_mask": [1, 1]})
    tp = turn_module.TurnProcessor(_bundle(), np.ones(7) / 7, emitter=type("E", (), {"emit": lambda self, e: None})())
    tp.start_turn("dia1_utt1")
    final = tp.end_turn("hello", [])
    assert final["faces_seen"] == 0
    assert all(v == v for v in final["emotion_probs"].values())  # no NaN


def test_track_expressions_and_mean_scene_embedding_accumulate_across_the_turn(monkeypatch):
    from meld_emotion.inference import turn as turn_module
    monkeypatch.setattr(turn_module, "detect_faces", lambda detector, frame: [(0, 0, 10, 10, 0.9)])
    tp = turn_module.TurnProcessor(_bundle(), np.ones(7) / 7, emitter=type("E", (), {"emit": lambda self, e: None})())
    tp.start_turn("dia1_utt1")
    tp.push_frame(_fake_frame())
    tp.push_frame(_fake_frame())
    assert tp.track_expressions == {0: ["joy", "joy"]}  # stub face encoder always says "joy"
    assert np.allclose(tp.mean_scene_embedding, 0.5)  # stub scene encoder always returns 0.5s
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_turn.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meld_emotion.inference.turn'`

- [ ] **Step 3: Implement**

Create `src/meld_emotion/inference/turn.py`:

```python
"""One inference core (design doc §2, §4.2, §4.3): TurnProcessor.push_frame
runs the same per-frame vision logic preprocess_clip uses -- shot-cut check
against the previous SAMPLED frame, then face detect/track/encode, then
scene encode -- and publishes a prior-corrected provisional expression
state. end_turn assembles the exact model input shape training used
(frame-position clamp, track-ID remap, dialogue-context text) and runs the
fused model for the final event, which never carries a response: the LM is
a separate, slower step so it never gates the state (design doc §2)."""
from typing import Optional

import numpy as np
import torch
from PIL import Image

from meld_emotion.data.labels import EMOTIONS, SENTIMENTS
from meld_emotion.data.preprocess import crop_face, letterbox
from meld_emotion.inference.loader import InferenceBundle
from meld_emotion.inference.prior import correct_provisional
from meld_emotion.training.dataset import collate, remap_track_ids
from meld_emotion.training.text import encode_text, format_context
from meld_emotion.vision.face_detector import detect_faces
from meld_emotion.vision.tracker import FaceTracker, SHOT_CUT_THRESHOLD, shot_change_score


class TurnProcessor:
    def __init__(self, bundle: InferenceBundle, prior: np.ndarray, emitter):
        self.bundle = bundle
        self.prior = prior
        self.emitter = emitter
        self.turn_id: Optional[str] = None
        self._tracker = FaceTracker()
        self._prev_frame = None
        self._face_features: list[np.ndarray] = []
        self._face_frame_idx: list[int] = []
        self._face_track_raw: list[int] = []
        self._face_probs_sum = np.zeros(7, dtype=np.float64)
        self._face_probs_count = 0
        self._scene_features: list[np.ndarray] = []
        self.track_expressions: dict[int, list[str]] = {}
        self.last_boxes: list[tuple] = []      # demo-only convenience: not part of the event schema
        self.last_track_ids: list[int] = []

    def start_turn(self, turn_id: str) -> None:
        self.turn_id = turn_id
        self._tracker.reset()  # one tracker per TurnProcessor lifetime; reset (not replaced) at each turn boundary
        self._prev_frame = None
        self._face_features, self._face_frame_idx, self._face_track_raw = [], [], []
        self._face_probs_sum = np.zeros(7, dtype=np.float64)
        self._face_probs_count = 0
        self._scene_features = []
        self.track_expressions = {}
        self.last_boxes, self.last_track_ids = [], []

    @property
    def mean_scene_embedding(self) -> np.ndarray:
        if not self._scene_features:
            return np.zeros(self.bundle.scene_encoder.encode.__self__.feature_dim
                            if hasattr(self.bundle.scene_encoder, "feature_dim") else 512, dtype=np.float32)
        return np.mean(self._scene_features, axis=0)

    def push_frame(self, frame_bgr: np.ndarray) -> dict:
        sample_i = len(self._scene_features)
        change = shot_change_score(self._prev_frame, frame_bgr) if self._prev_frame is not None else 0.0
        if change >= SHOT_CUT_THRESHOLD:
            self._tracker.reset()
        self._prev_frame = frame_bgr

        boxes = detect_faces(self.bundle.face_detector, frame_bgr)
        track_ids = self._tracker.update(boxes)
        self.last_boxes, self.last_track_ids = boxes, track_ids  # demo-only: for drawing between sampled frames
        crops, valid_track_ids = [], []
        for (x, y, w, h, score), track_id in zip(boxes, track_ids):
            crop = crop_face(frame_bgr, (x, y, w, h))
            if crop is None:
                continue
            crops.append(Image.fromarray(crop[:, :, ::-1]))
            valid_track_ids.append(track_id)

        enc = self.bundle.face_encoder.encode_batch(crops)
        self._face_features.append(enc["features"])
        self._face_frame_idx += [sample_i] * len(valid_track_ids)
        self._face_track_raw += valid_track_ids
        if len(enc["probs"]):
            self._face_probs_sum += enc["probs"].sum(axis=0)
            self._face_probs_count += len(enc["probs"])

        meld_order = self.bundle.face_encoder.meld_labels
        for track_id, probs in zip(valid_track_ids, enc["probs"]):
            label = meld_order[int(np.argmax(probs))]
            self.track_expressions.setdefault(track_id, []).append(label)

        scene_img = Image.fromarray(letterbox(frame_bgr)[:, :, ::-1])
        self._scene_features.append(self.bundle.scene_encoder.encode(scene_img))

        raw = (self._face_probs_sum / self._face_probs_count) if self._face_probs_count else np.ones(7) / 7
        provisional = correct_provisional(raw, self.prior, self.bundle.face_encoder.meld_labels)
        event = {"turn_id": self.turn_id, "phase": "provisional", "frame": sample_i,
                 "faces_seen": len(crops), "provisional_expression": provisional}
        self.emitter.emit(event)
        return event

    def end_turn(self, text: str, context_prev: list[str], visual_cues: list[str] | None = None) -> dict:
        """visual_cues is computed by the caller (Task 8) from .mean_scene_embedding
        and .track_expressions -- both already fully populated once every
        push_frame for the turn has run, so the caller can compute them any
        time before calling end_turn. Passed in rather than patched onto the
        returned dict afterward, since the event is emitted exactly once,
        here, with everything it will ever carry."""
        cfg = self.bundle.config
        face_feat = (np.concatenate(self._face_features) if self._face_features
                    else np.zeros((0, 8), dtype=np.float32))
        face_frame = np.minimum(np.array(self._face_frame_idx, dtype=np.int64), cfg.max_frames - 1)
        face_track = remap_track_ids(np.array(self._face_track_raw, dtype=np.int64), cfg.max_track_slots)
        scene_feat = (np.stack(self._scene_features).astype(np.float32) if self._scene_features
                     else np.zeros((0, 4), dtype=np.float32))
        scene_frame = np.minimum(np.arange(len(self._scene_features), dtype=np.int64), cfg.max_frames - 1)

        context = format_context(context_prev, cfg.context_k)
        enc = encode_text(self.bundle.tokenizer, context, text, cfg.max_text_tokens)
        item = {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "face_feat": torch.from_numpy(np.ascontiguousarray(face_feat, dtype=np.float32)),
            "face_frame": torch.from_numpy(face_frame),
            "face_track": torch.from_numpy(face_track),
            "scene_feat": torch.from_numpy(np.ascontiguousarray(scene_feat, dtype=np.float32)),
            "scene_frame": torch.from_numpy(scene_frame),
            "emotion": torch.tensor(0, dtype=torch.long),
            "sentiment": torch.tensor(0, dtype=torch.long),
            "clip": self.turn_id,
        }
        pad_id = self.bundle.tokenizer.pad_token_id if self.bundle.tokenizer is not None else 0
        batch = collate([item], pad_id=pad_id)
        batch = {k: (v.to(self.bundle.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            out = self.bundle.model(batch)
        emotion_probs = torch.softmax(out["emotion_logits"], dim=-1)[0].cpu().numpy()
        sentiment_probs = torch.softmax(out["sentiment_logits"], dim=-1)[0].cpu().numpy()
        emotion = EMOTIONS[int(emotion_probs.argmax())]
        sentiment = SENTIMENTS[int(sentiment_probs.argmax())]

        event = {"turn_id": self.turn_id, "phase": "final", "text": text, "emotion": emotion,
                 "emotion_probs": {e: float(p) for e, p in zip(EMOTIONS, emotion_probs)},
                 "sentiment": sentiment,
                 "sentiment_probs": {s: float(p) for s, p in zip(SENTIMENTS, sentiment_probs)},
                 "faces_seen": max(self._face_frame_idx.count(i) for i in range(len(self._scene_features)))
                              if self._face_frame_idx else 0,
                 "visual_cues": visual_cues or []}
        self.emitter.emit(event)
        return event
```

Note on `visual_cues`: `turn.py` never imports `gloss.py`. The caller
(Task 8) computes cues from `TurnProcessor.mean_scene_embedding`/
`.track_expressions` via `gloss.scene_gloss`/`gloss.face_gloss` (using the
CLIP bank it already built once, not re-embedded per turn) and passes the
result into `end_turn`, so the `final` event is correct the one time it is
built and emitted — never patched after the fact.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_turn.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/meld_emotion/inference/turn.py tests/test_turn.py
git commit -m "Add TurnProcessor: incremental per-frame vision to provisional/final events"
```

---

### Task 6: Response LM — selection benchmark and streaming wrapper

**Files:**
- Create: `src/meld_emotion/inference/responder.py`
- Create: `scripts/select_lm.py`
- Test: `tests/test_responder.py`

**Interfaces:**
- Produces: `SYSTEM_PROMPT: str`; `build_prompt(context_prev: list[str],
  text: str, emotion: str, top2_probs: list[tuple[str, float]], sentiment:
  str, visual_cues: list[str]) -> list[dict]` (chat-format messages);
  `Responder(model_repo: str | None = None, max_tokens: int = 40, *,
  model=None, tokenizer=None)` with `.stream(messages: list[dict],
  generate_fn=None) -> Iterator[str]` (yields incremental text deltas —
  `generate_fn` defaults to `mlx_lm.stream_generate`, injectable for
  offline tests).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_responder.py`:

```python
from types import SimpleNamespace

import pytest


def test_build_prompt_uses_last_4_context_lines_and_never_speaker_names():
    from meld_emotion.inference.responder import build_prompt
    messages = build_prompt(["l0", "l1", "l2", "l3", "l4"], "current line", "surprise",
                            [("surprise", 0.6), ("neutral", 0.2)], "negative", ["two people"])
    assert messages[0]["role"] == "system"
    user = messages[1]["content"]
    assert "l0" not in user  # only the last 4 (l1..l4) are kept — l0 is the 5th-from-last, dropped
    assert "l4" in user and "current line" in user
    assert "surprise" in user and "60%" in user and "negative" in user and "two people" in user
    assert "Speaker" not in user and "speaker" not in user


def test_build_prompt_stays_short():
    from meld_emotion.inference.responder import build_prompt
    messages = build_prompt(["a short line"] * 4, "another short line", "joy",
                            [("joy", 0.9), ("neutral", 0.05)], "positive", ["one person"])
    total_chars = sum(len(m["content"]) for m in messages)
    assert total_chars < 800  # ~200 tokens at a generous 4 chars/token


def test_responder_stream_yields_incremental_text_and_uses_the_chat_template():
    from meld_emotion.inference.responder import Responder

    class _StubTokenizer:
        def apply_chat_template(self, messages, add_generation_prompt, enable_thinking=None):
            assert add_generation_prompt is True
            return "PROMPT"

    def fake_generate(model, tokenizer, prompt, max_tokens):
        assert prompt == "PROMPT" and max_tokens == 40
        for text in ["Wow", " really?"]:
            yield SimpleNamespace(text=text, finish_reason=None)
        yield SimpleNamespace(text="", finish_reason="stop")

    responder = Responder(model=object(), tokenizer=_StubTokenizer())
    chunks = list(responder.stream([{"role": "user", "content": "hi"}], generate_fn=fake_generate))
    assert chunks == ["Wow", " really?", ""]
    assert "".join(chunks) == "Wow really?"


def test_responder_stream_falls_back_when_enable_thinking_is_unsupported():
    from meld_emotion.inference.responder import Responder

    class _StubTokenizerNoThinking:
        def apply_chat_template(self, messages, add_generation_prompt):
            return "PROMPT_NO_THINKING"

    def fake_generate(model, tokenizer, prompt, max_tokens):
        assert prompt == "PROMPT_NO_THINKING"
        yield SimpleNamespace(text="ok", finish_reason="stop")

    responder = Responder(model=object(), tokenizer=_StubTokenizerNoThinking())
    assert list(responder.stream([{"role": "user", "content": "hi"}], generate_fn=fake_generate)) == ["ok"]


@pytest.mark.network
def test_responder_loads_a_real_mlx_model_and_streams_a_real_response():
    from meld_emotion.inference.responder import Responder
    responder = Responder(model_repo="mlx-community/Qwen2.5-1.5B-Instruct-4bit", max_tokens=10)
    chunks = list(responder.stream([{"role": "user", "content": "Say hi in one word."}]))
    assert len("".join(chunks)) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_responder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meld_emotion.inference.responder'`

- [ ] **Step 3: Implement**

Create `src/meld_emotion/inference/responder.py`:

```python
"""Response generation (design doc §4.4): a fixed persona, prompted (not
fine-tuned) instruction LM, served locally via mlx-lm at 4-bit, streaming
tokens into the event stream. By the time this runs, the cross-modal
reasoning is already done (turn.py) -- its only job is a short, in-
character reactive line."""
SYSTEM_PROMPT = ("You are a small, friendly character robot. React to what the person just "
                 "said and how they seem to feel, in at most two sentences. Don't summarize "
                 "what they said back to them -- react to it. Never invent facts you weren't told.")


def build_prompt(context_prev: list[str], text: str, emotion: str, top2_probs: list[tuple[str, float]],
                 sentiment: str, visual_cues: list[str]) -> list[dict]:
    lines = "\n".join(context_prev[-4:])
    top2_str = ", ".join(f"{e} ({p:.0%})" for e, p in top2_probs)
    cues_str = ", ".join(visual_cues) if visual_cues else "none"
    user = (f"{lines}\n{text}\n\n"
           f"[Detected emotion: {top2_str}. Sentiment: {sentiment}. Visual cues: {cues_str}.]")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


class Responder:
    def __init__(self, model_repo: str | None = None, max_tokens: int = 40, *, model=None, tokenizer=None):
        if model is None or tokenizer is None:
            from mlx_lm import load
            model, tokenizer = load(model_repo)
        self.model, self.tokenizer, self.max_tokens = model, tokenizer, max_tokens

    def stream(self, messages: list[dict], generate_fn=None):
        if generate_fn is None:
            from mlx_lm import stream_generate as generate_fn
        try:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        for response in generate_fn(self.model, self.tokenizer, prompt, max_tokens=self.max_tokens):
            yield response.text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_responder.py -v`
Expected: 4 passed, 1 deselected

- [ ] **Step 5: Write the LM selection benchmark script**

Create `scripts/select_lm.py`:

```python
#!/usr/bin/env python3
"""Benchmark response-LM candidates on this machine and pick the largest
that meets the replay latency targets (design doc §2, §4.1). Candidates
are real, verified mlx-community 4-bit checkpoints, ordered largest to
smallest; the smallest is the design doc's explicit latency fallback.

Usage:
    uv run --with mlx-lm python scripts/select_lm.py
"""
import json
import time
from pathlib import Path

from meld_emotion.config import REPO_ROOT

CANDIDATES = [
    "mlx-community/Qwen3-4B-4bit",
    "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "mlx-community/Phi-3.5-mini-instruct-4bit",
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
]
FIRST_TOKEN_TARGET_S = 1.0
DONE_TARGET_S = 2.5
MAX_TOKENS = 40
SAMPLE_MESSAGES = [
    {"role": "system", "content": "You are a small, friendly character robot. React in at most two sentences."},
    {"role": "user", "content": "You did WHAT?\n\n[Detected emotion: surprise (60%), neutral (20%). "
                                "Sentiment: negative. Visual cues: two people, dimly lit room.]"},
]


def benchmark(repo: str) -> dict:
    from mlx_lm import load, stream_generate
    model, tokenizer = load(repo)
    try:
        prompt = tokenizer.apply_chat_template(SAMPLE_MESSAGES, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tokenizer.apply_chat_template(SAMPLE_MESSAGES, add_generation_prompt=True)
    start = time.perf_counter()
    first_token_s, n_tokens = None, 0
    for response in stream_generate(model, tokenizer, prompt, max_tokens=MAX_TOKENS):
        if first_token_s is None:
            first_token_s = time.perf_counter() - start
        n_tokens += 1
    done_s = time.perf_counter() - start
    return {"repo": repo, "first_token_s": round(first_token_s, 3) if first_token_s else None,
           "done_s": round(done_s, 3), "n_tokens": n_tokens,
           "meets_target": bool(first_token_s and first_token_s <= FIRST_TOKEN_TARGET_S and done_s <= DONE_TARGET_S)}


def main():
    results = [benchmark(repo) for repo in CANDIDATES]
    for r in results:
        print(f"{r['repo']}: first_token={r['first_token_s']}s done={r['done_s']}s "
             f"tokens={r['n_tokens']} meets_target={r['meets_target']}")
    winners = [r for r in results if r["meets_target"]]
    chosen = winners[0] if winners else results[-1]
    print(f"\nChosen: {chosen['repo']}")
    out = {"candidates": results, "chosen": chosen["repo"]}
    (REPO_ROOT / "results").mkdir(parents=True, exist_ok=True)
    Path(REPO_ROOT / "results" / "lm_selection.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run it for real and record the choice**

Run: `uv run --with mlx-lm python scripts/select_lm.py`
Expected: four lines of measured timings (first download will pull each
4-bit checkpoint, ~1-3GB apiece) and a `Chosen: mlx-community/...` line;
`results/lm_selection.json` written. `CANDIDATES` is ordered largest-first
so the loop naturally prefers the largest model that clears both targets;
record the chosen repo — later tasks pass it as `--model-repo`.

- [ ] **Step 7: Commit**

```bash
git add src/meld_emotion/inference/responder.py scripts/select_lm.py tests/test_responder.py
git commit -m "Add mlx-lm response wrapper and the latency-driven model selection benchmark"
```

---

### Task 7: Curated demo clip candidates

**Files:**
- Modify: `pyproject.toml` (add `pythonpath` so `tests/` can `import scripts`)
- Create: `scripts/pick_demo_clips.py`
- Test: `tests/test_pick_demo_clips.py`

This is the first task whose tests import a top-level `scripts` module
(`from scripts.pick_demo_clips import ...`), which pytest cannot resolve
without `rootdir` on `sys.path` — every later task's tests
(`test_consistency_check.py`, `test_measure_latency.py`,
`test_param_budget.py`, `test_response_rubric.py`) need the same fix, so it
belongs here, once, ahead of all of them.

- [ ] **Step 1: Make `scripts` importable from tests**

Edit `pyproject.toml`'s `[tool.pytest.ini_options]` (already exists from
the Stage 1 training plan) to add `pythonpath`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "-m 'not network'"
markers = [
    "network: downloads pretrained weights from Hugging Face; deselected by default, run with `uv run pytest -m network`",
]
```

Run: `uv run pytest -v`
Expected: same pass count as before this edit (currently 6 passed, 1
deselected after Task 6 — this step only adds an import path, no new
tests yet) — confirms the edit didn't break test discovery.

```bash
git add pyproject.toml
git commit -m "Make scripts/ importable from tests (pythonpath) ahead of the scripts.* test modules"
```

**Interfaces:**
- Consumes: `load_inference_bundle` (Task 2); `meld_emotion.training.dataset
  .{load_ok_rows, collate, MeldFeatureDataset}`; `meld_emotion.training.train
  .evaluate`; `meld_emotion.data.labels.EMOTIONS`.
- Produces: `predict_test_split(bundle, features_dir: Path) -> list[dict]`
  (one dict per ok test row: `clip, true_emotion, pred_emotion, max_n_faces,
  total_faces`); `stratified_candidates(predictions: list[dict], per_emotion:
  int = 2, seed: int = 0) -> list[dict]`; `hard_case_candidates(predictions:
  list[dict]) -> dict[str, list[dict]]` (keys `"many_faces"` — top-5 by
  `max_n_faces`, `"zero_faces"` — up to 5 with `total_faces == 0`);
  `wrong_prediction_candidates(predictions: list[dict], seed: int = 0) ->
  list[dict]` (up to 3 where `true_emotion != pred_emotion`, sampled).

- [ ] **Step 2: Write the failing tests**

Create `tests/test_pick_demo_clips.py`:

```python
def _predictions():
    return [
        {"clip": "dia1_utt0", "true_emotion": "joy", "pred_emotion": "joy", "max_n_faces": 1, "total_faces": 3},
        {"clip": "dia1_utt1", "true_emotion": "joy", "pred_emotion": "neutral", "max_n_faces": 6, "total_faces": 12},
        {"clip": "dia1_utt2", "true_emotion": "anger", "pred_emotion": "anger", "max_n_faces": 0, "total_faces": 0},
        {"clip": "dia1_utt3", "true_emotion": "anger", "pred_emotion": "sadness", "max_n_faces": 2, "total_faces": 4},
        {"clip": "dia1_utt4", "true_emotion": "neutral", "pred_emotion": "neutral", "max_n_faces": 1, "total_faces": 2},
    ]


def test_stratified_candidates_samples_per_emotion():
    from meld_emotion.training.dataset import EMOTION_TO_IDX
    from scripts.pick_demo_clips import stratified_candidates
    preds = _predictions()
    picked = stratified_candidates(preds, per_emotion=1, seed=0)
    emotions_picked = {p["true_emotion"] for p in picked}
    assert emotions_picked.issubset({"joy", "anger", "neutral"})
    assert len(picked) <= 3


def test_hard_case_candidates_finds_many_faces_and_zero_faces():
    from scripts.pick_demo_clips import hard_case_candidates
    result = hard_case_candidates(_predictions())
    assert result["many_faces"][0]["clip"] == "dia1_utt1"
    assert any(c["clip"] == "dia1_utt2" for c in result["zero_faces"])


def test_wrong_prediction_candidates_finds_mismatches_only():
    from scripts.pick_demo_clips import wrong_prediction_candidates
    result = wrong_prediction_candidates(_predictions())
    assert {c["clip"] for c in result} == {"dia1_utt1", "dia1_utt3"}


def test_predict_test_split_runs_the_real_forward_pass(synthetic_features):
    from conftest import StubTextEncoder
    from meld_emotion.inference.loader import InferenceBundle
    from meld_emotion.training.config import TrainConfig
    from meld_emotion.training.model import FusionModel
    from scripts.pick_demo_clips import predict_test_split
    root, train_manifest, dev_manifest = synthetic_features
    config = TrainConfig(d_model=32, n_heads=4, ff_dim=64, n_layers=1)
    model = FusionModel(config, StubTextEncoder(32), face_dim=8, scene_dim=4).eval()

    class _StubTokenizer:
        pad_token_id = 1
    bundle = InferenceBundle(model=model, config=config, tokenizer=_StubTokenizer(),
                             face_detector=None, face_encoder=None, scene_encoder=None, device="cpu")
    from conftest import whitespace_encode
    predictions = predict_test_split(bundle, root, split="dev", encode_fn=whitespace_encode)
    assert len(predictions) == 14   # dev fixture has 14 ok rows
    assert all(p["true_emotion"] in {"neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"}
              for p in predictions)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_pick_demo_clips.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.pick_demo_clips'`

- [ ] **Step 4: Implement**

Create `scripts/pick_demo_clips.py`:

```python
#!/usr/bin/env python3
"""Curated demo clip candidates for the replay window (design doc §8.1):
stratified by emotion, at least one wrong prediction, and heuristic
candidates for the hard shot compositions from design doc §5 (a many-face
ensemble frame, a zero-detected-face clip). This produces CANDIDATE lists
for a human to pick the final ~10 from -- design doc §5's specific hard
clips were identified by eye during exploration, not by a reusable rule,
so the many-faces/zero-faces heuristics here are a starting point, not a
guarantee of the exact same clips.

Usage:
    uv run python scripts/pick_demo_clips.py --checkpoint results/fusion/seed0/best.pt
"""
import argparse
import json
import random
from pathlib import Path

from meld_emotion.config import FEATURE_CACHE_DIR, REPO_ROOT
from meld_emotion.data.labels import EMOTIONS
from meld_emotion.inference.loader import load_inference_bundle
from meld_emotion.training.dataset import collate, load_ok_rows, MeldFeatureDataset
from torch.utils.data import DataLoader


def predict_test_split(bundle, features_dir: Path, split: str = "test", encode_fn=None) -> list[dict]:
    import functools
    import torch

    from meld_emotion.training.text import encode_text

    rows = load_ok_rows(Path(features_dir) / split / "manifest.jsonl")
    if encode_fn is None:
        encode_fn = functools.partial(encode_text, bundle.tokenizer, max_length=bundle.config.max_text_tokens)
    dataset = MeldFeatureDataset(rows, features_dir, bundle.config, encode_fn)
    loader = DataLoader(dataset, batch_size=16, shuffle=False,
                        collate_fn=lambda b: collate(b, pad_id=bundle.tokenizer.pad_token_id))
    predictions = []
    row_by_clip = {f"dia{r['dialogue_id']}_utt{r['utterance_id']}": r for r in rows}
    with torch.no_grad():
        for batch in loader:
            out = bundle.model(batch)
            preds = out["emotion_logits"].argmax(-1).tolist()
            for clip, pred_idx in zip(batch["clips"], preds):
                row = row_by_clip[clip]
                predictions.append({"clip": clip, "true_emotion": row["emotion"],
                                    "pred_emotion": EMOTIONS[pred_idx],
                                    "max_n_faces": row["n_faces"], "total_faces": row["n_faces"]})
    return predictions


def stratified_candidates(predictions: list[dict], per_emotion: int = 2, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    by_emotion: dict[str, list[dict]] = {}
    for p in predictions:
        by_emotion.setdefault(p["true_emotion"], []).append(p)
    out = []
    for group in by_emotion.values():
        out.extend(rng.sample(group, min(per_emotion, len(group))))
    return out


def hard_case_candidates(predictions: list[dict]) -> dict[str, list[dict]]:
    many_faces = sorted(predictions, key=lambda p: -p["max_n_faces"])[:5]
    zero_faces = [p for p in predictions if p["total_faces"] == 0][:5]
    return {"many_faces": many_faces, "zero_faces": zero_faces}


def wrong_prediction_candidates(predictions: list[dict], seed: int = 0, limit: int = 3) -> list[dict]:
    rng = random.Random(seed)
    wrong = [p for p in predictions if p["true_emotion"] != p["pred_emotion"]]
    return rng.sample(wrong, min(limit, len(wrong)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "results" / "fusion" / "seed0" / "best.pt")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "demo_clip_candidates.json")
    args = parser.parse_args()

    bundle = load_inference_bundle(args.checkpoint, device="cpu")  # cpu: this is a one-off batch pass, not the demo
    predictions = predict_test_split(bundle, FEATURE_CACHE_DIR, split="test")
    result = {"stratified": stratified_candidates(predictions), **hard_case_candidates(predictions),
             "wrong_predictions": wrong_prediction_candidates(predictions)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"stratified={len(result['stratified'])} many_faces={len(result['many_faces'])} "
         f"zero_faces={len(result['zero_faces'])} wrong={len(result['wrong_predictions'])}")
    print(f"-> {args.out}  (pick ~10 total by hand, save the final list as results/demo_clips.json: "
         f'a plain JSON list of clip ids, e.g. ["dia38_utt4", ...])')


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_pick_demo_clips.py -v`
Expected: 4 passed

- [ ] **Step 6: Run it for real and hand-curate the final list**

Run: `uv run python scripts/pick_demo_clips.py`
Expected: candidate counts printed, `results/demo_clip_candidates.json`
written. Open it, pick ~10 clips total covering: every emotion at least
once, at least one wrong prediction, the many-faces and zero-faces
candidates if they look like genuinely hard cases on inspection (use
`scripts/view_meld_clips.py --split test` to eyeball candidates), and save
the final list to `results/demo_clips.json` as a plain JSON array of clip
ids, e.g. `["dia38_utt4", "dia220_utt0", ...]`.

- [ ] **Step 7: Commit**

```bash
git add scripts/pick_demo_clips.py tests/test_pick_demo_clips.py
git commit -m "Add curated demo-clip candidate generation (stratified, hard-case, wrong-prediction)"
```

---

### Task 8: Replay demo

**Files:**
- Create: `scripts/replay_demo.py`
- Modify: `.gitignore` (already covers `/results/`; no change needed —
  noted here so the implementer doesn't add one)

**Interfaces:**
- Consumes: `load_inference_bundle` (Task 2); `load_prior` (Task 3);
  `build_clip_tokenizer`, `embed_prompt_bank`, `scene_gloss`, `face_gloss`
  (Task 4); `TurnProcessor` (Task 5); `Responder`, `build_prompt` (Task 6);
  `EventEmitter`, `LatencyStamps` (Task 1); `meld_emotion.data.preprocess
  .sample_frame_indices`; `meld_emotion.data.video_index.build_video_index`;
  `meld_emotion.config.split_video_dir`; `meld_emotion.data.manifest
  .read_manifest`.
- Produces: `run_one_clip(bundle, responder, prior, bank_embeddings,
  video_path: Path, row: dict, emitter: EventEmitter, *, show_window: bool
  = True) -> dict` (returns the final `done` event, so callers — including
  the consistency check — get the full result without parsing a stream);
  CLI `scripts/replay_demo.py --clips results/demo_clips.json`.

This task is the only one that touches `cv2.imshow`; it is a thin
orchestrator over everything built so far, styled after
`scripts/view_meld_clips.py`'s overlay (bottom text bar, face boxes) kept
deliberately minimal: video with face boxes + track IDs, a slim moving
provisional-expression bar strip, the transcript line arriving at clip
end, the final emotion/sentiment tags, the visual cues, and the response
text streaming in below the video — no extra chrome.

Because this script's correctness is end-to-end (it is exercised for real
by Task 9's consistency check and cannot be meaningfully unit-tested
against a mocked `cv2` window), it has no dedicated test file; `run_one_
clip` is validated by Task 9 running it against real curated clips and
checking its output against the batch-eval path.

- [ ] **Step 1: Implement**

Create `scripts/replay_demo.py`:

```python
#!/usr/bin/env python3
"""Replay demo (design doc §8.1): plays curated MELD test clips at real
speed, running the same inference core the live path will use, rendering
one minimal window (video + face boxes/track IDs + provisional bars +
transcript + final state + visual cues + streamed response).

Usage:
    uv run python scripts/replay_demo.py --clips results/demo_clips.json --model-repo mlx-community/Qwen2.5-3B-Instruct-4bit
"""
import argparse
import json
import time
from pathlib import Path

import cv2

from meld_emotion.config import FEATURE_CACHE_DIR, REPO_ROOT, split_video_dir
from meld_emotion.data.manifest import read_manifest
from meld_emotion.data.preprocess import sample_frame_indices
from meld_emotion.data.video_index import build_video_index
from meld_emotion.inference.events import EventEmitter, LatencyStamps
from meld_emotion.inference.gloss import build_clip_tokenizer, embed_prompt_bank, face_gloss, scene_gloss
from meld_emotion.inference.loader import load_inference_bundle
from meld_emotion.inference.prior import load_prior
from meld_emotion.inference.responder import Responder, build_prompt
from meld_emotion.inference.turn import TurnProcessor

WINDOW = "Replay demo  (space=pause, n=next, q=quit)"


def _draw_overlay(frame, tp: TurnProcessor, provisional_expression: dict) -> None:
    """Minimal overlay, styled after scripts/view_meld_clips.py: a face box
    + track-ID label per detected face (held from the last SAMPLED frame,
    per the design note that boxes are drawn every frame from the last
    sampled result, not re-detected every frame), plus a thin corner bar
    per emotion sized by its current provisional probability."""
    for (x, y, w, h, score), track_id in zip(tp.last_boxes, tp.last_track_ids):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"id{track_id}", (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    bar_x, bar_y, bar_w, row_h = 8, 8, 100, 14
    for i, (label, prob) in enumerate(provisional_expression.items()):
        y = bar_y + i * row_h
        cv2.rectangle(frame, (bar_x, y), (bar_x + int(bar_w * prob), y + row_h - 4), (200, 200, 0), -1)
        cv2.putText(frame, label[:3], (bar_x + bar_w + 4, y + row_h - 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def run_one_clip(bundle, responder, prior, bank_embeddings, clip_tokenizer, video_path: Path, row: dict,
                 emitter: EventEmitter, *, show_window: bool = True) -> tuple[dict, dict]:
    """Returns (final_event, done_event) -- the caller (and the consistency
    check) reads emotion_probs off final_event; done_event carries only the
    response text and latencies, per the event schema (Task 1)."""
    turn_id = f"dia{row['dialogue_id']}_utt{row['utterance_id']}"
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = set(sample_frame_indices(total_frames, fps))
    delay_ms = max(1, int(1000 / fps))

    stamps = LatencyStamps(turn_start=time.perf_counter())
    tp = TurnProcessor(bundle, prior, emitter)
    tp.start_turn(turn_id)

    frame_idx = -1
    last_provisional = {"neutral": 1.0}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx in wanted:
            last_provisional = tp.push_frame(frame)["provisional_expression"]
        if show_window:
            _draw_overlay(frame, tp, last_provisional)
            cv2.imshow(WINDOW, frame)
            if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                break
    cap.release()

    cues = [scene_gloss(tp.mean_scene_embedding, bank_embeddings)[0], face_gloss(tp.track_expressions)]
    context_prev = row["context_prev"]
    final = tp.end_turn(row["text"], context_prev, visual_cues=cues)
    stamps.state_emitted = time.perf_counter()

    top2_idx = sorted(final["emotion_probs"], key=final["emotion_probs"].get, reverse=True)[:2]
    top2 = [(e, final["emotion_probs"][e]) for e in top2_idx]
    messages = build_prompt(context_prev, row["text"], final["emotion"], top2, final["sentiment"], cues)

    response_text = ""
    for i, chunk in enumerate(responder.stream(messages)):
        if i == 0:
            stamps.first_token_emitted = time.perf_counter()
        response_text += chunk
        emitter.token(turn_id, chunk)
    stamps.done_emitted = time.perf_counter()
    done_event = {"turn_id": turn_id, "phase": "done", "response": response_text,
                 "latency_ms": stamps.as_ms()}
    emitter.emit(done_event)

    if show_window:
        # Hold the last frame with the final tag + streamed response overlaid
        # (design doc §8.1: "the final emotion + sentiment ... the streamed
        # response"), per the user's ask to show the emotion at the end of
        # the response text, until the next clip starts.
        end_frame = frame.copy()
        cv2.putText(end_frame, f"{response_text}  [{final['emotion']}]", (8, end_frame.shape[0] - 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(WINDOW, end_frame)
        cv2.waitKey(1500)
    return final, done_event


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", type=Path, default=REPO_ROOT / "results" / "demo_clips.json")
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "results" / "fusion" / "seed0" / "best.pt")
    parser.add_argument("--model-repo", required=True, help="from results/lm_selection.json's 'chosen' field")
    args = parser.parse_args()

    clip_ids = json.loads(args.clips.read_text())
    bundle = load_inference_bundle(args.checkpoint)
    prior = load_prior(REPO_ROOT / "src" / "meld_emotion" / "inference" / "face_prior.json")
    clip_tokenizer = build_clip_tokenizer()
    bank_embeddings = embed_prompt_bank(bundle.scene_encoder.model, clip_tokenizer, bundle.device)
    responder = Responder(model_repo=args.model_repo)

    rows_by_clip = {f"dia{r['dialogue_id']}_utt{r['utterance_id']}": r
                   for r in read_manifest(FEATURE_CACHE_DIR / "test" / "manifest.jsonl")}
    index = build_video_index(split_video_dir("test"))
    emitter = EventEmitter(open(REPO_ROOT / "results" / "replay_events.jsonl", "a"))

    for clip_id in clip_ids:
        row = rows_by_clip[clip_id]
        video_path = index[(row["dialogue_id"], row["utterance_id"])]
        print(f"playing {clip_id}: \"{row['text']}\" -> {row['emotion']}")
        final, done = run_one_clip(bundle, responder, prior, bank_embeddings, clip_tokenizer, video_path, row, emitter)
        print(f"  predicted={final['emotion']} response={done['response']!r} latency_ms={done['latency_ms']}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test on one real curated clip**

Run: `uv run python scripts/replay_demo.py --clips results/demo_clips.json --model-repo <chosen from results/lm_selection.json>`
Expected: a window opens playing the first curated clip at real speed with
face boxes drawn, then a printed final emotion/response line, and this
repeats per clip; `results/replay_events.jsonl` accumulates one line per
event across all four phases for every clip played.

- [ ] **Step 3: Commit**

```bash
git add scripts/replay_demo.py
git commit -m "Add the replay demo: curated clips through TurnProcessor + Responder, minimal overlay"
```

---

### Task 9: Consistency check

**Files:**
- Create: `scripts/consistency_check.py`
- Test: `tests/test_consistency_check.py`

**Interfaces:**
- Consumes: `run_one_clip` (Task 8, imported as a library function — the
  replay path); `meld_emotion.training.train.evaluate` and
  `meld_emotion.training.dataset.{MeldFeatureDataset, collate, load_ok_rows}`
  (the batch path, over the *same* checkpoint's cached features).
- Produces: `batch_predict_one(bundle, features_dir: Path, clip_id: str) ->
  dict` (`emotion_probs: dict[str, float]`, matching the `final` event's
  shape); `compare(batch_probs: dict, replay_probs: dict) -> dict`
  (`argmax_match: bool`, `max_abs_diff: float`); CLI that runs both paths
  over `results/demo_clips.json` and asserts every clip passes.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_consistency_check.py`:

```python
import pytest


def test_compare_reports_argmax_match_and_max_abs_diff():
    from scripts.consistency_check import compare
    a = {"neutral": 0.60, "joy": 0.20, "surprise": 0.10, "anger": 0.05, "sadness": 0.03, "disgust": 0.01, "fear": 0.01}
    b = {"neutral": 0.61, "joy": 0.19, "surprise": 0.10, "anger": 0.05, "sadness": 0.03, "disgust": 0.01, "fear": 0.01}
    result = compare(a, b)
    assert result["argmax_match"] is True
    assert result["max_abs_diff"] == pytest.approx(0.01, abs=1e-6)


def test_compare_detects_a_real_argmax_mismatch():
    from scripts.consistency_check import compare
    a = {"neutral": 0.51, "joy": 0.49, "surprise": 0.0, "anger": 0.0, "sadness": 0.0, "disgust": 0.0, "fear": 0.0}
    b = {"neutral": 0.40, "joy": 0.60, "surprise": 0.0, "anger": 0.0, "sadness": 0.0, "disgust": 0.0, "fear": 0.0}
    result = compare(a, b)
    assert result["argmax_match"] is False


def test_batch_predict_one_matches_a_direct_forward_pass(synthetic_features):
    import torch
    from conftest import StubTextEncoder, whitespace_encode
    from meld_emotion.data.labels import EMOTIONS
    from meld_emotion.inference.loader import InferenceBundle
    from meld_emotion.training.config import TrainConfig
    from meld_emotion.training.dataset import collate, load_ok_rows, MeldFeatureDataset
    from meld_emotion.training.model import FusionModel
    from scripts.consistency_check import batch_predict_one

    root, train_manifest, dev_manifest = synthetic_features
    config = TrainConfig(d_model=32, n_heads=4, ff_dim=64, n_layers=1)
    model = FusionModel(config, StubTextEncoder(32), face_dim=8, scene_dim=4).eval()

    class _StubTokenizer:
        pad_token_id = 1
    bundle = InferenceBundle(model=model, config=config, tokenizer=_StubTokenizer(),
                             face_detector=None, face_encoder=None, scene_encoder=None, device="cpu")
    rows = load_ok_rows(train_manifest)
    row = rows[5]
    clip_id = f"dia{row['dialogue_id']}_utt{row['utterance_id']}"

    result = batch_predict_one(bundle, root, clip_id, split="train", encode_fn=whitespace_encode)
    ds = MeldFeatureDataset([row], root, config, whitespace_encode)
    with torch.no_grad():
        out = model(collate([ds[0]], pad_id=1))
    expected = torch.softmax(out["emotion_logits"], dim=-1)[0]
    for i, e in enumerate(EMOTIONS):
        assert result["emotion_probs"][e] == pytest.approx(expected[i].item(), abs=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_consistency_check.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.consistency_check'`

- [ ] **Step 3: Implement**

Create `scripts/consistency_check.py`:

```python
#!/usr/bin/env python3
"""Batch-vs-replay consistency check (design doc §8.1): the replay path's
final prediction for a curated clip must match the batch-eval path's
prediction from the cached features, on the same checkpoint. Tiny numeric
drift is expected (the cache encoded JPEG-saved crops; replay encodes
in-memory crops), so this asserts argmax equality and a small probability
tolerance, not bitwise equality.

Usage:
    uv run python scripts/consistency_check.py --clips results/demo_clips.json --model-repo <chosen LM>
"""
import argparse
import functools
import json
from pathlib import Path

import torch

from meld_emotion.config import FEATURE_CACHE_DIR, REPO_ROOT, split_video_dir
from meld_emotion.data.labels import EMOTIONS
from meld_emotion.data.manifest import read_manifest
from meld_emotion.data.video_index import build_video_index
from meld_emotion.inference.events import EventEmitter, LatencyStamps
from meld_emotion.inference.gloss import build_clip_tokenizer, embed_prompt_bank
from meld_emotion.inference.loader import load_inference_bundle
from meld_emotion.inference.prior import load_prior
from meld_emotion.inference.responder import Responder
from meld_emotion.training.dataset import collate, load_ok_rows, MeldFeatureDataset
from meld_emotion.training.text import encode_text
from scripts.replay_demo import run_one_clip

MAX_ABS_DIFF = 0.02


def batch_predict_one(bundle, features_dir: Path, clip_id: str, split: str = "test", encode_fn=None) -> dict:
    rows = load_ok_rows(Path(features_dir) / split / "manifest.jsonl")
    row = next(r for r in rows if f"dia{r['dialogue_id']}_utt{r['utterance_id']}" == clip_id)
    if encode_fn is None:
        encode_fn = functools.partial(encode_text, bundle.tokenizer, max_length=bundle.config.max_text_tokens)
    dataset = MeldFeatureDataset([row], features_dir, bundle.config, encode_fn)
    with torch.no_grad():
        out = bundle.model(collate([dataset[0]], pad_id=bundle.tokenizer.pad_token_id))
    probs = torch.softmax(out["emotion_logits"], dim=-1)[0]
    return {"emotion_probs": {e: probs[i].item() for i, e in enumerate(EMOTIONS)}}


def compare(batch_probs: dict, replay_probs: dict) -> dict:
    batch_argmax = max(batch_probs, key=batch_probs.get)
    replay_argmax = max(replay_probs, key=replay_probs.get)
    max_abs_diff = max(abs(batch_probs[e] - replay_probs[e]) for e in batch_probs)
    return {"argmax_match": batch_argmax == replay_argmax, "max_abs_diff": max_abs_diff}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", type=Path, default=REPO_ROOT / "results" / "demo_clips.json")
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "results" / "fusion" / "seed0" / "best.pt")
    parser.add_argument("--model-repo", required=True)
    args = parser.parse_args()

    clip_ids = json.loads(args.clips.read_text())
    bundle = load_inference_bundle(args.checkpoint)
    prior = load_prior(REPO_ROOT / "src" / "meld_emotion" / "inference" / "face_prior.json")
    clip_tokenizer = build_clip_tokenizer()
    bank_embeddings = embed_prompt_bank(bundle.scene_encoder.model, clip_tokenizer, bundle.device)
    responder = Responder(model_repo=args.model_repo)

    rows_by_clip = {f"dia{r['dialogue_id']}_utt{r['utterance_id']}": r
                   for r in read_manifest(FEATURE_CACHE_DIR / "test" / "manifest.jsonl")}
    index = build_video_index(split_video_dir("test"))
    emitter = EventEmitter(open(REPO_ROOT / "results" / "consistency_events.jsonl", "w"))

    results = []
    for clip_id in clip_ids:
        row = rows_by_clip[clip_id]
        video_path = index[(row["dialogue_id"], row["utterance_id"])]
        batch = batch_predict_one(bundle, FEATURE_CACHE_DIR, clip_id, split="test")
        final_event, _done_event = run_one_clip(bundle, responder, prior, bank_embeddings, clip_tokenizer,
                                                video_path, row, emitter, show_window=False)
        result = compare(batch["emotion_probs"], final_event["emotion_probs"])
        results.append({"clip": clip_id, **result})
    passed = all(r["argmax_match"] and r["max_abs_diff"] < MAX_ABS_DIFF for r in results)
    Path(REPO_ROOT / "results" / "consistency_check.json").write_text(json.dumps(results, indent=2))
    print(f"{'PASS' if passed else 'FAIL'}: {sum(r['argmax_match'] for r in results)}/{len(results)} argmax matches")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_consistency_check.py -v`
Expected: 3 passed

- [ ] **Step 5: Run the real consistency check on the curated clips**

Run: `uv run python scripts/consistency_check.py --clips results/demo_clips.json --model-repo <chosen LM>`
Expected: `PASS: N/N argmax matches` (N = number of curated clips) and
`results/consistency_check.json` written. If any clip fails, debug before
proceeding — a genuine mismatch here means the replay path's frame
sampling, face detection, or track-ID handling has drifted from what
`preprocess_clip` did when building the cache, which would make the demo
inconsistent with the reported metrics.

- [ ] **Step 6: Commit**

```bash
git add scripts/consistency_check.py tests/test_consistency_check.py
git commit -m "Add batch-vs-replay consistency check on the curated clips"
```

---

### Task 10: Latency measurement, parameter budget, and response rubric

**Files:**
- Create: `scripts/measure_latency.py`, `scripts/param_budget.py`,
  `scripts/response_rubric.py`
- Test: `tests/test_measure_latency.py`, `tests/test_param_budget.py`,
  `tests/test_response_rubric.py`

**Interfaces:**
- Consumes: `run_one_clip` (Task 8); `results/replay_events.jsonl` (Task
  8's own output, already has every latency-relevant timestamp implicitly
  via the `done` event's `latency_ms`, but per-sampled-frame vision cost
  needs its own direct timing since it's not in the event schema);
  `results/lm_selection.json` (Task 6).
- Produces: `percentile(values: list[float], p: float) -> float`;
  `summarise_latencies(done_events: list[dict]) -> dict` (`state_p50/p95,
  first_token_p50/p95, done_p50/p95`, all in ms); `time_one_frame(bundle,
  frame_bgr) -> float` (seconds, one `TurnProcessor.push_frame`-equivalent
  vision cost, isolated from decode); `count_lm_params(model_repo: str) ->
  int` (from the HF config, no weights download); `build_budget_table(
  lm_repo: str, lm_params: int) -> str` (markdown, matches design doc
  §4.1's table); `sample_response_rows(replay_events_path: Path, n: int =
  50, seed: int = 0) -> list[dict]` (`turn_id, prompt, response,
  tag_consistent: None, uses_visual_cue: None, concise: None,
  in_character: None, no_invented_facts: None`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_measure_latency.py`:

```python
import pytest


def test_percentile_matches_known_values():
    from scripts.measure_latency import percentile
    values = list(range(1, 101))  # 1..100
    assert percentile(values, 50) == pytest.approx(50.5, abs=0.5)
    assert percentile(values, 95) == pytest.approx(95.5, abs=0.5)


def test_summarise_latencies_computes_p50_p95_per_event():
    from scripts.measure_latency import summarise_latencies
    done_events = [
        {"latency_ms": {"state": 80, "first_token": 600, "done": 1800}},
        {"latency_ms": {"state": 90, "first_token": 700, "done": 2000}},
        {"latency_ms": {"state": 100, "first_token": 800, "done": 2200}},
    ]
    summary = summarise_latencies(done_events)
    assert summary["state_p50"] == pytest.approx(90, abs=1)
    assert summary["done_p95"] >= summary["done_p50"] >= 1800
```

Create `tests/test_param_budget.py`:

```python
def test_count_lm_params_reads_the_hf_config_without_downloading_weights(monkeypatch):
    from scripts.param_budget import count_lm_params

    class _StubConfig:
        num_hidden_layers = 2
        hidden_size = 8
        intermediate_size = 16
        vocab_size = 100

    monkeypatch.setattr("scripts.param_budget.AutoConfig",
                        type("_A", (), {"from_pretrained": staticmethod(lambda repo: _StubConfig())}))
    n = count_lm_params("fake/repo")
    assert n > 0   # exact figure depends on the architecture-specific estimate; just must be positive and finite


def test_build_budget_table_includes_every_component_and_the_total():
    from scripts.param_budget import build_budget_table
    table = build_budget_table("mlx-community/Qwen2.5-3B-Instruct-4bit", 3_000_000_000)
    assert "RoBERTa" in table and "CLIP" in table and "Qwen2.5-3B-Instruct-4bit" in table
    assert "Total" in table
```

Create `tests/test_response_rubric.py`:

```python
import json


def test_sample_response_rows_reads_done_events_and_adds_empty_rubric_fields(tmp_path):
    from scripts.response_rubric import sample_response_rows
    path = tmp_path / "events.jsonl"
    lines = [json.dumps({"turn_id": f"dia{i}_utt0", "phase": "done", "response": f"response {i}"})
            for i in range(5)]
    path.write_text("\n".join(lines) + "\n")
    rows = sample_response_rows(path, n=3, seed=0)
    assert len(rows) == 3
    for row in rows:
        assert set(row) == {"turn_id", "prompt", "response",
                            "tag_consistent", "uses_visual_cue", "concise", "in_character", "no_invented_facts"}
        assert all(row[f] is None for f in
                  ("tag_consistent", "uses_visual_cue", "concise", "in_character", "no_invented_facts"))


def test_sample_response_rows_is_capped_at_the_available_count(tmp_path):
    from scripts.response_rubric import sample_response_rows
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps({"turn_id": "dia0_utt0", "phase": "done", "response": "hi"}) + "\n")
    assert len(sample_response_rows(path, n=50, seed=0)) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_measure_latency.py tests/test_param_budget.py tests/test_response_rubric.py -v`
Expected: FAIL, each with `ModuleNotFoundError` for its own module.

- [ ] **Step 3: Implement**

Create `scripts/measure_latency.py`:

```python
#!/usr/bin/env python3
"""Measured (not estimated) latency: p50/p95 per end-of-turn event from a
completed replay run's event log, plus the per-sampled-frame vision cost
in isolation (design doc §9). Run scripts/replay_demo.py first to produce
results/replay_events.jsonl.

Usage:
    uv run python scripts/measure_latency.py
"""
import json
import time
from pathlib import Path

import numpy as np

from meld_emotion.config import REPO_ROOT


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p))


def summarise_latencies(done_events: list[dict]) -> dict:
    out = {}
    for key in ("state", "first_token", "done"):
        values = [e["latency_ms"][key] for e in done_events if e["latency_ms"].get(key) is not None]
        out[f"{key}_p50"] = percentile(values, 50) if values else None
        out[f"{key}_p95"] = percentile(values, 95) if values else None
    return out


def time_one_frame(bundle, frame_bgr) -> float:
    """Isolated cost of one push_frame-equivalent vision pass (detect + track
    + encode faces + encode scene), excluding video decode. Design doc §9:
    must stay under the 333ms (~3fps) frame interval."""
    from PIL import Image

    from meld_emotion.data.preprocess import crop_face, letterbox
    from meld_emotion.vision.face_detector import detect_faces

    start = time.perf_counter()
    boxes = detect_faces(bundle.face_detector, frame_bgr)
    crops = [Image.fromarray(crop[:, :, ::-1]) for (x, y, w, h, s) in boxes
            if (crop := crop_face(frame_bgr, (x, y, w, h))) is not None]
    bundle.face_encoder.encode_batch(crops)
    scene_img = Image.fromarray(letterbox(frame_bgr)[:, :, ::-1])
    bundle.scene_encoder.encode(scene_img)
    return time.perf_counter() - start


def main():
    events_path = REPO_ROOT / "results" / "replay_events.jsonl"
    done_events = [json.loads(l) for l in events_path.read_text().splitlines() if l.strip()]
    done_events = [e for e in done_events if e.get("phase") == "done"]
    summary = summarise_latencies(done_events)
    for key, value in summary.items():
        print(f"{key}: {value}")
    (REPO_ROOT / "results" / "latency.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

Create `scripts/param_budget.py`:

```python
#!/usr/bin/env python3
"""Final parameter-count table with the chosen response LM (design doc
§4.1). Everything before the LM is already known and measured (~375M,
Stage 1 plan's Global Constraints); this script adds the LM's own count
from its HF config (no weight download needed) and renders the table.

Usage:
    uv run python scripts/param_budget.py --model-repo <chosen from results/lm_selection.json>
"""
import argparse
import json
from pathlib import Path

from transformers import AutoConfig

from meld_emotion.config import REPO_ROOT

PRE_LM_PARAMS = 375_000_000  # RoBERTa + face ViT + CLIP (both towers) + fusion transformer + projectors + heads


def count_lm_params(model_repo: str) -> int:
    """A parameter estimate from the HF config alone (no weights needed):
    embeddings + per-layer attention/FFN, the same style of estimate used
    throughout this project's plans (e.g. the Stage 1 plan's RoBERTa math)."""
    cfg = AutoConfig.from_pretrained(model_repo)
    hidden = cfg.hidden_size
    layers = cfg.num_hidden_layers
    inter = getattr(cfg, "intermediate_size", 4 * hidden)
    vocab = cfg.vocab_size
    embeddings = vocab * hidden
    per_layer = 4 * hidden * hidden + 2 * hidden * inter  # attention proj + FFN, biases/norms negligible
    return int(embeddings + layers * per_layer)


def build_budget_table(lm_repo: str, lm_params: int) -> str:
    rows = [
        ("RoBERTa-base (text)", "125M", "Fine-tuned, top half"),
        ("Face/expression encoder (ViT-Base)", "86M", "Frozen"),
        ("CLIP ViT-B/32 (image + text towers)", "151M", "Frozen"),
        ("Fusion transformer + projectors + heads", "~12M", "From scratch"),
        ("Face detector (YuNet)", "75K", "Zero-shot"),
        (f"Response LM ({lm_repo})", f"{lm_params / 1e9:.2f}B", "Prompted, not fine-tuned"),
    ]
    total = PRE_LM_PARAMS + lm_params
    lines = ["| Component | Params | Trained? |", "|---|---|---|"]
    lines += [f"| {name} | {params} | {trained} |" for name, params, trained in rows]
    lines.append(f"| **Total** | **{total / 1e9:.2f}B** | |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-repo", required=True)
    args = parser.parse_args()
    lm_params = count_lm_params(args.model_repo)
    table = build_budget_table(args.model_repo, lm_params)
    print(table)
    (REPO_ROOT / "results" / "param_budget.json").write_text(
        json.dumps({"model_repo": args.model_repo, "lm_params": lm_params,
                   "pre_lm_params": PRE_LM_PARAMS, "total": PRE_LM_PARAMS + lm_params}, indent=2))
    (REPO_ROOT / "docs" / "results").mkdir(parents=True, exist_ok=True)
    Path(REPO_ROOT / "docs" / "results" / "param_budget.md").write_text(table + "\n")


if __name__ == "__main__":
    main()
```

Create `scripts/response_rubric.py`:

```python
#!/usr/bin/env python3
"""Response quality rubric (design doc §4.4): 50 sampled prompt/response
pairs from a completed replay run, with empty fields for hand scoring.
This file IS a plan deliverable, not just a script side-effect.

Usage:
    uv run python scripts/response_rubric.py
"""
import json
import random
from pathlib import Path

from meld_emotion.config import REPO_ROOT

RUBRIC_FIELDS = ("tag_consistent", "uses_visual_cue", "concise", "in_character", "no_invented_facts")


def sample_response_rows(events_path: Path, n: int = 50, seed: int = 0) -> list[dict]:
    events = [json.loads(l) for l in Path(events_path).read_text().splitlines() if l.strip()]
    done = [e for e in events if e.get("phase") == "done"]
    rng = random.Random(seed)
    sampled = rng.sample(done, min(n, len(done)))
    return [{"turn_id": e["turn_id"], "prompt": e.get("prompt", ""), "response": e["response"],
            **{f: None for f in RUBRIC_FIELDS}} for e in sampled]


def main():
    events_path = REPO_ROOT / "results" / "replay_events.jsonl"
    rows = sample_response_rows(events_path)
    out_path = REPO_ROOT / "results" / "response_rubric.jsonl"
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_measure_latency.py tests/test_param_budget.py tests/test_response_rubric.py -v`
Expected: 6 passed

- [ ] **Step 5: Run the full offline suite**

Run: `uv run pytest -v`
Expected: every test from the data plan, the Stage 1 plan, and this plan
passes; the `network` tests (Task 6's real `mlx_lm` load/stream test)
deselected by default.

- [ ] **Step 6: Run the three reporting scripts for real**

```bash
uv run python scripts/measure_latency.py            # after Task 8/9 have produced results/replay_events.jsonl
uv run python scripts/param_budget.py --model-repo <chosen from results/lm_selection.json>
uv run python scripts/response_rubric.py
```
Expected: `results/latency.json`, `results/param_budget.json`,
`docs/results/param_budget.md`, and `results/response_rubric.jsonl` all
written with real numbers; hand-score the rubric file afterward (fill in
the five `null` fields per row) — that scored file, not the empty one, is
what goes in the write-up.

- [ ] **Step 7: Commit**

```bash
git add scripts/measure_latency.py scripts/param_budget.py scripts/response_rubric.py \
       tests/test_measure_latency.py tests/test_param_budget.py tests/test_response_rubric.py
git commit -m "Add latency measurement, parameter-budget table, and response rubric generation"
```

---

## Running this plan for real

Prerequisite: `results/fusion/seed0/best.pt` exists (it does — dev
wF1=0.621, beating text_only_k4's 0.610 and text_only_k0's 0.587, per
`docs/results/stage1_ablations.md`).

```bash
# 1. Once: the face-head prior (Task 3, Step 5) and the LM choice (Task 6, Step 6)
uv run python -c "..."   # see Task 3 Step 5
uv run --with mlx-lm python scripts/select_lm.py

# 2. Curated clips (Task 7)
uv run python scripts/pick_demo_clips.py
# -> hand-pick ~10 into results/demo_clips.json

# 3. The demo itself (Task 8)
uv run python scripts/replay_demo.py --clips results/demo_clips.json --model-repo <chosen>

# 4. Verification and reporting (Tasks 9, 10)
uv run python scripts/consistency_check.py --clips results/demo_clips.json --model-repo <chosen>
uv run python scripts/measure_latency.py
uv run python scripts/param_budget.py --model-repo <chosen>
uv run python scripts/response_rubric.py
# -> hand-score results/response_rubric.jsonl
```

**If the LM misses the latency target:** the design doc's own fallback is
the smallest candidate (`Qwen2.5-1.5B-Instruct-4bit`); `select_lm.py`
already tries it last. There is no code change needed to use it — just
pass its repo id as `--model-repo` everywhere above.

## Self-Review

**Spec coverage.** §4.4 prompt/gloss/serving/evaluation → Tasks 4, 6, 10
(rubric); §4.5 event schema (with the `final`-carries-no-response
resolution) → Task 1, consumed everywhere; §8.1 replay window, consistency
check → Tasks 8, 9; §9 latency/vision-cost/parameter-count reporting →
Task 10; §2's three separate latency targets and "state never gated by the
LM" → Task 5 (`final` has no `response`) and Task 8 (LM only starts after
`end_turn` returns); §4.1's LM selection rule → Task 6. The live webcam
path, Stage 2, and the tri-modal/RL extension are explicitly out of scope
per the briefing and design doc build-priority ordering.

**Placeholder scan.** No TBD/TODO markers, no "add appropriate handling"
prose, no code block that describes behavior without showing it. `end_turn`
takes `visual_cues` as a parameter (computed by the caller from properties
that are already populated once every frame is pushed) rather than
patching the returned dict after emission, specifically so the `final`
event is built and emitted exactly once, correctly, with no post-hoc
mutation gap; `run_one_clip` returns `(final_event, done_event)` so Task
9's consistency check reads `emotion_probs` from the right place without
re-deriving it. Every code block is complete and runnable as written.

**Type consistency.** `InferenceBundle` (Task 2) fields (`model, config,
tokenizer, face_detector, face_encoder, scene_encoder, device`) are read
identically by Tasks 5, 7, 8, 9, 10. `TurnProcessor.push_frame`/`.end_turn`
(Task 5) emit exactly the `provisional`/`final` shapes Task 1 defines, and
`.last_boxes`/`.last_track_ids` (demo-only) are consumed only by Task 8's
overlay. `Responder.stream` (Task 6) yields plain `str` deltas, consumed
identically by Task 8's token-emission loop. `correct_provisional`'s output
dict keys (Task 3, `EMOTIONS` order) match what Task 1's `.provisional()`
and Task 5's event both expect. `gloss.scene_gloss`/`face_gloss` (Task 4)
are called with `TurnProcessor.mean_scene_embedding`/`.track_expressions`
(Task 5) with matching shapes/types, and their output feeds `end_turn`'s
new `visual_cues` parameter directly. `batch_predict_one`'s (Task 9)
`emotion_probs` shape matches `predict_test_split`'s per-row shape (Task 7)
and `turn.py`'s `final["emotion_probs"]` (Task 5) — all three are
`dict[str, float]` keyed by `EMOTIONS`, which is exactly what Task 9's
`compare()` consumes from both the batch and replay paths.
