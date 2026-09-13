# Response LM + Replay Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the trained Stage 2 checkpoint into a real-time inference
core that emits the design doc's four-phase event stream, add a prompted,
streamed response LM with a visual gloss, and ship the replay demo on ~10
curated MELD test clips — with a batch/replay consistency check, measured
per-event latency, the final parameter-budget table, and a hand-scorable
response rubric. This is design doc build-priority item 3.

**Explicitly out of scope:** the live webcam/microphone/VAD/Whisper path
(design doc build-priority item 4 — a separate, later plan), any
retraining, and the tri-modal/RL extension. Stage 2 is done; its winning
checkpoint is this plan's input.

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
  already uses for images). (b) The fusion model's own vision-only reading of the turn (Task 3
  with text masked): "N faces visible, reading <top-1> or <top-2>". Not
  the face head's per-face labels — that head is prior-skewed on MELD and
  stale after Stage 2 (see next bullet).
- **Never use the face head's probabilities** (`FaceEmotionEncoder
  .encode_batch()["probs"]`). Measured fact: on MELD dev the frozen head's
  raw argmax scores wF1 0.12 — below always-neutral — because it almost
  never says "neutral" on talking faces; and after Stage 2 the head is
  stale (its CLS input changed). The `provisional` event is the **fusion
  model with text masked** (`force_drop_text=True`) over the face tokens
  accumulated so far — what modality dropout trained it to do, and 0.30+
  wF1 on dev vs the head's 0.12. Task 3 is the single code path for it.
- **The submitted model is a Stage 2 checkpoint with `use_scene=False`
  and `use_track_id=False`** (`results/stage2_fusion_faces_only/seed1/
  best.pt`: dev wF1 0.634, test 0.642; `docs/results/stage1_ablations.md`).
  Consequences: the classifier sees text + face tokens only (CLIP is kept
  *solely* for the scene gloss; the tracker only for drawing boxes); the
  face features must come from the **fine-tuned** ViT, i.e.
  `FaceEmotionEncoder(weights_path=<that best.pt>)`, which the loader does
  automatically; and the batch path capped crops at 64 per clip
  (`training.crops.subsample_faces`), which replay must mirror
  (`config.face_trainable_layers > 0` is the switch — Task 3).
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
- **Parameter budget** (design doc §4.1): ceiling 6B. Task 10 counts the
  loaded bundle's components directly (RoBERTa + fusion ≈ 136M, face ViT
  86M, CLIP both towers 151M, YuNet 75K ≈ 373M) and adds the chosen LM
  from its HF config; `results/param_budget.json` records the measured
  total. `mlx-community/Qwen3-4B-4bit` is already in this machine's HF
  cache (2.1 GB) — the top candidate needs no download.
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
- **Inputs this plan consumes, exactly as the training plans write them:**
  `results/stage2_fusion_faces_only/seed1/best.pt` — a `torch.save` dict
  with keys `model_state` (the `FusionModel` state), `face_encoder_state`
  (the whole fine-tuned ViT + classifier state, loadable into
  `FaceEmotionEncoder.model`), `config` (a `TrainConfig` field dict),
  `epoch, dev_weighted_f1, face_dim (768), scene_dim (512), stage (2),
  base_checkpoint`; `results/stage2_fusion_faces_only/seed1/
  test_predictions.jsonl` — one `{"clip", "pred", "probs"}` per ok test row
  (`probs` in `EMOTIONS` order), the batch path's own output;
  `data/meld/features/test/manifest.jsonl` rows (keys `dialogue_id,
  utterance_id, speaker, text, context_prev, emotion, sentiment, status,
  feature_path, duration_s, n_frames, n_faces, n_shot_cuts`); real MELD test
  videos under `meld_emotion.config.split_video_dir("test")`, indexed with
  `meld_emotion.data.video_index.build_video_index` (never a global glob —
  IDs collide across splits). The cached `.npz` features are **not** an
  input: they were encoded by the frozen ViT and are stale for this model.

---

## File Structure

```
src/meld_emotion/inference/
  __init__.py
  events.py       # Event dataclasses-as-dicts + JSON-lines emitter (Task 1)
  loader.py       # checkpoint -> InferenceBundle (model, tokenizer, vision encoders) (Task 2)
  predict.py      # model-input builder + predict(); provisional = text masked (Task 3)
  gloss.py        # CLIP scene-cue bank + vision-only face reading (Task 4)
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
  test_events.py, test_loader.py, test_predict.py, test_gloss.py, test_turn.py,
  test_responder.py, test_pick_demo_clips.py, test_consistency_check.py,
  test_measure_latency.py, test_param_budget.py, test_response_rubric.py
  conftest.py           # extends the training-plan conftest: synthetic Stage 1 and
                        # Stage 2 checkpoint fixtures (Task 2)
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
- Modify: `tests/conftest.py` (append a synthetic-checkpoint fixture)
- Test: `tests/test_loader.py`

**Interfaces:**
- Consumes: `meld_emotion.training.config.TrainConfig`;
  `meld_emotion.training.model.FusionModel`; `meld_emotion.training.text
  .build_text_encoder`, `.build_tokenizer`; `meld_emotion.training.train
  .resolve_device`; `meld_emotion.vision.encoders.FaceEmotionEncoder`
  (with its `weights_path=` argument, added by the Stage 2 plan),
  `.SceneEncoder`; `meld_emotion.vision.face_detector.build_face_detector`.
- Produces: `InferenceBundle` (dataclass: `model: FusionModel`, `config:
  TrainConfig`, `tokenizer`, `face_detector`, `face_encoder`,
  `scene_encoder`, `device: str`, `face_dim: int`, `scene_dim: int`,
  `checkpoint_path: Path`, `stage: int`); `load_inference_bundle(
  checkpoint_path: Path, device: str = "auto", *, text_encoder_factory=None,
  tokenizer_factory=None, face_encoder_factory=None,
  scene_encoder_factory=None, face_detector_factory=None) ->
  InferenceBundle`. The factories are injection points (each `None` builds
  the real component); tests inject stubs so no network/weights are needed
  offline. When the checkpoint carries `face_encoder_state` (a Stage 2
  checkpoint), the default face-encoder factory is
  `FaceEmotionEncoder(device=device, weights_path=checkpoint_path)` so the
  demo runs the *fine-tuned* ViT; a Stage 1 checkpoint gets the pretrained
  one.

- [ ] **Step 1: Add the synthetic-checkpoint fixture**

Append to `tests/conftest.py`:

(The file exists from the training plans — add these at the end, do not
replace anything.)

```python
import torch as _torch
from dataclasses import asdict as _asdict


def write_synthetic_checkpoint(path, d_model=32, n_heads=4, ff_dim=64, n_layers=1,
                               face_dim=8, scene_dim=4, use_text=True, stage2=False):
    """A checkpoint in exactly train.py's save format (train_stage2.py's when
    stage2=True: adds face_encoder_state / stage / base_checkpoint), small
    enough to build and load in a unit test."""
    from meld_emotion.training.config import TrainConfig
    from meld_emotion.training.model import FusionModel
    config = TrainConfig(d_model=d_model, n_heads=n_heads, ff_dim=ff_dim, n_layers=n_layers,
                         use_text=use_text, use_scene=not stage2, use_track_id=not stage2,
                         face_trainable_layers=4 if stage2 else 0)
    text_encoder = StubTextEncoder(d_model) if use_text else None
    model = FusionModel(config, text_encoder, face_dim=face_dim, scene_dim=scene_dim)
    ckpt = {"model_state": model.state_dict(), "config": _asdict(config), "epoch": 1,
            "dev_weighted_f1": 0.5, "face_dim": face_dim, "scene_dim": scene_dim}
    if stage2:
        ckpt.update({"face_encoder_state": {"dummy.weight": _torch.zeros(1)}, "stage": 2,
                     "base_checkpoint": "results/fusion_faces_only/seed0/best.pt"})
    _torch.save(ckpt, path)
    return config


@pytest.fixture
def synthetic_checkpoint(tmp_path):
    path = tmp_path / "best.pt"
    config = write_synthetic_checkpoint(path)
    return path, config


@pytest.fixture
def synthetic_stage2_checkpoint(tmp_path):
    path = tmp_path / "stage2.pt"
    config = write_synthetic_checkpoint(path, stage2=True)
    return path, config
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_loader.py`:

```python
import numpy as np
import torch

from conftest import StubTextEncoder


class _StubFaceDetector:
    pass


class _StubFaceEncoder:
    feature_dim = 8
    meld_labels = ("neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear")

    def encode_batch(self, images):
        return {"features": np.zeros((len(images), 8), dtype=np.float32),
                "probs": np.zeros((len(images), 7), dtype=np.float32)}


class _StubSceneEncoder:
    feature_dim = 4

    def encode(self, image):
        return np.zeros(4, dtype=np.float32)


class _StubTokenizer:
    pad_token_id = 1
    cls_token_id = 0


def _stub_factories():
    return dict(text_encoder_factory=lambda: StubTextEncoder(32),
                tokenizer_factory=lambda: _StubTokenizer(),
                face_encoder_factory=lambda: _StubFaceEncoder(),
                scene_encoder_factory=lambda: _StubSceneEncoder(),
                face_detector_factory=lambda: _StubFaceDetector())


def test_loads_a_matching_model_and_reproduces_the_checkpoints_own_state(synthetic_checkpoint):
    from meld_emotion.inference.loader import load_inference_bundle
    path, config = synthetic_checkpoint
    bundle = load_inference_bundle(path, device="cpu", **_stub_factories())
    assert bundle.device == "cpu" and bundle.stage == 1
    assert (bundle.face_dim, bundle.scene_dim, bundle.checkpoint_path) == (8, 4, path)
    assert bundle.config.d_model == config.d_model and bundle.config.n_layers == config.n_layers
    assert bundle.tokenizer.pad_token_id == 1
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for key, value in bundle.model.state_dict().items():
        assert torch.equal(value, ckpt["model_state"][key])


def test_vision_only_checkpoint_skips_the_text_encoder(tmp_path):
    from conftest import write_synthetic_checkpoint
    from meld_emotion.inference.loader import load_inference_bundle
    path = tmp_path / "vision_only.pt"
    write_synthetic_checkpoint(path, use_text=False)
    factories = _stub_factories()
    factories["text_encoder_factory"] = lambda: (_ for _ in ()).throw(AssertionError("must not be called"))
    factories["tokenizer_factory"] = lambda: (_ for _ in ()).throw(AssertionError("must not be called"))
    bundle = load_inference_bundle(path, device="cpu", **factories)
    assert bundle.tokenizer is None and bundle.model.text_encoder is None


def test_resolve_device_auto_is_used_when_device_not_given(synthetic_checkpoint, monkeypatch):
    from meld_emotion.inference import loader as loader_module
    from meld_emotion.inference.loader import load_inference_bundle
    path, _ = synthetic_checkpoint
    monkeypatch.setattr(loader_module, "resolve_device", lambda name: "cpu" if name == "auto" else name)
    assert load_inference_bundle(path, **_stub_factories()).device == "cpu"


def test_stage2_checkpoint_loads_the_fine_tuned_face_weights_by_default(synthetic_checkpoint,
                                                                         synthetic_stage2_checkpoint, monkeypatch):
    from meld_emotion.inference import loader as loader_module
    from meld_emotion.inference.loader import load_inference_bundle
    seen = []

    class _RecordingFaceEncoder(_StubFaceEncoder):
        def __init__(self, device=None, weights_path=None):
            seen.append(weights_path)

    monkeypatch.setattr(loader_module, "FaceEmotionEncoder", _RecordingFaceEncoder)
    factories = _stub_factories()
    del factories["face_encoder_factory"]                 # use the default (real) factory path
    s2_path, s2_config = synthetic_stage2_checkpoint
    bundle = load_inference_bundle(s2_path, device="cpu", **factories)
    assert bundle.stage == 2 and seen[-1] == s2_path
    assert not bundle.config.use_scene and not bundle.config.use_track_id
    s1_path, _ = synthetic_checkpoint
    load_inference_bundle(s1_path, device="cpu", **factories)
    assert seen[-1] is None                                # Stage 1 checkpoint: pretrained ViT, no override
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meld_emotion.inference.loader'`

- [ ] **Step 4: Implement**

Create `src/meld_emotion/inference/loader.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_loader.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/meld_emotion/inference/loader.py tests/conftest.py tests/test_loader.py
git commit -m "Add checkpoint loader producing an inference bundle; Stage 2 checkpoints load the fine-tuned face ViT"
```

---

### Task 3: Model input builder and predictor

The one place that turns a turn's accumulated tokens into the exact batch
the training pipeline produced, and runs the model on it. Both
`push_frame` (provisional state: same batch, **text masked**) and
`end_turn` (final state) go through it, so the provisional state is the
fusion model's own vision-only prediction — not the face head's softmax,
which is prior-skewed on MELD and stale after Stage 2.

**Files:**
- Create: `src/meld_emotion/inference/predict.py`
- Test: `tests/test_predict.py`

**Interfaces:**
- Consumes: `meld_emotion.training.dataset.{collate, remap_track_ids}`;
  `meld_emotion.training.crops.subsample_faces`; `meld_emotion.data.labels
  .{EMOTIONS, SENTIMENTS}`; `TrainConfig`.
- Produces: `dummy_text_encoding(tokenizer) -> dict` (`{"input_ids":
  [cls_id], "attention_mask": [1]}` — one real token so the text encoder
  runs on something well-formed; the fusion mask then drops it);
  `build_item(config, enc: dict, face_feat: np.ndarray, face_frame_idx:
  list[int], face_track_raw: list[int], scene_feat: np.ndarray, *,
  face_dim: int, scene_dim: int, clip: str) -> dict` (a Stage 1 dataset
  item: frame clamp, track remap, empty `(0, face_dim)`/`(0, scene_dim)`
  when a modality has no tokens, modality switched off per `config`, and
  the Stage 2 per-clip face cap applied iff `config.face_trainable_layers >
  0`); `predict(model, item, pad_id: int, device: str, force_drop_text:
  bool = False) -> tuple[dict, dict]` (`emotion_probs`, `sentiment_probs`,
  both keyed in `EMOTIONS`/`SENTIMENTS` order, floats summing to 1).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_predict.py`:

```python
import numpy as np
import pytest
import torch

from conftest import StubTextEncoder
from meld_emotion.data.labels import EMOTIONS, SENTIMENTS


def _model(config):
    from meld_emotion.training.model import FusionModel
    return FusionModel(config, StubTextEncoder(32) if config.use_text else None, face_dim=8, scene_dim=4).eval()


def _cfg(**kw):
    from meld_emotion.training.config import TrainConfig
    return TrainConfig(**{**dict(d_model=32, n_heads=4, ff_dim=64, n_layers=1), **kw})


ENC = {"input_ids": [0, 5, 6, 1], "attention_mask": [1, 1, 1, 1]}


def test_build_item_clamps_frames_remaps_tracks_and_keeps_dims_from_the_checkpoint():
    from meld_emotion.inference.predict import build_item
    item = build_item(_cfg(max_frames=32, max_track_slots=16), ENC,
                      face_feat=np.ones((3, 8), np.float32), face_frame_idx=[0, 40, 44],
                      face_track_raw=[7, 55, 7], scene_feat=np.ones((2, 4), np.float32),
                      face_dim=8, scene_dim=4, clip="dia1_utt1")
    assert item["face_frame"].tolist() == [0, 31, 31] and item["face_track"].tolist() == [0, 1, 0]
    assert item["scene_frame"].tolist() == [0, 1] and item["clip"] == "dia1_utt1"
    assert item["input_ids"].tolist() == ENC["input_ids"] and item["emotion"].item() == 0


def test_build_item_uses_checkpoint_dims_for_empty_modalities():
    from meld_emotion.inference.predict import build_item
    item = build_item(_cfg(), ENC, face_feat=np.zeros((0, 8), np.float32), face_frame_idx=[], face_track_raw=[],
                      scene_feat=np.zeros((0, 4), np.float32), face_dim=768, scene_dim=512, clip="x")
    assert item["face_feat"].shape == (0, 768) and item["scene_feat"].shape == (0, 512)


def test_build_item_drops_modalities_the_config_switched_off():
    from meld_emotion.inference.predict import build_item
    item = build_item(_cfg(use_scene=False), ENC, face_feat=np.ones((2, 8), np.float32), face_frame_idx=[0, 1],
                      face_track_raw=[0, 0], scene_feat=np.ones((2, 4), np.float32), face_dim=8, scene_dim=4, clip="x")
    assert item["scene_feat"].shape == (0, 4) and item["face_feat"].shape == (2, 8)


def test_build_item_applies_the_stage2_face_cap_only_for_stage2_configs():
    from meld_emotion.inference.predict import build_item
    feats = np.arange(100, dtype=np.float32)[:, None].repeat(8, 1)
    kw = dict(face_frame_idx=list(range(100)), face_track_raw=[0] * 100, scene_feat=np.zeros((0, 4), np.float32),
              face_dim=8, scene_dim=4, clip="x")
    stage2 = build_item(_cfg(face_trainable_layers=4, max_faces_per_clip=64), ENC, feats, **kw)
    stage1 = build_item(_cfg(face_trainable_layers=0, max_faces_per_clip=64), ENC, feats, **kw)
    assert stage2["face_feat"].shape[0] == 64 == len(stage2["face_frame"]) == len(stage2["face_track"])
    assert stage2["face_feat"][0, 0] == 0 and stage2["face_feat"][-1, 0] == 99      # uniform, endpoints kept
    assert stage1["face_feat"].shape[0] == 100


def test_predict_returns_normalised_distributions_and_text_masking_changes_them():
    from meld_emotion.inference.predict import build_item, dummy_text_encoding, predict
    cfg = _cfg()
    model = _model(cfg)
    item = build_item(cfg, ENC, np.random.default_rng(0).standard_normal((3, 8)).astype(np.float32), [0, 1, 2],
                      [0, 0, 0], np.random.default_rng(1).standard_normal((2, 4)).astype(np.float32),
                      face_dim=8, scene_dim=4, clip="x")
    emo, sent = predict(model, item, pad_id=1, device="cpu")
    assert list(emo) == list(EMOTIONS) and list(sent) == list(SENTIMENTS)
    assert sum(emo.values()) == pytest.approx(1.0, abs=1e-5) and sum(sent.values()) == pytest.approx(1.0, abs=1e-5)
    masked, _ = predict(model, item, pad_id=1, device="cpu", force_drop_text=True)
    assert masked != emo

    class _Tok:
        cls_token_id = 0
    assert dummy_text_encoding(_Tok()) == {"input_ids": [0], "attention_mask": [1]}
    dummy_item = build_item(cfg, dummy_text_encoding(_Tok()), item["face_feat"].numpy(), [0, 1, 2], [0, 0, 0],
                            item["scene_feat"].numpy(), face_dim=8, scene_dim=4, clip="x")
    provisional, _ = predict(model, dummy_item, pad_id=1, device="cpu", force_drop_text=True)
    assert provisional == pytest.approx(masked, abs=1e-6)   # text masked -> the text content is irrelevant


def test_predict_is_finite_with_no_visual_tokens_at_all():
    from meld_emotion.inference.predict import build_item, predict
    cfg = _cfg()
    item = build_item(cfg, ENC, np.zeros((0, 8), np.float32), [], [], np.zeros((0, 4), np.float32),
                      face_dim=8, scene_dim=4, clip="x")
    emo, _ = predict(_model(cfg), item, pad_id=1, device="cpu")
    assert all(np.isfinite(v) for v in emo.values())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_predict.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meld_emotion.inference.predict'`

- [ ] **Step 3: Implement**

Create `src/meld_emotion/inference/predict.py`:

```python
"""Turn a turn's accumulated tokens into the exact batch the training
pipeline built, and run the fusion model on it (design doc §4.3).

Used twice per turn: with `force_drop_text=True` after every sampled frame
(the provisional, vision-only state -- the fusion model's own reading of
the faces so far, which is what modality dropout trained it to do) and
with the real text at end-of-turn (the final state).
"""
import numpy as np
import torch

from meld_emotion.data.labels import EMOTIONS, SENTIMENTS
from meld_emotion.training.config import TrainConfig
from meld_emotion.training.crops import subsample_faces
from meld_emotion.training.dataset import collate, remap_track_ids


def dummy_text_encoding(tokenizer) -> dict:
    cls_id = getattr(tokenizer, "cls_token_id", None)
    return {"input_ids": [0 if cls_id is None else int(cls_id)], "attention_mask": [1]}


def build_item(config: TrainConfig, enc: dict, face_feat: np.ndarray, face_frame_idx: list[int],
               face_track_raw: list[int], scene_feat: np.ndarray, *, face_dim: int, scene_dim: int,
               clip: str) -> dict:
    face_feat = np.asarray(face_feat, dtype=np.float32).reshape(-1, face_dim)
    face_frame = np.asarray(face_frame_idx, dtype=np.int64)
    face_track = np.asarray(face_track_raw, dtype=np.int64)
    scene_feat = np.asarray(scene_feat, dtype=np.float32).reshape(-1, scene_dim)

    if config.face_trainable_layers > 0:
        # The Stage 2 batch path capped crops per clip (crops.py); replay must match it.
        keep = subsample_faces(len(face_feat), config.max_faces_per_clip)
        face_feat, face_frame, face_track = face_feat[keep], face_frame[keep], face_track[keep]
    if not config.use_faces:
        face_feat, face_frame, face_track = face_feat[:0], face_frame[:0], face_track[:0]
    if not config.use_scene:
        scene_feat = scene_feat[:0]

    return {
        "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
        "face_feat": torch.from_numpy(np.ascontiguousarray(face_feat)),
        "face_frame": torch.from_numpy(np.minimum(face_frame, config.max_frames - 1)),
        "face_track": torch.from_numpy(remap_track_ids(face_track, config.max_track_slots)),
        "scene_feat": torch.from_numpy(np.ascontiguousarray(scene_feat)),
        "scene_frame": torch.from_numpy(np.minimum(np.arange(len(scene_feat), dtype=np.int64), config.max_frames - 1)),
        "emotion": torch.tensor(0, dtype=torch.long),
        "sentiment": torch.tensor(0, dtype=torch.long),
        "clip": clip,
    }


@torch.no_grad()
def predict(model, item: dict, pad_id: int, device: str, force_drop_text: bool = False) -> tuple[dict, dict]:
    batch = collate([item], pad_id=pad_id)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    out = model(batch, force_drop_text=force_drop_text)
    emotion = torch.softmax(out["emotion_logits"].float(), dim=-1)[0].cpu().numpy()
    sentiment = torch.softmax(out["sentiment_logits"].float(), dim=-1)[0].cpu().numpy()
    return ({e: float(p) for e, p in zip(EMOTIONS, emotion)},
            {s: float(p) for s, p in zip(SENTIMENTS, sentiment)})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_predict.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/meld_emotion/inference/predict.py tests/test_predict.py
git commit -m "Add the shared model-input builder and predictor (provisional = fusion with text masked)"
```

---

### Task 4: Visual gloss

**Files:**
- Create: `src/meld_emotion/inference/gloss.py`
- Test: `tests/test_gloss.py`

**Interfaces:**
- Consumes: `meld_emotion.vision.encoders.CLIP_MODEL_ID`; a loaded
  `SceneEncoder.model` (the shared `CLIPModel` from `loader.py` — this
  module never loads its own `CLIPModel`, only the lightweight tokenizer).
- Produces: `SCENE_PROMPT_BANK: tuple[str, ...]` (24 fixed phrases);
  `build_clip_tokenizer() -> CLIPTokenizer`; `embed_prompt_bank(clip_model,
  clip_tokenizer, device: str) -> torch.Tensor` (`(len(bank), 512)`,
  L2-normalised); `scene_gloss(mean_scene_embedding: np.ndarray,
  bank_embeddings: torch.Tensor, margin: float = 0.05) -> list[str]` (top-2
  phrases if their softmax-over-bank probability gap clears `margin`, else
  just the top-1); `face_gloss(faces_seen: int, vision_only_expression:
  dict[str, float] | None) -> str` (e.g. `"2 faces visible, reading
  surprise (52%) or neutral (23%)"`, or `"no faces visible"`).

The gloss is **LM-only** (design doc §4.4). Its face half is the fusion
model's own vision-only reading of the turn (Task 3 with text masked) —
not the face head's per-face labels, which are prior-skewed on MELD and
stale after Stage 2. CLIP is kept solely for the scene half; the
submitted classifier (`use_scene=False`) never sees it.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gloss.py`:

```python
import numpy as np
import pytest
import torch


def test_scene_gloss_picks_the_closest_bank_phrase():
    from meld_emotion.inference.gloss import SCENE_PROMPT_BANK, scene_gloss
    bank = torch.nn.functional.normalize(torch.eye(len(SCENE_PROMPT_BANK), 8), dim=-1)
    query = bank[3].numpy() * 5.0  # far from unit norm on purpose; scene_gloss must normalise it
    assert scene_gloss(query, bank, margin=0.05)[0] == SCENE_PROMPT_BANK[3]


def test_scene_gloss_reports_only_the_top1_when_the_next_is_within_margin():
    from meld_emotion.inference.gloss import scene_gloss
    bank = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.999, 0.045], [0.0, 1.0]]), dim=-1)
    assert len(scene_gloss(np.array([1.0, 0.0], dtype=np.float32), bank, margin=0.5)) == 1


def test_face_gloss_reads_the_vision_only_top2_and_counts_faces():
    from meld_emotion.inference.gloss import face_gloss
    probs = {"neutral": 0.23, "joy": 0.05, "surprise": 0.52, "anger": 0.1, "sadness": 0.05, "disgust": 0.03, "fear": 0.02}
    assert face_gloss(2, probs) == "2 faces visible, reading surprise (52%) or neutral (23%)"
    assert face_gloss(1, probs).startswith("1 face visible, reading surprise")
    assert face_gloss(0, probs) == "no faces visible"
    assert face_gloss(0, None) == "no faces visible"


def test_embed_prompt_bank_is_l2_normalised_and_unwraps_pooler_output():
    from meld_emotion.inference.gloss import SCENE_PROMPT_BANK, embed_prompt_bank

    class _StubOut:
        def __init__(self, n):
            self.pooler_output = torch.randn(n, 8) * 10  # deliberately not unit norm

    class _StubModel:
        def get_text_features(self, **kwargs):
            return _StubOut(len(SCENE_PROMPT_BANK))

    class _StubTokenizer:
        def __call__(self, texts, return_tensors, padding):
            return {"input_ids": torch.zeros(len(texts), 1, dtype=torch.long)}

    embeddings = embed_prompt_bank(_StubModel(), _StubTokenizer(), "cpu")
    norms = embeddings.norm(dim=-1)
    assert embeddings.shape == (len(SCENE_PROMPT_BANK), 8)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_gloss.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meld_emotion.inference.gloss'`

- [ ] **Step 3: Implement**

Create `src/meld_emotion/inference/gloss.py`:

```python
"""Visual gloss (design doc §4.4): human-readable cues for the response LM
from tensors already computed. (a) CLIP zero-shot scene cues against a
fixed prompt bank embedded once; (b) the fusion model's vision-only
reading of the faces. Feeds the LM only -- never the classifier."""
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
        feats = out if isinstance(out, torch.Tensor) else out.pooler_output   # transformers 5 wraps it
    return torch.nn.functional.normalize(feats.float().cpu(), dim=-1)


def scene_gloss(mean_scene_embedding, bank_embeddings: torch.Tensor, margin: float = 0.05) -> list[str]:
    query = torch.nn.functional.normalize(
        torch.as_tensor(mean_scene_embedding, dtype=torch.float32).reshape(1, -1), dim=-1)
    probs = torch.softmax((query @ bank_embeddings.T).squeeze(0) * 100.0, dim=-1)   # CLIP's logit scale
    top2 = torch.topk(probs, min(2, len(probs)))
    if len(top2.values) < 2 or (top2.values[0] - top2.values[1]).item() < margin:
        return [SCENE_PROMPT_BANK[top2.indices[0]]]
    return [SCENE_PROMPT_BANK[i] for i in top2.indices.tolist()]


def face_gloss(faces_seen: int, vision_only_expression: dict | None) -> str:
    if faces_seen <= 0 or not vision_only_expression:
        return "no faces visible"
    top = sorted(vision_only_expression.items(), key=lambda kv: kv[1], reverse=True)[:2]
    reading = " or ".join(f"{label} ({p:.0%})" for label, p in top)
    return f"{faces_seen} face{'s' if faces_seen != 1 else ''} visible, reading {reading}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_gloss.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/meld_emotion/inference/gloss.py tests/test_gloss.py
git commit -m "Add CLIP scene-cue gloss and the vision-only face reading for the LM prompt"
```

---

### Task 5: TurnProcessor — incremental vision pipeline and fused prediction

**Files:**
- Create: `src/meld_emotion/inference/turn.py`
- Test: `tests/test_turn.py`

**Interfaces:**
- Consumes: `InferenceBundle` (Task 2); `build_item`, `predict`,
  `dummy_text_encoding` (Task 3); `meld_emotion.data.preprocess.{letterbox,
  crop_face}`; `meld_emotion.training.text.{format_context, encode_text}`;
  `meld_emotion.vision.face_detector.detect_faces`;
  `meld_emotion.vision.tracker.{FaceTracker, SHOT_CUT_THRESHOLD,
  shot_change_score}`.
- Produces: `TurnProcessor(bundle: InferenceBundle, emitter)` with
  `.start_turn(turn_id: str)`, `.push_frame(frame_bgr: np.ndarray) -> dict`
  (the emitted `provisional` event; `provisional_expression` is the
  fusion model's vision-only prediction over every face token so far),
  `.end_turn(text: str, context_prev: list[str], visual_cues: list[str] |
  None = None) -> dict` (the emitted `final` event; `visual_cues` defaults
  to `[]` — the caller computes cues from the properties below and passes
  them in, since the event is built and emitted exactly once);
  `.last_provisional -> dict | None`; `.max_faces_seen -> int` (most faces
  in any sampled frame this turn); `.mean_scene_embedding -> np.ndarray`
  (for `gloss.scene_gloss`; zeros of `bundle.scene_dim` when no frames);
  `.last_boxes` / `.last_track_ids` (demo-only, for drawing between sampled
  frames).

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
    feature_dim = 8

    def encode_batch(self, images):
        n = len(images)
        feats = np.stack([np.full(8, 0.1 * (i + 1), np.float32) for i in range(n)]) if n else np.zeros((0, 8), np.float32)
        return {"features": feats, "probs": np.zeros((n, 7), dtype=np.float32)}


class _StubSceneEncoder:
    feature_dim = 4

    def encode(self, image):
        return np.full(4, 0.5, dtype=np.float32)


class _StubTokenizer:
    pad_token_id = 1
    cls_token_id = 0

    def __call__(self, *args, **kwargs):
        raise AssertionError("turn.py must go through encode_text, not call the tokenizer directly")


def _bundle(use_faces=True, use_scene=True, stage2=False):
    from meld_emotion.inference.loader import InferenceBundle
    from meld_emotion.training.config import TrainConfig
    from meld_emotion.training.model import FusionModel
    config = TrainConfig(d_model=32, n_heads=4, ff_dim=64, n_layers=1, use_faces=use_faces, use_scene=use_scene,
                         context_k=2, face_trainable_layers=4 if stage2 else 0)
    model = FusionModel(config, StubTextEncoder(32), face_dim=8, scene_dim=4).eval()
    return InferenceBundle(model=model, config=config, tokenizer=_StubTokenizer(), face_detector=_StubFaceDetector(),
                           face_encoder=_StubFaceEncoder(), scene_encoder=_StubSceneEncoder(), device="cpu",
                           face_dim=8, scene_dim=4, checkpoint_path=None, stage=2 if stage2 else 1)


def _fake_frame(color=(50, 50, 50)):
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:] = color
    return frame


class _Collector:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


def test_push_frame_emits_a_provisional_event_from_the_masked_fusion_model(monkeypatch):
    from meld_emotion.inference import turn as turn_module
    from meld_emotion.inference.predict import build_item, dummy_text_encoding, predict
    monkeypatch.setattr(turn_module, "detect_faces", lambda detector, frame: [(0, 0, 10, 10, 0.9)])
    bundle, collector = _bundle(), _Collector()
    tp = turn_module.TurnProcessor(bundle, emitter=collector)
    tp.start_turn("dia1_utt1")
    event = tp.push_frame(_fake_frame())
    assert event["phase"] == "provisional" and event["turn_id"] == "dia1_utt1" and event["frame"] == 0
    assert event["faces_seen"] == 1 and collector.events[-1] == event
    # the provisional state IS the fusion model with text masked over the tokens so far
    item = build_item(bundle.config, dummy_text_encoding(bundle.tokenizer), np.full((1, 8), 0.1, np.float32), [0], [0],
                      np.full((1, 4), 0.5, np.float32), face_dim=8, scene_dim=4, clip="dia1_utt1")
    expected, _ = predict(bundle.model, item, pad_id=1, device="cpu", force_drop_text=True)
    assert event["provisional_expression"] == pytest.approx(expected, abs=1e-6)
    assert tp.last_provisional == event["provisional_expression"]


def test_shot_cut_between_sampled_frames_resets_the_tracker_before_detection(monkeypatch):
    from meld_emotion.inference import turn as turn_module
    order = []
    monkeypatch.setattr(turn_module, "detect_faces", lambda detector, frame: order.append("detect") or [(0, 0, 10, 10, 0.9)])

    class _Tracker:
        def reset(self): order.append("reset")
        def update(self, boxes): order.append("update"); return [0] * len(boxes)
    monkeypatch.setattr(turn_module, "FaceTracker", lambda: _Tracker())
    monkeypatch.setattr(turn_module, "shot_change_score", lambda a, b: 0.9)  # every transition is "a cut"

    tp = turn_module.TurnProcessor(_bundle(), emitter=_Collector())
    tp.start_turn("dia1_utt1")
    tp.push_frame(_fake_frame((10, 10, 10)))
    tp.push_frame(_fake_frame((200, 200, 200)))
    assert order == ["reset", "detect", "update", "reset", "detect", "update"]   # start_turn reset; cut reset before detect


def test_end_turn_builds_the_training_shaped_batch_and_emits_final_without_response(monkeypatch):
    from meld_emotion.inference import turn as turn_module
    monkeypatch.setattr(turn_module, "detect_faces", lambda detector, frame: [(0, 0, 10, 10, 0.9)])
    monkeypatch.setattr(turn_module, "encode_text",
                        lambda tok, ctx, cur, max_len: {"input_ids": [0, 5, 6, 1], "attention_mask": [1, 1, 1, 1]})
    collector = _Collector()
    tp = turn_module.TurnProcessor(_bundle(), emitter=collector)
    tp.start_turn("dia1_utt1")
    tp.push_frame(_fake_frame())
    tp.push_frame(_fake_frame())
    final = tp.end_turn("You did WHAT?", ["earlier line one", "earlier line two"])
    assert final["phase"] == "final" and "response" not in final and "latency_ms" not in final
    assert final["text"] == "You did WHAT?" and final["faces_seen"] == 1
    assert set(final["emotion_probs"]) == {"neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"}
    assert final["emotion"] in final["emotion_probs"] and final["sentiment"] in final["sentiment_probs"]
    assert collector.events[-1] == final


def test_end_turn_carries_the_caller_supplied_visual_cues_and_defaults_to_empty(monkeypatch):
    from meld_emotion.inference import turn as turn_module
    monkeypatch.setattr(turn_module, "detect_faces", lambda detector, frame: [])
    monkeypatch.setattr(turn_module, "encode_text",
                        lambda tok, ctx, cur, max_len: {"input_ids": [0, 1], "attention_mask": [1, 1]})
    tp = turn_module.TurnProcessor(_bundle(), emitter=_Collector())
    tp.start_turn("dia1_utt1")
    assert tp.end_turn("hello", [], visual_cues=["two people", "no faces visible"])["visual_cues"] == ["two people", "no faces visible"]
    tp.start_turn("dia1_utt2")
    assert tp.end_turn("hello again", [])["visual_cues"] == []


def test_end_turn_with_zero_frames_still_produces_finite_predictions(monkeypatch):
    from meld_emotion.inference import turn as turn_module
    monkeypatch.setattr(turn_module, "encode_text",
                        lambda tok, ctx, cur, max_len: {"input_ids": [0, 1], "attention_mask": [1, 1]})
    tp = turn_module.TurnProcessor(_bundle(), emitter=_Collector())
    tp.start_turn("dia1_utt1")
    final = tp.end_turn("hello", [])
    assert final["faces_seen"] == 0 and tp.last_provisional is None
    assert all(np.isfinite(v) for v in final["emotion_probs"].values())
    assert np.allclose(tp.mean_scene_embedding, 0.0) and tp.mean_scene_embedding.shape == (4,)


def test_state_accumulates_across_the_turn_and_resets_on_start_turn(monkeypatch):
    from meld_emotion.inference import turn as turn_module
    monkeypatch.setattr(turn_module, "detect_faces", lambda detector, frame: [(0, 0, 10, 10, 0.9), (20, 20, 10, 10, 0.9)])
    tp = turn_module.TurnProcessor(_bundle(), emitter=_Collector())
    tp.start_turn("dia1_utt1")
    tp.push_frame(_fake_frame())
    tp.push_frame(_fake_frame())
    assert tp.max_faces_seen == 2 and np.allclose(tp.mean_scene_embedding, 0.5)
    assert len(tp.last_boxes) == 2 and len(tp.last_track_ids) == 2
    tp.start_turn("dia1_utt2")
    assert tp.max_faces_seen == 0 and tp.last_provisional is None and tp.last_boxes == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_turn.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meld_emotion.inference.turn'`

- [ ] **Step 3: Implement**

Create `src/meld_emotion/inference/turn.py`:

```python
"""One inference core (design doc §2, §4.2, §4.3). push_frame runs the same
per-frame vision logic preprocess_clip uses -- shot-cut check against the
previous SAMPLED frame, then face detect/track/encode, then scene encode
for the gloss -- and publishes a provisional state: the fusion model's own
vision-only prediction over the face tokens so far (text masked). end_turn
adds the real text and runs the model for the final event, which never
carries a response: the LM is a separate, slower step so it never gates
the state (design doc §2)."""
from typing import Optional

import numpy as np
from PIL import Image

from meld_emotion.data.preprocess import crop_face, letterbox
from meld_emotion.inference.loader import InferenceBundle
from meld_emotion.inference.predict import build_item, dummy_text_encoding, predict
from meld_emotion.training.text import encode_text, format_context
from meld_emotion.vision.face_detector import detect_faces
from meld_emotion.vision.tracker import FaceTracker, SHOT_CUT_THRESHOLD, shot_change_score


class TurnProcessor:
    def __init__(self, bundle: InferenceBundle, emitter):
        self.bundle = bundle
        self.emitter = emitter
        self.turn_id: Optional[str] = None
        self._tracker = FaceTracker()
        self._pad_id = bundle.tokenizer.pad_token_id if bundle.tokenizer is not None else 0
        self._reset_state()

    def start_turn(self, turn_id: str) -> None:
        self.turn_id = turn_id
        self._tracker.reset()
        self._reset_state()

    def _reset_state(self) -> None:
        self._prev_frame = None
        self._face_features: list[np.ndarray] = []
        self._face_frame_idx: list[int] = []
        self._face_track_raw: list[int] = []
        self._scene_features: list[np.ndarray] = []
        self.last_provisional: Optional[dict] = None
        self.max_faces_seen = 0
        self.last_boxes: list[tuple] = []
        self.last_track_ids: list[int] = []

    @property
    def mean_scene_embedding(self) -> np.ndarray:
        if not self._scene_features:
            return np.zeros(self.bundle.scene_dim, dtype=np.float32)
        return np.mean(self._scene_features, axis=0).astype(np.float32)

    def _item(self, enc: dict) -> dict:
        face_feat = np.concatenate(self._face_features) if self._face_features \
            else np.zeros((0, self.bundle.face_dim), np.float32)
        scene_feat = np.stack(self._scene_features) if self._scene_features \
            else np.zeros((0, self.bundle.scene_dim), np.float32)
        return build_item(self.bundle.config, enc, face_feat, self._face_frame_idx, self._face_track_raw, scene_feat,
                          face_dim=self.bundle.face_dim, scene_dim=self.bundle.scene_dim, clip=self.turn_id)

    def push_frame(self, frame_bgr: np.ndarray) -> dict:
        sample_i = len(self._scene_features)
        # Shot-cut check against the previous SAMPLED frame, before detection (as preprocess_clip does).
        if self._prev_frame is not None and shot_change_score(self._prev_frame, frame_bgr) >= SHOT_CUT_THRESHOLD:
            self._tracker.reset()
        self._prev_frame = frame_bgr

        boxes = detect_faces(self.bundle.face_detector, frame_bgr)
        track_ids = self._tracker.update(boxes)
        self.last_boxes, self.last_track_ids = boxes, track_ids
        crops, kept_tracks = [], []
        for (x, y, w, h, score), track_id in zip(boxes, track_ids):
            crop = crop_face(frame_bgr, (x, y, w, h))
            if crop is not None:
                crops.append(Image.fromarray(crop[:, :, ::-1]))   # OpenCV BGR -> PIL RGB
                kept_tracks.append(track_id)
        enc = self.bundle.face_encoder.encode_batch(crops)
        if len(kept_tracks):
            self._face_features.append(np.asarray(enc["features"], np.float32))
            self._face_frame_idx += [sample_i] * len(kept_tracks)
            self._face_track_raw += kept_tracks
        self.max_faces_seen = max(self.max_faces_seen, len(kept_tracks))

        self._scene_features.append(np.asarray(self.bundle.scene_encoder.encode(
            Image.fromarray(letterbox(frame_bgr)[:, :, ::-1])), np.float32))

        provisional, _ = predict(self.bundle.model, self._item(dummy_text_encoding(self.bundle.tokenizer)),
                                 self._pad_id, self.bundle.device, force_drop_text=True)
        self.last_provisional = provisional
        event = {"turn_id": self.turn_id, "phase": "provisional", "frame": sample_i,
                 "faces_seen": len(kept_tracks), "provisional_expression": provisional}
        self.emitter.emit(event)
        return event

    def end_turn(self, text: str, context_prev: list[str], visual_cues: list[str] | None = None) -> dict:
        cfg = self.bundle.config
        enc = encode_text(self.bundle.tokenizer, format_context(context_prev, cfg.context_k), text, cfg.max_text_tokens)
        emotion_probs, sentiment_probs = predict(self.bundle.model, self._item(enc), self._pad_id, self.bundle.device)
        event = {"turn_id": self.turn_id, "phase": "final", "text": text,
                 "emotion": max(emotion_probs, key=emotion_probs.get), "emotion_probs": emotion_probs,
                 "sentiment": max(sentiment_probs, key=sentiment_probs.get), "sentiment_probs": sentiment_probs,
                 "faces_seen": self.max_faces_seen, "visual_cues": list(visual_cues or [])}
        self.emitter.emit(event)
        return event
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_turn.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/meld_emotion/inference/turn.py tests/test_turn.py
git commit -m "Add TurnProcessor: incremental per-frame vision, vision-only provisional state, final event"
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
    "mlx-community/Qwen3-4B-4bit",              # already in ~/.cache/huggingface on the dev machine
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

The batch-path predictions for the submitted model already exist:
`results/stage2_fusion_faces_only/seed1/test_predictions.jsonl` (one line
per ok test row, `{"clip", "pred", "probs"}`, written by the `--eval-test`
run with the fine-tuned ViT over the JPEG crops). This task **reads** them;
it must not re-predict from the cached `.npz` features, which were encoded
by the *frozen* ViT and are stale for a Stage 2 model.

This is the first task whose tests import a top-level `scripts` module
(`from scripts.pick_demo_clips import ...`), which pytest cannot resolve
without `rootdir` on `sys.path` — every later task's tests need the same
fix, so it belongs here, once.

- [ ] **Step 1: Make `scripts` importable from tests**

Edit `pyproject.toml`'s `[tool.pytest.ini_options]` to add `pythonpath`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "-m 'not network'"
markers = [
    "network: downloads pretrained weights from Hugging Face; deselected by default, run with `uv run pytest -m network`",
]
```

Run: `uv run pytest -q`
Expected: the same pass count as before this edit — confirms the edit didn't break discovery.

```bash
git add pyproject.toml
git commit -m "Make scripts/ importable from tests (pythonpath) ahead of the scripts.* test modules"
```

**Interfaces:**
- Consumes: `meld_emotion.training.dataset.load_ok_rows`; the
  `test_predictions.jsonl` layout above.
- Produces: `load_test_predictions(predictions_path: Path, manifest_path:
  Path) -> list[dict]` (one dict per predicted clip: `clip, true_emotion,
  pred_emotion, n_faces, n_frames, duration_s`, joined on clip id);
  `stratified_candidates(predictions, per_emotion: int = 2, seed: int = 0)
  -> list[dict]`; `hard_case_candidates(predictions) -> dict[str,
  list[dict]]` (`"many_faces"` — top-5 by `n_faces`, `"zero_faces"` — up to
  5 with `n_faces == 0`, `"long"` — up to 3 with `duration_s > 15`);
  `wrong_prediction_candidates(predictions, seed: int = 0, limit: int = 3)
  -> list[dict]`.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_pick_demo_clips.py`:

```python
import json


def _predictions():
    return [
        {"clip": "dia1_utt0", "true_emotion": "joy", "pred_emotion": "joy", "n_faces": 3, "n_frames": 3, "duration_s": 2.0},
        {"clip": "dia1_utt1", "true_emotion": "joy", "pred_emotion": "neutral", "n_faces": 12, "n_frames": 3, "duration_s": 2.0},
        {"clip": "dia1_utt2", "true_emotion": "anger", "pred_emotion": "anger", "n_faces": 0, "n_frames": 3, "duration_s": 2.0},
        {"clip": "dia1_utt3", "true_emotion": "anger", "pred_emotion": "sadness", "n_faces": 4, "n_frames": 3, "duration_s": 20.0},
        {"clip": "dia1_utt4", "true_emotion": "neutral", "pred_emotion": "neutral", "n_faces": 2, "n_frames": 3, "duration_s": 2.0},
    ]


def test_stratified_candidates_samples_per_emotion():
    from scripts.pick_demo_clips import stratified_candidates
    picked = stratified_candidates(_predictions(), per_emotion=1, seed=0)
    assert {p["true_emotion"] for p in picked} == {"joy", "anger", "neutral"} and len(picked) == 3


def test_hard_case_candidates_finds_many_faces_zero_faces_and_long_clips():
    from scripts.pick_demo_clips import hard_case_candidates
    result = hard_case_candidates(_predictions())
    assert result["many_faces"][0]["clip"] == "dia1_utt1"
    assert [c["clip"] for c in result["zero_faces"]] == ["dia1_utt2"]
    assert [c["clip"] for c in result["long"]] == ["dia1_utt3"]


def test_wrong_prediction_candidates_finds_mismatches_only():
    from scripts.pick_demo_clips import wrong_prediction_candidates
    assert {c["clip"] for c in wrong_prediction_candidates(_predictions())} == {"dia1_utt1", "dia1_utt3"}


def test_load_test_predictions_joins_the_batch_output_with_the_manifest(synthetic_features, tmp_path):
    from meld_emotion.training.dataset import load_ok_rows
    from scripts.pick_demo_clips import load_test_predictions
    root, _, dev_manifest = synthetic_features
    rows = load_ok_rows(dev_manifest)
    preds_path = tmp_path / "test_predictions.jsonl"
    with open(preds_path, "w") as f:
        for r in rows:
            f.write(json.dumps({"clip": f"dia{r['dialogue_id']}_utt{r['utterance_id']}", "pred": "joy",
                                "probs": [0.1, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1]}) + "\n")
    predictions = load_test_predictions(preds_path, dev_manifest)
    assert len(predictions) == len(rows) == 14
    first = predictions[0]
    assert first["pred_emotion"] == "joy" and first["true_emotion"] == rows[0]["emotion"]
    assert first["n_faces"] == rows[0]["n_faces"] and first["duration_s"] == rows[0]["duration_s"]
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
candidates for the hard cases from design doc §5 (a many-face ensemble
clip, a zero-detected-face clip, an over-long mis-cut clip). Reads the
submitted model's own batch predictions (test_predictions.jsonl from the
--eval-test run) -- it never re-predicts from the cached features, which
are stale for a Stage 2 checkpoint. Produces CANDIDATE lists for a human
to pick the final ~10 from.

Usage:
    uv run python scripts/pick_demo_clips.py
    uv run python scripts/pick_demo_clips.py --predictions results/<run>/test_predictions.jsonl
"""
import argparse
import json
import random
from pathlib import Path

from meld_emotion.config import FEATURE_CACHE_DIR, REPO_ROOT
from meld_emotion.training.dataset import load_ok_rows

DEFAULT_PREDICTIONS = REPO_ROOT / "results" / "stage2_fusion_faces_only" / "seed1" / "test_predictions.jsonl"


def load_test_predictions(predictions_path: Path, manifest_path: Path) -> list[dict]:
    rows = {f"dia{r['dialogue_id']}_utt{r['utterance_id']}": r for r in load_ok_rows(manifest_path)}
    out = []
    with open(predictions_path) as f:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)
            row = rows[p["clip"]]
            out.append({"clip": p["clip"], "true_emotion": row["emotion"], "pred_emotion": p["pred"],
                        "n_faces": row["n_faces"], "n_frames": row["n_frames"], "duration_s": row["duration_s"]})
    return out


def stratified_candidates(predictions: list[dict], per_emotion: int = 2, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    by_emotion: dict[str, list[dict]] = {}
    for p in predictions:
        by_emotion.setdefault(p["true_emotion"], []).append(p)
    out = []
    for emotion in sorted(by_emotion):
        out.extend(rng.sample(by_emotion[emotion], min(per_emotion, len(by_emotion[emotion]))))
    return out


def hard_case_candidates(predictions: list[dict]) -> dict[str, list[dict]]:
    return {"many_faces": sorted(predictions, key=lambda p: -p["n_faces"])[:5],
            "zero_faces": [p for p in predictions if p["n_faces"] == 0][:5],
            "long": [p for p in predictions if p["duration_s"] > 15][:3]}


def wrong_prediction_candidates(predictions: list[dict], seed: int = 0, limit: int = 3) -> list[dict]:
    rng = random.Random(seed)
    wrong = [p for p in predictions if p["true_emotion"] != p["pred_emotion"]]
    return rng.sample(wrong, min(limit, len(wrong)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "demo_clip_candidates.json")
    args = parser.parse_args()

    predictions = load_test_predictions(args.predictions, FEATURE_CACHE_DIR / "test" / "manifest.jsonl")
    result = {"stratified": stratified_candidates(predictions), **hard_case_candidates(predictions),
              "wrong_predictions": wrong_prediction_candidates(predictions)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"{len(predictions)} predictions; stratified={len(result['stratified'])} many_faces={len(result['many_faces'])} "
          f"zero_faces={len(result['zero_faces'])} long={len(result['long'])} wrong={len(result['wrong_predictions'])}")
    print(f"-> {args.out}  (pick ~10 by hand into results/demo_clips.json: a JSON list of clip ids)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_pick_demo_clips.py -v`
Expected: 4 passed

- [ ] **Step 6: Run it for real and hand-curate the final list**

Run: `uv run python scripts/pick_demo_clips.py`
Expected: `2610 predictions; stratified=14 many_faces=5 zero_faces=<n> long=3 wrong=3` and
`results/demo_clip_candidates.json`. Pick ~10 clips covering every emotion
at least once, at least one wrong prediction, and the many-faces /
zero-faces / long candidates that look genuinely hard on inspection
(`uv run python scripts/view_meld_clips.py --split test` to eyeball). Save
the list to `results/demo_clips.json`, e.g. `["dia38_utt4", "dia220_utt0", ...]`.

- [ ] **Step 7: Commit**

```bash
git add scripts/pick_demo_clips.py tests/test_pick_demo_clips.py
git commit -m "Add curated demo-clip candidates from the submitted model's own test predictions"
```

---

### Task 8: Replay demo

**Files:**
- Create: `scripts/replay_demo.py`

**Interfaces:**
- Consumes: `load_inference_bundle` (Task 2); `build_clip_tokenizer`,
  `embed_prompt_bank`, `scene_gloss`, `face_gloss` (Task 4);
  `TurnProcessor` (Task 5); `Responder`, `build_prompt` (Task 6);
  `EventEmitter`, `LatencyStamps` (Task 1); `meld_emotion.data.preprocess
  .sample_frame_indices`; `meld_emotion.data.video_index.build_video_index`;
  `meld_emotion.config.split_video_dir`; `meld_emotion.data.manifest
  .read_manifest`.
- Produces: `DEFAULT_CHECKPOINT`, `DEFAULT_PREDICTIONS: Path` (the
  submitted model: `results/stage2_fusion_faces_only/seed1/best.pt` and its
  `test_predictions.jsonl`); `run_one_clip(bundle, responder,
  bank_embeddings, video_path: Path, row: dict, emitter: EventEmitter, *,
  show_window: bool = True) -> tuple[dict, dict]` (`final_event`,
  `done_event`); CLI `scripts/replay_demo.py --clips results/demo_clips.json
  --model-repo <chosen>`.

This is the only task that touches `cv2.imshow`; it is a thin orchestrator
over everything built so far, styled after `scripts/view_meld_clips.py`'s
overlay and kept minimal: video with face boxes + track IDs, a slim
provisional-expression bar strip, the transcript line at clip end, the
final emotion/sentiment tags, the visual cues, and the response streaming
in below. Its correctness is end-to-end (Task 9 runs `run_one_clip`
against real curated clips and checks the output against the batch path),
so it has no dedicated unit test.

**Latency is measured from end-of-turn** (design doc §2): `LatencyStamps.
turn_start` is stamped *after* the last frame is read — a clip plays for
~3 s in real time, and that playback is not "state latency".

- [ ] **Step 1: Implement**

Create `scripts/replay_demo.py`:

```python
#!/usr/bin/env python3
"""Replay demo (design doc §8.1): plays curated MELD test clips at real
speed, running the same inference core the live path will use, rendering
one minimal window (video + face boxes/track IDs + provisional bars +
transcript + final state + visual cues + streamed response).

Usage:
    uv run --with mlx-lm python scripts/replay_demo.py --clips results/demo_clips.json --model-repo <chosen LM>
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
from meld_emotion.inference.responder import Responder, build_prompt
from meld_emotion.inference.turn import TurnProcessor

DEFAULT_CHECKPOINT = REPO_ROOT / "results" / "stage2_fusion_faces_only" / "seed1" / "best.pt"
DEFAULT_PREDICTIONS = DEFAULT_CHECKPOINT.parent / "test_predictions.jsonl"
WINDOW = "Replay demo  (q=quit)"


def _draw_overlay(frame, tp: TurnProcessor, provisional: dict | None, caption: str) -> None:
    """Face boxes + track IDs held from the last SAMPLED frame, a thin bar per
    emotion sized by the current provisional probability, and a caption bar."""
    for (x, y, w, h, score), track_id in zip(tp.last_boxes, tp.last_track_ids):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"id{track_id}", (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if provisional:
        bar_x, bar_y, bar_w, row_h = 8, 8, 100, 14
        for i, (label, prob) in enumerate(provisional.items()):
            y = bar_y + i * row_h
            cv2.rectangle(frame, (bar_x, y), (bar_x + int(bar_w * prob), y + row_h - 4), (200, 200, 0), -1)
            cv2.putText(frame, label[:4], (bar_x + bar_w + 4, y + row_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    if caption:
        h = frame.shape[0]
        cv2.rectangle(frame, (0, h - 28), (frame.shape[1], h), (0, 0, 0), -1)
        cv2.putText(frame, caption[:110], (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def run_one_clip(bundle, responder, bank_embeddings, video_path: Path, row: dict, emitter: EventEmitter, *,
                 show_window: bool = True) -> tuple[dict, dict]:
    turn_id = f"dia{row['dialogue_id']}_utt{row['utterance_id']}"
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    wanted = set(sample_frame_indices(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), fps))
    delay_ms = max(1, int(1000 / fps))

    tp = TurnProcessor(bundle, emitter)
    tp.start_turn(turn_id)
    frame_idx, last_frame = -1, None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        last_frame = frame
        if frame_idx in wanted:
            tp.push_frame(frame)                         # provisional event; also updates tp.last_provisional
        if show_window:
            _draw_overlay(frame, tp, tp.last_provisional, "")
            cv2.imshow(WINDOW, frame)
            if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                break
    cap.release()

    # --- end of turn: the text arrives; everything below is what §2's targets measure ---
    stamps = LatencyStamps(turn_start=time.perf_counter())
    cues = scene_gloss(tp.mean_scene_embedding, bank_embeddings) + [face_gloss(tp.max_faces_seen, tp.last_provisional)]
    final = tp.end_turn(row["text"], row["context_prev"], visual_cues=cues)
    stamps.state_emitted = time.perf_counter()

    top2 = sorted(final["emotion_probs"].items(), key=lambda kv: kv[1], reverse=True)[:2]
    messages = build_prompt(row["context_prev"], row["text"], final["emotion"], top2, final["sentiment"], cues)
    response_text = ""
    for i, chunk in enumerate(responder.stream(messages)):
        if i == 0:
            stamps.first_token_emitted = time.perf_counter()
        response_text += chunk
        emitter.token(turn_id, chunk)
        if show_window and last_frame is not None:
            shown = last_frame.copy()
            _draw_overlay(shown, tp, tp.last_provisional, f"{final['emotion']}/{final['sentiment']}: {response_text}")
            cv2.imshow(WINDOW, shown)
            cv2.waitKey(1)
    stamps.done_emitted = time.perf_counter()
    done_event = {"turn_id": turn_id, "phase": "done", "response": response_text, "latency_ms": stamps.as_ms()}
    emitter.emit(done_event)

    if show_window and last_frame is not None:
        shown = last_frame.copy()
        _draw_overlay(shown, tp, tp.last_provisional,
                      f"{final['emotion']}/{final['sentiment']} | {'; '.join(cues)} | {response_text}")
        cv2.imshow(WINDOW, shown)
        cv2.waitKey(2000)
    return final, done_event


def warm_up(bundle, video_path: Path) -> None:
    """MPS compiles kernels on first use: the first sampled frame of a cold
    process costs ~1 s instead of ~0.2 s. Push one real frame through a
    throwaway TurnProcessor so the first curated clip is measured warm."""
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if ret:
        tp = TurnProcessor(bundle, EventEmitter(open(REPO_ROOT / "results" / "warmup_events.jsonl", "w")))
        tp.start_turn("warmup")
        tp.push_frame(frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", type=Path, default=REPO_ROOT / "results" / "demo_clips.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-repo", required=True, help="from results/lm_selection.json's 'chosen' field")
    parser.add_argument("--events", type=Path, default=REPO_ROOT / "results" / "replay_events.jsonl")
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()

    clip_ids = json.loads(args.clips.read_text())
    bundle = load_inference_bundle(args.checkpoint)
    bank_embeddings = embed_prompt_bank(bundle.scene_encoder.model, build_clip_tokenizer(), bundle.device)
    responder = Responder(model_repo=args.model_repo)
    rows_by_clip = {f"dia{r['dialogue_id']}_utt{r['utterance_id']}": r
                    for r in read_manifest(FEATURE_CACHE_DIR / "test" / "manifest.jsonl")}
    index = build_video_index(split_video_dir("test"))
    args.events.parent.mkdir(parents=True, exist_ok=True)
    emitter = EventEmitter(open(args.events, "a"))
    first = rows_by_clip[clip_ids[0]]
    warm_up(bundle, index[(first["dialogue_id"], first["utterance_id"])])

    for clip_id in clip_ids:
        row = rows_by_clip[clip_id]
        print(f"playing {clip_id}: \"{row['text']}\" -> true={row['emotion']}")
        final, done = run_one_clip(bundle, responder, bank_embeddings, index[(row["dialogue_id"], row["utterance_id"])],
                                   row, emitter, show_window=not args.no_window)
        print(f"  predicted={final['emotion']} cues={final['visual_cues']} response={done['response']!r} "
              f"latency_ms={done['latency_ms']}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test on the curated clips**

Run: `uv run --with mlx-lm python scripts/replay_demo.py --clips results/demo_clips.json --model-repo <chosen>`
Expected: a window plays each curated clip at real speed with face boxes
and moving provisional bars; at clip end the caption shows the final
tag, the cues, and the response streaming in; one printed summary line per
clip with `latency_ms` values that reflect end-of-turn processing only;
`results/replay_events.jsonl` gains provisional/final/token/done lines.
**Measured on the M1 while writing this plan** (real checkpoint, real
clips, MPS, fp32): `state` ≈ 200–250 ms (RoBERTa over ~100 tokens +
fusion — about 2× the design doc's 100 ms target; report the measured
number, don't tune to it), per-sampled-frame vision cost p95 ≈ 180–315 ms
once warm (under the 333 ms frame interval, but not by much — `warm_up`
exists because the cold first frame is ~1 s), and batch-vs-replay
agreement on 4/4 trial clips with max |Δp| ≤ 0.005.

- [ ] **Step 3: Commit**

```bash
git add scripts/replay_demo.py
git commit -m "Add the replay demo: curated clips through TurnProcessor + Responder, latency from end-of-turn"
```

---

### Task 9: Consistency check

**Files:**
- Create: `scripts/consistency_check.py`
- Test: `tests/test_consistency_check.py`

**Interfaces:**
- Consumes: `run_one_clip` (Task 8, imported as a library function — the
  replay path); the submitted model's batch-path predictions
  `results/stage2_fusion_faces_only/seed1/test_predictions.jsonl` (its
  `probs` list is in `EMOTIONS` order — the batch path with the in-loop
  ViT over the JPEG crops, capped at 64 per clip, which Task 3's
  `build_item` mirrors).
- Produces: `batch_predict_one(predictions_path: Path, clip_id: str) ->
  dict` (`emotion_probs: dict[str, float]`, keyed like the `final` event);
  `compare(batch_probs: dict, replay_probs: dict) -> dict` (`argmax_match:
  bool`, `max_abs_diff: float`); CLI that runs the replay path over
  `results/demo_clips.json` and asserts every clip passes.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_consistency_check.py`:

```python
import json

import pytest


def test_compare_reports_argmax_match_and_max_abs_diff():
    from scripts.consistency_check import compare
    a = {"neutral": 0.60, "joy": 0.20, "surprise": 0.10, "anger": 0.05, "sadness": 0.03, "disgust": 0.01, "fear": 0.01}
    b = {"neutral": 0.61, "joy": 0.19, "surprise": 0.10, "anger": 0.05, "sadness": 0.03, "disgust": 0.01, "fear": 0.01}
    result = compare(a, b)
    assert result["argmax_match"] is True and result["max_abs_diff"] == pytest.approx(0.01, abs=1e-6)


def test_compare_detects_a_real_argmax_mismatch():
    from scripts.consistency_check import compare
    a = {"neutral": 0.51, "joy": 0.49, "surprise": 0.0, "anger": 0.0, "sadness": 0.0, "disgust": 0.0, "fear": 0.0}
    b = {"neutral": 0.40, "joy": 0.60, "surprise": 0.0, "anger": 0.0, "sadness": 0.0, "disgust": 0.0, "fear": 0.0}
    assert compare(a, b)["argmax_match"] is False


def test_batch_predict_one_reads_the_stored_probs_in_emotion_order(tmp_path):
    from meld_emotion.data.labels import EMOTIONS
    from scripts.consistency_check import batch_predict_one
    path = tmp_path / "test_predictions.jsonl"
    probs = [0.05, 0.6, 0.1, 0.05, 0.1, 0.05, 0.05]
    path.write_text(json.dumps({"clip": "dia1_utt1", "pred": "joy", "probs": probs}) + "\n"
                    + json.dumps({"clip": "dia1_utt2", "pred": "anger", "probs": probs[::-1]}) + "\n")
    result = batch_predict_one(path, "dia1_utt1")
    assert list(result["emotion_probs"]) == list(EMOTIONS)
    assert result["emotion_probs"]["joy"] == pytest.approx(0.6)
    with pytest.raises(KeyError):
        batch_predict_one(path, "dia9_utt9")
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
prediction (test_predictions.jsonl from the same checkpoint's --eval-test
run). Tiny numeric drift is expected -- the batch path read JPEG-saved
crops, replay encodes in-memory crops -- so this asserts argmax equality
and a small probability tolerance, not bitwise equality.

Usage:
    uv run python scripts/consistency_check.py --clips results/demo_clips.json --model-repo <chosen LM>
"""
import argparse
import json
from pathlib import Path

from meld_emotion.config import FEATURE_CACHE_DIR, REPO_ROOT, split_video_dir
from meld_emotion.data.labels import EMOTIONS
from meld_emotion.data.manifest import read_manifest
from meld_emotion.data.video_index import build_video_index
from meld_emotion.inference.events import EventEmitter
from meld_emotion.inference.gloss import build_clip_tokenizer, embed_prompt_bank
from meld_emotion.inference.loader import load_inference_bundle
from meld_emotion.inference.responder import Responder
from scripts.replay_demo import DEFAULT_CHECKPOINT, DEFAULT_PREDICTIONS, run_one_clip

MAX_ABS_DIFF = 0.02


def batch_predict_one(predictions_path: Path, clip_id: str) -> dict:
    with open(predictions_path) as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                if p["clip"] == clip_id:
                    return {"emotion_probs": {e: float(v) for e, v in zip(EMOTIONS, p["probs"])}}
    raise KeyError(f"{clip_id} not in {predictions_path}")


def compare(batch_probs: dict, replay_probs: dict) -> dict:
    batch_argmax = max(batch_probs, key=batch_probs.get)
    replay_argmax = max(replay_probs, key=replay_probs.get)
    return {"argmax_match": batch_argmax == replay_argmax,
            "max_abs_diff": max(abs(batch_probs[e] - replay_probs[e]) for e in batch_probs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", type=Path, default=REPO_ROOT / "results" / "demo_clips.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--model-repo", required=True)
    args = parser.parse_args()

    clip_ids = json.loads(args.clips.read_text())
    bundle = load_inference_bundle(args.checkpoint)
    bank_embeddings = embed_prompt_bank(bundle.scene_encoder.model, build_clip_tokenizer(), bundle.device)
    responder = Responder(model_repo=args.model_repo)
    rows_by_clip = {f"dia{r['dialogue_id']}_utt{r['utterance_id']}": r
                    for r in read_manifest(FEATURE_CACHE_DIR / "test" / "manifest.jsonl")}
    index = build_video_index(split_video_dir("test"))
    emitter = EventEmitter(open(REPO_ROOT / "results" / "consistency_events.jsonl", "w"))

    results = []
    for clip_id in clip_ids:
        row = rows_by_clip[clip_id]
        batch = batch_predict_one(args.predictions, clip_id)
        final_event, _ = run_one_clip(bundle, responder, bank_embeddings, index[(row["dialogue_id"], row["utterance_id"])],
                                      row, emitter, show_window=False)
        results.append({"clip": clip_id, **compare(batch["emotion_probs"], final_event["emotion_probs"])})
    passed = all(r["argmax_match"] and r["max_abs_diff"] < MAX_ABS_DIFF for r in results)
    Path(REPO_ROOT / "results" / "consistency_check.json").write_text(json.dumps(results, indent=2))
    print(f"{'PASS' if passed else 'FAIL'}: {sum(r['argmax_match'] for r in results)}/{len(results)} argmax matches; "
          f"max |dp| = {max(r['max_abs_diff'] for r in results):.4f}")
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
Expected: `PASS: N/N argmax matches; max |dp| = 0.00xx` and
`results/consistency_check.json` written. A genuine mismatch means the
replay path's frame sampling, face detection, per-clip face cap, or
track-ID handling has drifted from the batch path — debug before
proceeding, since it would make the demo inconsistent with the reported
metrics.

- [ ] **Step 6: Commit**

```bash
git add scripts/consistency_check.py tests/test_consistency_check.py
git commit -m "Add batch-vs-replay consistency check against the submitted model's test predictions"
```

---

### Task 10: Latency measurement, parameter budget, and response rubric

**Files:**
- Create: `scripts/measure_latency.py`, `scripts/param_budget.py`,
  `scripts/response_rubric.py`
- Test: `tests/test_measure_latency.py`, `tests/test_param_budget.py`,
  `tests/test_response_rubric.py`

**Interfaces:**
- Consumes: `results/replay_events.jsonl` (Task 8); `TurnProcessor` (Task
  5); `load_inference_bundle` (Task 2); `meld_emotion.training.text
  .count_parameters`; `results/lm_selection.json` (Task 6).
- Produces: `percentile(values, p) -> float`; `summarise_latencies(
  done_events) -> dict` (`state_p50/p95, first_token_p50/p95,
  done_p50/p95`, ms); `time_one_frame(tp: TurnProcessor, frame_bgr) ->
  float` (seconds for one real `push_frame`, decode excluded);
  `count_lm_params(model_repo) -> int` (from the HF config, no weights);
  `count_bundle_params(bundle) -> dict[str, int]` (`text_and_fusion`,
  `face_encoder`, `scene_encoder`, `face_detector`); `build_budget_table(
  counts: dict, lm_repo: str, lm_params: int, stage: int) -> str`
  (markdown, the design doc §4.1 table with **measured** counts);
  `sample_response_rows(replay_events_path, n=50, seed=0) -> list[dict]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_measure_latency.py`:

```python
import numpy as np
import pytest


def test_percentile_matches_known_values():
    from scripts.measure_latency import percentile
    values = list(range(1, 101))
    assert percentile(values, 50) == pytest.approx(50.5, abs=0.5)
    assert percentile(values, 95) == pytest.approx(95.5, abs=0.5)


def test_summarise_latencies_computes_p50_p95_per_event():
    from scripts.measure_latency import summarise_latencies
    done_events = [{"latency_ms": {"state": 80, "first_token": 600, "done": 1800}},
                   {"latency_ms": {"state": 90, "first_token": 700, "done": 2000}},
                   {"latency_ms": {"state": 100, "first_token": 800, "done": 2200}}]
    summary = summarise_latencies(done_events)
    assert summary["state_p50"] == pytest.approx(90, abs=1)
    assert summary["done_p95"] >= summary["done_p50"] >= 1800


def test_time_one_frame_runs_a_real_push_frame(monkeypatch):
    from meld_emotion.inference import turn as turn_module
    from scripts.measure_latency import time_one_frame
    from test_turn import _Collector, _bundle
    monkeypatch.setattr(turn_module, "detect_faces", lambda detector, frame: [(0, 0, 10, 10, 0.9)])
    tp = turn_module.TurnProcessor(_bundle(), emitter=_Collector())
    tp.start_turn("t")
    seconds = time_one_frame(tp, np.zeros((48, 64, 3), dtype=np.uint8))
    assert seconds > 0 and tp.max_faces_seen == 1
```

Create `tests/test_param_budget.py`:

```python
import torch.nn as nn


def test_count_lm_params_reads_the_hf_config_without_downloading_weights(monkeypatch):
    from scripts.param_budget import count_lm_params

    class _StubConfig:
        num_hidden_layers, hidden_size, intermediate_size, vocab_size = 2, 8, 16, 100

    monkeypatch.setattr("scripts.param_budget.AutoConfig",
                        type("_A", (), {"from_pretrained": staticmethod(lambda repo: _StubConfig())}))
    assert count_lm_params("fake/repo") == 100 * 8 + 2 * (4 * 8 * 8 + 2 * 8 * 16)


def test_count_bundle_params_measures_each_loaded_component():
    from types import SimpleNamespace
    from scripts.param_budget import count_bundle_params
    bundle = SimpleNamespace(model=nn.Linear(4, 3), face_encoder=SimpleNamespace(model=nn.Linear(2, 2)),
                             scene_encoder=SimpleNamespace(model=nn.Linear(3, 1)))
    counts = count_bundle_params(bundle)
    assert counts == {"text_and_fusion": 15, "face_encoder": 6, "scene_encoder": 4, "face_detector": 75_000}


def test_build_budget_table_includes_every_component_and_the_total():
    from scripts.param_budget import build_budget_table
    counts = {"text_and_fusion": 136_000_000, "face_encoder": 86_000_000, "scene_encoder": 151_000_000, "face_detector": 75_000}
    table = build_budget_table(counts, "mlx-community/Qwen3-4B-4bit", 4_000_000_000, stage=2)
    assert "RoBERTa" in table and "CLIP" in table and "Qwen3-4B-4bit" in table and "Fine-tuned top 4" in table
    assert "**4.37B**" in table
```

Create `tests/test_response_rubric.py`:

```python
import json


def test_sample_response_rows_reads_done_events_and_adds_empty_rubric_fields(tmp_path):
    from scripts.response_rubric import sample_response_rows
    path = tmp_path / "events.jsonl"
    lines = [json.dumps({"turn_id": f"dia{i}_utt0", "phase": "done", "response": f"response {i}"}) for i in range(5)]
    path.write_text("\n".join(lines) + "\n")
    rows = sample_response_rows(path, n=3, seed=0)
    assert len(rows) == 3
    for row in rows:
        assert set(row) == {"turn_id", "prompt", "response", "tag_consistent", "uses_visual_cue", "concise",
                            "in_character", "no_invented_facts"}
        assert all(row[f] is None for f in ("tag_consistent", "uses_visual_cue", "concise", "in_character", "no_invented_facts"))


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
"""Measured (not estimated) latency (design doc §9): p50/p95 per
end-of-turn event from a completed replay run's event log, plus the
per-sampled-frame vision cost in isolation on the real model. Run
scripts/replay_demo.py first to produce results/replay_events.jsonl.

Usage:
    uv run python scripts/measure_latency.py
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np

from meld_emotion.config import REPO_ROOT, split_video_dir
from meld_emotion.data.video_index import build_video_index
from meld_emotion.inference.events import EventEmitter
from meld_emotion.inference.loader import load_inference_bundle
from meld_emotion.inference.turn import TurnProcessor
from scripts.replay_demo import DEFAULT_CHECKPOINT


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p))


def summarise_latencies(done_events: list[dict]) -> dict:
    out = {}
    for key in ("state", "first_token", "done"):
        values = [e["latency_ms"][key] for e in done_events if e["latency_ms"].get(key) is not None]
        out[f"{key}_p50"] = percentile(values, 50) if values else None
        out[f"{key}_p95"] = percentile(values, 95) if values else None
    return out


def time_one_frame(tp: TurnProcessor, frame_bgr) -> float:
    """One real push_frame (detect + track + face encode + scene encode +
    vision-only fusion forward), decode excluded. Design doc §9: must stay
    under the 333 ms (~3 fps) frame interval."""
    start = time.perf_counter()
    tp.push_frame(frame_bgr)
    return time.perf_counter() - start


def main():
    events_path = REPO_ROOT / "results" / "replay_events.jsonl"
    done_events = [e for e in (json.loads(l) for l in events_path.read_text().splitlines() if l.strip())
                   if e.get("phase") == "done"]
    summary = summarise_latencies(done_events)

    # per-frame vision cost on the real model, over the sampled frames of the curated clips
    clip_ids = json.loads((REPO_ROOT / "results" / "demo_clips.json").read_text())
    bundle = load_inference_bundle(DEFAULT_CHECKPOINT)
    index = build_video_index(split_video_dir("test"))
    tp = TurnProcessor(bundle, EventEmitter(open(REPO_ROOT / "results" / "latency_frame_events.jsonl", "w")))
    per_frame = []
    for clip_id in clip_ids:
        dia, utt = (int(p[3:]) for p in clip_id.split("_"))
        cap = cv2.VideoCapture(str(index[(dia, utt)]))
        step = max(1, round((cap.get(cv2.CAP_PROP_FPS) or 24.0) / 3.0))
        tp.start_turn(clip_id)
        i = -1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            i += 1
            if i % step == 0:
                per_frame.append(time_one_frame(tp, frame))
        cap.release()
    summary["vision_per_frame_p50_ms"] = percentile(per_frame, 50) * 1000
    summary["vision_per_frame_p95_ms"] = percentile(per_frame, 95) * 1000
    summary["n_turns"], summary["n_frames"] = len(done_events), len(per_frame)
    for key, value in summary.items():
        print(f"{key}: {value}")
    Path(REPO_ROOT / "results" / "latency.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

Create `scripts/param_budget.py`:

```python
#!/usr/bin/env python3
"""Final parameter-count table (design doc §4.1), measured from the loaded
inference bundle (text encoder + fusion, face ViT, CLIP both towers) plus
YuNet's fixed 75K and the chosen response LM's count from its HF config.

Usage:
    uv run python scripts/param_budget.py --model-repo <chosen from results/lm_selection.json>
"""
import argparse
import json
from pathlib import Path

from transformers import AutoConfig

from meld_emotion.config import REPO_ROOT
from meld_emotion.inference.loader import load_inference_bundle
from meld_emotion.training.text import count_parameters
from scripts.replay_demo import DEFAULT_CHECKPOINT

FACE_DETECTOR_PARAMS = 75_000   # YuNet 2023mar (232 KB ONNX)


def count_lm_params(model_repo: str) -> int:
    """From the HF config alone: embeddings + per-layer attention/FFN
    (biases/norms negligible). Same estimate style as the Stage 1 plan."""
    cfg = AutoConfig.from_pretrained(model_repo)
    hidden, layers = cfg.hidden_size, cfg.num_hidden_layers
    inter = getattr(cfg, "intermediate_size", 4 * hidden)
    return int(cfg.vocab_size * hidden + layers * (4 * hidden * hidden + 2 * hidden * inter))


def count_bundle_params(bundle) -> dict[str, int]:
    return {"text_and_fusion": count_parameters(bundle.model)[0],
            "face_encoder": count_parameters(bundle.face_encoder.model)[0],
            "scene_encoder": count_parameters(bundle.scene_encoder.model)[0],
            "face_detector": FACE_DETECTOR_PARAMS}


def build_budget_table(counts: dict, lm_repo: str, lm_params: int, stage: int) -> str:
    face_note = "Fine-tuned top 4 layers (Stage 2)" if stage >= 2 else "Frozen"
    rows = [("RoBERTa-base + fusion transformer + projectors + heads", counts["text_and_fusion"], "RoBERTa top half fine-tuned; fusion from scratch"),
            ("Face/expression encoder (ViT-Base)", counts["face_encoder"], face_note),
            ("CLIP ViT-B/32 (image + text towers; gloss only)", counts["scene_encoder"], "Frozen"),
            ("Face detector (YuNet)", counts["face_detector"], "Zero-shot"),
            (f"Response LM ({lm_repo.split('/')[-1]})", lm_params, "Prompted, not fine-tuned")]
    total = sum(n for _, n, _ in rows)
    lines = ["| Component | Params | Trained? |", "|---|---|---|"]
    lines += [f"| {name} | {n / 1e6:,.1f}M | {trained} |" if n < 1e9 else f"| {name} | {n / 1e9:.2f}B | {trained} |"
              for name, n, trained in rows]
    lines.append(f"| **Total** | **{total / 1e9:.2f}B** | ceiling 6B |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    bundle = load_inference_bundle(args.checkpoint, device="cpu")
    counts = count_bundle_params(bundle)
    lm_params = count_lm_params(args.model_repo)
    table = build_budget_table(counts, args.model_repo, lm_params, bundle.stage)
    print(table)
    (REPO_ROOT / "results").mkdir(exist_ok=True)
    (REPO_ROOT / "results" / "param_budget.json").write_text(json.dumps(
        {"model_repo": args.model_repo, "lm_params": lm_params, **counts,
         "total": sum(counts.values()) + lm_params, "stage": bundle.stage}, indent=2))
    (REPO_ROOT / "docs" / "results").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "docs" / "results" / "param_budget.md").write_text(table + "\n")


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
    finals = {e["turn_id"]: e for e in events if e.get("phase") == "final"}
    done = [e for e in events if e.get("phase") == "done"]
    sampled = random.Random(seed).sample(done, min(n, len(done)))
    rows = []
    for e in sampled:
        final = finals.get(e["turn_id"], {})
        prompt = f"{final.get('text', '')} [{final.get('emotion', '')}/{final.get('sentiment', '')}; " \
                 f"cues: {', '.join(final.get('visual_cues', []))}]"
        rows.append({"turn_id": e["turn_id"], "prompt": prompt, "response": e["response"], **{f: None for f in RUBRIC_FIELDS}})
    return rows


def main():
    rows = sample_response_rows(REPO_ROOT / "results" / "replay_events.jsonl")
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
Expected: 8 passed

- [ ] **Step 5: Run the full offline suite**

Run: `uv run pytest -q`
Expected: every test from the data, Stage 1, Stage 2 and this plan passes; `network` tests deselected.

- [ ] **Step 6: Run the three reporting scripts for real**

```bash
uv run python scripts/measure_latency.py            # after Task 8 has produced results/replay_events.jsonl
uv run python scripts/param_budget.py --model-repo <chosen from results/lm_selection.json>
uv run python scripts/response_rubric.py
```
Expected: `results/latency.json` (with `vision_per_frame_p95_ms` well under
333), `results/param_budget.json`, `docs/results/param_budget.md`, and
`results/response_rubric.jsonl`; hand-score the rubric afterwards — the
scored file goes in the write-up. The rubric samples from *all* `done`
events in the log, so run the replay over more than the curated 10 (e.g.
`--clips` with a 50-clip list drawn from `demo_clip_candidates.json`,
`--no-window`) before generating it.

- [ ] **Step 7: Commit**

```bash
git add scripts/measure_latency.py scripts/param_budget.py scripts/response_rubric.py \
       tests/test_measure_latency.py tests/test_param_budget.py tests/test_response_rubric.py
git commit -m "Add latency measurement (incl. per-frame vision cost), measured parameter budget, and response rubric"
```

---

## Running this plan for real

Prerequisites (all exist): `results/stage2_fusion_faces_only/seed1/best.pt`
(dev wF1 0.634, test 0.642) and its `test_predictions.jsonl`;
`data/meld/features/test/manifest.jsonl`; the test videos.

```bash
# 1. Once: the LM choice (Task 6, Step 6)
uv run --with mlx-lm python scripts/select_lm.py

# 2. Curated clips (Task 7) -> hand-pick ~10 into results/demo_clips.json
uv run python scripts/pick_demo_clips.py

# 3. The demo itself (Task 8)
uv run --with mlx-lm python scripts/replay_demo.py --clips results/demo_clips.json --model-repo <chosen>

# 4. Verification and reporting (Tasks 9, 10)
uv run --with mlx-lm python scripts/consistency_check.py --clips results/demo_clips.json --model-repo <chosen>
uv run --with mlx-lm python scripts/replay_demo.py --clips results/rubric_clips.json --model-repo <chosen> --no-window   # ~50 clips for the rubric
uv run python scripts/measure_latency.py
uv run python scripts/param_budget.py --model-repo <chosen>
uv run python scripts/response_rubric.py
# -> hand-score results/response_rubric.jsonl
```

**If the LM misses the latency target:** the design doc's fallback is the
smallest candidate (`Qwen2.5-1.5B-Instruct-4bit`); `select_lm.py` already
tries it last, and no code changes — just pass its repo id as
`--model-repo`.

## Self-Review

**Spec coverage.** §4.4 prompt/gloss/serving/evaluation → Tasks 4, 6, 10
(rubric); §4.5 event schema (with the `final`-carries-no-response
resolution) → Task 1, consumed everywhere; §8.1 replay window and
batch-vs-replay consistency → Tasks 8, 9; §9 latency (per event *and* per
sampled frame), parameter count → Task 10; §2's separate latency targets,
measured from end-of-turn, and "state never gated by the LM" → Tasks 5, 8;
§4.1's LM selection rule → Task 6; §6's Stage 2 outcome (fine-tuned face
ViT, no scene token) → Tasks 2, 3, 5 load and mirror it. The live path and
the tri-modal/RL extension are out of scope per the design doc's build
priority.

**Placeholder scan.** No TBD/TODO markers; every code block is complete and
runnable as written, with exact commands and expected output.

**Type consistency.** `InferenceBundle` (Task 2: `model, config, tokenizer,
face_detector, face_encoder, scene_encoder, device, face_dim, scene_dim,
checkpoint_path, stage`) is read identically by Tasks 3, 5, 8, 9, 10.
`build_item`/`predict` (Task 3) are the only path from accumulated tokens
to a batch, used by `push_frame` and `end_turn` (Task 5), and they mirror
Stage 2's per-clip face cap via `config.face_trainable_layers`.
`TurnProcessor` (Task 5) emits exactly the `provisional`/`final` shapes
Task 1 defines; `.last_provisional`, `.max_faces_seen`,
`.mean_scene_embedding` feed `face_gloss`/`scene_gloss` (Task 4), whose
output is passed into `end_turn`'s `visual_cues`. `run_one_clip` (Task 8)
returns `(final_event, done_event)`; Task 9 compares `final_event
["emotion_probs"]` (keyed by `EMOTIONS`) against `batch_predict_one`'s
dict built from `test_predictions.jsonl`'s `probs` list in the same order.
`Responder.stream` (Task 6) yields `str` deltas consumed by Task 8's token
loop. `DEFAULT_CHECKPOINT`/`DEFAULT_PREDICTIONS` are defined once (Task 8)
and imported by Tasks 9 and 10. The synthetic checkpoint fixtures
(conftest, Task 2) write exactly `train.py`'s / `train_stage2.py`'s save
formats.
