# Briefing for the response-LM + replay-demo plan

For whoever writes `docs/superpowers/plans/<date>-response-lm-replay-demo.md`.
Everything below is verified against the code and data as of 2026-09-12; the
design doc is `docs/superpowers/specs/2026-09-11-multimodal-emotion-prototype-design.md`.

## 1. Scope of this plan — and only this

**In:** the inference core that turns one turn into the §4.5 event stream
(provisional → final → token… → done); the response LM (local, streamed);
the visual gloss (§4.4 a + b); the replay demo on ~10 curated test clips
(§8.1); the batch-vs-replay consistency check; latency measurement
(p50/p95 per event, §2/§9); the parameter-budget table (§4.1); the 50-sample
response rubric file (§4.4).

**Out:** the live webcam/VAD/Whisper path (next plan), Stage 2 training,
any retraining, any change to how test was selected. Do not add the
tri-modal extension or RL.

## 2. Read first

- Spec §2 (the turn model and the three latency targets), §4.4, §4.5, §8.1, §9.
- `docs/superpowers/plans/2026-09-12-stage1-training.md` — for the *format*
  a plan must have here: header block, Global Constraints, per-task
  Files/Interfaces, TDD steps with exact commands and expected output, no
  placeholders, a self-review at the end. Every code block must be
  runnable as written; the previous plans were verified by extracting every
  block and running the tests before hand-off. Do the same.
- `docs/results/stage1_ablations.md` — the numbers the demo is showing off.

## 3. Existing code to reuse — exact names, do not reimplement

Model and checkpoint (`meld_emotion.training`):
- `train.load_checkpoint(path, model)`; checkpoint keys: `model_state`,
  `config` (a `TrainConfig` as dict → `TrainConfig(**ckpt["config"])`),
  `epoch`, `dev_weighted_f1`, `face_dim` (768), `scene_dim` (512).
- `model.FusionModel(config, text_encoder, face_dim=768, scene_dim=512)`;
  `forward(batch) -> {"emotion_logits", "sentiment_logits"}`; softmax them.
- `text.build_text_encoder(config.text_model, config.text_trainable_layers)`,
  `text.build_tokenizer`, `text.encode_text(tokenizer, context, current, max_length)`,
  `text.format_context(context_prev, k)` — the *only* way to build the text
  input; it must match training byte-for-byte.
- `dataset.collate(batch, pad_id)` with a one-item list builds an inference
  batch; `dataset.remap_track_ids(ids, config.max_track_slots)`; frame
  positions are clamped `min(idx, config.max_frames - 1)`. Item keys:
  `input_ids, attention_mask, face_feat (N,768), face_frame (N,),
  face_track (N,), scene_feat (M,512), scene_frame (M,), emotion,
  sentiment, clip` — for inference pass dummy labels (0).
- Labels: `meld_emotion.data.labels.EMOTIONS` =
  `("neutral","joy","surprise","anger","sadness","disgust","fear")`, `SENTIMENTS`.
- The best checkpoint: `results/fusion/seed0/best.pt` (519 MB; includes all
  of RoBERTa). Loading it needs the model built the same way, then
  `load_state_dict`.

Vision (`meld_emotion.vision`, `meld_emotion.data.preprocess`) — the replay
path must reproduce `preprocess_clip`'s per-frame logic *incrementally*
(one sampled frame at a time), not call `preprocess_clip` (it reads the
whole file and writes JPEGs):
- `preprocess.sample_frame_indices(total_frames, fps)` (3 fps, 15 s cap),
  `preprocess.letterbox(frame)`, `preprocess.crop_face(frame, box)` (20 %
  margin, 224²).
- `face_detector.build_face_detector()` / `detect_faces(detector, frame_bgr)`
  → `(x, y, w, h, score)`; threshold 0.75 is baked in.
- `tracker.FaceTracker()` (`update(boxes) -> track_ids`, `reset()`),
  `tracker.shot_change_score(prev, curr)`, `tracker.SHOT_CUT_THRESHOLD = 0.3`.
- `encoders.FaceEmotionEncoder(device)` / `SceneEncoder(device)`:
  `.encode_batch(list_of_PIL_RGB)` → faces: `{"features": (N,768),
  "probs": (N,7)}`; scenes: `(M,512)`. `.meld_labels` gives the face head's
  column order in MELD spelling. Crops come from OpenCV (BGR) — convert with
  `Image.fromarray(crop[:, :, ::-1])` before encoding.
- Data locations: test videos under `split_video_dir("test")` (index them
  with `data.video_index.build_video_index` — never a global glob, IDs
  collide across splits); text/context/labels from
  `data/meld/features/test/manifest.jsonl` (`context_prev` is up to 8
  previous lines, oldest first; training used the last 4).

## 4. Facts that will bite (all verified)

- **The frozen face head's raw probabilities are prior-skewed on MELD.** It
  almost never says "neutral" on talking faces (0.10 on true-neutral dev
  clips) and defaults to sad/joy; raw argmax scores wF1 0.12 on dev, below
  always-neutral. The §2 *provisional* state must not display these raw.
  Compute the head's mean probability vector over the dev cache once
  (`face_probs` in the `.npz` files), store it as a small JSON, and show
  `p / prior` renormalised (or the top-2 relative to prior). Say in the
  plan that this is a calibration, not a learned component.
- **CLIP under transformers 5:** `model.get_text_features(...)` and
  `get_image_features(...)` return `BaseModelOutputWithPooling`; take
  `.pooler_output` (512-d). Neither is L2-normalised — normalise both sides
  before cosine. Cosine margins between prompt-bank entries are small
  (~0.2 vs 0.22); use CLIP's logit scale (×100) then softmax over the bank
  and report the top-2 with a probability margin, not raw cosines. The
  cached `scene_features` are the same projected 512-d vectors, so the
  gloss can be developed against the cache without decoding video.
- **CLIP's text tower is 63.4M params** and is loaded already
  (`SceneEncoder.model` is the full `CLIPModel`); it counts in the budget.
- **MPS:** fp32 only. The training-time SDPA/dropout gotcha does not apply
  at inference (eval mode), but build the text encoder the same way the
  checkpoint was built (`build_text_encoder`) so state-dict keys match.
- **mlx-lm runs in its own runtime** (MLX, not torch). Load once, keep it
  resident; a 3–4B model at 4-bit is ~2–2.5 GB. Use the chat template with
  any "thinking" mode disabled (Qwen3: `enable_thinking=False`);
  `max_tokens` ≈ 40; stream tokens. Verify the exact `mlx_lm` API
  (`load`, `stream_generate`) at implementation — it changes between
  versions. Count the LM's parameters from its config and put the final
  number in the budget table (spec §4.1 ceiling: 6B total; everything
  before the LM is ~375M).
- **Latency targets are per event and the events are separate** (§2):
  `final` state ≤100 ms after end-of-turn on replay; first token ≤1 s;
  done ≤2.5 s. The LM must never gate the `final` event. Measure with
  `time.perf_counter()` at emission and report p50/p95 over the demo clips
  in `results/latency.json`. Include the per-sampled-frame vision cost too —
  it must stay under the 333 ms frame interval (ViT-B is ~20 ms/face on
  MPS, so this holds unless a frame has >10 faces).
- **Consistency check (§8.1):** the replay path's `final` prediction for a
  demo clip must equal the batch path's prediction from the cached
  features with the same checkpoint. Expect tiny numeric drift — the cache
  encoded JPEG-saved crops, replay encodes in-memory crops — so assert
  argmax equality and `max|Δprob| < 0.02`, not bitwise equality. Use the
  same `sample_frame_indices` and the same detector threshold or the
  frame sets won't line up.
- **Curated demo clips:** ~10 from *test*, stratified by emotion, including
  the hard cases from spec §5 (an ensemble shot, a face-away clip, a dark
  one) and at least one the model gets wrong. `results/fusion/seed0/`
  will contain `test_predictions.jsonl` after the `--eval-test` run
  (`{"clip", "pred", "probs"}` per test row) — pick from that.
- **Speaker names are never used** (path parity, §4.3). Don't put them in
  the prompt either.
- **Deadline is ~2026-09-15.** Keep the plan to what §8.1 needs; the live
  path is a separate plan. No timelines in the doc.

## 5. Design points that are easy to get wrong

- One inference core, two thin frontends (§8): put the turn logic in a
  class (e.g. `TurnProcessor`) with `push_frame(frame_bgr) -> provisional
  event` and `end_turn(text, context_prev) -> final event`, and a separate
  `Responder.stream(prompt) -> token events`. The replay script and the
  future live script both just drive that.
- Event stream = JSON lines exactly as §4.5 (`phase: provisional | final |
  token | done`; `latency_ms: {state, first_token, done}` on `done`). Make
  the emitter a tiny class that writes to any file-like (stdout, a file, a
  list in tests).
- The gloss feeds the **LM only** (§4.4) — never the classifier.
- The demo window: reuse the overlay style of `scripts/view_meld_clips.py`;
  play at real speed with `cv2.waitKey(delay_ms)`; run the pipeline only on
  every `step`-th frame (step = `round(fps / 3)`), and draw boxes/track
  IDs on every frame from the last sampled result.
- Prompt: fixed persona system prompt (small friendly character robot;
  react in ≤2 sentences; don't summarise), then the last 4 lines, the
  current line, the predicted emotion with top-2 probabilities, the
  sentiment, and the gloss phrases. Keep it under ~200 tokens so prefill
  stays fast.
- Response eval: write `results/response_rubric.jsonl` with the 50 sampled
  prompts + responses and empty rubric fields (tag-consistent, uses a
  visual cue when present, ≤2 sentences, in character, no invented facts)
  for hand scoring. That file is a deliverable of the plan.

## 6. Suggested layout

```
src/meld_emotion/inference/
  loader.py      # checkpoint -> (FusionModel, tokenizer, encoders) on a device
  gloss.py       # prompt bank, CLIP text embeddings (cached), scene + face gloss
  prior.py       # face-head prior from the dev cache; provisional-state correction
  turn.py        # TurnProcessor: incremental frames -> provisional; end_turn -> final
  events.py      # event dataclasses / JSON-lines emitter / latency stamps
  responder.py   # mlx-lm wrapper: prompt building + streaming tokens
scripts/
  pick_demo_clips.py, replay_demo.py, measure_latency.py, param_budget.py
tests/ ...       # stub LM (yields fixed tokens), stub encoders; offline in seconds;
                 # `network` marker for anything that downloads
```

## 7. Verification bar before hand-off

Extract every code block from the plan into a scratch copy of the package
and run its tests (that is how the two previous plans caught their bugs).
The plan's smoke steps should include: one real clip through
`TurnProcessor` on MPS printing all four event phases; the consistency
check passing on the curated clips; a latency JSON with p50/p95; and the
LM latency test that picks the model per §4.1's selection rule.
