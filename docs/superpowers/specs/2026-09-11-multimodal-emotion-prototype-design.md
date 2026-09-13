# Real-Time Multimodal Emotion Prototype — Design

**Date:** 2026-09-11 (revised)
**Track:** Text + Vision
**Status:** Design accepted — next step is the implementation plan

## 1. Problem framing

We're building a small end-to-end prototype for emotion-aware interaction with a
character robot: combine an utterance's text with the speaker's facial
expression, produce a structured emotion state plus a short grounded response,
and do it as a live, ongoing interaction rather than a one-shot batch
classification.

The assignment asks for exactly two outputs per turn, and this design maps to
them directly:

1. **Structured state/tags containing a MELD emotion category** — §4.3 (the
   predictor) and §4.5 (the state schema).
2. **Short response text grounded in the multimodal input** — §4.4.

MELD (Multimodal EmotionLines Dataset) is the shared foundation: ~13,700
utterances across 1,433 dialogues from *Friends*, each labeled with one of 7
emotions (anger, disgust, fear, joy, neutral, sadness, surprise) plus a 3-way
sentiment, and each utterance packaged as a short video clip with synced audio
and a transcript line.

## 2. Real-time definition

A **turn** is one utterance: a few seconds of video arriving together with a
line of text (MELD's own unit — verified average clip length 3.1s, average
utterance length 7.9 words). A turn ends when the clip ends (replay) or when
voice-activity detection sees end-of-speech (live).

"Real-time" here means voice-assistant-grade turn responsiveness, not
animation-loop timing — nothing at this model scale on a laptop could honestly
promise sub-100ms, and the assignment doesn't require it. Concretely:

**During the turn — input is consumed as it arrives.**
- Frames are sampled at ~3fps and pushed through the vision pipeline (§4.2)
  as they arrive, not after the clip finishes. By end-of-turn, all vision
  work for the turn is already done.
- A **provisional expression state** is published after every sampled frame:
  the fusion model's own prediction with the text masked, over the face
  tokens accumulated so far — exactly the input modality dropout (§4.3)
  trained it to handle. (The face encoder's own head was the original plan;
  measured on MELD it is prior-skewed — raw argmax weighted-F1 0.12 on dev,
  never predicting neutral — so it is not used for anything the user sees.)
  One extra fusion forward per sampled frame (~10 ms), and it is what makes
  "accept input over time" visibly true — the state moves while the person
  is still talking.

**At end of turn — three events, each with a latency target.**

| Event | Target (replay) | Target (live) |
|---|---|---|
| Final fused emotion + sentiment state (§4.5) | ≤100 ms | ≤500 ms (includes ASR) |
| First response token | ≤1.0 s | ≤1.0 s after the state |
| Response complete (streamed) | ≤2.5 s | ≤2.5 s after the state |

The state and the response are deliberately **separate events**. The response
LM is the slowest component by an order of magnitude; if it gated the state,
the robot couldn't react (face, posture, LEDs) until it was ready to speak.
Splitting them is both the honest latency story and the better interaction.

Everything runs on-device (16GB M1 MacBook); no remote inference. These are
targets, not claims — measured p50/p95 per event on both paths is a required
deliverable (§9).

## 3. Scope

**In scope:**
- Text + vision fusion producing a 7-way emotion distribution and a 3-way
  sentiment distribution per turn (§4.3), plus a provisional vision-only
  expression state during the turn (§2).
- A short grounded response per turn from a local LM, streamed (§4.4).
- Two demo paths sharing one inference core (§8).
- Full training/evaluation pipeline against MELD with ablations, reported
  against published baselines (§6, §7).
- Measured latency and resource requirements (§9).
- This document's evidence, trade-offs, and limitations.

**Explicitly out of scope for this submission:**
- The tri-modal (+voice-as-a-fusion-input) extension and reinforcement
  learning — deferred per the assignment's "core first" guidance. The fusion
  module (a variable-size token set with modality-type embeddings, §4.3) would
  admit a third modality without redesign, but it is not built or evaluated.
- Identity-based face recognition. MELD's `Speaker` field is not a closed set
  (guest/one-off characters appear alongside the main cast), so identity
  enrollment doesn't generalize, and the detect-everything + attention
  approach in §4.2 doesn't need it.
- Audio-visual active speaker detection (lip-sync matching) — a research
  problem in its own right; superseded by detection + attention.

**Build priority (what gets cut first if time runs short):**
1. Data layer, preprocessing, feature cache, text(+context) baseline.
2. Fusion model, Stage 1 training, ablation table, batch eval — the
   submittable core.
3. Response LM, visual gloss, replay demo, latency measurement.
4. Live path.
5. Stage 2 training on Modal — only once Stage 1 has produced a strong
   benchmark.
6. Stretch: appearance-gated tracking and the ±track-ID ablation,
   RoBERTa-large, other-show clips.

Cut order is the reverse: 6 first, then Stage 2, then the live path degrades
from VAD to push-to-talk.

## 4. Architecture

### 4.1 Components and parameter budget

| Component | Role | Params | Trained? |
|---|---|---|---|
| Text encoder — RoBERTa-base | Encode current utterance + previous *k* utterances (§4.3) | ~125M | Fine-tuned, top half of layers (Stage 1) |
| Face/expression encoder — ViT-Base (`dima806/facial_emotions_image_detection`) | Encode each detected face crop. Its own 7-way head is kept only as the zero-training vision-only baseline (§7) — it is prior-skewed on MELD (§2) | ~86M | Frozen (Stage 1); top 4 layers unfrozen (Stage 2, §6) |
| Scene encoder — CLIP ViT-B/32 image tower | Encode the letterboxed full frame: setting, hands, number of people, activity | ~88M | Frozen |
| CLIP text tower | Encodes the fixed visual-gloss prompt bank (§4.4). Computed once and cached, but counted: the gloss requires it | ~63M | Frozen |
| Modality projectors | Linear maps from face (768-d), scene (512-d) features into the fusion width | ~1M | From scratch |
| Face detector — OpenCV YuNet | Locate faces per sampled frame | ~75K | Zero-shot |
| Face tracker | Per-turn track IDs: IoU matching + shot-cut reset | 0 (geometry) | N/A |
| Fusion transformer — single-stream, 2–4 layers, d=768 | Joint self-attention over text + face + scene tokens | ~15–30M | From scratch |
| Emotion head (7-way) + sentiment head (3-way) | Linear heads on the fused token | <1M | From scratch |
| Response LM — ~3–4B instruction-tuned, 4-bit via MLX | Short grounded response, streamed | ~3–4B | Prompted, not fine-tuned |
| VAD — webrtcvad (live only) | End-of-turn detection | 0 (non-learned) | N/A |
| ASR — Whisper-base (live only) | Speech-to-text for the live path | ~74M | Zero-shot |

**Total: ~3.4–4.4B on the replay path, ~3.5–4.5B on the live path** (ASR is
required on that path, so it's counted there; MELD supplies ground-truth text
on replay). Comfortably under the 6B ceiling. The exact figure is fixed once
the LM is chosen.

**Response LM selection rule.** Candidates are ~3–4B instruction-tuned models
with permissive licenses (Qwen3-4B / Qwen2.5-3B / Phi-3.5-mini class), with a
~1.5–2B model of the same family as the latency fallback. The choice is made
by a short measured test on the M1: prefill + decode speed at 4-bit via
`mlx-lm`, with any "thinking" mode disabled. The largest candidate that meets
the §2 targets wins. Quantization is the lever for capability within the
latency budget; it does not change the counted parameter total.

**Why two vision encoders.** `dima806` is fine-tuned to read expressions from
cropped, face-dominated images; nothing in its training rewarded learning
about hands, room settings, or group composition. Asking it to also encode
whole busy scenes would be relying on a narrow specialist outside its
competence. CLIP's image tower, trained on a large diverse image–caption
corpus, is suited to exactly the general scene context the global token is
for. It doesn't need to know about emotion — that inference happens in the
fusion transformer, which combines general context with expression-specific
face tokens and the text.

**Sizing principle: the ceiling is a constraint, not a target.** What decides
where parameters help is *how many times a component runs per turn*:
- The response LM and text encoder run **once** per turn — upsizing them is
  cheap in latency terms. The LM is where the most headroom is spent, since
  its output quality is directly visible and evaluable.
- The face encoder runs **once per detected face, per sampled frame** —
  potentially 15–25 forward passes in a turn. It is kept at ViT-Base
  deliberately; upsizing it has a multiplicative latency cost.
- The fusion transformer is small regardless of budget: its job is cheap
  fusion over already-good pretrained representations, not representation
  learning, and a larger from-scratch transformer is just harder to optimize
  on ~10K examples.

### 4.2 Vision pipeline (per sampled frame, ~3fps)

MELD's raw footage is uncontrolled TV video — confirmed by inspection (§5) to
range from clean single-subject shots to 7-person ensembles to dark,
partially-occluded multi-person scenes. No fixed heuristic (largest face,
most-central face) survives this variety, so the pipeline detects everything
and lets attention sort it out.

1. **Detect** faces on the full-resolution frame (YuNet, confidence ≥ 0.75).
2. **Encode every face crop** (with a ~20% margin, resized to 224²) through the
   face/expression encoder → one pooled 768-d token per face. (Its own
   7-way softmax is computed but unused at serving time — see §2.)
3. **Encode the full frame** through the frozen scene encoder,
   **unconditionally** — not only when zero faces are found. A face the
   detector misses is therefore not a total loss of signal; the scene token
   still carries posture, position, partial profile, and context. The frame
   is **letterboxed** to 224², not center-cropped: a 16:9 center-crop
   discards ~44% of the width, and faces at the frame edges are a verified
   edge case (§5).
4. **Track** faces frame-to-frame with IoU matching (tolerating one missed
   frame before a track drops), assigning a per-turn track ID so the fusion
   transformer gets an explicit "same person over time" signal. A **shot-cut
   detector** (HSV-histogram difference between consecutive sampled frames
   above a threshold) hard-resets all tracks, so a multi-camera cut that puts
   a different face in the same screen position doesn't get linked under one
   ID. Appearance-gated matching (rejecting a positional match whose face
   embedding clearly differs) is a stretch addition. Even a wrong track ID
   corrupts only the continuity signal — the per-frame content encoding is
   still correct for whichever face is actually in the crop — and since the
   target is one label per turn, the model doesn't depend on perfect
   continuity. Whether the track-ID embedding helps at all is an ablation
   (§7), not an assumption.
5. **Emit tokens.** Each sampled frame contributes its face tokens and one
   scene token to the turn's variable-length token set (§4.3), each carrying
   a modality-type embedding, a frame-position embedding (sampled-frame index
   within the turn, capped at 32), and — for face tokens — a track-ID
   embedding. A frame with 0, 1, or 5 faces (some false positives) is the
   expected input shape, not an edge case.

The 0.75 detector threshold was tuned against visible false positives during
exploration (§5). It trades some real low-confidence secondary detections
(blurred or partial background faces) for fewer false positives, and pairs
with the always-on scene token, which softly absorbs whatever context the
stricter detector no longer commits to.

### 4.3 Fusion and prediction

**Text input with dialogue context.** The text encoder sees the previous *k*
utterances of the same dialogue followed by the current one:

```
<s> u[t-k] </s> ... </s> u[t-1] </s></s> u[t] </s>
```

with the current utterance last, context truncated from the oldest end to fit
256 tokens, *k* = 4 by default and ablated at *k* = 0. On replay the context is
the CSV dialogue; on the live path it is the ASR transcripts of the previous
live turns. This is the single largest known lever on MELD — context-aware
text models sit several F1 points above per-utterance ones — and it costs no
parameters. It is also the natural behavior of a robot that remembers the
conversation.

Speaker names are **deliberately not used**, even though MELD provides them
and they are known to help slightly (the model would learn actor priors): the
live path has no names, and keeping the two paths identical is worth more
than that gain. Recorded as a trade-off in §11.

**Token set.** All text-encoder output tokens (up to 256) + all face tokens
+ all scene tokens across the turn's sampled frames (≤32 frames, so ≤ ~250
visual tokens) + one learned `[FUSE]` token. Face and scene features pass
through their modality projectors into d=768. Each token carries a
modality-type embedding (text / face / scene / fuse); visual tokens add the
frame-position and (face only) track-ID embeddings from §4.2.

**Fusion transformer — single-stream.** 2–4 pre-LN transformer encoder
layers, d=768, 8 heads, standard self-attention over the joint set. Every
text token attends to every visual token and vice versa in every layer — this
is full bidirectional crossover; a dual-stream (co-attention) design would
carry the same information flow at ~2× the parameters and with two readouts to
reconcile, and at this data scale the difference is noise. The `[FUSE]`
output feeds both heads.

**Modality dropout.** During training, with p=0.15 all visual tokens are
dropped, and with p=0.15 all text tokens are dropped (never both). This
makes the model robust to the verified no-face-detected case, and it means a
single checkpoint can produce text-only and vision-only predictions by
masking at inference — used in the demo's interpretability panel and as a
sanity check. (The reported ablation numbers in §7 come from separately
trained models; masked predictions are reported alongside, labeled as such.)

**Heads and loss.** Emotion (7-way) and sentiment (3-way) linear heads on
the `[FUSE]` output, trained jointly:

```
L = CE_emotion(w) + λ · CE_sentiment          λ ∈ [0.3, 0.5], tuned on dev
```

Emotion class weights `w ∝ (1/freq)^α` with α ∈ {0, 0.5, 1} tuned on dev —
train is 47.2% neutral down to 2.7% fear/disgust (§5), and weighting trades
weighted-F1 for macro-F1, so its strength is a hyperparameter, not a fixed
choice. Label smoothing 0.1 is an option.

**Metrics.** Weighted-F1 is primary (comparable to the published MELD
literature), macro-F1 secondary, plus per-class F1 and a confusion matrix.
Sentiment accuracy and F1 are reported as secondary outputs. The heads emit
full probability distributions, not just argmax — richer state for a
downstream consumer (§4.5).

### 4.4 Response generation

A separate small instruction-tuned LM, prompted (not fine-tuned), produces
the response. Discriminative prediction gets a trained head (sample-efficient,
fast, evaluable); generative text gets the LM (its actual strength). By the
time the LM runs, the cross-modal reasoning is already done — its job is a
short, in-character reactive line given an already-determined emotional
context.

**Prompt inputs:** a fixed persona system prompt (a small, friendly character
robot; reply in ≤2 sentences; react, don't summarize), the last *k*
utterances, the current utterance, the predicted emotion with its top-2
probabilities, the sentiment, and the **visual gloss**.

**Visual gloss** — human-readable cues derived from tensors already computed,
with no extra forward passes:
- (a) **CLIP zero-shot scene cues.** A fixed, hand-written bank of ~20–30
  scene descriptors ("one person", "a group of people", "a dimly lit room",
  "people sitting on a couch", "someone leaning in close", "an animated
  gesture", …) is embedded once with CLIP's text tower. At end-of-turn, the
  mean scene embedding over the turn is scored against the bank; the top 2
  above a margin become phrases.
- (b) **The vision-only reading of the faces** — the last provisional state
  (§2), i.e. the fusion model with text masked — as its top-2: "2 faces
  visible, reading neutral (47%) or anger (16%)".

The gloss is an input to the LM **only**. The classifier sees embeddings,
never these labels, so its accuracy isn't bottlenecked by a hand-written
vocabulary, and the LM's grounding stays inspectable.

**Serving.** `mlx-lm`, 4-bit weights, streaming tokens, `max_new_tokens`
≈ 40, any thinking mode off. Tokens are emitted into the event stream (§4.5)
as they decode, so the first-token and completion latencies in §2 are
measured directly.

**Evaluation.** A 50-sample hand-scored rubric on replay outputs: consistent
with the predicted tag; references a visual cue when one is present; ≤2
sentences; in character; no invented facts. Pass rate per criterion is
reported. An LLM-judge pass over the same samples is optional and secondary.

**Not fine-tuned, and why.** MELD's next utterance is the next line of
*Friends*, not a robot's response, so it's not a usable supervision target.
A LoRA on a synthesized response set is a possible later step; its parameters
would count toward the budget but are negligible.

### 4.5 Structured state schema

One inference core emits a JSON-lines event stream that both demo frontends
consume. Two event kinds:

```json
{"turn_id": "dia38_utt4", "phase": "provisional", "frame": 5,
 "faces_seen": 2,
 "provisional_expression": {"surprise": 0.51, "neutral": 0.30, "joy": 0.08,
                            "anger": 0.04, "sadness": 0.03, "fear": 0.02, "disgust": 0.02}}
```

```json
{"turn_id": "dia38_utt4", "phase": "final",
 "text": "You did WHAT?",
 "emotion": "surprise",
 "emotion_probs": {"surprise": 0.60, "neutral": 0.20, "joy": 0.12, "anger": 0.03,
                   "fear": 0.02, "sadness": 0.02, "disgust": 0.01},
 "sentiment": "negative",
 "sentiment_probs": {"negative": 0.55, "neutral": 0.35, "positive": 0.10},
 "faces_seen": 2,
 "visual_cues": ["two people", "dimly lit room", "faces: surprised, neutral"],
 "response": "Wait — say that again, slowly.",
 "latency_ms": {"state": 84, "first_token": 610, "done": 1820}}
```

The `final` event is emitted as soon as the state is ready with `response`
absent; response tokens follow as `{"turn_id": ..., "phase": "token",
"text": "..."}` events, and a closing `{"phase": "done", ...}` carries the
full response and the measured latencies.

## 5. Data — verified facts and loader rules

Verified directly against the real MELD.Raw archive, not from documentation.

**Layout.** Clips are pre-cut per utterance as `dia<D>_utt<U>.mp4` at
1280×720, ~24fps, under `train_splits/`, `dev_splits_complete/`, and
`output_repeated_splits_test/`.

**`(Dialogue_ID, Utterance_ID)` is not a unique key across splits.** IDs
restart at 0 in each split: 1,740 keys collide between train and test, 675
between train and dev, 670 between dev and test. A loader that indexes the
whole archive into one map silently pairs labels with another split's video.
**Loader rule: index each split's own directory only.** (The exploration
viewer in §12 originally indexed globally and must be fixed; the footage
findings below stand regardless of which split a frame came from, but any
per-clip claim made with it should be re-checked with the fixed viewer.)

**Resolution.** Within their own directories: train 9,989/9,989 rows resolve,
test 2,610/2,610, dev 1,108/1,109 — `dia110_utt7` has no file and is dropped.
The archive also contains a few unreferenced mp4s (dev: 1,112 files, test:
2,615); they are ignored.

**Clip boundaries — a small number of clips are mis-cut.** The CSV
`StartTime`/`EndTime` columns are never read: every clip is pre-cut, so the
loader takes duration from the container. But the official clips were cut
*from* those timestamps, so where a timestamp is wrong the clip is wrong
too. A scan of all 13,707 referenced files: median clip 2.46s, p95 7.9s,
p99 11.8s — and **37 clips exceed 15s**. Most of those are genuine long
lines (dev's six are 15–29s carrying 17–22 words each) that the 15s decode
cap simply truncates; a few are clearly mis-cut (`test/dia38_utt4` is 305s
of footage for the 7-word "Oh it's great, it's a role on";
`test/dia220_utt0` is 235s for "What's that smell?"). At the other end,
**404 clips are under 0.5s**, 63 of which carry ≥5 words that cannot fit in
1–2 frames. For the clearly mis-cut rows (well under 1%) the text and label
are correct but the vision signal is noise. Preprocessing caps decoding at
15s (a 305s clip costs 45 sampled frames, not 900), and the manifest records
every clip's container `duration_s` and `n_frames` so training can treat
implausible clips as vision-missing — the same condition modality dropout
(§4.3) already trains for. (An earlier revision of this document called `dia38_utt4` a 2.38s file
and then a "genuine outlier utterance"; both were wrong — the first was the
cross-split ID collision above catching out the check looking for it.)

**Decodability and format.** All 13,707 referenced files open and decode
except `train/dia125_utt3` (truncated container, "moov atom not found"),
which preprocessing records as `decode_failed` and training drops. 13,630
clips are 1280×720 at 23.98fps; 67 run at 25fps and 76 are 496×384 or
560×432 — letterboxing and the detector's per-frame input size handle both
without special-casing.

**Class imbalance (train).** neutral 47.2%, joy 17.4%, surprise 12.1%, anger
11.1%, sadness 6.8%, disgust 2.7%, fear 2.7%.

**Utterance length.** avg 7.9 words (min 1, max 69). Average clip 3.1s.

**Shot composition** is highly inconsistent (visually confirmed on real
frames): a clean single-subject medium shot; a 7-person Central Perk ensemble
where the speaker is not the most visually salient face; a 3-person
conversation with one participant facing away from camera entirely (no usable
face); a dark car interior with partial occlusion at both frame edges. The
face detector on these frames: found the speaker at 0.94 confidence in the
ensemble shot, correctly produced zero detections for the person facing away,
and produced one plausible false positive (a hand, 0.65) that the 0.75
threshold removes.

**Speakers.** `Speaker` is not a closed set (one-off/guest characters appear
alongside the main cast), which rules out identity enrollment. The same six
main actors dominate all three splits — an identity-leakage risk for any
fine-tuned face encoder (§11).

## 6. Training plan

Training is staged so that a strong, fully-ablated result exists before any
GPU money is spent, and so that no single run is long enough for a failure to
cost a day.

**Preprocessing pass (once, local CPU, multiprocess).** For every clip:
decode at 3fps → detect faces → track → save face crops (224², JPEG),
letterboxed frames (224²), and per-frame metadata (boxes, scores, track IDs,
cut flags). Roughly 90K sampled frames and an estimated ~135K face crops
(~1.5 faces/frame — to be measured), a few GB on disk. Estimated 30–60 min.

**Feature cache (once, local MPS).** Frozen face-encoder pooled features
(768-d) and softmax, and frozen CLIP scene features (512-d, the projected
embedding, so the same cached vector serves both fusion and the gloss).
~1.5KB per token → a few hundred MB. Estimated 45–60 min locally.

**Stage 1 — local, MPS.** Train from scratch: fusion transformer, projectors,
both heads. Fine-tune: top half of RoBERTa-base (text is cheap to keep in the
loop — ~10K short sequences per epoch). Vision features come from the cache,
so the vision encoders are not in the loop. Estimated 2–3 min/epoch, 10–15
epochs with early stopping on dev weighted-F1 → ~30 min per run. **Every
ablation in §7 is one such run**, which is what makes the full table
affordable. Goal: a benchmark that stands against the published text-only and
multimodal numbers before Stage 2 starts.

**Stage 2 — Modal GPU (A10G/A100), gated on Stage 1.** Only after Stage 1 has
produced a strong benchmark. Unfreeze the top 4 layers of the face ViT
(optionally: full RoBERTa fine-tune, or RoBERTa-large at +230M params for an
expected ~1–2 F1), initialize fusion and heads from the Stage 1 checkpoint,
lower learning rate. The face encoder is back in the loop, so each epoch
re-encodes all crops: estimated ~1 hour per run on an A100. Kept only if it
beats Stage 1 on dev. The inference path and parameter count are unchanged
(except by the RoBERTa-large option).

**Common.** AdamW, linear warmup + cosine decay, gradient clipping, fixed
seeds, dev-based model selection, test evaluated once per reported
configuration. PyTorch on MPS for local work; MPS op coverage is incomplete
and some ops fall back to CPU (`PYTORCH_ENABLE_MPS_FALLBACK=1`), so
acceleration is validated empirically, not assumed.

**Budget check.** The hard constraint is that training completes in under 24
hours. Stage 1 runs are minutes; Stage 2 runs are about an hour; the
preprocessing and cache passes are under two hours combined. The constraint is
met with a wide margin on either machine.

## 7. Evaluation and expectation calibration

**Published baselines (verified, not assumed).** On MELD, text alone is the
strongest single modality at roughly 58–66% weighted-F1 depending on the text
model; vision alone is markedly weaker, around 42%; multimodal fusion adds
about 1–2 points over the best single modality. Dialogue-context text models
sit at the top of the text range. Our numbers are reported against these
directly. The write-up will **not** imply that vision dominates or that fusion
produces a dramatic jump — the value case is a working, interpretable,
real-time multimodal architecture with a measurable (if modest) fusion gain
over a text-only ablation.

**Ablation table.** Each row is a separately trained Stage 1 model (3 seeds
where cheap; mean ± std reported):

| Model | What it isolates |
|---|---|
| Text-only, *k*=0 | Per-utterance text baseline |
| Text-only, *k*=4 | Value of dialogue context |
| Vision-only, zero-training | `dima806` head averaged over faces and frames — no training at all |
| Vision-only, trained | Fusion over face + scene tokens, no text |
| **Fusion (full)** | The submitted model |
| Fusion − scene token | Value of the global frame |
| Fusion − context (*k*=0) | Context's contribution inside fusion |
| Fusion − track-ID embedding | Stretch; also compared specifically on clips containing a detected cut |
| Stage 2 fusion | If run; vs. Stage 1 on the same test split |

Modality-masked predictions from the full model (§4.3) are reported alongside,
labeled as masked rather than retrained.

**Protocol.** All numbers come from **batch evaluation over cached features**
— not from the real-time replay path, which would take ~2.25 hours per pass
over the 2,610 test clips. Dev selects; test is evaluated once per reported
configuration. Per-class F1 and the confusion matrix are reported for the
submitted model; sentiment metrics alongside.

**Response quality** is evaluated per §4.4. **Latency and resources** per §9.

## 8. Demo interfaces

One inference core emits the event stream in §4.5; each demo is a thin
frontend on it. This satisfies the assignment's "one input travelling through
the system to both outputs" — the same turn produces the state event and the
streamed response.

1. **Replay (real-time).** ~10 curated test clips, stratified by emotion and
   including the hard cases from §5 (ensemble shot, face-away, dark car),
   played at real speed with frames fed incrementally. One window shows:
   video with face boxes and track IDs; the provisional expression bars
   moving during the clip; the ground-truth text arriving at clip end; the
   final emotion + sentiment distributions; the visual cues; the streamed
   response; and the measured latency per event.

   **Consistency check.** For these clips, the batch-eval path (§7) and the
   replay path must produce identical final predictions — asserted and
   reported, as a guard against train/serve skew.

2. **Live.** Webcam sampled at ~3fps + microphone. webrtcvad marks end of
   speech; Whisper-base transcribes the turn; the transcript enters the same
   core, with context = the previous live turns. ASR is a convenience for
   producing text from speech; the fusion model still only sees text +
   vision, so this is not a tri-modal system. **Push-to-talk is the fallback**
   if VAD segmentation proves unreliable.

3. **Optional stretch.** Replay a small curated set of clips from other TV
   shows as generalization evidence — faces never seen in MELD. Not a
   replacement for the live path. Sourcing clips from other copyrighted shows
   gets a one-line acknowledgment in the write-up.

## 9. Hardware and resource reporting

- Development and inference machine: 16GB M1 MacBook. Modal GPU (A10G/A100)
  for Stage 2 only.
- MELD.Raw extracted footprint: ~20GB (10.1GB compressed, deleted after
  verified extraction). Preprocessing outputs and feature caches: a few GB.

**Reported (measured, not estimated):**
- Preprocessing and feature-cache wall-clock time.
- Stage 1: per-epoch and per-run wall-clock, peak memory. Stage 2: GPU-hours.
- Inference, both paths, p50/p95: per-sampled-frame vision cost (must stay
  under the 333ms frame interval to keep up), and the three end-of-turn
  events in §2 (state, first token, done).
- Peak resident memory on the live path with all models loaded.
- Final parameter count table with the chosen LM.

## 10. External and generated components — ownership

Reused pretrained, not trained by us (licenses recorded from each source at
implementation):

| Component | Source | License |
|---|---|---|
| RoBERTa-base | HF `roberta-base` | MIT |
| Face/expression encoder | HF `dima806/facial_emotions_image_detection` | per model card |
| CLIP ViT-B/32 (image + text towers) | OpenAI CLIP | MIT |
| YuNet face detector | opencv/opencv_zoo | per repo |
| Whisper-base | OpenAI | MIT |
| webrtcvad | Google WebRTC | BSD |
| Response LM | chosen per §4.1 | permissive (Apache-2.0/MIT) required |

Trained by us:
- Fusion transformer (from scratch)
- Modality projectors (from scratch)
- Emotion and sentiment heads (from scratch)
- Top half of RoBERTa-base (fine-tuned, Stage 1)
- Top 4 layers of the face ViT (fine-tuned, Stage 2, if run)

## 11. Known limitations

- Vision is a genuinely weaker signal than text on this dataset (§7); the
  design does not claim facial expression dominates the prediction.
- Face detection + tracking reduces but does not eliminate missing the
  speaker's face; the scene token is a mitigation, not a guarantee.
- Tracking is approximate at ~3fps. Shot-cut reset handles hard cuts;
  appearance gating is a stretch; whether the track-ID embedding helps at all
  is an open empirical question (§7).
- The 0.75 detector threshold trades real secondary detections for fewer
  false positives — deliberate, not free.
- **Actor-identity leakage.** The same six actors dominate all splits. Frozen
  vision encoders (Stage 1) limit the model's ability to learn actor→emotion
  priors; Stage 2 unfreezing raises that risk, and only the other-show
  stretch test would expose it.
- **Mis-cut and truncated clips.** A few MELD clips are cut to the wrong
  span — two run 4–5 minutes for a one-line utterance, ~63 are too short to
  contain their words (§5) — and 37 clips longer than 15s are truncated by
  the decode cap. Labels and text are used for all of them; for the mis-cut
  ones the vision tokens are noise, which modality dropout tolerates but
  does not fix. The manifest's `duration_s`/`n_frames` let training mask
  them explicitly; whether that helps is a cheap ablation.
- Speaker names are unused for path parity (§4.3), forfeiting a known small
  gain on MELD.
- Live path: ASR errors propagate into the text signal; the dialogue context
  is ASR history; VAD end-of-turn adds latency and can mis-segment (hence
  push-to-talk as fallback).
- MPS op coverage is incomplete; some local training ops may run on CPU.
- The response LM is prompted, not fine-tuned; response quality is bounded by
  the base model's instruction-following. The visual-gloss vocabulary is a
  fixed hand-written bank.
- The §2 latency figures are targets; the measured p50/p95 in §9 are the
  claims.

## 12. Tooling delivered during design and exploration

- `scripts/extract_meld_raw.sh` — extracts MELD.Raw.tar.gz, including the
  nested per-split archives.
- `scripts/view_meld_clips.py` — interactive labeled-clip viewer (text /
  emotion / sentiment overlay, stratified-by-emotion sampling, optional
  `--faces` face-detection overlay). **Must be changed to index within the
  selected split's directory** (§5); its current global index can pair a
  label with another split's clip.
- `scripts/download_face_model.sh` — downloads the OpenCV YuNet detector.

These were built to ground the design in real data rather than documentation
assumptions, and remain useful for qualitative error analysis later
(visualizing predictions against ground truth on held-out clips).
