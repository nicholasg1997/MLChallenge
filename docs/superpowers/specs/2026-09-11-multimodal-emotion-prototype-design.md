# Real-Time Multimodal Emotion Prototype — Design

**Date:** 2026-09-11
**Track:** Text + Vision
**Timeline:** weekend-scale

## 1. Problem framing

We're building a small end-to-end prototype for emotion-aware interaction with a
character robot: combine an utterance's text with the speaker's facial
expression, produce a structured emotion tag plus a short grounded response,
and do it in a way that's usable as a live, ongoing interaction rather than a
one-shot batch classification.

MELD (Multimodal EmotionLines Dataset) is the shared foundation: ~13,700
utterances across 1,433 dialogues from *Friends*, each labeled with one of 7
emotions (anger, disgust, fear, joy, neutral, sadness, surprise) plus
sentiment, and each utterance packaged as a short video clip with synced
audio and a transcript line.

## 2. Real-time definition

A **turn** is one utterance: a few seconds of video arriving together with a
line of text (MELD's own unit — verified average clip length 3.1s, average
utterance length 7.9 words). "Real-time" means:

- Processing starts as soon as a turn's input starts arriving — frames are
  consumed incrementally as the clip/stream plays, not after the whole clip
  has finished.
- The system emits an emotion tag + response text within roughly **1-2
  seconds of the turn ending.**

This is deliberately not sub-100ms animation-loop timing — nothing at this
model scale on a laptop CPU/MPS setup could honestly promise that, and the
assignment doesn't require it. 1-2s matches the responsiveness of a
voice-assistant-style interaction, which is the right bar for "an utterance
comes in, the robot reacts."

## 3. Scope

**In scope:**
- Text + vision fusion producing a 7-way emotion tag (+ probability
  distribution) and a short grounded response.
- Two demo paths sharing one inference core (see §7).
- Full training/evaluation pipeline against MELD, with results reported
  against published baselines.
- This document's evidence, trade-offs, and limitations.

**Explicitly out of scope for this submission:**
- The tri-modal (+voice-as-a-fusion-input) extension and reinforcement
  learning — deliberately deferred per the assignment's own "core first"
  guidance. The fusion module's design (a variable-size token set with
  modality-type embeddings) would allow a third modality to be added later
  without a redesign, but it is not built or evaluated here.
- Identity-based face recognition (e.g., matching detected faces to named
  speakers) — MELD's speaker field is not a closed set (guest/one-off
  characters appear alongside the main cast), so identity enrollment
  doesn't generalize, and the detection+attention approach below doesn't
  need it.
- Audio-visual active speaker detection (lip-sync matching) — real signal,
  but a research problem in its own right; superseded by the
  detection+attention+tracking approach in §5.

## 4. Architecture

### 4.1 Components

| Component | Role | Params | Trained? |
|---|---|---|---|
| Text encoder (RoBERTa-base class) | Encode utterance text | ~125M | Fine-tuned (partial freeze) |
| Face/expression encoder (ViT-Base, e.g. dima806/facial_emotions_image_detection) | Encode each face crop *and* the global frame | ~86M | Fine-tuned (partial freeze) |
| Face detector (OpenCV YuNet) | Locate faces per sampled frame | ~1M | Not trained (zero-shot) |
| Face tracker | Assign per-clip track IDs across frames | 0 (pure geometry) | N/A |
| Fusion transformer (2-4 layers, cross-attention) | Let text and visual tokens attend to each other | ~10-25M | Trained from scratch |
| Emotion classifier head | 7-way softmax on fused representation | <1M | Trained from scratch |
| Response LM (small instruction-tuned model, ~3-4B) | Generate short grounded response text | ~3-4B | Prompted, not fine-tuned initially |
| ASR (Whisper-base, live demo only) | Speech-to-text for the live webcam path | ~74M | Not trained (zero-shot) |

**Estimated total: ~3.3-4.3B parameters**, depending on the final response-LM
size chosen after empirical latency testing — comfortably under the 6B
ceiling, with meaningful headroom remaining. ASR is included in this count
whenever the live demo path is active, since it's a required component of
that inference path per the assignment's counting rule; it is not part of
the replay-path count (MELD provides ground-truth text directly, no ASR
needed).

Sizing principle: **the ceiling is a constraint, not a target.** The
component that actually decides where extra parameters help is *how many
times it runs per turn*, not the raw budget headroom:
- The response LM and text encoder each run **once** per turn, so upsizing
  them is cheap in latency terms — the response LM is where we've
  deliberately spent the most headroom, since its output quality is directly
  visible and evaluable.
- The face/expression encoder runs **once per detected face, per sampled
  frame** — potentially 15-25 forward passes in a single turn. This is
  intentionally kept small; upsizing it would have an outsized, multiplicative
  latency cost for comparatively little benefit.
- The fusion transformer is trained from scratch and deliberately small
  regardless of budget: its job is cheap fusion over already-good pretrained
  representations, not representation learning, and a larger from-scratch
  transformer would just be harder to optimize on ~10K training examples.
- Quantization (4-bit/8-bit, e.g. via MLX) is the intended lever for getting
  more real capability out of the response LM within a given latency/memory
  budget, without changing its counted parameter total.

### 4.2 Vision pipeline (per sampled frame, ~3fps within a clip)

1. Run the face detector on the frame. MELD's raw footage is uncontrolled TV
   video — confirmed directly by inspection (see §6) to range from clean
   single-subject shots to 7-person ensembles to dark, partially-occluded
   multi-person scenes. No fixed heuristic (largest face, most-central face)
   survives this variety.
2. Encode **every** detected face crop through the face/expression encoder.
3. **Also** encode the full frame through the same encoder, unconditionally
   — not only as a fallback when zero faces are detected. This means a face
   the detector misses isn't a total loss of signal; the global embedding
   still carries some information (position, posture, partial profile) even
   when no clean crop exists for that person.
4. A lightweight frame-to-frame tracker (IoU/centroid matching, tolerant of
   one missed frame before dropping a track) assigns a per-clip track ID to
   each face. Every face token gets a track-ID embedding *and* a
   frame-position embedding, giving the fusion transformer an explicit
   "same person over time" signal rather than requiring it to infer
   continuity purely from visual similarity. This is intentionally simple —
   pure geometry, no pretrained model, no audio — and approximate: at ~3fps
   sampling, people move more between samples than in a dense stream, so
   track continuity is a soft inductive bias, not ground truth identity.
5. All tokens (text tokens + all face tokens across all sampled frames + one
   global-frame token per sampled frame) feed the fusion transformer as one
   variable-length set with modality-type, frame-position, and track-ID
   embeddings. The architecture is inherently permutation-invariant/set-based
   — a frame with 0 faces, 1 face, or 5 faces (some of them false positives)
   is the expected input shape, not an edge case requiring special handling.

Detector confidence is thresholded at 0.75 (tuned empirically against
visible false positives during data exploration — see §6). This trades away
some real, low-confidence secondary detections (partially visible or
blurred background faces) along with false positives, but pairs well with
the always-on global-frame token, which softly absorbs whatever context the
stricter face detector no longer commits to.

### 4.3 Output heads

- **Emotion tag**: a linear classifier head on the fused representation,
  trained with **class-weighted cross-entropy** (train-split distribution is
  47.2% neutral down to 2.7% fear/disgust — verified directly, see §6),
  evaluated on **macro-F1 and weighted-F1** against the held-out test split,
  not raw accuracy (which would be misleadingly inflated by the imbalance).
  This head outputs a full probability distribution over all 7 emotions, not
  just a top-1 label — richer structured state for a downstream consumer to
  react to.
- **Response text**: a separate, small pretrained instruction-tuned LM,
  prompted (not fine-tuned initially) with the utterance text, the
  classifier's predicted emotion label/distribution, and optionally a short
  textual gloss of salient visual cues. Kept deliberately separate from the
  classifier: discriminative prediction gets a trained head (sample-efficient,
  fast, evaluable), generative response text gets the LM (its actual
  strength). By the time the LM runs, the hard cross-modal reasoning is
  already done, so its job is simplified to producing a short, in-character
  reactive line given an already-determined emotional context.

## 5. Data pipeline & verified edge cases

Verified directly against the real MELD.Raw archive (not assumed from
documentation):

- Train/dev/test: 9,989 / 1,109 / 2,610 utterances across 1,038 / 114 / 280
  dialogues. **100% of CSV rows resolve to a real video file** (`dia<D>_utt<U>.mp4`,
  confirmed against the extracted archive).
- Class imbalance (train split): neutral 47.2%, joy 17.4%, surprise 12.1%,
  anger 11.1%, sadness 6.8%, disgust 2.7%, fear 2.7%.
- Utterance length: avg 7.9 words (min 1, max 69).
- Clip duration has a long, corrupted-looking tail: average 3.1s, but test
  split's max is **304.94 seconds** against a handful of near-zero-duration
  rows. The loader must sanity-check and cap the read window rather than
  trust `EndTime - StartTime` blindly, or a single bad row could hang frame
  sampling on a 5-minute decode for one utterance.
- Shot composition is highly inconsistent (visually confirmed on real
  frames): a clean single-subject medium shot, a 7-person Central Perk
  ensemble shot where the actual speaker is not the most visually salient
  face, a 3-person conversation where one participant faces away from
  camera entirely (no usable face), and a dark car-interior scene with
  partial occlusion at both frame edges.
- The face detector, tested against these exact frames: correctly found the
  true speaker at 0.94 confidence in the crowded ensemble shot; correctly
  produced zero detections for the person facing away from camera; produced
  one plausible false positive (a hand mistaken for a face at 0.65
  confidence, filtered out by the 0.75 threshold).
- `Speaker` is not a closed set (one-off/guest characters appear alongside
  the main cast), ruling out identity-enrollment approaches.

## 6. Training plan

- Fine-tune with most of each pretrained encoder frozen, training only the
  top layers/adapters plus the fusion transformer and classifier head from
  scratch. This is both a compute-saving choice and a generalization
  safeguard — fully fine-tuning 100M+ parameter encoders on ~10K training
  examples risks overfitting.
- PyTorch with the MPS backend (Apple M1 GPU) wherever supported. MPS op
  coverage isn't complete — some ops silently fall back to CPU or need
  `PYTORCH_ENABLE_MPS_FALLBACK=1` — so this will be validated empirically
  during implementation rather than assumed to accelerate everything.
- Local training only; no Modal needed given the actual trainable-parameter
  count (fusion transformer + heads + adapters, not full encoder backbones).

## 7. Evaluation & expectation calibration

Published MELD baselines (verified, not assumed): text-alone unimodal
performance is the strongest single modality at roughly 58-66% F1;
vision-alone is markedly weaker, around 42% F1; multimodal fusion adds only
about 1-2 points of F1 over the best single modality. We report our numbers
against these baselines directly, and the write-up will **not** imply the
vision track dominates or that fusion produces a dramatic accuracy jump —
the value case for this project is a working, interpretable, real-time
multimodal architecture with a measurable (if modest) fusion gain over a
text-only ablation, not beating text alone by a wide margin.

## 8. Demo interfaces

Two paths, sharing one inference core:

1. **Live**: webcam + live speech-to-text (Whisper-base), processed through
   the same face pipeline and fusion model, emitting a live tag + response.
   This is the interactive "character robot" demo the assignment's framing
   describes. ASR is a live-path convenience for producing the text input
   from speech — it does not make this a tri-modal system; the fusion model
   itself still only ever sees text + vision.
2. **Replay**: MELD test-split clips fed utterance-by-utterance, streamed
   incrementally rather than processed as a batch, for reproducible
   accuracy evaluation against ground truth.
3. **Optional stretch** (only if time remains after 1 and 2 are solid):
   replaying a small curated set of clips from other TV shows, as
   generalization evidence — does the system work on faces/voices never
   seen in MELD. Not a replacement for the live path. Sourcing clips from
   other copyrighted shows for a demo is a minor consideration worth a
   one-line acknowledgment in the final write-up.

## 9. Hardware & resource requirements

- Development machine: 16GB M1 MacBook.
- MELD.Raw extracted footprint: ~20GB disk (10.1GB compressed download,
  deleted after verified extraction).
- Estimated total inference-path parameters: ~3.3-4.3B (see §4.1), under the
  6B ceiling.
- Actual measured training wall-clock time and peak memory usage will be
  reported once implemented, not estimated in advance.

## 10. External/generated components — ownership

Reused pretrained, not trained by us:
- Text encoder backbone (RoBERTa-base class)
- Face/expression encoder backbone (ViT-Base, e.g. dima806/facial_emotions_image_detection)
- Face detector (OpenCV YuNet)
- ASR (Whisper-base)
- Response LM (small instruction-tuned model)

Trained by us, from scratch or fine-tuned:
- Cross-attention fusion transformer (from scratch)
- Emotion classifier head (from scratch)
- Top layers/adapters of the text and face encoders (fine-tuned)

## 11. Known limitations

- Vision is a genuinely weaker signal than text in this dataset (per
  published baselines, §7) — the design should not be read as claiming
  facial expression dominates the prediction.
- Face detection + tracking materially reduces but does not eliminate the
  risk of missing the speaker's face in a given frame; the global-frame
  token is a mitigation, not a guarantee, and mitigates rather than
  measures the residual risk.
- Face tracking is approximate at ~3fps sampling, not frame-perfect
  identity.
- The 0.75 confidence threshold trades some real secondary detections for
  fewer false positives (§4.2) — a deliberate, not free, choice.
- MPS acceleration coverage on Apple Silicon is incomplete; some training
  ops may run on CPU.
- The response LM is prompted, not fine-tuned, in this submission — response
  text quality is bounded by the base model's instruction-following, not
  adapted to this specific character-robot use case.

## 12. Tooling delivered during design/exploration

- `scripts/extract_meld_raw.sh` — extracts MELD.Raw.tar.gz (including nested
  per-split archives).
- `scripts/view_meld_clips.py` — interactive labeled-clip viewer (text/
  emotion/sentiment overlay, stratified-by-emotion sampling, optional
  `--faces` face-detection overlay with live box + count).
- `scripts/download_face_model.sh` — downloads the OpenCV YuNet face
  detector model.

These were built to ground this design in real data rather than
documentation assumptions, and remain useful for qualitative
error-analysis later in the project (e.g., visualizing model predictions
against ground truth on held-out clips).
