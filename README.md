# Real-time text + vision emotion prototype (MELD)

A turn-based emotion system for the *Text + Vision* track: video frames are consumed
as they arrive, and when the person finishes a line the system emits a 7-way emotion
+ 3-way sentiment state and then streams a short in-character reply from a local LM.
Two frontends share one inference core — a **replay** over curated MELD test clips and
a **live** webcam/microphone demo — and everything runs on a 16 GB M1 MacBook Air
(**3.54B parameters** on the local inference path, ceiling 6B). Training used one
A10G on Modal for about 3.5 GPU-hours in total.

## Results

MELD test (2,610 utterances), weighted F1 over the 7 emotions; dev is mean ± std over seeds.
Full table: [docs/results/stage1_ablations.md](docs/results/stage1_ablations.md).

| Model | dev wF1 | test wF1 | test macro-F1 |
|---|---|---|---|
| text only, no context | 0.593 ± 0.005 | 0.607 | 0.438 |
| text only, 4 previous lines | 0.615 ± 0.004 | 0.622 | 0.452 |
| vision only (frozen face features + trained head) | 0.357 | — | — |
| text + faces, frozen face encoder (Stage 1) | 0.621 ± 0.007 | 0.625 | 0.458 |
| **text + faces, face ViT top-4 fine-tuned (Stage 2) — submitted** | **0.630 ± 0.004** | **0.638 ± 0.005** | **0.471** |

Per-class test F1 of the submitted model: neutral .77, joy .61, surprise .59, anger .52,
sadness .41, fear .28, disgust .18.

## Key findings

- **Dialogue context is the biggest single win (+2.2 dev / +1.5 test)**; the faces add
  another +1.6 test over the best text-only model, and the fine-tuning is where most of
  that comes from (+1.3 test over the same architecture with a frozen encoder).
- **The face channel only matters where the text is unsure** — and there it matters a lot.
  Binning the test set by the text-only model's confidence
  ([docs/results/where_fusion_helps.md](docs/results/where_fusion_helps.md)):

  | text confidence | n | text only | Stage 1 fusion | Stage 2 fusion |
  |---|---|---|---|---|
  | < 0.5 | 713 | 35.2% | 39.3% | **43.9%** |
  | 0.5 – 0.9 | 1652 | 68.4% | 66.9% | 68.6% |
  | ≥ 0.9 | 245 | 92.7% | 92.7% | 92.7% |

  When Stage 2 overrides the text model (19% of clips) it is right 203 times to the
  text model's 137; Stage 1's overrides were a coin flip (168 vs 175). Fine-tuning did
  not make the face features a better standalone expression classifier (the model's
  text-masked reading scores 0.29 either way) — it made the face *vote* trustworthy at
  the moments it is cast.
- **The full-frame scene token hurt, so it was dropped.** With only six actors and a
  handful of sets, CLIP frame embeddings let the model memorise scenes; removing them
  was the best Stage 1 variant. Track identity embeddings were likewise neutral.
- **The off-the-shelf face-expression head is unusable on MELD** (weighted F1 0.12; it
  reads "sadness" on a third of actors mid-word) but is the right thing to show on a
  webcam, where posed expressions are what it was trained on. The live demo uses it
  for the on-screen face reading only; the classifier never sees it.
- **Batch and streaming paths agree**: the replay's final prediction matches the batch
  evaluation on all 10 curated clips (max |Δp| 0.018, run on the same hardware —
  the bf16 Modal predictions differ by up to 0.025, enough to flip a genuine tie).
- **Prompt structure beat model size for the reply.** A 1.5B–3B instruct model reads
  a raw transcript as a script and narrates it in the third person; separating
  *earlier lines / the line said to you / a detected-state hint* fixed that. Qwen2.5-3B
  (4-bit, mlx) was chosen as the largest candidate meeting the latency targets on a
  warm, realistic-prompt benchmark.

## Measured latency (M1 MacBook Air, MPS + mlx)

Targets from the design doc were state ≤ 100 ms (replay) / ≤ 500 ms incl. ASR (live),
first token ≤ 1.0 s, reply done ≤ 2.5 s. Raw numbers: [docs/results/measured/](docs/results/measured/).

| | state | first token | reply done | vision / sampled frame |
|---|---|---|---|---|
| replay, p50 / p95 (10 clips) | 37 / 43 ms | 0.89 / 1.11 s | 1.5 / 1.8 s | 126 / 506 ms |
| live over a clip file (single turn) | 564 ms, of which ASR 521 ms | 1.48 s | 2.10 s | 164 ms |

The vision p95 is the curated 26-face classroom clip; a single webcam face costs
~100–160 ms per frame at 3 fps. Peak resident memory on the live path: 2.8 GB process
RSS + 2.8 GB mlx (LM + ASR).

## How it works

`[FUSE] + RoBERTa-base tokens (current line, 4 previous lines) + one token per detected face`
→ 2-layer fusion transformer → emotion and sentiment heads. Faces are detected by YuNet,
tracked across ~3 fps samples with a shot-cut reset, and encoded by a ViT-Base expression
model whose top 4 layers are fine-tuned in Stage 2. Modality dropout (p = 0.15) trains the
same model to answer from text or faces alone — that is what produces the *provisional*
state after every frame while the person is still talking. The reply LM (Qwen2.5-3B-Instruct,
4-bit) is prompted, not fine-tuned, and receives the transcript, the state and a short
visual gloss (CLIP zero-shot scene phrases + the face reading); it never gates the state.
Design: [docs/superpowers/specs/…design.md](docs/superpowers/specs/2026-09-11-multimodal-emotion-prototype-design.md).

## Running it

```bash
uv sync --extra live --extra mlx          # macOS; drop --extra mlx elsewhere (torch paths remain)
bash scripts/download_face_model.sh       # YuNet weights -> models/
bash scripts/download_checkpoint.sh       # the submitted model (886 MB) from the v1.0 release -> results/
uv run pytest                             # 181 offline tests
```

The remaining weights (RoBERTa, the face ViT, CLIP, Whisper, the reply LM; ~4 GB) are
pulled from Hugging Face on first use.

Data and training (MELD.Raw is ~10 GB; preprocessing is ~2.5 h on the M1):

```bash
bash scripts/extract_meld_raw.sh                       # expects data/meld/raw/MELD.Raw.tar.gz
for s in train dev test; do
  uv run python scripts/preprocess_meld.py --split $s   # decode @3 fps, detect/track/crop faces
  uv run python scripts/build_feature_cache.py --split $s
  uv run python scripts/build_manifest.py --split $s
done
uv run python scripts/train_meld.py --ablation fusion_faces_only --seed 0      # Stage 1 (or scripts/modal_train.py)
uv run python scripts/train_stage2.py --init-from results/fusion_faces_only/seed0/best.pt --eval-test   # or modal_stage2.py
```

Demos (the checkpoint from `download_checkpoint.sh`; the replay additionally needs the MELD
test videos and features from the pipeline above, the live demo needs nothing else):

```bash
uv run --extra mlx python scripts/select_lm.py                                   # picks the reply LM
uv run --extra mlx python scripts/replay_demo.py                                 # curated clips, one window
uv run python -m scripts.consistency_check                                       # batch == replay?
uv run --extra live --extra mlx python scripts/live_demo.py                      # webcam + mic
uv run --extra live --extra mlx python scripts/live_demo.py --source clip.mp4    # same loop over any video
```

Live demo keys: `q` quits; `--push-to-talk` makes space start/end a turn if VAD misfires
in the room; `--face-threshold` / `--face-ema` tune the on-screen face reading.

## What was left out, and known limits

- No audio modality and no text-to-speech (the reply is streamed as text).
- The face channel is a small vote by design of the data: in scripted comedy the face
  rarely contradicts the line, so the model was never taught to trust a face over words.
  On the webcam, a clearly negative sentence said with a smile still reads as negative.
- The provisional (faces-only) state is prior-heavy — neutral 0.4–0.6 almost always — so
  the live demo shows the expression head's reading instead; that head flickers while
  you are speaking and is accurate when you hold an expression.
- ~37 MELD clips are mis-cut (up to 305 s for a one-line utterance); decoding is capped at
  15 s. `train/dia125_utt3` is unreadable and dev `dia110_utt7` is missing.
- Test metrics come from a bf16 run on an A10G; fp32 on MPS moves probabilities by up
  to 0.025.

## Layout

`src/meld_emotion/` — `data/` (labels, video index, preprocessing, feature cache, manifest),
`vision/` (YuNet, tracker, encoders), `training/` (config, dataset, model, Stage 1 and
Stage 2 loops), `inference/` (loader, predictor, turn processor, events, gloss, responder,
ASR, VAD, live session, overlay). `scripts/` — the entry points above plus Modal runners.
`docs/` — the design spec, the four implementation plans, and measured results.
