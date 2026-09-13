# Batch vs. replay consistency check

**Result: PASS — 10/10 argmax matches, max |Δp| = 0.0176** (tolerance: argmax equality +
`max_abs_diff < 0.02`), batch path and replay path both run locally (MPS, fp32) from
`results/stage2_fusion_faces_only/seed1/best.pt`.

Run: `uv run python -m scripts.consistency_check --clips results/demo_clips.json`. Raw
per-clip output: `results/consistency_check.json`. The batch side is `train_stage2.py`'s
own `evaluate()` over the preprocessing pass's JPEG crops; the replay side is
`TurnProcessor` over the raw video. They share the checkpoint and nothing else — different
crop source, different face-encoder wrapper, batched-and-padded vs. one clip at a time.

| clip | argmax | max \|Δp\| | note |
|---|---|---|---|
| dia168_utt11 | anger / anger | 0.0008 | |
| dia153_utt4 | anger / anger | 0.0011 | near-tie with fear (0.33 vs 0.32) |
| dia96_utt11 | neutral / neutral | 0.0005 | |
| dia261_utt0 | surprise / surprise | 0.0001 | |
| dia167_utt6 | joy / joy | 0.0023 | 26 faces per frame (capped at 64 crops/clip) |
| dia65_utt3 | neutral / neutral | 0.0022 | |
| dia124_utt12 | joy / joy | 0.0176 | the one clip where the JPEG round-trip is visible |
| dia128_utt0 | neutral / neutral | 0.0000 | zero faces detected |
| dia125_utt22 | anger / anger | 0.0015 | |
| dia90_utt1 | surprise / surprise | 0.0004 | 16.7 s clip, capped at 15 s on both paths |

## Where the drift comes from (measured, 2026-09-13)

An earlier version of this check compared replay against the Modal `--eval-test` file
(`test_predictions.jsonl`, CUDA + bf16 autocast) and got 9/10 with max |Δp| 0.0254, which was
written up as "JPEG plus cross-hardware precision". Isolating the two by rerunning the replay
path with its crops JPEG-encoded in memory, and rerunning the batch code path locally in fp32:

| comparison | max \|Δp\| over the 10 clips |
|---|---|
| replay (in-memory crops) vs replay (JPEG round-tripped crops), same hardware | 0.0023 on 9 clips; 0.0176 on dia124_utt12 |
| local batch path (fp32) vs replay, same hardware | identical to the row above |
| CUDA-bf16 batch file vs local fp32 batch path, **same JPEG crops** | up to 0.0245; flips the dia153_utt4 near-tie |

So the JPEG asymmetry is real but small, and everything outside the tolerance was
bf16-on-CUDA vs fp32-on-MPS numerics — a property of the stored file, not of the serving
path. The check therefore computes its batch reference locally; `--predictions <file>`
still compares against a stored file when that is what you want to know.

## What this establishes

The demo's inference core (`inference/turn.py` → `predict.py`) reproduces the batch
evaluation the reported test numbers come from, on the same hardware, to within 0.02
probability on every curated clip including the hard cases (26 faces, zero faces, >15 s).
The residual limitation for the write-up: test metrics were computed in bf16 on an A10G,
and bf16 moves individual probabilities by up to ~0.025, enough to flip a genuine tie.
