#!/usr/bin/env python3
"""Batch-vs-replay consistency check (design doc §8.1): the replay path's
final prediction for a curated clip must match the batch-eval path's
prediction -- the guard against train/serve skew.

Both paths run on THIS machine, in fp32, from the same checkpoint: the
batch path is train_stage2.py's own evaluate() over the preprocessing
pass's JPEG crops (or train.py's over the cached features for a Stage 1
checkpoint); the replay path is TurnProcessor over the raw video. Tiny
numeric drift is expected -- the batch path saw JPEG-saved crops, replay
encodes in-memory crops -- so this asserts argmax equality and a small
probability tolerance, not bitwise equality.

`--predictions` compares against a stored test_predictions.jsonl instead
(e.g. the Modal --eval-test run). Expect larger drift there: those were
computed on CUDA under bf16 autocast, which alone moves probabilities by
up to ~0.025 and can flip a near-tied argmax (docs/results/consistency_check.md).

Usage:
    uv run python -m scripts.consistency_check --clips results/demo_clips.json
"""
import argparse
import functools
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from meld_emotion.config import FEATURE_CACHE_DIR, PREPROCESSED_DIR, REPO_ROOT, split_video_dir
from meld_emotion.data.labels import EMOTIONS
from meld_emotion.data.manifest import read_manifest
from meld_emotion.data.video_index import build_video_index
from meld_emotion.inference.events import EventEmitter
from meld_emotion.inference.gloss import build_clip_tokenizer, embed_prompt_bank
from meld_emotion.inference.loader import load_inference_bundle
from meld_emotion.inference.responder import Responder
from meld_emotion.training.config import TrainConfig
from meld_emotion.training.crops import MeldCropDataset, collate_crops
from meld_emotion.training.dataset import MeldFeatureDataset, collate, infer_feature_dims, load_ok_rows
from meld_emotion.training.model import FusionModel
from meld_emotion.training.text import build_text_encoder, build_tokenizer, encode_text
from meld_emotion.training.train import evaluate, resolve_device
from scripts.replay_demo import DEFAULT_CHECKPOINT, run_one_clip

MAX_ABS_DIFF = 0.02


def batch_predict_one(predictions_path: Path, clip_id: str) -> dict:
    with open(predictions_path) as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                if p["clip"] == clip_id:
                    return {"emotion_probs": {e: float(v) for e, v in zip(EMOTIONS, p["probs"])}}
    raise KeyError(f"{clip_id} not in {predictions_path}")


def local_batch_predictions(checkpoint_path: Path, rows: list[dict], device: str = "auto", *,
                            features_dir: Path = FEATURE_CACHE_DIR, preprocessed_dir: Path = PREPROCESSED_DIR,
                            batch_size: int = 4, text_encoder_factory=None, tokenizer_factory=None) -> dict[str, dict]:
    """The batch-eval path itself, on this machine, for just `rows`:
    clip id -> {"emotion_probs": {...}}. A Stage 2 checkpoint re-encodes the
    preprocessing JPEG crops through its fine-tuned ViT (train_stage2.py);
    a Stage 1 checkpoint reads the cached features (train.py)."""
    device = resolve_device(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = TrainConfig(**ckpt["config"])
    tokenizer = (tokenizer_factory or (lambda: build_tokenizer(config.text_model)))()
    encode_fn = functools.partial(encode_text, tokenizer, max_length=config.max_text_tokens)
    build_text = text_encoder_factory or (lambda: build_text_encoder(config.text_model, config.text_trainable_layers))
    text_encoder = build_text() if config.use_text else None
    _, scene_dim = infer_feature_dims(rows, features_dir)

    if "face_encoder_state" in ckpt:
        from meld_emotion.training.face_encoder import TrainableFaceEncoder
        from meld_emotion.training.stage2_model import Stage2Model
        face_encoder = TrainableFaceEncoder(trainable_layers=config.face_trainable_layers)
        face_encoder.model.load_state_dict(ckpt["face_encoder_state"])
        fusion = FusionModel(config, text_encoder, face_dim=face_encoder.feature_dim, scene_dim=scene_dim)
        fusion.load_state_dict(ckpt["model_state"])
        model = Stage2Model(fusion, face_encoder)
        dataset = MeldCropDataset(rows, preprocessed_dir, features_dir, config, encode_fn, train=False)
        collate_fn = functools.partial(collate_crops, pad_id=tokenizer.pad_token_id)
    else:
        model = FusionModel(config, text_encoder, face_dim=ckpt["face_dim"], scene_dim=scene_dim)
        model.load_state_dict(ckpt["model_state"])
        dataset = MeldFeatureDataset(rows, features_dir, config, encode_fn)
        collate_fn = functools.partial(collate, pad_id=tokenizer.pad_token_id)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    with torch.no_grad():
        ev = evaluate(model.to(device).eval(), loader, device)
    return {clip: {"emotion_probs": dict(zip(EMOTIONS, map(float, probs)))}
            for clip, probs in zip(ev["clips"], ev["emotion_probs"])}


def compare(batch_probs: dict, replay_probs: dict) -> dict:
    batch_argmax = max(batch_probs, key=batch_probs.get)
    replay_argmax = max(replay_probs, key=replay_probs.get)
    return {"argmax_match": batch_argmax == replay_argmax,
            "max_abs_diff": max(abs(batch_probs[e] - replay_probs[e]) for e in batch_probs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", type=Path, default=REPO_ROOT / "results" / "demo_clips.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--predictions", type=Path, default=None,
                        help="compare against this stored test_predictions.jsonl instead of re-running the "
                             "batch path locally (a CUDA/bf16 file will show precision drift, see module doc)")
    parser.add_argument("--model-repo", default=None,
                        help="optional -- this check only reads final_event['emotion_probs'], never the LM's "
                             "response, so it runs without loading any LM unless a repo is given")
    args = parser.parse_args()

    clip_ids = json.loads(args.clips.read_text())
    manifest = FEATURE_CACHE_DIR / "test" / "manifest.jsonl"
    rows_by_clip = {f"dia{r['dialogue_id']}_utt{r['utterance_id']}": r for r in read_manifest(manifest)}
    if args.predictions is not None:
        batch = {clip_id: batch_predict_one(args.predictions, clip_id) for clip_id in clip_ids}
        batch_source = str(args.predictions)
    else:
        ok_rows = [r for r in load_ok_rows(manifest) if f"dia{r['dialogue_id']}_utt{r['utterance_id']}" in clip_ids]
        batch = local_batch_predictions(args.checkpoint, ok_rows)
        batch_source = "local batch path (fp32)"

    bundle = load_inference_bundle(args.checkpoint)
    bank_embeddings = embed_prompt_bank(bundle.scene_encoder.model, build_clip_tokenizer(), bundle.device)
    responder = Responder(model_repo=args.model_repo) if args.model_repo else None
    index = build_video_index(split_video_dir("test"))

    results = []
    with open(REPO_ROOT / "results" / "consistency_events.jsonl", "w") as f:
        emitter = EventEmitter(f)
        for clip_id in clip_ids:
            row = rows_by_clip[clip_id]
            final_event, _, _ = run_one_clip(bundle, bank_embeddings, index[(row["dialogue_id"], row["utterance_id"])],
                                             row, emitter, responder=responder, show_window=False)
            results.append({"clip": clip_id, **compare(batch[clip_id]["emotion_probs"], final_event["emotion_probs"])})
    passed = all(r["argmax_match"] and r["max_abs_diff"] < MAX_ABS_DIFF for r in results)
    Path(REPO_ROOT / "results" / "consistency_check.json").write_text(
        json.dumps({"batch_source": batch_source, "device": bundle.device, "tolerance": MAX_ABS_DIFF,
                    "passed": passed, "clips": results}, indent=2))
    print(f"{'PASS' if passed else 'FAIL'} vs {batch_source}: {sum(r['argmax_match'] for r in results)}/{len(results)} "
          f"argmax matches; max |dp| = {max(r['max_abs_diff'] for r in results):.4f}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
