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
