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
