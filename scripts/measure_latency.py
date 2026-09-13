#!/usr/bin/env python3
"""Measured (not estimated) latency (design doc §9): p50/p95 per
end-of-turn event from a completed replay run's event log, plus the
per-sampled-frame vision cost in isolation on the real model. Run
scripts/replay_demo.py first to produce results/replay_events.jsonl.

Usage:
    uv run python -m scripts.measure_latency
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np

from meld_emotion.config import REPO_ROOT, split_video_dir
from meld_emotion.data.preprocess import sample_frame_indices
from meld_emotion.data.video_index import build_video_index
from meld_emotion.inference.events import EventEmitter
from meld_emotion.inference.loader import load_inference_bundle
from meld_emotion.inference.turn import TurnProcessor
from scripts.replay_demo import DEFAULT_CHECKPOINT, warm_up


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
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

    if clip_ids:
        # MPS cold-start compiles kernels on first use -- warm both push_frame and end_turn
        # before the first measured frame (same warm_up used before the replay demo's own
        # curated run, scripts.replay_demo.warm_up).
        first_dia, first_utt = (int(p[3:]) for p in clip_ids[0].split("_"))
        warm_up(bundle, index[(first_dia, first_utt)])

    per_frame = []
    with open(REPO_ROOT / "results" / "latency_frame_events.jsonl", "w") as f:
        tp = TurnProcessor(bundle, EventEmitter(f))
        for clip_id in clip_ids:
            dia, utt = (int(p[3:]) for p in clip_id.split("_"))
            cap = cv2.VideoCapture(str(index[(dia, utt)]))
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            # sample_frame_indices, not a re-derived step: it also applies the 15 s decode
            # cap (preprocess.MAX_DECODE_SECONDS), which a manual `i % step` loop misses.
            wanted = set(sample_frame_indices(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), fps))
            tp.start_turn(clip_id)
            i = -1
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                i += 1
                if i in wanted:
                    per_frame.append(time_one_frame(tp, frame))
            cap.release()
    p50, p95 = percentile(per_frame, 50), percentile(per_frame, 95)
    summary["vision_per_frame_p50_ms"] = p50 * 1000 if p50 is not None else None
    summary["vision_per_frame_p95_ms"] = p95 * 1000 if p95 is not None else None
    summary["n_turns"], summary["n_frames"] = len(done_events), len(per_frame)
    for key, value in summary.items():
        print(f"{key}: {value}")
    Path(REPO_ROOT / "results" / "latency.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
