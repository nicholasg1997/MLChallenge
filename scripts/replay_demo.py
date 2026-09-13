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


def run_one_clip(bundle, bank_embeddings, video_path: Path, row: dict, emitter: EventEmitter, *,
                 responder=None, show_window: bool = True) -> tuple[dict | None, dict | None, bool]:
    """Runs one curated clip through the inference core. `responder=None`
    skips the LM entirely (no `token` events; `done_event["response"] ==
    ""`) -- the correctness gates (consistency_check.py) only need
    `final_event["emotion_probs"]` and shouldn't have to load a multi-GB LM.
    Returns `(final_event, done_event, quit_requested)`; if the user presses
    `q` mid-clip (only possible when `show_window=True`), the frame-reading
    loop stops early and `end_turn`/the LM/the 2-second hold are skipped
    entirely -- `final_event`/`done_event` come back `None` and
    `quit_requested` is `True` so the caller can stop its own clip loop
    instead of playing the remaining clips."""
    turn_id = f"dia{row['dialogue_id']}_utt{row['utterance_id']}"
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    wanted = set(sample_frame_indices(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), fps))
    delay_ms = max(1, int(1000 / fps))

    tp = TurnProcessor(bundle, emitter)
    tp.start_turn(turn_id)
    frame_idx, last_frame = -1, None
    quit_requested = False
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
                quit_requested = True
                break
    cap.release()
    if quit_requested:
        return None, None, True

    # --- end of turn: the text arrives; everything below is what §2's targets measure ---
    stamps = LatencyStamps(turn_start=time.perf_counter())
    cues = scene_gloss(tp.mean_scene_embedding, bank_embeddings) + [face_gloss(tp.max_faces_seen, tp.last_provisional)]
    final = tp.end_turn(row["text"], row["context_prev"], visual_cues=cues)
    stamps.state_emitted = time.perf_counter()

    top2 = sorted(final["emotion_probs"].items(), key=lambda kv: kv[1], reverse=True)[:2]
    messages = build_prompt(row["context_prev"], row["text"], final["emotion"], top2, final["sentiment"], cues)
    response_text = ""
    if responder is not None:
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
    # messages[-1] is the user-turn content build_prompt built (persona + last 4 context
    # lines + current line + emotion/sentiment/cues) -- the real prompt sent to the LM,
    # recorded here so response_rubric.py can score against it instead of an approximation.
    done_event = emitter.done(turn_id, response_text, stamps.as_ms(), prompt=messages[-1]["content"])

    if show_window and last_frame is not None:
        shown = last_frame.copy()
        _draw_overlay(shown, tp, tp.last_provisional,
                      f"{final['emotion']}/{final['sentiment']} | {'; '.join(cues)} | {response_text}")
        cv2.imshow(WINDOW, shown)
        cv2.waitKey(2000)
    return final, done_event, False


def warm_up(bundle, video_path: Path) -> None:
    """MPS compiles kernels on first use: the first sampled frame of a cold
    process costs ~1 s instead of ~0.2 s, and the same is true of end_turn's
    RoBERTa forward over a real multi-token sequence (push_frame's dummy
    text encoding is only ever one token, a shape MPS has already compiled
    by the time end_turn runs for real). Push one real frame AND run one
    real end_turn through a throwaway TurnProcessor so both code paths are
    warm before the first curated clip is measured."""
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if ret:
        with open(REPO_ROOT / "results" / "warmup_events.jsonl", "w") as f:
            tp = TurnProcessor(bundle, EventEmitter(f))
            tp.start_turn("warmup")
            tp.push_frame(frame)
            tp.end_turn("warmup text", [])


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
    first = rows_by_clip[clip_ids[0]]
    warm_up(bundle, index[(first["dialogue_id"], first["utterance_id"])])

    with open(args.events, "a") as f:
        emitter = EventEmitter(f)
        for clip_id in clip_ids:
            row = rows_by_clip[clip_id]
            print(f"playing {clip_id}: \"{row['text']}\" -> true={row['emotion']}")
            final, done, quit_requested = run_one_clip(bundle, bank_embeddings, index[(row["dialogue_id"], row["utterance_id"])],
                                                        row, emitter, responder=responder, show_window=not args.no_window)
            if quit_requested:
                print("  q pressed -- stopping (remaining clips skipped)")
                break
            print(f"  predicted={final['emotion']} cues={final['visual_cues']} response={done['response']!r} "
                  f"latency_ms={done['latency_ms']}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
