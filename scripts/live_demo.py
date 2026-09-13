#!/usr/bin/env python3
"""Live demo (design doc §8.1 #2): webcam at ~3 fps + microphone, VAD marks
end of speech, Whisper transcribes the turn, the transcript enters the
same inference core as the replay (context = the previous live turns), the
state event is published, and the response LM streams on a background
thread while the camera keeps running. No TTS. Push-to-talk (space) is
the fallback if VAD segmentation misbehaves in the room.

`--source <video file>` runs the identical loop over a file's frames and
audio track instead of the camera and mic -- the headless test, and the
way to replay clips from other shows.

Usage:
    uv run --extra live --extra mlx python scripts/live_demo.py
    uv run --extra live --extra mlx python scripts/live_demo.py --source path/to/clip.mp4 --no-window
    uv run --extra live python scripts/live_demo.py --no-lm --asr-backend torch   # no mlx at all

Keys: q quits; space toggles a push-to-talk turn when --push-to-talk is set.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from meld_emotion.config import REPO_ROOT
from meld_emotion.inference.asr import DEFAULT_ASR_MODEL, Transcriber
from meld_emotion.inference.audio import EndpointDetector, build_vad
from meld_emotion.inference.events import EventEmitter
from meld_emotion.inference.gloss import FACE_READING_THRESHOLD, build_clip_tokenizer, embed_prompt_bank
from meld_emotion.inference.latency import percentile, summarise_latencies
from meld_emotion.inference.live import LiveSession, warm_up
from meld_emotion.inference.loader import DEFAULT_CHECKPOINT, load_inference_bundle
from meld_emotion.inference.overlay import draw_overlay
from meld_emotion.inference.responder import Responder
from meld_emotion.inference.sources import CameraSource, FileSource
from meld_emotion.training.train import peak_memory_mb

WINDOW = "Live demo  (q=quit, space=push-to-talk)"
LM_SELECTION = REPO_ROOT / "results" / "lm_selection.json"


def default_model_repo() -> str | None:
    return json.loads(LM_SELECTION.read_text())["chosen"] if LM_SELECTION.exists() else None


def build_source(spec: str):
    return CameraSource(int(spec)) if spec.isdigit() else FileSource(Path(spec))


def write_report(path: Path, events_path: Path, session: LiveSession, *, device: str, model_repo: str | None,
                 asr_model: str, asr_backend: str, asr_params: int) -> dict:
    done = [e for e in (json.loads(l) for l in events_path.read_text().splitlines() if l.strip())
            if e.get("phase") == "done" and e["turn_id"].startswith("live")]
    report = summarise_latencies(done)
    report.update({
        "asr_p50_ms": (percentile(session.stats["asr_s"], 50) or 0) * 1000 if session.stats["asr_s"] else None,
        "asr_p95_ms": (percentile(session.stats["asr_s"], 95) or 0) * 1000 if session.stats["asr_s"] else None,
        "vision_per_frame_p50_ms": (percentile(session.stats["vision_s"], 50) or 0) * 1000 if session.stats["vision_s"] else None,
        "vision_per_frame_p95_ms": (percentile(session.stats["vision_s"], 95) or 0) * 1000 if session.stats["vision_s"] else None,
        "n_turns": len(done), "n_frames": len(session.stats["vision_s"]), "device": device,
        "model_repo": model_repo, "asr_model": asr_model, "asr_backend": asr_backend, "asr_params": asr_params,
        **peak_memory_mb(device)})
    try:                                   # the LM's Metal buffers live outside the process RSS
        import mlx.core as mx
        report["peak_mlx_mb"] = mx.get_peak_memory() / (1024 * 1024)
    except (ImportError, AttributeError):
        pass
    path.write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="0", help="camera index (default 0) or a video file path")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-repo", default=default_model_repo(),
                        help="response LM (default: results/lm_selection.json's 'chosen')")
    parser.add_argument("--no-lm", action="store_true", help="skip the response LM: state + transcript only")
    parser.add_argument("--asr-model", default=DEFAULT_ASR_MODEL)
    parser.add_argument("--asr-backend", default="auto", choices=["auto", "mlx", "torch"],
                        help="auto = mlx-whisper when installed (faster on Apple silicon), else transformers")
    parser.add_argument("--push-to-talk", action="store_true", help="space starts/ends a turn instead of VAD")
    parser.add_argument("--face-threshold", type=float, default=FACE_READING_THRESHOLD,
                        help="faces-only probability a non-neutral emotion needs before the face is read as it")
    parser.add_argument("--vad-aggressiveness", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--end-silence-ms", type=int, default=800)
    parser.add_argument("--events", type=Path, default=REPO_ROOT / "results" / "live_events.jsonl")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "results" / "live_run.json")
    parser.add_argument("--max-turns", type=int, default=None, help="stop after this many completed turns")
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()
    if not args.no_lm and not args.model_repo:
        parser.error("--model-repo is required (or run scripts/select_lm.py first, or pass --no-lm)")

    bundle = load_inference_bundle(args.checkpoint)
    bank_embeddings = embed_prompt_bank(bundle.scene_encoder.model, build_clip_tokenizer(), bundle.device)
    transcriber = Transcriber(args.asr_model, device=bundle.device, backend=args.asr_backend)
    responder = None if args.no_lm else Responder(model_repo=args.model_repo)
    endpoint = EndpointDetector(build_vad(args.vad_aggressiveness), end_silence_ms=args.end_silence_ms)

    args.events.parent.mkdir(parents=True, exist_ok=True)
    with open(args.events, "w") as f:
        emitter = EventEmitter(f)
        session = LiveSession(bundle, bank_embeddings, transcriber, endpoint, emitter, responder=responder,
                              push_to_talk=args.push_to_talk, face_threshold=args.face_threshold)
        warm_up(bundle, transcriber, responder)
        source = build_source(args.source)
        print(f"ready: source={args.source} device={bundle.device} lm={args.model_repo if responder else 'off'} "
              f"asr={args.asr_model} ({transcriber.backend}) mode={'push-to-talk' if args.push_to_talk else 'VAD'}")
        turns_done, frame = 0, None
        try:
            for frame, pcm_frames in source.frames():
                session.on_frame(frame, time.perf_counter())
                for pcm in pcm_frames:
                    if session.on_audio(pcm) is not None:
                        turns_done += 1
                        print(f"[{session.tp.turn_id}] {session.lines[0]}")
                session.drain_tokens()
                if args.max_turns and turns_done >= args.max_turns and session.status == session._idle_status():
                    break
                if not args.no_window:
                    shown = frame.copy()
                    draw_overlay(shown, session.tp.last_boxes, session.tp.last_track_ids, session.face_label(),
                                 session.lines, status=session.status)
                    cv2.imshow(WINDOW, shown)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord(" ") and args.push_to_talk and session.toggle_push_to_talk() is not None:
                        turns_done += 1
                        print(f"[{session.tp.turn_id}] {session.lines[0]}")
            # The source can end (file EOF, camera gone) while a response is still streaming.
            deadline = time.perf_counter() + 30
            while session.status == "responding" and time.perf_counter() < deadline:
                session.drain_tokens()
                if not args.no_window and frame is not None:
                    shown = frame.copy()
                    draw_overlay(shown, session.tp.last_boxes, session.tp.last_track_ids, session.face_label(),
                                 session.lines, status=session.status)
                    cv2.imshow(WINDOW, shown)
                    if cv2.waitKey(30) & 0xFF == ord("q"):
                        break
                else:
                    time.sleep(0.03)
        finally:
            source.close()
            session.close()
            session.drain_tokens()
            cv2.destroyAllWindows()
    report = write_report(args.report, args.events, session, device=bundle.device,
                          model_repo=None if args.no_lm else args.model_repo,
                          asr_model=args.asr_model, asr_backend=transcriber.backend, asr_params=transcriber.n_params)
    for line in session.lines:
        print(line)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
