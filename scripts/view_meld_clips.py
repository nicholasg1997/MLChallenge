"""Interactively watch MELD clips with the ground-truth utterance text, speaker,
emotion, and sentiment overlaid on the video -- for eyeballing what the raw data
actually looks like (framing, faces, audio) before committing to a model design.

Usage:
    python scripts/view_meld_clips.py --split dev --per-emotion 2
    python scripts/view_meld_clips.py --split train --emotion anger --per-emotion 5

Controls while a clip is playing: space = pause/resume, n = next clip, q = quit.

Pass --faces to also run a face detector (OpenCV YuNet) on every frame,
drawing a box per detected face plus a running count. Requires the model
file from scripts/download_face_model.sh.
"""
import argparse
import csv
import random
import sys
import textwrap

import cv2

from meld_emotion.config import (
    LABELS_DIR, RAW_EXTRACTED_DIR, FACE_DETECTOR_MODEL_PATH as FACE_MODEL_PATH, SPLIT_DIRS,
)

CONFIDENCE_THRESHOLD = 0.75  # Confidence threshold for facial recognition

# BGR (OpenCV order), not RGB.
EMOTION_COLORS = {
    "anger": (0, 0, 255),
    "disgust": (0, 128, 0),
    "fear": (128, 0, 128),
    "joy": (0, 215, 255),
    "neutral": (200, 200, 200),
    "sadness": (255, 80, 0),
    "surprise": (0, 255, 255),
}


def load_rows(split):
    path = LABELS_DIR / f"{split}_sent_emo.csv"
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_video_index(split_dir):
    """Map (dialogue_id, utterance_id) -> path by scanning ONE split's directory
    for dia<D>_utt<U>.* files. Must be scoped to a single split -- see the
    SPLIT_DIRS comment above for why a global index across splits is wrong.
    """
    index = {}
    for p in split_dir.glob("dia*_utt*.*"):
        try:
            dia_part, utt_part = p.stem.split("_")
            key = (int(dia_part.replace("dia", "")), int(utt_part.replace("utt", "")))
        except ValueError:
            continue
        index[key] = p
    return index


def sample_rows(rows, per_emotion, seed):
    rng = random.Random(seed)
    by_emotion = {}
    for r in rows:
        by_emotion.setdefault(r["Emotion"], []).append(r)
    sampled = []
    for group in by_emotion.values():
        sampled.extend(rng.sample(group, min(per_emotion, len(group))))
    rng.shuffle(sampled)
    return sampled


def build_face_detector():
    """Loads the OpenCV YuNet face detector (ONNX, runs on OpenCV's own DNN
    backend -- no GPU/graph-service dependency, unlike mediapipe's Tasks API,
    which crashes on this platform: see download_face_model.sh for why)."""
    return cv2.FaceDetectorYN.create(str(FACE_MODEL_PATH),
                                     "",
                                     (320, 320),
                                     score_threshold=CONFIDENCE_THRESHOLD)


def detect_faces(detector, frame_bgr):
    """Returns a list of (x, y, w, h, score) boxes in pixel coordinates."""
    h, w = frame_bgr.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame_bgr)
    if faces is None:
        return []
    return [(int(f[0]), int(f[1]), int(f[2]), int(f[3]), float(f[-1])) for f in faces]


def play_clip(path, row, detector=None):
    """Returns False if the user quit, True to continue to the next clip.

    detector, if given, turns on the face-box overlay.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  [skip] could not open {path}")
        return True

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    delay_ms = max(1, int(1000 / fps))
    emotion = row["Emotion"]
    color = EMOTION_COLORS.get(emotion, (255, 255, 255))
    header = f'{row["Speaker"]}  |  {emotion.upper()} / {row["Sentiment"]}'
    lines = textwrap.wrap(row["Utterance"], width=48) or [""]
    tag = f'dia{row["Dialogue_ID"]}_utt{row["Utterance_ID"]}'

    window = "MELD clip viewer  (space=pause, n=next, q=quit)"
    paused = False
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]

            if detector is not None:
                boxes = detect_faces(detector, frame)
                for (fx, fy, fw, fh, score) in boxes:
                    cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
                    cv2.putText(frame, f"{score:.2f}", (fx, max(0, fy - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, f"faces: {len(boxes)}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            bar_h = 40 + 28 * len(lines)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            cv2.putText(frame, header, (10, h - bar_h + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (10, h - bar_h + 52 + 28 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, tag, (w - 170, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.imshow(window, frame)

        key = cv2.waitKey(delay_ms if not paused else 50) & 0xFF
        if key == ord("q"):
            cap.release()
            return False
        if key == ord("n"):
            cap.release()
            return True
        if key == ord(" "):
            paused = not paused

    cap.release()
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    parser.add_argument("--per-emotion", type=int, default=2, help="clips to sample per emotion class")
    parser.add_argument("--emotion", help="only sample this emotion")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--faces", action="store_true",
                         help="overlay face detection boxes + a per-frame face count")
    args = parser.parse_args()

    split_dir = RAW_EXTRACTED_DIR / SPLIT_DIRS[args.split]
    if not split_dir.exists():
        sys.exit(f"Split directory not found: {split_dir}\n"
                  f"Run scripts/extract_meld_raw.sh first.")

    detector = None
    if args.faces:
        if not FACE_MODEL_PATH.exists():
            sys.exit(f"Face model not found: {FACE_MODEL_PATH}\n"
                      f"Run scripts/download_face_model.sh first.")
        print("Loading face detector...")
        detector = build_face_detector()

    rows = load_rows(args.split)
    if args.emotion:
        rows = [r for r in rows if r["Emotion"] == args.emotion]
        if not rows:
            sys.exit(f"No rows with emotion={args.emotion!r} in {args.split}")

    print(f"Indexing {args.split} video files...")
    index = build_video_index(split_dir)
    print(f"  found {len(index)} video files under {split_dir}")
    if not index:
        sys.exit("No video files matched the dia<D>_utt<U>.* pattern -- check the extracted layout.")

    sampled = sample_rows(rows, args.per_emotion, args.seed)
    print(f"Showing {len(sampled)} clips from {args.split} (space=pause, n=next, q=quit)")

    for row in sampled:
        key = (int(row["Dialogue_ID"]), int(row["Utterance_ID"]))
        path = index.get(key)
        if path is None:
            print(f"  [missing] dia{key[0]}_utt{key[1]} not found in extracted files")
            continue
        print(f'  dia{key[0]}_utt{key[1]}: "{row["Utterance"]}" -> {row["Emotion"]}')
        if not play_clip(path, row, detector=detector):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
