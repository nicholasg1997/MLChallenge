"""Per-clip preprocessing: decode at ~3fps, detect + track faces, crop and
letterbox, persist crops and metadata to disk. Plus a multiprocess driver
for a whole split.

Frames are read sequentially and every `step`-th one kept. Seeking with
CAP_PROP_POS_FRAMES was measured 3x slower on MELD's H.264 clips and is not
guaranteed frame-accurate, so it is deliberately not used.
"""
import json
import multiprocessing as mp
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from meld_emotion.config import PREPROCESSED_DIR
from meld_emotion.vision.face_detector import build_face_detector, detect_faces
from meld_emotion.vision.tracker import SHOT_CUT_THRESHOLD, FaceTracker, shot_change_score

SAMPLE_FPS = 3.0
MAX_DECODE_SECONDS = 15.0  # guard only: MELD clips are pre-cut (design doc §5)
CROP_SIZE = 224
FACE_MARGIN = 0.2


# ---------- pure-logic pieces ----------

def sample_frame_indices(total_frames: int, fps: float, sample_fps: float = SAMPLE_FPS,
                         max_seconds: float = MAX_DECODE_SECONDS) -> list[int]:
    if fps <= 0 or total_frames <= 0:
        return []
    capped_frames = min(total_frames, int(fps * max_seconds))
    step = max(1, round(fps / sample_fps))
    return list(range(0, capped_frames, step))


def letterbox(frame_bgr, size: int = CROP_SIZE):
    """Resize to fit inside size x size, pad the rest with black. Never crops
    (design doc §4.2: faces at frame edges are a verified edge case)."""
    h, w = frame_bgr.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(frame_bgr, (new_w, new_h))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def crop_face(frame_bgr, box: tuple, margin: float = FACE_MARGIN, size: int = CROP_SIZE):
    h, w = frame_bgr.shape[:2]
    x, y, bw, bh = box[:4]
    mx, my = int(bw * margin), int(bh * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(w, x + bw + mx), min(h, y + bh + my)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size))


def clip_dir_for(out_dir: Path, split: str, dialogue_id: int, utterance_id: int) -> Path:
    return Path(out_dir) / split / f"dia{dialogue_id}_utt{utterance_id}"


# ---------- one clip ----------

def _write_metadata(clip_dir: Path, meta: dict) -> None:
    clip_dir.mkdir(parents=True, exist_ok=True)
    with open(clip_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def preprocess_clip(video_path: Path, split: str, dialogue_id: int, utterance_id: int,
                    detector=None, out_dir: Path = PREPROCESSED_DIR, overwrite: bool = False) -> dict:
    clip_dir = clip_dir_for(out_dir, split, dialogue_id, utterance_id)
    meta_path = clip_dir / "metadata.json"
    if meta_path.exists() and not overwrite:
        with open(meta_path) as f:
            return json.load(f)

    base = {"split": split, "dialogue_id": dialogue_id, "utterance_id": utterance_id,
            "video": Path(video_path).name}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        meta = {**base, "status": "decode_failed", "fps": 0.0, "total_frames": 0,
                "n_shot_cuts": 0, "frames": []}
        _write_metadata(clip_dir, meta)
        return meta

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    wanted = sample_frame_indices(total_frames, fps)
    wanted_set = set(wanted)
    last_wanted = wanted[-1] if wanted else -1

    if detector is None:
        detector = build_face_detector()
    tracker = FaceTracker()
    prev_frame = None
    frames_meta = []
    n_shot_cuts = 0

    clip_dir.mkdir(parents=True, exist_ok=True)
    frame_idx = -1
    while frame_idx < last_wanted:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx not in wanted_set:
            continue
        sample_i = len(frames_meta)

        change = shot_change_score(prev_frame, frame) if prev_frame is not None else 0.0
        shot_cut = change >= SHOT_CUT_THRESHOLD
        if shot_cut:
            tracker.reset()
            n_shot_cuts += 1
        prev_frame = frame

        boxes = detect_faces(detector, frame)
        track_ids = tracker.update(boxes)
        face_records = []
        for (x, y, w, h, score), track_id in zip(boxes, track_ids):
            crop = crop_face(frame, (x, y, w, h))
            if crop is None:
                continue
            face_name = f"frame{sample_i}_face{track_id}.jpg"
            cv2.imwrite(str(clip_dir / face_name), crop)
            face_records.append({"track_id": track_id, "box": [x, y, w, h],
                                 "score": score, "path": face_name})

        scene_name = f"frame{sample_i}_scene.jpg"
        cv2.imwrite(str(clip_dir / scene_name), letterbox(frame))

        frames_meta.append({"sample_index": sample_i, "source_frame_index": frame_idx,
                            "shot_change": change, "shot_cut": shot_cut,
                            "faces": face_records, "scene_path": scene_name})
    cap.release()

    meta = {**base, "status": "ok" if frames_meta else "no_frames", "fps": fps,
            "total_frames": total_frames, "n_shot_cuts": n_shot_cuts, "frames": frames_meta}
    _write_metadata(clip_dir, meta)
    return meta


# ---------- a whole split, optionally multiprocess ----------

_WORKER_DETECTOR = None


def _init_worker():
    global _WORKER_DETECTOR
    cv2.setNumThreads(1)  # one process per core already; avoid OpenCV thread oversubscription
    _WORKER_DETECTOR = build_face_detector()


def _process_one(job: tuple) -> str:
    video_path, split, dialogue_id, utterance_id, out_dir, overwrite = job
    meta = preprocess_clip(Path(video_path), split, dialogue_id, utterance_id,
                           detector=_WORKER_DETECTOR, out_dir=Path(out_dir), overwrite=overwrite)
    return meta["status"]


def preprocess_split(split: str, utterances, index: dict[tuple[int, int], Path],
                     out_dir: Path = PREPROCESSED_DIR,
                     workers: int = 1, overwrite: bool = False, detector=None,
                     progress_every: int = 200) -> Counter:
    """Preprocess every utterance that has a video file. `detector` is only
    used when workers <= 1 (each pool worker builds its own)."""
    counts: Counter = Counter()
    jobs = []
    for u in utterances:
        path = index.get((u.dialogue_id, u.utterance_id))
        if path is None:
            counts["missing_video"] += 1
            continue
        jobs.append((str(path), split, u.dialogue_id, u.utterance_id, str(out_dir), overwrite))

    if workers <= 1:
        global _WORKER_DETECTOR
        _WORKER_DETECTOR = detector if detector is not None else build_face_detector()
        results = map(_process_one, jobs)
        pool = None
    else:
        pool = mp.get_context("spawn").Pool(workers, initializer=_init_worker)
        results = pool.imap_unordered(_process_one, jobs, chunksize=8)

    try:
        for i, status in enumerate(results, 1):
            counts[status] += 1
            if progress_every and i % progress_every == 0:
                print(f"  {i}/{len(jobs)} {dict(counts)}", flush=True)
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
    return counts
