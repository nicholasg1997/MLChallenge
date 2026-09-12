"""Per-split manifest: one JSON line per labeled utterance joining its labels
and dialogue context to its preprocessed metadata and cached features, plus
the pipeline statistics design doc §6 says will be measured, not assumed."""
import json
from collections import Counter
from pathlib import Path

from meld_emotion.data.cache import cache_path_for
from meld_emotion.data.labels import Utterance, context_window, group_by_dialogue
from meld_emotion.data.preprocess import MAX_DECODE_SECONDS, clip_dir_for

CONTEXT_MAX = 8  # previous utterances stored; training slices context_prev[-k:] for any k <= 8


def build_manifest(split: str, utterances: list[Utterance], index: dict[tuple[int, int], Path],
                   preprocessed_dir: Path, cache_dir: Path) -> list[dict]:
    by_dialogue = group_by_dialogue(utterances)
    rows = []
    for u in utterances:
        window = context_window(by_dialogue, u.dialogue_id, u.utterance_id, k=CONTEXT_MAX)
        clip_dir = clip_dir_for(preprocessed_dir, split, u.dialogue_id, u.utterance_id)
        npz_path = cache_path_for(cache_dir, split, clip_dir.name)
        row = {
            "split": split, "dialogue_id": u.dialogue_id, "utterance_id": u.utterance_id,
            "speaker": u.speaker, "text": u.text, "context_prev": window[:-1],
            "emotion": u.emotion, "sentiment": u.sentiment,
            "status": None, "feature_path": None, "duration_s": 0.0,
            "n_frames": 0, "n_faces": 0, "n_shot_cuts": 0,
        }
        meta_path = clip_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            fps = meta.get("fps") or 0.0
            # Container duration, not the CSV's. Rows far above MAX_DECODE_SECONDS or
            # far below their word count are mis-cut clips whose vision is unreliable.
            row["duration_s"] = round(meta.get("total_frames", 0) / fps, 2) if fps else 0.0
            row["n_frames"] = len(meta["frames"])
            row["n_faces"] = sum(len(frame["faces"]) for frame in meta["frames"])
            row["n_shot_cuts"] = meta.get("n_shot_cuts", 0)
            if meta["status"] != "ok":
                row["status"] = meta["status"]
            elif not npz_path.exists():
                row["status"] = "not_cached"
            else:
                row["status"] = "ok"
                row["feature_path"] = str(npz_path.relative_to(cache_dir))
        elif (u.dialogue_id, u.utterance_id) not in index:
            row["status"] = "missing_video"
        else:
            row["status"] = "not_preprocessed"
        rows.append(row)
    return rows


def write_manifest(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_manifest(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def manifest_stats(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["status"] == "ok"]
    n_frames = sum(r["n_frames"] for r in ok)
    n_faces = sum(r["n_faces"] for r in ok)
    n_cuts = sum(r["n_shot_cuts"] for r in ok)
    return {
        "utterances": len(rows),
        "status": dict(Counter(r["status"] for r in rows)),
        "faces_per_frame": n_faces / n_frames if n_frames else 0.0,
        "zero_face_clip_fraction": sum(1 for r in ok if r["n_faces"] == 0) / len(ok) if ok else 0.0,
        "clips_with_a_cut_fraction": sum(1 for r in ok if r["n_shot_cuts"] > 0) / len(ok) if ok else 0.0,
        "shot_cuts_per_frame": n_cuts / n_frames if n_frames else 0.0,
        "truncated_clips": sum(1 for r in ok if r["duration_s"] > MAX_DECODE_SECONDS),
    }
