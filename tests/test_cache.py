import json

import numpy as np
from PIL import Image


class _StubFaceEncoder:
    feature_dim = 768

    def encode_batch(self, images):
        n = len(images)
        return {"features": np.full((n, 768), 0.5, np.float32),
                "probs": np.full((n, 7), 1 / 7, np.float32)}


class _StubSceneEncoder:
    feature_dim = 512

    def encode_batch(self, images):
        return np.full((len(images), 512), 0.25, np.float32)


def _make_fixture_clip(root, name="dia1_utt1", n_frames=2, faces_per_frame=1, status="ok"):
    clip_dir = root / "dev" / name
    clip_dir.mkdir(parents=True)
    frames = []
    for i in range(n_frames):
        faces = []
        for j in range(faces_per_frame):
            fname = f"frame{i}_face{j}.jpg"
            Image.new("RGB", (224, 224), (120, 80, 60)).save(clip_dir / fname)
            faces.append({"track_id": j, "box": [0, 0, 50, 50], "score": 0.9, "path": fname})
        sname = f"frame{i}_scene.jpg"
        Image.new("RGB", (224, 224), (30, 40, 50)).save(clip_dir / sname)
        frames.append({"sample_index": i, "source_frame_index": i * 8, "shot_change": 0.0,
                       "shot_cut": i == 1, "faces": faces, "scene_path": sname})
    dia, utt = (int(p[3:]) for p in name.split("_"))
    meta = {"split": "dev", "dialogue_id": dia, "utterance_id": utt, "video": f"{name}.mp4",
            "status": status, "fps": 24.0, "total_frames": n_frames * 8,
            "n_shot_cuts": 1 if n_frames > 1 else 0, "frames": frames if status == "ok" else []}
    with open(clip_dir / "metadata.json", "w") as f:
        json.dump(meta, f)
    return clip_dir


def test_build_clip_cache_produces_expected_shapes(tmp_path):
    from meld_emotion.data.cache import build_clip_cache
    clip_dir = _make_fixture_clip(tmp_path)
    cache = build_clip_cache(clip_dir, _StubFaceEncoder(), _StubSceneEncoder())
    assert cache["dialogue_id"] == 1 and cache["utterance_id"] == 1
    assert cache["face_features"].shape == (2, 768)
    assert cache["face_probs"].shape == (2, 7)
    assert cache["face_track_ids"].tolist() == [0, 0]
    assert cache["face_frame_idx"].tolist() == [0, 1]
    assert cache["scene_features"].shape == (2, 512)
    assert cache["scene_frame_idx"].tolist() == [0, 1]
    assert cache["shot_cut"].tolist() == [False, True]


def test_build_clip_cache_with_no_faces_keeps_correct_empty_widths(tmp_path):
    from meld_emotion.data.cache import build_clip_cache
    clip_dir = _make_fixture_clip(tmp_path, faces_per_frame=0)
    cache = build_clip_cache(clip_dir, _StubFaceEncoder(), _StubSceneEncoder())
    assert cache["face_features"].shape == (0, 768)
    assert cache["face_probs"].shape == (0, 7)
    assert cache["face_track_ids"].shape == (0,)
    assert cache["scene_features"].shape == (2, 512)


def test_build_clip_cache_returns_none_for_a_failed_clip(tmp_path):
    from meld_emotion.data.cache import build_clip_cache
    clip_dir = _make_fixture_clip(tmp_path, status="decode_failed")
    assert build_clip_cache(clip_dir, _StubFaceEncoder(), _StubSceneEncoder()) is None


def test_save_clip_cache_round_trips(tmp_path):
    from meld_emotion.data.cache import build_clip_cache, save_clip_cache
    clip_dir = _make_fixture_clip(tmp_path)
    cache = build_clip_cache(clip_dir, _StubFaceEncoder(), _StubSceneEncoder())
    out_path = tmp_path / "cache_out" / "dia1_utt1.npz"
    save_clip_cache(cache, out_path)
    loaded = np.load(out_path)
    assert loaded["face_features"].shape == (2, 768)
    assert loaded["scene_features"].dtype == np.float32
    assert int(loaded["dialogue_id"]) == 1
    assert loaded["shot_cut"].tolist() == [False, True]


def test_build_split_cache_caches_ok_clips_skips_failed_and_resumes(tmp_path):
    from meld_emotion.data.cache import build_split_cache
    preprocessed_dir = tmp_path / "preprocessed"
    cache_dir = tmp_path / "cache"
    _make_fixture_clip(preprocessed_dir, name="dia1_utt1")
    _make_fixture_clip(preprocessed_dir, name="dia1_utt2", status="decode_failed")
    first = build_split_cache("dev", _StubFaceEncoder(), _StubSceneEncoder(),
                              preprocessed_dir=preprocessed_dir, cache_dir=cache_dir)
    assert first == {"cached": 1, "skipped_not_ok": 1}
    assert (cache_dir / "dev" / "dia1_utt1.npz").exists()
    assert not (cache_dir / "dev" / "dia1_utt2.npz").exists()
    second = build_split_cache("dev", _StubFaceEncoder(), _StubSceneEncoder(),
                               preprocessed_dir=preprocessed_dir, cache_dir=cache_dir)
    assert second == {"skipped_existing": 1, "skipped_not_ok": 1}
