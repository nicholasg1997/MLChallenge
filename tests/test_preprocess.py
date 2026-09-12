import cv2
import numpy as np
import pytest


# ---------- pure-logic pieces ----------

def test_samples_at_approximately_3fps():
    from meld_emotion.data.preprocess import sample_frame_indices
    indices = sample_frame_indices(total_frames=72, fps=24.0, sample_fps=3.0, max_seconds=15.0)
    assert indices == [0, 8, 16, 24, 32, 40, 48, 56, 64]


def test_caps_long_clips_at_max_seconds():
    from meld_emotion.data.preprocess import sample_frame_indices
    indices = sample_frame_indices(total_frames=7200, fps=24.0, sample_fps=3.0, max_seconds=15.0)
    assert len(indices) == 45
    assert max(indices) < 24 * 15


def test_returns_empty_for_zero_fps():
    from meld_emotion.data.preprocess import sample_frame_indices
    assert sample_frame_indices(total_frames=100, fps=0.0) == []


def test_letterbox_pads_to_a_square_of_the_requested_size():
    from meld_emotion.data.preprocess import letterbox
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    out = letterbox(frame, size=224)
    assert out.shape == (224, 224, 3)


def test_crop_face_returns_a_square_crop_of_the_requested_size():
    from meld_emotion.data.preprocess import crop_face
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    crop = crop_face(frame, box=(100, 100, 50, 60), size=224)
    assert crop.shape == (224, 224, 3)


def test_crop_face_returns_none_for_a_box_entirely_outside_the_frame():
    from meld_emotion.data.preprocess import crop_face
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = crop_face(frame, box=(500, 500, 10, 10), size=224)
    assert crop is None


# ---------- whole-clip pipeline on synthetic video ----------

def _write_synthetic_video(path, n_frames=24, fps=24.0, cut_at=None, size=(320, 240)):
    """MJPG AVI of solid-colour frames: red, switching to green at frame `cut_at`."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(n_frames):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        channel = 1 if (cut_at is not None and i >= cut_at) else 2  # BGR: 2=red, 1=green
        frame[:, :, channel] = 255
        writer.write(frame)
    writer.release()


class _OneFaceDetector:
    """Stub with cv2.FaceDetectorYN's interface: always one face at a fixed box."""
    def setInputSize(self, size):
        pass

    def detect(self, frame):
        row = [40.0, 30.0, 60.0, 60.0] + [0.0] * 10 + [0.95]
        return (1, np.array([row], dtype=np.float32))


def test_preprocess_clip_writes_crops_scene_frames_and_metadata(tmp_path):
    from meld_emotion.data.preprocess import preprocess_clip
    video = tmp_path / "dia0_utt0.avi"
    _write_synthetic_video(video)
    meta = preprocess_clip(video, "dev", 0, 0, detector=_OneFaceDetector(), out_dir=tmp_path / "pre")
    assert meta["status"] == "ok"
    assert [f["sample_index"] for f in meta["frames"]] == [0, 1, 2]          # 24 frames @ 24fps, step 8
    assert [f["source_frame_index"] for f in meta["frames"]] == [0, 8, 16]
    assert all(len(f["faces"]) == 1 for f in meta["frames"])
    clip_dir = tmp_path / "pre" / "dev" / "dia0_utt0"
    assert (clip_dir / "metadata.json").exists()
    assert (clip_dir / "frame0_face0.jpg").exists()
    assert (clip_dir / "frame0_scene.jpg").exists()
    assert cv2.imread(str(clip_dir / "frame0_scene.jpg")).shape == (224, 224, 3)
    assert cv2.imread(str(clip_dir / "frame0_face0.jpg")).shape == (224, 224, 3)


def test_preprocess_clip_flags_a_shot_cut_and_restarts_track_ids(tmp_path):
    from meld_emotion.data.preprocess import preprocess_clip
    video = tmp_path / "dia0_utt1.avi"
    _write_synthetic_video(video, cut_at=12)
    meta = preprocess_clip(video, "dev", 0, 1, detector=_OneFaceDetector(), out_dir=tmp_path / "pre")
    assert [f["shot_cut"] for f in meta["frames"]] == [False, False, True]   # frames 0, 8 red; 16 green
    assert meta["frames"][2]["shot_change"] > 0.9
    assert meta["n_shot_cuts"] == 1
    track_ids = [f["faces"][0]["track_id"] for f in meta["frames"]]
    assert track_ids[0] == track_ids[1]
    assert track_ids[2] != track_ids[1]   # same box, but the cut reset the tracker -> new identity


def test_preprocess_clip_records_decode_failure_for_a_non_video_file(tmp_path):
    from meld_emotion.data.preprocess import preprocess_clip
    bogus = tmp_path / "dia0_utt2.mp4"
    bogus.write_text("not a video")
    meta = preprocess_clip(bogus, "dev", 0, 2, detector=_OneFaceDetector(), out_dir=tmp_path / "pre")
    assert meta["status"] == "decode_failed"
    assert meta["frames"] == []
    assert (tmp_path / "pre" / "dev" / "dia0_utt2" / "metadata.json").exists()


def test_preprocess_clip_resumes_from_existing_metadata_without_reprocessing(tmp_path):
    from meld_emotion.data.preprocess import preprocess_clip
    video = tmp_path / "dia0_utt3.avi"
    _write_synthetic_video(video)
    first = preprocess_clip(video, "dev", 0, 3, detector=_OneFaceDetector(), out_dir=tmp_path / "pre")

    class _Explodes:
        def setInputSize(self, size):
            raise AssertionError("detector must not run when metadata already exists")

        def detect(self, frame):
            raise AssertionError("detector must not run when metadata already exists")

    second = preprocess_clip(video, "dev", 0, 3, detector=_Explodes(), out_dir=tmp_path / "pre")
    assert second == first


def test_preprocess_split_counts_ok_and_missing_videos(tmp_path):
    from meld_emotion.data.labels import Utterance
    from meld_emotion.data.preprocess import preprocess_split
    v0 = tmp_path / "dia0_utt0.avi"
    v1 = tmp_path / "dia0_utt1.avi"
    _write_synthetic_video(v0)
    _write_synthetic_video(v1)
    index = {(0, 0): v0, (0, 1): v1}                       # (0, 2) has no video file
    utterances = [Utterance("dev", 0, i, "Joey", f"line {i}", "neutral", "neutral") for i in range(3)]
    counts = preprocess_split("dev", utterances, index, out_dir=tmp_path / "pre",
                              workers=1, detector=_OneFaceDetector())
    assert counts == {"ok": 2, "missing_video": 1}


def test_preprocess_clip_finds_the_ensemble_faces_in_a_real_clip(tmp_path):
    from meld_emotion.config import split_video_dir
    from meld_emotion.data.video_index import build_video_index
    from meld_emotion.data.preprocess import preprocess_clip

    dev_dir = split_video_dir("dev")
    if not dev_dir.exists():
        pytest.skip("requires the extracted MELD dataset (see scripts/extract_meld_raw.sh)")

    # dev dia1_utt1 was verified (with the per-split index) to be a Central Perk
    # ensemble shot with 6-7 faces per sampled frame.
    video_path = build_video_index(dev_dir)[(1, 1)]
    meta = preprocess_clip(video_path, split="dev", dialogue_id=1, utterance_id=1, out_dir=tmp_path)

    assert meta["status"] == "ok"
    assert len(meta["frames"]) >= 4
    assert max(len(f["faces"]) for f in meta["frames"]) >= 4
    assert (tmp_path / "dev" / "dia1_utt1" / "metadata.json").exists()
