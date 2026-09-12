import json

import pytest

from meld_emotion.data.labels import Utterance


def _fixture(tmp_path):
    pre = tmp_path / "pre"
    cache = tmp_path / "cache"
    utterances = [Utterance("dev", 0, i, "Joey", f"t{i}", "joy", "positive") for i in range(3)]
    index = {(0, 0): tmp_path / "dia0_utt0.mp4", (0, 1): tmp_path / "dia0_utt1.mp4"}  # (0, 2): no video

    ok_dir = pre / "dev" / "dia0_utt0"
    ok_dir.mkdir(parents=True)
    with open(ok_dir / "metadata.json", "w") as f:
        json.dump({"status": "ok", "fps": 24.0, "total_frames": 48, "n_shot_cuts": 1, "frames": [
            {"sample_index": 0, "shot_cut": False, "faces": [{"track_id": 0}, {"track_id": 1}]},
            {"sample_index": 1, "shot_cut": True, "faces": []},
        ]}, f)
    (cache / "dev").mkdir(parents=True)
    (cache / "dev" / "dia0_utt0.npz").write_bytes(b"")

    bad_dir = pre / "dev" / "dia0_utt1"
    bad_dir.mkdir(parents=True)
    with open(bad_dir / "metadata.json", "w") as f:
        json.dump({"status": "decode_failed", "n_shot_cuts": 0, "frames": []}, f)

    return utterances, index, pre, cache


def test_manifest_rows_carry_status_context_paths_and_counts(tmp_path):
    from meld_emotion.data.manifest import build_manifest
    utterances, index, pre, cache = _fixture(tmp_path)
    rows = build_manifest("dev", utterances, index, preprocessed_dir=pre, cache_dir=cache)
    assert [r["status"] for r in rows] == ["ok", "decode_failed", "missing_video"]
    assert rows[0]["feature_path"] == "dev/dia0_utt0.npz"
    assert rows[1]["feature_path"] is None
    assert (rows[0]["n_frames"], rows[0]["n_faces"], rows[0]["n_shot_cuts"]) == (2, 2, 1)
    assert rows[0]["duration_s"] == 2.0          # 48 frames @ 24fps, from the container
    assert rows[1]["duration_s"] == 0.0          # decode_failed: no fps to divide by
    assert rows[0]["context_prev"] == []
    assert rows[2]["context_prev"] == ["t0", "t1"]
    assert rows[2]["text"] == "t2" and rows[2]["emotion"] == "joy" and rows[2]["sentiment"] == "positive"


def test_manifest_distinguishes_not_cached_and_not_preprocessed(tmp_path):
    from meld_emotion.data.manifest import build_manifest
    utterances, index, pre, cache = _fixture(tmp_path)
    (cache / "dev" / "dia0_utt0.npz").unlink()
    index[(0, 2)] = tmp_path / "dia0_utt2.mp4"        # video exists but was never preprocessed
    rows = build_manifest("dev", utterances, index, preprocessed_dir=pre, cache_dir=cache)
    assert rows[0]["status"] == "not_cached"
    assert rows[2]["status"] == "not_preprocessed"


def test_manifest_stats(tmp_path):
    from meld_emotion.data.manifest import build_manifest, manifest_stats
    utterances, index, pre, cache = _fixture(tmp_path)
    stats = manifest_stats(build_manifest("dev", utterances, index, preprocessed_dir=pre, cache_dir=cache))
    assert stats["utterances"] == 3
    assert stats["status"] == {"ok": 1, "decode_failed": 1, "missing_video": 1}
    assert stats["faces_per_frame"] == pytest.approx(1.0)          # 2 faces over 2 frames
    assert stats["zero_face_clip_fraction"] == pytest.approx(0.0)
    assert stats["clips_with_a_cut_fraction"] == pytest.approx(1.0)
    assert stats["shot_cuts_per_frame"] == pytest.approx(0.5)
    assert stats["truncated_clips"] == 0


def test_write_and_read_manifest_round_trip(tmp_path):
    from meld_emotion.data.manifest import build_manifest, read_manifest, write_manifest
    utterances, index, pre, cache = _fixture(tmp_path)
    rows = build_manifest("dev", utterances, index, preprocessed_dir=pre, cache_dir=cache)
    out = tmp_path / "manifest.jsonl"
    write_manifest(rows, out)
    assert read_manifest(out) == rows
