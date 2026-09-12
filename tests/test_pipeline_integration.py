"""One chained test across the full preprocess -> cache -> manifest contract.

Every per-stage test file hand-rolls its own fixture for the *previous*
stage's output (e.g. test_cache.py invents a metadata.json by hand rather than
deriving it from a real preprocess_clip() call), so a field rename in one
stage could silently break the next while every existing test still passes.
This test instead runs the real preprocess_clip -> build_clip_cache ->
build_manifest pipeline end to end, stubbing only the two external boundaries
(face detector, vision encoders) that every other test in this suite already
stubs for the same reason.

Note: `_write_synthetic_video` and `_OneFaceDetector` are imported from
test_preprocess.py rather than duplicated, so this test always exercises the
same synthetic-video fixture the preprocess unit tests use -- if that helper
changes shape, this test changes with it instead of silently drifting.
"""
from meld_emotion.data.cache import build_clip_cache, cache_path_for, save_clip_cache
from meld_emotion.data.labels import Utterance
from meld_emotion.data.manifest import build_manifest
from meld_emotion.data.preprocess import clip_dir_for, preprocess_clip

from test_cache import _StubFaceEncoder, _StubSceneEncoder
from test_preprocess import _OneFaceDetector, _write_synthetic_video


def test_full_pipeline_from_real_preprocess_output_through_cache_to_manifest(tmp_path):
    split, dialogue_id, utterance_id = "dev", 2, 5
    preprocessed_dir = tmp_path / "pre"
    cache_dir = tmp_path / "cache"

    video_path = tmp_path / f"dia{dialogue_id}_utt{utterance_id}.avi"
    _write_synthetic_video(video_path)

    # Stage 1: real preprocess_clip, stubbing only the face detector.
    meta = preprocess_clip(video_path, split, dialogue_id, utterance_id,
                           detector=_OneFaceDetector(), out_dir=preprocessed_dir)
    assert meta["status"] == "ok"
    clip_dir = clip_dir_for(preprocessed_dir, split, dialogue_id, utterance_id)
    assert (clip_dir / "metadata.json").exists()

    expected_n_frames = len(meta["frames"])
    expected_n_faces = sum(len(f["faces"]) for f in meta["frames"])
    assert expected_n_frames > 0 and expected_n_faces > 0

    # Stage 2: real build_clip_cache/save_clip_cache, stubbing only the encoders.
    cache = build_clip_cache(clip_dir, _StubFaceEncoder(), _StubSceneEncoder())
    assert cache is not None
    out_path = cache_path_for(cache_dir, split, clip_dir.name)
    save_clip_cache(cache, out_path)
    assert out_path.exists()

    # Stage 3: real build_manifest, no stubs at all.
    utterances = [Utterance(split, dialogue_id, utterance_id, "Joey", "hi", "joy", "positive")]
    index = {(dialogue_id, utterance_id): video_path}
    rows = build_manifest(split, utterances, index,
                          preprocessed_dir=preprocessed_dir, cache_dir=cache_dir)

    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "ok"
    assert row["feature_path"] == str(out_path.relative_to(cache_dir))
    assert row["n_frames"] == expected_n_frames
    assert row["n_faces"] == expected_n_faces
