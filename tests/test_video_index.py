def test_indexes_files_by_dialogue_and_utterance_id(tmp_path):
    from meld_emotion.data.video_index import build_video_index
    (tmp_path / "dia0_utt0.mp4").touch()
    (tmp_path / "dia38_utt4.mp4").touch()
    index = build_video_index(tmp_path)
    assert index[(0, 0)] == tmp_path / "dia0_utt0.mp4"
    assert index[(38, 4)] == tmp_path / "dia38_utt4.mp4"


def test_skips_files_that_dont_match_the_naming_pattern(tmp_path):
    from meld_emotion.data.video_index import build_video_index
    (tmp_path / "dia0_utt0.mp4").touch()
    (tmp_path / "readme.txt").touch()
    index = build_video_index(tmp_path)
    assert len(index) == 1


def test_does_not_see_files_in_a_sibling_split_directory(tmp_path):
    # Regression test: a global index across splits previously let a colliding
    # (Dialogue_ID, Utterance_ID) from one split silently resolve to another
    # split's video file.
    from meld_emotion.data.video_index import build_video_index
    split_a = tmp_path / "split_a"
    split_b = tmp_path / "split_b"
    split_a.mkdir()
    split_b.mkdir()
    (split_a / "dia38_utt4.mp4").touch()
    (split_b / "dia38_utt4.mp4").touch()
    index_a = build_video_index(split_a)
    assert index_a[(38, 4)] == split_a / "dia38_utt4.mp4"
