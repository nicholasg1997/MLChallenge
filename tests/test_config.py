import pytest
from meld_emotion.config import SPLIT_DIRS, SPLITS, REPO_ROOT, split_video_dir


def test_split_dirs_cover_all_three_splits():
    assert set(SPLIT_DIRS.keys()) == {"train", "dev", "test"}


def test_splits_matches_split_dirs_keys():
    assert set(SPLITS) == set(SPLIT_DIRS.keys())


def test_split_video_dir_builds_expected_path():
    assert split_video_dir("dev") == (
        REPO_ROOT / "data" / "meld" / "raw" / "extracted" / "MELD.Raw" / "dev_splits_complete"
    )


def test_split_video_dir_rejects_unknown_split():
    with pytest.raises(ValueError):
        split_video_dir("bogus")
