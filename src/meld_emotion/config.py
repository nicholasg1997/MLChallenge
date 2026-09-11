"""Shared paths and constants for the MELD data/feature pipeline."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data" / "meld"
LABELS_DIR = DATA_DIR / "labels"
RAW_EXTRACTED_DIR = DATA_DIR / "raw" / "extracted"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
FEATURE_CACHE_DIR = DATA_DIR / "features"

MODELS_DIR = REPO_ROOT / "models"
FACE_DETECTOR_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"

# (Dialogue_ID, Utterance_ID) is NOT unique across MELD's splits -- IDs
# restart at 0 in each split. Every video lookup must be scoped to one
# split's own directory; never build a single index spanning all three.
SPLIT_DIRS = {
    "train": "MELD.Raw/train_splits",
    "dev": "MELD.Raw/dev_splits_complete",
    "test": "MELD.Raw/output_repeated_splits_test",
}
SPLITS = tuple(SPLIT_DIRS.keys())


def split_video_dir(split: str) -> Path:
    if split not in SPLIT_DIRS:
        raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")
    return RAW_EXTRACTED_DIR / SPLIT_DIRS[split]
