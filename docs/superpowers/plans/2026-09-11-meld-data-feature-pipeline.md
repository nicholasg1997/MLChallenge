# MELD Data & Feature Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the raw MELD.Raw archive into a verified, cached set of frozen
vision features (face-crop embeddings + scene embeddings) and dialogue-context
text strings for all three splits, ready for the training plan that follows.

**Architecture:** A small installable package (`src/meld_emotion/`) with four
layers: label/text loading, per-split video indexing, a vision pipeline
(detection → tracking → crop/letterbox), and a caching layer that runs frozen
pretrained encoders over the preprocessed images and persists the results as
`.npz` files. Each layer is unit-tested in isolation with synthetic fixtures;
the pipeline's end-to-end behavior is checked against real downloaded MELD
clips in a skippable integration test.

**Tech Stack:** Python 3.11, pytest, OpenCV (video decode, YuNet face
detection), NumPy, PyTorch + Hugging Face `transformers` (pretrained face and
CLIP scene encoders), Pillow.

## Global Constraints

- Everything runs on-device, no remote inference (design doc §2).
- Total inference-path parameters must stay under 6B (design doc §4.1) — not
  exercised by this plan directly, but no task here may introduce a new
  learned component without recording its parameter count for the running
  total in the next plan.
- Frames are sampled at ~3fps (design doc §2, §4.2).
- A 15-second decode cap applies per clip regardless of its labeled duration
  (design doc §5) — MELD has at least one genuine outlier-length utterance
  (`dia38_utt4`, ~305s).
- **`(Dialogue_ID, Utterance_ID)` is never a unique key across MELD splits** —
  IDs restart at 0 in each split (1,740 collide between train/test alone).
  Every index, cache, or lookup in this plan is scoped to one split's own
  directory; nothing may build a single index spanning multiple splits
  (design doc §5; this is the exact bug already found and fixed in
  `scripts/view_meld_clips.py`).
- `requires-python = ">=3.11"` (pyproject.toml).

---

## File Structure

```
src/meld_emotion/
  __init__.py
  config.py                  # shared paths, SPLIT_DIRS (Task 1)
  data/
    __init__.py
    labels.py                # Utterance loading + dialogue context window (Task 2)
    video_index.py           # per-split video file index (Task 3)
    preprocess.py            # decode/detect/track/crop pipeline (Task 6)
    cache.py                 # frozen feature cache builder (Task 8)
  vision/
    __init__.py
    face_detector.py         # YuNet wrapper (Task 4)
    tracker.py                # IoU tracker + shot-cut detector (Task 5)
    encoders.py                # face + scene pretrained encoders (Task 7)
scripts/
  view_meld_clips.py          # MODIFIED: use shared config/video_index/face_detector (Tasks 1, 3, 4)
  preprocess_meld.py           # NEW: CLI driver for Task 6
  build_feature_cache.py        # NEW: CLI driver for Task 8
tests/
  test_config.py
  test_labels.py
  test_video_index.py
  test_face_detector.py
  test_tracker.py
  test_preprocess.py
  test_encoders.py
  test_cache.py
```

---

### Task 1: Project scaffolding and shared config

**Files:**
- Create: `src/meld_emotion/__init__.py`
- Create: `src/meld_emotion/config.py`
- Modify: `pyproject.toml`
- Modify: `scripts/view_meld_clips.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `REPO_ROOT: Path`, `RAW_EXTRACTED_DIR: Path`, `LABELS_DIR: Path`,
  `PREPROCESSED_DIR: Path`, `FEATURE_CACHE_DIR: Path`,
  `FACE_DETECTOR_MODEL_PATH: Path`, `SPLIT_DIRS: dict[str, str]`,
  `SPLITS: tuple[str, ...]`, `split_video_dir(split: str) -> Path`.

- [ ] **Step 1: Add pytest and configure `src` on the test path**

Edit `pyproject.toml`:

```toml
[project]
name = "mlchallenge"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "opencv-python==5.0.0.93",
    "pytest>=8.0",
]

[tool.pytest.ini_options]
pythonpath = ["src"]
```

- [ ] **Step 2: Install and verify pytest runs (with nothing to collect yet)**

Run: `.venv/bin/python3 -m pip install -e . --no-build-isolation 2>/dev/null; .venv/bin/python3 -m pip install pytest>=8.0`
Then run: `.venv/bin/python3 -m pytest --collect-only`
Expected: `no tests ran` (no errors) — confirms pytest is importable.

- [ ] **Step 3: Write the failing test for the config module**

Create `tests/test_config.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it fails**

Run: `.venv/bin/python3 -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meld_emotion'`

- [ ] **Step 5: Create the package and config module**

Create `src/meld_emotion/__init__.py` (empty file).

Create `src/meld_emotion/config.py`:

```python
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
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/python3 -m pytest tests/test_config.py -v`
Expected: 4 passed

- [ ] **Step 7: Point `view_meld_clips.py` at the shared config (remove duplication)**

In `scripts/view_meld_clips.py`, replace the local `RAW_EXTRACTED_DIR`,
`FACE_MODEL_PATH`, and `SPLIT_DIRS` definitions with an import, so the split
directory mapping exists in exactly one place:

```python
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

from meld_emotion.config import RAW_EXTRACTED_DIR, FACE_DETECTOR_MODEL_PATH as FACE_MODEL_PATH, SPLIT_DIRS
```

Remove the old lines:
```python
REPO_ROOT = Path(__file__).resolve().parents[1]
LABELS_DIR = REPO_ROOT / "data" / "meld" / "labels"
RAW_EXTRACTED_DIR = REPO_ROOT / "data" / "meld" / "raw" / "extracted"
FACE_MODEL_PATH = REPO_ROOT / "models" / "face_detection_yunet_2023mar.onnx"
SPLIT_DIRS = { ... }
```

Keep `LABELS_DIR = REPO_ROOT / "data" / "meld" / "labels"` for now (it moves
to the new `labels.py` module in Task 2) — just change its `REPO_ROOT` to
`from meld_emotion.config import REPO_ROOT`.

- [ ] **Step 8: Verify the viewer script still parses and runs `--help` cleanly**

Run: `.venv/bin/python3 scripts/view_meld_clips.py --help`
Expected: the same usage text as before, no import errors.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml src/meld_emotion/__init__.py src/meld_emotion/config.py tests/test_config.py scripts/view_meld_clips.py
git commit -m "Add meld_emotion package with shared config; dedupe SPLIT_DIRS out of view_meld_clips.py"
```

---

### Task 2: Label loading and dialogue context window

**Files:**
- Create: `src/meld_emotion/data/__init__.py`
- Create: `src/meld_emotion/data/labels.py`
- Test: `tests/test_labels.py`

**Interfaces:**
- Consumes: nothing from Task 1 directly (uses its own `labels_dir` parameter
  rather than importing `LABELS_DIR`, so it stays independently testable).
- Produces: `Utterance` (frozen dataclass: `split, dialogue_id, utterance_id,
  speaker, text, emotion, sentiment`), `load_split(split: str, labels_dir:
  Path) -> list[Utterance]`, `group_by_dialogue(utterances: list[Utterance])
  -> dict[int, list[Utterance]]`, `context_window(by_dialogue: dict[int,
  list[Utterance]], dialogue_id: int, utterance_id: int, k: int) -> list[str]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_labels.py`:

```python
import csv

FIELDNAMES = ["Sr No.", "Utterance", "Speaker", "Emotion", "Sentiment",
              "Dialogue_ID", "Utterance_ID", "Season", "Episode", "StartTime", "EndTime"]


def _row(dia, utt, text, emotion="neutral", sentiment="neutral", speaker="Chandler"):
    return {"Sr No.": "1", "Utterance": text, "Speaker": speaker, "Emotion": emotion,
            "Sentiment": sentiment, "Dialogue_ID": str(dia), "Utterance_ID": str(utt),
            "Season": "1", "Episode": "1", "StartTime": "00:00:00,000", "EndTime": "00:00:01,000"}


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def test_load_split_parses_rows_into_utterances(tmp_path):
    from meld_emotion.data.labels import load_split
    _write_csv(tmp_path / "dev_sent_emo.csv", [
        _row(0, 0, "Hello there", emotion="joy"),
        _row(0, 1, "Not really", emotion="anger"),
    ])
    utterances = load_split("dev", labels_dir=tmp_path)
    assert len(utterances) == 2
    assert utterances[0].text == "Hello there"
    assert utterances[0].emotion == "joy"
    assert utterances[0].split == "dev"
    assert utterances[1].dialogue_id == 0
    assert utterances[1].utterance_id == 1


def test_context_window_returns_previous_k_plus_current(tmp_path):
    from meld_emotion.data.labels import load_split, group_by_dialogue, context_window
    _write_csv(tmp_path / "dev_sent_emo.csv", [
        _row(0, 0, "line zero"), _row(0, 1, "line one"),
        _row(0, 2, "line two"), _row(0, 3, "line three"),
    ])
    by_dialogue = group_by_dialogue(load_split("dev", labels_dir=tmp_path))
    assert context_window(by_dialogue, dialogue_id=0, utterance_id=3, k=2) == [
        "line one", "line two", "line three"
    ]


def test_context_window_clamps_at_dialogue_start(tmp_path):
    from meld_emotion.data.labels import load_split, group_by_dialogue, context_window
    _write_csv(tmp_path / "dev_sent_emo.csv", [_row(0, 0, "line zero"), _row(0, 1, "line one")])
    by_dialogue = group_by_dialogue(load_split("dev", labels_dir=tmp_path))
    assert context_window(by_dialogue, dialogue_id=0, utterance_id=1, k=4) == ["line zero", "line one"]


def test_context_window_k_zero_returns_only_current(tmp_path):
    from meld_emotion.data.labels import load_split, group_by_dialogue, context_window
    _write_csv(tmp_path / "dev_sent_emo.csv", [_row(0, 0, "line zero"), _row(0, 1, "line one")])
    by_dialogue = group_by_dialogue(load_split("dev", labels_dir=tmp_path))
    assert context_window(by_dialogue, dialogue_id=0, utterance_id=1, k=0) == ["line one"]


def test_group_by_dialogue_sorts_by_utterance_id(tmp_path):
    from meld_emotion.data.labels import load_split, group_by_dialogue
    _write_csv(tmp_path / "dev_sent_emo.csv", [_row(0, 2, "second"), _row(0, 0, "first"), _row(0, 1, "middle")])
    by_dialogue = group_by_dialogue(load_split("dev", labels_dir=tmp_path))
    assert [u.text for u in by_dialogue[0]] == ["first", "middle", "second"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python3 -m pytest tests/test_labels.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meld_emotion.data'`

- [ ] **Step 3: Implement**

Create `src/meld_emotion/data/__init__.py` (empty file).

Create `src/meld_emotion/data/labels.py`:

```python
"""Loading MELD's per-split label CSVs and building dialogue-context text."""
import csv
from dataclasses import dataclass
from pathlib import Path

EMOTIONS = ("neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear")
SENTIMENTS = ("neutral", "positive", "negative")


@dataclass(frozen=True)
class Utterance:
    split: str
    dialogue_id: int
    utterance_id: int
    speaker: str
    text: str
    emotion: str
    sentiment: str


def load_split(split: str, labels_dir: Path) -> list[Utterance]:
    path = labels_dir / f"{split}_sent_emo.csv"
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(Utterance(
                split=split,
                dialogue_id=int(row["Dialogue_ID"]),
                utterance_id=int(row["Utterance_ID"]),
                speaker=row["Speaker"],
                text=row["Utterance"],
                emotion=row["Emotion"],
                sentiment=row["Sentiment"],
            ))
    return out


def group_by_dialogue(utterances: list[Utterance]) -> dict[int, list[Utterance]]:
    by_dialogue: dict[int, list[Utterance]] = {}
    for u in utterances:
        by_dialogue.setdefault(u.dialogue_id, []).append(u)
    for group in by_dialogue.values():
        group.sort(key=lambda u: u.utterance_id)
    return by_dialogue


def context_window(by_dialogue: dict[int, list[Utterance]], dialogue_id: int,
                    utterance_id: int, k: int) -> list[str]:
    """Up to k previous utterance texts (oldest first) followed by the current one."""
    dialogue = by_dialogue[dialogue_id]
    idx = next(i for i, u in enumerate(dialogue) if u.utterance_id == utterance_id)
    start = max(0, idx - k)
    return [u.text for u in dialogue[start:idx + 1]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python3 -m pytest tests/test_labels.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/meld_emotion/data/__init__.py src/meld_emotion/data/labels.py tests/test_labels.py
git commit -m "Add MELD label loading and dialogue context window"
```

---

### Task 3: Per-split video index

**Files:**
- Create: `src/meld_emotion/data/video_index.py`
- Modify: `scripts/view_meld_clips.py`
- Test: `tests/test_video_index.py`

**Interfaces:**
- Produces: `build_video_index(split_dir: Path) -> dict[tuple[int, int], Path]`.

- [ ] **Step 1: Write the failing tests (including the collision regression)**

Create `tests/test_video_index.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python3 -m pytest tests/test_video_index.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meld_emotion.data.video_index'`

- [ ] **Step 3: Implement**

Create `src/meld_emotion/data/video_index.py`:

```python
"""Per-split video file indexing.

(Dialogue_ID, Utterance_ID) is NOT a unique key across MELD's splits -- IDs
restart at 0 in each split. build_video_index must always be called with ONE
split's own directory. Never merge indices from multiple splits.
"""
from pathlib import Path


def build_video_index(split_dir: Path) -> dict[tuple[int, int], Path]:
    index = {}
    for p in split_dir.glob("dia*_utt*.*"):
        try:
            dia_part, utt_part = p.stem.split("_")
            key = (int(dia_part.replace("dia", "")), int(utt_part.replace("utt", "")))
        except ValueError:
            continue
        index[key] = p
    return index
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python3 -m pytest tests/test_video_index.py -v`
Expected: 3 passed

- [ ] **Step 5: Point `view_meld_clips.py` at the shared implementation**

In `scripts/view_meld_clips.py`, remove the local `build_video_index`
function definition and import it instead:

```python
from meld_emotion.data.video_index import build_video_index
```

(This assumes Task 1 Step 7's `sys.path.insert` is already present above this
import.)

- [ ] **Step 6: Verify the viewer script still runs against real data**

Run: `.venv/bin/python3 scripts/view_meld_clips.py --split dev --per-emotion 1 --seed 0` and immediately press `q`.
Expected: prints `Indexing dev video files...`, `found 1112 video files under ...`, then one sampled-clip line per emotion, no traceback.

- [ ] **Step 7: Commit**

```bash
git add src/meld_emotion/data/video_index.py tests/test_video_index.py scripts/view_meld_clips.py
git commit -m "Extract per-split video index into shared module with a collision regression test"
```

---

### Task 4: Face detector wrapper

**Files:**
- Create: `src/meld_emotion/vision/__init__.py`
- Create: `src/meld_emotion/vision/face_detector.py`
- Modify: `scripts/view_meld_clips.py`
- Test: `tests/test_face_detector.py`

**Interfaces:**
- Consumes: `FACE_DETECTOR_MODEL_PATH` from `meld_emotion.config` (Task 1).
- Produces: `DEFAULT_CONFIDENCE_THRESHOLD: float`, `build_face_detector(score_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD)`,
  `detect_faces(detector, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int, float]]`
  (each tuple is `x, y, w, h, score`).

- [ ] **Step 1: Add numpy as an explicit dependency**

Edit `pyproject.toml` dependencies to add `"numpy>=1.26"` (already present
transitively via opencv-python, but the test suite imports it directly).

- [ ] **Step 2: Write the failing tests**

Create `tests/test_face_detector.py`. These test the wrapper's parsing logic
against a stub detector object, not the real YuNet model, so they run without
the downloaded model file:

```python
import numpy as np


class _StubDetector:
    """Mimics cv2.FaceDetectorYN's setInputSize/detect interface."""
    def __init__(self, faces):
        self._faces = faces

    def setInputSize(self, size):
        pass

    def detect(self, frame):
        return (1, self._faces)


def test_detect_faces_returns_empty_list_when_none_found():
    from meld_emotion.vision.face_detector import detect_faces
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert detect_faces(_StubDetector(None), frame) == []


def test_detect_faces_parses_box_and_score_from_yunet_row():
    from meld_emotion.vision.face_detector import detect_faces
    # YuNet rows are [x, y, w, h, <10 landmark coords>, score]
    row = [10.0, 20.0, 30.0, 40.0] + [0.0] * 10 + [0.87]
    faces = np.array([row], dtype=np.float32)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = detect_faces(_StubDetector(faces), frame)
    assert len(result) == 1
    x, y, w, h, score = result[0]
    assert (x, y, w, h) == (10, 20, 30, 40)
    assert score == pytest.approx(0.87, abs=1e-4)


def test_detect_faces_handles_multiple_rows():
    from meld_emotion.vision.face_detector import detect_faces
    row1 = [0.0, 0.0, 10.0, 10.0] + [0.0] * 10 + [0.9]
    row2 = [50.0, 50.0, 20.0, 20.0] + [0.0] * 10 + [0.6]
    faces = np.array([row1, row2], dtype=np.float32)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = detect_faces(_StubDetector(faces), frame)
    assert len(result) == 2
```

Add `import pytest` at the top of the file (needed for `pytest.approx`).

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python3 -m pytest tests/test_face_detector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meld_emotion.vision'`

- [ ] **Step 4: Implement**

Create `src/meld_emotion/vision/__init__.py` (empty file).

Create `src/meld_emotion/vision/face_detector.py`:

```python
"""OpenCV YuNet face detector wrapper.

Uses OpenCV's own DNN backend rather than mediapipe's Tasks API, which
crashes with a GPU graph-service error on this platform (see
scripts/download_face_model.sh for the reproduction).
"""
import cv2

from meld_emotion.config import FACE_DETECTOR_MODEL_PATH

DEFAULT_CONFIDENCE_THRESHOLD = 0.75


def build_face_detector(score_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
    return cv2.FaceDetectorYN.create(str(FACE_DETECTOR_MODEL_PATH), "", (320, 320),
                                      score_threshold=score_threshold)


def detect_faces(detector, frame_bgr) -> list[tuple[int, int, int, int, float]]:
    """Returns a list of (x, y, w, h, score) boxes in pixel coordinates."""
    h, w = frame_bgr.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame_bgr)
    if faces is None:
        return []
    return [(int(f[0]), int(f[1]), int(f[2]), int(f[3]), float(f[-1])) for f in faces]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python3 -m pytest tests/test_face_detector.py -v`
Expected: 3 passed

- [ ] **Step 6: Point `view_meld_clips.py` at the shared implementation**

In `scripts/view_meld_clips.py`, remove the local `build_face_detector` and
`detect_faces` function definitions and the local `CONFIDENCE_THRESHOLD`
constant, and import instead:

```python
from meld_emotion.vision.face_detector import build_face_detector, detect_faces
```

If the script's `--faces` flag needs a different default threshold than
`DEFAULT_CONFIDENCE_THRESHOLD`, pass it explicitly:
`build_face_detector(score_threshold=0.75)`.

- [ ] **Step 7: Verify the viewer script's face overlay still works against real data**

Run: `.venv/bin/python3 scripts/view_meld_clips.py --split dev --per-emotion 1 --seed 0 --faces` and press `q` after the first window appears.
Expected: no traceback, face boxes render as before.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/meld_emotion/vision/__init__.py src/meld_emotion/vision/face_detector.py tests/test_face_detector.py scripts/view_meld_clips.py
git commit -m "Extract face detector wrapper into shared module with stub-based unit tests"
```

---

### Task 5: Face tracker with shot-cut reset

**Files:**
- Create: `src/meld_emotion/vision/tracker.py`
- Test: `tests/test_tracker.py`

**Interfaces:**
- Produces: `iou(box_a: tuple, box_b: tuple) -> float`, class `FaceTracker`
  with `__init__(self, iou_threshold: float = 0.3, max_missed: int = 1)`,
  `.update(self, boxes: list[tuple]) -> list[int]` (one track ID per input
  box, same order), `.reset(self)`; function
  `is_shot_cut(prev_frame_bgr, curr_frame_bgr, threshold: float = 0.5) -> bool`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tracker.py`:

```python
import numpy as np
import pytest


def test_iou_of_identical_boxes_is_one():
    from meld_emotion.vision.tracker import iou
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    from meld_emotion.vision.tracker import iou
    assert iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0


def test_tracker_assigns_same_id_to_a_box_that_moves_slightly():
    from meld_emotion.vision.tracker import FaceTracker
    tracker = FaceTracker(iou_threshold=0.3)
    ids_frame1 = tracker.update([(10, 10, 50, 50)])
    ids_frame2 = tracker.update([(12, 11, 50, 50)])
    assert ids_frame1 == ids_frame2


def test_tracker_assigns_different_ids_to_far_apart_boxes_in_same_frame():
    from meld_emotion.vision.tracker import FaceTracker
    tracker = FaceTracker()
    ids = tracker.update([(0, 0, 20, 20), (200, 200, 20, 20)])
    assert ids[0] != ids[1]


def test_tracker_keeps_a_track_alive_through_one_missed_frame():
    from meld_emotion.vision.tracker import FaceTracker
    tracker = FaceTracker(max_missed=1)
    ids1 = tracker.update([(10, 10, 50, 50)])
    tracker.update([])  # missed detection
    ids3 = tracker.update([(11, 11, 50, 50)])
    assert ids1 == ids3


def test_tracker_drops_a_track_after_too_many_missed_frames():
    from meld_emotion.vision.tracker import FaceTracker
    tracker = FaceTracker(max_missed=1)
    ids1 = tracker.update([(10, 10, 50, 50)])
    tracker.update([])
    tracker.update([])  # second consecutive miss -- track should drop
    ids4 = tracker.update([(10, 10, 50, 50)])
    assert ids1 != ids4


def test_reset_forces_new_ids_even_for_a_box_in_the_same_position():
    from meld_emotion.vision.tracker import FaceTracker
    tracker = FaceTracker()
    ids1 = tracker.update([(10, 10, 50, 50)])
    tracker.reset()
    ids2 = tracker.update([(10, 10, 50, 50)])
    assert ids1 != ids2


def test_identical_frames_are_not_a_shot_cut():
    from meld_emotion.vision.tracker import is_shot_cut
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    assert is_shot_cut(frame, frame) is False


def test_very_different_frames_are_a_shot_cut():
    from meld_emotion.vision.tracker import is_shot_cut
    frame_a = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_a[:, :, 2] = 255  # solid red (BGR)
    frame_b = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_b[:, :, 1] = 255  # solid green
    assert is_shot_cut(frame_a, frame_b) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python3 -m pytest tests/test_tracker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meld_emotion.vision.tracker'`

- [ ] **Step 3: Implement**

Create `src/meld_emotion/vision/tracker.py`:

```python
"""Frame-to-frame face tracking: greedy IoU matching plus a shot-cut reset.

Pure geometry -- no pretrained model, no audio. Track continuity is a soft
inductive bias for the fusion model (design doc §4.2), not ground truth
identity, and is approximate at the ~3fps sampling rate this pipeline uses.
"""
from dataclasses import dataclass

import cv2


def iou(box_a: tuple, box_b: tuple) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_x1, inter_y1 = max(ax, bx), max(ay, by)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union_area = aw * ah + bw * bh - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


@dataclass
class _Track:
    track_id: int
    box: tuple
    missed_frames: int = 0


class FaceTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 1):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._tracks: list[_Track] = []
        self._next_id = 0

    def reset(self):
        """Drops all active tracks -- call this on a detected shot cut."""
        self._tracks = []

    def update(self, boxes: list[tuple]) -> list[int]:
        """boxes: list of (x, y, w, h[, score, ...]). Returns a track ID per
        input box, in the same order."""
        xywh_boxes = [tuple(b[:4]) for b in boxes]
        assigned: list = [None] * len(xywh_boxes)

        pairs = []
        for bi, box in enumerate(xywh_boxes):
            for ti, track in enumerate(self._tracks):
                score = iou(box, track.box)
                if score >= self.iou_threshold:
                    pairs.append((score, bi, ti))
        pairs.sort(reverse=True)

        used_boxes, used_tracks = set(), set()
        for score, bi, ti in pairs:
            if bi in used_boxes or ti in used_tracks:
                continue
            used_boxes.add(bi)
            used_tracks.add(ti)
            self._tracks[ti].box = xywh_boxes[bi]
            self._tracks[ti].missed_frames = 0
            assigned[bi] = self._tracks[ti].track_id

        for bi, box in enumerate(xywh_boxes):
            if assigned[bi] is None:
                track = _Track(track_id=self._next_id, box=box)
                self._next_id += 1
                self._tracks.append(track)
                assigned[bi] = track.track_id

        for ti, track in enumerate(self._tracks):
            if ti not in used_tracks:
                track.missed_frames += 1
        self._tracks = [t for t in self._tracks if t.missed_frames <= self.max_missed]

        return assigned


def is_shot_cut(prev_frame_bgr, curr_frame_bgr, threshold: float = 0.5) -> bool:
    """HSV-histogram Bhattacharyya distance between two frames; True means a
    hard camera cut (used to reset tracking so a different person appearing
    in the same screen position doesn't inherit the previous track ID)."""
    def hist(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(h, h)
        return h
    diff = cv2.compareHist(hist(prev_frame_bgr), hist(curr_frame_bgr), cv2.HISTCMP_BHATTACHARYYA)
    return diff >= threshold
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python3 -m pytest tests/test_tracker.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/meld_emotion/vision/tracker.py tests/test_tracker.py
git commit -m "Add IoU face tracker with shot-cut reset"
```

---

### Task 6: Preprocessing pipeline

**Files:**
- Create: `src/meld_emotion/data/preprocess.py`
- Create: `scripts/preprocess_meld.py`
- Test: `tests/test_preprocess.py`

**Interfaces:**
- Consumes: `build_face_detector`, `detect_faces` (Task 4); `FaceTracker`,
  `is_shot_cut` (Task 5); `build_video_index` (Task 3); `load_split`,
  `Utterance` (Task 2); `PREPROCESSED_DIR` (Task 1).
- Produces: `sample_frame_indices(total_frames: int, fps: float, sample_fps:
  float = 3.0, max_seconds: float = 15.0) -> list[int]`,
  `letterbox(frame_bgr, size: int = 224) -> np.ndarray`,
  `crop_face(frame_bgr, box: tuple, margin: float = 0.2, size: int = 224) ->
  np.ndarray | None`, `preprocess_clip(video_path: Path, split: str,
  dialogue_id: int, utterance_id: int, detector=None, out_dir: Path) ->
  dict` (writes `<out_dir>/<split>/dia<D>_utt<U>/` with face crop JPEGs,
  scene JPEGs, and `metadata.json`).

- [ ] **Step 1: Write the failing tests for the pure-logic pieces**

Create `tests/test_preprocess.py`:

```python
import numpy as np
import pytest


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python3 -m pytest tests/test_preprocess.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meld_emotion.data.preprocess'`

- [ ] **Step 3: Implement the pure-logic pieces and the clip pipeline**

Create `src/meld_emotion/data/preprocess.py`:

```python
"""Per-clip preprocessing: decode at ~3fps, detect + track faces, crop and
letterbox, persist crops and metadata to disk."""
import json
from pathlib import Path

import cv2
import numpy as np

from meld_emotion.config import PREPROCESSED_DIR
from meld_emotion.vision.face_detector import build_face_detector, detect_faces
from meld_emotion.vision.tracker import FaceTracker, is_shot_cut

SAMPLE_FPS = 3.0
MAX_DECODE_SECONDS = 15.0
CROP_SIZE = 224
FACE_MARGIN = 0.2


def sample_frame_indices(total_frames: int, fps: float, sample_fps: float = SAMPLE_FPS,
                          max_seconds: float = MAX_DECODE_SECONDS) -> list[int]:
    if fps <= 0 or total_frames <= 0:
        return []
    capped_frames = min(total_frames, int(fps * max_seconds))
    step = max(1, round(fps / sample_fps))
    return list(range(0, capped_frames, step))


def letterbox(frame_bgr, size: int = CROP_SIZE):
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
    x, y, bw, bh = box
    mx, my = int(bw * margin), int(bh * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(w, x + bw + mx), min(h, y + bh + my)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size))


def preprocess_clip(video_path: Path, split: str, dialogue_id: int, utterance_id: int,
                     detector=None, out_dir: Path = PREPROCESSED_DIR) -> dict:
    detector = detector or build_face_detector()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"split": split, "dialogue_id": dialogue_id, "utterance_id": utterance_id,
                "status": "decode_failed", "frames": []}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    indices = sample_frame_indices(total_frames, fps)

    clip_dir = out_dir / split / f"dia{dialogue_id}_utt{utterance_id}"
    clip_dir.mkdir(parents=True, exist_ok=True)

    tracker = FaceTracker()
    prev_frame = None
    frames_meta = []

    for sample_i, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        if prev_frame is not None and is_shot_cut(prev_frame, frame):
            tracker.reset()
        prev_frame = frame

        boxes = detect_faces(detector, frame)
        track_ids = tracker.update(boxes)

        face_records = []
        for (x, y, w, h, score), track_id in zip(boxes, track_ids):
            crop = crop_face(frame, (x, y, w, h))
            if crop is None:
                continue
            face_path = clip_dir / f"frame{sample_i}_face{track_id}.jpg"
            cv2.imwrite(str(face_path), crop)
            face_records.append({"track_id": track_id, "box": [x, y, w, h],
                                  "score": score, "path": face_path.name})

        scene_path = clip_dir / f"frame{sample_i}_scene.jpg"
        cv2.imwrite(str(scene_path), letterbox(frame))

        frames_meta.append({"sample_index": sample_i, "source_frame_index": frame_idx,
                             "faces": face_records, "scene_path": scene_path.name})

    cap.release()
    meta = {"split": split, "dialogue_id": dialogue_id, "utterance_id": utterance_id,
            "status": "ok", "fps": fps, "total_frames": total_frames, "frames": frames_meta}
    with open(clip_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python3 -m pytest tests/test_preprocess.py -v`
Expected: 6 passed

- [ ] **Step 5: Write the skippable end-to-end integration test against real data**

Append to `tests/test_preprocess.py`:

```python
def test_preprocess_clip_produces_metadata_and_crops_on_a_real_clip(tmp_path):
    from meld_emotion.config import split_video_dir
    from meld_emotion.data.video_index import build_video_index
    from meld_emotion.data.preprocess import preprocess_clip

    dev_dir = split_video_dir("dev")
    if not dev_dir.exists():
        pytest.skip("requires the extracted MELD dataset (see scripts/extract_meld_raw.sh)")

    index = build_video_index(dev_dir)
    video_path = index[(1, 1)]  # verified 7-8-face "joy" ensemble shot from data exploration
    meta = preprocess_clip(video_path, split="dev", dialogue_id=1, utterance_id=1, out_dir=tmp_path)

    assert meta["status"] == "ok"
    assert len(meta["frames"]) > 0
    assert any(len(f["faces"]) > 0 for f in meta["frames"])
    clip_dir = tmp_path / "dev" / "dia1_utt1"
    assert (clip_dir / "metadata.json").exists()
```

- [ ] **Step 6: Run the integration test**

Run: `.venv/bin/python3 -m pytest tests/test_preprocess.py -v -k real_clip`
Expected: 1 passed (if MELD is extracted at `data/meld/raw/extracted`) or 1 skipped otherwise.

- [ ] **Step 7: Add the CLI driver for running preprocessing over a full split**

Create `scripts/preprocess_meld.py`:

```python
#!/usr/bin/env python3
"""Preprocess every clip in one MELD split: decode, detect+track faces, crop
and letterbox, write crops + metadata.json per clip.

Usage:
    .venv/bin/python3 scripts/preprocess_meld.py --split dev
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meld_emotion.config import LABELS_DIR, PREPROCESSED_DIR, split_video_dir
from meld_emotion.data.labels import load_split
from meld_emotion.data.video_index import build_video_index
from meld_emotion.data.preprocess import preprocess_clip
from meld_emotion.vision.face_detector import build_face_detector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "dev", "test"], required=True)
    parser.add_argument("--limit", type=int, default=None, help="process only the first N rows (for testing)")
    args = parser.parse_args()

    utterances = load_split(args.split, labels_dir=LABELS_DIR)
    if args.limit:
        utterances = utterances[:args.limit]
    index = build_video_index(split_video_dir(args.split))
    detector = build_face_detector()

    ok, failed, missing = 0, 0, 0
    for i, u in enumerate(utterances):
        key = (u.dialogue_id, u.utterance_id)
        path = index.get(key)
        if path is None:
            missing += 1
            continue
        meta = preprocess_clip(path, split=args.split, dialogue_id=u.dialogue_id,
                                utterance_id=u.utterance_id, detector=detector,
                                out_dir=PREPROCESSED_DIR)
        if meta["status"] == "ok":
            ok += 1
        else:
            failed += 1
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(utterances)} (ok={ok} failed={failed} missing={missing})")

    print(f"Done. ok={ok} failed={failed} missing={missing} total={len(utterances)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Smoke-test the CLI on a small slice of real data**

Run: `.venv/bin/python3 scripts/preprocess_meld.py --split dev --limit 5`
Expected: `Done. ok=5 failed=0 missing=0 total=5` (or `missing=1` if the slice
happens to include the known-missing `dia110_utt7` row), and
`data/meld/preprocessed/dev/` contains 5 clip subdirectories each with a
`metadata.json`.

- [ ] **Step 9: Commit**

```bash
git add src/meld_emotion/data/preprocess.py tests/test_preprocess.py scripts/preprocess_meld.py
git commit -m "Add MELD clip preprocessing pipeline (detect, track, crop, letterbox) with CLI driver"
```

---

### Task 7: Pretrained vision encoders

**Files:**
- Create: `src/meld_emotion/vision/encoders.py`
- Test: `tests/test_encoders.py`

**Interfaces:**
- Produces: `FACE_EMOTION_LABELS: tuple[str, ...]` (7 labels, in the order
  the pretrained model's logits are indexed), class `FaceEmotionEncoder`
  with `__init__(self, device: str = "cpu")` and `.encode(self, image:
  PIL.Image) -> dict` (keys `"features"`: shape `(768,)` float32 ndarray,
  `"probs"`: shape `(7,)` float32 ndarray summing to 1); class
  `SceneEncoder` with `__init__(self, device: str = "cpu")` and
  `.encode(self, image: PIL.Image) -> np.ndarray` (shape `(512,)`).

**Note:** these tests download pretrained weights from Hugging Face on first
run (cached under `~/.cache/huggingface` afterward) — requires internet
access once.

- [ ] **Step 1: Add ML dependencies**

Edit `pyproject.toml` dependencies to add `"torch>=2.2"`,
`"transformers>=4.40"`, `"pillow>=10.0"`.

Run: `.venv/bin/python3 -m pip install torch transformers pillow`

- [ ] **Step 2: Write the failing tests**

Create `tests/test_encoders.py`:

```python
import numpy as np
from PIL import Image


def test_face_encoder_outputs_768d_feature_and_7way_probs():
    from meld_emotion.vision.encoders import FaceEmotionEncoder
    encoder = FaceEmotionEncoder()
    image = Image.new("RGB", (224, 224), color=(128, 100, 90))
    result = encoder.encode(image)
    assert result["features"].shape == (768,)
    assert result["probs"].shape == (7,)
    assert np.isclose(result["probs"].sum(), 1.0, atol=1e-3)


def test_scene_encoder_outputs_512d_feature():
    from meld_emotion.vision.encoders import SceneEncoder
    encoder = SceneEncoder()
    image = Image.new("RGB", (224, 224), color=(50, 60, 70))
    features = encoder.encode(image)
    assert features.shape == (512,)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python3 -m pytest tests/test_encoders.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meld_emotion.vision.encoders'`

- [ ] **Step 4: Implement**

Create `src/meld_emotion/vision/encoders.py`:

```python
"""Pretrained, frozen vision encoders: per-face expression features (dima806)
and general-purpose scene features (CLIP). See design doc §4.1 for why these
are two different models, not one applied twice."""
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, CLIPModel, CLIPProcessor

FACE_MODEL_ID = "dima806/facial_emotions_image_detection"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
FACE_EMOTION_LABELS = ("sad", "disgust", "angry", "neutral", "fear", "surprise", "happy")


class FaceEmotionEncoder:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(FACE_MODEL_ID)
        self.model = AutoModelForImageClassification.from_pretrained(FACE_MODEL_ID).to(device).eval()

    @torch.no_grad()
    def encode(self, image) -> dict:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        pooled = outputs.hidden_states[-1][:, 0, :].squeeze(0)  # CLS token
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
        return {"features": pooled.cpu().numpy().astype("float32"),
                "probs": probs.cpu().numpy().astype("float32")}


class SceneEncoder:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        self.model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()

    @torch.no_grad()
    def encode(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        features = self.model.get_image_features(**inputs).squeeze(0)
        return features.cpu().numpy().astype("float32")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python3 -m pytest tests/test_encoders.py -v`
Expected: 2 passed. If `result["features"].shape` is not `(768,)`, print
`outputs.hidden_states[-1].shape` to check the actual hidden size and CLS
position for this specific checkpoint, and adjust the indexing — the model
card doesn't guarantee the exact pooling convention used here.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/meld_emotion/vision/encoders.py tests/test_encoders.py
git commit -m "Add frozen face-expression and CLIP scene encoders"
```

---

### Task 8: Frozen feature cache builder

**Files:**
- Create: `src/meld_emotion/data/cache.py`
- Create: `scripts/build_feature_cache.py`
- Test: `tests/test_cache.py`

**Interfaces:**
- Consumes: `FaceEmotionEncoder`, `SceneEncoder` (Task 7); `PREPROCESSED_DIR`,
  `FEATURE_CACHE_DIR` (Task 1); clip directory layout from Task 6
  (`metadata.json` + face/scene JPEGs).
- Produces: `build_clip_cache(clip_dir: Path, face_encoder: FaceEmotionEncoder,
  scene_encoder: SceneEncoder) -> dict` (keys: `dialogue_id`, `utterance_id`,
  `face_features` `(N_faces, 768)` float32, `face_probs` `(N_faces, 7)`
  float32, `face_track_ids` `(N_faces,)` int64, `face_frame_idx` `(N_faces,)`
  int64, `scene_features` `(N_frames, 512)` float32, `scene_frame_idx`
  `(N_frames,)` int64), `save_clip_cache(cache: dict, out_path: Path)`,
  `build_split_cache(split: str, preprocessed_dir: Path, cache_dir: Path,
  device: str = "cpu") -> int` (returns count of clips cached).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cache.py`:

```python
import json

import numpy as np
from PIL import Image


def _make_fixture_clip(tmp_path):
    clip_dir = tmp_path / "dev" / "dia1_utt1"
    clip_dir.mkdir(parents=True)
    Image.new("RGB", (224, 224), (120, 80, 60)).save(clip_dir / "frame0_face0.jpg")
    Image.new("RGB", (224, 224), (30, 40, 50)).save(clip_dir / "frame0_scene.jpg")
    meta = {
        "split": "dev", "dialogue_id": 1, "utterance_id": 1, "status": "ok",
        "frames": [{"sample_index": 0, "source_frame_index": 0,
                    "faces": [{"track_id": 0, "box": [0, 0, 50, 50], "score": 0.9,
                               "path": "frame0_face0.jpg"}],
                    "scene_path": "frame0_scene.jpg"}],
    }
    with open(clip_dir / "metadata.json", "w") as f:
        json.dump(meta, f)
    return clip_dir


def test_build_clip_cache_produces_expected_shapes(tmp_path):
    from meld_emotion.data.cache import build_clip_cache
    from meld_emotion.vision.encoders import FaceEmotionEncoder, SceneEncoder
    clip_dir = _make_fixture_clip(tmp_path)
    cache = build_clip_cache(clip_dir, FaceEmotionEncoder(), SceneEncoder())
    assert cache["face_features"].shape == (1, 768)
    assert cache["face_probs"].shape == (1, 7)
    assert cache["scene_features"].shape == (1, 512)
    assert cache["face_track_ids"].tolist() == [0]
    assert cache["dialogue_id"] == 1
    assert cache["utterance_id"] == 1


def test_save_clip_cache_round_trips(tmp_path):
    from meld_emotion.data.cache import build_clip_cache, save_clip_cache
    from meld_emotion.vision.encoders import FaceEmotionEncoder, SceneEncoder
    clip_dir = _make_fixture_clip(tmp_path)
    cache = build_clip_cache(clip_dir, FaceEmotionEncoder(), SceneEncoder())
    out_path = tmp_path / "cache_out" / "dia1_utt1.npz"
    save_clip_cache(cache, out_path)
    loaded = np.load(out_path)
    assert loaded["face_features"].shape == (1, 768)
    assert int(loaded["dialogue_id"]) == 1


def test_build_split_cache_processes_every_clip_in_the_preprocessed_dir(tmp_path):
    from meld_emotion.data.cache import build_split_cache
    preprocessed_dir = tmp_path / "preprocessed"
    cache_dir = tmp_path / "cache"
    _make_fixture_clip(preprocessed_dir)
    count = build_split_cache("dev", preprocessed_dir=preprocessed_dir, cache_dir=cache_dir)
    assert count == 1
    assert (cache_dir / "dev" / "dia1_utt1.npz").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python3 -m pytest tests/test_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meld_emotion.data.cache'`

- [ ] **Step 3: Implement**

Create `src/meld_emotion/data/cache.py`:

```python
"""Runs the frozen vision encoders over preprocessed crops and persists the
resulting features as one .npz per clip."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from meld_emotion.config import PREPROCESSED_DIR, FEATURE_CACHE_DIR
from meld_emotion.vision.encoders import FaceEmotionEncoder, SceneEncoder


def build_clip_cache(clip_dir: Path, face_encoder: FaceEmotionEncoder,
                      scene_encoder: SceneEncoder) -> dict:
    with open(clip_dir / "metadata.json") as f:
        meta = json.load(f)

    face_features, face_probs, face_track_ids, face_frame_idx = [], [], [], []
    scene_features, scene_frame_idx = [], []

    for frame in meta["frames"]:
        for face in frame["faces"]:
            image = Image.open(clip_dir / face["path"]).convert("RGB")
            result = face_encoder.encode(image)
            face_features.append(result["features"])
            face_probs.append(result["probs"])
            face_track_ids.append(face["track_id"])
            face_frame_idx.append(frame["sample_index"])

        scene_image = Image.open(clip_dir / frame["scene_path"]).convert("RGB")
        scene_features.append(scene_encoder.encode(scene_image))
        scene_frame_idx.append(frame["sample_index"])

    return {
        "dialogue_id": meta["dialogue_id"],
        "utterance_id": meta["utterance_id"],
        "face_features": np.array(face_features, dtype="float32") if face_features else np.zeros((0, 768), "float32"),
        "face_probs": np.array(face_probs, dtype="float32") if face_probs else np.zeros((0, 7), "float32"),
        "face_track_ids": np.array(face_track_ids, dtype="int64"),
        "face_frame_idx": np.array(face_frame_idx, dtype="int64"),
        "scene_features": np.array(scene_features, dtype="float32"),
        "scene_frame_idx": np.array(scene_frame_idx, dtype="int64"),
    }


def save_clip_cache(cache: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **cache)


def build_split_cache(split: str, preprocessed_dir: Path = PREPROCESSED_DIR,
                       cache_dir: Path = FEATURE_CACHE_DIR, device: str = "cpu") -> int:
    face_encoder = FaceEmotionEncoder(device=device)
    scene_encoder = SceneEncoder(device=device)
    split_dir = preprocessed_dir / split
    count = 0
    for clip_dir in sorted(split_dir.iterdir()):
        if not (clip_dir / "metadata.json").exists():
            continue
        cache = build_clip_cache(clip_dir, face_encoder, scene_encoder)
        save_clip_cache(cache, cache_dir / split / f"{clip_dir.name}.npz")
        count += 1
    return count
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python3 -m pytest tests/test_cache.py -v`
Expected: 3 passed

- [ ] **Step 5: Add the CLI driver**

Create `scripts/build_feature_cache.py`:

```python
#!/usr/bin/env python3
"""Build the frozen vision feature cache for one MELD split from already
preprocessed clips (run scripts/preprocess_meld.py first).

Usage:
    .venv/bin/python3 scripts/build_feature_cache.py --split dev
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meld_emotion.data.cache import build_split_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "dev", "test"], required=True)
    parser.add_argument("--device", default="cpu", help="'cpu' or 'mps'")
    args = parser.parse_args()
    count = build_split_cache(args.split, device=args.device)
    print(f"Cached {count} clips for split={args.split}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Smoke-test the CLI against the small real slice from Task 6**

Run: `.venv/bin/python3 scripts/build_feature_cache.py --split dev`
Expected: `Cached 5 clips for split=dev` (matching however many clips Task
6 Step 8 preprocessed), and `data/meld/features/dev/` contains 5 `.npz` files.

- [ ] **Step 7: Commit**

```bash
git add src/meld_emotion/data/cache.py tests/test_cache.py scripts/build_feature_cache.py
git commit -m "Add frozen vision feature cache builder with CLI driver"
```

---

## Running the full pipeline for real (not part of the automated test suite)

Once all 8 tasks are merged, the full data/feature pipeline for a split is:

```bash
.venv/bin/python3 scripts/preprocess_meld.py --split dev
.venv/bin/python3 scripts/build_feature_cache.py --split dev --device mps
```

Repeat for `train` and `test`. Preprocessing all three splits and building
the full feature cache is expected to take under two hours combined (design
doc §6) — run it in the background and verify clip counts against the known
row counts (train 9989, dev 1109, test 2610, allowing for the one known
missing dev row) before starting the next plan's training work.

## Self-Review

**Spec coverage.** Design doc §5 (data verification, per-split indexing,
duration handling) → Tasks 2, 3, 6. §4.2 (face detection, tracking, shot-cut
reset, letterboxing) → Tasks 4, 5, 6. §4.1 (face + scene encoders, why two
models) → Task 7. §6 preprocessing/cache steps → Tasks 6, 8. §4.3 dialogue
context → Task 2. Text encoding (RoBERTa) itself, the trainable fusion model,
Stage 1 training, ablations, and batch evaluation are out of scope for this
plan by design — they're the next plan, since RoBERTa's top layers are
fine-tuned during training rather than cached (design doc §6), so they don't
belong in a frozen-feature pipeline.

**Placeholder scan.** No TBD/TODO markers; every step has complete, runnable
code. Task 7 Step 5 includes a debugging note (verify the CLS-pooling
assumption against the real model) rather than a placeholder — it's guidance
for what to do if the test fails, not an unwritten step.

**Type consistency.** Verified `build_video_index(split_dir: Path)` (Task 3)
signature matches its use in Task 6's `preprocess_meld.py` CLI. Verified
`detect_faces(detector, frame_bgr)` (Task 4) and `FaceTracker.update(boxes)` /
`is_shot_cut(prev, curr)` (Task 5) signatures match their use in Task 6's
`preprocess_clip`. Verified `FaceEmotionEncoder`/`SceneEncoder` (Task 7)
constructor and `.encode()` signatures match their use in Task 8's
`build_clip_cache`. Verified the face-crop/scene-JPEG filename convention
written in Task 6 (`frame{i}_face{track_id}.jpg`, `frame{i}_scene.jpg`,
`metadata.json`'s `faces[].path` / `frames[].scene_path` fields) matches
exactly what Task 8 reads.
