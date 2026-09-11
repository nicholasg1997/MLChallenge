# MELD Data & Feature Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the raw MELD.Raw archive into a verified, cached set of frozen
vision features (face-crop embeddings + expression probabilities + scene
embeddings) and a per-split manifest that joins every utterance's labels and
dialogue context to its cached features, ready for the training plan that
follows.

**Architecture:** An installable package (`src/meld_emotion/`) with four
layers: label/text loading, per-split video indexing, a vision pipeline
(sequential decode → detection → tracking → crop/letterbox), and a caching
layer that runs the frozen pretrained encoders in batches over the
preprocessed images and persists one `.npz` per clip. A final manifest step
joins labels to cache files and reports the pipeline statistics the design
doc says will be measured rather than assumed. Every layer is unit-tested
offline with synthetic fixtures (synthetic videos, stub detectors, stub
encoders); tests that download weights are marked `network`, and tests that
need the extracted MELD archive skip when it is absent.

**Tech Stack:** Python 3.11, uv, pytest, OpenCV (video decode, YuNet face
detection), NumPy, PyTorch + Hugging Face `transformers` (pretrained face and
CLIP scene encoders), Pillow.

## Global Constraints

- Everything runs on-device, no remote inference (design doc §2).
- Total inference-path parameters must stay under 6B (design doc §4.1). This
  plan introduces exactly three pretrained components and no trained ones:
  face encoder ViT-Base ~86M, CLIP ViT-B/32 image + text towers ~151M, YuNet
  ~75K. The running total is carried into the training plan.
- Frames are sampled at ~3fps (design doc §2, §4.2).
- MELD clips are pre-cut per utterance; the CSV `StartTime`/`EndTime` columns
  are redundant and are **never read** (design doc §5). `dia38_utt4` in the
  *test* split is a genuine 304.97s outlier utterance matching its 304.94s
  CSV timestamp almost exactly — not a bad timestamp (a 2.38s file with the
  same name exists, but it's `train_splits/dia38_utt4.mp4`, an unrelated clip
  from a different split; that mismatch is the cross-split collision this
  same Global Constraints section warns about below, not a property of the
  test row). A 15-second decode cap is kept purely as a guard against
  outliers like this one.
- **`(Dialogue_ID, Utterance_ID)` is never a unique key across MELD splits** —
  IDs restart at 0 in each split (1,740 collide between train/test alone).
  Every index, cache, or lookup in this plan is scoped to one split's own
  directory; nothing may build a single index spanning multiple splits
  (design doc §5).
- Dependencies are managed with uv: `uv add` / `uv add --dev`, never `pip`,
  so `uv.lock` stays authoritative. Every command runs via `uv run`.
- `uv run pytest` with no arguments must pass **offline in seconds**. Tests
  that download weights carry `@pytest.mark.network` (deselected by default);
  tests that need the extracted MELD archive `pytest.skip` when it is absent.
- `requires-python = ">=3.11"` (pyproject.toml).

---

## File Structure

```
pyproject.toml                  # MODIFIED (Task 2): hatchling build, dev group, pytest markers
src/meld_emotion/
  __init__.py                   # Task 1 (done)
  config.py                     # shared paths, SPLIT_DIRS -- Task 1 (done)
  data/
    __init__.py
    labels.py                   # Utterance loading + dialogue context window (Task 3)
    video_index.py              # per-split video file index (Task 4)
    preprocess.py               # decode/detect/track/crop pipeline + multiprocess driver (Task 7)
    cache.py                    # batched frozen-feature cache builder (Task 9)
    manifest.py                 # labels <-> features join + pipeline stats (Task 10)
  vision/
    __init__.py
    face_detector.py            # YuNet wrapper (Task 5)
    tracker.py                  # IoU tracker + shot-change scoring (Task 6)
    encoders.py                 # face + scene pretrained encoders, batched (Task 8)
scripts/
  view_meld_clips.py            # MODIFIED: Tasks 2, 4, 5 (dedupe onto the package)
  preprocess_meld.py            # NEW: CLI driver for Task 7
  build_feature_cache.py        # NEW: CLI driver for Task 9
  build_manifest.py             # NEW: CLI driver for Task 10
tests/
  test_config.py                # Task 1 (done)
  test_package.py               # Task 2
  test_labels.py                # Task 3
  test_video_index.py           # Task 4
  test_face_detector.py         # Task 5
  test_tracker.py               # Task 6
  test_preprocess.py            # Task 7
  test_encoders.py              # Task 8 (network)
  test_cache.py                 # Task 9
  test_manifest.py              # Task 10
```

---

### Task 1: Project scaffolding and shared config — ✅ completed

Landed in commits `dff5efa` and `50bedc9` on `meld-data-feature-pipeline`.
Kept here so later tasks' readers know exactly what exists. (Step 7 added a
`sys.path.insert` hack to `scripts/view_meld_clips.py`; Task 2 removes it.)

**Files:**
- Created: `src/meld_emotion/__init__.py` (empty)
- Created: `src/meld_emotion/config.py`
- Modified: `pyproject.toml` (added `pytest`, `[tool.pytest.ini_options] pythonpath = ["src"]`)
- Modified: `scripts/view_meld_clips.py` (imports paths + `SPLIT_DIRS` from the package)
- Test: `tests/test_config.py` (4 tests, passing)

**Interfaces:**
- Produces: `REPO_ROOT: Path`, `DATA_DIR: Path`, `LABELS_DIR: Path`,
  `RAW_EXTRACTED_DIR: Path`, `PREPROCESSED_DIR: Path`, `FEATURE_CACHE_DIR:
  Path`, `MODELS_DIR: Path`, `FACE_DETECTOR_MODEL_PATH: Path`, `SPLIT_DIRS:
  dict[str, str]`, `SPLITS: tuple[str, ...]`, `split_video_dir(split: str)
  -> Path` (raises `ValueError` on an unknown split).

`src/meld_emotion/config.py` as landed:

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

- [x] **Step 1–9:** package created, `tests/test_config.py` passing (4 tests), viewer deduplicated onto `config.py`, committed.

---

### Task 2: Make the package installable and drop the path hack

Right now `import meld_emotion` only works inside pytest (via
`pythonpath = ["src"]`); `pyproject.toml` has no build backend, so nothing is
actually installed into the environment and every script needs a
`sys.path.insert`. This task makes the package a real editable install so
scripts and tests import it the normal way, moves `pytest` to a dev group,
and registers the `network` marker used by Task 8.

**Files:**
- Modify: `pyproject.toml`
- Modify: `scripts/view_meld_clips.py`
- Test: `tests/test_package.py`

**Interfaces:**
- Produces: `meld_emotion` importable from any working directory inside the
  uv environment; pytest marker `network` (deselected by default).

- [ ] **Step 1: Write the failing test**

Create `tests/test_package.py`:

```python
import os
import subprocess
import sys


def test_package_imports_from_a_neutral_cwd_without_path_hacks(tmp_path):
    # A fresh interpreter, cwd outside the repo, PYTHONPATH stripped: this can
    # only pass if the package is genuinely installed into the environment.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-c", "import meld_emotion.config as c; print(c.SPLITS)"],
        cwd=tmp_path, capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "train" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_package.py -v`
Expected: FAIL — the assertion message contains `ModuleNotFoundError: No module named 'meld_emotion'`

- [ ] **Step 3: Rewrite `pyproject.toml`**

Replace the whole file with:

```toml
[project]
name = "mlchallenge"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "opencv-python==5.0.0.93",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/meld_emotion"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-m 'not network'"
markers = [
    "network: downloads pretrained weights from Hugging Face; deselected by default, run with `uv run pytest -m network`",
]
```

- [ ] **Step 4: Sync the environment (installs the package editable, re-locks)**

Run: `uv sync`
Expected: output includes `Built mlchallenge @ file://...` and `+ mlchallenge==0.1.0 (from file://...)`.
Then run: `uv pip show mlchallenge | head -3`
Expected: `Name: mlchallenge`, `Version: 0.1.0`, and an `Editable project location:` line pointing at this checkout.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_package.py -v`
Expected: 1 passed

- [ ] **Step 6: Remove the path hack from `scripts/view_meld_clips.py`**

Replace these lines near the top of the script:

```python
import sys
import textwrap
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import cv2
```

with:

```python
import sys
import textwrap

import cv2
```

(`sys` stays — the script still uses `sys.exit`. Nothing else in the script
uses `Path`.)

- [ ] **Step 7: Verify the viewer still runs**

Run: `uv run python scripts/view_meld_clips.py --help`
Expected: the usage text, no traceback.

- [ ] **Step 8: Run the whole suite**

Run: `uv run pytest -v`
Expected: 5 passed (4 config + 1 package)

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml uv.lock scripts/view_meld_clips.py tests/test_package.py
git commit -m "Make meld_emotion an editable install (hatchling); move pytest to dev group; drop sys.path hack"
```

---

### Task 3: Label loading and dialogue context window

**Files:**
- Create: `src/meld_emotion/data/__init__.py`
- Create: `src/meld_emotion/data/labels.py`
- Test: `tests/test_labels.py`

**Interfaces:**
- Consumes: nothing (takes `labels_dir` as a parameter so it is testable
  with a temp directory).
- Produces: `EMOTIONS: tuple[str, ...]` (7, MELD spelling), `SENTIMENTS:
  tuple[str, ...]` (3), `Utterance` (frozen dataclass: `split, dialogue_id,
  utterance_id, speaker, text, emotion, sentiment`), `load_split(split: str,
  labels_dir: Path) -> list[Utterance]` (raises `ValueError` on an unknown
  emotion/sentiment), `group_by_dialogue(utterances: list[Utterance]) ->
  dict[int, list[Utterance]]` (sorted by utterance_id), `context_window(
  by_dialogue, dialogue_id: int, utterance_id: int, k: int) -> list[str]` (up
  to `k` previous texts, oldest first, then the current text last).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_labels.py`:

```python
import csv

import pytest

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
        _row(0, 0, "Hello there", emotion="joy", sentiment="positive"),
        _row(0, 1, "Not really", emotion="anger", sentiment="negative"),
    ])
    utterances = load_split("dev", labels_dir=tmp_path)
    assert len(utterances) == 2
    assert utterances[0].text == "Hello there"
    assert utterances[0].emotion == "joy"
    assert utterances[0].sentiment == "positive"
    assert utterances[0].split == "dev"
    assert utterances[1].dialogue_id == 0
    assert utterances[1].utterance_id == 1


def test_load_split_rejects_an_unknown_emotion_label(tmp_path):
    from meld_emotion.data.labels import load_split
    _write_csv(tmp_path / "dev_sent_emo.csv", [_row(0, 0, "hi", emotion="confused")])
    with pytest.raises(ValueError, match="confused"):
        load_split("dev", labels_dir=tmp_path)


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

Run: `uv run pytest tests/test_labels.py -v`
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
            if row["Emotion"] not in EMOTIONS:
                raise ValueError(f"{path.name}: unknown emotion {row['Emotion']!r} "
                                 f"(dia{row['Dialogue_ID']}_utt{row['Utterance_ID']})")
            if row["Sentiment"] not in SENTIMENTS:
                raise ValueError(f"{path.name}: unknown sentiment {row['Sentiment']!r} "
                                 f"(dia{row['Dialogue_ID']}_utt{row['Utterance_ID']})")
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

Run: `uv run pytest tests/test_labels.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/meld_emotion/data/__init__.py src/meld_emotion/data/labels.py tests/test_labels.py
git commit -m "Add MELD label loading with validation and dialogue context window"
```

---

### Task 4: Per-split video index

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

Run: `uv run pytest tests/test_video_index.py -v`
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

Run: `uv run pytest tests/test_video_index.py -v`
Expected: 3 passed

- [ ] **Step 5: Point `scripts/view_meld_clips.py` at the shared implementation**

Delete the script's local `build_video_index` function (the whole `def
build_video_index(split_dir): ...` block) and add this import next to the
existing `from meld_emotion.config import (...)` line:

```python
from meld_emotion.data.video_index import build_video_index
```

- [ ] **Step 6: Verify the viewer still runs against real data**

Run: `uv run python scripts/view_meld_clips.py --split dev --per-emotion 1 --seed 0` and press `q` when the first window appears.
Expected: prints `Indexing dev video files...`, then `  found 1112 video files under .../MELD.Raw/dev_splits_complete`, then one sampled-clip line per emotion, no traceback.

- [ ] **Step 7: Commit**

```bash
git add src/meld_emotion/data/video_index.py tests/test_video_index.py scripts/view_meld_clips.py
git commit -m "Extract per-split video index into shared module with a collision regression test"
```

---

### Task 5: Face detector wrapper

**Files:**
- Create: `src/meld_emotion/vision/__init__.py`
- Create: `src/meld_emotion/vision/face_detector.py`
- Modify: `pyproject.toml` (via `uv add numpy`)
- Modify: `scripts/view_meld_clips.py`
- Test: `tests/test_face_detector.py`

**Interfaces:**
- Consumes: `FACE_DETECTOR_MODEL_PATH` from `meld_emotion.config` (Task 1).
- Produces: `DEFAULT_CONFIDENCE_THRESHOLD: float = 0.75`,
  `build_face_detector(score_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD)`
  (returns a `cv2.FaceDetectorYN`), `detect_faces(detector, frame_bgr:
  np.ndarray) -> list[tuple[int, int, int, int, float]]` (each tuple is `x,
  y, w, h, score` in pixels; empty list when nothing is found). Any object
  with `setInputSize((w, h))` and `detect(frame) -> (retval, faces_or_None)`
  is accepted as `detector`, which is how later tasks stub it.

- [ ] **Step 1: Add numpy as an explicit dependency**

Run: `uv add "numpy>=1.26"`
Expected: `pyproject.toml` gains `"numpy>=1.26"` under `dependencies`; `uv.lock` updated (numpy was already present transitively via opencv, so nothing new downloads).

- [ ] **Step 2: Write the failing tests**

Create `tests/test_face_detector.py`. These test the wrapper's parsing logic
against a stub detector, not the real YuNet model, so they run without the
downloaded model file:

```python
import numpy as np
import pytest


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

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_face_detector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meld_emotion.vision'`

- [ ] **Step 4: Implement**

Create `src/meld_emotion/vision/__init__.py` (empty file).

Create `src/meld_emotion/vision/face_detector.py`:

```python
"""OpenCV YuNet face detector wrapper (~75K params, zero-shot).

Uses OpenCV's own DNN backend rather than mediapipe's Tasks API, which
crashes with a GPU graph-service error on this platform (see
scripts/download_face_model.sh).
"""
import cv2

from meld_emotion.config import FACE_DETECTOR_MODEL_PATH

DEFAULT_CONFIDENCE_THRESHOLD = 0.75  # tuned against visible false positives (design doc §4.2, §5)


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

Run: `uv run pytest tests/test_face_detector.py -v`
Expected: 3 passed

- [ ] **Step 6: Point `scripts/view_meld_clips.py` at the shared implementation**

Delete the script's local `CONFIDENCE_THRESHOLD` constant and its local
`build_face_detector` and `detect_faces` functions, and add this import next
to the other `meld_emotion` imports:

```python
from meld_emotion.vision.face_detector import build_face_detector, detect_faces
```

The script's existing `build_face_detector()` call (no arguments) keeps the
0.75 default.

- [ ] **Step 7: Verify the viewer's face overlay still works against real data**

Run: `uv run python scripts/view_meld_clips.py --split dev --per-emotion 1 --seed 0 --faces` and press `q` after the first window appears.
Expected: no traceback; green face boxes with scores render as before.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock src/meld_emotion/vision/__init__.py src/meld_emotion/vision/face_detector.py tests/test_face_detector.py scripts/view_meld_clips.py
git commit -m "Extract face detector wrapper into shared module with stub-based unit tests"
```

---

### Task 6: Face tracker with shot-change scoring

**Files:**
- Create: `src/meld_emotion/vision/tracker.py`
- Test: `tests/test_tracker.py`

**Interfaces:**
- Produces: `iou(box_a: tuple, box_b: tuple) -> float`; class `FaceTracker`
  with `__init__(self, iou_threshold: float = 0.3, max_missed: int = 1)`,
  `.update(self, boxes: list[tuple]) -> list[int]` (one track ID per input
  box, same order; boxes may carry extra trailing fields such as `score`),
  `.reset(self)`; `SHOT_CUT_THRESHOLD: float = 0.3`;
  `shot_change_score(prev_frame_bgr, curr_frame_bgr) -> float` (Bhattacharyya
  distance between HSV histograms, 0.0 identical … 1.0 totally different);
  `is_shot_cut(prev_frame_bgr, curr_frame_bgr, threshold: float =
  SHOT_CUT_THRESHOLD) -> bool`.

The score is exposed separately from the boolean so Task 7 can persist it per
frame and the threshold can be re-checked against the full dataset later
(Task 10 stats). The 0.3 default was chosen from real MELD footage, not
guessed: over 259 sampled-frame transitions in dev `dia0`–`dia2`, same-shot
transitions cluster at 0.05–0.10 with a tail to ~0.25, while visible camera
cuts (8 faces → 1, 7 → 1, 1 → 11 between consecutive sampled frames) score
0.34–0.47 — every one of them *below* a naive 0.5 threshold. There is a gap
between 0.25 and 0.30 in that sample. (Sitcom cuts are between shots of the
same set, so the colour histograms differ far less than a synthetic red→green
cut.)

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
    # Regression: a track must NOT be aged in the same update() that created
    # it, otherwise a brand-new track dies on its very first miss.
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


def _red():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :, 2] = 255  # BGR
    return frame


def _green():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :, 1] = 255
    return frame


def test_shot_change_score_is_zero_for_identical_frames():
    from meld_emotion.vision.tracker import shot_change_score
    assert shot_change_score(_red(), _red()) == pytest.approx(0.0, abs=1e-6)


def test_shot_change_score_is_high_for_a_colour_cut():
    from meld_emotion.vision.tracker import shot_change_score
    assert shot_change_score(_red(), _green()) > 0.9


def test_identical_frames_are_not_a_shot_cut():
    from meld_emotion.vision.tracker import is_shot_cut
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    assert is_shot_cut(frame, frame) is False


def test_very_different_frames_are_a_shot_cut():
    from meld_emotion.vision.tracker import is_shot_cut
    assert is_shot_cut(_red(), _green()) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tracker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meld_emotion.vision.tracker'`

- [ ] **Step 3: Implement**

Create `src/meld_emotion/vision/tracker.py`:

```python
"""Frame-to-frame face tracking: greedy IoU matching plus shot-change scoring.

Pure geometry -- no pretrained model, no audio. Track continuity is a soft
inductive bias for the fusion model (design doc §4.2), not ground-truth
identity, and is approximate at the ~3fps sampling rate this pipeline uses.
"""
from dataclasses import dataclass

import cv2

SHOT_CUT_THRESHOLD = 0.3  # Bhattacharyya distance; real MELD cuts score 0.34-0.47, same-shot <0.25


def iou(box_a: tuple, box_b: tuple) -> float:
    ax, ay, aw, ah = box_a[:4]
    bx, by, bw, bh = box_b[:4]
    inter_x1, inter_y1 = max(ax, bx), max(ay, by)
    inter_x2, inter_y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
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
        input box, in the same order.

        Order matters: (1) age every EXISTING track, (2) matches reset their
        age to 0, (3) unmatched boxes become new tracks with age 0, (4) prune.
        Creating new tracks before ageing would age them in the same call
        that created them.
        """
        xywh_boxes = [tuple(b[:4]) for b in boxes]

        for track in self._tracks:
            track.missed_frames += 1

        pairs = sorted(
            ((iou(box, track.box), bi, ti)
             for bi, box in enumerate(xywh_boxes)
             for ti, track in enumerate(self._tracks)),
            reverse=True,
        )
        assigned: list = [None] * len(xywh_boxes)
        used_tracks: set[int] = set()
        for score, bi, ti in pairs:
            if score < self.iou_threshold:
                break  # sorted descending: nothing further can qualify
            if assigned[bi] is not None or ti in used_tracks:
                continue
            track = self._tracks[ti]
            track.box = xywh_boxes[bi]
            track.missed_frames = 0
            used_tracks.add(ti)
            assigned[bi] = track.track_id

        for bi, box in enumerate(xywh_boxes):
            if assigned[bi] is None:
                track = _Track(track_id=self._next_id, box=box)
                self._next_id += 1
                self._tracks.append(track)
                assigned[bi] = track.track_id

        self._tracks = [t for t in self._tracks if t.missed_frames <= self.max_missed]
        return assigned


def _hsv_hist(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def shot_change_score(prev_frame_bgr, curr_frame_bgr) -> float:
    """Bhattacharyya distance between the two frames' HSV histograms:
    0.0 for identical content, approaching 1.0 for a hard camera cut."""
    return float(cv2.compareHist(_hsv_hist(prev_frame_bgr), _hsv_hist(curr_frame_bgr),
                                 cv2.HISTCMP_BHATTACHARYYA))


def is_shot_cut(prev_frame_bgr, curr_frame_bgr, threshold: float = SHOT_CUT_THRESHOLD) -> bool:
    return shot_change_score(prev_frame_bgr, curr_frame_bgr) >= threshold
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tracker.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add src/meld_emotion/vision/tracker.py tests/test_tracker.py
git commit -m "Add IoU face tracker (age-then-match ordering) and HSV shot-change scoring"
```

---

### Task 7: Preprocessing pipeline with multiprocess driver

**Files:**
- Create: `src/meld_emotion/data/preprocess.py`
- Create: `scripts/preprocess_meld.py`
- Test: `tests/test_preprocess.py`

**Interfaces:**
- Consumes: `build_face_detector`, `detect_faces` (Task 5); `FaceTracker`,
  `SHOT_CUT_THRESHOLD`, `shot_change_score` (Task 6); `build_video_index`
  (Task 4); `load_split`, `Utterance` (Task 3); `LABELS_DIR`,
  `PREPROCESSED_DIR`, `split_video_dir` (Task 1).
- Produces: `SAMPLE_FPS = 3.0`, `MAX_DECODE_SECONDS = 15.0`, `CROP_SIZE =
  224`; `sample_frame_indices(total_frames: int, fps: float, sample_fps:
  float = SAMPLE_FPS, max_seconds: float = MAX_DECODE_SECONDS) ->
  list[int]`; `letterbox(frame_bgr, size: int = CROP_SIZE) -> np.ndarray`;
  `crop_face(frame_bgr, box: tuple, margin: float = 0.2, size: int =
  CROP_SIZE) -> np.ndarray | None`; `clip_dir_for(out_dir: Path, split: str,
  dialogue_id: int, utterance_id: int) -> Path` (=
  `out_dir/<split>/dia<D>_utt<U>`); `preprocess_clip(video_path: Path, split:
  str, dialogue_id: int, utterance_id: int, detector=None, out_dir: Path =
  PREPROCESSED_DIR, overwrite: bool = False) -> dict`;
  `preprocess_split(split: str, utterances: list[Utterance], index:
  dict[tuple[int, int], Path], out_dir: Path = PREPROCESSED_DIR, workers: int
  = 1, overwrite: bool = False, detector=None, progress_every: int = 200) ->
  Counter` (keys: `ok`, `no_frames`, `decode_failed`, `missing_video`).

**`metadata.json` layout** written per clip (Tasks 9 and 10 read exactly this):

```json
{"split": "dev", "dialogue_id": 1, "utterance_id": 1, "video": "dia1_utt1.mp4",
 "status": "ok",                       // or "no_frames" | "decode_failed"
 "fps": 24.0, "total_frames": 39, "n_shot_cuts": 0,
 "frames": [
   {"sample_index": 0, "source_frame_index": 0,
    "shot_change": 0.0, "shot_cut": false,
    "faces": [{"track_id": 0, "box": [x, y, w, h], "score": 0.94, "path": "frame0_face0.jpg"}],
    "scene_path": "frame0_scene.jpg"}
 ]}
```

`sample_index` is contiguous over the frames actually kept; `source_frame_index`
is the position in the original video. Failed clips get a `metadata.json` too
(with `frames: []`) so failures are persisted, not just counted.

- [ ] **Step 1: Write the failing unit tests**

Create `tests/test_preprocess.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_preprocess.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meld_emotion.data.preprocess'`

- [ ] **Step 3: Implement**

Create `src/meld_emotion/data/preprocess.py`:

```python
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
            clip_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(clip_dir / face_name), crop)
            face_records.append({"track_id": track_id, "box": [x, y, w, h],
                                 "score": score, "path": face_name})

        scene_name = f"frame{sample_i}_scene.jpg"
        clip_dir.mkdir(parents=True, exist_ok=True)
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


def preprocess_split(split: str, utterances, index: dict, out_dir: Path = PREPROCESSED_DIR,
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

    for i, status in enumerate(results, 1):
        counts[status] += 1
        if progress_every and i % progress_every == 0:
            print(f"  {i}/{len(jobs)} {dict(counts)}", flush=True)

    if pool is not None:
        pool.close()
        pool.join()
    return counts
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_preprocess.py -v`
Expected: 11 passed

- [ ] **Step 5: Append the skippable integration test against real data**

Append to `tests/test_preprocess.py`:

```python
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
```

- [ ] **Step 6: Run the integration test**

Run: `uv run pytest tests/test_preprocess.py -v -k real_clip`
Expected: 1 passed (MELD is extracted at `data/meld/raw/extracted`), or 1 skipped otherwise.

- [ ] **Step 7: Add the CLI driver**

Create `scripts/preprocess_meld.py`:

```python
#!/usr/bin/env python3
"""Preprocess every clip in one MELD split: decode at ~3fps, detect + track
faces, crop and letterbox, write crops + metadata.json per clip under
data/meld/preprocessed/<split>/. Resumable: clips with an existing
metadata.json are skipped unless --overwrite.

Usage:
    uv run python scripts/preprocess_meld.py --split dev --workers 8
"""
import argparse
import os
import time

from meld_emotion.config import LABELS_DIR, PREPROCESSED_DIR, SPLITS, split_video_dir
from meld_emotion.data.labels import load_split
from meld_emotion.data.preprocess import preprocess_split
from meld_emotion.data.video_index import build_video_index


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--limit", type=int, default=None, help="process only the first N rows")
    parser.add_argument("--overwrite", action="store_true", help="redo clips that already have metadata")
    args = parser.parse_args()

    utterances = load_split(args.split, labels_dir=LABELS_DIR)
    if args.limit:
        utterances = utterances[:args.limit]
    index = build_video_index(split_video_dir(args.split))
    print(f"{args.split}: {len(utterances)} utterances, {len(index)} video files, "
          f"{args.workers} workers -> {PREPROCESSED_DIR / args.split}")

    started = time.perf_counter()
    counts = preprocess_split(args.split, utterances, index, out_dir=PREPROCESSED_DIR,
                              workers=args.workers, overwrite=args.overwrite)
    elapsed = time.perf_counter() - started
    print(f"Done in {elapsed / 60:.1f} min: {dict(counts)} (total={len(utterances)})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Smoke-test the CLI on a small slice of real data, single- and multi-process**

Run: `uv run python scripts/preprocess_meld.py --split dev --limit 5 --workers 1`
Expected: `Done in 0.0 min: {'ok': 5} (total=5)` and `data/meld/preprocessed/dev/` contains 5 clip directories each with a `metadata.json`.

Run: `uv run python scripts/preprocess_meld.py --split dev --limit 40 --workers 4`
Expected: `Done in ... {'ok': 40} (total=40)` — the first 5 are resumed instantly, the rest processed in parallel, no traceback (this exercises the spawn-based pool on macOS).

- [ ] **Step 9: Commit**

```bash
git add src/meld_emotion/data/preprocess.py tests/test_preprocess.py scripts/preprocess_meld.py
git commit -m "Add MELD clip preprocessing (sequential decode, detect, track, cut flags, letterbox) with resumable multiprocess driver"
```

---

### Task 8: Pretrained vision encoders (batched)

**Files:**
- Create: `src/meld_emotion/vision/encoders.py`
- Modify: `pyproject.toml` (via `uv add`)
- Test: `tests/test_encoders.py`

**Interfaces:**
- Produces: `FACE_MODEL_ID`, `CLIP_MODEL_ID`, `FACE_TO_MELD_LABEL: dict[str,
  str]`, `default_device() -> str` (`"mps"` if available else `"cpu"`);
  class `FaceEmotionEncoder(device: str | None = None, batch_size: int =
  64)` with attributes `.labels: tuple[str, ...]` (the checkpoint's own 7
  names, in logit order), `.meld_labels: tuple[str, ...]` (same order,
  MELD spelling), `.feature_dim: int` (768), and methods
  `.encode_batch(images: list[PIL.Image]) -> dict` (keys `"features"`:
  `(N, 768)` float32, `"probs"`: `(N, 7)` float32 rows summing to 1; `N=0`
  gives correctly-shaped empty arrays) and `.encode(image) -> dict`
  (`(768,)`, `(7,)`); class `SceneEncoder(device=None, batch_size=64)` with
  `.feature_dim: int` (512), `.encode_batch(images) -> np.ndarray` `(N,
  512)`, `.encode(image) -> np.ndarray` `(512,)`.

**Note:** the tests download weights from Hugging Face on first run (~340MB
face model, ~600MB CLIP; cached under `~/.cache/huggingface` afterwards) and
are marked `network`, so `uv run pytest` skips them unless `-m network`.

Verified facts this task relies on: the checkpoint is
`ViTForImageClassification` with `id2label` = `sad, disgust, angry, neutral,
fear, surprise, happy` and `hidden_size` 768; CLIP ViT-B/32's
`projection_dim` is 512.

- [ ] **Step 1: Add ML dependencies**

Run: `uv add "torch>=2.2" "transformers>=4.40" "pillow>=10.0"`
Expected: `pyproject.toml` and `uv.lock` updated; torch's arm64 wheel installs (a few hundred MB).

- [ ] **Step 2: Write the failing tests**

Create `tests/test_encoders.py`:

```python
import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.network


def _images(n, seed=0):
    rng = np.random.default_rng(seed)
    return [Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(n)]


@pytest.fixture(scope="module")
def face_encoder():
    from meld_emotion.vision.encoders import FaceEmotionEncoder
    return FaceEmotionEncoder(device="cpu")


@pytest.fixture(scope="module")
def scene_encoder():
    from meld_emotion.vision.encoders import SceneEncoder
    return SceneEncoder(device="cpu")


def test_face_labels_come_from_the_checkpoint_and_map_onto_meld(face_encoder):
    assert set(face_encoder.labels) == {"sad", "disgust", "angry", "neutral", "fear", "surprise", "happy"}
    assert set(face_encoder.meld_labels) == {"sadness", "disgust", "anger", "neutral", "fear", "surprise", "joy"}
    assert face_encoder.feature_dim == 768


def test_face_encode_batch_shapes_and_probabilities(face_encoder):
    out = face_encoder.encode_batch(_images(3))
    assert out["features"].shape == (3, 768)
    assert out["probs"].shape == (3, 7)
    assert out["features"].dtype == np.float32
    assert np.allclose(out["probs"].sum(axis=1), 1.0, atol=1e-4)


def test_face_encode_batch_of_nothing_returns_empty_arrays_with_the_right_width(face_encoder):
    out = face_encoder.encode_batch([])
    assert out["features"].shape == (0, 768)
    assert out["probs"].shape == (0, 7)


def test_face_probs_match_the_checkpoints_own_forward_pass(face_encoder):
    # Guards the CLS-pooling path: our features must be exactly what the
    # checkpoint's classifier consumes (post-LayerNorm CLS), so the probs we
    # return equal the model's own prediction.
    image = _images(1)[0]
    ours = face_encoder.encode(image)["probs"]
    inputs = face_encoder.processor(images=image, return_tensors="pt")
    with torch.no_grad():
        theirs = torch.softmax(face_encoder.model(**inputs).logits, dim=-1)[0].numpy()
    assert np.allclose(ours, theirs, atol=1e-5)


def test_scene_encoder_shapes(scene_encoder):
    assert scene_encoder.feature_dim == 512
    assert scene_encoder.encode_batch(_images(2)).shape == (2, 512)
    assert scene_encoder.encode(_images(1)[0]).shape == (512,)
    assert scene_encoder.encode_batch([]).shape == (0, 512)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_encoders.py -m network -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meld_emotion.vision.encoders'`

- [ ] **Step 4: Implement**

Create `src/meld_emotion/vision/encoders.py`:

```python
"""Pretrained, frozen vision encoders (design doc §4.1).

- FaceEmotionEncoder: dima806/facial_emotions_image_detection (ViT-Base,
  ~86M). Per face crop -> the 768-d post-LayerNorm CLS feature the
  checkpoint's classifier consumes, plus its own 7-way expression
  probabilities (used for the provisional state, the zero-training
  vision-only baseline, and the visual gloss -- never as classifier input).
- SceneEncoder: CLIP ViT-B/32 image tower (~88M). Per letterboxed frame ->
  the 512-d projected image embedding. The full CLIPModel (incl. the ~63M
  text tower) is loaded because the visual-gloss prompt bank needs the text
  tower later; both towers are counted in the parameter budget.

Both run under torch.no_grad and are never trained in this pipeline.
"""
import numpy as np
import torch
from transformers import (AutoImageProcessor, AutoModelForImageClassification,
                          CLIPImageProcessor, CLIPModel)

FACE_MODEL_ID = "dima806/facial_emotions_image_detection"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_BATCH_SIZE = 64

# The checkpoint's label names -> MELD's. Same seven categories, different spelling.
FACE_TO_MELD_LABEL = {
    "sad": "sadness", "disgust": "disgust", "angry": "anger", "neutral": "neutral",
    "fear": "fear", "surprise": "surprise", "happy": "joy",
}


def default_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


class FaceEmotionEncoder:
    def __init__(self, device: str | None = None, batch_size: int = DEFAULT_BATCH_SIZE):
        self.device = device or default_device()
        self.batch_size = batch_size
        self.processor = AutoImageProcessor.from_pretrained(FACE_MODEL_ID)
        self.model = AutoModelForImageClassification.from_pretrained(FACE_MODEL_ID).to(self.device).eval()
        if not (hasattr(self.model, "vit") and hasattr(self.model, "classifier")):
            raise TypeError(f"{FACE_MODEL_ID} loaded as {type(self.model).__name__}; "
                            f"expected ViTForImageClassification with .vit and .classifier")
        cfg = self.model.config
        self.labels = tuple(cfg.id2label[i] for i in range(cfg.num_labels))
        self.meld_labels = tuple(FACE_TO_MELD_LABEL[label] for label in self.labels)
        self.feature_dim = cfg.hidden_size

    @torch.no_grad()
    def encode_batch(self, images: list) -> dict:
        features, probs = [], []
        for chunk in _chunks(list(images), self.batch_size):
            inputs = self.processor(images=chunk, return_tensors="pt").to(self.device)
            hidden = self.model.vit(pixel_values=inputs["pixel_values"]).last_hidden_state
            cls = hidden[:, 0, :]  # post-LayerNorm CLS: exactly what the classifier sees
            logits = self.model.classifier(cls)
            features.append(cls.float().cpu().numpy())
            probs.append(torch.softmax(logits, dim=-1).float().cpu().numpy())
        if not features:
            return {"features": np.zeros((0, self.feature_dim), np.float32),
                    "probs": np.zeros((0, len(self.labels)), np.float32)}
        return {"features": np.concatenate(features).astype(np.float32),
                "probs": np.concatenate(probs).astype(np.float32)}

    def encode(self, image) -> dict:
        out = self.encode_batch([image])
        return {"features": out["features"][0], "probs": out["probs"][0]}


class SceneEncoder:
    def __init__(self, device: str | None = None, batch_size: int = DEFAULT_BATCH_SIZE):
        self.device = device or default_device()
        self.batch_size = batch_size
        self.processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)
        self.model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(self.device).eval()
        self.feature_dim = self.model.config.projection_dim

    @torch.no_grad()
    def encode_batch(self, images: list) -> np.ndarray:
        features = []
        for chunk in _chunks(list(images), self.batch_size):
            inputs = self.processor(images=chunk, return_tensors="pt").to(self.device)
            features.append(self.model.get_image_features(pixel_values=inputs["pixel_values"])
                            .float().cpu().numpy())
        if not features:
            return np.zeros((0, self.feature_dim), np.float32)
        return np.concatenate(features).astype(np.float32)

    def encode(self, image) -> np.ndarray:
        return self.encode_batch([image])[0]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_encoders.py -m network -v`
Expected: 5 passed (first run downloads weights; allow a few minutes).

- [ ] **Step 6: Confirm the default suite still skips them and passes offline**

Run: `uv run pytest -v`
Expected: all Task 1–7 tests pass; `test_encoders.py` reports `5 deselected`.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/meld_emotion/vision/encoders.py tests/test_encoders.py
git commit -m "Add batched frozen face-expression and CLIP scene encoders with checkpoint-derived labels"
```

---

### Task 9: Frozen feature cache builder (batched, resumable)

**Files:**
- Create: `src/meld_emotion/data/cache.py`
- Create: `scripts/build_feature_cache.py`
- Test: `tests/test_cache.py`

**Interfaces:**
- Consumes: `FaceEmotionEncoder`, `SceneEncoder`, `default_device` (Task 8);
  `PREPROCESSED_DIR`, `FEATURE_CACHE_DIR` (Task 1); the clip directory
  layout and `metadata.json` fields from Task 7. Any objects exposing
  `.feature_dim` and `.encode_batch(images)` with the Task 8 return shapes
  are accepted as encoders (how the tests stub them).
- Produces: `build_clip_cache(clip_dir: Path, face_encoder, scene_encoder)
  -> dict | None` (`None` when `metadata.json` status is not `"ok"`); dict
  keys: `dialogue_id: int`, `utterance_id: int`, `face_features (N_faces,
  768) float32`, `face_probs (N_faces, 7) float32`, `face_track_ids
  (N_faces,) int64`, `face_frame_idx (N_faces,) int64`, `scene_features
  (N_frames, 512) float32`, `scene_frame_idx (N_frames,) int64`, `shot_cut
  (N_frames,) bool`; `save_clip_cache(cache: dict, out_path: Path)`;
  `cache_path_for(cache_dir: Path, split: str, clip_name: str) -> Path` (=
  `cache_dir/<split>/<clip_name>.npz`); `build_split_cache(split: str,
  face_encoder, scene_encoder, preprocessed_dir: Path = PREPROCESSED_DIR,
  cache_dir: Path = FEATURE_CACHE_DIR, overwrite: bool = False,
  progress_every: int = 500) -> Counter` (keys `cached`, `skipped_existing`,
  `skipped_not_ok`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cache.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meld_emotion.data.cache'`

- [ ] **Step 3: Implement**

Create `src/meld_emotion/data/cache.py`:

```python
"""Runs the frozen vision encoders over preprocessed crops, one batch per
clip, and persists the results as one .npz per clip. Resumable."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from meld_emotion.config import FEATURE_CACHE_DIR, PREPROCESSED_DIR


def cache_path_for(cache_dir: Path, split: str, clip_name: str) -> Path:
    return Path(cache_dir) / split / f"{clip_name}.npz"


def build_clip_cache(clip_dir: Path, face_encoder, scene_encoder) -> dict | None:
    with open(clip_dir / "metadata.json") as f:
        meta = json.load(f)
    if meta["status"] != "ok":
        return None

    face_images, face_track_ids, face_frame_idx = [], [], []
    scene_images, scene_frame_idx, shot_cut = [], [], []
    for frame in meta["frames"]:
        for face in frame["faces"]:
            face_images.append(Image.open(clip_dir / face["path"]).convert("RGB"))
            face_track_ids.append(face["track_id"])
            face_frame_idx.append(frame["sample_index"])
        scene_images.append(Image.open(clip_dir / frame["scene_path"]).convert("RGB"))
        scene_frame_idx.append(frame["sample_index"])
        shot_cut.append(bool(frame["shot_cut"]))

    faces = face_encoder.encode_batch(face_images)   # (0, D) arrays when there are no faces
    return {
        "dialogue_id": meta["dialogue_id"],
        "utterance_id": meta["utterance_id"],
        "face_features": faces["features"],
        "face_probs": faces["probs"],
        "face_track_ids": np.array(face_track_ids, dtype=np.int64),
        "face_frame_idx": np.array(face_frame_idx, dtype=np.int64),
        "scene_features": scene_encoder.encode_batch(scene_images),
        "scene_frame_idx": np.array(scene_frame_idx, dtype=np.int64),
        "shot_cut": np.array(shot_cut, dtype=bool),
    }


def save_clip_cache(cache: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **cache)


def build_split_cache(split: str, face_encoder, scene_encoder,
                      preprocessed_dir: Path = PREPROCESSED_DIR, cache_dir: Path = FEATURE_CACHE_DIR,
                      overwrite: bool = False, progress_every: int = 500) -> Counter:
    counts: Counter = Counter()
    clip_dirs = sorted(p for p in (Path(preprocessed_dir) / split).iterdir()
                       if (p / "metadata.json").exists())
    for i, clip_dir in enumerate(clip_dirs, 1):
        out_path = cache_path_for(cache_dir, split, clip_dir.name)
        if out_path.exists() and not overwrite:
            counts["skipped_existing"] += 1
        else:
            cache = build_clip_cache(clip_dir, face_encoder, scene_encoder)
            if cache is None:
                counts["skipped_not_ok"] += 1
            else:
                save_clip_cache(cache, out_path)
                counts["cached"] += 1
        if progress_every and i % progress_every == 0:
            print(f"  {i}/{len(clip_dirs)} {dict(counts)}", flush=True)
    return counts
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cache.py -v`
Expected: 5 passed

- [ ] **Step 5: Add the CLI driver**

Create `scripts/build_feature_cache.py`:

```python
#!/usr/bin/env python3
"""Build the frozen vision feature cache for one MELD split from already
preprocessed clips (run scripts/preprocess_meld.py first). Resumable: clips
with an existing .npz are skipped unless --overwrite.

Usage:
    uv run python scripts/build_feature_cache.py --split dev
"""
import argparse
import time

from meld_emotion.config import FEATURE_CACHE_DIR, SPLITS
from meld_emotion.data.cache import build_split_cache
from meld_emotion.vision.encoders import FaceEmotionEncoder, SceneEncoder, default_device


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--device", default=default_device(), help="'mps' or 'cpu'")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print(f"Loading encoders on {args.device}...")
    face_encoder = FaceEmotionEncoder(device=args.device)
    scene_encoder = SceneEncoder(device=args.device)

    started = time.perf_counter()
    counts = build_split_cache(args.split, face_encoder, scene_encoder, overwrite=args.overwrite)
    elapsed = time.perf_counter() - started
    print(f"Done in {elapsed / 60:.1f} min: {dict(counts)} -> {FEATURE_CACHE_DIR / args.split}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Smoke-test the CLI against the slice preprocessed in Task 7**

Run: `uv run python scripts/build_feature_cache.py --split dev`
Expected: `Done in ... min: {'cached': 40} -> .../data/meld/features/dev` (matching however many clips Task 7 Step 8 preprocessed) and `data/meld/features/dev/` contains that many `.npz` files.
Run it again: expected `{'skipped_existing': 40}` in well under a second of work.

- [ ] **Step 7: Commit**

```bash
git add src/meld_emotion/data/cache.py tests/test_cache.py scripts/build_feature_cache.py
git commit -m "Add batched, resumable frozen-feature cache builder with CLI driver"
```

---

### Task 10: Per-split manifest and pipeline statistics

The training plan needs one file per split that says, for every labeled
utterance: its text, its dialogue context, its labels, and where (if
anywhere) its cached features live. This task also produces the numbers the
design doc (§6) says will be measured rather than assumed — faces per frame,
zero-face clip rate, shot-cut rate.

**Files:**
- Create: `src/meld_emotion/data/manifest.py`
- Create: `scripts/build_manifest.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Consumes: `Utterance`, `load_split`, `group_by_dialogue`, `context_window`
  (Task 3); `build_video_index` (Task 4); `clip_dir_for` and the
  `metadata.json` fields (Task 7); `cache_path_for` (Task 9); `LABELS_DIR`,
  `PREPROCESSED_DIR`, `FEATURE_CACHE_DIR`, `split_video_dir` (Task 1).
- Produces: `CONTEXT_MAX = 8`; `build_manifest(split: str, utterances:
  list[Utterance], index: dict[tuple[int, int], Path], preprocessed_dir:
  Path, cache_dir: Path) -> list[dict]` — one row per utterance, in CSV
  order, with keys `split, dialogue_id, utterance_id, speaker, text,
  context_prev (list[str], up to CONTEXT_MAX previous texts oldest-first,
  current excluded), emotion, sentiment, status, feature_path (str relative
  to cache_dir, or None), n_frames, n_faces, n_shot_cuts`; `status` is one
  of `ok | not_cached | no_frames | decode_failed | not_preprocessed |
  missing_video`; `write_manifest(rows, out_path: Path)` /
  `read_manifest(path: Path) -> list[dict]` (JSON lines);
  `manifest_stats(rows) -> dict` with keys `utterances`, `status`
  (dict), `faces_per_frame`, `zero_face_clip_fraction`,
  `clips_with_a_cut_fraction`, `shot_cuts_per_frame` (the last four over
  `ok` rows only).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_manifest.py`:

```python
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
        json.dump({"status": "ok", "n_shot_cuts": 1, "frames": [
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


def test_write_and_read_manifest_round_trip(tmp_path):
    from meld_emotion.data.manifest import build_manifest, read_manifest, write_manifest
    utterances, index, pre, cache = _fixture(tmp_path)
    rows = build_manifest("dev", utterances, index, preprocessed_dir=pre, cache_dir=cache)
    out = tmp_path / "manifest.jsonl"
    write_manifest(rows, out)
    assert read_manifest(out) == rows
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meld_emotion.data.manifest'`

- [ ] **Step 3: Implement**

Create `src/meld_emotion/data/manifest.py`:

```python
"""Per-split manifest: one JSON line per labeled utterance joining its labels
and dialogue context to its preprocessed metadata and cached features, plus
the pipeline statistics design doc §6 says will be measured, not assumed."""
import json
from collections import Counter
from pathlib import Path

from meld_emotion.data.cache import cache_path_for
from meld_emotion.data.labels import Utterance, context_window, group_by_dialogue
from meld_emotion.data.preprocess import clip_dir_for

CONTEXT_MAX = 8  # previous utterances stored; training slices context_prev[-k:] for any k <= 8


def build_manifest(split: str, utterances: list[Utterance], index: dict,
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
            "status": None, "feature_path": None, "n_frames": 0, "n_faces": 0, "n_shot_cuts": 0,
        }
        meta_path = clip_dir / "metadata.json"
        if (u.dialogue_id, u.utterance_id) not in index:
            row["status"] = "missing_video"
        elif not meta_path.exists():
            row["status"] = "not_preprocessed"
        else:
            with open(meta_path) as f:
                meta = json.load(f)
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
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_manifest.py -v`
Expected: 4 passed

- [ ] **Step 5: Add the CLI driver**

Create `scripts/build_manifest.py`:

```python
#!/usr/bin/env python3
"""Write data/meld/features/<split>/manifest.jsonl joining every labeled
utterance to its cached features, and print pipeline statistics.

Usage:
    uv run python scripts/build_manifest.py --split dev
"""
import argparse
import json

from meld_emotion.config import FEATURE_CACHE_DIR, LABELS_DIR, PREPROCESSED_DIR, SPLITS, split_video_dir
from meld_emotion.data.labels import load_split
from meld_emotion.data.manifest import build_manifest, manifest_stats, write_manifest
from meld_emotion.data.video_index import build_video_index


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=SPLITS, required=True)
    args = parser.parse_args()

    utterances = load_split(args.split, labels_dir=LABELS_DIR)
    index = build_video_index(split_video_dir(args.split))
    rows = build_manifest(args.split, utterances, index,
                          preprocessed_dir=PREPROCESSED_DIR, cache_dir=FEATURE_CACHE_DIR)
    out_path = FEATURE_CACHE_DIR / args.split / "manifest.jsonl"
    write_manifest(rows, out_path)
    print(f"Wrote {len(rows)} rows -> {out_path}")
    print(json.dumps(manifest_stats(rows), indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Smoke-test against the slice cached in Task 9**

Run: `uv run python scripts/build_manifest.py --split dev`
Expected: `Wrote 1109 rows -> .../features/dev/manifest.jsonl`, then a stats JSON whose `status` shows `"ok": 40` (the Task 7/9 slice), `"not_preprocessed"` for the rest, and `"missing_video": 1` (`dia110_utt7`). `faces_per_frame` should be a plausible 1–3 for those 40 clips.

- [ ] **Step 7: Run the complete offline suite one last time**

Run: `uv run pytest -v`
Expected: every test in Tasks 1–7, 9, 10 passes; Task 8's 5 tests deselected; total wall-clock a few seconds.

- [ ] **Step 8: Commit**

```bash
git add src/meld_emotion/data/manifest.py tests/test_manifest.py scripts/build_manifest.py
git commit -m "Add per-split manifest joining labels to cached features, with pipeline statistics"
```

---

## Running the full pipeline for real (not part of the automated test suite)

Once all tasks are merged, the full data/feature pipeline per split is:

```bash
uv run python scripts/preprocess_meld.py --split dev --workers 8
uv run python scripts/build_feature_cache.py --split dev
uv run python scripts/build_manifest.py --split dev
```

Repeat for `train` and `test`. Design doc §6 budgets under two hours for all
three splits combined (preprocessing ~30–60 min with workers, cache ~45–60
min on MPS); both steps are resumable, so a crash mid-way costs only the
clips not yet done.

**Acceptance checks, from the manifest stats:**
- `status.ok` + known exceptions equals the row count: dev `ok` = 1108 with
  `missing_video` = 1 (`dia110_utt7`); train `ok` = 9989; test `ok` = 2610.
  Any `decode_failed` / `no_frames` rows are listed in the manifest — inspect
  them with `scripts/view_meld_clips.py` before deciding whether to drop them.
- `faces_per_frame` (the design doc guessed ~1.5; the first 24 dev clips are
  ensemble scenes at 4.4, so expect the split-wide number to land in between)
  and `zero_face_clip_fraction`
  (expected a few percent) go into the write-up as the measured values the
  design doc §6 left open.
- `shot_cuts_per_frame` sanity-checks `SHOT_CUT_THRESHOLD` (0.3, chosen from
  a 24-clip sample — see Task 6): if it is ~0 the threshold never fires; if it
  is ≳0.3 it fires on ordinary motion. Every frame's raw `shot_change` score
  is stored in its `metadata.json`, so the split-wide histogram can be
  plotted without decoding anything; re-run `preprocess_meld.py --overwrite`
  if the threshold changes (track IDs depend on the resets).

## Self-Review

**Spec coverage.** Design doc §5 (per-split indexing, timestamps ignored,
decode failures recorded, `dia110_utt7`) → Tasks 3, 4, 7, 10. §4.2 (detect
every face at 0.75, letterboxed scene frame, IoU tracking with shot-cut reset,
cut flags for the §7 ablation) → Tasks 5, 6, 7. §4.1 (two frozen encoders,
why; parameter counts) → Task 8. §6 preprocessing pass, feature cache, and
the measured-not-assumed statistics → Tasks 7, 9, 10. §4.3 dialogue context
→ Task 3 (window) and Task 10 (`context_prev` stored per row). §2's
provisional expression state and §7's zero-training vision-only baseline both
need the face encoder's own probabilities in MELD label order → Task 8
(`meld_labels`) and Task 9 (`face_probs` cached). Text encoding (RoBERTa),
the fusion model, Stage 1/2 training, ablations, and batch evaluation are the
next plan: RoBERTa's top layers are fine-tuned during training rather than
cached (design doc §6), so they don't belong in a frozen-feature pipeline.

**Placeholder scan.** No TBD/TODO markers; every step has complete, runnable
code and an exact command with expected output.

**Type consistency.** `build_video_index(split_dir: Path) -> dict[(int, int),
Path]` (Task 4) is what Task 7's `preprocess_split(index=...)` and Task 10's
`build_manifest(index=...)` consume. `detect_faces(detector, frame_bgr) ->
list[(x, y, w, h, score)]` (Task 5) is unpacked exactly that way in Task 7.
`FaceTracker.update(boxes) -> list[int]` and `shot_change_score(prev, curr)
-> float` / `SHOT_CUT_THRESHOLD` (Task 6) match their use in Task 7.
`clip_dir_for` (Task 7) and `cache_path_for` (Task 9) are the only two
places that spell out the on-disk layout, and Task 10 uses both rather than
re-deriving paths. The `metadata.json` fields written in Task 7
(`status`, `n_shot_cuts`, `frames[].sample_index`, `frames[].shot_cut`,
`frames[].faces[].{track_id, path}`, `frames[].scene_path`) are exactly the
fields Task 9's `build_clip_cache` and Task 10's `build_manifest` read.
`FaceEmotionEncoder.encode_batch` / `SceneEncoder.encode_batch` (Task 8)
return the `(N, 768)`/`(N, 7)`/`(N, 512)` arrays — including `N=0` — that
Task 9 stores without reshaping, and the stubs in `tests/test_cache.py`
mirror those shapes.
