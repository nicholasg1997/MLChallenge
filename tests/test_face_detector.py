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
