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
