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
