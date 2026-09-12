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
