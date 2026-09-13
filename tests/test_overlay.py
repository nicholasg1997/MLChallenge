import numpy as np


def test_draw_overlay_draws_boxes_label_status_and_captions_in_place():
    from meld_emotion.inference.overlay import draw_overlay
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    draw_overlay(frame, [(50, 40, 30, 30, 0.9)], [3], "face: joy (34%)", ["a caption", "", "second line"], status="listening")
    assert frame.any()
    assert frame[40, 50:80].any() and frame[40:70, 50].any()   # the box edges
    assert frame[-30:, :].any()                                 # the caption band


def test_draw_overlay_is_a_no_op_with_nothing_to_draw():
    from meld_emotion.inference.overlay import draw_overlay
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    draw_overlay(frame, [], [], None, [])
    assert not frame.any()


def test_face_label_reports_none_neutral_or_the_thresholded_emotion():
    from meld_emotion.inference.overlay import face_label_for
    smiling = {"neutral": 0.44, "joy": 0.34, "surprise": 0.1, "anger": 0.05, "sadness": 0.04, "disgust": 0.02, "fear": 0.01}
    assert face_label_for(0, smiling, 0.25) == "face: none"
    assert face_label_for(1, smiling, 0.25) == "face: joy (34%)"
    assert face_label_for(1, smiling, 0.4) == "face: neutral (44%)"
