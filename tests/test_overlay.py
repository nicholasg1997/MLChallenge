import numpy as np


def test_draw_overlay_draws_boxes_bars_status_and_captions_in_place():
    from meld_emotion.inference.overlay import draw_overlay
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    provisional = {"neutral": 0.5, "joy": 0.2, "surprise": 0.1, "anger": 0.1, "sadness": 0.05, "disgust": 0.03, "fear": 0.02}
    draw_overlay(frame, [(50, 40, 30, 30, 0.9)], [3], provisional, ["a caption", "", "second line"], status="listening")
    assert frame.any()                                # something was drawn
    assert frame[40, 50:80].any() and frame[40:70, 50].any()   # the box edges
    assert frame[-3, 5:].any() or frame[-30:, :].any()          # the caption band


def test_draw_overlay_is_a_no_op_with_nothing_to_draw():
    from meld_emotion.inference.overlay import draw_overlay
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    draw_overlay(frame, [], [], None, [])
    assert not frame.any()
