import numpy as np


def _detector(flags, **kw):
    """An EndpointDetector whose VAD answers from a scripted list of voiced flags."""
    from meld_emotion.inference.audio import EndpointDetector
    it = iter(flags)
    return EndpointDetector(lambda frame: next(it), **kw)


def _frame(value: int) -> bytes:
    return np.full(480, value, dtype=np.int16).tobytes()


def test_turn_starts_after_start_ms_of_speech_and_keeps_the_pre_roll():
    # 3 unvoiced frames, then voiced: start_ms=90 -> 3 voiced frames to open; pre_roll_ms=60 -> 2 frames kept
    flags = [False, False, False, True, True, True]
    det = _detector(flags, start_ms=90, pre_roll_ms=60, end_silence_ms=60)
    events = [det.feed(_frame(i)) for i in range(6)]
    assert events[:5] == [None] * 5 and events[5] == ("start", None)
    assert det.speaking
    # the buffer is the pre-roll deque: the last 2 frames fed, the opening frame included
    assert det.force_end() == _frame(4) + _frame(5)


def test_turn_ends_after_end_silence_and_returns_the_whole_utterance():
    flags = [True, True, True, True, False, False, False]
    det = _detector(flags, start_ms=60, pre_roll_ms=0, end_silence_ms=90)
    out = [det.feed(_frame(i)) for i in range(7)]
    assert out[1] == ("start", None)
    assert out[-1][0] == "end"
    assert out[-1][1] == b"".join(_frame(i) for i in range(1, 7))   # frames 1..6: from the start frame to the last silence
    assert not det.speaking


def test_short_voiced_blips_do_not_start_a_turn():
    flags = [True, False, True, False, True, False]
    det = _detector(flags, start_ms=60)
    assert all(det.feed(_frame(0)) is None for _ in flags)


def test_a_pause_shorter_than_end_silence_does_not_split_the_turn():
    flags = [True, True, False, False, True, True, False, False, False, False]
    det = _detector(flags, start_ms=60, pre_roll_ms=0, end_silence_ms=120)
    out = [det.feed(_frame(0)) for _ in flags]
    assert sum(1 for o in out if o and o[0] == "start") == 1
    assert sum(1 for o in out if o and o[0] == "end") == 1


def test_max_turn_ends_a_turn_that_never_goes_quiet():
    det = _detector([True] * 50, start_ms=30, pre_roll_ms=0, end_silence_ms=3000, max_turn_s=0.3)   # 10 frames
    out = [det.feed(_frame(0)) for _ in range(12)]
    ends = [o for o in out if o and o[0] == "end"]
    assert len(ends) == 1 and len(ends[0][1]) == 10 * 960


def test_force_end_returns_none_when_no_turn_is_open():
    det = _detector([False])
    assert det.force_end() is None


def test_pcm16_to_float_scales_to_unit_range():
    from meld_emotion.inference.audio import pcm16_to_float
    audio = pcm16_to_float(np.array([0, 16384, -32768], dtype=np.int16).tobytes())
    assert audio.dtype == np.float32
    assert list(audio) == [0.0, 0.5, -1.0]
