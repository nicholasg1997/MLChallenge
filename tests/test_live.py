import time

import numpy as np
import pytest
import torch

from test_turn import _bundle, _fake_frame, _Collector


@pytest.fixture(autouse=True)
def _stub_vision_and_text(monkeypatch):
    """The stub bundle has no YuNet or RoBERTa tokenizer: one face per frame, whitespace text ids."""
    from conftest import whitespace_encode
    from meld_emotion.inference import turn as turn_module
    monkeypatch.setattr(turn_module, "detect_faces", lambda detector, frame: [(0, 0, 10, 10, 0.9)])
    monkeypatch.setattr(turn_module, "encode_text",
                        lambda tokenizer, context, current, max_length: whitespace_encode(context, current))


class _FakeTranscriber:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []

    def transcribe(self, audio):
        self.calls.append(len(audio))
        return self.texts.pop(0)

    def warm_up(self):
        pass


class _FakeResponder:
    def __init__(self, chunks=("Oh", " wow", "!")):
        self.chunks = chunks
        self.prompts = []

    def stream(self, messages):
        self.prompts.append(messages)
        for c in self.chunks:
            time.sleep(0.005)
            yield c


def _session(texts=("hello there",), responder=None, **kw):
    from meld_emotion.inference.audio import EndpointDetector
    from meld_emotion.inference.live import LiveSession
    bundle = _bundle(use_scene=False)
    bank = torch.nn.functional.normalize(torch.randn(3, 4), dim=-1)
    flags = kw.pop("flags", [])
    it = iter(flags)
    endpoint = EndpointDetector(lambda f: next(it, False), start_ms=60, pre_roll_ms=0, end_silence_ms=60)
    collector = _Collector()
    return LiveSession(bundle, bank, _FakeTranscriber(texts), endpoint, collector, responder=responder, **kw), collector


def _pcm():
    return np.zeros(480, dtype=np.int16).tobytes()


def test_frames_are_sampled_at_the_interval_and_emit_provisional_events():
    session, events = _session(sample_interval_s=0.1)
    assert session.on_frame(_fake_frame(), now=0.0) is True
    assert session.on_frame(_fake_frame(), now=0.05) is False
    assert session.on_frame(_fake_frame(), now=0.11) is True
    assert [e["phase"] for e in events.events] == ["provisional", "provisional"]
    assert events.events[0]["turn_id"] == "live001"
    assert len(session.stats["vision_s"]) == 2


def test_vad_end_of_speech_runs_asr_then_final_then_done_without_an_lm():
    flags = [True, True, False, False]
    session, events = _session(texts=("what are you doing",), flags=flags)
    session.on_frame(_fake_frame(), now=0.0)
    results = [session.on_audio(_pcm()) for _ in flags]
    assert session.status == "listening"
    final = results[-1]
    assert final is not None and final["phase"] == "final" and final["text"] == "what are you doing"
    assert final["turn_id"] == "live001"
    phases = [e["phase"] for e in events.events]
    assert phases[-2:] == ["final", "done"]
    done = events.events[-1]
    assert done["response"] == "" and done["latency_ms"]["state"] is not None and done["latency_ms"]["first_token"] is None
    assert "what are you doing" in done["prompt"]
    assert session.context == ["what are you doing"]
    assert session.tp.turn_id == "live002"                     # watching continues under a new turn id
    assert len(session.stats["asr_s"]) == 1


def test_speech_start_resets_the_vision_window_to_this_utterance():
    flags = [True, True]
    session, events = _session(flags=flags)
    session.on_frame(_fake_frame(), now=0.0)                  # idle frame: accumulated before speech
    assert session.tp.max_faces_seen == 1 and len(session.tp._scene_features) == 1
    session.on_audio(_pcm()); session.on_audio(_pcm())          # speech start
    assert session.status == "speaking"
    assert len(session.tp._scene_features) == 0               # window reset; same turn id
    assert session.tp.turn_id == "live001"


def test_empty_transcript_ends_the_turn_quietly():
    flags = [True, True, False, False]
    session, events = _session(texts=("",), flags=flags)
    results = [session.on_audio(_pcm()) for _ in flags]
    assert results[-1] is None
    assert session.status == "listening" and session.context == []
    assert all(e["phase"] != "final" for e in events.events)


def test_response_streams_from_the_worker_into_the_caption_and_context_feeds_the_next_prompt():
    responder = _FakeResponder()
    flags = [True, True, False, False] * 2
    session, events = _session(texts=("first line", "second line"), flags=flags, responder=responder)
    try:
        for _ in range(4):
            session.on_audio(_pcm())
        assert session.status == "responding"
        deadline = time.time() + 2
        while session.status != "listening" and time.time() < deadline:
            session.drain_tokens()
            time.sleep(0.01)
        assert session.status == "listening"
        assert session.lines[1] == "Oh wow!"
        assert session.lines[2].startswith("state ")
        done = [e for e in events.events if e["phase"] == "done"][0]
        assert done["response"] == "Oh wow!" and done["latency_ms"]["first_token"] is not None
        assert [e["text"] for e in events.events if e["phase"] == "token"] == ["Oh", " wow", "!"]

        for _ in range(4):
            session.on_audio(_pcm())
        deadline = time.time() + 2
        while session.status != "listening" and time.time() < deadline:
            session.drain_tokens()
            time.sleep(0.01)
        assert "first line" in responder.prompts[1][-1]["content"]      # the earlier turn is context now
        assert "second line" in responder.prompts[1][-1]["content"]
        assert session.context == ["first line", "second line"]
    finally:
        session.close()


def test_push_to_talk_collects_audio_between_toggles_and_never_runs_the_vad():
    # every frame "voiced": in VAD mode this would open and close turns on its own
    session, events = _session(texts=("ptt line",), flags=[True] * 50, push_to_talk=True)
    assert session.status == "space to talk"
    for _ in range(6):
        assert session.on_audio(_pcm()) is None          # idle: audio is dropped, no VAD turn starts
    assert session.tp.turn_id == "live001" and session.status == "space to talk"
    assert session.toggle_push_to_talk() is None
    assert session.status == "recording (space to send)"
    for _ in range(5):
        assert session.on_audio(_pcm()) is None
    final = session.toggle_push_to_talk()
    assert final["text"] == "ptt line"
    assert session.transcriber.calls == [5 * 480]
    assert session.status == "space to talk"


def test_empty_transcript_drops_that_windows_faces():
    flags = [True, True, False, False]
    session, events = _session(texts=("",), flags=flags)
    session.on_audio(_pcm()); session.on_audio(_pcm())       # speech start
    session.on_frame(_fake_frame(), now=0.0)                  # a face seen during the (silent) turn
    assert session.tp.max_faces_seen == 1
    session.on_audio(_pcm()); session.on_audio(_pcm())       # end: ASR hears nothing
    assert session.tp.max_faces_seen == 0 and session.tp.turn_id == "live001"


def test_done_event_carries_the_asr_split_and_the_caption_shows_it():
    flags = [True, True, False, False]
    session, events = _session(texts=("hello",), flags=flags)
    for _ in flags:
        session.on_audio(_pcm())
    done = [e for e in events.events if e["phase"] == "done"][0]
    assert 0 <= done["latency_ms"]["asr"] <= done["latency_ms"]["state"]
    assert session.lines[2].startswith("state ") and "(ASR " in session.lines[2]


def test_face_label_follows_the_threshold():
    session, _ = _session(face_threshold=0.0)
    assert session.face_label() == "face: none"
    session.on_frame(_fake_frame(), now=0.0)
    assert session.face_label().startswith("face: ") and "none" not in session.face_label()


def test_warm_up_touches_every_model_and_emits_nothing():
    from meld_emotion.inference.live import warm_up
    responder = _FakeResponder()
    session, events = _session()
    warm_up(session.bundle, session.transcriber, responder)
    assert len(responder.prompts) == 1
    assert events.events == []                                 # the throwaway processor's events go nowhere
    assert session.tp.turn_id == "live001" and len(session.tp._scene_features) == 0
