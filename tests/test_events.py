import io
import json

import pytest


def _lines(buf):
    return [json.loads(l) for l in buf.getvalue().strip().splitlines()]


def test_provisional_event_has_exactly_the_spec_fields():
    from meld_emotion.inference.events import EventEmitter
    buf = io.StringIO()
    EventEmitter(buf).provisional("dia1_utt1", 3, 2, {"joy": 0.6, "neutral": 0.4})
    events = _lines(buf)
    assert len(events) == 1
    assert events[0] == {"turn_id": "dia1_utt1", "phase": "provisional", "frame": 3,
                         "faces_seen": 2, "provisional_expression": {"joy": 0.6, "neutral": 0.4}}


def test_final_event_never_carries_response_or_latency():
    from meld_emotion.inference.events import EventEmitter
    buf = io.StringIO()
    EventEmitter(buf).final("dia1_utt1", "hi there", "joy", {"joy": 0.9}, "positive",
                            {"positive": 0.9}, 1, ["one person"])
    event = _lines(buf)[0]
    assert event["phase"] == "final"
    assert "response" not in event and "latency_ms" not in event


def test_token_and_done_events_carry_the_response_and_latencies():
    from meld_emotion.inference.events import EventEmitter
    buf = io.StringIO()
    emitter = EventEmitter(buf)
    emitter.token("dia1_utt1", "Wow")
    emitter.token("dia1_utt1", " really?")
    emitter.done("dia1_utt1", "Wow really?", {"state": 80, "first_token": 600, "done": 1800})
    events = _lines(buf)
    assert [e["phase"] for e in events] == ["token", "token", "done"]
    assert events[0]["text"] == "Wow" and events[1]["text"] == " really?"
    assert events[2] == {"turn_id": "dia1_utt1", "phase": "done", "response": "Wow really?",
                         "latency_ms": {"state": 80, "first_token": 600, "done": 1800}}


def test_latency_stamps_report_none_until_set_then_millisecond_deltas():
    from meld_emotion.inference.events import LatencyStamps
    stamps = LatencyStamps(turn_start=10.0)
    assert stamps.as_ms() == {"state": None, "first_token": None, "done": None}
    stamps.state_emitted = 10.084
    stamps.first_token_emitted = 10.61
    stamps.done_emitted = 11.82
    ms = stamps.as_ms()
    assert ms == {"state": 84, "first_token": 610, "done": 1820}
