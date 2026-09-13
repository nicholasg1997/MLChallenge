import numpy as np
import pytest


def test_percentile_matches_known_values():
    from scripts.measure_latency import percentile
    values = list(range(1, 101))
    assert percentile(values, 50) == pytest.approx(50.5, abs=0.5)
    assert percentile(values, 95) == pytest.approx(95.5, abs=0.5)


def test_summarise_latencies_computes_p50_p95_per_event():
    from scripts.measure_latency import summarise_latencies
    done_events = [{"latency_ms": {"state": 80, "first_token": 600, "done": 1800}},
                   {"latency_ms": {"state": 90, "first_token": 700, "done": 2000}},
                   {"latency_ms": {"state": 100, "first_token": 800, "done": 2200}}]
    summary = summarise_latencies(done_events)
    assert summary["state_p50"] == pytest.approx(90, abs=1)
    assert summary["done_p95"] >= summary["done_p50"] >= 1800


def test_time_one_frame_runs_a_real_push_frame(monkeypatch):
    from meld_emotion.inference import turn as turn_module
    from scripts.measure_latency import time_one_frame
    from test_turn import _Collector, _bundle
    monkeypatch.setattr(turn_module, "detect_faces", lambda detector, frame: [(0, 0, 10, 10, 0.9)])
    tp = turn_module.TurnProcessor(_bundle(), emitter=_Collector())
    tp.start_turn("t")
    seconds = time_one_frame(tp, np.zeros((48, 64, 3), dtype=np.uint8))
    assert seconds > 0 and tp.max_faces_seen == 1
