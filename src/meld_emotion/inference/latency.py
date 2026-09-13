"""Latency summaries over the event stream (design doc §9): p50/p95 per
end-of-turn event from `done` events' latency_ms, used by the replay
measurement and the live demo's run report."""
import numpy as np


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, p))


def summarise_latencies(done_events: list[dict]) -> dict:
    out = {}
    for key in ("state", "first_token", "done"):
        values = [e["latency_ms"][key] for e in done_events if e["latency_ms"].get(key) is not None]
        out[f"{key}_p50"] = percentile(values, 50)
        out[f"{key}_p95"] = percentile(values, 95)
    return out
