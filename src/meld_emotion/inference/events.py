"""JSON-lines event stream shared by both demo frontends (design doc §4.5).

Four phases per turn: `provisional` (during the turn, one per sampled
frame), `final` (state is ready -- response is deliberately absent so the
LM never gates it), `token` (one incremental response chunk), `done`
(closes the turn with the full response and the three measured latencies).
"""
import json
from dataclasses import dataclass
from typing import TextIO


class EventEmitter:
    def __init__(self, out: TextIO):
        self.out = out

    def emit(self, event: dict) -> None:
        self.out.write(json.dumps(event) + "\n")
        self.out.flush()

    def provisional(self, turn_id: str, frame: int, faces_seen: int,
                    provisional_expression: dict) -> dict:
        event = {"turn_id": turn_id, "phase": "provisional", "frame": frame,
                 "faces_seen": faces_seen, "provisional_expression": provisional_expression}
        self.emit(event)
        return event

    def final(self, turn_id: str, text: str, emotion: str, emotion_probs: dict,
              sentiment: str, sentiment_probs: dict, faces_seen: int,
              visual_cues: list[str]) -> dict:
        event = {"turn_id": turn_id, "phase": "final", "text": text, "emotion": emotion,
                 "emotion_probs": emotion_probs, "sentiment": sentiment,
                 "sentiment_probs": sentiment_probs, "faces_seen": faces_seen,
                 "visual_cues": visual_cues}
        self.emit(event)
        return event

    def token(self, turn_id: str, text: str) -> None:
        self.emit({"turn_id": turn_id, "phase": "token", "text": text})

    def done(self, turn_id: str, response: str, latency_ms: dict, prompt: str | None = None) -> dict:
        event = {"turn_id": turn_id, "phase": "done", "response": response,
                 "latency_ms": latency_ms}
        if prompt is not None:
            # Additive field (not part of the design doc's original `done` schema): the
            # exact user-turn prompt text sent to the LM, so response_rubric.py can score
            # against the real prompt instead of reconstructing an approximation.
            event["prompt"] = prompt
        self.emit(event)
        return event


@dataclass
class LatencyStamps:
    turn_start: float
    state_emitted: float | None = None
    first_token_emitted: float | None = None
    done_emitted: float | None = None

    def as_ms(self) -> dict:
        def delta(stamp):
            return None if stamp is None else round((stamp - self.turn_start) * 1000)
        return {"state": delta(self.state_emitted), "first_token": delta(self.first_token_emitted),
                "done": delta(self.done_emitted)}
