"""The live path's turn loop (design doc §2, §8.1 #2): webcam frames sampled
at ~3 fps into the same TurnProcessor the replay uses, microphone audio
into an EndpointDetector, and at end-of-speech: ASR -> end_turn (the state
event, whose latency includes ASR) -> the response LM on a background
thread so the camera keeps running while it streams. Context for the text
encoder is the previous live turns' transcripts.

Vision runs between turns too (the face reading keeps moving); the window
the classifier sees is reset at speech start, which is what MELD's clips
were -- one utterance's worth of frames.

The face reading shown on screen and handed to the LM as the face cue comes
from `face_reader` when one is given: the ORIGINAL face encoder's own 7-way
head (a posed-expression classifier), averaged over the faces in the frame
and EMA-smoothed. The fusion model's text-masked reading, used on MELD
replay, is prior-locked on a live face (surprise/anger/sadness/fear never
left 0.04-0.10 over a 658-frame session), while the head responds to posed
expressions -- and is known-bad on MELD actors mid-sentence, which is why
the replay does not use it. The provisional *event* is the fusion model's
either way."""
import queue
import threading
import time
from typing import Optional

import numpy as np
from PIL import Image

from meld_emotion.inference.audio import pcm16_to_float
from meld_emotion.inference.events import LatencyStamps
from meld_emotion.inference.gloss import (FACE_HEAD_READING_THRESHOLD, FACE_READING_THRESHOLD, face_gloss,
                                          scene_gloss)
from meld_emotion.inference.responder import build_prompt
from meld_emotion.inference.turn import TurnProcessor

SAMPLE_INTERVAL_S = 1 / 3          # ~3 fps, as in preprocessing (SAMPLE_FPS)


class ResponseWorker:
    """Streams responses on one background thread, in submission order.
    Tokens are mirrored onto `.tokens` as (turn_id, chunk) -- chunk None
    closes the turn -- for the display loop to drain."""
    def __init__(self, responder, emitter):
        self.responder, self.emitter = responder, emitter
        self._jobs: queue.Queue = queue.Queue()
        self.tokens: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, turn_id: str, messages: list[dict], stamps: LatencyStamps) -> None:
        self._jobs.put((turn_id, messages, stamps))

    def close(self, timeout: float = 30.0) -> None:
        self._jobs.put(None)
        self._thread.join(timeout)

    def _run(self) -> None:
        while (job := self._jobs.get()) is not None:
            turn_id, messages, stamps = job
            text = ""
            try:
                for i, chunk in enumerate(self.responder.stream(messages)):
                    if i == 0:
                        stamps.first_token_emitted = time.perf_counter()
                    text += chunk
                    self.emitter.token(turn_id, chunk)
                    self.tokens.put((turn_id, chunk))
            finally:
                stamps.done_emitted = time.perf_counter()
                self.emitter.done(turn_id, text, stamps.as_ms(), prompt=messages[-1]["content"])
                self.tokens.put((turn_id, None))


class LiveSession:
    def __init__(self, bundle, bank_embeddings, transcriber, endpoint, emitter, *, responder=None,
                 push_to_talk: bool = False, face_reader=None, face_threshold: float | None = None,
                 face_ema: float = 0.5, sample_interval_s: float = SAMPLE_INTERVAL_S):
        self.bundle, self.bank_embeddings, self.transcriber, self.endpoint, self.emitter = \
            bundle, bank_embeddings, transcriber, endpoint, emitter
        self.tp = TurnProcessor(bundle, emitter)
        self.worker = ResponseWorker(responder, emitter) if responder is not None else None
        self.push_to_talk = push_to_talk
        self.face_reader, self.face_ema = face_reader, face_ema
        self.face_threshold = face_threshold if face_threshold is not None else \
            (FACE_HEAD_READING_THRESHOLD if face_reader is not None else FACE_READING_THRESHOLD)
        self.face_probs: Optional[dict] = None      # the reader's smoothed 7-way reading, MELD labels
        self.sample_interval_s = sample_interval_s
        self.context: list[str] = []          # previous live turns' transcripts
        self.turn_n = 0
        self.status = self._idle_status()
        self.lines: list[str] = []            # caption lines for the overlay
        self.stats = {"vision_s": [], "asr_s": []}
        self._last_sample = -float("inf")
        self._ptt_buffer: Optional[list[bytes]] = None
        self._responding: Optional[str] = None
        self._stamps: dict[str, LatencyStamps] = {}
        self.tp.start_turn(self._next_turn_id())

    def _next_turn_id(self) -> str:
        self.turn_n += 1
        return f"live{self.turn_n:03d}"

    def _idle_status(self) -> str:
        return "space to talk" if self.push_to_talk else "listening"

    # --- vision ---
    def on_frame(self, frame_bgr: np.ndarray, now: float) -> bool:
        """Pushes the frame through the vision pipeline if a sampling slot has
        elapsed. Returns True when it did."""
        if now - self._last_sample < self.sample_interval_s:
            return False
        self._last_sample = now
        start = time.perf_counter()
        self.tp.push_frame(frame_bgr)
        if self.face_reader is not None and self.tp.last_crops:
            probs = self.face_reader.encode_batch(self.tp.last_crops)["probs"].mean(axis=0)
            new = dict(zip(self.face_reader.meld_labels, map(float, probs)))
            self.face_probs = new if self.face_probs is None else \
                {k: self.face_ema * new[k] + (1 - self.face_ema) * self.face_probs[k] for k in new}
        self.stats["vision_s"].append(time.perf_counter() - start)
        return True

    def face_expression(self) -> Optional[dict]:
        """What the face label and the LM's face cue are read from."""
        return self.face_probs if self.face_reader is not None else self.tp.last_provisional

    # --- audio: VAD-driven, or collected between push-to-talk presses ---
    def on_audio(self, pcm_frame: bytes) -> Optional[dict]:
        """One 30 ms microphone frame. Returns the final event when this frame ended a turn."""
        if self.push_to_talk:
            if self._ptt_buffer is not None:
                self._ptt_buffer.append(pcm_frame)
            return None                                   # the VAD never runs in push-to-talk mode
        event = self.endpoint.feed(pcm_frame)
        if event is None:
            return None
        kind, audio = event
        if kind == "start":
            self._begin_turn()
            return None
        return self.end_turn(audio)

    # --- audio: push-to-talk ---
    def toggle_push_to_talk(self) -> Optional[dict]:
        if self._ptt_buffer is None:
            self._ptt_buffer = []
            self._begin_turn()
            self.status = "recording (space to send)"
            return None
        audio, self._ptt_buffer = b"".join(self._ptt_buffer), None
        return self.end_turn(audio)

    def _begin_turn(self) -> None:
        self.tp.start_turn(self.tp.turn_id)    # the classifier's window = this utterance's frames
        self.status = "speaking"

    def end_turn(self, pcm: bytes) -> Optional[dict]:
        """End-of-speech: ASR, then the state event, then hand the LM its prompt.
        Returns the final event, or None when ASR heard nothing."""
        turn_id = self.tp.turn_id
        stamps = LatencyStamps(turn_start=time.perf_counter())
        self.status = "transcribing"
        text = self.transcriber.transcribe(pcm16_to_float(pcm))
        stamps.asr_done = time.perf_counter()
        self.stats["asr_s"].append(stamps.asr_done - stamps.turn_start)
        if not text:
            self.tp.start_turn(turn_id)            # nothing was said: drop this window's faces too
            self.status = self._idle_status()
            return None
        cues = scene_gloss(self.tp.mean_scene_embedding, self.bank_embeddings) + \
            [face_gloss(self.tp.max_faces_seen, self.face_expression(), self.face_threshold)]
        final = self.tp.end_turn(text, self.context, visual_cues=cues)
        stamps.state_emitted = time.perf_counter()
        top2 = sorted(final["emotion_probs"].items(), key=lambda kv: kv[1], reverse=True)[:2]
        messages = build_prompt(self.context, text, final["emotion"], top2, final["sentiment"], cues)
        self.context.append(text)
        self.lines = [f"\"{text}\"  ->  {final['emotion']} / {final['sentiment']}  |  {'; '.join(cues)}", ""]
        self._stamps[turn_id] = stamps
        if self.worker is not None:
            self.worker.submit(turn_id, messages, stamps)
            self._responding, self.status = turn_id, "responding"
        else:
            stamps.done_emitted = time.perf_counter()
            self.emitter.done(turn_id, "", stamps.as_ms(), prompt=messages[-1]["content"])
            self._finish_lines(turn_id)
            self.status = self._idle_status()
        self.tp.start_turn(self._next_turn_id())    # keep watching while the LM talks
        return final

    def drain_tokens(self) -> None:
        """Move streamed response chunks from the worker into the caption."""
        if self.worker is None:
            return
        while True:
            try:
                turn_id, chunk = self.worker.tokens.get_nowait()
            except queue.Empty:
                return
            if chunk is None:
                self._finish_lines(turn_id)
                if turn_id == self._responding:
                    self._responding, self.status = None, self._idle_status()
            elif turn_id == self._responding and len(self.lines) >= 2:
                self.lines[1] += chunk

    def _finish_lines(self, turn_id: str) -> None:
        ms = self._stamps[turn_id].as_ms()
        self.lines = self.lines[:2] + [f"state {ms['state']} ms (ASR {ms.get('asr')} ms)  "
                                       f"first token {ms['first_token']} ms  done {ms['done']} ms"]

    def face_label(self) -> str:
        from meld_emotion.inference.overlay import face_label_for
        return face_label_for(len(self.tp.last_track_ids), self.face_expression(), self.face_threshold)

    def close(self) -> None:
        if self.worker is not None:
            self.worker.close()


def warm_up(bundle, transcriber, responder=None, face_reader=None) -> None:
    """MPS compiles kernels on first use (~1-2 s per model): run every path
    once on synthetic input, through a throwaway TurnProcessor whose events
    go nowhere, before the first real turn is measured -- the face and scene
    encoders, the text-side end_turn, Whisper, and the LM."""
    import io
    from meld_emotion.inference.events import EventEmitter
    grey = Image.new("RGB", (224, 224), (128, 128, 128))
    bundle.face_encoder.encode_batch([grey])
    bundle.scene_encoder.encode(grey)
    if face_reader is not None:
        face_reader.encode_batch([grey])
    tp = TurnProcessor(bundle, EventEmitter(io.StringIO()))
    tp.start_turn("warmup")
    tp.push_frame(np.full((240, 320, 3), 128, dtype=np.uint8))
    tp.end_turn("warm up", [])
    transcriber.warm_up()
    if responder is not None:
        for _ in responder.stream(build_prompt([], "hello", "neutral", [("neutral", 0.6), ("joy", 0.2)], "neutral", [])):
            pass
