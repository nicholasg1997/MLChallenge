"""Turn segmentation for the live path (design doc §2, §8.1): a VAD-driven
endpoint detector over 30 ms PCM frames, plus the microphone stream that
feeds it. The detector is pure logic over an injected `is_speech(frame)`
so it is unit-testable without webrtcvad or a microphone.

A turn starts after `start_ms` of consecutive voiced audio (with `pre_roll_ms`
of audio before that kept, so the first syllable is not clipped) and ends
after `end_silence_ms` of unvoiced audio -- or at `max_turn_s`, the same
15 s cap the model's training clips had (preprocess.MAX_DECODE_SECONDS)."""
import queue
from collections import deque
from typing import Callable, Optional

import numpy as np

SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000     # 480 int16 samples = 960 bytes


class EndpointDetector:
    def __init__(self, is_speech: Callable[[bytes], bool], *, start_ms: int = 150, end_silence_ms: int = 800,
                 pre_roll_ms: int = 300, max_turn_s: float = 15.0, frame_ms: int = FRAME_MS):
        self.is_speech = is_speech
        self.start_frames = max(1, start_ms // frame_ms)
        self.end_frames = max(1, end_silence_ms // frame_ms)
        self.max_frames = int(max_turn_s * 1000 / frame_ms)
        self._pre_roll: deque = deque(maxlen=max(1, pre_roll_ms // frame_ms))
        self.speaking = False
        self._voiced_run = 0
        self._unvoiced_run = 0
        self._buffer: list[bytes] = []

    def feed(self, frame: bytes) -> Optional[tuple[str, Optional[bytes]]]:
        """One 30 ms frame in; `("start", None)` when a turn begins,
        `("end", pcm_bytes)` when it ends, else None."""
        voiced = bool(self.is_speech(frame))
        if not self.speaking:
            self._pre_roll.append(frame)
            self._voiced_run = self._voiced_run + 1 if voiced else 0
            if self._voiced_run >= self.start_frames:
                self.speaking, self._voiced_run, self._unvoiced_run = True, 0, 0
                self._buffer = list(self._pre_roll)       # includes this frame
                self._pre_roll.clear()
                return ("start", None)
            return None
        self._buffer.append(frame)
        self._unvoiced_run = 0 if voiced else self._unvoiced_run + 1
        if self._unvoiced_run >= self.end_frames or len(self._buffer) >= self.max_frames:
            return ("end", self._finish())
        return None

    def force_end(self) -> Optional[bytes]:
        """Push-to-talk release: end the turn now, if one is open."""
        return self._finish() if self.speaking else None

    def _finish(self) -> bytes:
        audio = b"".join(self._buffer)
        self.speaking, self._buffer, self._unvoiced_run = False, [], 0
        return audio


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def build_vad(aggressiveness: int = 2) -> Callable[[bytes], bool]:
    import webrtcvad
    vad = webrtcvad.Vad(aggressiveness)
    return lambda frame: vad.is_speech(frame, SAMPLE_RATE)


class MicrophoneStream:
    """16 kHz mono int16 capture in 30 ms blocks onto a queue, from a
    PortAudio callback thread. `drain()` returns whatever has arrived."""
    def __init__(self, device=None):
        import sounddevice as sd
        self._frames: queue.Queue = queue.Queue()
        self._stream = sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                                         blocksize=FRAME_SAMPLES, device=device, callback=self._on_audio)

    def _on_audio(self, indata, frames, time_info, status) -> None:
        self._frames.put(bytes(indata))

    def __enter__(self):
        self._stream.start()
        return self

    def __exit__(self, *exc):
        self._stream.stop()
        self._stream.close()

    def drain(self) -> list[bytes]:
        out = []
        while True:
            try:
                out.append(self._frames.get_nowait())
            except queue.Empty:
                return out
