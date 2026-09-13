"""Frame + audio sources for the live loop. Both yield `(frame_bgr,
[pcm_frames])` at wall-clock pace: the webcam/microphone pair for the real
demo, and a video file (its own audio track, resampled to 16 kHz mono)
for running the identical loop headlessly -- also the way to replay clips
from other shows (design doc §8.1 #3)."""
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from meld_emotion.inference.audio import FRAME_SAMPLES, SAMPLE_RATE

FRAME_BYTES = FRAME_SAMPLES * 2


class CameraSource:
    def __init__(self, index: int = 0, mic_device=None):
        import cv2
        from meld_emotion.inference.audio import MicrophoneStream
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"camera {index} did not open")
        self.mic = MicrophoneStream(device=mic_device)

    def frames(self) -> Iterator[tuple[np.ndarray, list[bytes]]]:
        with self.mic:
            while True:
                ret, frame = self.cap.read()        # blocks for the next camera frame: paces the loop
                if not ret:
                    return
                yield frame, self.mic.drain()

    def close(self) -> None:
        self.cap.release()


class FileSource:
    """Decodes video and audio together with PyAV, delivering each video
    frame at its own timestamp and the audio decoded so far alongside it.
    After the file ends, `tail_silence_s` of silent frames let the endpoint
    detector close the last turn."""
    def __init__(self, path: Path, tail_silence_s: float = 1.0):
        import av
        self.container = av.open(str(path))
        self.video = self.container.streams.video[0]
        self.audio = self.container.streams.audio[0]
        self.resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        self.tail_silence_s = tail_silence_s
        self._pending = bytearray()

    def _chunks(self) -> list[bytes]:
        n = len(self._pending) // FRAME_BYTES
        out = [bytes(self._pending[i * FRAME_BYTES:(i + 1) * FRAME_BYTES]) for i in range(n)]
        del self._pending[:n * FRAME_BYTES]
        return out

    def frames(self) -> Iterator[tuple[np.ndarray, list[bytes]]]:
        t0, last = time.perf_counter(), None
        for packet in self.container.demux(self.video, self.audio):
            for frame in packet.decode():
                if packet.stream.type == "audio":
                    for resampled in self.resampler.resample(frame):
                        self._pending += resampled.to_ndarray().tobytes()
                    continue
                last = frame.to_ndarray(format="bgr24")
                delay = t0 + float(frame.pts * frame.time_base) - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                yield last, self._chunks()
        if last is None:
            return
        silence = bytes(FRAME_BYTES)
        for _ in range(int(self.tail_silence_s * 1000 / 30)):
            time.sleep(0.03)
            yield last, [silence]

    def close(self) -> None:
        self.container.close()
