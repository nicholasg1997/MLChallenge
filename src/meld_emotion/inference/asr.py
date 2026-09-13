"""Speech-to-text for the live path (design doc §8.1): Whisper small.en on
the local device. ASR is a convenience for producing text from speech; the
fusion model still only sees text + vision.

Two runtimes for the same weights. mlx-whisper (default when importable;
Apple silicon) measured 0.35-0.47 s per MELD utterance on the M1; the
transformers/MPS path 0.6-0.7 s (2 s cold -- hence warm_up()). Whisper
always encodes a padded 30 s window, so the cost is per utterance, not per
second of speech."""
import importlib.util

import numpy as np

from meld_emotion.inference.audio import SAMPLE_RATE

DEFAULT_ASR_MODEL = "openai/whisper-small.en"
MLX_REPOS = {"openai/whisper-small.en": "mlx-community/whisper-small.en-mlx",
             "openai/whisper-base.en": "mlx-community/whisper-base.en-mlx",
             "openai/whisper-tiny.en": "mlx-community/whisper-tiny.en-mlx"}
MIN_AUDIO_SAMPLES = SAMPLE_RATE // 10          # <100 ms: nothing Whisper should be asked about


class _TorchWhisper:
    def __init__(self, model_id: str, device: str, max_new_tokens: int):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device).eval()
        self.device, self.max_new_tokens = device, max_new_tokens
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def transcribe(self, audio: np.ndarray) -> str:
        import torch
        with torch.no_grad():
            feats = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.to(self.device)
            ids = self.model.generate(feats, max_new_tokens=self.max_new_tokens)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


class _MlxWhisper:
    def __init__(self, model_id: str):
        import mlx.core as mx
        import mlx_whisper
        from mlx.utils import tree_flatten
        from mlx_whisper.load_models import load_model
        self.repo = MLX_REPOS.get(model_id, model_id)
        self._transcribe = mlx_whisper.transcribe
        model = load_model(self.repo, dtype=mx.float16)      # lru-cached inside mlx_whisper
        self.n_params = sum(v.size for _, v in tree_flatten(model.parameters()))

    def transcribe(self, audio: np.ndarray) -> str:
        return self._transcribe(audio, path_or_hf_repo=self.repo, fp16=True,
                                condition_on_previous_text=False)["text"].strip()


class Transcriber:
    def __init__(self, model_id: str = DEFAULT_ASR_MODEL, device: str = "cpu", backend: str = "auto",
                 max_new_tokens: int = 96):
        if backend not in ("auto", "mlx", "torch"):
            raise ValueError(f"backend must be auto/mlx/torch, got {backend!r}")
        use_mlx = backend == "mlx" or (backend == "auto" and importlib.util.find_spec("mlx_whisper") is not None)
        self.backend = "mlx" if use_mlx else "torch"
        self._impl = _MlxWhisper(model_id) if use_mlx else _TorchWhisper(model_id, device, max_new_tokens)
        self.model_id, self.device, self.n_params = model_id, device, self._impl.n_params

    def transcribe(self, audio: np.ndarray) -> str:
        """`audio`: float32 mono at 16 kHz. Returns the stripped transcript ('' for silence)."""
        if len(audio) < MIN_AUDIO_SAMPLES:
            return ""
        return self._impl.transcribe(np.asarray(audio, dtype=np.float32)).strip()

    def warm_up(self) -> None:
        self.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))
