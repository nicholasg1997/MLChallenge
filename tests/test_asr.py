import numpy as np
import pytest


class _Impl:
    def __init__(self, *a, **k):
        self.n_params, self.calls = 7, []

    def transcribe(self, audio):
        self.calls.append(len(audio))
        return "  hi there  "


@pytest.fixture
def stub_impls(monkeypatch):
    from meld_emotion.inference import asr
    monkeypatch.setattr(asr, "_MlxWhisper", _Impl)
    monkeypatch.setattr(asr, "_TorchWhisper", _Impl)
    return asr


def test_auto_prefers_mlx_when_importable(stub_impls, monkeypatch):
    monkeypatch.setattr(stub_impls.importlib.util, "find_spec", lambda name: object())
    assert stub_impls.Transcriber().backend == "mlx"
    monkeypatch.setattr(stub_impls.importlib.util, "find_spec", lambda name: None)
    assert stub_impls.Transcriber().backend == "torch"


def test_explicit_backend_wins_and_bad_backend_rejected(stub_impls, monkeypatch):
    monkeypatch.setattr(stub_impls.importlib.util, "find_spec", lambda name: object())
    assert stub_impls.Transcriber(backend="torch").backend == "torch"
    with pytest.raises(ValueError):
        stub_impls.Transcriber(backend="cuda")


def test_short_audio_is_not_sent_to_the_model_and_transcripts_are_stripped(stub_impls):
    t = stub_impls.Transcriber(backend="torch")
    assert t.transcribe(np.zeros(100, np.float32)) == ""
    assert t.transcribe(np.zeros(16000, np.float32)) == "hi there"
    assert t._impl.calls == [16000] and t.n_params == 7
