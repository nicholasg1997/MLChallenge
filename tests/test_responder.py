from types import SimpleNamespace

import pytest


def test_build_prompt_uses_last_4_context_lines_and_never_speaker_names():
    from meld_emotion.inference.responder import build_prompt
    messages = build_prompt(["l0", "l1", "l2", "l3", "l4"], "current line", "surprise",
                            [("surprise", 0.6), ("neutral", 0.2)], "negative", ["two people"])
    assert messages[0]["role"] == "system"
    user = messages[1]["content"]
    assert "l0" not in user  # only the last 4 (l1..l4) are kept — l0 is the 5th-from-last, dropped
    assert "l4" in user and "current line" in user
    assert "surprise" in user and "60%" in user and "negative" in user and "two people" in user
    assert "Speaker" not in user and "speaker" not in user


def test_build_prompt_stays_short():
    from meld_emotion.inference.responder import build_prompt
    messages = build_prompt(["a short line"] * 4, "another short line", "joy",
                            [("joy", 0.9), ("neutral", 0.05)], "positive", ["one person"])
    total_chars = sum(len(m["content"]) for m in messages)
    assert total_chars < 800  # ~200 tokens at a generous 4 chars/token


def test_responder_stream_yields_incremental_text_and_uses_the_chat_template():
    from meld_emotion.inference.responder import Responder

    class _StubTokenizer:
        def apply_chat_template(self, messages, add_generation_prompt, enable_thinking=None):
            assert add_generation_prompt is True
            return "PROMPT"

    def fake_generate(model, tokenizer, prompt, max_tokens):
        assert prompt == "PROMPT" and max_tokens == 40
        for text in ["Wow", " really?"]:
            yield SimpleNamespace(text=text, finish_reason=None)
        yield SimpleNamespace(text="", finish_reason="stop")

    responder = Responder(model=object(), tokenizer=_StubTokenizer())
    chunks = list(responder.stream([{"role": "user", "content": "hi"}], generate_fn=fake_generate))
    assert chunks == ["Wow", " really?", ""]
    assert "".join(chunks) == "Wow really?"


def test_responder_stream_falls_back_when_enable_thinking_is_unsupported():
    from meld_emotion.inference.responder import Responder

    class _StubTokenizerNoThinking:
        def apply_chat_template(self, messages, add_generation_prompt):
            return "PROMPT_NO_THINKING"

    def fake_generate(model, tokenizer, prompt, max_tokens):
        assert prompt == "PROMPT_NO_THINKING"
        yield SimpleNamespace(text="ok", finish_reason="stop")

    responder = Responder(model=object(), tokenizer=_StubTokenizerNoThinking())
    assert list(responder.stream([{"role": "user", "content": "hi"}], generate_fn=fake_generate)) == ["ok"]


@pytest.mark.network
def test_responder_loads_a_real_mlx_model_and_streams_a_real_response():
    from meld_emotion.inference.responder import Responder
    responder = Responder(model_repo="mlx-community/Qwen2.5-1.5B-Instruct-4bit", max_tokens=10)
    chunks = list(responder.stream([{"role": "user", "content": "Say hi in one word."}]))
    assert len("".join(chunks)) > 0
