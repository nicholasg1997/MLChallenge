import pytest
import torch


def test_format_context_takes_the_last_k_previous_utterances():
    from meld_emotion.training.text import format_context
    assert format_context(["a", "b", "c", "d"], k=2) == "c </s> d"
    assert format_context(["a", "b"], k=4) == "a </s> b"


def test_format_context_is_empty_for_k_zero_or_no_history():
    from meld_emotion.training.text import format_context
    assert format_context(["a", "b"], k=0) == ""
    assert format_context([], k=4) == ""


def test_count_parameters_separates_frozen_from_trainable():
    from meld_emotion.training.text import count_parameters
    m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    for p in m[0].parameters():
        p.requires_grad = False
    total, trainable = count_parameters(m)
    assert total == 4 * 3 + 3 + 3 * 2 + 2
    assert trainable == 3 * 2 + 2


@pytest.mark.network
def test_encode_text_puts_context_first_current_last_and_truncates_oldest_context():
    from meld_emotion.training.text import build_tokenizer, encode_text, format_context
    tok = build_tokenizer("roberta-base")
    ctx = format_context(["line zero", "line one", "line two", "line three"], k=4)
    enc = encode_text(tok, ctx, "current line", max_length=16)
    assert len(enc["input_ids"]) == 16 and len(enc["attention_mask"]) == 16
    decoded = tok.decode(enc["input_ids"])
    assert decoded.endswith("</s></s>current line</s>")
    assert "line zero" not in decoded and "line three" in decoded   # oldest context dropped first


@pytest.mark.network
def test_encode_text_without_context_is_a_plain_single_sequence():
    from meld_emotion.training.text import build_tokenizer, encode_text
    tok = build_tokenizer("roberta-base")
    enc = encode_text(tok, "", "current line", max_length=16)
    assert tok.decode(enc["input_ids"]) == "<s>current line</s>"


@pytest.mark.network
def test_build_text_encoder_freezes_all_but_the_top_layers():
    from meld_emotion.training.text import build_text_encoder, count_parameters
    enc = build_text_encoder("roberta-base", trainable_layers=6)
    assert enc.hidden_size == 768
    total, trainable = count_parameters(enc)
    assert 120_000_000 < total < 130_000_000
    assert 40_000_000 < trainable < 45_000_000          # 6 of 12 layers ≈ 42.5M
    assert not any(p.requires_grad for p in enc.embeddings.parameters())
    assert all(p.requires_grad for p in enc.encoder.layer[-1].parameters())
    assert not any(p.requires_grad for p in enc.encoder.layer[0].parameters())
    out = enc(input_ids=torch.tensor([[0, 100, 2]]), attention_mask=torch.tensor([[1, 1, 1]]))
    assert out.last_hidden_state.shape == (1, 3, 768)
