"""Dialogue-context text formatting and the RoBERTa text encoder.

Format (design doc §4.3): <s> u[t-k] </s> ... </s> u[t-1] </s></s> u[t] </s>
The context is the tokenizer's *first* sequence, the current utterance the
second, so `truncation="only_first"` with `truncation_side="left"` drops the
oldest context first and never touches the current utterance.
"""
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

CONTEXT_SEPARATOR = " </s> "   # RoBERTa's sep token; tokenizes to the real </s> id


def format_context(context_prev: list[str], k: int) -> str:
    if k <= 0 or not context_prev:
        return ""
    return CONTEXT_SEPARATOR.join(context_prev[-k:])


def build_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.truncation_side = "left"
    return tokenizer


def encode_text(tokenizer, context: str, current: str, max_length: int) -> dict[str, list[int]]:
    if context:
        enc = tokenizer(context, current, truncation="only_first", max_length=max_length)
    else:
        enc = tokenizer(current, truncation=True, max_length=max_length)
    return {"input_ids": list(enc["input_ids"]), "attention_mask": list(enc["attention_mask"])}


def build_text_encoder(model_name: str, trainable_layers: int) -> nn.Module:
    """Pretrained encoder with everything frozen except the top `trainable_layers`
    transformer layers (design doc §6: partial fine-tune on ~10K examples)."""
    # attn_implementation="eager": with the bottom layers frozen, MPS's fused
    # scaled_dot_product_attention picks a no-grad kernel that raises
    # "does not support dropout" in train mode (verified on torch 2.14).
    model = AutoModel.from_pretrained(model_name, add_pooling_layer=False, attn_implementation="eager")
    for p in model.parameters():
        p.requires_grad = False
    n_layers = model.config.num_hidden_layers
    for layer in model.encoder.layer[n_layers - trainable_layers:]:
        for p in layer.parameters():
            p.requires_grad = True
    model.hidden_size = model.config.hidden_size
    return model


def count_parameters(module: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable
