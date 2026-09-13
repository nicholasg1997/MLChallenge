import torch.nn as nn


def test_count_lm_params_reads_the_hf_config_without_downloading_weights(monkeypatch):
    from scripts.param_budget import count_lm_params

    class _StubConfig:   # GQA (2 heads, 1 KV head), gated FFN, tied embeddings -- the Qwen2.5 layout
        num_hidden_layers, hidden_size, intermediate_size, vocab_size = 2, 8, 16, 100
        num_attention_heads, num_key_value_heads, tie_word_embeddings = 2, 1, True

    monkeypatch.setattr("scripts.param_budget.AutoConfig",
                        type("_A", (), {"from_pretrained": staticmethod(lambda repo: _StubConfig())}))
    head_dim = 8 // 2
    attention = 2 * 8 * 2 * head_dim + 2 * 8 * 1 * head_dim
    assert count_lm_params("fake/repo") == 100 * 8 + 2 * (attention + 3 * 8 * 16)


def test_count_bundle_params_measures_each_loaded_component():
    from types import SimpleNamespace
    from scripts.param_budget import count_bundle_params
    bundle = SimpleNamespace(model=nn.Linear(4, 3), face_encoder=SimpleNamespace(model=nn.Linear(2, 2)),
                             scene_encoder=SimpleNamespace(model=nn.Linear(3, 1)))
    counts = count_bundle_params(bundle)
    assert counts == {"text_and_fusion": 15, "face_encoder": 6, "scene_encoder": 4, "face_detector": 75_000}


def test_build_budget_table_includes_every_component_and_the_total():
    from scripts.param_budget import build_budget_table
    counts = {"text_and_fusion": 136_000_000, "face_encoder": 86_000_000, "scene_encoder": 151_000_000, "face_detector": 75_000}
    table = build_budget_table(counts, "mlx-community/Qwen3-4B-4bit", 4_000_000_000, stage=2)
    assert "RoBERTa" in table and "CLIP" in table and "Qwen3-4B-4bit" in table and "Fine-tuned top 4" in table
    assert "**4.37B**" in table
