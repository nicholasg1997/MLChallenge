#!/usr/bin/env python3
"""Final parameter-count table (design doc §4.1), measured from the loaded
inference bundle (text encoder + fusion, face ViT, CLIP both towers) plus
YuNet's fixed 75K and the chosen response LM's count from its HF config.

Usage:
    uv run python scripts/param_budget.py --model-repo <chosen from results/lm_selection.json>
"""
import argparse
import json
from pathlib import Path

from transformers import AutoConfig

from meld_emotion.config import REPO_ROOT
from meld_emotion.inference.loader import load_inference_bundle
from meld_emotion.training.text import count_parameters
from scripts.replay_demo import DEFAULT_CHECKPOINT

FACE_DETECTOR_PARAMS = 75_000   # YuNet 2023mar (232 KB ONNX)


def count_lm_params(model_repo: str) -> int:
    """From the HF config alone: embeddings + per-layer attention/FFN
    (biases/norms negligible). Same estimate style as the Stage 1 plan."""
    cfg = AutoConfig.from_pretrained(model_repo)
    hidden, layers = cfg.hidden_size, cfg.num_hidden_layers
    inter = getattr(cfg, "intermediate_size", 4 * hidden)
    return int(cfg.vocab_size * hidden + layers * (4 * hidden * hidden + 2 * hidden * inter))


def count_bundle_params(bundle) -> dict[str, int]:
    return {"text_and_fusion": count_parameters(bundle.model)[0],
            "face_encoder": count_parameters(bundle.face_encoder.model)[0],
            "scene_encoder": count_parameters(bundle.scene_encoder.model)[0],
            "face_detector": FACE_DETECTOR_PARAMS}


def build_budget_table(counts: dict, lm_repo: str, lm_params: int, stage: int) -> str:
    face_note = "Fine-tuned top 4 layers (Stage 2)" if stage >= 2 else "Frozen"
    rows = [("RoBERTa-base + fusion transformer + projectors + heads", counts["text_and_fusion"], "RoBERTa top half fine-tuned; fusion from scratch"),
            ("Face/expression encoder (ViT-Base)", counts["face_encoder"], face_note),
            ("CLIP ViT-B/32 (image + text towers; gloss only)", counts["scene_encoder"], "Frozen"),
            ("Face detector (YuNet)", counts["face_detector"], "Zero-shot"),
            (f"Response LM ({lm_repo.split('/')[-1]})", lm_params, "Prompted, not fine-tuned")]
    total = sum(n for _, n, _ in rows)
    lines = ["| Component | Params | Trained? |", "|---|---|---|"]
    lines += [f"| {name} | {n / 1e6:,.1f}M | {trained} |" if n < 1e9 else f"| {name} | {n / 1e9:.2f}B | {trained} |"
              for name, n, trained in rows]
    lines.append(f"| **Total** | **{total / 1e9:.2f}B** | ceiling 6B |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    bundle = load_inference_bundle(args.checkpoint, device="cpu")
    counts = count_bundle_params(bundle)
    lm_params = count_lm_params(args.model_repo)
    table = build_budget_table(counts, args.model_repo, lm_params, bundle.stage)
    print(table)
    (REPO_ROOT / "results").mkdir(exist_ok=True)
    (REPO_ROOT / "results" / "param_budget.json").write_text(json.dumps(
        {"model_repo": args.model_repo, "lm_params": lm_params, **counts,
         "total": sum(counts.values()) + lm_params, "stage": bundle.stage}, indent=2))
    (REPO_ROOT / "docs" / "results").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "docs" / "results" / "param_budget.md").write_text(table + "\n")


if __name__ == "__main__":
    main()
