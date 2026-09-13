#!/usr/bin/env python3
"""Benchmark response-LM candidates on this machine and pick the largest
that meets the replay latency targets (design doc §2, §4.1). Candidates
are real, verified mlx-community 4-bit checkpoints, ordered largest to
smallest; the smallest is the design doc's explicit latency fallback.

Usage:
    uv run --with mlx-lm python scripts/select_lm.py
"""
import json
import time
from pathlib import Path

from meld_emotion.config import REPO_ROOT

CANDIDATES = [
    "mlx-community/Qwen3-4B-4bit",               # 4B params; already in ~/.cache/huggingface on the dev machine
    "mlx-community/Phi-3.5-mini-instruct-4bit",  # 3.8B params
    "mlx-community/Qwen2.5-3B-Instruct-4bit",    # 3.09B params
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",  # 1.5B params
]
FIRST_TOKEN_TARGET_S = 1.0
DONE_TARGET_S = 2.5
MAX_TOKENS = 40
SAMPLE_MESSAGES = [
    {"role": "system", "content": "You are a small, friendly character robot. React in at most two sentences."},
    {"role": "user", "content": "You did WHAT?\n\n[Detected emotion: surprise (60%), neutral (20%). "
                                "Sentiment: negative. Visual cues: two people, dimly lit room.]"},
]


def benchmark(repo: str) -> dict:
    from mlx_lm import load, stream_generate
    model, tokenizer = load(repo)
    try:
        prompt = tokenizer.apply_chat_template(SAMPLE_MESSAGES, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tokenizer.apply_chat_template(SAMPLE_MESSAGES, add_generation_prompt=True)
    start = time.perf_counter()
    first_token_s, n_tokens = None, 0
    for response in stream_generate(model, tokenizer, prompt, max_tokens=MAX_TOKENS):
        if first_token_s is None:
            first_token_s = time.perf_counter() - start
        n_tokens += 1
    done_s = time.perf_counter() - start
    return {"repo": repo, "first_token_s": round(first_token_s, 3) if first_token_s else None,
           "done_s": round(done_s, 3), "n_tokens": n_tokens,
           "meets_target": bool(first_token_s and first_token_s <= FIRST_TOKEN_TARGET_S and done_s <= DONE_TARGET_S)}


def main():
    results = [benchmark(repo) for repo in CANDIDATES]
    for r in results:
        print(f"{r['repo']}: first_token={r['first_token_s']}s done={r['done_s']}s "
             f"tokens={r['n_tokens']} meets_target={r['meets_target']}")
    winners = [r for r in results if r["meets_target"]]
    chosen = winners[0] if winners else results[-1]
    print(f"\nChosen: {chosen['repo']}")
    out = {"candidates": results, "chosen": chosen["repo"]}
    (REPO_ROOT / "results").mkdir(parents=True, exist_ok=True)
    Path(REPO_ROOT / "results" / "lm_selection.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
