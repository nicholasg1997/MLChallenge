#!/usr/bin/env python3
"""Benchmark response-LM candidates on this machine and pick the largest
that meets the replay latency targets (design doc §2, §4.1). Candidates
are real, verified mlx-community 4-bit checkpoints, ordered largest to
smallest; the smallest is the design doc's explicit latency fallback.

Each candidate is timed the way the demo pays for it: through Responder
(same chat template and streaming path), on a prompt the size of a real
turn (four context lines + cues), after one discarded warm-up generation
(the first generation on a freshly loaded mlx model compiles Metal kernels
and runs ~2x slower than every later one). Median of three runs.

Usage:
    uv run --with mlx-lm python scripts/select_lm.py
"""
import json
import statistics
import time
from pathlib import Path

from meld_emotion.config import REPO_ROOT
from meld_emotion.inference.responder import Responder, build_prompt

CANDIDATES = [
    "mlx-community/Qwen3-4B-4bit",               # 4.02B params
    "mlx-community/Phi-3.5-mini-instruct-4bit",  # 3.82B params
    "mlx-community/Qwen2.5-3B-Instruct-4bit",    # 3.09B params
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",  # 1.54B params (latency fallback)
]
FIRST_TOKEN_TARGET_S = 1.0
DONE_TARGET_S = 2.5
N_RUNS = 3
# A real replay turn (test/dia0): four lines of context, a current line, the fused state, two cues.
SAMPLE_MESSAGES = build_prompt(
    ["Why do all you're coffee mugs have numbers on the bottom?",
     "Oh. That's so Monica can keep track. That way if one on them is missing, she can be like, 'Where's number 27?!'",
     "Y'know what?", "I think I'm gonna go get a coffee."],
    "Really? You'd-you'd do that for me?!", "surprise", [("surprise", 0.6), ("neutral", 0.2)], "positive",
    ["people in a kitchen", "2 faces visible, reading neutral (56%) or joy (25%)"])


def time_one(responder: Responder, messages: list[dict]) -> tuple[float | None, float, int]:
    start = time.perf_counter()
    first_token_s, n_tokens = None, 0
    for _ in responder.stream(messages):
        if first_token_s is None:
            first_token_s = time.perf_counter() - start
        n_tokens += 1
    return first_token_s, time.perf_counter() - start, n_tokens


def benchmark(repo: str, n_runs: int = N_RUNS) -> dict:
    responder = Responder(model_repo=repo)
    time_one(responder, SAMPLE_MESSAGES)                         # warm-up, discarded
    runs = [time_one(responder, SAMPLE_MESSAGES) for _ in range(n_runs)]
    firsts = [r[0] for r in runs if r[0] is not None]
    first_token_s = statistics.median(firsts) if firsts else None
    done_s = statistics.median(r[1] for r in runs)
    return {"repo": repo, "first_token_s": round(first_token_s, 3) if first_token_s else None,
            "done_s": round(done_s, 3), "n_tokens": round(statistics.median(r[2] for r in runs)), "n_runs": n_runs,
            "meets_target": bool(first_token_s and first_token_s <= FIRST_TOKEN_TARGET_S and done_s <= DONE_TARGET_S)}


def main():
    results = [benchmark(repo) for repo in CANDIDATES]
    for r in results:
        print(f"{r['repo']}: first_token={r['first_token_s']}s done={r['done_s']}s "
              f"tokens={r['n_tokens']} meets_target={r['meets_target']}  (median of {r['n_runs']}, warm)")
    winners = [r for r in results if r["meets_target"]]
    chosen = winners[0] if winners else results[-1]
    print(f"\nChosen: {chosen['repo']}")
    out = {"candidates": results, "chosen": chosen["repo"],
           "targets": {"first_token_s": FIRST_TOKEN_TARGET_S, "done_s": DONE_TARGET_S}}
    (REPO_ROOT / "results").mkdir(parents=True, exist_ok=True)
    Path(REPO_ROOT / "results" / "lm_selection.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
