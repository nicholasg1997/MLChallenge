#!/usr/bin/env python3
"""Response quality rubric (design doc §4.4): 50 sampled prompt/response
pairs from a completed replay run, with empty fields for hand scoring.
This file IS a plan deliverable, not just a script side-effect.

Usage:
    uv run python scripts/response_rubric.py
"""
import json
import random
from pathlib import Path

from meld_emotion.config import REPO_ROOT

RUBRIC_FIELDS = ("tag_consistent", "uses_visual_cue", "concise", "in_character", "no_invented_facts")


def sample_response_rows(events_path: Path, n: int = 50, seed: int = 0) -> list[dict]:
    events = [json.loads(l) for l in Path(events_path).read_text().splitlines() if l.strip()]
    finals = {e["turn_id"]: e for e in events if e.get("phase") == "final"}
    done = [e for e in events if e.get("phase") == "done"]
    sampled = random.Random(seed).sample(done, min(n, len(done)))
    rows = []
    for e in sampled:
        final = finals.get(e["turn_id"], {})
        prompt = f"{final.get('text', '')} [{final.get('emotion', '')}/{final.get('sentiment', '')}; " \
                 f"cues: {', '.join(final.get('visual_cues', []))}]"
        rows.append({"turn_id": e["turn_id"], "prompt": prompt, "response": e["response"], **{f: None for f in RUBRIC_FIELDS}})
    return rows


def main():
    rows = sample_response_rows(REPO_ROOT / "results" / "replay_events.jsonl")
    out_path = REPO_ROOT / "results" / "response_rubric.jsonl"
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
