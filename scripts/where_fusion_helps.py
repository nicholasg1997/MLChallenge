#!/usr/bin/env python3
"""Where does the face channel change the answer? Bins the test set by the
text-only model's confidence and compares accuracy per bin for text-only,
Stage 1 fusion and Stage 2 fusion (each run's test_predictions.jsonl), plus
who is right when the fused model overrides the text model.

Usage:
    uv run python -m scripts.where_fusion_helps
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from meld_emotion.config import FEATURE_CACHE_DIR, REPO_ROOT

RUNS = {"text-only (k=4)": "text_only_k4/seed0", "Stage 1 fusion": "fusion_faces_only/seed0",
        "Stage 2 fusion": "stage2_fusion_faces_only/seed1"}
BINS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]


def load_predictions(run: str) -> dict[str, dict]:
    path = REPO_ROOT / "results" / run / "test_predictions.jsonl"
    return {p["clip"]: p for p in (json.loads(l) for l in path.read_text().splitlines() if l.strip())}


def main():
    truth = {f"dia{r['dialogue_id']}_utt{r['utterance_id']}": r["emotion"]
             for r in (json.loads(l) for l in (FEATURE_CACHE_DIR / "test" / "manifest.jsonl").read_text().splitlines())
             if r["status"] == "ok"}
    preds = {name: load_predictions(run) for name, run in RUNS.items()}
    clips = [c for c in truth if all(c in p for p in preds.values())]
    text = preds["text-only (k=4)"]
    conf = np.array([max(text[c]["probs"]) for c in clips])

    def acc(p, idx):
        return float(np.mean([p[clips[i]]["pred"] == truth[clips[i]] for i in idx]))

    lines = [f"{len(clips)} test clips, binned by the text-only model's max probability.", "",
             "| text confidence | n | " + " | ".join(RUNS) + " |", "|---|---|" + "---|" * len(RUNS)]
    for lo, hi in BINS:
        idx = [i for i, v in enumerate(conf) if lo <= v < hi]
        lines.append(f"| [{lo:.1f}, {min(hi, 1.0):.1f}) | {len(idx)} | " +
                     " | ".join(f"{acc(p, idx):.1%}" for p in preds.values()) + " |")
    lines.append("")
    for name in ("Stage 1 fusion", "Stage 2 fusion"):
        fused = preds[name]
        dis = [c for c in clips if fused[c]["pred"] != text[c]["pred"]]
        f_right = sum(fused[c]["pred"] == truth[c] for c in dis)
        t_right = sum(text[c]["pred"] == truth[c] for c in dis)
        lines.append(f"- **{name} overrides the text model on {len(dis)} clips ({len(dis) / len(clips):.1%})**: "
                     f"fused right {f_right}, text right {t_right}, neither {len(dis) - f_right - t_right} "
                     f"(mean text confidence there {conf[[clips.index(c) for c in dis]].mean():.2f} vs {conf.mean():.2f} overall).")
    wins = Counter(truth[c] for c in clips if preds["Stage 2 fusion"][c]["pred"] == truth[c] != text[c]["pred"])
    lines.append(f"- Stage 2's correct overrides by true class: {dict(wins.most_common())}.")
    out = "\n".join(lines) + "\n"
    print(out)
    (REPO_ROOT / "docs" / "results" / "where_fusion_helps.md").write_text("# Where the face channel changes the answer\n\n" + out)


if __name__ == "__main__":
    main()
