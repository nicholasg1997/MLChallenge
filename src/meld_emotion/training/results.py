"""Aggregate results/<name>/seed*/results.json into the ablation table
(design doc §7): mean ± std over seeds, dev and (when evaluated) test."""
import json
import statistics
from pathlib import Path

ORDER = ["vision_only_zero_training", "text_only_k0", "text_only_k4", "vision_only", "fusion",
         "fusion_no_scene", "fusion_no_context", "fusion_no_trackid", "fusion_faces_only"]


def collect_results(results_dir: Path) -> list[dict]:
    out = []
    for path in sorted(Path(results_dir).glob("*/seed*/results.json")):
        with open(path) as f:
            r = json.load(f)
        r["name"] = r["config"]["name"]
        r["seed"] = r["config"]["seed"]
        out.append(r)
    return out


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.mean(values), (statistics.pstdev(values) if len(values) > 1 else 0.0)


def summarise(results: list[dict]) -> list[dict]:
    by_name: dict[str, list[dict]] = {}
    for r in results:
        by_name.setdefault(r["name"], []).append(r)
    rows = []
    for name, runs in by_name.items():
        row = {"name": name, "n_seeds": len(runs)}
        for split in ("dev", "test"):
            for metric in ("weighted_f1", "macro_f1"):
                vals = [r[split]["emotion"][metric] for r in runs if split in r]
                row[f"{split}_{metric}_mean"], row[f"{split}_{metric}_std"] = _mean_std(vals)
        row["dev_sentiment_f1_mean"], _ = _mean_std([r["dev"]["sentiment"]["weighted_f1"] for r in runs])
        row["best_epoch_mean"], _ = _mean_std([r["best_epoch"] for r in runs])
        row["wall_clock_min_mean"], _ = _mean_std([r["wall_clock_s"] / 60 for r in runs])
        rows.append(row)
    rows.sort(key=lambda r: ORDER.index(r["name"]) if r["name"] in ORDER else len(ORDER))
    return rows


def _fmt(mean, std=None) -> str:
    if mean is None:
        return "—"
    return f"{mean:.3f}" if std is None else f"{mean:.3f} ± {std:.3f}"


def ablation_table(summary: list[dict], baseline: dict | None = None) -> str:
    lines = ["| Model | seeds | dev wF1 | dev mF1 | test wF1 | test mF1 | dev sent. F1 | best ep. | min/run |",
             "|---|---|---|---|---|---|---|---|---|"]
    if baseline:
        b = baseline.get("dev", {})
        t = baseline.get("test", {})
        lines.append(f"| vision_only_zero_training | — | {_fmt(b.get('weighted_f1'))} | {_fmt(b.get('macro_f1'))} "
                     f"| {_fmt(t.get('weighted_f1'))} | {_fmt(t.get('macro_f1'))} | — | — | 0 |")
    for r in summary:
        lines.append(f"| {r['name']} | {r['n_seeds']} | {_fmt(r['dev_weighted_f1_mean'], r['dev_weighted_f1_std'])} "
                     f"| {_fmt(r['dev_macro_f1_mean'], r['dev_macro_f1_std'])} "
                     f"| {_fmt(r['test_weighted_f1_mean'], r['test_weighted_f1_std'])} "
                     f"| {_fmt(r['test_macro_f1_mean'], r['test_macro_f1_std'])} "
                     f"| {_fmt(r['dev_sentiment_f1_mean'])} | {r['best_epoch_mean']:.1f} | {r['wall_clock_min_mean']:.1f} |")
    return "\n".join(lines)
