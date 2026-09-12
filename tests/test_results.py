import json


def _write(root, name, seed, dev_w, dev_m, test_w=None):
    d = root / name / f"seed{seed}"
    d.mkdir(parents=True)
    r = {"config": {"name": name, "seed": seed}, "best_epoch": 3, "wall_clock_s": 120.0,
         "dev": {"emotion": {"weighted_f1": dev_w, "macro_f1": dev_m}, "sentiment": {"weighted_f1": 0.7}}}
    if test_w is not None:
        r["test"] = {"emotion": {"weighted_f1": test_w, "macro_f1": test_w - 0.1}, "sentiment": {"weighted_f1": 0.7}}
    (d / "results.json").write_text(json.dumps(r))


def test_collect_and_summarise_average_over_seeds(tmp_path):
    from meld_emotion.training.results import collect_results, summarise
    _write(tmp_path, "fusion", 0, 0.60, 0.40, test_w=0.62)
    _write(tmp_path, "fusion", 1, 0.62, 0.42, test_w=0.64)
    _write(tmp_path, "text_only_k4", 0, 0.58, 0.38)
    results = collect_results(tmp_path)
    assert {(r["name"], r["seed"]) for r in results} == {("fusion", 0), ("fusion", 1), ("text_only_k4", 0)}
    summary = {row["name"]: row for row in summarise(results)}
    assert summary["fusion"]["n_seeds"] == 2
    assert abs(summary["fusion"]["dev_weighted_f1_mean"] - 0.61) < 1e-9
    assert abs(summary["fusion"]["test_weighted_f1_mean"] - 0.63) < 1e-9
    assert summary["text_only_k4"]["test_weighted_f1_mean"] is None
    assert summary["fusion"]["wall_clock_min_mean"] == 2.0


def test_ablation_table_renders_markdown_with_baseline_row(tmp_path):
    from meld_emotion.training.results import ablation_table, collect_results, summarise
    _write(tmp_path, "fusion", 0, 0.60, 0.40)
    baseline = {"dev": {"weighted_f1": 0.35, "macro_f1": 0.20}}
    table = ablation_table(summarise(collect_results(tmp_path)), baseline)
    assert table.startswith("| Model |")
    assert "| fusion |" in table and "vision_only_zero_training" in table
    assert "0.600" in table and "0.350" in table
