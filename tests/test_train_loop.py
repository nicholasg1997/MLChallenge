import json

import torch

from conftest import StubTextEncoder, whitespace_encode


def _cfg(**kw):
    from meld_emotion.training.config import TrainConfig
    defaults = dict(d_model=32, n_heads=4, ff_dim=64, n_layers=1, batch_size=8, epochs=2,
                    patience=5, device="cpu", modality_dropout=0.15)
    return TrainConfig(**{**defaults, **kw})


def test_scheduler_warms_up_then_decays_to_zero():
    from meld_emotion.training.train import build_scheduler
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=1.0)
    sched = build_scheduler(opt, total_steps=100, warmup_fraction=0.1)
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"]); opt.step(); sched.step()
    assert lrs[0] < lrs[5] < lrs[9]                 # warming up
    assert abs(lrs[10] - 1.0) < 1e-6                 # peak after warmup
    assert lrs[50] < lrs[10] and lrs[-1] < 0.01      # cosine decay to ~0


def test_optimizer_uses_two_learning_rates(synthetic_features):
    from meld_emotion.training.model import FusionModel
    from meld_emotion.training.train import build_optimizer
    cfg = _cfg(lr_text=1e-5, lr_fusion=1e-3)
    model = FusionModel(cfg, StubTextEncoder(32), face_dim=8, scene_dim=4)
    opt = build_optimizer(model, cfg)
    lrs = sorted(g["lr"] for g in opt.param_groups)
    assert lrs == [1e-5, 1e-3]


def test_train_end_to_end_on_synthetic_features_writes_all_artifacts(synthetic_features, tmp_path):
    import functools
    from torch.utils.data import DataLoader
    from meld_emotion.training.train import load_checkpoint, train, evaluate
    from meld_emotion.training.model import FusionModel
    from meld_emotion.training.dataset import MeldFeatureDataset, BucketBatchSampler, collate, load_ok_rows
    root, _, _ = synthetic_features
    out = tmp_path / "run"
    cfg = _cfg(seed=1)
    results = train(cfg, root, out, test_split="dev",
                    encode_fn=whitespace_encode, text_encoder=StubTextEncoder(32), pad_id=1, log=lambda *_: None)
    assert (out / "best.pt").exists() and (out / "results.json").exists()
    assert len((out / "log.jsonl").read_text().strip().splitlines()) == 2
    for key in ("dev", "dev_masked_text_only", "dev_masked_vision_only", "test"):
        assert 0.0 <= results[key]["emotion"]["weighted_f1"] <= 1.0
        assert len(results[key]["emotion"]["confusion"]) == 7
    assert results["train_rows"] == 28 and results["dev_rows"] == 14
    assert results["params_trainable"] > 0 and results["wall_clock_s"] > 0 and results["peak_rss_mb"] > 10
    assert results["best_epoch"] in (1, 2) and results["config"]["seed"] == 1
    assert json.load(open(out / "results.json"))["best_epoch"] == results["best_epoch"]
    # the checkpoint reloads into a fresh model and reproduces the dev prediction
    model = FusionModel(_cfg(seed=1), StubTextEncoder(32), face_dim=8, scene_dim=4)
    load_checkpoint(out / "best.pt", model)
    # verify the reloaded model reproduces the dev prediction from the original train() call
    dev_rows = load_ok_rows(root / "dev" / "manifest.jsonl")
    collate_fn = functools.partial(collate, pad_id=1)
    ds = MeldFeatureDataset(dev_rows, root, cfg, whitespace_encode)
    sampler = BucketBatchSampler(ds.lengths, cfg.batch_size, shuffle=False, seed=cfg.seed)
    dev_loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn, num_workers=0)
    reloaded_eval = evaluate(model, dev_loader, device="cpu")
    original_f1 = results["dev"]["emotion"]["weighted_f1"]
    reloaded_f1 = reloaded_eval["emotion"]["weighted_f1"]
    assert abs(original_f1 - reloaded_f1) < 1e-5, f"F1 mismatch: original={original_f1:.6f}, reloaded={reloaded_f1:.6f}"


def test_train_text_only_and_vision_only_presets_run(synthetic_features, tmp_path):
    from meld_emotion.training.train import train
    root, _, _ = synthetic_features
    r1 = train(_cfg(use_faces=False, use_scene=False, context_k=0, epochs=1), root, tmp_path / "t",
               encode_fn=whitespace_encode, text_encoder=StubTextEncoder(32), pad_id=1, log=lambda *_: None)
    assert "dev_masked_text_only" not in r1                       # single modality: no masked variants
    r2 = train(_cfg(use_text=False, epochs=1), root, tmp_path / "v",
               encode_fn=whitespace_encode, text_encoder=None, pad_id=1, log=lambda *_: None)
    assert r2["dev"]["emotion"]["n"] == 14


def test_max_train_rows_limits_the_training_set(synthetic_features, tmp_path):
    from meld_emotion.training.train import train
    root, _, _ = synthetic_features
    r = train(_cfg(epochs=1), root, tmp_path / "s", max_train_rows=10,
              encode_fn=whitespace_encode, text_encoder=StubTextEncoder(32), pad_id=1, log=lambda *_: None)
    assert r["train_rows"] == 10


def test_early_stopping_triggers_when_patience_exceeded(synthetic_features, tmp_path):
    """Verify that early stopping (patience exceeded) actually triggers and halts training before epochs."""
    from meld_emotion.training.train import train
    root, _, _ = synthetic_features
    # patience=1 means stop after 1 epoch without improvement; epochs=5 gives opportunity to trigger
    # With synthetic data and random initialization, dev F1 typically does not improve monotonically,
    # so early stopping should trigger.
    r = train(_cfg(patience=1, epochs=5, seed=2), root, tmp_path / "early_stop",
              encode_fn=whitespace_encode, text_encoder=StubTextEncoder(32), pad_id=1, log=lambda *_: None)

    # Verify early stopping actually triggered by checking that we stopped before epochs
    log_lines = (tmp_path / "early_stop" / "log.jsonl").read_text().strip().splitlines()
    epochs_in_log = len(log_lines)

    # With patience=1 on synthetic data, we expect to stop before epoch 5
    assert r["epochs_run"] < 5, f"Expected early stopping to trigger (epochs_run < 5), got {r['epochs_run']}"
    assert epochs_in_log < 5, f"Expected log to have < 5 lines, got {epochs_in_log}"
    assert epochs_in_log == r["epochs_run"], f"Log lines ({epochs_in_log}) should match epochs_run ({r['epochs_run']})"
