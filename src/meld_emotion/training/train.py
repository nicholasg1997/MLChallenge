"""Stage 1 training loop (design doc §6): AdamW with two learning rates,
linear warmup + cosine decay, gradient clipping, dev-selected early stopping
on weighted-F1, best checkpoint, JSON-lines log, and a results.json that
records everything the write-up needs (metrics, masked-modality variants,
wall-clock, memory, parameter counts)."""
import functools
import json
import math
import random
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from meld_emotion.data.labels import EMOTIONS, SENTIMENTS
from meld_emotion.training.config import TrainConfig
from meld_emotion.training.dataset import (BucketBatchSampler, MeldFeatureDataset, collate,
                                           infer_feature_dims, load_ok_rows)
from meld_emotion.training.metrics import class_weights, compute_metrics, joint_loss
from meld_emotion.training.model import FusionModel
from meld_emotion.training.text import build_text_encoder, build_tokenizer, count_parameters, encode_text


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def peak_memory_mb(device: str) -> dict:
    """ru_maxrss is bytes on macOS but kilobytes on Linux; report both host and GPU peaks."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out = {"peak_rss_mb": rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024}
    if device == "cuda":
        out["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return out


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_optimizer(model: FusionModel, config: TrainConfig) -> torch.optim.AdamW:
    groups = []
    text_params = [p for p in model.text_parameters() if p.requires_grad]
    if text_params:
        groups.append({"params": text_params, "lr": config.lr_text})
    groups.append({"params": [p for p in model.fusion_parameters() if p.requires_grad], "lr": config.lr_fusion})
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def build_scheduler(optimizer, total_steps: int, warmup_fraction: float):
    warmup = max(1, int(total_steps * warmup_fraction))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _to_device(batch: dict, device: str) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model: FusionModel, loader: DataLoader, device: str,
             force_drop_text: bool = False, force_drop_vision: bool = False) -> dict:
    model.eval()
    clips, e_true, e_pred, e_probs, s_true, s_pred = [], [], [], [], [], []
    for batch in loader:
        batch = _to_device(batch, device)
        out = model(batch, force_drop_text=force_drop_text, force_drop_vision=force_drop_vision)
        probs = torch.softmax(out["emotion_logits"], dim=-1)
        clips += batch["clips"]
        e_true += batch["emotion"].tolist(); e_pred += probs.argmax(-1).tolist(); e_probs += probs.cpu().tolist()
        s_true += batch["sentiment"].tolist(); s_pred += out["sentiment_logits"].argmax(-1).tolist()
    return {"emotion": compute_metrics(np.array(e_true), np.array(e_pred), EMOTIONS),
            "sentiment": compute_metrics(np.array(s_true), np.array(s_pred), SENTIMENTS),
            "clips": clips, "emotion_pred": e_pred, "emotion_probs": e_probs}


def _metrics_only(ev: dict) -> dict:
    return {"emotion": ev["emotion"], "sentiment": ev["sentiment"]}


def load_checkpoint(path: Path, model: FusionModel) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return ckpt


def train(config: TrainConfig, features_dir: Path, out_dir: Path, *,
          train_split: str = "train", dev_split: str = "dev", test_split: str | None = None,
          encode_fn=None, text_encoder=None, pad_id: int | None = None,
          max_train_rows: int | None = None, log=print, on_epoch_end=None) -> dict:
    """`on_epoch_end(record)` is called after each epoch's log line and
    checkpoint are written -- the Modal runner uses it to commit the results
    volume so progress survives a killed container."""
    features_dir, out_dir = Path(features_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()   # a warm Modal container can carry over a previous job's peak otherwise
    seed_everything(config.seed)
    started = time.perf_counter()

    # --- text side: real RoBERTa unless stubs are injected ---
    if config.use_text and (encode_fn is None or text_encoder is None or pad_id is None):
        tokenizer = build_tokenizer(config.text_model)
        encode_fn = encode_fn or functools.partial(encode_text, tokenizer, max_length=config.max_text_tokens)
        text_encoder = text_encoder or build_text_encoder(config.text_model, config.text_trainable_layers)
        pad_id = tokenizer.pad_token_id if pad_id is None else pad_id
    if not config.use_text:
        encode_fn = encode_fn or (lambda context, current: {"input_ids": [0], "attention_mask": [1]})
        pad_id = 0 if pad_id is None else pad_id
        text_encoder = None

    # --- data ---
    train_rows = load_ok_rows(features_dir / train_split / "manifest.jsonl")
    if max_train_rows:
        train_rows = train_rows[:max_train_rows]
    dev_rows = load_ok_rows(features_dir / dev_split / "manifest.jsonl")
    face_dim, scene_dim = infer_feature_dims(train_rows, features_dir)
    collate_fn = functools.partial(collate, pad_id=pad_id)

    def make_loader(rows, batch_size, shuffle):
        ds = MeldFeatureDataset(rows, features_dir, config, encode_fn)
        sampler = BucketBatchSampler(ds.lengths, batch_size, shuffle=shuffle, seed=config.seed)
        return DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn, num_workers=0), sampler

    train_loader, train_sampler = make_loader(train_rows, config.batch_size, shuffle=True)
    dev_loader, _ = make_loader(dev_rows, config.batch_size * 2, shuffle=False)

    # --- model / optimisation ---
    model = FusionModel(config, text_encoder, face_dim=face_dim, scene_dim=scene_dim).to(device)
    params_total, params_trainable = count_parameters(model)
    optimizer = build_optimizer(model, config)
    total_steps = config.epochs * len(train_loader)
    scheduler = build_scheduler(optimizer, total_steps, config.warmup_fraction)
    emotion_weight = class_weights(np.array([b for b in (EMOTIONS.index(r["emotion"]) for r in train_rows)]),
                                   len(EMOTIONS), config.class_weight_alpha).to(device)
    log(f"[{config.name}] device={device} train={len(train_rows)} dev={len(dev_rows)} "
        f"params={params_total:,} trainable={params_trainable:,} steps/epoch={len(train_loader)}")

    # --- loop ---
    best_f1, best_epoch, epochs_without_gain = -1.0, 0, 0
    log_path = out_dir / "log.jsonl"
    log_path.write_text("")
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch)
        epoch_started = time.perf_counter()
        losses = []
        for batch in train_loader:
            batch = _to_device(batch, device)
            out = model(batch)
            loss, parts = joint_loss(out, batch, emotion_weight, config.sentiment_lambda, config.label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        dev_eval = evaluate(model, dev_loader, device)
        dev_f1 = dev_eval["emotion"]["weighted_f1"]
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "dev_weighted_f1": dev_f1,
                  "dev_macro_f1": dev_eval["emotion"]["macro_f1"], "dev_sentiment_f1": dev_eval["sentiment"]["weighted_f1"],
                  "epoch_s": time.perf_counter() - epoch_started, "lr": scheduler.get_last_lr()[-1]}
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        log(f"  epoch {epoch}: loss={record['train_loss']:.4f} dev_wF1={dev_f1:.4f} "
            f"dev_mF1={record['dev_macro_f1']:.4f} ({record['epoch_s']:.0f}s)")
        if dev_f1 > best_f1:
            best_f1, best_epoch, epochs_without_gain = dev_f1, epoch, 0
            torch.save({"model_state": model.state_dict(), "config": asdict(config), "epoch": epoch,
                        "dev_weighted_f1": dev_f1, "face_dim": face_dim, "scene_dim": scene_dim}, out_dir / "best.pt")
        else:
            epochs_without_gain += 1
        if on_epoch_end is not None:
            on_epoch_end(record)
        if epochs_without_gain >= config.patience:
            log(f"  early stop: no dev improvement for {config.patience} epochs")
            break

    # --- final evaluation from the best checkpoint ---
    load_checkpoint(out_dir / "best.pt", model)
    model.to(device)
    results = {"config": asdict(config), "device": device, "best_epoch": best_epoch, "epochs_run": epoch,
               "train_rows": len(train_rows), "dev_rows": len(dev_rows),
               "params_total": params_total, "params_trainable": params_trainable,
               "dev": _metrics_only(evaluate(model, dev_loader, device))}
    if config.use_text and config.use_vision:
        results["dev_masked_text_only"] = _metrics_only(evaluate(model, dev_loader, device, force_drop_vision=True))
        results["dev_masked_vision_only"] = _metrics_only(evaluate(model, dev_loader, device, force_drop_text=True))
    if test_split:
        test_rows = load_ok_rows(features_dir / test_split / "manifest.jsonl")
        test_loader, _ = make_loader(test_rows, config.batch_size * 2, shuffle=False)
        test_eval = evaluate(model, test_loader, device)
        results["test"] = _metrics_only(test_eval)
        results["test_rows"] = len(test_rows)
        with open(out_dir / "test_predictions.jsonl", "w") as f:
            for clip, pred, probs in zip(test_eval["clips"], test_eval["emotion_pred"], test_eval["emotion_probs"]):
                f.write(json.dumps({"clip": clip, "pred": EMOTIONS[pred], "probs": probs}) + "\n")
    results["wall_clock_s"] = time.perf_counter() - started
    results.update(peak_memory_mb(device))
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"[{config.name}] best epoch {best_epoch}: dev wF1={results['dev']['emotion']['weighted_f1']:.4f} "
        f"mF1={results['dev']['emotion']['macro_f1']:.4f} in {results['wall_clock_s'] / 60:.1f} min -> {out_dir}")
    return results
