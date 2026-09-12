"""Stage 2 training loop (design doc §6): the Stage 1 loop with the face
ViT in front of the fusion model, three learning rates (face < text <
fusion), bf16 autocast on CUDA, and a checkpoint that also carries the
fine-tuned face-encoder weights for the demo."""
import contextlib
import functools
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from meld_emotion.data.labels import EMOTIONS
from meld_emotion.training.config import TrainConfig
from meld_emotion.training.crops import MeldCropDataset, collate_crops
from meld_emotion.training.dataset import BucketBatchSampler, infer_feature_dims, load_ok_rows
from meld_emotion.training.metrics import class_weights, joint_loss
from meld_emotion.training.model import FusionModel
from meld_emotion.training.stage2_model import Stage2Model, init_fusion_from_stage1
from meld_emotion.training.text import build_text_encoder, build_tokenizer, count_parameters, encode_text
from meld_emotion.training.train import (_metrics_only, _to_device, build_scheduler, evaluate, peak_memory_mb,
                                         resolve_device, seed_everything)


def build_stage2_optimizer(model: Stage2Model, config: TrainConfig) -> torch.optim.AdamW:
    groups = [{"params": [p for p in model.face_parameters() if p.requires_grad], "lr": config.lr_face}]
    text_params = [p for p in model.text_parameters() if p.requires_grad]
    if text_params:
        groups.append({"params": text_params, "lr": config.lr_text})
    groups.append({"params": [p for p in model.fusion_parameters() if p.requires_grad], "lr": config.lr_fusion})
    return torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=config.weight_decay)


def _autocast(device: str, enabled: bool):
    if device == "cuda" and enabled:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def train_stage2(config: TrainConfig, features_dir: Path, preprocessed_dir: Path, out_dir: Path, *,
                 train_split: str = "train", dev_split: str = "dev", test_split: str | None = None,
                 encode_fn=None, text_encoder=None, face_encoder=None, pad_id: int | None = None,
                 init_from: Path | None = None, max_train_rows: int | None = None,
                 log=print, on_epoch_end=None) -> dict:
    features_dir, preprocessed_dir, out_dir = Path(features_dir), Path(preprocessed_dir), Path(out_dir)
    init_from = Path(init_from) if init_from is not None else (Path(config.init_from) if config.init_from else None)
    if init_from is None or not init_from.exists():
        raise FileNotFoundError(f"Stage 2 needs a Stage 1 checkpoint to initialise from; got {init_from}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    seed_everything(config.seed)
    started = time.perf_counter()

    # --- text side (same as Stage 1) ---
    if config.use_text and (encode_fn is None or text_encoder is None or pad_id is None):
        tokenizer = build_tokenizer(config.text_model)
        encode_fn = encode_fn or functools.partial(encode_text, tokenizer, max_length=config.max_text_tokens)
        text_encoder = text_encoder or build_text_encoder(config.text_model, config.text_trainable_layers)
        pad_id = tokenizer.pad_token_id if pad_id is None else pad_id
    if not config.use_text:
        encode_fn = encode_fn or (lambda context, current: {"input_ids": [0], "attention_mask": [1]})
        pad_id = 0 if pad_id is None else pad_id
        text_encoder = None

    # --- face encoder in the loop ---
    if face_encoder is None:
        from meld_emotion.training.face_encoder import TrainableFaceEncoder
        face_encoder = TrainableFaceEncoder(trainable_layers=config.face_trainable_layers)

    # --- data ---
    train_rows = load_ok_rows(features_dir / train_split / "manifest.jsonl")
    if max_train_rows:
        train_rows = train_rows[:max_train_rows]
    dev_rows = load_ok_rows(features_dir / dev_split / "manifest.jsonl")
    _, scene_dim = infer_feature_dims(train_rows, features_dir)
    collate_fn = functools.partial(collate_crops, pad_id=pad_id)

    def make_loader(rows, batch_size, shuffle, train):
        ds = MeldCropDataset(rows, preprocessed_dir, features_dir, config, encode_fn, train=train)
        sampler = BucketBatchSampler(ds.lengths, batch_size, shuffle=shuffle, seed=config.seed)
        return DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn, num_workers=config.loader_workers,
                          persistent_workers=config.loader_workers > 0), sampler

    train_loader, train_sampler = make_loader(train_rows, config.batch_size, shuffle=True, train=True)
    dev_loader, _ = make_loader(dev_rows, config.batch_size, shuffle=False, train=False)

    # --- model: Stage 1 weights for everything but the ViT ---
    fusion = FusionModel(config, text_encoder, face_dim=face_encoder.feature_dim, scene_dim=scene_dim)
    base_ckpt = init_fusion_from_stage1(fusion, init_from)
    model = Stage2Model(fusion, face_encoder).to(device)
    params_total, params_trainable = count_parameters(model)
    face_trainable = sum(p.numel() for p in model.face_parameters() if p.requires_grad)
    optimizer = build_stage2_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config.epochs * len(train_loader), config.warmup_fraction)
    emotion_weight = class_weights(np.array([EMOTIONS.index(r["emotion"]) for r in train_rows]),
                                   len(EMOTIONS), config.class_weight_alpha).to(device)
    log(f"[{config.name}] device={device} train={len(train_rows)} dev={len(dev_rows)} params={params_total:,} "
        f"trainable={params_trainable:,} (face {face_trainable:,}) steps/epoch={len(train_loader)} "
        f"init={init_from} (stage1 dev wF1 {base_ckpt.get('dev_weighted_f1', float('nan')):.4f})")

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
            with _autocast(device, config.amp_bf16):
                out = model(batch)
            out = {k: v.float() for k, v in out.items()}
            loss, _ = joint_loss(out, batch, emotion_weight, config.sentiment_lambda, config.label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        with _autocast(device, config.amp_bf16):
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
            torch.save({"model_state": model.fusion.state_dict(), "face_encoder_state": model.face_encoder.export_state(),
                        "config": asdict(config), "epoch": epoch, "dev_weighted_f1": dev_f1,
                        "face_dim": face_encoder.feature_dim, "scene_dim": scene_dim, "stage": 2,
                        "base_checkpoint": str(init_from)}, out_dir / "best.pt")
        else:
            epochs_without_gain += 1
        if on_epoch_end is not None:
            on_epoch_end(record)
        if epochs_without_gain >= config.patience:
            log(f"  early stop: no dev improvement for {config.patience} epochs")
            break

    # --- final evaluation from the best checkpoint ---
    ckpt = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
    model.fusion.load_state_dict(ckpt["model_state"])
    model.face_encoder.load_state_dict(ckpt["face_encoder_state"]) if not hasattr(model.face_encoder, "model") \
        else model.face_encoder.model.load_state_dict(ckpt["face_encoder_state"])
    model.to(device)
    with _autocast(device, config.amp_bf16):
        results = {"config": asdict(config), "stage": 2, "base_checkpoint": str(init_from), "device": device,
                   "best_epoch": best_epoch, "epochs_run": epoch, "train_rows": len(train_rows), "dev_rows": len(dev_rows),
                   "params_total": params_total, "params_trainable": params_trainable, "face_trainable_params": face_trainable,
                   "dev": _metrics_only(evaluate(model, dev_loader, device))}
        if config.use_text and config.use_vision:
            results["dev_masked_text_only"] = _metrics_only(evaluate(model, dev_loader, device, force_drop_vision=True))
            results["dev_masked_vision_only"] = _metrics_only(evaluate(model, dev_loader, device, force_drop_text=True))
        if test_split:
            test_rows = load_ok_rows(features_dir / test_split / "manifest.jsonl")
            test_loader, _ = make_loader(test_rows, config.batch_size, shuffle=False, train=False)
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
