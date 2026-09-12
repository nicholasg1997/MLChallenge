#!/usr/bin/env python3
"""Stage 2 on Modal (A10G): extract the split tarballs to local disk in the
container, then run train_stage2() with the ViT in the loop.

    uv run --with modal modal run --detach scripts/modal_stage2.py --bases fusion --seeds 0
    uv run --with modal modal run --detach scripts/modal_stage2.py --bases fusion,fusion_no_scene --seeds 0,1,2 --eval-test
Results: same layout as Stage 1 under /out/stage2_<base>/seed<N>/ on meld-results.
"""
import modal

app = modal.App("meld-stage2")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.2", "torchvision>=0.17", "transformers>=4.40,<6", "scikit-learn>=1.4", "numpy>=1.26",
                 "pillow>=10.0", "opencv-python-headless>=4.9")
    .add_local_python_source("meld_emotion")
)
features = modal.Volume.from_name("meld-features", create_if_missing=True)
crops = modal.Volume.from_name("meld-crops", create_if_missing=True)
results = modal.Volume.from_name("meld-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("meld-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="A10G", cpu=4, memory=16384, timeout=4 * 3600,   # CPU/RAM are billed too
              volumes={"/vol": features, "/crops": crops, "/out": results, "/root/.cache/huggingface": hf_cache})
def run(base: str, seed: int, eval_test: bool = False, epochs: int | None = None, patience: int | None = None) -> dict:
    import os, subprocess, time
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from meld_emotion.training.config import stage2_config
    from meld_emotion.training.train_stage2 import train_stage2

    started = time.perf_counter()
    for split in ("train", "dev") + (("test",) if eval_test else ()):
        subprocess.run(["tar", "-xf", f"/crops/crops_{split}.tar", "-C", "/tmp"], check=True)
    print(f"crops extracted in {time.perf_counter() - started:.0f}s", flush=True)

    overrides = {"seed": seed, "device": "cuda", "loader_workers": 4,
                 **({"epochs": epochs} if epochs else {}), **({"patience": patience} if patience else {})}
    config = stage2_config(base, **overrides)
    out_dir = f"/out/{config.name}/seed{seed}"
    r = train_stage2(config, "/vol/features", "/tmp", out_dir, init_from=f"/out/{base}/seed0/best.pt",
                     test_split="test" if eval_test else None, on_epoch_end=lambda record: results.commit())
    results.commit()
    return {"run": config.name, "seed": seed, "best_epoch": r["best_epoch"],
            "dev_weighted_f1": r["dev"]["emotion"]["weighted_f1"], "dev_macro_f1": r["dev"]["emotion"]["macro_f1"],
            "test_weighted_f1": r.get("test", {}).get("emotion", {}).get("weighted_f1"),
            "minutes": round(r["wall_clock_s"] / 60, 1), "peak_gpu_mb": round(r.get("peak_gpu_mb", 0))}


@app.local_entrypoint()
def main(bases: str = "fusion", seeds: str = "0", eval_test: bool = False, epochs: int = 0, patience: int = 0):
    jobs = [(b, int(s), eval_test, epochs or None, patience or None) for b in bases.split(",") for s in seeds.split(",")]
    print(f"launching {len(jobs)} Stage 2 run(s) on A10G: {jobs}")
    for out in run.starmap(jobs):
        print(out)
