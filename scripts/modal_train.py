#!/usr/bin/env python3
"""Stage 1 trainings on Modal (A10G), ablations × seeds in parallel.

One-time setup -- upload the cached features (~1 GB, a few minutes):
    uv run --with modal modal volume create meld-features
    uv run --with modal modal volume put meld-features data/meld/features /features

Run (each ablation×seed gets its own GPU; ~10-15 min each):
    uv run --with modal modal run scripts/modal_train.py --ablations fusion,text_only_k4 --seeds 0
    uv run --with modal modal run scripts/modal_train.py --ablations fusion --seeds 0,1,2 --eval-test

Fetch results into the local results/ tree, then build the table as usual:
    uv run --with modal modal volume get meld-results / results/ --force
    uv run python scripts/ablation_table.py
"""
import modal

app = modal.App("meld-stage1")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.2", "transformers>=4.40,<6", "scikit-learn>=1.4", "numpy>=1.26",
                 "pillow>=10.0", "opencv-python-headless>=4.9")   # cv2: imported transitively via data.preprocess
    .add_local_python_source("meld_emotion")
)
features = modal.Volume.from_name("meld-features", create_if_missing=True)
results = modal.Volume.from_name("meld-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("meld-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=3 * 3600,
              volumes={"/vol": features, "/out": results, "/root/.cache/huggingface": hf_cache})
def run(ablation: str, seed: int, eval_test: bool = False, epochs: int | None = None) -> dict:
    from meld_emotion.training.config import config_for
    from meld_emotion.training.train import train

    overrides = {"seed": seed, "device": "cuda", **({"epochs": epochs} if epochs else {})}
    out_dir = f"/out/{ablation}/seed{seed}"
    r = train(config_for(ablation, **overrides), "/vol/features", out_dir,
              test_split="test" if eval_test else None,
              on_epoch_end=lambda record: results.commit())   # log.jsonl + best.pt survive a killed job
    results.commit()
    return {"ablation": ablation, "seed": seed, "best_epoch": r["best_epoch"],
            "dev_weighted_f1": r["dev"]["emotion"]["weighted_f1"],
            "dev_macro_f1": r["dev"]["emotion"]["macro_f1"],
            "test_weighted_f1": r.get("test", {}).get("emotion", {}).get("weighted_f1"),
            "minutes": round(r["wall_clock_s"] / 60, 1), "peak_gpu_mb": round(r.get("peak_gpu_mb", 0))}


@app.local_entrypoint()
def main(ablations: str = "fusion", seeds: str = "0", eval_test: bool = False, epochs: int = 0):
    jobs = [(a, int(s), eval_test, epochs or None) for a in ablations.split(",") for s in seeds.split(",")]
    print(f"launching {len(jobs)} run(s) on A10G: {jobs}")
    for out in run.starmap(jobs):
        print(out)
