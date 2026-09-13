| Component | Params | Trained? |
|---|---|---|
| RoBERTa-base + fusion transformer + projectors + heads | 135.7M | RoBERTa top half fine-tuned; fusion from scratch |
| Face/expression encoder (ViT-Base) | 85.8M | Fine-tuned top 4 layers (Stage 2) |
| CLIP ViT-B/32 (image + text towers; gloss only) | 151.3M | Frozen |
| Face detector (YuNet) | 0.1M | Zero-shot |
| Response LM (Qwen2.5-1.5B-Instruct-4bit) | 1.27B | Prompted, not fine-tuned |
| **Total** | **1.64B** | ceiling 6B |
