# Where the face channel changes the answer

2610 test clips, binned by the text-only model's max probability.

| text confidence | n | text-only (k=4) | Stage 1 fusion | Stage 2 fusion |
|---|---|---|---|---|
| [0.0, 0.5) | 713 | 35.2% | 39.3% | 43.9% |
| [0.5, 0.7) | 744 | 58.6% | 54.8% | 58.7% |
| [0.7, 0.9) | 908 | 76.5% | 75.7% | 76.9% |
| [0.9, 1.0) | 245 | 92.7% | 92.7% | 92.7% |

- **Stage 1 fusion overrides the text model on 509 clips (19.5%)**: fused right 168, text right 175, neither 166 (mean text confidence there 0.48 vs 0.65 overall).
- **Stage 2 fusion overrides the text model on 504 clips (19.3%)**: fused right 203, text right 137, neither 164 (mean text confidence there 0.46 vs 0.65 overall).
- Stage 2's correct overrides by true class: {'neutral': 69, 'anger': 58, 'joy': 30, 'surprise': 20, 'sadness': 17, 'fear': 5, 'disgust': 4}.
