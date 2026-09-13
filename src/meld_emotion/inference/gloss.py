"""Visual gloss (design doc §4.4): human-readable cues for the response LM
from tensors already computed. (a) CLIP zero-shot scene cues against a
fixed prompt bank embedded once; (b) the fusion model's vision-only
reading of the faces. Feeds the LM only -- never the classifier."""
import torch
from transformers import CLIPTokenizer

from meld_emotion.vision.encoders import CLIP_MODEL_ID

SCENE_PROMPT_BANK = (
    "one person", "two people talking", "a group of people", "a dimly lit room",
    "a brightly lit room", "people sitting on a couch", "someone standing",
    "someone leaning in close", "an animated gesture", "someone with arms crossed",
    "a person laughing", "a person crying", "people in a kitchen", "people in a coffee shop",
    "an empty background", "someone pointing", "a crowded room", "two people arguing",
    "a quiet, calm scene", "someone holding an object", "people hugging",
    "a person looking away from the camera", "a close-up of a face", "a wide shot of a room",
)


def build_clip_tokenizer() -> CLIPTokenizer:
    return CLIPTokenizer.from_pretrained(CLIP_MODEL_ID)


def embed_prompt_bank(clip_model, clip_tokenizer, device: str) -> torch.Tensor:
    inputs = clip_tokenizer(list(SCENE_PROMPT_BANK), return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = clip_model.get_text_features(**inputs)
        feats = out if isinstance(out, torch.Tensor) else out.pooler_output   # transformers 5 wraps it
    return torch.nn.functional.normalize(feats.float().cpu(), dim=-1)


def scene_gloss(mean_scene_embedding, bank_embeddings: torch.Tensor, margin: float = 0.05) -> list[str]:
    query = torch.nn.functional.normalize(
        torch.as_tensor(mean_scene_embedding, dtype=torch.float32).reshape(1, -1), dim=-1)
    probs = torch.softmax((query @ bank_embeddings.T).squeeze(0) * 100.0, dim=-1)   # CLIP's logit scale
    top2 = torch.topk(probs, min(2, len(probs)))
    if len(top2.values) < 2 or (top2.values[0] - top2.values[1]).item() < margin:
        return [SCENE_PROMPT_BANK[top2.indices[0]]]
    return [SCENE_PROMPT_BANK[i] for i in top2.indices.tolist()]


FACE_READING_THRESHOLD = 0.25   # a non-neutral emotion is "read" once its faces-only probability clears this


def face_reading(vision_only_expression: dict | None, threshold: float = FACE_READING_THRESHOLD) -> tuple[str, float]:
    """The faces-only state as one label: the top non-neutral emotion when it
    clears `threshold`, else neutral. The faces-only distribution is
    prior-heavy on MELD (neutral 0.4-0.6 almost always; masked vision-only
    wF1 0.27), so the argmax is uninformative and the signal is whether any
    other emotion has risen above the noise floor -- on a live face, a relaxed
    expression reads joy 0.11-0.21 and a deliberate smile 0.25-0.40."""
    if not vision_only_expression:
        return "neutral", 0.0
    label, prob = max(((k, v) for k, v in vision_only_expression.items() if k != "neutral"), key=lambda kv: kv[1])
    if prob >= threshold:
        return label, prob
    return "neutral", vision_only_expression.get("neutral", 0.0)


def face_gloss(faces_seen: int, vision_only_expression: dict | None, threshold: float = FACE_READING_THRESHOLD) -> str:
    if faces_seen <= 0 or not vision_only_expression:
        return "no faces visible"
    label, prob = face_reading(vision_only_expression, threshold)
    return f"{faces_seen} face{'s' if faces_seen != 1 else ''} visible, reading {label} ({prob:.0%})"
