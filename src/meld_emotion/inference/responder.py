"""Response generation (design doc §4.4): a fixed persona, prompted (not
fine-tuned) instruction LM, served locally via mlx-lm at 4-bit, streaming
tokens into the event stream. By the time this runs, the cross-modal
reasoning is already done (turn.py) -- its only job is a short, in-
character reactive line.

The prompt keeps three things visibly separate for a small model: earlier
lines (context only), the line just spoken TO the robot, and the detected
state (a parenthetical hint). Without that separation a 1.5B-3B model reads
the transcript as a script and narrates it in the third person."""
SYSTEM_PROMPT = ("You are a small, friendly robot companion talking with one person. Reply to them "
                 "directly, in the second person, in one or two short sentences, the way a warm friend "
                 "reacts in the moment. Do not narrate, describe, or summarize what they said. Do not "
                 "mention the emotion label or the camera hints. Never invent facts you weren't told.")


def build_prompt(context_prev: list[str], text: str, emotion: str, top2_probs: list[tuple[str, float]],
                 sentiment: str, visual_cues: list[str]) -> list[dict]:
    hint = " or ".join(f"{e} ({p:.0%})" for e, p in top2_probs)
    cues_str = ", ".join(visual_cues) if visual_cues else "nothing notable"
    parts = []
    if context_prev:
        parts.append("Earlier in the conversation they said:\n" + "\n".join(f"- {line}" for line in context_prev[-4:]))
    parts.append(f'They just said to you: "{text}"')
    parts.append(f"(They seem {hint}; overall {sentiment}. Camera: {cues_str}.)")
    parts.append("Your reply:")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n\n".join(parts)}]


class Responder:
    def __init__(self, model_repo: str | None = None, max_tokens: int = 40, *, model=None, tokenizer=None):
        if model is None or tokenizer is None:
            from mlx_lm import load
            model, tokenizer = load(model_repo)
        self.model, self.tokenizer, self.max_tokens = model, tokenizer, max_tokens

    def stream(self, messages: list[dict], generate_fn=None):
        if generate_fn is None:
            from mlx_lm import stream_generate as generate_fn
        try:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        for response in generate_fn(self.model, self.tokenizer, prompt, max_tokens=self.max_tokens):
            yield response.text
