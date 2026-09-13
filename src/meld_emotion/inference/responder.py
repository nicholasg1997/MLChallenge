"""Response generation (design doc §4.4): a fixed persona, prompted (not
fine-tuned) instruction LM, served locally via mlx-lm at 4-bit, streaming
tokens into the event stream. By the time this runs, the cross-modal
reasoning is already done (turn.py) -- its only job is a short, in-
character reactive line."""
SYSTEM_PROMPT = ("You are a small, friendly character robot. React to what the person just "
                 "said and how they seem to feel, in at most two sentences. Don't summarize "
                 "what they said back to them -- react to it. Never invent facts you weren't told.")


def build_prompt(context_prev: list[str], text: str, emotion: str, top2_probs: list[tuple[str, float]],
                 sentiment: str, visual_cues: list[str]) -> list[dict]:
    lines = "\n".join(context_prev[-4:])
    top2_str = ", ".join(f"{e} ({p:.0%})" for e, p in top2_probs)
    cues_str = ", ".join(visual_cues) if visual_cues else "none"
    user = (f"{lines}\n{text}\n\n"
           f"[Detected emotion: {top2_str}. Sentiment: {sentiment}. Visual cues: {cues_str}.]")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


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
