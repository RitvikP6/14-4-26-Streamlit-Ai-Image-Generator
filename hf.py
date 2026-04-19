"""Hugging Face helper utilities."""

from __future__ import annotations

from huggingface_hub import InferenceClient

import config


def _make_client() -> InferenceClient:
    """Build an InferenceClient across huggingface_hub versions."""
    token = config.get_hf_api_key() or None

    # Newer versions support the explicit provider argument.
    try:
        return InferenceClient(provider="hf-inference", api_key=token)
    except TypeError:
        # Older versions only support token/api_key kwargs.
        return InferenceClient(token=token)


def generate_response(prompt: str, temperature: float = 0.4, max_tokens: int = 220) -> str:
    """Generate a text completion used to enhance image prompts.

    Returns a best-effort string and never raises. On API failures we return
    the incoming prompt so the app can continue.
    """
    client = _make_client()
    model = config.HF_TEXT_MODEL
    cleaned_prompt = (prompt or "").strip()

    if not cleaned_prompt:
        return ""

    # Preferred path: chat-completions API.
    try:
        out = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": cleaned_prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = (
            out.choices[0].message.content
            if out and getattr(out, "choices", None)
            else None
        )
        if text:
            return text.strip()
    except Exception:
        pass

    # Fallback path: text_generation API.
    try:
        out = client.text_generation(
            prompt=cleaned_prompt,
            model=model,
            max_new_tokens=max_tokens,
            temperature=temperature,
            return_full_text=False,
        )
        if isinstance(out, str) and out.strip():
            return out.strip()
    except Exception:
        pass

    # Never block app flow if enhancement fails.
    return cleaned_prompt
