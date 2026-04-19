"""Hugging Face helper utilities for prompt enhancement."""

from __future__ import annotations

from huggingface_hub import InferenceClient

import config


def _make_client() -> InferenceClient:
    """Build an InferenceClient across huggingface_hub versions."""
    token = config.get_hf_api_key() or None
    try:
        return InferenceClient(provider="hf-inference", api_key=token)
    except TypeError:
        return InferenceClient(token=token)


def generate_response(prompt: str, temperature: float = 0.4, max_tokens: int = 220) -> str:
    """Generate a best-effort enhanced prompt and never raise."""
    cleaned_prompt = (prompt or "").strip()
    if not cleaned_prompt:
        return ""

    client = _make_client()
    model = config.HF_TEXT_MODEL

    # Newer API path.
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

    # Older API path.
    try:
        out = client.chat_completion(
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

    # Text generation fallback.
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

    return cleaned_prompt
