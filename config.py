"""App configuration and secrets resolution."""

from __future__ import annotations

import os


def _normalize_token(value: str) -> str:
    """Normalize token values from env/secrets."""
    token = (value or "").strip().strip("'").strip('"')
    if token.lower().startswith("bearer "):
        token = token.split(" ", 1)[1].strip()
    return token


def get_hf_api_key() -> str:
    """Resolve Hugging Face token from env vars and Streamlit secrets."""
    keys = (
        "HF_API_KEY",
        "HF_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    )

    for key in keys:
        token = _normalize_token(os.getenv(key, ""))
        if token:
            return token

    try:
        import streamlit as st

        for key in keys:
            value = st.secrets.get(key, "")
            if isinstance(value, str):
                token = _normalize_token(value)
                if token:
                    return token
    except Exception:
        pass

    return ""


HF_API_KEY = get_hf_api_key()
HF_TEXT_MODEL = os.getenv("HF_TEXT_MODEL", "mistralai/Mistral-7B-Instruct-v0.2").strip()
HF_MODELS = [x.strip() for x in os.getenv("HF_MODELS", HF_TEXT_MODEL).split(",") if x.strip()]
