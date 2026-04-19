"""Configuration values for the Streamlit app.

All secrets are loaded from environment variables so the code can run
locally, in Streamlit Cloud, or any CI environment without edits.
"""

from __future__ import annotations

import os


def get_hf_api_key() -> str:
    """Resolve Hugging Face token from env vars and Streamlit secrets."""
    keys = (
        "HF_API_KEY",
        "HF_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    )

    for key in keys:
        value = os.getenv(key, "")
        if value and value.strip():
            return value.strip()

    # Optional: Streamlit secrets for deployed apps.
    try:
        import streamlit as st

        for key in keys:
            value = st.secrets.get(key, "")
            if isinstance(value, str) and value.strip():
                return value.strip()
    except Exception:
        pass

    return ""


# Hugging Face access token (recommended for reliable inference access).
HF_API_KEY = get_hf_api_key()

# Model used for prompt-enhancement text generation.
HF_TEXT_MODEL = os.getenv("HF_TEXT_MODEL", "mistralai/Mistral-7B-Instruct-v0.2").strip()
