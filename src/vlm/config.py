"""Provider configuration for cloud VLM inference.

Credentials are loaded from ``.env`` via ``python-dotenv``. Clients are created
lazily so one missing provider package or key does not break the rest of the
pipeline.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

Provider = Literal["auto", "gemini", "openai", "grok", "claude", "github"]

OPENAI_COMPATIBLE_PROVIDERS = {"openai", "grok", "github"}

PROVIDER_ALIASES = {
    "anthropic": "claude",
    "xai": "grok",
    "openapi": "openai",
    "github_models": "github",
    "github-models": "github",
}

PROVIDER_CREDENTIALS = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "grok": ("XAI_API_KEY",),
    "claude": ("ANTHROPIC_API_KEY",),
    "github": ("GITHUB_TOKEN",),
}

PROVIDER_BASE_URLS = {
    "openai": "OPENAI_BASE_URL",
    "grok": "XAI_BASE_URL",
    "github": "GITHUB_MODELS_BASE_URL",
}

DEFAULT_BASE_URLS = {
    "grok": "https://api.x.ai/v1",
    "github": "https://models.inference.ai.azure.com",
}

# Exact matches are useful for defaults and older experiment commands. Prefix
# fallback in ``detect_provider`` handles new model names without code changes.
MODEL_FAMILIES = {
    "gemini": [
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemma-4-26b-a4b-it",
    ],
    "openai": [
        "gpt-4o",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-5",
        "gpt-5-mini",
    ],
    "grok": [
        "grok-4",
        "grok-4.1",
        "grok-4.1-fast",
    ],
    "claude": [
        "claude-3-5-sonnet-latest",
        "claude-3-7-sonnet-latest",
        "claude-sonnet-4",
        "claude-opus-4",
    ],
    "github": [
        "Phi-4-multimodal-instruct",
        "Llama-3.2-11B-Vision-Instruct",
    ],
}


def normalize_provider(provider: str | None) -> Provider:
    """Normalize provider aliases used in Makefile or CLI arguments."""
    if provider is None or provider == "":
        return "auto"
    normalized = provider.strip().lower().replace("_", "-")
    normalized = PROVIDER_ALIASES.get(normalized, normalized)
    if normalized not in {"auto", *PROVIDER_CREDENTIALS}:
        raise ValueError(
            f"Unknown VLM provider {provider!r}. Choose one of: "
            "auto, gemini, openai, grok, claude, github."
        )
    return normalized  # type: ignore[return-value]


def detect_provider(model_name: str, provider: str | None = "auto") -> Provider:
    """Return the provider for a model, honoring an explicit provider first."""
    requested = normalize_provider(provider)
    if requested != "auto":
        return requested

    model = model_name.strip()
    model_lower = model.lower()
    for family, models in MODEL_FAMILIES.items():
        if model in models:
            return family  # type: ignore[return-value]

    if model_lower.startswith(("gemini", "gemma")):
        return "gemini"
    if model_lower.startswith("grok"):
        return "grok"
    if model_lower.startswith("claude"):
        return "claude"
    if model_lower.startswith(("gpt", "o1", "o3", "o4", "o5")):
        return "openai"
    if model_lower.startswith(("phi", "llama")):
        return "github"

    return "openai"


def _first_env(names: Iterable[str]) -> tuple[str | None, str | None]:
    for name in names:
        value = os.getenv(name)
        if value:
            return name, value
    return None, None


def missing_credentials_for_provider(provider: str, model_name: str | None = None) -> list[str]:
    """Return human-readable missing credential requirements."""
    resolved_provider = detect_provider(model_name or "", provider)
    names = PROVIDER_CREDENTIALS[resolved_provider]
    _name, value = _first_env(names)
    if value:
        return []
    if len(names) == 1:
        return [names[0]]
    return [" or ".join(names)]


def require_credentials(provider: str, model_name: str | None = None) -> str:
    """Return the API key for provider, or raise a clear setup error."""
    resolved_provider = detect_provider(model_name or "", provider)
    names = PROVIDER_CREDENTIALS[resolved_provider]
    name, value = _first_env(names)
    if value:
        return value
    expected = " or ".join(names)
    raise RuntimeError(
        f"Missing credentials for VLM provider {resolved_provider!r}. "
        f"Set {expected} in .env or in the shell environment."
    )


def get_gemini_client():
    """Return a configured Google GenAI client."""
    api_key = require_credentials("gemini")
    from google import genai

    return genai.Client(api_key=api_key)


def get_openai_compatible_client(provider: str):
    """Return an OpenAI-compatible client for OpenAI, Grok/xAI, or GitHub Models."""
    resolved_provider = detect_provider("", provider)
    if resolved_provider not in OPENAI_COMPATIBLE_PROVIDERS:
        raise ValueError(f"{resolved_provider!r} is not OpenAI-compatible.")

    api_key = require_credentials(resolved_provider)
    base_url_var = PROVIDER_BASE_URLS[resolved_provider]
    base_url = os.getenv(base_url_var) or DEFAULT_BASE_URLS.get(resolved_provider)

    from openai import OpenAI

    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def get_anthropic_client():
    """Return a configured Anthropic client for Claude models."""
    api_key = require_credentials("claude")
    try:
        from anthropic import Anthropic
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Claude VLM inference requires the `anthropic` package. "
            "Update the environment with `conda env update -f environment.yml --prune`."
        ) from exc

    return Anthropic(api_key=api_key)
