"""Dispatch VLM prompts to the selected cloud provider."""

from __future__ import annotations

import base64

from .config import (
    OPENAI_COMPATIBLE_PROVIDERS,
    detect_provider,
    get_anthropic_client,
    get_gemini_client,
    get_openai_compatible_client,
)


def build_gemini_contents(prompt: str, images: list[dict[str, str]]) -> list[object]:
    """Build Google GenAI content parts."""
    from google.genai import types

    parts: list[object] = [prompt]
    for image in images:
        parts.append(
            types.Part.from_bytes(
                data=base64.b64decode(image["b64"]),
                mime_type=image["mime"],
            )
        )
    return parts


def build_openai_messages(prompt: str, images: list[dict[str, str]]) -> list[dict[str, object]]:
    """Build OpenAI-compatible chat messages with inline base64 images."""
    content: list[dict[str, object]] = []
    for image in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image['mime']};base64,{image['b64']}",
                },
            }
        )
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def build_anthropic_messages(prompt: str, images: list[dict[str, str]]) -> list[dict[str, object]]:
    """Build Anthropic messages for Claude vision models."""
    content: list[dict[str, object]] = []
    for image in images:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image["mime"],
                    "data": image["b64"],
                },
            }
        )
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _anthropic_text(response: object) -> str:
    """Extract text blocks from an Anthropic response object."""
    chunks: list[str] = []
    for block in getattr(response, "content", []):
        text = getattr(block, "text", None)
        if text is not None:
            chunks.append(str(text))
    return "".join(chunks)


def run_model(
    model_name: str,
    prompt: str,
    images: list[dict[str, str]],
    *,
    provider: str = "auto",
    max_tokens: int = 4096,
) -> str:
    """Run one inference call and return the model's text reply."""
    resolved_provider = detect_provider(model_name, provider)

    if resolved_provider == "gemini":
        client = get_gemini_client()
        response = client.models.generate_content(
            model=model_name,
            contents=build_gemini_contents(prompt, images),
        )
        return response.text

    if resolved_provider == "claude":
        client = get_anthropic_client()
        response = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            messages=build_anthropic_messages(prompt, images),
        )
        return _anthropic_text(response)

    if resolved_provider in OPENAI_COMPATIBLE_PROVIDERS:
        client = get_openai_compatible_client(resolved_provider)
        response = client.chat.completions.create(
            model=model_name,
            messages=build_openai_messages(prompt, images),
        )
        return response.choices[0].message.content

    raise ValueError(f"No VLM router branch for provider {resolved_provider!r}.")
