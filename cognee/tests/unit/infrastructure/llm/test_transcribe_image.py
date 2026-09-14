"""transcribe_image builds the correct multimodal request (prompt, image, token cap).

Only litellm.acompletion (the network boundary) is mocked, so the adapter's real request
construction — base64 encoding, MIME detection, message assembly, prompt/cap wiring — is exercised.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter import (
    GenericAPIAdapter,
)

IMAGE = Path(__file__).parents[3] / "test_data" / "revenue_table.png"


def _adapter() -> GenericAPIAdapter:
    return GenericAPIAdapter(
        api_key="test-key",
        model="openai/gpt-5-mini",
        max_completion_tokens=1024,
        name="test",
    )


def _fake_response():
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])


def _content_part(kwargs, part_type):
    parts = kwargs["messages"][0]["content"]
    return next(part for part in parts if part["type"] == part_type)


@pytest.mark.asyncio
async def test_transcribe_image_builds_request_with_prompt_and_cap():
    """Caller-supplied prompt, cap, and reasoning effort land in the litellm request."""
    fake = AsyncMock(return_value=_fake_response())

    with patch("litellm.acompletion", fake):
        await _adapter().transcribe_image(
            str(IMAGE),
            prompt="EXTRACT ENTITIES",
            max_completion_tokens=777,
            reasoning_effort="low",
        )

    kwargs = fake.call_args.kwargs
    assert _content_part(kwargs, "text")["text"] == "EXTRACT ENTITIES"
    assert kwargs["max_completion_tokens"] == 777
    assert kwargs["reasoning_effort"] == "low"


@pytest.mark.asyncio
async def test_transcribe_image_defaults_are_backwards_compatible():
    """With no prompt/cap the request keeps the legacy caption prompt and 300-token cap."""
    fake = AsyncMock(return_value=_fake_response())

    with patch("litellm.acompletion", fake):
        await _adapter().transcribe_image(str(IMAGE))

    kwargs = fake.call_args.kwargs
    assert _content_part(kwargs, "text")["text"] == "What's in this image?"
    assert kwargs["max_completion_tokens"] == 300
