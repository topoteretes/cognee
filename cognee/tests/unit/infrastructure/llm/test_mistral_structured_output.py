"""Regression test: MistralAdapter returns the instructor result directly.

``self.aclient`` is an ``instructor.from_litellm(...)`` client, so calling
``chat.completions.create(..., response_model=...)`` returns the *validated
response_model instance*, not a raw ``ChatCompletion``. The adapter used to do
``response.choices[0].message.content`` on that instance, which has no
``choices`` attribute -> ``AttributeError`` (retried by tenacity, then reraised),
making structured output completely broken under ``LLM_PROVIDER="mistral"``.
Every sibling adapter (OpenAI, Gemini, Anthropic, Generic) returns the result
directly. This test drives the method with a stubbed instructor client.
"""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

# The mistral adapter imports `mistralai` at module load; skip when the optional
# `mistral` extra isn't installed (e.g. default CI) rather than erroring on collect.
pytest.importorskip("mistralai")

from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.mistral import (  # noqa: E402
    adapter as mistral_adapter_module,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.mistral.adapter import (  # noqa: E402
    MistralAdapter,
)


class _Answer(BaseModel):
    value: str


@pytest.mark.asyncio
async def test_returns_validated_model_from_instructor(monkeypatch):
    @asynccontextmanager
    async def _noop_rate_limiter(*_args, **_kwargs):
        yield

    monkeypatch.setattr(
        mistral_adapter_module, "llm_rate_limiter_context_manager", _noop_rate_limiter
    )

    expected = _Answer(value="ok")
    create = AsyncMock(return_value=expected)
    # Stub self: an instructor client returns the validated model instance.
    stub = SimpleNamespace(
        llm_args={},
        model="mistral/mistral-small-latest",
        aclient=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )

    result = await MistralAdapter.acreate_structured_output(
        stub, text_input="hello", system_prompt="sys", response_model=_Answer
    )

    # The instructor result is returned as-is (no .choices dereference).
    assert result is expected
    create.assert_awaited_once()
    assert create.await_args.kwargs["response_model"] is _Answer
