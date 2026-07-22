"""LLM adapter that delegates completions to the host via MCP sampling.

When cognee runs as an MCP server, this routes structured-output requests through
the host client's ``sampling/createMessage`` capability instead of a provider API,
so no ``LLM_API_KEY`` is needed (issue #3644).

MCP sampling returns free text — the protocol guarantees no JSON-schema or tool
mode — so structured output is produced by embedding the response model's JSON
Schema in the prompt and validating/repairing the reply. Audio transcription and
image description have no sampling equivalent.
"""

import json
from typing import Any

from pydantic import BaseModel, ValidationError

from cognee.infrastructure.llm.exceptions import MCPSamplingUnavailableError
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.llm_interface import (
    LLMInterface,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.types import (
    TranscriptionReturnType,
)
from cognee.shared.logging_utils import get_logger

from .session_context import get_sampling_session

logger = get_logger()


def _extract_json(text: str) -> str:
    """Extract a JSON object from a free-text reply that may be fenced or prosey.

    Takes the substring between the first ``{`` and the last ``}``; that alone
    strips Markdown code fences and surrounding prose.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text.strip()


class MCPSamplingAdapter(LLMInterface):
    """Delegate completions to the host harness via MCP sampling."""

    def __init__(
        self,
        model: str,
        max_completion_tokens: int,
        structured_output_retries: int = 5,
    ) -> None:
        # `model` is a preference hint only; the host chooses the actual model.
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.structured_output_retries = max(1, structured_output_retries)

    async def _sample(self, session: Any, system_prompt: str, user_text: str) -> str:
        """One ``sampling/createMessage`` round trip; returns the reply text."""
        # Messages must be typed: ``ServerSession.create_message`` validates them
        # (SEP-1577, ``msg.content_as_list``) before pydantic coercion, so plain
        # dicts crash on mcp >= 1.25. mcp is installed in any real sampling
        # deployment; the dict fallback only ever reaches mock sessions in
        # environments without mcp.
        try:
            from mcp.types import SamplingMessage, TextContent  # ty:ignore[unresolved-import]

            messages: list[Any] = [
                SamplingMessage(role="user", content=TextContent(type="text", text=user_text))
            ]
        except ImportError:
            messages = [{"role": "user", "content": {"type": "text", "text": user_text}}]

        result = await session.create_message(
            messages=messages,
            max_tokens=self.max_completion_tokens,
            system_prompt=system_prompt,
        )
        # CreateMessageResult.content is a single text block.
        text = getattr(result.content, "text", None)
        return text if text is not None else str(result.content)

    async def acreate_structured_output(
        self,
        text_input: str,
        system_prompt: str,
        response_model: type[BaseModel | str],
        **kwargs: Any,
    ) -> BaseModel | str:
        """
        Generate structured output by delegating to the host via MCP sampling.

        The response model's JSON Schema is embedded in the prompt and the host's
        free-text reply is validated (and repaired on failure) against it.

        Parameters:
        -----------

            - text_input (str): The input text from the user to be processed.
            - system_prompt (str): The system prompt that guides the model's response.
            - response_model (type[BaseModel | str]): The model type that structures the
              response, or ``str`` for a plain-text reply.

        Returns:
        --------

            - BaseModel | str: The validated structured output, or the raw reply text when
              ``response_model`` is ``str``.
        """
        session = get_sampling_session()
        if session is None:
            raise MCPSamplingUnavailableError()

        # Plain-string responses need no schema.
        if response_model is str:
            return await self._sample(session, system_prompt, text_input)

        if not (isinstance(response_model, type) and issubclass(response_model, BaseModel)):
            raise TypeError(
                "MCPSamplingAdapter.acreate_structured_output expects a Pydantic model or `str`, "
                f"got {response_model!r}"
            )

        schema = json.dumps(response_model.model_json_schema())
        instructions = (
            "Respond with ONLY a single JSON object that validates against this JSON Schema. "
            "Do not include any prose, explanation, or Markdown code fences.\n\n"
            f"JSON Schema:\n{schema}"
        )
        prompt = f"{system_prompt}\n\n{instructions}"

        last_error: Exception | None = None
        for attempt in range(self.structured_output_retries):
            raw = await self._sample(session, prompt, text_input)
            try:
                return response_model.model_validate_json(_extract_json(raw))
            except (ValidationError, ValueError) as error:
                last_error = error
                logger.warning(
                    f"MCP sampling structured output failed validation "
                    f"(attempt {attempt + 1}/{self.structured_output_retries}): {error}"
                )
                # Repair: show the validation error and ask for corrected JSON.
                prompt = (
                    f"{system_prompt}\n\n{instructions}\n\n"
                    f"The previous response did not validate: {error}. Return corrected JSON only."
                )

        raise ValueError(
            "MCP sampling could not produce structured output matching "
            f"{response_model.__name__} after {self.structured_output_retries} attempts. "
            f"Last error: {last_error}"
        )

    async def create_transcript(self, input: str, **kwargs: Any) -> TranscriptionReturnType | None:
        """Audio transcription has no MCP sampling equivalent — graceful no-audio."""
        logger.debug("create_transcript is not supported over MCP sampling; returning None")
        return None

    async def transcribe_image(self, input: str) -> Any:
        """Image description has no MCP sampling equivalent; use a vision-capable provider."""
        raise NotImplementedError(
            "Image description is not supported over MCP sampling. Configure a vision-capable "
            "LLM provider instead."
        )
