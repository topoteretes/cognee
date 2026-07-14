"""LLM adapter that delegates completions to the host via MCP sampling.

When cognee runs as an MCP server, this adapter routes structured-output requests
through the host client's ``sampling/createMessage`` capability instead of a
provider API, so no ``LLM_API_KEY`` is needed (issue #3644).

Limitations, documented honestly:

* MCP sampling returns **free text** — the protocol guarantees no JSON-schema or
  tool mode. Structured output is therefore produced by embedding the response
  model's JSON Schema in the prompt and validating/repairing the reply, rather
  than relying on native schema enforcement.
* The sampling call must run in the server process, inside the MCP request scope
  (that is where the client connection lives). LLM calls that cognee runs in a
  detached background task or subprocess cannot reach the host session and will
  raise :class:`MCPSamplingUnavailableError`.
* Audio transcription and image description have no sampling equivalent.
"""

import json
from typing import Any

from pydantic import BaseModel, ValidationError

from cognee.infrastructure.llm.exceptions import MCPSamplingUnavailableError
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.llm_interface import (
    LLMInterface,
)
from cognee.shared.logging_utils import get_logger

from .session_context import get_sampling_session

logger = get_logger()


def _extract_json(text: str) -> str:
    """Best-effort extraction of a JSON object from a free-text sampling reply.

    Hosts often wrap JSON in Markdown fences or add prose; take the substring
    between the first ``{`` and the last ``}`` after stripping code fences.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Drop the opening fence (``` or ```json) and the trailing fence.
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned
        if cleaned.endswith("```"):
            cleaned = cleaned[: -len("```")]
        cleaned = cleaned.strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def _result_text(result: Any) -> str:
    """Pull the text out of an MCP ``CreateMessageResult`` (duck-typed)."""
    content = getattr(result, "content", result)
    if isinstance(content, list):
        parts = [getattr(block, "text", "") for block in content]
        return "".join(p for p in parts if p)
    text = getattr(content, "text", None)
    return text if text is not None else str(content)


class McpSamplingAdapter(LLMInterface):
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
        messages = [{"role": "user", "content": {"type": "text", "text": user_text}}]
        result = await session.create_message(
            messages=messages,
            max_tokens=self.max_completion_tokens,
            system_prompt=system_prompt,
        )
        return _result_text(result)

    async def acreate_structured_output(
        self, text_input: str, system_prompt: str, response_model: type, **kwargs: Any
    ) -> Any:
        # Peer adapters accept provider-specific params via **kwargs, and
        # LLMGateway forwards them (e.g. the LLM router passes request params),
        # so accept them for drop-in parity. MCP sampling has no provider knobs
        # (the host owns the model/params), so they are intentionally ignored.
        session = get_sampling_session()
        if session is None:
            raise MCPSamplingUnavailableError()

        # Plain-string responses need no schema.
        if response_model is str:
            return await self._sample(session, system_prompt, text_input)

        if not (isinstance(response_model, type) and issubclass(response_model, BaseModel)):
            raise TypeError(
                "McpSamplingAdapter.acreate_structured_output expects a Pydantic model or `str`, "
                f"got {response_model!r}"
            )

        schema = json.dumps(response_model.model_json_schema())
        schema_instructions = (
            "Respond with ONLY a single JSON object that validates against this JSON Schema. "
            "Do not include any prose, explanation, or Markdown code fences.\n\n"
            f"JSON Schema:\n{schema}"
        )
        prompt = f"{system_prompt}\n\n{schema_instructions}"

        last_error: Exception | None = None
        for attempt in range(self.structured_output_retries):
            raw = await self._sample(session, prompt, text_input)
            candidate = _extract_json(raw)
            try:
                return response_model.model_validate_json(candidate)
            except (ValidationError, ValueError) as error:
                last_error = error
                logger.warning(
                    "MCP sampling structured output failed validation (attempt %d/%d): %s",
                    attempt + 1,
                    self.structured_output_retries,
                    error,
                )
                # Repair prompt: show the model its error and ask for corrected JSON.
                prompt = (
                    f"{system_prompt}\n\n{schema_instructions}\n\n"
                    "Your previous response did not validate against the schema: "
                    f"{error}. Return corrected JSON only."
                )

        raise ValueError(
            "MCP sampling could not produce structured output matching "
            f"{response_model.__name__} after {self.structured_output_retries} attempts. "
            f"Last error: {last_error}"
        )

    async def create_transcript(self, input: str):
        """Audio transcription has no MCP sampling equivalent — graceful no-audio."""
        logger.debug("create_transcript is not supported over MCP sampling; returning None")
        return None

    async def transcribe_image(self, input: str) -> Any:
        raise NotImplementedError(
            "Image description is not supported over MCP sampling. Configure a vision-capable "
            "LLM provider instead."
        )
