"""DeepSeek API adapter — single-call, self-discovering V4 cloud / local R1."""
import logging
import re
from typing import Any

import litellm
from openai import AsyncOpenAI
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_delay,
    wait_exponential_jitter,
)

from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.llm_interface import (
    LLMInterface,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.types import (
    TranscriptionReturnType,
)
from cognee.shared.logging_utils import get_logger
from cognee.shared.rate_limiting import llm_rate_limiter_context_manager
from cognee.infrastructure.files.utils.open_data_file import open_data_file

logger = get_logger()

# Strips <think>...</think> blocks from local R1 model responses.
# Handles: <think>, <THINK>, <think >, </think>, </THINK>
THINK_PATTERN = re.compile(
    r"<\s*think[^>]*>.*?<\s*/\s*think\s*>", re.DOTALL | re.IGNORECASE
)


class DeepSeekAPIAdapter(LLMInterface):
    """Adapter for DeepSeek API — single-call, self-discovering parsing.

    Makes one API call per request and inspects the response to decide
    whether <think> tag stripping is needed (local R1) or not (V4 cloud).
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str,
        name: str,
        max_completion_tokens: int,
        instructor_mode: str | None = None,
        llm_args: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint
        self.max_completion_tokens = max_completion_tokens
        self.llm_args: dict[str, Any] = llm_args or {}

        # instructor_mode is accepted for Cognee factory compatibility but not used.
        # The adapter uses native JSON mode (response_format) instead of instructor.
        if instructor_mode is not None:
            logger.debug(
                "instructor_mode=%s provided but not used — "
                "DeepSeek adapter uses native response_format=json_object.",
                instructor_mode,
            )

        # Single async client — one call per request, inspect response to discover
        # whether <think> tag stripping is needed (local R1) or not (V4 cloud).
        self.client = AsyncOpenAI(base_url=self.endpoint, api_key=self.api_key)

    @retry(
        stop=stop_after_delay(128),
        wait=wait_exponential_jitter(8, 128),
        retry=retry_if_not_exception_type(
            (litellm.exceptions.NotFoundError, litellm.exceptions.AuthenticationError)
        ),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
        reraise=True,
    )
    async def acreate_structured_output(
        self,
        text_input: str,
        system_prompt: str,
        response_model: type[BaseModel],
        **kwargs,
    ) -> BaseModel:
        """Single-call structured output. Inspects response to decide parse path.

        V4 cloud: clean JSON → parsed directly.
        Local R1: <think> tags inline → stripped first, then parsed.
        """
        merged_kwargs = {**self.llm_args, **kwargs}
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text_input},
        ]

        async with llm_rate_limiter_context_manager():
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **merged_kwargs,
            )
            content = response.choices[0].message.content or ""

            # V4 cloud: clean JSON — parse directly.
            # Local R1: <think> tags inline — strip first.
            if THINK_PATTERN.search(content):
                clean = THINK_PATTERN.sub("", content).strip()

                # Strip any markdown fences the model may have added
                if clean.startswith("```"):
                    clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
                    if clean.endswith("```"):
                        clean = clean[:-3]
                    clean = clean.strip()
                    if clean.startswith("json"):
                        clean = clean[4:].strip()

                return response_model.model_validate_json(clean)

            # No <think> tags — parse the clean JSON directly.
            return response_model.model_validate_json(content)

    @retry(
        stop=stop_after_delay(128),
        wait=wait_exponential_jitter(2, 128),
        retry=retry_if_not_exception_type(
            (litellm.exceptions.NotFoundError, litellm.exceptions.AuthenticationError)
        ),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
        reraise=True,
    )
    async def transcribe_image(self, input: str, **kwargs: Any) -> str:
        """DeepSeek V4 supports image input via OpenAI-compatible vision API."""
        import base64

        async with open_data_file(input, mode="rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                        },
                    ],
                }
            ],
            max_completion_tokens=300,
        )

        if not hasattr(response, "choices") or not response.choices:
            raise ValueError("Image transcription failed. No response received.")
        return response.choices[0].message.content

    async def create_transcript(self, input: str, **kwargs: Any) -> TranscriptionReturnType:
        """DeepSeek does not provide an audio transcription endpoint."""
        raise NotImplementedError(
            "DeepSeek does not support audio transcription. "
            "Use a separate LLM provider with OpenAI Whisper for transcription tasks."
        )
