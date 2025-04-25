import litellm
from pydantic import BaseModel
from typing import Type, Optional
from litellm import acompletion, JSONSchemaValidationError

from cognee.shared.logging_utils import get_logger
from cognee.modules.observability.get_observe import get_observe
from cognee.exceptions import InvalidValueError
from cognee.infrastructure.llm.llm_interface import LLMInterface
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.infrastructure.llm.rate_limiter import (
    rate_limit_async,
    sleep_and_retry_async,
)

logger = get_logger()
observe = get_observe()


class GeminiAdapter(LLMInterface):
    MAX_RETRIES = 5

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int,
        endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        streaming: bool = False,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.api_version = api_version
        self.streaming = streaming
        self.max_tokens = max_tokens

    @observe(as_type="generation")
    @sleep_and_retry_async()
    @rate_limit_async
    async def acreate_structured_output(
        self, text_input: str, system_prompt: str, response_model: Type[BaseModel]
    ) -> BaseModel:
        try:
            if response_model is str:
                response_schema = {"type": "string"}
            else:
                response_schema = response_model

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_input},
            ]

            try:
                response = await acompletion(
                    model=f"{self.model}",
                    messages=messages,
                    api_key=self.api_key,
                    max_tokens=self.max_tokens,
                    temperature=0.1,
                    response_format=response_schema,
                    timeout=100,
                    num_retries=self.MAX_RETRIES,
                )

                if response.choices and response.choices[0].message.content:
                    content = response.choices[0].message.content
                    if response_model is str:
                        return content
                    return response_model.model_validate_json(content)

            except litellm.exceptions.BadRequestError as e:
                logger.error(f"Bad request error: {str(e)}")
                raise ValueError(f"Invalid request: {str(e)}")

            raise ValueError("Failed to get valid response after retries")

        except JSONSchemaValidationError as e:
            logger.error(f"Schema validation failed: {str(e)}")
            logger.debug(f"Raw response: {e.raw_response}")
            raise ValueError(f"Response failed schema validation: {str(e)}")

    def show_prompt(self, text_input: str, system_prompt: str) -> str:
        """Format and display the prompt for a user query."""
        if not text_input:
            text_input = "No user input provided."
        if not system_prompt:
            raise InvalidValueError(message="No system prompt path provided.")
        system_prompt = read_query_prompt(system_prompt)

        formatted_prompt = (
            f"""System Prompt:\n{system_prompt}\n\nUser Input:\n{text_input}\n"""
            if system_prompt
            else None
        )
        return formatted_prompt
