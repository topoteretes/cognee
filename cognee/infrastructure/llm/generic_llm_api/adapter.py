"""Adapter for Generic API LLM provider API"""

from typing import Type

from pydantic import BaseModel
import instructor
from cognee.infrastructure.llm.llm_interface import LLMInterface
from cognee.infrastructure.llm.config import get_llm_config
from cognee.infrastructure.llm.rate_limiter import rate_limit_async, sleep_and_retry_async
import litellm


class GenericAPIAdapter(LLMInterface):
    """Adapter for Generic API LLM provider API"""

    name: str
    model: str
    api_key: str

    def __init__(self, endpoint, api_key: str, model: str, name: str, max_tokens: int):
        self.name = name
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint
        self.max_tokens = max_tokens

        self.aclient = instructor.from_litellm(
            litellm.acompletion, mode=instructor.Mode.JSON, api_key=api_key
        )

    @sleep_and_retry_async()
    @rate_limit_async
    async def acreate_structured_output(
        self, text_input: str, system_prompt: str, response_model: Type[BaseModel]
    ) -> BaseModel:
        """Generate a response from a user query."""

        return await self.aclient.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"""Use the given format to
                extract information from the following input: {text_input}. """,
                },
                {
                    "role": "system",
                    "content": system_prompt,
                },
            ],
            max_retries=5,
            api_base=self.endpoint,
            response_model=response_model,
        )
