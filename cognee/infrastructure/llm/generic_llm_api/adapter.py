"""Adapter for Generic API LLM provider API"""

import asyncio
from typing import List, Type
from pydantic import BaseModel
import instructor
from cognee.infrastructure.llm.llm_interface import LLMInterface
from cognee.infrastructure.llm.config import get_llm_config
import litellm


class GenericAPIAdapter(LLMInterface):
    """Adapter for Generic API LLM provider API"""

    name: str
    model: str
    api_key: str

    def __init__(self, endpoint, api_key: str, model: str, name: str):
        self.name = name
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint

        llm_config = get_llm_config()

        if llm_config.llm_provider == "ollama":
            self.aclient = instructor.from_litellm(litellm.acompletion)

        else:
            self.aclient = instructor.from_litellm(litellm.acompletion)

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
