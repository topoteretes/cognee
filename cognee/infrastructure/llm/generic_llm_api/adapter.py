"""Adapter for Generic API LLM provider API"""

from typing import Type

from pydantic import BaseModel
import instructor
from cognee.infrastructure.llm.llm_interface import LLMInterface
from cognee.infrastructure.llm.config import get_llm_config
from cognee.infrastructure.llm.rate_limiter import rate_limit_async, sleep_and_retry_async
import litellm


class GenericAPIAdapter(LLMInterface):
    """
    Adapter for Generic API LLM provider API.

    This class initializes the API adapter with necessary credentials and configurations for
    interacting with a language model. It provides methods for creating structured outputs
    based on user input and system prompts.

    Public methods:
    - acreate_structured_output(text_input: str, system_prompt: str, response_model:
    Type[BaseModel]) -> BaseModel
    """

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
        """
        Generate a response from a user query.

        This asynchronous method sends a user query and a system prompt to a language model and
        retrieves the generated response. It handles API communication and retries up to a
        specified limit in case of request failures.

        Parameters:
        -----------

            - text_input (str): The input text from the user to generate a response for.
            - system_prompt (str): A prompt that provides context or instructions for the
              response generation.
            - response_model (Type[BaseModel]): A Pydantic model that defines the structure of
              the expected response.

        Returns:
        --------

            - BaseModel: An instance of the specified response model containing the structured
              output from the language model.
        """

        return await self.aclient.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"""{text_input}""",
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
