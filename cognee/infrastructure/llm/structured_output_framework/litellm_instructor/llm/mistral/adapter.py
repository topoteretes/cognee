import litellm
import instructor
from pydantic import BaseModel
from typing import Type, Optional
from litellm import acompletion, JSONSchemaValidationError

from cognee.shared.logging_utils import get_logger
from cognee.modules.observability.get_observe import get_observe
from cognee.infrastructure.llm.exceptions import MissingSystemPromptPathError
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.llm_interface import (
    LLMInterface,
)
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.config import get_llm_config
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.rate_limiter import (
    rate_limit_async,
    sleep_and_retry_async,
)

logger = get_logger()
observe = get_observe()


class MistralAdapter(LLMInterface):
    """
    Adapter for Mistral AI API, for structured output generation and prompt display.

    Public methods:
    - acreate_structured_output
    - show_prompt
    """

    name = "Mistral"
    model: str
    api_key: str
    max_completion_tokens: int

    def __init__(self, api_key: str, model: str, max_completion_tokens: int, endpoint: str = None):
        from mistralai import Mistral

        self.model = model
        self.max_completion_tokens = max_completion_tokens

        self.aclient = instructor.from_litellm(
            litellm.acompletion,
            mode=instructor.Mode.MISTRAL_TOOLS,
            api_key=get_llm_config().llm_api_key,
        )

    @sleep_and_retry_async()
    @rate_limit_async
    async def acreate_structured_output(
        self, text_input: str, system_prompt: str, response_model: Type[BaseModel]
    ) -> BaseModel:
        """
        Generate a response from the user query.

        Parameters:
        -----------
            - text_input (str): The input text from the user to be processed.
            - system_prompt (str): A prompt that sets the context for the query.
            - response_model (Type[BaseModel]): The model to structure the response according to
              its format.

        Returns:
        --------
            - BaseModel: An instance of BaseModel containing the structured response.
        """
        return await self.aclient.chat.completions.create(
            model=self.model,
            max_tokens=self.max_completion_tokens,
            max_retries=5,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": f"""Use the given format to extract information
            from the following input: {text_input}""",
                },
            ],
            response_model=response_model,
        )

    def show_prompt(self, text_input: str, system_prompt: str) -> str:
        """
        Format and display the prompt for a user query.

        Parameters:
        -----------
            - text_input (str): Input text from the user to be included in the prompt.
            - system_prompt (str): The system prompt that will be shown alongside the user input.

        Returns:
        --------
            - str: The formatted prompt string combining system prompt and user input.
        """
        if not text_input:
            text_input = "No user input provided."
        if not system_prompt:
            raise MissingSystemPromptPathError()

        system_prompt = LLMGateway.read_query_prompt(system_prompt)

        formatted_prompt = (
            f"""System Prompt:\n{system_prompt}\n\nUser Input:\n{text_input}\n"""
            if system_prompt
            else None
        )

        return formatted_prompt
