import logging
from typing import Type
from pydantic import BaseModel
import litellm
import instructor
from cognee.shared.logging_utils import get_logger
from tenacity import (
    retry,
    stop_after_delay,
    wait_exponential_jitter,
    retry_if_not_exception_type,
    before_sleep_log,
)

from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.llm_interface import (
    LLMInterface,
)
from cognee.shared.rate_limiting import llm_rate_limiter_context_manager
from cognee.infrastructure.llm.config import get_llm_config

logger = get_logger()


class AnthropicAdapter(LLMInterface):
    """
    Adapter for interfacing with the Anthropic API, enabling structured output generation
    and prompt display.
    """

    name = "Anthropic"
    model: str
    default_instructor_mode = "anthropic_tools"

    def __init__(self, max_completion_tokens: int, model: str = None, instructor_mode: str = None):
        import anthropic

        self.instructor_mode = instructor_mode if instructor_mode else self.default_instructor_mode

        self.aclient = instructor.patch(
            create=anthropic.AsyncAnthropic(api_key=get_llm_config().llm_api_key).messages.create,
            mode=instructor.Mode(self.instructor_mode),
        )

        self.model = model
        self.max_completion_tokens = max_completion_tokens

    @retry(
        stop=stop_after_delay(128),
        wait=wait_exponential_jitter(8, 128),
        retry=retry_if_not_exception_type(litellm.exceptions.NotFoundError),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
        reraise=True,
    )
    async def acreate_structured_output(
        self, text_input: str, system_prompt: str, response_model: Type[BaseModel]
    ) -> BaseModel:
        """
        Generate a response from a user query.

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
        async with llm_rate_limiter_context_manager():
            return await self.aclient(
                model=self.model,
                max_tokens=4096,
                max_retries=2,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Use the given format to extract information
                    from the following input: {text_input}. {system_prompt}""",
                    }
                ],
                response_model=response_model,
            )
