import logging
from typing import Type
from pydantic import BaseModel
import litellm
import instructor
import anthropic
from cognee.shared.logging_utils import get_logger
from cognee.modules.observability.get_observe import get_observe
from tenacity import (
    retry,
    stop_after_delay,
    wait_exponential_jitter,
    retry_if_not_exception_type,
    before_sleep_log,
)

from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter import (
    GenericAPIAdapter,
)
from cognee.shared.rate_limiting import llm_rate_limiter_context_manager
from cognee.infrastructure.llm.config import get_llm_config

logger = get_logger()
observe = get_observe()


class AnthropicAdapter(GenericAPIAdapter):
    """
    Adapter for interfacing with the Anthropic API, enabling structured output generation
    and prompt display.
    """

    default_instructor_mode = "anthropic_tools"

    def __init__(
        self, api_key: str, model: str, max_completion_tokens: int, instructor_mode: str = None
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            max_completion_tokens=max_completion_tokens,
            name="Anthropic",
        )
        self.instructor_mode = instructor_mode if instructor_mode else self.default_instructor_mode

        self.aclient = instructor.patch(
            create=anthropic.AsyncAnthropic(api_key=self.api_key).messages.create,
            mode=instructor.Mode(self.instructor_mode),
        )

    @observe(as_type="generation")
    @retry(
        stop=stop_after_delay(128),
        wait=wait_exponential_jitter(8, 128),
        retry=retry_if_not_exception_type(litellm.exceptions.NotFoundError),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
        reraise=True,
    )
    async def acreate_structured_output(
        self, text_input: str, system_prompt: str, response_model: Type[BaseModel], **kwargs
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
