"""Adapter for Generic API LLM provider API"""

import litellm
import instructor
from typing import Type
from pydantic import BaseModel
from openai import ContentFilterFinishReasonError
from litellm.exceptions import ContentPolicyViolationError
from instructor.core import InstructorRetryException

from cognee.infrastructure.llm.exceptions import ContentPolicyFilterError
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.llm_interface import (
    LLMInterface,
)
import logging
from cognee.shared.logging_utils import get_logger
from tenacity import (
    retry,
    stop_after_delay,
    wait_exponential_jitter,
    retry_if_not_exception_type,
    before_sleep_log,
)

logger = get_logger()


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

    def __init__(
        self,
        endpoint,
        api_key: str,
        model: str,
        name: str,
        max_completion_tokens: int,
        fallback_model: str = None,
        fallback_api_key: str = None,
        fallback_endpoint: str = None,
    ):
        self.name = name
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint
        self.max_completion_tokens = max_completion_tokens

        self.fallback_model = fallback_model
        self.fallback_api_key = fallback_api_key
        self.fallback_endpoint = fallback_endpoint

        self.aclient = instructor.from_litellm(litellm.acompletion, mode=instructor.Mode.JSON)

    @retry(
        stop=stop_after_delay(128),
        wait=wait_exponential_jitter(2, 128),
        retry=retry_if_not_exception_type(litellm.exceptions.NotFoundError),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
        reraise=True,
    )
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

        try:
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
                api_key=self.api_key,
                api_base=self.endpoint,
                response_model=response_model,
            )
        except (
            ContentFilterFinishReasonError,
            ContentPolicyViolationError,
            InstructorRetryException,
        ) as error:
            if (
                isinstance(error, InstructorRetryException)
                and "content management policy" not in str(error).lower()
            ):
                raise error

            if not (self.fallback_model and self.fallback_api_key and self.fallback_endpoint):
                raise ContentPolicyFilterError(
                    f"The provided input contains content that is not aligned with our content policy: {text_input}"
                ) from error

            try:
                return await self.aclient.chat.completions.create(
                    model=self.fallback_model,
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
                    api_key=self.fallback_api_key,
                    api_base=self.fallback_endpoint,
                    response_model=response_model,
                )
            except (
                ContentFilterFinishReasonError,
                ContentPolicyViolationError,
                InstructorRetryException,
            ) as error:
                if (
                    isinstance(error, InstructorRetryException)
                    and "content management policy" not in str(error).lower()
                ):
                    raise error
                else:
                    raise ContentPolicyFilterError(
                        f"The provided input contains content that is not aligned with our content policy: {text_input}"
                    ) from error
