"""Adapter for MiniMax LLM provider API"""

import litellm
import instructor
from typing import Any, Dict, Type, Optional
from pydantic import BaseModel
from openai import ContentFilterFinishReasonError
from litellm.exceptions import ContentPolicyViolationError
from instructor.core import InstructorRetryException

import logging
from cognee.shared.rate_limiting import llm_rate_limiter_context_manager

from tenacity import (
    retry,
    stop_after_delay,
    wait_exponential_jitter,
    retry_if_not_exception_type,
    before_sleep_log,
)

from cognee.infrastructure.llm.exceptions import ContentPolicyFilterError
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter import (
    GenericAPIAdapter,
)
from cognee.shared.logging_utils import get_logger
from cognee.modules.observability.get_observe import get_observe

logger = get_logger()
observe = get_observe()

MINIMAX_DEFAULT_BASE_URL = "https://api.minimax.io/v1"


class MiniMaxAdapter(GenericAPIAdapter):
    """
    Adapter for MiniMax LLM provider API.

    MiniMax provides OpenAI-compatible chat completion endpoints.
    Supported models: MiniMax-M2.5, MiniMax-M2.5-highspeed (204K context).

    Public methods:
    - acreate_structured_output
    """

    default_instructor_mode = "json_mode"

    def __init__(
        self,
        api_key: str,
        model: str,
        max_completion_tokens: int,
        endpoint: str = None,
        instructor_mode: str = None,
        llm_args: Optional[Dict[str, Any]] = None,
    ):
        # Default to MiniMax API endpoint if none provided
        if not endpoint:
            endpoint = MINIMAX_DEFAULT_BASE_URL

        # litellm requires the "openai/" prefix for OpenAI-compatible endpoints
        if not model.startswith("openai/"):
            model = f"openai/{model}"

        super().__init__(
            api_key=api_key,
            model=model,
            max_completion_tokens=max_completion_tokens,
            name="MiniMax",
            endpoint=endpoint,
            llm_args=llm_args,
        )
        self.llm_args = llm_args
        self.instructor_mode = instructor_mode if instructor_mode else self.default_instructor_mode

        self.aclient = instructor.from_litellm(
            litellm.acompletion, mode=instructor.Mode(self.instructor_mode)
        )

    @observe(as_type="generation")
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
        self, text_input: str, system_prompt: str, response_model: Type[BaseModel], **kwargs
    ) -> BaseModel:
        """
        Generate a response from a user query.

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

        merged_kwargs = {**self.llm_args, **kwargs}

        # MiniMax requires temperature in (0.0, 1.0]; 0 is rejected
        if "temperature" not in merged_kwargs or merged_kwargs.get("temperature", 0) == 0:
            merged_kwargs["temperature"] = 1.0

        # MiniMax does not support response_format; remove if present
        merged_kwargs.pop("response_format", None)

        try:
            async with llm_rate_limiter_context_manager():
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
                    api_key=self.api_key,
                    max_retries=2,
                    api_base=self.endpoint,
                    response_model=response_model,
                    **merged_kwargs,
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

            raise ContentPolicyFilterError(
                f"The provided input contains content that is not aligned with our content policy: {text_input}"
            ) from error
