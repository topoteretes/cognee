"""Adapter for Gemini API LLM provider"""

import logging
from typing import Any

import instructor
import litellm
from instructor.core import InstructorRetryException
from litellm.exceptions import ContentPolicyViolationError
from openai import ContentFilterFinishReasonError
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_delay,
    wait_exponential_jitter,
)

from cognee.infrastructure.llm.exceptions import ContentPolicyFilterError
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter import (
    GenericAPIAdapter,
)
from cognee.modules.observability.get_observe import get_observe
from cognee.shared.logging_utils import get_logger
from cognee.shared.rate_limiting import llm_rate_limiter_context_manager

logger = get_logger()
observe = get_observe()


class GeminiAdapter(GenericAPIAdapter):
    """
    Adapter for Gemini API LLM provider.

    This class initializes the API adapter with necessary credentials and configurations for
    interacting with the gemini LLM models. It provides methods for creating structured outputs
    based on user input and system prompts, as well as multimodal processing capabilities.

    Public methods:
    - acreate_structured_output(text_input: str, system_prompt: str, response_model: Type[BaseModel]) -> BaseModel
    - create_transcript(input) -> BaseModel: Transcribe audio files to text
    - transcribe_image(input) -> BaseModel: Inherited from GenericAPIAdapter
    """

    default_instructor_mode = "json_mode"

    def __init__(
        self,
        api_key: str,
        model: str,
        max_completion_tokens: int,
        endpoint: str | None = None,
        api_version: str | None = None,
        transcription_model: str | None = None,
        instructor_mode: str | None = None,
        fallback_model: str | None = None,
        fallback_api_key: str | None = None,
        fallback_endpoint: str | None = None,
        llm_args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            max_completion_tokens=max_completion_tokens,
            name="Gemini",
            endpoint=endpoint,
            api_version=api_version,
            transcription_model=transcription_model,
            fallback_model=fallback_model,
            fallback_api_key=fallback_api_key,
            fallback_endpoint=fallback_endpoint,
            llm_args=llm_args,
        )
        self.llm_args: dict[str, Any] = llm_args or {}
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
        self, text_input: str, system_prompt: str, response_model: type[BaseModel], **kwargs
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

        merged_kwargs = {**self.llm_args, **kwargs}
        try:
            async with llm_rate_limiter_context_manager():
                return await self.aclient.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": f"""{text_input}""",
                        },
                    ],
                    api_key=self.api_key,
                    max_retries=2,
                    api_base=self.endpoint,
                    api_version=self.api_version,
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

            if not (self.fallback_model and self.fallback_api_key and self.fallback_endpoint):
                raise ContentPolicyFilterError(
                    f"The provided input contains content that is not aligned with our content policy: {text_input}"
                )

            try:
                async with llm_rate_limiter_context_manager():
                    return await self.aclient.chat.completions.create(
                        model=self.fallback_model,
                        messages=[
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            {
                                "role": "user",
                                "content": f"""{text_input}""",
                            },
                        ],
                        max_retries=2,
                        api_key=self.fallback_api_key,
                        api_base=self.fallback_endpoint,
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
                else:
                    raise ContentPolicyFilterError(
                        f"The provided input contains content that is not aligned with our content policy: {text_input}"
                    )
