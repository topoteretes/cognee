"""Adapter for Generic API LLM provider API"""

import base64
import mimetypes
import litellm
import instructor
from typing import Type, Optional
from pydantic import BaseModel
from openai import ContentFilterFinishReasonError
from litellm.exceptions import ContentPolicyViolationError
from instructor.core import InstructorRetryException

from cognee.infrastructure.llm.exceptions import ContentPolicyFilterError
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.llm_interface import (
    LLMInterface,
)
from cognee.infrastructure.files.utils.open_data_file import open_data_file
from cognee.modules.observability.get_observe import get_observe
import logging
from cognee.shared.rate_limiting import llm_rate_limiter_context_manager
from cognee.shared.logging_utils import get_logger
from tenacity import (
    retry,
    stop_after_delay,
    wait_exponential_jitter,
    retry_if_not_exception_type,
    before_sleep_log,
)

from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.types import (
    TranscriptionReturnType,
)

logger = get_logger()
observe = get_observe()


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

    MAX_RETRIES = 5
    default_instructor_mode = "json_mode"

    def __init__(
        self,
        api_key: str,
        model: str,
        max_completion_tokens: int,
        name: str,
        endpoint: str = None,
        api_version: str = None,
        transcription_model: str = None,
        image_transcribe_model: str = None,
        instructor_mode: str = None,
        fallback_model: str = None,
        fallback_api_key: str = None,
        fallback_endpoint: str = None,
    ):
        self.name = name
        self.model = model
        self.api_key = api_key
        self.api_version = api_version
        self.endpoint = endpoint
        self.max_completion_tokens = max_completion_tokens
        self.transcription_model = transcription_model or model
        self.image_transcribe_model = image_transcribe_model or model
        self.fallback_model = fallback_model
        self.fallback_api_key = fallback_api_key
        self.fallback_endpoint = fallback_endpoint

        self.instructor_mode = instructor_mode if instructor_mode else self.default_instructor_mode

        self.aclient = instructor.from_litellm(
            litellm.acompletion, mode=instructor.Mode(self.instructor_mode)
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
                    max_retries=2,
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
                async with llm_rate_limiter_context_manager():
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
                        max_retries=2,
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

    @observe(as_type="transcription")
    @retry(
        stop=stop_after_delay(128),
        wait=wait_exponential_jitter(2, 128),
        retry=retry_if_not_exception_type(litellm.exceptions.NotFoundError),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
        reraise=True,
    )
    async def create_transcript(self, input) -> TranscriptionReturnType:
        """
        Generate an audio transcript from a user query.

        This method creates a transcript from the specified audio file, raising a
        FileNotFoundError if the file does not exist. The audio file is processed and the
        transcription is retrieved from the API.

        Parameters:
        -----------
            - input: The path to the audio file that needs to be transcribed.

        Returns:
        --------
            The generated transcription of the audio file.
        """
        async with open_data_file(input, mode="rb") as audio_file:
            encoded_string = base64.b64encode(audio_file.read()).decode("utf-8")
        mime_type, _ = mimetypes.guess_type(input)
        if not mime_type or not mime_type.startswith("audio/"):
            raise ValueError(
                f"Could not determine MIME type for audio file: {input}. Is the extension correct?"
            )
        response = await litellm.acompletion(
            model=self.transcription_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "file": {"file_data": f"data:{mime_type};base64,{encoded_string}"},
                        },
                        {"type": "text", "text": "Transcribe the following audio precisely."},
                    ],
                }
            ],
            api_key=self.api_key,
            api_version=self.api_version,
            max_completion_tokens=self.max_completion_tokens,
            api_base=self.endpoint,
            max_retries=self.MAX_RETRIES,
        )

        return TranscriptionReturnType(response.choices[0].message.content, response)

    @observe(as_type="transcribe_image")
    @retry(
        stop=stop_after_delay(128),
        wait=wait_exponential_jitter(2, 128),
        retry=retry_if_not_exception_type(litellm.exceptions.NotFoundError),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
        reraise=True,
    )
    async def transcribe_image(self, input) -> BaseModel:
        """
        Generate a transcription of an image from a user query.

        This method encodes the image and sends a request to the API to obtain a
        description of the contents of the image.

        Parameters:
        -----------
            - input: The path to the image file that needs to be transcribed.

        Returns:
        --------
            - BaseModel: A structured output generated by the model, returned as an instance of
              BaseModel.
        """
        async with open_data_file(input, mode="rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        mime_type, _ = mimetypes.guess_type(input)
        if not mime_type or not mime_type.startswith("image/"):
            raise ValueError(
                f"Could not determine MIME type for image file: {input}. Is the extension correct?"
            )
        response = await litellm.acompletion(
            model=self.image_transcribe_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What's in this image?",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{encoded_image}",
                            },
                        },
                    ],
                }
            ],
            api_key=self.api_key,
            api_base=self.endpoint,
            api_version=self.api_version,
            max_completion_tokens=300,
            max_retries=self.MAX_RETRIES,
        )
        return response
