import litellm
import instructor
from pydantic import BaseModel
from typing import Type, Optional
from litellm import JSONSchemaValidationError
from cognee.infrastructure.files.utils.open_data_file import open_data_file
from cognee.shared.logging_utils import get_logger
from cognee.modules.observability.get_observe import get_observe
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter import (
    GenericAPIAdapter,
)
from cognee.infrastructure.llm.config import get_llm_config
from cognee.shared.rate_limiting import llm_rate_limiter_context_manager

import logging
from tenacity import (
    retry,
    stop_after_delay,
    wait_exponential_jitter,
    retry_if_not_exception_type,
    before_sleep_log,
)
from ..types import TranscriptionReturnType
from mistralai import Mistral

logger = get_logger()
observe = get_observe()


class MistralAdapter(GenericAPIAdapter):
    """
    Adapter for Mistral AI API, for structured output generation and prompt display.

    Public methods:
    - acreate_structured_output
    - show_prompt
    """

    default_instructor_mode = "mistral_tools"

    def __init__(
        self,
        api_key: str,
        model: str,
        max_completion_tokens: int,
        endpoint: str = None,
        transcription_model: str = None,
        image_transcribe_model: str = None,
        instructor_mode: str = None,
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            max_completion_tokens=max_completion_tokens,
            name="Mistral",
            endpoint=endpoint,
            transcription_model=transcription_model,
            image_transcribe_model=image_transcribe_model,
        )

        self.instructor_mode = instructor_mode if instructor_mode else self.default_instructor_mode

        self.aclient = instructor.from_litellm(
            litellm.acompletion,
            mode=instructor.Mode(self.instructor_mode),
            api_key=get_llm_config().llm_api_key,
        )
        self.mistral_client = Mistral(api_key=self.api_key)

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
        try:
            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": f"""Use the given format to extract information
                from the following input: {text_input}""",
                },
            ]
            try:
                async with llm_rate_limiter_context_manager():
                    response = await self.aclient.chat.completions.create(
                        model=self.model,
                        max_tokens=self.max_completion_tokens,
                        max_retries=2,
                        messages=messages,
                        response_model=response_model,
                    )
                if response.choices and response.choices[0].message.content:
                    content = response.choices[0].message.content
                    return response_model.model_validate_json(content)
                else:
                    raise ValueError("Failed to get valid response after retries")
            except litellm.exceptions.BadRequestError as e:
                logger.error(f"Bad request error: {str(e)}")
                raise ValueError(f"Invalid request: {str(e)}")

        except JSONSchemaValidationError as e:
            logger.error(f"Schema validation failed: {str(e)}")
            logger.debug(f"Raw response: {e.raw_response}")
            raise ValueError(f"Response failed schema validation: {str(e)}")

    @observe(as_type="transcription")
    @retry(
        stop=stop_after_delay(128),
        wait=wait_exponential_jitter(2, 128),
        retry=retry_if_not_exception_type(litellm.exceptions.NotFoundError),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
        reraise=True,
    )
    async def create_transcript(self, input) -> Optional[TranscriptionReturnType]:
        """
        Generate an audio transcript from a user query.

        This method creates a transcript from the specified audio file.
        The audio file is processed and the transcription is retrieved from the API.

        Parameters:
        -----------
            - input: The path to the audio file that needs to be transcribed.

        Returns:
        --------
            The generated transcription of the audio file.
        """
        transcription_model = self.transcription_model
        if self.transcription_model.startswith("mistral"):
            transcription_model = self.transcription_model.split("/")[-1]
        file_name = input.split("/")[-1]
        async with open_data_file(input, mode="rb") as f:
            transcription_response = self.mistral_client.audio.transcriptions.complete(
                model=transcription_model,
                file={
                    "content": f,
                    "file_name": file_name,
                },
            )

            return TranscriptionReturnType(transcription_response.text, transcription_response)
